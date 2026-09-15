// SPDX-License-Identifier: Apache-2.0
// SLH-DSA-SHA2-128s fused GPU signer — kernel (Card SLH-002, P1).
//
// Mapping: signature-per-warp. Within a warp:
//   - FORS: the k=14 trees are built one at a time; the 2^a=4096 leaves
//     (PRF then F) and each treehash level are striped across the 32 lanes.
//   - HT: for each of the d=7 XMSS trees, each lane owns 512/32 = 16 whole
//     WOTS leaves (35 chains x 15 F steps, sequential per lane — uniform
//     work, so lanes stay in lockstep); treehash levels striped as in FORS.
// Tree levels live in per-warp global scratch (ping-pong buffers); chain
// state and single-block hashing stay in registers.

#include "slh_sha2_128s_sign.cuh"

using namespace slh128s;

// Extract FORS index i (a=12 bits, big-endian bit order) from md.
// base_2b(md, 12, 14): bits [12i, 12i+12). md is padded to 24 bytes so the
// 3-byte window read is always in bounds.
__device__ __forceinline__
uint32_t fors_index(const uint8_t* md, int i) {
    int bitoff = A * i;
    int byteoff = bitoff >> 3;
    uint32_t v = ((uint32_t)md[byteoff] << 16) |
                 ((uint32_t)md[byteoff + 1] << 8) |
                 (uint32_t)md[byteoff + 2];
    return (v >> (12 - (bitoff & 7))) & 0xFFFu;
}

// Card SLH-002 P2 refactor: the signing body (FORS + hypertree from a fully
// prepared SignCase) is extracted into a per-warp __device__ function shared
// by BOTH kernel entries — the P1 entry (host-prologue SignCase array, the
// byte-exact differential/KAT gate path, semantics unchanged) and the P2
// keyslot entry (device prologue). The body is byte-identical to the P1
// kernel that passed the D-097 gates.
static __device__ void slh_sign_case_warp(const SignCase& cs, const int lane,
                                          uint8_t* __restrict__ sig,
                                          uint4* bufA, uint4* bufB,
                                          uint4* small) {
    // pk.seed midstate — cached ONCE, reused by every F/PRF/H/T call.
    uint32_t pkw[4], mid[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) pkw[i] = slh_load_be32(cs.pk_seed + 4 * i);
    slh_sha256_midstate(pkw, mid);

    const uint4 sk_seed = slh_load_node(cs.sk_seed);
    const uint64_t idx_tree = cs.idx_tree;
    const uint32_t idx_leaf = cs.idx_leaf;

    if (lane == 0) {   // R goes verbatim into sig[0:16)
        #pragma unroll
        for (int i = 0; i < 16; ++i) sig[i] = cs.r[i];
    }

    // ---------------- FORS (layer 0, tree idx_tree, keypair idx_leaf) ------
    for (int t = 0; t < K; ++t) {
        const uint32_t tgt = fors_index(cs.md, t);
        uint8_t* tsig = sig + SIG_FORS_OFF + t * SIG_FORS_TREE;

        for (int j = lane; j < FORS_LEAVES; j += 32) {
            uint32_t tidx = ((uint32_t)t << A) + j;
            // fors_skGen: PRF with FORS_PRF ADRS (w2 stays 0)
            uint4 sk = slh_tweak16(mid, 0, idx_tree, T_FORS_PRF,
                                   idx_leaf, 0, tidx, sk_seed);
            if ((uint32_t)j == tgt) slh_store_be16(tsig, sk);
            bufA[j] = slh_tweak16(mid, 0, idx_tree, T_FORS_TREE,
                                  idx_leaf, 0, tidx, sk);
        }
        __syncwarp();
        if (lane == 0)
            slh_store_be16(tsig + N, bufA[tgt ^ 1]);           // auth level 0
        uint4* cur = bufA;
        uint4* nxt = bufB;
        for (int z = 1; z <= A; ++z) {
            int width = FORS_LEAVES >> z;
            for (int j = lane; j < width; j += 32) {
                nxt[j] = slh_tweak32(mid, 0, idx_tree, T_FORS_TREE, idx_leaf,
                                     z, ((uint32_t)t << (A - z)) + j,
                                     cur[2 * j], cur[2 * j + 1]);
            }
            __syncwarp();
            if (z < A && lane == 0)
                slh_store_be16(tsig + (1 + z) * N, nxt[(tgt >> z) ^ 1]);
            uint4* tmp = cur; cur = nxt; nxt = tmp;
        }
        if (lane == 0) small[1 + t] = cur[0];                  // tree root
        __syncwarp();
    }

    // pk_fors = T(FORS_ROOTS, roots[0..13]) — one 246-byte hash, lane 0.
    if (lane == 0) {
        uint8_t buf[22 + K * N];
        slh_adrsc_bytes(0, idx_tree, T_FORS_ROOTS, idx_leaf, 0, 0, buf);
        for (int t = 0; t < K; ++t)
            slh_store_be16(buf + 22 + t * N, small[1 + t]);
        uint32_t out[8];
        slh_sha256_mid_bytes(mid, buf, 22 + K * N, out);
        small[0] = make_uint4(out[0], out[1], out[2], out[3]);
    }
    __syncwarp();

    // ---------------- Hypertree: d=7 XMSS trees, bottom to top -------------
    uint64_t t_tree = idx_tree;
    uint32_t t_leaf = idx_leaf;
    for (int layer = 0; layer < D; ++layer) {
        // message for this layer = FORS pk (layer 0) / previous root
        uint4 msg = small[0];
        const uint32_t msgw[4] = {msg.x, msg.y, msg.z, msg.w};
        const uint32_t csum13 = slh_wots_csum13(msgw);
        uint8_t* lsig = sig + SIG_HT_OFF + layer * SIG_HT_LAYER;

        for (int leaf = lane; leaf < XMSS_LEAVES; leaf += 32) {
            // WOTS pkGen for this leaf; the signing leaf also captures its
            // chain value at step lengths[c] (that IS wots_sign, for free).
            uint8_t tbuf[22 + LEN * N];   // T input: ADRSc || len chain ends
            slh_adrsc_bytes((uint32_t)layer, t_tree, T_WOTS_PK,
                            (uint32_t)leaf, 0, 0, tbuf);
            const bool is_sig_leaf = ((uint32_t)leaf == t_leaf);
            for (int c = 0; c < LEN; ++c) {
                uint4 v = slh_tweak16(mid, layer, t_tree, T_WOTS_PRF,
                                      leaf, c, 0, sk_seed);
                const uint32_t dlen = slh_wots_digit(c, msgw, csum13);
                #pragma unroll
                for (int s = 0; s < W - 1; ++s) {
                    if (is_sig_leaf && dlen == (uint32_t)s)
                        slh_store_be16(lsig + c * N, v);
                    v = slh_tweak16(mid, layer, t_tree, T_WOTS_HASH,
                                    leaf, c, s, v);
                }
                if (is_sig_leaf && dlen == W - 1)
                    slh_store_be16(lsig + c * N, v);
                slh_store_be16(tbuf + 22 + c * N, v);
            }
            uint32_t out[8];
            slh_sha256_mid_bytes(mid, tbuf, 22 + LEN * N, out);
            bufA[leaf] = make_uint4(out[0], out[1], out[2], out[3]);
        }
        __syncwarp();

        if (lane == 0)
            slh_store_be16(lsig + LEN * N, bufA[t_leaf ^ 1]);  // auth level 0
        uint4* cur = bufA;
        uint4* nxt = bufB;
        for (int z = 1; z <= HP; ++z) {
            int width = XMSS_LEAVES >> z;
            for (int j = lane; j < width; j += 32) {
                nxt[j] = slh_tweak32(mid, layer, t_tree, T_TREE, 0, z, j,
                                     cur[2 * j], cur[2 * j + 1]);
            }
            __syncwarp();
            if (z < HP && lane == 0)
                slh_store_be16(lsig + (LEN + z) * N, nxt[(t_leaf >> z) ^ 1]);
            uint4* tmp = cur; cur = nxt; nxt = tmp;
        }
        if (lane == 0) small[0] = cur[0];   // root -> next layer's message
        __syncwarp();
        t_leaf = (uint32_t)(t_tree & ((1u << HP) - 1));
        t_tree >>= HP;
    }
}

// P1 entry: host-prepared SignCase array (differential-oracle harness path;
// byte-exact gate of record, D-097). Unchanged semantics.
__global__ void __launch_bounds__(128)
slh_sha2_128s_sign_kernel(const SignCase* __restrict__ cases, int n_cases,
                          uint8_t* __restrict__ sigs,
                          uint8_t* __restrict__ scratch) {
    const int warp_gid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_gid >= n_cases) return;

    uint8_t* sig = sigs + (size_t)warp_gid * SIG_BYTES;
    uint4* bufA = (uint4*)(scratch + (size_t)warp_gid * SCRATCH_BYTES_PER_WARP);
    uint4* bufB = bufA + SCRATCH_BUFA_NODES;
    uint4* small = bufB + SCRATCH_BUFB_NODES;  // [0]=pk_fors/root, [1..14]=FORS roots

    slh_sign_case_warp(cases[warp_gid], lane, sig, bufA, bufB, small);
}

// Card SLH-002 P2 entry: keyslot key block + DEVICE prologue from the raw
// message. Lane 0 verifies key == mirror (fail-closed: mismatch -> status 1,
// no signature bytes), computes R = PRF_msg (HMAC-SHA-256 with SK.prf) and
// the H_msg/MGF1 digest split, parks the SignCase in the warp's scratch
// small area (small[16..], disjoint from small[0..14] which the body owns),
// then all lanes run the shared signing body.
__global__ void __launch_bounds__(128)
slh_sha2_128s_sign_keyslot_kernel(const uint8_t* __restrict__ key128,
                                  const KeyslotSignReq* __restrict__ reqs,
                                  int n_cases,
                                  const uint8_t* __restrict__ msgs,
                                  uint8_t* __restrict__ sigs,
                                  uint8_t* __restrict__ status,
                                  uint8_t* __restrict__ scratch) {
    const int warp_gid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_gid >= n_cases) return;

    uint8_t* sig = sigs + (size_t)warp_gid * SIG_BYTES;
    uint4* bufA = (uint4*)(scratch + (size_t)warp_gid * SCRATCH_BYTES_PER_WARP);
    uint4* bufB = bufA + SCRATCH_BUFA_NODES;
    uint4* small = bufB + SCRATCH_BUFB_NODES;
    // SignCase parked past the body's small[0..14] working set (88 B <= 256 B).
    static_assert(sizeof(SignCase) <= (SCRATCH_SMALL_NODES - 16) * 16,
                  "SignCase must fit the spare small-area scratch");
    SignCase* cs = reinterpret_cast<SignCase*>(small + 16);

    int key_ok = 0;
    if (lane == 0) {
        // Integrity: key block vs mirror (rowhammer/fault defense). The 64
        // key bytes are sk_seed||sk_prf||pk_seed||pk_root; mirror follows.
        key_ok = 1;
        for (int i = 0; i < 64; ++i)
            if (key128[i] != key128[64 + i]) { key_ok = 0; break; }
        if (key_ok) {
            const KeyslotSignReq& rq = reqs[warp_gid];
            const uint8_t* sk_seed = key128;
            const uint8_t* sk_prf  = key128 + 16;
            const uint8_t* pk_seed = key128 + 32;
            const uint8_t* pk_root = key128 + 48;
            const uint8_t* opt_rand = rq.hedged ? rq.addrnd : pk_seed;
            slh_sign_prologue_device(sk_prf, pk_seed, pk_root, opt_rand,
                                     msgs + rq.msg_offset, rq.msg_len, *cs);
            #pragma unroll
            for (int i = 0; i < 16; ++i) cs->sk_seed[i] = sk_seed[i];
            #pragma unroll
            for (int i = 0; i < 16; ++i) cs->pk_seed[i] = pk_seed[i];
        }
        status[warp_gid] = key_ok ? 0 : 1;   // 1 = key/mirror mismatch
    }
    key_ok = __shfl_sync(0xFFFFFFFFu, key_ok, 0);
    __syncwarp();
    if (!key_ok) return;   // fail-closed: no signature bytes on key fault

    slh_sign_case_warp(*cs, lane, sig, bufA, bufB, small);
}

extern "C" cudaError_t slh_sha2_128s_sign_launch(
    const SignCase* d_cases, int n_cases, uint8_t* d_sigs,
    uint8_t* d_scratch, cudaStream_t stream) {
    if (n_cases <= 0) return cudaErrorInvalidValue;
    const int block = 128;                       // 4 warps = 4 signatures
    const int grid = (n_cases + (block / 32) - 1) / (block / 32);
    slh_sha2_128s_sign_kernel<<<grid, block, 0, stream>>>(
        d_cases, n_cases, d_sigs, d_scratch);
    return cudaGetLastError();
}

// Card SLH-002 P2: keyslot entry launcher (device prologue).
extern "C" cudaError_t slh_sha2_128s_sign_keyslot_launch(
    const uint8_t* d_key, const KeyslotSignReq* d_reqs, int n_cases,
    const uint8_t* d_msgs, uint8_t* d_sigs, uint8_t* d_status,
    uint8_t* d_scratch, cudaStream_t stream) {
    if (n_cases <= 0 || !d_key || !d_reqs || !d_msgs || !d_sigs || !d_status ||
        !d_scratch) {
        return cudaErrorInvalidValue;
    }
    const int block = 128;                       // 4 warps = 4 signatures
    const int grid = (n_cases + (block / 32) - 1) / (block / 32);
    slh_sha2_128s_sign_keyslot_kernel<<<grid, block, 0, stream>>>(
        d_key, d_reqs, n_cases, d_msgs, d_sigs, d_status, d_scratch);
    return cudaGetLastError();
}
