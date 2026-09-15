// SPDX-License-Identifier: Apache-2.0
// SLH-DSA-SHA2-128s fused GPU signer — device side (Card SLH-002, P1).
//
// Standalone: no ring, no engine ABI. One signature per warp; all ~2.2M
// SHA-256 compressions per signature happen inside the kernel from a
// pk.seed midstate cached in registers.
//
// Byte-exact reference: core/pqc/slh_dsa/python (Sha2Backend + slh_adrs.py).
// Value representation: 16-byte hash nodes are held as uint4 of BIG-ENDIAN
// uint32 words (word i = bytes 4i..4i+3 of the node).

#ifndef SLH_SHA2_128S_SIGN_CUH
#define SLH_SHA2_128S_SIGN_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// SLH-DSA-SHA2-128s parameters (FIPS 205 Table 2)
// ---------------------------------------------------------------------------
namespace slh128s {

constexpr int N        = 16;   // hash bytes
constexpr int H_TOTAL  = 63;
constexpr int D        = 7;
constexpr int HP       = 9;    // h' = h/d
constexpr int A        = 12;   // FORS tree height
constexpr int K        = 14;   // FORS trees
constexpr int LG_W     = 4;
constexpr int W        = 16;
constexpr int LEN1     = 32;
constexpr int LEN2     = 3;
constexpr int LEN      = 35;   // WOTS chains
constexpr int MD_BYTES = 21;   // ceil(k*a/8)
constexpr int SIG_BYTES  = 7856;
constexpr int FORS_LEAVES = 1 << A;    // 4096
constexpr int XMSS_LEAVES = 1 << HP;   // 512

// Signature layout offsets (bytes)
constexpr int SIG_FORS_OFF  = N;                       // 16
constexpr int SIG_FORS_TREE = (1 + A) * N;             // 208 per tree
constexpr int SIG_HT_OFF    = N + K * SIG_FORS_TREE;   // 2928
constexpr int SIG_HT_LAYER  = (LEN + HP) * N;          // 704 per layer

// ADRS types (FIPS 205 Table 1)
constexpr uint32_t T_WOTS_HASH  = 0;
constexpr uint32_t T_WOTS_PK    = 1;
constexpr uint32_t T_TREE       = 2;
constexpr uint32_t T_FORS_TREE  = 3;
constexpr uint32_t T_FORS_ROOTS = 4;
constexpr uint32_t T_WOTS_PRF   = 5;
constexpr uint32_t T_FORS_PRF   = 6;

// Per-warp scratch: FORS level ping-pong (4096 + 2048 nodes, reused by the
// much smaller XMSS trees) + 32-node small area (pk_fors/root msg + roots).
constexpr int SCRATCH_BUFA_NODES  = FORS_LEAVES;       // 4096
constexpr int SCRATCH_BUFB_NODES  = FORS_LEAVES / 2;   // 2048
constexpr int SCRATCH_SMALL_NODES = 32;
constexpr int SCRATCH_NODES_PER_WARP =
    SCRATCH_BUFA_NODES + SCRATCH_BUFB_NODES + SCRATCH_SMALL_NODES;  // 6176
constexpr size_t SCRATCH_BYTES_PER_WARP = (size_t)SCRATCH_NODES_PER_WARP * 16;

// Exact SHA-256 compression count the kernel performs per signature
// (excludes the per-lane midstate setup, counted separately below):
//   FORS: k*(2^a PRF + 2^a F + (2^a - 1) H) = 14*(4096+4096+4095) = 172,018
//         + pk_fors T: tail 22+224=246 B -> 5 blocks - 1 midstate  =       4
//   HT:   d*( 2^h' * (len PRF + len*(w-1) F + T(22+560 -> 10)) + (2^h'-1) H )
//       = 7*( 512*(35 + 525 + 10) + 511 ) = 7*292,351               = 2,046,457
constexpr long long COMPRESSIONS_PER_SIG = 172018LL + 4 + 2046457LL;  // 2,218,479
// Midstate setup: 1 compression per lane, 32 lanes per signature.
constexpr long long MIDSTATE_COMPRESSIONS_PER_SIG = 32;

// Per-signature kernel input. Host computes R = PRF_msg(sk_prf, opt_rand, M)
// and the H_msg digest split (2 of ~2.2M hashes); the kernel owns FORS + HT.
struct SignCase {
    uint8_t  sk_seed[16];
    uint8_t  pk_seed[16];
    uint8_t  r[16];       // randomizer R (goes verbatim into sig[0:16))
    uint8_t  md[24];      // 21 bytes used; padded with zeros (bit reader
                          // may touch up to md[21])
    uint64_t idx_tree;    // 54-bit tree index from H_msg
    uint32_t idx_leaf;    // 9-bit leaf index from H_msg
    uint32_t _pad;
};
static_assert(sizeof(SignCase) == 88, "SignCase ABI: 88 bytes");

// Card SLH-002 P2: everything from here to the KeyslotSignReq descriptor is
// DEVICE code (register-promoted SHA-256, tweakable hashes, device prologue)
// and uses CUDA intrinsics — visible only under nvcc. Host TUs (e.g.
// engine_host.cpp) include this header for the constants, descriptors and
// launcher declarations only.
#ifdef __CUDACC__

// ---------------------------------------------------------------------------
// Register-promoted SHA-256 (D-095 hard requirement 1): constexpr K table,
// fully unrolled rounds, state and message schedule in registers. Same round
// math as the validated SLH-001 microbench (7.39e9 compressions/s A100 class).
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t slh_rotr32(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

__device__ __forceinline__
void slh_sha256_compress(uint32_t state[8], const uint32_t block[16]) {
    constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    uint32_t W[64];
    #pragma unroll
    for (int t = 0; t < 16; ++t) W[t] = block[t];
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        uint32_t s0 = slh_rotr32(W[t-15], 7) ^ slh_rotr32(W[t-15], 18) ^ (W[t-15] >> 3);
        uint32_t s1 = slh_rotr32(W[t-2], 17) ^ slh_rotr32(W[t-2], 19) ^ (W[t-2] >> 10);
        W[t] = W[t-16] + s0 + W[t-7] + s1;
    }
    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t S1 = slh_rotr32(e, 6) ^ slh_rotr32(e, 11) ^ slh_rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K[t] + W[t];
        uint32_t S0 = slh_rotr32(a, 2) ^ slh_rotr32(a, 13) ^ slh_rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Midstate = SHA-256 state after absorbing pk_seed(16) || 0^48 as one block
// (the Sha2Backend padding block; hard requirement 3).
__device__ __forceinline__
void slh_sha256_midstate(const uint32_t pk_seed_words[4], uint32_t mid[8]) {
    mid[0] = 0x6a09e667u; mid[1] = 0xbb67ae85u; mid[2] = 0x3c6ef372u;
    mid[3] = 0xa54ff53au; mid[4] = 0x510e527fu; mid[5] = 0x9b05688cu;
    mid[6] = 0x1f83d9abu; mid[7] = 0x5be0cd19u;
    uint32_t blk[16];
    #pragma unroll
    for (int i = 0; i < 4; ++i) blk[i] = pk_seed_words[i];
    #pragma unroll
    for (int i = 4; i < 16; ++i) blk[i] = 0;
    slh_sha256_compress(mid, blk);
}

// ---------------------------------------------------------------------------
// ADRSc (FIPS 205 §11.2, slh_adrs.py compressed()):
//   22 bytes = layer(1) || tree(8, BE) || type(1) || w1(4, BE) || w2(4, BE)
//              || w3(4, BE)
// where (w1,w2,w3) are the three type-specific ADRS words: w1 = key-pair
// address, w2 = chain address / tree height, w3 = hash address / tree index.
// ---------------------------------------------------------------------------

// Pack ADRSc into big-endian block words W[0..5]. The 22-byte ADRSc ends
// mid-word: W[5]'s high 16 bits are ADRSc bytes 20..21, low 16 bits are
// message bytes 0..1 (caller ORs them in).
__device__ __forceinline__
void slh_adrsc_words(uint32_t layer, uint64_t tree, uint32_t type,
                     uint32_t w1, uint32_t w2, uint32_t w3, uint32_t W[16]) {
    W[0] = (layer << 24) | (uint32_t)((tree >> 40) & 0xFFFFFFu);
    W[1] = (uint32_t)(tree >> 8);
    W[2] = ((uint32_t)(tree & 0xFFu) << 24) | ((type & 0xFFu) << 16) | (w1 >> 16);
    W[3] = (w1 << 16) | (w2 >> 16);
    W[4] = (w2 << 16) | (w3 >> 16);
    W[5] = (w3 << 16);
}

// Byte form of ADRSc, for the long T() inputs that go through the generic
// streaming hash.
__device__ __forceinline__
void slh_adrsc_bytes(uint32_t layer, uint64_t tree, uint32_t type,
                     uint32_t w1, uint32_t w2, uint32_t w3, uint8_t out[22]) {
    out[0] = (uint8_t)layer;
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[1 + i] = (uint8_t)(tree >> (56 - 8 * i));
    out[9] = (uint8_t)type;
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[10 + i] = (uint8_t)(w1 >> (24 - 8 * i));
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[14 + i] = (uint8_t)(w2 >> (24 - 8 * i));
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[18 + i] = (uint8_t)(w3 >> (24 - 8 * i));
}

// ---------------------------------------------------------------------------
// The tweakable hashes at 128s shapes. F / PRF (16-byte input) and H
// (32-byte input) each fit in EXACTLY ONE compression from the midstate:
//   total message = 64 (seed block) + 22 (ADRSc) + msg; padding 0x80 and the
//   8-byte bit length land inside the same block for msg <= 33 bytes.
// ---------------------------------------------------------------------------

// F(pk.seed, ADRS, m1) / PRF(pk.seed, sk.seed, ADRS): 16-byte message.
__device__ __forceinline__
uint4 slh_tweak16(const uint32_t mid[8], uint32_t layer, uint64_t tree,
                  uint32_t type, uint32_t w1, uint32_t w2, uint32_t w3,
                  uint4 m) {
    uint32_t W[16];
    slh_adrsc_words(layer, tree, type, w1, w2, w3, W);
    W[5] |= m.x >> 16;
    W[6]  = (m.x << 16) | (m.y >> 16);
    W[7]  = (m.y << 16) | (m.z >> 16);
    W[8]  = (m.z << 16) | (m.w >> 16);
    W[9]  = (m.w << 16) | 0x8000u;       // 0x80 pad at byte 38
    #pragma unroll
    for (int i = 10; i < 15; ++i) W[i] = 0;
    W[15] = (64 + 22 + 16) * 8;          // 816-bit total length
    uint32_t st[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) st[i] = mid[i];
    slh_sha256_compress(st, W);
    return make_uint4(st[0], st[1], st[2], st[3]);   // truncate to n=16
}

// H(pk.seed, ADRS, m1, m2): 32-byte message.
__device__ __forceinline__
uint4 slh_tweak32(const uint32_t mid[8], uint32_t layer, uint64_t tree,
                  uint32_t type, uint32_t w1, uint32_t w2, uint32_t w3,
                  uint4 m1, uint4 m2) {
    uint32_t W[16];
    slh_adrsc_words(layer, tree, type, w1, w2, w3, W);
    W[5] |= m1.x >> 16;
    W[6]  = (m1.x << 16) | (m1.y >> 16);
    W[7]  = (m1.y << 16) | (m1.z >> 16);
    W[8]  = (m1.z << 16) | (m1.w >> 16);
    W[9]  = (m1.w << 16) | (m2.x >> 16);
    W[10] = (m2.x << 16) | (m2.y >> 16);
    W[11] = (m2.y << 16) | (m2.z >> 16);
    W[12] = (m2.z << 16) | (m2.w >> 16);
    W[13] = (m2.w << 16) | 0x8000u;      // 0x80 pad at byte 54
    W[14] = 0;
    W[15] = (64 + 22 + 32) * 8;          // 944-bit total length
    uint32_t st[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) st[i] = mid[i];
    slh_sha256_compress(st, W);
    return make_uint4(st[0], st[1], st[2], st[3]);
}

// Generic streaming hash from the midstate over a byte buffer (T() with long
// inputs: WOTS pk 582 B, FORS roots 246 B). Cold path (~14 of every ~570
// compressions), so byte-buffer assembly cost is acceptable.
__device__ inline
void slh_sha256_mid_bytes(const uint32_t mid[8], const uint8_t* data,
                          uint32_t len, uint32_t out[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = mid[i];
    uint32_t W[16];
    uint32_t off = 0;
    while (len - off >= 64) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t* p = data + off + i * 4;
            W[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                   ((uint32_t)p[2] << 8) | (uint32_t)p[3];
        }
        slh_sha256_compress(out, W);
        off += 64;
    }
    uint32_t rem = len - off;
    uint8_t buf[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) buf[i] = 0;
    for (uint32_t i = 0; i < rem; ++i) buf[i] = data[off + i];
    buf[rem] = 0x80;
    uint64_t bits = (uint64_t)(64 + len) * 8;   // + midstate block
    if (rem >= 56) {   // length field does not fit: extra block
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t* p = buf + i * 4;
            W[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                   ((uint32_t)p[2] << 8) | (uint32_t)p[3];
        }
        slh_sha256_compress(out, W);
        #pragma unroll
        for (int i = 0; i < 64; ++i) buf[i] = 0;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) buf[56 + i] = (uint8_t)(bits >> (56 - 8 * i));
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t* p = buf + i * 4;
        W[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
               ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    }
    slh_sha256_compress(out, W);
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void slh_store_be16(uint8_t* p, uint4 v) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) p[i]      = (uint8_t)(v.x >> (24 - 8 * i));
    #pragma unroll
    for (int i = 0; i < 4; ++i) p[4 + i]  = (uint8_t)(v.y >> (24 - 8 * i));
    #pragma unroll
    for (int i = 0; i < 4; ++i) p[8 + i]  = (uint8_t)(v.z >> (24 - 8 * i));
    #pragma unroll
    for (int i = 0; i < 4; ++i) p[12 + i] = (uint8_t)(v.w >> (24 - 8 * i));
}

__device__ __forceinline__ uint32_t slh_load_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

__device__ __forceinline__ uint4 slh_load_node(const uint8_t* p) {
    return make_uint4(slh_load_be32(p), slh_load_be32(p + 4),
                      slh_load_be32(p + 8), slh_load_be32(p + 12));
}

// WOTS chain length (base-w digit) for chain c, from the 16-byte message
// held as 4 BE words + the precomputed 13-bit shifted checksum. Recomputed
// on demand instead of a lengths[35] local array (keeps it in registers).
__device__ __forceinline__
uint32_t slh_wots_digit(int c, const uint32_t msgw[4], uint32_t csum13) {
    if (c < LEN1) return (msgw[c >> 3] >> (28 - 4 * (c & 7))) & 0xFu;
    // checksum digits: base_2b(toByte(csum << 4, 2), 4, 3)
    return (csum13 >> (12 - 4 * (c - LEN1))) & 0xFu;
}

__device__ __forceinline__
uint32_t slh_wots_csum13(const uint32_t msgw[4]) {
    uint32_t csum = 0;
    #pragma unroll
    for (int c = 0; c < LEN1; ++c)
        csum += (W - 1) - ((msgw[c >> 3] >> (28 - 4 * (c & 7))) & 0xFu);
    return csum << 4;   // left-shift per Alg 7: 8 - ((len2*lg_w) % 8) = 4
}

// ---------------------------------------------------------------------------
// Card SLH-002 P2: DEVICE-side sign prologue (FIPS 205 Alg 19 lines 1-7).
//
// P1 computed R = PRF_msg and the H_msg digest split on the HOST (harness
// scaffolding). P2 moves the prologue ON DEVICE: host-side HMAC would require
// SK.prf resident in host RAM, breaking key residency for PQ keys. The cost
// is ~a dozen SHA-256 compressions on one lane per signature — trivial next
// to the ~2.2M the tree walk performs.
//
// These helpers are P2-only; the P1 kernel and its byte-exact gates are
// untouched.
// ---------------------------------------------------------------------------

// Full SHA-256 (own IV + padding) over up to three concatenated byte
// segments. Cold path (prologue only) — byte-buffer assembly is acceptable.
__device__ inline
void slh_sha256_segs(const uint8_t* a, uint32_t la,
                     const uint8_t* b, uint32_t lb,
                     const uint8_t* c, uint32_t lc,
                     uint8_t out[32]) {
    uint32_t st[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    const uint8_t* segs[3] = {a, b, c};
    const uint32_t lens[3] = {la, lb, lc};
    uint8_t buf[64];
    uint32_t fill = 0;
    uint64_t total = 0;
    uint32_t W[16];
    for (int s = 0; s < 3; ++s) {
        const uint8_t* p = segs[s];
        uint32_t len = lens[s];
        total += len;
        for (uint32_t i = 0; i < len; ++i) {
            buf[fill++] = p[i];
            if (fill == 64) {
                #pragma unroll
                for (int t = 0; t < 16; ++t)
                    W[t] = ((uint32_t)buf[4*t] << 24) | ((uint32_t)buf[4*t+1] << 16) |
                           ((uint32_t)buf[4*t+2] << 8) | (uint32_t)buf[4*t+3];
                slh_sha256_compress(st, W);
                fill = 0;
            }
        }
    }
    // padding
    uint64_t bits = total * 8;
    buf[fill++] = 0x80;
    if (fill > 56) {
        while (fill < 64) buf[fill++] = 0;
        #pragma unroll
        for (int t = 0; t < 16; ++t)
            W[t] = ((uint32_t)buf[4*t] << 24) | ((uint32_t)buf[4*t+1] << 16) |
                   ((uint32_t)buf[4*t+2] << 8) | (uint32_t)buf[4*t+3];
        slh_sha256_compress(st, W);
        fill = 0;
    }
    while (fill < 56) buf[fill++] = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) buf[56 + i] = (uint8_t)(bits >> (56 - 8 * i));
    #pragma unroll
    for (int t = 0; t < 16; ++t)
        W[t] = ((uint32_t)buf[4*t] << 24) | ((uint32_t)buf[4*t+1] << 16) |
               ((uint32_t)buf[4*t+2] << 8) | (uint32_t)buf[4*t+3];
    slh_sha256_compress(st, W);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[4*i]   = (uint8_t)(st[i] >> 24);
        out[4*i+1] = (uint8_t)(st[i] >> 16);
        out[4*i+2] = (uint8_t)(st[i] >> 8);
        out[4*i+3] = (uint8_t)st[i];
    }
}

// HMAC-SHA-256 with a 16-byte key (SK.prf; n = 16 at 128s) over
// opt_rand[16] || msg[msg_len]. PRF_msg per FIPS 205 §11.2.1.
__device__ inline
void slh_hmac_sha256_prfmsg(const uint8_t sk_prf[16],
                            const uint8_t opt_rand[16],
                            const uint8_t* msg, uint32_t msg_len,
                            uint8_t out[32]) {
    uint8_t ipad[64], opad[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) { ipad[i] = 0x36; opad[i] = 0x5c; }
    #pragma unroll
    for (int i = 0; i < 16; ++i) { ipad[i] ^= sk_prf[i]; opad[i] ^= sk_prf[i]; }
    uint8_t inner[32];
    slh_sha256_segs(ipad, 64, opt_rand, 16, msg, msg_len, inner);
    slh_sha256_segs(opad, 64, inner, 32, nullptr, 0, out);
}

// Device prologue: build the SignCase fields (r, md, idx_tree, idx_leaf) from
// keyslot material + raw message. Byte-exact mirror of the P1 harness
// `prologue()` (which the differential gates proved against the oracle).
//   opt_rand = addrnd (hedged) or pk_seed (deterministic, FIPS 205 Alg 19).
// digest = MGF1-SHA-256(R || pk_seed || SHA-256(R || pk_seed || pk_root || M),
// 30); at 128s the 30 requested bytes fit ONE MGF1 block (counter 0).
__device__ inline
void slh_sign_prologue_device(const uint8_t sk_prf[16],
                              const uint8_t pk_seed[16],
                              const uint8_t pk_root[16],
                              const uint8_t opt_rand[16],
                              const uint8_t* msg, uint32_t msg_len,
                              SignCase& out) {
    // R = HMAC-SHA256(sk_prf, opt_rand || M)[:16]
    uint8_t rfull[32];
    slh_hmac_sha256_prfmsg(sk_prf, opt_rand, msg, msg_len, rfull);
    #pragma unroll
    for (int i = 0; i < 16; ++i) out.r[i] = rfull[i];

    // inner = SHA256(R || pk_seed || pk_root || M)
    uint8_t head[48];
    #pragma unroll
    for (int i = 0; i < 16; ++i) head[i]      = rfull[i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) head[16 + i] = pk_seed[i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) head[32 + i] = pk_root[i];
    uint8_t inner[32];
    slh_sha256_segs(head, 48, msg, msg_len, nullptr, 0, inner);

    // digest[0:30] = SHA256(R || pk_seed || inner || ctr=0)[:30]  (MGF1, 1 blk)
    uint8_t seed[68];
    #pragma unroll
    for (int i = 0; i < 16; ++i) seed[i]      = rfull[i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) seed[16 + i] = pk_seed[i];
    #pragma unroll
    for (int i = 0; i < 32; ++i) seed[32 + i] = inner[i];
    seed[64] = 0; seed[65] = 0; seed[66] = 0; seed[67] = 0;   // counter 0 BE
    uint8_t digest[32];
    slh_sha256_segs(seed, 68, nullptr, 0, nullptr, 0, digest);
    static_assert(MD_BYTES + 7 + 2 <= 32, "128s digest split must fit one MGF1 block");

    #pragma unroll
    for (int i = 0; i < 24; ++i) out.md[i] = 0;
    #pragma unroll
    for (int i = 0; i < MD_BYTES; ++i) out.md[i] = digest[i];
    // idx_tree = toInt(digest[21:28]) mod 2^(h-h') — 56-bit BE & (2^54 - 1)
    uint64_t it = 0;
    #pragma unroll
    for (int i = 0; i < 7; ++i) it = (it << 8) | digest[MD_BYTES + i];
    out.idx_tree = it & ((1ULL << (H_TOTAL - HP)) - 1);
    // idx_leaf = toInt(digest[28:30]) mod 2^h'
    out.idx_leaf = (uint32_t)(((digest[28] << 8) | digest[29]) & (XMSS_LEAVES - 1));
    out._pad = 0;
}

#endif  // __CUDACC__

// Card SLH-002 P2: per-request descriptor for the keyslot (engine) entry.
// One key per launch (the engine batches per keyslot); messages live in a
// concatenated device buffer.
struct KeyslotSignReq {
    uint32_t msg_offset;   // offset into d_msgs
    uint32_t msg_len;      // <= 65535
    uint8_t  addrnd[16];   // per-request additional randomness (hedged mode)
    uint8_t  hedged;       // 1 = opt_rand = addrnd; 0 = opt_rand = pk_seed
    uint8_t  _pad[3];
};
static_assert(sizeof(KeyslotSignReq) == 28, "KeyslotSignReq ABI: 28 bytes");

}  // namespace slh128s

// Kernel + launcher (defined in slh_sha2_128s_sign.cu)
extern "C" cudaError_t slh_sha2_128s_sign_launch(
    const slh128s::SignCase* d_cases, int n_cases,
    uint8_t* d_sigs,        // n_cases * 7856 bytes
    uint8_t* d_scratch,     // n_cases * SCRATCH_BYTES_PER_WARP
    cudaStream_t stream);

// Card SLH-002 P2: keyslot entry — device prologue from raw messages.
// d_key is a 128-byte block: key[64] (sk_seed||sk_prf||pk_seed||pk_root)
// followed by mirror[64] (redundant copy). Every warp verifies key == mirror
// before signing; on mismatch it writes status = 1 for its case and produces
// NO signature bytes (fail-closed). d_status[i] = 0 on success.
extern "C" cudaError_t slh_sha2_128s_sign_keyslot_launch(
    const uint8_t* d_key,                       // 128 B: key || mirror
    const slh128s::KeyslotSignReq* d_reqs, int n_cases,
    const uint8_t* d_msgs,                      // concatenated messages
    uint8_t* d_sigs,                            // n_cases * 7856 bytes
    uint8_t* d_status,                          // n_cases bytes
    uint8_t* d_scratch,                         // n_cases * SCRATCH_BYTES_PER_WARP
    cudaStream_t stream);

#endif  // SLH_SHA2_128S_SIGN_CUH
