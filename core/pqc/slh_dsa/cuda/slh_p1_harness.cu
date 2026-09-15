// SPDX-License-Identifier: Apache-2.0
// Card SLH-002 P1 harness: batch-sign SLH-DSA-SHA2-128s cases on GPU.
//
// Modes:
//   slh_p1_harness sign  <cases.bin> <sigs.bin>   differential/KAT signing
//   slh_p1_harness sign-keyslot <cases.bin> <sigs.bin>
//                                                 Card SLH-002 P2: DEVICE
//                                                 prologue path (keyslot
//                                                 kernel; all cases must
//                                                 share one key)
//   slh_p1_harness bench <batch> [reps]           cuda-event-timed batches
//   slh_p1_harness prim                           chained SHA-256 primitive
//                                                 rate (SLH-001 pattern) with
//                                                 THIS header's compress
//
// Host-side crypto here is TEST SCAFFOLDING (the *_ref/oracle tier): a
// self-contained SHA-256/HMAC/MGF1 used only to compute R = PRF_msg and the
// H_msg digest split (2 hashes of ~2.2M per signature). The assembled
// signature is still byte-compared against the full Python oracle output.
//
// cases.bin format (little-endian):
//   magic "SLH1", u32 count, then per case:
//     sk_seed[16] sk_prf[16] pk_seed[16] pk_root[16] addrnd[16]
//     u8 hedged, u8 pad[3], u32 msg_len, u8 msg[msg_len]
// sigs.bin: count * 7856 raw signature bytes.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "slh_sha2_128s_sign.cuh"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "FATAL CUDA error %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err__));                    \
            exit(2);                                                         \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Host SHA-256 (scaffolding only — never called from GPU paths)
// ---------------------------------------------------------------------------
namespace hostsha {

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

struct Sha256 {
    uint32_t h[8];
    uint8_t buf[64];
    uint64_t total = 0;
    size_t fill = 0;
    Sha256() {
        static const uint32_t init[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        memcpy(h, init, sizeof(h));
    }
    void compress(const uint8_t* p) {
        uint32_t W[64];
        for (int i = 0; i < 16; ++i)
            W[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) |
                   ((uint32_t)p[4*i+2] << 8) | p[4*i+3];
        for (int t = 16; t < 64; ++t) {
            uint32_t s0 = rotr(W[t-15], 7) ^ rotr(W[t-15], 18) ^ (W[t-15] >> 3);
            uint32_t s1 = rotr(W[t-2], 17) ^ rotr(W[t-2], 19) ^ (W[t-2] >> 10);
            W[t] = W[t-16] + s0 + W[t-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int t = 0; t < 64; ++t) {
            uint32_t S1 = rotr(e,6)^rotr(e,11)^rotr(e,25);
            uint32_t ch = (e&f)^((~e)&g);
            uint32_t t1 = hh + S1 + ch + K[t] + W[t];
            uint32_t S0 = rotr(a,2)^rotr(a,13)^rotr(a,22);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    void update(const uint8_t* data, size_t len) {
        total += len;
        while (len) {
            size_t take = 64 - fill;
            if (take > len) take = len;
            memcpy(buf + fill, data, take);
            fill += take; data += take; len -= take;
            if (fill == 64) { compress(buf); fill = 0; }
        }
    }
    void final(uint8_t out[32]) {
        uint64_t bits = total * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t z = 0;
        while (fill != 56) update(&z, 1);
        uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) lenb[i] = (uint8_t)(bits >> (56 - 8*i));
        update(lenb, 8);
        for (int i = 0; i < 8; ++i) {
            out[4*i]   = (uint8_t)(h[i] >> 24);
            out[4*i+1] = (uint8_t)(h[i] >> 16);
            out[4*i+2] = (uint8_t)(h[i] >> 8);
            out[4*i+3] = (uint8_t)h[i];
        }
    }
};

static void sha256(const uint8_t* d, size_t l, uint8_t out[32]) {
    Sha256 s; s.update(d, l); s.final(out);
}

// HMAC-SHA256 with key_len <= 64 (n=16 here).
static void hmac_sha256(const uint8_t* key, size_t key_len,
                        const uint8_t* msg, size_t msg_len, uint8_t out[32]) {
    if (key_len > 64) { fprintf(stderr, "FATAL: hmac key >64B\n"); exit(2); }
    uint8_t ipad[64], opad[64];
    memset(ipad, 0x36, 64); memset(opad, 0x5c, 64);
    for (size_t i = 0; i < key_len; ++i) { ipad[i] ^= key[i]; opad[i] ^= key[i]; }
    uint8_t inner[32];
    Sha256 si; si.update(ipad, 64); si.update(msg, msg_len); si.final(inner);
    Sha256 so; so.update(opad, 64); so.update(inner, 32); so.final(out);
}

// MGF1-SHA256 (RFC 8017 B.2.1)
static void mgf1(const uint8_t* seed, size_t seed_len, uint8_t* out,
                 size_t out_len) {
    uint32_t counter = 0;
    size_t off = 0;
    while (off < out_len) {
        uint8_t ctr[4] = {(uint8_t)(counter >> 24), (uint8_t)(counter >> 16),
                          (uint8_t)(counter >> 8), (uint8_t)counter};
        uint8_t d[32];
        Sha256 s; s.update(seed, seed_len); s.update(ctr, 4); s.final(d);
        size_t take = out_len - off < 32 ? out_len - off : 32;
        memcpy(out + off, d, take);
        off += take; counter += 1;
    }
}

}  // namespace hostsha

// ---------------------------------------------------------------------------
// Host-side sign prologue: R = PRF_msg, H_msg digest split (Alg 19 lines 1-7)
// ---------------------------------------------------------------------------
struct HostCase {
    uint8_t sk_seed[16], sk_prf[16], pk_seed[16], pk_root[16], addrnd[16];
    uint8_t hedged;
    std::vector<uint8_t> msg;
};

static void prologue(const HostCase& hc, slh128s::SignCase& out) {
    using namespace hostsha;
    memcpy(out.sk_seed, hc.sk_seed, 16);
    memcpy(out.pk_seed, hc.pk_seed, 16);
    const uint8_t* opt_rand = hc.hedged ? hc.addrnd : hc.pk_seed;

    // R = HMAC-SHA256(sk_prf, opt_rand || M)[:16]
    std::vector<uint8_t> pm(16 + hc.msg.size());
    memcpy(pm.data(), opt_rand, 16);
    memcpy(pm.data() + 16, hc.msg.data(), hc.msg.size());
    uint8_t rfull[32];
    hmac_sha256(hc.sk_prf, 16, pm.data(), pm.size(), rfull);
    memcpy(out.r, rfull, 16);

    // digest = MGF1(R || pk_seed || SHA256(R || pk_seed || pk_root || M), 30)
    std::vector<uint8_t> hm(48 + hc.msg.size());
    memcpy(hm.data(), rfull, 16);
    memcpy(hm.data() + 16, hc.pk_seed, 16);
    memcpy(hm.data() + 32, hc.pk_root, 16);
    memcpy(hm.data() + 48, hc.msg.data(), hc.msg.size());
    uint8_t inner[32];
    sha256(hm.data(), hm.size(), inner);
    uint8_t seed[64];
    memcpy(seed, rfull, 16);
    memcpy(seed + 16, hc.pk_seed, 16);
    memcpy(seed + 32, inner, 32);
    uint8_t digest[30];
    mgf1(seed, 64, digest, 30);

    memset(out.md, 0, sizeof(out.md));
    memcpy(out.md, digest, slh128s::MD_BYTES);   // 21 bytes
    // idx_tree = toInt(digest[21:28]) mod 2^(h-h') = 56-bit BE & (2^54 - 1)
    uint64_t it = 0;
    for (int i = 0; i < 7; ++i) it = (it << 8) | digest[21 + i];
    out.idx_tree = it & ((1ULL << 54) - 1);
    // idx_leaf = toInt(digest[28:30]) mod 2^h' = 16-bit BE & 511
    out.idx_leaf = (uint32_t)(((digest[28] << 8) | digest[29]) & 511);
    out._pad = 0;
}

// ---------------------------------------------------------------------------
// GPU batch sign
// ---------------------------------------------------------------------------
static void run_batch(const std::vector<slh128s::SignCase>& cases,
                      std::vector<uint8_t>& sigs_out,
                      float* kernel_ms /*nullable*/) {
    using namespace slh128s;
    const int n = (int)cases.size();
    SignCase* d_cases;
    uint8_t *d_sigs, *d_scratch;
    CUDA_CHECK(cudaMalloc(&d_cases, sizeof(SignCase) * n));
    CUDA_CHECK(cudaMalloc(&d_sigs, (size_t)n * SIG_BYTES));
    CUDA_CHECK(cudaMalloc(&d_scratch, (size_t)n * SCRATCH_BYTES_PER_WARP));
    CUDA_CHECK(cudaMemcpy(d_cases, cases.data(), sizeof(SignCase) * n,
                          cudaMemcpyHostToDevice));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(slh_sha2_128s_sign_launch(d_cases, n, d_sigs, d_scratch, 0));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    if (kernel_ms) *kernel_ms = ms;
    sigs_out.resize((size_t)n * SIG_BYTES);
    CUDA_CHECK(cudaMemcpy(sigs_out.data(), d_sigs, (size_t)n * SIG_BYTES,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cases));
    CUDA_CHECK(cudaFree(d_sigs));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
}

// ---------------------------------------------------------------------------
// sign mode
// ---------------------------------------------------------------------------
static int read_cases_file(const char* in_path, std::vector<HostCase>& out) {
    FILE* f = fopen(in_path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", in_path); return 2; }
    char magic[4];
    uint32_t count;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SLH1", 4) != 0 ||
        fread(&count, 4, 1, f) != 1) {
        fprintf(stderr, "FATAL: bad case file header\n"); fclose(f); return 2;
    }
    out.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        HostCase& hc = out[i];
        uint8_t pad[3];
        uint32_t msg_len;
        if (fread(hc.sk_seed, 1, 16, f) != 16 || fread(hc.sk_prf, 1, 16, f) != 16 ||
            fread(hc.pk_seed, 1, 16, f) != 16 || fread(hc.pk_root, 1, 16, f) != 16 ||
            fread(hc.addrnd, 1, 16, f) != 16 || fread(&hc.hedged, 1, 1, f) != 1 ||
            fread(pad, 1, 3, f) != 3 || fread(&msg_len, 4, 1, f) != 1) {
            fprintf(stderr, "FATAL: truncated case %u\n", i); fclose(f); return 2;
        }
        hc.msg.resize(msg_len);
        if (msg_len && fread(hc.msg.data(), 1, msg_len, f) != msg_len) {
            fprintf(stderr, "FATAL: truncated msg in case %u\n", i); fclose(f); return 2;
        }
    }
    fclose(f);
    return 0;
}

static int mode_sign(const char* in_path, const char* out_path) {
    std::vector<HostCase> host_cases;
    int rc = read_cases_file(in_path, host_cases);
    if (rc) return rc;
    std::vector<slh128s::SignCase> cases(host_cases.size());
    for (size_t i = 0; i < host_cases.size(); ++i)
        prologue(host_cases[i], cases[i]);

    std::vector<uint8_t> sigs;
    float ms;
    run_batch(cases, sigs, &ms);
    fprintf(stderr, "signed %zu cases, kernel %.2f ms\n", cases.size(), ms);

    FILE* o = fopen(out_path, "wb");
    if (!o) { fprintf(stderr, "FATAL: cannot write %s\n", out_path); return 2; }
    if (fwrite(sigs.data(), 1, sigs.size(), o) != sigs.size()) {
        fprintf(stderr, "FATAL: short write\n"); return 2;
    }
    fclose(o);
    return 0;
}

// ---------------------------------------------------------------------------
// sign-keyslot mode (Card SLH-002 P2): DEVICE-prologue path. All cases must
// share one key (the engine batches per keyslot); the harness uploads
// key[64] || mirror[64] and per-request (msg, addrnd, hedged) descriptors and
// runs slh_sha2_128s_sign_keyslot_launch — R/H_msg are computed ON DEVICE.
// The differential script byte-compares against the full Python oracle.
// ---------------------------------------------------------------------------
static int mode_sign_keyslot(const char* in_path, const char* out_path) {
    using namespace slh128s;
    std::vector<HostCase> hcs;
    int rc = read_cases_file(in_path, hcs);
    if (rc) return rc;
    if (hcs.empty()) { fprintf(stderr, "FATAL: no cases\n"); return 2; }

    uint8_t key128[128];
    memcpy(key128 +  0, hcs[0].sk_seed, 16);
    memcpy(key128 + 16, hcs[0].sk_prf, 16);
    memcpy(key128 + 32, hcs[0].pk_seed, 16);
    memcpy(key128 + 48, hcs[0].pk_root, 16);
    memcpy(key128 + 64, key128, 64);   // mirror
    for (size_t i = 1; i < hcs.size(); ++i) {
        if (memcmp(hcs[i].sk_seed, hcs[0].sk_seed, 16) ||
            memcmp(hcs[i].sk_prf, hcs[0].sk_prf, 16) ||
            memcmp(hcs[i].pk_seed, hcs[0].pk_seed, 16) ||
            memcmp(hcs[i].pk_root, hcs[0].pk_root, 16)) {
            fprintf(stderr, "FATAL: sign-keyslot requires one shared key; "
                            "case %zu differs\n", i);
            return 2;
        }
    }

    const int n = (int)hcs.size();
    std::vector<KeyslotSignReq> reqs(n);
    std::vector<uint8_t> msgs;
    for (int i = 0; i < n; ++i) {
        reqs[i].msg_offset = (uint32_t)msgs.size();
        reqs[i].msg_len = (uint32_t)hcs[i].msg.size();
        memcpy(reqs[i].addrnd, hcs[i].addrnd, 16);
        reqs[i].hedged = hcs[i].hedged;
        memset(reqs[i]._pad, 0, sizeof(reqs[i]._pad));
        msgs.insert(msgs.end(), hcs[i].msg.begin(), hcs[i].msg.end());
    }
    if (msgs.empty()) msgs.push_back(0);   // keep cudaMalloc(0) out of the path

    uint8_t *d_key, *d_msgs, *d_sigs, *d_status, *d_scratch;
    KeyslotSignReq* d_reqs;
    CUDA_CHECK(cudaMalloc(&d_key, 128));
    CUDA_CHECK(cudaMalloc(&d_reqs, sizeof(KeyslotSignReq) * n));
    CUDA_CHECK(cudaMalloc(&d_msgs, msgs.size()));
    CUDA_CHECK(cudaMalloc(&d_sigs, (size_t)n * SIG_BYTES));
    CUDA_CHECK(cudaMalloc(&d_status, n));
    CUDA_CHECK(cudaMalloc(&d_scratch, (size_t)n * SCRATCH_BYTES_PER_WARP));
    CUDA_CHECK(cudaMemcpy(d_key, key128, 128, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reqs, reqs.data(), sizeof(KeyslotSignReq) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_msgs, msgs.data(), msgs.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(slh_sha2_128s_sign_keyslot_launch(d_key, d_reqs, n, d_msgs,
                                                 d_sigs, d_status, d_scratch,
                                                 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> status(n);
    CUDA_CHECK(cudaMemcpy(status.data(), d_status, n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (status[i] != 0) {
            fprintf(stderr, "FATAL: keyslot kernel status[%d]=%u "
                            "(key/mirror fault)\n", i, status[i]);
            return 2;
        }
    }
    std::vector<uint8_t> sigs((size_t)n * SIG_BYTES);
    CUDA_CHECK(cudaMemcpy(sigs.data(), d_sigs, sigs.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_msgs));
    CUDA_CHECK(cudaFree(d_sigs));
    CUDA_CHECK(cudaFree(d_status));
    CUDA_CHECK(cudaFree(d_scratch));

    fprintf(stderr, "signed %d cases (device prologue)\n", n);
    FILE* o = fopen(out_path, "wb");
    if (!o) { fprintf(stderr, "FATAL: cannot write %s\n", out_path); return 2; }
    if (fwrite(sigs.data(), 1, sigs.size(), o) != sigs.size()) {
        fprintf(stderr, "FATAL: short write\n"); return 2;
    }
    fclose(o);
    return 0;
}

// ---------------------------------------------------------------------------
// bench mode — batch of DISTINCT signs; full signatures written to a device
// buffer and copied out (that consumed output is the DCE proof).
// ---------------------------------------------------------------------------
static uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97f4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int mode_bench(int batch, int reps) {
    using namespace slh128s;
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("# Hardware ID: %s, sm_count=%d, cc=%d.%d, CUDA runtime\n",
           p.name, p.multiProcessorCount, p.major, p.minor);
    printf("# Execution shape: 1 signature/warp, 128 threads/block (4 warps),"
           " grid=%d blocks, batch=%d, scratch=%.1f MiB\n",
           (batch + 3) / 4, batch,
           batch * (double)SCRATCH_BYTES_PER_WARP / (1 << 20));

    uint64_t seed = 0x51AB002ULL;
    std::vector<slh128s::SignCase> cases(batch);
    for (int i = 0; i < batch; ++i) {
        HostCase hc;
        hc.msg.resize(32);
        auto fill16 = [&](uint8_t* d) {
            for (int j = 0; j < 2; ++j) {
                uint64_t v = splitmix64(seed);
                memcpy(d + 8 * j, &v, 8);
            }
        };
        fill16(hc.sk_seed); fill16(hc.sk_prf); fill16(hc.pk_seed);
        fill16(hc.pk_root); fill16(hc.addrnd);
        for (int j = 0; j < 4; ++j) {
            uint64_t v = splitmix64(seed);
            memcpy(hc.msg.data() + 8 * j, &v, 8);
        }
        hc.hedged = (uint8_t)(i & 1);
        prologue(hc, cases[i]);
    }

    // warmup
    std::vector<uint8_t> sigs;
    float ms;
    run_batch(cases, sigs, &ms);
    uint64_t chk = 0;
    for (size_t i = 0; i < sigs.size(); i += 257) chk = chk * 31 + sigs[i];
    printf("# warmup kernel_ms=%.2f output_checksum=%016llx\n", ms,
           (unsigned long long)chk);

    const long long comps =
        (COMPRESSIONS_PER_SIG + MIDSTATE_COMPRESSIONS_PER_SIG);
    for (int r = 0; r < reps; ++r) {
        run_batch(cases, sigs, &ms);
        double signs_s = batch / (ms / 1000.0);
        double comp_s = signs_s * (double)comps;
        printf("rep=%d batch=%d kernel_ms=%.2f signs_per_s=%.1f "
               "kernel_compressions_per_sig=%lld achieved_compressions_per_s=%.3e\n",
               r, batch, ms, signs_s, comps, comp_s);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// prim mode — chained, DCE-proof single-block midstate compression rate with
// THIS header's slh_sha256_compress (SLH-001 sha2_slh_bench pattern).
// ---------------------------------------------------------------------------
__global__ void prim_bench_kernel(const uint32_t* __restrict__ in,
                                  uint32_t* __restrict__ out,
                                  int n_msgs, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int m = idx; m < n_msgs; m += stride) {
        uint32_t midstate[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        uint32_t block[16];
        const uint32_t* mp = in + (size_t)m * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) block[i] = mp[i];
        uint32_t chain = 0, acc = 0;
        for (int it = 0; it < iters; ++it) {
            uint32_t st[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) st[i] = midstate[i];
            block[0] ^= chain;
            slh128s::slh_sha256_compress(st, block);
            chain = st[0];
            acc ^= st[0] ^ st[7];
        }
        out[m] = acc;
    }
}

static int mode_prim() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("gpu=%s sm=%d\n", p.name, p.multiProcessorCount);
    int sizes[] = {100000, 1000000, 4000000};
    for (int s = 0; s < 3; ++s) {
        int n = sizes[s], iters = 100;
        uint32_t *din, *dout;
        CUDA_CHECK(cudaMalloc(&din, (size_t)n * 64));
        CUDA_CHECK(cudaMalloc(&dout, (size_t)n * 4));
        CUDA_CHECK(cudaMemset(din, 0x3C, (size_t)n * 64));
        int block = 256, grid = p.multiProcessorCount * 8;
        prim_bench_kernel<<<grid, block>>>(din, dout, n, 2);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        prim_bench_kernel<<<grid, block>>>(din, dout, n, iters);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        CUDA_CHECK(cudaGetLastError());
        printf("batch=%d iters=%d time_ms=%.2f sha256_compressions_per_s=%.3e\n",
               n, iters, ms, (double)n * iters / (ms / 1000.0));
        CUDA_CHECK(cudaFree(din));
        CUDA_CHECK(cudaFree(dout));
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 4 && strcmp(argv[1], "sign") == 0)
        return mode_sign(argv[2], argv[3]);
    if (argc >= 4 && strcmp(argv[1], "sign-keyslot") == 0)
        return mode_sign_keyslot(argv[2], argv[3]);
    if (argc >= 3 && strcmp(argv[1], "bench") == 0)
        return mode_bench(atoi(argv[2]), argc >= 4 ? atoi(argv[3]) : 3);
    if (argc >= 2 && strcmp(argv[1], "prim") == 0)
        return mode_prim();
    fprintf(stderr,
            "usage: %s sign <cases.bin> <sigs.bin> | "
            "sign-keyslot <cases.bin> <sigs.bin> | bench <batch> [reps] | prim\n",
            argv[0]);
    return 2;
}
