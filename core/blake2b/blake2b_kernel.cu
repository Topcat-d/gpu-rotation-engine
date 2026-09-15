/*
 * BLAKE2b CUDA Fused Kernel
 *
 * Single kernel processes 12-round compression with message schedule in registers.
 * Expected: 35M+ ops/sec on RTX 2060 SUPER (FASTEST hash - ARX operations!)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// BLAKE2b initialization vectors (IV)
__constant__ uint64_t IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// BLAKE2b SIGMA permutations (12 rounds)
__constant__ int SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

// Rotate right for 64-bit
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

/*
 * BLAKE2b G function (mixing function)
 *
 * ARX operations: Add-Rotate-XOR (extremely GPU-friendly!)
 * Rotation constants: R1=32, R2=24, R3=16, R4=63
 */
__device__ __forceinline__ void G(
    uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d,
    uint64_t x, uint64_t y
) {
    a = a + b + x;
    d = rotr64(d ^ a, 32);  // R1 = 32

    c = c + d;
    b = rotr64(b ^ c, 24);  // R2 = 24

    a = a + b + y;
    d = rotr64(d ^ a, 16);  // R3 = 16

    c = c + d;
    b = rotr64(b ^ c, 63);  // R4 = 63
}

/*
 * BLAKE2b Compression Kernel
 *
 * Each thread processes one message block:
 * - Loads state (H0..H7) and block (16 x 64-bit words)
 * - Executes 12 compression rounds with G function
 * - Applies counter and final flag
 * - Stores updated state
 *
 * All operations in registers - minimal global memory traffic.
 */
__global__ void blake2b_compress_kernel(
    const uint64_t* __restrict__ state_in,   // [B, 8] int64
    const uint64_t* __restrict__ blocks,     // [B, 16] int64 (message words)
    const uint64_t* __restrict__ counters,   // [B] int64 (bytes processed)
    const bool* __restrict__ final_flags,    // [B] bool (is final block?)
    uint64_t* __restrict__ state_out,        // [B, 8] int64
    int B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    // Load initial state (8 x 64-bit words)
    uint64_t h[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        h[j] = state_in[i*8 + j];
    }

    // Load message block (16 x 64-bit words)
    uint64_t m[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        m[j] = blocks[i*16 + j];
    }

    // Load counter and final flag
    uint64_t t = counters[i];
    bool f = final_flags[i];

    // Initialize working variables: v[0..7] = h[0..7], v[8..15] = IV[0..7]
    uint64_t v[16];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = h[j];
        v[j+8] = IV[j];
    }

    // Mix in the offset counter (t) and final block flag
    v[12] ^= t;              // Low 64 bits of counter
    v[13] ^= 0;              // High 64 bits (always 0 for reasonable sizes)
    if (f) {
        v[14] ^= 0xFFFFFFFFFFFFFFFFULL;  // Final block flag
    }

    // 12 rounds of mixing
    #pragma unroll
    for (int round = 0; round < 12; ++round) {
        // Column step
        G(v[0], v[4], v[ 8], v[12], m[SIGMA[round][ 0]], m[SIGMA[round][ 1]]);
        G(v[1], v[5], v[ 9], v[13], m[SIGMA[round][ 2]], m[SIGMA[round][ 3]]);
        G(v[2], v[6], v[10], v[14], m[SIGMA[round][ 4]], m[SIGMA[round][ 5]]);
        G(v[3], v[7], v[11], v[15], m[SIGMA[round][ 6]], m[SIGMA[round][ 7]]);

        // Diagonal step
        G(v[0], v[5], v[10], v[15], m[SIGMA[round][ 8]], m[SIGMA[round][ 9]]);
        G(v[1], v[6], v[11], v[12], m[SIGMA[round][10]], m[SIGMA[round][11]]);
        G(v[2], v[7], v[ 8], v[13], m[SIGMA[round][12]], m[SIGMA[round][13]]);
        G(v[3], v[4], v[ 9], v[14], m[SIGMA[round][14]], m[SIGMA[round][15]]);
    }

    // XOR the two halves back into state: h[i] = h[i] ^ v[i] ^ v[i+8]
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        state_out[i*8 + j] = h[j] ^ v[j] ^ v[j+8];
    }
}

/*
 * Host launcher function
 *
 * Chooses optimal thread/block configuration and launches kernel.
 */
extern "C" void launch_blake2b_compress(
    const uint64_t* state_in,
    const uint64_t* blocks,
    const uint64_t* counters,
    const bool* final_flags,
    uint64_t* state_out,
    int B,
    cudaStream_t stream
) {
    // Optimal configuration for RTX GPUs
    const int threads = 256;
    const int blocks_grid = (B + threads - 1) / threads;

    blake2b_compress_kernel<<<blocks_grid, threads, 0, stream>>>(
        state_in, blocks, counters, final_flags, state_out, B
    );
}
