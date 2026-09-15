/*
 * SHA-256 CUDA Fused Kernel
 *
 * Single kernel processes entire 64-round compression with W[64] in registers.
 * Expected: 10-13M ops/sec on RTX 2060 SUPER
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// SHA-256 constants (K[64])
__constant__ uint32_t K[64] = {
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

// Rotate right using funnel shift (native GPU instruction)
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

/*
 * SHA-256 Compression Kernel
 *
 * Each thread processes one message block:
 * - Loads state (a..h) and block (16 words)
 * - Expands W[0..15] to W[0..63] in registers
 * - Executes 64 compression rounds
 * - Stores updated state
 *
 * All operations in registers - minimal global memory traffic.
 */
__global__ void sha256_compress_kernel(
    const uint32_t* __restrict__ state_in,   // [B, 8] int32
    const uint32_t* __restrict__ blocks,     // [B, 16] int32
    uint32_t* __restrict__ state_out,        // [B, 8] int32
    int B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    // Load initial state (8 words)
    uint32_t a = state_in[i*8 + 0];
    uint32_t b = state_in[i*8 + 1];
    uint32_t c = state_in[i*8 + 2];
    uint32_t d = state_in[i*8 + 3];
    uint32_t e = state_in[i*8 + 4];
    uint32_t f = state_in[i*8 + 5];
    uint32_t g = state_in[i*8 + 6];
    uint32_t h = state_in[i*8 + 7];

    // Save original state for final addition
    uint32_t a0 = a, b0 = b, c0 = c, d0 = d;
    uint32_t e0 = e, f0 = f, g0 = g, h0 = h;

    // Message schedule W[64] - kept in registers!
    uint32_t W[64];

    // Load first 16 words from message block
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        W[t] = blocks[i*16 + t];
    }

    // Expand to 64 words
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        uint32_t s0 = rotr(W[t-15], 7) ^ rotr(W[t-15], 18) ^ (W[t-15] >> 3);
        uint32_t s1 = rotr(W[t-2], 17) ^ rotr(W[t-2], 19) ^ (W[t-2] >> 10);
        W[t] = W[t-16] + s0 + W[t-7] + s1;
    }

    // 64 compression rounds
    // Partial unroll (8 rounds) to balance registers vs instruction cache
    #pragma unroll 8
    for (int t = 0; t < 64; ++t) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K[t] + W[t];

        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        // Rotate working variables
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    // Add to original state
    state_out[i*8 + 0] = a0 + a;
    state_out[i*8 + 1] = b0 + b;
    state_out[i*8 + 2] = c0 + c;
    state_out[i*8 + 3] = d0 + d;
    state_out[i*8 + 4] = e0 + e;
    state_out[i*8 + 5] = f0 + f;
    state_out[i*8 + 6] = g0 + g;
    state_out[i*8 + 7] = h0 + h;
}

/*
 * Host launcher function
 *
 * Chooses optimal thread/block configuration and launches kernel.
 */
extern "C" void launch_sha256_compress(
    const uint32_t* state_in,
    const uint32_t* blocks,
    uint32_t* state_out,
    int B,
    cudaStream_t stream
) {
    // Optimal configuration for RTX GPUs
    const int threads = 256;
    const int blocks_grid = (B + threads - 1) / threads;

    sha256_compress_kernel<<<blocks_grid, threads, 0, stream>>>(
        state_in, blocks, state_out, B
    );
}
