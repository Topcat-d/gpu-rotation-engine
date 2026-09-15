/*
 * SHA-512 CUDA Fused Kernel
 *
 * Single kernel processes entire 80-round compression with W[80] in registers.
 * Expected: 8-12M ops/sec on RTX 2060 SUPER (slower than SHA-256 due to 64-bit ops)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// SHA-512 constants (K[80])
__constant__ uint64_t K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

// Rotate right for 64-bit
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

/*
 * SHA-512 Compression Kernel
 *
 * Each thread processes one message block:
 * - Loads state (H0..H7) and block (16 x 64-bit words)
 * - Expands W[0..15] to W[0..79] in registers
 * - Executes 80 compression rounds
 * - Stores updated state
 *
 * All operations in registers - minimal global memory traffic.
 */
__global__ void sha512_compress_kernel(
    const uint64_t* __restrict__ state_in,   // [B, 8] int64
    const uint64_t* __restrict__ blocks,     // [B, 16] int64
    uint64_t* __restrict__ state_out,        // [B, 8] int64
    int B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    // Load initial state (8 x 64-bit words)
    uint64_t a = state_in[i*8 + 0];
    uint64_t b = state_in[i*8 + 1];
    uint64_t c = state_in[i*8 + 2];
    uint64_t d = state_in[i*8 + 3];
    uint64_t e = state_in[i*8 + 4];
    uint64_t f = state_in[i*8 + 5];
    uint64_t g = state_in[i*8 + 6];
    uint64_t h = state_in[i*8 + 7];

    // Save original state for final addition
    uint64_t a0 = a, b0 = b, c0 = c, d0 = d;
    uint64_t e0 = e, f0 = f, g0 = g, h0 = h;

    // Message schedule W[80] - kept in registers!
    uint64_t W[80];

    // Load first 16 words from message block
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        W[t] = blocks[i*16 + t];
    }

    // Expand to 80 words
    #pragma unroll
    for (int t = 16; t < 80; ++t) {
        uint64_t s0 = rotr64(W[t-15], 1) ^ rotr64(W[t-15], 8) ^ (W[t-15] >> 7);
        uint64_t s1 = rotr64(W[t-2], 19) ^ rotr64(W[t-2], 61) ^ (W[t-2] >> 6);
        W[t] = W[t-16] + s0 + W[t-7] + s1;
    }

    // 80 compression rounds
    // Partial unroll (8 rounds) to balance registers vs instruction cache
    #pragma unroll 8
    for (int t = 0; t < 80; ++t) {
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ ((~e) & g);
        uint64_t temp1 = h + S1 + ch + K[t] + W[t];

        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t temp2 = S0 + maj;

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
extern "C" void launch_sha512_compress(
    const uint64_t* state_in,
    const uint64_t* blocks,
    uint64_t* state_out,
    int B,
    cudaStream_t stream
) {
    // Optimal configuration for RTX GPUs
    const int threads = 256;
    const int blocks_grid = (B + threads - 1) / threads;

    sha512_compress_kernel<<<blocks_grid, threads, 0, stream>>>(
        state_in, blocks, state_out, B
    );
}
