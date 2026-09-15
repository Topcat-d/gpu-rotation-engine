/*
 * SHA-3-256 CUDA Fused Kernel (Keccak-f[1600] permutation)
 *
 * Single kernel processes entire 24-round Keccak permutation.
 * Expected: 8-10M ops/sec on RTX 2060 SUPER
 *
 * Keccak-f[1600] state: 5x5 matrix of 64-bit lanes (1600 bits total)
 * Rate (r) = 136 bytes for SHA-3-256
 * Capacity (c) = 64 bytes
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Keccak round constants (24 rounds)
__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation offsets for rho step (ROW-MAJOR layout to match PyTorch!)
// PyTorch state[x, y] is stored at memory position x*5 + y (ROW-MAJOR)
// Indexed as rho_offsets[x][y] -> stored linearly as x*5 + y
__constant__ int RHO_OFFSETS[25] = {
    0,  36, 3,  41, 18,   // x=0, y=0..4
    1,  44, 10, 45, 2,    // x=1, y=0..4
    62, 6,  43, 15, 61,   // x=2, y=0..4
    28, 55, 25, 21, 56,   // x=3, y=0..4
    27, 20, 39, 8,  14    // x=4, y=0..4
};

// Permutation indices for pi step (ROW-MAJOR layout to match PyTorch!)
// Pi transformation: state'[y, (2*x + 3*y) % 5] = state[x, y]
// For ROW-MAJOR: input at j = x*5 + y, output at x'*5 + y' where x'=y, y'=(2x+3y)%5
// Precomputed for all 25 positions
__constant__ int PI_INDICES[25] = {
    0,  8,  11, 19, 22,   // x=0, y=0..4 -> new positions
    2,  5,  13, 16, 24,   // x=1, y=0..4 -> new positions
    4,  7,  10, 18, 21,   // x=2, y=0..4 -> new positions
    1,  9,  12, 15, 23,   // x=3, y=0..4 -> new positions
    3,  6,  14, 17, 20    // x=4, y=0..4 -> new positions
};

// Rotate left for 64-bit
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

/*
 * Keccak-f[1600] Permutation Kernel
 *
 * Each thread processes one 1600-bit state through 24 rounds:
 * - State represented as 5x5 matrix of 64-bit lanes (25 words)
 * - 5 steps per round: theta, rho, pi, chi, iota
 * - All operations in registers for maximum performance
 */
__global__ void keccak_f1600_kernel(
    uint64_t* __restrict__ state,   // [B, 25] int64
    int B
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    // Load state into registers (5x5 matrix = 25 lanes)
    uint64_t A[25];
    #pragma unroll
    for (int j = 0; j < 25; ++j) {
        A[j] = state[i*25 + j];
    }

    // 24 rounds of Keccak-f[1600]
    #pragma unroll 4
    for (int round = 0; round < 24; ++round) {
        uint64_t C[5], D[5], B[25];

        // Theta step
        // NOTE: PyTorch uses ROW-MAJOR indexing: state[x,y] -> x*5 + y
        // C[x] = state[x,0] ^ state[x,1] ^ state[x,2] ^ state[x,3] ^ state[x,4]
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            C[x] = A[x*5] ^ A[x*5+1] ^ A[x*5+2] ^ A[x*5+3] ^ A[x*5+4];
        }

        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x+4) % 5] ^ rotl64(C[(x+1) % 5], 1);
        }

        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            #pragma unroll
            for (int y = 0; y < 5; ++y) {
                A[x*5 + y] ^= D[x];  // ROW-MAJOR: x*5 + y (match PyTorch!)
            }
        }

        // Rho and Pi steps (combined)
        #pragma unroll
        for (int j = 0; j < 25; ++j) {
            B[PI_INDICES[j]] = rotl64(A[j], RHO_OFFSETS[j]);
        }

        // Chi step
        // ROW-MAJOR: state[x,y] -> x*5 + y
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            #pragma unroll
            for (int y = 0; y < 5; ++y) {
                int idx = x*5 + y;  // ROW-MAJOR to match PyTorch!
                A[idx] = B[idx] ^ ((~B[((x+1)%5)*5 + y]) & B[((x+2)%5)*5 + y]);
            }
        }

        // Iota step
        A[0] ^= RC[round];
    }

    // Store state back to global memory
    #pragma unroll
    for (int j = 0; j < 25; ++j) {
        state[i*25 + j] = A[j];
    }
}

/*
 * Host launcher function
 */
extern "C" void launch_keccak_f1600(
    uint64_t* state,
    int B,
    cudaStream_t stream
) {
    // Optimal configuration for RTX GPUs
    const int threads = 256;
    const int blocks = (B + threads - 1) / threads;

    keccak_f1600_kernel<<<blocks, threads, 0, stream>>>(state, B);
}
