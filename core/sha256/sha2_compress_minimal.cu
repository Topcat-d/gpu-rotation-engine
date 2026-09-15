// SPDX-License-Identifier: Apache-2.0
// Smoke - Minimal SHA-2 compression kernel (no padding, no message schedule on stack)
// Host provides fully-padded blocks; device just compresses

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// SHA-256 constants
__constant__ uint32_t K256[64] = {
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

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

// SHA-256 initial state (IV)
extern "C" __device__ void sha256_init_state(uint32_t H[8]) {
    H[0] = 0x6a09e667;
    H[1] = 0xbb67ae85;
    H[2] = 0x3c6ef372;
    H[3] = 0xa54ff53a;
    H[4] = 0x510e527f;
    H[5] = 0x9b05688c;
    H[6] = 0x1f83d9ab;
    H[7] = 0x5be0cd19;
}

// Store SHA-256 state as big-endian digest
extern "C" __device__ void sha256_store_digest(const uint32_t H[8], uint8_t out32[32]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t w = H[i];
        out32[i*4 + 0] = (w >> 24) & 0xFF;
        out32[i*4 + 1] = (w >> 16) & 0xFF;
        out32[i*4 + 2] = (w >> 8)  & 0xFF;
        out32[i*4 + 3] = (w >> 0)  & 0xFF;
    }
}

// Core SHA-256 compression function (takes uint32 words directly)
__device__ void sha256_compress_block_core(uint32_t state[8], const uint32_t block[16]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    uint32_t W[64];

    // Load + expand message schedule in registers
    #pragma unroll
    for (int t = 0; t < 16; ++t) W[t] = block[t];

    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        uint32_t s0 = rotr32(W[t-15], 7) ^ rotr32(W[t-15], 18) ^ (W[t-15] >> 3);
        uint32_t s1 = rotr32(W[t-2], 17) ^ rotr32(W[t-2], 19) ^ (W[t-2] >> 10);
        W[t] = W[t-16] + s0 + W[t-7] + s1;
    }

    // 64 rounds
    #pragma unroll 8
    for (int t = 0; t < 64; ++t) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K256[t] + W[t];

        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Byte-based wrapper for fast HKDF kernel (converts bytes to words)
extern "C" __device__ void sha256_compress_block(uint32_t state[8], const uint8_t* block_bytes) {
    // Convert bytes to big-endian uint32 words
    uint32_t block[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t* p = block_bytes + i*4;
        block[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                   ((uint32_t)p[2] << 8)  | ((uint32_t)p[3]);
    }
    sha256_compress_block_core(state, block);
}

// Kernel: compress N blocks sequentially for one message
// Each thread processes one message's worth of blocks
extern "C" __global__
void sha256_compress_sequential(
    const uint32_t* __restrict__ blocks_in,  // [batch * num_blocks * 16] big-endian
    uint32_t* __restrict__ state_out,        // [batch * 8] final state
    const int32_t* __restrict__ num_blocks,  // [batch] blocks per message
    const int32_t* __restrict__ block_off,   // [batch] offset to first block
    int32_t batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    // Initial state (SHA-256 IV)
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    int n_blks = num_blocks[idx];
    int offset = block_off[idx];

    // Compress each block
    for (int b = 0; b < n_blks; ++b) {
        const uint32_t* blk = blocks_in + (offset + b) * 16;
        uint32_t block_local[16];

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block_local[i] = blk[i];
        }

        sha256_compress_block_core(state, block_local);
    }

    // Write final state
    uint32_t* out = state_out + idx * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = state[i];
    }
}
