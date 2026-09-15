#pragma once

// Card 26.32: SHA3-256 Device Implementation for Persistent Engine
// Single-block SHA3-256 (Keccak) for input_len in [0, 32] bytes
// Output is always 32 bytes digest
//
// SHA3-256 parameters:
// - State: 1600 bits (5x5 array of 64-bit lanes)
// - Rate: 1088 bits (136 bytes)
// - Capacity: 512 bits (64 bytes)
// - Output: 256 bits (32 bytes)
// - Rounds: 24 Keccak-f[1600] rounds

#include <stdint.h>

namespace smoke {
namespace hash {

// Keccak round constants (FIPS 202)
__device__ __constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation offsets for rho step
__device__ __constant__ int KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

// Pi step permutation indices
__device__ __constant__ int KECCAK_PI[25] = {
     0, 10, 20,  5, 15,
    16,  1, 11, 21,  6,
     7, 17,  2, 12, 22,
    23,  8, 18,  3, 13,
    14, 24,  9, 19,  4
};

// Rotate left 64-bit
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] permutation (24 rounds)
__device__ void keccak_f1600(uint64_t state[25]) {
    uint64_t C[5], D[5], B[25];

    for (int round = 0; round < 24; round++) {
        // Theta step
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        #pragma unroll
        for (int i = 0; i < 25; i++) {
            state[i] ^= D[i % 5];
        }

        // Rho and Pi steps combined
        #pragma unroll
        for (int i = 0; i < 25; i++) {
            B[KECCAK_PI[i]] = rotl64(state[i], KECCAK_RHO[i]);
        }

        // Chi step
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            int base = y * 5;
            #pragma unroll
            for (int x = 0; x < 5; x++) {
                state[base + x] = B[base + x] ^ ((~B[base + (x + 1) % 5]) & B[base + (x + 2) % 5]);
            }
        }

        // Iota step
        state[0] ^= KECCAK_RC[round];
    }
}

// Load 8 bytes (little-endian) to uint64_t
__device__ __forceinline__ uint64_t le64_load(const uint8_t* p) {
    return ((uint64_t)p[0])       | ((uint64_t)p[1] << 8)  |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

// Store uint64_t as 8 bytes (little-endian)
__device__ __forceinline__ void le64_store(uint8_t* p, uint64_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32);
    p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48);
    p[7] = (uint8_t)(v >> 56);
}

// SHA3-256 for short inputs (0-32 bytes)
// For SHA3-256: rate = 136 bytes, output = 32 bytes
// Padding: input || 0x06 || 0x00...0x00 || 0x80 (domain sep + pad10*1)
//
// Since input_len <= 32 < 136, everything fits in one block.
//
// Parameters:
//   input     - pointer to input bytes (must be valid for input_len bytes)
//   input_len - number of input bytes, MUST be in [0, 32]
//   output    - pointer to 32-byte output buffer
//
__device__ void sha3_256_hash_short(
    const uint8_t* input,
    uint16_t input_len,
    uint8_t output[32]
) {
    // Initialize state to zero
    uint64_t state[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) {
        state[i] = 0;
    }

    // Build padded block (rate = 136 bytes for SHA3-256)
    // We need to XOR: input || 0x06 || 0x00...0x00 || 0x80
    // where 0x06 is SHA3 domain separator and 0x80 is at byte 135 (last byte of rate)
    uint8_t block[136];

    // Zero the block
    #pragma unroll
    for (int i = 0; i < 136; i++) {
        block[i] = 0;
    }

    // Copy input
    for (uint16_t i = 0; i < input_len; i++) {
        block[i] = input[i];
    }

    // SHA3 domain separator (0x06) immediately after input
    block[input_len] = 0x06;

    // pad10*1: set last bit of rate block
    block[135] |= 0x80;

    // XOR block into state (17 lanes = 136 bytes / 8)
    #pragma unroll
    for (int i = 0; i < 17; i++) {
        state[i] ^= le64_load(&block[i * 8]);
    }

    // Apply Keccak-f[1600]
    keccak_f1600(state);

    // Squeeze: extract first 32 bytes (4 lanes)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        le64_store(&output[i * 8], state[i]);
    }
}

} // namespace hash
} // namespace smoke
