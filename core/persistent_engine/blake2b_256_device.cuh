#pragma once

// Card 26.33: BLAKE2b-256 Device Implementation for Persistent Engine
// Single-block BLAKE2b-256 for input_len in [0, 32] bytes
// Output is always 32 bytes digest
//
// BLAKE2b-256 parameters:
// - State: 8 x 64-bit words (512 bits)
// - Block size: 128 bytes
// - Rounds: 12
// - Output: 256 bits (32 bytes)
// - ARX operations (Add-Rotate-XOR) - extremely GPU-friendly

#include <stdint.h>

namespace smoke {
namespace hash {

// BLAKE2b initialization vectors (IV) - from SHA-512
__device__ __constant__ uint64_t BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// BLAKE2b SIGMA permutations (12 rounds)
__device__ __constant__ uint8_t BLAKE2B_SIGMA[12][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
    {11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4},
    { 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8},
    { 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13},
    { 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9},
    {12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11},
    {13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10},
    { 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5},
    {10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0},
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3}
};

// Rotate right for 64-bit
__device__ __forceinline__ uint64_t blake2b_rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

// BLAKE2b G function (mixing function)
// ARX operations: Add-Rotate-XOR
// Rotation constants: R1=32, R2=24, R3=16, R4=63
__device__ __forceinline__ void blake2b_G(
    uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d,
    uint64_t x, uint64_t y
) {
    a = a + b + x;
    d = blake2b_rotr64(d ^ a, 32);  // R1 = 32

    c = c + d;
    b = blake2b_rotr64(b ^ c, 24);  // R2 = 24

    a = a + b + y;
    d = blake2b_rotr64(d ^ a, 16);  // R3 = 16

    c = c + d;
    b = blake2b_rotr64(b ^ c, 63);  // R4 = 63
}

// BLAKE2b compression function (12 rounds)
__device__ void blake2b_compress(
    uint64_t h[8],          // State (modified in place)
    const uint64_t m[16],   // Message block (16 words = 128 bytes)
    uint64_t t,             // Counter (bytes compressed so far including this block)
    bool f                  // Final block flag
) {
    // Initialize working variables: v[0..7] = h[0..7], v[8..15] = IV[0..7]
    uint64_t v[16];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = h[j];
        v[j+8] = BLAKE2B_IV[j];
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
        blake2b_G(v[0], v[4], v[ 8], v[12], m[BLAKE2B_SIGMA[round][ 0]], m[BLAKE2B_SIGMA[round][ 1]]);
        blake2b_G(v[1], v[5], v[ 9], v[13], m[BLAKE2B_SIGMA[round][ 2]], m[BLAKE2B_SIGMA[round][ 3]]);
        blake2b_G(v[2], v[6], v[10], v[14], m[BLAKE2B_SIGMA[round][ 4]], m[BLAKE2B_SIGMA[round][ 5]]);
        blake2b_G(v[3], v[7], v[11], v[15], m[BLAKE2B_SIGMA[round][ 6]], m[BLAKE2B_SIGMA[round][ 7]]);

        // Diagonal step
        blake2b_G(v[0], v[5], v[10], v[15], m[BLAKE2B_SIGMA[round][ 8]], m[BLAKE2B_SIGMA[round][ 9]]);
        blake2b_G(v[1], v[6], v[11], v[12], m[BLAKE2B_SIGMA[round][10]], m[BLAKE2B_SIGMA[round][11]]);
        blake2b_G(v[2], v[7], v[ 8], v[13], m[BLAKE2B_SIGMA[round][12]], m[BLAKE2B_SIGMA[round][13]]);
        blake2b_G(v[3], v[4], v[ 9], v[14], m[BLAKE2B_SIGMA[round][14]], m[BLAKE2B_SIGMA[round][15]]);
    }

    // XOR the two halves back into state: h[i] = h[i] ^ v[i] ^ v[i+8]
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        h[j] = h[j] ^ v[j] ^ v[j+8];
    }
}

// Load 8 bytes (little-endian) to uint64_t
__device__ __forceinline__ uint64_t blake2b_le64_load(const uint8_t* p) {
    return ((uint64_t)p[0])       | ((uint64_t)p[1] << 8)  |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

// Store uint64_t as 8 bytes (little-endian)
__device__ __forceinline__ void blake2b_le64_store(uint8_t* p, uint64_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32);
    p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48);
    p[7] = (uint8_t)(v >> 56);
}

// BLAKE2b-256 for short inputs (0-32 bytes)
// For BLAKE2b-256: block_size = 128 bytes, output = 32 bytes
//
// Parameters:
//   input     - pointer to input bytes (must be valid for input_len bytes)
//   input_len - number of input bytes, MUST be in [0, 32]
//   output    - pointer to 32-byte output buffer
//
__device__ void blake2b_256_hash_short(
    const uint8_t* input,
    uint16_t input_len,
    uint8_t output[32]
) {
    // Initialize state with IV
    uint64_t h[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        h[i] = BLAKE2B_IV[i];
    }

    // BLAKE2b parameter block XOR into h[0]:
    // h[0] ^= 0x01010000 ^ (kk << 8) ^ nn
    // For unkeyed BLAKE2b-256: kk=0 (no key), nn=32 (output length)
    // So: h[0] ^= 0x01010000 ^ 0 ^ 32 = 0x01010020
    h[0] ^= 0x01010020ULL;

    // Build message block (128 bytes, zero-padded)
    uint8_t block[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        block[i] = 0;
    }

    // Copy input
    for (uint16_t i = 0; i < input_len; i++) {
        block[i] = input[i];
    }

    // Load block into 16 x 64-bit words
    uint64_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        m[i] = blake2b_le64_load(&block[i * 8]);
    }

    // Compress with t=input_len (bytes in this block) and f=true (final block)
    blake2b_compress(h, m, (uint64_t)input_len, true);

    // Extract first 32 bytes (4 x 64-bit words) as output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        blake2b_le64_store(&output[i * 8], h[i]);
    }
}

} // namespace hash
} // namespace smoke
