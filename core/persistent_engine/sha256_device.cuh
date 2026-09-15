#pragma once

// Card 26.19: SHA-256 Device Implementation for Persistent Engine
// Single-block SHA-256 for input_len ∈ [0, 32] bytes
// Output is always 32 bytes digest

#include <stdint.h>

namespace smoke {
namespace hash {

// SHA-256 round constants (FIPS 180-4 Section 4.2.2)
__device__ __constant__ uint32_t K256[64] = {
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

// SHA-256 initial hash values (FIPS 180-4 Section 5.3.3)
__device__ __constant__ uint32_t SHA256_IV[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Rotate right
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 functions (FIPS 180-4 Section 4.1.2)
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t Sigma0(uint32_t x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__device__ __forceinline__ uint32_t Sigma1(uint32_t x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// Convert 4 bytes (big-endian) to uint32_t
__device__ __forceinline__ uint32_t be32_load(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | ((uint32_t)p[3]);
}

// Store uint32_t as 4 bytes (big-endian)
__device__ __forceinline__ void be32_store(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v);
}

// SHA-256 compression function for a single 64-byte block
// H is updated in place (8 x uint32_t)
__device__ inline void sha256_compress_block(uint32_t H[8], const uint8_t block[64]) {
    uint32_t W[64];

    // Message schedule: first 16 words from block (big-endian)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = be32_load(&block[i * 4]);
    }

    // Message schedule: remaining 48 words
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
    }

    // Working variables
    uint32_t a = H[0];
    uint32_t b = H[1];
    uint32_t c = H[2];
    uint32_t d = H[3];
    uint32_t e = H[4];
    uint32_t f = H[5];
    uint32_t g = H[6];
    uint32_t h = H[7];

    // 64 rounds
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + K256[i] + W[i];
        uint32_t T2 = Sigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // Add compressed chunk to current hash value
    H[0] += a;
    H[1] += b;
    H[2] += c;
    H[3] += d;
    H[4] += e;
    H[5] += f;
    H[6] += g;
    H[7] += h;
}

// SHA-256 for short inputs (0-32 bytes)
// This produces exactly one 64-byte block after padding:
//   [input][0x80][zeros...][length_bits_be64]
//   where length_bits = input_len * 8, stored at bytes 56-63
//
// For input_len ∈ [0, 55], one block suffices (length at 56-63)
// For input_len ∈ [56, 63], would need two blocks - but we limit to 32.
//
// Parameters:
//   input     - pointer to input bytes (must be valid for input_len bytes)
//   input_len - number of input bytes, MUST be in [0, 32]
//   output    - pointer to 32-byte output buffer
//
__device__ inline void sha256_hash_short(
    const uint8_t* input,
    uint16_t input_len,
    uint8_t output[32]
) {
    // Build padded block (64 bytes)
    uint8_t block[64];

    // Copy input
    for (uint16_t i = 0; i < input_len; i++) {
        block[i] = input[i];
    }

    // Padding: 0x80 followed by zeros
    block[input_len] = 0x80;
    for (uint16_t i = input_len + 1; i < 56; i++) {
        block[i] = 0;
    }

    // Length in bits (big-endian, 64-bit) at bytes 56-63
    // input_len <= 32, so length_bits = input_len * 8 <= 256, fits in low bytes
    uint64_t length_bits = (uint64_t)input_len * 8;
    block[56] = 0;
    block[57] = 0;
    block[58] = 0;
    block[59] = 0;
    block[60] = (uint8_t)(length_bits >> 24);
    block[61] = (uint8_t)(length_bits >> 16);
    block[62] = (uint8_t)(length_bits >> 8);
    block[63] = (uint8_t)(length_bits);

    // Initialize hash state
    uint32_t H[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        H[i] = SHA256_IV[i];
    }

    // Compress single block
    sha256_compress_block(H, block);

    // Output digest (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        be32_store(&output[i * 4], H[i]);
    }
}

} // namespace hash
} // namespace smoke
