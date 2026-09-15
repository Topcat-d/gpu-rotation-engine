#pragma once

// Card 26.31: SHA-512 Device Implementation for Persistent Engine
// Single-block SHA-512 for input_len in [0, 32] bytes
// Output is always 64 bytes digest
//
// Key differences from SHA-256:
// - 64-bit words (vs 32-bit)
// - 80 rounds (vs 64)
// - 128-byte block size (vs 64)
// - 64-byte output (vs 32)

#include <stdint.h>

namespace smoke {
namespace hash {

// SHA-512 round constants (FIPS 180-4 Section 4.2.3)
// 80 constants, each 64 bits
__device__ __constant__ uint64_t K512[80] = {
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

// SHA-512 initial hash values (FIPS 180-4 Section 5.3.5)
__device__ __constant__ uint64_t SHA512_IV[8] = {
    0x6a09e667f3bcc908ULL,
    0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL,
    0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL,
    0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL,
    0x5be0cd19137e2179ULL
};

// Rotate right 64-bit
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

// SHA-512 functions (FIPS 180-4 Section 4.1.3)
__device__ __forceinline__ uint64_t Ch512(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint64_t Maj512(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint64_t Sigma0_512(uint64_t x) {
    return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39);
}

__device__ __forceinline__ uint64_t Sigma1_512(uint64_t x) {
    return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41);
}

__device__ __forceinline__ uint64_t sigma0_512(uint64_t x) {
    return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7);
}

__device__ __forceinline__ uint64_t sigma1_512(uint64_t x) {
    return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6);
}

// Convert 8 bytes (big-endian) to uint64_t
__device__ __forceinline__ uint64_t be64_load(const uint8_t* p) {
    return ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48) |
           ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32) |
           ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16) |
           ((uint64_t)p[6] << 8)  | ((uint64_t)p[7]);
}

// Store uint64_t as 8 bytes (big-endian)
__device__ __forceinline__ void be64_store(uint8_t* p, uint64_t v) {
    p[0] = (uint8_t)(v >> 56);
    p[1] = (uint8_t)(v >> 48);
    p[2] = (uint8_t)(v >> 40);
    p[3] = (uint8_t)(v >> 32);
    p[4] = (uint8_t)(v >> 24);
    p[5] = (uint8_t)(v >> 16);
    p[6] = (uint8_t)(v >> 8);
    p[7] = (uint8_t)(v);
}

// SHA-512 compression function for a single 128-byte block
// H is updated in place (8 x uint64_t)
__device__ void sha512_compress_block(uint64_t H[8], const uint8_t block[128]) {
    uint64_t W[80];

    // Message schedule: first 16 words from block (big-endian)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = be64_load(&block[i * 8]);
    }

    // Message schedule: remaining 64 words
    #pragma unroll
    for (int i = 16; i < 80; i++) {
        W[i] = sigma1_512(W[i-2]) + W[i-7] + sigma0_512(W[i-15]) + W[i-16];
    }

    // Working variables
    uint64_t a = H[0];
    uint64_t b = H[1];
    uint64_t c = H[2];
    uint64_t d = H[3];
    uint64_t e = H[4];
    uint64_t f = H[5];
    uint64_t g = H[6];
    uint64_t h = H[7];

    // 80 rounds
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        uint64_t T1 = h + Sigma1_512(e) + Ch512(e, f, g) + K512[i] + W[i];
        uint64_t T2 = Sigma0_512(a) + Maj512(a, b, c);
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

// SHA-512 for short inputs (0-32 bytes)
// This produces exactly one 128-byte block after padding:
//   [input][0x80][zeros...][length_bits_be128]
//   where length_bits = input_len * 8, stored at bytes 112-127
//
// For input_len in [0, 111], one block suffices (length at 112-127)
// For input_len in [112, 127], would need two blocks - but we limit to 32.
//
// Parameters:
//   input     - pointer to input bytes (must be valid for input_len bytes)
//   input_len - number of input bytes, MUST be in [0, 32]
//   output    - pointer to 64-byte output buffer
//
__device__ void sha512_hash_short(
    const uint8_t* input,
    uint16_t input_len,
    uint8_t output[64]
) {
    // Build padded block (128 bytes)
    uint8_t block[128];

    // Copy input
    for (uint16_t i = 0; i < input_len; i++) {
        block[i] = input[i];
    }

    // Padding: 0x80 followed by zeros
    block[input_len] = 0x80;
    for (uint16_t i = input_len + 1; i < 112; i++) {
        block[i] = 0;
    }

    // Length in bits (big-endian, 128-bit) at bytes 112-127
    // input_len <= 32, so length_bits = input_len * 8 <= 256, fits in low bytes
    // High 64 bits are 0, low 64 bits contain the length
    uint64_t length_bits = (uint64_t)input_len * 8;

    // High 64 bits (bytes 112-119): all zeros
    block[112] = 0;
    block[113] = 0;
    block[114] = 0;
    block[115] = 0;
    block[116] = 0;
    block[117] = 0;
    block[118] = 0;
    block[119] = 0;

    // Low 64 bits (bytes 120-127): length in bits, big-endian
    block[120] = (uint8_t)(length_bits >> 56);
    block[121] = (uint8_t)(length_bits >> 48);
    block[122] = (uint8_t)(length_bits >> 40);
    block[123] = (uint8_t)(length_bits >> 32);
    block[124] = (uint8_t)(length_bits >> 24);
    block[125] = (uint8_t)(length_bits >> 16);
    block[126] = (uint8_t)(length_bits >> 8);
    block[127] = (uint8_t)(length_bits);

    // Initialize hash state
    uint64_t H[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        H[i] = SHA512_IV[i];
    }

    // Compress single block
    sha512_compress_block(H, block);

    // Output digest (big-endian, 64 bytes = 8 x 8 bytes)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        be64_store(&output[i * 8], H[i]);
    }
}

} // namespace hash
} // namespace smoke
