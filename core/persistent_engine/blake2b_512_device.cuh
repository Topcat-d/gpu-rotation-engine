#pragma once

// Card 141: BLAKE2b-512 Device Implementation for Persistent Engine
// Single-block BLAKE2b-512 for input_len in [0, 32] bytes
// Output is always 64 bytes digest
//
// BLAKE2b-512 parameters:
// - State: 8 x 64-bit words (512 bits)
// - Block size: 128 bytes
// - Rounds: 12
// - Output: 512 bits (64 bytes)
// - ARX operations (Add-Rotate-XOR) - extremely GPU-friendly
//
// NOTE: G mixing function, compress, constants, and helpers are defined in
// blake2b_256_device.cuh (blake2b_G, blake2b_compress, blake2b_le64_load,
// blake2b_le64_store, BLAKE2B_IV, BLAKE2B_SIGMA).
// This file only defines the blake2b_512_hash_short wrapper with nn=64 params.

#include <stdint.h>

namespace smoke {
namespace hash {

// BLAKE2b-512 for short inputs (0-32 bytes)
// For BLAKE2b-512: block_size = 128 bytes, output = 64 bytes
//
// Parameters:
//   input     - pointer to input bytes (must be valid for input_len bytes)
//   input_len - number of input bytes, MUST be in [0, 32]
//   output    - pointer to 64-byte output buffer
//
__device__ void blake2b_512_hash_short(
    const uint8_t* input,
    uint16_t input_len,
    uint8_t output[64]
) {
    // Initialize state with IV
    uint64_t h[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        h[i] = BLAKE2B_IV[i];
    }

    // BLAKE2b parameter block XOR into h[0]:
    // h[0] ^= 0x01010000 ^ (kk << 8) ^ nn
    // For unkeyed BLAKE2b-512: kk=0 (no key), nn=64 (output length)
    // So: h[0] ^= 0x01010000 ^ 0 ^ 64 = 0x01010040
    h[0] ^= 0x01010040ULL;

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

    // Extract all 64 bytes (8 x 64-bit words) as output
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        blake2b_le64_store(&output[i * 8], h[i]);
    }
}

} // namespace hash
} // namespace smoke
