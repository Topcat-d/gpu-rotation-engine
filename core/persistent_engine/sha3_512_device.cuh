#pragma once

// Card 140: SHA3-512 Device Implementation for Persistent Engine
// Single-block SHA3-512 (Keccak) for input_len in [0, 32] bytes
// Output is always 64 bytes digest
//
// SHA3-512 parameters:
// - State: 1600 bits (5x5 array of 64-bit lanes)
// - Rate: 576 bits (72 bytes)
// - Capacity: 1024 bits (128 bytes)
// - Output: 512 bits (64 bytes)
// - Rounds: 24 Keccak-f[1600] rounds
//
// NOTE: Keccak-f[1600] permutation and helpers are defined in sha3_256_device.cuh
// (keccak_f1600, le64_load, le64_store, rotl64, constants).
// This file only defines the sha3_512_hash_short absorb/squeeze wrapper.

#include <stdint.h>

namespace smoke {
namespace hash {

// SHA3-512 for short inputs (0-32 bytes)
// For SHA3-512: rate = 72 bytes (576 bits), output = 64 bytes (512 bits)
// Padding: input || 0x06 || 0x00...0x00 || 0x80 (domain sep + pad10*1)
//
// Since input_len <= 32 < 72, everything fits in one block.
//
// Parameters:
//   input     - pointer to input bytes (must be valid for input_len bytes)
//   input_len - number of input bytes, MUST be in [0, 32]
//   output    - pointer to 64-byte output buffer
//
__device__ void sha3_512_hash_short(
    const uint8_t* input,
    uint16_t input_len,
    uint8_t output[64]
) {
    // Initialize state to zero
    uint64_t state[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) {
        state[i] = 0;
    }

    // Build padded block (rate = 72 bytes for SHA3-512)
    // We need to XOR: input || 0x06 || 0x00...0x00 || 0x80
    // where 0x06 is SHA3 domain separator and 0x80 is at byte 71 (last byte of rate)
    uint8_t block[72];

    // Zero the block
    #pragma unroll
    for (int i = 0; i < 72; i++) {
        block[i] = 0;
    }

    // Copy input
    for (uint16_t i = 0; i < input_len; i++) {
        block[i] = input[i];
    }

    // SHA3 domain separator (0x06) immediately after input
    block[input_len] = 0x06;

    // pad10*1: set last bit of rate block
    block[71] |= 0x80;

    // XOR block into state (9 lanes = 72 bytes / 8)
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        state[i] ^= le64_load(&block[i * 8]);
    }

    // Apply Keccak-f[1600]
    keccak_f1600(state);

    // Squeeze: extract first 64 bytes (8 lanes)
    // SHA3-512 output = 64 bytes <= rate = 72 bytes, so single squeeze is sufficient
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        le64_store(&output[i * 8], state[i]);
    }
}

} // namespace hash
} // namespace smoke
