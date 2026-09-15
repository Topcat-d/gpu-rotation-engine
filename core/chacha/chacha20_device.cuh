// SPDX-License-Identifier: Apache-2.0
// Smoke ChaCha20 Device Implementation
// Card 26.66: Device-side ChaCha20 stream cipher
//
// Features:
// - RFC 8439 compliant ChaCha20 implementation
// - Uses precomputed key from keyslot (32 bytes)
// - 12-byte nonce, 4-byte counter
// - Short-message support (up to 20 bytes inline)
//
// ChaCha20 structure:
// - State: 16 32-bit words
//   [0-3]:   Constants "expand 32-byte k"
//   [4-11]:  256-bit key (8 words)
//   [12]:    32-bit counter
//   [13-15]: 96-bit nonce (3 words)
// - 20 rounds (10 double-rounds)
// - Little-endian throughout

#ifndef SMOKE_CHACHA20_DEVICE_CUH
#define SMOKE_CHACHA20_DEVICE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace smoke {
namespace chacha {

// ============================================================================
// ChaCha20 Constants (RFC 8439)
// ============================================================================

// "expand 32-byte k" in little-endian
__device__ __constant__ uint32_t CHACHA_CONSTANTS[4] = {
    0x61707865,  // "expa"
    0x3320646e,  // "nd 3"
    0x79622d32,  // "2-by"
    0x6b206574   // "te k"
};

// ============================================================================
// ChaCha20 Quarter Round
// ============================================================================

__device__ __forceinline__
void quarter_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8)  | (d >> 24);
    c += d; b ^= c; b = (b << 7)  | (b >> 25);
}

// ============================================================================
// ChaCha20 Block Function
// ============================================================================

// Generate 64-byte keystream block
// key: 32 bytes (8 words, little-endian)
// nonce: 12 bytes (3 words, little-endian)
// counter: 32-bit block counter
// output: 64-byte keystream block
__device__
void chacha20_block(const uint32_t key[8], const uint32_t nonce[3],
                    uint32_t counter, uint8_t output[64]) {
    // Initialize state
    uint32_t state[16];

    // Constants
    state[0] = CHACHA_CONSTANTS[0];
    state[1] = CHACHA_CONSTANTS[1];
    state[2] = CHACHA_CONSTANTS[2];
    state[3] = CHACHA_CONSTANTS[3];

    // Key
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[4 + i] = key[i];
    }

    // Counter and nonce
    state[12] = counter;
    state[13] = nonce[0];
    state[14] = nonce[1];
    state[15] = nonce[2];

    // Working state
    uint32_t working[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        working[i] = state[i];
    }

    // 20 rounds (10 double-rounds)
    for (int i = 0; i < 10; i++) {
        // Column rounds
        quarter_round(working[0], working[4], working[8],  working[12]);
        quarter_round(working[1], working[5], working[9],  working[13]);
        quarter_round(working[2], working[6], working[10], working[14]);
        quarter_round(working[3], working[7], working[11], working[15]);

        // Diagonal rounds
        quarter_round(working[0], working[5], working[10], working[15]);
        quarter_round(working[1], working[6], working[11], working[12]);
        quarter_round(working[2], working[7], working[8],  working[13]);
        quarter_round(working[3], working[4], working[9],  working[14]);
    }

    // Add original state
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        working[i] += state[i];
    }

    // Serialize to little-endian bytes
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        output[i*4 + 0] = (uint8_t)(working[i]);
        output[i*4 + 1] = (uint8_t)(working[i] >> 8);
        output[i*4 + 2] = (uint8_t)(working[i] >> 16);
        output[i*4 + 3] = (uint8_t)(working[i] >> 24);
    }
}

// ============================================================================
// ChaCha20 Encryption
// ============================================================================

// Encrypt/decrypt using ChaCha20
// key: 32 bytes (stored as raw bytes, converted to words internally)
// nonce: 12 bytes
// counter: initial counter (typically 0 or 1)
// plaintext: input data
// plaintext_len: length in bytes
// ciphertext: output buffer (same size as plaintext)
extern "C" __device__
void chacha20_encrypt(
    const uint8_t key[32],
    const uint8_t nonce[12],
    uint32_t initial_counter,
    const uint8_t* plaintext,
    uint16_t plaintext_len,
    uint8_t* ciphertext)
{
    // Convert key bytes to words (little-endian)
    uint32_t key_words[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        key_words[i] = ((uint32_t)key[i*4]) |
                       ((uint32_t)key[i*4 + 1] << 8) |
                       ((uint32_t)key[i*4 + 2] << 16) |
                       ((uint32_t)key[i*4 + 3] << 24);
    }

    // Convert nonce bytes to words (little-endian)
    uint32_t nonce_words[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        nonce_words[i] = ((uint32_t)nonce[i*4]) |
                         ((uint32_t)nonce[i*4 + 1] << 8) |
                         ((uint32_t)nonce[i*4 + 2] << 16) |
                         ((uint32_t)nonce[i*4 + 3] << 24);
    }

    // Process in 64-byte blocks
    uint32_t counter = initial_counter;
    uint16_t offset = 0;

    while (offset < plaintext_len) {
        // Generate keystream block
        uint8_t keystream[64];
        chacha20_block(key_words, nonce_words, counter, keystream);

        // XOR with plaintext
        uint16_t block_len = (plaintext_len - offset > 64) ? 64 : (plaintext_len - offset);
        for (uint16_t i = 0; i < block_len; i++) {
            ciphertext[offset + i] = plaintext[offset + i] ^ keystream[i];
        }

        offset += block_len;
        counter++;
    }
}

// ============================================================================
// Persistent Engine Interface
// ============================================================================

// Short-message encryption for persistent engine
// Input layout: nonce[12] + plaintext[up to 20 bytes]
// Output: ciphertext (same length as plaintext)
extern "C" __device__
void chacha20_encrypt_short(
    const uint8_t key[32],
    const uint8_t* input,      // nonce[12] + plaintext
    uint16_t input_len,        // total input length (12 + plaintext_len)
    uint8_t* output)           // ciphertext
{
    if (input_len < 12) return;  // Need at least nonce

    const uint8_t* nonce = input;
    const uint8_t* plaintext = input + 12;
    uint16_t plaintext_len = input_len - 12;

    // Use counter = 0 (standard ChaCha20 starting counter per RFC 8439)
    chacha20_encrypt(key, nonce, 0, plaintext, plaintext_len, output);
}

// Card 26.66: ChaCha20 encryption using key from keyslot
// Avoids per-request key conversion overhead
// key: 32 bytes from keyslot (private_d field)
extern "C" __device__
void chacha20_encrypt_short_keyslot(
    const uint8_t key[32],     // Key from keyslot
    const uint8_t* input,      // nonce[12] + plaintext
    uint16_t input_len,        // total input length (12 + plaintext_len)
    uint8_t* output,           // ciphertext
    uint16_t* output_len)      // output length = plaintext_len
{
    if (input_len < 12) {
        *output_len = 0;
        return;
    }

    const uint8_t* nonce = input;
    const uint8_t* plaintext = input + 12;
    uint16_t plaintext_len = input_len - 12;

    // Limit to 20 bytes for inline mode (32 - 12 = 20)
    if (plaintext_len > 20) {
        plaintext_len = 20;
    }

    // Use counter = 0 (standard ChaCha20 starting counter)
    chacha20_encrypt(key, nonce, 0, plaintext, plaintext_len, output);

    *output_len = plaintext_len;
}

} // namespace chacha
} // namespace smoke

#endif // SMOKE_CHACHA20_DEVICE_CUH
