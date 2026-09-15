// SPDX-License-Identifier: Apache-2.0
// Smoke AES-256-GCM Device Implementation
// Card 26.65: Device-side AES-256-GCM authenticated encryption
//
// Features:
// - Uses precomputed AES round keys from keyslot (Card 26.64)
// - GHASH for authentication tag generation
// - Supports short messages (inline mode)
//
// AES-GCM structure:
// - J0 = IV || 0x00000001 (IV is 12 bytes)
// - H = AES(K, 0^128) is the hash subkey
// - Counter starts at 2 for encryption (J0 is used for tag only)
// - Tag = AES(K, J0) XOR GHASH(H, A || pad || C || pad || len(A) || len(C))

#ifndef SMOKE_AES256_GCM_DEVICE_CUH
#define SMOKE_AES256_GCM_DEVICE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "aes256_ctr_device.cuh"

namespace smoke {
namespace aes {

// ============================================================================
// GF(2^128) Multiplication for GHASH
// ============================================================================

// GCM uses GF(2^128) with reducing polynomial x^128 + x^7 + x^2 + x + 1
// Representation: MSB-first (bit 0 is most significant)

// Multiply two 128-bit values in GF(2^128)
// X and Y are 16-byte arrays in big-endian order
__device__
void gf128_mul(const uint8_t X[16], const uint8_t Y[16], uint8_t result[16]) {
    uint8_t Z[16] = {0};  // Initialize to zero
    uint8_t V[16];

    // Copy Y to V (we'll shift V)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        V[i] = Y[i];
    }

    // For each bit of X (128 bits total)
    for (int i = 0; i < 128; i++) {
        int byte_idx = i / 8;
        int bit_idx = 7 - (i % 8);  // MSB first

        // If bit i of X is set, Z = Z XOR V
        if (X[byte_idx] & (1 << bit_idx)) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                Z[j] ^= V[j];
            }
        }

        // Check if LSB of V is set (need to reduce)
        bool lsb_set = V[15] & 0x01;

        // Right shift V by 1 (MSB-first representation)
        for (int j = 15; j > 0; j--) {
            V[j] = (V[j] >> 1) | ((V[j-1] & 0x01) << 7);
        }
        V[0] >>= 1;

        // If LSB was set, XOR with R = 0xE1 || 0^120
        // This is the reduction polynomial x^128 + x^7 + x^2 + x + 1
        // In MSB-first: 0xE1000000...00
        if (lsb_set) {
            V[0] ^= 0xE1;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i] = Z[i];
    }
}

// GHASH: Compute authentication hash
// H: hash subkey (AES(K, 0^128))
// data: concatenation of A || pad(A) || C || pad(C) || [len(A)]_64 || [len(C)]_64
// data_len: total length of padded data
// result: 16-byte output
__device__
void ghash(const uint8_t H[16], const uint8_t* data, uint32_t data_len, uint8_t result[16]) {
    // Initialize Y to zero
    uint8_t Y[16] = {0};

    // Process each 16-byte block
    uint32_t num_blocks = (data_len + 15) / 16;

    for (uint32_t i = 0; i < num_blocks; i++) {
        uint8_t block[16] = {0};
        uint32_t offset = i * 16;
        uint32_t copy_len = (data_len - offset > 16) ? 16 : (data_len - offset);

        // Copy block (zero-pad if needed)
        for (uint32_t j = 0; j < copy_len; j++) {
            block[j] = data[offset + j];
        }

        // Y = (Y XOR block) * H
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            Y[j] ^= block[j];
        }

        uint8_t temp[16];
        gf128_mul(Y, H, temp);

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            Y[j] = temp[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i] = Y[i];
    }
}

// ============================================================================
// AES-256-GCM Seal (Encrypt + Tag)
// ============================================================================

// AES-GCM seal with precomputed round keys
// roundkeys: 60 words from keyslot (precomputed)
// nonce: 12 bytes
// aad: additional authenticated data (can be NULL if aad_len == 0)
// aad_len: length of AAD
// plaintext: data to encrypt
// plaintext_len: length of plaintext
// ciphertext: output buffer (same size as plaintext)
// tag: 16-byte output authentication tag
extern "C" __device__
void aes256_gcm_seal(
    const uint32_t roundkeys[60],
    const uint8_t nonce[12],
    const uint8_t* aad,
    uint16_t aad_len,
    const uint8_t* plaintext,
    uint16_t plaintext_len,
    uint8_t* ciphertext,
    uint8_t tag[16])
{
    // Step 1: Compute H = AES(K, 0^128)
    uint8_t zeros[16] = {0};
    uint8_t H[16];
    aes256_encrypt_block(roundkeys, zeros, H);

    // Step 2: Construct J0 = nonce || 0x00000001
    uint8_t J0[16];
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        J0[i] = nonce[i];
    }
    J0[12] = 0x00;
    J0[13] = 0x00;
    J0[14] = 0x00;
    J0[15] = 0x01;

    // Step 3: Encrypt using counter mode starting at counter 2
    // CTR block = nonce || counter (big-endian 4-byte counter)
    uint32_t counter = 2;  // Start at 2 (J0 is counter 1, used for tag)
    uint16_t offset = 0;

    while (offset < plaintext_len) {
        // Build counter block
        uint8_t ctr_block[16];
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            ctr_block[i] = nonce[i];
        }
        ctr_block[12] = (uint8_t)(counter >> 24);
        ctr_block[13] = (uint8_t)(counter >> 16);
        ctr_block[14] = (uint8_t)(counter >> 8);
        ctr_block[15] = (uint8_t)(counter);

        // Generate keystream block
        uint8_t keystream[16];
        aes256_encrypt_block(roundkeys, ctr_block, keystream);

        // XOR with plaintext
        uint16_t block_len = (plaintext_len - offset > 16) ? 16 : (plaintext_len - offset);
        for (uint16_t i = 0; i < block_len; i++) {
            ciphertext[offset + i] = plaintext[offset + i] ^ keystream[i];
        }

        offset += block_len;
        counter++;
    }

    // Step 4: Compute GHASH over AAD || pad || C || pad || len(A) || len(C)
    // For simplicity with short messages, we'll build the data inline

    // Calculate padded lengths
    uint16_t aad_padded = ((aad_len + 15) / 16) * 16;
    uint16_t ct_padded = ((plaintext_len + 15) / 16) * 16;
    uint32_t ghash_len = aad_padded + ct_padded + 16;  // +16 for lengths

    // Build GHASH input (max reasonable size for inline: 128 bytes)
    // This limits AAD + plaintext but is fine for MVP
    uint8_t ghash_input[128] = {0};
    uint32_t pos = 0;

    // Copy AAD (already zero-padded by initialization)
    for (uint16_t i = 0; i < aad_len && pos < 128; i++) {
        ghash_input[pos++] = aad[i];
    }
    pos = aad_padded;  // Skip to padded position

    // Copy ciphertext
    for (uint16_t i = 0; i < plaintext_len && pos < 112; i++) {
        ghash_input[pos++] = ciphertext[i];
    }
    pos = aad_padded + ct_padded;

    // Append lengths (in bits, big-endian 64-bit each)
    uint64_t aad_bits = (uint64_t)aad_len * 8;
    uint64_t ct_bits = (uint64_t)plaintext_len * 8;

    ghash_input[pos++] = (uint8_t)(aad_bits >> 56);
    ghash_input[pos++] = (uint8_t)(aad_bits >> 48);
    ghash_input[pos++] = (uint8_t)(aad_bits >> 40);
    ghash_input[pos++] = (uint8_t)(aad_bits >> 32);
    ghash_input[pos++] = (uint8_t)(aad_bits >> 24);
    ghash_input[pos++] = (uint8_t)(aad_bits >> 16);
    ghash_input[pos++] = (uint8_t)(aad_bits >> 8);
    ghash_input[pos++] = (uint8_t)(aad_bits);

    ghash_input[pos++] = (uint8_t)(ct_bits >> 56);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 48);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 40);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 32);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 24);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 16);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 8);
    ghash_input[pos++] = (uint8_t)(ct_bits);

    // Compute GHASH
    uint8_t S[16];
    ghash(H, ghash_input, ghash_len, S);

    // Step 5: Tag = AES(K, J0) XOR S
    uint8_t E_J0[16];
    aes256_encrypt_block(roundkeys, J0, E_J0);

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        tag[i] = E_J0[i] ^ S[i];
    }
}

// Simplified seal for persistent engine short messages
// Input layout: nonce[12] + plaintext[up to 16 bytes]
// No AAD support for simplicity
// Output: ciphertext[N] + tag[16]
extern "C" __device__
void aes256_gcm_seal_short(
    const uint32_t roundkeys[60],
    const uint8_t* input,          // nonce[12] + plaintext
    uint16_t input_len,            // total input length
    uint8_t* output,               // ciphertext + tag[16]
    uint16_t* output_len)          // output length = plaintext_len + 16
{
    if (input_len < 12) {
        *output_len = 0;
        return;
    }

    const uint8_t* nonce = input;
    const uint8_t* plaintext = input + 12;
    uint16_t plaintext_len = input_len - 12;

    // Fail closed above the inline limit. Silent prefix sealing is never safe.
    if (plaintext_len > 16) {
        *output_len = 0;
        return;
    }

    // Seal
    aes256_gcm_seal(
        roundkeys,
        nonce,
        nullptr,       // no AAD
        0,             // no AAD length
        plaintext,
        plaintext_len,
        output,        // ciphertext
        output + plaintext_len  // tag at end
    );

    *output_len = plaintext_len + 16;
}

// AES-256-GCM seal for payload-slab plaintext (SEAL-SLAB-01).
// No AAD in the native ABI. Output is ciphertext || tag in the dedicated seal
// output region. Bounds and output-segment ownership are enforced by dispatch
// before this function is called.
extern "C" __device__
void aes256_gcm_seal_slab(
    const uint32_t roundkeys[60],
    const uint8_t nonce[12],
    const uint8_t* payload_slab,
    uint32_t payload_offset,
    uint32_t plaintext_len,
    uint8_t* output_slab,
    uint32_t out_offset,
    uint32_t* out_len)
{
    *out_len = 0;
    if (plaintext_len == 0 || plaintext_len > 65535) {
        return;
    }

    const uint8_t* plaintext = payload_slab + payload_offset;
    uint8_t* ciphertext = output_slab + out_offset;

    uint8_t zeros[16] = {0};
    uint8_t H[16];
    aes256_encrypt_block(roundkeys, zeros, H);

    uint8_t J0[16];
    #pragma unroll
    for (int i = 0; i < 12; i++) J0[i] = nonce[i];
    J0[12] = 0x00; J0[13] = 0x00; J0[14] = 0x00; J0[15] = 0x01;

    uint32_t counter = 2;
    uint32_t offset = 0;
    while (offset < plaintext_len) {
        uint8_t ctr_block[16];
        #pragma unroll
        for (int i = 0; i < 12; i++) ctr_block[i] = nonce[i];
        ctr_block[12] = (uint8_t)(counter >> 24);
        ctr_block[13] = (uint8_t)(counter >> 16);
        ctr_block[14] = (uint8_t)(counter >> 8);
        ctr_block[15] = (uint8_t)(counter);

        uint8_t keystream[16];
        aes256_encrypt_block(roundkeys, ctr_block, keystream);
        uint32_t block_len =
            (plaintext_len - offset > 16) ? 16 : (plaintext_len - offset);
        for (uint32_t i = 0; i < block_len; i++) {
            ciphertext[offset + i] = plaintext[offset + i] ^ keystream[i];
        }
        offset += block_len;
        counter++;
    }

    // Incremental GHASH over ciphertext, followed by the AEAD length block.
    uint8_t Y[16] = {0};
    uint32_t num_blocks = (plaintext_len + 15) / 16;
    for (uint32_t blk = 0; blk < num_blocks; blk++) {
        uint8_t block[16] = {0};
        uint32_t block_offset = blk * 16;
        uint32_t copy_len =
            (plaintext_len - block_offset > 16) ? 16 : (plaintext_len - block_offset);
        for (uint32_t i = 0; i < copy_len; i++) {
            block[i] = ciphertext[block_offset + i];
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) Y[i] ^= block[i];
        uint8_t product[16];
        gf128_mul(Y, H, product);
        #pragma unroll
        for (int i = 0; i < 16; i++) Y[i] = product[i];
    }

    uint8_t len_block[16] = {0};
    uint64_t ct_bits = (uint64_t)plaintext_len * 8;
    len_block[8]  = (uint8_t)(ct_bits >> 56);
    len_block[9]  = (uint8_t)(ct_bits >> 48);
    len_block[10] = (uint8_t)(ct_bits >> 40);
    len_block[11] = (uint8_t)(ct_bits >> 32);
    len_block[12] = (uint8_t)(ct_bits >> 24);
    len_block[13] = (uint8_t)(ct_bits >> 16);
    len_block[14] = (uint8_t)(ct_bits >> 8);
    len_block[15] = (uint8_t)(ct_bits);
    #pragma unroll
    for (int i = 0; i < 16; i++) Y[i] ^= len_block[i];
    uint8_t S[16];
    gf128_mul(Y, H, S);

    uint8_t E_J0[16];
    aes256_encrypt_block(roundkeys, J0, E_J0);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        ciphertext[plaintext_len + i] = E_J0[i] ^ S[i];
    }
    *out_len = plaintext_len + 16;
}

// ============================================================================
// Constant-time comparison (DECRYPT-ABI-01)
// ============================================================================

// OR-accumulate pattern: no early exit, constant time regardless of mismatch position
__device__
uint8_t constant_time_equal(const uint8_t* a, const uint8_t* b, uint32_t len) {
    uint8_t diff = 0;
    for (uint32_t i = 0; i < len; i++) {
        diff |= a[i] ^ b[i];
    }
    return (diff == 0) ? 1 : 0;
}

// ============================================================================
// AES-256-GCM Open (Decrypt + Verify) — DECRYPT-ABI-01 / AES-GCM-OPEN-SHORT-01
// ============================================================================

// AES-GCM open (decrypt) for short inline messages
// Input layout: nonce[12] || ciphertext[N] || tag[16]
// N = input_len - 28 (max 4 bytes in inline mode)
//
// INVARIANT: Auth-before-plaintext.
// Tag verification MUST complete before ANY plaintext bytes are written.
// On auth failure: output is zeroed, output_len=0, auth_ok=0.
//
// AAD: NOT supported in inline mode. Caller must reject non-empty AAD
// before calling this function.
extern "C" __device__
void aes256_gcm_open_short(
    const uint32_t roundkeys[60],
    const uint8_t* input,          // nonce[12] || ciphertext[N] || tag[16]
    uint16_t input_len,            // total input length (must be >= 28)
    uint8_t* output,               // plaintext buffer (same size as ciphertext)
    uint16_t* output_len,          // output: plaintext length (0 on auth fail)
    uint8_t* auth_ok)              // output: 1 if tag matches, 0 if not
{
    *output_len = 0;
    *auth_ok = 0;

    // Validate minimum input size: nonce[12] + tag[16] = 28
    if (input_len < 28) {
        return;
    }

    const uint8_t* nonce = input;
    uint16_t ct_len = input_len - 28;  // ciphertext length (can be 0)
    const uint8_t* ciphertext = input + 12;
    const uint8_t* provided_tag = input + 12 + ct_len;

    // Step 1: Compute H = AES(K, 0^128) — hash subkey
    uint8_t zeros[16] = {0};
    uint8_t H[16];
    aes256_encrypt_block(roundkeys, zeros, H);

    // Step 2: Construct J0 = nonce || 0x00000001
    uint8_t J0[16];
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        J0[i] = nonce[i];
    }
    J0[12] = 0x00;
    J0[13] = 0x00;
    J0[14] = 0x00;
    J0[15] = 0x01;

    // Step 3: Compute GHASH over ciphertext (no AAD in inline mode)
    // GHASH input: ct || pad(ct) || len(A)_64 || len(C)_64
    uint16_t ct_padded = ((ct_len + 15) / 16) * 16;
    uint32_t ghash_len = ct_padded + 16;  // +16 for length block

    uint8_t ghash_input[128] = {0};
    uint32_t pos = 0;

    // Copy ciphertext (zero-padded by initialization)
    for (uint16_t i = 0; i < ct_len && pos < 112; i++) {
        ghash_input[pos++] = ciphertext[i];
    }
    pos = ct_padded;

    // Append lengths (AAD bits = 0, CT bits = ct_len * 8), big-endian 64-bit each
    uint64_t ct_bits = (uint64_t)ct_len * 8;
    // AAD length = 0, already zero from initialization
    pos += 8;  // skip 8 zero bytes for AAD length

    ghash_input[pos++] = (uint8_t)(ct_bits >> 56);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 48);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 40);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 32);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 24);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 16);
    ghash_input[pos++] = (uint8_t)(ct_bits >> 8);
    ghash_input[pos++] = (uint8_t)(ct_bits);

    // Compute GHASH
    uint8_t S[16];
    ghash(H, ghash_input, ghash_len, S);

    // Step 4: Compute expected tag = AES(K, J0) XOR S
    uint8_t E_J0[16];
    aes256_encrypt_block(roundkeys, J0, E_J0);

    uint8_t expected_tag[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        expected_tag[i] = E_J0[i] ^ S[i];
    }

    // Step 5: Constant-time tag verification (MUST happen before plaintext write)
    uint8_t tag_match = constant_time_equal(expected_tag, provided_tag, 16);

    if (!tag_match) {
        // Auth failure: zero output, return nothing
        for (uint16_t i = 0; i < 64; i++) {
            output[i] = 0;
        }
        *output_len = 0;
        *auth_ok = 0;
        return;
    }

    // Step 6: Tag verified — now decrypt using AES-CTR starting at counter=2
    // ONLY reached after successful auth verification
    *auth_ok = 1;

    uint32_t counter = 2;
    uint16_t offset = 0;

    while (offset < ct_len) {
        uint8_t ctr_block[16];
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            ctr_block[i] = nonce[i];
        }
        ctr_block[12] = (uint8_t)(counter >> 24);
        ctr_block[13] = (uint8_t)(counter >> 16);
        ctr_block[14] = (uint8_t)(counter >> 8);
        ctr_block[15] = (uint8_t)(counter);

        uint8_t keystream[16];
        aes256_encrypt_block(roundkeys, ctr_block, keystream);

        uint16_t block_len = (ct_len - offset > 16) ? 16 : (ct_len - offset);
        for (uint16_t i = 0; i < block_len; i++) {
            output[offset + i] = ciphertext[offset + i] ^ keystream[i];
        }

        offset += block_len;
        counter++;
    }

    *output_len = ct_len;
}

// ============================================================================
// AES-256-GCM Open (Decrypt) — Slab variant (DECRYPT-SLAB-01)
// ============================================================================
//
// For payloads > 4 bytes ciphertext, data is in the payload slab.
// Input layout:
//   inline_input[0:11] = nonce (12 bytes)
//   payload_slab[offset : offset+payload_len] = ciphertext[N] || tag[16]
//
// Output goes to output_slab[out_offset : out_offset+N]
//
// Same auth-before-plaintext invariant as open_short.
extern "C" __device__
void aes256_gcm_open_slab(
    const uint32_t roundkeys[60],
    const uint8_t nonce[12],               // From inline input
    const uint8_t* payload_slab,           // Mapped slab memory
    uint32_t payload_offset,               // Offset into slab
    uint32_t payload_len,                  // ciphertext_len + 16 (tag)
    uint8_t* output_slab,                  // Output slab for plaintext
    uint32_t out_offset,                   // Output offset in slab
    uint32_t* out_len,                     // Output: plaintext length
    uint8_t* auth_ok)                      // Output: 1 if tag matches
{
    *out_len = 0;
    *auth_ok = 0;

    if (payload_len < 16) {
        return;  // Must have at least tag[16]
    }

    const uint8_t* slab_data = payload_slab + payload_offset;
    uint32_t ct_len = payload_len - 16;
    const uint8_t* ciphertext = slab_data;
    const uint8_t* provided_tag = slab_data + ct_len;

    // Step 1: H = AES(K, 0^128)
    uint8_t zeros[16] = {0};
    uint8_t H[16];
    aes256_encrypt_block(roundkeys, zeros, H);

    // Step 2: J0 = nonce || 0x00000001
    uint8_t J0[16];
    #pragma unroll
    for (int i = 0; i < 12; i++) J0[i] = nonce[i];
    J0[12] = 0x00; J0[13] = 0x00; J0[14] = 0x00; J0[15] = 0x01;

    // Step 3: GHASH over ciphertext (no AAD)
    // For large payloads, compute GHASH incrementally
    uint8_t Y[16] = {0};

    // Process ciphertext blocks
    uint32_t num_ct_blocks = (ct_len + 15) / 16;
    for (uint32_t blk = 0; blk < num_ct_blocks; blk++) {
        uint8_t block[16] = {0};
        uint32_t blk_offset = blk * 16;
        uint32_t copy_len = (ct_len - blk_offset > 16) ? 16 : (ct_len - blk_offset);
        for (uint32_t j = 0; j < copy_len; j++) {
            block[j] = ciphertext[blk_offset + j];
        }
        #pragma unroll
        for (int j = 0; j < 16; j++) Y[j] ^= block[j];
        uint8_t temp[16];
        gf128_mul(Y, H, temp);
        #pragma unroll
        for (int j = 0; j < 16; j++) Y[j] = temp[j];
    }

    // Final length block: [len(A)]_64 || [len(C)]_64
    uint8_t len_block[16] = {0};
    uint64_t ct_bits = (uint64_t)ct_len * 8;
    len_block[8]  = (uint8_t)(ct_bits >> 56);
    len_block[9]  = (uint8_t)(ct_bits >> 48);
    len_block[10] = (uint8_t)(ct_bits >> 40);
    len_block[11] = (uint8_t)(ct_bits >> 32);
    len_block[12] = (uint8_t)(ct_bits >> 24);
    len_block[13] = (uint8_t)(ct_bits >> 16);
    len_block[14] = (uint8_t)(ct_bits >> 8);
    len_block[15] = (uint8_t)(ct_bits);

    #pragma unroll
    for (int j = 0; j < 16; j++) Y[j] ^= len_block[j];
    uint8_t S[16];
    gf128_mul(Y, H, S);

    // Step 4: expected_tag = AES(K, J0) XOR S
    uint8_t E_J0[16];
    aes256_encrypt_block(roundkeys, J0, E_J0);
    uint8_t expected_tag[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) expected_tag[i] = E_J0[i] ^ S[i];

    // Step 5: Constant-time tag verify
    uint8_t tag_match = constant_time_equal(expected_tag, provided_tag, 16);

    if (!tag_match) {
        *out_len = 0;
        *auth_ok = 0;
        return;
    }

    // Step 6: Decrypt (only after auth passes)
    *auth_ok = 1;
    uint8_t* output_ptr = output_slab + out_offset;
    uint32_t counter = 2;
    uint32_t offset = 0;

    while (offset < ct_len) {
        uint8_t ctr_block[16];
        #pragma unroll
        for (int i = 0; i < 12; i++) ctr_block[i] = nonce[i];
        ctr_block[12] = (uint8_t)(counter >> 24);
        ctr_block[13] = (uint8_t)(counter >> 16);
        ctr_block[14] = (uint8_t)(counter >> 8);
        ctr_block[15] = (uint8_t)(counter);

        uint8_t keystream[16];
        aes256_encrypt_block(roundkeys, ctr_block, keystream);

        uint32_t block_len = (ct_len - offset > 16) ? 16 : (ct_len - offset);
        for (uint32_t i = 0; i < block_len; i++) {
            output_ptr[offset + i] = ciphertext[offset + i] ^ keystream[i];
        }
        offset += block_len;
        counter++;
    }

    *out_len = ct_len;
}

} // namespace aes
} // namespace smoke

#endif // SMOKE_AES256_GCM_DEVICE_CUH
