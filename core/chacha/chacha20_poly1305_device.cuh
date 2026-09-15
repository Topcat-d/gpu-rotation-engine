// SPDX-License-Identifier: Apache-2.0
// Smoke ChaCha20-Poly1305 AEAD Device Implementation
// Card 26.68: ChaCha20-Poly1305 AEAD per RFC 8439
//
// Features:
// - RFC 8439 compliant AEAD construction
// - 256-bit key, 96-bit nonce, 128-bit authentication tag
// - Short-message support (up to 16 bytes plaintext inline)
//
// AEAD Construction (RFC 8439 Section 2.8):
// 1. Generate Poly1305 one-time key: ChaCha20(key, nonce, counter=0)[0:32]
// 2. Encrypt plaintext: ChaCha20(key, nonce, counter=1)
// 3. Compute tag: Poly1305(otk, aad || pad || ct || pad || len(aad) || len(ct))
//
// Input layout for short messages:
//   input[0:11]  = nonce (12 bytes)
//   input[12:N]  = plaintext (0-16 bytes for inline)
//
// Output layout:
//   output[0:N]  = ciphertext
//   output[N:N+16] = authentication tag

#ifndef SMOKE_CHACHA20_POLY1305_DEVICE_CUH
#define SMOKE_CHACHA20_POLY1305_DEVICE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../poly1305/poly1305_device.cuh"
#include "../poly1305/poly1305_26.cuh"

namespace smoke {
namespace aead {

// ============================================================================
// ChaCha20 Block Function (for key generation)
// ============================================================================

// Generate a 64-byte ChaCha20 keystream block
// Replicates the logic from chacha20_device.cuh but operates directly here
// to avoid namespace issues
__device__
void chacha20_block_for_keygen(
    const uint8_t key[32],
    const uint8_t nonce[12],
    uint32_t counter,
    uint8_t output[64])
{
    // ChaCha20 constants
    const uint32_t CONSTANTS[4] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
    };

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

    // Initialize state
    uint32_t state[16];
    state[0] = CONSTANTS[0];
    state[1] = CONSTANTS[1];
    state[2] = CONSTANTS[2];
    state[3] = CONSTANTS[3];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[4 + i] = key_words[i];
    }
    state[12] = counter;
    state[13] = nonce_words[0];
    state[14] = nonce_words[1];
    state[15] = nonce_words[2];

    // Working state
    uint32_t working[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        working[i] = state[i];
    }

    // Quarter round helper (inline)
    #define QR(a, b, c, d) do { \
        a += b; d ^= a; d = (d << 16) | (d >> 16); \
        c += d; b ^= c; b = (b << 12) | (b >> 20); \
        a += b; d ^= a; d = (d << 8)  | (d >> 24); \
        c += d; b ^= c; b = (b << 7)  | (b >> 25); \
    } while(0)

    // 20 rounds (10 double-rounds)
    for (int i = 0; i < 10; i++) {
        // Column rounds
        QR(working[0], working[4], working[8],  working[12]);
        QR(working[1], working[5], working[9],  working[13]);
        QR(working[2], working[6], working[10], working[14]);
        QR(working[3], working[7], working[11], working[15]);
        // Diagonal rounds
        QR(working[0], working[5], working[10], working[15]);
        QR(working[1], working[6], working[11], working[12]);
        QR(working[2], working[7], working[8],  working[13]);
        QR(working[3], working[4], working[9],  working[14]);
    }

    #undef QR

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
// Pad length to 16-byte boundary
// ============================================================================

__device__ __forceinline__
uint32_t pad16(uint32_t len) {
    return (16 - (len & 15)) & 15;
}

// ============================================================================
// ChaCha20-Poly1305 AEAD Encrypt
// ============================================================================

// Encrypt with ChaCha20-Poly1305 AEAD (no AAD, short message)
// key: 32-byte ChaCha20 key
// nonce: 12-byte nonce
// plaintext: input data
// pt_len: plaintext length (0-16 for inline)
// ciphertext: output buffer (pt_len bytes)
// tag: 16-byte authentication tag
__device__
void chacha20_poly1305_encrypt_short(
    const uint8_t key[32],
    const uint8_t nonce[12],
    const uint8_t* plaintext,
    uint16_t pt_len,
    uint8_t* ciphertext,
    uint8_t tag[16])
{
    // Step 1: Generate Poly1305 one-time key using ChaCha20 with counter=0
    uint8_t keystream0[64];
    chacha20_block_for_keygen(key, nonce, 0, keystream0);

    // First 32 bytes become the Poly1305 key
    uint8_t otk[32];
    for (int i = 0; i < 32; i++) {
        otk[i] = keystream0[i];
    }

    // Step 2: Encrypt plaintext with ChaCha20 starting at counter=1
    if (pt_len > 0) {
        uint8_t keystream1[64];
        chacha20_block_for_keygen(key, nonce, 1, keystream1);

        // XOR plaintext with keystream
        for (uint16_t i = 0; i < pt_len; i++) {
            ciphertext[i] = plaintext[i] ^ keystream1[i];
        }
    }

    // Step 3: Compute Poly1305 tag over ciphertext
    // RFC 8439 Section 2.8 MAC input format:
    //   - AAD (empty in our case)
    //   - Pad AAD to 16-byte boundary (0 bytes since 0 % 16 == 0)
    //   - Ciphertext
    //   - Pad ciphertext to 16-byte boundary
    //   - 8-byte AAD length (little-endian) = 0
    //   - 8-byte ciphertext length (little-endian)
    //
    // Note: pad16(x) returns empty if len(x) % 16 == 0

    uint8_t mac_input[64];  // Max: 0 + 16 + 16 + 16 = 48 bytes for pt_len <= 16
    uint32_t mac_len = 0;

    // AAD is empty (0 bytes), and pad16(0) = 0 bytes (0 % 16 == 0)
    // So we add nothing here

    // Ciphertext
    for (uint16_t i = 0; i < pt_len; i++) {
        mac_input[mac_len++] = ciphertext[i];
    }

    // Ciphertext padding (pad to 16-byte boundary)
    uint32_t ct_pad = pad16(pt_len);
    for (uint32_t i = 0; i < ct_pad; i++) {
        mac_input[mac_len++] = 0;
    }

    // 8-byte AAD length (little-endian, = 0)
    for (int i = 0; i < 8; i++) {
        mac_input[mac_len++] = 0;
    }

    // 8-byte ciphertext length (little-endian)
    uint64_t ct_len64 = pt_len;
    mac_input[mac_len++] = (uint8_t)(ct_len64);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 8);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 16);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 24);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 32);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 40);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 48);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 56);

    // Compute Poly1305 tag
    smoke::poly1305::poly1305_mac(otk, mac_input, (uint16_t)mac_len, tag);
}

// ============================================================================
// Persistent Engine Interface
// ============================================================================

// ChaCha20-Poly1305 encrypt for persistent engine
// Input layout: nonce[12] + plaintext[0-16]
// Output: ciphertext[0-16] + tag[16]
// CONTRACT (D-089): callers MUST reject input_len > 28 (12 nonce + 16 pt)
// with INVALID_INPUT_LEN before calling. This device helper independently
// fails closed rather than silently sealing a prefix.
extern "C" __device__
void chacha20_poly1305_encrypt_short_keyslot(
    const uint8_t key[32],     // Key from keyslot
    const uint8_t* input,      // nonce[12] + plaintext
    uint16_t input_len,        // 12 + pt_len
    uint8_t* output,           // ciphertext + tag
    uint16_t* output_len)      // ct_len + 16
{
    // Extract nonce and plaintext
    const uint8_t* nonce = input;
    const uint8_t* plaintext = input + 12;
    uint16_t pt_len = (input_len > 12) ? (input_len - 12) : 0;

    // OOB backstop — never turn an oversized request into a successful prefix.
    if (pt_len > 16) {
        *output_len = 0;
        return;
    }

    // Encrypt: output = ciphertext || tag
    chacha20_poly1305_encrypt_short(key, nonce, plaintext, pt_len, output, output + pt_len);

    *output_len = pt_len + 16;
}

// ChaCha20-Poly1305 seal for payload-slab plaintext (SEAL-SLAB-01).
// No AAD in native ABI v1. Output is ciphertext || tag in the dedicated seal
// output region. Dispatch owns all slab/key/length validation.
extern "C" __device__
void chacha20_poly1305_encrypt_slab(
    const uint8_t key[32],
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

    uint8_t keystream0[64];
    chacha20_block_for_keygen(key, nonce, 0, keystream0);
    uint8_t otk[32];
    for (int i = 0; i < 32; i++) otk[i] = keystream0[i];

    uint32_t offset = 0;
    uint32_t counter = 1;
    while (offset < plaintext_len) {
        uint8_t keystream[64];
        chacha20_block_for_keygen(key, nonce, counter, keystream);
        uint32_t block_len =
            (plaintext_len - offset > 64) ? 64 : (plaintext_len - offset);
        for (uint32_t i = 0; i < block_len; i++) {
            ciphertext[offset + i] = plaintext[offset + i] ^ keystream[i];
        }
        offset += block_len;
        counter++;
    }

    Poly1305Key26 pk;
    Poly1305Acc acc = {0, 0, 0, 0, 0};
    poly1305_clamp(pk, otk);
    pk.pad0 = ld_le32(otk + 16);
    pk.pad1 = ld_le32(otk + 20);
    pk.pad2 = ld_le32(otk + 24);
    pk.pad3 = ld_le32(otk + 28);

    uint32_t full_blocks = plaintext_len / 16;
    uint32_t tail = plaintext_len & 15;
    uint32_t t0, t1, t2, t3, t4;
    for (uint32_t block = 0; block < full_blocks; block++) {
        poly1305_load_block26(
            ciphertext + block * 16, true, t0, t1, t2, t3, t4);
        poly1305_acc_1block(acc, pk, t0, t1, t2, t3, t4);
    }
    if (tail > 0) {
        uint8_t partial[16] = {0};
        for (uint32_t i = 0; i < tail; i++) {
            partial[i] = ciphertext[full_blocks * 16 + i];
        }
        poly1305_load_block26(partial, true, t0, t1, t2, t3, t4);
        poly1305_acc_1block(acc, pk, t0, t1, t2, t3, t4);
    }

    uint8_t len_block[16] = {0};
    uint64_t ct_len64 = plaintext_len;
    len_block[8]  = (uint8_t)(ct_len64);
    len_block[9]  = (uint8_t)(ct_len64 >> 8);
    len_block[10] = (uint8_t)(ct_len64 >> 16);
    len_block[11] = (uint8_t)(ct_len64 >> 24);
    len_block[12] = (uint8_t)(ct_len64 >> 32);
    len_block[13] = (uint8_t)(ct_len64 >> 40);
    len_block[14] = (uint8_t)(ct_len64 >> 48);
    len_block[15] = (uint8_t)(ct_len64 >> 56);
    poly1305_load_block26(len_block, true, t0, t1, t2, t3, t4);
    poly1305_acc_1block(acc, pk, t0, t1, t2, t3, t4);
    poly1305_finish(acc, pk, ciphertext + plaintext_len);
    *out_len = plaintext_len + 16;
}

// ============================================================================
// Constant-time comparison (CHACHA-OPEN-01)
// ============================================================================

__device__
uint8_t chacha_constant_time_equal(const uint8_t* a, const uint8_t* b, uint32_t len) {
    uint8_t diff = 0;
    for (uint32_t i = 0; i < len; i++) {
        diff |= a[i] ^ b[i];
    }
    return (diff == 0) ? 1 : 0;
}

// ============================================================================
// ChaCha20-Poly1305 AEAD Decrypt (CHACHA-OPEN-01)
// ============================================================================

// Decrypt with ChaCha20-Poly1305 AEAD (no AAD, short message)
// Auth-before-plaintext invariant: tag verified BEFORE any plaintext written.
__device__
void chacha20_poly1305_decrypt_short(
    const uint8_t key[32],
    const uint8_t nonce[12],
    const uint8_t* ciphertext,
    uint16_t ct_len,
    const uint8_t provided_tag[16],
    uint8_t* plaintext,
    uint16_t* out_len,
    uint8_t* auth_ok)
{
    *out_len = 0;
    *auth_ok = 0;

    // Step 1: Generate Poly1305 one-time key using ChaCha20 with counter=0
    uint8_t keystream0[64];
    chacha20_block_for_keygen(key, nonce, 0, keystream0);
    uint8_t otk[32];
    for (int i = 0; i < 32; i++) otk[i] = keystream0[i];

    // Step 2: Compute expected Poly1305 tag over ciphertext
    // RFC 8439 Section 2.8 MAC input: AAD || pad(AAD) || CT || pad(CT) || len(AAD) || len(CT)
    uint8_t mac_input[64];
    uint32_t mac_len = 0;

    // AAD is empty, pad16(0) = 0 → nothing

    // Ciphertext
    for (uint16_t i = 0; i < ct_len; i++) {
        mac_input[mac_len++] = ciphertext[i];
    }

    // Ciphertext padding
    uint32_t ct_pad = pad16(ct_len);
    for (uint32_t i = 0; i < ct_pad; i++) {
        mac_input[mac_len++] = 0;
    }

    // 8-byte AAD length (= 0)
    for (int i = 0; i < 8; i++) mac_input[mac_len++] = 0;

    // 8-byte ciphertext length (little-endian)
    uint64_t ct_len64 = ct_len;
    mac_input[mac_len++] = (uint8_t)(ct_len64);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 8);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 16);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 24);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 32);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 40);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 48);
    mac_input[mac_len++] = (uint8_t)(ct_len64 >> 56);

    // Compute expected tag
    uint8_t expected_tag[16];
    smoke::poly1305::poly1305_mac(otk, mac_input, (uint16_t)mac_len, expected_tag);

    // Step 3: Constant-time tag verification (MUST happen before plaintext write)
    uint8_t tag_match = chacha_constant_time_equal(expected_tag, provided_tag, 16);

    if (!tag_match) {
        for (uint16_t i = 0; i < 64; i++) plaintext[i] = 0;
        *out_len = 0;
        *auth_ok = 0;
        return;
    }

    // Step 4: Decrypt — XOR ciphertext with ChaCha20 keystream (counter=1)
    *auth_ok = 1;
    if (ct_len > 0) {
        uint8_t keystream1[64];
        chacha20_block_for_keygen(key, nonce, 1, keystream1);
        for (uint16_t i = 0; i < ct_len; i++) {
            plaintext[i] = ciphertext[i] ^ keystream1[i];
        }
    }
    *out_len = ct_len;
}

// Persistent engine interface for ChaCha20-Poly1305 decrypt
// Input layout: nonce[12] || ciphertext[N] || tag[16]
extern "C" __device__
void chacha20_poly1305_decrypt_short_keyslot(
    const uint8_t key[32],
    const uint8_t* input,
    uint16_t input_len,
    uint8_t* output,
    uint16_t* output_len,
    uint8_t* auth_ok)
{
    *output_len = 0;
    *auth_ok = 0;

    if (input_len < 28) return;  // nonce[12] + tag[16] minimum

    const uint8_t* nonce = input;
    uint16_t ct_len = input_len - 28;
    const uint8_t* ciphertext = input + 12;
    const uint8_t* tag = input + 12 + ct_len;

    if (ct_len > 16) {
        return;  // Fail closed; never authenticate/decrypt a prefix.
    }

    chacha20_poly1305_decrypt_short(key, nonce, ciphertext, ct_len, tag,
                                     output, output_len, auth_ok);
}

// Slab variant for large payloads (CHACHA-OPEN-01)
extern "C" __device__
void chacha20_poly1305_decrypt_slab(
    const uint8_t key[32],
    const uint8_t nonce[12],
    const uint8_t* payload_slab,
    uint32_t payload_offset,
    uint32_t payload_len,       // ct_len + 16
    uint8_t* output_slab,
    uint32_t out_offset,
    uint32_t* out_len,
    uint8_t* auth_ok)
{
    *out_len = 0;
    *auth_ok = 0;

    if (payload_len < 16) return;

    const uint8_t* slab_data = payload_slab + payload_offset;
    uint32_t ct_len = payload_len - 16;
    const uint8_t* ciphertext = slab_data;
    const uint8_t* provided_tag = slab_data + ct_len;

    // Step 1: Generate OTK
    uint8_t keystream0[64];
    chacha20_block_for_keygen(key, nonce, 0, keystream0);
    uint8_t otk[32];
    for (int i = 0; i < 32; i++) otk[i] = keystream0[i];

    // Step 2: Compute Poly1305 tag over AEAD construction (CHACHA-SLAB-LARGE-01)
    // RFC 8439 §2.8: Poly1305(otk, aad || pad(aad) || ct || pad(ct) || len_aad || len_ct)
    // No AAD in v1 native mode, so: ct || pad(ct) || 0^8 || len(ct) LE64
    //
    // Uses streaming Poly1305 via poly1305_26.cuh incremental API to handle
    // arbitrary payload sizes without stack buffers.
    uint8_t expected_tag[16] = {0};
    {
        Poly1305Key26 pk;
        Poly1305Acc acc = {0, 0, 0, 0, 0};

        // Initialize key from OTK: r = otk[0:16], s = otk[16:32]
        poly1305_clamp(pk, otk);
        pk.pad0 = ld_le32(otk + 16);
        pk.pad1 = ld_le32(otk + 20);
        pk.pad2 = ld_le32(otk + 24);
        pk.pad3 = ld_le32(otk + 28);

        // Absorb ciphertext in 16-byte blocks (full blocks with hibit=1)
        uint32_t full_blocks = ct_len / 16;
        uint32_t tail = ct_len & 15;
        uint32_t t0, t1, t2, t3, t4;

        for (uint32_t b = 0; b < full_blocks; b++) {
            poly1305_load_block26(ciphertext + b * 16, true, t0, t1, t2, t3, t4);
            poly1305_acc_1block(acc, pk, t0, t1, t2, t3, t4);
        }

        // Partial final block of ciphertext (if any) — pad with zeros
        if (tail > 0) {
            uint8_t partial[16] = {0};
            for (uint32_t i = 0; i < tail; i++) partial[i] = ciphertext[full_blocks * 16 + i];
            poly1305_load_block26(partial, true, t0, t1, t2, t3, t4);
            poly1305_acc_1block(acc, pk, t0, t1, t2, t3, t4);
        }

        // Pad ciphertext to 16-byte boundary (already handled by partial block above)
        // RFC 8439: pad is implicit — the partial block is already zero-padded

        // AAD length (0) || CT length (LE64) — one 16-byte block
        uint8_t len_block[16] = {0};
        uint64_t ct_len64 = ct_len;
        len_block[8]  = (uint8_t)(ct_len64);
        len_block[9]  = (uint8_t)(ct_len64 >> 8);
        len_block[10] = (uint8_t)(ct_len64 >> 16);
        len_block[11] = (uint8_t)(ct_len64 >> 24);
        len_block[12] = (uint8_t)(ct_len64 >> 32);
        len_block[13] = (uint8_t)(ct_len64 >> 40);
        len_block[14] = (uint8_t)(ct_len64 >> 48);
        len_block[15] = (uint8_t)(ct_len64 >> 56);
        poly1305_load_block26(len_block, true, t0, t1, t2, t3, t4);
        poly1305_acc_1block(acc, pk, t0, t1, t2, t3, t4);

        // Finalize: fold carries + add pad
        poly1305_finish(acc, pk, expected_tag);
    }

    // Step 3: Constant-time tag compare
    uint8_t tag_match = chacha_constant_time_equal(expected_tag, provided_tag, 16);
    if (!tag_match) {
        *out_len = 0;
        *auth_ok = 0;
        return;
    }

    // Step 4: Decrypt with ChaCha20 keystream (counter=1+)
    *auth_ok = 1;
    uint8_t* output_ptr = output_slab + out_offset;

    uint32_t offset = 0;
    uint32_t counter = 1;
    while (offset < ct_len) {
        uint8_t ks[64];
        chacha20_block_for_keygen(key, nonce, counter, ks);
        uint32_t block_len = (ct_len - offset > 64) ? 64 : (ct_len - offset);
        for (uint32_t i = 0; i < block_len; i++) {
            output_ptr[offset + i] = ciphertext[offset + i] ^ ks[i];
        }
        offset += block_len;
        counter++;
    }

    *out_len = ct_len;
}

} // namespace aead
} // namespace smoke

#endif // SMOKE_CHACHA20_POLY1305_DEVICE_CUH
