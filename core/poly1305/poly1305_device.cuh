// SPDX-License-Identifier: Apache-2.0
// Smoke Poly1305 Device Implementation
// Card 26.67: Device-side Poly1305 message authenticator
//
// Features:
// - RFC 8439 compliant Poly1305 implementation
// - One-time authenticator with 32-byte key (r[16] + s[16])
// - Produces 16-byte authentication tag
// - Short-message support (up to 32 bytes inline)
//
// Poly1305 algorithm:
// - r = key[0:16] with clamping applied
// - s = key[16:32]
// - Accumulator starts at 0
// - For each 16-byte block: acc = (acc + block) * r mod (2^130 - 5)
// - Final tag = (acc + s) mod 2^128
//
// Key clamping (RFC 8439):
// - r[3], r[7], r[11], r[15] have top 4 bits cleared
// - r[4], r[8], r[12] have bottom 2 bits cleared
//
// Implementation uses 5 x 26-bit limbs for 130-bit arithmetic

#ifndef SMOKE_POLY1305_DEVICE_CUH
#define SMOKE_POLY1305_DEVICE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace smoke {
namespace poly1305 {

// ============================================================================
// Poly1305 State
// ============================================================================

struct Poly1305State {
    uint32_t r[5];      // Clamped r in 26-bit limbs
    uint32_t h[5];      // Accumulator in 26-bit limbs
    uint32_t pad[4];    // s as 4 x 32-bit words
};

// ============================================================================
// Helper Functions
// ============================================================================

// Clamp r according to RFC 8439
__device__ __forceinline__
void clamp_r(uint8_t r[16]) {
    r[3] &= 0x0f;
    r[7] &= 0x0f;
    r[11] &= 0x0f;
    r[15] &= 0x0f;
    r[4] &= 0xfc;
    r[8] &= 0xfc;
    r[12] &= 0xfc;
}

// Initialize Poly1305 state from 32-byte key
__device__
void poly1305_init(Poly1305State* st, const uint8_t key[32]) {
    // Extract and clamp r
    uint8_t r_bytes[16];
    for (int i = 0; i < 16; i++) {
        r_bytes[i] = key[i];
    }
    clamp_r(r_bytes);

    // Load r into 26-bit limbs (little-endian)
    uint32_t t0 = ((uint32_t)r_bytes[0]) | ((uint32_t)r_bytes[1] << 8) |
                  ((uint32_t)r_bytes[2] << 16) | ((uint32_t)r_bytes[3] << 24);
    uint32_t t1 = ((uint32_t)r_bytes[4]) | ((uint32_t)r_bytes[5] << 8) |
                  ((uint32_t)r_bytes[6] << 16) | ((uint32_t)r_bytes[7] << 24);
    uint32_t t2 = ((uint32_t)r_bytes[8]) | ((uint32_t)r_bytes[9] << 8) |
                  ((uint32_t)r_bytes[10] << 16) | ((uint32_t)r_bytes[11] << 24);
    uint32_t t3 = ((uint32_t)r_bytes[12]) | ((uint32_t)r_bytes[13] << 8) |
                  ((uint32_t)r_bytes[14] << 16) | ((uint32_t)r_bytes[15] << 24);

    st->r[0] = t0 & 0x3ffffff;
    st->r[1] = ((t0 >> 26) | (t1 << 6)) & 0x3ffffff;
    st->r[2] = ((t1 >> 20) | (t2 << 12)) & 0x3ffffff;
    st->r[3] = ((t2 >> 14) | (t3 << 18)) & 0x3ffffff;
    st->r[4] = (t3 >> 8) & 0x3ffffff;

    // Initialize accumulator to zero
    st->h[0] = 0;
    st->h[1] = 0;
    st->h[2] = 0;
    st->h[3] = 0;
    st->h[4] = 0;

    // Store s (pad) as 4 x 32-bit words
    const uint8_t* s = key + 16;
    st->pad[0] = ((uint32_t)s[0]) | ((uint32_t)s[1] << 8) |
                 ((uint32_t)s[2] << 16) | ((uint32_t)s[3] << 24);
    st->pad[1] = ((uint32_t)s[4]) | ((uint32_t)s[5] << 8) |
                 ((uint32_t)s[6] << 16) | ((uint32_t)s[7] << 24);
    st->pad[2] = ((uint32_t)s[8]) | ((uint32_t)s[9] << 8) |
                 ((uint32_t)s[10] << 16) | ((uint32_t)s[11] << 24);
    st->pad[3] = ((uint32_t)s[12]) | ((uint32_t)s[13] << 8) |
                 ((uint32_t)s[14] << 16) | ((uint32_t)s[15] << 24);
}

// Process a 16-byte block (or final partial block)
// hibit = 1 for normal blocks, 0 for final block
__device__
void poly1305_blocks(Poly1305State* st, const uint8_t* m, uint32_t bytes, uint32_t hibit) {
    const uint32_t r0 = st->r[0];
    const uint32_t r1 = st->r[1];
    const uint32_t r2 = st->r[2];
    const uint32_t r3 = st->r[3];
    const uint32_t r4 = st->r[4];

    // Precompute r * 5 for reduction
    const uint32_t s1 = r1 * 5;
    const uint32_t s2 = r2 * 5;
    const uint32_t s3 = r3 * 5;
    const uint32_t s4 = r4 * 5;

    uint32_t h0 = st->h[0];
    uint32_t h1 = st->h[1];
    uint32_t h2 = st->h[2];
    uint32_t h3 = st->h[3];
    uint32_t h4 = st->h[4];

    while (bytes >= 16) {
        // Load message block as 26-bit limbs
        uint32_t t0 = ((uint32_t)m[0]) | ((uint32_t)m[1] << 8) |
                      ((uint32_t)m[2] << 16) | ((uint32_t)m[3] << 24);
        uint32_t t1 = ((uint32_t)m[4]) | ((uint32_t)m[5] << 8) |
                      ((uint32_t)m[6] << 16) | ((uint32_t)m[7] << 24);
        uint32_t t2 = ((uint32_t)m[8]) | ((uint32_t)m[9] << 8) |
                      ((uint32_t)m[10] << 16) | ((uint32_t)m[11] << 24);
        uint32_t t3 = ((uint32_t)m[12]) | ((uint32_t)m[13] << 8) |
                      ((uint32_t)m[14] << 16) | ((uint32_t)m[15] << 24);

        // h += m (with high bit)
        h0 += t0 & 0x3ffffff;
        h1 += ((t0 >> 26) | (t1 << 6)) & 0x3ffffff;
        h2 += ((t1 >> 20) | (t2 << 12)) & 0x3ffffff;
        h3 += ((t2 >> 14) | (t3 << 18)) & 0x3ffffff;
        h4 += (t3 >> 8) | (hibit << 24);

        // h = h * r mod (2^130 - 5)
        uint64_t d0 = (uint64_t)h0 * r0 + (uint64_t)h1 * s4 + (uint64_t)h2 * s3 + (uint64_t)h3 * s2 + (uint64_t)h4 * s1;
        uint64_t d1 = (uint64_t)h0 * r1 + (uint64_t)h1 * r0 + (uint64_t)h2 * s4 + (uint64_t)h3 * s3 + (uint64_t)h4 * s2;
        uint64_t d2 = (uint64_t)h0 * r2 + (uint64_t)h1 * r1 + (uint64_t)h2 * r0 + (uint64_t)h3 * s4 + (uint64_t)h4 * s3;
        uint64_t d3 = (uint64_t)h0 * r3 + (uint64_t)h1 * r2 + (uint64_t)h2 * r1 + (uint64_t)h3 * r0 + (uint64_t)h4 * s4;
        uint64_t d4 = (uint64_t)h0 * r4 + (uint64_t)h1 * r3 + (uint64_t)h2 * r2 + (uint64_t)h3 * r1 + (uint64_t)h4 * r0;

        // Partial reduction mod 2^130 - 5
        uint32_t c;
        c = (uint32_t)(d0 >> 26); h0 = (uint32_t)d0 & 0x3ffffff; d1 += c;
        c = (uint32_t)(d1 >> 26); h1 = (uint32_t)d1 & 0x3ffffff; d2 += c;
        c = (uint32_t)(d2 >> 26); h2 = (uint32_t)d2 & 0x3ffffff; d3 += c;
        c = (uint32_t)(d3 >> 26); h3 = (uint32_t)d3 & 0x3ffffff; d4 += c;
        c = (uint32_t)(d4 >> 26); h4 = (uint32_t)d4 & 0x3ffffff; h0 += c * 5;
        c = h0 >> 26; h0 &= 0x3ffffff; h1 += c;

        m += 16;
        bytes -= 16;
    }

    st->h[0] = h0;
    st->h[1] = h1;
    st->h[2] = h2;
    st->h[3] = h3;
    st->h[4] = h4;
}

// Finalize and produce tag
__device__
void poly1305_finish(Poly1305State* st, uint8_t tag[16]) {
    uint32_t h0 = st->h[0];
    uint32_t h1 = st->h[1];
    uint32_t h2 = st->h[2];
    uint32_t h3 = st->h[3];
    uint32_t h4 = st->h[4];

    // Full carry
    uint32_t c;
    c = h1 >> 26; h1 &= 0x3ffffff; h2 += c;
    c = h2 >> 26; h2 &= 0x3ffffff; h3 += c;
    c = h3 >> 26; h3 &= 0x3ffffff; h4 += c;
    c = h4 >> 26; h4 &= 0x3ffffff; h0 += c * 5;
    c = h0 >> 26; h0 &= 0x3ffffff; h1 += c;

    // Compute h + -p = h - (2^130 - 5) = h - 2^130 + 5
    uint32_t g0 = h0 + 5; c = g0 >> 26; g0 &= 0x3ffffff;
    uint32_t g1 = h1 + c; c = g1 >> 26; g1 &= 0x3ffffff;
    uint32_t g2 = h2 + c; c = g2 >> 26; g2 &= 0x3ffffff;
    uint32_t g3 = h3 + c; c = g3 >> 26; g3 &= 0x3ffffff;
    uint32_t g4 = h4 + c - (1u << 26);

    // Select h if h < p, or h + -p if h >= p
    // mask = 0xffffffff if g4 underflowed (h < p), else 0
    uint32_t mask = (g4 >> 31) - 1;  // 0 if h < p, 0xffffffff if h >= p
    g0 &= mask;
    g1 &= mask;
    g2 &= mask;
    g3 &= mask;
    g4 &= mask;
    mask = ~mask;
    h0 = (h0 & mask) | g0;
    h1 = (h1 & mask) | g1;
    h2 = (h2 & mask) | g2;
    h3 = (h3 & mask) | g3;
    h4 = (h4 & mask) | g4;

    // h = h mod 2^128 (drop high bits) + s
    // Convert to bytes: h0 | (h1 << 26) | (h2 << 52) | (h3 << 78) | (h4 << 104)
    uint64_t f0 = (uint64_t)h0 | ((uint64_t)h1 << 26) | ((uint64_t)h2 << 52);
    uint64_t f1 = ((uint64_t)h2 >> 12) | ((uint64_t)h3 << 14) | ((uint64_t)h4 << 40);

    // pad is 16 bytes little-endian stored as 4x u32: pad[0..3]
    uint64_t pad0 = (uint64_t)st->pad[0] | ((uint64_t)st->pad[1] << 32);
    uint64_t pad1 = (uint64_t)st->pad[2] | ((uint64_t)st->pad[3] << 32);

    // 128-bit add: (f1:f0) += (pad1:pad0) with proper carry
    uint64_t lo = f0 + pad0;
    uint64_t carry = (lo < f0) ? 1ull : 0ull;
    uint64_t hi = f1 + pad1 + carry;

    // Write tag (little-endian)
    tag[0] = (uint8_t)(lo);
    tag[1] = (uint8_t)(lo >> 8);
    tag[2] = (uint8_t)(lo >> 16);
    tag[3] = (uint8_t)(lo >> 24);
    tag[4] = (uint8_t)(lo >> 32);
    tag[5] = (uint8_t)(lo >> 40);
    tag[6] = (uint8_t)(lo >> 48);
    tag[7] = (uint8_t)(lo >> 56);
    tag[8] = (uint8_t)(hi);
    tag[9] = (uint8_t)(hi >> 8);
    tag[10] = (uint8_t)(hi >> 16);
    tag[11] = (uint8_t)(hi >> 24);
    tag[12] = (uint8_t)(hi >> 32);
    tag[13] = (uint8_t)(hi >> 40);
    tag[14] = (uint8_t)(hi >> 48);
    tag[15] = (uint8_t)(hi >> 56);
}

// ============================================================================
// Poly1305 MAC
// ============================================================================

// Compute Poly1305 MAC
// key: 32 bytes (r[16] + s[16])
// msg: message to authenticate
// msg_len: message length
// tag: output 16-byte tag
extern "C" __device__
void poly1305_mac(
    const uint8_t key[32],
    const uint8_t* msg,
    uint16_t msg_len,
    uint8_t tag[16])
{
    Poly1305State st;
    poly1305_init(&st, key);

    // Process full 16-byte blocks
    if (msg_len >= 16) {
        uint32_t full_blocks = msg_len & ~15u;
        poly1305_blocks(&st, msg, full_blocks, 1);
        msg += full_blocks;
        msg_len -= full_blocks;
    }

    // Process final partial block (if any)
    if (msg_len > 0) {
        uint8_t block[16] = {0};
        for (uint16_t i = 0; i < msg_len; i++) {
            block[i] = msg[i];
        }
        block[msg_len] = 1;  // Pad with 0x01

        // Load as 26-bit limbs and add to accumulator
        uint32_t t0 = ((uint32_t)block[0]) | ((uint32_t)block[1] << 8) |
                      ((uint32_t)block[2] << 16) | ((uint32_t)block[3] << 24);
        uint32_t t1 = ((uint32_t)block[4]) | ((uint32_t)block[5] << 8) |
                      ((uint32_t)block[6] << 16) | ((uint32_t)block[7] << 24);
        uint32_t t2 = ((uint32_t)block[8]) | ((uint32_t)block[9] << 8) |
                      ((uint32_t)block[10] << 16) | ((uint32_t)block[11] << 24);
        uint32_t t3 = ((uint32_t)block[12]) | ((uint32_t)block[13] << 8) |
                      ((uint32_t)block[14] << 16) | ((uint32_t)block[15] << 24);

        st.h[0] += t0 & 0x3ffffff;
        st.h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3ffffff;
        st.h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3ffffff;
        st.h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3ffffff;
        st.h[4] += (t3 >> 8);  // No high bit for final block

        // h = h * r mod (2^130 - 5)
        const uint32_t r0 = st.r[0], r1 = st.r[1], r2 = st.r[2], r3 = st.r[3], r4 = st.r[4];
        const uint32_t s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;
        uint32_t h0 = st.h[0], h1 = st.h[1], h2 = st.h[2], h3 = st.h[3], h4 = st.h[4];

        uint64_t d0 = (uint64_t)h0 * r0 + (uint64_t)h1 * s4 + (uint64_t)h2 * s3 + (uint64_t)h3 * s2 + (uint64_t)h4 * s1;
        uint64_t d1 = (uint64_t)h0 * r1 + (uint64_t)h1 * r0 + (uint64_t)h2 * s4 + (uint64_t)h3 * s3 + (uint64_t)h4 * s2;
        uint64_t d2 = (uint64_t)h0 * r2 + (uint64_t)h1 * r1 + (uint64_t)h2 * r0 + (uint64_t)h3 * s4 + (uint64_t)h4 * s3;
        uint64_t d3 = (uint64_t)h0 * r3 + (uint64_t)h1 * r2 + (uint64_t)h2 * r1 + (uint64_t)h3 * r0 + (uint64_t)h4 * s4;
        uint64_t d4 = (uint64_t)h0 * r4 + (uint64_t)h1 * r3 + (uint64_t)h2 * r2 + (uint64_t)h3 * r1 + (uint64_t)h4 * r0;

        uint32_t c;
        c = (uint32_t)(d0 >> 26); st.h[0] = (uint32_t)d0 & 0x3ffffff; d1 += c;
        c = (uint32_t)(d1 >> 26); st.h[1] = (uint32_t)d1 & 0x3ffffff; d2 += c;
        c = (uint32_t)(d2 >> 26); st.h[2] = (uint32_t)d2 & 0x3ffffff; d3 += c;
        c = (uint32_t)(d3 >> 26); st.h[3] = (uint32_t)d3 & 0x3ffffff; d4 += c;
        c = (uint32_t)(d4 >> 26); st.h[4] = (uint32_t)d4 & 0x3ffffff; st.h[0] += c * 5;
        c = st.h[0] >> 26; st.h[0] &= 0x3ffffff; st.h[1] += c;
    }

    // Finalize
    poly1305_finish(&st, tag);
}

// ============================================================================
// Persistent Engine Interface
// ============================================================================

// Short-message MAC for persistent engine
// Input layout: message[up to 32 bytes]
// Output: 16-byte tag
//
// We use the keyslot's private_d[32] as the Poly1305 key
extern "C" __device__
void poly1305_mac_short_keyslot(
    const uint8_t key[32],     // Key from keyslot (r[16] + s[16])
    const uint8_t* input,      // Message to authenticate
    uint16_t input_len,        // Message length
    uint8_t* output,           // 16-byte tag
    uint16_t* output_len)      // Output length (always 16)
{
    poly1305_mac(key, input, input_len, output);
    *output_len = 16;
}

} // namespace poly1305
} // namespace smoke

#endif // SMOKE_POLY1305_DEVICE_CUH
