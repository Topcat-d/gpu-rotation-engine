/**
 * Poly1305 MAC - Standalone Implementation
 *
 * RFC 7539 compliant Poly1305 message authentication code
 * Uses 26-bit limb representation for 130-bit field arithmetic
 *
 * Performance targets (RTX 2060 SUPER):
 * - Batched throughput: ≥100 Gb/s @ 16KB messages
 * - Small messages: ≥40 Gb/s @ 512B
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Toggles
#ifndef POLY1305_VEC4_ENABLE
#define POLY1305_VEC4_ENABLE 1  // Use vectorized loads for message blocks
#endif

#ifndef POLY1305_LAUNCH_BOUND
#define POLY1305_LAUNCH_BOUND 3  // Threads per SM preference
#endif

// ============================================================================
// Poly1305 Field Arithmetic (130-bit, 2^130 - 5 prime)
// ============================================================================

/**
 * Poly1305 uses a 130-bit prime field: 2^130 - 5
 * We represent 130-bit values as 5 limbs of 26 bits each:
 * h0 (26 bits) | h1 (26 bits) | h2 (26 bits) | h3 (26 bits) | h4 (26 bits)
 *
 * This allows addition without overflow in 32-bit integers
 * and simplifies modular reduction.
 */

struct Poly1305State {
    uint32_t h[5];  // Accumulator (5x 26-bit limbs)
    uint32_t r[5];  // Clamped key (5x 26-bit limbs)
    uint32_t s[4];  // Secret for final add (128 bits)
};

/**
 * Clamp r according to RFC 7539:
 * r[3], r[7], r[11], r[15] &= 15
 * r[4], r[8], r[12] &= 252
 */
__device__ __forceinline__ void poly1305_clamp_r(uint8_t r_bytes[16]) {
    r_bytes[3] &= 15;
    r_bytes[7] &= 15;
    r_bytes[11] &= 15;
    r_bytes[15] &= 15;
    r_bytes[4] &= 252;
    r_bytes[8] &= 252;
    r_bytes[12] &= 252;
}

/**
 * Convert 16-byte little-endian r to 5x 26-bit limbs
 */
__device__ __forceinline__ void poly1305_r_to_limbs(
    const uint8_t r_bytes[16],
    uint32_t r[5]
) {
    // Read as 32-bit words (little-endian)
    uint32_t r0 = r_bytes[0] | (r_bytes[1] << 8) | (r_bytes[2] << 16) | (r_bytes[3] << 24);
    uint32_t r1 = r_bytes[4] | (r_bytes[5] << 8) | (r_bytes[6] << 16) | (r_bytes[7] << 24);
    uint32_t r2 = r_bytes[8] | (r_bytes[9] << 8) | (r_bytes[10] << 16) | (r_bytes[11] << 24);
    uint32_t r3 = r_bytes[12] | (r_bytes[13] << 8) | (r_bytes[14] << 16) | (r_bytes[15] << 24);

    // Split into 26-bit limbs
    r[0] = r0 & 0x3ffffff;
    r[1] = ((r0 >> 26) | (r1 << 6)) & 0x3ffffff;
    r[2] = ((r1 >> 20) | (r2 << 12)) & 0x3ffffff;
    r[3] = ((r2 >> 14) | (r3 << 18)) & 0x3ffffff;
    r[4] = (r3 >> 8) & 0x3ffffff;
}

/**
 * Convert 16-byte message block to 5x 26-bit limbs + padding bit
 * If final block (len < 16), only use len bytes and set padding = 0
 */
__device__ __forceinline__ void poly1305_block_to_limbs(
    const uint8_t block[16],
    uint32_t limbs[5],
    int len,  // 1-16
    bool is_final
) {
    // Zero-pad if partial block
    uint8_t padded[16] = {0};
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        padded[i] = (i < len) ? block[i] : 0;
    }

    uint32_t b0 = padded[0] | (padded[1] << 8) | (padded[2] << 16) | (padded[3] << 24);
    uint32_t b1 = padded[4] | (padded[5] << 8) | (padded[6] << 16) | (padded[7] << 24);
    uint32_t b2 = padded[8] | (padded[9] << 8) | (padded[10] << 16) | (padded[11] << 24);
    uint32_t b3 = padded[12] | (padded[13] << 8) | (padded[14] << 16) | (padded[15] << 24);

    limbs[0] = b0 & 0x3ffffff;
    limbs[1] = ((b0 >> 26) | (b1 << 6)) & 0x3ffffff;
    limbs[2] = ((b1 >> 20) | (b2 << 12)) & 0x3ffffff;
    limbs[3] = ((b2 >> 14) | (b3 << 18)) & 0x3ffffff;
    limbs[4] = (b3 >> 8) & 0x3ffffff;

    // Add padding bit (2^128 for full blocks, 2^(8*len) for partial)
    if (!is_final || len == 16) {
        limbs[4] |= (1u << 24);  // Padding bit for 128 bits
    } else {
        // For partial final block, add 1 << (8*len) in limb representation
        uint32_t padding_bit_pos = len * 8;
        uint32_t limb_idx = padding_bit_pos / 26;
        uint32_t bit_offset = padding_bit_pos % 26;
        if (limb_idx < 5) {
            limbs[limb_idx] |= (1u << bit_offset);
        }
    }
}

/**
 * Multiply and reduce: h = (h + c) * r mod (2^130 - 5)
 *
 * Uses 26-bit limbs to avoid overflow during multiplication
 * Lazy reduction after multiply
 */
__device__ __forceinline__ void poly1305_multiply_add(
    uint32_t h[5],
    const uint32_t c[5],
    const uint32_t r[5]
) {
    // Add message block to accumulator
    uint64_t h0 = h[0] + (uint64_t)c[0];
    uint64_t h1 = h[1] + (uint64_t)c[1];
    uint64_t h2 = h[2] + (uint64_t)c[2];
    uint64_t h3 = h[3] + (uint64_t)c[3];
    uint64_t h4 = h[4] + (uint64_t)c[4];

    // Multiply h * r (130-bit x 130-bit = 260-bit product)
    // We compute this as 5x5 limb multiplication
    uint64_t d0 = h0 * (uint64_t)r[0] + h1 * (uint64_t)r[4] * 5 + h2 * (uint64_t)r[3] * 5 +
                  h3 * (uint64_t)r[2] * 5 + h4 * (uint64_t)r[1] * 5;
    uint64_t d1 = h0 * (uint64_t)r[1] + h1 * (uint64_t)r[0] + h2 * (uint64_t)r[4] * 5 +
                  h3 * (uint64_t)r[3] * 5 + h4 * (uint64_t)r[2] * 5;
    uint64_t d2 = h0 * (uint64_t)r[2] + h1 * (uint64_t)r[1] + h2 * (uint64_t)r[0] +
                  h3 * (uint64_t)r[4] * 5 + h4 * (uint64_t)r[3] * 5;
    uint64_t d3 = h0 * (uint64_t)r[3] + h1 * (uint64_t)r[2] + h2 * (uint64_t)r[1] +
                  h3 * (uint64_t)r[0] + h4 * (uint64_t)r[4] * 5;
    uint64_t d4 = h0 * (uint64_t)r[4] + h1 * (uint64_t)r[3] + h2 * (uint64_t)r[2] +
                  h3 * (uint64_t)r[1] + h4 * (uint64_t)r[0];

    // Propagate carries (lazy reduction)
    uint64_t carry;
    carry = d0 >> 26; d0 &= 0x3ffffff; d1 += carry;
    carry = d1 >> 26; d1 &= 0x3ffffff; d2 += carry;
    carry = d2 >> 26; d2 &= 0x3ffffff; d3 += carry;
    carry = d3 >> 26; d3 &= 0x3ffffff; d4 += carry;
    carry = d4 >> 26; d4 &= 0x3ffffff;

    // Reduce modulo 2^130 - 5 (carry from limb 4 wraps around with factor 5)
    d0 += carry * 5;
    carry = d0 >> 26; d0 &= 0x3ffffff; d1 += carry;

    h[0] = (uint32_t)d0;
    h[1] = (uint32_t)d1;
    h[2] = (uint32_t)d2;
    h[3] = (uint32_t)d3;
    h[4] = (uint32_t)d4;
}

/**
 * Finalize: reduce fully, add s, output 16-byte tag
 */
__device__ __forceinline__ void poly1305_finalize(
    uint32_t h[5],
    const uint32_t s[4],
    uint8_t tag[16]
) {
    // Fully reduce h modulo 2^130 - 5
    uint64_t h0 = h[0];
    uint64_t h1 = h[1];
    uint64_t h2 = h[2];
    uint64_t h3 = h[3];
    uint64_t h4 = h[4];

    // Final carry propagation
    uint64_t carry;
    carry = h1 >> 26; h1 &= 0x3ffffff; h2 += carry;
    carry = h2 >> 26; h2 &= 0x3ffffff; h3 += carry;
    carry = h3 >> 26; h3 &= 0x3ffffff; h4 += carry;
    carry = h4 >> 26; h4 &= 0x3ffffff; h0 += carry * 5;
    carry = h0 >> 26; h0 &= 0x3ffffff; h1 += carry;

    // Compute h + 5 (to check if h >= 2^130 - 5)
    uint64_t g0 = h0 + 5;
    carry = g0 >> 26; g0 &= 0x3ffffff;
    uint64_t g1 = h1 + carry; carry = g1 >> 26; g1 &= 0x3ffffff;
    uint64_t g2 = h2 + carry; carry = g2 >> 26; g2 &= 0x3ffffff;
    uint64_t g3 = h3 + carry; carry = g3 >> 26; g3 &= 0x3ffffff;
    uint64_t g4 = h4 + carry - (1ull << 26);

    // If g4 didn't underflow, use g; otherwise use h
    uint64_t mask = (g4 >> 63) - 1;  // All 1s if g4 >= 0, else all 0s
    h0 = (h0 & ~mask) | (g0 & mask);
    h1 = (h1 & ~mask) | (g1 & mask);
    h2 = (h2 & ~mask) | (g2 & mask);
    h3 = (h3 & ~mask) | (g3 & mask);
    h4 = (h4 & ~mask) | (g4 & mask);

    // Convert 5x 26-bit limbs to 4x 32-bit words
    uint32_t f0 = (uint32_t)((h0) | (h1 << 26));
    uint32_t f1 = (uint32_t)((h1 >> 6) | (h2 << 20));
    uint32_t f2 = (uint32_t)((h2 >> 12) | (h3 << 14));
    uint32_t f3 = (uint32_t)((h3 >> 18) | (h4 << 8));

    // Add s (little-endian 128-bit secret)
    uint64_t sum0 = (uint64_t)f0 + s[0];
    uint64_t sum1 = (uint64_t)f1 + s[1] + (sum0 >> 32);
    uint64_t sum2 = (uint64_t)f2 + s[2] + (sum1 >> 32);
    uint64_t sum3 = (uint64_t)f3 + s[3] + (sum2 >> 32);

    f0 = (uint32_t)sum0;
    f1 = (uint32_t)sum1;
    f2 = (uint32_t)sum2;
    f3 = (uint32_t)sum3;

    // Write tag (little-endian)
    tag[0] = f0; tag[1] = f0 >> 8; tag[2] = f0 >> 16; tag[3] = f0 >> 24;
    tag[4] = f1; tag[5] = f1 >> 8; tag[6] = f1 >> 16; tag[7] = f1 >> 24;
    tag[8] = f2; tag[9] = f2 >> 8; tag[10] = f2 >> 16; tag[11] = f2 >> 24;
    tag[12] = f3; tag[13] = f3 >> 8; tag[14] = f3 >> 16; tag[15] = f3 >> 24;
}

// ============================================================================
// Poly1305 MAC Kernel (Batched)
// ============================================================================

/**
 * Poly1305 MAC - one thread per message
 *
 * Each thread processes one complete message independently
 * Optimal for small-to-medium messages (64B - 16KB)
 *
 * @param messages      Input messages [batch_size, max_msg_len]
 * @param msg_lens      Message lengths [batch_size]
 * @param keys          Poly1305 keys (32 bytes each) [batch_size, 32]
 * @param tags          Output tags (16 bytes each) [batch_size, 16]
 * @param batch_size    Number of messages
 */
__launch_bounds__(256, POLY1305_LAUNCH_BOUND)
extern "C" __global__ void poly1305_mac_kernel(
    const uint8_t* __restrict__ messages,
    const int* __restrict__ msg_lens,
    const uint8_t* __restrict__ keys,
    uint8_t* __restrict__ tags,
    int batch_size,
    int max_msg_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Load key (32 bytes: 16-byte r + 16-byte s)
    const uint8_t* key = keys + tid * 32;
    uint8_t r_bytes[16], s_bytes[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        r_bytes[i] = key[i];
        s_bytes[i] = key[16 + i];
    }

    // Clamp r
    poly1305_clamp_r(r_bytes);

    // Convert r to limbs
    uint32_t r[5];
    poly1305_r_to_limbs(r_bytes, r);

    // Convert s to 32-bit words (little-endian)
    uint32_t s[4];
    s[0] = s_bytes[0] | (s_bytes[1] << 8) | (s_bytes[2] << 16) | (s_bytes[3] << 24);
    s[1] = s_bytes[4] | (s_bytes[5] << 8) | (s_bytes[6] << 16) | (s_bytes[7] << 24);
    s[2] = s_bytes[8] | (s_bytes[9] << 8) | (s_bytes[10] << 16) | (s_bytes[11] << 24);
    s[3] = s_bytes[12] | (s_bytes[13] << 8) | (s_bytes[14] << 16) | (s_bytes[15] << 24);

    // Initialize accumulator
    uint32_t h[5] = {0, 0, 0, 0, 0};

    // Process message in 16-byte blocks
    int msg_len = msg_lens[tid];
    const uint8_t* msg = messages + tid * max_msg_len;

    int num_full_blocks = msg_len / 16;
    int remaining = msg_len % 16;

    // Process full blocks
    for (int b = 0; b < num_full_blocks; b++) {
        uint32_t block_limbs[5];
        poly1305_block_to_limbs(msg + b * 16, block_limbs, 16, false);
        poly1305_multiply_add(h, block_limbs, r);
    }

    // Process final partial block (if any)
    if (remaining > 0) {
        uint32_t block_limbs[5];
        poly1305_block_to_limbs(msg + num_full_blocks * 16, block_limbs, remaining, true);
        poly1305_multiply_add(h, block_limbs, r);
    }

    // Finalize and write tag
    uint8_t tag[16];
    poly1305_finalize(h, s, tag);

    uint8_t* out_tag = tags + tid * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out_tag[i] = tag[i];
    }
}

// ============================================================================
// Kernel Launcher
// ============================================================================

extern "C" void poly1305_mac_launcher(
    const uint8_t* messages,
    const int* msg_lens,
    const uint8_t* keys,
    uint8_t* tags,
    int batch_size,
    int max_msg_len,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    poly1305_mac_kernel<<<blocks, threads, 0, stream>>>(
        messages, msg_lens, keys, tags, batch_size, max_msg_len
    );
}
