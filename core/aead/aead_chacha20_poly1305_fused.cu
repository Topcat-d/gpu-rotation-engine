/**
 * ChaCha20-Poly1305 AEAD - Fused Warp-Per-Message Kernel
 *
 * Single-pass AEAD: encrypts and MACs in one kernel launch
 * - Warp generates ChaCha20 keystream
 * - XORs plaintext → ciphertext
 * - Feeds ciphertext blocks immediately to Poly1305 MAC
 * - Finalizes tag with AAD length || CT length
 *
 * Performance: Target ~1,000+ Gb/s (same as standalone kernels)
 * Register budget: ~48 registers (fusion overhead)
 *
 * RFC 8439 compliant:
 * - ChaCha20 counter starts at 1 for encryption
 * - Poly1305 key (r||s) from ChaCha20 counter=0
 * - MAC = Poly1305(AAD || pad16 || CT || pad16 || len(AAD) || len(CT))
 *
 * Author: PyTorch Crypto Project
 * Date: October 25, 2025
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32
#define CHACHA_BLOCK_SIZE 64  // ChaCha20: 64 bytes
#define POLY_BLOCK_SIZE 16    // Poly1305: 16 bytes

// ===== Phase Q-1 toggles =====
#ifndef LAUNCH_TUNE_BOUND
#define LAUNCH_TUNE_BOUND 3   // try 3 (was 4). Set to 4 to revert.
#endif

// Try handling 32B per lane when aligned; falls back to 16B path automatically
#ifndef AEAD_VEC32_ENABLE
#define AEAD_VEC32_ENABLE 1
#endif

// ============================================================================
// Streaming (.cs) vector loads/stores (Turing+, sm_70+)
// Reduces L2 cache pollution for one-time-use data (plaintext/ciphertext)
// ============================================================================
__device__ __forceinline__ uint4 ld_cs_uint4(const uint4* p) {
#if __CUDA_ARCH__ >= 700
  uint4 v;
  asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
#else
  return *p;  // Fallback for older architectures
#endif
}

__device__ __forceinline__ void st_cs_uint4(uint4* p, const uint4 v) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
#else
  *p = v;  // Fallback for older architectures
#endif
}

// ============================================================================
// ChaCha20 Core (same as standalone)
// ============================================================================

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return __funnelshift_l(x, x, n);  // GPU intrinsic
}

__device__ __forceinline__ void quarterround(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

__device__ __forceinline__ void chacha20_block(
    uint32_t out[16],
    const uint32_t key[8],
    const uint32_t nonce[3],
    uint32_t counter
) {
    // Initialize state
    uint32_t x[16];

    // Constants
    x[0] = 0x61707865;
    x[1] = 0x3320646e;
    x[2] = 0x79622d32;
    x[3] = 0x6b206574;

    // Key
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x[4 + i] = key[i];
    }

    // Counter + nonce
    x[12] = counter;
    x[13] = nonce[0];
    x[14] = nonce[1];
    x[15] = nonce[2];

    // Save initial state
    uint32_t initial[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        initial[i] = x[i];
    }

    // 20 rounds (10 double rounds)
    #pragma unroll 10
    for (int i = 0; i < 10; i++) {
        // Column rounds
        quarterround(x[0], x[4], x[8], x[12]);
        quarterround(x[1], x[5], x[9], x[13]);
        quarterround(x[2], x[6], x[10], x[14]);
        quarterround(x[3], x[7], x[11], x[15]);

        // Diagonal rounds
        quarterround(x[0], x[5], x[10], x[15]);
        quarterround(x[1], x[6], x[11], x[12]);
        quarterround(x[2], x[7], x[8], x[13]);
        quarterround(x[3], x[4], x[9], x[14]);
    }

    // Add initial state
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out[i] = x[i] + initial[i];
    }
}

// ============================================================================
// Poly1305 Core (26-bit limbs, poly1305-donna style)
// ============================================================================

#define LIMB_MASK 0x3ffffffu

// LE64 load helper
__device__ __forceinline__ uint64_t load64_le(const uint8_t *p) {
  return (uint64_t)p[0]        | ((uint64_t)p[1] << 8)  | ((uint64_t)p[2] << 16) |
         ((uint64_t)p[3] << 24)| ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
         ((uint64_t)p[6] << 48)| ((uint64_t)p[7] << 56);
}

// Split a 16-byte block -> 5 limbs (adds the implicit 1 bit: m4 += 1<<24)
__device__ __forceinline__
void poly1305_load_block_26(const uint8_t b[16],
                            uint32_t &m0, uint32_t &m1, uint32_t &m2,
                            uint32_t &m3, uint32_t &m4) {
  uint64_t lo = load64_le(b + 0);   // bytes 0..7
  uint64_t hi = load64_le(b + 8);   // bytes 8..15

  m0 = (uint32_t)( ( lo              ) & LIMB_MASK );
  m1 = (uint32_t)( ( lo >> 26        ) & LIMB_MASK );
  m2 = (uint32_t)( ((lo >> 52) | (hi << 12)) & LIMB_MASK );
  m3 = (uint32_t)( ( hi >> 14        ) & LIMB_MASK );
  m4 = (uint32_t)( ( hi >> 40        )          );      // ≤ 24 bits
  m4 += (1u << 24);                                     // implicit 1 bit
}

// Clamp r at the byte level (RFC 7539), then split with the SAME mapping
__device__ __forceinline__
void poly1305_clamp_r_26(const uint8_t rbytes[16],
                         uint32_t &r0, uint32_t &r1, uint32_t &r2,
                         uint32_t &r3, uint32_t &r4,
                         uint32_t &r1_5, uint32_t &r2_5,
                         uint32_t &r3_5, uint32_t &r4_5) {
  uint8_t r[16];
  #pragma unroll
  for (int i=0;i<16;i++) r[i] = rbytes[i];

  // byte-level clamp per RFC 7539 / donna
  r[3]  &= 15;   r[7]  &= 15;   r[11] &= 15;   r[15] &= 15;
  r[4]  &= 252;  r[8]  &= 252;  r[12] &= 252;

  // reuse the same limb splitter (but WITHOUT the implicit 1 bit!)
  uint64_t lo = load64_le(r + 0);
  uint64_t hi = load64_le(r + 8);

  r0 = (uint32_t)( ( lo              ) & LIMB_MASK );
  r1 = (uint32_t)( ( lo >> 26        ) & LIMB_MASK );
  r2 = (uint32_t)( ((lo >> 52) | (hi << 12)) & LIMB_MASK );
  r3 = (uint32_t)( ( hi >> 14        ) & LIMB_MASK );
  r4 = (uint32_t)( ( hi >> 40        )          );      // ≤ 24 bits

  r1_5 = r1 * 5u; r2_5 = r2 * 5u; r3_5 = r3 * 5u; r4_5 = r4 * 5u;
}

// One Poly1305 block step: h = (h + m) * r mod (2^130-5)
// h0..h4, r0..r4, r1_5..r4_5 are 26-bit; m0..m4 from load above
__device__ __forceinline__
void poly1305_block_26(uint32_t &h0, uint32_t &h1, uint32_t &h2,
                       uint32_t &h3, uint32_t &h4,
                       const uint32_t r0, const uint32_t r1,
                       const uint32_t r2, const uint32_t r3,
                       const uint32_t r4,
                       const uint32_t r1_5, const uint32_t r2_5,
                       const uint32_t r3_5, const uint32_t r4_5,
                       const uint32_t m0, const uint32_t m1,
                       const uint32_t m2, const uint32_t m3,
                       const uint32_t m4) {
  uint64_t t0, t1, t2, t3, t4;
  uint64_t c;

  // (h + m)
  uint64_t x0 = (uint64_t)h0 + m0;
  uint64_t x1 = (uint64_t)h1 + m1;
  uint64_t x2 = (uint64_t)h2 + m2;
  uint64_t x3 = (uint64_t)h3 + m3;
  uint64_t x4 = (uint64_t)h4 + m4;

  // multiply with 5-fold trick
  t0 = x0*r0 + x1*r4_5 + x2*r3_5 + x3*r2_5 + x4*r1_5;
  t1 = x0*r1 + x1*r0   + x2*r4_5 + x3*r3_5 + x4*r2_5;
  t2 = x0*r2 + x1*r1   + x2*r0   + x3*r4_5 + x4*r3_5;
  t3 = x0*r3 + x1*r2   + x2*r1   + x3*r0   + x4*r4_5;
  t4 = x0*r4 + x1*r3   + x2*r2   + x3*r1   + x4*r0;

  // carry propagate to 26-bit limbs
  c = (t0 >> 26); h0 = (uint32_t)(t0 & LIMB_MASK); t1 += c;
  c = (t1 >> 26); h1 = (uint32_t)(t1 & LIMB_MASK); t2 += c;
  c = (t2 >> 26); h2 = (uint32_t)(t2 & LIMB_MASK); t3 += c;
  c = (t3 >> 26); h3 = (uint32_t)(t3 & LIMB_MASK); t4 += c;
  c = (t4 >> 26); h4 = (uint32_t)(t4 & LIMB_MASK);
  h0 += (uint32_t)(c * 5u);
  c = h0 >> 26; h0 &= LIMB_MASK; h1 += (uint32_t)c;
}

// Final reduction to canonical and add s to form 128-bit tag
__device__ __forceinline__
void poly1305_finish_26(uint32_t h0, uint32_t h1, uint32_t h2,
                        uint32_t h3, uint32_t h4,
                        const uint8_t sbytes[16],
                        uint64_t &tag_lo, uint64_t &tag_hi) {
  // g = h + 5 (mod 2^130)
  uint64_t c;
  uint32_t g0 = h0 + 5u;     c = g0 >> 26; g0 &= LIMB_MASK;
  uint32_t g1 = h1 + (uint32_t)c; c = g1 >> 26; g1 &= LIMB_MASK;
  uint32_t g2 = h2 + (uint32_t)c; c = g2 >> 26; g2 &= LIMB_MASK;
  uint32_t g3 = h3 + (uint32_t)c; c = g3 >> 26; g3 &= LIMB_MASK;
  uint32_t g4 = h4 + (uint32_t)c - (1u << 26);

  // constant-time select: if g4 underflowed, use h; else use g
  uint32_t mask = (g4 >> 31) - 1u; // all 1s if non-underflow, else 0
  h0 = (h0 & ~mask) | (g0 & mask);
  h1 = (h1 & ~mask) | (g1 & mask);
  h2 = (h2 & ~mask) | (g2 & mask);
  h3 = (h3 & ~mask) | (g3 & mask);
  h4 = (h4 & ~mask) | ((g4 + (1u<<26)) & mask); // re-add 2^26 for canonical

  // serialize to 128-bit little-endian
  uint64_t f0 = ((uint64_t)h0)        | ((uint64_t)h1 << 26) | ((uint64_t)h2 << 52);
  uint64_t f1 = ((uint64_t)(h2 >> 12))| ((uint64_t)h3 << 14) | ((uint64_t)h4 << 40);

  // add s (little-endian)
  uint64_t s0 =  (uint64_t) sbytes[0]       | ((uint64_t)sbytes[1] << 8)
               | ((uint64_t) sbytes[2] <<16)| ((uint64_t)sbytes[3] <<24)
               | ((uint64_t) sbytes[4] <<32)| ((uint64_t)sbytes[5] <<40)
               | ((uint64_t) sbytes[6] <<48)| ((uint64_t)sbytes[7] <<56);
  uint64_t s1 =  (uint64_t) sbytes[8]       | ((uint64_t)sbytes[9] << 8)
               | ((uint64_t)sbytes[10]<<16) | ((uint64_t)sbytes[11]<<24)
               | ((uint64_t)sbytes[12]<<32) | ((uint64_t)sbytes[13]<<40)
               | ((uint64_t)sbytes[14]<<48) | ((uint64_t)sbytes[15]<<56);

  tag_lo = f0 + s0;
  uint64_t carry = (tag_lo < f0);
  tag_hi = f1 + s1 + carry;

  // Debug: Print intermediate values (enable for debugging only)
  #ifdef POLY_DEBUG
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[GPU poly_finish] h0=0x%07x h1=0x%07x h2=0x%07x h3=0x%07x h4=0x%07x\n",
           h0, h1, h2, h3, h4);
    printf("[GPU poly_finish] f0=0x%016llx f1=0x%016llx\n", f0, f1);
    printf("[GPU poly_finish] s0=0x%016llx s1=0x%016llx\n", s0, s1);
    printf("[GPU poly_finish] carry=%llu\n", carry);
    printf("[GPU poly_finish] tag_lo=0x%016llx tag_hi=0x%016llx\n", tag_lo, tag_hi);
  }
  #endif
}

// Process a partial block (< 16 bytes) with zero-padding
__device__ __forceinline__
void poly1305_process_partial_block(const uint8_t* src, int nbytes,
                                     uint32_t &h0, uint32_t &h1, uint32_t &h2,
                                     uint32_t &h3, uint32_t &h4,
                                     const uint32_t r0, const uint32_t r1, const uint32_t r2,
                                     const uint32_t r3, const uint32_t r4,
                                     const uint32_t r1_5, const uint32_t r2_5,
                                     const uint32_t r3_5, const uint32_t r4_5) {
    uint8_t tmp[16] = {0};
    #pragma unroll
    for (int i = 0; i < nbytes; ++i) {
        tmp[i] = src[i];
    }
    uint32_t m0, m1, m2, m3, m4;
    poly1305_load_block_26(tmp, m0, m1, m2, m3, m4);  // Adds implicit 1 bit
    poly1305_block_26(h0, h1, h2, h3, h4, r0, r1, r2, r3, r4,
                      r1_5, r2_5, r3_5, r4_5, m0, m1, m2, m3, m4);
}

// ============================================================================
// Debug Tracing
// ============================================================================

struct Poly1305Trace {
    uint32_t h0, h1, h2, h3, h4;
};

// ============================================================================
// Fused AEAD Kernel
// ============================================================================

__launch_bounds__(256, LAUNCH_TUNE_BOUND)
extern "C" __global__ void aead_chacha20_poly1305_fused(
    const uint8_t*  __restrict__ plaintext,
    uint8_t*        __restrict__ ciphertext,
    uint8_t*        __restrict__ tags,
    const uint32_t* __restrict__ keys,
    const uint32_t* __restrict__ nonces,
    const uint8_t*  __restrict__ aad_data,     // Additional authenticated data
    const int*      __restrict__ tile_map,
    int num_tiles,
    int msg_size_bytes,
    int aad_len,  // AAD length in bytes (0 = no AAD)
    Poly1305Trace*  __restrict__ trace_buffer,  // Debug: trace h values
    int trace_capacity  // Max trace entries per message
) {
    const int warps_per_cta = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int tile_id = blockIdx.x * warps_per_cta + warp_id;

    if (tile_id >= num_tiles) return;

    const int msg_id = tile_map[tile_id * 2 + 0];
    const int start_block = tile_map[tile_id * 2 + 1];

    // Step 1: Lane 0 loads key and nonce, broadcasts to warp
    uint32_t key[8];
    uint32_t nonce[3];

    if (lane_id == 0) {
        for (int i = 0; i < 8; i++) {
            key[i] = keys[msg_id * 8 + i];
        }
        for (int i = 0; i < 3; i++) {
            nonce[i] = nonces[msg_id * 3 + i];
        }
    }

    // Broadcast key and nonce
    for (int i = 0; i < 8; i++) {
        key[i] = __shfl_sync(0xFFFFFFFF, key[i], 0);
    }
    for (int i = 0; i < 3; i++) {
        nonce[i] = __shfl_sync(0xFFFFFFFF, nonce[i], 0);
    }

    // Step 2: Generate Poly1305 key (r||s) from ChaCha20 counter=0 (only lane 0)
    uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0;
    uint32_t r1_5 = 0, r2_5 = 0, r3_5 = 0, r4_5 = 0;
    uint32_t s_u32[4];  // Store s as uint32 for easier broadcast

    if (lane_id == 0) {
        uint32_t poly_key_block[16];
        chacha20_block(poly_key_block, key, nonce, 0);

        // Extract OTK as bytes (r || s)
        uint8_t otk[32];
        for (int i = 0; i < 8; i++) {
            uint32_t word = poly_key_block[i];
            otk[i*4 + 0] = (word >> 0) & 0xFF;
            otk[i*4 + 1] = (word >> 8) & 0xFF;
            otk[i*4 + 2] = (word >> 16) & 0xFF;
            otk[i*4 + 3] = (word >> 24) & 0xFF;
        }

        // Clamp r and split to 26-bit limbs (with r*5 precomputation)
        poly1305_clamp_r_26(otk, r0, r1, r2, r3, r4, r1_5, r2_5, r3_5, r4_5);

        // Extract s as uint32 words for easier broadcast
        s_u32[0] = poly_key_block[4];
        s_u32[1] = poly_key_block[5];
        s_u32[2] = poly_key_block[6];
        s_u32[3] = poly_key_block[7];
    }

    // Broadcast r, r*5, and s to warp
    r0 = __shfl_sync(0xFFFFFFFF, r0, 0);
    r1 = __shfl_sync(0xFFFFFFFF, r1, 0);
    r2 = __shfl_sync(0xFFFFFFFF, r2, 0);
    r3 = __shfl_sync(0xFFFFFFFF, r3, 0);
    r4 = __shfl_sync(0xFFFFFFFF, r4, 0);
    r1_5 = __shfl_sync(0xFFFFFFFF, r1_5, 0);
    r2_5 = __shfl_sync(0xFFFFFFFF, r2_5, 0);
    r3_5 = __shfl_sync(0xFFFFFFFF, r3_5, 0);
    r4_5 = __shfl_sync(0xFFFFFFFF, r4_5, 0);
    for (int i = 0; i < 4; i++) {
        s_u32[i] = __shfl_sync(0xFFFFFFFF, s_u32[i], 0);
    }

    // Convert s to bytes for poly1305_finish_26
    uint8_t s_bytes[16];
    for (int i = 0; i < 4; i++) {
        s_bytes[i*4 + 0] = (s_u32[i] >> 0) & 0xFF;
        s_bytes[i*4 + 1] = (s_u32[i] >> 8) & 0xFF;
        s_bytes[i*4 + 2] = (s_u32[i] >> 16) & 0xFF;
        s_bytes[i*4 + 3] = (s_u32[i] >> 24) & 0xFF;
    }

    // Step 3: Initialize Poly1305 accumulator (lane 0 only, serial processing)
    uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0, h4 = 0;
    int trace_index = 0;  // Track trace entries for this message

    // Step 3a: Process AAD blocks (lane 0 only, serial before encryption)
    if (lane_id == 0 && aad_len > 0) {
        const uint8_t* aad_ptr = aad_data + msg_id * aad_len;
        int aad_full_blocks = aad_len / POLY_BLOCK_SIZE;
        int aad_tail = aad_len % POLY_BLOCK_SIZE;

        // Process full 16-byte AAD blocks
        for (int i = 0; i < aad_full_blocks; ++i) {
            uint32_t m0, m1, m2, m3, m4;
            poly1305_load_block_26(aad_ptr + i * POLY_BLOCK_SIZE, m0, m1, m2, m3, m4);
            poly1305_block_26(h0, h1, h2, h3, h4, r0, r1, r2, r3, r4,
                              r1_5, r2_5, r3_5, r4_5, m0, m1, m2, m3, m4);
        }

        // Process partial AAD block (with zero-padding)
        if (aad_tail > 0) {
            poly1305_process_partial_block(aad_ptr + aad_full_blocks * POLY_BLOCK_SIZE, aad_tail,
                                           h0, h1, h2, h3, h4, r0, r1, r2, r3, r4,
                                           r1_5, r2_5, r3_5, r4_5);
        }
    }

    // Sync to ensure Poly1305 accumulator is ready before ciphertext processing
    __syncwarp();

    // Step 4: Process this tile's blocks (warp-parallel ChaCha20 + Poly1305)
    const int blocks_per_tile = 32;  // Warp size
    const int total_blocks = (msg_size_bytes + CHACHA_BLOCK_SIZE - 1) / CHACHA_BLOCK_SIZE;

    // Each lane processes ONE ChaCha20 block
    const int block_id = start_block + lane_id;

    if (block_id < total_blocks) {
        // Generate ChaCha20 keystream for this block (counter starts at 1)
        uint32_t keystream[16];
        chacha20_block(keystream, key, nonce, block_id + 1);

        // XOR plaintext with keystream → ciphertext
        const uint8_t* pt = plaintext + msg_id * msg_size_bytes + block_id * CHACHA_BLOCK_SIZE;
        uint8_t* ct = ciphertext + msg_id * msg_size_bytes + block_id * CHACHA_BLOCK_SIZE;

        uint32_t* ks_bytes = (uint32_t*)keystream;
        uint4* pt_vec = (uint4*)pt;
        uint4* ct_vec = (uint4*)ct;

#if AEAD_VEC32_ENABLE
        // Phase Q-1: 32B vector path (process 2x uint4 per iteration)
        // Check alignment for 32B loads
        const bool vec32_ok = (((uintptr_t)pt % 32u) == 0u) &&
                              (((uintptr_t)ct % 32u) == 0u) &&
                              (CHACHA_BLOCK_SIZE >= 32);

        if (vec32_ok) {
            // Process 2x uint4 (32B) per iteration → 2 iterations for 64B block
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                // Load 2x uint4 = 32B plaintext
                uint4 p0 = ld_cs_uint4(&pt_vec[i * 2 + 0]);
                uint4 p1 = ld_cs_uint4(&pt_vec[i * 2 + 1]);

                // Prepare 2x uint4 = 32B keystream
                uint4 k0, k1;
                k0.x = ks_bytes[(i * 2 + 0) * 4 + 0];
                k0.y = ks_bytes[(i * 2 + 0) * 4 + 1];
                k0.z = ks_bytes[(i * 2 + 0) * 4 + 2];
                k0.w = ks_bytes[(i * 2 + 0) * 4 + 3];
                k1.x = ks_bytes[(i * 2 + 1) * 4 + 0];
                k1.y = ks_bytes[(i * 2 + 1) * 4 + 1];
                k1.z = ks_bytes[(i * 2 + 1) * 4 + 2];
                k1.w = ks_bytes[(i * 2 + 1) * 4 + 3];

                // XOR 32B
                uint4 c0, c1;
                c0.x = p0.x ^ k0.x; c0.y = p0.y ^ k0.y;
                c0.z = p0.z ^ k0.z; c0.w = p0.w ^ k0.w;
                c1.x = p1.x ^ k1.x; c1.y = p1.y ^ k1.y;
                c1.z = p1.z ^ k1.z; c1.w = p1.w ^ k1.w;

                // Store 32B ciphertext
                st_cs_uint4(&ct_vec[i * 2 + 0], c0);
                st_cs_uint4(&ct_vec[i * 2 + 1], c1);
            }
        } else
#endif
        {
            // Fallback: 16B vector path (4x uint4 = 64 bytes)
            // Use streaming loads/stores to avoid L2 pollution (one-time-use data)
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint4 plain = ld_cs_uint4(&pt_vec[i]);  // Streaming load
                uint4 ks;
                ks.x = ks_bytes[i * 4 + 0];
                ks.y = ks_bytes[i * 4 + 1];
                ks.z = ks_bytes[i * 4 + 2];
                ks.w = ks_bytes[i * 4 + 3];

                uint4 cipher;
                cipher.x = plain.x ^ ks.x;
                cipher.y = plain.y ^ ks.y;
                cipher.z = plain.z ^ ks.z;
                cipher.w = plain.w ^ ks.w;

                st_cs_uint4(&ct_vec[i], cipher);  // Streaming store
            }
        }

    }

    // CRITICAL: Wait for all lanes to finish writing ciphertext before Poly1305 reads it
    __syncwarp();

    // ========================================================================
    // Poly1305 Processing (ONLY LANE 0 - Poly1305 is serial!)
    // ========================================================================
    // CRITICAL: Poly1305 is a multiplicative chain, NOT additive.
    // We cannot sum accumulators from different lanes - must process serially.
    if (lane_id == 0) {
        // Process ALL ciphertext blocks for this message
        for (int blk = 0; blk < total_blocks; blk++) {
            // Read ciphertext for this ChaCha20 block
            const uint8_t* ct_block = ciphertext + msg_id * msg_size_bytes + blk * CHACHA_BLOCK_SIZE;

            int block_start_byte = blk * CHACHA_BLOCK_SIZE;
            int block_end_byte = (block_start_byte + CHACHA_BLOCK_SIZE < msg_size_bytes)
                                 ? block_start_byte + CHACHA_BLOCK_SIZE
                                 : msg_size_bytes;
            int bytes_in_block = block_end_byte - block_start_byte;
            int poly_blocks_in_chacha_block = (bytes_in_block + POLY_BLOCK_SIZE - 1) / POLY_BLOCK_SIZE;

            for (int poly_blk = 0; poly_blk < poly_blocks_in_chacha_block; poly_blk++) {
                int poly_start_byte = poly_blk * POLY_BLOCK_SIZE;
                int poly_end_byte = (poly_start_byte + POLY_BLOCK_SIZE < bytes_in_block)
                                    ? poly_start_byte + POLY_BLOCK_SIZE
                                    : bytes_in_block;
                int bytes_in_poly_block = poly_end_byte - poly_start_byte;

                // Prepare 16-byte block (zero-padded if partial)
                uint8_t poly_input[16] = {0};
                for (int i = 0; i < bytes_in_poly_block; i++) {
                    poly_input[i] = ct_block[poly_start_byte + i];
                }

                // Load block to limbs (auto-adds implicit 1 bit at 2^128)
                uint32_t m0, m1, m2, m3, m4;
                poly1305_load_block_26(poly_input, m0, m1, m2, m3, m4);

                // Process block: h = (h + m) * r mod (2^130-5)
                poly1305_block_26(h0, h1, h2, h3, h4, r0, r1, r2, r3, r4,
                                  r1_5, r2_5, r3_5, r4_5, m0, m1, m2, m3, m4);

                // Trace: Record h after this block
                if (trace_buffer != nullptr && trace_index < trace_capacity) {
                    int trace_offset = msg_id * trace_capacity + trace_index;
                    trace_buffer[trace_offset].h0 = h0;
                    trace_buffer[trace_offset].h1 = h1;
                    trace_buffer[trace_offset].h2 = h2;
                    trace_buffer[trace_offset].h3 = h3;
                    trace_buffer[trace_offset].h4 = h4;
                    trace_index++;
                }
            }
        }
    }

    // Step 6: Lane 0 finalizes tag
    if (lane_id == 0 && tile_id < num_tiles) {
        // For first tile of each message, finalize tag
        // (In warp-per-message, first tile handles tag computation)
        if (start_block == 0) {
            // ================================================================
            // Poly1305 length block: [ lenAD (LE64) | lenCT (LE64) ]
            // RFC 7539 §2.8.1: this block is processed WITHOUT the 1<<128 bit.
            // ================================================================
            uint64_t lenAD = (uint64_t)aad_len;  // bytes, not bits!
            uint64_t lenCT = (uint64_t)msg_size_bytes;  // UNPADDED bytes!

            // Build 16-byte length block as bytes
            uint8_t len_block[16];
            len_block[0] = (lenAD >> 0) & 0xFF;
            len_block[1] = (lenAD >> 8) & 0xFF;
            len_block[2] = (lenAD >> 16) & 0xFF;
            len_block[3] = (lenAD >> 24) & 0xFF;
            len_block[4] = (lenAD >> 32) & 0xFF;
            len_block[5] = (lenAD >> 40) & 0xFF;
            len_block[6] = (lenAD >> 48) & 0xFF;
            len_block[7] = (lenAD >> 56) & 0xFF;
            len_block[8] = (lenCT >> 0) & 0xFF;
            len_block[9] = (lenCT >> 8) & 0xFF;
            len_block[10] = (lenCT >> 16) & 0xFF;
            len_block[11] = (lenCT >> 24) & 0xFF;
            len_block[12] = (lenCT >> 32) & 0xFF;
            len_block[13] = (lenCT >> 40) & 0xFF;
            len_block[14] = (lenCT >> 48) & 0xFF;
            len_block[15] = (lenCT >> 56) & 0xFF;

            // Load length block to limbs (includes implicit 1 bit like all other blocks)
            uint32_t m0, m1, m2, m3, m4;
            poly1305_load_block_26(len_block, m0, m1, m2, m3, m4);

            // Process length block
            poly1305_block_26(h0, h1, h2, h3, h4, r0, r1, r2, r3, r4,
                              r1_5, r2_5, r3_5, r4_5, m0, m1, m2, m3, m4);

            // Trace: Record h after length block
            if (trace_buffer != nullptr && trace_index < trace_capacity) {
                int trace_offset = msg_id * trace_capacity + trace_index;
                trace_buffer[trace_offset].h0 = h0;
                trace_buffer[trace_offset].h1 = h1;
                trace_buffer[trace_offset].h2 = h2;
                trace_buffer[trace_offset].h3 = h3;
                trace_buffer[trace_offset].h4 = h4;
                trace_index++;
            }

            // Final reduction and tag generation (poly1305-donna style)
            uint64_t tag_lo, tag_hi;
            poly1305_finish_26(h0, h1, h2, h3, h4, s_bytes, tag_lo, tag_hi);

            // Write tag (16 bytes, little-endian)
            uint8_t* tag = tags + msg_id * 16;
            tag[0] = (tag_lo >> 0) & 0xFF;
            tag[1] = (tag_lo >> 8) & 0xFF;
            tag[2] = (tag_lo >> 16) & 0xFF;
            tag[3] = (tag_lo >> 24) & 0xFF;
            tag[4] = (tag_lo >> 32) & 0xFF;
            tag[5] = (tag_lo >> 40) & 0xFF;
            tag[6] = (tag_lo >> 48) & 0xFF;
            tag[7] = (tag_lo >> 56) & 0xFF;
            tag[8] = (tag_hi >> 0) & 0xFF;
            tag[9] = (tag_hi >> 8) & 0xFF;
            tag[10] = (tag_hi >> 16) & 0xFF;
            tag[11] = (tag_hi >> 24) & 0xFF;
            tag[12] = (tag_hi >> 32) & 0xFF;
            tag[13] = (tag_hi >> 40) & 0xFF;
            tag[14] = (tag_hi >> 48) & 0xFF;
            tag[15] = (tag_hi >> 56) & 0xFF;
        }
    }
}

// ============================================================================
// Launcher
// ============================================================================

extern "C" void aead_chacha20_poly1305_fused_launcher(
    const uint8_t* plaintext,
    uint8_t* ciphertext,
    uint8_t* tags,
    const uint32_t* keys,
    const uint32_t* nonces,
    const uint8_t* aad_data,
    const int* tile_map,
    int num_tiles,
    int msg_size_bytes,
    int aad_len,
    cudaStream_t stream,
    Poly1305Trace* trace_buffer,  // Optional: nullptr to disable
    int trace_capacity  // Max traces per message
) {
    const int threads_per_block = 256;  // 8 warps per block
    const int num_blocks = (num_tiles + 7) / 8;

    aead_chacha20_poly1305_fused<<<num_blocks, threads_per_block, 0, stream>>>(
        plaintext, ciphertext, tags,
        keys, nonces, aad_data, tile_map,
        num_tiles, msg_size_bytes, aad_len,
        trace_buffer, trace_capacity
    );
}
