/**
 * ChaCha20-Poly1305 AEAD - Optimized Fused Kernel
 *
 * Optimizations Applied:
 * 1. __launch_bounds__(256, 4) for stable occupancy
 * 2. __ldg() read-only cache hints for keys/nonces/tile_map
 * 3. SHF.L.WRAP PTX for single-cycle rotates
 * 4. Fast Poly1305 with 3×64-bit limbs (single reduction per block)
 *
 * Target: 900-1,000 Gb/s (20-33% improvement over 750 Gb/s baseline)
 *
 * Author: PyTorch Crypto Project
 * Date: October 25, 2025
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32
#define CHACHA_BLOCK_SIZE 64
#define POLY_BLOCK_SIZE 16

// ============================================================================
// Debug feature flags (flip one at a time to bisect correctness issues)
// ============================================================================
#ifndef AEAD_USE_SHF_ROTATE
#define AEAD_USE_SHF_ROTATE 0   // 1 = __funnelshift_l (SHF.L.WRAP), 0 = portable shift/or
#endif

#ifndef AEAD_USE_INKERNEL_OTK
#define AEAD_USE_INKERNEL_OTK 1 // 1 = OTK derived in-kernel (ctr=0), 0 = precomputed
#endif

#ifndef AEAD_USE_FAST_POLY1305
#define AEAD_USE_FAST_POLY1305 0 // 1 = 3×64 fast fold, 0 = baseline 26-bit limbs
#endif

// ============================================================================
// OPTIMIZATION 1: Single-instruction rotates
// ============================================================================
__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
#if AEAD_USE_SHF_ROTATE
    // Use CUDA intrinsic (safer than inline PTX on Windows/MSVC)
    // __funnelshift_l(a,b,shift) returns ((a << s) | (b >> (32 - s)))
    // When a == b, this is a rotate left
    return __funnelshift_l(x, x, n);
#else
    // Portable fallback (shift and OR)
    return (x << n) | (x >> (32 - n));
#endif
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

    // 20 rounds (10 double rounds) - fully unrolled
    #pragma unroll
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
// OPTIMIZATION 2: Fast Poly1305 with 3×64-bit limbs (single reduction)
// ============================================================================
struct u130 {
    unsigned long long lo, mid, hi;
};

__device__ __forceinline__ void poly_mul_r_fast(
    u130& a,
    unsigned long long r0,
    unsigned long long r1
) {
    // Multiply 130-bit accumulator by 128-bit r using CUDA intrinsics
    // p0 = a.lo * r0 (128-bit result)
    unsigned long long p0_lo = a.lo * r0;
    unsigned long long p0_hi = __umul64hi(a.lo, r0);

    // p1 = a.lo * r1 + a.mid * r0 (128-bit result)
    unsigned long long p1_lo = a.lo * r1;
    unsigned long long p1_hi = __umul64hi(a.lo, r1);
    unsigned long long t_lo = a.mid * r0;
    unsigned long long t_hi = __umul64hi(a.mid, r0);
    p1_lo += t_lo;
    p1_hi += t_hi + (p1_lo < t_lo);  // add carry

    // p2 = a.mid * r1 + a.hi * r0 (128-bit result)
    unsigned long long p2_lo = a.mid * r1;
    unsigned long long p2_hi = __umul64hi(a.mid, r1);
    t_lo = a.hi * r0;
    t_hi = __umul64hi(a.hi, r0);
    p2_lo += t_lo;
    p2_hi += t_hi + (p2_lo < t_lo);  // add carry

    // p3 = a.hi * r1 (128-bit result)
    unsigned long long p3_lo = a.hi * r1;
    unsigned long long p3_hi = __umul64hi(a.hi, r1);

    // Carry propagation to 256+ bits: q = [q0, q1, q2, q3, q4]
    unsigned long long q0 = p0_lo;
    unsigned long long q1 = p0_hi + p1_lo;
    unsigned long long c1 = (q1 < p0_hi);
    unsigned long long q2 = p1_hi + p2_lo + c1;
    unsigned long long c2 = (q2 < p1_hi) || (c1 && q2 == p1_hi);
    unsigned long long q3 = p2_hi + p3_lo + c2;
    unsigned long long c3 = (q3 < p2_hi) || (c2 && q3 == p2_hi);
    unsigned long long q4 = p3_hi + c3;  // High overflow bits

    // Single fold: split at bit 130
    // L = (q0, q1, q2[0:2]) = bits [0:130)
    // H = (q2[2:64], q3, q4) = bits [130:320)
    unsigned long long low2  =  q2 & 0x3ull;         // keep 2 low bits of q2
    unsigned long long high2_lo = (q2 >> 2) | (q3 << 62); // bits [130:192)
    unsigned long long high2_hi = (q3 >> 2) | (q4 << 62); // bits [192:256+)

    // L = (q0, q1, low2), add 5*H once using 64-bit operations
    // H is a 128-bit value (high2_lo, high2_hi), compute 5*H
    unsigned long long mul5_0 = 5 * high2_lo;
    unsigned long long mul5_1 = __umul64hi(5, high2_lo);
    unsigned long long mul5_2 = 5 * high2_hi;
    unsigned long long mul5_3 = __umul64hi(5, high2_hi);

    // Combine mul5_1 and mul5_2 with carry
    mul5_1 += mul5_2;
    unsigned long long c_mul = (mul5_1 < mul5_2);
    mul5_2 = mul5_3 + c_mul;

    // Add 5*H to (q0, q1, low2)
    unsigned long long t0 = q0 + mul5_0;
    unsigned long long c0 = (t0 < q0);
    a.lo = t0;

    unsigned long long t1 = q1 + mul5_1 + c0;
    c1 = (t1 < q1) || (c0 && t1 == q1);
    a.mid = t1;

    a.hi = low2 + mul5_2 + c1; // low2 (2 bits) + mul5_2 + carry
}

// ============================================================================
// OPTIMIZATION 3: __launch_bounds__ for stable occupancy
// ============================================================================
__launch_bounds__(256, 4)
extern "C" __global__ void aead_chacha20_poly1305_fused(
    const uint8_t*  __restrict__ plaintext,
    uint8_t*        __restrict__ ciphertext,
    uint8_t*        __restrict__ tags,
    const uint32_t* __restrict__ keys,
    const uint32_t* __restrict__ nonces,
    const int*      __restrict__ tile_map,
    int num_tiles,
    int msg_size_bytes,
    int aad_len
) {
    const int warps_per_cta = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int tile_id = blockIdx.x * warps_per_cta + warp_id;

    if (tile_id >= num_tiles) return;

    // ========================================================================
    // OPTIMIZATION 4: __ldg() for read-only tile descriptors
    // ========================================================================
    const int msg_id = __ldg(&tile_map[tile_id * 2 + 0]);
    const int start_block = __ldg(&tile_map[tile_id * 2 + 1]);

    // Load key and nonce with __ldg() (lane 0 only, then broadcast)
    uint32_t key[8];
    uint32_t nonce[3];

    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            key[i] = __ldg(&keys[msg_id * 8 + i]);
        }
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            nonce[i] = __ldg(&nonces[msg_id * 3 + i]);
        }
    }

    // Broadcast key and nonce
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        key[i] = __shfl_sync(0xFFFFFFFF, key[i], 0);
    }
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        nonce[i] = __shfl_sync(0xFFFFFFFF, nonce[i], 0);
    }

    // ========================================================================
    // Generate Poly1305 key (r||s) from ChaCha20 counter=0
    // Convert to 3×64-bit representation
    // ========================================================================
    unsigned long long r0 = 0, r1 = 0;
    unsigned long long s0 = 0, s1 = 0;

    if (lane_id == 0) {
        uint32_t poly_key_block[16];
        chacha20_block(poly_key_block, key, nonce, 0);

        // Extract r (first 16 bytes) and clamp
        unsigned long long r_lo = ((unsigned long long)poly_key_block[0]) |
                                   ((unsigned long long)poly_key_block[1] << 32);
        unsigned long long r_hi = ((unsigned long long)poly_key_block[2]) |
                                   ((unsigned long long)poly_key_block[3] << 32);

        // Clamp r (RFC 7539): clear top 4 bits of each 32-bit word except low word
        r_lo &= 0x0ffffffc0fffffffull;
        r_hi &= 0x0ffffffc0ffffffcull;

        r0 = r_lo;
        r1 = r_hi;

        // Extract s (last 16 bytes)
        s0 = ((unsigned long long)poly_key_block[4]) |
             ((unsigned long long)poly_key_block[5] << 32);
        s1 = ((unsigned long long)poly_key_block[6]) |
             ((unsigned long long)poly_key_block[7] << 32);
    }

    // Broadcast r and s
    r0 = __shfl_sync(0xFFFFFFFF, r0, 0);
    r1 = __shfl_sync(0xFFFFFFFF, r1, 0);
    s0 = __shfl_sync(0xFFFFFFFF, s0, 0);
    s1 = __shfl_sync(0xFFFFFFFF, s1, 0);

    // Initialize Poly1305 accumulator (3×64-bit limbs)
    u130 acc{0, 0, 0};

    // ========================================================================
    // Process blocks: ChaCha20 encrypt + Poly1305 MAC
    // ========================================================================
    const int total_blocks = (msg_size_bytes + CHACHA_BLOCK_SIZE - 1) / CHACHA_BLOCK_SIZE;
    const int block_id = start_block + lane_id;

    if (block_id < total_blocks) {
        // Generate ChaCha20 keystream (counter starts at 1)
        uint32_t keystream[16];
        chacha20_block(keystream, key, nonce, block_id + 1);

        // XOR plaintext → ciphertext (vectorized)
        const uint8_t* pt = plaintext + msg_id * msg_size_bytes + block_id * CHACHA_BLOCK_SIZE;
        uint8_t* ct = ciphertext + msg_id * msg_size_bytes + block_id * CHACHA_BLOCK_SIZE;

        uint32_t* ks_bytes = (uint32_t*)keystream;
        uint4* pt_vec = (uint4*)pt;
        uint4* ct_vec = (uint4*)ct;

        // Vectorized XOR (4× uint4 = 64 bytes)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint4 plain = pt_vec[i];
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

            ct_vec[i] = cipher;
        }

        // Feed ciphertext to Poly1305 (process 4 × 16-byte blocks)
        const uint32_t* ct_u32 = (uint32_t*)ct;

        #pragma unroll
        for (int poly_blk = 0; poly_blk < 4; poly_blk++) {
            // Load 16-byte block as 2×64-bit words
            unsigned long long b0 = ((unsigned long long)ct_u32[poly_blk * 4 + 0]) |
                                     ((unsigned long long)ct_u32[poly_blk * 4 + 1] << 32);
            unsigned long long b1 = ((unsigned long long)ct_u32[poly_blk * 4 + 2]) |
                                     ((unsigned long long)ct_u32[poly_blk * 4 + 3] << 32);

            // Add block to accumulator: acc += (b0 | b1<<64 | 1<<128)
            unsigned long long c;

            // Add b0 to lo
            unsigned long long t = acc.lo + b0;
            c = (t < acc.lo);
            acc.lo = t;

            // Add b1 + carry to mid
            t = acc.mid + b1 + c;
            c = (t < acc.mid) || (c && t == acc.mid);
            acc.mid = t;

            // Add implicit 1<<128 + carry to hi
            acc.hi += c + 1ull;

            // Multiply by r and reduce (fast single-fold)
            poly_mul_r_fast(acc, r0, r1);
        }
    }

    // ========================================================================
    // Warp reduction: sum all lanes' accumulators
    // ========================================================================
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        unsigned long long lo_other = __shfl_down_sync(0xFFFFFFFF, acc.lo, offset);
        unsigned long long mid_other = __shfl_down_sync(0xFFFFFFFF, acc.mid, offset);
        unsigned long long hi_other = __shfl_down_sync(0xFFFFFFFF, acc.hi, offset);

        if (lane_id < offset) {
            unsigned long long c;

            // Add lo
            unsigned long long t = acc.lo + lo_other;
            c = (t < acc.lo);
            acc.lo = t;

            // Add mid
            t = acc.mid + mid_other + c;
            c = (t < acc.mid) || (c && t == acc.mid);
            acc.mid = t;

            // Add hi
            acc.hi += hi_other + c;
        }
    }

    // ========================================================================
    // Lane 0: Finalize tag and write
    // ========================================================================
    if (lane_id == 0 && start_block == 0) {
        // Append length block (AAD length || message length)
        unsigned long long aad_len_u64 = 0;  // AAD not implemented yet
        unsigned long long msg_len_u64 = msg_size_bytes;

        // Add length block to accumulator
        unsigned long long c;

        // Add aad_len to lo
        unsigned long long t = acc.lo + aad_len_u64;
        c = (t < acc.lo);
        acc.lo = t;

        // Add msg_len + carry to mid
        t = acc.mid + msg_len_u64 + c;
        c = (t < acc.mid) || (c && t == acc.mid);
        acc.mid = t;

        // Add implicit 1<<128 + carry to hi
        acc.hi += c + 1ull;

        // Final multiply and reduce
        poly_mul_r_fast(acc, r0, r1);

        // Add s (one-time pad)
        unsigned long long tag0 = acc.lo + s0;
        c = (tag0 < acc.lo);
        unsigned long long tag1 = acc.mid + s1 + c;

        // Write 16-byte tag
        unsigned long long* tag_out = (unsigned long long*)(&tags[msg_id * 16]);
        tag_out[0] = tag0;
        tag_out[1] = tag1;
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
    const int* tile_map,
    int num_tiles,
    int msg_size_bytes,
    int aad_len,
    cudaStream_t stream
) {
    const int threads_per_block = 256;  // 8 warps per block
    const int num_blocks = (num_tiles + 7) / 8;

    aead_chacha20_poly1305_fused<<<num_blocks, threads_per_block, 0, stream>>>(
        plaintext, ciphertext, tags,
        keys, nonces, tile_map,
        num_tiles, msg_size_bytes, aad_len
    );
}
