// =======================================================================
// CARD35B.1: 4-bit fixed-window Montgomery exponentiation (skeleton)
// =======================================================================
//
// Goal:
//   Replace square-and-multiply in modexp_1024() with a 4-bit window
//   variant that reduces the number of Montgomery multiplications per
//   exponent.
//
//   - Baseline ~1536 mont ops per 1024-bit exponent
//   - 4-bit window target ~1280 mont ops per exponent (~17% fewer)
//   - Two exponents (dp, dq) -> ~3.0k -> ~2.6k ops/sign
//
// Strategy (per lane group / per message):
//   1. Work entirely in Montgomery domain (like existing modexp_1024).
//   2. Precompute table:
//          T[0] = R mod n   (Montgomery "1")
//          T[1] = base_mont
//          T[i] = T[i-1] * base_mont  (i = 2..15)
//   3. Scan exponent from MSB to LSB in 4-bit chunks.
//   4. For each window:
//          - 4 squarings
//          - 1 multiply by T[window_value] if window != 0
//
// This implementation assumes:
//   - 1024-bit modulus, 32 limbs
//   - mont_mul_1024_serial(result, a, b, n, n_prime, lane_id) exists
//   - lane_id is 0..31, with lane 0 orchestrating serial work
//   - exp_bits is a 1024-bit exponent (dp or dq) laid out in 32 limbs
//

#pragma once

#include <cstdint>

namespace rsa2048_crt {

// Helper: get 4-bit window from exponent (little-endian 32-bit limbs).
// exp_limbs: 32 x uint32_t, representing 1024-bit exponent (LSW = limb 0).
// bit_index: 0..1023 (the START bit of the 4-bit window)
__device__ __forceinline__
uint32_t get_4bit_window(const uint32_t* exp_limbs, int bit_index)
{
    int limb_idx = bit_index >> 5;      // /32
    int bit_off  = bit_index & 31;      // %32

    uint64_t limb = exp_limbs[limb_idx];
    uint64_t next = 0;
    if (limb_idx + 1 < 32) {
        next = exp_limbs[limb_idx + 1];
    }

    // Combine to safely grab bits across limb boundary.
    uint64_t combined = limb | (next << 32);
    uint64_t shifted  = combined >> bit_off;
    return static_cast<uint32_t>(shifted & 0xF);  // 4 bits
}


// Windowed Montgomery exponentiation in 1024-bit field.
// out:        1024-bit result, normal (non-Montgomery) domain
// base:       1024-bit base, normal domain
// exp_limbs:  1024-bit exponent, 32 x uint32_t
// n:          modulus
// R2:         R^2 mod n (for Montgomery conversion)
// n_prime:    Montgomery n' = -n^(-1) mod 2^32
// lane_id:    warp lane
//
// NOTE: Requires shared memory to be configured by caller:
//   needed = (32 + 32 + 16*32) * sizeof(uint32_t) = 2304 bytes
//
__device__ void modexp_1024_window4(
    uint32_t* out,
    const uint32_t* base,
    const uint32_t* exp_limbs,
    const uint32_t* n,
    const uint32_t* R2,
    uint32_t n_prime,
    int lane_id)
{
    // Shared scratch for one message per warp/block.
    // Layout:
    //   one_mont:   32 limbs  (Montgomery "1" = R mod n)
    //   base_mont:  32 limbs  (base in Montgomery domain)
    //   table:      16 * 32 limbs (precomputed powers)
    __shared__ uint32_t one_mont[RSA_1024_LIMBS];
    __shared__ uint32_t base_mont[RSA_1024_LIMBS];
    __shared__ uint32_t table[16 * RSA_1024_LIMBS];  // table[w][limb] = table[w*32 + limb]

    // 1) Convert "1" into Montgomery domain: one_mont = 1 * R mod n
    //    We do this by converting the integer "1" using R2.
    uint32_t one_normal[RSA_1024_LIMBS];
    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            one_normal[i] = (i == 0) ? 1 : 0;
        }
    }
    __syncwarp();
    // Broadcast one_normal to all lanes
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        one_normal[i] = __shfl_sync(0xFFFFFFFF, one_normal[i], 0);
    }

    // to_mont: one_mont = one_normal * R2 mod n (using serial Montgomery mul)
    mont_mul_1024_serial(one_mont, one_normal, R2, n, n_prime, lane_id);

    // 2) Convert base into Montgomery domain: base_mont = base * R2 mod n
    mont_mul_1024_serial(base_mont, base, R2, n, n_prime, lane_id);

    // 3) Precompute table T[0..15] in Montgomery domain (lane 0 serial)
    if (lane_id == 0) {
        // T[0] = one_mont (Montgomery "1")
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            table[0 * RSA_1024_LIMBS + i] = one_mont[i];
        }
        // T[1] = base_mont
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            table[1 * RSA_1024_LIMBS + i] = base_mont[i];
        }
        // T[w] = T[w-1] * base_mont for w = 2..15
        for (int w = 2; w < 16; ++w) {
            mont_mul_1024_serial(
                &table[w * RSA_1024_LIMBS],
                &table[(w - 1) * RSA_1024_LIMBS],
                base_mont,
                n,
                n_prime,
                /*lane_id=*/0);
        }
    }
    __syncwarp();

    // 4) Exponentiation loop (MSB -> LSB, windows of 4 bits).
    //    Result is kept in Montgomery domain.
    uint32_t result_mont[RSA_1024_LIMBS];
    uint32_t temp[RSA_1024_LIMBS];

    if (lane_id == 0) {
        // Initialize result = 1 (Montgomery)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            result_mont[i] = one_mont[i];
        }

        // Start from the highest 4-bit aligned window (bits 1020..1023)
        // Process 1024 bits = 256 windows of 4 bits each
        // Window indices: 1020, 1016, 1012, ..., 4, 0
        for (int bit = 1020; bit >= 0; bit -= 4) {
            // 4 squarings: result = result^16
            for (int k = 0; k < 4; ++k) {
                mont_mul_1024_serial(
                    temp,
                    result_mont,
                    result_mont,
                    n,
                    n_prime,
                    /*lane_id=*/0);
                // Copy temp -> result_mont
                for (int i = 0; i < RSA_1024_LIMBS; i++) {
                    result_mont[i] = temp[i];
                }
            }

            // Get the 4-bit window value
            uint32_t w = get_4bit_window(exp_limbs, bit);

            // Multiply by table[w] if w != 0
            // (For constant-time, always multiply and conditionally copy)
            if (w != 0) {
                mont_mul_1024_serial(
                    temp,
                    result_mont,
                    &table[w * RSA_1024_LIMBS],
                    n,
                    n_prime,
                    /*lane_id=*/0);
                // Copy temp -> result_mont
                for (int i = 0; i < RSA_1024_LIMBS; i++) {
                    result_mont[i] = temp[i];
                }
            }
        }

        // Copy result_mont to shared for broadcast
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            one_mont[i] = result_mont[i];  // Reuse one_mont as temp storage
        }
    }
    __syncwarp();

    // 5) Convert back from Montgomery domain: out = result_mont * 1 mod n
    //    This is done by multiplying by 1 (in normal form), which gives
    //    result_mont * 1 * R^(-1) mod n = result * R * R^(-1) = result
    uint32_t one_val[RSA_1024_LIMBS];
    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            one_val[i] = (i == 0) ? 1 : 0;
        }
    }
    __syncwarp();
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        one_val[i] = __shfl_sync(0xFFFFFFFF, one_val[i], 0);
    }

    // Reload result_mont from shared (it was stored in one_mont)
    uint32_t result_final[RSA_1024_LIMBS];
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        result_final[i] = one_mont[i];
    }

    // from_mont: out = result_mont * 1 mod n
    mont_mul_1024_serial(out, result_final, one_val, n, n_prime, lane_id);

    // After mont_mul_1024_serial, result is broadcast to all lanes
}

}  // namespace rsa2048_crt
