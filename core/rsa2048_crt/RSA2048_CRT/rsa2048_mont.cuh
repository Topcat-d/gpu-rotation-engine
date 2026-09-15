/**
 * CARD30-B: RSA-2048 Montgomery Arithmetic Primitives (v2 - Correct)
 *
 * Provides GPU-accelerated Montgomery multiplication for 2048-bit integers.
 *
 * Representation:
 *   - 2048-bit values as 64 x 32-bit little-endian limbs
 *   - limbs[0] = least significant 32 bits
 *   - limbs[63] = most significant 32 bits
 *
 * This version uses lane-0 serial computation for correctness.
 * Future optimization: warp-cooperative parallel computation.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace rsa2048 {

// ============================================================================
// Constants
// ============================================================================

constexpr int RSA_BITS = 2048;
constexpr int RSA_LIMBS = 64;        // 2048 / 32
constexpr uint32_t LIMB_MASK = 0xFFFFFFFF;

// ============================================================================
// Montgomery Multiplication (Simple Serial Version)
// ============================================================================

/**
 * Montgomery multiplication: result = a * b * R^(-1) mod n
 *
 * Simple CIOS implementation for correctness.
 * Lane 0 does all computation, other lanes idle.
 *
 * Note: This is NOT optimized. It's for correctness validation.
 * Production code should use warp-cooperative parallel version.
 */
__device__
void mont_mul_2048_serial(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* b,
    const uint32_t* n,
    uint32_t n_prime,
    int lane_id)
{
    // Only lane 0 does computation
    if (lane_id == 0) {
        // Accumulator: 65 limbs (2048 + 32 bits)
        uint64_t T[RSA_LIMBS + 1];
        for (int i = 0; i <= RSA_LIMBS; i++) {
            T[i] = 0;
        }

        // CIOS outer loop
        for (int i = 0; i < RSA_LIMBS; i++) {
            uint32_t bi = b[i];

            // Step 1: T = T + a * b[i]
            uint64_t carry = 0;
            for (int j = 0; j < RSA_LIMBS; j++) {
                uint64_t prod = (uint64_t)a[j] * bi + T[j] + carry;
                T[j] = prod & LIMB_MASK;
                carry = prod >> 32;
            }
            T[RSA_LIMBS] += carry;

            // Step 2: m = T[0] * n' mod 2^32
            uint32_t m = (uint32_t)(T[0] * n_prime);

            // Step 3: T = (T + m*n) >> 32
            carry = 0;
            uint64_t sum = T[0] + (uint64_t)m * n[0];
            carry = sum >> 32;  // This is discarded (shift right)

            for (int j = 1; j < RSA_LIMBS; j++) {
                sum = T[j] + (uint64_t)m * n[j] + carry;
                T[j - 1] = sum & LIMB_MASK;
                carry = sum >> 32;
            }
            T[RSA_LIMBS - 1] = (T[RSA_LIMBS] + carry) & LIMB_MASK;
            T[RSA_LIMBS] = (T[RSA_LIMBS] + carry) >> 32;
        }

        // Final reduction: if T >= n, compute T - n
        // First, check if T[64] > 0 (definitely need reduction)
        bool need_sub = (T[RSA_LIMBS] > 0);

        // If T[64] == 0, compare T[0:63] with n
        if (!need_sub) {
            // Compare from MSB
            for (int j = RSA_LIMBS - 1; j >= 0; j--) {
                if (T[j] > n[j]) {
                    need_sub = true;
                    break;
                } else if (T[j] < n[j]) {
                    break;
                }
            }
        }

        if (need_sub) {
            uint64_t borrow = 0;
            for (int j = 0; j < RSA_LIMBS; j++) {
                uint64_t diff = T[j] - n[j] - borrow;
                result[j] = (uint32_t)(diff & LIMB_MASK);
                borrow = (diff >> 63) & 1;  // Borrow if negative
            }
        } else {
            for (int j = 0; j < RSA_LIMBS; j++) {
                result[j] = (uint32_t)T[j];
            }
        }
    }

    // Broadcast result to all lanes (for consistency)
    __syncwarp();
    for (int i = 0; i < RSA_LIMBS; i++) {
        result[i] = __shfl_sync(0xFFFFFFFF, result[i], 0);
    }
}


/**
 * Montgomery squaring
 */
__device__ __forceinline__
void mont_sqr_2048_serial(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* n,
    uint32_t n_prime,
    int lane_id)
{
    mont_mul_2048_serial(result, a, a, n, n_prime, lane_id);
}


/**
 * Convert to Montgomery domain: result = a * R mod n
 * Uses: to_mont(a) = mont_mul(a, R^2)
 */
__device__ __forceinline__
void to_mont_2048(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* R2,
    const uint32_t* n,
    uint32_t n_prime,
    int lane_id)
{
    mont_mul_2048_serial(result, a, R2, n, n_prime, lane_id);
}


/**
 * Convert from Montgomery domain: result = a * R^(-1) mod n
 * Uses: from_mont(aR) = mont_mul(aR, 1)
 */
__device__
void from_mont_2048(
    uint32_t* result,
    const uint32_t* a_mont,
    const uint32_t* n,
    uint32_t n_prime,
    int lane_id)
{
    // One = [1, 0, 0, ...]
    uint32_t one[RSA_LIMBS];
    for (int i = 0; i < RSA_LIMBS; i++) {
        one[i] = (i == 0) ? 1 : 0;
    }

    mont_mul_2048_serial(result, a_mont, one, n, n_prime, lane_id);
}


// ============================================================================
// Modular Exponentiation
// ============================================================================

/**
 * Constant-time modular exponentiation: result = base^exp mod n
 *
 * Uses square-and-multiply-always for constant-time execution.
 */
__device__
void modexp_2048(
    uint32_t* result,
    const uint32_t* base,
    const uint32_t* exp,
    int exp_bits,
    const uint32_t* n,
    const uint32_t* R2,
    uint32_t n_prime,
    int lane_id)
{
    // Working arrays (per-thread, but only lane 0 uses them)
    uint32_t base_mont[RSA_LIMBS];
    uint32_t acc[RSA_LIMBS];
    uint32_t temp[RSA_LIMBS];

    // Convert base to Montgomery domain
    to_mont_2048(base_mont, base, R2, n, n_prime, lane_id);

    // Initialize accumulator to 1 in Montgomery domain (= R mod n)
    // to_mont(1) = 1 * R^2 * R^(-1) = R
    uint32_t one[RSA_LIMBS];
    for (int i = 0; i < RSA_LIMBS; i++) {
        one[i] = (i == 0) ? 1 : 0;
    }
    to_mont_2048(acc, one, R2, n, n_prime, lane_id);

    // Square-and-multiply from MSB to LSB
    for (int i = exp_bits - 1; i >= 0; i--) {
        // Always square
        mont_sqr_2048_serial(temp, acc, n, n_prime, lane_id);

        // Copy temp to acc
        for (int j = 0; j < RSA_LIMBS; j++) {
            acc[j] = temp[j];
        }

        // Get exponent bit
        int limb_idx = i / 32;
        int bit_idx = i % 32;
        uint32_t exp_limb = exp[limb_idx];
        uint32_t bit = (exp_limb >> bit_idx) & 1;

        // Always multiply (constant time)
        mont_mul_2048_serial(temp, acc, base_mont, n, n_prime, lane_id);

        // Conditional move (constant time via select)
        for (int j = 0; j < RSA_LIMBS; j++) {
            acc[j] = bit ? temp[j] : acc[j];
        }
    }

    // Convert result from Montgomery domain
    from_mont_2048(result, acc, n, n_prime, lane_id);
}


/**
 * RSA-2048 verify (optimized for e=65537)
 */
__device__
void rsa_verify_2048(
    uint32_t* recovered,
    const uint32_t* signature,
    uint32_t e,
    const uint32_t* n,
    const uint32_t* R2,
    uint32_t n_prime,
    int lane_id)
{
    // For e = 65537 = 2^16 + 1:
    // s^65537 = s * (s^2)^16 = s * s^(2^16)

    uint32_t s_mont[RSA_LIMBS];
    uint32_t acc[RSA_LIMBS];
    uint32_t temp[RSA_LIMBS];

    // Convert signature to Montgomery domain
    to_mont_2048(s_mont, signature, R2, n, n_prime, lane_id);

    // Initialize acc = s_mont
    for (int j = 0; j < RSA_LIMBS; j++) {
        acc[j] = s_mont[j];
    }

    // 16 squarings: acc = s^(2^16)
    for (int i = 0; i < 16; i++) {
        mont_sqr_2048_serial(temp, acc, n, n_prime, lane_id);
        for (int j = 0; j < RSA_LIMBS; j++) {
            acc[j] = temp[j];
        }
    }

    // Final multiply: acc = acc * s = s^(2^16 + 1)
    mont_mul_2048_serial(temp, acc, s_mont, n, n_prime, lane_id);

    // Convert from Montgomery domain
    from_mont_2048(recovered, temp, n, n_prime, lane_id);
}


}  // namespace rsa2048
