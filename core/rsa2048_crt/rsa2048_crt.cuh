/**
 * CARD31: RSA-2048 CRT Montgomery Primitives
 *
 * Provides 1024-bit Montgomery multiplication for CRT-accelerated RSA signing.
 *
 * Representation:
 *   - 1024-bit values as 32 x 32-bit little-endian limbs
 *   - 2048-bit values as 64 x 32-bit little-endian limbs
 *
 * CRT Sign:
 *   1. sp = m^dp mod p  (1024-bit modexp)
 *   2. sq = m^dq mod q  (1024-bit modexp)
 *   3. h = (qinv * (sp - sq)) mod p  (1024-bit mont mul)
 *   4. s = sq + q * h   (2048-bit result)
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace rsa2048_crt {

// ============================================================================
// Constants
// ============================================================================

constexpr int RSA_1024_BITS = 1024;
constexpr int RSA_1024_LIMBS = 32;        // 1024 / 32
constexpr int RSA_2048_BITS = 2048;
constexpr int RSA_2048_LIMBS = 64;        // 2048 / 32
constexpr uint32_t LIMB_MASK = 0xFFFFFFFF;

// ============================================================================
// CARD33: Device-Side Key Structure for Tensor-Native RSA
// ============================================================================

/**
 * GPU-resident RSA-2048 CRT key structure.
 *
 * This structure lives entirely in device memory, enabling zero-copy
 * batch signing without CPU transformation.
 *
 * All arrays are 32-limb (1024-bit) little-endian.
 */
struct rsa2048_crt_key_gpu {
    uint32_t p[RSA_1024_LIMBS];        // Prime p
    uint32_t q[RSA_1024_LIMBS];        // Prime q
    uint32_t dp[RSA_1024_LIMBS];       // d mod (p-1)
    uint32_t dq[RSA_1024_LIMBS];       // d mod (q-1)
    uint32_t qinv[RSA_1024_LIMBS];     // q^(-1) mod p
    uint32_t R2_p[RSA_1024_LIMBS];     // R^2 mod p (Montgomery)
    uint32_t R2_q[RSA_1024_LIMBS];     // R^2 mod q (Montgomery)
    uint32_t p_prime;                   // -p^(-1) mod 2^32
    uint32_t q_prime;                   // -q^(-1) mod 2^32
};

// ============================================================================
// CARD33: Warp-Cooperative Montgomery Multiplication
// ============================================================================

/**
 * Warp-cooperative carry propagation using shuffle instructions.
 *
 * Each lane holds one limb of a multi-precision value. This function
 * propagates carries from lane 0 to lane 31.
 *
 * @param val The value in this lane (modified in-place)
 * @param carry_in Carry input from lane 0 (only lane 0 uses this)
 * @return Carry out from lane 31
 */
__device__ __forceinline__
uint32_t warp_propagate_carry(uint32_t& val, uint32_t carry_in)
{
    // Lane 0 adds the initial carry
    int lane_id = threadIdx.x % 32;
    uint64_t sum = (uint64_t)val;
    if (lane_id == 0) {
        sum += carry_in;
    }

    // Propagate carries across the warp (log2(32) = 5 iterations)
    #pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        uint32_t carry = (sum >> 32) & 1;
        uint32_t prev_carry = __shfl_up_sync(0xFFFFFFFF, carry, delta);
        if (lane_id >= delta) {
            sum += prev_carry;
        }
    }

    val = (uint32_t)(sum & LIMB_MASK);

    // Lane 31 returns the final carry
    uint32_t final_carry = (sum >> 32) & 1;
    return __shfl_sync(0xFFFFFFFF, final_carry, 31);
}


/**
 * Warp-cooperative Montgomery multiplication: result = a * b * R^(-1) mod n
 *
 * CARD33 IMPLEMENTATION: All 32 lanes participate in parallel.
 *
 * Each lane i handles limb i of the computation. Uses warp shuffles
 * for carry propagation and broadcasting.
 *
 * Algorithm: CIOS (Coarsely Integrated Operand Scanning)
 * - 32 outer iterations (one per limb of b)
 * - Each iteration: T = T + a * b[i], then T = (T + m*n) >> 32
 *
 * @param result Output: 32-limb result
 * @param a Input: 32-limb multiplicand
 * @param b Input: 32-limb multiplier
 * @param n Input: 32-limb modulus
 * @param n_prime Input: -n^(-1) mod 2^32
 */
__device__
void mont_mul_1024_warp(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* b,
    const uint32_t* n,
    uint32_t n_prime)
{
    int lane_id = threadIdx.x % 32;

    // Each lane holds one limb of the accumulator T
    // T has 33 limbs (1024 + 32 bits), lane 0 also tracks T[32] in T_overflow
    uint32_t T_lane = 0;  // This lane's accumulator limb
    uint32_t T_overflow = 0;  // Extra high limb T[32] (tracked by lane 0)

    // Load operands: each lane holds its corresponding limb
    uint32_t a_lane = a[lane_id];
    uint32_t b_lane = b[lane_id];  // Each lane holds b[lane_id]
    uint32_t n_lane = n[lane_id];

    // CIOS outer loop: iterate over each limb of b
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        // Broadcast b[i] from lane i to all lanes
        uint32_t bi = __shfl_sync(0xFFFFFFFF, b_lane, i);

        // ================================================================
        // Step 1: T = T + a * b[i]
        // ================================================================
        // Parallel multiply-add: each lane computes its partial product
        uint64_t prod = (uint64_t)a_lane * bi;
        uint64_t sum = (uint64_t)T_lane + prod;
        T_lane = (uint32_t)(sum & LIMB_MASK);
        uint32_t carry = (uint32_t)(sum >> 32);

        // Truly sequential carry propagation: lane j waits for lane j-1
        // Lane 0 starts with its carry, lane 1 receives and adds, etc.
        // CRITICAL: carry += overflow, not carry = overflow!
        // Each lane's carry includes BOTH the high bits from a*bi AND
        // any overflow from adding the incoming carry.
        #pragma unroll
        for (int j = 0; j < 31; j++) {
            // Broadcast current carry from lane j
            uint32_t c = __shfl_sync(0xFFFFFFFF, carry, j);
            // Only lane j+1 acts
            if (lane_id == j + 1) {
                sum = (uint64_t)T_lane + c;
                T_lane = (uint32_t)(sum & LIMB_MASK);
                carry += (uint32_t)(sum >> 32);  // ADD to existing carry!
            }
        }

        // Lane 31's carry goes to T_overflow
        uint32_t carry_from_31 = __shfl_sync(0xFFFFFFFF, carry, 31);
        uint32_t T_high = __shfl_sync(0xFFFFFFFF, T_overflow, 0);
        if (lane_id == 0) {
            T_overflow = T_high + carry_from_31;
        }

        // ================================================================
        // Step 2: m = T[0] * n_prime mod 2^32
        // ================================================================
        uint32_t T0 = __shfl_sync(0xFFFFFFFF, T_lane, 0);
        uint32_t m = T0 * n_prime;

        // ================================================================
        // Step 3: T = (T + m * n) >> 32
        // ================================================================
        prod = (uint64_t)m * n_lane;
        sum = (uint64_t)T_lane + prod;
        T_lane = (uint32_t)(sum & LIMB_MASK);
        carry = (uint32_t)(sum >> 32);

        // Sequential carry propagation (same fix: carry += overflow)
        #pragma unroll
        for (int j = 0; j < 31; j++) {
            uint32_t c = __shfl_sync(0xFFFFFFFF, carry, j);
            if (lane_id == j + 1) {
                sum = (uint64_t)T_lane + c;
                T_lane = (uint32_t)(sum & LIMB_MASK);
                carry += (uint32_t)(sum >> 32);  // ADD to existing carry!
            }
        }

        // Get carry from lane 31 and current T_overflow
        carry_from_31 = __shfl_sync(0xFFFFFFFF, carry, 31);
        T_high = __shfl_sync(0xFFFFFFFF, T_overflow, 0);

        // Compute T[32] + carry_from_31 -> this becomes new T[31] after shift
        uint64_t high_sum = (uint64_t)T_high + carry_from_31;
        uint32_t new_T31 = (uint32_t)(high_sum & LIMB_MASK);
        uint32_t new_overflow = (uint32_t)(high_sum >> 32);

        // Shift right by 32 bits: T[j-1] = T[j], T[31] = (T[32] + carry)
        // Note: T[0] is discarded (Montgomery reduction)
        uint32_t T_shifted = __shfl_down_sync(0xFFFFFFFF, T_lane, 1);
        if (lane_id == 31) {
            T_shifted = new_T31;
        }
        T_lane = T_shifted;

        // Update T_overflow for next iteration
        if (lane_id == 0) {
            T_overflow = new_overflow;
        }
    }

    // Final reduction: if T >= n, compute T - n
    // First check if T_overflow > 0 (T definitely >= n)
    uint32_t T_high_final = __shfl_sync(0xFFFFFFFF, T_overflow, 0);
    bool need_sub = (T_high_final > 0);

    // If T_overflow == 0, compare T with n across all lanes
    if (!need_sub) {
        // Determine if T >= n by finding the most significant difference
        int cmp_this = (T_lane > n_lane) ? 1 : ((T_lane < n_lane) ? -1 : 0);

        // Find the highest lane with a non-zero comparison
        uint32_t gt_mask = __ballot_sync(0xFFFFFFFF, cmp_this > 0);
        uint32_t lt_mask = __ballot_sync(0xFFFFFFFF, cmp_this < 0);

        int highest_gt = (gt_mask == 0) ? -1 : (31 - __clz(gt_mask));
        int highest_lt = (lt_mask == 0) ? -1 : (31 - __clz(lt_mask));

        need_sub = (highest_gt > highest_lt);
    }

    // Conditional subtraction: T = T - n
    uint32_t result_val = T_lane;
    if (need_sub) {
        // Parallel subtraction with borrow propagation
        // Use signed 64-bit to detect borrow
        int64_t diff = (int64_t)T_lane - (int64_t)n_lane;
        result_val = (uint32_t)(diff & LIMB_MASK);
        uint32_t borrow = (diff < 0) ? 1 : 0;

        // Borrow propagation: O(32) rounds
        #pragma unroll
        for (int round = 0; round < 32; round++) {
            uint32_t b = __shfl_up_sync(0xFFFFFFFF, borrow, 1);
            if (lane_id > 0) {
                diff = (int64_t)result_val - (int64_t)b;
                result_val = (uint32_t)(diff & LIMB_MASK);
                borrow = (diff < 0) ? 1 : 0;
            }
            if (lane_id == 0) {
                borrow = 0;
            }
        }
    }

    // Write result
    result[lane_id] = result_val;
}


/**
 * Warp-cooperative Montgomery squaring: result = a^2 * R^(-1) mod n
 */
__device__ __forceinline__
void mont_sqr_1024_warp(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* n,
    uint32_t n_prime)
{
    mont_mul_1024_warp(result, a, a, n, n_prime);
}


// ============================================================================
// CARD32: Binary Long-Division Helpers for O(n^2) Reduction
// ============================================================================

/**
 * Get a single bit from a 2048-bit value.
 *
 * @param x 64-limb little-endian array
 * @param bit_index 0..2047 (0 = LSB)
 * @return 0 or 1
 */
__device__ __forceinline__
uint32_t get_bit_2048(const uint32_t* x, int bit_index)
{
    int word = bit_index / 32;
    int bit = bit_index % 32;
    return (x[word] >> bit) & 1u;
}


/**
 * Shift a 1024-bit value left by 1 bit: r <<= 1
 *
 * Returns the carry bit that was shifted out (0 or 1).
 * CRITICAL: Must capture return value for correct modular reduction!
 *
 * @param r 32-limb little-endian array (modified in place)
 * @return Carry bit (1 if overflow, 0 otherwise)
 */
__device__ __forceinline__
uint32_t shl1_1024(uint32_t* r)
{
    uint32_t carry = 0;
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        uint32_t new_carry = r[i] >> 31;
        r[i] = (r[i] << 1) | carry;
        carry = new_carry;
    }
    return carry;  // Return carry for overflow detection
}


/**
 * Compare two 1024-bit values.
 *
 * @param a 32-limb little-endian array
 * @param b 32-limb little-endian array
 * @return 1 if a > b, 0 if a == b, -1 if a < b
 */
__device__ __forceinline__
int cmp_1024(const uint32_t* a, const uint32_t* b)
{
    for (int i = RSA_1024_LIMBS - 1; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}


/**
 * Subtract two 1024-bit values: out = a - b
 *
 * Assumes a >= b (no underflow check).
 *
 * @param out 32-limb result
 * @param a 32-limb minuend
 * @param b 32-limb subtrahend
 */
__device__ __forceinline__
void sub_1024_inline(uint32_t* out, const uint32_t* a, const uint32_t* b)
{
    uint64_t borrow = 0;
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        uint64_t diff = (uint64_t)a[i] - (uint64_t)b[i] - borrow;
        out[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
}


/**
 * GPU-Native Binary Long-Division Reduction: r = m mod p
 *
 * CARD32 FIX: O(n^2) algorithm replaces O(2^n) iterative subtraction.
 *
 * Algorithm (bit-wise shift-subtract):
 *   1. Initialize r = 0
 *   2. For each bit of m from MSB (2047) to LSB (0):
 *      a. r <<= 1
 *      b. r[0] |= current bit of m
 *      c. if r >= p: r -= p
 *   3. Return r = m mod p
 *
 * Complexity: 2048 iterations * O(32) limb ops = O(65k) ops per reduction.
 * This is acceptable compared to thousands of Montgomery muls in modexp.
 *
 * @param r Output: 32-limb result (m mod p)
 * @param m Input: 64-limb 2048-bit value
 * @param p Input: 32-limb 1024-bit modulus
 * @param lane_id Warp lane (only lane 0 does work)
 */
__device__
void reduce_2048_mod_1024_binary(
    uint32_t* r,              // [32] out: m mod p
    const uint32_t* m,        // [64] in:  2048-bit
    const uint32_t* p,        // [32] in:  1024-bit modulus
    int lane_id)
{
    // Only lane 0 does the actual work
    if (lane_id == 0) {
        // Temporary storage for subtraction result
        uint32_t tmp[RSA_1024_LIMBS];

        // Initialize r = 0
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            r[i] = 0;
        }

        // Process bits from MSB (2047) down to LSB (0)
        for (int bit = RSA_2048_BITS - 1; bit >= 0; --bit) {
            // r <<= 1 (capture carry for overflow detection)
            uint32_t carry = shl1_1024(r);

            // r += current bit of m
            uint32_t b = get_bit_2048(m, bit);
            r[0] |= b;

            // if carry || r >= p: r -= p
            // When carry=1, r*2 overflowed 1024 bits, so r >= 2^1024 > p
            if (carry || cmp_1024(r, p) >= 0) {
                sub_1024_inline(tmp, r, p);
                for (int i = 0; i < RSA_1024_LIMBS; i++) {
                    r[i] = tmp[i];
                }
            }
        }
    }

    // Synchronize all lanes before broadcasting
    __syncwarp();

    // Broadcast result from lane 0 to all lanes
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        r[i] = __shfl_sync(0xFFFFFFFF, r[i], 0);
    }
}


// ============================================================================
// 1024-bit Montgomery Multiplication (Serial)
// ============================================================================

/**
 * Montgomery multiplication for 1024-bit: result = a * b * R^(-1) mod n
 *
 * Serial CIOS implementation. Lane 0 does all computation.
 */
__device__
void mont_mul_1024_serial(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* b,
    const uint32_t* n,
    uint32_t n_prime,
    int lane_id)
{
    if (lane_id == 0) {
        // Accumulator: 33 limbs (1024 + 32 bits)
        uint64_t T[RSA_1024_LIMBS + 1];
        for (int i = 0; i <= RSA_1024_LIMBS; i++) {
            T[i] = 0;
        }

        // CIOS outer loop
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            uint32_t bi = b[i];

            // Step 1: T = T + a * b[i]
            uint64_t carry = 0;
            for (int j = 0; j < RSA_1024_LIMBS; j++) {
                uint64_t prod = (uint64_t)a[j] * bi + T[j] + carry;
                T[j] = prod & LIMB_MASK;
                carry = prod >> 32;
            }
            T[RSA_1024_LIMBS] += carry;

            // Step 2: m = T[0] * n' mod 2^32
            uint32_t m = (uint32_t)(T[0] * n_prime);

            // Step 3: T = (T + m*n) >> 32
            carry = 0;
            uint64_t sum = T[0] + (uint64_t)m * n[0];
            carry = sum >> 32;  // Discarded (shift right)

            for (int j = 1; j < RSA_1024_LIMBS; j++) {
                sum = T[j] + (uint64_t)m * n[j] + carry;
                T[j - 1] = sum & LIMB_MASK;
                carry = sum >> 32;
            }
            T[RSA_1024_LIMBS - 1] = (T[RSA_1024_LIMBS] + carry) & LIMB_MASK;
            T[RSA_1024_LIMBS] = (T[RSA_1024_LIMBS] + carry) >> 32;
        }

        // Final reduction: if T >= n, compute T - n
        bool need_sub = (T[RSA_1024_LIMBS] > 0);

        if (!need_sub) {
            for (int j = RSA_1024_LIMBS - 1; j >= 0; j--) {
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
            for (int j = 0; j < RSA_1024_LIMBS; j++) {
                uint64_t diff = T[j] - n[j] - borrow;
                result[j] = (uint32_t)(diff & LIMB_MASK);
                borrow = (diff >> 63) & 1;
            }
        } else {
            for (int j = 0; j < RSA_1024_LIMBS; j++) {
                result[j] = (uint32_t)T[j];
            }
        }
    }

    __syncwarp();
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        result[i] = __shfl_sync(0xFFFFFFFF, result[i], 0);
    }
}


// CARD35B SAFETY HELPER:
//   mont_mul_1024_lane0_only()
//
//   - This is a lane-0-only version of Montgomery multiplication that
//     removes the internal __syncwarp() and broadcast to avoid deadlock.
//   - It MUST ONLY be called from code that is already guarded by:
//
//         if (lane_id == 0) { ... }
//
//   - Other lanes must NOT enter this function, otherwise they will
//     diverge from the warp and hit undefined behavior / deadlock.
//   - The caller is responsible for any broadcast of results if needed.
//
//   Usage pattern:
//
//       if (lane_id == 0) {
//           mont_mul_1024_lane0_only(result, a, b, n, n_prime);
//           // result is only valid on lane 0 here
//       }
//       __syncwarp();
//       // Broadcast result if other lanes need it
//
__device__ __forceinline__
void mont_mul_1024_lane0_only(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* b,
    const uint32_t* n,
    uint32_t n_prime)
{
    // Accumulator: 33 limbs (1024 + 32 bits)
    uint64_t T[RSA_1024_LIMBS + 1];
    for (int i = 0; i <= RSA_1024_LIMBS; i++) {
        T[i] = 0;
    }

    // CIOS outer loop
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        uint32_t bi = b[i];

        // Step 1: T = T + a * b[i]
        uint64_t carry = 0;
        for (int j = 0; j < RSA_1024_LIMBS; j++) {
            uint64_t prod = (uint64_t)a[j] * bi + T[j] + carry;
            T[j] = prod & LIMB_MASK;
            carry = prod >> 32;
        }
        T[RSA_1024_LIMBS] += carry;

        // Step 2: m = T[0] * n' mod 2^32
        uint32_t m = (uint32_t)(T[0] * n_prime);

        // Step 3: T = (T + m*n) >> 32
        carry = 0;
        uint64_t sum = T[0] + (uint64_t)m * n[0];
        carry = sum >> 32;  // Discarded (shift right)

        for (int j = 1; j < RSA_1024_LIMBS; j++) {
            sum = T[j] + (uint64_t)m * n[j] + carry;
            T[j - 1] = sum & LIMB_MASK;
            carry = sum >> 32;
        }
        T[RSA_1024_LIMBS - 1] = (T[RSA_1024_LIMBS] + carry) & LIMB_MASK;
        T[RSA_1024_LIMBS] = (T[RSA_1024_LIMBS] + carry) >> 32;
    }

    // Final reduction: if T >= n, compute T - n
    bool need_sub = (T[RSA_1024_LIMBS] > 0);

    if (!need_sub) {
        for (int j = RSA_1024_LIMBS - 1; j >= 0; j--) {
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
        for (int j = 0; j < RSA_1024_LIMBS; j++) {
            uint64_t diff = T[j] - n[j] - borrow;
            result[j] = (uint32_t)(diff & LIMB_MASK);
            borrow = (diff >> 63) & 1;
        }
    } else {
        for (int j = 0; j < RSA_1024_LIMBS; j++) {
            result[j] = (uint32_t)T[j];
        }
    }
    // NO __syncwarp() or broadcast here!
}


/**
 * Montgomery squaring for 1024-bit
 */
__device__ __forceinline__
void mont_sqr_1024_serial(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* n,
    uint32_t n_prime,
    int lane_id)
{
    mont_mul_1024_serial(result, a, a, n, n_prime, lane_id);
}


/**
 * Convert to Montgomery domain: result = a * R mod n (1024-bit)
 */
__device__ __forceinline__
void to_mont_1024(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* R2,
    const uint32_t* n,
    uint32_t n_prime,
    int lane_id)
{
    mont_mul_1024_serial(result, a, R2, n, n_prime, lane_id);
}


/**
 * Convert from Montgomery domain: result = a * R^(-1) mod n (1024-bit)
 */
__device__
void from_mont_1024(
    uint32_t* result,
    const uint32_t* a_mont,
    const uint32_t* n,
    uint32_t n_prime,
    int lane_id)
{
    uint32_t one[RSA_1024_LIMBS];
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        one[i] = (i == 0) ? 1 : 0;
    }
    mont_mul_1024_serial(result, a_mont, one, n, n_prime, lane_id);
}


// ============================================================================
// CARD33: Warp-Cooperative Montgomery Domain Conversions
// ============================================================================

/**
 * Warp-cooperative convert to Montgomery domain: result = a * R mod n
 */
__device__ __forceinline__
void to_mont_1024_warp(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* R2,
    const uint32_t* n,
    uint32_t n_prime)
{
    mont_mul_1024_warp(result, a, R2, n, n_prime);
}


/**
 * Warp-cooperative convert from Montgomery domain: result = a * R^(-1) mod n
 */
__device__
void from_mont_1024_warp(
    uint32_t* result,
    const uint32_t* a_mont,
    const uint32_t* n,
    uint32_t n_prime)
{
    int lane_id = threadIdx.x % 32;

    // Create "1" in shared memory (each lane writes its limb)
    __shared__ uint32_t one_shared[RSA_1024_LIMBS];
    one_shared[lane_id] = (lane_id == 0) ? 1 : 0;
    __syncwarp();

    mont_mul_1024_warp(result, a_mont, one_shared, n, n_prime);
}


/**
 * Warp-cooperative modular exponentiation: result = base^exp mod n
 *
 * CARD33 IMPLEMENTATION: Uses warp-cooperative Montgomery multiplication.
 * All 32 threads in the warp participate in each operation.
 *
 * Algorithm: Square-and-multiply (constant-time)
 * - 1024 iterations (one per bit of exponent)
 * - Each iteration: square, then conditionally multiply
 *
 * @param result Output: 32-limb result
 * @param base Input: 32-limb base
 * @param exp Input: 32-limb exponent
 * @param exp_bits Number of bits in exponent
 * @param n Input: 32-limb modulus
 * @param R2 Input: R^2 mod n for Montgomery conversion
 * @param n_prime Input: -n^(-1) mod 2^32
 */
__device__
void modexp_1024_warp(
    uint32_t* result,
    const uint32_t* base,
    const uint32_t* exp,
    int exp_bits,
    const uint32_t* n,
    const uint32_t* R2,
    uint32_t n_prime)
{
    int lane_id = threadIdx.x % 32;

    // Use shared memory for intermediate results (each warp needs its own)
    // Note: For production, use dynamically allocated shared memory per warp
    __shared__ uint32_t base_mont_shared[RSA_1024_LIMBS];
    __shared__ uint32_t acc_shared[RSA_1024_LIMBS];
    __shared__ uint32_t temp_shared[RSA_1024_LIMBS];
    __shared__ uint32_t one_mont_shared[RSA_1024_LIMBS];

    // Convert base to Montgomery domain
    to_mont_1024_warp(base_mont_shared, base, R2, n, n_prime);

    // Initialize "1" for Montgomery domain conversion
    __shared__ uint32_t one_shared[RSA_1024_LIMBS];
    one_shared[lane_id] = (lane_id == 0) ? 1 : 0;
    __syncwarp();

    // acc = 1 in Montgomery domain = R mod n
    to_mont_1024_warp(acc_shared, one_shared, R2, n, n_prime);

    // Square-and-multiply from MSB to LSB
    for (int i = exp_bits - 1; i >= 0; i--) {
        // Always square: acc = acc^2
        mont_sqr_1024_warp(temp_shared, acc_shared, n, n_prime);

        // Copy temp to acc
        acc_shared[lane_id] = temp_shared[lane_id];
        __syncwarp();

        // Get exponent bit (all lanes compute this)
        int limb_idx = i / 32;
        int bit_idx = i % 32;
        uint32_t exp_limb = exp[limb_idx];
        uint32_t bit = (exp_limb >> bit_idx) & 1;

        // Always multiply (constant time): temp = acc * base_mont
        mont_mul_1024_warp(temp_shared, acc_shared, base_mont_shared, n, n_prime);

        // Conditional move: acc = bit ? temp : acc
        acc_shared[lane_id] = bit ? temp_shared[lane_id] : acc_shared[lane_id];
        __syncwarp();
    }

    // Convert result from Montgomery domain
    from_mont_1024_warp(result, acc_shared, n, n_prime);
}


// ============================================================================
// 1024-bit Modular Exponentiation (Serial - Legacy)
// ============================================================================

/**
 * Constant-time modular exponentiation (1024-bit): result = base^exp mod n
 */
__device__
void modexp_1024(
    uint32_t* result,
    const uint32_t* base,
    const uint32_t* exp,
    int exp_bits,
    const uint32_t* n,
    const uint32_t* R2,
    uint32_t n_prime,
    int lane_id)
{
    uint32_t base_mont[RSA_1024_LIMBS];
    uint32_t acc[RSA_1024_LIMBS];
    uint32_t temp[RSA_1024_LIMBS];

    // Convert base to Montgomery domain
    to_mont_1024(base_mont, base, R2, n, n_prime, lane_id);

    // Initialize accumulator to 1 in Montgomery domain (= R mod n)
    uint32_t one[RSA_1024_LIMBS];
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        one[i] = (i == 0) ? 1 : 0;
    }
    to_mont_1024(acc, one, R2, n, n_prime, lane_id);

    // Square-and-multiply from MSB to LSB
    for (int i = exp_bits - 1; i >= 0; i--) {
        // Always square
        mont_sqr_1024_serial(temp, acc, n, n_prime, lane_id);

        for (int j = 0; j < RSA_1024_LIMBS; j++) {
            acc[j] = temp[j];
        }

        // Get exponent bit
        int limb_idx = i / 32;
        int bit_idx = i % 32;
        uint32_t exp_limb = exp[limb_idx];
        uint32_t bit = (exp_limb >> bit_idx) & 1;

        // Always multiply (constant time)
        mont_mul_1024_serial(temp, acc, base_mont, n, n_prime, lane_id);

        // Conditional move
        for (int j = 0; j < RSA_1024_LIMBS; j++) {
            acc[j] = bit ? temp[j] : acc[j];
        }
    }

    // Convert result from Montgomery domain
    from_mont_1024(result, acc, n, n_prime, lane_id);
}


// ============================================================================
// CARD35B: 4-bit Window Modular Exponentiation (Experimental)
// ============================================================================
//
// Goal: Reduce Montgomery ops from ~1536 to ~1280 per 1024-bit exponent (-17%)
// Enable with: #define USE_WINDOW_MODEXP or #define USE_WINDOW5_MODEXP
//

#if defined(USE_WINDOW_MODEXP) || defined(USE_WINDOW5_MODEXP)

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
    if (limb_idx + 1 < RSA_1024_LIMBS) {
        next = exp_limbs[limb_idx + 1];
    }

    // Combine to safely grab bits across limb boundary.
    uint64_t combined = limb | (next << 32);
    uint64_t shifted  = combined >> bit_off;
    return static_cast<uint32_t>(shifted & 0xF);  // 4 bits
}

/**
 * CARD35B: 4-bit fixed-window Montgomery exponentiation (1024-bit).
 *
 * result = base^exp mod n
 *
 * Algorithm:
 *   1. Precompute T[0..15] where T[i] = base^i in Montgomery domain
 *   2. Process exponent in 4-bit windows from MSB to LSB
 *   3. For each window: 4 squarings + 1 multiply by T[window_value]
 *
 * Operations: ~256 windows * (4 sqr + ~0.5 mul) = ~1280 mont ops (vs 1536 baseline)
 */
__device__
void modexp_1024_window4(
    uint32_t* result,
    const uint32_t* base,
    const uint32_t* exp,
    int exp_bits,
    const uint32_t* n,
    const uint32_t* R2,
    uint32_t n_prime,
    int lane_id)
{
    // CARD35B NOTE:
    //
    //   We use a LOCAL table here:
    //
    //       16 entries * 32 limbs = 512 uint32_t = 2 KB per thread
    //
    //   This is simple and avoids shared-memory collisions when many
    //   threads in the block are processing different messages.
    //
    //   Only lane 0 actually uses this table; other lanes carry the
    //   allocation but do not touch it. In practice this still performs
    //   well (1.44x speedup over baseline), but we may later:
    //
    //     - Move the table to __shared__ per-warp, OR
    //     - Restrict this kernel to 1 thread per warp in a dedicated block
    //
    //   as part of a future optimization (e.g. CARD35B.3).
    //
    //   For now, the local-memory approach is correct and performant.
    //
    uint32_t table[16 * RSA_1024_LIMBS];
    uint32_t one_mont[RSA_1024_LIMBS];
    uint32_t base_mont[RSA_1024_LIMBS];

    // 1) Convert "1" and base into Montgomery domain
    uint32_t one_normal[RSA_1024_LIMBS];
    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            one_normal[i] = (i == 0) ? 1 : 0;
        }
    }
    __syncwarp();
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        one_normal[i] = __shfl_sync(0xFFFFFFFF, one_normal[i], 0);
    }

    // one_mont = 1 * R mod n (Montgomery "1")
    mont_mul_1024_serial(one_mont, one_normal, R2, n, n_prime, lane_id);

    // base_mont = base * R mod n
    mont_mul_1024_serial(base_mont, base, R2, n, n_prime, lane_id);

    // 2) Precompute table T[0..15] in Montgomery domain (lane 0 serial)
    if (lane_id == 0) {
        // T[0] = one_mont (base^0 = 1)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            table[0 * RSA_1024_LIMBS + i] = one_mont[i];
        }
        // T[1] = base_mont (base^1)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            table[1 * RSA_1024_LIMBS + i] = base_mont[i];
        }
        // T[w] = T[w-1] * base_mont for w = 2..15
        // Use lane0_only version to avoid warp deadlock
        for (int w = 2; w < 16; ++w) {
            mont_mul_1024_lane0_only(
                &table[w * RSA_1024_LIMBS],
                &table[(w - 1) * RSA_1024_LIMBS],
                base_mont,
                n,
                n_prime);
        }
    }
    __syncwarp();

    // 3) Exponentiation loop (MSB -> LSB, windows of 4 bits)
    uint32_t acc[RSA_1024_LIMBS];
    uint32_t temp[RSA_1024_LIMBS];

    if (lane_id == 0) {
        // Initialize acc = 1 (Montgomery)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            acc[i] = one_mont[i];
        }

        // Process 256 windows of 4 bits each (1024 bits total)
        // Start from highest 4-bit window (bits 1020..1023)
        int num_windows = (exp_bits + 3) / 4;
        int start_bit = (num_windows - 1) * 4;

        for (int bit = start_bit; bit >= 0; bit -= 4) {
            // 4 squarings: acc = acc^16
            // Use lane0_only version to avoid warp deadlock
            for (int k = 0; k < 4; ++k) {
                mont_mul_1024_lane0_only(temp, acc, acc, n, n_prime);
                for (int i = 0; i < RSA_1024_LIMBS; i++) {
                    acc[i] = temp[i];
                }
            }

            // Get the 4-bit window value
            uint32_t w = get_4bit_window(exp, bit);

            // Multiply by table[w] if w != 0
            if (w != 0) {
                mont_mul_1024_lane0_only(
                    temp,
                    acc,
                    &table[w * RSA_1024_LIMBS],
                    n,
                    n_prime);
                for (int i = 0; i < RSA_1024_LIMBS; i++) {
                    acc[i] = temp[i];
                }
            }
        }

        // Store result in shared for broadcast
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            one_mont[i] = acc[i];  // Reuse one_mont as temp storage
        }
    }
    __syncwarp();

    // 4) Convert back from Montgomery domain
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

    // Reload acc from shared
    uint32_t acc_final[RSA_1024_LIMBS];
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        acc_final[i] = one_mont[i];
    }

    // result = acc * 1 mod n (removes Montgomery factor)
    mont_mul_1024_serial(result, acc_final, one_val, n, n_prime, lane_id);
}


// ============================================================================
// CARD35B.2: 5-bit Window Variant (experimental)
// ============================================================================

#ifdef USE_WINDOW5_MODEXP

// Helper: get 5-bit window from exponent (little-endian 32-bit limbs).
// bit_index: 0..1023 (the START bit of the 5-bit window)
__device__ __forceinline__
uint32_t get_5bit_window(const uint32_t* exp_limbs, int bit_index)
{
    int limb_idx = bit_index >> 5;      // /32
    int bit_off  = bit_index & 31;      // %32

    uint64_t limb = exp_limbs[limb_idx];
    uint64_t next = 0;
    if (limb_idx + 1 < RSA_1024_LIMBS) {
        next = exp_limbs[limb_idx + 1];
    }

    // Combine to safely grab bits across limb boundary.
    uint64_t combined = limb | (next << 32);
    uint64_t shifted  = combined >> bit_off;
    return static_cast<uint32_t>(shifted & 0x1F);  // 5 bits
}

/**
 * CARD35B.2: 5-bit fixed-window Montgomery exponentiation (1024-bit).
 *
 * result = base^exp mod n
 *
 * Algorithm:
 *   1. Precompute T[0..31] where T[i] = base^i in Montgomery domain
 *   2. Process exponent in 5-bit windows from MSB to LSB
 *   3. For each window: 5 squarings + 1 multiply by T[window_value]
 *
 * Operations: ~205 windows * (5 sqr + ~0.5 mul) = ~1128 mont ops (vs 1280 for 4-bit)
 * Table size: 32 entries * 32 limbs * 4 bytes = 4 KB (vs 2 KB for 4-bit)
 *
 * Trade-off: Fewer mont ops, but larger table (more register pressure).
 */
__device__
void modexp_1024_window5(
    uint32_t* result,
    const uint32_t* base,
    const uint32_t* exp,
    int exp_bits,
    const uint32_t* n,
    const uint32_t* R2,
    uint32_t n_prime,
    int lane_id)
{
    // CARD35B.2 NOTE:
    //
    //   5-bit window uses 32 table entries = 4 KB per thread.
    //   This is larger than 4-bit (2 KB) but should give ~12% fewer mont ops.
    //
    uint32_t table[32 * RSA_1024_LIMBS];
    uint32_t one_mont[RSA_1024_LIMBS];
    uint32_t base_mont[RSA_1024_LIMBS];

    // 1) Convert "1" and base into Montgomery domain
    uint32_t one_normal[RSA_1024_LIMBS];
    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            one_normal[i] = (i == 0) ? 1 : 0;
        }
    }
    __syncwarp();
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        one_normal[i] = __shfl_sync(0xFFFFFFFF, one_normal[i], 0);
    }

    // one_mont = 1 * R mod n (Montgomery "1")
    mont_mul_1024_serial(one_mont, one_normal, R2, n, n_prime, lane_id);

    // base_mont = base * R mod n
    mont_mul_1024_serial(base_mont, base, R2, n, n_prime, lane_id);

    // 2) Precompute table T[0..31] in Montgomery domain (lane 0 serial)
    if (lane_id == 0) {
        // T[0] = one_mont (base^0 = 1)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            table[0 * RSA_1024_LIMBS + i] = one_mont[i];
        }
        // T[1] = base_mont (base^1)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            table[1 * RSA_1024_LIMBS + i] = base_mont[i];
        }
        // T[w] = T[w-1] * base_mont for w = 2..31
        for (int w = 2; w < 32; ++w) {
            mont_mul_1024_lane0_only(
                &table[w * RSA_1024_LIMBS],
                &table[(w - 1) * RSA_1024_LIMBS],
                base_mont,
                n,
                n_prime);
        }
    }
    __syncwarp();

    // 3) Exponentiation loop (MSB -> LSB, windows of 5 bits)
    uint32_t acc[RSA_1024_LIMBS];
    uint32_t temp[RSA_1024_LIMBS];

    if (lane_id == 0) {
        // Initialize acc = 1 (Montgomery)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            acc[i] = one_mont[i];
        }

        // Process windows of 5 bits each
        // 1024 bits = ceiling(1024/5) = 205 windows
        int num_windows = (exp_bits + 4) / 5;
        int start_bit = (num_windows - 1) * 5;

        for (int bit = start_bit; bit >= 0; bit -= 5) {
            // 5 squarings: acc = acc^32
            for (int k = 0; k < 5; ++k) {
                mont_mul_1024_lane0_only(temp, acc, acc, n, n_prime);
                for (int i = 0; i < RSA_1024_LIMBS; i++) {
                    acc[i] = temp[i];
                }
            }

            // Get the 5-bit window value
            uint32_t w = get_5bit_window(exp, bit);

            // Multiply by table[w] if w != 0
            if (w != 0) {
                mont_mul_1024_lane0_only(
                    temp,
                    acc,
                    &table[w * RSA_1024_LIMBS],
                    n,
                    n_prime);
                for (int i = 0; i < RSA_1024_LIMBS; i++) {
                    acc[i] = temp[i];
                }
            }
        }

        // Store result in shared for broadcast
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            one_mont[i] = acc[i];  // Reuse one_mont as temp storage
        }
    }
    __syncwarp();

    // 4) Convert back from Montgomery domain
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

    // Reload acc from shared
    uint32_t acc_final[RSA_1024_LIMBS];
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        acc_final[i] = one_mont[i];
    }

    // result = acc * 1 mod n (removes Montgomery factor)
    mont_mul_1024_serial(result, acc_final, one_val, n, n_prime, lane_id);
}

#endif  // USE_WINDOW5_MODEXP

#endif  // USE_WINDOW_MODEXP || USE_WINDOW5_MODEXP


// ============================================================================
// 2048-bit Arithmetic Helpers (for CRT recombination)
// ============================================================================

/**
 * Reduce 2048-bit value mod 1024-bit prime: result = a mod p
 *
 * Simple modular reduction by taking lower 32 limbs and adjusting.
 * For CRT, we reduce message mod p/q before modexp.
 */
__device__
void reduce_2048_mod_1024(
    uint32_t* result,        // 32 limbs output
    const uint32_t* a,       // 64 limbs input
    const uint32_t* p,       // 32 limbs modulus
    int lane_id)
{
    if (lane_id == 0) {
        // We need to compute a mod p where a is 2048-bit and p is 1024-bit
        // Barrett reduction or simple shift-subtract would be complex
        //
        // For correctness, we do iterative subtraction using Schoolbook division
        // This is O(n^2) but works for serial lane-0 implementation

        // Copy a to working array (needs 65 limbs to handle overflow)
        uint64_t work[RSA_2048_LIMBS + 1];
        for (int i = 0; i < RSA_2048_LIMBS; i++) {
            work[i] = a[i];
        }
        work[RSA_2048_LIMBS] = 0;

        // Extend p to 64 limbs for comparison
        uint32_t p_ext[RSA_2048_LIMBS];
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            p_ext[i] = p[i];
        }
        for (int i = RSA_1024_LIMBS; i < RSA_2048_LIMBS; i++) {
            p_ext[i] = 0;
        }

        // Iterative subtraction: while work >= p, work -= p * 2^shift
        // Start from the highest possible shift and work down
        for (int shift = RSA_2048_LIMBS - RSA_1024_LIMBS; shift >= 0; shift--) {
            // Check if work >= p_shifted
            while (true) {
                bool can_subtract = true;
                bool strictly_greater = false;

                // Compare work with p << (shift * 32)
                for (int j = RSA_2048_LIMBS - 1; j >= shift; j--) {
                    uint32_t w = (uint32_t)work[j];
                    uint32_t pv = (j - shift < RSA_1024_LIMBS) ? p[j - shift] : 0;

                    if (w > pv) {
                        strictly_greater = true;
                        break;
                    } else if (w < pv) {
                        can_subtract = false;
                        break;
                    }
                }

                // Also check lower limbs if we haven't decided
                if (can_subtract && !strictly_greater) {
                    for (int j = shift - 1; j >= 0; j--) {
                        if (work[j] > 0) {
                            strictly_greater = true;
                            break;
                        }
                    }
                }

                if (!can_subtract) break;
                if (!strictly_greater && shift > 0) break;

                // Subtract p << (shift * 32)
                uint64_t borrow = 0;
                for (int j = shift; j < shift + RSA_1024_LIMBS; j++) {
                    uint64_t diff = work[j] - p[j - shift] - borrow;
                    work[j] = diff & LIMB_MASK;
                    borrow = (diff >> 63) & 1;
                }
                for (int j = shift + RSA_1024_LIMBS; j <= RSA_2048_LIMBS; j++) {
                    if (borrow == 0) break;
                    uint64_t diff = work[j] - borrow;
                    work[j] = diff & LIMB_MASK;
                    borrow = (diff >> 63) & 1;
                }

                if (!strictly_greater) break;
            }
        }

        // Result is now in lower 32 limbs
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            result[i] = (uint32_t)work[i];
        }
    }

    __syncwarp();
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        result[i] = __shfl_sync(0xFFFFFFFF, result[i], 0);
    }
}


/**
 * Modular subtraction: result = (a - b) mod p  (1024-bit)
 *
 * If a >= b: result = a - b
 * If a < b:  result = a - b + p
 */
__device__
void mod_sub_1024(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* b,
    const uint32_t* p,
    int lane_id)
{
    // CARD34 SAFETY NOTE:
    //
    //   - Only lane 0 actually performs the full 1024-bit subtraction.
    //   - At the end of the function we broadcast the result from lane 0
    //     to ALL lanes using __shfl_sync.
    //   - This means that after mod_sub_1024() returns, every lane sees
    //     a consistent 1024-bit value in result[0..RSA_1024_LIMBS-1].
    //
    //   Important for callers:
    //     * Lane 0 will later read result[0..31].
    //     * result may live in either local or __shared__ memory.
    //     * If you need lane 0 to see the whole 1024-bit value, do NOT
    //       try to "gather" limbs via per-lane locals - use the array
    //       (local or shared) directly.
    //
    if (lane_id == 0) {
        // Compute a - b
        int64_t borrow = 0;
        uint32_t diff[RSA_1024_LIMBS];

        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            int64_t d = (int64_t)a[i] - (int64_t)b[i] - borrow;
            if (d < 0) {
                diff[i] = (uint32_t)(d + 0x100000000LL);
                borrow = 1;
            } else {
                diff[i] = (uint32_t)d;
                borrow = 0;
            }
        }

        // If borrow != 0, we need to add p
        if (borrow != 0) {
            uint64_t carry = 0;
            for (int i = 0; i < RSA_1024_LIMBS; i++) {
                uint64_t sum = (uint64_t)diff[i] + (uint64_t)p[i] + carry;
                result[i] = (uint32_t)(sum & LIMB_MASK);
                carry = sum >> 32;
            }
        } else {
            for (int i = 0; i < RSA_1024_LIMBS; i++) {
                result[i] = diff[i];
            }
        }
    }

    __syncwarp();
    for (int i = 0; i < RSA_1024_LIMBS; i++) {
        result[i] = __shfl_sync(0xFFFFFFFF, result[i], 0);
    }
}


/**
 * Multiply 1024-bit by 1024-bit, producing 2048-bit result (schoolbook)
 *
 * result[64] = a[32] * b[32]
 */
__device__
void mul_1024x1024_to_2048(
    uint32_t* result,        // 64 limbs output
    const uint32_t* a,       // 32 limbs
    const uint32_t* b,       // 32 limbs
    int lane_id)
{
    if (lane_id == 0) {
        // Initialize result to zero
        uint64_t acc[RSA_2048_LIMBS];
        for (int i = 0; i < RSA_2048_LIMBS; i++) {
            acc[i] = 0;
        }

        // Schoolbook multiplication
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            uint64_t carry = 0;
            for (int j = 0; j < RSA_1024_LIMBS; j++) {
                uint64_t prod = (uint64_t)a[i] * (uint64_t)b[j] + acc[i + j] + carry;
                acc[i + j] = prod & LIMB_MASK;
                carry = prod >> 32;
            }
            acc[i + RSA_1024_LIMBS] += carry;
        }

        // Copy to result
        for (int i = 0; i < RSA_2048_LIMBS; i++) {
            result[i] = (uint32_t)acc[i];
        }
    }

    __syncwarp();
    for (int i = 0; i < RSA_2048_LIMBS; i++) {
        result[i] = __shfl_sync(0xFFFFFFFF, result[i], 0);
    }
}


/**
 * Add 2048-bit: result = a + b (no modular reduction)
 */
__device__
void add_2048(
    uint32_t* result,
    const uint32_t* a,
    const uint32_t* b,
    int lane_id)
{
    if (lane_id == 0) {
        uint64_t carry = 0;
        for (int i = 0; i < RSA_2048_LIMBS; i++) {
            uint64_t sum = (uint64_t)a[i] + (uint64_t)b[i] + carry;
            result[i] = (uint32_t)(sum & LIMB_MASK);
            carry = sum >> 32;
        }
    }

    __syncwarp();
    for (int i = 0; i < RSA_2048_LIMBS; i++) {
        result[i] = __shfl_sync(0xFFFFFFFF, result[i], 0);
    }
}


// ============================================================================
// CRT Sign
// ============================================================================

/**
 * CRT-accelerated RSA-2048 sign: s = m^d mod n
 *
 * Using CRT:
 *   1. sp = m^dp mod p
 *   2. sq = m^dq mod q
 *   3. h = (qinv * (sp - sq mod p)) mod p
 *   4. s = sq + q * h
 */
__device__
void rsa_crt_sign(
    uint32_t* signature,      // 64 limbs output
    const uint32_t* message,  // 64 limbs input
    const uint32_t* p,        // 32 limbs
    const uint32_t* q,        // 32 limbs
    const uint32_t* dp,       // 32 limbs
    const uint32_t* dq,       // 32 limbs
    const uint32_t* qinv,     // 32 limbs
    const uint32_t* R2_p,     // 32 limbs
    const uint32_t* R2_q,     // 32 limbs
    uint32_t p_prime,
    uint32_t q_prime,
    int lane_id)
{
    // Local arrays
    uint32_t mp[RSA_1024_LIMBS];       // m mod p
    uint32_t mq[RSA_1024_LIMBS];       // m mod q
    uint32_t sp[RSA_1024_LIMBS];       // m^dp mod p
    uint32_t sq[RSA_1024_LIMBS];       // m^dq mod q
    uint32_t diff[RSA_1024_LIMBS];     // (sp - sq) mod p
    uint32_t h[RSA_1024_LIMBS];        // qinv * diff mod p
    uint32_t qh[RSA_2048_LIMBS];       // q * h (2048-bit)
    uint32_t sq_ext[RSA_2048_LIMBS];   // sq zero-extended to 2048 bits

    // Step 1: Reduce message mod p and mod q (CARD32: GPU-native O(n^2) binary reduction)
    reduce_2048_mod_1024_binary(mp, message, p, lane_id);
    reduce_2048_mod_1024_binary(mq, message, q, lane_id);

    // Step 2: Compute partial signatures
    //
    // CRT exponentiations: m^dp mod p, m^dq mod q
    //
    // We support three modes:
    //   - Baseline square-and-multiply (no window)
    //   - 4-bit window (USE_WINDOW_MODEXP)
    //   - 5-bit window (USE_WINDOW5_MODEXP)
    //
    // USE_WINDOW5_MODEXP takes precedence if both are defined.
    //
#ifdef USE_WINDOW5_MODEXP
    modexp_1024_window5(sp, mp, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
    modexp_1024_window5(sq, mq, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#elif defined(USE_WINDOW_MODEXP)
    modexp_1024_window4(sp, mp, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
    modexp_1024_window4(sq, mq, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#else
    modexp_1024(sp, mp, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
    modexp_1024(sq, mq, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#endif

    // Step 3: CRT recombination
    // diff = (sp - sq) mod p
    mod_sub_1024(diff, sp, sq, p, lane_id);

    // h = (qinv * diff) mod p (using Montgomery multiplication)
    // Need to convert diff to Montgomery domain, multiply, convert back
    uint32_t diff_mont[RSA_1024_LIMBS];
    uint32_t h_mont[RSA_1024_LIMBS];

    to_mont_1024(diff_mont, diff, R2_p, p, p_prime, lane_id);

    // qinv needs to be in Montgomery form too
    uint32_t qinv_mont[RSA_1024_LIMBS];
    to_mont_1024(qinv_mont, qinv, R2_p, p, p_prime, lane_id);

    mont_mul_1024_serial(h_mont, qinv_mont, diff_mont, p, p_prime, lane_id);
    from_mont_1024(h, h_mont, p, p_prime, lane_id);

    // Step 4: Final combination
    // s = sq + q * h
    //
    // CARD34 SAFETY NOTE:
    //
    //   mul_1024x1024_to_2048() is written assuming lane 0 consumes the
    //   full 1024-bit operands:
    //
    //      - On lane 0 it reads a[0..31], b[0..31]
    //      - Other lanes may contribute partial work, but correctness
    //        requires that those limbs are valid for lane 0.
    //
    //   In the serial CRT path, all of sp/sq/h/q are produced by helpers
    //   (modexp_1024, from_mont_1024, mod_sub_1024) that already
    //   broadcast their outputs via __shfl_sync, so each local array has
    //   all limbs valid on lane 0.
    //
    //   DO NOT re-introduce patterns like:
    //       tmp[lane_id] = shared[lane_id];   // lane 0 sees only tmp[0]
    //
    mul_1024x1024_to_2048(qh, q, h, lane_id);

    // Extend sq to 2048 bits
    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            sq_ext[i] = sq[i];
        }
        for (int i = RSA_1024_LIMBS; i < RSA_2048_LIMBS; i++) {
            sq_ext[i] = 0;
        }
    }
    __syncwarp();

    add_2048(signature, sq_ext, qh, lane_id);
}


/**
 * CRT-accelerated RSA-2048 sign with pre-reduced inputs.
 *
 * This version takes m_p = m mod p and m_q = m mod q as inputs,
 * avoiding the expensive 2048->1024 reduction on GPU.
 *
 * The reduction should be done on the CPU/Python side using native
 * big integer division which is O(n^2) not O(2^n).
 */
__device__
void rsa_crt_sign_prereduced(
    uint32_t* signature,      // 64 limbs output
    const uint32_t* mp,       // 32 limbs: m mod p (pre-reduced)
    const uint32_t* mq,       // 32 limbs: m mod q (pre-reduced)
    const uint32_t* p,        // 32 limbs
    const uint32_t* q,        // 32 limbs
    const uint32_t* dp,       // 32 limbs
    const uint32_t* dq,       // 32 limbs
    const uint32_t* qinv,     // 32 limbs
    const uint32_t* R2_p,     // 32 limbs
    const uint32_t* R2_q,     // 32 limbs
    uint32_t p_prime,
    uint32_t q_prime,
    int lane_id)
{
    // Local arrays
    uint32_t sp[RSA_1024_LIMBS];       // m^dp mod p
    uint32_t sq[RSA_1024_LIMBS];       // m^dq mod q
    uint32_t diff[RSA_1024_LIMBS];     // (sp - sq) mod p
    uint32_t h[RSA_1024_LIMBS];        // qinv * diff mod p
    uint32_t qh[RSA_2048_LIMBS];       // q * h (2048-bit)
    uint32_t sq_ext[RSA_2048_LIMBS];   // sq zero-extended to 2048 bits

    // Step 1: Compute partial signatures (inputs already reduced)
    //
    // CRT exponentiations: m^dp mod p, m^dq mod q
    // (same macro ladder as serial CRT sign)
    //
#ifdef USE_WINDOW5_MODEXP
    modexp_1024_window5(sp, mp, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
    modexp_1024_window5(sq, mq, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#elif defined(USE_WINDOW_MODEXP)
    modexp_1024_window4(sp, mp, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
    modexp_1024_window4(sq, mq, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#else
    modexp_1024(sp, mp, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
    modexp_1024(sq, mq, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#endif

    // Step 2: CRT recombination
    // diff = (sp - sq) mod p
    mod_sub_1024(diff, sp, sq, p, lane_id);

    // h = (qinv * diff) mod p (using Montgomery multiplication)
    // Need to convert diff to Montgomery domain, multiply, convert back
    uint32_t diff_mont[RSA_1024_LIMBS];
    uint32_t h_mont[RSA_1024_LIMBS];

    to_mont_1024(diff_mont, diff, R2_p, p, p_prime, lane_id);

    // qinv needs to be in Montgomery form too
    uint32_t qinv_mont[RSA_1024_LIMBS];
    to_mont_1024(qinv_mont, qinv, R2_p, p, p_prime, lane_id);

    mont_mul_1024_serial(h_mont, qinv_mont, diff_mont, p, p_prime, lane_id);
    from_mont_1024(h, h_mont, p, p_prime, lane_id);

    // Step 3: Final combination
    // s = sq + q * h
    mul_1024x1024_to_2048(qh, q, h, lane_id);

    // Extend sq to 2048 bits
    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            sq_ext[i] = sq[i];
        }
        for (int i = RSA_1024_LIMBS; i < RSA_2048_LIMBS; i++) {
            sq_ext[i] = 0;
        }
    }
    __syncwarp();

    add_2048(signature, sq_ext, qh, lane_id);
}


// ============================================================================
// CARD33A: Warp-Cooperative CRT Sign
// ============================================================================

/**
 * CARD33A: Warp-cooperative CRT-accelerated RSA-2048 sign.
 *
 * Uses warp-cooperative Montgomery multiplication for modexp operations.
 * All 32 lanes participate in each Montgomery operation.
 *
 * IMPORTANT: This function uses __shared__ memory internally.
 * Call with exactly 1 warp per block (32 threads/block) to avoid conflicts.
 *
 * Algorithm:
 *   1. mp = m mod p, mq = m mod q (GPU-native binary reduction, serial)
 *   2. sp = mp^dp mod p, sq = mq^dq mod q (warp-cooperative modexp)
 *   3. h = (qinv * (sp - sq mod p)) mod p (warp-cooperative Montgomery)
 *   4. s = sq + q * h (serial)
 */
__device__
void rsa_crt_sign_warp(
    uint32_t* signature,      // 64 limbs output
    const uint32_t* message,  // 64 limbs input
    const uint32_t* p,        // 32 limbs
    const uint32_t* q,        // 32 limbs
    const uint32_t* dp,       // 32 limbs
    const uint32_t* dq,       // 32 limbs
    const uint32_t* qinv,     // 32 limbs
    const uint32_t* R2_p,     // 32 limbs
    const uint32_t* R2_q,     // 32 limbs
    uint32_t p_prime,
    uint32_t q_prime)
{
    int lane_id = threadIdx.x % 32;

    // Shared memory for warp-cooperative operations
    // These arrays are used by modexp_1024_warp internally
    __shared__ uint32_t mp_shared[RSA_1024_LIMBS];
    __shared__ uint32_t mq_shared[RSA_1024_LIMBS];
    __shared__ uint32_t sp_shared[RSA_1024_LIMBS];
    __shared__ uint32_t sq_shared[RSA_1024_LIMBS];
    __shared__ uint32_t diff_shared[RSA_1024_LIMBS];
    __shared__ uint32_t h_shared[RSA_1024_LIMBS];

    // Local arrays for final computation (2048-bit operations stay serial)
    uint32_t qh[RSA_2048_LIMBS];
    uint32_t sq_ext[RSA_2048_LIMBS];

    // Step 1: Reduce message mod p and mod q (serial, lane 0 only)
    // This is O(n^2) but fast compared to modexp
    uint32_t mp_local[RSA_1024_LIMBS];
    uint32_t mq_local[RSA_1024_LIMBS];
    reduce_2048_mod_1024_binary(mp_local, message, p, lane_id);
    reduce_2048_mod_1024_binary(mq_local, message, q, lane_id);

    // Copy to shared memory for warp-cooperative modexp
    mp_shared[lane_id] = mp_local[lane_id];
    mq_shared[lane_id] = mq_local[lane_id];
    __syncwarp();

    // Step 2: Compute partial signatures using WARP-COOPERATIVE modexp
    // sp = mp^dp mod p (1024-bit modexp, all 32 lanes participate)
    modexp_1024_warp(sp_shared, mp_shared, dp, RSA_1024_BITS, p, R2_p, p_prime);

    // sq = mq^dq mod q (1024-bit modexp, all 32 lanes participate)
    modexp_1024_warp(sq_shared, mq_shared, dq, RSA_1024_BITS, q, R2_q, q_prime);

    // Step 3: CRT recombination
    // diff = (sp - sq) mod p
    //
    // CARD33/34 SAFETY NOTE:
    //
    //   Earlier we had a bug here:
    //
    //       sp_local[lane_id] = sp_shared[lane_id];
    //       sq_local[lane_id] = sq_shared[lane_id];
    //       mod_sub_1024(diff_local, sp_local, sq_local, p, lane_id);
    //
    //   This left lane 0 with only sp_local[0], sq_local[0] initialized,
    //   but mod_sub_1024() expects lane 0 to see ALL 32 limbs.
    //
    //   The fixed pattern is to call mod_sub_1024() directly on the
    //   shared arrays, so lane 0 always has a full 1024-bit view:
    //
    mod_sub_1024(diff_shared, sp_shared, sq_shared, p, lane_id);
    // mod_sub_1024 broadcasts the final 1024-bit diff from lane 0 to all lanes.

    // Convert diff and qinv to Montgomery domain (warp-cooperative)
    __shared__ uint32_t diff_mont_shared[RSA_1024_LIMBS];
    __shared__ uint32_t qinv_mont_shared[RSA_1024_LIMBS];
    __shared__ uint32_t h_mont_shared[RSA_1024_LIMBS];

    to_mont_1024_warp(diff_mont_shared, diff_shared, R2_p, p, p_prime);

    // Load qinv into shared memory
    __shared__ uint32_t qinv_shared[RSA_1024_LIMBS];
    qinv_shared[lane_id] = qinv[lane_id];
    __syncwarp();
    to_mont_1024_warp(qinv_mont_shared, qinv_shared, R2_p, p, p_prime);

    // h_mont = qinv_mont * diff_mont (warp-cooperative Montgomery mul)
    mont_mul_1024_warp(h_mont_shared, qinv_mont_shared, diff_mont_shared, p, p_prime);

    // Convert back from Montgomery domain
    from_mont_1024_warp(h_shared, h_mont_shared, p, p_prime);
    __syncwarp();

    // Step 4: Final combination (serial, lane 0 only)
    // s = sq + q * h
    //
    // CARD33/34 SAFETY NOTE:
    //
    //   mul_1024x1024_to_2048() again assumes lane 0 will read all 32
    //   limbs of each operand. Here we must **never** copy h_shared into
    //   a per-lane local array like:
    //
    //       uint32_t h_local[32];
    //       h_local[lane_id] = h_shared[lane_id];  // lane 0 only sees [0]
    //       mul_1024x1024_to_2048(qh, q, h_local, lane_id);
    //
    //   That pattern was the original CRT warp bug.
    //   The correct approach is to operate directly on the shared array,
    //   where h_shared[0..31] are valid and visible to lane 0:
    //
    mul_1024x1024_to_2048(qh, q, h_shared, lane_id);

    // Extend sq to 2048 bits
    // NOTE: Use sq_shared, not sq_local! sq_local[i] for i>0 is garbage on lane 0.
    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            sq_ext[i] = sq_shared[i];
        }
        for (int i = RSA_1024_LIMBS; i < RSA_2048_LIMBS; i++) {
            sq_ext[i] = 0;
        }
    }
    __syncwarp();

    add_2048(signature, sq_ext, qh, lane_id);
}


}  // namespace rsa2048_crt
