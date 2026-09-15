/*
 * P-256 Scalar Order (n) Constants and Modular Arithmetic
 *
 * CARD20: Full ECDSA Sign Pipeline
 *
 * This file provides:
 * - Curve order n and related Montgomery constants
 * - mod-n arithmetic operations (mul, add, sub, inv)
 * - All operations in single Montgomery domain (a*R mod n)
 *
 * The curve order n for P-256 (secp256r1):
 *   n = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
 */

#ifndef P256_SCALAR_ORDER_CUH
#define P256_SCALAR_ORDER_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// P-256 Curve Order Constants (Little-Endian Limbs)
// ============================================================================

// Curve order n = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
__constant__ uint32_t P256_N[8] = {
    0xfc632551u, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
    0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
};

// n - 2 (for Fermat inverse: a^(n-2) mod n)
__constant__ uint32_t P256_N_MINUS_2[8] = {
    0xfc63254fu, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
    0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
};

// Montgomery constant R mod n = 2^256 mod n
// R mod n = 0x00000000ffffffff00000000000000004319055258e8617b0c46353d039cdaaf
__constant__ uint32_t P256_N_R[8] = {
    0x039cdaafu, 0x0c46353du, 0x58e8617bu, 0x43190552u,
    0x00000000u, 0x00000000u, 0xffffffffu, 0x00000000u
};

// Montgomery constant R^2 mod n
// R^2 mod n = 0x66e12d94f3d956202845b2392b6bec594699799c49bd6fa683244c95be79eea2
__constant__ uint32_t P256_N_R2[8] = {
    0xbe79eea2u, 0x83244c95u, 0x49bd6fa6u, 0x4699799cu,
    0x2b6bec59u, 0x2845b239u, 0xf3d95620u, 0x66e12d94u
};

// Montgomery constant n0' = -n^(-1) mod 2^32
// For P-256 order n, n0' = 0xee00bc4f
__constant__ uint32_t P256_N0_PRIME = 0xee00bc4fu;

// Montgomery one (1*R mod n) = R mod n
__constant__ uint32_t P256_N_MONT_ONE[8] = {
    0x039cdaafu, 0x0c46353du, 0x58e8617bu, 0x43190552u,
    0x00000000u, 0x00000000u, 0xffffffffu, 0x00000000u
};

// ============================================================================
// Scalar Oracle Functions (mod n arithmetic on lane 0)
// ============================================================================

/*
 * Montgomery multiplication mod n (scalar, lane 0 only)
 * Computes: out = (A * B * R^-1) mod n using CIOS algorithm
 */
__device__ __forceinline__
void mont_mul_scalar_n(uint32_t out[8], const uint32_t A[8], const uint32_t B[8])
{
    const uint32_t n0_prime = 0xee00bc4fu;

    // Local copy of n
    const uint32_t N[8] = {
        0xfc632551u, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
        0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
    };

    // CIOS Montgomery multiplication
    unsigned long long T[9] = {0};

    for (int i = 0; i < 8; ++i) {
        // Step A: T = T + A[i] * B
        unsigned long long carry = 0;
        for (int j = 0; j < 8; ++j) {
            unsigned long long prod = (unsigned long long)A[i] * B[j];
            unsigned long long sum = T[j] + (prod & 0xFFFFFFFFull) + carry;
            T[j] = (uint32_t)sum;
            carry = (sum >> 32) + (prod >> 32);
        }
        T[8] += carry;

        // Step B: m = T[0] * n0'
        uint32_t m = (uint32_t)T[0] * n0_prime;

        // Step B: T = T + m * N
        carry = 0;
        for (int j = 0; j < 8; ++j) {
            unsigned long long prod = (unsigned long long)m * N[j];
            unsigned long long sum = T[j] + (prod & 0xFFFFFFFFull) + carry;
            T[j] = (uint32_t)sum;
            carry = (sum >> 32) + (prod >> 32);
        }
        T[8] += carry;

        // Step C: T = T >> 32
        for (int k = 0; k < 8; ++k) T[k] = T[k + 1];
        T[8] = 0;
    }

    // Final reduction: if T >= N, subtract N
    bool ge = (T[8] != 0);
    if (!ge) {
        for (int i = 7; i >= 0; --i) {
            if (T[i] > N[i]) { ge = true; break; }
            if (T[i] < N[i]) { ge = false; break; }
        }
    }

    unsigned long long borrow = 0;
    for (int i = 0; i < 8; ++i) {
        unsigned long long xi = T[i];
        unsigned long long yi = ge ? N[i] : 0ull;
        unsigned long long diff = xi - yi - borrow;
        out[i] = (uint32_t)diff;
        borrow = (diff >> 63);
    }
}

/*
 * Montgomery addition mod n (scalar, lane 0 only)
 * Computes: out = (A + B) mod n
 * Note: A and B are in Montgomery domain, result is in Montgomery domain
 */
__device__ __forceinline__
void mont_add_scalar_n(uint32_t out[8], const uint32_t A[8], const uint32_t B[8])
{
    const uint32_t N[8] = {
        0xfc632551u, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
        0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
    };

    // Add A + B
    unsigned long long carry = 0;
    uint32_t sum[8];
    for (int i = 0; i < 8; ++i) {
        unsigned long long s = (unsigned long long)A[i] + B[i] + carry;
        sum[i] = (uint32_t)s;
        carry = s >> 32;
    }

    // Check if sum >= N
    bool ge = (carry != 0);
    if (!ge) {
        for (int i = 7; i >= 0; --i) {
            if (sum[i] > N[i]) { ge = true; break; }
            if (sum[i] < N[i]) { ge = false; break; }
        }
    }

    // Conditional subtract N
    unsigned long long borrow = 0;
    for (int i = 0; i < 8; ++i) {
        unsigned long long xi = sum[i];
        unsigned long long yi = ge ? N[i] : 0ull;
        unsigned long long diff = xi - yi - borrow;
        out[i] = (uint32_t)diff;
        borrow = (diff >> 63);
    }
}

/*
 * Montgomery subtraction mod n (scalar, lane 0 only)
 * Computes: out = (A - B) mod n
 */
__device__ __forceinline__
void mont_sub_scalar_n(uint32_t out[8], const uint32_t A[8], const uint32_t B[8])
{
    const uint32_t N[8] = {
        0xfc632551u, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
        0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
    };

    // Subtract A - B
    long long borrow = 0;
    uint32_t diff[8];
    for (int i = 0; i < 8; ++i) {
        long long d = (long long)A[i] - B[i] - borrow;
        diff[i] = (uint32_t)d;
        borrow = (d < 0) ? 1 : 0;
    }

    // If borrow, add N back
    if (borrow) {
        unsigned long long carry = 0;
        for (int i = 0; i < 8; ++i) {
            unsigned long long s = (unsigned long long)diff[i] + N[i] + carry;
            out[i] = (uint32_t)s;
            carry = s >> 32;
        }
    } else {
        for (int i = 0; i < 8; ++i) out[i] = diff[i];
    }
}

/*
 * To Montgomery domain mod n (scalar, lane 0 only)
 * Computes: out = (A * R) mod n = mont_mul(A, R^2)
 */
__device__ __forceinline__
void to_mont_n(uint32_t out[8], const uint32_t A[8])
{
    // R^2 mod n
    const uint32_t R2[8] = {
        0xbe79eea2u, 0x83244c95u, 0x49bd6fa6u, 0x4699799cu,
        0x2b6bec59u, 0x2845b239u, 0xf3d95620u, 0x66e12d94u
    };
    mont_mul_scalar_n(out, A, R2);
}

/*
 * From Montgomery domain mod n (scalar, lane 0 only)
 * Computes: out = (A * 1) mod n = mont_mul(A, 1)
 */
__device__ __forceinline__
void from_mont_n(uint32_t out[8], const uint32_t A[8])
{
    const uint32_t one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    mont_mul_scalar_n(out, A, one);
}

/*
 * Montgomery Fermat inverse mod n (scalar, lane 0 only)
 * Computes: out = A^(-1) mod n = A^(n-2) mod n using binary exponentiation
 */
__device__ __forceinline__
void mont_fermat_inv_n(uint32_t out[8], const uint32_t A[8])
{
    // n - 2 in little-endian limbs
    const uint32_t n_minus_2[8] = {
        0xfc63254fu, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
        0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
    };

    // R mod n (Montgomery one)
    const uint32_t R[8] = {
        0x039cdaafu, 0x0c46353du, 0x58e8617bu, 0x43190552u,
        0x00000000u, 0x00000000u, 0xffffffffu, 0x00000000u
    };

    // Initialize result = R (Montgomery representation of 1)
    uint32_t result[8];
    for (int i = 0; i < 8; ++i) result[i] = R[i];

    // Copy base
    uint32_t base[8];
    for (int i = 0; i < 8; ++i) base[i] = A[i];

    // Binary exponentiation: scan bits of n-2 from LSB to MSB
    for (int limb_idx = 0; limb_idx < 8; limb_idx++) {
        uint32_t exp_limb = n_minus_2[limb_idx];
        for (int bit = 0; bit < 32; bit++) {
            // If bit is set, multiply result by base
            if (exp_limb & (1u << bit)) {
                uint32_t temp[8];
                mont_mul_scalar_n(temp, result, base);
                for (int i = 0; i < 8; ++i) result[i] = temp[i];
            }
            // Square base for next bit (skip last iteration)
            if (limb_idx < 7 || bit < 31) {
                uint32_t temp[8];
                mont_mul_scalar_n(temp, base, base);
                for (int i = 0; i < 8; ++i) base[i] = temp[i];
            }
        }
    }

    // Copy result to output
    for (int i = 0; i < 8; ++i) out[i] = result[i];
}

/*
 * Reduce x coordinate mod n (for ECDSA r computation)
 * Input: x in normal domain (not Montgomery)
 * Output: r = x mod n in normal domain
 *
 * Since P-256 has x < p and n < p, we need at most one subtraction.
 */
__device__ __forceinline__
void reduce_x_mod_n(uint32_t out[8], const uint32_t x[8])
{
    const uint32_t N[8] = {
        0xfc632551u, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
        0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
    };

    // Check if x >= N
    bool ge = false;
    for (int i = 7; i >= 0; --i) {
        if (x[i] > N[i]) { ge = true; break; }
        if (x[i] < N[i]) { ge = false; break; }
    }

    // Conditional subtract N
    if (ge) {
        unsigned long long borrow = 0;
        for (int i = 0; i < 8; ++i) {
            unsigned long long diff = (unsigned long long)x[i] - N[i] - borrow;
            out[i] = (uint32_t)diff;
            borrow = (diff >> 63);
        }
    } else {
        for (int i = 0; i < 8; ++i) out[i] = x[i];
    }
}

/*
 * Check if value is zero (scalar, lane 0 only)
 */
__device__ __forceinline__
bool is_zero_scalar(const uint32_t A[8])
{
    for (int i = 0; i < 8; ++i) {
        if (A[i] != 0) return false;
    }
    return true;
}

#endif // P256_SCALAR_ORDER_CUH
