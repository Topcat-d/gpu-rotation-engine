// ==============================================================================
// P-256 Batch Affine Conversion - Reference Implementation
// ==============================================================================
// Purpose: Batch Jacobian → Affine conversion using Montgomery's trick
// Algorithm: Single inversion for N points (vs N inversions)
// Speedup: ~6x for N=8, ~10x for N=16, ~30x for N=100
//
// This is a NEW implementation built on top of the FROZEN golden reference.
// Do NOT modify p256_scalar_ref_golden.cuh - instead, modify this file.
// ==============================================================================

#ifndef P256_BATCH_AFFINE_REF_CUH
#define P256_BATCH_AFFINE_REF_CUH

#include "p256_scalar_ref_golden.cuh"
#include <vector>
#include <array>

// ==============================================================================
// Additional Helper Functions Not in Golden Reference
// ==============================================================================

// Check if field element is zero
__device__ __forceinline__ bool fe_is_zero_scalar(const uint32_t a[8]) {
    for (int i = 0; i < 8; ++i) {
        if (a[i] != 0) return false;
    }
    return true;
}

// Set field element to zero
__device__ __forceinline__ void fe_set_zero_scalar(uint32_t r[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = 0;
}

// Set field element to one (Montgomery "one")
__device__ __forceinline__ void fe_set_one_mont_scalar(uint32_t r[8]) {
    // Montgomery representation of 1 is R mod p
    // For P-256, R mod p = 1 (since R = 2^256 and 2^256 mod p = 1)
    r[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; ++i) r[i] = 0;
}

// Field inversion using Fermat's little theorem: a^{-1} = a^{p-2} mod p
// This uses binary exponentiation (square-and-multiply)
__device__ void fe_inv_scalar(uint32_t r[8], const uint32_t a[8]) {
    // P-256 prime p - 2 in little-endian limbs
    const uint32_t p_minus_2[8] = {
        0xfffffffd, 0xffffffff, 0xffffffff, 0x00000000,
        0x00000000, 0x00000000, 0x00000001, 0xffffffff
    };

    uint32_t result[8];
    fe_set_one_mont_scalar(result);  // result = 1 (Montgomery)

    uint32_t base[8];
    fe_copy_scalar(base, a);  // base = a

    // Binary exponentiation: compute a^(p-2) mod p
    for (int limb = 0; limb < 8; ++limb) {
        uint32_t exp_limb = p_minus_2[limb];

        for (int bit = 0; bit < 32; ++bit) {
            if (exp_limb & 1) {
                // result = result * base
                fe_mont_mul_scalar(result, result, base);
            }

            // base = base^2
            fe_mont_sqr_scalar(base, base);

            exp_limb >>= 1;
        }
    }

    fe_copy_scalar(r, result);
}

// ==============================================================================
// Affine Point Struct
// ==============================================================================

struct P256_Affine_Scalar {
    uint32_t x[8];  // Montgomery field element
    uint32_t y[8];  // Montgomery field element
};

// ==============================================================================
// Batch Jacobian → Affine Conversion
// ==============================================================================

/**
 * Batch convert N Jacobian points to affine using Montgomery's trick.
 *
 * Algorithm:
 *   1. Build prefix products: prefix[i] = Z[0] * Z[1] * ... * Z[i-1]
 *   2. Invert total product: inv_total = (prefix[N])^{-1}
 *   3. Walk backwards to recover each Z[i]^{-1}
 *   4. Convert each point: x = X * Z^{-2}, y = Y * Z^{-3}
 *
 * Cost:
 *   - Single-point conversion: N inversions + 2N multiplications
 *   - Batch conversion: 1 inversion + O(N) multiplications
 *   - Speedup: ~6-30x depending on N
 *
 * Inputs:
 *   - in[i]: Jacobian points (X, Y, Z) in Montgomery domain
 *   - N: number of points
 *
 * Outputs:
 *   - out[i]: Affine points (x, y) in Montgomery domain
 *   - Points with Z=0 (infinity) are mapped to (0, 0)
 */
__device__ void batch_affine_from_jacobian_scalar(
    const P256_Jacobian_Scalar* in,
    P256_Affine_Scalar* out,
    int N
) {
    if (N <= 0) return;

    // Allocate prefix product array (size N+1)
    // prefix[i] = Z[0] * Z[1] * ... * Z[i-1]
    uint32_t* prefix = new uint32_t[(N + 1) * 8];

    // Step 1: Build prefix products
    fe_set_one_mont_scalar(prefix);  // prefix[0] = 1

    for (int i = 0; i < N; ++i) {
        uint32_t* prev = prefix + i * 8;
        uint32_t* curr = prefix + (i + 1) * 8;

        // If Z[i] == 0 (infinity), treat as 1 for prefix product
        if (fe_is_zero_scalar(in[i].Z)) {
            fe_copy_scalar(curr, prev);
        } else {
            fe_mont_mul_scalar(curr, prev, in[i].Z);
        }
    }

    // Step 2: Invert total product
    uint32_t inv_total[8];
    fe_inv_scalar(inv_total, prefix + N * 8);

    // Step 3: Walk backwards to recover each Z[i]^{-1} and convert
    for (int i = N - 1; i >= 0; --i) {
        // If this point is infinity, output (0, 0)
        if (fe_is_zero_scalar(in[i].Z)) {
            fe_set_zero_scalar(out[i].x);
            fe_set_zero_scalar(out[i].y);
            continue;
        }

        // Recover Z[i]^{-1} = prefix[i] * inv_total
        uint32_t invZi[8];
        fe_mont_mul_scalar(invZi, prefix + i * 8, inv_total);

        // Update inv_total for next iteration: inv_total *= Z[i]
        fe_mont_mul_scalar(inv_total, inv_total, in[i].Z);

        // Compute invZi^2 and invZi^3
        uint32_t invZi2[8], invZi3[8];
        fe_mont_sqr_scalar(invZi2, invZi);           // invZi2 = invZi^2
        fe_mont_mul_scalar(invZi3, invZi2, invZi);   // invZi3 = invZi^3

        // Convert to affine: x = X * invZi^2, y = Y * invZi^3
        fe_mont_mul_scalar(out[i].x, in[i].X, invZi2);
        fe_mont_mul_scalar(out[i].y, in[i].Y, invZi3);

        // Optional: Convert from Montgomery to normal domain
        // from_mont_scalar(out[i].x, out[i].x);
        // from_mont_scalar(out[i].y, out[i].y);
    }

    // Cleanup
    delete[] prefix;
}

// ==============================================================================
// Host-Side Wrapper (for CPU testing via pybind11)
// ==============================================================================

#ifndef __CUDA_ARCH__
// This version runs on CPU for testing (no __device__ qualifier)

#include <cstring>

// Host-side constants (copy of device constants)
static const uint32_t P256_P_HOST[8] = {
    0xffffffffu, 0xffffffffu, 0xffffffffu, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000001u, 0xffffffffu
};

static const uint32_t P256_R2_HOST[8] = {
    0x00000003u, 0x00000000u, 0xffffffffu, 0xfffffffbu,
    0xfffffffeu, 0xffffffffu, 0xfffffffdu, 0x00000004u
};

inline bool fe_is_zero_host(const uint32_t a[8]) {
    for (int i = 0; i < 8; ++i) {
        if (a[i] != 0) return false;
    }
    return true;
}

inline void fe_set_zero_host(uint32_t r[8]) {
    for (int i = 0; i < 8; ++i) r[i] = 0;
}

inline void fe_set_one_mont_host(uint32_t r[8]) {
    r[0] = 1;
    for (int i = 1; i < 8; ++i) r[i] = 0;
}

inline void fe_copy_host(uint32_t r[8], const uint32_t a[8]) {
    memcpy(r, a, 8 * sizeof(uint32_t));
}

inline int fe_cmp_host(const uint32_t a[8], const uint32_t b[8]) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// Host version of Montgomery multiplication (CIOS)
inline void mont_mul_host(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint64_t t[17] = {0};

    // Schoolbook multiply
    for (int i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t uv = t[i + j] + (uint64_t)a[j] * (uint64_t)b[i] + carry;
            t[i + j] = (uint32_t)uv;
            carry = uv >> 32;
        }
        uint64_t acc_sb = t[i + 8] + carry;
        t[i + 8] = (uint32_t)acc_sb;
        uint64_t carry_sb = acc_sb >> 32;
        t[i + 9] += carry_sb;
    }

    // CIOS reduction
    const uint32_t m0 = 0x00000001u;  // mont_inv32_p for P-256
    for (int i = 0; i < 8; ++i) {
        uint32_t m = (uint32_t)t[i] * m0;
        uint64_t carry = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t uv = t[i + j] + (uint64_t)m * (uint64_t)P256_P_HOST[j] + carry;
            t[i + j] = (uint32_t)uv;
            carry = uv >> 32;
        }
        uint64_t acc = t[i + 8] + carry;
        t[i + 8] = (uint32_t)acc;
        uint64_t carry2 = acc >> 32;
        t[i + 9] += carry2;
    }

    // Take high 8 limbs
    for (int i = 0; i < 8; ++i) {
        r[i] = (uint32_t)t[i + 8];
    }

    // Final conditional subtract
    uint32_t final_carry = (uint32_t)t[16];
    bool needs_subtract = (final_carry != 0) || (fe_cmp_host(r, P256_P_HOST) >= 0);

    if (needs_subtract) {
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)r[i] - (uint64_t)P256_P_HOST[i] - borrow;
            r[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
    }
}

inline void mont_sqr_host(uint32_t r[8], const uint32_t a[8]) {
    mont_mul_host(r, a, a);
}

// Field inversion using Fermat's little theorem: a^{-1} = a^{p-2} mod p
inline void fe_inv_host(uint32_t r[8], const uint32_t a[8]) {
    const uint32_t p_minus_2[8] = {
        0xfffffffd, 0xffffffff, 0xffffffff, 0x00000000,
        0x00000000, 0x00000000, 0x00000001, 0xffffffff
    };

    uint32_t result[8];
    fe_set_one_mont_host(result);

    uint32_t base[8];
    fe_copy_host(base, a);

    // Binary exponentiation: compute a^(p-2) mod p
    for (int limb = 0; limb < 8; ++limb) {
        uint32_t exp_limb = p_minus_2[limb];

        for (int bit = 0; bit < 32; ++bit) {
            if (exp_limb & 1) {
                mont_mul_host(result, result, base);
            }
            mont_sqr_host(base, base);
            exp_limb >>= 1;
        }
    }

    fe_copy_host(r, result);
}

// Batch affine conversion using Montgomery's trick (inline helper)
inline void batch_affine_from_jacobian_host_impl(
    const P256_Jacobian_Scalar* in,
    P256_Affine_Scalar* out,
    int N
) {
    if (N <= 0) return;

    // Allocate prefix product array (size N+1)
    uint32_t* prefix = new uint32_t[(N + 1) * 8];

    // Step 1: Build prefix products
    fe_set_one_mont_host(prefix);  // prefix[0] = 1

    for (int i = 0; i < N; ++i) {
        uint32_t* prev = prefix + i * 8;
        uint32_t* curr = prefix + (i + 1) * 8;

        // If Z[i] == 0 (infinity), treat as 1 for prefix product
        if (fe_is_zero_host(in[i].Z)) {
            fe_copy_host(curr, prev);
        } else {
            mont_mul_host(curr, prev, in[i].Z);
        }
    }

    // Step 2: Invert total product
    uint32_t inv_total[8];
    fe_inv_host(inv_total, prefix + N * 8);

    // Step 3: Walk backwards to recover each Z[i]^{-1} and convert
    for (int i = N - 1; i >= 0; --i) {
        // If this point is infinity, output (0, 0)
        if (fe_is_zero_host(in[i].Z)) {
            fe_set_zero_host(out[i].x);
            fe_set_zero_host(out[i].y);
            continue;
        }

        // Recover Z[i]^{-1} = prefix[i] * inv_total
        uint32_t invZi[8];
        mont_mul_host(invZi, prefix + i * 8, inv_total);

        // Update inv_total for next iteration: inv_total *= Z[i]
        mont_mul_host(inv_total, inv_total, in[i].Z);

        // Compute invZi^2 and invZi^3
        uint32_t invZi2[8], invZi3[8];
        mont_sqr_host(invZi2, invZi);
        mont_mul_host(invZi3, invZi2, invZi);

        // Convert to affine: x = X * invZi^2, y = Y * invZi^3
        mont_mul_host(out[i].x, in[i].X, invZi2);
        mont_mul_host(out[i].y, in[i].Y, invZi3);
    }

    // Cleanup
    delete[] prefix;
}

#endif // __CUDA_ARCH__

#endif // P256_BATCH_AFFINE_REF_CUH
