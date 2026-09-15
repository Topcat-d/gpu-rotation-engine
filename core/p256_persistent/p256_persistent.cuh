/*
 * P-256 Persistent Engine Signing - Card 08 + Card 13 + Card 14 + Card 15
 *
 * Device-callable ECDSA signing for the persistent kernel.
 *
 * This header provides:
 *   p256_sign_persistent() - Compute ECDSA signature given hash and private key
 *
 * Dependencies: Reuses arithmetic from p256_scalar_order.cuh
 *
 * Algorithm (ECDSA P-256):
 *   1. Generate nonce k via:
 *      - Device-side RFC 6979 (Card 13) when nonce_mode = DETERMINISTIC
 *      - Device-side ChaCha20 CSPRNG (Card 15) when nonce_mode = RANDOM
 *   2. Compute R = k*G (fixed-base scalar multiplication)
 *   3. Convert R to affine (x, y) via Z-inversion
 *   4. r = x mod n
 *   5. s = k^(-1) * (hash + r*d) mod n
 *   6. Apply low-s normalization if enabled (Card 14)
 *   7. Return (r, s)
 *
 * Card 13: RFC 6979 nonce generation is now done entirely on the GPU
 * using the device-side HMAC-SHA256 implementation.
 *
 * Card 14: Added nonce mode selection (DETERMINISTIC/RANDOM) and
 * canonical low-s signature normalization.
 *
 * Card 15: RANDOM mode now uses a real ChaCha20-based CSPRNG for
 * non-deterministic nonce generation (research/stress testing mode).
 */

#ifndef P256_PERSISTENT_CUH
#define P256_PERSISTENT_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Card 13: Include device-side RFC 6979
#include "p256_rfc6979_device.cuh"

// Card 15: Include device-side CSPRNG for random nonces
#include "p256_csprng_device.cuh"

namespace smoke {
namespace p256 {

// ============================================================================
// P-256 Field Prime Constants (Little-Endian Limbs)
// ============================================================================

// P-256 prime p = 2^256 - 2^224 + 2^192 + 2^96 - 1
static __device__ __constant__ uint32_t P256_P[8] = {
    0xffffffffu, 0xffffffffu, 0xffffffffu, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000001u, 0xffffffffu
};

// R^2 mod p for Montgomery (2^512 mod p)
static __device__ __constant__ uint32_t P256_R2[8] = {
    0x00000003u, 0x00000000u, 0xffffffffu, 0xfffffffbu,
    0xfffffffeu, 0xffffffffu, 0xfffffffdu, 0x00000004u
};

// R mod p for Montgomery (2^256 mod p)
static __device__ __constant__ uint32_t P256_R[8] = {
    0x00000001u, 0x00000000u, 0x00000000u, 0xffffffffu,
    0xffffffffu, 0xffffffffu, 0xfffffffeu, 0x00000000u
};

// Base point Gx, Gy (little-endian)
static __device__ __constant__ uint32_t P256_GX[8] = {
    0xd898c296u, 0xf4a13945u, 0x2deb33a0u, 0x77037d81u,
    0x63a440f2u, 0xf8bce6e5u, 0xe12c4247u, 0x6b17d1f2u
};

static __device__ __constant__ uint32_t P256_GY[8] = {
    0x37bf51f5u, 0xcbb64068u, 0x6b315eceu, 0x2bce3357u,
    0x7c0f9e16u, 0x8ee7eb4au, 0xfe1a7f9bu, 0x4fe342e2u
};

// n0' = -(p^-1) mod 2^32
static __device__ __constant__ uint32_t P256_N0PRIME_P = 0x00000001u;

// ============================================================================
// P-256 Curve Order Constants (Little-Endian Limbs)
// ============================================================================

// Curve order n
static __device__ __constant__ uint32_t P256_N[8] = {
    0xfc632551u, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
    0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
};

// R^2 mod n for Montgomery (2^512 mod n)
static __device__ __constant__ uint32_t P256_N_R2[8] = {
    0xbe79eea2u, 0x83244c95u, 0x49bd6fa6u, 0x4699799cu,
    0x2b6bec59u, 0x2845b239u, 0xf3d95620u, 0x66e12d94u
};

// R mod n (Montgomery one)
static __device__ __constant__ uint32_t P256_N_R[8] = {
    0x039cdaafu, 0x0c46353du, 0x58e8617bu, 0x43190552u,
    0x00000000u, 0x00000000u, 0xffffffffu, 0x00000000u
};

// n0' = -(n^-1) mod 2^32
static __device__ __constant__ uint32_t P256_N0PRIME_N = 0xee00bc4fu;

// Card 14: n/2 (floor(n/2)) for low-s check
// n/2 = 0x7fffffff800000007fffffffffffffffde737d56d38bcf4279dce5617e3192a8
static __device__ __constant__ uint32_t P256_N_HALF[8] = {
    0x7e3192a8u, 0x79dce561u, 0xd38bcf42u, 0xde737d56u,
    0xffffffffu, 0x7fffffffu, 0x80000000u, 0x7fffffffu
};

// ============================================================================
// Helper Functions
// ============================================================================

__device__ __forceinline__ void copy256(uint32_t dst[8], const uint32_t src[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = src[i];
}

__device__ __forceinline__ bool is_zero256(const uint32_t a[8]) {
    uint32_t z = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) z |= a[i];
    return (z == 0);
}

__device__ __forceinline__ int cmp256(const uint32_t a[8], const uint32_t b[8]) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

// ============================================================================
// Montgomery Multiplication (mod p)
// ============================================================================

__device__ void mont_mul_p(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t t[9] = {0};

    for (int i = 0; i < 8; ++i) {
        uint64_t C = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)a[i] * (uint64_t)b[j];
            uint64_t sum = (uint64_t)t[j] + prod + C;
            t[j] = (uint32_t)sum;
            C = sum >> 32;
        }
        uint64_t sum_hi = (uint64_t)t[8] + C;
        t[8] = (uint32_t)sum_hi;

        uint32_t m = t[0] * P256_N0PRIME_P;

        C = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)m * (uint64_t)P256_P[j];
            uint64_t sum = (uint64_t)t[j] + prod + C;
            if (j > 0) t[j-1] = (uint32_t)sum;
            C = sum >> 32;
        }
        sum_hi = (uint64_t)t[8] + C;
        t[7] = (uint32_t)sum_hi;
        t[8] = (uint32_t)(sum_hi >> 32);
    }

    // Conditional subtract p
    bool need_sub = (t[8] != 0);
    if (!need_sub) {
        for (int i = 7; i >= 0; --i) {
            if (t[i] > P256_P[i]) { need_sub = true; break; }
            else if (t[i] < P256_P[i]) break;
        }
    }

    if (need_sub) {
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)t[i] - (uint64_t)P256_P[i] - borrow;
            r[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = t[i];
    }
}

__device__ __forceinline__ void mont_sqr_p(uint32_t r[8], const uint32_t a[8]) {
    mont_mul_p(r, a, a);
}

__device__ __forceinline__ void to_mont_p(uint32_t r[8], const uint32_t a[8]) {
    mont_mul_p(r, a, P256_R2);
}

__device__ __forceinline__ void from_mont_p(uint32_t r[8], const uint32_t a[8]) {
    uint32_t one[8] = {1,0,0,0,0,0,0,0};
    mont_mul_p(r, a, one);
}

// ============================================================================
// Modular Addition/Subtraction (mod p)
// ============================================================================

__device__ void mod_add_p(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint64_t carry = 0;
    uint32_t sum[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t s = (uint64_t)a[i] + (uint64_t)b[i] + carry;
        sum[i] = (uint32_t)s;
        carry = s >> 32;
    }

    bool need_sub = (carry != 0);
    if (!need_sub && cmp256(sum, P256_P) >= 0) need_sub = true;

    if (need_sub) {
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)sum[i] - (uint64_t)P256_P[i] - borrow;
            r[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = sum[i];
    }
}

__device__ void mod_sub_p(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint64_t borrow = 0;
    uint32_t diff[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t d = (uint64_t)a[i] - (uint64_t)b[i] - borrow;
        diff[i] = (uint32_t)d;
        borrow = (d >> 63) & 1;
    }

    if (borrow) {
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint64_t s = (uint64_t)diff[i] + (uint64_t)P256_P[i] + carry;
            r[i] = (uint32_t)s;
            carry = s >> 32;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = diff[i];
    }
}

// ============================================================================
// Montgomery Multiplication (mod n) - for ECDSA
// ============================================================================

__device__ void mont_mul_n(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint64_t T[9] = {0};

    for (int i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)a[i] * b[j];
            uint64_t sum = T[j] + (prod & 0xFFFFFFFFull) + carry;
            T[j] = (uint32_t)sum;
            carry = (sum >> 32) + (prod >> 32);
        }
        T[8] += carry;

        uint32_t m = (uint32_t)T[0] * P256_N0PRIME_N;

        carry = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)m * P256_N[j];
            uint64_t sum = T[j] + (prod & 0xFFFFFFFFull) + carry;
            T[j] = (uint32_t)sum;
            carry = (sum >> 32) + (prod >> 32);
        }
        T[8] += carry;

        for (int k = 0; k < 8; ++k) T[k] = T[k + 1];
        T[8] = 0;
    }

    // Final reduction
    bool ge = (T[8] != 0);
    if (!ge) {
        for (int i = 7; i >= 0; --i) {
            if (T[i] > P256_N[i]) { ge = true; break; }
            if (T[i] < P256_N[i]) { ge = false; break; }
        }
    }

    uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t xi = T[i];
        uint64_t yi = ge ? P256_N[i] : 0ull;
        uint64_t diff = xi - yi - borrow;
        r[i] = (uint32_t)diff;
        borrow = (diff >> 63);
    }
}

__device__ __forceinline__ void to_mont_n(uint32_t r[8], const uint32_t a[8]) {
    mont_mul_n(r, a, P256_N_R2);
}

__device__ __forceinline__ void from_mont_n(uint32_t r[8], const uint32_t a[8]) {
    uint32_t one[8] = {1,0,0,0,0,0,0,0};
    mont_mul_n(r, a, one);
}

__device__ void mod_add_n(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint64_t carry = 0;
    uint32_t sum[8];
    for (int i = 0; i < 8; ++i) {
        uint64_t s = (uint64_t)a[i] + b[i] + carry;
        sum[i] = (uint32_t)s;
        carry = s >> 32;
    }

    bool ge = (carry != 0);
    if (!ge) {
        for (int i = 7; i >= 0; --i) {
            if (sum[i] > P256_N[i]) { ge = true; break; }
            if (sum[i] < P256_N[i]) { ge = false; break; }
        }
    }

    uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t xi = sum[i];
        uint64_t yi = ge ? P256_N[i] : 0ull;
        uint64_t diff = xi - yi - borrow;
        r[i] = (uint32_t)diff;
        borrow = (diff >> 63);
    }
}

// ============================================================================
// Card 14: Low-s Normalization Helpers
// ============================================================================

// Check if s > n/2 (for canonical low-s signatures)
__device__ __forceinline__ bool is_greater_than_half_n(const uint32_t s[8]) {
    return cmp256(s, P256_N_HALF) > 0;
}

// Compute r = n - a (for low-s: if s > n/2, return n - s)
__device__ void mod_sub_from_n(uint32_t r[8], const uint32_t a[8]) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t diff = (uint64_t)P256_N[i] - (uint64_t)a[i] - borrow;
        r[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
    // Result is always valid since 0 < a < n implies 0 < (n - a) < n
}

// ============================================================================
// Modular Inversion (mod n)
// ============================================================================

// Fermat inverse mod n: a^(n-2) mod n
__device__ void fermat_inv_n(uint32_t r[8], const uint32_t a_mont[8]) {
    // n - 2 (little-endian limbs)
    const uint32_t n_minus_2[8] = {
        0xfc63254fu, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
        0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
    };

    // Initialize result = R (Montgomery 1)
    uint32_t result[8];
    copy256(result, P256_N_R);

    uint32_t base[8];
    copy256(base, a_mont);

    // Binary exponentiation
    for (int limb_idx = 0; limb_idx < 8; limb_idx++) {
        uint32_t exp_limb = n_minus_2[limb_idx];
        for (int bit = 0; bit < 32; bit++) {
            if (exp_limb & (1u << bit)) {
                uint32_t temp[8];
                mont_mul_n(temp, result, base);
                copy256(result, temp);
            }
            if (limb_idx < 7 || bit < 31) {
                uint32_t temp[8];
                mont_mul_n(temp, base, base);
                copy256(base, temp);
            }
        }
    }
    copy256(r, result);
}

// Fermat inverse mod p: a^(p-2) mod p
__device__ void fermat_inv_p(uint32_t r[8], const uint32_t a_mont[8]) {
    // p - 2 (little-endian limbs)
    // p = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
    // p-2 = 0xffffffff00000001000000000000000000000000fffffffffffffffffffffffd
    const uint32_t p_minus_2[8] = {
        0xfffffffdu, 0xffffffffu, 0xffffffffu, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000001u, 0xffffffffu
    };

    uint32_t result[8];
    copy256(result, P256_R);  // Montgomery 1 mod p

    uint32_t base[8];
    copy256(base, a_mont);

    for (int limb_idx = 0; limb_idx < 8; limb_idx++) {
        uint32_t exp_limb = p_minus_2[limb_idx];
        for (int bit = 0; bit < 32; bit++) {
            if (exp_limb & (1u << bit)) {
                uint32_t temp[8];
                mont_mul_p(temp, result, base);
                copy256(result, temp);
            }
            if (limb_idx < 7 || bit < 31) {
                uint32_t temp[8];
                mont_mul_p(temp, base, base);
                copy256(base, temp);
            }
        }
    }
    copy256(r, result);
}

// ============================================================================
// Point Operations (Jacobian Coordinates, Montgomery Domain)
// ============================================================================

__device__ void set_infinity(uint32_t X[8], uint32_t Y[8], uint32_t Z[8]) {
    copy256(X, P256_R);  // X = R (arbitrary non-zero)
    copy256(Y, P256_R);  // Y = R
    for (int i = 0; i < 8; ++i) Z[i] = 0;  // Z = 0 marks infinity
}

// Point doubling: 2P -> R (Jacobian, Montgomery)
// Uses the formula for a=-3 curves (P-256 has a=-3)
// Reference: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#doubling-dbl-2001-b
__device__ void point_double(
    const uint32_t X1[8], const uint32_t Y1[8], const uint32_t Z1[8],
    uint32_t X3[8], uint32_t Y3[8], uint32_t Z3[8]
) {
    if (is_zero256(Z1)) {
        set_infinity(X3, Y3, Z3);
        return;
    }

    uint32_t delta[8], gamma[8], beta[8], alpha[8], t[8], t2[8];

    // delta = Z1^2
    mont_sqr_p(delta, Z1);

    // gamma = Y1^2
    mont_sqr_p(gamma, Y1);

    // beta = X1 * gamma
    mont_mul_p(beta, X1, gamma);

    // For a=-3: alpha = 3*(X1 - delta)*(X1 + delta) = 3*(X1^2 - Z1^4)
    // This is equivalent to 3*X1^2 + a*Z1^4 = 3*X1^2 - 3*Z1^4 for a=-3
    mod_sub_p(t, X1, delta);   // t = X1 - Z1^2
    mod_add_p(t2, X1, delta);  // t2 = X1 + Z1^2
    mont_mul_p(alpha, t, t2);  // alpha = (X1 - Z1^2)*(X1 + Z1^2) = X1^2 - Z1^4
    mod_add_p(t, alpha, alpha);  // t = 2*alpha
    mod_add_p(alpha, t, alpha);  // alpha = 3*alpha = 3*(X1^2 - Z1^4)

    // X3 = alpha^2 - 8*beta
    mont_sqr_p(X3, alpha);
    mod_add_p(t, beta, beta);   // 2*beta
    mod_add_p(t, t, t);         // 4*beta
    mod_add_p(t, t, t);         // 8*beta
    mod_sub_p(X3, X3, t);

    // Z3 = (Y1 + Z1)^2 - gamma - delta
    mod_add_p(t, Y1, Z1);
    mont_sqr_p(Z3, t);
    mod_sub_p(Z3, Z3, gamma);
    mod_sub_p(Z3, Z3, delta);

    // Y3 = alpha * (4*beta - X3) - 8*gamma^2
    mod_add_p(t, beta, beta);   // 2*beta
    mod_add_p(t, t, t);         // 4*beta
    mod_sub_p(t, t, X3);        // 4*beta - X3
    mont_mul_p(Y3, alpha, t);   // alpha * (4*beta - X3)
    mont_sqr_p(t, gamma);       // gamma^2
    mod_add_p(t, t, t);         // 2*gamma^2
    mod_add_p(t, t, t);         // 4*gamma^2
    mod_add_p(t, t, t);         // 8*gamma^2
    mod_sub_p(Y3, Y3, t);       // Y3 = alpha*(4*beta - X3) - 8*gamma^2
}

// Mixed addition: P (Jacobian) + Q (Affine) -> R (Jacobian)
__device__ void point_add_mixed(
    const uint32_t X1[8], const uint32_t Y1[8], const uint32_t Z1[8],
    const uint32_t x2[8], const uint32_t y2[8],  // Affine, Montgomery domain
    uint32_t X3[8], uint32_t Y3[8], uint32_t Z3[8]
) {
    if (is_zero256(Z1)) {
        // P1 = infinity, return P2 in Jacobian form
        copy256(X3, x2);
        copy256(Y3, y2);
        copy256(Z3, P256_R);  // Z = 1 in Montgomery
        return;
    }

    uint32_t Z1Z1[8], U2[8], S2[8], H[8], HH[8], I[8], J[8], r[8], V[8], t[8];

    // Z1Z1 = Z1^2
    mont_sqr_p(Z1Z1, Z1);

    // U2 = x2 * Z1Z1
    mont_mul_p(U2, x2, Z1Z1);

    // S2 = y2 * Z1 * Z1Z1
    mont_mul_p(t, Z1, Z1Z1);
    mont_mul_p(S2, y2, t);

    // H = U2 - X1
    mod_sub_p(H, U2, X1);

    // r = 2*(S2 - Y1)
    mod_sub_p(t, S2, Y1);
    mod_add_p(r, t, t);

    if (is_zero256(H)) {
        if (is_zero256(r)) {
            // P1 == P2, need doubling
            point_double(X1, Y1, Z1, X3, Y3, Z3);
        } else {
            // P1 == -P2, result is infinity
            set_infinity(X3, Y3, Z3);
        }
        return;
    }

    // HH = H^2
    mont_sqr_p(HH, H);

    // I = 4*HH
    mod_add_p(I, HH, HH);
    mod_add_p(I, I, I);

    // J = H*I
    mont_mul_p(J, H, I);

    // V = X1*I
    mont_mul_p(V, X1, I);

    // X3 = r^2 - J - 2*V
    mont_sqr_p(X3, r);
    mod_sub_p(X3, X3, J);
    mod_add_p(t, V, V);
    mod_sub_p(X3, X3, t);

    // Y3 = r*(V - X3) - 2*Y1*J
    mod_sub_p(t, V, X3);
    mont_mul_p(Y3, r, t);
    mont_mul_p(t, Y1, J);
    mod_add_p(t, t, t);
    mod_sub_p(Y3, Y3, t);

    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    mod_add_p(t, Z1, H);
    mont_sqr_p(Z3, t);
    mod_sub_p(Z3, Z3, Z1Z1);
    mod_sub_p(Z3, Z3, HH);
}

// ============================================================================
// Fixed-Base Scalar Multiplication (k * G)
// ============================================================================

// Simple double-and-add (no precomputed table for Card 08)
// Future optimization: use window method with precomputed G table
__device__ void scalar_mul_g(
    const uint32_t k[8],
    uint32_t Qx[8], uint32_t Qy[8], uint32_t Qz[8]
) {
    // Initialize accumulator to infinity
    set_infinity(Qx, Qy, Qz);

    // Convert G to Montgomery domain
    uint32_t Gx_mont[8], Gy_mont[8];
    to_mont_p(Gx_mont, P256_GX);
    to_mont_p(Gy_mont, P256_GY);

    // Double-and-add from MSB to LSB
    for (int limb = 7; limb >= 0; --limb) {
        uint32_t limb_val = k[limb];
        for (int bit = 31; bit >= 0; --bit) {
            // Double
            uint32_t temp_X[8], temp_Y[8], temp_Z[8];
            point_double(Qx, Qy, Qz, temp_X, temp_Y, temp_Z);
            copy256(Qx, temp_X);
            copy256(Qy, temp_Y);
            copy256(Qz, temp_Z);

            // Add if bit is set
            if (limb_val & (1u << bit)) {
                point_add_mixed(Qx, Qy, Qz, Gx_mont, Gy_mont, temp_X, temp_Y, temp_Z);
                copy256(Qx, temp_X);
                copy256(Qy, temp_Y);
                copy256(Qz, temp_Z);
            }
        }
    }
}

// ============================================================================
// Jacobian to Affine Conversion (with Z-inversion)
// ============================================================================

// Convert Jacobian (X, Y, Z) to affine (x, y), both in Montgomery domain
// Returns false if point is at infinity
__device__ bool jacobian_to_affine(
    const uint32_t X[8], const uint32_t Y[8], const uint32_t Z[8],
    uint32_t x_out[8], uint32_t y_out[8]
) {
    if (is_zero256(Z)) {
        return false;  // Point at infinity
    }

    // z_inv = Z^(-1) mod p
    uint32_t z_inv[8];
    fermat_inv_p(z_inv, Z);

    // z_inv2 = z_inv^2
    uint32_t z_inv2[8];
    mont_sqr_p(z_inv2, z_inv);

    // z_inv3 = z_inv^3 = z_inv2 * z_inv
    uint32_t z_inv3[8];
    mont_mul_p(z_inv3, z_inv2, z_inv);

    // x = X * z_inv2
    mont_mul_p(x_out, X, z_inv2);

    // y = Y * z_inv3
    mont_mul_p(y_out, Y, z_inv3);

    return true;
}

// ============================================================================
// ECDSA Signature Computation
// ============================================================================

// Reduce x coordinate mod n (for r computation)
__device__ void reduce_x_mod_n(uint32_t r[8], const uint32_t x[8]) {
    // Check if x >= n
    bool ge = false;
    for (int i = 7; i >= 0; --i) {
        if (x[i] > P256_N[i]) { ge = true; break; }
        if (x[i] < P256_N[i]) { ge = false; break; }
    }

    if (ge) {
        uint64_t borrow = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)x[i] - P256_N[i] - borrow;
            r[i] = (uint32_t)diff;
            borrow = (diff >> 63);
        }
    } else {
        copy256(r, x);
    }
}

// Generate deterministic nonce from hash and counter
// Simple counter-based nonce for Card 08 (NOT RFC 6979)
__device__ void generate_nonce(
    uint32_t k[8],
    const uint8_t hash[32],
    const uint8_t priv_d[32],
    uint32_t counter
) {
    // XOR hash with private key and counter for simple deterministic nonce
    // WARNING: This is NOT cryptographically secure! Replace with RFC 6979 later.
    for (int i = 0; i < 8; ++i) {
        uint32_t h_limb = ((uint32_t)hash[i*4+0]) |
                          ((uint32_t)hash[i*4+1] << 8) |
                          ((uint32_t)hash[i*4+2] << 16) |
                          ((uint32_t)hash[i*4+3] << 24);
        uint32_t d_limb = ((uint32_t)priv_d[i*4+0]) |
                          ((uint32_t)priv_d[i*4+1] << 8) |
                          ((uint32_t)priv_d[i*4+2] << 16) |
                          ((uint32_t)priv_d[i*4+3] << 24);
        k[i] = h_limb ^ d_limb ^ (counter * 0x9e3779b9u + i);  // Simple mixing
    }
    // Ensure k is in valid range [1, n-1]
    reduce_x_mod_n(k, k);
    if (is_zero256(k)) k[0] = 1;  // Prevent k=0
}

// ============================================================================
// Main Signing Function
// ============================================================================

/*
 * p256_sign_persistent - Device-callable ECDSA P-256 signing
 *
 * Computes ECDSA signature (r, s) for a given hash and private key.
 *
 * Parameters:
 *   out_r[32]: Output signature r component (big-endian bytes)
 *   out_s[32]: Output signature s component (big-endian bytes)
 *   hash[32]: Message hash (big-endian bytes)
 *   priv_d[32]: Private key (big-endian bytes)
 *   nonce_counter: Unused in Card 13 (kept for API compatibility)
 *   ext_nonce[32]: Optional override nonce (big-endian bytes, NULL = use mode-based nonce)
 *   low_s_enabled: Card 14 - If true, normalize s to low-s form (s <= n/2)
 *   use_random_nonce: Card 15 - If true, use CSPRNG for nonce instead of RFC 6979
 *   rng_seed[32]: Card 15 - CSPRNG seed (used only when use_random_nonce=true)
 *   rng_stream_id: Card 15 - Unique stream ID for CSPRNG (prevents nonce reuse)
 *
 * Returns: true if signature is valid, false if invalid (r=0 or s=0)
 *
 * Card 13: By default, nonce k is generated using device-side RFC 6979.
 * The ext_nonce parameter is kept for debug/override purposes only.
 *
 * Card 14: When low_s_enabled is true, if s > n/2, the function returns
 * s' = n - s instead. This produces BIP 62 / BIP 146 canonical signatures
 * for compatibility with wallets, FIDO/WebAuthn, and strict verifiers.
 *
 * Card 15: When use_random_nonce is true, nonce k is generated using a
 * ChaCha20-based CSPRNG with the provided seed and stream ID. This is
 * useful for research, fuzzing, and stress testing. The stream ID must
 * be unique per signing operation to prevent nonce reuse.
 */
__device__ bool p256_sign_persistent(
    uint8_t out_r[32],
    uint8_t out_s[32],
    const uint8_t hash[32],
    const uint8_t priv_d[32],
    uint32_t nonce_counter,
    const uint8_t* ext_nonce = nullptr,
    bool low_s_enabled = true,
    bool use_random_nonce = false,
    const uint8_t* rng_seed = nullptr,
    uint64_t rng_stream_id = 0
) {
    // Convert inputs from big-endian bytes to little-endian limbs
    uint32_t h[8], d[8];
    for (int i = 0; i < 8; ++i) {
        // Big-endian byte order: byte 0 is MSB
        int byte_base = (7 - i) * 4;
        h[i] = ((uint32_t)hash[byte_base+3]) |
               ((uint32_t)hash[byte_base+2] << 8) |
               ((uint32_t)hash[byte_base+1] << 16) |
               ((uint32_t)hash[byte_base+0] << 24);
        d[i] = ((uint32_t)priv_d[byte_base+3]) |
               ((uint32_t)priv_d[byte_base+2] << 8) |
               ((uint32_t)priv_d[byte_base+1] << 16) |
               ((uint32_t)priv_d[byte_base+0] << 24);
    }

    // Generate nonce k
    uint32_t k[8];
    if (ext_nonce != nullptr) {
        // Debug/override: Use externally provided nonce
        for (int i = 0; i < 8; ++i) {
            int byte_base = (7 - i) * 4;
            k[i] = ((uint32_t)ext_nonce[byte_base+3]) |
                   ((uint32_t)ext_nonce[byte_base+2] << 8) |
                   ((uint32_t)ext_nonce[byte_base+1] << 16) |
                   ((uint32_t)ext_nonce[byte_base+0] << 24);
        }
    } else if (use_random_nonce && rng_seed != nullptr) {
        // Card 15: Use ChaCha20-based CSPRNG for random nonce
        P256RngState rng;
        p256_rng_init(rng, rng_seed, rng_stream_id);

        if (!p256_generate_random_k(k, rng)) {
            // CSPRNG failed to generate valid k (astronomically unlikely)
            for (int i = 0; i < 32; ++i) { out_r[i] = 0; out_s[i] = 0; }
            return false;
        }
    } else {
        // Card 13: Use device-side RFC 6979 deterministic nonce (default)
        if (!rfc6979_generate_k_limbs_device(k, priv_d, hash)) {
            // RFC 6979 failed (should never happen with valid inputs)
            for (int i = 0; i < 32; ++i) { out_r[i] = 0; out_s[i] = 0; }
            return false;
        }
    }

    // Compute R = k*G
    uint32_t Rx[8], Ry[8], Rz[8];
    scalar_mul_g(k, Rx, Ry, Rz);

    // Convert R to affine
    uint32_t x_affine[8], y_affine[8];
    if (!jacobian_to_affine(Rx, Ry, Rz, x_affine, y_affine)) {
        // Point at infinity (should not happen with valid k)
        for (int i = 0; i < 32; ++i) { out_r[i] = 0; out_s[i] = 0; }
        return false;
    }

    // Convert x from Montgomery to standard domain
    uint32_t x_std[8];
    from_mont_p(x_std, x_affine);

    // r = x mod n
    uint32_t r[8];
    reduce_x_mod_n(r, x_std);

    if (is_zero256(r)) {
        for (int i = 0; i < 32; ++i) { out_r[i] = 0; out_s[i] = 0; }
        return false;
    }

    // Convert k, h, d, r to Montgomery domain mod n
    uint32_t k_mont[8], h_mont[8], d_mont[8], r_mont[8];
    to_mont_n(k_mont, k);
    to_mont_n(h_mont, h);
    to_mont_n(d_mont, d);
    to_mont_n(r_mont, r);

    // k_inv = k^(-1) mod n
    uint32_t k_inv[8];
    fermat_inv_n(k_inv, k_mont);

    // rd = r * d mod n
    uint32_t rd[8];
    mont_mul_n(rd, r_mont, d_mont);

    // z = h + r*d mod n
    uint32_t z[8];
    mod_add_n(z, h_mont, rd);

    // s = k_inv * z mod n
    uint32_t s_mont[8];
    mont_mul_n(s_mont, k_inv, z);

    // Convert s from Montgomery to standard domain
    uint32_t s[8];
    from_mont_n(s, s_mont);

    if (is_zero256(s)) {
        for (int i = 0; i < 32; ++i) { out_r[i] = 0; out_s[i] = 0; }
        return false;
    }

    // Card 14: Apply low-s normalization if enabled
    // If s > n/2, replace s with n - s for canonical signatures
    if (low_s_enabled && is_greater_than_half_n(s)) {
        uint32_t s_low[8];
        mod_sub_from_n(s_low, s);
        copy256(s, s_low);
    }

    // Convert outputs to big-endian bytes
    for (int i = 0; i < 8; ++i) {
        int byte_base = (7 - i) * 4;
        out_r[byte_base+3] = (uint8_t)(r[i]);
        out_r[byte_base+2] = (uint8_t)(r[i] >> 8);
        out_r[byte_base+1] = (uint8_t)(r[i] >> 16);
        out_r[byte_base+0] = (uint8_t)(r[i] >> 24);
        out_s[byte_base+3] = (uint8_t)(s[i]);
        out_s[byte_base+2] = (uint8_t)(s[i] >> 8);
        out_s[byte_base+1] = (uint8_t)(s[i] >> 16);
        out_s[byte_base+0] = (uint8_t)(s[i] >> 24);
    }

    return true;
}

} // namespace p256
} // namespace smoke

#endif // P256_PERSISTENT_CUH
