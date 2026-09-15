/*
 * Card 26: P-256 Warp-Cooperative Persistent Signer
 *
 * This header provides warp-cooperative ECDSA signing for the persistent engine.
 * It uses the same comb table + warp-coop Montgomery approach as the fast PyTorch
 * kernel but is device-callable for use in persistent loops.
 *
 * Architecture:
 *   - R = k*G uses warp-coop scalar multiplication with comb table (fast)
 *   - Signature computation (r, s) uses lane-0 scalar order operations (proven correct)
 *   - Combined: ~4-10x speedup over thread-only implementation per signature
 *
 * Trade-off:
 *   - Thread-only: 32 concurrent signatures, ~8.6ms each = ~3,700 sig/sec
 *   - Warp-coop: 1 signature at a time, but ~0.5-2ms each = ~500-2000 sig/sec per warp
 *   - Warp-coop wins when combined with multi-CTA (many warps)
 *
 * Build flag:
 *   -DP256_WARP_COOP_ENABLED=1: Enable warp-coop signer
 *   -DP256_WARP_COOP_ENABLED=0: Use thread-only fallback (default)
 *
 * Requires:
 *   - Comb table loaded to constant memory before kernel launch
 *   - Shared memory: 32 bytes per warp (8 limbs × 4 bytes)
 */

#ifndef P256_SIGN_WARP_COOP_CUH
#define P256_SIGN_WARP_COOP_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Card 26.2 DEBUG: Debug prints in warp-coop functions
// Must be defined BEFORE including p256_field_warp_coop.cuh
// Set to 1 for verbose debug output, 0 for production
#define P256_WARP_COOP_DEBUG 0

// ============================================================================
// Card 26: Wiring Flag
// ============================================================================
#ifndef P256_WARP_COOP_ENABLED
#define P256_WARP_COOP_ENABLED 0  // Default: disabled until multi-CTA is ready
#endif

#if P256_WARP_COOP_ENABLED

// ============================================================================
// Include existing warp-coop infrastructure from ecdsa_p256
// ============================================================================

// IMPORTANT: These includes bring in the validated warp-coop implementations
// from the PyTorch-based ECDSA path that achieves 80k-140k sig/sec

// Guard against duplicate constant declarations
#ifndef P256_WARP_COOP_CONSTANTS_INCLUDED
#define P256_WARP_COOP_CONSTANTS_INCLUDED

// Warp-coop field operations (Montgomery multiply, etc.)
// Note: This header defines __constant__ arrays that may conflict with
// p256_persistent.cuh. Include order matters.
#include "../ecdsa_p256/p256_field_warp_coop.cuh"

// KERNEL-P256-JACOBIAN-01: Use true warp-parallel Jacobian ops by default.
// Define P256_USE_ORACLE_JACOBIAN to fall back to lane-0 scalar oracle path.
#ifdef P256_USE_ORACLE_JACOBIAN
#include "../ecdsa_p256/jacobian_double_oracle.cuh"
#endif

// Order-n arithmetic for ECDSA signature computation
#include "../ecdsa_p256/p256_scalar_order.cuh"

// RFC 6979 nonce generation
#include "p256_rfc6979_device.cuh"

#endif // P256_WARP_COOP_CONSTANTS_INCLUDED

namespace smoke {
namespace p256 {
namespace warp_coop {

// ============================================================================
// Comb table reference
// ============================================================================
// Table symbols are defined in p256_field_warp_coop.cuh (included above)
// Data is loaded via load_p256_comb_table() before kernel launch
//
// Card 26 FIX: Removed redundant extern declarations that conflicted with
// the actual definitions in p256_field_warp_coop.cuh

// Table configuration (w=8 => 255 entries)
constexpr int COMB_TABLE_ENTRIES = 255;  // 2^8 - 1

// ============================================================================
// SoA table lookup (reused from p256_sign_persistent.cu)
// ============================================================================

__device__ __forceinline__
void soa_lookup_bucket_warp(int entry, uint32_t Px[8], uint32_t Py[8], int lane_id) {
    // AOS layout: all 8 limbs of one point are contiguous [(entry-1)*8 + limb]
    // One constant cache line read per group instead of 8 serialized reads.
    // For w=8: 255 entries (indices 0-254 map to multipliers 1-255)
    // Phase 3b: All 32 lanes do lookups; each 8-lane group gets the same entry
    if (entry > 0 && entry <= COMB_TABLE_ENTRIES) {
        int local_lane = lane_id & 7;
        int offset = (entry - 1) * 8 + local_lane;
        Px[local_lane] = P256_G_TABLE_X_SOA[offset];
        Py[local_lane] = P256_G_TABLE_Y_SOA[offset];
    }
}

// ============================================================================
// Signed window recoding for comb method
// ============================================================================

template<int W, int MAX_L>
__device__ __forceinline__
int recode_signed_fixed_window_warp(const uint32_t k[8], int16_t digits[MAX_L]) {
    int L = (256 + W - 1) / W;
    uint32_t carry = 0;

    for (int i = 0; i < L; ++i) {
        int bit_pos = i * W;
        int limb_idx = bit_pos / 32;
        int bit_offset = bit_pos % 32;

        uint32_t window;
        if (bit_offset + W <= 32) {
            window = (k[limb_idx] >> bit_offset) & ((1u << W) - 1);
        } else {
            uint32_t lo = k[limb_idx] >> bit_offset;
            uint32_t hi = (limb_idx + 1 < 8) ? k[limb_idx + 1] : 0;
            window = (lo | (hi << (32 - bit_offset))) & ((1u << W) - 1);
        }

        window += carry;
        carry = 0;

        if (window >= (1u << (W - 1))) {
            window = (1u << W) - window;
            carry = 1;
            digits[i] = -(int16_t)window;
        } else {
            digits[i] = (int16_t)window;
        }
    }

    // Handle final carry overflow (possible when W evenly divides 256, e.g. w=8)
    if (carry && L < MAX_L) {
        digits[L] = 1;
        L++;
    }

    return L;
}

// ============================================================================
// Check if Z coordinate is zero
// ============================================================================

__device__ __forceinline__
bool warp_is_zero_z(const uint32_t Z[8], int lane_id) {
    uint32_t my_z = (lane_id < 8) ? Z[lane_id] : 0;
    uint32_t all_z = my_z;
    for (int offset = 4; offset > 0; offset /= 2) {
        all_z |= __shfl_xor_sync(0xFF, all_z, offset);
    }
    all_z = __shfl_sync(0xFFFFFFFF, all_z, 0);
    return (all_z == 0);
}

// ============================================================================
// Warp-cooperative scalar multiplication: R = k * G
// ============================================================================

__device__ void scalar_mul_g_warp_coop(
    const uint32_t k[8],
    uint32_t Rx[8], uint32_t Ry[8], uint32_t Rz[8],
    int lane_id
) {
    // Card 26.5: Count scalar multiplication calls
    CARD26_SCALAR_MUL_INCREMENT();

    constexpr int W = 8;
    constexpr int MAX_L = (256 + W - 1) / W + 1;  // +1 for carry overflow when W divides 256

    // Phase 3b: Each group leader (lanes 0,8,16,24) recodes its own scalar
    const int local_lane = lane_id & 7;
    const bool is_group_leader = (local_lane == 0);

    int16_t digits[MAX_L];
    int L = 0;
    if (is_group_leader) {
        L = recode_signed_fixed_window_warp<W, MAX_L>(k, digits);
    }
    L = __shfl_sync(0xFFFFFFFF, L, 0, 8);  // width=8: broadcast within group

    // Initialize accumulator to infinity
    for (int i = 0; i < 8; ++i) {
        Rx[i] = 0;
        Ry[i] = 0;
        Rz[i] = 0;
    }
    bool initialized = false;

    // Process windows from high to low
    for (int j = L - 1; j >= 0; --j) {
        // Each group leader broadcasts its digit within its group
        int d_j = 0;
        if (is_group_leader) {
            d_j = (int)digits[j];
        }
        d_j = __shfl_sync(0xFFFFFFFF, d_j, 0, 8);  // width=8

        // W doublings (only after initialization)
        if (initialized) {
            for (int dd = 0; dd < W; ++dd) {
#ifdef P256_USE_ORACLE_JACOBIAN
                jacobian_double_oracle_compat(Rx, Ry, Rz, lane_id);
#else
                jacobian_double_warp(Rx, Ry, Rz, lane_id);
#endif
            }
        }

        // Conditional add — groups may diverge here (d_j differs per group).
        // Width=8 shuffles inside Jacobian ops are group-local, so no deadlock.
        if (d_j != 0) {
            int entry = (d_j > 0) ? d_j : -d_j;
            uint32_t Px[8], Py[8];
            soa_lookup_bucket_warp(entry, Px, Py, lane_id);

            if (d_j < 0) {
                uint32_t neg = warp_negate_u32(Py[local_lane], lane_id);
                Py[local_lane] = neg;
            }

            if (!initialized) {
                Rx[local_lane] = Px[local_lane];
                Ry[local_lane] = Py[local_lane];
                Rz[local_lane] = P256_MONT_ONE[local_lane];
                initialized = true;
            } else {
#ifdef P256_USE_ORACLE_JACOBIAN
                jacobian_add_affine_oracle_compat(Rx, Ry, Rz, Px, Py, lane_id);
#else
                jacobian_add_affine_warp(Rx, Ry, Rz, Px, Py, lane_id);
#endif
            }
        }
        // No __syncwarp needed — width=8 shuffles are group-local
    }
}

#if P256_FASTPATH_FULLWINDOW
// ============================================================================
// SPEED-H2-01: zero-doubling fixed-base scalar multiplication R = k*G.
//
// Mechanical transform of scalar_mul_g_warp_coop above: SAME signed-window
// recode, SAME add/negate primitives, SAME per-lane SoA operand layout — but
// the doubling loop is REMOVED and the 1D comb lookup (entry = |digit|*G) is
// replaced by the 2D full-window lookup (entry = |digit| * 2^(8*window) * G).
// Because each window's table entry is already pre-scaled by 2^(8*window),
// summing the per-window contributions reconstructs k*G with no doublings:
//     k*G = sum_j digit_j * 2^(8j) * G
// Window 32 (the recode's carry digit, weight 2^256) is covered by the 33rd
// table window — see the table generator / P256_FULLWINDOW_WINDOWS=33.
//
// Correctness is NOT asserted by this comment: the acceptance gate is NIST
// test vectors + the table-free double-and-add oracle at 4096/4096 on GPU
// (Phase C, operator-run on the 3060). Ships behind P256_FASTPATH_FULLWINDOW,
// default OFF, with the fail-closed load gate — it cannot silently promote.
// ============================================================================
__device__ void scalar_mul_g_fullwindow_warp(
    const uint32_t k[8],
    uint32_t Rx[8], uint32_t Ry[8], uint32_t Rz[8],
    int lane_id
) {
    CARD26_SCALAR_MUL_INCREMENT();

    constexpr int W = 8;
    constexpr int MAX_L = (256 + W - 1) / W + 1;  // 33: includes the carry window
    const int local_lane = lane_id & 7;
    const bool is_group_leader = (local_lane == 0);

    int16_t digits[MAX_L];
    int L = 0;
    if (is_group_leader) {
        L = recode_signed_fixed_window_warp<W, MAX_L>(k, digits);
    }
    L = __shfl_sync(0xFFFFFFFF, L, 0, 8);

    // Accumulator = point at infinity.
    for (int i = 0; i < 8; ++i) { Rx[i] = 0; Ry[i] = 0; Rz[i] = 0; }
    bool initialized = false;

    // Window order is irrelevant without doublings; iterate low->high. Each
    // window contributes digit_j * 2^(8j) * G directly from the table.
    for (int j = 0; j < L; ++j) {
        int d_j = 0;
        if (is_group_leader) d_j = (int)digits[j];
        d_j = __shfl_sync(0xFFFFFFFF, d_j, 0, 8);

        if (d_j != 0) {
            int entry = (d_j > 0) ? d_j : -d_j;         // |digit| in [1,128]
            uint32_t Px[8], Py[8];
            fullwindow_lookup_warp(j, entry, Px, Py, lane_id);

            if (d_j < 0) {
                uint32_t neg = warp_negate_u32(Py[local_lane], lane_id);
                Py[local_lane] = neg;
            }

            if (!initialized) {
                Rx[local_lane] = Px[local_lane];
                Ry[local_lane] = Py[local_lane];
                Rz[local_lane] = P256_MONT_ONE[local_lane];
                initialized = true;
            } else {
                // Same mixed-add the comb path uses (oracle-compat, group-local
                // width-8 shuffles): accumulate the pre-scaled window point.
                jacobian_add_affine_oracle_compat(Rx, Ry, Rz, Px, Py, lane_id);
            }
        }
    }
    // If k recoded to all-zero digits (k==0), R stays at infinity — the
    // caller's existing r==0 / phase checks fail closed, same as the comb path.
}
#endif // P256_FASTPATH_FULLWINDOW

// ============================================================================
// Convert Jacobian to affine (warp-cooperative)
// ============================================================================

__device__ void jacobian_to_affine_warp(
    const uint32_t X[8], const uint32_t Y[8], const uint32_t Z[8],
    uint32_t x_out[8], uint32_t y_out[8],
    int lane_id
) {
    // Card 26.5: Count Jacobian-to-affine conversions
    CARD26_J2A_INCREMENT();

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[j2a] Entry\n");
#endif

    // Card 26 FIX: Use working warp-cooperative Fermat inversion
    // Zinv = Z^(-1) = Z^(p-2) mod p in Montgomery domain
    uint32_t Zinv[8];

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[j2a] Calling warp_fermat_inverse_p256...\n");
#endif
    warp_fermat_inverse_p256(Z, Zinv, lane_id);

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[j2a] Fermat inverse done!\n");
    __syncwarp(0xFFFFFFFF);
    if (lane_id == 0) printf("[j2a] About to compute Zinv^2...\n");
#endif

    __syncwarp(0xFFFFFFFF);

    // Zinv^2 = Zinv * Zinv
    // Card 26.2 FIX: ALL 32 lanes must call warp_mont_mul_oracle_compat to avoid deadlock
    // (it has internal __syncwarp calls)
    uint32_t Zinv2[8];
    uint32_t zinv2_result = warp_mont_mul_oracle_compat(Zinv, Zinv, lane_id);
    if (lane_id < 8) {
        Zinv2[lane_id] = zinv2_result;
    }
    __syncwarp(0xFFFFFFFF);
#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[j2a] Zinv^2 done\n");
#endif

    // Zinv^3 = Zinv^2 * Zinv
    // Card 26.2 FIX: ALL 32 lanes must call warp_mont_mul_oracle_compat
    uint32_t Zinv3[8];
    uint32_t zinv3_result = warp_mont_mul_oracle_compat(Zinv2, Zinv, lane_id);
    if (lane_id < 8) {
        Zinv3[lane_id] = zinv3_result;
    }
    __syncwarp(0xFFFFFFFF);

    // x_mont = X * Zinv^2 (still in Montgomery domain)
    // Card 26.2 FIX: ALL 32 lanes must call warp_mont_mul_oracle_compat
    uint32_t x_mont[8];
    uint32_t xmont_result = warp_mont_mul_oracle_compat(X, Zinv2, lane_id);
    if (lane_id < 8) {
        x_mont[lane_id] = xmont_result;
    }
    __syncwarp(0xFFFFFFFF);

    // y_mont = Y * Zinv^3 (still in Montgomery domain)
    // Card 26.2 FIX: ALL 32 lanes must call warp_mont_mul_oracle_compat
    uint32_t y_mont[8];
    uint32_t ymont_result = warp_mont_mul_oracle_compat(Y, Zinv3, lane_id);
    if (lane_id < 8) {
        y_mont[lane_id] = ymont_result;
    }
    __syncwarp(0xFFFFFFFF);

    // De-Montgomery to normal domain: x = x_mont * 1
    // Card 26.2 FIX: ALL 32 lanes must call warp_mont_mul_oracle_compat
    uint32_t x_demont = warp_mont_mul_oracle_compat(x_mont, P256_ONE_LITERAL, lane_id);
    if (lane_id < 8) {
        x_out[lane_id] = x_demont;
    }
    __syncwarp(0xFFFFFFFF);

    // De-Montgomery to normal domain: y = y_mont * 1
    // Card 26.2 FIX: ALL 32 lanes must call warp_mont_mul_oracle_compat
    uint32_t y_demont = warp_mont_mul_oracle_compat(y_mont, P256_ONE_LITERAL, lane_id);
    if (lane_id < 8) {
        y_out[lane_id] = y_demont;
    }
    __syncwarp(0xFFFFFFFF);

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[j2a] EXIT SUCCESS\n");
#endif
}

// ============================================================================
// Card 26.6: Convert Jacobian to affine using PRECOMPUTED Zinv
// ============================================================================
// This version skips the expensive Fermat inversion since Zinv was computed
// via Card14 batch inversion. Used in the batched signing path.
//
// CRITICAL: Zinv must be precomputed in Montgomery form by batch inversion!
// ===========================================================================

__device__ void jacobian_to_affine_with_zinv_warp(
    const uint32_t X[8], const uint32_t Y[8],
    const uint32_t Zinv[8],  // Precomputed Z^(-1) in Montgomery form
    uint32_t x_out[8], uint32_t y_out[8],
    int lane_id
) {
    // Card 26.5: Count J2A calls (even with precomputed Zinv)
    CARD26_J2A_INCREMENT();

    // Phase 3c: Use group-local oracle multiply (width=8, no full-warp sync)
    const int ll = lane_id & 7;

    // Zinv^2 = Zinv * Zinv
    uint32_t Zinv2[8];
    uint32_t zinv2_result = warp_mont_mul_oracle_compat_group(Zinv, Zinv, lane_id);
    Zinv2[ll] = zinv2_result;

    // Zinv^3 = Zinv^2 * Zinv
    uint32_t Zinv3[8];
    uint32_t zinv3_result = warp_mont_mul_oracle_compat_group(Zinv2, Zinv, lane_id);
    Zinv3[ll] = zinv3_result;

    // x_mont = X * Zinv^2
    uint32_t x_mont[8];
    uint32_t xmont_result = warp_mont_mul_oracle_compat_group(X, Zinv2, lane_id);
    x_mont[ll] = xmont_result;

    // y_mont = Y * Zinv^3
    uint32_t y_mont[8];
    uint32_t ymont_result = warp_mont_mul_oracle_compat_group(Y, Zinv3, lane_id);
    y_mont[ll] = ymont_result;

    // De-Montgomery: x = x_mont * 1
    uint32_t x_demont = warp_mont_mul_oracle_compat_group(x_mont, P256_ONE_LITERAL, lane_id);
    x_out[ll] = x_demont;

    // De-Montgomery: y = y_mont * 1
    uint32_t y_demont = warp_mont_mul_oracle_compat_group(y_mont, P256_ONE_LITERAL, lane_id);
    y_out[ll] = y_demont;
}

// ============================================================================
// ECDSA signature from point (lane-0 only, reuses scalar_order ops)
// ============================================================================

__device__ bool ecdsa_sign_from_r_warp(
    uint32_t sig_r[8], uint32_t sig_s[8],
    const uint32_t x_R[8],  // x-coordinate of R = k*G
    const uint32_t k[8],    // nonce
    const uint32_t h[8],    // hash
    const uint32_t d[8],    // private key
    int lane_id
) {
    // Phase 3b: Gather inputs to each group leader (width=8)
    const int local_lane = lane_id & 7;
    const bool is_group_leader = (local_lane == 0);

    uint32_t x_full[8], k_full[8], h_full[8], d_full[8];
    for (int i = 0; i < 8; ++i) {
        x_full[i] = __shfl_sync(0xFFFFFFFF, x_R[local_lane], i, 8);
        k_full[i] = __shfl_sync(0xFFFFFFFF, k[i], 0, 8);
        h_full[i] = __shfl_sync(0xFFFFFFFF, h[i], 0, 8);
        d_full[i] = __shfl_sync(0xFFFFFFFF, d[i], 0, 8);
    }

    // N5 x_full_zero (2026-05-16): record whether the gathered x_full
    // (affine x-coordinate of R = k*G) is all-zero BEFORE the mod-n
    // reduction. Splits r_zero into (x_full == 0: upstream affine/Zinv
    // produced zero) vs (x_full != 0 yet r == 0: gather/reduction
    // pathology). Lane 0 of warp only — once per op.
    if (lane_id == 0 && g_telemetry_ptr != nullptr) {
        bool x_full_is_zero = true;
        for (int j = 0; j < 8; ++j) {
            if (x_full[j] != 0u) { x_full_is_zero = false; break; }
        }
        if (x_full_is_zero) {
            atomicAdd((unsigned long long*)&g_telemetry_ptr->dbg_p256_x_full_zero,
                      (unsigned long long)1);
        }
    }

    uint32_t r[8] = {0}, s[8] = {0};
    bool valid = false;

    if (is_group_leader) {
        // Use the proven lane-0 ECDSA signature computation
        // r = x mod n
        reduce_x_mod_n(r, x_full);

        if (!is_zero_scalar(r)) {
            // Montgomery domain computations
            uint32_t k_mont[8], h_mont[8], d_mont[8], r_mont[8];
            to_mont_n(k_mont, k_full);
            to_mont_n(h_mont, h_full);
            to_mont_n(d_mont, d_full);
            to_mont_n(r_mont, r);

            // k_inv = k^(-1) mod n
            uint32_t k_inv[8];
            mont_fermat_inv_n(k_inv, k_mont);

            // rd = r * d mod n
            uint32_t rd[8];
            mont_mul_scalar_n(rd, r_mont, d_mont);

            // z = h + rd mod n
            uint32_t z[8];
            mont_add_scalar_n(z, h_mont, rd);

            // s = k_inv * z mod n
            uint32_t s_mont[8];
            mont_mul_scalar_n(s_mont, k_inv, z);

            // Back to normal domain
            from_mont_n(s, s_mont);

            valid = !is_zero_scalar(s);
        } else {
            // N5 r_zero counter (2026-05-15): lane 0 of warp increments
            // once per op when reduce_x_mod_n(r, x_full) yielded r == 0.
            // Splits phase3_fail into r==0 (upstream affine/Zinv path)
            // vs s==0 (Montgomery chain bug).
            if (lane_id == 0 && g_telemetry_ptr != nullptr) {
                atomicAdd((unsigned long long*)&g_telemetry_ptr->dbg_p256_r_zero,
                          (unsigned long long)1);
            }
        }
    }
    // Broadcast results within each group (width=8)
    valid = __shfl_sync(0xFFFFFFFF, valid ? 1 : 0, 0, 8) != 0;
    for (int i = 0; i < 8; ++i) {
        sig_r[i] = __shfl_sync(0xFFFFFFFF, r[i], 0, 8);
        sig_s[i] = __shfl_sync(0xFFFFFFFF, s[i], 0, 8);
    }

    return valid;
}

// ============================================================================
// Main warp-cooperative signing function
// ============================================================================

/*
 * p256_sign_warp_coop - Warp-cooperative ECDSA P-256 signing
 *
 * All 32 warp threads must call this function together.
 * Uses comb table + warp-cooperative Montgomery for scalar multiplication.
 * Uses lane-0 scalar operations for ECDSA signature (proven correct).
 *
 * Parameters:
 *   out_r[32]: Output signature r component (big-endian bytes)
 *   out_s[32]: Output signature s component (big-endian bytes)
 *   hash[32]: Message hash (big-endian bytes)
 *   priv_d[32]: Private key (big-endian bytes)
 *   lane_id: Thread's lane within warp (0-31)
 *   low_s_enabled: If true, normalize s to low-s form (not implemented yet)
 *
 * Returns: true if signature is valid (all lanes return same value)
 */
// Card 26.2: Debug flag for warp-coop internal beacons
// Set to 1 to enable printf debugging inside the signer
#ifndef P256_WARP_COOP_DEBUG
#define P256_WARP_COOP_DEBUG 0
#endif

__device__ bool p256_sign_warp_coop(
    uint8_t out_r[32],
    uint8_t out_s[32],
    const uint8_t hash[32],
    const uint8_t priv_d[32],
    int lane_id,
    bool low_s_enabled = true
) {
#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Entry, lane_id=%d\n", lane_id);
    }
    __syncwarp(0xFFFFFFFF);
#endif

    // ========================================================================
    // Step 1: Convert inputs from big-endian to little-endian limbs
    // ========================================================================
    uint32_t h[8], d[8];
    if (lane_id < 8) {
        int byte_base = (7 - lane_id) * 4;
        h[lane_id] = ((uint32_t)hash[byte_base+3]) |
                     ((uint32_t)hash[byte_base+2] << 8) |
                     ((uint32_t)hash[byte_base+1] << 16) |
                     ((uint32_t)hash[byte_base+0] << 24);
        d[lane_id] = ((uint32_t)priv_d[byte_base+3]) |
                     ((uint32_t)priv_d[byte_base+2] << 8) |
                     ((uint32_t)priv_d[byte_base+1] << 16) |
                     ((uint32_t)priv_d[byte_base+0] << 24);
    }
    __syncwarp(0xFFFFFFFF);  // Card 26.2 FIX: Use full mask for all threads

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Step 1 done (input conversion)\n");
    }
    __syncwarp(0xFFFFFFFF);
#endif

    // Gather to all lanes
    uint32_t h_full[8], d_full[8];
    for (int i = 0; i < 8; ++i) {
        h_full[i] = __shfl_sync(0xFFFFFFFF, h[lane_id < 8 ? lane_id : 0], i);
        d_full[i] = __shfl_sync(0xFFFFFFFF, d[lane_id < 8 ? lane_id : 0], i);
    }

    // ========================================================================
    // Step 2: Generate nonce k using RFC 6979 (lane 0)
    // ========================================================================
#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Step 2 starting (RFC 6979 nonce)\n");
    }
    __syncwarp(0xFFFFFFFF);
#endif

    uint32_t k[8];
    bool k_valid = false;
    if (lane_id == 0) {
        k_valid = rfc6979_generate_k_limbs_device(k, priv_d, hash);
    }
    __syncwarp(0xFFFFFFFF);

    k_valid = __shfl_sync(0xFFFFFFFF, k_valid ? 1 : 0, 0) != 0;
    if (!k_valid) {
        return false;
    }

    // Broadcast k
    for (int i = 0; i < 8; ++i) {
        k[i] = __shfl_sync(0xFFFFFFFF, k[i], 0);
    }

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Step 2 done (nonce generated)\n");
    }
    __syncwarp(0xFFFFFFFF);
#endif

    // ========================================================================
    // Step 3: Compute R = k*G (warp-cooperative)
    // ========================================================================
#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Step 3 starting (k*G scalar mul)\n");
    }
    __syncwarp(0xFFFFFFFF);
#endif

    uint32_t Rx[8], Ry[8], Rz[8];
#if P256_FASTPATH_FULLWINDOW
    scalar_mul_g_fullwindow_warp(k, Rx, Ry, Rz, lane_id);  // SPEED-H2-01 (default OFF)
#else
    scalar_mul_g_warp_coop(k, Rx, Ry, Rz, lane_id);
#endif

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Step 3 done (k*G complete)\n");
    }
    // Card 26.2: Check warp convergence before sync
    unsigned mask = __activemask();
    if (lane_id == 0) {
        printf("[warp_coop] Active mask before sync: 0x%08x\n", mask);
    }
    __syncwarp(0xFFFFFFFF);
#endif

    // ========================================================================
    // Step 4: Convert to affine (warp-cooperative)
    // ========================================================================
#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Step 4 starting (Jacobian to affine)\n");
    }
    __syncwarp(0xFFFFFFFF);
    if (lane_id == 0) {
        printf("[warp_coop] Calling jacobian_to_affine_warp...\n");
    }
    __syncwarp(0xFFFFFFFF);
    // Card 26.2: Print array values to confirm they're valid
    if (lane_id == 0) {
        printf("[warp_coop] Rz[0]=0x%08x Rz[7]=0x%08x (about to call j2a)\n", Rz[0], Rz[7]);
    }
#endif

    uint32_t x_affine[8], y_affine[8];
#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[warp_coop] >>> CALLING j2a <<<\n");
#endif
    jacobian_to_affine_warp(Rx, Ry, Rz, x_affine, y_affine, lane_id);

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Step 4 done (affine conversion)\n");
    }
    __syncwarp(0xFFFFFFFF);
#endif

    // ========================================================================
    // Step 5: Compute signature (r, s) (lane-0 with broadcast)
    // ========================================================================
#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Step 5 starting (ECDSA signature)\n");
    }
    __syncwarp(0xFFFFFFFF);
#endif

    uint32_t sig_r[8], sig_s[8];
    bool valid = ecdsa_sign_from_r_warp(sig_r, sig_s, x_affine, k, h_full, d_full, lane_id);

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) {
        printf("[warp_coop] Step 5 done (signature computed, valid=%d)\n", valid ? 1 : 0);
    }
    __syncwarp(0xFFFFFFFF);
#endif

    if (!valid) {
        return false;
    }

    // ========================================================================
    // Step 6: Convert outputs to big-endian bytes
    // FIX: Only lane 0 writes output because caller may have per-thread buffers.
    // sig_r/sig_s are already broadcast to all lanes by ecdsa_sign_from_r_warp.
    // ========================================================================
    if (lane_id == 0) {
        for (int i = 0; i < 8; ++i) {
            int byte_base = (7 - i) * 4;
            out_r[byte_base+3] = (uint8_t)(sig_r[i]);
            out_r[byte_base+2] = (uint8_t)(sig_r[i] >> 8);
            out_r[byte_base+1] = (uint8_t)(sig_r[i] >> 16);
            out_r[byte_base+0] = (uint8_t)(sig_r[i] >> 24);
            out_s[byte_base+3] = (uint8_t)(sig_s[i]);
            out_s[byte_base+2] = (uint8_t)(sig_s[i] >> 8);
            out_s[byte_base+1] = (uint8_t)(sig_s[i] >> 16);
            out_s[byte_base+0] = (uint8_t)(sig_s[i] >> 24);
        }
    }

    return true;
}

// ============================================================================
// Card 26.6: Phase-based signing for batched Z-inversion
// ============================================================================
// These functions split p256_sign_warp_coop into phases so batch inversion
// can be applied to multiple signatures at once:
//   Phase 1: Input conversion, nonce generation, k*G → outputs Jacobian point
//   Phase 2: Batch inversion (called externally on all Z coords)
//   Phase 3: Apply Zinv, convert to affine, compute signature
//
// Usage in engine loop:
//   for each request:
//     phase1() → get Jacobian R, nonce k, hash h, key d
//     store Z[i] for batch inv
//   batch_inv(Z[], Zinv[])  // 1 Fermat inv for all!
//   for each request:
//     phase3(R, Zinv[i], k, h, d) → get signature
// ============================================================================

// Intermediate state for batched signing (per-signature state between phases)
struct WarpCoopSignState {
    uint32_t Rx[8], Ry[8], Rz[8];  // Jacobian point R = k*G
    uint32_t k[8];                  // Nonce
    uint32_t h[8];                  // Hash (little-endian limbs)
    uint32_t d[8];                  // Private key (little-endian limbs)
    bool valid;                     // Nonce generation success
};

/*
 * Phase 1: Generate nonce and compute R = k*G (Jacobian)
 *
 * Outputs:
 *   state.Rx, Ry, Rz: Jacobian point R = k*G
 *   state.k: Nonce (little-endian limbs)
 *   state.h: Hash (little-endian limbs)
 *   state.d: Private key (little-endian limbs)
 *   state.valid: true if nonce generation succeeded
 */
__device__ void p256_sign_warp_coop_phase1(
    const uint8_t hash[32],
    const uint8_t priv_d[32],
    WarpCoopSignState& state,
    int lane_id
) {
    // Phase 3b: Each group converts its own input and generates its own nonce
    const int local_lane = lane_id & 7;
    const bool is_group_leader = (local_lane == 0);

    // Step 1: Convert inputs from big-endian to little-endian limbs
    uint32_t h[8], d[8];
    {
        int byte_base = (7 - local_lane) * 4;
        h[local_lane] = ((uint32_t)hash[byte_base+3]) |
                     ((uint32_t)hash[byte_base+2] << 8) |
                     ((uint32_t)hash[byte_base+1] << 16) |
                     ((uint32_t)hash[byte_base+0] << 24);
        d[local_lane] = ((uint32_t)priv_d[byte_base+3]) |
                     ((uint32_t)priv_d[byte_base+2] << 8) |
                     ((uint32_t)priv_d[byte_base+1] << 16) |
                     ((uint32_t)priv_d[byte_base+0] << 24);
    }

    // Gather within each group (width=8)
    for (int i = 0; i < 8; ++i) {
        state.h[i] = __shfl_sync(0xFFFFFFFF, h[local_lane], i, 8);
        state.d[i] = __shfl_sync(0xFFFFFFFF, d[local_lane], i, 8);
    }

    // Step 2: Generate nonce k — SIMT multi-leader (rfc6979 is pure scalar)
    uint32_t k[8];
    bool k_valid = false;
    if (is_group_leader) {
        k_valid = rfc6979_generate_k_limbs_device(k, priv_d, hash);
    }

    k_valid = __shfl_sync(0xFFFFFFFF, k_valid ? 1 : 0, 0, 8) != 0;
    state.valid = k_valid;

    if (!k_valid) {
        return;
    }

    // Broadcast k within group
    for (int i = 0; i < 8; ++i) {
        state.k[i] = __shfl_sync(0xFFFFFFFF, k[i], 0, 8);
    }

    // Step 3: Compute R = k*G (warp-cooperative, each group runs independently)
#if P256_FASTPATH_FULLWINDOW
    scalar_mul_g_fullwindow_warp(state.k, state.Rx, state.Ry, state.Rz, lane_id);  // SPEED-H2-01 (default OFF)
#else
    scalar_mul_g_warp_coop(state.k, state.Rx, state.Ry, state.Rz, lane_id);
#endif
}

/*
 * Phase 3: Apply precomputed Zinv, convert to affine, compute signature
 *
 * Inputs:
 *   state: Jacobian point R, nonce k, hash h, key d from phase1
 *   Zinv: Precomputed Z^(-1) in Montgomery form from batch inversion
 *
 * Outputs:
 *   out_r, out_s: Signature (big-endian bytes)
 *
 * Returns: true if signature is valid
 */
__device__ bool p256_sign_warp_coop_phase3(
    const WarpCoopSignState& state,
    const uint32_t Zinv[8],  // Precomputed Z^(-1) from batch inversion
    uint8_t out_r[32],
    uint8_t out_s[32],
    int lane_id,
    bool low_s_enabled = true
) {
    if (!state.valid) {
        return false;
    }

    // Step 4: Convert to affine using PRECOMPUTED Zinv
    uint32_t x_affine[8], y_affine[8];
    jacobian_to_affine_with_zinv_warp(state.Rx, state.Ry, Zinv, x_affine, y_affine, lane_id);

    // Step 5: Compute signature (r, s)
    uint32_t sig_r[8], sig_s[8];
    bool valid = ecdsa_sign_from_r_warp(sig_r, sig_s, x_affine, state.k, state.h, state.d, lane_id);

    if (!valid) {
        return false;
    }

    // Step 6: Convert outputs to big-endian bytes
    // Phase 3b: Each group leader writes its own output
    if ((lane_id & 7) == 0) {
        for (int i = 0; i < 8; ++i) {
            int byte_base = (7 - i) * 4;
            out_r[byte_base+3] = (uint8_t)(sig_r[i]);
            out_r[byte_base+2] = (uint8_t)(sig_r[i] >> 8);
            out_r[byte_base+1] = (uint8_t)(sig_r[i] >> 16);
            out_r[byte_base+0] = (uint8_t)(sig_r[i] >> 24);
            out_s[byte_base+3] = (uint8_t)(sig_s[i]);
            out_s[byte_base+2] = (uint8_t)(sig_s[i] >> 8);
            out_s[byte_base+1] = (uint8_t)(sig_s[i] >> 16);
            out_s[byte_base+0] = (uint8_t)(sig_s[i] >> 24);
        }
    }

    return true;
}

} // namespace warp_coop
} // namespace p256
} // namespace smoke

#endif // P256_WARP_COOP_ENABLED

#endif // P256_SIGN_WARP_COOP_CUH
