// dilithium_sign.h - GPU API for Dilithium signing
// Card 17: GPU-backed sign/verify
//
// This file provides GPU kernels for:
//   - sample_y_gamma1_gpu: Sample y from gamma1 distribution
//   - compute_z_gpu: z = y + c*s1
//   - check_norms_gpu: Check ||z||_inf and ||r0||_inf bounds
//   - make_hint_gpu: MakeHint per FIPS 204
//   - use_hint_gpu: UseHint per FIPS 204
//
// FIPS 204 Section 6.2/6.3 (ML-DSA Sign/Verify)

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "dilithium_params.h"

namespace smoke {
namespace dilithium {

// =============================================================================
// GPU API for Signing Operations
// =============================================================================

// Sample y vector from gamma1 distribution.
//
// FIPS 204 ExpandMask:
// - For ML-DSA-44: coefficients in [-(gamma1-1), gamma1-1] = [-131071, 131071]
// - For ML-DSA-65/87: coefficients in [-524287, 524287]
// - Uses SHAKE-256 XOF from (rhoprime || nonce)
// - Nonce = kappa * l + j for polynomial j in iteration kappa
//
// Inputs:
//   params: mode parameters
//   d_rhoprime: 32-byte seed on device
//   kappa: rejection loop counter
//
// Outputs:
//   d_y: y vector in time domain, shape (l, n), encoded in [0, q)
//        Size: l * n uint32_t
//
void sample_y_gamma1_gpu(const Params& params,
                          const uint8_t* d_rhoprime,
                          uint32_t kappa,
                          uint32_t* d_y,
                          cudaStream_t stream = 0);

// Compute z = y + c*s1 (polynomial-wise).
//
// For each polynomial j in [0, l):
//   z[j] = y[j] + c * s1[j] (mod q)
// where c * s1[j] is polynomial multiplication via NTT.
//
// Inputs:
//   params: mode parameters
//   d_y: y vector, shape (l, n), dtype uint32
//   d_c: challenge polynomial, shape (n,), dtype int8 in {-1, 0, 1}
//   d_s1: secret s1 vector, shape (l, n), dtype uint32
//
// Outputs:
//   d_z: response vector z, shape (l, n), dtype uint32
//
void compute_z_gpu(const Params& params,
                   const uint32_t* d_y,
                   const int8_t* d_c,
                   const uint32_t* d_s1,
                   uint32_t* d_z,
                   cudaStream_t stream = 0);

// Compute r = w - c*s2 (polynomial-wise).
//
// For each polynomial i in [0, k):
//   r[i] = w[i] - c * s2[i] (mod q)
//
// Inputs:
//   params: mode parameters
//   d_w: w vector, shape (k, n), dtype uint32
//   d_c: challenge polynomial, shape (n,), dtype int8 in {-1, 0, 1}
//   d_s2: secret s2 vector, shape (k, n), dtype uint32
//
// Outputs:
//   d_r: result vector, shape (k, n), dtype uint32
//
void compute_r_gpu(const Params& params,
                   const uint32_t* d_w,
                   const int8_t* d_c,
                   const uint32_t* d_s2,
                   uint32_t* d_r,
                   cudaStream_t stream = 0);

// Compute c*t0 (for hint computation).
//
// For each polynomial i in [0, k):
//   ct0[i] = c * t0[i] (mod q)
//
// Inputs:
//   params: mode parameters
//   d_c: challenge polynomial, shape (n,), dtype int8 in {-1, 0, 1}
//   d_t0: t0 vector, shape (k, n), dtype uint32
//
// Outputs:
//   d_ct0: result vector, shape (k, n), dtype uint32
//
void compute_ct0_gpu(const Params& params,
                     const int8_t* d_c,
                     const uint32_t* d_t0,
                     uint32_t* d_ct0,
                     cudaStream_t stream = 0);

// Check norm bounds for rejection sampling.
//
// Returns two flags:
//   z_ok: ||z||_inf < gamma1 - beta
//   r0_ok: ||r0||_inf < gamma2 - beta
//
// Inputs:
//   params: mode parameters
//   d_z: response vector, shape (l, n)
//   d_r0: low bits of r, shape (k, n), encoded in [0, q)
//
// Outputs:
//   z_ok: pointer to single uint8 (0 = fail, 1 = pass)
//   r0_ok: pointer to single uint8 (0 = fail, 1 = pass)
//
// Note: Outputs are on DEVICE. Copy back to host for rejection decision.
//
void check_norms_gpu(const Params& params,
                     const uint32_t* d_z,
                     const uint32_t* d_r0,
                     uint8_t* d_z_ok,
                     uint8_t* d_r0_ok,
                     cudaStream_t stream = 0);

// Compute MakeHint on GPU.
//
// FIPS 204 Algorithm 40:
//   For each coefficient:
//     h = 1 if HighBits(r) != HighBits(r + z)
//     h = 0 otherwise
//
// In signing: z = -c*t0, r = w - c*s2 + c*t0
//
// Inputs:
//   params: mode parameters
//   d_neg_ct0: -c*t0, shape (k, n), encoded in [0, q)
//   d_r_hint: w - c*s2 + c*t0, shape (k, n), encoded in [0, q)
//
// Outputs:
//   d_h: hint bits, shape (k, n), dtype uint8 in {0, 1}
//   d_hint_count: pointer to single uint32 (total 1s in h)
//
void make_hint_gpu(const Params& params,
                   const uint32_t* d_neg_ct0,
                   const uint32_t* d_r_hint,
                   uint8_t* d_h,
                   uint32_t* d_hint_count,
                   cudaStream_t stream = 0);

// Compute UseHint on GPU for verification.
//
// FIPS 204 Algorithm 41:
//   If h = 0: return HighBits(r)
//   If h = 1: adjust HighBits based on sign of LowBits(r)
//
// In verification: r = w' = A*z - c*t
//
// Inputs:
//   params: mode parameters
//   d_h: hint bits, shape (k, n), dtype uint8 in {0, 1}
//   d_w_prime: A*z - c*t, shape (k, n), dtype uint32
//
// Outputs:
//   d_w1_recon: recovered w1 = UseHint(h, w'), shape (k, n), dtype uint32
//
void use_hint_gpu(const Params& params,
                  const uint8_t* d_h,
                  const uint32_t* d_w_prime,
                  uint32_t* d_w1_recon,
                  cudaStream_t stream = 0);

// Compute c*t for verification (t = t1 * 2^d).
//
// For each polynomial i in [0, k):
//   ct[i] = c * (t1[i] * 2^d) (mod q)
//
// Inputs:
//   params: mode parameters
//   d_c: challenge polynomial, shape (n,), dtype int8 in {-1, 0, 1}
//   d_t1: t1 vector from public key, shape (k, n), dtype uint32
//
// Outputs:
//   d_ct: result vector c*t, shape (k, n), dtype uint32
//
void compute_ct_gpu(const Params& params,
                    const int8_t* d_c,
                    const uint32_t* d_t1,
                    uint32_t* d_ct,
                    cudaStream_t stream = 0);

// Compute w' = A*z - c*t for verification.
//
// Inputs:
//   params: mode parameters
//   d_Az: A*z result, shape (k, n)
//   d_ct: c*t result, shape (k, n)
//
// Outputs:
//   d_w_prime: w' = Az - ct, shape (k, n)
//
void compute_w_prime_gpu(const Params& params,
                         const uint32_t* d_Az,
                         const uint32_t* d_ct,
                         uint32_t* d_w_prime,
                         cudaStream_t stream = 0);

// =============================================================================
// Buffer size helpers
// =============================================================================

inline std::size_t z_vec_coeff_count(const Params& params) {
    return static_cast<std::size_t>(params.l) *
           static_cast<std::size_t>(params.n);
}

inline std::size_t hint_vec_coeff_count(const Params& params) {
    return static_cast<std::size_t>(params.k) *
           static_cast<std::size_t>(params.n);
}

// =============================================================================
// CPU reference functions (for testing)
// =============================================================================

// Infinity norm with signed decode
int64_t infinity_norm_cpu(const uint32_t* coeffs, std::size_t count, uint32_t q);

// Check z norm bound: ||z||_inf < gamma1 - beta
bool check_z_norm_cpu(const uint32_t* z, std::size_t count,
                      uint32_t gamma1, uint32_t beta, uint32_t q);

// Check r0 norm bound: ||r0||_inf < gamma2 - beta
bool check_r0_norm_cpu(const uint32_t* r0, std::size_t count,
                       uint32_t gamma2, uint32_t beta, uint32_t q);

} // namespace dilithium
} // namespace smoke
