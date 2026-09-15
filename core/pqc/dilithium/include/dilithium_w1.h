// dilithium_w1.h - w = A*y and w1 = highbits(w) computation
// Card 16.1: Real w and w1 for Dilithium signing
//
// FIPS 204 Section 8.4: HighBits and LowBits decomposition
// For w decomposition, we use alpha = 2*gamma2

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "dilithium_params.h"

namespace smoke {
namespace dilithium {

// =============================================================================
// GPU API for w and w1 computation
// =============================================================================

// Compute w = A * y in time domain.
//
// Algorithm:
//   1. Forward NTT on y -> y_hat
//   2. Pointwise matvec: w_hat[i] = sum_j(A_hat[i,j] * y_hat[j])
//   3. Inverse NTT on w_hat -> w
//
// Inputs:
//   params: mode parameters
//   d_A_hat: A in NTT domain, layout:
//       A_hat[(i, j, coeff)] -> d_A_hat[((i * l) + j) * n + coeff]
//       Size: k * l * n uint32_t
//   d_y: y in time domain, layout:
//       y[j, coeff] -> d_y[j * n + coeff]
//       Size: l * n uint32_t
//
// Outputs:
//   d_w: w in time domain, layout:
//       w[i, coeff] -> d_w[i * n + coeff]
//       Size: k * n uint32_t
//
// All pointers are device pointers.
void compute_w_gpu(const Params& params,
                   const uint32_t* d_A_hat,
                   const uint32_t* d_y,
                   uint32_t* d_w,
                   cudaStream_t stream = 0);

// Decompose w into (w1, w0) using FIPS 204 HighBits/LowBits.
//
// FIPS 204 Algorithm 35/36:
//   alpha = 2 * gamma2
//   r' = r mod q
//   r0 = r' mod+- alpha  (centered mod)
//   if r' - r0 == q - 1:
//       r1 = 0, r0 = r0 - 1
//   else:
//       r1 = (r' - r0) / alpha
//
// Inputs:
//   params: mode parameters (uses gamma2)
//   d_w: w in time domain, shape (k, n)
//       Size: k * n uint32_t
//
// Outputs:
//   d_w1: highbits(w), shape (k, n)
//       Size: k * n uint32_t
//   d_w0: lowbits(w), encoded in [0, q), shape (k, n)
//       Size: k * n uint32_t
//
// All pointers are device pointers.
void decompose_w_gpu(const Params& params,
                     const uint32_t* d_w,
                     uint32_t* d_w1,
                     uint32_t* d_w0,
                     cudaStream_t stream = 0);

// =============================================================================
// Buffer size helpers
// =============================================================================

inline std::size_t w_vec_coeff_count(const Params& params) {
    return static_cast<std::size_t>(params.k) *
           static_cast<std::size_t>(params.n);
}

inline std::size_t y_vec_coeff_count(const Params& params) {
    return static_cast<std::size_t>(params.l) *
           static_cast<std::size_t>(params.n);
}

// =============================================================================
// CPU reference functions (for testing)
// =============================================================================

// CPU version of highbits/lowbits decomposition for a single coefficient
// Returns: (w1, w0_encoded) where w0_encoded is in [0, q)
void decompose_coeff_cpu(uint32_t a, uint32_t q, uint32_t gamma2,
                         uint32_t& w1_out, uint32_t& w0_out);

// Verify recomposition: w ≡ w1 * alpha + w0_decoded (mod q)
// Returns true if recomposition matches original
bool verify_w_recomposition(uint32_t w, uint32_t w1, uint32_t w0_enc,
                            uint32_t q, uint32_t gamma2);

} // namespace dilithium
} // namespace smoke
