// dilithium_decompose.h - Decomposition functions for Dilithium / ML-DSA
// Card 14.2: power2round decomposition (t -> t1, t0)
//
// FIPS 204 Section 8.4: Power2Round
//   For coefficient a in [0, q):
//     a1 = (a + 2^(d-1)) >> d    (high bits, public in t1)
//     a0 = a - a1 * 2^d          (low bits, secret in t0)
//
// The decomposition satisfies: a = a1 * 2^d + a0 (mod q)
// with a0 centered around 0 (in range roughly [-2^(d-1), 2^(d-1)]).
//
// All modes use d=13, so t1 contains the top 10 bits of each coefficient.

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "dilithium_params.h"

namespace smoke {
namespace dilithium {

// =============================================================================
// Power2Round Decomposition
// =============================================================================

// GPU: Decompose t vector (k polynomials) into t1 and t0.
//
// For each coefficient a in t:
//   a1 = (a + 2^(d-1)) >> d         (rounded high bits)
//   a0 = a - a1 * 2^d               (low bits, can be negative)
//   a0 is encoded in [0, q) as: a0 >= 0 ? a0 : q + a0
//
// Inputs:
//   params : Dilithium parameters (q, d, k, n)
//   d_t    : Input t vector, shape [k * n], time domain, in [0, q)
//
// Outputs:
//   d_t1   : High bits, shape [k * n], values in [0, 2^(23-d)]
//   d_t0   : Low bits encoded in [0, q), shape [k * n]
//
void power2round_vec_gpu(const Params& params,
                         const uint32_t* d_t,
                         uint32_t* d_t1,
                         uint32_t* d_t0,
                         cudaStream_t stream = 0);

// GPU: Decompose a single polynomial (n coefficients)
void power2round_poly_gpu(const Params& params,
                          const uint32_t* d_poly_in,
                          uint32_t* d_poly_t1,
                          uint32_t* d_poly_t0,
                          cudaStream_t stream = 0);

// =============================================================================
// Buffer Size Helpers
// =============================================================================

// Coefficient count for t vector: k * n
inline std::size_t t_vec_coeff_count(const Params& params) {
    return static_cast<std::size_t>(params.k) *
           static_cast<std::size_t>(params.n);
}

// Buffer size in bytes for t, t1, or t0 vector
inline std::size_t t_vec_buffer_size(const Params& params) {
    return t_vec_coeff_count(params) * sizeof(uint32_t);
}

// =============================================================================
// Range Constants
// =============================================================================

// Maximum value of t1 coefficient: (q-1 + 2^(d-1)) >> d
// For q=8380417, d=13: max_t1 = (8380416 + 4096) >> 13 = 1023
inline uint32_t max_t1_value(const Params& params) {
    uint32_t d = params.d;
    return (params.q - 1 + (1u << (d - 1))) >> d;
}

// Bound for |t0| after decoding: 2^(d-1)
// For d=13: bound = 4096
inline uint32_t t0_bound(const Params& params) {
    return 1u << (params.d - 1);
}

} // namespace dilithium
} // namespace smoke
