// dilithium_expandA.h - ExpandA interface for Dilithium
// Card 12: Generate matrix A in NTT domain from seed rho
//
// Dilithium's matrix A is k x l polynomials in R_q = Z_q[X]/(X^256 + 1).
// ExpandA uses SHAKE-128 to deterministically expand seed rho into A,
// then applies NTT to produce A-hat (A in NTT domain).
//
// LAYOUT CONVENTION (LOCKED IN):
//   A_hat[(i, j, coeff)] -> d_A_hat[ ((i * params.l) + j) * N + coeff ]
//
//   where:
//     0 <= i < params.k  (rows)
//     0 <= j < params.l  (columns)
//     0 <= coeff < N     (polynomial coefficients, N=256)
//
//   This is row-major on (i, j), then coefficient index inside each poly.

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "dilithium_params.h"

namespace smoke {
namespace dilithium {

// ============================================================================
// Matrix A Layout Helpers
// ============================================================================

// Total number of polynomials in A: k * l
inline std::size_t expand_A_poly_count(const Params& params) {
    return static_cast<std::size_t>(params.k) *
           static_cast<std::size_t>(params.l);
}

// Total number of coefficients in A: k * l * n
inline std::size_t expand_A_coeff_count(const Params& params) {
    return expand_A_poly_count(params) *
           static_cast<std::size_t>(params.n);
}

// Flattened index for coefficient (i, j, c) in A
// Returns: ((i * l) + j) * n + c
inline std::size_t expand_A_index(const Params& params,
                                  std::size_t i,    // row index [0, k)
                                  std::size_t j,    // col index [0, l)
                                  std::size_t c) {  // coeff index [0, n)
    return ((i * params.l) + j) * params.n + c;
}

// Pointer to polynomial (i, j) within flattened A buffer
inline uint32_t* expand_A_poly_ptr(uint32_t* d_A_hat,
                                   const Params& params,
                                   std::size_t i,
                                   std::size_t j) {
    return d_A_hat + ((i * params.l) + j) * params.n;
}

// ============================================================================
// ExpandA API
// ============================================================================

// Dilithium seed length (SEEDBYTES = 32)
constexpr std::size_t SEEDBYTES = 32;

// GPU: Expand A from rho into NTT domain.
//
// This function:
//   1. Uses SHAKE-128 with domain separation (rho || j || i) for each (i, j)
//   2. Applies rejection sampling to produce uniform polynomials in R_q
//   3. Applies NTT to convert to frequency domain
//
// Parameters:
//   params  - Dilithium parameters (k, l, n, q)
//   rho     - 32-byte seed (host pointer)
//   d_A_hat - Device pointer to k*l*n uint32_t buffer (must be pre-allocated)
//   stream  - CUDA stream for async execution
//
// Output:
//   d_A_hat contains A in NTT domain with layout ((i*l)+j)*n + coeff
void expand_A_ntt_gpu(const Params& params,
                      const uint8_t rho[SEEDBYTES],
                      uint32_t* d_A_hat,
                      cudaStream_t stream = 0);

// ============================================================================
// Domain Separation (internal helper exposed for testing)
// ============================================================================

// Build ExpandA SHAKE input: rho || j || i (34 bytes)
// Note: Dilithium uses (j, i) order, not (i, j)!
inline void make_expandA_input(uint8_t* dst,
                               const uint8_t rho[SEEDBYTES],
                               uint8_t i,
                               uint8_t j) {
    // Copy rho (32 bytes)
    for (int k = 0; k < 32; ++k) {
        dst[k] = rho[k];
    }
    // Append j, i as single bytes (Dilithium FIPS 204 convention)
    dst[32] = j;
    dst[33] = i;
}

} // namespace dilithium
} // namespace smoke
