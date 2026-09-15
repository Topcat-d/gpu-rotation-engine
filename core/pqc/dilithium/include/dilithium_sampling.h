// dilithium_sampling.h - Secret sampling for Dilithium / ML-DSA
// Card 13: poly_eta sampling for s1, s2 vectors
//
// FIPS 204: Secrets are sampled as polynomials with coefficients in [-eta, eta].
// We encode coefficients as uint32_t in [0, q) with canonical lift:
//   coeff in [-eta, eta] -> coeff mod q (negative values become q + coeff)
//
// Domain separation: rhoprime || nonce (2 bytes, little-endian)
// XOF: SHAKE-256 (per FIPS 204 for secret sampling)
//
// Nonce policy:
//   s1[i] uses nonce = i           (i = 0..l-1)
//   s2[i] uses nonce = l + i       (i = 0..k-1)

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "dilithium_params.h"

namespace smoke {
namespace dilithium {

// =============================================================================
// Core Primitive: sample_poly_eta
// =============================================================================

// Sample a single polynomial with coefficients in [-eta, eta].
// Coefficients are encoded in [0, q): negative values become q + coeff.
//
// For Card 13, this runs on CPU and uploads to GPU.
// Future cards may implement full GPU sampling.
//
// Args:
//   params   : Dilithium parameters (contains q, n, eta)
//   rhoprime : 32-byte seed
//   nonce    : 16-bit nonce (incremented for each polynomial)
//   d_poly   : Device pointer to n uint32_t coefficients
//   stream   : CUDA stream (default 0)
void sample_poly_eta_gpu(const Params& params,
                         const uint8_t rhoprime[32],
                         uint16_t nonce,
                         uint32_t* d_poly,
                         cudaStream_t stream = 0);

// =============================================================================
// Vector Sampling: s1, s2
// =============================================================================

// Layout: row-major in poly index, then coeff index
//   s1[i, coeff] -> d_s1[i * n + coeff], i = 0..(l - 1)
//   s2[i, coeff] -> d_s2[i * n + coeff], i = 0..(k - 1)

// Sample s1 vector (l polynomials)
// Nonces: 0, 1, ..., l-1
void sample_s1_gpu(const Params& params,
                   const uint8_t rhoprime[32],
                   uint32_t* d_s1,  // size = l * n
                   cudaStream_t stream = 0);

// Sample s2 vector (k polynomials)
// Nonces: l, l+1, ..., l+k-1
void sample_s2_gpu(const Params& params,
                   const uint8_t rhoprime[32],
                   uint32_t* d_s2,  // size = k * n
                   cudaStream_t stream = 0);

// =============================================================================
// Count Helpers
// =============================================================================

inline std::size_t s1_poly_count(const Params& params) {
    return static_cast<std::size_t>(params.l);
}

inline std::size_t s2_poly_count(const Params& params) {
    return static_cast<std::size_t>(params.k);
}

inline std::size_t s1_coeff_count(const Params& params) {
    return s1_poly_count(params) * static_cast<std::size_t>(params.n);
}

inline std::size_t s2_coeff_count(const Params& params) {
    return s2_poly_count(params) * static_cast<std::size_t>(params.n);
}

inline std::size_t secret_coeff_count(const Params& params) {
    return s1_coeff_count(params) + s2_coeff_count(params);
}

// =============================================================================
// Domain Separation Helper
// =============================================================================

// Build input buffer for poly_eta sampling: rhoprime || nonce_le
// dst must be at least 34 bytes (32 + 2)
inline void make_secret_input(uint8_t* dst,
                              const uint8_t rhoprime[32],
                              uint16_t nonce) {
    for (int i = 0; i < 32; ++i) {
        dst[i] = rhoprime[i];
    }
    dst[32] = static_cast<uint8_t>(nonce & 0xff);
    dst[33] = static_cast<uint8_t>(nonce >> 8);
}

constexpr std::size_t SECRET_INPUT_LEN = 34;  // 32 + 2

} // namespace dilithium
} // namespace smoke
