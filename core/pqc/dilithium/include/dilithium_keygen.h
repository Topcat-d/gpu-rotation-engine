// dilithium_keygen.h - Keygen core interface for Dilithium / ML-DSA
// Card 14: Compute t = A*s1 + s2 using NTT-domain matrix-vector multiplication
//
// This module ties together:
//   - ExpandA (matrix A in NTT domain)
//   - Secret sampling (s1, s2 vectors)
//   - NTT-domain matvec computation
//
// The keygen core produces t = A*s1 + s2 in time domain.
// Future Card 14.2 will add t1/t0 decomposition (power2round) and key packing.

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "dilithium_params.h"

namespace smoke {
namespace dilithium {

// =============================================================================
// Core Computation: t = A*s1 + s2 (NTT-domain matvec)
// =============================================================================

// Compute t = A * s1 + s2 in R_q^k, using NTT-domain matrix-vector multiply.
//
// Algorithm:
//   1. Forward NTT: s1_hat = NTT(s1), s2_hat = NTT(s2)
//   2. Matvec: t_hat[i] = sum_j(A_hat[i,j] * s1_hat[j]) + s2_hat[i]
//      (pointwise multiply and add in frequency domain)
//   3. Inverse NTT: t = INTT(t_hat)
//
// Inputs:
//   params  : Dilithium parameters (k, l, n, q)
//   d_A_hat : A in NTT domain, layout: d_A_hat[((i * l) + j) * n + coeff]
//             size = k * l * n uint32_t
//   d_s1    : s1 vector in TIME domain, layout: d_s1[j * n + coeff], j=0..l-1
//             size = l * n uint32_t
//   d_s2    : s2 vector in TIME domain, layout: d_s2[i * n + coeff], i=0..k-1
//             size = k * n uint32_t
//
// Output:
//   d_t     : t vector in TIME domain, layout: d_t[i * n + coeff], i=0..k-1
//             size = k * n uint32_t
//
// Notes:
//   - A_hat is expected to already be in NTT domain (from expand_A_ntt_gpu)
//   - s1, s2 are in time domain (from sample_s1_gpu, sample_s2_gpu)
//   - Output t is in time domain
//   - All operations use the AUTO NTT implementation (warp-optimized when available)
//
void compute_t_As1_plus_s2_ntt_gpu(const Params& params,
                                   const uint32_t* d_A_hat,
                                   const uint32_t* d_s1,
                                   const uint32_t* d_s2,
                                   uint32_t* d_t,
                                   cudaStream_t stream = 0);

// =============================================================================
// Full Keygen Core: orchestrate ExpandA + sampling + matvec
// =============================================================================

// Full keygen core computation: expand A, sample s1/s2, compute t.
//
// This function orchestrates:
//   1. expand_A_ntt_gpu(params, rho, d_A_hat, stream)
//   2. sample_s1_gpu(params, rhoprime, d_s1, stream)
//   3. sample_s2_gpu(params, rhoprime, d_s2, stream)
//   4. compute_t_As1_plus_s2_ntt_gpu(params, d_A_hat, d_s1, d_s2, d_t, stream)
//
// Caller must pre-allocate all device buffers:
//   d_A_hat : k * l * n uint32_t
//   d_s1    : l * n uint32_t
//   d_s2    : k * n uint32_t
//   d_t     : k * n uint32_t
//
// Seeds:
//   rho      : 32-byte seed for matrix A expansion
//   rhoprime : 32-byte seed for secret sampling
//
void keygen_core_gpu(const Params& params,
                     const uint8_t rho[32],
                     const uint8_t rhoprime[32],
                     uint32_t* d_A_hat,
                     uint32_t* d_s1,
                     uint32_t* d_s2,
                     uint32_t* d_t,
                     cudaStream_t stream = 0);

// =============================================================================
// Buffer Size Helpers
// =============================================================================

// Total coefficients in t vector: k * n
inline std::size_t t_coeff_count(const Params& params) {
    return static_cast<std::size_t>(params.k) *
           static_cast<std::size_t>(params.n);
}

// Total bytes for t vector buffer
inline std::size_t t_buffer_size(const Params& params) {
    return t_coeff_count(params) * sizeof(uint32_t);
}

// =============================================================================
// Card 14.2: Full Keygen with Power2Round and Packing on GPU
// =============================================================================

// Full keygen producing packed pk and sk bytes on device.
//
// This is the optimized path with:
//   - keygen_core_gpu (ExpandA + sampling + matvec)
//   - power2round_vec_gpu (t -> t1, t0 on GPU)
//   - pack_pk_gpu / pack_sk_gpu (packing on GPU)
//   - Single D->H copy for final pk/sk bytes
//
// Caller provides device buffers for output:
//   d_pk : pk_simple_size(params) bytes
//   d_sk : sk_simple_size(params) bytes
//
// Seeds are passed as host pointers (copied to device internally).
//
void keygen_full_gpu(const Params& params,
                     const uint8_t rho[32],
                     const uint8_t rhoprime[32],
                     uint8_t* d_pk,
                     uint8_t* d_sk,
                     cudaStream_t stream = 0);

// =============================================================================
// Card 14.2: Batched Keygen (B > 1)
// =============================================================================

// Batched keygen producing B keypairs.
//
// Each item is independent - derived from its own (rho, rhoprime) seed pair.
//
// Inputs:
//   params      : Dilithium parameters
//   h_rho       : Host pointer to B * 32 bytes (rho seeds, row-major)
//   h_rhoprime  : Host pointer to B * 32 bytes (rhoprime seeds, row-major)
//   batch_size  : Number of keypairs to generate (B)
//
// Outputs:
//   d_pk        : Device buffer for B packed public keys [B * pk_simple_size]
//   d_sk        : Device buffer for B packed secret keys [B * sk_simple_size]
//
// Performance: Amortizes kernel launch overhead over B items.
//
void keygen_full_batch_gpu(const Params& params,
                           const uint8_t* h_rho,
                           const uint8_t* h_rhoprime,
                           uint8_t* d_pk,
                           uint8_t* d_sk,
                           int batch_size,
                           cudaStream_t stream = 0);

} // namespace dilithium
} // namespace smoke
