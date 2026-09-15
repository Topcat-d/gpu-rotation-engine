// dilithium_pack.h - GPU packing for Dilithium keys
// Card 14.2: Pack pk and sk on GPU for efficient keygen
//
// SIMPLE LAYOUT (matching dilithium_pack_ref.py):
//   pk = rho (32 bytes) || t1 (k * n * 4 bytes, little-endian uint32)
//   sk = rho (32) || rhoprime (32) || s1 || s2 || t0 (all as LE uint32)
//
// This avoids D->H copies for intermediate arrays by packing directly on GPU.

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "dilithium_params.h"

namespace smoke {
namespace dilithium {

// =============================================================================
// Buffer Size Helpers
// =============================================================================

// Public key size: 32 + k * n * 4
inline std::size_t pk_simple_size(const Params& params) {
    return 32 + static_cast<std::size_t>(params.k) * params.n * sizeof(uint32_t);
}

// Secret key size: 32 + 32 + l*n*4 + k*n*4 + k*n*4
inline std::size_t sk_simple_size(const Params& params) {
    return 32 + 32 +
           static_cast<std::size_t>(params.l) * params.n * sizeof(uint32_t) +
           static_cast<std::size_t>(params.k) * params.n * sizeof(uint32_t) +
           static_cast<std::size_t>(params.k) * params.n * sizeof(uint32_t);
}

// =============================================================================
// GPU Packing Functions
// =============================================================================

// Pack public key on GPU:
//   d_pk = rho || t1 (as little-endian uint32)
//
// Inputs:
//   params  : Dilithium parameters
//   d_rho   : Device pointer to rho (32 bytes)
//   d_t1    : Device pointer to t1 coefficients (k * n uint32_t)
//
// Output:
//   d_pk    : Device pointer to output pk buffer (pk_simple_size bytes)
//
void pack_pk_gpu(const Params& params,
                 const uint8_t* d_rho,
                 const uint32_t* d_t1,
                 uint8_t* d_pk,
                 cudaStream_t stream = 0);

// Pack secret key on GPU:
//   d_sk = rho || rhoprime || s1 || s2 || t0 (all as little-endian uint32)
//
// Inputs:
//   params     : Dilithium parameters
//   d_rho      : Device pointer to rho (32 bytes)
//   d_rhoprime : Device pointer to rhoprime (32 bytes)
//   d_s1       : Device pointer to s1 coefficients (l * n uint32_t)
//   d_s2       : Device pointer to s2 coefficients (k * n uint32_t)
//   d_t0       : Device pointer to t0 coefficients (k * n uint32_t)
//
// Output:
//   d_sk       : Device pointer to output sk buffer (sk_simple_size bytes)
//
void pack_sk_gpu(const Params& params,
                 const uint8_t* d_rho,
                 const uint8_t* d_rhoprime,
                 const uint32_t* d_s1,
                 const uint32_t* d_s2,
                 const uint32_t* d_t0,
                 uint8_t* d_sk,
                 cudaStream_t stream = 0);

// =============================================================================
// Batch Packing (for batched keygen)
// =============================================================================

// Pack B public keys on GPU.
// All arrays are batched: d_rho[B][32], d_t1[B][k*n], d_pk[B][pk_size]
void pack_pk_batch_gpu(const Params& params,
                       const uint8_t* d_rho,      // [B * 32]
                       const uint32_t* d_t1,      // [B * k * n]
                       uint8_t* d_pk,             // [B * pk_size]
                       int batch_size,
                       cudaStream_t stream = 0);

// Pack B secret keys on GPU.
void pack_sk_batch_gpu(const Params& params,
                       const uint8_t* d_rho,      // [B * 32]
                       const uint8_t* d_rhoprime, // [B * 32]
                       const uint32_t* d_s1,      // [B * l * n]
                       const uint32_t* d_s2,      // [B * k * n]
                       const uint32_t* d_t0,      // [B * k * n]
                       uint8_t* d_sk,             // [B * sk_size]
                       int batch_size,
                       cudaStream_t stream = 0);

} // namespace dilithium
} // namespace smoke
