// dilithium_sampling.cpp - Secret sampling for Dilithium / ML-DSA
// Card 13.1: This file provides CPU-fallback stubs when CUDA is not available.
//
// The actual GPU implementation is in cuda/dilithium_sampling.cu.
// When building with CUDA, the .cu file provides the real implementations.
// When building without CUDA, this file provides stub implementations.
//
// In production, always use the CUDA build for GPU acceleration.

#include "../include/dilithium_sampling.h"
#include "../include/dilithium_params.h"

#include <stdexcept>

// ============================================================================
// Build Configuration
// ============================================================================
//
// If DILITHIUM_HAS_CUDA is defined, the real implementation comes from
// cuda/dilithium_sampling.cu and this file should NOT be linked.
//
// If DILITHIUM_HAS_CUDA is NOT defined, we provide stub implementations
// that throw errors, since GPU sampling without CUDA doesn't make sense.

#ifndef DILITHIUM_HAS_CUDA

namespace smoke {
namespace dilithium {

void sample_poly_eta_gpu(const Params& /* params */,
                         const uint8_t /* rhoprime */ [32],
                         uint16_t /* nonce */,
                         uint32_t* /* d_poly */,
                         cudaStream_t /* stream */) {
    throw std::runtime_error(
        "sample_poly_eta_gpu: CUDA not available. "
        "Build with DILITHIUM_HAS_CUDA to enable GPU sampling."
    );
}

void sample_s1_gpu(const Params& /* params */,
                   const uint8_t /* rhoprime */ [32],
                   uint32_t* /* d_s1 */,
                   cudaStream_t /* stream */) {
    throw std::runtime_error(
        "sample_s1_gpu: CUDA not available. "
        "Build with DILITHIUM_HAS_CUDA to enable GPU sampling."
    );
}

void sample_s2_gpu(const Params& /* params */,
                   const uint8_t /* rhoprime */ [32],
                   uint32_t* /* d_s2 */,
                   cudaStream_t /* stream */) {
    throw std::runtime_error(
        "sample_s2_gpu: CUDA not available. "
        "Build with DILITHIUM_HAS_CUDA to enable GPU sampling."
    );
}

} // namespace dilithium
} // namespace smoke

#endif // !DILITHIUM_HAS_CUDA
