// dilithium_ntt.cuh - CUDA NTT interface for Dilithium
// Batched NTT/INTT for n=256, q=8380417
// Card 10: Adds warp-cooperative optimized kernel

#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "dilithium_params.h"

namespace smoke {
namespace dilithium {

// ============================================================================
// NTT Implementation Selection
// ============================================================================

enum class NTTImpl : uint8_t {
    BASELINE = 0,  // Single-thread per polynomial (Card 09, "boring but correct")
    WARP     = 1,  // Warp-cooperative (Card 10, 32 threads per polynomial)
    AUTO     = 2   // Runtime selection (defaults to WARP when stable)
};

// Initialize NTT twiddle factors in constant memory
// Must be called once before any NTT operations
void ntt_init();

// Check if NTT is initialized
bool ntt_is_initialized();

// ============================================================================
// Primary NTT API (with implementation selection)
// ============================================================================

// Forward NTT (in-place)
// Transforms polynomials from coefficient to NTT domain
// d_coeffs: device pointer to batch_size * 256 uint32_t elements in [0, q)
void ntt_forward_batch(uint32_t* d_coeffs,
                       std::size_t batch_size,
                       const Params& params,
                       cudaStream_t stream = 0,
                       NTTImpl impl = NTTImpl::AUTO);

// Inverse NTT (in-place)
// Transforms polynomials from NTT domain back to coefficient domain
// Includes scaling by n^-1 mod q
void ntt_inverse_batch(uint32_t* d_coeffs,
                       std::size_t batch_size,
                       const Params& params,
                       cudaStream_t stream = 0,
                       NTTImpl impl = NTTImpl::AUTO);

// ============================================================================
// Explicit Baseline Entrypoints (for testing/validation)
// ============================================================================

void ntt_forward_batch_baseline(uint32_t* d_coeffs,
                                std::size_t batch_size,
                                const Params& params,
                                cudaStream_t stream = 0);

void ntt_inverse_batch_baseline(uint32_t* d_coeffs,
                                std::size_t batch_size,
                                const Params& params,
                                cudaStream_t stream = 0);

// Pointwise multiply in NTT domain
// c[i] = a[i] * b[i] mod q for all i in [0, batch_size * 256)
// Result is in Montgomery form if inputs are
void pointwise_multiply_batch(uint32_t* d_out,
                              const uint32_t* d_a,
                              const uint32_t* d_b,
                              std::size_t batch_size,
                              const Params& params,
                              cudaStream_t stream = 0);

// Pointwise add in coefficient or NTT domain
// c[i] = (a[i] + b[i]) mod q
void pointwise_add_batch(uint32_t* d_out,
                         const uint32_t* d_a,
                         const uint32_t* d_b,
                         std::size_t batch_size,
                         const Params& params,
                         cudaStream_t stream = 0);

// Pointwise subtract in coefficient or NTT domain
// c[i] = (a[i] - b[i]) mod q (result in [0, q))
void pointwise_sub_batch(uint32_t* d_out,
                         const uint32_t* d_a,
                         const uint32_t* d_b,
                         std::size_t batch_size,
                         const Params& params,
                         cudaStream_t stream = 0);

// DEBUG: Get intermediate NTT stage output (only when DEBUG_DILITHIUM defined)
#ifdef DEBUG_DILITHIUM
void ntt_debug_dump_stage(const uint32_t* d_coeffs,
                          std::size_t batch_size,
                          std::size_t stage,
                          uint32_t* d_debug_out,
                          cudaStream_t stream = 0);
#endif

} // namespace dilithium
} // namespace smoke
