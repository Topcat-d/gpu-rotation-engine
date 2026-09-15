// dilithium_shake.h - SHAKE XOF interface for Dilithium
// Card 11: GPU Keccak + CPU Oracle for SHAKE-128/256
//
// Dilithium uses SHAKE everywhere:
// - ExpandA: SHAKE-128 to expand seed rho into matrix A
// - Secret sampling: SHAKE-256 for s1, s2
// - Challenge: SHAKE-256 for c_tilde
//
// This header defines a Dilithium-focused SHAKE XOF API.

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace smoke {
namespace dilithium {

// ============================================================================
// SHAKE Variants
// ============================================================================

enum class SHAKEKind : uint8_t {
    SHAKE128 = 0,  // 128-bit security, used for ExpandA
    SHAKE256 = 1   // 256-bit security, used for secrets & challenges
};

// ============================================================================
// XOF Job Descriptor
// ============================================================================

// A single XOF job: absorb input -> squeeze output
struct SHAKEJob {
    const uint8_t* d_in;    // Device pointer to input bytes
    std::size_t    in_len;  // Input length in bytes

    uint8_t*       d_out;   // Device pointer to output buffer
    std::size_t    out_len; // Desired output length in bytes
};

// ============================================================================
// Batched SHAKE XOF API
// ============================================================================

// Process multiple independent SHAKE XOF calls in parallel.
// Each job[i] is processed independently: absorb(d_in) -> squeeze(d_out).
//
// Parameters:
//   kind       - SHAKE128 or SHAKE256
//   jobs       - Host array of job descriptors (d_in/d_out are device pointers)
//   batch_size - Number of jobs to process
//   stream     - CUDA stream for async execution
//
// Notes:
//   - Jobs are independent and processed in parallel
//   - d_in and d_out must be valid device pointers
//   - Output is written to d_out[0..out_len-1]
void shake_xof_batch(SHAKEKind kind,
                     const SHAKEJob* jobs,
                     std::size_t batch_size,
                     cudaStream_t stream = 0);

// ============================================================================
// Single-Job Convenience Wrappers
// ============================================================================

// SHAKE-128 XOF: absorb input, squeeze out_len bytes
inline void shake128_xof(const uint8_t* d_in, std::size_t in_len,
                         uint8_t* d_out, std::size_t out_len,
                         cudaStream_t stream = 0) {
    SHAKEJob job = {d_in, in_len, d_out, out_len};
    shake_xof_batch(SHAKEKind::SHAKE128, &job, 1, stream);
}

// SHAKE-256 XOF: absorb input, squeeze out_len bytes
inline void shake256_xof(const uint8_t* d_in, std::size_t in_len,
                         uint8_t* d_out, std::size_t out_len,
                         cudaStream_t stream = 0) {
    SHAKEJob job = {d_in, in_len, d_out, out_len};
    shake_xof_batch(SHAKEKind::SHAKE256, &job, 1, stream);
}

// ============================================================================
// Dilithium-Specific SHAKE Usage (Future Cards)
// ============================================================================

// These will be implemented in Card 12 (ExpandA) and beyond:
//
// void expand_a_from_rho(const uint8_t* d_rho, uint32_t* d_A_ntt, ...);
// void sample_s1_s2(const uint8_t* d_rhoprime, uint32_t* d_s1, uint32_t* d_s2, ...);
// void challenge_hash(const uint8_t* d_mu, const uint8_t* d_w1, uint8_t* d_c_tilde, ...);

} // namespace dilithium
} // namespace smoke
