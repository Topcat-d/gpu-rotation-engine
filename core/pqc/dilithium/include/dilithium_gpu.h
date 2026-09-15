// dilithium_gpu.h - GPU API for Dilithium operations
// Job structures and entrypoints for keygen/sign/verify

#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "dilithium_params.h"

namespace smoke {
namespace dilithium {

// ============================================================================
// Job Structures
// ============================================================================

// Keygen job: generate batch_size keypairs
struct KeygenJob {
    Mode mode;
    std::size_t batch_size;

    // Device pointers - caller allocates, sized per mode
    uint8_t* d_public_keys;  // batch_size * pk_size(mode)
    uint8_t* d_secret_keys;  // batch_size * sk_size(mode)

    // Optional: seeds for deterministic generation (for testing)
    const uint8_t* d_seeds;  // batch_size * 32, or nullptr for random
};

// Sign job: sign batch_size messages
struct SignJob {
    Mode mode;
    std::size_t batch_size;

    // Messages (concatenated, fixed-size for now)
    const uint8_t* d_messages;
    std::size_t message_len;  // Length of each message (same for all)

    // Secret keys (batch_size * sk_size)
    const uint8_t* d_secret_keys;

    // Output: signatures (batch_size * sig_size)
    uint8_t* d_signatures;

    // Optional: actual signature lengths if variable (for future)
    std::size_t* d_sig_lens;
};

// Verify job: verify batch_size signatures
struct VerifyJob {
    Mode mode;
    std::size_t batch_size;

    // Messages
    const uint8_t* d_messages;
    std::size_t message_len;

    // Signatures (batch_size * sig_size)
    const uint8_t* d_signatures;

    // Public keys (batch_size * pk_size)
    const uint8_t* d_public_keys;

    // Output: verification results
    // 0 = INVALID, 1 = VALID
    uint8_t* d_results;
};

// ============================================================================
// Size Helpers
// ============================================================================

// Get public key size for a mode
inline std::size_t pk_size(Mode mode) {
    return get_params(mode).pk_size;
}

// Get secret key size for a mode
inline std::size_t sk_size(Mode mode) {
    return get_params(mode).sk_size;
}

// Get signature size for a mode
inline std::size_t sig_size(Mode mode) {
    return get_params(mode).sig_size;
}

// ============================================================================
// GPU Operations (STUBS - to be implemented)
// ============================================================================

// Generate keypairs on GPU
// Throws std::runtime_error until implemented
void keygen_gpu(const KeygenJob& job, cudaStream_t stream = 0);

// Sign messages on GPU
// Throws std::runtime_error until implemented
void sign_gpu(const SignJob& job, cudaStream_t stream = 0);

// Verify signatures on GPU
// Throws std::runtime_error until implemented
void verify_gpu(const VerifyJob& job, cudaStream_t stream = 0);

// ============================================================================
// Batch Allocation Helpers
// ============================================================================

// Allocate device memory for a keygen job
// Returns 0 on success, CUDA error code on failure
int alloc_keygen_buffers(KeygenJob& job);

// Free device memory from a keygen job
void free_keygen_buffers(KeygenJob& job);

// Similar for sign/verify...
int alloc_sign_buffers(SignJob& job);
void free_sign_buffers(SignJob& job);

int alloc_verify_buffers(VerifyJob& job);
void free_verify_buffers(VerifyJob& job);

} // namespace dilithium
} // namespace smoke
