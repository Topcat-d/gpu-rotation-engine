// dilithium_gpu.cpp - GPU API stubs for Dilithium
// These will throw until fully implemented in later cards

#include "../include/dilithium_gpu.h"
#include <stdexcept>
#include <cuda_runtime.h>

namespace smoke {
namespace dilithium {

// ============================================================================
// GPU Operations (STUBS)
// ============================================================================

void keygen_gpu(const KeygenJob& job, cudaStream_t stream) {
    (void)job;
    (void)stream;
    throw std::runtime_error(
        "Dilithium GPU keygen not implemented yet. "
        "See PIPELINE.md - requires Layer C (keygen) completion."
    );
}

void sign_gpu(const SignJob& job, cudaStream_t stream) {
    (void)job;
    (void)stream;
    throw std::runtime_error(
        "Dilithium GPU sign not implemented yet. "
        "See PIPELINE.md - requires Layer D (sign/verify) completion."
    );
}

void verify_gpu(const VerifyJob& job, cudaStream_t stream) {
    (void)job;
    (void)stream;
    throw std::runtime_error(
        "Dilithium GPU verify not implemented yet. "
        "See PIPELINE.md - requires Layer D (sign/verify) completion."
    );
}

// ============================================================================
// Buffer Allocation Helpers
// ============================================================================

int alloc_keygen_buffers(KeygenJob& job) {
    const Params& p = get_params(job.mode);

    cudaError_t err;

    err = cudaMalloc(&job.d_public_keys, job.batch_size * p.pk_size);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&job.d_secret_keys, job.batch_size * p.sk_size);
    if (err != cudaSuccess) {
        cudaFree(job.d_public_keys);
        job.d_public_keys = nullptr;
        return err;
    }

    return cudaSuccess;
}

void free_keygen_buffers(KeygenJob& job) {
    if (job.d_public_keys) {
        cudaFree(job.d_public_keys);
        job.d_public_keys = nullptr;
    }
    if (job.d_secret_keys) {
        cudaFree(job.d_secret_keys);
        job.d_secret_keys = nullptr;
    }
}

int alloc_sign_buffers(SignJob& job) {
    const Params& p = get_params(job.mode);

    cudaError_t err = cudaMalloc(&job.d_signatures, job.batch_size * p.sig_size);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

void free_sign_buffers(SignJob& job) {
    if (job.d_signatures) {
        cudaFree(job.d_signatures);
        job.d_signatures = nullptr;
    }
}

int alloc_verify_buffers(VerifyJob& job) {
    cudaError_t err = cudaMalloc(&job.d_results, job.batch_size);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

void free_verify_buffers(VerifyJob& job) {
    if (job.d_results) {
        cudaFree(job.d_results);
        job.d_results = nullptr;
    }
}

} // namespace dilithium
} // namespace smoke
