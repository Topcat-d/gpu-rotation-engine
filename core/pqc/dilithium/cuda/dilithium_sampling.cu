// dilithium_sampling.cu - CUDA-native poly_eta sampling for Dilithium
// Card 13.1: GPU secret sampling via SHAKE-256 + nibble extraction
//
// Implements poly_eta sampling entirely on GPU:
// 1. SHAKE-256 XOF: rhoprime(32) || nonce(2) -> XOF bytes
// 2. Nibble extraction: 2 coefficients per byte
// 3. Rejection: accept v <= 2*eta
// 4. Encoding: [-eta, eta] -> [0, q)
//
// This file provides:
// - Device function: poly_eta_sample_device
// - Kernel: kernel_poly_eta_from_xof
// - Host helper: sample_poly_eta_gpu_cuda

#include "../include/dilithium_sampling.h"
#include "../include/dilithium_shake.h"
#include "../include/dilithium_params.h"

#include <cstdio>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace smoke {
namespace dilithium {

namespace {

// ============================================================================
// Device Nibble Sampler
// ============================================================================

// Must match Python dilithium_sampling_ref.py exactly!
// - Low nibble first, then high nibble
// - Accept v <= 2*eta, reject otherwise
// - Map [0, 2*eta] -> [-eta, eta]
// - Encode: negative -> q + coeff

__device__ void poly_eta_sample_device(uint32_t* poly,
                                       const uint8_t* xof_bytes,
                                       std::size_t xof_len,
                                       uint32_t q,
                                       uint32_t eta,
                                       std::size_t n) {
    std::size_t count = 0;
    std::size_t pos = 0;

    const uint32_t max_val = 2 * eta;  // Maximum acceptable value

    while (count < n && pos < xof_len) {
        uint8_t byte = xof_bytes[pos++];

        // Low nibble first
        uint32_t v0 = byte & 0x0F;
        if (v0 <= max_val && count < n) {
            int32_t coeff = static_cast<int32_t>(v0) - static_cast<int32_t>(eta);
            uint32_t encoded;
            if (coeff >= 0) {
                encoded = static_cast<uint32_t>(coeff);
            } else {
                // coeff is negative: encode as q + coeff
                encoded = static_cast<uint32_t>(static_cast<int64_t>(q) + coeff);
            }
            poly[count++] = encoded;
        }

        // High nibble second
        uint32_t v1 = (byte >> 4) & 0x0F;
        if (v1 <= max_val && count < n) {
            int32_t coeff = static_cast<int32_t>(v1) - static_cast<int32_t>(eta);
            uint32_t encoded;
            if (coeff >= 0) {
                encoded = static_cast<uint32_t>(coeff);
            } else {
                encoded = static_cast<uint32_t>(static_cast<int64_t>(q) + coeff);
            }
            poly[count++] = encoded;
        }
    }

    // Zero-pad if needed (shouldn't happen with sufficient XOF output)
    while (count < n) {
        poly[count++] = 0;
    }
}

// ============================================================================
// Kernel: Convert XOF bytes to poly_eta
// ============================================================================

// One block per polynomial, single thread per block (sampling is sequential)
__global__ void kernel_poly_eta_from_xof(
    uint32_t* __restrict__ d_polys,      // Output: polynomials [num_polys * n]
    const uint8_t* __restrict__ d_xof,   // Input: XOF bytes [num_polys * xof_len]
    std::size_t num_polys,
    std::size_t n,
    std::size_t xof_len,
    uint32_t q,
    uint32_t eta
) {
    std::size_t poly_idx = blockIdx.x;
    if (poly_idx >= num_polys) return;

    // Only thread 0 does the work (sequential sampling)
    if (threadIdx.x != 0) return;

    // Point to this polynomial's output and XOF segment
    uint32_t* poly = d_polys + poly_idx * n;
    const uint8_t* xof = d_xof + poly_idx * xof_len;

    poly_eta_sample_device(poly, xof, xof_len, q, eta, n);
}

} // anonymous namespace

// ============================================================================
// Host API: CUDA-native poly_eta sampling
// ============================================================================

// Internal helper: Run SHAKE-256 XOF on GPU for a single (rhoprime, nonce)
static void shake256_secret_gpu(const uint8_t rhoprime[32],
                                uint16_t nonce,
                                uint8_t* d_xof,
                                std::size_t xof_len,
                                cudaStream_t stream) {
    // Build input: rhoprime(32) || nonce_le(2) = 34 bytes
    constexpr std::size_t in_len = 34;
    uint8_t h_in[in_len];
    std::memcpy(h_in, rhoprime, 32);
    h_in[32] = static_cast<uint8_t>(nonce & 0xFF);
    h_in[33] = static_cast<uint8_t>(nonce >> 8);

    // Copy input to device
    uint8_t* d_in = nullptr;
    cudaMalloc(&d_in, in_len);
    cudaMemcpyAsync(d_in, h_in, in_len, cudaMemcpyHostToDevice, stream);

    // Run SHAKE-256 via batch API (batch of 1)
    SHAKEJob job;
    job.d_in = d_in;
    job.in_len = in_len;
    job.d_out = d_xof;
    job.out_len = xof_len;

    shake_xof_batch(SHAKEKind::SHAKE256, &job, 1, stream);

    cudaFree(d_in);
}

// Internal helper: Run SHAKE-256 XOF for multiple polynomials at once
static void shake256_secrets_batch_gpu(const uint8_t rhoprime[32],
                                       const uint16_t* nonces,
                                       std::size_t num_polys,
                                       uint8_t* d_xof_all,
                                       std::size_t xof_len_per_poly,
                                       cudaStream_t stream) {
    constexpr std::size_t in_len = 34;

    // Prepare all inputs on host
    std::vector<uint8_t> h_inputs(num_polys * in_len);
    for (std::size_t i = 0; i < num_polys; ++i) {
        uint8_t* inp = h_inputs.data() + i * in_len;
        std::memcpy(inp, rhoprime, 32);
        inp[32] = static_cast<uint8_t>(nonces[i] & 0xFF);
        inp[33] = static_cast<uint8_t>(nonces[i] >> 8);
    }

    // Copy all inputs to device (contiguous)
    uint8_t* d_inputs_flat = nullptr;
    cudaMalloc(&d_inputs_flat, num_polys * in_len);
    cudaMemcpyAsync(d_inputs_flat, h_inputs.data(), num_polys * in_len,
                    cudaMemcpyHostToDevice, stream);

    // Build job array
    std::vector<SHAKEJob> jobs(num_polys);
    for (std::size_t i = 0; i < num_polys; ++i) {
        jobs[i].d_in = d_inputs_flat + i * in_len;
        jobs[i].in_len = in_len;
        jobs[i].d_out = d_xof_all + i * xof_len_per_poly;
        jobs[i].out_len = xof_len_per_poly;
    }

    // Run batched SHAKE-256
    shake_xof_batch(SHAKEKind::SHAKE256, jobs.data(), num_polys, stream);

    cudaFree(d_inputs_flat);
}

// ============================================================================
// Public API: sample_poly_eta_gpu (CUDA-native)
// ============================================================================

void sample_poly_eta_gpu(const Params& params,
                         const uint8_t rhoprime[32],
                         uint16_t nonce,
                         uint32_t* d_poly,
                         cudaStream_t stream) {
    if (!d_poly) {
        throw std::invalid_argument("sample_poly_eta_gpu: d_poly is null");
    }

    const std::size_t n = params.n;
    const uint32_t q = params.q;
    const uint32_t eta = params.eta;

    // XOF length: need enough bytes for n coefficients with rejections
    // Each byte gives 2 nibbles; worst case all reject (max_val = 2*eta)
    // For eta=2: accept rate ~31% (5/16), for eta=4: ~56% (9/16)
    // Use generous buffer: 4 * n bytes (can sample 8*n nibbles)
    const std::size_t xof_len = 4 * n;

    // Allocate XOF buffer on device
    uint8_t* d_xof = nullptr;
    cudaMalloc(&d_xof, xof_len);

    // 1) Run SHAKE-256 on GPU
    shake256_secret_gpu(rhoprime, nonce, d_xof, xof_len, stream);

    // 2) Launch kernel to convert XOF -> poly_eta
    dim3 grid(1);   // 1 polynomial
    dim3 block(1);  // 1 thread (sequential sampling)
    kernel_poly_eta_from_xof<<<grid, block, 0, stream>>>(
        d_poly, d_xof, 1, n, xof_len, q, eta);

    // 3) Cleanup
    cudaStreamSynchronize(stream);
    cudaFree(d_xof);
}

// ============================================================================
// Public API: sample_s1_gpu / sample_s2_gpu (CUDA-native, batched)
// ============================================================================

void sample_s1_gpu(const Params& params,
                   const uint8_t rhoprime[32],
                   uint32_t* d_s1,
                   cudaStream_t stream) {
    if (!d_s1) {
        throw std::invalid_argument("sample_s1_gpu: d_s1 is null");
    }

    const std::size_t n = params.n;
    const std::size_t l = params.l;
    const uint32_t q = params.q;
    const uint32_t eta = params.eta;

    const std::size_t xof_len = 4 * n;  // Per polynomial

    // Build nonces for s1: 0, 1, ..., l-1
    std::vector<uint16_t> nonces(l);
    for (std::size_t i = 0; i < l; ++i) {
        nonces[i] = static_cast<uint16_t>(i);
    }

    // Allocate XOF buffer for all l polynomials
    uint8_t* d_xof_all = nullptr;
    cudaMalloc(&d_xof_all, l * xof_len);

    // 1) Batched SHAKE-256 for all l nonces
    shake256_secrets_batch_gpu(rhoprime, nonces.data(), l, d_xof_all, xof_len, stream);

    // 2) Launch kernel: 1 block per polynomial
    dim3 grid(static_cast<unsigned int>(l));
    dim3 block(1);
    kernel_poly_eta_from_xof<<<grid, block, 0, stream>>>(
        d_s1, d_xof_all, l, n, xof_len, q, eta);

    // 3) Cleanup
    cudaStreamSynchronize(stream);
    cudaFree(d_xof_all);
}

void sample_s2_gpu(const Params& params,
                   const uint8_t rhoprime[32],
                   uint32_t* d_s2,
                   cudaStream_t stream) {
    if (!d_s2) {
        throw std::invalid_argument("sample_s2_gpu: d_s2 is null");
    }

    const std::size_t n = params.n;
    const std::size_t k = params.k;
    const std::size_t l = params.l;
    const uint32_t q = params.q;
    const uint32_t eta = params.eta;

    const std::size_t xof_len = 4 * n;  // Per polynomial

    // Build nonces for s2: l, l+1, ..., l+k-1
    std::vector<uint16_t> nonces(k);
    for (std::size_t i = 0; i < k; ++i) {
        nonces[i] = static_cast<uint16_t>(l + i);
    }

    // Allocate XOF buffer for all k polynomials
    uint8_t* d_xof_all = nullptr;
    cudaMalloc(&d_xof_all, k * xof_len);

    // 1) Batched SHAKE-256 for all k nonces
    shake256_secrets_batch_gpu(rhoprime, nonces.data(), k, d_xof_all, xof_len, stream);

    // 2) Launch kernel: 1 block per polynomial
    dim3 grid(static_cast<unsigned int>(k));
    dim3 block(1);
    kernel_poly_eta_from_xof<<<grid, block, 0, stream>>>(
        d_s2, d_xof_all, k, n, xof_len, q, eta);

    // 3) Cleanup
    cudaStreamSynchronize(stream);
    cudaFree(d_xof_all);
}

} // namespace dilithium
} // namespace smoke
