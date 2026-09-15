// dilithium_keygen.cu - CUDA keygen core for Dilithium / ML-DSA
// Card 14: Compute t = A*s1 + s2 using NTT-domain matrix-vector multiplication
//
// This file implements:
//   - kernel_matvec_As1_plus_s2_ntt: NTT-domain matvec kernel
//   - compute_t_As1_plus_s2_ntt_gpu: Full t computation
//   - keygen_core_gpu: Orchestrate ExpandA + sampling + matvec

#include "../include/dilithium_keygen.h"
#include "../include/dilithium_ntt.cuh"
#include "../include/dilithium_expandA.h"
#include "../include/dilithium_sampling.h"
#include "../include/dilithium_decompose.h"
#include "../include/dilithium_pack.h"

#include <stdexcept>

namespace smoke {
namespace dilithium {

namespace {

constexpr uint32_t N = 256;
constexpr uint32_t Q = 8380417;

// =============================================================================
// Device Functions
// =============================================================================

// Modular addition: (a + b) mod q
__device__ __forceinline__
uint32_t add_mod_q_keygen(uint32_t a, uint32_t b) {
    uint32_t r = a + b;
    return (r >= Q) ? (r - Q) : r;
}

// Modular multiplication: (a * b) mod q
__device__ __forceinline__
uint32_t mul_mod_q_keygen(uint32_t a, uint32_t b) {
    uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    return static_cast<uint32_t>(prod % Q);
}

// =============================================================================
// Matvec Kernel: t_hat[i] = sum_j(A_hat[i,j] * s1_hat[j]) + s2_hat[i]
// =============================================================================

// NTT-domain matrix-vector multiplication with addition.
//
// Computes: t_hat[i, coeff] = (sum_{j=0}^{l-1} A_hat[i,j,coeff] * s1_hat[j,coeff]) + s2_hat[i,coeff]
//
// Grid config: gridDim.x = k (number of output polynomials / rows)
// Block config: blockDim.x = threads per row (covers n coefficients)
//
// Memory layout:
//   A_hat: [((i * l) + j) * n + coeff]  (k*l*n total)
//   s1_hat: [j * n + coeff]             (l*n total)
//   s2_hat: [i * n + coeff]             (k*n total)
//   t_hat: [i * n + coeff]              (k*n total)
//
__global__ void kernel_matvec_As1_plus_s2_ntt(
    uint32_t* __restrict__ d_t_hat,
    const uint32_t* __restrict__ d_A_hat,
    const uint32_t* __restrict__ d_s1_hat,
    const uint32_t* __restrict__ d_s2_hat,
    int k, int l, int n, uint32_t q)
{
    int row = blockIdx.x;  // i index (0..k-1)
    if (row >= k) return;

    // Each thread handles multiple coefficients (strided)
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int coeff = tid; coeff < n; coeff += stride) {
        // Accumulate sum_j(A_hat[i,j,coeff] * s1_hat[j,coeff])
        uint64_t acc = 0;

        for (int j = 0; j < l; ++j) {
            // A_hat[i,j,coeff] = d_A_hat[((i * l) + j) * n + coeff]
            std::size_t idx_A = (static_cast<std::size_t>(row) * l + j) * n + coeff;
            // s1_hat[j,coeff] = d_s1_hat[j * n + coeff]
            std::size_t idx_s1 = static_cast<std::size_t>(j) * n + coeff;

            uint32_t a_ij = d_A_hat[idx_A];
            uint32_t s1_j = d_s1_hat[idx_s1];

            acc += static_cast<uint64_t>(a_ij) * static_cast<uint64_t>(s1_j);
        }

        // Reduce acc mod q
        uint32_t sum = static_cast<uint32_t>(acc % q);

        // Add s2_hat[i,coeff]
        std::size_t idx_s2 = static_cast<std::size_t>(row) * n + coeff;
        uint32_t s2_i = d_s2_hat[idx_s2];

        uint32_t t_hat_val = sum + s2_i;
        if (t_hat_val >= q) t_hat_val -= q;

        // Write output
        std::size_t idx_t = static_cast<std::size_t>(row) * n + coeff;
        d_t_hat[idx_t] = t_hat_val;
    }
}

} // anonymous namespace

// =============================================================================
// Public API: compute_t_As1_plus_s2_ntt_gpu
// =============================================================================

void compute_t_As1_plus_s2_ntt_gpu(const Params& params,
                                   const uint32_t* d_A_hat,
                                   const uint32_t* d_s1,
                                   const uint32_t* d_s2,
                                   uint32_t* d_t,
                                   cudaStream_t stream)
{
    if (!d_A_hat || !d_s1 || !d_s2 || !d_t) {
        throw std::invalid_argument("compute_t_As1_plus_s2_ntt_gpu: null pointer");
    }

    const int k = params.k;
    const int l = params.l;
    const int n = params.n;
    const uint32_t q = params.q;

    // Buffer sizes
    std::size_t size_s1 = static_cast<std::size_t>(l) * n;
    std::size_t size_s2 = static_cast<std::size_t>(k) * n;

    // Allocate temporary buffers on device:
    // s1_hat (l * n), s2_hat (k * n), t_hat (k * n)
    uint32_t* d_s1_hat = nullptr;
    uint32_t* d_s2_hat = nullptr;
    uint32_t* d_t_hat  = nullptr;

    cudaError_t err;
    err = cudaMallocAsync(&d_s1_hat, size_s1 * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocAsync failed for d_s1_hat");
    }

    err = cudaMallocAsync(&d_s2_hat, size_s2 * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_s1_hat, stream);
        throw std::runtime_error("cudaMallocAsync failed for d_s2_hat");
    }

    err = cudaMallocAsync(&d_t_hat, size_s2 * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_s1_hat, stream);
        cudaFreeAsync(d_s2_hat, stream);
        throw std::runtime_error("cudaMallocAsync failed for d_t_hat");
    }

    // Copy s1, s2 into temporary buffers for in-place NTT
    cudaMemcpyAsync(d_s1_hat, d_s1, size_s1 * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_s2_hat, d_s2, size_s2 * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);

    // Forward NTT on s1_hat (l polynomials) and s2_hat (k polynomials)
    ntt_forward_batch(d_s1_hat, l, params, stream, NTTImpl::AUTO);
    ntt_forward_batch(d_s2_hat, k, params, stream, NTTImpl::AUTO);

    // Matvec kernel: t_hat = A_hat * s1_hat + s2_hat
    // 1 block per row, 256 threads (1 thread per coefficient)
    dim3 grid(k, 1, 1);
    dim3 block(256, 1, 1);  // n=256 threads to cover all coefficients

    kernel_matvec_As1_plus_s2_ntt<<<grid, block, 0, stream>>>(
        d_t_hat,
        d_A_hat,
        d_s1_hat,
        d_s2_hat,
        k, l, n, q);

    // Inverse NTT on t_hat (k polynomials) -> d_t (time domain)
    // First copy t_hat to d_t, then do in-place INTT
    cudaMemcpyAsync(d_t, d_t_hat, size_s2 * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);
    ntt_inverse_batch(d_t, k, params, stream, NTTImpl::AUTO);

    // Cleanup temporary buffers
    cudaFreeAsync(d_s1_hat, stream);
    cudaFreeAsync(d_s2_hat, stream);
    cudaFreeAsync(d_t_hat, stream);
}

// =============================================================================
// Public API: keygen_core_gpu
// =============================================================================

void keygen_core_gpu(const Params& params,
                     const uint8_t rho[32],
                     const uint8_t rhoprime[32],
                     uint32_t* d_A_hat,
                     uint32_t* d_s1,
                     uint32_t* d_s2,
                     uint32_t* d_t,
                     cudaStream_t stream)
{
    if (!d_A_hat || !d_s1 || !d_s2 || !d_t) {
        throw std::invalid_argument("keygen_core_gpu: null pointer");
    }

    // 1) Expand A from rho into NTT domain
    expand_A_ntt_gpu(params, rho, d_A_hat, stream);

    // 2) Sample s1 vector (l polynomials) in time domain
    sample_s1_gpu(params, rhoprime, d_s1, stream);

    // 3) Sample s2 vector (k polynomials) in time domain
    sample_s2_gpu(params, rhoprime, d_s2, stream);

    // 4) Compute t = A*s1 + s2 (NTT-domain matvec)
    compute_t_As1_plus_s2_ntt_gpu(params, d_A_hat, d_s1, d_s2, d_t, stream);
}

// =============================================================================
// Card 14.2: Full Keygen with GPU Power2Round and Packing
// =============================================================================

void keygen_full_gpu(const Params& params,
                     const uint8_t rho[32],
                     const uint8_t rhoprime[32],
                     uint8_t* d_pk,
                     uint8_t* d_sk,
                     cudaStream_t stream)
{
    if (!d_pk || !d_sk) {
        throw std::invalid_argument("keygen_full_gpu: null output pointer");
    }

    const int k = params.k;
    const int l = params.l;
    const int n = params.n;

    // Buffer sizes
    const std::size_t A_size = static_cast<std::size_t>(k) * l * n;
    const std::size_t s1_size = static_cast<std::size_t>(l) * n;
    const std::size_t s2_size = static_cast<std::size_t>(k) * n;
    const std::size_t t_size = static_cast<std::size_t>(k) * n;

    // Allocate device buffers for intermediates
    uint8_t* d_rho = nullptr;
    uint8_t* d_rhoprime = nullptr;
    uint32_t* d_A_hat = nullptr;
    uint32_t* d_s1 = nullptr;
    uint32_t* d_s2 = nullptr;
    uint32_t* d_t = nullptr;
    uint32_t* d_t1 = nullptr;
    uint32_t* d_t0 = nullptr;

    cudaError_t err;

    // Allocate seeds on device
    err = cudaMallocAsync(&d_rho, 32, stream);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed for d_rho");

    err = cudaMallocAsync(&d_rhoprime, 32, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        throw std::runtime_error("cudaMalloc failed for d_rhoprime");
    }

    // Allocate coefficient buffers
    err = cudaMallocAsync(&d_A_hat, A_size * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        throw std::runtime_error("cudaMalloc failed for d_A_hat");
    }

    err = cudaMallocAsync(&d_s1, s1_size * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        throw std::runtime_error("cudaMalloc failed for d_s1");
    }

    err = cudaMallocAsync(&d_s2, s2_size * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        cudaFreeAsync(d_s1, stream);
        throw std::runtime_error("cudaMalloc failed for d_s2");
    }

    err = cudaMallocAsync(&d_t, t_size * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        cudaFreeAsync(d_s1, stream);
        cudaFreeAsync(d_s2, stream);
        throw std::runtime_error("cudaMalloc failed for d_t");
    }

    err = cudaMallocAsync(&d_t1, t_size * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        cudaFreeAsync(d_s1, stream);
        cudaFreeAsync(d_s2, stream);
        cudaFreeAsync(d_t, stream);
        throw std::runtime_error("cudaMalloc failed for d_t1");
    }

    err = cudaMallocAsync(&d_t0, t_size * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        cudaFreeAsync(d_s1, stream);
        cudaFreeAsync(d_s2, stream);
        cudaFreeAsync(d_t, stream);
        cudaFreeAsync(d_t1, stream);
        throw std::runtime_error("cudaMalloc failed for d_t0");
    }

    // Copy seeds to device
    cudaMemcpyAsync(d_rho, rho, 32, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_rhoprime, rhoprime, 32, cudaMemcpyHostToDevice, stream);

    // 1) Keygen core: ExpandA + sampling + matvec
    keygen_core_gpu(params, rho, rhoprime, d_A_hat, d_s1, d_s2, d_t, stream);

    // 2) Power2Round: t -> (t1, t0) on GPU
    power2round_vec_gpu(params, d_t, d_t1, d_t0, stream);

    // 3) Pack pk and sk on GPU
    pack_pk_gpu(params, d_rho, d_t1, d_pk, stream);
    pack_sk_gpu(params, d_rho, d_rhoprime, d_s1, d_s2, d_t0, d_sk, stream);

    // Cleanup intermediates
    cudaFreeAsync(d_rho, stream);
    cudaFreeAsync(d_rhoprime, stream);
    cudaFreeAsync(d_A_hat, stream);
    cudaFreeAsync(d_s1, stream);
    cudaFreeAsync(d_s2, stream);
    cudaFreeAsync(d_t, stream);
    cudaFreeAsync(d_t1, stream);
    cudaFreeAsync(d_t0, stream);
}

// =============================================================================
// Card 14.2: Batched Keygen
// =============================================================================

void keygen_full_batch_gpu(const Params& params,
                           const uint8_t* h_rho,
                           const uint8_t* h_rhoprime,
                           uint8_t* d_pk,
                           uint8_t* d_sk,
                           int batch_size,
                           cudaStream_t stream)
{
    if (!h_rho || !h_rhoprime) {
        throw std::invalid_argument("keygen_full_batch_gpu: null seed pointer");
    }
    if (!d_pk || !d_sk) {
        throw std::invalid_argument("keygen_full_batch_gpu: null output pointer");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("keygen_full_batch_gpu: batch_size must be > 0");
    }

    const int k = params.k;
    const int l = params.l;
    const int n = params.n;
    const int B = batch_size;

    // Buffer sizes per item
    const std::size_t A_coeffs = static_cast<std::size_t>(k) * l * n;
    const std::size_t s1_coeffs = static_cast<std::size_t>(l) * n;
    const std::size_t s2_coeffs = static_cast<std::size_t>(k) * n;
    const std::size_t t_coeffs = static_cast<std::size_t>(k) * n;
    const std::size_t pk_size = pk_simple_size(params);
    const std::size_t sk_size = sk_simple_size(params);

    // Allocate batched device buffers
    uint8_t* d_rho = nullptr;
    uint8_t* d_rhoprime = nullptr;
    uint32_t* d_A_hat = nullptr;
    uint32_t* d_s1 = nullptr;
    uint32_t* d_s2 = nullptr;
    uint32_t* d_t = nullptr;
    uint32_t* d_t1 = nullptr;
    uint32_t* d_t0 = nullptr;

    cudaError_t err;

    err = cudaMallocAsync(&d_rho, static_cast<std::size_t>(B) * 32, stream);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed for d_rho batch");

    err = cudaMallocAsync(&d_rhoprime, static_cast<std::size_t>(B) * 32, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        throw std::runtime_error("cudaMalloc failed for d_rhoprime batch");
    }

    err = cudaMallocAsync(&d_A_hat, static_cast<std::size_t>(B) * A_coeffs * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        throw std::runtime_error("cudaMalloc failed for d_A_hat batch");
    }

    err = cudaMallocAsync(&d_s1, static_cast<std::size_t>(B) * s1_coeffs * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        throw std::runtime_error("cudaMalloc failed for d_s1 batch");
    }

    err = cudaMallocAsync(&d_s2, static_cast<std::size_t>(B) * s2_coeffs * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        cudaFreeAsync(d_s1, stream);
        throw std::runtime_error("cudaMalloc failed for d_s2 batch");
    }

    err = cudaMallocAsync(&d_t, static_cast<std::size_t>(B) * t_coeffs * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        cudaFreeAsync(d_s1, stream);
        cudaFreeAsync(d_s2, stream);
        throw std::runtime_error("cudaMalloc failed for d_t batch");
    }

    err = cudaMallocAsync(&d_t1, static_cast<std::size_t>(B) * t_coeffs * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        cudaFreeAsync(d_s1, stream);
        cudaFreeAsync(d_s2, stream);
        cudaFreeAsync(d_t, stream);
        throw std::runtime_error("cudaMalloc failed for d_t1 batch");
    }

    err = cudaMallocAsync(&d_t0, static_cast<std::size_t>(B) * t_coeffs * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_rho, stream);
        cudaFreeAsync(d_rhoprime, stream);
        cudaFreeAsync(d_A_hat, stream);
        cudaFreeAsync(d_s1, stream);
        cudaFreeAsync(d_s2, stream);
        cudaFreeAsync(d_t, stream);
        cudaFreeAsync(d_t1, stream);
        throw std::runtime_error("cudaMalloc failed for d_t0 batch");
    }

    // Copy all seeds to device
    cudaMemcpyAsync(d_rho, h_rho, static_cast<std::size_t>(B) * 32, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_rhoprime, h_rhoprime, static_cast<std::size_t>(B) * 32, cudaMemcpyHostToDevice, stream);

    // Process each item in batch
    // TODO: Future optimization - true batched kernels for ExpandA, sampling, matvec
    for (int b = 0; b < B; ++b) {
        const uint8_t* rho_b = h_rho + b * 32;
        const uint8_t* rhoprime_b = h_rhoprime + b * 32;
        uint32_t* d_A_hat_b = d_A_hat + b * A_coeffs;
        uint32_t* d_s1_b = d_s1 + b * s1_coeffs;
        uint32_t* d_s2_b = d_s2 + b * s2_coeffs;
        uint32_t* d_t_b = d_t + b * t_coeffs;
        uint32_t* d_t1_b = d_t1 + b * t_coeffs;
        uint32_t* d_t0_b = d_t0 + b * t_coeffs;

        // Keygen core for this item
        keygen_core_gpu(params, rho_b, rhoprime_b, d_A_hat_b, d_s1_b, d_s2_b, d_t_b, stream);

        // Power2Round for this item
        power2round_vec_gpu(params, d_t_b, d_t1_b, d_t0_b, stream);
    }

    // Batch pack all pk and sk
    pack_pk_batch_gpu(params, d_rho, d_t1, d_pk, B, stream);
    pack_sk_batch_gpu(params, d_rho, d_rhoprime, d_s1, d_s2, d_t0, d_sk, B, stream);

    // Cleanup
    cudaFreeAsync(d_rho, stream);
    cudaFreeAsync(d_rhoprime, stream);
    cudaFreeAsync(d_A_hat, stream);
    cudaFreeAsync(d_s1, stream);
    cudaFreeAsync(d_s2, stream);
    cudaFreeAsync(d_t, stream);
    cudaFreeAsync(d_t1, stream);
    cudaFreeAsync(d_t0, stream);
}

} // namespace dilithium
} // namespace smoke
