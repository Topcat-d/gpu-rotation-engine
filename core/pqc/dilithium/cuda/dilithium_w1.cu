// dilithium_w1.cu - CUDA w = A*y and w1 = highbits(w) computation
// Card 16.1: Real w and w1 for Dilithium signing
//
// This file implements:
//   - compute_w_gpu: w = A * y via NTT-domain multiplication
//   - decompose_w_gpu: HighBits/LowBits decomposition per FIPS 204
//
// FIPS 204 Section 8.4: HighBits and LowBits decomposition
// For w decomposition, alpha = 2 * gamma2

#include "../include/dilithium_w1.h"
#include "../include/dilithium_ntt.cuh"

#include <stdexcept>

namespace smoke {
namespace dilithium {

namespace {

constexpr uint32_t N = 256;
constexpr uint32_t Q = 8380417;

// =============================================================================
// Device Functions
// =============================================================================

// Modular multiplication: (a * b) mod q
__device__ __forceinline__
uint32_t mul_mod_q_w1(uint32_t a, uint32_t b) {
    uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    return static_cast<uint32_t>(prod % Q);
}

// =============================================================================
// Matvec Kernel: w_hat[i] = sum_j(A_hat[i,j] * y_hat[j])
// =============================================================================
//
// This is the same as keygen matvec but WITHOUT the s2 addition.
//
// Grid config: gridDim.x = k (number of output polynomials / rows)
// Block config: blockDim.x = threads per row (covers n coefficients)
//
// Memory layout:
//   A_hat: [((i * l) + j) * n + coeff]  (k*l*n total)
//   y_hat: [j * n + coeff]              (l*n total)
//   w_hat: [i * n + coeff]              (k*n total)
//
__global__ void kernel_matvec_Ay_ntt(
    uint32_t* __restrict__ d_w_hat,
    const uint32_t* __restrict__ d_A_hat,
    const uint32_t* __restrict__ d_y_hat,
    int k, int l, int n, uint32_t q)
{
    int row = blockIdx.x;  // i index (0..k-1)
    if (row >= k) return;

    // Each thread handles multiple coefficients (strided)
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int coeff = tid; coeff < n; coeff += stride) {
        // Accumulate sum_j(A_hat[i,j,coeff] * y_hat[j,coeff])
        uint64_t acc = 0;

        for (int j = 0; j < l; ++j) {
            // A_hat[i,j,coeff] = d_A_hat[((i * l) + j) * n + coeff]
            std::size_t idx_A = (static_cast<std::size_t>(row) * l + j) * n + coeff;
            // y_hat[j,coeff] = d_y_hat[j * n + coeff]
            std::size_t idx_y = static_cast<std::size_t>(j) * n + coeff;

            uint32_t a_ij = d_A_hat[idx_A];
            uint32_t y_j = d_y_hat[idx_y];

            acc += static_cast<uint64_t>(a_ij) * static_cast<uint64_t>(y_j);
        }

        // Reduce acc mod q
        uint32_t w_hat_val = static_cast<uint32_t>(acc % q);

        // Write output
        std::size_t idx_w = static_cast<std::size_t>(row) * n + coeff;
        d_w_hat[idx_w] = w_hat_val;
    }
}

// =============================================================================
// HighBits/LowBits Decomposition Kernel
// =============================================================================
//
// FIPS 204 Algorithm 35 (Decompose) / Algorithm 36 (HighBits) / Algorithm 37 (LowBits):
//
//   alpha = 2 * gamma2
//   r' = r mod q (input already in [0, q))
//   r0 = r' mod+- alpha  (centered modulo in (-alpha/2, alpha/2])
//   if r' - r0 == q - 1:
//       r1 = 0, r0 = r0 - 1
//   else:
//       r1 = (r' - r0) / alpha
//
// Note: We encode r0 (which can be negative) in [0, q) for storage.
//
__device__ __forceinline__
void decompose_coeff_device(uint32_t r, uint32_t q, uint32_t gamma2,
                            uint32_t& r1_out, uint32_t& r0_out)
{
    // alpha = 2 * gamma2
    uint32_t alpha = 2 * gamma2;

    // r' = r mod q (input should already be in [0, q))
    // But ensure it for safety
    r = r % q;

    // r0 = r mod+- alpha (centered modulo)
    // First compute r mod alpha
    uint32_t r0_pos = r % alpha;

    // Convert to centered form: (-alpha/2, alpha/2]
    // If r0_pos > alpha/2, subtract alpha
    int32_t r0_centered;
    if (r0_pos > (alpha / 2)) {
        r0_centered = static_cast<int32_t>(r0_pos) - static_cast<int32_t>(alpha);
    } else {
        r0_centered = static_cast<int32_t>(r0_pos);
    }

    // Check special case: r - r0 == q - 1
    // Compute (r - r0) in full precision
    int64_t r_minus_r0 = static_cast<int64_t>(r) - static_cast<int64_t>(r0_centered);

    uint32_t r1;
    int32_t r0_final;

    if (r_minus_r0 == static_cast<int64_t>(q) - 1) {
        // Special case: set r1 = 0, r0 = r0 - 1
        r1 = 0;
        r0_final = r0_centered - 1;
    } else {
        // Normal case: r1 = (r - r0) / alpha
        r1 = static_cast<uint32_t>(r_minus_r0 / static_cast<int64_t>(alpha));
        r0_final = r0_centered;
    }

    // Encode r0 in [0, q): negative values become q + r0
    uint32_t r0_enc;
    if (r0_final >= 0) {
        r0_enc = static_cast<uint32_t>(r0_final);
    } else {
        r0_enc = static_cast<uint32_t>(static_cast<int64_t>(q) + r0_final);
    }

    r1_out = r1;
    r0_out = r0_enc;
}

__global__ void kernel_decompose_w_vec(
    const uint32_t* __restrict__ d_w,
    uint32_t* __restrict__ d_w1,
    uint32_t* __restrict__ d_w0,
    std::size_t coeff_count,
    uint32_t q,
    uint32_t gamma2)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coeff_count) return;

    uint32_t w = d_w[idx];
    uint32_t w1, w0;
    decompose_coeff_device(w, q, gamma2, w1, w0);

    d_w1[idx] = w1;
    d_w0[idx] = w0;
}

} // anonymous namespace

// =============================================================================
// Public API: compute_w_gpu
// =============================================================================

void compute_w_gpu(const Params& params,
                   const uint32_t* d_A_hat,
                   const uint32_t* d_y,
                   uint32_t* d_w,
                   cudaStream_t stream)
{
    if (!d_A_hat || !d_y || !d_w) {
        throw std::invalid_argument("compute_w_gpu: null pointer");
    }

    const int k = params.k;
    const int l = params.l;
    const int n = params.n;
    const uint32_t q = params.q;

    // Buffer sizes
    std::size_t size_y = static_cast<std::size_t>(l) * n;
    std::size_t size_w = static_cast<std::size_t>(k) * n;

    // Allocate temporary buffers on device:
    // y_hat (l * n), w_hat (k * n)
    uint32_t* d_y_hat = nullptr;
    uint32_t* d_w_hat = nullptr;

    cudaError_t err;
    err = cudaMallocAsync(&d_y_hat, size_y * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocAsync failed for d_y_hat");
    }

    err = cudaMallocAsync(&d_w_hat, size_w * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_y_hat, stream);
        throw std::runtime_error("cudaMallocAsync failed for d_w_hat");
    }

    // Copy y into temporary buffer for in-place NTT
    cudaMemcpyAsync(d_y_hat, d_y, size_y * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);

    // Forward NTT on y_hat (l polynomials)
    ntt_forward_batch(d_y_hat, l, params, stream, NTTImpl::AUTO);

    // Matvec kernel: w_hat = A_hat * y_hat
    // 1 block per row, 256 threads (1 thread per coefficient)
    dim3 grid(k, 1, 1);
    dim3 block(256, 1, 1);  // n=256 threads to cover all coefficients

    kernel_matvec_Ay_ntt<<<grid, block, 0, stream>>>(
        d_w_hat,
        d_A_hat,
        d_y_hat,
        k, l, n, q);

    // Inverse NTT on w_hat (k polynomials) -> d_w (time domain)
    // First copy w_hat to d_w, then do in-place INTT
    cudaMemcpyAsync(d_w, d_w_hat, size_w * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);
    ntt_inverse_batch(d_w, k, params, stream, NTTImpl::AUTO);

    // Cleanup temporary buffers
    cudaFreeAsync(d_y_hat, stream);
    cudaFreeAsync(d_w_hat, stream);
}

// =============================================================================
// Public API: decompose_w_gpu
// =============================================================================

void decompose_w_gpu(const Params& params,
                     const uint32_t* d_w,
                     uint32_t* d_w1,
                     uint32_t* d_w0,
                     cudaStream_t stream)
{
    if (!d_w || !d_w1 || !d_w0) {
        throw std::invalid_argument("decompose_w_gpu: null pointer");
    }

    std::size_t coeff_count = w_vec_coeff_count(params);
    uint32_t q = params.q;
    uint32_t gamma2 = params.gamma2;

    const int threads = 256;
    const int blocks = static_cast<int>((coeff_count + threads - 1) / threads);

    kernel_decompose_w_vec<<<blocks, threads, 0, stream>>>(
        d_w, d_w1, d_w0, coeff_count, q, gamma2);
}

// =============================================================================
// CPU Reference Functions (for testing)
// =============================================================================

void decompose_coeff_cpu(uint32_t r, uint32_t q, uint32_t gamma2,
                         uint32_t& w1_out, uint32_t& w0_out)
{
    // alpha = 2 * gamma2
    uint32_t alpha = 2 * gamma2;

    // r' = r mod q
    r = r % q;

    // r0 = r mod+- alpha (centered modulo)
    uint32_t r0_pos = r % alpha;

    // Convert to centered form: (-alpha/2, alpha/2]
    int32_t r0_centered;
    if (r0_pos > (alpha / 2)) {
        r0_centered = static_cast<int32_t>(r0_pos) - static_cast<int32_t>(alpha);
    } else {
        r0_centered = static_cast<int32_t>(r0_pos);
    }

    // Check special case: r - r0 == q - 1
    int64_t r_minus_r0 = static_cast<int64_t>(r) - static_cast<int64_t>(r0_centered);

    uint32_t r1;
    int32_t r0_final;

    if (r_minus_r0 == static_cast<int64_t>(q) - 1) {
        r1 = 0;
        r0_final = r0_centered - 1;
    } else {
        r1 = static_cast<uint32_t>(r_minus_r0 / static_cast<int64_t>(alpha));
        r0_final = r0_centered;
    }

    // Encode r0 in [0, q)
    uint32_t r0_enc;
    if (r0_final >= 0) {
        r0_enc = static_cast<uint32_t>(r0_final);
    } else {
        r0_enc = static_cast<uint32_t>(static_cast<int64_t>(q) + r0_final);
    }

    w1_out = r1;
    w0_out = r0_enc;
}

bool verify_w_recomposition(uint32_t w, uint32_t w1, uint32_t w0_enc,
                            uint32_t q, uint32_t gamma2)
{
    // alpha = 2 * gamma2
    uint32_t alpha = 2 * gamma2;

    // Decode w0 from [0, q) encoding
    // If w0_enc > q/2, it's negative
    int64_t w0_decoded;
    if (w0_enc > q / 2) {
        w0_decoded = static_cast<int64_t>(w0_enc) - static_cast<int64_t>(q);
    } else {
        w0_decoded = static_cast<int64_t>(w0_enc);
    }

    // Recompose: w' = w1 * alpha + w0
    int64_t w_reconstructed = static_cast<int64_t>(w1) * static_cast<int64_t>(alpha) + w0_decoded;

    // Reduce mod q
    w_reconstructed = ((w_reconstructed % static_cast<int64_t>(q)) + q) % q;

    // Compare with original
    uint32_t w_orig = w % q;

    return (static_cast<uint32_t>(w_reconstructed) == w_orig);
}

} // namespace dilithium
} // namespace smoke
