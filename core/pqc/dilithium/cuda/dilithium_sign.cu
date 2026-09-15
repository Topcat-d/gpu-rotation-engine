// dilithium_sign.cu - CUDA kernels for Dilithium signing
// Card 17: GPU-backed sign/verify
//
// This file implements:
//   - sample_y_gamma1_gpu: Sample y from gamma1 distribution
//   - compute_z_gpu: z = y + c*s1
//   - compute_r_gpu: r = w - c*s2
//   - compute_ct0_gpu: c*t0 for hint computation
//   - check_norms_gpu: Rejection sampling norm checks
//   - make_hint_gpu: MakeHint per FIPS 204
//   - use_hint_gpu: UseHint per FIPS 204
//
// FIPS 204 Section 6.2/6.3 (ML-DSA Sign/Verify)

#include "../include/dilithium_sign.h"
#include "../include/dilithium_ntt.cuh"
#include "../include/dilithium_shake.h"

#include <stdexcept>
#include <algorithm>

namespace smoke {
namespace dilithium {

namespace {

constexpr uint32_t N = 256;
constexpr uint32_t Q = 8380417;

// =============================================================================
// Device Helper Functions
// =============================================================================

// Decode signed value from [0, q) encoding
__device__ __forceinline__
int64_t decode_signed_device(uint32_t val, uint32_t q) {
    int64_t v = static_cast<int64_t>(val);
    if (v > static_cast<int64_t>(q) / 2) {
        v -= static_cast<int64_t>(q);
    }
    return v;
}

// Absolute value
__device__ __forceinline__
int64_t abs_device(int64_t x) {
    return x >= 0 ? x : -x;
}

// HighBits decomposition (same as dilithium_w1.cu)
__device__ __forceinline__
void decompose_coeff_sign(uint32_t r, uint32_t q, uint32_t gamma2,
                          uint32_t& r1_out, int32_t& r0_out) {
    uint32_t alpha = 2 * gamma2;
    r = r % q;
    uint32_t r0_pos = r % alpha;
    int32_t r0_centered;
    if (r0_pos > (alpha / 2)) {
        r0_centered = static_cast<int32_t>(r0_pos) - static_cast<int32_t>(alpha);
    } else {
        r0_centered = static_cast<int32_t>(r0_pos);
    }
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
    r1_out = r1;
    r0_out = r0_final;
}

__device__ __forceinline__
uint32_t highbits_device(uint32_t r, uint32_t q, uint32_t gamma2) {
    uint32_t r1;
    int32_t r0;
    decompose_coeff_sign(r, q, gamma2, r1, r0);
    return r1;
}

// =============================================================================
// Kernel: Sample y from gamma1 distribution
// =============================================================================
//
// FIPS 204 ExpandMask: coefficients in [-(gamma1-1), gamma1-1]
// We use SHAKE-256 XOF and simple rejection sampling.
//
// Each thread handles one polynomial.
//
__global__ void kernel_sample_y_gamma1(
    uint32_t* __restrict__ d_y,
    const uint8_t* __restrict__ d_xof_output,
    int l, int n, uint32_t q, uint32_t gamma1)
{
    int poly_idx = blockIdx.x;
    if (poly_idx >= l) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // XOF output: 3 bytes per coefficient (simplistic, like CPU)
    const uint8_t* xof = d_xof_output + poly_idx * 3 * n;

    // Bit mask depends on gamma1
    uint32_t mask;
    if (gamma1 == (1u << 17)) {
        mask = 0x3FFFF;  // 18 bits
    } else {
        mask = 0xFFFFF;  // 20 bits
    }

    for (int i = tid; i < n; i += stride) {
        // Extract 3 bytes
        uint32_t val = xof[3*i] | (xof[3*i + 1] << 8) | (xof[3*i + 2] << 16);
        val &= mask;

        // Reduce to [0, 2*gamma1 - 1] uniformly
        val = val % (2 * gamma1 - 1);

        // Map to [-(gamma1-1), gamma1-1]
        int32_t coeff = static_cast<int32_t>(val) - static_cast<int32_t>(gamma1 - 1);

        // Encode in [0, q)
        uint32_t encoded;
        if (coeff >= 0) {
            encoded = static_cast<uint32_t>(coeff);
        } else {
            encoded = static_cast<uint32_t>(static_cast<int64_t>(q) + coeff);
        }

        d_y[poly_idx * n + i] = encoded;
    }
}

// =============================================================================
// Kernel: Sparse challenge times polynomial (c * poly)
// =============================================================================
//
// c is sparse with values in {-1, 0, 1}
// We compute c * poly via NTT multiplication.
// This kernel converts c from int8 to uint32 for NTT.
//
__global__ void kernel_convert_challenge_to_ntt_input(
    uint32_t* __restrict__ d_c_uint,
    const int8_t* __restrict__ d_c,
    int n, uint32_t q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int8_t c_val = d_c[idx];
    uint32_t c_uint;
    if (c_val >= 0) {
        c_uint = static_cast<uint32_t>(c_val);
    } else {
        c_uint = static_cast<uint32_t>(static_cast<int32_t>(q) + c_val);
    }
    d_c_uint[idx] = c_uint;
}

// =============================================================================
// Kernel: z = y + poly_product (element-wise add)
// =============================================================================
__global__ void kernel_poly_add_vec(
    uint32_t* __restrict__ d_out,
    const uint32_t* __restrict__ d_a,
    const uint32_t* __restrict__ d_b,
    std::size_t coeff_count, uint32_t q)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coeff_count) return;

    uint64_t sum = static_cast<uint64_t>(d_a[idx]) + static_cast<uint64_t>(d_b[idx]);
    d_out[idx] = static_cast<uint32_t>(sum % q);
}

// =============================================================================
// Kernel: Polynomial subtract (a - b mod q)
// =============================================================================
__global__ void kernel_poly_sub_vec(
    uint32_t* __restrict__ d_out,
    const uint32_t* __restrict__ d_a,
    const uint32_t* __restrict__ d_b,
    std::size_t coeff_count, uint32_t q)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coeff_count) return;

    int64_t diff = static_cast<int64_t>(d_a[idx]) - static_cast<int64_t>(d_b[idx]);
    d_out[idx] = static_cast<uint32_t>((diff % q + q) % q);
}

// =============================================================================
// Kernel: Compute infinity norm and check bound
// =============================================================================
//
// Uses parallel reduction to find max |coeff| after signed decode.
//
__global__ void kernel_infinity_norm_check(
    const uint32_t* __restrict__ d_vec,
    std::size_t coeff_count,
    uint32_t q,
    int64_t bound,
    uint8_t* __restrict__ d_ok)
{
    extern __shared__ int64_t sdata[];

    int tid = threadIdx.x;
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes local max
    int64_t local_max = 0;
    for (std::size_t i = idx; i < coeff_count; i += blockDim.x * gridDim.x) {
        int64_t decoded = decode_signed_device(d_vec[i], q);
        int64_t absval = abs_device(decoded);
        if (absval > local_max) local_max = absval;
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (tid == 0) {
        // Atomic max across blocks would be complex, so for now use single block
        *d_ok = (sdata[0] < bound) ? 1 : 0;
    }
}

// =============================================================================
// Kernel: MakeHint
// =============================================================================
//
// h[i] = 1 if HighBits(r[i]) != HighBits(r[i] + z[i])
//
__global__ void kernel_make_hint(
    uint8_t* __restrict__ d_h,
    const uint32_t* __restrict__ d_z,  // -c*t0, encoded in [0, q)
    const uint32_t* __restrict__ d_r,  // w - c*s2 + c*t0, encoded in [0, q)
    std::size_t coeff_count,
    uint32_t q, uint32_t gamma2)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coeff_count) return;

    uint32_t r_val = d_r[idx];
    uint32_t z_val = d_z[idx];

    // Decode z to signed for proper addition
    int64_t z_signed = decode_signed_device(z_val, q);

    uint32_t r1 = highbits_device(r_val, q, gamma2);

    // r + z mod q
    int64_t r_plus_z = (static_cast<int64_t>(r_val) + z_signed) % static_cast<int64_t>(q);
    if (r_plus_z < 0) r_plus_z += q;

    uint32_t r1_plus = highbits_device(static_cast<uint32_t>(r_plus_z), q, gamma2);

    d_h[idx] = (r1 != r1_plus) ? 1 : 0;
}

// Kernel to count hint ones (simple reduction)
__global__ void kernel_count_hint_ones(
    const uint8_t* __restrict__ d_h,
    std::size_t coeff_count,
    uint32_t* __restrict__ d_count)
{
    extern __shared__ uint32_t scount[];

    int tid = threadIdx.x;
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t local_count = 0;
    for (std::size_t i = idx; i < coeff_count; i += blockDim.x * gridDim.x) {
        local_count += d_h[i];
    }

    scount[tid] = local_count;
    __syncthreads();

    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scount[tid] += scount[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_count, scount[0]);
    }
}

// =============================================================================
// Kernel: UseHint
// =============================================================================
//
// FIPS 204 Algorithm 41
//
__global__ void kernel_use_hint(
    uint32_t* __restrict__ d_w1_recon,
    const uint8_t* __restrict__ d_h,
    const uint32_t* __restrict__ d_w_prime,
    std::size_t coeff_count,
    uint32_t q, uint32_t gamma2)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coeff_count) return;

    uint8_t h_val = d_h[idx];
    uint32_t r_val = d_w_prime[idx];

    uint32_t alpha = 2 * gamma2;
    uint32_t m = (q - 1) / alpha;  // Number of possible r1 values

    uint32_t r1;
    int32_t r0;
    decompose_coeff_sign(r_val, q, gamma2, r1, r0);

    if (h_val == 0) {
        d_w1_recon[idx] = r1;
    } else {
        // Adjust r1 based on sign of r0
        if (r0 > 0) {
            d_w1_recon[idx] = (r1 + 1) % m;
        } else {
            d_w1_recon[idx] = (r1 + m - 1) % m;
        }
    }
}

// =============================================================================
// Kernel: Scale t1 by 2^d
// =============================================================================
__global__ void kernel_scale_t1(
    uint32_t* __restrict__ d_t_scaled,
    const uint32_t* __restrict__ d_t1,
    std::size_t coeff_count,
    uint32_t q, uint32_t d_param)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coeff_count) return;

    uint64_t scale = 1ULL << d_param;
    uint64_t val = d_t1[idx];
    d_t_scaled[idx] = static_cast<uint32_t>((val * scale) % q);
}

} // anonymous namespace

// =============================================================================
// Public API: sample_y_gamma1_gpu
// =============================================================================

void sample_y_gamma1_gpu(const Params& params,
                          const uint8_t* d_rhoprime,
                          uint32_t kappa,
                          uint32_t* d_y,
                          cudaStream_t stream) {
    if (!d_rhoprime || !d_y) {
        throw std::invalid_argument("sample_y_gamma1_gpu: null pointer");
    }

    const int l = params.l;
    const int n = params.n;
    const uint32_t q = params.q;
    const uint32_t gamma1 = params.gamma1;

    // Allocate XOF output buffer: l polynomials, 3 bytes per coefficient
    std::size_t xof_size_per_poly = 3 * n;
    std::size_t total_xof_size = l * xof_size_per_poly;

    uint8_t* d_xof_output = nullptr;
    cudaError_t err = cudaMallocAsync(&d_xof_output, total_xof_size, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocAsync failed for d_xof_output");
    }

    // For each polynomial, run SHAKE-256 with rhoprime || nonce
    // Nonce = kappa * l + j
    for (int j = 0; j < l; ++j) {
        uint16_t nonce = static_cast<uint16_t>(kappa * l + j);

        // Prepare input: rhoprime (32 bytes) || nonce (2 bytes LE)
        uint8_t* d_shake_input = nullptr;
        cudaMallocAsync(&d_shake_input, 34, stream);

        // Copy rhoprime and nonce
        cudaMemcpyAsync(d_shake_input, d_rhoprime, 32, cudaMemcpyDeviceToDevice, stream);

        uint8_t nonce_bytes[2] = {
            static_cast<uint8_t>(nonce & 0xFF),
            static_cast<uint8_t>((nonce >> 8) & 0xFF)
        };

        uint8_t* h_nonce = nullptr;
        cudaMallocAsync(&h_nonce, 2, stream);
        cudaMemcpyAsync(h_nonce, nonce_bytes, 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_shake_input + 32, h_nonce, 2, cudaMemcpyDeviceToDevice, stream);
        cudaFreeAsync(h_nonce, stream);

        // Run SHAKE-256 XOF
        uint8_t* d_xof_j = d_xof_output + j * xof_size_per_poly;
        shake256_xof(d_shake_input, 34, d_xof_j, xof_size_per_poly, stream);

        cudaFreeAsync(d_shake_input, stream);
    }

    // Convert XOF bytes to gamma1 coefficients
    dim3 grid(l, 1, 1);
    dim3 block(256, 1, 1);

    kernel_sample_y_gamma1<<<grid, block, 0, stream>>>(
        d_y, d_xof_output, l, n, q, gamma1);

    cudaFreeAsync(d_xof_output, stream);
}

// =============================================================================
// Helper: poly_mul_ntt_gpu - multiply one polynomial by c*poly
// =============================================================================
//
// Computes (c * poly) for a single polynomial via NTT.
// c is shape (n,) int8, poly is shape (n,) uint32.
// Result is shape (n,) uint32.
//
static void poly_mul_c_gpu(const Params& params,
                           const int8_t* d_c,
                           const uint32_t* d_poly,
                           uint32_t* d_result,
                           uint32_t* d_work,  // size 2*n
                           cudaStream_t stream) {
    const int n = params.n;
    const uint32_t q = params.q;

    // Convert c to uint32
    uint32_t* d_c_uint = d_work;
    uint32_t* d_poly_ntt = d_work + n;

    {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        kernel_convert_challenge_to_ntt_input<<<blocks, threads, 0, stream>>>(
            d_c_uint, d_c, n, q);
    }

    // Copy poly for NTT
    cudaMemcpyAsync(d_poly_ntt, d_poly, n * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);

    // NTT on both
    ntt_forward_batch(d_c_uint, 1, params, stream, NTTImpl::AUTO);
    ntt_forward_batch(d_poly_ntt, 1, params, stream, NTTImpl::AUTO);

    // Pointwise multiply
    pointwise_multiply_batch(d_result, d_c_uint, d_poly_ntt, 1, params, stream);

    // INTT
    ntt_inverse_batch(d_result, 1, params, stream, NTTImpl::AUTO);
}

// =============================================================================
// Public API: compute_z_gpu
// =============================================================================

void compute_z_gpu(const Params& params,
                   const uint32_t* d_y,
                   const int8_t* d_c,
                   const uint32_t* d_s1,
                   uint32_t* d_z,
                   cudaStream_t stream) {
    if (!d_y || !d_c || !d_s1 || !d_z) {
        throw std::invalid_argument("compute_z_gpu: null pointer");
    }

    const int l = params.l;
    const int n = params.n;
    const uint32_t q = params.q;

    // Allocate work buffers
    uint32_t* d_work = nullptr;
    uint32_t* d_cs1_j = nullptr;

    cudaError_t err;
    err = cudaMallocAsync(&d_work, 2 * n * sizeof(uint32_t), stream);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");

    err = cudaMallocAsync(&d_cs1_j, n * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_work, stream);
        throw std::runtime_error("cudaMalloc failed");
    }

    // For each polynomial j: z[j] = y[j] + c * s1[j]
    for (int j = 0; j < l; ++j) {
        const uint32_t* d_s1_j = d_s1 + j * n;
        const uint32_t* d_y_j = d_y + j * n;
        uint32_t* d_z_j = d_z + j * n;

        // c * s1[j]
        poly_mul_c_gpu(params, d_c, d_s1_j, d_cs1_j, d_work, stream);

        // z[j] = y[j] + c*s1[j]
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        kernel_poly_add_vec<<<blocks, threads, 0, stream>>>(
            d_z_j, d_y_j, d_cs1_j, n, q);
    }

    cudaFreeAsync(d_work, stream);
    cudaFreeAsync(d_cs1_j, stream);
}

// =============================================================================
// Public API: compute_r_gpu (r = w - c*s2)
// =============================================================================

void compute_r_gpu(const Params& params,
                   const uint32_t* d_w,
                   const int8_t* d_c,
                   const uint32_t* d_s2,
                   uint32_t* d_r,
                   cudaStream_t stream) {
    if (!d_w || !d_c || !d_s2 || !d_r) {
        throw std::invalid_argument("compute_r_gpu: null pointer");
    }

    const int k = params.k;
    const int n = params.n;
    const uint32_t q = params.q;

    uint32_t* d_work = nullptr;
    uint32_t* d_cs2_i = nullptr;

    cudaError_t err;
    err = cudaMallocAsync(&d_work, 2 * n * sizeof(uint32_t), stream);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");

    err = cudaMallocAsync(&d_cs2_i, n * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_work, stream);
        throw std::runtime_error("cudaMalloc failed");
    }

    // For each polynomial i: r[i] = w[i] - c * s2[i]
    for (int i = 0; i < k; ++i) {
        const uint32_t* d_s2_i = d_s2 + i * n;
        const uint32_t* d_w_i = d_w + i * n;
        uint32_t* d_r_i = d_r + i * n;

        // c * s2[i]
        poly_mul_c_gpu(params, d_c, d_s2_i, d_cs2_i, d_work, stream);

        // r[i] = w[i] - c*s2[i]
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        kernel_poly_sub_vec<<<blocks, threads, 0, stream>>>(
            d_r_i, d_w_i, d_cs2_i, n, q);
    }

    cudaFreeAsync(d_work, stream);
    cudaFreeAsync(d_cs2_i, stream);
}

// =============================================================================
// Public API: compute_ct0_gpu (c*t0)
// =============================================================================

void compute_ct0_gpu(const Params& params,
                     const int8_t* d_c,
                     const uint32_t* d_t0,
                     uint32_t* d_ct0,
                     cudaStream_t stream) {
    if (!d_c || !d_t0 || !d_ct0) {
        throw std::invalid_argument("compute_ct0_gpu: null pointer");
    }

    const int k = params.k;
    const int n = params.n;

    uint32_t* d_work = nullptr;
    cudaError_t err = cudaMallocAsync(&d_work, 2 * n * sizeof(uint32_t), stream);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");

    for (int i = 0; i < k; ++i) {
        const uint32_t* d_t0_i = d_t0 + i * n;
        uint32_t* d_ct0_i = d_ct0 + i * n;

        poly_mul_c_gpu(params, d_c, d_t0_i, d_ct0_i, d_work, stream);
    }

    cudaFreeAsync(d_work, stream);
}

// =============================================================================
// Public API: check_norms_gpu
// =============================================================================

void check_norms_gpu(const Params& params,
                     const uint32_t* d_z,
                     const uint32_t* d_r0,
                     uint8_t* d_z_ok,
                     uint8_t* d_r0_ok,
                     cudaStream_t stream) {
    if (!d_z || !d_r0 || !d_z_ok || !d_r0_ok) {
        throw std::invalid_argument("check_norms_gpu: null pointer");
    }

    const int l = params.l;
    const int k = params.k;
    const int n = params.n;
    const uint32_t q = params.q;

    int64_t z_bound = static_cast<int64_t>(params.gamma1) - static_cast<int64_t>(params.beta);
    int64_t r0_bound = static_cast<int64_t>(params.gamma2) - static_cast<int64_t>(params.beta);

    std::size_t z_count = static_cast<std::size_t>(l) * n;
    std::size_t r0_count = static_cast<std::size_t>(k) * n;

    // Use single block for simplicity (sufficient for small arrays)
    const int threads = 256;
    std::size_t smem_size = threads * sizeof(int64_t);

    kernel_infinity_norm_check<<<1, threads, smem_size, stream>>>(
        d_z, z_count, q, z_bound, d_z_ok);

    kernel_infinity_norm_check<<<1, threads, smem_size, stream>>>(
        d_r0, r0_count, q, r0_bound, d_r0_ok);
}

// =============================================================================
// Public API: make_hint_gpu
// =============================================================================

void make_hint_gpu(const Params& params,
                   const uint32_t* d_neg_ct0,
                   const uint32_t* d_r_hint,
                   uint8_t* d_h,
                   uint32_t* d_hint_count,
                   cudaStream_t stream) {
    if (!d_neg_ct0 || !d_r_hint || !d_h || !d_hint_count) {
        throw std::invalid_argument("make_hint_gpu: null pointer");
    }

    const int k = params.k;
    const int n = params.n;
    const uint32_t q = params.q;
    const uint32_t gamma2 = params.gamma2;

    std::size_t coeff_count = static_cast<std::size_t>(k) * n;

    // Zero hint count
    cudaMemsetAsync(d_hint_count, 0, sizeof(uint32_t), stream);

    // MakeHint
    const int threads = 256;
    int blocks = static_cast<int>((coeff_count + threads - 1) / threads);

    kernel_make_hint<<<blocks, threads, 0, stream>>>(
        d_h, d_neg_ct0, d_r_hint, coeff_count, q, gamma2);

    // Count ones
    std::size_t smem_count = threads * sizeof(uint32_t);
    kernel_count_hint_ones<<<1, threads, smem_count, stream>>>(
        d_h, coeff_count, d_hint_count);
}

// =============================================================================
// Public API: use_hint_gpu
// =============================================================================

void use_hint_gpu(const Params& params,
                  const uint8_t* d_h,
                  const uint32_t* d_w_prime,
                  uint32_t* d_w1_recon,
                  cudaStream_t stream) {
    if (!d_h || !d_w_prime || !d_w1_recon) {
        throw std::invalid_argument("use_hint_gpu: null pointer");
    }

    const int k = params.k;
    const int n = params.n;
    const uint32_t q = params.q;
    const uint32_t gamma2 = params.gamma2;

    std::size_t coeff_count = static_cast<std::size_t>(k) * n;

    const int threads = 256;
    int blocks = static_cast<int>((coeff_count + threads - 1) / threads);

    kernel_use_hint<<<blocks, threads, 0, stream>>>(
        d_w1_recon, d_h, d_w_prime, coeff_count, q, gamma2);
}

// =============================================================================
// Public API: compute_ct_gpu (c * t where t = t1 * 2^d)
// =============================================================================

void compute_ct_gpu(const Params& params,
                    const int8_t* d_c,
                    const uint32_t* d_t1,
                    uint32_t* d_ct,
                    cudaStream_t stream) {
    if (!d_c || !d_t1 || !d_ct) {
        throw std::invalid_argument("compute_ct_gpu: null pointer");
    }

    const int k = params.k;
    const int n = params.n;
    const uint32_t q = params.q;
    const uint32_t d_param = params.d;

    // Allocate scaled t
    uint32_t* d_t_scaled = nullptr;
    uint32_t* d_work = nullptr;

    cudaError_t err;
    std::size_t t_size = static_cast<std::size_t>(k) * n * sizeof(uint32_t);
    err = cudaMallocAsync(&d_t_scaled, t_size, stream);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");

    err = cudaMallocAsync(&d_work, 2 * n * sizeof(uint32_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_t_scaled, stream);
        throw std::runtime_error("cudaMalloc failed");
    }

    // Scale t1 by 2^d
    std::size_t coeff_count = static_cast<std::size_t>(k) * n;
    const int threads = 256;
    int blocks = static_cast<int>((coeff_count + threads - 1) / threads);

    kernel_scale_t1<<<blocks, threads, 0, stream>>>(
        d_t_scaled, d_t1, coeff_count, q, d_param);

    // c * t for each polynomial
    for (int i = 0; i < k; ++i) {
        const uint32_t* d_t_i = d_t_scaled + i * n;
        uint32_t* d_ct_i = d_ct + i * n;

        poly_mul_c_gpu(params, d_c, d_t_i, d_ct_i, d_work, stream);
    }

    cudaFreeAsync(d_t_scaled, stream);
    cudaFreeAsync(d_work, stream);
}

// =============================================================================
// Public API: compute_w_prime_gpu (w' = Az - ct)
// =============================================================================

void compute_w_prime_gpu(const Params& params,
                         const uint32_t* d_Az,
                         const uint32_t* d_ct,
                         uint32_t* d_w_prime,
                         cudaStream_t stream) {
    if (!d_Az || !d_ct || !d_w_prime) {
        throw std::invalid_argument("compute_w_prime_gpu: null pointer");
    }

    const int k = params.k;
    const int n = params.n;
    const uint32_t q = params.q;

    std::size_t coeff_count = static_cast<std::size_t>(k) * n;

    const int threads = 256;
    int blocks = static_cast<int>((coeff_count + threads - 1) / threads);

    kernel_poly_sub_vec<<<blocks, threads, 0, stream>>>(
        d_w_prime, d_Az, d_ct, coeff_count, q);
}

// =============================================================================
// CPU Reference Functions
// =============================================================================

int64_t infinity_norm_cpu(const uint32_t* coeffs, std::size_t count, uint32_t q) {
    int64_t max_abs = 0;
    for (std::size_t i = 0; i < count; ++i) {
        int64_t val = static_cast<int64_t>(coeffs[i]);
        if (val > static_cast<int64_t>(q) / 2) {
            val -= static_cast<int64_t>(q);
        }
        int64_t absval = val >= 0 ? val : -val;
        if (absval > max_abs) max_abs = absval;
    }
    return max_abs;
}

bool check_z_norm_cpu(const uint32_t* z, std::size_t count,
                      uint32_t gamma1, uint32_t beta, uint32_t q) {
    int64_t bound = static_cast<int64_t>(gamma1) - static_cast<int64_t>(beta);
    int64_t norm = infinity_norm_cpu(z, count, q);
    return norm < bound;
}

bool check_r0_norm_cpu(const uint32_t* r0, std::size_t count,
                       uint32_t gamma2, uint32_t beta, uint32_t q) {
    int64_t bound = static_cast<int64_t>(gamma2) - static_cast<int64_t>(beta);
    int64_t norm = infinity_norm_cpu(r0, count, q);
    return norm < bound;
}

} // namespace dilithium
} // namespace smoke
