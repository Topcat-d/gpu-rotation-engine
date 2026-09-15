// dilithium_sign_batch.cu - Batch-dimension CUDA kernels for Dilithium signing
// Card 19B: True batch-dimension kernels for launch amortization
//
// This file implements:
//   - Batch y sampling: grid(B, l)
//   - Batch NTT/matvec: single launch for all lanes
//   - Batch z/r/norms/hints: grid(B, k) or flat kernels
//   - Sign round GPU entrypoint: one call per round stage
//
// Key invariant: Within each sign round, every stage is ONE kernel launch
// (or a small constant number), regardless of batch size B.

#include "../include/dilithium_sign_batch.h"
#include "../include/dilithium_ntt.cuh"
#include "../include/dilithium_shake.h"
#include "../include/dilithium_shake_batch.h"  // Card 19C: Batched SHAKE

#include <cstdint>
#include <stdexcept>

namespace smoke {
namespace dilithium {

namespace {

constexpr uint32_t N = 256;
constexpr uint32_t Q = 8380417;

// =============================================================================
// Device Helper Functions
// =============================================================================

__device__ __forceinline__
int64_t decode_signed(uint32_t val, uint32_t q) {
    int64_t v = static_cast<int64_t>(val);
    if (v > static_cast<int64_t>(q) / 2) {
        v -= static_cast<int64_t>(q);
    }
    return v;
}

__device__ __forceinline__
uint64_t abs_val(int64_t x) {
    return x >= 0 ? static_cast<uint64_t>(x) : static_cast<uint64_t>(-x);
}

__device__ __forceinline__
uint32_t mod_q(uint64_t x) {
    return static_cast<uint32_t>(x % Q);
}

__device__ __forceinline__
uint32_t add_mod_q(uint32_t a, uint32_t b) {
    return mod_q(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}

__device__ __forceinline__
uint32_t sub_mod_q(uint32_t a, uint32_t b) {
    int64_t diff = static_cast<int64_t>(a) - static_cast<int64_t>(b);
    return static_cast<uint32_t>((diff % Q + Q) % Q);
}

__device__ __forceinline__
uint32_t mul_mod_q(uint32_t a, uint32_t b) {
    return mod_q(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

// HighBits decomposition
__device__ __forceinline__
void decompose_coeff(uint32_t r, uint32_t gamma2,
                     uint32_t& r1_out, int32_t& r0_out) {
    uint32_t alpha = 2 * gamma2;
    r = r % Q;
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
    if (r_minus_r0 == static_cast<int64_t>(Q) - 1) {
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
uint32_t highbits(uint32_t r, uint32_t gamma2) {
    uint32_t r1;
    int32_t r0;
    decompose_coeff(r, gamma2, r1, r0);
    return r1;
}

// =============================================================================
// Batch Kernel: Sample y from gamma1 distribution
// =============================================================================
// Grid: (B, l)
// Block: 256 threads
// Each block handles one polynomial for one batch lane
//
__global__ void kernel_sample_y_batch(
    uint32_t* __restrict__ d_y,           // [B, l, n]
    const uint8_t* __restrict__ d_xof,    // [B, l, 3*n] XOF output
    const uint8_t* __restrict__ d_active, // [B] mask
    int B, int l, int n,
    uint32_t gamma1)
{
    int b = blockIdx.x;
    int j = blockIdx.y;

    if (b >= B || j >= l) return;
    if (d_active[b] == 0) return;  // Skip inactive lanes

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pointer to XOF output for this (b, j)
    const uint8_t* xof = d_xof + (b * l + j) * 3 * n;
    uint32_t* y_out = d_y + (b * l + j) * n;

    // Bit mask depends on gamma1
    uint32_t mask = (gamma1 == (1u << 17)) ? 0x3FFFF : 0xFFFFF;

    for (int i = tid; i < n; i += stride) {
        // Extract 3 bytes
        uint32_t val = xof[3*i] | (xof[3*i + 1] << 8) | (xof[3*i + 2] << 16);
        val &= mask;

        // Reduce to [0, 2*gamma1 - 1]
        val = val % (2 * gamma1 - 1);

        // Map to [-(gamma1-1), gamma1-1]
        int32_t coeff = static_cast<int32_t>(val) - static_cast<int32_t>(gamma1 - 1);

        // Encode in [0, q)
        uint32_t encoded;
        if (coeff >= 0) {
            encoded = static_cast<uint32_t>(coeff);
        } else {
            encoded = static_cast<uint32_t>(static_cast<int64_t>(Q) + coeff);
        }

        y_out[i] = encoded;
    }
}

// =============================================================================
// Batch Kernel: Matrix-vector product w_hat = A_hat * y_hat
// =============================================================================
// Grid: (B, k)
// Block: 256 threads
// Each block computes one output polynomial w_hat[b, i, :]
//
__global__ void kernel_matvec_batch(
    uint32_t* __restrict__ d_w_hat,       // [B, k, n] output
    const uint32_t* __restrict__ d_A_hat, // [k, l, n] shared matrix
    const uint32_t* __restrict__ d_y_hat, // [B, l, n] input
    const uint8_t* __restrict__ d_active, // [B] mask
    int B, int k, int l, int n)
{
    int b = blockIdx.x;
    int i = blockIdx.y;

    if (b >= B || i >= k) return;
    if (d_active[b] == 0) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    uint32_t* w_out = d_w_hat + (b * k + i) * n;

    for (int c = tid; c < n; c += stride) {
        uint64_t acc = 0;

        // Sum over l: A_hat[i, j, c] * y_hat[b, j, c]
        for (int j = 0; j < l; ++j) {
            uint32_t a_val = d_A_hat[(i * l + j) * n + c];
            uint32_t y_val = d_y_hat[(b * l + j) * n + c];
            acc += static_cast<uint64_t>(a_val) * static_cast<uint64_t>(y_val);
        }

        w_out[c] = static_cast<uint32_t>(acc % Q);
    }
}

// =============================================================================
// Batch Kernel: Decompose w -> (w1, w0)
// =============================================================================
// Flat kernel over all coefficients
//
__global__ void kernel_decompose_batch(
    uint32_t* __restrict__ d_w1,          // [B, k, n]
    int32_t* __restrict__ d_w0,           // [B, k, n] (signed for norm check)
    const uint32_t* __restrict__ d_w,     // [B, k, n]
    const uint8_t* __restrict__ d_active, // [B]
    int B, int k, int n, uint32_t gamma2)
{
    int total = B * k * n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Map idx to (b, i, c)
    int poly_n = k * n;
    int b = idx / poly_n;

    if (d_active[b] == 0) return;

    uint32_t w_val = d_w[idx];
    uint32_t r1;
    int32_t r0;
    decompose_coeff(w_val, gamma2, r1, r0);

    d_w1[idx] = r1;
    d_w0[idx] = r0;
}

// =============================================================================
// Batch Kernel: Convert challenge c from int8 to uint32
// =============================================================================
// Grid: (B)
// Block: 256 threads
//
__global__ void kernel_convert_c_batch(
    uint32_t* __restrict__ d_c_uint,      // [B, n]
    const int8_t* __restrict__ d_c,       // [B, n]
    const uint8_t* __restrict__ d_active, // [B]
    int B, int n)
{
    int b = blockIdx.x;
    if (b >= B) return;
    if (d_active[b] == 0) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    const int8_t* c_in = d_c + b * n;
    uint32_t* c_out = d_c_uint + b * n;

    for (int i = tid; i < n; i += stride) {
        int8_t c_val = c_in[i];
        uint32_t c_uint;
        if (c_val >= 0) {
            c_uint = static_cast<uint32_t>(c_val);
        } else {
            c_uint = static_cast<uint32_t>(static_cast<int32_t>(Q) + c_val);
        }
        c_out[i] = c_uint;
    }
}

// =============================================================================
// Batch Kernel: Pointwise multiply-add z = y + (c_ntt * s1_ntt via INTT)
// =============================================================================
// This is for computing c*s1 where c is already in NTT domain
// Grid: (B, l)
// Block: 256 threads
//
__global__ void kernel_pointwise_mul_batch(
    uint32_t* __restrict__ d_out,         // [B, poly_count, n]
    const uint32_t* __restrict__ d_a,     // [B, n] (c_ntt, broadcast across polys)
    const uint32_t* __restrict__ d_b,     // [poly_count, n] (s1_ntt, shared)
    const uint8_t* __restrict__ d_active, // [B]
    int B, int poly_count, int n)
{
    int b = blockIdx.x;
    int p = blockIdx.y;

    if (b >= B || p >= poly_count) return;
    if (d_active[b] == 0) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    const uint32_t* a_ptr = d_a + b * n;        // c_ntt for lane b
    const uint32_t* b_ptr = d_b + p * n;        // s1_ntt[p]
    uint32_t* out_ptr = d_out + (b * poly_count + p) * n;

    for (int i = tid; i < n; i += stride) {
        out_ptr[i] = mul_mod_q(a_ptr[i], b_ptr[i]);
    }
}

// =============================================================================
// Batch Kernel: z = y + cs1 (add after INTT of cs1)
// =============================================================================
__global__ void kernel_add_batch(
    uint32_t* __restrict__ d_out,         // [B, poly_count, n]
    const uint32_t* __restrict__ d_a,     // [B, poly_count, n]
    const uint32_t* __restrict__ d_b,     // [B, poly_count, n]
    const uint8_t* __restrict__ d_active, // [B]
    int B, int poly_count, int n)
{
    int total = B * poly_count * n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int poly_n = poly_count * n;
    int b = idx / poly_n;
    if (d_active[b] == 0) return;

    d_out[idx] = add_mod_q(d_a[idx], d_b[idx]);
}

// =============================================================================
// Batch Kernel: Subtract r = w - cs2
// =============================================================================
__global__ void kernel_sub_batch(
    uint32_t* __restrict__ d_out,         // [B, poly_count, n]
    const uint32_t* __restrict__ d_a,     // [B, poly_count, n]
    const uint32_t* __restrict__ d_b,     // [B, poly_count, n]
    const uint8_t* __restrict__ d_active, // [B]
    int B, int poly_count, int n)
{
    int total = B * poly_count * n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int poly_n = poly_count * n;
    int b = idx / poly_n;
    if (d_active[b] == 0) return;

    d_out[idx] = sub_mod_q(d_a[idx], d_b[idx]);
}

// =============================================================================
// Batch Kernel: Check norms and compute pass flag
// =============================================================================
// Grid: (B)
// Block: 256 threads with reduction
//
__global__ void kernel_check_z_norm_batch(
    uint8_t* __restrict__ d_pass,         // [B] output flags
    const uint32_t* __restrict__ d_z,     // [B, l, n]
    const uint8_t* __restrict__ d_active, // [B]
    int B, int l, int n,
    int64_t bound)
{
    extern __shared__ int64_t smax[];

    int b = blockIdx.x;
    if (b >= B) return;
    if (d_active[b] == 0) {
        // Lane already converged or gave up - don't touch d_pass
        // (d_pass preserves the final status from when the lane was active)
        return;
    }

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int total = l * n;

    const uint32_t* z_ptr = d_z + b * total;

    // Each thread finds local max
    int64_t local_max = 0;
    for (int i = tid; i < total; i += stride) {
        int64_t decoded = decode_signed(z_ptr[i], Q);
        uint64_t absval = abs_val(decoded);
        if (static_cast<int64_t>(absval) > local_max) {
            local_max = static_cast<int64_t>(absval);
        }
    }

    smax[tid] = local_max;
    __syncthreads();

    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_pass[b] = (smax[0] < bound) ? 1 : 0;
    }
}

// =============================================================================
// Batch Kernel: Check r0 norm (used with w0 from decompose)
// =============================================================================
__global__ void kernel_check_r0_norm_batch(
    uint8_t* __restrict__ d_pass,         // [B] output flags (AND with existing)
    const int32_t* __restrict__ d_r0,     // [B, k, n] signed
    const uint8_t* __restrict__ d_active, // [B]
    int B, int k, int n,
    int64_t bound)
{
    extern __shared__ int64_t smax[];

    int b = blockIdx.x;
    if (b >= B) return;
    if (d_active[b] == 0 || d_pass[b] == 0) {
        // Already failed z norm or inactive
        return;
    }

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int total = k * n;

    const int32_t* r0_ptr = d_r0 + b * total;

    int64_t local_max = 0;
    for (int i = tid; i < total; i += stride) {
        int64_t absval = static_cast<int64_t>(r0_ptr[i] >= 0 ? r0_ptr[i] : -r0_ptr[i]);
        if (absval > local_max) local_max = absval;
    }

    smax[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (smax[0] >= bound) {
            d_pass[b] = 0;  // Fail if norm too large
        }
    }
}

// =============================================================================
// Batch Kernel: MakeHint
// =============================================================================
// Grid: (B, k)
// Block: 256 threads
//
__global__ void kernel_make_hint_batch(
    uint8_t* __restrict__ d_h,            // [B, k, n]
    uint32_t* __restrict__ d_hint_count,  // [B] count of hints per lane
    const uint32_t* __restrict__ d_neg_ct0, // [B, k, n]
    const uint32_t* __restrict__ d_r_hint,  // [B, k, n]
    const uint8_t* __restrict__ d_active, // [B]
    const uint8_t* __restrict__ d_pass,   // [B] norm pass flags
    int B, int k, int n, uint32_t gamma2)
{
    extern __shared__ uint32_t scount[];

    int b = blockIdx.x;
    int p = blockIdx.y;  // poly index in k

    if (b >= B || p >= k) return;
    if (d_active[b] == 0 || d_pass[b] == 0) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int offset = (b * k + p) * n;

    uint32_t local_count = 0;

    for (int i = tid; i < n; i += stride) {
        uint32_t r_val = d_r_hint[offset + i];
        uint32_t z_val = d_neg_ct0[offset + i];

        int64_t z_signed = decode_signed(z_val, Q);

        uint32_t r1 = highbits(r_val, gamma2);

        int64_t r_plus_z = (static_cast<int64_t>(r_val) + z_signed) % Q;
        if (r_plus_z < 0) r_plus_z += Q;

        uint32_t r1_plus = highbits(static_cast<uint32_t>(r_plus_z), gamma2);

        uint8_t h_val = (r1 != r1_plus) ? 1 : 0;
        d_h[offset + i] = h_val;
        local_count += h_val;
    }

    // Reduction for this poly
    scount[tid] = local_count;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scount[tid] += scount[tid + s];
        __syncthreads();
    }

    // Atomic add to lane's hint count
    if (tid == 0) {
        atomicAdd(&d_hint_count[b], scount[0]);
    }
}

// =============================================================================
// Batch Kernel: Check omega bound and finalize pass
// =============================================================================
__global__ void kernel_check_omega_batch(
    uint8_t* __restrict__ d_pass,         // [B] (modify in place)
    const uint32_t* __restrict__ d_hint_count, // [B]
    const uint8_t* __restrict__ d_active, // [B]
    int B, uint32_t omega)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    if (d_active[b] == 0) return;

    if (d_pass[b] == 1 && d_hint_count[b] > omega) {
        d_pass[b] = 0;
    }
}

// =============================================================================
// Batch Kernel: Update active mask based on pass flags
// =============================================================================
__global__ void kernel_update_active_batch(
    uint8_t* __restrict__ d_active,       // [B]
    uint16_t* __restrict__ d_attempts,    // [B]
    const uint8_t* __restrict__ d_pass,   // [B]
    int B, uint16_t max_attempts)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    if (d_active[b] == 0) return;

    if (d_pass[b] == 1) {
        // Success - mark inactive
        d_active[b] = 0;
    } else {
        // Failed - increment attempt
        d_attempts[b]++;
        if (d_attempts[b] >= max_attempts) {
            d_active[b] = 0;  // Give up
        }
    }
}

// =============================================================================
// Batch Kernel: Count active lanes (reduction)
// =============================================================================
__global__ void kernel_count_active(
    uint32_t* __restrict__ d_active_count,
    const uint8_t* __restrict__ d_active,
    int B)
{
    extern __shared__ uint32_t scount[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    uint32_t local = 0;
    for (int i = tid; i < B; i += stride) {
        local += d_active[i];
    }

    scount[tid] = local;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scount[tid] += scount[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        *d_active_count = scount[0];
    }
}

} // anonymous namespace

// =============================================================================
// Public API: Batch Sample Y
// =============================================================================

void sample_y_gamma1_batch_gpu(
    const Params& params,
    const uint8_t* d_rhoprime,     // [32] seed on device
    const uint16_t* d_attempts,    // [B] per-lane attempt counter
    const uint8_t* d_active,       // [B] mask
    uint32_t* d_y,                 // [B, l, n] output
    int B,
    cudaStream_t stream)
{
    const int l = params.l;
    const int n = params.n;
    const uint32_t gamma1 = params.gamma1;

    // Card 19C: Use batched SHAKE implementation (3 constant launches)
    // Allocate workspace for seeds + XOF output
    size_t work_size = sample_y_workspace_size(B, l);
    uint8_t* d_work = nullptr;
    cudaError_t err = cudaMalloc(&d_work, work_size);
    if (err != cudaSuccess || d_work == nullptr) {
        // Allocation failed - cannot proceed
        return;
    }

    // Call the batched v2 implementation
    sample_y_gamma1_batch_v2_gpu(
        d_rhoprime, d_attempts, d_active,
        d_y, d_work,
        B, l, n, gamma1,
        stream);

    // Sync stream before freeing to ensure work is complete
    cudaStreamSynchronize(stream);
    cudaFree(d_work);
}

// =============================================================================
// Public API: Batch Matvec (w_hat = A_hat * y_hat)
// =============================================================================

void matvec_batch_gpu(
    const Params& params,
    const uint32_t* d_A_hat,       // [k, l, n] shared
    const uint32_t* d_y_hat,       // [B, l, n]
    uint32_t* d_w_hat,             // [B, k, n]
    const uint8_t* d_active,       // [B]
    int B,
    cudaStream_t stream)
{
    dim3 grid(B, params.k);
    dim3 block(256);

    kernel_matvec_batch<<<grid, block, 0, stream>>>(
        d_w_hat, d_A_hat, d_y_hat, d_active,
        B, params.k, params.l, params.n);
}

// =============================================================================
// Public API: Batch Decompose (w -> w1, w0)
// =============================================================================

void decompose_batch_gpu(
    const Params& params,
    const uint32_t* d_w,           // [B, k, n]
    uint32_t* d_w1,                // [B, k, n]
    int32_t* d_w0,                 // [B, k, n]
    const uint8_t* d_active,       // [B]
    int B,
    cudaStream_t stream)
{
    int total = B * params.k * params.n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    kernel_decompose_batch<<<blocks, threads, 0, stream>>>(
        d_w1, d_w0, d_w, d_active,
        B, params.k, params.n, params.gamma2);
}

// =============================================================================
// Public API: Batch Compute Z (z = y + c*s1)
// =============================================================================

void compute_z_batch_gpu(
    const Params& params,
    const uint32_t* d_y,           // [B, l, n]
    const int8_t* d_c,             // [B, n]
    const uint32_t* d_s1_ntt,      // [l, n] s1 in NTT domain (shared)
    uint32_t* d_z,                 // [B, l, n] output
    const uint8_t* d_active,       // [B]
    int B,
    uint32_t* d_work,              // Workspace [B, l, n] + [B, n]
    cudaStream_t stream)
{
    const int l = params.l;
    const int n = params.n;

    // Work buffers
    uint32_t* d_c_ntt = d_work;                    // [B, n]
    uint32_t* d_cs1 = d_work + B * n;              // [B, l, n]

    // Step 1: Convert c to uint32 and NTT
    dim3 grid_c(B);
    dim3 block_c(256);
    kernel_convert_c_batch<<<grid_c, block_c, 0, stream>>>(
        d_c_ntt, d_c, d_active, B, n);

    // NTT on all c's
    ntt_forward_batch(d_c_ntt, B, params, stream, NTTImpl::AUTO);

    // Step 2: Pointwise multiply c_ntt * s1_ntt for each poly
    dim3 grid_mul(B, l);
    kernel_pointwise_mul_batch<<<grid_mul, block_c, 0, stream>>>(
        d_cs1, d_c_ntt, d_s1_ntt, d_active, B, l, n);

    // Step 3: INTT on cs1
    ntt_inverse_batch(d_cs1, B * l, params, stream, NTTImpl::AUTO);

    // Step 4: z = y + cs1
    int total = B * l * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_add_batch<<<blocks, threads, 0, stream>>>(
        d_z, d_y, d_cs1, d_active, B, l, n);
}

// =============================================================================
// Public API: Check Z Norm Batch
// =============================================================================

void check_z_norm_batch_gpu(
    const Params& params,
    const uint32_t* d_z,           // [B, l, n]
    uint8_t* d_pass,               // [B] output
    const uint8_t* d_active,       // [B]
    int B,
    cudaStream_t stream)
{
    int64_t bound = static_cast<int64_t>(params.gamma1) - static_cast<int64_t>(params.beta);

    dim3 grid(B);
    dim3 block(256);
    size_t smem = 256 * sizeof(int64_t);

    kernel_check_z_norm_batch<<<grid, block, smem, stream>>>(
        d_pass, d_z, d_active, B, params.l, params.n, bound);
}

// =============================================================================
// Public API: Update Active Mask
// =============================================================================

void update_active_batch_gpu(
    uint8_t* d_active,             // [B]
    uint16_t* d_attempts,          // [B]
    const uint8_t* d_pass,         // [B]
    int B,
    uint16_t max_attempts,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    kernel_update_active_batch<<<blocks, threads, 0, stream>>>(
        d_active, d_attempts, d_pass, B, max_attempts);
}

// =============================================================================
// Public API: Count Active Lanes
// =============================================================================

uint32_t count_active_batch_gpu(
    const uint8_t* d_active,       // [B]
    int B,
    cudaStream_t stream)
{
    uint32_t* d_count = nullptr;
    cudaError_t err = cudaMalloc(&d_count, sizeof(uint32_t));
    if (err != cudaSuccess || d_count == nullptr) {
        return 0;  // Failed to allocate
    }
    cudaMemsetAsync(d_count, 0, sizeof(uint32_t), stream);

    dim3 block(256);
    size_t smem = 256 * sizeof(uint32_t);

    kernel_count_active<<<1, block, smem, stream>>>(d_count, d_active, B);

    uint32_t h_count;
    cudaMemcpyAsync(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_count);
    return h_count;
}

// =============================================================================
// Batch Kernel: Compute r = w - c*s2
// =============================================================================
// Same pattern as compute_z but with subtraction
//
void compute_r_batch_gpu(
    const Params& params,
    const uint32_t* d_w,           // [B, k, n]
    const int8_t* d_c,             // [B, n]
    const uint32_t* d_s2_ntt,      // [k, n] s2 in NTT domain (shared)
    uint32_t* d_r,                 // [B, k, n] output
    const uint8_t* d_active,       // [B]
    int B,
    uint32_t* d_work,              // Workspace [B, k, n] + [B, n]
    cudaStream_t stream)
{
    const int k = params.k;
    const int n = params.n;

    uint32_t* d_c_ntt = d_work;                    // [B, n]
    uint32_t* d_cs2 = d_work + B * n;              // [B, k, n]

    // Convert c to uint32 and NTT
    dim3 grid_c(B);
    dim3 block(256);
    kernel_convert_c_batch<<<grid_c, block, 0, stream>>>(
        d_c_ntt, d_c, d_active, B, n);

    ntt_forward_batch(d_c_ntt, B, params, stream, NTTImpl::AUTO);

    // Pointwise multiply c_ntt * s2_ntt for each poly
    dim3 grid_mul(B, k);
    kernel_pointwise_mul_batch<<<grid_mul, block, 0, stream>>>(
        d_cs2, d_c_ntt, d_s2_ntt, d_active, B, k, n);

    // INTT on cs2
    ntt_inverse_batch(d_cs2, B * k, params, stream, NTTImpl::AUTO);

    // r = w - cs2
    int total = B * k * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_sub_batch<<<blocks, threads, 0, stream>>>(
        d_r, d_w, d_cs2, d_active, B, k, n);
}

// =============================================================================
// Batch Kernel: Compute ct0 = c * t0 (for hints)
// =============================================================================
void compute_ct0_batch_gpu(
    const Params& params,
    const int8_t* d_c,             // [B, n]
    const uint32_t* d_t0_ntt,      // [k, n] t0 in NTT domain (shared)
    uint32_t* d_ct0,               // [B, k, n] output
    const uint8_t* d_active,       // [B]
    int B,
    uint32_t* d_work,              // [B, n] for c_ntt
    cudaStream_t stream)
{
    const int k = params.k;
    const int n = params.n;

    uint32_t* d_c_ntt = d_work;

    dim3 block(256);

    // Convert c and NTT
    kernel_convert_c_batch<<<B, block, 0, stream>>>(
        d_c_ntt, d_c, d_active, B, n);

    ntt_forward_batch(d_c_ntt, B, params, stream, NTTImpl::AUTO);

    // Pointwise multiply c_ntt * t0_ntt
    dim3 grid_mul(B, k);
    kernel_pointwise_mul_batch<<<grid_mul, block, 0, stream>>>(
        d_ct0, d_c_ntt, d_t0_ntt, d_active, B, k, n);

    // INTT
    ntt_inverse_batch(d_ct0, B * k, params, stream, NTTImpl::AUTO);
}

// =============================================================================
// PUBLIC API: RoundA - Produces w1 for all active lanes
// =============================================================================

void sign_roundA_w1_batch_gpu(
    const Params& params,
    const uint8_t* d_rhoprime,
    const uint32_t* d_A_hat,
    const uint8_t* d_active,
    const uint16_t* d_attempts,
    uint32_t* d_y,
    uint32_t* d_w,
    uint32_t* d_w1,
    int32_t* d_w0,
    uint32_t* d_work,
    int B,
    cudaStream_t stream)
{
    const int l = params.l;
    const int k = params.k;
    const int n = params.n;

    // Work buffer for y in NTT domain
    uint32_t* d_y_hat = d_work;  // [B, l, n]

    // Step 1: Sample y for all active lanes
    sample_y_gamma1_batch_gpu(params, d_rhoprime, d_attempts, d_active,
                               d_y, B, stream);

    // Step 2: NTT(y) -> y_hat
    // Copy y to work buffer for NTT
    cudaMemcpyAsync(d_y_hat, d_y, B * l * n * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);
    ntt_forward_batch(d_y_hat, B * l, params, stream, NTTImpl::AUTO);

    // Step 3: w_hat = A_hat * y_hat (matvec in NTT domain)
    // Output directly to d_w (will be overwritten after INTT)
    matvec_batch_gpu(params, d_A_hat, d_y_hat, d_w, d_active, B, stream);

    // Step 4: INTT(w_hat) -> w (in-place)
    ntt_inverse_batch(d_w, B * k, params, stream, NTTImpl::AUTO);

    // Step 5: Decompose w -> (w1, w0)
    decompose_batch_gpu(params, d_w, d_w1, d_w0, d_active, B, stream);
}

// =============================================================================
// PUBLIC API: RoundB - Consumes c, produces z/h/pass, updates mask
// =============================================================================

void sign_roundB_zh_batch_gpu(
    const Params& params,
    const uint32_t* d_s1_ntt,
    const uint32_t* d_s2_ntt,
    const uint32_t* d_t0_ntt,
    const uint32_t* d_y,
    const uint32_t* d_w,
    const int32_t* d_w0,
    const int8_t* d_c,
    uint8_t* d_active,
    uint16_t* d_attempts,
    uint32_t* d_z,
    uint8_t* d_h,
    uint8_t* d_pass,
    uint32_t* d_hint_count,
    uint32_t* d_active_count,
    uint16_t max_attempts,
    uint32_t* d_work,
    int B,
    cudaStream_t stream)
{
    const int l = params.l;
    const int k = params.k;
    const int n = params.n;
    const uint32_t gamma2 = params.gamma2;
    const uint32_t omega = params.omega;

    // cs buffer needs max(l, k) since cs1 uses l polys and cs2 uses k polys
    int cs_polys = (l > k) ? l : k;

    // Workspace layout:
    // [0, B*n): c_ntt
    // [B*n, B*n + B*max(l,k)*n): cs (shared for cs1 and cs2)
    // [B*n + B*max(l,k)*n, ...): r, r0, ct0, etc.
    size_t work_offset = 0;
    uint32_t* d_c_ntt = d_work + work_offset; work_offset += B * n;
    uint32_t* d_cs = d_work + work_offset; work_offset += B * cs_polys * n;
    uint32_t* d_r = d_work + work_offset; work_offset += B * k * n;
    int32_t* d_r0 = reinterpret_cast<int32_t*>(d_work + work_offset); work_offset += B * k * n;
    uint32_t* d_ct0 = d_work + work_offset; work_offset += B * k * n;
    uint32_t* d_neg_ct0 = d_work + work_offset; work_offset += B * k * n;
    uint32_t* d_r_hint = d_work + work_offset;

    dim3 block(256);

    // -------------------------------------------------------------------------
    // Step 1: Compute z = y + c*s1
    // -------------------------------------------------------------------------
    // Convert c to uint32 and NTT
    kernel_convert_c_batch<<<B, block, 0, stream>>>(
        d_c_ntt, d_c, d_active, B, n);
    ntt_forward_batch(d_c_ntt, B, params, stream, NTTImpl::AUTO);

    // c_ntt * s1_ntt for each poly (uses first l polys of cs buffer)
    dim3 grid_mul_l(B, l);
    kernel_pointwise_mul_batch<<<grid_mul_l, block, 0, stream>>>(
        d_cs, d_c_ntt, d_s1_ntt, d_active, B, l, n);

    // INTT(cs1)
    ntt_inverse_batch(d_cs, B * l, params, stream, NTTImpl::AUTO);

    // z = y + cs1
    {
        int total = B * l * n;
        int blocks = (total + 255) / 256;
        kernel_add_batch<<<blocks, block, 0, stream>>>(
            d_z, d_y, d_cs, d_active, B, l, n);
    }

    // -------------------------------------------------------------------------
    // Step 2: Check z norm
    // -------------------------------------------------------------------------
    check_z_norm_batch_gpu(params, d_z, d_pass, d_active, B, stream);

    // -------------------------------------------------------------------------
    // Step 3: Compute r = w - c*s2 (only for lanes that passed z norm)
    // -------------------------------------------------------------------------
    // c_ntt already computed, reuse
    // cs buffer is sized for max(l,k), so safe to reuse for k polys
    dim3 grid_mul_k(B, k);

    kernel_pointwise_mul_batch<<<grid_mul_k, block, 0, stream>>>(
        d_cs, d_c_ntt, d_s2_ntt, d_active, B, k, n);
    ntt_inverse_batch(d_cs, B * k, params, stream, NTTImpl::AUTO);

    // r = w - cs2
    {
        int total = B * k * n;
        int blocks = (total + 255) / 256;
        kernel_sub_batch<<<blocks, block, 0, stream>>>(
            d_r, d_w, d_cs, d_active, B, k, n);
    }

    // -------------------------------------------------------------------------
    // Step 4: Decompose r and check r0 norm (FIPS 204 requirement)
    // -------------------------------------------------------------------------
    // NOTE: FIPS 204 requires checking ||r0||_∞ < γ2 - β where r0 = LowBits(r)
    // and r = w - c*s2. We MUST decompose r, not use w0 from RoundA!

    // Decompose r -> (r1, r0), we only need r0 for the norm check
    // Use d_ct0 as temporary storage for r1 (we compute ct0 later in step 5)
    // d_r must remain intact for hint computation in step 6
    {
        int total = B * k * n;
        int blocks_flat = (total + 255) / 256;
        kernel_decompose_batch<<<blocks_flat, block, 0, stream>>>(
            d_ct0, d_r0, d_r, d_active, B, k, n, gamma2);
    }

    int64_t r0_bound = static_cast<int64_t>(gamma2) - static_cast<int64_t>(params.beta);
    size_t smem = 256 * sizeof(int64_t);
    kernel_check_r0_norm_batch<<<B, block, smem, stream>>>(
        d_pass, d_r0, d_active, B, k, n, r0_bound);

    // -------------------------------------------------------------------------
    // Step 5: Compute ct0 = c*t0 (for hints)
    // -------------------------------------------------------------------------
    kernel_pointwise_mul_batch<<<grid_mul_k, block, 0, stream>>>(
        d_ct0, d_c_ntt, d_t0_ntt, d_active, B, k, n);
    ntt_inverse_batch(d_ct0, B * k, params, stream, NTTImpl::AUTO);

    // -------------------------------------------------------------------------
    // Step 6: Make hints
    // -------------------------------------------------------------------------
    // Zero hint counts
    cudaMemsetAsync(d_hint_count, 0, B * sizeof(uint32_t), stream);

    // Compute neg_ct0 = q - ct0 (for MakeHint)
    // r_hint = r + ct0
    {
        int total = B * k * n;
        int blocks = (total + 255) / 256;

        // Simple kernel for neg_ct0 and r_hint
        // We'll use a combined kernel to avoid extra launches
        // For now, do it inline
    }

    // MakeHint: compare highbits(r) vs highbits(r + neg_ct0)
    // Note: MakeHint uses (-ct0, r+ct0) in FIPS 204
    // neg_ct0[i] = (q - ct0[i]) % q
    // r_hint[i] = (r[i] + ct0[i]) % q

    // Launch MakeHint kernel
    dim3 grid_hint(B, k);
    size_t smem_hint = 256 * sizeof(uint32_t);

    // We need to compute neg_ct0 and r_hint first
    // Use a simple element-wise kernel
    auto compute_hint_inputs = [&]() {
        int total = B * k * n;
        int blocks_flat = (total + 255) / 256;

        // We'll compute in-place: neg_ct0 from ct0, r_hint = r + ct0
        // This requires a custom kernel - let's add it inline
    };

    // For simplicity, compute on CPU side for now (will optimize later)
    // Actually, let's add the kernels we need

    // Use existing make_hint_batch kernel which takes neg_ct0 and r_hint
    // We need to compute these first
    // neg_ct0 = q - ct0 (element-wise)
    // r_hint = r + ct0 (element-wise)

    // Compute neg_ct0 kernel (simple negate mod q)
    {
        int total = B * k * n;
        int blocks_flat = (total + 255) / 256;

        // Inline kernel for neg_ct0 = q - ct0
        // and r_hint = r + ct0
        // We'll reuse the sub/add kernels with q as the first operand

        // Actually, let's just do r_hint = r + ct0 first
        kernel_add_batch<<<blocks_flat, block, 0, stream>>>(
            d_r_hint, d_r, d_ct0, d_active, B, k, n);

        // neg_ct0: subtract ct0 from q (represented as 0 - ct0 mod q = q - ct0)
        // We can use a negate kernel, or compute q - ct0 inline
        // For now, compute in place using subtraction from q constant
        // This requires a new kernel - let's add it
    }

    // Simplified: use existing make_hint_batch which expects neg_ct0 and r_hint
    // For now, skip the neg_ct0 computation and use ct0 directly with modified logic
    // Actually the kernel already handles the signed decode, so we just need:
    // d_h[i] = 1 if highbits(r[i]) != highbits(r[i] + ct0[i] - ct0[i] + neg_ct0[i])
    // This is getting complex. Let me use a simpler approach.

    // The MakeHint kernel in our code uses:
    // h[i] = 1 if highbits(r_val) != highbits(r_val + z_signed)
    // where z_val is neg_ct0 (so z_signed = -ct0)
    // and r_val is r_hint = r + ct0

    // So we need:
    // neg_ct0 = q - ct0 (so decode_signed(neg_ct0) = -ct0)
    // r_hint = r + ct0

    // r_hint already computed above
    // For neg_ct0, we need to negate ct0 mod q

    // Let's add a simple negate kernel call
    // Actually easier: just compute neg_ct0 = (Q - ct0) % Q
    // which equals sub_mod_q(0, ct0) but we need a kernel for that

    // Simplest: create d_zero buffer and use sub_batch
    // But that's wasteful. Let's just skip for now and mark hints as 0
    // NO - that breaks correctness.

    // OK, let me add a proper negate kernel inline
    // Actually the make_hint_batch kernel already exists and handles this
    // It takes d_neg_ct0 and d_r_hint
    // We need to compute d_neg_ct0 = negate(d_ct0)

    // For now, allocate a zero buffer and subtract ct0 from it
    // This is a bit wasteful but works
    cudaMemsetAsync(d_neg_ct0, 0, B * k * n * sizeof(uint32_t), stream);
    {
        int total = B * k * n;
        int blocks_flat = (total + 255) / 256;
        kernel_sub_batch<<<blocks_flat, block, 0, stream>>>(
            d_neg_ct0, d_neg_ct0, d_ct0, d_active, B, k, n);
        // Now d_neg_ct0 = 0 - ct0 = q - ct0 mod q
    }

    // Now call MakeHint
    kernel_make_hint_batch<<<grid_hint, block, smem_hint, stream>>>(
        d_h, d_hint_count, d_neg_ct0, d_r_hint, d_active, d_pass,
        B, k, n, gamma2);

    // -------------------------------------------------------------------------
    // Step 7: Check omega bound
    // -------------------------------------------------------------------------
    {
        int blocks_flat = (B + 255) / 256;
        kernel_check_omega_batch<<<blocks_flat, block, 0, stream>>>(
            d_pass, d_hint_count, d_active, B, omega);
    }

    // -------------------------------------------------------------------------
    // Step 8: Update active mask and count
    // -------------------------------------------------------------------------
    update_active_batch_gpu(d_active, d_attempts, d_pass, B, max_attempts, stream);

    // Count active lanes and write result to device memory
    uint32_t h_active_count = count_active_batch_gpu(d_active, B, stream);
    cudaMemcpyAsync(d_active_count, &h_active_count, sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);  // Ensure h_active_count stays valid
}

} // namespace dilithium
} // namespace smoke
