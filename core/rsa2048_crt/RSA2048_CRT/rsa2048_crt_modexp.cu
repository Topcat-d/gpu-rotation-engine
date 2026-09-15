/**
 * CARD31: RSA-2048 CRT Modular Exponentiation Kernel
 *
 * CRT-accelerated RSA signing using two 1024-bit modexps.
 *
 * Build:
 *   python build_rsa2048_crt.py build_ext --inplace
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Include CRT Montgomery primitives
#include "rsa2048_crt.cuh"

using namespace rsa2048_crt;

// ============================================================================
// Batch CRT Sign Kernel
// ============================================================================

/**
 * Batch RSA-2048 CRT sign kernel.
 *
 * Each warp processes one message using CRT:
 *   1. sp = m^dp mod p
 *   2. sq = m^dq mod q
 *   3. h = (qinv * (sp - sq)) mod p
 *   4. s = sq + q * h
 */
__global__
void rsa2048_crt_sign_batch_kernel(
    const uint32_t* __restrict__ m_in,      // [batch_size, 64] messages
    const uint32_t* __restrict__ p,         // [32] prime p
    const uint32_t* __restrict__ q,         // [32] prime q
    const uint32_t* __restrict__ dp,        // [32] d mod (p-1)
    const uint32_t* __restrict__ dq,        // [32] d mod (q-1)
    const uint32_t* __restrict__ qinv,      // [32] q^(-1) mod p
    const uint32_t* __restrict__ R2_p,      // [32] R^2 mod p
    const uint32_t* __restrict__ R2_q,      // [32] R^2 mod q
    uint32_t p_prime,
    uint32_t q_prime,
    uint32_t* __restrict__ out,             // [batch_size, 64] signatures
    int batch_size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= batch_size) return;

    // Pointers to this message's data
    const uint32_t* m = m_in + warp_id * RSA_2048_LIMBS;
    uint32_t* sig_out = out + warp_id * RSA_2048_LIMBS;

    // Local arrays for computation
    uint32_t m_local[RSA_2048_LIMBS];
    uint32_t p_local[RSA_1024_LIMBS];
    uint32_t q_local[RSA_1024_LIMBS];
    uint32_t dp_local[RSA_1024_LIMBS];
    uint32_t dq_local[RSA_1024_LIMBS];
    uint32_t qinv_local[RSA_1024_LIMBS];
    uint32_t R2_p_local[RSA_1024_LIMBS];
    uint32_t R2_q_local[RSA_1024_LIMBS];
    uint32_t sig_local[RSA_2048_LIMBS];

    // Load data (lane 0 loads, then broadcast)
    if (lane_id == 0) {
        // Load message (2048-bit)
        for (int i = 0; i < RSA_2048_LIMBS; i++) {
            m_local[i] = m[i];
        }

        // Load CRT parameters (1024-bit each)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            p_local[i] = p[i];
            q_local[i] = q[i];
            dp_local[i] = dp[i];
            dq_local[i] = dq[i];
            qinv_local[i] = qinv[i];
            R2_p_local[i] = R2_p[i];
            R2_q_local[i] = R2_q[i];
        }
    }
    __syncwarp();

    // Perform CRT sign
    rsa_crt_sign(
        sig_local,
        m_local,
        p_local,
        q_local,
        dp_local,
        dq_local,
        qinv_local,
        R2_p_local,
        R2_q_local,
        p_prime,
        q_prime,
        lane_id);

    // Write result
    if (lane_id == 0) {
        for (int i = 0; i < RSA_2048_LIMBS; i++) {
            sig_out[i] = sig_local[i];
        }
    }
}


// ============================================================================
// PyTorch Interface
// ============================================================================

/**
 * RSA-2048 CRT sign batch: s = m^d mod n (using CRT acceleration)
 */
torch::Tensor rsa2048_crt_sign_batch(
    torch::Tensor m_tensor,       // [batch, 64]
    torch::Tensor p_tensor,       // [32]
    torch::Tensor q_tensor,       // [32]
    torch::Tensor dp_tensor,      // [32]
    torch::Tensor dq_tensor,      // [32]
    torch::Tensor qinv_tensor,    // [32]
    torch::Tensor R2_p_tensor,    // [32]
    torch::Tensor R2_q_tensor,    // [32]
    int64_t p_prime,
    int64_t q_prime)
{
    TORCH_CHECK(m_tensor.device().is_cuda(), "m_tensor must be CUDA");
    TORCH_CHECK(m_tensor.dim() == 2, "m_tensor must be 2D [batch, 64]");
    TORCH_CHECK(m_tensor.size(1) == RSA_2048_LIMBS, "m_tensor must have 64 limbs");

    int batch_size = m_tensor.size(0);

    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(m_tensor.device());
    torch::Tensor out = torch::zeros({batch_size, RSA_2048_LIMBS}, options);

    // Get raw pointers
    const uint32_t* m_ptr = reinterpret_cast<const uint32_t*>(m_tensor.data_ptr<int32_t>());
    const uint32_t* p_ptr = reinterpret_cast<const uint32_t*>(p_tensor.data_ptr<int32_t>());
    const uint32_t* q_ptr = reinterpret_cast<const uint32_t*>(q_tensor.data_ptr<int32_t>());
    const uint32_t* dp_ptr = reinterpret_cast<const uint32_t*>(dp_tensor.data_ptr<int32_t>());
    const uint32_t* dq_ptr = reinterpret_cast<const uint32_t*>(dq_tensor.data_ptr<int32_t>());
    const uint32_t* qinv_ptr = reinterpret_cast<const uint32_t*>(qinv_tensor.data_ptr<int32_t>());
    const uint32_t* R2_p_ptr = reinterpret_cast<const uint32_t*>(R2_p_tensor.data_ptr<int32_t>());
    const uint32_t* R2_q_ptr = reinterpret_cast<const uint32_t*>(R2_q_tensor.data_ptr<int32_t>());
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>());

    // Launch kernel: 1 warp per message
    int threads_per_block = 128;  // 4 warps per block
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    rsa2048_crt_sign_batch_kernel<<<num_blocks, threads_per_block>>>(
        m_ptr,
        p_ptr,
        q_ptr,
        dp_ptr,
        dq_ptr,
        qinv_ptr,
        R2_p_ptr,
        R2_q_ptr,
        (uint32_t)p_prime,
        (uint32_t)q_prime,
        out_ptr,
        batch_size);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return out;
}


// ============================================================================
// CARD33A: Warp-Cooperative CRT Sign Kernel
// ============================================================================

/**
 * CARD33A: Warp-cooperative batch RSA-2048 CRT sign kernel.
 *
 * Uses warp-cooperative Montgomery multiplication for modexp.
 * All 32 lanes participate in each Montgomery operation.
 *
 * IMPORTANT: Uses 32 threads per block (1 warp per block) to avoid
 * shared memory conflicts. Each block processes exactly 1 message.
 */
__global__
void rsa2048_crt_sign_warp_batch_kernel(
    const uint32_t* __restrict__ m_in,      // [batch_size, 64] messages
    const uint32_t* __restrict__ p,         // [32] prime p
    const uint32_t* __restrict__ q,         // [32] prime q
    const uint32_t* __restrict__ dp,        // [32] d mod (p-1)
    const uint32_t* __restrict__ dq,        // [32] d mod (q-1)
    const uint32_t* __restrict__ qinv,      // [32] q^(-1) mod p
    const uint32_t* __restrict__ R2_p,      // [32] R^2 mod p
    const uint32_t* __restrict__ R2_q,      // [32] R^2 mod q
    uint32_t p_prime,
    uint32_t q_prime,
    uint32_t* __restrict__ out,             // [batch_size, 64] signatures
    int batch_size)
{
    // Each block is exactly 1 warp (32 threads), processes 1 message
    int msg_id = blockIdx.x;
    int lane_id = threadIdx.x;

    if (msg_id >= batch_size) return;

    // Pointers to this message's data
    const uint32_t* m = m_in + msg_id * RSA_2048_LIMBS;
    uint32_t* sig_out = out + msg_id * RSA_2048_LIMBS;

    // Local array for output
    uint32_t sig_local[RSA_2048_LIMBS];

    // Use SHARED memory for CRT parameters so ALL lanes can access ALL limbs
    // This is critical: local arrays only have 1 valid limb per lane!
    __shared__ uint32_t p_shared[RSA_1024_LIMBS];
    __shared__ uint32_t q_shared[RSA_1024_LIMBS];
    __shared__ uint32_t dp_shared[RSA_1024_LIMBS];
    __shared__ uint32_t dq_shared[RSA_1024_LIMBS];
    __shared__ uint32_t qinv_shared[RSA_1024_LIMBS];
    __shared__ uint32_t R2_p_shared[RSA_1024_LIMBS];
    __shared__ uint32_t R2_q_shared[RSA_1024_LIMBS];
    __shared__ uint32_t m_shared[RSA_2048_LIMBS];

    // Parallel load into shared memory: each lane loads its limbs
    p_shared[lane_id] = p[lane_id];
    q_shared[lane_id] = q[lane_id];
    dp_shared[lane_id] = dp[lane_id];
    dq_shared[lane_id] = dq[lane_id];
    qinv_shared[lane_id] = qinv[lane_id];
    R2_p_shared[lane_id] = R2_p[lane_id];
    R2_q_shared[lane_id] = R2_q[lane_id];

    // Load 2048-bit message (64 limbs, 2 per lane)
    m_shared[lane_id] = m[lane_id];
    m_shared[lane_id + 32] = m[lane_id + 32];
    __syncwarp();

    // Call warp-cooperative CRT sign with shared memory arrays
    rsa_crt_sign_warp(
        sig_local,
        m_shared,
        p_shared,
        q_shared,
        dp_shared,
        dq_shared,
        qinv_shared,
        R2_p_shared,
        R2_q_shared,
        p_prime,
        q_prime);

    // Write result (parallel: each lane writes its limbs)
    sig_out[lane_id] = sig_local[lane_id];
    sig_out[lane_id + 32] = sig_local[lane_id + 32];
}


/**
 * CARD33A: PyTorch interface for warp-cooperative CRT sign.
 */
torch::Tensor rsa2048_crt_sign_warp_batch(
    torch::Tensor m_tensor,       // [batch, 64]
    torch::Tensor p_tensor,       // [32]
    torch::Tensor q_tensor,       // [32]
    torch::Tensor dp_tensor,      // [32]
    torch::Tensor dq_tensor,      // [32]
    torch::Tensor qinv_tensor,    // [32]
    torch::Tensor R2_p_tensor,    // [32]
    torch::Tensor R2_q_tensor,    // [32]
    int64_t p_prime,
    int64_t q_prime)
{
    TORCH_CHECK(m_tensor.device().is_cuda(), "m_tensor must be CUDA");
    TORCH_CHECK(m_tensor.dim() == 2, "m_tensor must be 2D [batch, 64]");
    TORCH_CHECK(m_tensor.size(1) == RSA_2048_LIMBS, "m_tensor must have 64 limbs");

    int batch_size = m_tensor.size(0);

    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(m_tensor.device());
    torch::Tensor out = torch::zeros({batch_size, RSA_2048_LIMBS}, options);

    // Get raw pointers
    const uint32_t* m_ptr = reinterpret_cast<const uint32_t*>(m_tensor.data_ptr<int32_t>());
    const uint32_t* p_ptr = reinterpret_cast<const uint32_t*>(p_tensor.data_ptr<int32_t>());
    const uint32_t* q_ptr = reinterpret_cast<const uint32_t*>(q_tensor.data_ptr<int32_t>());
    const uint32_t* dp_ptr = reinterpret_cast<const uint32_t*>(dp_tensor.data_ptr<int32_t>());
    const uint32_t* dq_ptr = reinterpret_cast<const uint32_t*>(dq_tensor.data_ptr<int32_t>());
    const uint32_t* qinv_ptr = reinterpret_cast<const uint32_t*>(qinv_tensor.data_ptr<int32_t>());
    const uint32_t* R2_p_ptr = reinterpret_cast<const uint32_t*>(R2_p_tensor.data_ptr<int32_t>());
    const uint32_t* R2_q_ptr = reinterpret_cast<const uint32_t*>(R2_q_tensor.data_ptr<int32_t>());
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>());

    // Launch kernel: 1 warp per block, 1 message per block
    // This ensures no shared memory conflicts between warps
    int threads_per_block = 32;  // Exactly 1 warp
    int num_blocks = batch_size;  // 1 block per message

    rsa2048_crt_sign_warp_batch_kernel<<<num_blocks, threads_per_block>>>(
        m_ptr,
        p_ptr,
        q_ptr,
        dp_ptr,
        dq_ptr,
        qinv_ptr,
        R2_p_ptr,
        R2_q_ptr,
        (uint32_t)p_prime,
        (uint32_t)q_prime,
        out_ptr,
        batch_size);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return out;
}


// ============================================================================
// Pre-Reduced Kernel (avoids O(2^n) reduction bug)
// ============================================================================

/**
 * Batch RSA-2048 CRT sign kernel with PRE-REDUCED inputs.
 *
 * Takes m_p = m mod p and m_q = m mod q as 32-limb inputs,
 * avoiding the catastrophically slow reduce_2048_mod_1024.
 */
__global__
void rsa2048_crt_sign_prereduced_batch_kernel(
    const uint32_t* __restrict__ mp_in,     // [batch_size, 32] m mod p
    const uint32_t* __restrict__ mq_in,     // [batch_size, 32] m mod q
    const uint32_t* __restrict__ p,         // [32] prime p
    const uint32_t* __restrict__ q,         // [32] prime q
    const uint32_t* __restrict__ dp,        // [32] d mod (p-1)
    const uint32_t* __restrict__ dq,        // [32] d mod (q-1)
    const uint32_t* __restrict__ qinv,      // [32] q^(-1) mod p
    const uint32_t* __restrict__ R2_p,      // [32] R^2 mod p
    const uint32_t* __restrict__ R2_q,      // [32] R^2 mod q
    uint32_t p_prime,
    uint32_t q_prime,
    uint32_t* __restrict__ out,             // [batch_size, 64] signatures
    int batch_size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= batch_size) return;

    // Pointers to this message's data
    const uint32_t* mp = mp_in + warp_id * RSA_1024_LIMBS;
    const uint32_t* mq = mq_in + warp_id * RSA_1024_LIMBS;
    uint32_t* sig_out = out + warp_id * RSA_2048_LIMBS;

    // Local arrays for computation
    uint32_t mp_local[RSA_1024_LIMBS];
    uint32_t mq_local[RSA_1024_LIMBS];
    uint32_t p_local[RSA_1024_LIMBS];
    uint32_t q_local[RSA_1024_LIMBS];
    uint32_t dp_local[RSA_1024_LIMBS];
    uint32_t dq_local[RSA_1024_LIMBS];
    uint32_t qinv_local[RSA_1024_LIMBS];
    uint32_t R2_p_local[RSA_1024_LIMBS];
    uint32_t R2_q_local[RSA_1024_LIMBS];
    uint32_t sig_local[RSA_2048_LIMBS];

    // Load data (lane 0 loads, then broadcast)
    if (lane_id == 0) {
        // Load pre-reduced messages (1024-bit each)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            mp_local[i] = mp[i];
            mq_local[i] = mq[i];
        }

        // Load CRT parameters (1024-bit each)
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            p_local[i] = p[i];
            q_local[i] = q[i];
            dp_local[i] = dp[i];
            dq_local[i] = dq[i];
            qinv_local[i] = qinv[i];
            R2_p_local[i] = R2_p[i];
            R2_q_local[i] = R2_q[i];
        }
    }
    __syncwarp();

    // Perform CRT sign with pre-reduced inputs
    rsa_crt_sign_prereduced(
        sig_local,
        mp_local,
        mq_local,
        p_local,
        q_local,
        dp_local,
        dq_local,
        qinv_local,
        R2_p_local,
        R2_q_local,
        p_prime,
        q_prime,
        lane_id);

    // Write result
    if (lane_id == 0) {
        for (int i = 0; i < RSA_2048_LIMBS; i++) {
            sig_out[i] = sig_local[i];
        }
    }
}


/**
 * RSA-2048 CRT sign batch with pre-reduced inputs
 */
torch::Tensor rsa2048_crt_sign_prereduced_batch(
    torch::Tensor mp_tensor,      // [batch, 32] m mod p
    torch::Tensor mq_tensor,      // [batch, 32] m mod q
    torch::Tensor p_tensor,       // [32]
    torch::Tensor q_tensor,       // [32]
    torch::Tensor dp_tensor,      // [32]
    torch::Tensor dq_tensor,      // [32]
    torch::Tensor qinv_tensor,    // [32]
    torch::Tensor R2_p_tensor,    // [32]
    torch::Tensor R2_q_tensor,    // [32]
    int64_t p_prime,
    int64_t q_prime)
{
    TORCH_CHECK(mp_tensor.device().is_cuda(), "mp_tensor must be CUDA");
    TORCH_CHECK(mp_tensor.dim() == 2, "mp_tensor must be 2D [batch, 32]");
    TORCH_CHECK(mp_tensor.size(1) == RSA_1024_LIMBS, "mp_tensor must have 32 limbs");
    TORCH_CHECK(mq_tensor.size(1) == RSA_1024_LIMBS, "mq_tensor must have 32 limbs");

    int batch_size = mp_tensor.size(0);

    // Create output tensor (2048-bit = 64 limbs)
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(mp_tensor.device());
    torch::Tensor out = torch::zeros({batch_size, RSA_2048_LIMBS}, options);

    // Get raw pointers
    const uint32_t* mp_ptr = reinterpret_cast<const uint32_t*>(mp_tensor.data_ptr<int32_t>());
    const uint32_t* mq_ptr = reinterpret_cast<const uint32_t*>(mq_tensor.data_ptr<int32_t>());
    const uint32_t* p_ptr = reinterpret_cast<const uint32_t*>(p_tensor.data_ptr<int32_t>());
    const uint32_t* q_ptr = reinterpret_cast<const uint32_t*>(q_tensor.data_ptr<int32_t>());
    const uint32_t* dp_ptr = reinterpret_cast<const uint32_t*>(dp_tensor.data_ptr<int32_t>());
    const uint32_t* dq_ptr = reinterpret_cast<const uint32_t*>(dq_tensor.data_ptr<int32_t>());
    const uint32_t* qinv_ptr = reinterpret_cast<const uint32_t*>(qinv_tensor.data_ptr<int32_t>());
    const uint32_t* R2_p_ptr = reinterpret_cast<const uint32_t*>(R2_p_tensor.data_ptr<int32_t>());
    const uint32_t* R2_q_ptr = reinterpret_cast<const uint32_t*>(R2_q_tensor.data_ptr<int32_t>());
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>());

    // Launch kernel: 1 warp per message
    int threads_per_block = 128;  // 4 warps per block
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    rsa2048_crt_sign_prereduced_batch_kernel<<<num_blocks, threads_per_block>>>(
        mp_ptr,
        mq_ptr,
        p_ptr,
        q_ptr,
        dp_ptr,
        dq_ptr,
        qinv_ptr,
        R2_p_ptr,
        R2_q_ptr,
        (uint32_t)p_prime,
        (uint32_t)q_prime,
        out_ptr,
        batch_size);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return out;
}


// ============================================================================
// Test Kernel: Binary Reduction Only
// ============================================================================

/**
 * Test kernel for binary reduction: r = m mod p
 */
__global__
void test_binary_reduction_kernel(
    const uint32_t* __restrict__ m_in,      // [batch, 64] 2048-bit messages
    const uint32_t* __restrict__ p,         // [32] 1024-bit modulus
    uint32_t* __restrict__ r_out,           // [batch, 32] results
    int batch_size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= batch_size) return;

    const uint32_t* m = m_in + warp_id * RSA_2048_LIMBS;
    uint32_t* r = r_out + warp_id * RSA_1024_LIMBS;

    // Load data into local arrays
    uint32_t m_local[RSA_2048_LIMBS];
    uint32_t p_local[RSA_1024_LIMBS];
    uint32_t r_local[RSA_1024_LIMBS];

    if (lane_id == 0) {
        for (int i = 0; i < RSA_2048_LIMBS; i++) {
            m_local[i] = m[i];
        }
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            p_local[i] = p[i];
        }
    }
    __syncwarp();

    // Perform binary reduction
    reduce_2048_mod_1024_binary(r_local, m_local, p_local, lane_id);

    // Write result
    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            r[i] = r_local[i];
        }
    }
}


/**
 * Python interface for testing binary reduction
 */
torch::Tensor test_binary_reduction(
    torch::Tensor m_tensor,   // [batch, 64]
    torch::Tensor p_tensor)   // [32]
{
    TORCH_CHECK(m_tensor.device().is_cuda(), "m_tensor must be CUDA");
    TORCH_CHECK(m_tensor.dim() == 2 && m_tensor.size(1) == RSA_2048_LIMBS,
                "m_tensor must be [batch, 64]");
    TORCH_CHECK(p_tensor.size(0) == RSA_1024_LIMBS, "p_tensor must have 32 limbs");

    int batch_size = m_tensor.size(0);

    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(m_tensor.device());
    torch::Tensor out = torch::zeros({batch_size, RSA_1024_LIMBS}, options);

    const uint32_t* m_ptr = reinterpret_cast<const uint32_t*>(m_tensor.data_ptr<int32_t>());
    const uint32_t* p_ptr = reinterpret_cast<const uint32_t*>(p_tensor.data_ptr<int32_t>());
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>());

    int threads_per_block = 128;
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    test_binary_reduction_kernel<<<num_blocks, threads_per_block>>>(
        m_ptr, p_ptr, out_ptr, batch_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return out;
}


// ============================================================================
// CARD33: Warp-Cooperative Montgomery Test Kernel
// ============================================================================

/**
 * Test kernel for warp-cooperative Montgomery multiplication.
 *
 * Computes: result = a * b * R^(-1) mod n using warp-cooperative algorithm.
 * Compares against serial implementation for correctness verification.
 */
__global__
void test_warp_mont_mul_kernel(
    const uint32_t* __restrict__ a_in,      // [batch_size, 32]
    const uint32_t* __restrict__ b_in,      // [batch_size, 32]
    const uint32_t* __restrict__ n,         // [32] modulus
    uint32_t n_prime,
    uint32_t* __restrict__ out_warp,        // [batch_size, 32] warp result
    uint32_t* __restrict__ out_serial,      // [batch_size, 32] serial result
    int batch_size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= batch_size) return;

    // Pointers to this batch element's data
    const uint32_t* a = a_in + warp_id * RSA_1024_LIMBS;
    const uint32_t* b = b_in + warp_id * RSA_1024_LIMBS;
    uint32_t* result_warp = out_warp + warp_id * RSA_1024_LIMBS;
    uint32_t* result_serial = out_serial + warp_id * RSA_1024_LIMBS;

    // Load into shared memory for warp-cooperative function
    __shared__ uint32_t a_shared[RSA_1024_LIMBS];
    __shared__ uint32_t b_shared[RSA_1024_LIMBS];
    __shared__ uint32_t n_shared[RSA_1024_LIMBS];
    __shared__ uint32_t result_shared[RSA_1024_LIMBS];

    // Each lane loads one limb
    a_shared[lane_id] = a[lane_id];
    b_shared[lane_id] = b[lane_id];
    n_shared[lane_id] = n[lane_id];
    __syncwarp();

    // Warp-cooperative multiplication
    mont_mul_1024_warp(result_shared, a_shared, b_shared, n_shared, n_prime);

    // Write warp result
    result_warp[lane_id] = result_shared[lane_id];
    __syncwarp();

    // Serial multiplication (for comparison)
    uint32_t a_local[RSA_1024_LIMBS];
    uint32_t b_local[RSA_1024_LIMBS];
    uint32_t n_local[RSA_1024_LIMBS];
    uint32_t result_local[RSA_1024_LIMBS];

    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            a_local[i] = a[i];
            b_local[i] = b[i];
            n_local[i] = n[i];
        }
    }
    __syncwarp();

    mont_mul_1024_serial(result_local, a_local, b_local, n_local, n_prime, lane_id);

    // Write serial result
    if (lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            result_serial[i] = result_local[i];
        }
    }
}


/**
 * Python wrapper for warp Montgomery test.
 */
std::tuple<torch::Tensor, torch::Tensor> test_warp_mont_mul(
    torch::Tensor a_tensor,   // [batch, 32]
    torch::Tensor b_tensor,   // [batch, 32]
    torch::Tensor n_tensor,   // [32]
    int64_t n_prime)
{
    TORCH_CHECK(a_tensor.device().is_cuda(), "a_tensor must be CUDA");
    TORCH_CHECK(a_tensor.dim() == 2 && a_tensor.size(1) == RSA_1024_LIMBS,
                "a_tensor must be [batch, 32]");
    TORCH_CHECK(b_tensor.dim() == 2 && b_tensor.size(1) == RSA_1024_LIMBS,
                "b_tensor must be [batch, 32]");
    TORCH_CHECK(n_tensor.size(0) == RSA_1024_LIMBS, "n_tensor must have 32 limbs");

    int batch_size = a_tensor.size(0);

    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(a_tensor.device());
    torch::Tensor out_warp = torch::zeros({batch_size, RSA_1024_LIMBS}, options);
    torch::Tensor out_serial = torch::zeros({batch_size, RSA_1024_LIMBS}, options);

    const uint32_t* a_ptr = reinterpret_cast<const uint32_t*>(a_tensor.data_ptr<int32_t>());
    const uint32_t* b_ptr = reinterpret_cast<const uint32_t*>(b_tensor.data_ptr<int32_t>());
    const uint32_t* n_ptr = reinterpret_cast<const uint32_t*>(n_tensor.data_ptr<int32_t>());
    uint32_t* out_warp_ptr = reinterpret_cast<uint32_t*>(out_warp.data_ptr<int32_t>());
    uint32_t* out_serial_ptr = reinterpret_cast<uint32_t*>(out_serial.data_ptr<int32_t>());

    int threads_per_block = 32;  // One warp per block for simplicity
    int num_blocks = batch_size;

    test_warp_mont_mul_kernel<<<num_blocks, threads_per_block>>>(
        a_ptr, b_ptr, n_ptr, (uint32_t)n_prime,
        out_warp_ptr, out_serial_ptr, batch_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return std::make_tuple(out_warp, out_serial);
}


// ============================================================================
// CARD35C.0: p||q Modexp Debug Harness
// ============================================================================
//
// Test kernel for verifying p||q parallel structure before full CRT.
// One block == one message, 2 warps:
//   Warp 0 -> p-branch: mp = m mod p, sp = m^dp mod p
//   Warp 1 -> q-branch: mq = m mod q, sq = m^dq mod q
//

__global__
void test_pq_modexp_kernel(
    const uint32_t* __restrict__ m_limbs,   // [RSA_2048_LIMBS] single message
    const uint32_t* __restrict__ p,         // [RSA_1024_LIMBS]
    const uint32_t* __restrict__ q,         // [RSA_1024_LIMBS]
    const uint32_t* __restrict__ dp,        // [RSA_1024_LIMBS]
    const uint32_t* __restrict__ dq,        // [RSA_1024_LIMBS]
    const uint32_t* __restrict__ R2_p,      // [RSA_1024_LIMBS]
    const uint32_t* __restrict__ R2_q,      // [RSA_1024_LIMBS]
    uint32_t p_prime,
    uint32_t q_prime,
    uint32_t* __restrict__ sp_out,          // [RSA_1024_LIMBS]
    uint32_t* __restrict__ sq_out           // [RSA_1024_LIMBS]
) {
    using namespace rsa2048_crt;

    int lane_id = threadIdx.x & 31;  // 0..31
    int warp_id = threadIdx.x >> 5;  // 0 or 1

    // Shared 1024-bit buffers for each branch
    __shared__ uint32_t mp_shared[RSA_1024_LIMBS];
    __shared__ uint32_t mq_shared[RSA_1024_LIMBS];
    __shared__ uint32_t sp_shared[RSA_1024_LIMBS];
    __shared__ uint32_t sq_shared[RSA_1024_LIMBS];

    // --- Warp 0: p-branch ---
    if (warp_id == 0) {
        // 1) Reduce m -> mp = m mod p
        reduce_2048_mod_1024_binary(mp_shared, m_limbs, p, lane_id);

        // 2) Modexp: sp = mp^dp mod p
#ifdef USE_WINDOW5_MODEXP
        modexp_1024_window5(sp_shared, mp_shared, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
#elif defined(USE_WINDOW_MODEXP)
        modexp_1024_window4(sp_shared, mp_shared, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
#else
        modexp_1024(sp_shared, mp_shared, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
#endif
    }

    // --- Warp 1: q-branch ---
    if (warp_id == 1) {
        // 1) Reduce m -> mq = m mod q
        reduce_2048_mod_1024_binary(mq_shared, m_limbs, q, lane_id);

        // 2) Modexp: sq = mq^dq mod q
#ifdef USE_WINDOW5_MODEXP
        modexp_1024_window5(sq_shared, mq_shared, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#elif defined(USE_WINDOW_MODEXP)
        modexp_1024_window4(sq_shared, mq_shared, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#else
        modexp_1024(sq_shared, mq_shared, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#endif
    }

    // Make sure both warps are done before we write out
    __syncthreads();

    // Lane 0 of each warp copies result back to global for Python comparison
    if (warp_id == 0 && lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            sp_out[i] = sp_shared[i];
        }
    }
    if (warp_id == 1 && lane_id == 0) {
        for (int i = 0; i < RSA_1024_LIMBS; i++) {
            sq_out[i] = sq_shared[i];
        }
    }
}

// PyTorch wrapper for test_pq_modexp_kernel
std::tuple<torch::Tensor, torch::Tensor> test_pq_modexp(
    torch::Tensor m_tensor,      // [64] int32 - single 2048-bit message
    torch::Tensor p_tensor,      // [32] int32
    torch::Tensor q_tensor,      // [32] int32
    torch::Tensor dp_tensor,     // [32] int32
    torch::Tensor dq_tensor,     // [32] int32
    torch::Tensor R2_p_tensor,   // [32] int32
    torch::Tensor R2_q_tensor,   // [32] int32
    int64_t p_prime,
    int64_t q_prime
) {
    TORCH_CHECK(m_tensor.device().is_cuda(), "m must be on CUDA");
    TORCH_CHECK(m_tensor.numel() == RSA_2048_LIMBS, "m must have 64 limbs");

    auto device = m_tensor.device();

    // Output tensors
    auto sp_out = torch::empty({RSA_1024_LIMBS}, torch::dtype(torch::kInt32).device(device));
    auto sq_out = torch::empty({RSA_1024_LIMBS}, torch::dtype(torch::kInt32).device(device));

    const uint32_t* m_ptr = reinterpret_cast<const uint32_t*>(m_tensor.data_ptr<int32_t>());
    const uint32_t* p_ptr = reinterpret_cast<const uint32_t*>(p_tensor.data_ptr<int32_t>());
    const uint32_t* q_ptr = reinterpret_cast<const uint32_t*>(q_tensor.data_ptr<int32_t>());
    const uint32_t* dp_ptr = reinterpret_cast<const uint32_t*>(dp_tensor.data_ptr<int32_t>());
    const uint32_t* dq_ptr = reinterpret_cast<const uint32_t*>(dq_tensor.data_ptr<int32_t>());
    const uint32_t* R2_p_ptr = reinterpret_cast<const uint32_t*>(R2_p_tensor.data_ptr<int32_t>());
    const uint32_t* R2_q_ptr = reinterpret_cast<const uint32_t*>(R2_q_tensor.data_ptr<int32_t>());
    uint32_t* sp_ptr = reinterpret_cast<uint32_t*>(sp_out.data_ptr<int32_t>());
    uint32_t* sq_ptr = reinterpret_cast<uint32_t*>(sq_out.data_ptr<int32_t>());

    // Launch with 2 warps (64 threads), 1 block (1 message)
    dim3 block(64);
    dim3 grid(1);

    test_pq_modexp_kernel<<<grid, block>>>(
        m_ptr, p_ptr, q_ptr, dp_ptr, dq_ptr, R2_p_ptr, R2_q_ptr,
        (uint32_t)p_prime, (uint32_t)q_prime,
        sp_ptr, sq_ptr
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return std::make_tuple(sp_out, sq_out);
}


// ============================================================================
// CARD35C.1: Full p||q CRT Sign Kernel (2 warps per message)
// ============================================================================
//
// Block layout:
//   blockDim.x = 64 (2 warps)
//   warp 0 -> p-branch + CRT recombination
//   warp 1 -> q-branch
//
// One block handles ONE message. We launch gridDim.x = batch_size.
//

__global__
void rsa2048_crt_sign_warp_pq_kernel(
    const uint32_t* __restrict__ m_arr,      // [batch][64] messages
    const uint32_t* __restrict__ p,          // [32] prime p
    const uint32_t* __restrict__ q,          // [32] prime q
    const uint32_t* __restrict__ dp,         // [32] d mod (p-1)
    const uint32_t* __restrict__ dq,         // [32] d mod (q-1)
    const uint32_t* __restrict__ qinv,       // [32] q^(-1) mod p
    const uint32_t* __restrict__ R2_p,       // [32] R^2 mod p
    const uint32_t* __restrict__ R2_q,       // [32] R^2 mod q
    uint32_t p_prime,
    uint32_t q_prime,
    uint32_t* __restrict__ sig_arr,          // [batch][64] signatures
    int batch_size
) {
    using namespace rsa2048_crt;

    int lane_id = threadIdx.x & 31;   // 0..31
    int warp_id = threadIdx.x >> 5;   // 0 or 1
    int msg_idx = blockIdx.x;         // which message

    if (msg_idx >= batch_size) return;

    // Stride into message array (64 limbs per 2048-bit)
    const uint32_t* m_limbs = m_arr + msg_idx * RSA_2048_LIMBS;
    uint32_t* s_limbs = sig_arr + msg_idx * RSA_2048_LIMBS;

    // Shared 1024-bit buffers for each branch
    __shared__ uint32_t sp_shared[RSA_1024_LIMBS];
    __shared__ uint32_t sq_shared[RSA_1024_LIMBS];

    // --- Phase 1: p/q branches in parallel ---

    if (warp_id == 0) {
        // p-branch: mp = m mod p, sp = mp^dp mod p
        uint32_t mp_local[RSA_1024_LIMBS];
        reduce_2048_mod_1024_binary(mp_local, m_limbs, p, lane_id);

#ifdef USE_WINDOW5_MODEXP
        modexp_1024_window5(sp_shared, mp_local, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
#elif defined(USE_WINDOW_MODEXP)
        modexp_1024_window4(sp_shared, mp_local, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
#else
        modexp_1024(sp_shared, mp_local, dp, RSA_1024_BITS, p, R2_p, p_prime, lane_id);
#endif
    }

    if (warp_id == 1) {
        // q-branch: mq = m mod q, sq = mq^dq mod q
        uint32_t mq_local[RSA_1024_LIMBS];
        reduce_2048_mod_1024_binary(mq_local, m_limbs, q, lane_id);

#ifdef USE_WINDOW5_MODEXP
        modexp_1024_window5(sq_shared, mq_local, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#elif defined(USE_WINDOW_MODEXP)
        modexp_1024_window4(sq_shared, mq_local, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#else
        modexp_1024(sq_shared, mq_local, dq, RSA_1024_BITS, q, R2_q, q_prime, lane_id);
#endif
    }

    // Make sure both warps have finished sp/sq
    __syncthreads();

    // --- Phase 2: CRT recombination (warp 0 only) ---
    //
    // The helpers (mod_sub_1024, to_mont_1024, etc.) use __syncwarp() internally,
    // which is fine since all 32 threads of warp 0 execute together.
    //
    if (warp_id == 0) {
        // Step 1: diff = (sp - sq) mod p
        uint32_t diff[RSA_1024_LIMBS];
        mod_sub_1024(diff, sp_shared, sq_shared, p, lane_id);

        // Step 2: h = (qinv * diff) mod p using Montgomery multiplication
        uint32_t diff_mont[RSA_1024_LIMBS];
        uint32_t qinv_mont[RSA_1024_LIMBS];
        uint32_t h_mont[RSA_1024_LIMBS];
        uint32_t h[RSA_1024_LIMBS];

        to_mont_1024(diff_mont, diff, R2_p, p, p_prime, lane_id);
        to_mont_1024(qinv_mont, qinv, R2_p, p, p_prime, lane_id);
        mont_mul_1024_serial(h_mont, qinv_mont, diff_mont, p, p_prime, lane_id);
        from_mont_1024(h, h_mont, p, p_prime, lane_id);

        // Step 3: qh = q * h (2048-bit result)
        uint32_t qh[RSA_2048_LIMBS];
        mul_1024x1024_to_2048(qh, q, h, lane_id);

        // Step 4: s = sq + q*h
        // First extend sq to 2048 bits
        uint32_t sq_ext[RSA_2048_LIMBS];
        if (lane_id == 0) {
            for (int i = 0; i < RSA_1024_LIMBS; i++) {
                sq_ext[i] = sq_shared[i];
            }
            for (int i = RSA_1024_LIMBS; i < RSA_2048_LIMBS; i++) {
                sq_ext[i] = 0;
            }
        }
        __syncwarp();

        uint32_t sig_local[RSA_2048_LIMBS];
        add_2048(sig_local, sq_ext, qh, lane_id);

        // Write signature to global memory
        if (lane_id == 0) {
            for (int i = 0; i < RSA_2048_LIMBS; i++) {
                s_limbs[i] = sig_local[i];
            }
        }
    }
}


/**
 * CARD35C.1: PyTorch wrapper for p||q CRT sign batch.
 */
torch::Tensor rsa2048_crt_sign_warp_pq_batch(
    torch::Tensor m_tensor,       // [batch, 64]
    torch::Tensor p_tensor,       // [32]
    torch::Tensor q_tensor,       // [32]
    torch::Tensor dp_tensor,      // [32]
    torch::Tensor dq_tensor,      // [32]
    torch::Tensor qinv_tensor,    // [32]
    torch::Tensor R2_p_tensor,    // [32]
    torch::Tensor R2_q_tensor,    // [32]
    int64_t p_prime,
    int64_t q_prime)
{
    TORCH_CHECK(m_tensor.device().is_cuda(), "m_tensor must be CUDA");
    TORCH_CHECK(m_tensor.dim() == 2, "m_tensor must be 2D [batch, 64]");
    TORCH_CHECK(m_tensor.size(1) == RSA_2048_LIMBS, "m_tensor must have 64 limbs");

    int batch_size = m_tensor.size(0);

    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(m_tensor.device());
    torch::Tensor out = torch::zeros({batch_size, RSA_2048_LIMBS}, options);

    // Get raw pointers
    const uint32_t* m_ptr = reinterpret_cast<const uint32_t*>(m_tensor.data_ptr<int32_t>());
    const uint32_t* p_ptr = reinterpret_cast<const uint32_t*>(p_tensor.data_ptr<int32_t>());
    const uint32_t* q_ptr = reinterpret_cast<const uint32_t*>(q_tensor.data_ptr<int32_t>());
    const uint32_t* dp_ptr = reinterpret_cast<const uint32_t*>(dp_tensor.data_ptr<int32_t>());
    const uint32_t* dq_ptr = reinterpret_cast<const uint32_t*>(dq_tensor.data_ptr<int32_t>());
    const uint32_t* qinv_ptr = reinterpret_cast<const uint32_t*>(qinv_tensor.data_ptr<int32_t>());
    const uint32_t* R2_p_ptr = reinterpret_cast<const uint32_t*>(R2_p_tensor.data_ptr<int32_t>());
    const uint32_t* R2_q_ptr = reinterpret_cast<const uint32_t*>(R2_q_tensor.data_ptr<int32_t>());
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>());

    // Launch kernel: 2 warps per block (64 threads), 1 message per block
    dim3 block(64);
    dim3 grid(batch_size);

    rsa2048_crt_sign_warp_pq_kernel<<<grid, block>>>(
        m_ptr,
        p_ptr,
        q_ptr,
        dp_ptr,
        dq_ptr,
        qinv_ptr,
        R2_p_ptr,
        R2_q_ptr,
        (uint32_t)p_prime,
        (uint32_t)q_prime,
        out_ptr,
        batch_size);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return out;
}


// ============================================================================
// Module Registration
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RSA-2048 CRT GPU modular exponentiation (CARD35C: warp-pq parallel)";

    m.def("crt_sign_batch",
          &rsa2048_crt_sign_batch,
          "RSA-2048 CRT sign batch: s = m^d mod n (GPU-native serial)",
          py::arg("m"),
          py::arg("p"),
          py::arg("q"),
          py::arg("dp"),
          py::arg("dq"),
          py::arg("qinv"),
          py::arg("R2_p"),
          py::arg("R2_q"),
          py::arg("p_prime"),
          py::arg("q_prime"));

    m.def("crt_sign_warp_batch",
          &rsa2048_crt_sign_warp_batch,
          "CARD33A: RSA-2048 CRT sign batch (warp-cooperative Montgomery)",
          py::arg("m"),
          py::arg("p"),
          py::arg("q"),
          py::arg("dp"),
          py::arg("dq"),
          py::arg("qinv"),
          py::arg("R2_p"),
          py::arg("R2_q"),
          py::arg("p_prime"),
          py::arg("q_prime"));

    m.def("crt_sign_prereduced_batch",
          &rsa2048_crt_sign_prereduced_batch,
          "RSA-2048 CRT sign batch with pre-reduced inputs (CPU prereduce)",
          py::arg("mp"),
          py::arg("mq"),
          py::arg("p"),
          py::arg("q"),
          py::arg("dp"),
          py::arg("dq"),
          py::arg("qinv"),
          py::arg("R2_p"),
          py::arg("R2_q"),
          py::arg("p_prime"),
          py::arg("q_prime"));

    m.def("test_binary_reduction",
          &test_binary_reduction,
          "Test binary reduction: r = m mod p",
          py::arg("m"),
          py::arg("p"));

    m.def("test_warp_mont_mul",
          &test_warp_mont_mul,
          "Test warp-cooperative Montgomery mul vs serial",
          py::arg("a"),
          py::arg("b"),
          py::arg("n"),
          py::arg("n_prime"));

    m.def("test_pq_modexp",
          &test_pq_modexp,
          "CARD35C.0: p||q modexp debug harness (2 warps parallel)",
          py::arg("m"),
          py::arg("p"),
          py::arg("q"),
          py::arg("dp"),
          py::arg("dq"),
          py::arg("R2_p"),
          py::arg("R2_q"),
          py::arg("p_prime"),
          py::arg("q_prime"));

    m.def("crt_sign_warp_pq_batch",
          &rsa2048_crt_sign_warp_pq_batch,
          "CARD35C.1: RSA-2048 CRT sign batch (2 warps: p||q parallel)",
          py::arg("m"),
          py::arg("p"),
          py::arg("q"),
          py::arg("dp"),
          py::arg("dq"),
          py::arg("qinv"),
          py::arg("R2_p"),
          py::arg("R2_q"),
          py::arg("p_prime"),
          py::arg("q_prime"));
}
