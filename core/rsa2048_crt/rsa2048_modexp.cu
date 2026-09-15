/**
 * CARD30-C: RSA-2048 Modular Exponentiation Kernel (v2 - Correct)
 *
 * Batch GPU kernel for RSA-2048 sign/verify operations.
 * Each warp handles one modular exponentiation.
 *
 * Build:
 *   python build_rsa2048.py build_ext --inplace
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Include Montgomery primitives
#include "rsa2048_mont.cuh"

using namespace rsa2048;

// ============================================================================
// Batch Modular Exponentiation Kernel
// ============================================================================

/**
 * Batch RSA-2048 modexp kernel.
 *
 * Each warp processes one message.
 * Lane 0 does the computation, other lanes broadcast results.
 */
__global__
void rsa2048_modexp_batch_kernel(
    const uint32_t* __restrict__ m_in,
    const uint32_t* __restrict__ exp,
    int exp_bits,
    const uint32_t* __restrict__ n,
    const uint32_t* __restrict__ R2,
    uint32_t n_prime,
    uint32_t* __restrict__ out,
    int batch_size)
{
    // Each warp handles one message
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= batch_size) return;

    // Pointers to this message's data
    const uint32_t* m = m_in + warp_id * RSA_LIMBS;
    uint32_t* result_out = out + warp_id * RSA_LIMBS;

    // Local arrays for this warp's computation
    uint32_t m_local[RSA_LIMBS];
    uint32_t n_local[RSA_LIMBS];
    uint32_t R2_local[RSA_LIMBS];
    uint32_t exp_local[RSA_LIMBS];
    uint32_t result_local[RSA_LIMBS];

    // Load data (all lanes load for coalescing, but only lane 0 uses)
    if (lane_id == 0) {
        for (int i = 0; i < RSA_LIMBS; i++) {
            m_local[i] = m[i];
            n_local[i] = n[i];
            R2_local[i] = R2[i];
            exp_local[i] = exp[i];
        }
    }
    __syncwarp();

    // Perform modular exponentiation
    modexp_2048(
        result_local,
        m_local,
        exp_local,
        exp_bits,
        n_local,
        R2_local,
        n_prime,
        lane_id);

    // Write result (only lane 0 has correct values after broadcast)
    if (lane_id == 0) {
        for (int i = 0; i < RSA_LIMBS; i++) {
            result_out[i] = result_local[i];
        }
    }
}


/**
 * RSA-2048 verify kernel (optimized for e=65537).
 */
__global__
void rsa2048_verify_batch_kernel(
    const uint32_t* __restrict__ s_in,
    uint32_t e,
    const uint32_t* __restrict__ n,
    const uint32_t* __restrict__ R2,
    uint32_t n_prime,
    uint32_t* __restrict__ out,
    int batch_size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= batch_size) return;

    const uint32_t* s = s_in + warp_id * RSA_LIMBS;
    uint32_t* result_out = out + warp_id * RSA_LIMBS;

    uint32_t s_local[RSA_LIMBS];
    uint32_t n_local[RSA_LIMBS];
    uint32_t R2_local[RSA_LIMBS];
    uint32_t result_local[RSA_LIMBS];

    if (lane_id == 0) {
        for (int i = 0; i < RSA_LIMBS; i++) {
            s_local[i] = s[i];
            n_local[i] = n[i];
            R2_local[i] = R2[i];
        }
    }
    __syncwarp();

    rsa_verify_2048(
        result_local,
        s_local,
        e,
        n_local,
        R2_local,
        n_prime,
        lane_id);

    if (lane_id == 0) {
        for (int i = 0; i < RSA_LIMBS; i++) {
            result_out[i] = result_local[i];
        }
    }
}


// ============================================================================
// PyTorch Interface
// ============================================================================

/**
 * RSA-2048 sign batch: s = m^d mod n
 */
torch::Tensor rsa2048_sign_batch(
    torch::Tensor m_tensor,
    torch::Tensor d_tensor,
    torch::Tensor n_tensor,
    torch::Tensor R2_tensor,
    int64_t n_prime)
{
    TORCH_CHECK(m_tensor.device().is_cuda(), "m_tensor must be CUDA");
    TORCH_CHECK(m_tensor.dim() == 2, "m_tensor must be 2D [batch, 64]");
    TORCH_CHECK(m_tensor.size(1) == RSA_LIMBS, "m_tensor must have 64 limbs");

    int batch_size = m_tensor.size(0);

    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(m_tensor.device());
    torch::Tensor out = torch::zeros({batch_size, RSA_LIMBS}, options);

    // Get raw pointers
    const uint32_t* m_ptr = reinterpret_cast<const uint32_t*>(m_tensor.data_ptr<int32_t>());
    const uint32_t* d_ptr = reinterpret_cast<const uint32_t*>(d_tensor.data_ptr<int32_t>());
    const uint32_t* n_ptr = reinterpret_cast<const uint32_t*>(n_tensor.data_ptr<int32_t>());
    const uint32_t* R2_ptr = reinterpret_cast<const uint32_t*>(R2_tensor.data_ptr<int32_t>());
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>());

    // Launch kernel: 1 warp per message
    int threads_per_block = 128;  // 4 warps per block
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    rsa2048_modexp_batch_kernel<<<num_blocks, threads_per_block>>>(
        m_ptr,
        d_ptr,
        RSA_BITS,
        n_ptr,
        R2_ptr,
        (uint32_t)n_prime,
        out_ptr,
        batch_size);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }

    return out;
}


/**
 * RSA-2048 verify batch: m' = s^e mod n
 */
torch::Tensor rsa2048_verify_batch(
    torch::Tensor s_tensor,
    int64_t e,
    torch::Tensor n_tensor,
    torch::Tensor R2_tensor,
    int64_t n_prime)
{
    TORCH_CHECK(s_tensor.device().is_cuda(), "s_tensor must be CUDA");
    TORCH_CHECK(s_tensor.dim() == 2, "s_tensor must be 2D [batch, 64]");
    TORCH_CHECK(s_tensor.size(1) == RSA_LIMBS, "s_tensor must have 64 limbs");

    int batch_size = s_tensor.size(0);

    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(s_tensor.device());
    torch::Tensor out = torch::zeros({batch_size, RSA_LIMBS}, options);

    const uint32_t* s_ptr = reinterpret_cast<const uint32_t*>(s_tensor.data_ptr<int32_t>());
    const uint32_t* n_ptr = reinterpret_cast<const uint32_t*>(n_tensor.data_ptr<int32_t>());
    const uint32_t* R2_ptr = reinterpret_cast<const uint32_t*>(R2_tensor.data_ptr<int32_t>());
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>());

    int threads_per_block = 128;
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    rsa2048_verify_batch_kernel<<<num_blocks, threads_per_block>>>(
        s_ptr,
        (uint32_t)e,
        n_ptr,
        R2_ptr,
        (uint32_t)n_prime,
        out_ptr,
        batch_size);

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
    m.doc() = "RSA-2048 GPU modular exponentiation";

    m.def("sign_batch",
          &rsa2048_sign_batch,
          "RSA-2048 sign batch: s = m^d mod n",
          py::arg("m"),
          py::arg("d"),
          py::arg("n"),
          py::arg("R2"),
          py::arg("n_prime"));

    m.def("verify_batch",
          &rsa2048_verify_batch,
          "RSA-2048 verify batch: m' = s^e mod n",
          py::arg("s"),
          py::arg("e"),
          py::arg("n"),
          py::arg("R2"),
          py::arg("n_prime"));
}
