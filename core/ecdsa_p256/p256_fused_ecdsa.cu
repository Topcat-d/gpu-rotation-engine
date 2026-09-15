/*
 * CARD23: P-256 Fused ECDSA Signing Kernel
 *
 * Combines scalar multiplication (k*G) and signature computation (r, s)
 * into a single API call, eliminating intermediate tensor transfers.
 *
 * Input: k (nonce), h (message hash), d (private key)
 * Output: r, s (signature)
 *
 * Flow:
 *   1. GPU scalar mult: k*G -> (x, y)
 *   2. GPU mod-n: r = x mod n, s = k_inv * (h + r*d) mod n
 *   3. Return (r, s)
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

// Include the mod-n operations
#include "p256_scalar_order.cuh"

// Forward declarations from p256_sign_persistent module
// We'll call these via Python interop rather than linking directly
// This keeps the modules separate but provides a fused Python API

// ============================================================================
// ECDSA Sign Kernel (same as p256_ecdsa_sign.cu)
// ============================================================================

__device__ __forceinline__
bool ecdsa_sign_from_point_fused(
    uint32_t r_out[8],
    uint32_t s_out[8],
    const uint32_t x_R[8],
    const uint32_t k[8],
    const uint32_t h[8],
    const uint32_t d[8]
)
{
    // Step 1: r = x_R mod n
    uint32_t r[8];
    reduce_x_mod_n(r, x_R);

    if (is_zero_scalar(r)) {
        for (int i = 0; i < 8; ++i) { r_out[i] = 0; s_out[i] = 0; }
        return false;
    }

    // Step 2: Convert to Montgomery domain
    uint32_t k_mont[8], h_mont[8], d_mont[8], r_mont[8];
    to_mont_n(k_mont, k);
    to_mont_n(h_mont, h);
    to_mont_n(d_mont, d);
    to_mont_n(r_mont, r);

    // Step 3: k_inv = k^(-1) mod n
    uint32_t k_inv[8];
    mont_fermat_inv_n(k_inv, k_mont);

    // Step 4: rd = r * d mod n
    uint32_t rd[8];
    mont_mul_scalar_n(rd, r_mont, d_mont);

    // Step 5: z = h + r*d mod n
    uint32_t z[8];
    mont_add_scalar_n(z, h_mont, rd);

    // Step 6: s = k_inv * z mod n
    uint32_t s_mont[8];
    mont_mul_scalar_n(s_mont, k_inv, z);

    // Step 7: Convert s back to normal domain
    uint32_t s[8];
    from_mont_n(s, s_mont);

    if (is_zero_scalar(s)) {
        for (int i = 0; i < 8; ++i) { r_out[i] = 0; s_out[i] = 0; }
        return false;
    }

    for (int i = 0; i < 8; ++i) {
        r_out[i] = r[i];
        s_out[i] = s[i];
    }

    return true;
}

// ============================================================================
// Fused ECDSA Kernel (Phase 2: after scalar mult)
// ============================================================================

__global__
void fused_ecdsa_sign_phase2_kernel(
    const int32_t* __restrict__ k_batch,
    const int32_t* __restrict__ h_batch,
    const int32_t* __restrict__ d_batch,
    const int32_t* __restrict__ x_batch,  // x from k*G
    int32_t* __restrict__ r_out,
    int32_t* __restrict__ s_out,
    int batch_size
)
{
    int warp_id_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;

    if (warp_id_g >= batch_size) return;

    // Load inputs
    uint32_t k_limb = 0, h_limb = 0, d_limb = 0, x_limb = 0;
    if (lane_id < 8) {
        k_limb = (uint32_t)k_batch[warp_id_g * 8 + lane_id];
        h_limb = (uint32_t)h_batch[warp_id_g * 8 + lane_id];
        d_limb = (uint32_t)d_batch[warp_id_g * 8 + lane_id];
        x_limb = (uint32_t)x_batch[warp_id_g * 8 + lane_id];
    }
    __syncwarp(0xFFFFFFFF);

    // Gather all limbs
    uint32_t k[8], h[8], d[8], x_R[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        k[i] = __shfl_sync(0xFFFFFFFF, k_limb, i);
        h[i] = __shfl_sync(0xFFFFFFFF, h_limb, i);
        d[i] = __shfl_sync(0xFFFFFFFF, d_limb, i);
        x_R[i] = __shfl_sync(0xFFFFFFFF, x_limb, i);
    }

    // Compute signature on lane 0
    uint32_t r[8] = {0}, s[8] = {0};
    if (lane_id == 0) {
        ecdsa_sign_from_point_fused(r, s, x_R, k, h, d);
    }
    __syncwarp(0xFFFFFFFF);

    // Store results
    for (int i = 0; i < 8; ++i) {
        uint32_t r_i = (lane_id == 0) ? r[i] : 0;
        uint32_t s_i = (lane_id == 0) ? s[i] : 0;
        r_i = __shfl_sync(0xFFFFFFFF, r_i, 0);
        s_i = __shfl_sync(0xFFFFFFFF, s_i, 0);

        if (lane_id == i) {
            r_out[warp_id_g * 8 + lane_id] = (int32_t)r_i;
            s_out[warp_id_g * 8 + lane_id] = (int32_t)s_i;
        }
    }
}

// ============================================================================
// PyTorch Binding
// ============================================================================

/*
 * Fused ECDSA Sign (Phase 2 only - after scalar mult)
 *
 * This is the same as ecdsa_sign_batch but packaged for the fused API.
 * The full fusion happens at the Python level by combining this with
 * the scalar mult in a single call.
 */
torch::Tensor fused_ecdsa_sign_phase2(
    torch::Tensor k_tensor,
    torch::Tensor h_tensor,
    torch::Tensor d_tensor,
    torch::Tensor x_tensor
)
{
    TORCH_CHECK(k_tensor.dim() == 2, "k must be 2D [batch, 8]");
    TORCH_CHECK(h_tensor.dim() == 2, "h must be 2D [batch, 8]");
    TORCH_CHECK(d_tensor.dim() == 2, "d must be 2D [batch, 8]");
    TORCH_CHECK(x_tensor.dim() == 2, "x must be 2D [batch, 8]");

    int batch_size = k_tensor.size(0);

    auto k_cuda = k_tensor.to(torch::kCUDA).contiguous();
    auto h_cuda = h_tensor.to(torch::kCUDA).contiguous();
    auto d_cuda = d_tensor.to(torch::kCUDA).contiguous();
    auto x_cuda = x_tensor.to(torch::kCUDA).contiguous();

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto r_tensor = torch::zeros({batch_size, 8}, options);
    auto s_tensor = torch::zeros({batch_size, 8}, options);

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    fused_ecdsa_sign_phase2_kernel<<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        k_cuda.data_ptr<int32_t>(),
        h_cuda.data_ptr<int32_t>(),
        d_cuda.data_ptr<int32_t>(),
        x_cuda.data_ptr<int32_t>(),
        r_tensor.data_ptr<int32_t>(),
        s_tensor.data_ptr<int32_t>(),
        batch_size
    );

    return torch::stack({r_tensor, s_tensor}, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ecdsa_sign_phase2", &fused_ecdsa_sign_phase2,
          "CARD23: Fused ECDSA Sign Phase 2 (after scalar mult)");
}
