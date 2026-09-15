/*
 * P-256 ECDSA Signature Generation - CARD20
 *
 * Full ECDSA sign pipeline on GPU:
 *   Given (k, h, d):
 *     1. R = k*G (scalar multiplication)
 *     2. r = x(R) mod n
 *     3. k_inv = k^(-1) mod n
 *     4. s = k_inv * (h + r*d) mod n
 *     5. Return signature (r, s)
 *
 * Dependencies:
 *   - CARD19: 256-bit scalar correctness
 *   - CARD17: Single Montgomery domain table
 *   - Existing scalar multiplication kernel
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

// Include scalar multiplication functionality (reuse existing infrastructure)
#include "p256_scalar_order.cuh"

// Forward declaration of scalar multiplication
// (We'll reuse the existing persistent_sign_batch infrastructure)

// ============================================================================
// Debug Macro
// ============================================================================
// #define ECDSA_DEBUG 1

#if defined(ECDSA_DEBUG)
  #define ECDSA_DBG(...) printf(__VA_ARGS__)
#else
  #define ECDSA_DBG(...) do {} while (0)
#endif

// ============================================================================
// ECDSA Sign Kernel (Scalar Reference Implementation)
// ============================================================================
// This version uses lane-0 scalar operations for correctness.
// A warp-cooperative version can be optimized later.

/*
 * ECDSA Sign - Single Signature (Lane 0 Only)
 *
 * Computes signature (r, s) for a single message.
 * Called by lane 0 after scalar multiplication completes.
 *
 * Inputs (all in NORMAL domain, not Montgomery):
 *   x_R[8]: x-coordinate of R = k*G (from scalar multiply)
 *   k[8]: nonce (256-bit)
 *   h[8]: message hash (256-bit)
 *   d[8]: private key (256-bit)
 *
 * Outputs (in NORMAL domain):
 *   r_out[8]: r component of signature
 *   s_out[8]: s component of signature
 *
 * Returns: true if signature is valid, false if r=0 or s=0
 */
__device__ __forceinline__
bool ecdsa_sign_from_point(
    uint32_t r_out[8],
    uint32_t s_out[8],
    const uint32_t x_R[8],
    const uint32_t k[8],
    const uint32_t h[8],
    const uint32_t d[8]
)
{
    // Step 1: r = x_R mod n (already in normal domain)
    uint32_t r[8];
    reduce_x_mod_n(r, x_R);

    // Check r == 0 (invalid signature)
    if (is_zero_scalar(r)) {
        for (int i = 0; i < 8; ++i) { r_out[i] = 0; s_out[i] = 0; }
        return false;
    }

    ECDSA_DBG("[ECDSA] r[0]=%08x r[7]=%08x\n", r[0], r[7]);

    // Step 2: Convert k, h, d, r to Montgomery domain mod n
    uint32_t k_mont[8], h_mont[8], d_mont[8], r_mont[8];
    to_mont_n(k_mont, k);
    to_mont_n(h_mont, h);
    to_mont_n(d_mont, d);
    to_mont_n(r_mont, r);

    ECDSA_DBG("[ECDSA] k_mont[0]=%08x h_mont[0]=%08x\n", k_mont[0], h_mont[0]);

    // Step 3: k_inv = k^(-1) mod n (Fermat inverse in Montgomery domain)
    uint32_t k_inv[8];
    mont_fermat_inv_n(k_inv, k_mont);

    ECDSA_DBG("[ECDSA] k_inv[0]=%08x\n", k_inv[0]);

    // Step 4: rd = r * d mod n (Montgomery multiplication)
    uint32_t rd[8];
    mont_mul_scalar_n(rd, r_mont, d_mont);

    ECDSA_DBG("[ECDSA] rd[0]=%08x\n", rd[0]);

    // Step 5: z = h + r*d mod n (Montgomery addition)
    uint32_t z[8];
    mont_add_scalar_n(z, h_mont, rd);

    ECDSA_DBG("[ECDSA] z[0]=%08x\n", z[0]);

    // Step 6: s = k_inv * z mod n (Montgomery multiplication)
    uint32_t s_mont[8];
    mont_mul_scalar_n(s_mont, k_inv, z);

    ECDSA_DBG("[ECDSA] s_mont[0]=%08x\n", s_mont[0]);

    // Step 7: Convert s back to normal domain
    uint32_t s[8];
    from_mont_n(s, s_mont);

    ECDSA_DBG("[ECDSA] s[0]=%08x s[7]=%08x\n", s[0], s[7]);

    // Check s == 0 (invalid signature)
    if (is_zero_scalar(s)) {
        for (int i = 0; i < 8; ++i) { r_out[i] = 0; s_out[i] = 0; }
        return false;
    }

    // Copy results
    for (int i = 0; i < 8; ++i) {
        r_out[i] = r[i];
        s_out[i] = s[i];
    }

    return true;
}

// ============================================================================
// ECDSA Sign Batch Kernel
// ============================================================================
// This kernel computes ECDSA signatures for a batch of (k, h, d) inputs.
// It reuses the existing scalar multiplication from persistent_sign_batch.

/*
 * ECDSA Sign Kernel - Batch Processing
 *
 * Each warp processes one signature:
 *   1. Load k, h, d with shuffle gather (CARD19 pattern)
 *   2. Lane 0 computes signature using scalar operations
 *   3. Broadcast and store results
 *
 * Inputs (batch_size x 8 limbs each):
 *   k_batch: nonces [batch_size, 8]
 *   h_batch: message hashes [batch_size, 8]
 *   d_batch: private keys [batch_size, 8]
 *   x_batch: x-coordinates from k*G [batch_size, 8] (pre-computed)
 *
 * Outputs:
 *   r_out: r components [batch_size, 8]
 *   s_out: s components [batch_size, 8]
 */
__global__
void p256_ecdsa_sign_kernel(
    const int32_t* __restrict__ k_batch,
    const int32_t* __restrict__ h_batch,
    const int32_t* __restrict__ d_batch,
    const int32_t* __restrict__ x_batch,  // x-coordinate of k*G
    int32_t* __restrict__ r_out,
    int32_t* __restrict__ s_out,
    int batch_size
)
{
    // Warp-level work assignment
    int warp_id_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;

    if (warp_id_g >= batch_size) return;

    // ========================================================================
    // Load inputs with CARD19 shuffle gather pattern
    // ========================================================================
    uint32_t k_limb = 0, h_limb = 0, d_limb = 0, x_limb = 0;
    if (lane_id < 8) {
        k_limb = (uint32_t)k_batch[warp_id_g * 8 + lane_id];
        h_limb = (uint32_t)h_batch[warp_id_g * 8 + lane_id];
        d_limb = (uint32_t)d_batch[warp_id_g * 8 + lane_id];
        x_limb = (uint32_t)x_batch[warp_id_g * 8 + lane_id];
    }
    __syncwarp(0xFFFFFFFF);

    // Gather all 8 limbs to lane 0 (and all lanes for consistency)
    uint32_t k[8], h[8], d[8], x_R[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        k[i] = __shfl_sync(0xFFFFFFFF, k_limb, i);
        h[i] = __shfl_sync(0xFFFFFFFF, h_limb, i);
        d[i] = __shfl_sync(0xFFFFFFFF, d_limb, i);
        x_R[i] = __shfl_sync(0xFFFFFFFF, x_limb, i);
    }

    // ========================================================================
    // Lane 0 computes the signature
    // ========================================================================
    uint32_t r[8] = {0}, s[8] = {0};

    if (lane_id == 0) {
        bool valid = ecdsa_sign_from_point(r, s, x_R, k, h, d);
        if (!valid) {
            ECDSA_DBG("[ECDSA] Invalid signature (r=0 or s=0) for warp %d\n", warp_id_g);
        }
    }
    __syncwarp(0xFFFFFFFF);

    // ========================================================================
    // Broadcast and store results
    // ========================================================================
    uint32_t r_lane = 0, s_lane = 0;
    if (lane_id == 0) {
        // Store the limb for broadcast
        r_lane = r[0]; s_lane = s[0];
    }

    // Broadcast each limb from lane 0
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
 * ECDSA Sign Batch - PyTorch Interface
 *
 * This function takes precomputed k*G results and computes signatures.
 * The scalar multiplication is done separately using persistent_sign_batch.
 *
 * Inputs:
 *   k_tensor: nonces [batch_size, 8] (int32)
 *   h_tensor: message hashes [batch_size, 8] (int32)
 *   d_tensor: private keys [batch_size, 8] (int32)
 *   x_tensor: x-coordinates of k*G [batch_size, 8] (int32)
 *
 * Returns:
 *   signatures: [batch_size, 2, 8] where [i, 0, :] = r, [i, 1, :] = s
 */
torch::Tensor ecdsa_sign_batch(
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
    TORCH_CHECK(k_tensor.size(1) == 8, "k must have 8 limbs");
    TORCH_CHECK(h_tensor.size(0) == batch_size && h_tensor.size(1) == 8, "h shape mismatch");
    TORCH_CHECK(d_tensor.size(0) == batch_size && d_tensor.size(1) == 8, "d shape mismatch");
    TORCH_CHECK(x_tensor.size(0) == batch_size && x_tensor.size(1) == 8, "x shape mismatch");

    // Ensure inputs are on CUDA
    auto k_cuda = k_tensor.to(torch::kCUDA).contiguous();
    auto h_cuda = h_tensor.to(torch::kCUDA).contiguous();
    auto d_cuda = d_tensor.to(torch::kCUDA).contiguous();
    auto x_cuda = x_tensor.to(torch::kCUDA).contiguous();

    // Allocate output tensors [batch_size, 2, 8] for (r, s)
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto r_tensor = torch::zeros({batch_size, 8}, options);
    auto s_tensor = torch::zeros({batch_size, 8}, options);

    // Launch kernel
    // Each warp handles one signature
    int threads_per_block = 256;  // 8 warps per block
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    p256_ecdsa_sign_kernel<<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        k_cuda.data_ptr<int32_t>(),
        h_cuda.data_ptr<int32_t>(),
        d_cuda.data_ptr<int32_t>(),
        x_cuda.data_ptr<int32_t>(),
        r_tensor.data_ptr<int32_t>(),
        s_tensor.data_ptr<int32_t>(),
        batch_size
    );

    // Stack r and s into [batch_size, 2, 8]
    auto result = torch::stack({r_tensor, s_tensor}, 1);
    return result;
}

// ============================================================================
// Module Registration
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ecdsa_sign_batch", &ecdsa_sign_batch, "ECDSA Sign Batch (P-256)");
}
