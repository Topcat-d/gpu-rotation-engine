/**
 * SHA-256 Warp-Cooperative V3 Kernel - PyTorch Binding
 *
 * V3: Register-only rounds, 4× messages per warp
 * Expected: 3-6x speedup over Phase A/V2
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_UINT8(x) TORCH_CHECK(x.scalar_type()==at::kByte, #x " must be uint8")
#define CHECK_UINT32(x) TORCH_CHECK(x.scalar_type()==at::kInt, #x " must be uint32")

// Forward declaration of CUDA kernel
extern "C" __global__ void sha256_compress_warpcoop_v3_4x(
    const uint8_t* blocks,
    const uint32_t* H_in,
    uint32_t* H_out,
    int N);

/**
 * Compress batch of SHA-256 blocks using V3 register-only 4×warp kernel
 *
 * Args:
 *   blocks: [N, 64] tensor of uint8 (one 64-byte block per message)
 *   H_in: [N, 8] tensor of uint32 (initial state, e.g., SHA-256 IV)
 *
 * Returns:
 *   H_out: [N, 8] tensor of uint32 (final state after compression + feed-forward)
 */
torch::Tensor sha256_compress_v3_4x(
    torch::Tensor blocks,
    torch::Tensor H_in
) {
    CHECK_CUDA(blocks);
    CHECK_CUDA(H_in);
    CHECK_CONTIG(blocks);
    CHECK_CONTIG(H_in);
    CHECK_UINT8(blocks);
    CHECK_UINT32(H_in);

    const int64_t N = H_in.size(0);

    TORCH_CHECK(blocks.size(0) == N, "blocks and H_in must have same batch size");
    TORCH_CHECK(blocks.size(1) == 64, "blocks must be [N, 64]");
    TORCH_CHECK(H_in.size(1) == 8, "H_in must be [N, 8]");

    // Create output tensor [N, 8]
    auto H_out = torch::empty({N, 8}, H_in.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Get pointers
    const uint8_t* blocks_ptr = blocks.data_ptr<uint8_t>();
    const uint32_t* H_in_ptr = reinterpret_cast<const uint32_t*>(H_in.data_ptr<int32_t>());
    uint32_t* H_out_ptr = reinterpret_cast<uint32_t*>(H_out.data_ptr<int32_t>());

    // Launch configuration: one warp per up-to-4 messages
    int warps = (N + 3) / 4;
    dim3 grid(warps);
    dim3 block(32, 1, 1);    // exactly one warp per block

    // Prepare kernel arguments
    void* args[] = {
        (void*)&blocks_ptr,
        (void*)&H_in_ptr,
        (void*)&H_out_ptr,
        (void*)&N
    };

    // Launch kernel using cudaLaunchKernel API (works in .cpp files)
    cudaError_t launch_err = cudaLaunchKernel(
        (void*)sha256_compress_warpcoop_v3_4x,
        grid,
        block,
        args,
        0,              // shared memory bytes
        stream
    );
    TORCH_CHECK(launch_err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(launch_err));

    // Sync and check for kernel errors
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    TORCH_CHECK(sync_err == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(sync_err));

    return H_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sha256_compress_v3_4x", &sha256_compress_v3_4x,
          "SHA-256 compress batch using V3 register-only 4×warp kernel");
}
