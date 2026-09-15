/**
 * SHA-256 Warp-Cooperative Kernel - PyTorch Binding
 *
 * Exports warp-cooperative compress functions for use from Python.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type()==at::kInt, #x " must be int32")

// Forward declaration of CUDA kernel
extern "C" __global__ void sha256_compress_batch_warpcoop(
    const int32_t* blocks,
    const int32_t* num_blocks,
    const int32_t* block_offsets,
    int32_t* state_out,
    int batch_size);

/**
 * Compress batch of SHA-256 blocks using warp-cooperative kernel
 *
 * Args:
 *   blocks: [total_blocks*16] tensor of int32 words (big-endian)
 *   num_blocks: [batch_size] tensor of block counts per message
 *   block_offsets: [batch_size] tensor of block offsets per message
 *
 * Returns:
 *   state_out: [batch_size, 8] tensor of final SHA-256 state
 */
torch::Tensor sha256_compress_batch(
    torch::Tensor blocks,
    torch::Tensor num_blocks,
    torch::Tensor block_offsets
) {
    CHECK_CUDA(blocks);
    CHECK_CUDA(num_blocks);
    CHECK_CUDA(block_offsets);
    CHECK_CONTIG(blocks);
    CHECK_CONTIG(num_blocks);
    CHECK_CONTIG(block_offsets);
    CHECK_INT32(blocks);
    CHECK_INT32(num_blocks);
    CHECK_INT32(block_offsets);

    const int32_t batch_size = static_cast<int32_t>(num_blocks.numel());

    // Create output tensor [batch_size, 8]
    auto state_out = torch::empty({batch_size, 8}, blocks.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Get pointers
    const int32_t* blocks_ptr = blocks.data_ptr<int32_t>();
    const int32_t* num_blocks_ptr = num_blocks.data_ptr<int32_t>();
    const int32_t* block_offsets_ptr = block_offsets.data_ptr<int32_t>();
    int32_t* state_out_ptr = state_out.data_ptr<int32_t>();

    // Launch configuration
    dim3 grid(batch_size);   // One block per message
    dim3 block(32, 1, 1);    // 32 threads per block (explicit 3D)

    // Prepare kernel arguments
    void* args[] = {
        (void*)&blocks_ptr,
        (void*)&num_blocks_ptr,
        (void*)&block_offsets_ptr,
        (void*)&state_out_ptr,
        (void*)&batch_size
    };

    // Launch kernel
    cudaError_t st = cudaLaunchKernel(
        (void*)sha256_compress_batch_warpcoop,
        grid,
        block,
        args,
        0,              // shared memory bytes
        stream
    );
    TORCH_CHECK(st == cudaSuccess, "CUDA launch failed: ", cudaGetErrorString(st));

    // Sync and check for kernel errors
    cudaError_t sync_st = cudaStreamSynchronize(stream);
    TORCH_CHECK(sync_st == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(sync_st));

    return state_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sha256_compress_batch", &sha256_compress_batch,
          "SHA-256 compress batch using warp-cooperative kernel");
}
