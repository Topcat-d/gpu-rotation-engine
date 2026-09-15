// PyTorch binding for SHA-256 compress-only kernel
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type()==at::kInt, #x " must be int32")
#define CHECK_UINT32(x) TORCH_CHECK(x.scalar_type()==at::kInt, #x " must be uint32")

// Kernel forward declaration
extern "C" __global__ void sha256_compress_sequential(
    const uint32_t* blocks_in,
    uint32_t* state_out,
    const int32_t* num_blocks,
    const int32_t* block_off,
    int32_t batch);

torch::Tensor sha256_compress_batch(
    torch::Tensor blocks,      // [total_blocks, 16] int32 (interpreted as uint32)
    torch::Tensor num_blocks,  // [batch] int32
    torch::Tensor block_off)   // [batch] int32
{
    CHECK_CUDA(blocks); CHECK_CUDA(num_blocks); CHECK_CUDA(block_off);
    CHECK_CONTIG(blocks); CHECK_CONTIG(num_blocks); CHECK_CONTIG(block_off);
    CHECK_INT32(blocks); CHECK_INT32(num_blocks); CHECK_INT32(block_off);

    const int32_t batch = static_cast<int32_t>(num_blocks.numel());

    // Output: [batch, 8] uint32 states
    auto state_out = torch::empty({batch, 8}, blocks.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Launch configuration
    int threads = 256;
    int blocks_grid = (batch + threads - 1) / threads;

    // Cast int32 data pointer to uint32 (same bits, different interpretation)
    const uint32_t* blocks_ptr = reinterpret_cast<const uint32_t*>(blocks.data_ptr<int32_t>());
    uint32_t* state_ptr = reinterpret_cast<uint32_t*>(state_out.data_ptr<int32_t>());
    const int32_t* num_blocks_ptr = num_blocks.data_ptr<int32_t>();
    const int32_t* block_off_ptr = block_off.data_ptr<int32_t>();

    // Prepare kernel arguments (all as pointers for cudaLaunchKernel)
    void* args[] = {
        (void*)&blocks_ptr,
        (void*)&state_ptr,
        (void*)&num_blocks_ptr,
        (void*)&block_off_ptr,
        (void*)&batch
    };

    // Launch kernel
    cudaError_t st = cudaLaunchKernel(
        (void*)sha256_compress_sequential,
        dim3(blocks_grid), dim3(threads), args, 0, stream
    );
    TORCH_CHECK(st == cudaSuccess, "CUDA launch failed: ", cudaGetErrorString(st));

    // Sync and check for kernel errors
    cudaError_t sync_st = cudaStreamSynchronize(stream);
    TORCH_CHECK(sync_st == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(sync_st));

    return state_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sha256_compress_batch", &sha256_compress_batch, "SHA-256 compress batch (takes blocks, num_blocks, block_off)");
}
