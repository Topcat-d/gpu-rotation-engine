/*
 * PyBind11 Binding for SHA-512 CUDA Kernel
 *
 * Exposes compress() function to Python/PyTorch
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of CUDA launcher
extern "C" void launch_sha512_compress(
    const uint64_t* state_in,
    const uint64_t* blocks,
    uint64_t* state_out,
    int B,
    cudaStream_t stream
);

/*
 * compress - Main entry point from Python
 *
 * Args:
 *   blocks: [B, 16] int64 tensor (message blocks, 16 x 64-bit words)
 *   state: [B, 8] int64 tensor (hash state, H0-H7)
 *
 * Returns:
 *   [B, 8] int64 tensor (updated state)
 */
torch::Tensor compress(
    torch::Tensor blocks,
    torch::Tensor state
) {
    // Validate inputs
    TORCH_CHECK(blocks.dim() == 2, "blocks must be 2D");
    TORCH_CHECK(state.dim() == 2, "state must be 2D");
    TORCH_CHECK(blocks.size(1) == 16, "blocks must have 16 words per block");
    TORCH_CHECK(state.size(1) == 8, "state must have 8 words");
    TORCH_CHECK(blocks.size(0) == state.size(0), "batch size mismatch");
    TORCH_CHECK(blocks.dtype() == torch::kInt64, "blocks must be int64");
    TORCH_CHECK(state.dtype() == torch::kInt64, "state must be int64");
    TORCH_CHECK(blocks.is_cuda(), "blocks must be on CUDA");
    TORCH_CHECK(state.is_cuda(), "state must be on CUDA");
    TORCH_CHECK(blocks.is_contiguous(), "blocks must be contiguous");
    TORCH_CHECK(state.is_contiguous(), "state must be contiguous");

    int B = blocks.size(0);

    // Allocate output
    auto output = torch::empty_like(state);

    // Use default CUDA stream (compatible with all PyTorch versions)
    cudaStream_t stream = 0;

    // Launch kernel
    launch_sha512_compress(
        reinterpret_cast<const uint64_t*>(state.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(blocks.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(output.data_ptr<int64_t>()),
        B,
        stream
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress", &compress, "SHA-512 compression function (CUDA)");
}
