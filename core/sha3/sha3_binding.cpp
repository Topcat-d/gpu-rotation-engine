/*
 * PyBind11 binding for SHA-3-256 CUDA kernel
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// External CUDA launcher
extern "C" void launch_keccak_f1600(
    uint64_t* state,
    int B,
    cudaStream_t stream
);

/*
 * Python-facing function: permute Keccak-f[1600] state
 *
 * Args:
 *   state: [B, 25] int64 tensor (GPU)
 *
 * Returns:
 *   state: [B, 25] int64 tensor (GPU) - permuted in-place
 */
torch::Tensor keccak_permute(torch::Tensor state) {
    // Validate input
    TORCH_CHECK(state.is_cuda(), "state must be CUDA tensor");
    TORCH_CHECK(state.dtype() == torch::kInt64, "state must be int64");
    TORCH_CHECK(state.dim() == 2, "state must be [B, 25]");
    TORCH_CHECK(state.size(1) == 25, "state must be [B, 25]");

    int B = state.size(0);

    // Use default CUDA stream (compatible with all PyTorch versions)
    cudaStream_t stream = 0;

    // Launch kernel (in-place permutation)
    launch_keccak_f1600(
        reinterpret_cast<uint64_t*>(state.data_ptr<int64_t>()),
        B,
        stream
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(err));

    return state;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("permute", &keccak_permute, "Keccak-f[1600] permutation (CUDA)");
}
