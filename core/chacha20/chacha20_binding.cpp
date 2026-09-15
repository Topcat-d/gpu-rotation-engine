// ==============================================================================
// ChaCha20 CUDA - PyTorch C++ Binding
// ==============================================================================
//
// Purpose: Interface between Python/PyTorch and CUDA kernel
// Date: October 24, 2025
//
// This binding does NO crypto work - it just passes tensors to GPU kernel!
// All ChaCha20 operations happen in chacha20_kernel.cu
//
// ==============================================================================

#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of CUDA kernel launcher
extern "C" void chacha20_encrypt_cuda_launcher(
    const uint32_t* keys,
    const uint32_t* nonces,
    const uint32_t* counters,
    const uint8_t*  plaintext,
    uint8_t*        ciphertext,
    const uint32_t* lengths,
    int N,
    int max_len
);

// ==============================================================================
// PyTorch Binding Function
// ==============================================================================

torch::Tensor chacha20_encrypt(
    torch::Tensor keys,      // [N, 8] uint32 - ChaCha20 keys
    torch::Tensor nonces,    // [N, 3] uint32 - ChaCha20 nonces
    torch::Tensor counters,  // [N] uint32 - Block counters
    torch::Tensor plaintext, // [N, max_len] uint8 - Input data
    torch::Tensor lengths    // [N] uint32 - Actual message lengths
) {
    // Validate inputs (all must be CUDA tensors)
    TORCH_CHECK(keys.is_cuda(), "keys must be CUDA tensor");
    TORCH_CHECK(nonces.is_cuda(), "nonces must be CUDA tensor");
    TORCH_CHECK(counters.is_cuda(), "counters must be CUDA tensor");
    TORCH_CHECK(plaintext.is_cuda(), "plaintext must be CUDA tensor");
    TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");

    // Validate contiguous memory layout
    TORCH_CHECK(keys.is_contiguous(), "keys must be contiguous");
    TORCH_CHECK(nonces.is_contiguous(), "nonces must be contiguous");
    TORCH_CHECK(counters.is_contiguous(), "counters must be contiguous");
    TORCH_CHECK(plaintext.is_contiguous(), "plaintext must be contiguous");
    TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");

    // Validate dimensions
    TORCH_CHECK(keys.dim() == 2 && keys.size(1) == 8,
                "keys must be [N, 8] shape");
    TORCH_CHECK(nonces.dim() == 2 && nonces.size(1) == 3,
                "nonces must be [N, 3] shape");
    TORCH_CHECK(counters.dim() == 1,
                "counters must be 1D tensor");
    TORCH_CHECK(plaintext.dim() == 2,
                "plaintext must be [N, max_len] shape");
    TORCH_CHECK(lengths.dim() == 1,
                "lengths must be 1D tensor");

    // Validate dtypes
    TORCH_CHECK(keys.dtype() == torch::kInt32 || keys.dtype() == torch::kUInt32,
                "keys must be uint32 or int32");
    TORCH_CHECK(nonces.dtype() == torch::kInt32 || nonces.dtype() == torch::kUInt32,
                "nonces must be uint32 or int32");
    TORCH_CHECK(counters.dtype() == torch::kInt32 || counters.dtype() == torch::kUInt32,
                "counters must be uint32 or int32");
    TORCH_CHECK(plaintext.dtype() == torch::kUInt8 || plaintext.dtype() == torch::kInt8,
                "plaintext must be uint8 or int8");
    TORCH_CHECK(lengths.dtype() == torch::kInt32 || lengths.dtype() == torch::kUInt32,
                "lengths must be uint32 or int32");

    // Extract dimensions
    int N = keys.size(0);
    int max_len = plaintext.size(1);

    // Validate batch size consistency
    TORCH_CHECK(nonces.size(0) == N, "nonces batch size mismatch");
    TORCH_CHECK(counters.size(0) == N, "counters batch size mismatch");
    TORCH_CHECK(plaintext.size(0) == N, "plaintext batch size mismatch");
    TORCH_CHECK(lengths.size(0) == N, "lengths batch size mismatch");

    // Allocate output tensor (same shape as plaintext)
    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(plaintext.device());
    auto ciphertext = torch::empty({N, max_len}, options);

    // Call CUDA kernel (GPU does the ChaCha20 work!)
    chacha20_encrypt_cuda_launcher(
        keys.data_ptr<uint32_t>(),
        nonces.data_ptr<uint32_t>(),
        counters.data_ptr<uint32_t>(),
        plaintext.data_ptr<uint8_t>(),
        ciphertext.data_ptr<uint8_t>(),
        lengths.data_ptr<uint32_t>(),
        N,
        max_len
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(err));
    }

    return ciphertext;
}

// ==============================================================================
// Python Module Definition
// ==============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encrypt", &chacha20_encrypt, "ChaCha20 encrypt (CUDA)",
          py::arg("keys"),
          py::arg("nonces"),
          py::arg("counters"),
          py::arg("plaintext"),
          py::arg("lengths"));
}

// ==============================================================================
// End of ChaCha20 PyTorch Binding
// ==============================================================================
