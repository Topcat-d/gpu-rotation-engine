/*
 * HChaCha20 + XChaCha20 - PyTorch C++ Binding
 *
 * Exposes HChaCha20 subkey derivation and XChaCha20 encrypt/decrypt to Python
 *
 * Author: PyTorch Crypto Project
 * Date: October 30, 2025
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Declare CUDA launcher functions
extern "C" {
    void hchacha20_derive_subkeys_launcher(
        const int* keys,
        const int* nonces,
        int* subkeys,
        int batch_size,
        cudaStream_t stream
    );

    void xchacha20_xor_launcher(
        const uint8_t* input,
        uint8_t* output,
        const int* keys,
        const int* nonces_24,
        const int* counters,
        const int* data_lens,
        int batch_size,
        int max_data_len,
        cudaStream_t stream
    );
}

/*
 * HChaCha20: Derive subkeys from keys and 16-byte nonces
 *
 * Args:
 *   keys: [batch_size, 8] int32 (32-byte keys as 8 words)
 *   nonces: [batch_size, 4] int32 (16-byte nonces as 4 words)
 *
 * Returns:
 *   subkeys: [batch_size, 8] int32 (32-byte subkeys as 8 words)
 */
torch::Tensor hchacha20_derive_subkeys(
    torch::Tensor keys,
    torch::Tensor nonces
) {
    TORCH_CHECK(keys.device().is_cuda(), "keys must be a CUDA tensor");
    TORCH_CHECK(nonces.device().is_cuda(), "nonces must be a CUDA tensor");
    TORCH_CHECK(keys.dtype() == torch::kInt32, "keys must be int32");
    TORCH_CHECK(nonces.dtype() == torch::kInt32, "nonces must be int32");
    TORCH_CHECK(keys.dim() == 2 && keys.size(1) == 8, "keys must be [batch_size, 8]");
    TORCH_CHECK(nonces.dim() == 2 && nonces.size(1) == 4, "nonces must be [batch_size, 4]");
    TORCH_CHECK(keys.size(0) == nonces.size(0), "batch sizes must match");

    int batch_size = keys.size(0);

    // Allocate output
    auto subkeys = torch::empty({batch_size, 8}, torch::dtype(torch::kInt32).device(keys.device()));

    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    // Launch kernel
    hchacha20_derive_subkeys_launcher(
        keys.data_ptr<int32_t>(),
        nonces.data_ptr<int32_t>(),
        subkeys.data_ptr<int32_t>(),
        batch_size,
        cu_stream
    );

    // Wait for completion
    cudaStreamSynchronize(cu_stream);

    return subkeys;
}

/*
 * XChaCha20 Batched: Encrypt/decrypt multiple messages with 24-byte nonces
 *
 * Args:
 *   input: [batch_size, max_data_len] uint8
 *   keys: [batch_size, 8] int32 (32-byte keys)
 *   nonces_24: [batch_size, 6] int32 (24-byte nonces as 6 words)
 *   counters: [batch_size] int32
 *   data_lens: [batch_size] int32
 *
 * Returns:
 *   output: [batch_size, max_data_len] uint8
 */
torch::Tensor xchacha20_xor(
    torch::Tensor input,
    torch::Tensor keys,
    torch::Tensor nonces_24,
    torch::Tensor counters,
    torch::Tensor data_lens
) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(keys.device().is_cuda(), "keys must be a CUDA tensor");
    TORCH_CHECK(nonces_24.device().is_cuda(), "nonces_24 must be a CUDA tensor");
    TORCH_CHECK(counters.device().is_cuda(), "counters must be a CUDA tensor");
    TORCH_CHECK(data_lens.device().is_cuda(), "data_lens must be a CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(keys.dtype() == torch::kInt32, "keys must be int32");
    TORCH_CHECK(nonces_24.dtype() == torch::kInt32, "nonces_24 must be int32");
    TORCH_CHECK(counters.dtype() == torch::kInt32, "counters must be int32");
    TORCH_CHECK(data_lens.dtype() == torch::kInt32, "data_lens must be int32");

    TORCH_CHECK(input.dim() == 2, "input must be [batch_size, max_data_len]");
    TORCH_CHECK(keys.dim() == 2 && keys.size(1) == 8, "keys must be [batch_size, 8]");
    TORCH_CHECK(nonces_24.dim() == 2 && nonces_24.size(1) == 6, "nonces_24 must be [batch_size, 6]");
    TORCH_CHECK(counters.dim() == 1, "counters must be [batch_size]");
    TORCH_CHECK(data_lens.dim() == 1, "data_lens must be [batch_size]");

    int batch_size = input.size(0);
    int max_data_len = input.size(1);

    TORCH_CHECK(keys.size(0) == batch_size, "keys batch size mismatch");
    TORCH_CHECK(nonces_24.size(0) == batch_size, "nonces_24 batch size mismatch");
    TORCH_CHECK(counters.size(0) == batch_size, "counters batch size mismatch");
    TORCH_CHECK(data_lens.size(0) == batch_size, "data_lens batch size mismatch");

    // Allocate output
    auto output = torch::empty_like(input);

    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    // Launch kernel
    xchacha20_xor_launcher(
        input.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        keys.data_ptr<int32_t>(),
        nonces_24.data_ptr<int32_t>(),
        counters.data_ptr<int32_t>(),
        data_lens.data_ptr<int32_t>(),
        batch_size,
        max_data_len,
        cu_stream
    );

    // Wait for completion
    cudaStreamSynchronize(cu_stream);

    return output;
}

/*
 * XChaCha20 Single: Encrypt/decrypt single message (convenience wrapper)
 *
 * Args:
 *   input: [data_len] uint8
 *   key: [8] int32 (32-byte key)
 *   nonce_24: [6] int32 (24-byte nonce)
 *   counter: int32
 *
 * Returns:
 *   output: [data_len] uint8
 */
torch::Tensor xchacha20_xor_single(
    torch::Tensor input,
    torch::Tensor key,
    torch::Tensor nonce_24,
    int32_t counter
) {
    TORCH_CHECK(input.dim() == 1, "input must be 1D");
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 8, "key must be [8]");
    TORCH_CHECK(nonce_24.dim() == 1 && nonce_24.size(0) == 6, "nonce_24 must be [6]");

    // Convert to batch format
    int data_len = input.size(0);
    auto input_batched = input.unsqueeze(0);  // [1, data_len]
    auto keys = key.unsqueeze(0);  // [1, 8]
    auto nonces_24 = nonce_24.unsqueeze(0);  // [1, 6]
    auto counters = torch::tensor({counter}, torch::dtype(torch::kInt32).device(input.device()));
    auto data_lens = torch::tensor({data_len}, torch::dtype(torch::kInt32).device(input.device()));

    // Call batched version
    auto output_batched = xchacha20_xor(input_batched, keys, nonces_24, counters, data_lens);

    // Remove batch dimension
    return output_batched.squeeze(0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hchacha20_derive_subkeys", &hchacha20_derive_subkeys,
          "HChaCha20 subkey derivation (batched)");
    m.def("xchacha20_xor", &xchacha20_xor,
          "XChaCha20 XOR (batched encrypt/decrypt)");
    m.def("xchacha20_xor_single", &xchacha20_xor_single,
          "XChaCha20 XOR (single message)");
}
