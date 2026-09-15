/**
 * ChaCha20 Keystream - PyTorch C++ Binding
 *
 * Exposes standalone ChaCha20 keystream to Python
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declare CUDA launchers
extern "C" void chacha20_xor_launcher(
    const uint8_t* input,
    uint8_t* output,
    const uint32_t* keys,
    const uint32_t* nonces,
    const uint32_t* counters,
    const int* data_lens,
    int batch_size,
    int max_data_len,
    cudaStream_t stream
);

extern "C" void chacha20_keystream_launcher(
    uint8_t* keystream,
    const uint32_t* keys,
    const uint32_t* nonces,
    const uint32_t* counters,
    const int* keystream_lens,
    int batch_size,
    int max_keystream_len,
    cudaStream_t stream
);

/**
 * ChaCha20 XOR - Batched API
 *
 * @param input      [batch_size, max_data_len] uint8
 * @param keys       [batch_size, 8] int32 (32-byte key as 8 words)
 * @param nonces     [batch_size, 3] int32 (12-byte nonce as 3 words)
 * @param counters   [batch_size] int32 (initial counter, usually 0 or 1)
 * @param data_lens  [batch_size] int32 (actual data length per message)
 * @return output    [batch_size, max_data_len] uint8
 *
 * Example:
 *     output = chacha20_xor(plaintext, keys, nonces, counters, data_lens)
 *     # Decrypt: plaintext = chacha20_xor(ciphertext, keys, nonces, counters, data_lens)
 */
torch::Tensor chacha20_xor(
    torch::Tensor input,
    torch::Tensor keys,
    torch::Tensor nonces,
    torch::Tensor counters,
    torch::Tensor data_lens
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(keys.is_cuda(), "keys must be CUDA tensor");
    TORCH_CHECK(nonces.is_cuda(), "nonces must be CUDA tensor");
    TORCH_CHECK(counters.is_cuda(), "counters must be CUDA tensor");
    TORCH_CHECK(data_lens.is_cuda(), "data_lens must be CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(keys.dtype() == torch::kInt32, "keys must be int32");
    TORCH_CHECK(nonces.dtype() == torch::kInt32, "nonces must be int32");
    TORCH_CHECK(counters.dtype() == torch::kInt32, "counters must be int32");
    TORCH_CHECK(data_lens.dtype() == torch::kInt32, "data_lens must be int32");

    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch_size, max_data_len]");
    TORCH_CHECK(keys.dim() == 2, "keys must be 2D [batch_size, 8]");
    TORCH_CHECK(nonces.dim() == 2, "nonces must be 2D [batch_size, 3]");
    TORCH_CHECK(counters.dim() == 1, "counters must be 1D [batch_size]");
    TORCH_CHECK(data_lens.dim() == 1, "data_lens must be 1D [batch_size]");

    int batch_size = input.size(0);
    int max_data_len = input.size(1);

    TORCH_CHECK(keys.size(0) == batch_size, "keys batch mismatch");
    TORCH_CHECK(keys.size(1) == 8, "keys must be [batch_size, 8]");
    TORCH_CHECK(nonces.size(0) == batch_size, "nonces batch mismatch");
    TORCH_CHECK(nonces.size(1) == 3, "nonces must be [batch_size, 3]");
    TORCH_CHECK(counters.size(0) == batch_size, "counters batch mismatch");
    TORCH_CHECK(data_lens.size(0) == batch_size, "data_lens batch mismatch");

    // Allocate output
    auto output = torch::empty_like(input);

    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    // Launch kernel
    chacha20_xor_launcher(
        input.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(keys.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(nonces.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(counters.data_ptr<int32_t>()),
        data_lens.data_ptr<int32_t>(),
        batch_size,
        max_data_len,
        cu_stream
    );

    return output;
}

/**
 * ChaCha20 XOR - Single message convenience wrapper
 *
 * @param input    [data_len] uint8
 * @param key      [8] int32
 * @param nonce    [3] int32
 * @param counter  int32 scalar
 * @return output  [data_len] uint8
 */
torch::Tensor chacha20_xor_single(
    torch::Tensor input,
    torch::Tensor key,
    torch::Tensor nonce,
    int32_t counter
) {
    TORCH_CHECK(input.dim() == 1, "input must be 1D");
    TORCH_CHECK(key.dim() == 1, "key must be 1D");
    TORCH_CHECK(key.size(0) == 8, "key must be 8 words (32 bytes)");
    TORCH_CHECK(nonce.dim() == 1, "nonce must be 1D");
    TORCH_CHECK(nonce.size(0) == 3, "nonce must be 3 words (12 bytes)");

    // Add batch dimension
    auto input_batched = input.unsqueeze(0);
    auto keys_batched = key.unsqueeze(0);
    auto nonces_batched = nonce.unsqueeze(0);
    auto counters_batched = torch::tensor({counter},
                                         torch::TensorOptions()
                                             .dtype(torch::kInt32)
                                             .device(input.device()));
    auto data_lens = torch::tensor({(int)input.size(0)},
                                   torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(input.device()));

    // Call batched version
    auto output = chacha20_xor(input_batched, keys_batched, nonces_batched,
                              counters_batched, data_lens);

    // Remove batch dimension
    return output.squeeze(0);
}

/**
 * ChaCha20 Keystream - Batched API
 *
 * Generate pure keystream (no XOR)
 *
 * @param keys          [batch_size, 8] int32
 * @param nonces        [batch_size, 3] int32
 * @param counters      [batch_size] int32
 * @param keystream_lens [batch_size] int32
 * @param max_keystream_len int (maximum keystream length)
 * @return keystream    [batch_size, max_keystream_len] uint8
 */
torch::Tensor chacha20_keystream(
    torch::Tensor keys,
    torch::Tensor nonces,
    torch::Tensor counters,
    torch::Tensor keystream_lens,
    int max_keystream_len
) {
    // Validate inputs
    TORCH_CHECK(keys.is_cuda(), "keys must be CUDA tensor");
    TORCH_CHECK(nonces.is_cuda(), "nonces must be CUDA tensor");
    TORCH_CHECK(counters.is_cuda(), "counters must be CUDA tensor");
    TORCH_CHECK(keystream_lens.is_cuda(), "keystream_lens must be CUDA tensor");

    int batch_size = keys.size(0);

    TORCH_CHECK(keys.size(1) == 8, "keys must be [batch_size, 8]");
    TORCH_CHECK(nonces.size(0) == batch_size, "nonces batch mismatch");
    TORCH_CHECK(nonces.size(1) == 3, "nonces must be [batch_size, 3]");
    TORCH_CHECK(counters.size(0) == batch_size, "counters batch mismatch");
    TORCH_CHECK(keystream_lens.size(0) == batch_size, "keystream_lens batch mismatch");

    // Allocate output
    auto keystream = torch::empty({batch_size, max_keystream_len},
                                  torch::TensorOptions()
                                      .dtype(torch::kUInt8)
                                      .device(keys.device()));

    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    // Launch kernel
    chacha20_keystream_launcher(
        keystream.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(keys.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(nonces.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(counters.data_ptr<int32_t>()),
        keystream_lens.data_ptr<int32_t>(),
        batch_size,
        max_keystream_len,
        cu_stream
    );

    return keystream;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chacha20_xor", &chacha20_xor,
          "ChaCha20 XOR - batched (CUDA)");
    m.def("chacha20_xor_single", &chacha20_xor_single,
          "ChaCha20 XOR - single message (CUDA)");
    m.def("chacha20_keystream", &chacha20_keystream,
          "ChaCha20 keystream generation - batched (CUDA)");
}
