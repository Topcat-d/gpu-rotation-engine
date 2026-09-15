/**
 * Poly1305 MAC - PyTorch C++ Binding
 *
 * Exposes standalone Poly1305 MAC to Python via PyTorch
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declare CUDA launcher
extern "C" void poly1305_mac_launcher(
    const uint8_t* messages,
    const int* msg_lens,
    const uint8_t* keys,
    uint8_t* tags,
    int batch_size,
    int max_msg_len,
    cudaStream_t stream
);

/**
 * Poly1305 MAC - Batched one-shot API
 *
 * @param messages   [batch_size, max_msg_len] uint8 tensor
 * @param msg_lens   [batch_size] int32 tensor (actual length per message)
 * @param keys       [batch_size, 32] uint8 tensor (16-byte r + 16-byte s)
 * @return tags      [batch_size, 16] uint8 tensor
 */
torch::Tensor poly1305_mac(
    torch::Tensor messages,
    torch::Tensor msg_lens,
    torch::Tensor keys
) {
    // Validate inputs
    TORCH_CHECK(messages.is_cuda(), "messages must be CUDA tensor");
    TORCH_CHECK(msg_lens.is_cuda(), "msg_lens must be CUDA tensor");
    TORCH_CHECK(keys.is_cuda(), "keys must be CUDA tensor");

    TORCH_CHECK(messages.dtype() == torch::kUInt8, "messages must be uint8");
    TORCH_CHECK(msg_lens.dtype() == torch::kInt32, "msg_lens must be int32");
    TORCH_CHECK(keys.dtype() == torch::kUInt8, "keys must be uint8");

    TORCH_CHECK(messages.dim() == 2, "messages must be 2D [batch_size, max_msg_len]");
    TORCH_CHECK(msg_lens.dim() == 1, "msg_lens must be 1D [batch_size]");
    TORCH_CHECK(keys.dim() == 2, "keys must be 2D [batch_size, 32]");

    int batch_size = messages.size(0);
    int max_msg_len = messages.size(1);

    TORCH_CHECK(msg_lens.size(0) == batch_size, "msg_lens batch mismatch");
    TORCH_CHECK(keys.size(0) == batch_size, "keys batch mismatch");
    TORCH_CHECK(keys.size(1) == 32, "keys must be 32 bytes (16-byte r + 16-byte s)");

    // Allocate output
    auto tags = torch::empty({batch_size, 16},
                             torch::TensorOptions()
                                 .dtype(torch::kUInt8)
                                 .device(messages.device()));

    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    // Launch kernel
    poly1305_mac_launcher(
        messages.data_ptr<uint8_t>(),
        msg_lens.data_ptr<int32_t>(),
        keys.data_ptr<uint8_t>(),
        tags.data_ptr<uint8_t>(),
        batch_size,
        max_msg_len,
        cu_stream
    );

    return tags;
}

/**
 * Poly1305 MAC - Single message convenience wrapper
 *
 * @param message   [msg_len] uint8 tensor
 * @param key       [32] uint8 tensor
 * @return tag      [16] uint8 tensor
 */
torch::Tensor poly1305_mac_single(
    torch::Tensor message,
    torch::Tensor key
) {
    TORCH_CHECK(message.dim() == 1, "message must be 1D");
    TORCH_CHECK(key.dim() == 1, "key must be 1D");
    TORCH_CHECK(key.size(0) == 32, "key must be 32 bytes");

    // Add batch dimension
    auto messages_batched = message.unsqueeze(0);
    auto keys_batched = key.unsqueeze(0);
    auto msg_lens = torch::tensor({(int)message.size(0)},
                                   torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(message.device()));

    // Call batched version
    auto tags = poly1305_mac(messages_batched, msg_lens, keys_batched);

    // Remove batch dimension
    return tags.squeeze(0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("poly1305_mac", &poly1305_mac,
          "Poly1305 MAC - batched (CUDA)");
    m.def("poly1305_mac_single", &poly1305_mac_single,
          "Poly1305 MAC - single message (CUDA)");
}
