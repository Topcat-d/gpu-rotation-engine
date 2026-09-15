/**
 * ChaCha20-Poly1305 AEAD PyTorch Binding
 *
 * Provides both variants:
 * - Fused: Single kernel (high performance)
 * - Overlap: Two kernels on separate streams (easier debugging)
 *
 * Returns (ciphertext, tag, kernel_time_ms)
 *
 * Author: PyTorch Crypto Project
 * Date: October 25, 2025
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Forward declarations
extern "C" void aead_chacha20_poly1305_fused_launcher(
    const uint8_t* plaintext,
    uint8_t* ciphertext,
    uint8_t* tags,
    const uint32_t* keys,
    const uint32_t* nonces,
    const int* tile_map,
    int num_tiles,
    int msg_size_bytes,
    int aad_len,
    cudaStream_t stream
);

extern "C" void aead_chacha20_poly1305_w2warp_launcher(
    const uint8_t* plaintext,
    uint8_t* ciphertext,
    uint8_t* tags,
    const uint32_t* keys,
    const uint32_t* nonces,
    const int* tile_map,
    int num_tiles,
    int msg_size_bytes,
    int aad_len,
    cudaStream_t stream
);

extern "C" void chacha20_encrypt_cuda_launcher_warpmsg(
    const uint8_t* plaintext,
    uint8_t* ciphertext,
    const uint32_t* keys,
    const uint32_t* nonces,
    const uint32_t* counters,
    const int* tile_map,
    int num_tiles,
    int msg_size_bytes,
    cudaStream_t stream
);

extern "C" void poly1305_encrypt_cuda_launcher_warpmsg(
    const uint8_t* message,
    uint8_t* tags,
    const uint32_t* r_keys,
    const uint32_t* s_pads,
    const int* tile_map,
    int num_tiles,
    int msg_size_bytes,
    cudaStream_t stream
);

/**
 * Fused AEAD (single kernel)
 */
std::tuple<torch::Tensor, torch::Tensor, float> aead_fused(
    torch::Tensor plaintext,      // [N, msg_size_bytes]
    torch::Tensor key_words,       // [N, 8] int32
    torch::Tensor nonce_words,     // [N, 3] int32
    torch::Tensor tile_map,        // [num_tiles, 2] int32
    int msg_size_bytes,
    int aad_len
) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    const int N = plaintext.size(0);

    // Allocate output tensors
    auto ciphertext = torch::empty_like(plaintext);
    auto tags = torch::zeros({N, 16}, torch::dtype(torch::kUInt8).device(plaintext.device()));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start, cu_stream);

    // Launch fused kernel
    aead_chacha20_poly1305_fused_launcher(
        plaintext.data_ptr<uint8_t>(),
        ciphertext.data_ptr<uint8_t>(),
        tags.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(key_words.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(nonce_words.data_ptr<int32_t>()),
        tile_map.data_ptr<int32_t>(),
        tile_map.size(0),
        msg_size_bytes,
        aad_len,
        cu_stream
    );

    // Record stop
    cudaEventRecord(stop, cu_stream);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return std::make_tuple(ciphertext, tags, kernel_time_ms);
}

/**
 * Two-warp pipelined AEAD (single kernel with overlap)
 */
std::tuple<torch::Tensor, torch::Tensor, float> aead_w2warp(
    torch::Tensor plaintext,      // [N, msg_size_bytes]
    torch::Tensor key_words,       // [N, 8] int32
    torch::Tensor nonce_words,     // [N, 3] int32
    torch::Tensor tile_map,        // [num_tiles, 2] int32
    int msg_size_bytes,
    int aad_len
) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    const int N = plaintext.size(0);

    // Allocate output tensors
    auto ciphertext = torch::empty_like(plaintext);
    auto tags = torch::zeros({N, 16}, torch::dtype(torch::kUInt8).device(plaintext.device()));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start, cu_stream);

    // Launch two-warp pipeline kernel
    aead_chacha20_poly1305_w2warp_launcher(
        plaintext.data_ptr<uint8_t>(),
        ciphertext.data_ptr<uint8_t>(),
        tags.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(key_words.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(nonce_words.data_ptr<int32_t>()),
        tile_map.data_ptr<int32_t>(),
        tile_map.size(0),
        msg_size_bytes,
        aad_len,
        cu_stream
    );

    // Record stop
    cudaEventRecord(stop, cu_stream);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return std::make_tuple(ciphertext, tags, kernel_time_ms);
}

/**
 * Overlap AEAD (two kernels: ChaCha20 + Poly1305)
 */
std::tuple<torch::Tensor, torch::Tensor, float> aead_overlap(
    torch::Tensor plaintext,       // [N, msg_size_bytes]
    torch::Tensor key_words,        // [N, 8] int32
    torch::Tensor nonce_words,      // [N, 3] int32
    torch::Tensor counter,          // [N] int32 (all 1s for encryption)
    torch::Tensor tile_map_chacha,  // [num_tiles_chacha, 2] int32
    torch::Tensor r_keys,           // [N, 4] int32
    torch::Tensor s_pads,           // [N, 4] int32
    torch::Tensor tile_map_poly,    // [num_tiles_poly, 2] int32
    int msg_size_bytes
) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    const int N = plaintext.size(0);

    // Allocate output tensors
    auto ciphertext = torch::empty_like(plaintext);
    auto tags = torch::zeros({N, 16}, torch::dtype(torch::kUInt8).device(plaintext.device()));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start, cu_stream);

    // Step 1: ChaCha20 encryption
    chacha20_encrypt_cuda_launcher_warpmsg(
        plaintext.data_ptr<uint8_t>(),
        ciphertext.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(key_words.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(nonce_words.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(counter.data_ptr<int32_t>()),
        tile_map_chacha.data_ptr<int32_t>(),
        tile_map_chacha.size(0),
        msg_size_bytes,
        cu_stream
    );

    // Step 2: Poly1305 MAC (on ciphertext)
    poly1305_encrypt_cuda_launcher_warpmsg(
        ciphertext.data_ptr<uint8_t>(),
        tags.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(r_keys.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(s_pads.data_ptr<int32_t>()),
        tile_map_poly.data_ptr<int32_t>(),
        tile_map_poly.size(0),
        msg_size_bytes,
        cu_stream
    );

    // Record stop
    cudaEventRecord(stop, cu_stream);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return std::make_tuple(ciphertext, tags, kernel_time_ms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aead_fused", &aead_fused, "ChaCha20-Poly1305 AEAD (fused single-kernel)");
    m.def("aead_w2warp", &aead_w2warp, "ChaCha20-Poly1305 AEAD (two-warp pipelined)");
    m.def("aead_overlap", &aead_overlap, "ChaCha20-Poly1305 AEAD (overlap two-kernel)");
}
