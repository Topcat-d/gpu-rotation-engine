// Baseline AEAD binding (original fused kernel)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

struct Poly1305Trace {
    uint32_t h0, h1, h2, h3, h4;
};

extern "C" void aead_chacha20_poly1305_fused_launcher(
    const uint8_t* plaintext,
    uint8_t* ciphertext,
    uint8_t* tags,
    const uint32_t* keys,
    const uint32_t* nonces,
    const uint8_t* aad_data,
    const int* tile_map,
    int num_tiles,
    int msg_size_bytes,
    int aad_len,
    cudaStream_t stream,
    Poly1305Trace* trace_buffer,
    int trace_capacity
);

std::tuple<torch::Tensor, torch::Tensor, float> aead_fused(
    torch::Tensor plaintext,
    torch::Tensor key_words,
    torch::Tensor nonce_words,
    torch::Tensor tile_map,
    int msg_size_bytes,
    int aad_len,
    c10::optional<torch::Tensor> aad_data_opt = c10::nullopt
) {
    // One-time cache config: prefer larger L1 for streaming traffic (Turing+ combined cache)
    // This reduces L2 pollution and improves bandwidth for one-time-use plaintext/ciphertext
    static bool s_cache_cfg_set = false;
    if (!s_cache_cfg_set) {
        // Note: This requires the kernel symbol to be visible. If using a launcher wrapper,
        // you may need to declare the kernel symbol with extern "C" __global__ here.
        // For now, we set a global preference which affects all kernels in this context.
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        s_cache_cfg_set = true;
    }

    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    const int N = plaintext.size(0);

    auto ciphertext = torch::empty_like(plaintext);
    auto tags = torch::zeros({N, 16}, torch::dtype(torch::kUInt8).device(plaintext.device()));

    // Create AAD tensor if not provided (empty AAD)
    torch::Tensor aad_data;
    const uint8_t* aad_ptr = nullptr;
    if (aad_data_opt.has_value() && aad_len > 0) {
        aad_data = aad_data_opt.value();
        aad_ptr = aad_data.data_ptr<uint8_t>();
    } else {
        // Allocate empty AAD tensor (kernel will skip AAD processing if aad_len == 0)
        aad_data = torch::zeros({N, 1}, torch::dtype(torch::kUInt8).device(plaintext.device()));
        aad_ptr = aad_data.data_ptr<uint8_t>();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, cu_stream);

    aead_chacha20_poly1305_fused_launcher(
        plaintext.data_ptr<uint8_t>(),
        ciphertext.data_ptr<uint8_t>(),
        tags.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(key_words.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(nonce_words.data_ptr<int32_t>()),
        aad_ptr,
        tile_map.data_ptr<int32_t>(),
        tile_map.size(0),
        msg_size_bytes,
        aad_len,
        cu_stream,
        nullptr,  // No tracing
        0
    );

    cudaEventRecord(stop, cu_stream);
    cudaEventSynchronize(stop);

    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return std::make_tuple(ciphertext, tags, kernel_time_ms);
}

void encrypt_with_trace(
    torch::Tensor plaintext,
    torch::Tensor ciphertext,
    torch::Tensor tag,
    torch::Tensor key,
    torch::Tensor nonce,
    torch::Tensor trace_buffer,  // [max_traces, 5] uint32
    int max_traces
) {
    // Simple 1-message encryption with tracing (for debugging)
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    int msg_size_bytes = plaintext.size(0);

    // Convert key/nonce to uint32 words
    auto key_u32 = torch::zeros({8}, torch::dtype(torch::kInt32).device(key.device()));
    auto nonce_u32 = torch::zeros({3}, torch::dtype(torch::kInt32).device(nonce.device()));

    for (int i = 0; i < 8; i++) {
        uint32_t word = 0;
        for (int j = 0; j < 4; j++) {
            word |= (uint32_t)key[i*4+j].item<uint8_t>() << (j*8);
        }
        key_u32[i] = (int32_t)word;
    }

    for (int i = 0; i < 3; i++) {
        uint32_t word = 0;
        for (int j = 0; j < 4; j++) {
            word |= (uint32_t)nonce[i*4+j].item<uint8_t>() << (j*8);
        }
        nonce_u32[i] = (int32_t)word;
    }

    // Tile map: single message, single tile (warp 0 processes blocks 0-31)
    auto tile_map = torch::tensor({{0, 0}}, torch::dtype(torch::kInt32).device(plaintext.device()));

    // Empty AAD for tracing (can be extended later)
    auto aad_empty = torch::zeros({1, 1}, torch::dtype(torch::kUInt8).device(plaintext.device()));

    aead_chacha20_poly1305_fused_launcher(
        plaintext.data_ptr<uint8_t>(),
        ciphertext.data_ptr<uint8_t>(),
        tag.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(key_u32.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(nonce_u32.data_ptr<int32_t>()),
        aad_empty.data_ptr<uint8_t>(),
        tile_map.data_ptr<int32_t>(),
        1,  // num_tiles
        msg_size_bytes,
        0,  // aad_len
        cu_stream,
        reinterpret_cast<Poly1305Trace*>(trace_buffer.data_ptr<uint32_t>()),
        max_traces
    );

    cudaStreamSynchronize(cu_stream);
}

// ============================================================================
// Capture-safe "into" API for CUDA Graphs
// ============================================================================

void aead_fused_into(
    torch::Tensor plaintext,
    torch::Tensor ciphertext,
    torch::Tensor tags,
    torch::Tensor key_words,
    torch::Tensor nonce_words,
    torch::Tensor tile_map,
    torch::Tensor aad_data,  // Pre-allocated (even if empty) - REQUIRED for capture-safe!
    int msg_size_bytes,
    int aad_len
) {
    // No allocations, no events - capture-safe!

    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t cu_stream = stream.stream();

    // AAD pointer (even if aad_len==0, we pass the buffer)
    const uint8_t* aad_ptr = aad_data.data_ptr<uint8_t>();

    // Just call the launcher - no timing
    aead_chacha20_poly1305_fused_launcher(
        plaintext.data_ptr<uint8_t>(),
        ciphertext.data_ptr<uint8_t>(),
        tags.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(key_words.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(nonce_words.data_ptr<int32_t>()),
        aad_ptr,
        tile_map.data_ptr<int32_t>(),
        tile_map.size(0),
        msg_size_bytes,
        aad_len,
        cu_stream,
        nullptr,  // No tracing
        0
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aead_fused", &aead_fused, "ChaCha20-Poly1305 AEAD Baseline");
    m.def("aead_fused_into", &aead_fused_into, "ChaCha20-Poly1305 AEAD (into pre-allocated buffers, capture-safe)");
    m.def("encrypt_with_trace", &encrypt_with_trace, "ChaCha20-Poly1305 with Poly1305 tracing (debug)");
}
