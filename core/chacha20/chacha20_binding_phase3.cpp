// ==============================================================================
// ChaCha20 CUDA - PyTorch C++ Binding (Phase 3: 2D Grid + ILP)
// ==============================================================================

#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of Phase 3 CUDA kernel launcher
extern "C" void chacha20_encrypt_cuda_launcher_phase3(
    const uint8_t*  plaintext,
    uint8_t*        ciphertext,
    const uint32_t* keys,
    const uint32_t* nonces,
    const uint32_t* counters,
    const uint32_t* blocks_per_msg,
    const uint64_t* msg_offsets_bytes,
    int N,
    int max_tiles_per_message,
    int tile_blocks
);

// ==============================================================================
// PyTorch Binding Function (Phase 3)
// ==============================================================================

torch::Tensor chacha20_encrypt_phase3(
    torch::Tensor plaintext,        // [N, max_len] uint8
    torch::Tensor keys,             // [N, 8] uint32
    torch::Tensor nonces,           // [N, 3] uint32
    torch::Tensor counters,         // [N] uint32
    torch::Tensor blocks_per_msg,   // [N] uint32 - blocks per message
    torch::Tensor msg_offsets_bytes,// [N] uint64 - byte offset per message
    int tile_blocks                 // Tile size (e.g., 256)
) {
    // ===== INPUT VALIDATION =====

    // Check CUDA
    TORCH_CHECK(plaintext.is_cuda(), "plaintext must be CUDA tensor");
    TORCH_CHECK(keys.is_cuda(), "keys must be CUDA tensor");
    TORCH_CHECK(nonces.is_cuda(), "nonces must be CUDA tensor");
    TORCH_CHECK(counters.is_cuda(), "counters must be CUDA tensor");
    TORCH_CHECK(blocks_per_msg.is_cuda(), "blocks_per_msg must be CUDA tensor");
    TORCH_CHECK(msg_offsets_bytes.is_cuda(), "msg_offsets_bytes must be CUDA tensor");

    // Check contiguous
    TORCH_CHECK(plaintext.is_contiguous(), "plaintext must be contiguous");
    TORCH_CHECK(keys.is_contiguous(), "keys must be contiguous");
    TORCH_CHECK(nonces.is_contiguous(), "nonces must be contiguous");
    TORCH_CHECK(counters.is_contiguous(), "counters must be contiguous");
    TORCH_CHECK(blocks_per_msg.is_contiguous(), "blocks_per_msg must be contiguous");
    TORCH_CHECK(msg_offsets_bytes.is_contiguous(), "msg_offsets_bytes must be contiguous");

    // Check dimensions
    TORCH_CHECK(plaintext.dim() == 2, "plaintext must be [N, max_len] shape");
    TORCH_CHECK(keys.dim() == 2 && keys.size(1) == 8,
                "keys must be [N, 8] shape");
    TORCH_CHECK(nonces.dim() == 2 && nonces.size(1) == 3,
                "nonces must be [N, 3] shape");
    TORCH_CHECK(counters.dim() == 1, "counters must be 1D tensor");
    TORCH_CHECK(blocks_per_msg.dim() == 1, "blocks_per_msg must be 1D tensor");
    TORCH_CHECK(msg_offsets_bytes.dim() == 1, "msg_offsets_bytes must be 1D tensor");

    // Check dtypes
    TORCH_CHECK(plaintext.dtype() == torch::kUInt8,
                "plaintext must be uint8");
    TORCH_CHECK(keys.dtype() == torch::kUInt32,
                "keys must be uint32");
    TORCH_CHECK(nonces.dtype() == torch::kUInt32,
                "nonces must be uint32");
    TORCH_CHECK(counters.dtype() == torch::kUInt32,
                "counters must be uint32");
    TORCH_CHECK(blocks_per_msg.dtype() == torch::kUInt32,
                "blocks_per_msg must be uint32");
    TORCH_CHECK(msg_offsets_bytes.dtype() == torch::kInt64,  // Use int64 for large offsets
                "msg_offsets_bytes must be int64");

    // ===== EXTRACT DIMENSIONS =====

    int N = plaintext.size(0);
    int max_len = plaintext.size(1);

    // Validate batch size consistency
    TORCH_CHECK(keys.size(0) == N, "keys batch size mismatch");
    TORCH_CHECK(nonces.size(0) == N, "nonces batch size mismatch");
    TORCH_CHECK(counters.size(0) == N, "counters batch size mismatch");
    TORCH_CHECK(blocks_per_msg.size(0) == N, "blocks_per_msg batch size mismatch");
    TORCH_CHECK(msg_offsets_bytes.size(0) == N, "msg_offsets_bytes batch size mismatch");

    // Calculate max tiles per message for grid.y
    int max_tiles_per_message = 0;
    auto blocks_cpu = blocks_per_msg.cpu();
    auto blocks_accessor = blocks_cpu.accessor<uint32_t, 1>();
    for (int i = 0; i < N; i++) {
        int tiles = (blocks_accessor[i] + tile_blocks - 1) / tile_blocks;
        if (tiles > max_tiles_per_message) {
            max_tiles_per_message = tiles;
        }
    }

    TORCH_CHECK(max_tiles_per_message > 0, "max_tiles_per_message must be positive");

    // ===== ALLOCATE OUTPUT =====

    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(plaintext.device());
    auto ciphertext = torch::empty({N, max_len}, options);

    // ===== CALL CUDA KERNEL =====

    chacha20_encrypt_cuda_launcher_phase3(
        plaintext.data_ptr<uint8_t>(),
        ciphertext.data_ptr<uint8_t>(),
        keys.data_ptr<uint32_t>(),
        nonces.data_ptr<uint32_t>(),
        counters.data_ptr<uint32_t>(),
        blocks_per_msg.data_ptr<uint32_t>(),
        reinterpret_cast<const uint64_t*>(msg_offsets_bytes.data_ptr<int64_t>()),
        N,
        max_tiles_per_message,
        tile_blocks
    );

    // ===== CHECK FOR ERRORS =====

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
    m.def("encrypt_phase3", &chacha20_encrypt_phase3,
          "ChaCha20 encrypt Phase 3 (2D grid + ILP, CUDA)",
          py::arg("plaintext"),
          py::arg("keys"),
          py::arg("nonces"),
          py::arg("counters"),
          py::arg("blocks_per_msg"),
          py::arg("msg_offsets_bytes"),
          py::arg("tile_blocks") = 256);
}

// ==============================================================================
// End of ChaCha20 Phase 3 PyTorch Binding
// ==============================================================================
