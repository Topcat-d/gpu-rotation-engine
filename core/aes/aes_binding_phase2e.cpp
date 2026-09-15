// SPDX-License-Identifier: MIT
// Smoke AES Phase 2e binding (CTR only, 2× ILP)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <chrono>

// CTR launcher
extern "C" void launch_aes_ctr_encrypt_phase2(
    const uint4* in, uint4* out,
    const uint32_t* rk, int Nr,
    uint64_t nonce_hi, uint64_t base_counter,
    int blocks_per_msg, int num_msgs,
    int num_blocks, int threads_per_block);

std::tuple<torch::Tensor, float> aes_ctr_encrypt_phase2(
    torch::Tensor pt,
    torch::Tensor rk,
    int Nr,
    int64_t nonce_hi,
    int64_t base_counter,
    int blocks_per_msg,
    int threads_per_block)
{
    TORCH_CHECK(pt.is_cuda(), "pt must be CUDA");
    TORCH_CHECK(rk.is_cuda(), "rk must be CUDA");
    TORCH_CHECK(pt.dtype() == torch::kUInt8, "pt must be uint8");
    TORCH_CHECK(pt.dim() == 2, "pt: [num_blocks, 16]");
    TORCH_CHECK(pt.size(1) == 16, "pt block size must be 16");

    int64_t num_blocks_data = pt.size(0);
    int num_msgs = (num_blocks_data + blocks_per_msg - 1) / blocks_per_msg;

    torch::Tensor ct = torch::empty_like(pt);

    const uint4* d_in = reinterpret_cast<const uint4*>(pt.data_ptr<uint8_t>());
    uint4* d_out = reinterpret_cast<uint4*>(ct.data_ptr<uint8_t>());
    const uint32_t* d_rk = reinterpret_cast<const uint32_t*>(rk.data_ptr<int32_t>());

    int grid_size = (num_msgs + (threads_per_block / 32) - 1) / (threads_per_block / 32);

    auto start = std::chrono::high_resolution_clock::now();
    launch_aes_ctr_encrypt_phase2(
        d_in, d_out, d_rk, Nr,
        (uint64_t)nonce_hi, (uint64_t)base_counter,
        blocks_per_msg, num_msgs,
        grid_size, threads_per_block);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float elapsed_ms = std::chrono::duration<float, std::milli>(end - start).count();

    return std::make_tuple(ct, elapsed_ms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aes_ctr_encrypt_phase2", &aes_ctr_encrypt_phase2, "AES CTR Phase 2e (2× ILP)");
}
