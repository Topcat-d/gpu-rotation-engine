// SPDX-License-Identifier: MIT
// PyTorch binding for Smoke AES CUDA baseline

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Declare CUDA launcher functions (defined in aes_launcher.cu)
extern "C" void launch_aes_ecb_encrypt(
    const uint4* in, uint4* out,
    const uint32_t* rk, int Nr,
    int blocks_per_msg, int num_msgs,
    int num_blocks, int threads_per_block);

extern "C" void launch_aes_ctr_encrypt(
    const uint4* in, uint4* out,
    const uint32_t* rk, int Nr,
    uint64_t nonce_hi, uint64_t base_counter,
    int blocks_per_msg, int num_msgs,
    int num_blocks, int threads_per_block);

// Host wrapper for ECB
std::tuple<torch::Tensor, float> aes_ecb_encrypt(
    torch::Tensor plaintext,   // [N, 16] uint8
    torch::Tensor round_keys,  // [(Nr+1)*4] int32
    int Nr,
    int blocks_per_msg,
    int threads_per_block)
{
    TORCH_CHECK(plaintext.is_cuda(), "plaintext must be CUDA");
    TORCH_CHECK(round_keys.is_cuda(), "round_keys must be CUDA");
    TORCH_CHECK(plaintext.dtype() == torch::kUInt8, "plaintext uint8");
    TORCH_CHECK(round_keys.dtype() == torch::kInt32, "round_keys int32");
    TORCH_CHECK(plaintext.dim() == 2 && plaintext.size(1) == 16, "plaintext [N,16]");

    int total_blocks = plaintext.size(0);
    TORCH_CHECK(total_blocks % blocks_per_msg == 0, "total_blocks % blocks_per_msg != 0");
    int num_msgs = total_blocks / blocks_per_msg;

    auto ciphertext = torch::empty_like(plaintext);

    const uint4* in_ptr = reinterpret_cast<const uint4*>(plaintext.data_ptr<uint8_t>());
    uint4* out_ptr = reinterpret_cast<uint4*>(ciphertext.data_ptr<uint8_t>());
    const uint32_t* rk_ptr = reinterpret_cast<const uint32_t*>(round_keys.data_ptr<int32_t>());

    int warps_per_block = threads_per_block / 32;
    int num_blocks = (num_msgs + warps_per_block - 1) / warps_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    launch_aes_ecb_encrypt(
        in_ptr, out_ptr, rk_ptr, Nr, blocks_per_msg, num_msgs,
        num_blocks, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return std::make_tuple(ciphertext, ms);
}

// Host wrapper for CTR
std::tuple<torch::Tensor, float> aes_ctr_encrypt(
    torch::Tensor plaintext,
    torch::Tensor round_keys,
    int Nr,
    uint64_t nonce_hi,
    uint64_t base_counter,
    int blocks_per_msg,
    int threads_per_block)
{
    TORCH_CHECK(plaintext.is_cuda(), "plaintext must be CUDA");
    TORCH_CHECK(round_keys.is_cuda(), "round_keys must be CUDA");
    TORCH_CHECK(plaintext.dtype() == torch::kUInt8, "plaintext uint8");
    TORCH_CHECK(round_keys.dtype() == torch::kInt32, "round_keys int32");
    TORCH_CHECK(plaintext.dim() == 2 && plaintext.size(1) == 16, "plaintext [N,16]");

    int total_blocks = plaintext.size(0);
    TORCH_CHECK(total_blocks % blocks_per_msg == 0, "total_blocks % blocks_per_msg != 0");
    int num_msgs = total_blocks / blocks_per_msg;

    auto ciphertext = torch::empty_like(plaintext);

    const uint4* in_ptr = reinterpret_cast<const uint4*>(plaintext.data_ptr<uint8_t>());
    uint4* out_ptr = reinterpret_cast<uint4*>(ciphertext.data_ptr<uint8_t>());
    const uint32_t* rk_ptr = reinterpret_cast<const uint32_t*>(round_keys.data_ptr<int32_t>());

    int warps_per_block = threads_per_block / 32;
    int num_blocks = (num_msgs + warps_per_block - 1) / warps_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    launch_aes_ctr_encrypt(
        in_ptr, out_ptr, rk_ptr, Nr, nonce_hi, base_counter,
        blocks_per_msg, num_msgs, num_blocks, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return std::make_tuple(ciphertext, ms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aes_ecb_encrypt", &aes_ecb_encrypt, "AES ECB encrypt (CUDA, timed)");
    m.def("aes_ctr_encrypt", &aes_ctr_encrypt, "AES CTR encrypt (CUDA, timed)");
}
