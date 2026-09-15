// SPDX-License-Identifier: MIT
// Smoke AES kernel launcher wrappers (CUDA-compiled)

#include <cuda_runtime.h>
#include <stdint.h>

// Forward declare kernels (defined in aes_kernel.cu, linked together)
extern "C" __global__ void aes_ecb_encrypt_kernel(
    const uint4* __restrict__ in,
    uint4* __restrict__ out,
    const uint32_t* __restrict__ rk,
    int Nr,
    int blocks_per_msg,
    int num_msgs);

extern "C" __global__ void aes_ctr_encrypt_kernel(
    const uint4* __restrict__ in,
    uint4* __restrict__ out,
    const uint32_t* __restrict__ rk,
    int Nr,
    uint64_t nonce_hi,
    uint64_t base_counter,
    int blocks_per_msg,
    int num_msgs);

// Host-side launcher for ECB (callable from C++)
extern "C" void launch_aes_ecb_encrypt(
    const uint4* in,
    uint4* out,
    const uint32_t* rk,
    int Nr,
    int blocks_per_msg,
    int num_msgs,
    int num_blocks,
    int threads_per_block)
{
    aes_ecb_encrypt_kernel<<<num_blocks, threads_per_block>>>(
        in, out, rk, Nr, blocks_per_msg, num_msgs);
}

// Host-side launcher for CTR (callable from C++)
extern "C" void launch_aes_ctr_encrypt(
    const uint4* in,
    uint4* out,
    const uint32_t* rk,
    int Nr,
    uint64_t nonce_hi,
    uint64_t base_counter,
    int blocks_per_msg,
    int num_msgs,
    int num_blocks,
    int threads_per_block)
{
    aes_ctr_encrypt_kernel<<<num_blocks, threads_per_block>>>(
        in, out, rk, Nr, nonce_hi, base_counter, blocks_per_msg, num_msgs);
}
