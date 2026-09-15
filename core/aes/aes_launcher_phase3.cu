// SPDX-License-Identifier: MIT
// Smoke AES Phase 3 launcher (CTR only, round ping-pong)

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void aes_ctr_encrypt_phase3(
    const uint4* __restrict__ in,
    uint4* __restrict__ out,
    const uint32_t* __restrict__ rk,
    int Nr,
    uint64_t nonce_hi,
    uint64_t base_counter,
    int blocks_per_msg,
    int num_msgs);

extern "C" void launch_aes_ctr_encrypt_phase3(
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
    aes_ctr_encrypt_phase3<<<num_blocks, threads_per_block>>>(
        in, out, rk, Nr, nonce_hi, base_counter, blocks_per_msg, num_msgs);
}
