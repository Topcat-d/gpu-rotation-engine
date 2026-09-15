// SPDX-License-Identifier: Apache-2.0
// Smoke KDFs — HKDF (SHA-256 / SHA-512) shared-memory implementation
// Baseline: uses shared memory to avoid stack overflow, single-lane execution

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "hmac_sha2_device.cuh"

#ifndef SMOKE_WARP
#define SMOKE_WARP 32
#endif

__device__ __forceinline__ uint32_t lane_id() { return threadIdx.x & (SMOKE_WARP - 1); }

// ---- HKDF-Extract: PRK = HMAC(salt, IKM) ----------------------------

extern "C" __global__
void hkdf_extract_kernel(
    const uint8_t* __restrict__ salt_base,
    const uint8_t* __restrict__ ikm_base,
    const int32_t* __restrict__ salt_off,
    const int32_t* __restrict__ salt_len,
    const int32_t* __restrict__ ikm_off,
    const int32_t* __restrict__ ikm_len,
    uint8_t* __restrict__ prk_out,
    int32_t n_msgs,
    int32_t hash_len,
    int32_t prk_stride)
{
    const int32_t msg_idx = blockIdx.x;
    if (msg_idx >= n_msgs) return;

    // Use shared memory for output to avoid register pressure
    extern __shared__ uint8_t shmem[];

    if (lane_id() == 0) {
        const uint8_t* salt = salt_base + salt_off[msg_idx];
        const uint8_t* ikm  = ikm_base  + ikm_off[msg_idx];
        const int s_len = salt_len[msg_idx];
        const int i_len = ikm_len[msg_idx];
        uint8_t* prk_local = shmem; // Use shared memory

        if (hash_len == 32) {
            hmac_sha256_device(salt, s_len, ikm, i_len, prk_local);
        } else {
            hmac_sha512_device(salt, s_len, ikm, i_len, prk_local);
        }

        // Copy to global memory
        uint8_t* prk_global = prk_out + msg_idx * prk_stride;
        for (int i = 0; i < hash_len; ++i) {
            prk_global[i] = prk_local[i];
        }
    }
    __syncthreads();
}

// ---- HKDF-Expand: OKM blocks from PRK --------------------------------

extern "C" __global__
void hkdf_expand_kernel(
    const uint8_t* __restrict__ prk_base,
    const uint8_t* __restrict__ info_base,
    const int32_t* __restrict__ info_off,
    const int32_t* __restrict__ info_len,
    uint8_t* __restrict__ okm_base,
    const int32_t* __restrict__ okm_off,
    const int32_t* __restrict__ L_out,
    int32_t n_msgs,
    int32_t hash_len,
    int32_t prk_stride)
{
    const int32_t msg_idx = blockIdx.x;
    if (msg_idx >= n_msgs) return;

    extern __shared__ uint8_t shmem[];

    if (lane_id() == 0) {
        const uint8_t* prk  = prk_base + msg_idx * prk_stride;
        const uint8_t* info = info_base + info_off[msg_idx];
        const int inf_len = info_len[msg_idx];
        uint8_t* okm = okm_base + okm_off[msg_idx];
        int L = L_out[msg_idx];

        const int H = hash_len;
        const int N = (L + H - 1) / H;

        // Use shared memory for temporary buffers
        uint8_t* T_prev = shmem;           // [64]
        uint8_t* msg_buf = shmem + 64;     // [256] for T_prev || info || counter
        uint8_t* T_curr = shmem + 64 + 256; // [64]

        int t_prev_len = 0;

        for (int block_idx = 1; block_idx <= N; ++block_idx) {
            // Build msg = T_prev || info || counter
            int msg_len = 0;

            // Copy T_prev
            for (int i = 0; i < t_prev_len; ++i) {
                msg_buf[msg_len++] = T_prev[i];
            }
            // Copy info
            for (int i = 0; i < inf_len; ++i) {
                msg_buf[msg_len++] = info[i];
            }
            // Append counter
            msg_buf[msg_len++] = static_cast<uint8_t>(block_idx);

            // T_curr = HMAC(PRK, msg_buf)
            if (hash_len == 32) {
                hmac_sha256_device(prk, H, msg_buf, msg_len, T_curr);
            } else {
                hmac_sha512_device(prk, H, msg_buf, msg_len, T_curr);
            }

            // Copy to output
            const int copy_len = (block_idx < N) ? H : (L - (N - 1) * H);
            for (int i = 0; i < copy_len; ++i) {
                okm[(block_idx - 1) * H + i] = T_curr[i];
            }

            // T_prev = T_curr
            for (int i = 0; i < H; ++i) {
                T_prev[i] = T_curr[i];
            }
            t_prev_len = H;
        }
    }
    __syncthreads();
}

// ---- HKDF Fused (Extract + Expand) ----------------------------------

extern "C" __global__
void hkdf_extract_expand_kernel(
    const uint8_t* __restrict__ salt_base,
    const uint8_t* __restrict__ ikm_base,
    const uint8_t* __restrict__ info_base,
    const int32_t* __restrict__ salt_off,
    const int32_t* __restrict__ salt_len,
    const int32_t* __restrict__ ikm_off,
    const int32_t* __restrict__ ikm_len,
    const int32_t* __restrict__ info_off,
    const int32_t* __restrict__ info_len,
    uint8_t* __restrict__ okm_base,
    const int32_t* __restrict__ okm_off,
    const int32_t* __restrict__ L_out,
    int32_t n_msgs,
    int32_t hash_len)
{
    const int32_t msg_idx = blockIdx.x;
    if (msg_idx >= n_msgs) return;

    extern __shared__ uint8_t shmem[];

    if (lane_id() == 0) {
        const uint8_t* salt = salt_base + salt_off[msg_idx];
        const uint8_t* ikm  = ikm_base  + ikm_off[msg_idx];
        const uint8_t* info = info_base + info_off[msg_idx];
        const int s_len = salt_len[msg_idx];
        const int i_len = ikm_len[msg_idx];
        const int inf_len = info_len[msg_idx];
        uint8_t* okm = okm_base + okm_off[msg_idx];
        const int L = L_out[msg_idx];
        const int H = hash_len;

        // Use shared memory layout
        uint8_t* prk     = shmem;            // [64]
        uint8_t* T_prev  = shmem + 64;       // [64]
        uint8_t* msg_buf = shmem + 128;      // [256]
        uint8_t* T_curr  = shmem + 128 + 256; // [64]

        // Extract: PRK = HMAC(salt, IKM)
        if (hash_len == 32) {
            hmac_sha256_device(salt, s_len, ikm, i_len, prk);
        } else {
            hmac_sha512_device(salt, s_len, ikm, i_len, prk);
        }

        // Expand
        const int N = (L + H - 1) / H;
        int t_prev_len = 0;

        for (int block_idx = 1; block_idx <= N; ++block_idx) {
            int msg_len = 0;

            // T_prev
            for (int i = 0; i < t_prev_len; ++i) {
                msg_buf[msg_len++] = T_prev[i];
            }
            // info
            for (int i = 0; i < inf_len; ++i) {
                msg_buf[msg_len++] = info[i];
            }
            // counter
            msg_buf[msg_len++] = static_cast<uint8_t>(block_idx);

            // HMAC
            if (hash_len == 32) {
                hmac_sha256_device(prk, H, msg_buf, msg_len, T_curr);
            } else {
                hmac_sha512_device(prk, H, msg_buf, msg_len, T_curr);
            }

            // Copy to output
            const int copy_len = (block_idx < N) ? H : (L - (N - 1) * H);
            for (int i = 0; i < copy_len; ++i) {
                okm[(block_idx - 1) * H + i] = T_curr[i];
            }

            // Update T_prev
            for (int i = 0; i < H; ++i) {
                T_prev[i] = T_curr[i];
            }
            t_prev_len = H;
        }
    }
    __syncthreads();
}
