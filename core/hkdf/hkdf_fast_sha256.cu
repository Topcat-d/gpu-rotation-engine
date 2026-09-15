// SPDX-License-Identifier: Apache-2.0
// Smoke - Fast HKDF-SHA256 kernel (compress-only, lane-0 sequential)
// Host prebuilds all HMAC blocks with padding → GPU only does compress loops

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef SMOKE_WARP
#define SMOKE_WARP 32
#endif

// Device functions from sha2_compress_minimal.cu
extern "C" __device__ void sha256_init_state(uint32_t H[8]);
extern "C" __device__ void sha256_compress_block(uint32_t H[8], const uint8_t* block);
extern "C" __device__ void sha256_store_digest(const uint32_t H[8], uint8_t out32[32]);

__device__ __forceinline__ uint32_t lane_id() {
    return threadIdx.x & (SMOKE_WARP-1);
}

// Kernel: compute PRK = HMAC_SHA256(salt, IKM)
// Inputs: prepacked INNER blocks (ipad||IKM||padding) per message; OUTER is constructed on-device.
extern "C" __global__
__launch_bounds__(32, 8)
void hkdf_extract_fast_256_kernel(
    const uint8_t* __restrict__ inner_blocks_base, // concatenated blocks for all msgs
    const int32_t* __restrict__ inner_off,         // byte offset per msg
    const int32_t* __restrict__ inner_nblocks,     // blocks per msg
    const uint8_t* __restrict__ opad_block_base,   // opad 64B per msg (precomputed host w/ K0^opad)
    const uint8_t* __restrict__ opad2_block_base,  // optional second opad block if host needs (usually 0)
    const int32_t  opad_has_2blocks,               // 0 or 1 (normally 1 block for SHA256 opad)
    uint8_t* __restrict__ prk_out,                 // [n * 32]
    int32_t n_msgs)
{
    const int m = blockIdx.x;
    if (m >= n_msgs) return;  // Uniform early exit for whole warp

    const int lid = lane_id();
    // NO early return for lane != 0 - all lanes must participate in compress calls!

    // Shared memory for blocks (all lanes can access)
    __shared__ uint8_t inner_digest_shared[32];
    __shared__ uint8_t last_block_shared[64];

    // --- INNER: H(ipad || IKM || pad) ---
    uint32_t H[8];
    sha256_init_state(H);

    // Lane 0 does sequential work, but ALL lanes participate in compress
    const uint8_t* in_ptr = inner_blocks_base + inner_off[m];
    int nb = inner_nblocks[m];
    for (int b = 0; b < nb; ++b) {
        sha256_compress_block(H, in_ptr + b*64);  // All 32 lanes participate
    }

    // Lane 0 stores inner digest
    if (lid == 0) {
        sha256_store_digest(H, inner_digest_shared);
    }
    __syncwarp(0xFFFFFFFFu);

    // --- OUTER: H(opad || inner_digest || pad) ---
    sha256_init_state(H);

    // opad first block (all lanes participate)
    const uint8_t* opad1 = opad_block_base + m*64;
    sha256_compress_block(H, opad1);

    if (opad_has_2blocks) {
        const uint8_t* opad2 = opad2_block_base + m*64;
        sha256_compress_block(H, opad2);
    }

    // Lane 0 builds the final block
    if (lid == 0) {
        // Fill zeros
        #pragma unroll
        for (int i=0;i<64;i++) last_block_shared[i]=0;

        // inner digest at start
        #pragma unroll
        for (int i=0;i<32;i++) last_block_shared[i] = inner_digest_shared[i];

        // 0x80 after digest
        last_block_shared[32] = 0x80;

        // 8-byte big-endian length: (64 + 32) * 8 = 768 bits
        const uint64_t bitlen = (uint64_t)(64 + 32) * 8ULL;
        last_block_shared[63] = (uint8_t)( bitlen        & 0xFF);
        last_block_shared[62] = (uint8_t)((bitlen >> 8)  & 0xFF);
        last_block_shared[61] = (uint8_t)((bitlen >> 16) & 0xFF);
        last_block_shared[60] = (uint8_t)((bitlen >> 24) & 0xFF);
        last_block_shared[59] = (uint8_t)((bitlen >> 32) & 0xFF);
        last_block_shared[58] = (uint8_t)((bitlen >> 40) & 0xFF);
        last_block_shared[57] = (uint8_t)((bitlen >> 48) & 0xFF);
        last_block_shared[56] = (uint8_t)((bitlen >> 56) & 0xFF);
    }
    __syncwarp(0xFFFFFFFFu);

    // All lanes participate in final compress
    sha256_compress_block(H, last_block_shared);

    // Lane 0 stores PRK
    if (lid == 0) {
        sha256_store_digest(H, prk_out + m*32);
    }
}
