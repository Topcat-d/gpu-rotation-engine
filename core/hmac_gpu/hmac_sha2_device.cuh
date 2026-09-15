// SPDX-License-Identifier: Apache-2.0
// Smoke KDFs — Device-side HMAC-SHA2 (SHA-256 / SHA-512)
// These are __device__ functions callable from within HKDF kernels

#ifndef SMOKE_HMAC_SHA2_DEVICE_CUH
#define SMOKE_HMAC_SHA2_DEVICE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// SHA-256 Device-Side Implementation
// ============================================================================

// SHA-256 initial hash values
__device__ __constant__ uint32_t H256_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// SHA-256 round constants
__constant__ uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

// SHA-256 compression function (device-side)
__device__ __forceinline__
void sha256_compress_device(uint32_t state[8], const uint32_t block[16]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    uint32_t W[64];

    // Load message schedule
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        W[t] = block[t];
    }

    // Expand
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        uint32_t s0 = rotr32(W[t-15], 7) ^ rotr32(W[t-15], 18) ^ (W[t-15] >> 3);
        uint32_t s1 = rotr32(W[t-2], 17) ^ rotr32(W[t-2], 19) ^ (W[t-2] >> 10);
        W[t] = W[t-16] + s0 + W[t-7] + s1;
    }

    // 64 rounds
    #pragma unroll 8
    for (int t = 0; t < 64; ++t) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K256[t] + W[t];

        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    // Add to state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// SHA-256 padding and hash (device-side)
// Processes arbitrary-length message, returns 32-byte digest
__device__
void sha256_hash_device(const uint8_t* msg, size_t msg_len, uint8_t* digest) {
    uint32_t state[8];

    // Initialize state
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        state[i] = H256_INIT[i];
    }

    const size_t block_size = 64; // 512 bits
    size_t num_blocks = msg_len / block_size;

    // Process complete blocks
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        uint32_t block[16];
        const uint8_t* src = msg + blk * block_size;

        // Convert bytes to big-endian uint32_t
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block[i] = ((uint32_t)src[i*4 + 0] << 24) |
                       ((uint32_t)src[i*4 + 1] << 16) |
                       ((uint32_t)src[i*4 + 2] << 8) |
                       ((uint32_t)src[i*4 + 3]);
        }
        sha256_compress_device(state, block);
    }

    // Process final block with padding
    size_t remaining = msg_len % block_size;
    uint8_t final_block[128] = {0}; // Up to 2 blocks

    // Copy remaining bytes
    const uint8_t* src = msg + num_blocks * block_size;
    for (size_t i = 0; i < remaining; ++i) {
        final_block[i] = src[i];
    }

    // Append 0x80
    final_block[remaining] = 0x80;

    // Calculate total bit length
    uint64_t bit_len = msg_len * 8;

    // Determine if we need 1 or 2 blocks
    size_t pad_blocks = (remaining < 56) ? 1 : 2;

    // Append length in last 8 bytes (big-endian)
    size_t len_offset = pad_blocks * 64 - 8;
    for (int i = 0; i < 8; ++i) {
        final_block[len_offset + i] = (bit_len >> (56 - i*8)) & 0xFF;
    }

    // Process final block(s)
    for (size_t blk = 0; blk < pad_blocks; ++blk) {
        uint32_t block[16];
        const uint8_t* blk_src = final_block + blk * 64;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block[i] = ((uint32_t)blk_src[i*4 + 0] << 24) |
                       ((uint32_t)blk_src[i*4 + 1] << 16) |
                       ((uint32_t)blk_src[i*4 + 2] << 8) |
                       ((uint32_t)blk_src[i*4 + 3]);
        }
        sha256_compress_device(state, block);
    }

    // Convert state to bytes (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        digest[i*4 + 0] = (state[i] >> 24) & 0xFF;
        digest[i*4 + 1] = (state[i] >> 16) & 0xFF;
        digest[i*4 + 2] = (state[i] >> 8) & 0xFF;
        digest[i*4 + 3] = state[i] & 0xFF;
    }
}

// SHA-256 of (prefix[64] || msg[msg_len]) computed by STREAMING — the 64-byte
// prefix is absorbed as one block, then the message is absorbed block-by-block
// directly from its source pointer (e.g. the mapped payload slab). Only a
// 128-byte final block ever lives on the stack, regardless of msg_len. This is
// the fail-closed replacement for the old HMAC inner hash, which copied
// ipad||msg into a fixed 1088-byte stack buffer and SILENTLY produced no output
// for msg_len > 1024 (see the removed `if (inner_len <= max_inner)` guard).
__device__
void sha256_prefix_hash_device(const uint8_t prefix[64],
                               const uint8_t* msg, size_t msg_len,
                               uint8_t out32[32]) {
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[i] = H256_INIT[i];

    // Absorb the 64-byte prefix as exactly one block.
    {
        uint32_t block[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block[i] = ((uint32_t)prefix[i*4 + 0] << 24) |
                       ((uint32_t)prefix[i*4 + 1] << 16) |
                       ((uint32_t)prefix[i*4 + 2] << 8) |
                       ((uint32_t)prefix[i*4 + 3]);
        }
        sha256_compress_device(state, block);
    }

    // Absorb full 64-byte blocks of the message, streamed from source.
    const size_t block_size = 64;
    size_t num_blocks = msg_len / block_size;
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        uint32_t block[16];
        const uint8_t* src = msg + blk * block_size;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block[i] = ((uint32_t)src[i*4 + 0] << 24) |
                       ((uint32_t)src[i*4 + 1] << 16) |
                       ((uint32_t)src[i*4 + 2] << 8) |
                       ((uint32_t)src[i*4 + 3]);
        }
        sha256_compress_device(state, block);
    }

    // Final block(s): trailing message bytes + 0x80 + total bit length.
    // Total absorbed length = 64 (prefix) + msg_len.
    size_t remaining = msg_len % block_size;
    uint8_t final_block[128] = {0};
    const uint8_t* src = msg + num_blocks * block_size;
    for (size_t i = 0; i < remaining; ++i) final_block[i] = src[i];
    final_block[remaining] = 0x80;

    uint64_t bit_len = (uint64_t)(block_size + msg_len) * 8;
    size_t pad_blocks = (remaining < 56) ? 1 : 2;
    size_t len_offset = pad_blocks * 64 - 8;
    for (int i = 0; i < 8; ++i) {
        final_block[len_offset + i] = (bit_len >> (56 - i*8)) & 0xFF;
    }
    for (size_t blk = 0; blk < pad_blocks; ++blk) {
        uint32_t block[16];
        const uint8_t* bs = final_block + blk * 64;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block[i] = ((uint32_t)bs[i*4 + 0] << 24) |
                       ((uint32_t)bs[i*4 + 1] << 16) |
                       ((uint32_t)bs[i*4 + 2] << 8) |
                       ((uint32_t)bs[i*4 + 3]);
        }
        sha256_compress_device(state, block);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out32[i*4 + 0] = (state[i] >> 24) & 0xFF;
        out32[i*4 + 1] = (state[i] >> 16) & 0xFF;
        out32[i*4 + 2] = (state[i] >> 8) & 0xFF;
        out32[i*4 + 3] = state[i] & 0xFF;
    }
}

// HMAC-SHA256 (device-side)
extern "C" __device__
void hmac_sha256_device(
    const uint8_t* key, size_t key_len,
    const uint8_t* msg, size_t msg_len,
    uint8_t* out32)
{
    const size_t block_size = 64; // SHA-256 block size
    uint8_t key_block[64] = {0};

    // If key > block_size, hash it first
    if (key_len > block_size) {
        sha256_hash_device(key, key_len, key_block);
        // Rest already zeroed
    } else {
        // Copy key and zero-pad
        for (size_t i = 0; i < key_len; ++i) {
            key_block[i] = key[i];
        }
    }

    // ipad = key_block XOR 0x36
    uint8_t ipad[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        ipad[i] = key_block[i] ^ 0x36;
    }

    // opad = key_block XOR 0x5c
    uint8_t opad[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        opad[i] = key_block[i] ^ 0x5c;
    }

    // Inner hash: H(ipad || msg) — STREAMED, no length cap, no large stack
    // buffer. Handles arbitrary msg_len (KB-scale payload-slab messages), fixing
    // the prior fail-OPEN that skipped output entirely for msg_len > 1024.
    uint8_t inner_hash[32];
    sha256_prefix_hash_device(ipad, msg, msg_len, inner_hash);

    // Outer hash: H(opad || inner_hash) — fixed 96 bytes.
    uint8_t outer_msg[64 + 32];
    #pragma unroll
    for (int i = 0; i < 64; ++i) outer_msg[i] = opad[i];
    #pragma unroll
    for (int i = 0; i < 32; ++i) outer_msg[64 + i] = inner_hash[i];
    sha256_hash_device(outer_msg, 96, out32);

    // Scrub key-derived material from the stack (do not leak ipad/opad/key_block
    // or the inner digest across the thread's next work item).
    #pragma unroll
    for (int i = 0; i < 64; ++i) { key_block[i] = 0; ipad[i] = 0; opad[i] = 0; }
    #pragma unroll
    for (int i = 0; i < 32; ++i) inner_hash[i] = 0;
}

// ============================================================================
// SHA-512 Device-Side Implementation
// ============================================================================

// SHA-512 initial hash values
__device__ __constant__ uint64_t H512_INIT[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// SHA-512 round constants
__constant__ uint64_t K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

__device__ __forceinline__ uint64_t rotr64_dev(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__
void sha512_compress_device(uint64_t state[8], const uint64_t block[16]) {
    uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];

    uint64_t W[80];

    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        W[t] = block[t];
    }

    #pragma unroll
    for (int t = 16; t < 80; ++t) {
        uint64_t s0 = rotr64_dev(W[t-15], 1) ^ rotr64_dev(W[t-15], 8) ^ (W[t-15] >> 7);
        uint64_t s1 = rotr64_dev(W[t-2], 19) ^ rotr64_dev(W[t-2], 61) ^ (W[t-2] >> 6);
        W[t] = W[t-16] + s0 + W[t-7] + s1;
    }

    #pragma unroll 8
    for (int t = 0; t < 80; ++t) {
        uint64_t S1 = rotr64_dev(e, 14) ^ rotr64_dev(e, 18) ^ rotr64_dev(e, 41);
        uint64_t ch = (e & f) ^ ((~e) & g);
        uint64_t temp1 = h + S1 + ch + K512[t] + W[t];

        uint64_t S0 = rotr64_dev(a, 28) ^ rotr64_dev(a, 34) ^ rotr64_dev(a, 39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__
void sha512_hash_device(const uint8_t* msg, size_t msg_len, uint8_t* digest) {
    uint64_t state[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        state[i] = H512_INIT[i];
    }

    const size_t block_size = 128; // 1024 bits
    size_t num_blocks = msg_len / block_size;

    // Process complete blocks
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        uint64_t block[16];
        const uint8_t* src = msg + blk * block_size;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block[i] = ((uint64_t)src[i*8 + 0] << 56) |
                       ((uint64_t)src[i*8 + 1] << 48) |
                       ((uint64_t)src[i*8 + 2] << 40) |
                       ((uint64_t)src[i*8 + 3] << 32) |
                       ((uint64_t)src[i*8 + 4] << 24) |
                       ((uint64_t)src[i*8 + 5] << 16) |
                       ((uint64_t)src[i*8 + 6] << 8) |
                       ((uint64_t)src[i*8 + 7]);
        }
        sha512_compress_device(state, block);
    }

    // Final block with padding
    size_t remaining = msg_len % block_size;
    uint8_t final_block[256] = {0}; // Up to 2 blocks

    const uint8_t* src = msg + num_blocks * block_size;
    for (size_t i = 0; i < remaining; ++i) {
        final_block[i] = src[i];
    }

    final_block[remaining] = 0x80;

    uint64_t bit_len = msg_len * 8;
    size_t pad_blocks = (remaining < 112) ? 1 : 2;

    // Append length in last 16 bytes (big-endian 128-bit, but we only use lower 64)
    size_t len_offset = pad_blocks * 128 - 8;
    for (int i = 0; i < 8; ++i) {
        final_block[len_offset + i] = (bit_len >> (56 - i*8)) & 0xFF;
    }

    for (size_t blk = 0; blk < pad_blocks; ++blk) {
        uint64_t block[16];
        const uint8_t* blk_src = final_block + blk * 128;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block[i] = ((uint64_t)blk_src[i*8 + 0] << 56) |
                       ((uint64_t)blk_src[i*8 + 1] << 48) |
                       ((uint64_t)blk_src[i*8 + 2] << 40) |
                       ((uint64_t)blk_src[i*8 + 3] << 32) |
                       ((uint64_t)blk_src[i*8 + 4] << 24) |
                       ((uint64_t)blk_src[i*8 + 5] << 16) |
                       ((uint64_t)blk_src[i*8 + 6] << 8) |
                       ((uint64_t)blk_src[i*8 + 7]);
        }
        sha512_compress_device(state, block);
    }

    // Convert state to bytes (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        digest[i*8 + 0] = (state[i] >> 56) & 0xFF;
        digest[i*8 + 1] = (state[i] >> 48) & 0xFF;
        digest[i*8 + 2] = (state[i] >> 40) & 0xFF;
        digest[i*8 + 3] = (state[i] >> 32) & 0xFF;
        digest[i*8 + 4] = (state[i] >> 24) & 0xFF;
        digest[i*8 + 5] = (state[i] >> 16) & 0xFF;
        digest[i*8 + 6] = (state[i] >> 8) & 0xFF;
        digest[i*8 + 7] = state[i] & 0xFF;
    }
}

extern "C" __device__
void hmac_sha512_device(
    const uint8_t* key, size_t key_len,
    const uint8_t* msg, size_t msg_len,
    uint8_t* out64)
{
    const size_t block_size = 128; // SHA-512 block size
    uint8_t key_block[128] = {0};

    if (key_len > block_size) {
        sha512_hash_device(key, key_len, key_block);
    } else {
        for (size_t i = 0; i < key_len; ++i) {
            key_block[i] = key[i];
        }
    }

    uint8_t ipad[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        ipad[i] = key_block[i] ^ 0x36;
    }

    uint8_t opad[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        opad[i] = key_block[i] ^ 0x5c;
    }

    const size_t max_inner = 128 + 1024;
    uint8_t inner_msg[max_inner];
    size_t inner_len = 128 + msg_len;

    if (inner_len <= max_inner) {
        for (int i = 0; i < 128; ++i) {
            inner_msg[i] = ipad[i];
        }
        for (size_t i = 0; i < msg_len; ++i) {
            inner_msg[128 + i] = msg[i];
        }

        uint8_t inner_hash[64];
        sha512_hash_device(inner_msg, inner_len, inner_hash);

        uint8_t outer_msg[128 + 64];
        for (int i = 0; i < 128; ++i) {
            outer_msg[i] = opad[i];
        }
        for (int i = 0; i < 64; ++i) {
            outer_msg[128 + i] = inner_hash[i];
        }

        sha512_hash_device(outer_msg, 128 + 64, out64);
    }
}

// ============================================================================
// SHA-384 Device-Side Implementation (Card 26.52)
// SHA-384 is SHA-512 with different initial values, truncated to 48 bytes
// ============================================================================

// SHA-384 initial hash values (different from SHA-512!)
__device__ __constant__ uint64_t H384_INIT[8] = {
    0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL,
    0x9159015a3070dd17ULL, 0x152fecd8f70e5939ULL,
    0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL,
    0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL
};

__device__
void sha384_hash_device(const uint8_t* msg, size_t msg_len, uint8_t* digest) {
    // SHA-384 uses the same compression function as SHA-512
    // but with different initial values and truncated output (48 bytes)
    uint64_t state[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        state[i] = H384_INIT[i];
    }

    const size_t block_size = 128; // Same as SHA-512
    size_t num_blocks = msg_len / block_size;

    // Process complete blocks
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        uint64_t block[16];
        const uint8_t* src = msg + blk * block_size;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block[i] = ((uint64_t)src[i*8 + 0] << 56) |
                       ((uint64_t)src[i*8 + 1] << 48) |
                       ((uint64_t)src[i*8 + 2] << 40) |
                       ((uint64_t)src[i*8 + 3] << 32) |
                       ((uint64_t)src[i*8 + 4] << 24) |
                       ((uint64_t)src[i*8 + 5] << 16) |
                       ((uint64_t)src[i*8 + 6] << 8) |
                       ((uint64_t)src[i*8 + 7]);
        }
        sha512_compress_device(state, block);
    }

    // Final block with padding (same as SHA-512)
    size_t remaining = msg_len % block_size;
    uint8_t final_block[256] = {0};

    const uint8_t* src = msg + num_blocks * block_size;
    for (size_t i = 0; i < remaining; ++i) {
        final_block[i] = src[i];
    }

    final_block[remaining] = 0x80;

    uint64_t bit_len = msg_len * 8;
    size_t pad_blocks = (remaining < 112) ? 1 : 2;

    size_t len_offset = pad_blocks * 128 - 8;
    for (int i = 0; i < 8; ++i) {
        final_block[len_offset + i] = (bit_len >> (56 - i*8)) & 0xFF;
    }

    for (size_t blk = 0; blk < pad_blocks; ++blk) {
        uint64_t block[16];
        const uint8_t* blk_src = final_block + blk * 128;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            block[i] = ((uint64_t)blk_src[i*8 + 0] << 56) |
                       ((uint64_t)blk_src[i*8 + 1] << 48) |
                       ((uint64_t)blk_src[i*8 + 2] << 40) |
                       ((uint64_t)blk_src[i*8 + 3] << 32) |
                       ((uint64_t)blk_src[i*8 + 4] << 24) |
                       ((uint64_t)blk_src[i*8 + 5] << 16) |
                       ((uint64_t)blk_src[i*8 + 6] << 8) |
                       ((uint64_t)blk_src[i*8 + 7]);
        }
        sha512_compress_device(state, block);
    }

    // Truncate to 48 bytes (first 6 of 8 state words)
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        digest[i*8 + 0] = (state[i] >> 56) & 0xFF;
        digest[i*8 + 1] = (state[i] >> 48) & 0xFF;
        digest[i*8 + 2] = (state[i] >> 40) & 0xFF;
        digest[i*8 + 3] = (state[i] >> 32) & 0xFF;
        digest[i*8 + 4] = (state[i] >> 24) & 0xFF;
        digest[i*8 + 5] = (state[i] >> 16) & 0xFF;
        digest[i*8 + 6] = (state[i] >> 8) & 0xFF;
        digest[i*8 + 7] = state[i] & 0xFF;
    }
}

extern "C" __device__
void hmac_sha384_device(
    const uint8_t* key, size_t key_len,
    const uint8_t* msg, size_t msg_len,
    uint8_t* out48)
{
    const size_t block_size = 128; // SHA-384 uses same block size as SHA-512
    uint8_t key_block[128] = {0};

    if (key_len > block_size) {
        sha384_hash_device(key, key_len, key_block);
    } else {
        for (size_t i = 0; i < key_len; ++i) {
            key_block[i] = key[i];
        }
    }

    uint8_t ipad[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        ipad[i] = key_block[i] ^ 0x36;
    }

    uint8_t opad[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        opad[i] = key_block[i] ^ 0x5c;
    }

    const size_t max_inner = 128 + 1024;
    uint8_t inner_msg[max_inner];
    size_t inner_len = 128 + msg_len;

    if (inner_len <= max_inner) {
        for (int i = 0; i < 128; ++i) {
            inner_msg[i] = ipad[i];
        }
        for (size_t i = 0; i < msg_len; ++i) {
            inner_msg[128 + i] = msg[i];
        }

        uint8_t inner_hash[48];
        sha384_hash_device(inner_msg, inner_len, inner_hash);

        uint8_t outer_msg[128 + 48];
        for (int i = 0; i < 128; ++i) {
            outer_msg[i] = opad[i];
        }
        for (int i = 0; i < 48; ++i) {
            outer_msg[128 + i] = inner_hash[i];
        }

        sha384_hash_device(outer_msg, 128 + 48, out48);
    }
}

#endif // SMOKE_HMAC_SHA2_DEVICE_CUH