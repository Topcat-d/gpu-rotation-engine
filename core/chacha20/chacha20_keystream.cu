/**
 * ChaCha20 Keystream - Standalone Implementation (RFC 8439)
 *
 * Pure ChaCha20 keystream generation and XOR operations
 * Reuses Phase Q-1 optimizations from AEAD kernel
 *
 * Use cases:
 * - QUIC/TLS header protection
 * - Generic stream cipher (encrypt/decrypt = XOR)
 * - Building block for XChaCha20
 *
 * Performance targets (RTX 2060 SUPER):
 * - Sequential: ≥250 Gb/s @ 16KB
 * - Batched: ≥220 Gb/s @ 4KB
 * - Small: ≥40 Gb/s @ 512B
 *
 * Author: PyTorch Crypto Project
 * Date: October 29, 2025
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define CHACHA_BLOCK_SIZE 64  // ChaCha20: 64 bytes per block

// ============================================================================
// ChaCha20 Core (from Phase Q-1 AEAD)
// ============================================================================

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return __funnelshift_l(x, x, n);  // GPU intrinsic
}

__device__ __forceinline__ void quarterround(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

/**
 * ChaCha20 block function
 *
 * Generates 64 bytes of keystream from key + nonce + counter
 * RFC 8439 compliant (counter is 32-bit)
 */
__device__ __forceinline__ void chacha20_block(
    uint32_t out[16],
    const uint32_t key[8],
    const uint32_t nonce[3],
    uint32_t counter
) {
    // Initialize state
    uint32_t x[16];

    // Constants ("expand 32-byte k")
    x[0] = 0x61707865;
    x[1] = 0x3320646e;
    x[2] = 0x79622d32;
    x[3] = 0x6b206574;

    // Key (32 bytes = 8 words)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x[4 + i] = key[i];
    }

    // Counter (32-bit) + Nonce (96-bit = 3 words)
    x[12] = counter;
    x[13] = nonce[0];
    x[14] = nonce[1];
    x[15] = nonce[2];

    // Save initial state for final add
    uint32_t initial[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        initial[i] = x[i];
    }

    // 20 rounds (10 double rounds)
    #pragma unroll 10
    for (int i = 0; i < 10; i++) {
        // Column rounds
        quarterround(x[0], x[4], x[8], x[12]);
        quarterround(x[1], x[5], x[9], x[13]);
        quarterround(x[2], x[6], x[10], x[14]);
        quarterround(x[3], x[7], x[11], x[15]);

        // Diagonal rounds
        quarterround(x[0], x[5], x[10], x[15]);
        quarterround(x[1], x[6], x[11], x[12]);
        quarterround(x[2], x[7], x[8], x[13]);
        quarterround(x[3], x[4], x[9], x[14]);
    }

    // Add initial state
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        out[i] = x[i] + initial[i];
    }
}

// ============================================================================
// ChaCha20 Keystream XOR Kernel (Batched)
// ============================================================================

/**
 * ChaCha20 keystream XOR - one thread per message
 *
 * Each thread:
 * 1. Generates ChaCha20 keystream blocks
 * 2. XORs with input data
 * 3. Writes output
 *
 * Symmetric: encrypt and decrypt are identical (XOR is reversible)
 *
 * @param input         Input data [batch_size, data_len]
 * @param output        Output data [batch_size, data_len]
 * @param keys          Keys (32 bytes each) [batch_size, 8 words]
 * @param nonces        Nonces (12 bytes each) [batch_size, 3 words]
 * @param counters      Initial counters [batch_size] (usually 0 or 1)
 * @param data_lens     Data lengths [batch_size]
 * @param batch_size    Number of messages
 * @param max_data_len  Maximum data length (for array sizing)
 */
__launch_bounds__(256, 3)
extern "C" __global__ void chacha20_xor_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ keys,
    const uint32_t* __restrict__ nonces,
    const uint32_t* __restrict__ counters,
    const int* __restrict__ data_lens,
    int batch_size,
    int max_data_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Load key and nonce
    const uint32_t* key = keys + tid * 8;
    const uint32_t* nonce = nonces + tid * 3;
    uint32_t counter = counters[tid];

    // Get data pointers
    const uint8_t* in_ptr = input + tid * max_data_len;
    uint8_t* out_ptr = output + tid * max_data_len;
    int data_len = data_lens[tid];

    // Process in 64-byte blocks
    int num_blocks = (data_len + 63) / 64;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Generate keystream block
        uint32_t keystream[16];
        chacha20_block(keystream, key, nonce, counter + block_idx);

        // Convert to bytes for XOR
        uint8_t* ks_bytes = (uint8_t*)keystream;

        // XOR with input (handle partial last block)
        int block_offset = block_idx * 64;
        int block_len = min(64, data_len - block_offset);

        #pragma unroll
        for (int i = 0; i < block_len; i++) {
            out_ptr[block_offset + i] = in_ptr[block_offset + i] ^ ks_bytes[i];
        }
    }
}

/**
 * ChaCha20 keystream generation (no XOR)
 *
 * Generates pure keystream bytes
 * Useful for protocols that need keystream separately
 *
 * @param keystream     Output keystream [batch_size, keystream_len]
 * @param keys          Keys (32 bytes each) [batch_size, 8 words]
 * @param nonces        Nonces (12 bytes each) [batch_size, 3 words]
 * @param counters      Initial counters [batch_size]
 * @param keystream_lens Keystream lengths [batch_size]
 * @param batch_size    Number of messages
 * @param max_keystream_len Maximum keystream length
 */
__launch_bounds__(256, 3)
extern "C" __global__ void chacha20_keystream_kernel(
    uint8_t* __restrict__ keystream,
    const uint32_t* __restrict__ keys,
    const uint32_t* __restrict__ nonces,
    const uint32_t* __restrict__ counters,
    const int* __restrict__ keystream_lens,
    int batch_size,
    int max_keystream_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Load key and nonce
    const uint32_t* key = keys + tid * 8;
    const uint32_t* nonce = nonces + tid * 3;
    uint32_t counter = counters[tid];

    // Get output pointer
    uint8_t* ks_ptr = keystream + tid * max_keystream_len;
    int ks_len = keystream_lens[tid];

    // Generate keystream blocks
    int num_blocks = (ks_len + 63) / 64;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Generate keystream block
        uint32_t ks_block[16];
        chacha20_block(ks_block, key, nonce, counter + block_idx);

        // Write to output (handle partial last block)
        uint8_t* ks_bytes = (uint8_t*)ks_block;
        int block_offset = block_idx * 64;
        int block_len = min(64, ks_len - block_offset);

        #pragma unroll
        for (int i = 0; i < block_len; i++) {
            ks_ptr[block_offset + i] = ks_bytes[i];
        }
    }
}

// ============================================================================
// Kernel Launchers
// ============================================================================

extern "C" void chacha20_xor_launcher(
    const uint8_t* input,
    uint8_t* output,
    const uint32_t* keys,
    const uint32_t* nonces,
    const uint32_t* counters,
    const int* data_lens,
    int batch_size,
    int max_data_len,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    chacha20_xor_kernel<<<blocks, threads, 0, stream>>>(
        input, output, keys, nonces, counters, data_lens,
        batch_size, max_data_len
    );
}

extern "C" void chacha20_keystream_launcher(
    uint8_t* keystream,
    const uint32_t* keys,
    const uint32_t* nonces,
    const uint32_t* counters,
    const int* keystream_lens,
    int batch_size,
    int max_keystream_len,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    chacha20_keystream_kernel<<<blocks, threads, 0, stream>>>(
        keystream, keys, nonces, counters, keystream_lens,
        batch_size, max_keystream_len
    );
}
