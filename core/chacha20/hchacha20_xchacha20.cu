/*
 * HChaCha20 + XChaCha20 - Extended Nonce ChaCha20
 *
 * HChaCha20: Hash function to derive 32-byte subkey from key + 16-byte nonce
 * XChaCha20: ChaCha20 with 24-byte nonce (using HChaCha20 for nonce extension)
 *
 * Reference: draft-irtf-cfrg-xchacha-03
 * https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-xchacha-03
 *
 * Performance: Same as ChaCha20 (one extra HChaCha20 call per session)
 * Register usage: ~50-55 registers (similar to ChaCha20)
 *
 * Author: PyTorch Crypto Project
 * Date: October 30, 2025
 */

#include <cuda_runtime.h>
#include <cstdint>

// ChaCha20 constants ("expand 32-byte k")
#define CHACHA_CONST0 0x61707865  // "expa"
#define CHACHA_CONST1 0x3320646e  // "nd 3"
#define CHACHA_CONST2 0x79622d32  // "2-by"
#define CHACHA_CONST3 0x6b206574  // "te k"

// Optimized rotation using GPU intrinsic (Phase Q-1 optimization)
__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
    return __funnelshift_l(x, x, n);
}

// ChaCha20 quarterround (same as base ChaCha20)
__device__ __forceinline__ void quarterround(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d
) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

/*
 * HChaCha20: Derive 32-byte subkey from key + 16-byte nonce
 *
 * Input:
 *   key[8]: 32-byte key as 8 words (little-endian)
 *   nonce[4]: 16-byte nonce as 4 words (little-endian)
 *
 * Output:
 *   subkey[8]: 32-byte subkey as 8 words
 *
 * Algorithm:
 *   1. Initialize state: constants[4] + key[8] + nonce[4]
 *   2. Run 20 rounds (same as ChaCha20)
 *   3. Output: state[0:4] + state[12:16] (skip middle)
 */
__device__ __forceinline__ void hchacha20_block(
    uint32_t subkey[8],
    const uint32_t key[8],
    const uint32_t nonce[4]
) {
    // Initialize state
    uint32_t x[16];

    // Constants
    x[0] = CHACHA_CONST0;
    x[1] = CHACHA_CONST1;
    x[2] = CHACHA_CONST2;
    x[3] = CHACHA_CONST3;

    // Key
    for (int i = 0; i < 8; i++) {
        x[4 + i] = key[i];
    }

    // Nonce (16 bytes = 4 words)
    for (int i = 0; i < 4; i++) {
        x[12 + i] = nonce[i];
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

    // Output: first 128 bits + last 128 bits (skip middle)
    subkey[0] = x[0];
    subkey[1] = x[1];
    subkey[2] = x[2];
    subkey[3] = x[3];
    subkey[4] = x[12];
    subkey[5] = x[13];
    subkey[6] = x[14];
    subkey[7] = x[15];
}

/*
 * HChaCha20 Kernel - Batched Subkey Derivation
 *
 * Derives subkeys for multiple key-nonce pairs in parallel
 *
 * Input:
 *   keys: [batch_size, 8] int32 (32-byte keys as 8 words)
 *   nonces: [batch_size, 4] int32 (16-byte nonces as 4 words)
 *   batch_size: Number of key-nonce pairs
 *
 * Output:
 *   subkeys: [batch_size, 8] int32 (32-byte subkeys as 8 words)
 */
extern "C" __global__ void __launch_bounds__(256, 3) hchacha20_derive_subkeys(
    const int* __restrict__ keys,
    const int* __restrict__ nonces,
    int* __restrict__ subkeys,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size) return;

    // Load key (8 words)
    uint32_t key[8];
    for (int i = 0; i < 8; i++) {
        key[i] = (uint32_t)keys[idx * 8 + i];
    }

    // Load nonce (4 words)
    uint32_t nonce[4];
    for (int i = 0; i < 4; i++) {
        nonce[i] = (uint32_t)nonces[idx * 4 + i];
    }

    // Derive subkey
    uint32_t subkey[8];
    hchacha20_block(subkey, key, nonce);

    // Store subkey (8 words)
    for (int i = 0; i < 8; i++) {
        subkeys[idx * 8 + i] = (int)subkey[i];
    }
}

/*
 * XChaCha20 XOR Kernel - Encrypt/Decrypt with 24-byte nonce
 *
 * XChaCha20 = HChaCha20(key, nonce[:16]) + ChaCha20(subkey, [0,0,0,0] + nonce[16:24])
 *
 * Input:
 *   input: [batch_size, max_data_len] uint8
 *   keys: [batch_size, 8] int32 (32-byte keys)
 *   nonces_24: [batch_size, 6] int32 (24-byte nonces as 6 words)
 *   counters: [batch_size] int32
 *   data_lens: [batch_size] int32
 *   batch_size: Number of messages
 *   max_data_len: Maximum message length
 *
 * Output:
 *   output: [batch_size, max_data_len] uint8 (ciphertext or plaintext)
 */
extern "C" __global__ void __launch_bounds__(256, 3) xchacha20_xor_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const int* __restrict__ keys,
    const int* __restrict__ nonces_24,
    const int* __restrict__ counters,
    const int* __restrict__ data_lens,
    int batch_size,
    int max_data_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size) return;

    int data_len = data_lens[idx];
    if (data_len <= 0) return;

    // Load key (8 words = 32 bytes)
    uint32_t key[8];
    for (int i = 0; i < 8; i++) {
        key[i] = (uint32_t)keys[idx * 8 + i];
    }

    // Load 24-byte nonce (6 words)
    uint32_t nonce_24[6];
    for (int i = 0; i < 6; i++) {
        nonce_24[i] = (uint32_t)nonces_24[idx * 6 + i];
    }

    // Step 1: Derive subkey using HChaCha20(key, nonce[:16])
    uint32_t hchacha_nonce[4];
    hchacha_nonce[0] = nonce_24[0];
    hchacha_nonce[1] = nonce_24[1];
    hchacha_nonce[2] = nonce_24[2];
    hchacha_nonce[3] = nonce_24[3];

    uint32_t subkey[8];
    hchacha20_block(subkey, key, hchacha_nonce);

    // Step 2: Use ChaCha20 with subkey and remaining nonce (nonce[16:24])
    uint32_t chacha_nonce[3];
    chacha_nonce[0] = 0;  // Padding
    chacha_nonce[1] = nonce_24[4];
    chacha_nonce[2] = nonce_24[5];

    uint32_t counter = (uint32_t)counters[idx];

    // Process data in 64-byte blocks
    const uint8_t* in_ptr = input + idx * max_data_len;
    uint8_t* out_ptr = output + idx * max_data_len;

    for (int block_idx = 0; block_idx < (data_len + 63) / 64; block_idx++) {
        // Initialize ChaCha20 state
        uint32_t state[16];

        // Constants
        state[0] = CHACHA_CONST0;
        state[1] = CHACHA_CONST1;
        state[2] = CHACHA_CONST2;
        state[3] = CHACHA_CONST3;

        // Subkey (from HChaCha20)
        for (int i = 0; i < 8; i++) {
            state[4 + i] = subkey[i];
        }

        // Counter
        state[12] = counter + block_idx;

        // Nonce (8 bytes from nonce[16:24])
        state[13] = chacha_nonce[0];
        state[14] = chacha_nonce[1];
        state[15] = chacha_nonce[2];

        // Save initial state
        uint32_t initial[16];
        for (int i = 0; i < 16; i++) {
            initial[i] = state[i];
        }

        // 20 rounds (10 double rounds)
        #pragma unroll 10
        for (int i = 0; i < 10; i++) {
            // Column rounds
            quarterround(state[0], state[4], state[8], state[12]);
            quarterround(state[1], state[5], state[9], state[13]);
            quarterround(state[2], state[6], state[10], state[14]);
            quarterround(state[3], state[7], state[11], state[15]);

            // Diagonal rounds
            quarterround(state[0], state[5], state[10], state[15]);
            quarterround(state[1], state[6], state[11], state[12]);
            quarterround(state[2], state[7], state[8], state[13]);
            quarterround(state[3], state[4], state[9], state[14]);
        }

        // Add initial state
        for (int i = 0; i < 16; i++) {
            state[i] += initial[i];
        }

        // XOR keystream with plaintext/ciphertext
        int block_start = block_idx * 64;
        int block_end = min(block_start + 64, data_len);

        for (int pos = block_start; pos < block_end; pos++) {
            int state_idx = (pos - block_start) / 4;
            int byte_idx = (pos - block_start) % 4;

            uint8_t keystream_byte = (state[state_idx] >> (byte_idx * 8)) & 0xff;
            out_ptr[pos] = in_ptr[pos] ^ keystream_byte;
        }
    }
}

// Host launcher functions (called from C++ binding)

extern "C" void hchacha20_derive_subkeys_launcher(
    const int* keys,
    const int* nonces,
    int* subkeys,
    int batch_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    hchacha20_derive_subkeys<<<blocks, threads, 0, stream>>>(
        keys, nonces, subkeys, batch_size
    );
}

extern "C" void xchacha20_xor_launcher(
    const uint8_t* input,
    uint8_t* output,
    const int* keys,
    const int* nonces_24,
    const int* counters,
    const int* data_lens,
    int batch_size,
    int max_data_len,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    xchacha20_xor_kernel<<<blocks, threads, 0, stream>>>(
        input, output, keys, nonces_24, counters, data_lens,
        batch_size, max_data_len
    );
}
