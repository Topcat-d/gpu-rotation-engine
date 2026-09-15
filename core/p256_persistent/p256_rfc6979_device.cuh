/*
 * Card 13: Device-Side RFC 6979 Nonce Generation for P-256
 *
 * This header provides device-callable RFC 6979 deterministic nonce generation
 * using inline SHA-256 and HMAC-SHA-256 implementations.
 *
 * The implementation is bit-for-bit identical to the CPU RFC 6979 in
 * core/persistent_engine/rfc6979_p256.cpp for validation.
 *
 * Algorithm (RFC 6979 Section 3.2):
 *   Input: private key d, message hash h1
 *
 *   Step b: V = 0x01 * 32 bytes
 *   Step c: K = 0x00 * 32 bytes
 *   Step d: K = HMAC_K(V || 0x00 || d || h1)
 *   Step e: V = HMAC_K(V)
 *   Step f: K = HMAC_K(V || 0x01 || d || h1)
 *   Step g: V = HMAC_K(V)
 *   Step h: Loop until valid k:
 *           - T = HMAC_K(V); V = T
 *           - k = bits2int(T)
 *           - If 1 <= k < n: return k
 *           - Else: K = HMAC_K(V || 0x00); V = HMAC_K(V)
 */

#ifndef P256_RFC6979_DEVICE_CUH
#define P256_RFC6979_DEVICE_CUH

#include <cuda_runtime.h>
#include <stdint.h>

namespace smoke {
namespace p256 {

// ============================================================================
// SHA-256 Constants (Device)
// ============================================================================

__device__ __constant__ uint32_t SHA256_K_DEVICE[64] = {
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

__device__ __constant__ uint32_t SHA256_H0_DEVICE[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// P-256 curve order n (big-endian bytes for comparison)
__device__ __constant__ uint8_t P256_N_BE[32] = {
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xbc, 0xe6, 0xfa, 0xad, 0xa7, 0x17, 0x9e, 0x84,
    0xf3, 0xb9, 0xca, 0xc2, 0xfc, 0x63, 0x25, 0x51
};

// ============================================================================
// SHA-256 Helper Functions (Device)
// ============================================================================

#define ROTR32_D(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH_D(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ_D(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0_D(x)       (ROTR32_D(x, 2) ^ ROTR32_D(x, 13) ^ ROTR32_D(x, 22))
#define EP1_D(x)       (ROTR32_D(x, 6) ^ ROTR32_D(x, 11) ^ ROTR32_D(x, 25))
#define SIG0_D(x)      (ROTR32_D(x, 7) ^ ROTR32_D(x, 18) ^ ((x) >> 3))
#define SIG1_D(x)      (ROTR32_D(x, 17) ^ ROTR32_D(x, 19) ^ ((x) >> 10))

__device__ void sha256_transform_device(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h, t1, t2;

    // Prepare message schedule (big-endian)
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[i*4+0] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }

    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        W[i] = SIG1_D(W[i-2]) + W[i-7] + SIG0_D(W[i-15]) + W[i-16];
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    #pragma unroll 8
    for (int i = 0; i < 64; ++i) {
        t1 = h + EP1_D(e) + CH_D(e, f, g) + SHA256_K_DEVICE[i] + W[i];
        t2 = EP0_D(a) + MAJ_D(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/*
 * Device-side SHA-256 for arbitrary-length data.
 * Output: 32 bytes in hash[].
 */
__device__ void sha256_device(uint8_t hash[32], const uint8_t* data, int len) {
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        state[i] = SHA256_H0_DEVICE[i];
    }

    // Process full blocks
    int num_blocks = len / 64;
    for (int i = 0; i < num_blocks; ++i) {
        sha256_transform_device(state, data + i * 64);
    }

    // Padding
    uint8_t pad_block[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) pad_block[i] = 0;

    int remaining = len % 64;
    for (int i = 0; i < remaining; ++i) {
        pad_block[i] = data[num_blocks * 64 + i];
    }
    pad_block[remaining] = 0x80;

    uint64_t bit_len = (uint64_t)len * 8;
    if (remaining >= 56) {
        // Need two blocks
        sha256_transform_device(state, pad_block);
        #pragma unroll
        for (int i = 0; i < 64; ++i) pad_block[i] = 0;
        // Length goes in second block
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            pad_block[56 + i] = (uint8_t)(bit_len >> (56 - i * 8));
        }
        sha256_transform_device(state, pad_block);
    } else {
        // Length fits in first padding block
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            pad_block[56 + i] = (uint8_t)(bit_len >> (56 - i * 8));
        }
        sha256_transform_device(state, pad_block);
    }

    // Output hash (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        hash[i*4+0] = (uint8_t)(state[i] >> 24);
        hash[i*4+1] = (uint8_t)(state[i] >> 16);
        hash[i*4+2] = (uint8_t)(state[i] >> 8);
        hash[i*4+3] = (uint8_t)(state[i]);
    }
}

// ============================================================================
// HMAC-SHA-256 (Device)
// ============================================================================

#define SHA256_BLOCK_SIZE 64
#define SHA256_HASH_SIZE  32

/*
 * Device-side HMAC-SHA-256.
 * Output: 32 bytes in mac[].
 */
__device__ void hmac_sha256_device(
    uint8_t mac[32],
    const uint8_t* key, int key_len,
    const uint8_t* data, int data_len
) {
    uint8_t k_ipad[SHA256_BLOCK_SIZE];
    uint8_t k_opad[SHA256_BLOCK_SIZE];
    uint8_t key_block[SHA256_BLOCK_SIZE];

    // Prepare key (hash if > 64 bytes, pad if < 64 bytes)
    if (key_len > SHA256_BLOCK_SIZE) {
        sha256_device(key_block, key, key_len);
        #pragma unroll
        for (int i = SHA256_HASH_SIZE; i < SHA256_BLOCK_SIZE; ++i) {
            key_block[i] = 0;
        }
    } else {
        for (int i = 0; i < key_len; ++i) {
            key_block[i] = key[i];
        }
        for (int i = key_len; i < SHA256_BLOCK_SIZE; ++i) {
            key_block[i] = 0;
        }
    }

    // Compute ipad and opad
    #pragma unroll
    for (int i = 0; i < SHA256_BLOCK_SIZE; ++i) {
        k_ipad[i] = key_block[i] ^ 0x36;
        k_opad[i] = key_block[i] ^ 0x5c;
    }

    // Inner hash: H(k_ipad || data)
    // For RFC 6979, data_len is at most 97 bytes, so 256 is plenty
    uint8_t inner_data[256];
    #pragma unroll
    for (int i = 0; i < SHA256_BLOCK_SIZE; ++i) {
        inner_data[i] = k_ipad[i];
    }
    for (int i = 0; i < data_len; ++i) {
        inner_data[SHA256_BLOCK_SIZE + i] = data[i];
    }

    uint8_t inner_hash[SHA256_HASH_SIZE];
    sha256_device(inner_hash, inner_data, SHA256_BLOCK_SIZE + data_len);

    // Outer hash: H(k_opad || inner_hash)
    uint8_t outer_data[SHA256_BLOCK_SIZE + SHA256_HASH_SIZE];
    #pragma unroll
    for (int i = 0; i < SHA256_BLOCK_SIZE; ++i) {
        outer_data[i] = k_opad[i];
    }
    #pragma unroll
    for (int i = 0; i < SHA256_HASH_SIZE; ++i) {
        outer_data[SHA256_BLOCK_SIZE + i] = inner_hash[i];
    }

    sha256_device(mac, outer_data, SHA256_BLOCK_SIZE + SHA256_HASH_SIZE);
}

// ============================================================================
// Big Integer Comparison (Device)
// ============================================================================

// Compare two 32-byte big-endian integers
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
__device__ int cmp_be32_device(const uint8_t a[32], const uint8_t b[32]) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// Check if 32-byte big-endian integer is zero
__device__ bool is_zero_be32_device(const uint8_t a[32]) {
    uint32_t z = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        z |= a[i];
    }
    return (z == 0);
}

// ============================================================================
// RFC 6979 Device Implementation
// ============================================================================

/*
 * rfc6979_generate_k_device - Device-callable RFC 6979 nonce generation
 *
 * Generates deterministic nonce k for P-256 ECDSA using HMAC-SHA-256.
 * This implementation is bit-for-bit identical to the CPU version.
 *
 * Parameters:
 *   k_out[32]: Output nonce (big-endian bytes)
 *   private_key[32]: Private key d (big-endian bytes)
 *   hash[32]: Message hash h1 (big-endian bytes)
 *
 * Returns: true if valid k found, false otherwise (should never fail with valid inputs)
 */
__device__ bool rfc6979_generate_k_device(
    uint8_t k_out[32],
    const uint8_t private_key[32],
    const uint8_t hash[32]
) {
    // Step b: V = 0x01 * 32 bytes
    uint8_t V[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) V[i] = 0x01;

    // Step c: K = 0x00 * 32 bytes
    uint8_t K[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) K[i] = 0x00;

    // Step d: K = HMAC_K(V || 0x00 || int2octets(x) || bits2octets(h1))
    uint8_t hmac_data[97];  // V(32) + 0x00(1) + x(32) + h1(32) = 97
    #pragma unroll
    for (int i = 0; i < 32; ++i) hmac_data[i] = V[i];
    hmac_data[32] = 0x00;
    #pragma unroll
    for (int i = 0; i < 32; ++i) hmac_data[33 + i] = private_key[i];
    #pragma unroll
    for (int i = 0; i < 32; ++i) hmac_data[65 + i] = hash[i];
    hmac_sha256_device(K, K, 32, hmac_data, 97);

    // Step e: V = HMAC_K(V)
    uint8_t temp[32];
    hmac_sha256_device(temp, K, 32, V, 32);
    #pragma unroll
    for (int i = 0; i < 32; ++i) V[i] = temp[i];

    // Step f: K = HMAC_K(V || 0x01 || int2octets(x) || bits2octets(h1))
    #pragma unroll
    for (int i = 0; i < 32; ++i) hmac_data[i] = V[i];
    hmac_data[32] = 0x01;
    #pragma unroll
    for (int i = 0; i < 32; ++i) hmac_data[33 + i] = private_key[i];
    #pragma unroll
    for (int i = 0; i < 32; ++i) hmac_data[65 + i] = hash[i];
    hmac_sha256_device(temp, K, 32, hmac_data, 97);
    #pragma unroll
    for (int i = 0; i < 32; ++i) K[i] = temp[i];

    // Step g: V = HMAC_K(V)
    hmac_sha256_device(temp, K, 32, V, 32);
    #pragma unroll
    for (int i = 0; i < 32; ++i) V[i] = temp[i];

    // Step h: Generate k
    for (int retry = 0; retry < 100; ++retry) {
        // T = HMAC_K(V), then V = T
        uint8_t T[32];
        hmac_sha256_device(T, K, 32, V, 32);
        #pragma unroll
        for (int i = 0; i < 32; ++i) V[i] = T[i];

        // k = bits2int(T) - for P-256, T is already 32 bytes
        #pragma unroll
        for (int i = 0; i < 32; ++i) k_out[i] = T[i];

        // Check if k is in range [1, n-1]
        if (!is_zero_be32_device(k_out) && cmp_be32_device(k_out, P256_N_BE) < 0) {
            return true;  // Valid k found
        }

        // Invalid k, retry with:
        // K = HMAC_K(V || 0x00)
        uint8_t retry_data[33];
        #pragma unroll
        for (int i = 0; i < 32; ++i) retry_data[i] = V[i];
        retry_data[32] = 0x00;
        uint8_t new_K[32];
        hmac_sha256_device(new_K, K, 32, retry_data, 33);
        #pragma unroll
        for (int i = 0; i < 32; ++i) K[i] = new_K[i];

        // V = HMAC_K(V)
        uint8_t new_V[32];
        hmac_sha256_device(new_V, K, 32, V, 32);
        #pragma unroll
        for (int i = 0; i < 32; ++i) V[i] = new_V[i];
    }

    // Should never reach here with valid inputs
    return false;
}

/*
 * rfc6979_generate_k_limbs_device - Device-callable RFC 6979 with limb output
 *
 * Generates deterministic nonce k for P-256 ECDSA and outputs as little-endian limbs.
 * This is the format needed by the P-256 scalar multiplication.
 *
 * Parameters:
 *   k_out[8]: Output nonce as little-endian 32-bit limbs (limb[0] = LSW)
 *   private_key[32]: Private key d (big-endian bytes)
 *   hash[32]: Message hash h1 (big-endian bytes)
 *
 * Returns: true if valid k found, false otherwise
 */
__device__ bool rfc6979_generate_k_limbs_device(
    uint32_t k_out[8],
    const uint8_t private_key[32],
    const uint8_t hash[32]
) {
    uint8_t k_bytes[32];
    if (!rfc6979_generate_k_device(k_bytes, private_key, hash)) {
        return false;
    }

    // Convert big-endian bytes to little-endian limbs
    // k_bytes[0..3] = MSW, k_bytes[28..31] = LSW
    // k_out[0] = LSW, k_out[7] = MSW
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int byte_base = (7 - i) * 4;
        k_out[i] = ((uint32_t)k_bytes[byte_base+3]) |
                   ((uint32_t)k_bytes[byte_base+2] << 8) |
                   ((uint32_t)k_bytes[byte_base+1] << 16) |
                   ((uint32_t)k_bytes[byte_base+0] << 24);
    }

    return true;
}

} // namespace p256
} // namespace smoke

#endif // P256_RFC6979_DEVICE_CUH
