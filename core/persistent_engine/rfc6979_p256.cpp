/*
 * rfc6979_p256.cpp - RFC 6979 Deterministic Nonce for P-256 ECDSA
 *
 * Card 12: Implements RFC 6979 Section 3.2 using HMAC-SHA-256.
 *
 * Algorithm (from RFC 6979 Section 3.2):
 *   Input: private key x, message hash h1
 *
 *   Step a: h1 = H(m) - already done, hash is input parameter
 *   Step b: V = 0x01 0x01 ... 0x01 (32 bytes of 0x01)
 *   Step c: K = 0x00 0x00 ... 0x00 (32 bytes of 0x00)
 *   Step d: K = HMAC_K(V || 0x00 || int2octets(x) || bits2octets(h1))
 *   Step e: V = HMAC_K(V)
 *   Step f: K = HMAC_K(V || 0x01 || int2octets(x) || bits2octets(h1))
 *   Step g: V = HMAC_K(V)
 *   Step h: Loop until valid k:
 *           - T = empty
 *           - While len(T) < qlen: T = T || HMAC_K(V); V = HMAC_K(V)
 *           - k = bits2int(T)
 *           - If 1 <= k < q: return k
 *           - Else: K = HMAC_K(V || 0x00); V = HMAC_K(V)
 */

#include "rfc6979_p256.h"
#include <string.h>
#include <stdio.h>

// P-256 curve order n (big-endian bytes)
static const uint8_t P256_N_BE[32] = {
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xbc, 0xe6, 0xfa, 0xad, 0xa7, 0x17, 0x9e, 0x84,
    0xf3, 0xb9, 0xca, 0xc2, 0xfc, 0x63, 0x25, 0x51
};

// ============================================================================
// SHA-256 Implementation (minimal, self-contained)
// ============================================================================

static const uint32_t SHA256_K[64] = {
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

static const uint32_t SHA256_H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)       (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define EP1(x)       (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define SIG0(x)      (ROTR32(x, 7) ^ ROTR32(x, 18) ^ ((x) >> 3))
#define SIG1(x)      (ROTR32(x, 17) ^ ROTR32(x, 19) ^ ((x) >> 10))

static void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h, t1, t2;

    // Prepare message schedule
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[i*4+0] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }
    for (int i = 16; i < 64; ++i) {
        W[i] = SIG1(W[i-2]) + W[i-7] + SIG0(W[i-15]) + W[i-16];
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (int i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + SHA256_K[i] + W[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void sha256(uint8_t hash[32], const uint8_t* data, size_t len) {
    uint32_t state[8];
    memcpy(state, SHA256_H0, sizeof(state));

    // Process full blocks
    size_t num_blocks = len / 64;
    for (size_t i = 0; i < num_blocks; ++i) {
        sha256_transform(state, data + i * 64);
    }

    // Padding
    uint8_t pad_block[128] = {0};
    size_t remaining = len % 64;
    memcpy(pad_block, data + num_blocks * 64, remaining);
    pad_block[remaining] = 0x80;

    uint64_t bit_len = (uint64_t)len * 8;
    if (remaining >= 56) {
        // Need two blocks
        sha256_transform(state, pad_block);
        memset(pad_block, 0, 64);
        // Length goes in second block
        for (int i = 0; i < 8; ++i) {
            pad_block[56 + i] = (uint8_t)(bit_len >> (56 - i * 8));
        }
        sha256_transform(state, pad_block);
    } else {
        // Length fits in first padding block
        for (int i = 0; i < 8; ++i) {
            pad_block[56 + i] = (uint8_t)(bit_len >> (56 - i * 8));
        }
        sha256_transform(state, pad_block);
    }

    // Output hash
    for (int i = 0; i < 8; ++i) {
        hash[i*4+0] = (uint8_t)(state[i] >> 24);
        hash[i*4+1] = (uint8_t)(state[i] >> 16);
        hash[i*4+2] = (uint8_t)(state[i] >> 8);
        hash[i*4+3] = (uint8_t)(state[i]);
    }
}

// ============================================================================
// HMAC-SHA-256 Implementation
// ============================================================================

#define SHA256_BLOCK_SIZE 64
#define SHA256_HASH_SIZE  32

static void hmac_sha256(
    uint8_t mac[32],
    const uint8_t* key, size_t key_len,
    const uint8_t* data, size_t data_len
) {
    uint8_t k_ipad[SHA256_BLOCK_SIZE];
    uint8_t k_opad[SHA256_BLOCK_SIZE];
    uint8_t key_block[SHA256_BLOCK_SIZE];

    // Prepare key
    if (key_len > SHA256_BLOCK_SIZE) {
        sha256(key_block, key, key_len);
        memset(key_block + SHA256_HASH_SIZE, 0, SHA256_BLOCK_SIZE - SHA256_HASH_SIZE);
    } else {
        memcpy(key_block, key, key_len);
        memset(key_block + key_len, 0, SHA256_BLOCK_SIZE - key_len);
    }

    // Compute ipad and opad
    for (int i = 0; i < SHA256_BLOCK_SIZE; ++i) {
        k_ipad[i] = key_block[i] ^ 0x36;
        k_opad[i] = key_block[i] ^ 0x5c;
    }

    // Inner hash: H(k_ipad || data)
    uint8_t inner_data[SHA256_BLOCK_SIZE + 256];  // Enough for our use case
    memcpy(inner_data, k_ipad, SHA256_BLOCK_SIZE);
    memcpy(inner_data + SHA256_BLOCK_SIZE, data, data_len);

    uint8_t inner_hash[SHA256_HASH_SIZE];
    sha256(inner_hash, inner_data, SHA256_BLOCK_SIZE + data_len);

    // Outer hash: H(k_opad || inner_hash)
    uint8_t outer_data[SHA256_BLOCK_SIZE + SHA256_HASH_SIZE];
    memcpy(outer_data, k_opad, SHA256_BLOCK_SIZE);
    memcpy(outer_data + SHA256_BLOCK_SIZE, inner_hash, SHA256_HASH_SIZE);

    sha256(mac, outer_data, SHA256_BLOCK_SIZE + SHA256_HASH_SIZE);
}

// ============================================================================
// Big Integer Comparison
// ============================================================================

// Compare two 32-byte big-endian integers
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
static int cmp_be32(const uint8_t a[32], const uint8_t b[32]) {
    for (int i = 0; i < 32; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// Check if 32-byte big-endian integer is zero
static bool is_zero_be32(const uint8_t a[32]) {
    for (int i = 0; i < 32; ++i) {
        if (a[i] != 0) return false;
    }
    return true;
}

// ============================================================================
// RFC 6979 Implementation
// ============================================================================

// Uncomment for debug output
// #define RFC6979_DEBUG 1

#ifdef RFC6979_DEBUG
static void print_hex(const char* label, const uint8_t* data, int len) {
    fprintf(stderr, "%s: ", label);
    for (int i = 0; i < len; i++) fprintf(stderr, "%02x", data[i]);
    fprintf(stderr, "\n");
}
#endif

bool rfc6979_generate_k(
    uint8_t k_out[32],
    const uint8_t private_key[32],
    const uint8_t hash[32]
) {
    // RFC 6979 Section 3.2

#ifdef RFC6979_DEBUG
    print_hex("[RFC6979] private_key", private_key, 32);
    print_hex("[RFC6979] hash", hash, 32);
#endif

    // Step b: V = 0x01 0x01 ... 0x01 (32 bytes)
    uint8_t V[32];
    memset(V, 0x01, 32);

    // Step c: K = 0x00 0x00 ... 0x00 (32 bytes)
    uint8_t K[32];
    memset(K, 0x00, 32);

    // Step d: K = HMAC_K(V || 0x00 || int2octets(x) || bits2octets(h1))
    // For P-256: int2octets(x) = x as 32 bytes, bits2octets(h1) = h1 as 32 bytes
    uint8_t hmac_data[32 + 1 + 32 + 32];  // V || 0x00 || x || h1
    memcpy(hmac_data, V, 32);
    hmac_data[32] = 0x00;
    memcpy(hmac_data + 33, private_key, 32);
    memcpy(hmac_data + 65, hash, 32);
    hmac_sha256(K, K, 32, hmac_data, 97);

    // Step e: V = HMAC_K(V)
    // IMPORTANT: Use temp buffer to avoid aliasing issues
    uint8_t temp[32];
    hmac_sha256(temp, K, 32, V, 32);
    memcpy(V, temp, 32);

    // Step f: K = HMAC_K(V || 0x01 || int2octets(x) || bits2octets(h1))
    memcpy(hmac_data, V, 32);
    hmac_data[32] = 0x01;
    memcpy(hmac_data + 33, private_key, 32);
    memcpy(hmac_data + 65, hash, 32);
    hmac_sha256(temp, K, 32, hmac_data, 97);
    memcpy(K, temp, 32);

    // Step g: V = HMAC_K(V)
    hmac_sha256(temp, K, 32, V, 32);
    memcpy(V, temp, 32);

#ifdef RFC6979_DEBUG
    print_hex("[RFC6979] Step g: V", V, 32);
#endif

    // Step h: Generate k
    for (int retry = 0; retry < 100; ++retry) {
        // T = empty, build T until len(T) >= qlen (32 bytes for P-256)
        // For P-256, one HMAC output (32 bytes) is exactly what we need
        // IMPORTANT: Use temp buffer to avoid aliasing issues
        uint8_t T[32];
        hmac_sha256(T, K, 32, V, 32);
        memcpy(V, T, 32);

        // k = bits2int(T) - for P-256, T is already the right size
        memcpy(k_out, T, 32);

#ifdef RFC6979_DEBUG
        print_hex("[RFC6979] Step h: k candidate", k_out, 32);
#endif

        // Check if k is in range [1, n-1]
        // k must be >= 1 AND k < n
        if (!is_zero_be32(k_out) && cmp_be32(k_out, P256_N_BE) < 0) {
            return true;  // Valid k found
        }

        // Invalid k, retry with:
        // K = HMAC_K(V || 0x00)
        uint8_t retry_data[33];
        memcpy(retry_data, V, 32);
        retry_data[32] = 0x00;
        uint8_t new_K[32];
        hmac_sha256(new_K, K, 32, retry_data, 33);
        memcpy(K, new_K, 32);

        // V = HMAC_K(V)
        uint8_t new_V[32];
        hmac_sha256(new_V, K, 32, V, 32);
        memcpy(V, new_V, 32);
    }

    // Should never reach here with valid inputs
    return false;
}
