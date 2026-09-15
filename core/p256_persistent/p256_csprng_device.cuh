/*
 * P-256 Device-side CSPRNG for Random Nonce Generation - Card 15
 *
 * Provides a ChaCha20-based CSPRNG for generating random nonces k
 * when NonceMode::RANDOM is selected.
 *
 * Design:
 *   - Uses ChaCha20 quarter-round structure for fast mixing
 *   - State: 256-bit key (from host seed) + 128-bit counter
 *   - Each call to next_block() produces 256 bits of output
 *   - Counter is auto-incremented to prevent nonce reuse
 *
 * Security notes:
 *   - Seed MUST come from a secure source (os.urandom on host)
 *   - Counter ensures no (key, counter) pair is reused
 *   - Output is suitable for ECDSA nonce k after mod n reduction
 */

#ifndef P256_CSPRNG_DEVICE_CUH
#define P256_CSPRNG_DEVICE_CUH

#include <cuda_runtime.h>
#include <stdint.h>

namespace smoke {
namespace p256 {

// ============================================================================
// ChaCha20 Constants and State
// ============================================================================

// ChaCha20 "expand 32-byte k" constants
static __device__ __constant__ uint32_t CHACHA_SIGMA[4] = {
    0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u
};

// CSPRNG state for P-256 nonce generation
struct P256RngState {
    uint32_t key[8];      // 256-bit key from host seed
    uint32_t counter[2];  // 64-bit counter (plenty for our use)
    uint32_t nonce[2];    // 64-bit nonce/stream ID
};

// ============================================================================
// ChaCha20 Quarter Round
// ============================================================================

__device__ __forceinline__ void chacha_quarter_round(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d
) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8) | (d >> 24);
    c += d; b ^= c; b = (b << 7) | (b >> 25);
}

// ============================================================================
// ChaCha20 Block Function
// ============================================================================

// Produces 512 bits (16 x uint32_t) of output from state
__device__ void chacha20_block(
    const uint32_t key[8],
    uint64_t counter,
    const uint32_t nonce[2],
    uint32_t output[16]
) {
    // Initialize state: constants | key | counter | nonce
    uint32_t state[16];

    // Constants
    state[0] = CHACHA_SIGMA[0];
    state[1] = CHACHA_SIGMA[1];
    state[2] = CHACHA_SIGMA[2];
    state[3] = CHACHA_SIGMA[3];

    // Key (256 bits)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        state[4 + i] = key[i];
    }

    // Counter (64 bits) + Nonce (64 bits)
    state[12] = (uint32_t)counter;
    state[13] = (uint32_t)(counter >> 32);
    state[14] = nonce[0];
    state[15] = nonce[1];

    // Copy for final addition
    uint32_t working[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        working[i] = state[i];
    }

    // 20 rounds (10 double-rounds)
    for (int i = 0; i < 10; ++i) {
        // Column rounds
        chacha_quarter_round(working[0], working[4], working[8],  working[12]);
        chacha_quarter_round(working[1], working[5], working[9],  working[13]);
        chacha_quarter_round(working[2], working[6], working[10], working[14]);
        chacha_quarter_round(working[3], working[7], working[11], working[15]);

        // Diagonal rounds
        chacha_quarter_round(working[0], working[5], working[10], working[15]);
        chacha_quarter_round(working[1], working[6], working[11], working[12]);
        chacha_quarter_round(working[2], working[7], working[8],  working[13]);
        chacha_quarter_round(working[3], working[4], working[9],  working[14]);
    }

    // Add original state
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        output[i] = working[i] + state[i];
    }
}

// ============================================================================
// RNG Initialization
// ============================================================================

/*
 * Initialize RNG state from host-provided seed and stream ID.
 *
 * Parameters:
 *   state: Output RNG state
 *   seed[32]: 256-bit seed from host (big-endian bytes)
 *   stream_id: Unique stream identifier (e.g., request ID, thread ID)
 */
__device__ void p256_rng_init(
    P256RngState& state,
    const uint8_t seed[32],
    uint64_t stream_id
) {
    // Convert seed bytes to key words (little-endian)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        state.key[i] = ((uint32_t)seed[i*4 + 0]) |
                       ((uint32_t)seed[i*4 + 1] << 8) |
                       ((uint32_t)seed[i*4 + 2] << 16) |
                       ((uint32_t)seed[i*4 + 3] << 24);
    }

    // Initialize counter to 0
    state.counter[0] = 0;
    state.counter[1] = 0;

    // Use stream_id as nonce
    state.nonce[0] = (uint32_t)stream_id;
    state.nonce[1] = (uint32_t)(stream_id >> 32);
}

// ============================================================================
// RNG Next Block
// ============================================================================

/*
 * Generate next 256-bit random block.
 *
 * Parameters:
 *   state: RNG state (counter is auto-incremented)
 *   out[32]: Output 256 bits (big-endian bytes, suitable for k candidate)
 */
__device__ void p256_rng_next_block(
    P256RngState& state,
    uint8_t out[32]
) {
    // Generate ChaCha20 block
    uint32_t block[16];
    uint64_t ctr = ((uint64_t)state.counter[1] << 32) | state.counter[0];
    chacha20_block(state.key, ctr, state.nonce, block);

    // Increment counter for next call
    state.counter[0]++;
    if (state.counter[0] == 0) {
        state.counter[1]++;
    }

    // Take first 256 bits (8 words) and convert to big-endian bytes
    // This is suitable for direct use as a scalar candidate
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t w = block[i];
        out[i*4 + 0] = (uint8_t)(w >> 24);
        out[i*4 + 1] = (uint8_t)(w >> 16);
        out[i*4 + 2] = (uint8_t)(w >> 8);
        out[i*4 + 3] = (uint8_t)(w);
    }
}

/*
 * Generate next 256-bit random block as limbs.
 *
 * Parameters:
 *   state: RNG state (counter is auto-incremented)
 *   out_limbs[8]: Output 256 bits as little-endian limbs
 */
__device__ void p256_rng_next_limbs(
    P256RngState& state,
    uint32_t out_limbs[8]
) {
    // Generate ChaCha20 block
    uint32_t block[16];
    uint64_t ctr = ((uint64_t)state.counter[1] << 32) | state.counter[0];
    chacha20_block(state.key, ctr, state.nonce, block);

    // Increment counter for next call
    state.counter[0]++;
    if (state.counter[0] == 0) {
        state.counter[1]++;
    }

    // Take first 256 bits (8 words) directly as little-endian limbs
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_limbs[i] = block[i];
    }
}

// ============================================================================
// Scalar Reduction and Validation
// ============================================================================

// P-256 curve order n (for comparison)
static __device__ __constant__ uint32_t CSPRNG_P256_N[8] = {
    0xfc632551u, 0xf3b9cac2u, 0xa7179e84u, 0xbce6faadu,
    0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu
};

/*
 * Reduce a 256-bit value mod n to get a valid scalar.
 * Uses simple conditional subtraction (value is already < 2n typically).
 */
__device__ void p256_reduce_mod_n(uint32_t k[8]) {
    // Check if k >= n
    bool ge = false;
    for (int i = 7; i >= 0; --i) {
        if (k[i] > CSPRNG_P256_N[i]) { ge = true; break; }
        if (k[i] < CSPRNG_P256_N[i]) { ge = false; break; }
    }

    if (ge) {
        // k = k - n
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint64_t diff = (uint64_t)k[i] - (uint64_t)CSPRNG_P256_N[i] - borrow;
            k[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
    }
}

/*
 * Check if scalar is zero or invalid.
 */
__device__ bool p256_scalar_is_zero(const uint32_t k[8]) {
    uint32_t z = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        z |= k[i];
    }
    return (z == 0);
}

/*
 * Generate a random valid nonce k in [1, n-1] using CSPRNG.
 *
 * Parameters:
 *   k_out[8]: Output nonce as little-endian limbs
 *   state: RNG state
 *   max_attempts: Maximum retries if k=0 (default 3)
 *
 * Returns: true if valid k was generated, false if all attempts failed
 */
__device__ bool p256_generate_random_k(
    uint32_t k_out[8],
    P256RngState& state,
    int max_attempts = 3
) {
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        // Generate random 256 bits
        p256_rng_next_limbs(state, k_out);

        // Reduce mod n
        p256_reduce_mod_n(k_out);

        // Check if valid (non-zero)
        if (!p256_scalar_is_zero(k_out)) {
            return true;
        }
        // k=0 is extremely rare (probability 1/n), retry
    }
    return false;  // All attempts failed (astronomically unlikely)
}

} // namespace p256
} // namespace smoke

#endif // P256_CSPRNG_DEVICE_CUH
