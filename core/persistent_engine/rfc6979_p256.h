/*
 * rfc6979_p256.h - RFC 6979 Deterministic Nonce for P-256 ECDSA
 *
 * Card 12: Implements RFC 6979 Section 3.2 using HMAC-SHA-256.
 *
 * This provides deterministic, cryptographically secure nonce generation
 * for P-256 ECDSA signatures. Given the same (private_key, hash) pair,
 * it always produces the same nonce k, ensuring:
 *   - Reproducible signatures (GPU matches CPU golden implementation)
 *   - No reliance on random number generator quality
 *   - Protection against nonce reuse attacks
 *
 * Reference: https://www.rfc-editor.org/rfc/rfc6979
 */

#ifndef RFC6979_P256_H
#define RFC6979_P256_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * rfc6979_generate_k - Generate deterministic nonce k per RFC 6979
 *
 * Parameters:
 *   k_out[32]:       Output nonce (32 bytes, big-endian)
 *   private_key[32]: Private key d (32 bytes, big-endian)
 *   hash[32]:        Message hash (32 bytes, big-endian, already truncated)
 *
 * Returns: true on success, false on failure (should not happen with valid inputs)
 *
 * Notes:
 *   - Uses HMAC-SHA-256 as the underlying hash function
 *   - Output k is guaranteed to be in range [1, n-1] where n is P-256 curve order
 *   - Deterministic: same inputs always produce same k
 */
bool rfc6979_generate_k(
    uint8_t k_out[32],
    const uint8_t private_key[32],
    const uint8_t hash[32]
);

#ifdef __cplusplus
}
#endif

#endif /* RFC6979_P256_H */
