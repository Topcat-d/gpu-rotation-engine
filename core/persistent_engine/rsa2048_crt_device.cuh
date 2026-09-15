/**
 * Card 26.70: RSA-2048 CRT Device Wrapper for Persistent Engine
 *
 * Provides a simplified interface for RSA-2048 CRT signing
 * from the persistent engine loop.
 *
 * Input: 256-byte PKCS#1 v1.5 padded message (in payload slab)
 * Output: 256-byte signature (in output slab)
 * Key: RSAKeyslot containing CRT components
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "../rsa2048_crt/rsa2048_crt.cuh"
#include "engine_state.cuh"

namespace smoke {
namespace rsa {

/**
 * RSA-2048 CRT sign for persistent engine.
 *
 * Takes a 256-byte padded message and produces a 256-byte signature.
 * Uses CRT optimization for performance: two 1024-bit modexps instead of one 2048-bit.
 *
 * @param keyslot    Pointer to loaded RSA keyslot
 * @param message    Input: 256-byte PKCS#1 padded message (big-endian)
 * @param signature  Output: 256-byte signature (big-endian)
 *
 * Note: This function performs endianness conversion internally.
 * Message and signature are big-endian (network byte order).
 * Internal computation uses little-endian limbs.
 */
__device__
void rsa2048_sign_keyslot(
    const smoke::engine::RSAKeyslot* keyslot,
    const uint8_t* message,
    uint8_t* signature)
{
    int lane_id = threadIdx.x % 32;

    // Convert 256-byte big-endian message to 64 little-endian limbs
    uint32_t msg_limbs[64];
    if (lane_id == 0) {
        for (int i = 0; i < 64; i++) {
            // Big-endian byte order: MSB at index 0
            // Little-endian limb order: LSW at index 0
            int byte_offset = (63 - i) * 4;
            msg_limbs[i] = ((uint32_t)message[byte_offset + 3])       |
                           ((uint32_t)message[byte_offset + 2] << 8)  |
                           ((uint32_t)message[byte_offset + 1] << 16) |
                           ((uint32_t)message[byte_offset + 0] << 24);
        }
    }
    __syncwarp();

    // Broadcast message limbs from lane 0 to all lanes
    for (int i = 0; i < 64; i++) {
        msg_limbs[i] = __shfl_sync(0xFFFFFFFF, msg_limbs[i], 0);
    }

    // Output signature limbs
    uint32_t sig_limbs[64];

    // Call CRT sign function
    rsa2048_crt::rsa_crt_sign(
        sig_limbs,
        msg_limbs,
        keyslot->p,
        keyslot->q,
        keyslot->dp,
        keyslot->dq,
        keyslot->qinv,
        keyslot->R2_p,
        keyslot->R2_q,
        keyslot->p_prime,
        keyslot->q_prime,
        lane_id
    );

    // Convert 64 little-endian limbs back to 256-byte big-endian signature
    if (lane_id == 0) {
        for (int i = 0; i < 64; i++) {
            int byte_offset = (63 - i) * 4;
            signature[byte_offset + 3] = sig_limbs[i] & 0xFF;
            signature[byte_offset + 2] = (sig_limbs[i] >> 8) & 0xFF;
            signature[byte_offset + 1] = (sig_limbs[i] >> 16) & 0xFF;
            signature[byte_offset + 0] = (sig_limbs[i] >> 24) & 0xFF;
        }
    }
    __syncwarp();
}

/**
 * Validate RSA keyslot is loaded.
 *
 * @param keyslot Pointer to RSA keyslot
 * @return true if key is loaded and valid, false otherwise
 */
__device__ __forceinline__
bool rsa_keyslot_valid(const smoke::engine::RSAKeyslot* keyslot)
{
    return keyslot != nullptr && keyslot->key_loaded == 1;
}

} // namespace rsa
} // namespace smoke
