// SPDX-License-Identifier: Apache-2.0
// Smoke HKDF-SHA256 Device Implementation
// Card 26.69: HKDF-SHA256 for Persistent Engine
//
// RFC 5869 HKDF using SHA-256:
//   Extract: PRK = HMAC-SHA256(salt, IKM)
//   Expand:  OKM = HMAC-SHA256(PRK, T || info || counter)
//
// Input layout for persistent engine:
//   keyslot[0:32] = salt (32 bytes)
//   input[0:N]    = IKM (input key material, 32-64 bytes)
//
// Output:
//   output[0:32]  = OKM (derived key, 32 bytes)
//
// Note: This MVP uses empty info and derives exactly 32 bytes.

#ifndef SMOKE_HKDF_SHA256_DEVICE_CUH
#define SMOKE_HKDF_SHA256_DEVICE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../hmac_gpu/hmac_sha2_device.cuh"

namespace smoke {
namespace kdf {

// ============================================================================
// HKDF-Extract: PRK = HMAC-SHA256(salt, IKM)
// ============================================================================

__device__
void hkdf_sha256_extract(
    const uint8_t salt[32],
    const uint8_t* ikm,
    uint16_t ikm_len,
    uint8_t prk[32])
{
    // PRK = HMAC-SHA256(salt, IKM)
    // Note: If salt is not provided, RFC 5869 says to use HashLen zeros
    hmac_sha256_device(salt, 32, ikm, ikm_len, prk);
}

// ============================================================================
// HKDF-Expand: OKM = HMAC-SHA256(PRK, T || info || counter)
// ============================================================================

// Expand to exactly 32 bytes (one HMAC-SHA256 output)
// This is the common case for deriving a single 256-bit key
__device__
void hkdf_sha256_expand_32(
    const uint8_t prk[32],
    const uint8_t* info,
    uint16_t info_len,
    uint8_t okm[32])
{
    // T(1) = HMAC-SHA256(PRK, info || 0x01)
    // For L=32, we only need one block

    // Build message: info || 0x01
    uint8_t msg[256];  // Max info_len + 1
    uint16_t msg_len = 0;

    // Copy info
    for (uint16_t i = 0; i < info_len && i < 255; i++) {
        msg[msg_len++] = info[i];
    }

    // Append counter (0x01 for first block)
    msg[msg_len++] = 0x01;

    // OKM = HMAC-SHA256(PRK, msg)
    hmac_sha256_device(prk, 32, msg, msg_len, okm);
}

// ============================================================================
// HKDF-SHA256 Full (Extract + Expand)
// ============================================================================

// Full HKDF: salt + IKM -> 32-byte derived key
// info is empty in this MVP implementation
__device__
void hkdf_sha256_derive(
    const uint8_t salt[32],
    const uint8_t* ikm,
    uint16_t ikm_len,
    uint8_t okm[32])
{
    // Step 1: Extract
    uint8_t prk[32];
    hkdf_sha256_extract(salt, ikm, ikm_len, prk);

    // Step 2: Expand (empty info, 32 bytes output)
    hkdf_sha256_expand_32(prk, nullptr, 0, okm);
}

// ============================================================================
// Persistent Engine Interface
// ============================================================================

// HKDF-SHA256 for persistent engine
// Input: IKM (32-64 bytes)
// Keyslot: salt (32 bytes)
// Output: OKM (32 bytes)
extern "C" __device__
void hkdf_sha256_keyslot(
    const uint8_t salt[32],    // Salt from keyslot
    const uint8_t* input,      // IKM
    uint16_t input_len,        // IKM length (32-64)
    uint8_t* output,           // OKM (32 bytes)
    uint16_t* output_len)
{
    // Derive key using HKDF-SHA256
    hkdf_sha256_derive(salt, input, input_len, output);
    *output_len = 32;
}

} // namespace kdf
} // namespace smoke

#endif // SMOKE_HKDF_SHA256_DEVICE_CUH
