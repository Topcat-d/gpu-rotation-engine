// self_healing_device.cuh
// Card 19: Device-side HKDF-SHA256 key derivation for self-healing keyslots
// Uses root seed to re-derive keys when corruption is detected

#pragma once

#include <stdint.h>
#include "engine_state.cuh"

// Include HMAC-SHA256 device implementation
#include "../hmac_gpu/hmac_sha2_device.cuh"

// Include integrity functions for tag computation
#include "integrity_device.cuh"

namespace smoke {
namespace engine {

// Domain separation constants for HKDF
__device__ __constant__ uint8_t HKDF_SALT[16] = {
    's', 'm', 'o', 'k', 'e', ':', 'r', 'o', 'o', 't', '-', 's', 'e', 'e', 'd', 0x00
};
constexpr size_t HKDF_SALT_LEN = 15;  // Without null terminator

__device__ __constant__ uint8_t HKDF_INFO_PREFIX[14] = {
    's', 'm', 'o', 'k', 'e', ':', 'p', '2', '5', '6', '-', 'k', 'e', 'y'
};
constexpr size_t HKDF_INFO_PREFIX_LEN = 14;

// P-256 curve order n (for clamping derived key to valid range)
// n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
__device__ __constant__ uint8_t P256_ORDER[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xBC, 0xE6, 0xFA, 0xAD, 0xA7, 0x17, 0x9E, 0x84,
    0xF3, 0xB9, 0xCA, 0xC2, 0xFC, 0x63, 0x25, 0x51
};

// Compare two 32-byte big-endian values
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
__device__ __forceinline__
int compare_256bit(const uint8_t a[32], const uint8_t b[32]) {
    for (int i = 0; i < 32; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// Check if value is zero
__device__ __forceinline__
bool is_zero_256bit(const uint8_t v[32]) {
    uint8_t acc = 0;
    for (int i = 0; i < 32; ++i) {
        acc |= v[i];
    }
    return acc == 0;
}

// Simple subtraction for 256-bit values (a - b -> out)
// Assumes a >= b
__device__ __forceinline__
void subtract_256bit(uint8_t out[32], const uint8_t a[32], const uint8_t b[32]) {
    int borrow = 0;
    for (int i = 31; i >= 0; --i) {
        int diff = (int)a[i] - (int)b[i] - borrow;
        if (diff < 0) {
            diff += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out[i] = (uint8_t)diff;
    }
}

// HKDF-Extract: PRK = HMAC-SHA256(salt, IKM)
__device__ __forceinline__
void hkdf_extract_sha256(
    uint8_t prk[32],
    const uint8_t* salt, size_t salt_len,
    const uint8_t* ikm, size_t ikm_len)
{
    hmac_sha256_device(salt, salt_len, ikm, ikm_len, prk);
}

// HKDF-Expand: OKM = HMAC-SHA256(PRK, info || 0x01)
// For 32 bytes output, we only need one round
__device__ __forceinline__
void hkdf_expand_sha256_32(
    uint8_t okm[32],
    const uint8_t prk[32],
    const uint8_t* info, size_t info_len)
{
    // Build message: info || 0x01
    uint8_t msg[128];  // Should be enough for our use case
    size_t msg_len = info_len + 1;

    for (size_t i = 0; i < info_len; ++i) {
        msg[i] = info[i];
    }
    msg[info_len] = 0x01;  // Counter for first block

    hmac_sha256_device(prk, 32, msg, msg_len, okm);
}

// Derive P-256 private key from root seed using HKDF-SHA256
// Returns true if key is valid (1 <= d < n), false otherwise
__device__
bool derive_p256_key_from_root(
    uint8_t out_priv[32],
    const uint8_t root_seed[32],
    uint32_t slot_index,
    uint32_t integrity_version)
{
    // Step 1: HKDF-Extract
    // PRK = HMAC-SHA256(salt="smoke:root-seed", IKM=root_seed)
    uint8_t prk[32];
    hkdf_extract_sha256(prk, HKDF_SALT, HKDF_SALT_LEN, root_seed, 32);

    // Step 2: Build info string
    // info = "smoke:p256-key" || slot_index (4 bytes BE) || integrity_version (4 bytes BE)
    uint8_t info[22];  // 14 + 4 + 4 = 22

    // Copy prefix
    for (int i = 0; i < HKDF_INFO_PREFIX_LEN; ++i) {
        info[i] = HKDF_INFO_PREFIX[i];
    }

    // Append slot_index (big-endian)
    info[14] = (slot_index >> 24) & 0xFF;
    info[15] = (slot_index >> 16) & 0xFF;
    info[16] = (slot_index >> 8) & 0xFF;
    info[17] = slot_index & 0xFF;

    // Append integrity_version (big-endian)
    info[18] = (integrity_version >> 24) & 0xFF;
    info[19] = (integrity_version >> 16) & 0xFF;
    info[20] = (integrity_version >> 8) & 0xFF;
    info[21] = integrity_version & 0xFF;

    // Step 3: HKDF-Expand
    // OKM = HMAC-SHA256(PRK, info || 0x01)
    uint8_t okm[32];
    hkdf_expand_sha256_32(okm, prk, info, 22);

    // Step 4: Reduce mod n to get valid P-256 scalar
    // If OKM >= n, compute OKM - n
    // If result is 0, retry with incremented version (caller should handle)

    if (compare_256bit(okm, P256_ORDER) >= 0) {
        // okm >= n, subtract n
        subtract_256bit(out_priv, okm, P256_ORDER);
    } else {
        // okm < n, use directly
        for (int i = 0; i < 32; ++i) {
            out_priv[i] = okm[i];
        }
    }

    // Check if result is zero (invalid)
    if (is_zero_256bit(out_priv)) {
        return false;
    }

    return true;
}

// Self-heal a keyslot by re-deriving from root seed
// Returns true if self-heal succeeded, false if it failed
__device__
bool self_heal_keyslot(
    Keyslot* slot,
    uint32_t slot_index,
    const uint8_t root_seed[32],
    EngineIntegrityStatus* status)
{
    // Bump integrity version for fresh derivation
    uint32_t new_version = slot->integrity_version + 1;

    // Derive new key
    uint8_t new_priv[32];
    if (!derive_p256_key_from_root(new_priv, root_seed, slot_index, new_version)) {
        // Extremely unlikely - try one more time with next version
        new_version++;
        if (!derive_p256_key_from_root(new_priv, root_seed, slot_index, new_version)) {
            // Still failed - this should never happen in practice
            return false;
        }
    }

    // Update keyslot with derived key
    copy_key(slot->private_d, new_priv);
    copy_key(slot->private_d_mirror, new_priv);
    slot->integrity_version = new_version;

    // Recompute integrity tag
    compute_integrity_tag(
        slot->integrity_tag,
        slot_index,
        new_version,
        new_priv
    );

    // Update fault flags
    slot->fault_flags = FAULT_SELF_HEALED;  // Clear other faults, mark as self-healed

    // Update status
    status->slots[slot_index].integrity_version = new_version;
    status->slots[slot_index].fault_flags = FAULT_SELF_HEALED;
    status->slots[slot_index].self_heal_count++;

    return true;
}

// Attempt to self-heal a corrupted keyslot
// Called when check_and_repair_keyslot returns INTEGRITY_ZEROIZED
// Returns true if slot was successfully healed and is now usable
__device__
bool attempt_self_heal(
    EngineState* state,
    uint32_t slot_index)
{
    // Check preconditions
    if (state->root_seed_initialized == 0) {
        return false;  // No root seed, can't self-heal
    }

    if (state->tuning.self_healing_enabled == 0) {
        return false;  // Self-healing disabled
    }

    if (state->tuning.integrity_mode != IntegrityMode::INTEGRITY_STRICT) {
        return false;  // Only self-heal in STRICT mode
    }

    // Perform self-heal
    return self_heal_keyslot(
        &state->keyslots[slot_index],
        slot_index,
        state->root_seed,
        &state->integrity_status
    );
}

// Enhanced integrity check with self-healing capability
// Returns true if slot is usable (either was OK, repaired, or self-healed)
__device__
bool check_repair_or_heal_keyslot(
    EngineState* state,
    uint32_t slot_index)
{
    Keyslot* slot = &state->keyslots[slot_index];
    IntegrityMode mode = state->tuning.integrity_mode;

    if (mode == IntegrityMode::INTEGRITY_DISABLED) {
        return true;  // No checks, always usable
    }

    // Try standard repair first
    IntegrityCheckResult result = check_and_repair_keyslot(
        slot,
        slot_index,
        mode,
        state->tuning.integrity_max_repair_attempts,
        &state->integrity_status
    );

    // If OK or repaired, slot is usable
    if (result == INTEGRITY_OK ||
        result == INTEGRITY_MIRROR_REPAIRED ||
        result == INTEGRITY_PRIMARY_REPAIRED ||
        result == INTEGRITY_TAG_REPAIRED) {
        return true;
    }

    // If ZEROIZED and we have self-healing enabled, try to heal
    if (result == INTEGRITY_ZEROIZED) {
        if (attempt_self_heal(state, slot_index)) {
            // Successfully healed!
            return true;
        }
    }

    // Slot is not usable
    return false;
}

} // namespace engine
} // namespace smoke
