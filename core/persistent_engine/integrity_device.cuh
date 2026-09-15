// integrity_device.cuh
// Card 18: Device-side integrity tag computation and verification
// Uses SHA-256 to compute deterministic integrity tags for keyslots

#pragma once

#include <stdint.h>
#include "engine_state.cuh"

// Include device SHA-256 implementation
#include "../hmac_gpu/hmac_sha2_device.cuh"

namespace smoke {
namespace engine {

// Domain separation prefix for keyslot integrity tags
// Tag = SHA-256("smoke:keyslot-tag" || slot_index || integrity_version || private_d)
__device__ __constant__ uint8_t INTEGRITY_TAG_PREFIX[18] = {
    's', 'm', 'o', 'k', 'e', ':', 'k', 'e', 'y', 's', 'l', 'o', 't', '-', 't', 'a', 'g', 0x00
};
constexpr size_t INTEGRITY_TAG_PREFIX_LEN = 17;  // Without null terminator

// Compute integrity tag for a keyslot
// tag = SHA-256("smoke:keyslot-tag" || slot_index || integrity_version || private_d)
__device__ __forceinline__
void compute_integrity_tag(
    uint8_t tag_out[32],
    uint32_t slot_index,
    uint32_t integrity_version,
    const uint8_t private_d[32])
{
    // Build message: prefix (17) + slot_index (4) + version (4) + private_d (32) = 57 bytes
    uint8_t msg[57];

    // Copy prefix
    for (int i = 0; i < INTEGRITY_TAG_PREFIX_LEN; ++i) {
        msg[i] = INTEGRITY_TAG_PREFIX[i];
    }

    // Append slot_index (big-endian)
    msg[17] = (slot_index >> 24) & 0xFF;
    msg[18] = (slot_index >> 16) & 0xFF;
    msg[19] = (slot_index >> 8) & 0xFF;
    msg[20] = slot_index & 0xFF;

    // Append integrity_version (big-endian)
    msg[21] = (integrity_version >> 24) & 0xFF;
    msg[22] = (integrity_version >> 16) & 0xFF;
    msg[23] = (integrity_version >> 8) & 0xFF;
    msg[24] = integrity_version & 0xFF;

    // Append private_d
    for (int i = 0; i < 32; ++i) {
        msg[25 + i] = private_d[i];
    }

    // Compute SHA-256
    sha256_hash_device(msg, 57, tag_out);
}

// Verify integrity tag matches expected value
__device__ __forceinline__
bool verify_integrity_tag(
    const uint8_t expected_tag[32],
    uint32_t slot_index,
    uint32_t integrity_version,
    const uint8_t private_d[32])
{
    uint8_t computed_tag[32];
    compute_integrity_tag(computed_tag, slot_index, integrity_version, private_d);

    // Constant-time comparison
    uint8_t diff = 0;
    for (int i = 0; i < 32; ++i) {
        diff |= computed_tag[i] ^ expected_tag[i];
    }
    return diff == 0;
}

// Compare two 32-byte arrays (mirror comparison)
__device__ __forceinline__
bool compare_key_copies(const uint8_t a[32], const uint8_t b[32])
{
    uint8_t diff = 0;
    for (int i = 0; i < 32; ++i) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

// Copy 32 bytes from src to dst
__device__ __forceinline__
void copy_key(uint8_t dst[32], const uint8_t src[32])
{
    for (int i = 0; i < 32; ++i) {
        dst[i] = src[i];
    }
}

// Zero out 32 bytes
__device__ __forceinline__
void zeroize_key(uint8_t key[32])
{
    for (int i = 0; i < 32; ++i) {
        key[i] = 0;
    }
}

// Result of integrity check
enum IntegrityCheckResult : uint32_t {
    INTEGRITY_OK = 0,           // All checks passed
    INTEGRITY_MIRROR_REPAIRED,  // Mirror mismatch, repaired from primary
    INTEGRITY_PRIMARY_REPAIRED, // Primary corrupted, repaired from mirror
    INTEGRITY_TAG_REPAIRED,     // Tag recomputed after repair
    INTEGRITY_ZEROIZED,         // Both copies corrupted, slot zeroized
    INTEGRITY_SKIP,             // Integrity checking disabled
};

// Check and optionally repair a keyslot
// Returns the result of the check
// In STRICT mode, if result is INTEGRITY_ZEROIZED, the slot should not be used for signing
__device__
IntegrityCheckResult check_and_repair_keyslot(
    Keyslot* slot,
    uint32_t slot_index,
    IntegrityMode mode,
    uint32_t max_repair_attempts,
    EngineIntegrityStatus* status)
{
    if (mode == IntegrityMode::INTEGRITY_DISABLED) {
        return INTEGRITY_SKIP;
    }

    // Step 1: Check mirror match
    bool mirror_match = compare_key_copies(slot->private_d, slot->private_d_mirror);

    // Step 2: Verify integrity tag
    bool tag_valid = verify_integrity_tag(
        slot->integrity_tag,
        slot_index,
        slot->integrity_version,
        slot->private_d
    );

    // Fast path: everything is fine
    if (mirror_match && tag_valid) {
        return INTEGRITY_OK;
    }

    // Something is wrong - record fault
    if (!mirror_match) {
        slot->fault_flags |= FAULT_MIRROR_MISMATCH;
        status->slots[slot_index].fault_flags |= FAULT_MIRROR_MISMATCH;
    }
    if (!tag_valid) {
        slot->fault_flags |= FAULT_TAG_MISMATCH;
        status->slots[slot_index].fault_flags |= FAULT_TAG_MISMATCH;
    }

    // In MONITOR mode, we just record but don't repair
    if (mode == IntegrityMode::INTEGRITY_MONITOR) {
        if (!tag_valid && mirror_match) {
            // Tag doesn't match but copies are consistent - tag may just be stale
            return INTEGRITY_OK;  // Allow signing to continue
        }
        return mirror_match ? INTEGRITY_OK : INTEGRITY_ZEROIZED;
    }

    // STRICT mode: try to repair
    if (max_repair_attempts == 0) {
        // No repair allowed, zeroize
        zeroize_key(slot->private_d);
        zeroize_key(slot->private_d_mirror);
        zeroize_key(slot->integrity_tag);
        slot->fault_flags |= FAULT_HARD_ZEROIZED;
        status->slots[slot_index].fault_flags |= FAULT_HARD_ZEROIZED;
        return INTEGRITY_ZEROIZED;
    }

    // Try to repair
    if (!mirror_match) {
        // Check which copy matches the tag (if either)
        bool primary_matches_tag = verify_integrity_tag(
            slot->integrity_tag, slot_index, slot->integrity_version, slot->private_d
        );
        bool mirror_matches_tag = verify_integrity_tag(
            slot->integrity_tag, slot_index, slot->integrity_version, slot->private_d_mirror
        );

        if (primary_matches_tag && !mirror_matches_tag) {
            // Mirror is corrupted, repair from primary
            copy_key(slot->private_d_mirror, slot->private_d);
            slot->fault_flags |= FAULT_REPAIRED_FROM_MIRROR;
            slot->fault_flags &= ~FAULT_MIRROR_MISMATCH;
            status->slots[slot_index].fault_flags |= FAULT_REPAIRED_FROM_MIRROR;
            return INTEGRITY_MIRROR_REPAIRED;
        }
        else if (!primary_matches_tag && mirror_matches_tag) {
            // Primary is corrupted, repair from mirror
            copy_key(slot->private_d, slot->private_d_mirror);
            slot->fault_flags |= FAULT_REPAIRED_FROM_MIRROR;
            slot->fault_flags &= ~FAULT_MIRROR_MISMATCH;
            status->slots[slot_index].fault_flags |= FAULT_REPAIRED_FROM_MIRROR;
            return INTEGRITY_PRIMARY_REPAIRED;
        }
        else {
            // Neither matches or both match (both corrupted) - zeroize
            zeroize_key(slot->private_d);
            zeroize_key(slot->private_d_mirror);
            zeroize_key(slot->integrity_tag);
            slot->fault_flags |= FAULT_HARD_ZEROIZED;
            status->slots[slot_index].fault_flags |= FAULT_HARD_ZEROIZED;
            return INTEGRITY_ZEROIZED;
        }
    }
    else if (!tag_valid) {
        // Mirror matches but tag doesn't - recompute tag
        // (This shouldn't happen in normal operation, but handle it)
        compute_integrity_tag(
            slot->integrity_tag,
            slot_index,
            slot->integrity_version,
            slot->private_d
        );
        slot->fault_flags |= FAULT_REPAIRED_FROM_MIRROR;
        slot->fault_flags &= ~FAULT_TAG_MISMATCH;
        status->slots[slot_index].fault_flags |= FAULT_REPAIRED_FROM_MIRROR;
        return INTEGRITY_TAG_REPAIRED;
    }

    return INTEGRITY_OK;
}

// Run background integrity scan on all slots
__device__
void run_background_integrity_scan(EngineState* state)
{
    IntegrityMode mode = state->tuning.integrity_mode;
    if (mode == IntegrityMode::INTEGRITY_DISABLED) {
        return;
    }

    uint32_t max_repair = state->tuning.integrity_max_repair_attempts;

    // Scan all slots
    for (uint32_t i = 0; i < ENGINE_MAX_KEY_SLOTS; ++i) {
        check_and_repair_keyslot(
            &state->keyslots[i],
            i,
            mode,
            max_repair,
            &state->integrity_status
        );
    }

    // Update last scan count
    state->integrity_status.last_scan_op_count = state->total_sign_ops;
}

// Check if a slot is usable for signing (STRICT mode check)
// Returns true if slot is usable, false if it should not be used
__device__ __forceinline__
bool is_slot_usable(const Keyslot* slot, IntegrityMode mode)
{
    if (mode != IntegrityMode::INTEGRITY_STRICT) {
        return true;  // DISABLED and MONITOR always allow signing
    }

    // In STRICT mode, reject if slot is zeroized or has unrepaired tag mismatch
    uint32_t blocking_flags = FAULT_HARD_ZEROIZED | FAULT_TAG_MISMATCH;
    return (slot->fault_flags & blocking_flags) == 0;
}

// Compute and set integrity tag for a slot (called from host after key install)
// This is a kernel helper that can be invoked to set up the tag
__device__
void setup_keyslot_integrity(Keyslot* slot, uint32_t slot_index)
{
    // Copy primary to mirror
    copy_key(slot->private_d_mirror, slot->private_d);

    // Compute tag
    compute_integrity_tag(
        slot->integrity_tag,
        slot_index,
        slot->integrity_version,
        slot->private_d
    );

    // Clear fault flags
    slot->fault_flags = FAULT_NONE;
}

} // namespace engine
} // namespace smoke
