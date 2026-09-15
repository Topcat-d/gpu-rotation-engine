/**
 * Smoke Rotation Probe API — Diagnostic keyslot inspection
 *
 * Provides bit-level invariant checks on keyslot state for audit,
 * thesis proofs, and integration testing. NOT a hot-path API.
 *
 * Design:
 *   - C++ layer: raw slot snapshot + cheap inline checks (memcmp, byte compare)
 *   - Python layer: expensive checks (on-curve) using existing api.py constants
 *   - check_* fields: 0 = PASS, non-zero = failure code
 *   - overall_healthy computed from correctness checks ONLY (not heuristics)
 */

#ifndef SMOKE_ROTATION_PROBE_H
#define SMOKE_ROTATION_PROBE_H

#include <stdint.h>
#include "smoke_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-slot probe result */
typedef struct SmokeSlotProbeResult {
    uint32_t slot_index;
    uint32_t epoch;
    uint32_t key_type;
    uint16_t key_len;
    uint16_t reserved_pad;

    /* --- Correctness invariants (affect overall_healthy) --- */
    uint32_t check_scalar_range;       /* 1 <= d < n (d!=0, d<n strict) */
    uint32_t check_mirror_match;       /* private_d == private_d_mirror */
    uint32_t check_integrity_tag;      /* SHA-256 tag matches recomputed tag (NOT HMAC) */
    uint32_t check_no_fault_flags;     /* fault_flags == FAULT_NONE */
    uint32_t check_version_monotonic;  /* integrity_version >= 1 when loaded */
    uint32_t check_key_len_valid;      /* matches key_type expectation */

    /* --- Heuristic warnings (NOT in overall_healthy) --- */
    uint32_t check_packing_sanity;     /* top 4 bytes of d not all-zero (WARN only) */

    /* --- Python-side checks (set by wrapper, not C++) --- */
    uint32_t check_on_curve;           /* y^2 == x^3 - 3x + b (mod p) */
    uint32_t check_not_infinity;       /* pubkey not point-at-infinity */

    /* Debug metadata (no raw key bytes exposed) */
    uint8_t  d_top_byte;
    uint8_t  pub_x_top_byte;
    uint8_t  reserved_bytes[2];

    uint32_t fault_flags;
    uint32_t integrity_version;

    /* Overall health: AND of correctness checks only */
    uint32_t healthy;
} SmokeSlotProbeResult;

/* Engine-wide rotation probe result */
typedef struct SmokeRotationProbeResult {
    uint32_t active_epoch;
    uint32_t num_slots;
    uint32_t overall_healthy;          /* AND of all slot.healthy + cross-slot checks */
    uint32_t probe_error;              /* Non-zero if probe itself failed */

    SmokeSlotProbeResult slots[2];

    /* Cross-slot checks */
    uint32_t check_active_slot_loaded; /* Active slot key_type != EMPTY */
    uint32_t check_epoch_in_range;     /* active_epoch < num_slots */
    uint32_t check_epochs_distinct;    /* slot[0].epoch != slot[1].epoch (if both loaded) */

    char error_message[128];
} SmokeRotationProbeResult;

/* Sentinel: "unchecked" — Python-side checks default to this */
#define SMOKE_PROBE_UNCHECKED 0xFFFFFFFFu

SMOKE_API SmokeStatus smoke_engine_rotation_probe(
    SmokeEngineHandle engine, SmokeRotationProbeResult* out
);

#ifdef __cplusplus
}
#endif

#endif /* SMOKE_ROTATION_PROBE_H */
