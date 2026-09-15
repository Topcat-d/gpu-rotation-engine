/**
 * Smoke Engine ABI v1 (Frozen)
 *
 * Minimal C-stable ABI for GPU-accelerated cryptographic operations.
 * Synchronous API - each call blocks until complete.
 *
 * CARD: CARD-L3-ENGINE-ABI-02
 * Card 26.87: Added SMOKE_API for DLL export/import
 * Card 26.89: ABI Spec Freeze - v1 struct layouts are now stable
 *
 * ============================================================================
 * ABI FREEZE NOTICE (Card 26.89)
 * ============================================================================
 * The struct layouts in this header are FROZEN as of v1. Changes to struct
 * sizes, field offsets, or field types are BREAKING CHANGES that require:
 *   1. Incrementing SMOKE_ENGINE_ABI_VERSION
 *   2. Maintaining backward compatibility or explicit migration path
 *
 * Frozen struct sizes (v1):
 *   - SmokeErrorInfo:      264 bytes (code + internal_code + 128*2 message)
 *   - SmokeEngineConfig:   56 bytes
 *   - SmokeEngineStats:    64 bytes
 *   - SmokeP256KeyId:      8 bytes
 *   - SmokeP256PrivateKey: 64 bytes
 *
 * Reserved fields exist for future expansion without breaking ABI.
 * ============================================================================
 */

#ifndef SMOKE_ENGINE_H
#define SMOKE_ENGINE_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * DLL Export/Import Macros (Card 26.87)
 * ============================================================================
 * When building the DLL: define SMOKE_ABI_BUILDING_DLL
 * When using the DLL: define nothing (or SMOKE_ABI_USING_DLL for clarity)
 */
#ifdef _WIN32
    #ifdef SMOKE_ABI_BUILDING_DLL
        #define SMOKE_API __declspec(dllexport)
    #elif defined(SMOKE_ABI_USING_DLL)
        #define SMOKE_API __declspec(dllimport)
    #else
        #define SMOKE_API
    #endif
#else
    #ifdef SMOKE_ABI_BUILDING_DLL
        #define SMOKE_API __attribute__((visibility("default")))
    #else
        #define SMOKE_API
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 0. ABI Version (Card 26.89: Frozen at v1)
 * ============================================================================ */

#define SMOKE_ENGINE_ABI_VERSION  1

/* Version compatibility check - callers should use this before init */
#define SMOKE_ABI_VERSION_COMPATIBLE(caller_version) \
    ((caller_version) == SMOKE_ENGINE_ABI_VERSION)

/* ============================================================================
 * 1. Forward Declarations / Handles
 * ============================================================================ */

typedef struct SmokeEngine SmokeEngine;
typedef SmokeEngine* SmokeEngineHandle;

/* ============================================================================
 * 2. Status / Error Codes
 * ============================================================================ */

typedef enum SmokeStatus {
    SMOKE_STATUS_OK                   = 0,
    SMOKE_STATUS_ERROR_UNKNOWN        = 1,
    SMOKE_STATUS_ERROR_INVALID_ARG    = 2,
    SMOKE_STATUS_ERROR_NOT_INITIALIZED= 3,
    SMOKE_STATUS_ERROR_DEVICE_FAILURE = 4,
    SMOKE_STATUS_ERROR_TIMEOUT        = 5,
    SMOKE_STATUS_ERROR_KEY_MISSING    = 6,
    SMOKE_STATUS_ERROR_UNSUPPORTED    = 7,
    SMOKE_STATUS_ERROR_NOT_IMPLEMENTED= 8,  /* Card 26.78: CPU fallback disabled */
    SMOKE_STATUS_ERROR_VERIFY_FAILED  = 9,  /* Card 27.23: Signature verification failed */
    SMOKE_STATUS_ERROR_FENCE_REQUIRED = 10  /* Card 32C: Async rotation pending, call rotation_fence() first */
} SmokeStatus;

typedef struct SmokeErrorInfo {
    SmokeStatus code;           /* offset 0, 4 bytes */
    int32_t     internal_code;  /* offset 4, 4 bytes */
    char        message[256];   /* offset 8, 256 bytes - increased for detailed errors */
} SmokeErrorInfo;
/* Total size: 264 bytes (frozen v1) */

/* ============================================================================
 * 3. Engine Config / Stats
 * ============================================================================ */

typedef struct SmokeEngineConfig {
    uint32_t abi_version;       /* Must be SMOKE_ENGINE_ABI_VERSION */
    int32_t  device_index;      /* GPU index, -1 = default */
    uint32_t queue_depth;       /* Target queue depth (e.g. 1024) */
    uint32_t reserved_u32[4];
    uint64_t reserved_u64[4];
} SmokeEngineConfig;

typedef struct SmokeEngineStats {
    uint64_t total_sign_p256;
    uint64_t total_hash_sha256;
    uint64_t total_errors;
    uint64_t uptime_ms;
    uint64_t reserved_u64[4];
} SmokeEngineStats;

/* ============================================================================
 * 4. Buffer Types
 * ============================================================================ */

typedef struct SmokeBuffer {
    const uint8_t* data;
    size_t         len;
} SmokeBuffer;

typedef struct SmokeMutableBuffer {
    uint8_t* data;
    size_t   len;
} SmokeMutableBuffer;

/* ============================================================================
 * 5. Key Management (P-256)
 * ============================================================================ */

typedef struct SmokeP256KeyId {
    uint32_t id;
    uint32_t reserved;
} SmokeP256KeyId;

typedef struct SmokeP256PrivateKey {
    uint8_t d[32];          /* Big-endian private scalar */
    uint8_t reserved[32];
} SmokeP256PrivateKey;

SMOKE_API SmokeStatus smoke_engine_load_key_p256(
    SmokeEngineHandle           engine,
    const SmokeP256KeyId*       key_id,
    const SmokeP256PrivateKey*  key
);

/* N5 (2026-05-16): Upload the precomputed generator-point comb table
 * (SOA layout, w=8: 8 limbs * 255 entries per coordinate) to GPU
 * __constant__ memory. REQUIRED before smoke_fast_path_start when any
 * P-256 keyslot is loaded — otherwise warp-coop scalar multiplication
 * silently produces point-at-infinity and every signature emerges as
 * r=0 / CRYPTO_ERROR. table_size must equal 8*255 = 2040 uint32 each.
 * x_table and y_table point to flat host buffers, layout [limb*255+entry].
 * Call after smoke_fast_path_create and before smoke_fast_path_start.
 * Loading while a kernel is running returns INVALID_ARG.
 */
SMOKE_API SmokeStatus smoke_engine_load_p256_comb_table(
    SmokeEngineHandle  engine,
    const uint32_t*    x_table,
    const uint32_t*    y_table,
    size_t             table_size
);

/* SPEED-H2-01 (2026-07-20): load the full-window fixed-base table (zero-
 * doubling k*G) for engines built with P256_FASTPATH_FULLWINDOW. `blob` is
 * the entire p256_G_fullwindow_w8.bin (32-byte typed header + 256 KB AoS
 * Montgomery payload); the loader validates the header type and verifies the
 * whole-blob SHA-256 against a build-time compiled golden constant, refusing
 * fail-closed on any mismatch. `caller_fp_hex` is an optional secondary
 * cross-check (may be NULL). Returns SMOKE_STATUS_ERROR_VERIFY_FAILED on
 * fingerprint mismatch, INVALID_ARG on bad header/size, UNSUPPORTED if the
 * engine was not built with the fullwindow path. REQUIRED before
 * smoke_fast_path_start on such builds when a P-256 keyslot is loaded.
 * Call after smoke_fast_path_create and before smoke_fast_path_start;
 * loading while a persistent kernel runs returns INVALID_ARG. */
SMOKE_API SmokeStatus smoke_engine_load_p256_fullwindow_table(
    SmokeEngineHandle  engine,
    const uint8_t*     blob,
    size_t             blob_size,
    const char*        caller_fp_hex
);

/* ATTEST-NATIVE-01: load an HMAC key (1..64 bytes) into keyslot `slot`.
 * Required before smoke_engine_submit_and_drain with an HMAC opcode — the
 * fast-path HMAC branch reads the key from this slot. Same typed-keyslot path
 * the P-256/AES loaders use; fail-closed on bad args. */
SMOKE_API SmokeStatus smoke_engine_load_key_hmac(
    SmokeEngineHandle  engine,
    uint32_t           slot,
    const uint8_t*     key,
    uint32_t           key_len
);

/* ATTEST-NATIVE-01: native, outputs-returning variable-length hash/HMAC batch.
 * The stable, language-neutral surface over the C++ core
 * (fast_path_submit_and_drain_hashes): the ENTIRE submit+drain loop runs in
 * C++, messages are hashed via the payload slab (arbitrary length), and the
 * 32-byte digests are returned in SUBMIT ORDER. Caller owns all buffers.
 *
 * `offsets` is CSR: length item_count+1, offsets[0]==0, offsets[item_count]==
 * payload_bytes; each item i is payloads[offsets[i] .. offsets[i+1]). `digests`
 * must have capacity >= item_count*32. On any failure (bad CSR, out-of-range
 * request id, or a drain-deadline miss) the offending digests are left ZEROED,
 * accounting_ok is 0, and the call returns SMOKE_STATUS_ERROR_DEVICE_FAILURE.
 * Callers MUST branch on the return status before trusting `digests`. */
typedef struct SmokeSubmitBatch {
    const uint8_t*  payloads;       /* concatenated item bytes */
    const uint32_t* offsets;        /* item_count+1 CSR offsets */
    uint32_t        item_count;
    uint32_t        payload_bytes;  /* == offsets[item_count] (validated) */
    uint16_t        opcode;         /* 44 = HMAC_SHA256_VARLEN */
    uint32_t        key_slot;       /* HMAC key slot */
} SmokeSubmitBatch;

typedef struct SmokeDrainBatch {
    uint8_t*  digests;              /* caller-owned, item_count*32, submit order */
    uint64_t* sequence_numbers;     /* reserved; may be NULL (not populated in v1) */
    uint32_t  capacity;            /* item_count */
    uint32_t  completed_count;     /* OUT: digests written with status OK */
    uint32_t  accounting_ok;       /* OUT: 1 iff all submitted, received, OK */
} SmokeDrainBatch;

SMOKE_API SmokeStatus smoke_engine_submit_and_drain(
    SmokeEngineHandle       engine,
    const SmokeSubmitBatch* submit,
    SmokeDrainBatch*        drain
);

/* ============================================================================
 * Card SLH-002 P2: SLH-DSA-SHA2-128s (FIPS 205) — key load + batch sign
 * ============================================================================
 * Sign-only, SHA2-128s parameter set only (the one set that cleared the
 * D-095 >10x signs-per-dollar gate). Host-mediated dispatch: the persistent
 * kernel refuses opcode 53 inline; these entry points run the fused signer
 * as a batch kernel. The PRF_msg / H_msg prologue executes ON DEVICE from
 * the GPU-resident keyslot — SK.prf never has to sit in host RAM after load.
 *
 * Fail-closed: signing with an unloaded or faulted slot returns a structured
 * error; there is no default key. A build without SLH CUDA support keeps
 * these exports but returns SMOKE_STATUS_ERROR_UNSUPPORTED.
 */

#define SMOKE_SLH_DSA_128S_SIG_BYTES  7856
#define SMOKE_SLH_DSA_MAX_KEYSLOTS    4
#define SMOKE_SLH_DSA_MAX_MSG_BYTES   65535
#define SMOKE_OPCODE_SLH_DSA_SIGN     53

/* FIPS 205 SLH-DSA-SHA2-128s private key: four 16-byte components. */
typedef struct SmokeSlhDsaPrivateKey {
    uint8_t sk_seed[16];
    uint8_t sk_prf[16];
    uint8_t pk_seed[16];
    uint8_t pk_root[16];
} SmokeSlhDsaPrivateKey;

/**
 * Load an SLH-DSA-SHA2-128s private key into dedicated SLH keyslot `slot`
 * (0..3; separate array from P-256/AES/RSA/ML-DSA slots — no D-081 class
 * collisions). Key material goes GPU-resident with a mirror copy + integrity
 * tag; the engine's transient host copy is zeroized before return.
 *
 * @return SMOKE_STATUS_OK on success; INVALID_ARG / NOT_INITIALIZED /
 *         DEVICE_FAILURE / UNSUPPORTED (no SLH CUDA in this build) otherwise.
 */
SMOKE_API SmokeStatus smoke_engine_load_key_slh_dsa(
    SmokeEngineHandle             engine,
    uint32_t                      slot,
    const SmokeSlhDsaPrivateKey*  key,
    SmokeErrorInfo*               out_error
);

/**
 * Sign `num_msgs` messages with the SLH-DSA-SHA2-128s key in `key_slot`
 * (synchronous batch; one fused-kernel launch signs the whole batch).
 *
 * addrnds: per-message 16-byte additional randomness, num_msgs * 16 bytes
 *          contiguous (hedged signing, FIPS 205 Alg 19) — or NULL for
 *          deterministic signing (opt_rand = PK.seed, supplied on device).
 * sigs_out: caller buffer of num_msgs * SMOKE_SLH_DSA_128S_SIG_BYTES bytes;
 *          signature i lands at offset i * SMOKE_SLH_DSA_128S_SIG_BYTES.
 *
 * All-or-nothing (§4b: no partial output): on any error the buffer must not
 * be trusted and the status names the failure. Refuses with FENCE_REQUIRED
 * while an async rotation is pending (same latch as smoke_generic_submit).
 */
SMOKE_API SmokeStatus smoke_engine_submit_slh_dsa_sign(
    SmokeEngineHandle     engine,
    uint8_t               key_slot,
    const uint8_t* const* msgs,        /* num_msgs message pointers */
    const uint32_t*       msg_lens,    /* num_msgs lengths (0..65535 each) */
    const uint8_t*        addrnds,     /* num_msgs*16 bytes, or NULL */
    uint32_t              num_msgs,
    uint8_t*              sigs_out,    /* num_msgs * 7856 bytes */
    SmokeErrorInfo*       out_error
);

/* Card SLH-002 §5d item 5 — DEBUG/TEST HOOK ONLY. Flips one byte of the
 * DEVICE key mirror in SLH keyslot `key_slot` so the fused signer's
 * key==mirror integrity check trips on the next sign (fail-closed
 * fault-injection validation). Same family as the smoke_debug_p256_*
 * telemetry exports; never call from product code. Requires a loaded slot;
 * UNSUPPORTED without SLH CUDA in the build. */
SMOKE_API SmokeStatus smoke_debug_slh_corrupt_key_mirror(
    SmokeEngineHandle engine,
    uint8_t           key_slot
);

/* Card N5c (2026-06-01): query a device's multiprocessor (SM) count.
 * Pass device = -1 for the current/default device. Callers use this to
 * clamp num_ctas/num_shards to the SM count before smoke_fast_path_start —
 * the persistent kernel requires all CTAs co-resident, and
 * smoke_fast_path_start fails closed (D-077) if num_ctas > SM count.
 * Pure device query; no engine handle needed. out_sm_count is written
 * only on SMOKE_STATUS_OK. */
SMOKE_API SmokeStatus smoke_engine_query_sm_count(
    int32_t   device,
    int32_t*  out_sm_count
);

/* RRP-02 (2026-07-10): query a device's CUDA model name (e.g.
 * "NVIDIA L40S"). Pass device = -1 for the current/default device.
 * Feeds the daemon's validated-GPU boot gate: every new GPU class so
 * far has broken P-256 in a new way (D-082 L40S kernel miscompile,
 * A100 response drops), so callers refuse to run production shapes on
 * models they cannot identify. out_name receives a NUL-terminated
 * string and is written only on SMOKE_STATUS_OK. A name that does not
 * fit max_len (including the NUL) returns
 * SMOKE_STATUS_ERROR_INVALID_ARG rather than truncating — a truncated
 * model name could alias a different, unvalidated GPU in the caller's
 * allowlist. cudaDeviceProp.name is 256 bytes; pass max_len >= 256 to
 * never hit this. Pure device query; no engine handle needed. */
SMOKE_API SmokeStatus smoke_engine_query_device_name(
    int32_t   device,
    char*     out_name,
    int32_t   max_len
);

/* ============================================================================
 * 6. Engine Lifecycle
 * ============================================================================ */

SMOKE_API SmokeStatus smoke_engine_init(
    const SmokeEngineConfig* config,
    SmokeEngineHandle*       out_engine,
    SmokeErrorInfo*          out_error
);

SMOKE_API SmokeStatus smoke_engine_shutdown(
    SmokeEngineHandle engine
);

SMOKE_API SmokeStatus smoke_engine_get_stats(
    SmokeEngineHandle  engine,
    SmokeEngineStats*  out_stats
);

/* ============================================================================
 * 6b. Engine Measurement Export (SBE-002, 2026-08-17)
 * ============================================================================
 * Exports a measurement of the ENGINE'S RUNTIME IDENTITY: which ABI build
 * this is, and which key(s) are currently loaded (by public-key hash, never
 * private key material), and the active rotation epoch. This is the "what
 * key does THIS engine instance hold" half of an attested-key-custody
 * binding — it is deliberately NOT a measurement of the .so file on disk
 * (that is the deployment/launcher's job: on TDX this is MRTD/RTMR, not
 * something the engine can self-report) and NOT a claim that this export is
 * itself measured/attested. A caller (e.g. a TDX report_data composer) that
 * treats this value as proof the CALLING PROCESS is a specific measured
 * daemon is making an unsupported claim — see
 * engine/memory-bank/architecture/attested_key_custody.md for the honest
 * scope. This export exists so a verifier CAN pin engine identity once the
 * surrounding deployment supplies independent process/launch measurement.
 *
 * engine_measurement = SHA-256 over a stable, self-describing byte string:
 *   "smoke-engine-measurement-v1\0"
 *   || u32_le(abi_version)
 *   || u32_le(num_key_slots)
 *   || for each loaded slot in slot-index order:
 *        u32_le(slot_index) || u32_le(key_type) || sha256(pub_x || pub_y)
 *        (P-256 slots only in v1; non-P-256 slot types contribute
 *        u32_le(slot_index) || u32_le(key_type) || 32 zero bytes, since
 *        there is no ABI-level public-key export for them yet)
 *   || u32_le(active_epoch)
 * No private key material is ever read into this computation.
 */
typedef struct SmokeEngineMeasurement {
    uint8_t  engine_measurement[32]; /* SHA-256, see layout above */
    uint32_t active_epoch;
    uint8_t  num_key_slots;          /* slots considered (0..2 in v1) */
    uint8_t  reserved_u8[3];
    uint32_t abi_version;            /* echoes SMOKE_ENGINE_ABI_VERSION */
    uint8_t  reserved[16];
} SmokeEngineMeasurement;
/* Total size: 32 + 4 + 1 + 3 + 4 + 16 = 60 bytes */

/* Fail-closed: returns SMOKE_STATUS_ERROR_NOT_INITIALIZED if the persistent
 * engine is not running, SMOKE_STATUS_ERROR_INVALID_ARG on null args, and
 * SMOKE_STATUS_ERROR_DEVICE_FAILURE if the keyslot probe read fails. On any
 * non-OK status *out_measurement is left zeroed — callers MUST branch on
 * status before reading it (CLAUDE.md 4b). */
SMOKE_API SmokeStatus smoke_engine_get_measurement(
    SmokeEngineHandle         engine,
    SmokeEngineMeasurement*   out_measurement
);

/* ============================================================================
 * 7. P-256 Sign
 * ============================================================================ */

typedef struct SmokeP256SignRequest {
    SmokeP256KeyId  key_id;
    SmokeBuffer     msg;
    uint8_t         hash_precomputed;   /* 0 = raw, 1 = already hashed */
    uint8_t         reserved_u8[7];
    uint64_t        reserved_u64[4];
} SmokeP256SignRequest;

typedef struct SmokeP256SignResponse {
    uint8_t r[32];          /* Big-endian */
    uint8_t s[32];          /* Big-endian */
    uint8_t reserved[32];
} SmokeP256SignResponse;

SMOKE_API SmokeStatus smoke_engine_sign_p256(
    SmokeEngineHandle           engine,
    const SmokeP256SignRequest* req,
    SmokeP256SignResponse*      resp,
    SmokeErrorInfo*             out_error
);

/* ============================================================================
 * 8. SHA-256 Hash
 * ============================================================================ */

typedef struct SmokeSHA256HashRequest {
    SmokeBuffer input;
    uint64_t    reserved_u64[4];
} SmokeSHA256HashRequest;

typedef struct SmokeSHA256HashResponse {
    uint8_t digest[32];
    uint8_t reserved[32];
} SmokeSHA256HashResponse;

SMOKE_API SmokeStatus smoke_engine_hash_sha256(
    SmokeEngineHandle               engine,
    const SmokeSHA256HashRequest*   req,
    SmokeSHA256HashResponse*        resp,
    SmokeErrorInfo*                 out_error
);

/* ============================================================================
 * 9. Key Rotation (Card 29)
 * ============================================================================ */

SMOKE_API SmokeStatus smoke_engine_rotate_key(
    SmokeEngineHandle engine,
    const uint8_t*    private_key,
    uint32_t          key_len
);

SMOKE_API SmokeStatus smoke_engine_prepare_next_key(
    SmokeEngineHandle engine,
    const uint8_t*    private_key,
    uint32_t          key_len
);

/* Prepare the inactive rotation slot with a typed key. key_type 1 = P-256,
 * 2 = AES-256 / ChaCha20 (32-byte symmetric key). Commit selects the new
 * active epoch; callers must submit with its explicit key_slot. These
 * additive exports expose the existing typed rotation implementation. */
SMOKE_API SmokeStatus smoke_engine_prepare_next_key_typed(
    SmokeEngineHandle engine,
    uint32_t key_type,
    const uint8_t* key_data,
    uint32_t key_len
);

/* Same preparation without synchronizing; rotation_fence is required before
 * submitting work, including after commit_rotation_async. */
SMOKE_API SmokeStatus smoke_engine_prepare_next_key_typed_async(
    SmokeEngineHandle engine,
    uint32_t key_type,
    const uint8_t* key_data,
    uint32_t key_len
);

SMOKE_API SmokeStatus smoke_engine_commit_rotation(
    SmokeEngineHandle engine
);

SMOKE_API SmokeStatus smoke_engine_rollback_rotation(
    SmokeEngineHandle engine
);

/* ============================================================================
 * 10. Async Key Rotation (Card 32B + 32C)
 *
 * Zero-sync batch rotation: enqueue N rotations, fence once.
 *   prepare_next_key_async → commit_rotation_async → ... → rotation_fence
 *
 * CONTRACT (Card 32C — fail-closed guardrail):
 *   After any *_async call, a "pending fence" latch is set.
 *   While pending:
 *     - smoke_engine_sign_p256() returns SMOKE_STATUS_ERROR_FENCE_REQUIRED
 *     - smoke_generic_submit()   returns SMOKE_STATUS_ERROR_FENCE_REQUIRED
 *     - smoke_engine_submit_slh_dsa_sign() returns SMOKE_STATUS_ERROR_FENCE_REQUIRED
 *     - smoke_mldsa_sign() returns SMOKE_STATUS_ERROR_FENCE_REQUIRED;
 *       smoke_mldsa_sign_batch / _batch_deterministic sign 0 messages and
 *       set out_error->code to it (D-098 — the latch is engine-global and
 *       gates the separate ML-DSA/SLH keyslot arrays too)
 *     - fast_path_submit / fast_path_submit_batch throw (pybind11)
 *   The latch clears ONLY when rotation_fence() succeeds.
 *   If rotation_fence() fails, the latch stays set (fail-closed).
 * ============================================================================ */

SMOKE_API SmokeStatus smoke_engine_prepare_next_key_async(
    SmokeEngineHandle engine,
    const uint8_t*    private_key,
    uint32_t          key_len
);

SMOKE_API SmokeStatus smoke_engine_commit_rotation_async(
    SmokeEngineHandle engine
);

SMOKE_API SmokeStatus smoke_engine_rotation_fence(
    SmokeEngineHandle engine
);

// Card 64: Force recovery from stuck fence state.
// Operator-initiated: destroys/recreates comm_stream, re-syncs epoch, clears fence.
SMOKE_API SmokeStatus smoke_engine_rotation_force_recovery(
    SmokeEngineHandle engine
);

/* ============================================================================
 * 12. Audit Hooks (Card 210)
 *
 * Callback-based audit: register a function that fires on every ABI operation.
 * Key material is NEVER passed to the callback (redacted at C level).
 * ============================================================================ */

typedef void (*SmokeAuditCallback)(
    const char*   op_name,       /* Operation name (e.g. "sign_p256") */
    uint16_t      opcode,        /* Opcode number */
    uint8_t       status,        /* Result status code */
    uint8_t       key_slot,      /* Key slot used */
    uint32_t      input_len,     /* Input length in bytes */
    uint32_t      output_len,    /* Output length in bytes */
    uint64_t      duration_us,   /* Operation duration in microseconds */
    uint64_t      request_id,    /* Request ID (if applicable) */
    void*         user_data      /* Opaque user data from registration */
);

SMOKE_API int smoke_engine_set_audit_callback(
    SmokeAuditCallback cb,
    void*              user_data
);

SMOKE_API int smoke_engine_clear_audit_callback(void);

#ifdef __cplusplus
}
#endif

#endif /* SMOKE_ENGINE_H */
