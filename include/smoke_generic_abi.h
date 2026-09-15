/**
 * Smoke Engine Generic ABI v1 (Card 26.79, 26.82)
 *
 * Unified submit/poll interface for all GPU opcodes.
 * This enables a single ABI entry point to reach all 15 GPU operations.
 *
 * Card 26.82: True C ABI - no Python/PyTorch required in caller.
 * Card 26.89: ABI Spec Freeze - v1 struct layouts are now stable.
 *
 * Uses FAST PATH GPU internally, returns backend tags proving execution.
 *
 * ============================================================================
 * ABI FREEZE NOTICE (Card 26.89)
 * ============================================================================
 * The struct layouts in this header are FROZEN as of v1. Changes to struct
 * sizes, field offsets, or field types are BREAKING CHANGES that require:
 *   1. Incrementing SMOKE_GENERIC_ABI_VERSION
 *   2. Maintaining backward compatibility or explicit migration path
 *
 * Frozen struct sizes (v1):
 *   - SmokeGenericRequest:  64 bytes (cache-aligned)
 *   - SmokeGenericResponse: 128 bytes
 *   - SmokeOpInfo:          24 bytes (plus pointer)
 *
 * Frozen struct layouts verified by compile-time assertions in test_headers.c
 * ============================================================================
 */

#ifndef SMOKE_GENERIC_ABI_H
#define SMOKE_GENERIC_ABI_H

#include <stdint.h>
#include "smoke_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SMOKE_GENERIC_ABI_VERSION 1

/* ============================================================================
 * Backend Tag - Proves Execution Path
 * ============================================================================ */

typedef enum SmokeBackend {
    SMOKE_BACKEND_UNKNOWN      = 0,
    SMOKE_BACKEND_GPU_FASTPATH = 1,
    SMOKE_BACKEND_GPU_LEGACY   = 2,
    SMOKE_BACKEND_CPU_FALLBACK = 3,
} SmokeBackend;

/* ============================================================================
 * Generic Request (64 bytes, cache-aligned) - FROZEN v1
 * ============================================================================
 * Layout (all offsets verified by compile-time assertions):
 *   [0:4]   abi_version    uint32_t
 *   [4:6]   opcode         uint16_t
 *   [6:7]   flags          uint8_t
 *   [7:8]   key_slot       uint8_t
 *   [8:16]  client_id      uint64_t  (only bits 0..7 are transported — D-093)
 *   [16:18] input_len      uint16_t
 *   [18:20] reserved_u16   uint16_t
 *   [20:24] payload_offset uint32_t
 *   [24:28] payload_len    uint32_t
 *   [28:32] reserved_u32   uint32_t
 *   [32:64] input          uint8_t[32]
 */

typedef struct SmokeGenericRequest {
    uint32_t abi_version;      /* Must be SMOKE_GENERIC_ABI_VERSION */
    uint16_t opcode;           /* OpCode enum value */
    uint8_t  flags;            /* Operation flags */
    uint8_t  key_slot;         /* Keyslot index. Inline requests
                                  (payload_len == 0): bounds-checked against
                                  the engine keyslot count and forwarded to
                                  the ring (D5b-keyslot; older builds dropped
                                  it and ran under slot 0 — probe for
                                  smoke_fast_path_submit_inline_batch_slot
                                  before relying on non-zero inline slots).
                                  Slab requests: per-primitive bounds
                                  (RSA/ML-DSA: 0-3). */
    uint64_t client_id;        /* Caller-assigned correlation tag — ONLY THE
                                  LOW 8 BITS reach the engine (D-093). The
                                  transport slot (FastPathRequest.client_id,
                                  frozen 64-byte ring layout, Card 26.16) is
                                  uint8_t: smoke_generic_submit truncates with
                                  `& 0xFF` and the kernel echoes that 8-bit
                                  value in SmokeGenericResponse.client_id.
                                  Bits 8..63 are dropped at submit and NEVER
                                  round-trip; treat them as caller-side
                                  scratch. Correlate within a submit batch via
                                  low-8-bit indices (0..255) and route
                                  responses by request_id (B-321) — the
                                  client_id echo itself can mis-tag under load
                                  (D-052..D-054). */
    uint16_t input_len;        /* Inline input length (0-32) */
    uint16_t reserved_u16;
    uint32_t payload_offset;   /* Slab offset for large inputs (0=inline) */
    uint32_t payload_len;      /* Slab length (0=inline) */
    uint32_t reserved_u32;
    uint8_t  input[32];        /* Inline input buffer */
} SmokeGenericRequest;

/* ============================================================================
 * Generic Response (128 bytes) - FROZEN v1
 * ============================================================================
 * Layout (all offsets verified by compile-time assertions):
 *   [0:8]    client_id      uint64_t
 *   [8:16]   seq            uint64_t
 *   [16:17]  status         uint8_t
 *   [17:18]  backend        uint8_t
 *   [18:20]  opcode         uint16_t
 *   [20:22]  output_len     uint16_t
 *   [22:24]  reserved_u16   uint16_t
 *   [24:28]  output_offset  uint32_t
 *   [28:32]  output_bytes   uint32_t
 *   [32:40]  t_submit_us    uint64_t
 *   [40:48]  t_dequeue_us   uint64_t
 *   [48:56]  t_complete_us  uint64_t
 *   [56:64]  request_id     uint64_t  (B-321: was reserved_u64; now carries
 *                                      the kernel's request_id for response
 *                                      routing in cid-corruption scenarios)
 *   [64:128] output         uint8_t[64]
 */

typedef struct SmokeGenericResponse {
    uint64_t client_id;        /* Echoed from request — TWO caveats:
                                  (1) D-093: only the low 8 bits of the
                                  request's client_id are transported and
                                  echoed; this field is always in [0,255],
                                  never the caller's full 64-bit tag.
                                  (2) may be mis-tagged under residual kernel
                                  race conditions; do not use for routing
                                  correctness (B-318..B-320). Use request_id
                                  for routing if exactness is needed. */
    uint64_t seq;              /* Engine sequence number */
    uint8_t  status;           /* OpStatus */
    uint8_t  backend;          /* SmokeBackend - proves GPU vs CPU */
    uint16_t opcode;           /* Echoed from request */
    uint16_t output_len;       /* Inline output length (0-64) */
    uint16_t reserved_u16;
    uint32_t output_offset;    /* Slab offset for large outputs */
    uint32_t output_bytes;     /* Slab length */
    uint64_t t_submit_us;      /* When host submitted */
    uint64_t t_dequeue_us;     /* When kernel dequeued */
    uint64_t t_complete_us;    /* When kernel completed */
    uint64_t request_id;       /* B-321: kernel-assigned monotonic id; commit
                                  signal already validated by host-side poll.
                                  Routing key for scheduler. (Field offset 56
                                  was reserved_u64 — repurposed; layout
                                  unchanged at the byte level.) */
    uint8_t  output[64];       /* Inline output buffer */
} SmokeGenericResponse;

/* ============================================================================
 * Operation Info for Validation
 * ============================================================================ */

typedef struct SmokeOpInfo {
    uint16_t opcode;
    uint16_t min_input_len;
    uint16_t max_input_len;
    uint16_t output_len;
    uint8_t  requires_key;
    uint8_t  reserved[7];
    const char* name;
} SmokeOpInfo;

/* ============================================================================
 * Card 26.82: FAST PATH Lifecycle (C ABI)
 * ============================================================================
 * These functions manage the FAST PATH persistent kernel lifecycle.
 * Call sequence: init -> fast_path_create -> fast_path_start -> submit/poll -> stop -> shutdown
 */

/**
 * Create FAST PATH resources for the engine.
 * Must be called after smoke_engine_init() and before smoke_fast_path_start().
 * This shuts down the legacy kernel and allocates FAST PATH buffers.
 *
 * @param engine  Engine handle from smoke_engine_init()
 * @param out_error  Optional error info
 * @return SMOKE_STATUS_OK on success
 */
SMOKE_API SmokeStatus smoke_fast_path_create(
    SmokeEngineHandle engine,
    SmokeErrorInfo*   out_error
);

/**
 * Start the FAST PATH persistent kernel.
 * Must be called after smoke_fast_path_create().
 *
 * @param engine  Engine handle
 * @param out_error  Optional error info
 * @return SMOKE_STATUS_OK on success
 */
SMOKE_API SmokeStatus smoke_fast_path_start(
    SmokeEngineHandle engine,
    SmokeErrorInfo*   out_error
);

/**
 * Stop the FAST PATH persistent kernel but keep resources allocated.
 * Can be restarted with smoke_fast_path_start().
 *
 * @param engine  Engine handle
 * @return SMOKE_STATUS_OK on success
 */
SMOKE_API SmokeStatus smoke_fast_path_stop(
    SmokeEngineHandle engine
);

/* ============================================================================
 * Card 26.82: Generic Submit/Poll (C ABI)
 * ============================================================================
 * Submit and poll operations without Python/PyTorch dependencies.
 * Uses FAST PATH GPU internally. Never falls back to CPU.
 */

/**
 * Submit a batch of generic requests through FAST PATH.
 *
 * @param engine  Engine handle with FAST PATH started
 * @param reqs    Array of requests to submit
 * @param count   Number of requests in array
 * @param out_error  Optional error info
 * @return SMOKE_STATUS_OK on success, or error code
 *         Note: Partial submission is possible. Check out_error->internal_code for count submitted.
 *
 * D-093 client_id contract: only req->client_id & 0xFF is transported to the
 * kernel and echoed in responses; bits 8..63 are dropped at submit and do not
 * round-trip. A once-per-process [SMOKE][ABI-CONTRACT] stderr notice fires the
 * first time a request with high bits set is submitted. Route responses by
 * request_id (B-321), not client_id.
 */
SMOKE_API SmokeStatus smoke_generic_submit(
    SmokeEngineHandle             engine,
    const SmokeGenericRequest*    reqs,
    uint32_t                      count,
    SmokeErrorInfo*               out_error
);

/**
 * Poll for completed responses from FAST PATH.
 *
 * @param engine    Engine handle with FAST PATH started
 * @param out       Array to receive responses
 * @param max_out   Maximum number of responses to return
 * @return Number of responses written to out array (0 if none ready)
 */
SMOKE_API uint32_t smoke_generic_poll(
    SmokeEngineHandle       engine,
    SmokeGenericResponse*   out,
    uint32_t                max_out
);

/**
 * B-308 (2026-05-11): Blocking poll variant — wait inside the engine
 * layer until at least one response is ready or the deadline passes.
 *
 * Existing `smoke_generic_poll` is a single non-blocking drain;
 * callers that want all N responses for a submitted batch end up in
 * a tight spin-poll loop, paying one user→kernel syscall per check.
 * Profile of the Go IPC path at pack=1500 (B-307 / D-041) showed
 * ~330 poll calls per batch with ~328 of them returning zero, each
 * costing ~7 µs on Windows — i.e. ~2.3 ms of pure syscall overhead
 * per batch.
 *
 * This entry point moves the spin into C. The internal loop calls
 * `smoke::engine::fast_path_poll` repeatedly with `_mm_pause` between
 * checks (no syscalls, no sleeps) until a response is available or
 * the timeout expires. One syscall per "available burst" instead of
 * hundreds per batch.
 *
 * Semantics:
 *   - Returns 0 if timeout expired with no responses.
 *   - Returns N >= 1 with the first available burst of responses
 *     (caller may need to call again for the rest of the batch).
 *   - Backward-compatible: existing `smoke_generic_poll` keeps its
 *     non-blocking behavior unchanged.
 *
 * @param engine     engine handle with FAST PATH started
 * @param out        response buffer (must be at least max_out entries)
 * @param max_out    max responses to drain in this call
 * @param timeout_us maximum microseconds to wait before returning 0
 * @return number of responses written to `out`
 */
SMOKE_API uint32_t smoke_generic_poll_wait(
    SmokeEngineHandle       engine,
    SmokeGenericResponse*   out,
    uint32_t                max_out,
    uint32_t                timeout_us
);

/**
 * Check if FAST PATH is running on the engine.
 *
 * @param engine  Engine handle
 * @return 1 if FAST PATH is running, 0 otherwise
 */
SMOKE_API int smoke_fast_path_is_running(
    SmokeEngineHandle engine
);

/* ============================================================================
 * B-306 (2026-05-10): FAST PATH Configuration Controls (C ABI)
 * ============================================================================
 * Pre-start configuration of CTA count and ring shards. These call the
 * smoke::engine::fast_path_set_num_* namespace functions that were
 * previously only callable from pybind11. Without these wrappers the
 * engine refuses to start at the C ABI default (1 shard) since the
 * legacy single-ring kernel is disabled.
 *
 * Call order: smoke_fast_path_create -> set_num_ctas + set_num_shards
 *             -> smoke_fast_path_start.
 * Constraint: num_shards == num_ctas (per Card 26.2 exclusive shard
 *             ownership for zero contention).
 */

/**
 * Set the number of CTAs (service lanes) for the FAST PATH kernel.
 * Must be called AFTER smoke_fast_path_create() and BEFORE
 * smoke_fast_path_start().
 *
 * @param engine    Engine handle with FAST PATH created
 * @param num_ctas  Number of CTAs (typically 60 on RTX 4070 Ti)
 * @return SMOKE_STATUS_OK on success
 */
SMOKE_API SmokeStatus smoke_fast_path_set_num_ctas(
    SmokeEngineHandle engine,
    uint32_t          num_ctas
);

/**
 * Set the number of ring shards for the FAST PATH kernel.
 * Must be called AFTER smoke_fast_path_create() and BEFORE
 * smoke_fast_path_start(). Should equal num_ctas for zero contention.
 *
 * @param engine     Engine handle with FAST PATH created
 * @param num_shards Number of ring shards
 * @return SMOKE_STATUS_OK on success
 */
SMOKE_API SmokeStatus smoke_fast_path_set_num_shards(
    SmokeEngineHandle engine,
    uint32_t          num_shards
);

/* ============================================================================
 * B-306 (2026-05-10): AES-256 Key Load (C ABI)
 * ============================================================================
 * Counterpart to smoke_engine_load_key_p256 for AES-256 keyslots.
 * Required by non-Python IPC clients (Go cgo, Rust FFI) to bring up
 * AEAD workloads end-to-end without embedding Python.
 */

/**
 * Load a 32-byte AES-256 key into the engine's keyslot.
 *
 * @param engine    Engine handle with FAST PATH started
 * @param slot      Keyslot index (typically 0-3)
 * @param key       32-byte AES-256 key material
 * @param out_error Optional error info
 * @return SMOKE_STATUS_OK on success
 */
SMOKE_API SmokeStatus smoke_engine_load_key_aes256(
    SmokeEngineHandle engine,
    uint32_t          slot,
    const uint8_t     key[32],
    SmokeErrorInfo*   out_error
);

/* ============================================================================
 * B-307 (2026-05-10): Coalesced inline-input batched submit (C ABI)
 * ============================================================================
 * Direct exposure of `smoke::engine::fast_path_submit_batch`. The
 * existing `smoke_generic_submit` accepts a heterogeneous request
 * array and loops over each one, calling fast_path_submit_batch
 * with count=1 per request -- this pays N memory fences + N head
 * updates per shard. This batched-submit path takes a contiguous
 * inputs buffer (count * 32 bytes) plus shared opcode/flags/key_slot
 * and submits all N requests in one call: fences and head updates
 * are coalesced per shard, not per request.
 *
 * Constraint: all N requests share the same opcode, flags, key_slot,
 * and input_len. For AEAD batched seal (the B-307 hot path) this is
 * always the case. For mixed-opcode batches, use smoke_generic_submit.
 *
 * KEYSLOT (D5b-keyslot): this legacy symbol submits every request with
 * ring key_slot = 0. For a non-default keyslot use
 * smoke_fast_path_submit_inline_batch_slot below.
 *
 * IMPORTANT: with the default submit_policy=SHARD_LOCAL and
 * shards_per_batch=1, ALL count requests land on ONE shard. Set
 * shards_per_batch == num_shards (via smoke_fast_path_set_shards_per_batch)
 * for the batch to be distributed evenly across all shards. Without
 * this configuration the kernel only uses 1/N of its capacity per
 * batch.
 *
 * @return number of requests submitted (may be less than count if
 *         the ring fills up partway through).
 */
SMOKE_API uint32_t smoke_fast_path_submit_inline_batch(
    SmokeEngineHandle engine,
    const uint8_t*    inputs,        /* count * 32 bytes, zero-padded */
    uint32_t          count,
    uint8_t           flags,
    uint8_t           client_id,
    uint16_t          opcode,
    uint16_t          input_len      /* actual bytes used in each 32-byte slot */
);

/* ============================================================================
 * D5b-keyslot: Keyed coalesced inline-input batched submit (C ABI)
 * ============================================================================
 * smoke_fast_path_submit_inline_batch with an explicit shared key_slot.
 *
 * Until this change, EVERY inline submit path (this batched symbol AND
 * smoke_generic_submit's payload_len == 0 branch) hardcoded ring
 * key_slot = 0 — SmokeGenericRequest.key_slot was silently dropped, so
 * inline AEAD/keyed-hash ops always ran under slot 0's key regardless
 * of the slot the caller addressed. Multi-tenant per-op keyslot routing
 * (D5b) was therefore broken at the engine even though the kernel reads
 * req.key_slot correctly.
 *
 * Capability probe: resolving this symbol (dlsym/GetProcAddress) is the
 * supported way for a loader to detect that the engine build forwards
 * key_slot on inline submits — including smoke_generic_submit, which was
 * fixed in the same change. On builds without this symbol a caller MUST
 * fail closed for non-zero inline keyslots rather than submit (the ops
 * would run under slot 0: wrong output, CLAUDE.md §4).
 *
 * key_slot is bounds-checked against the engine keyslot count
 * (ENGINE_MAX_KEY_SLOTS): an out-of-range slot submits 0 requests, loudly
 * (the kernel itself has no device-side bounds check). Responses echo the
 * slot the kernel actually used in `slot_used` (Card 66) — assert on it
 * when validating isolation on real hardware.
 *
 * @return number of requests submitted (may be less than count if the
 *         ring fills up partway through; 0 for an out-of-range key_slot).
 */
SMOKE_API uint32_t smoke_fast_path_submit_inline_batch_slot(
    SmokeEngineHandle engine,
    const uint8_t*    inputs,        /* count * 32 bytes, zero-padded */
    uint32_t          count,
    uint8_t           flags,
    uint8_t           client_id,
    uint16_t          opcode,
    uint16_t          input_len,     /* actual bytes used in each 32-byte slot */
    uint8_t           key_slot       /* shared by the batch; < engine keyslot count */
);

/**
 * Set the submit-policy for FAST PATH.
 *   0 = RR_PER_REQ   -- each request round-robins to next shard
 *   1 = SHARD_LOCAL  -- segments of `shards_per_batch` shards per submit
 * Default at create-time is SHARD_LOCAL with shards_per_batch=1.
 * Must be set BEFORE smoke_fast_path_start.
 */
SMOKE_API SmokeStatus smoke_fast_path_set_submit_policy(
    SmokeEngineHandle engine,
    uint32_t          policy
);

/**
 * Set how many shards a SHARD_LOCAL batch touches.
 * For maximum throughput with a single batched submit, set this to
 * num_shards. For latency-sensitive small batches, keep it small.
 * Must be set BEFORE smoke_fast_path_start.
 */
SMOKE_API SmokeStatus smoke_fast_path_set_shards_per_batch(
    SmokeEngineHandle engine,
    uint32_t          k
);

/* ============================================================================
 * Card 26.85: Payload/Output Slab Accessors (C ABI)
 * ============================================================================
 * For operations with large inputs/outputs, callers write to the payload slab
 * and read from the output slab. Slabs are pinned+mapped memory (zero-copy).
 *
 * Usage pattern:
 *   1. Get payload slab pointer via smoke_engine_get_payload_slab()
 *   2. Write input data at chosen offset in slab
 *   3. Set request.payload_offset and request.payload_len
 *   4. Submit request via smoke_generic_submit()
 *   5. Poll response via smoke_generic_poll()
 *   6. If response.output_bytes > 0, read from output slab at response.output_offset
 */

/**
 * Get the payload slab pointer and size.
 * Callers write large input data here before submitting requests.
 *
 * @param engine    Engine handle with FAST PATH created
 * @param out_ptr   Receives host pointer to payload slab
 * @param out_size  Receives slab size in bytes (typically 4MB)
 * @return SMOKE_STATUS_OK on success
 */
SMOKE_API SmokeStatus smoke_engine_get_payload_slab(
    SmokeEngineHandle engine,
    uint8_t**         out_ptr,
    uint32_t*         out_size
);

/**
 * Get the output slab pointer and size.
 * Large outputs (> 64 bytes) are written here by the kernel.
 *
 * @param engine    Engine handle with FAST PATH created
 * @param out_ptr   Receives host pointer to output slab
 * @param out_size  Receives slab size in bytes (typically 64MB)
 * @return SMOKE_STATUS_OK on success
 */
SMOKE_API SmokeStatus smoke_engine_get_output_slab(
    SmokeEngineHandle engine,
    uint8_t**         out_ptr,
    uint32_t*         out_size
);

/* ============================================================================
 * Card 26.94: RSA-2048 CRT Key Loading (C ABI)
 * ============================================================================
 * Load RSA-2048 CRT key material into a keyslot for signing operations.
 *
 * Key blob format (904 bytes, frozen v1):
 *   [0:128]     p        - CRT prime p (128 bytes, big-endian)
 *   [128:256]   q        - CRT prime q (128 bytes, big-endian)
 *   [256:384]   dp       - d mod (p-1) (128 bytes, big-endian)
 *   [384:512]   dq       - d mod (q-1) (128 bytes, big-endian)
 *   [512:640]   qinv     - q^(-1) mod p (128 bytes, big-endian)
 *   [640:768]   R2_p     - R^2 mod p Montgomery precompute (128 bytes, big-endian)
 *   [768:896]   R2_q     - R^2 mod q Montgomery precompute (128 bytes, big-endian)
 *   [896:900]   p_prime  - -p^(-1) mod 2^32 (4 bytes, little-endian)
 *   [900:904]   q_prime  - -q^(-1) mod 2^32 (4 bytes, little-endian)
 *
 * The R2_p, R2_q, p_prime, q_prime values are Montgomery precomputes that must
 * be computed by the caller. See test_rsa2048_sign_persistent_kat.py for
 * compute_mont_constants() reference implementation.
 */

#define SMOKE_RSA2048_CRT_KEY_BLOB_SIZE 904

/**
 * Load RSA-2048 CRT key into a keyslot.
 *
 * @param engine    Engine handle with FAST PATH created
 * @param slot      Keyslot index (0-3, RSA has limited slots due to key size)
 * @param key_blob  904-byte key blob in frozen format (see above)
 * @param key_len   Must be SMOKE_RSA2048_CRT_KEY_BLOB_SIZE (904)
 * @param out_error Optional error info
 * @return SMOKE_STATUS_OK on success,
 *         SMOKE_STATUS_ERROR_INVALID_ARG if slot out of range or wrong key_len,
 *         SMOKE_STATUS_ERROR_NOT_INITIALIZED if FAST PATH not created
 */
SMOKE_API SmokeStatus smoke_load_key_rsa2048_crt(
    SmokeEngineHandle engine,
    uint8_t           slot,
    const uint8_t*    key_blob,
    uint32_t          key_len,
    SmokeErrorInfo*   out_error
);

/* ============================================================================
 * Card 27.12: ML-DSA (Dilithium) Key Loading and Signing (C ABI)
 * ============================================================================
 * ML-DSA is NIST's post-quantum digital signature standard (FIPS 204).
 * Three security levels are supported: ML-DSA-44, ML-DSA-65, ML-DSA-87.
 *
 * MLDSA SIGN REQUEST FORMAT (opcode 50):
 *   req.input[0]     = mode (0=ML-DSA-44, 1=ML-DSA-65, 2=ML-DSA-87)
 *   req.input[1..3]  = reserved (must be 0)
 *   req.input[4..7]  = msg_len (uint32_t, little-endian) - OPTIONAL if payload_len set
 *   req.key_slot     = keyslot index (0-3)
 *   req.payload_offset = offset to message in payload slab
 *   req.payload_len    = message length in bytes
 *
 * RESPONSE:
 *   resp.output_offset = offset to signature in output slab
 *   resp.output_bytes  = signature length (mode-dependent: 2420/3309/4627)
 *   resp.status        = 0 on success, error code on failure
 *
 * Note: ML-DSA signatures are too large for inline output (64 bytes max).
 * Signatures are always written to the output slab.
 */

/* ML-DSA signature sizes (NIST FIPS 204) */
#define SMOKE_MLDSA44_SIG_BYTES   2420
#define SMOKE_MLDSA65_SIG_BYTES   3309
#define SMOKE_MLDSA87_SIG_BYTES   4627

/* ML-DSA private key sizes (expanded form) */
#define SMOKE_MLDSA44_SK_BYTES    2560
#define SMOKE_MLDSA65_SK_BYTES    4032
#define SMOKE_MLDSA87_SK_BYTES    4896

/* ML-DSA public key sizes (NIST packed format) */
#define SMOKE_MLDSA44_PK_BYTES    1312
#define SMOKE_MLDSA65_PK_BYTES    1952
#define SMOKE_MLDSA87_PK_BYTES    2592

/* Card 27.23: ML-DSA public key sizes (simple format: rho || t1) */
#define SMOKE_MLDSA44_PK_SIMPLE_BYTES  (32 + 4*256*4)   /* 4128 bytes */
#define SMOKE_MLDSA65_PK_SIMPLE_BYTES  (32 + 6*256*4)   /* 6176 bytes */
#define SMOKE_MLDSA87_PK_SIMPLE_BYTES  (32 + 8*256*4)   /* 8224 bytes */

/* ML-DSA mode identifiers (in req.input[0]) */
#define SMOKE_MLDSA_MODE_44       0    /* NIST Level 2 */
#define SMOKE_MLDSA_MODE_65       1    /* NIST Level 3 (recommended default) */
#define SMOKE_MLDSA_MODE_87       2    /* NIST Level 5 */

/* Maximum ML-DSA keyslots (limited due to large key size) */
#define SMOKE_MLDSA_MAX_KEYSLOTS  4

/* ML-DSA opcode */
#define SMOKE_OPCODE_MLDSA_SIGN   50

/**
 * Load ML-DSA private key into a keyslot.
 *
 * The key blob format depends on the mode:
 *   - ML-DSA-44: 2560 bytes (SMOKE_MLDSA44_SK_BYTES)
 *   - ML-DSA-65: 4032 bytes (SMOKE_MLDSA65_SK_BYTES)
 *   - ML-DSA-87: 4896 bytes (SMOKE_MLDSA87_SK_BYTES)
 *
 * Key blob is the standard NIST expanded secret key format:
 *   rho || K || tr || s1 || s2 || t0
 * where components are packed as per FIPS 204.
 *
 * @param engine    Engine handle with FAST PATH created
 * @param slot      Keyslot index (0-3)
 * @param mode      ML-DSA mode (0=44, 1=65, 2=87)
 * @param key_blob  Secret key bytes in NIST format
 * @param key_len   Must match SMOKE_MLDSAxx_SK_BYTES for the mode
 * @param out_error Optional error info
 * @return SMOKE_STATUS_OK on success,
 *         SMOKE_STATUS_ERROR_INVALID_ARG if slot/mode/key_len invalid,
 *         SMOKE_STATUS_ERROR_NOT_INITIALIZED if FAST PATH not created
 */
SMOKE_API SmokeStatus smoke_load_key_mldsa_private(
    SmokeEngineHandle engine,
    uint8_t           slot,
    uint8_t           mode,
    const uint8_t*    key_blob,
    uint32_t          key_len,
    SmokeErrorInfo*   out_error
);

/**
 * Card 27.19: Sign a message using ML-DSA (host-mediated dispatch).
 *
 * This function performs ML-DSA signing synchronously. It uses host-mediated
 * dispatch because sign_full_batch_gpu uses CUDA Dynamic Parallelism which
 * cannot be called from inside the persistent kernel.
 *
 * @param engine      Engine handle with FAST PATH started
 * @param mode        ML-DSA mode (0=44, 1=65, 2=87)
 * @param key_slot    Keyslot index (0-3) with loaded key
 * @param msg         Message to sign
 * @param msg_len     Message length in bytes
 * @param sig_out     Output buffer for signature (must be large enough)
 * @param sig_len_out Receives actual signature length
 * @param out_error   Optional error info
 * @return SMOKE_STATUS_OK on success,
 *         SMOKE_STATUS_ERROR_FENCE_REQUIRED while an async rotation is
 *         pending (Card 32C — same engine-global latch as
 *         smoke_generic_submit; also enforced by smoke_mldsa_sign_batch
 *         and smoke_mldsa_sign_batch_deterministic, which sign 0 messages
 *         and set out_error->code)
 *
 * Note: Caller should allocate sig_out to at least:
 *   - ML-DSA-44: 2420 bytes (SMOKE_MLDSA44_SIG_BYTES)
 *   - ML-DSA-65: 3309 bytes (SMOKE_MLDSA65_SIG_BYTES)
 *   - ML-DSA-87: 4627 bytes (SMOKE_MLDSA87_SIG_BYTES)
 */
SMOKE_API SmokeStatus smoke_mldsa_sign(
    SmokeEngineHandle engine,
    uint8_t           mode,
    uint8_t           key_slot,
    const uint8_t*    msg,
    uint32_t          msg_len,
    uint8_t*          sig_out,
    uint32_t*         sig_len_out,
    SmokeErrorInfo*   out_error
);

/**
 * Card 27.23: Verify a signature using ML-DSA.
 *
 * This function performs ML-DSA signature verification. Unlike signing,
 * verification takes the public key directly as a parameter (not from a keyslot)
 * since verification is often done on messages from untrusted sources.
 *
 * @param engine      Engine handle with FAST PATH started
 * @param mode        ML-DSA mode (0=44, 1=65, 2=87)
 * @param pk          Public key bytes (simple format: rho || t1)
 * @param pk_len      Public key length in bytes
 * @param msg         Message that was signed
 * @param msg_len     Message length in bytes
 * @param sig         Signature bytes (FIPS 204 format)
 * @param sig_len     Signature length in bytes
 * @param out_error   Optional error info
 * @return SMOKE_STATUS_OK if signature is valid,
 *         SMOKE_STATUS_ERROR_VERIFY_FAILED if signature is invalid,
 *         SMOKE_STATUS_ERROR_INVALID_ARG for malformed inputs
 *
 * Expected public key sizes:
 *   - ML-DSA-44: rho(32) + t1(4*256*4) = 4128 bytes
 *   - ML-DSA-65: rho(32) + t1(6*256*4) = 6176 bytes
 *   - ML-DSA-87: rho(32) + t1(8*256*4) = 8224 bytes
 *
 * Expected signature sizes:
 *   - ML-DSA-44: 2420 bytes (SMOKE_MLDSA44_SIG_BYTES)
 *   - ML-DSA-65: 3309 bytes (SMOKE_MLDSA65_SIG_BYTES)
 *   - ML-DSA-87: 4627 bytes (SMOKE_MLDSA87_SIG_BYTES)
 */
SMOKE_API SmokeStatus smoke_mldsa_verify(
    SmokeEngineHandle engine,
    uint8_t           mode,
    const uint8_t*    pk,
    uint32_t          pk_len,
    const uint8_t*    msg,
    uint32_t          msg_len,
    const uint8_t*    sig,
    uint32_t          sig_len,
    SmokeErrorInfo*   out_error
);

/* ============================================================================
 * Card 26.86: Operation Schema Query (C ABI)
 * ============================================================================
 * Query operation constraints so callers know how to build valid requests.
 */

/**
 * Get operation info for a given opcode.
 *
 * @param opcode    Operation code to query
 * @param out_info  Receives operation constraints
 * @return SMOKE_STATUS_OK if opcode is valid, SMOKE_STATUS_ERROR_UNSUPPORTED otherwise
 */
SMOKE_API SmokeStatus smoke_get_op_info(
    uint16_t      opcode,
    SmokeOpInfo*  out_info
);

/**
 * Get the number of supported operations.
 *
 * @return Number of operations in the schema table
 */
SMOKE_API uint32_t smoke_get_op_count(void);

/**
 * Get operation info by index (for enumeration).
 *
 * @param index     Index (0 to smoke_get_op_count()-1)
 * @param out_info  Receives operation constraints
 * @return SMOKE_STATUS_OK if index is valid
 */
SMOKE_API SmokeStatus smoke_get_op_info_by_index(
    uint32_t      index,
    SmokeOpInfo*  out_info
);

/* ============================================================================
 * Card 26.91: Zero-Copy Read Helpers
 * ============================================================================
 * Convenience functions for reading output data from responses.
 * These are useful for handling large outputs that spill to the output slab.
 */

/**
 * Get pointer to response output data (zero-copy).
 *
 * For responses with output_bytes > 0, the output is in the slab.
 * For responses with output_bytes == 0, the output is inline.
 *
 * This function returns the correct pointer regardless of where the output is.
 *
 * @param engine    Engine handle with FAST PATH created
 * @param response  Response from smoke_generic_poll()
 * @param out_ptr   Receives pointer to output data (valid until next poll)
 * @param out_len   Receives length of output data
 * @return SMOKE_STATUS_OK on success
 *
 * Note: The returned pointer is valid only until the next smoke_generic_poll() call.
 */
SMOKE_API SmokeStatus smoke_response_get_output(
    SmokeEngineHandle             engine,
    const SmokeGenericResponse*   response,
    const uint8_t**               out_ptr,
    uint32_t*                     out_len
);

/**
 * Copy response output data to caller buffer.
 *
 * This is a convenience function that handles both inline and slab outputs.
 *
 * @param engine    Engine handle with FAST PATH created
 * @param response  Response from smoke_generic_poll()
 * @param dst       Destination buffer (must be at least *dst_len bytes)
 * @param dst_len   On input: max bytes to copy. On output: bytes copied.
 * @return SMOKE_STATUS_OK on success, SMOKE_STATUS_ERROR_INVALID_ARG if buffer too small
 */
SMOKE_API SmokeStatus smoke_response_copy_output(
    SmokeEngineHandle             engine,
    const SmokeGenericResponse*   response,
    uint8_t*                      dst,
    uint32_t*                     dst_len
);

/* ============================================================================
 * B-317 (2026-05-12): Per-shard read-only debug probe (C ABI)
 * ============================================================================
 * Returns a snapshot of every active shard's request-ring and response-buffer
 * indices, as they appear in host-mapped memory at the instant of the call.
 *
 * Read-only / no side effects / no kernel interaction. Used to localize
 * engine-side progress stalls (B-316 found smoke-ipc-go scheduler deadlocks
 * are engine-side; B-317 narrows to which shard / which ring is responsible).
 *
 * Per-shard fields, all read from `FastPathRing` and `FastPathResponseBuffer`
 * in host-mapped memory (no copy):
 *
 *   req_head        : host's request-ring producer index (host writes here
 *                     when calling fast_path_submit_batch). Monotonic.
 *   req_tail        : kernel's request-ring consumer index (kernel advances
 *                     after dequeuing a request). Monotonic.
 *   req_pending     : req_head - req_tail (work the host produced that the
 *                     kernel has not yet dequeued). Should be small under
 *                     normal flow; a value at FAST_PATH_RING_SIZE means the
 *                     ring is full and the kernel hasn't caught up.
 *   resp_write_idx  : kernel's response-buffer producer index (kernel writes
 *                     here after completing a request). Monotonic.
 *   resp_read_idx   : host's response-buffer consumer index (host advances
 *                     after reading a response). Monotonic.
 *   resp_pending    : resp_write_idx - resp_read_idx (responses the kernel
 *                     produced that the host has not yet read).
 *
 * The caller passes a buffer of `max_shards` `SmokeShardDiag` entries; the
 * implementation fills the first min(num_shards, max_shards) entries and
 * returns how many were written. Entries are filled in shard-id order
 * (entry[i].shard_id == i).
 *
 * Returns 0 if engine is null, FAST PATH not created, or num_shards == 0.
 */

typedef struct SmokeShardDiag {
    uint32_t shard_id;
    uint32_t req_head;
    uint32_t req_tail;
    uint32_t req_pending;
    uint32_t resp_write_idx;
    uint32_t resp_read_idx;
    uint32_t resp_pending;
    uint32_t reserved;
} SmokeShardDiag;

SMOKE_API uint32_t smoke_debug_shard_diag(
    SmokeEngineHandle engine,
    SmokeShardDiag*   out_array,
    uint32_t          max_shards
);

#ifdef __cplusplus
}
#endif

#endif /* SMOKE_GENERIC_ABI_H */
