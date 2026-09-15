#pragma once

#include <stdint.h>
#include <stddef.h>  // B-319b: offsetof for the commit-layout static_asserts
#include <cuda_runtime.h>
#include "engine_constants.cuh"

namespace smoke {
namespace engine {

// Request: "Sign this hash with the active keyslot"
// Card 12: Added has_nonce and nonce_k for RFC 6979 deterministic nonces.
// Card 26.18: Renamed hash -> input for multi-op genericity.
struct Request {
    uint32_t request_id;      // host-assigned, unique-ish
    uint8_t  input[32];       // Card 26.18: Generic input (was 'hash')
    uint8_t  has_nonce;       // Card 12: 1 if nonce_k is provided, 0 otherwise
    uint8_t  reserved_req[3]; // padding for alignment
    uint8_t  nonce_k[32];     // Card 12: RFC 6979 deterministic nonce (if has_nonce)
};

// Response: ECDSA signature (r, s).
// Card 05+: kernel produces stub signature based on keyslot.
// Card 07+: real P-256 ECDSA signing.
struct EngineResponse {
    uint32_t request_id;      // echoes request_id
    uint8_t  status;          // EngineResponseStatus
    uint8_t  reserved[3];     // padding / future
    uint8_t  sig_r[32];       // ECDSA signature r component
    uint8_t  sig_s[32];       // ECDSA signature s component
};

// Card 17: Keyslot fault flags (bitmask)
// Used to track integrity issues detected in keyslots
// Card 19: Added FAULT_SELF_HEALED for HKDF-derived key recovery
enum KeyslotFaultFlags : uint32_t {
    FAULT_NONE                 = 0,
    FAULT_MIRROR_MISMATCH      = 1u << 0,  // priv_d != priv_d_mirror
    FAULT_TAG_MISMATCH         = 1u << 1,  // HMAC tag verification failed
    FAULT_REPAIRED_FROM_MIRROR = 1u << 2,  // Successfully repaired from mirror
    FAULT_HARD_ZEROIZED        = 1u << 3,  // Unrecoverable, slot zeroized
    FAULT_SELF_HEALED          = 1u << 4,  // Card 19: Key re-derived from root seed
};

// Card 26.57: Keyslot type enum for multi-algorithm support
// Identifies the key type stored in a keyslot for validation and dispatch
enum class KeyslotType : uint32_t {
    KEYSLOT_EMPTY   = 0,   // No key loaded
    KEYSLOT_P256    = 1,   // P-256 ECDSA private key (32 bytes)
    KEYSLOT_AES256  = 2,   // AES-256 symmetric key (32 bytes)
    KEYSLOT_HMAC    = 3,   // HMAC key (variable length, stored in private_d)
    KEYSLOT_RSA2048 = 4,   // RSA-2048 CRT key (future: requires extended storage)
};

// Card 26.57: Key size constraints per type
constexpr uint16_t KEYSLOT_P256_KEY_LEN   = 32;  // P-256 private scalar
constexpr uint16_t KEYSLOT_AES256_KEY_LEN = 32;  // AES-256 key
constexpr uint16_t KEYSLOT_HMAC_MAX_LEN   = 64;  // HMAC key max (fits in private_d + mirror)

// Card 17: Integrity mode for keyslot protection
enum class IntegrityMode : uint32_t {
    INTEGRITY_DISABLED = 0,  // No tags, no scans (baseline performance)
    INTEGRITY_MONITOR  = 1,  // Check & report, but don't auto-heal
    INTEGRITY_STRICT   = 2,  // Check, auto-heal if possible, else zeroize & fault
};

// P-256 keyslot with integrity protection (Card 17)
// Card 26.57: Extended for multi-algorithm support (key_type field)
// Card 26.64: Added AES expanded round keys cache
// Previously just "Keyslot", now includes mirror copy and integrity tag
struct Keyslot {
    uint32_t epoch;              // Monotonically increasing epoch ID
    uint32_t key_type;           // Card 26.57: KeyslotType enum (was 'reserved')
    uint8_t  private_d[32];      // Key material (P-256 d, AES key, HMAC key, etc.)
    uint8_t  pub_x[32];          // P-256 public key X (unused for symmetric keys)
    uint8_t  pub_y[32];          // P-256 public key Y (unused for symmetric keys)

    // Card 17: Integrity protection fields
    uint8_t  private_d_mirror[32];  // Redundant mirror copy for rowhammer defense
    uint8_t  integrity_tag[32];     // HMAC-SHA256 tag over key material + slot id
    uint32_t integrity_version;     // Monotonically incrementing per key install
    uint32_t fault_flags;           // Bitmask of KeyslotFaultFlags
    uint16_t key_len;               // Card 26.57: Actual key length (for variable-length keys)
    uint16_t key_reserved;          // Card 26.57: Padding for alignment

    // Card 26.64: AES-256 expanded round keys (precomputed on key load)
    // 60 words = 240 bytes for AES-256 (15 round keys × 4 words each)
    // Only valid when key_type == KEYSLOT_AES256
    uint32_t aes_roundkeys[60];
};

// ============================================================================
// Card 26.70: RSA-2048 CRT Keyslot
// ============================================================================
// RSA keys are much larger than P256/AES (~904 bytes vs 32 bytes)
// so we define a separate keyslot structure for RSA operations.
//
// These keyslots are stored in a dedicated array, not in the main Keyslot array.

constexpr uint32_t RSA_1024_LIMBS = 32;   // 1024-bit = 32 x 32-bit limbs
constexpr uint32_t RSA_MAX_KEYSLOTS = 4;  // Small number due to large key size

struct RSAKeyslot {
    uint32_t epoch;              // Monotonically increasing epoch ID
    uint32_t key_loaded;         // 1 if key is loaded, 0 otherwise

    // CRT key components (all 1024-bit = 32 limbs, little-endian)
    uint32_t p[RSA_1024_LIMBS];        // Prime p
    uint32_t q[RSA_1024_LIMBS];        // Prime q
    uint32_t dp[RSA_1024_LIMBS];       // d mod (p-1)
    uint32_t dq[RSA_1024_LIMBS];       // d mod (q-1)
    uint32_t qinv[RSA_1024_LIMBS];     // q^(-1) mod p

    // Montgomery precomputed values
    uint32_t R2_p[RSA_1024_LIMBS];     // R^2 mod p
    uint32_t R2_q[RSA_1024_LIMBS];     // R^2 mod q
    uint32_t p_prime;                   // -p^(-1) mod 2^32
    uint32_t q_prime;                   // -q^(-1) mod 2^32

    // Integrity protection (mirrored from standard keyslot design)
    uint32_t fault_flags;              // KeyslotFaultFlags
    uint32_t reserved;                 // Padding for alignment
};
// sizeof(RSAKeyslot) = 8 + 7*128 + 8 + 8 = 920 bytes

// ============================================================================
// Card 27.13 + 27.18: ML-DSA (Dilithium) Keyslot with Precomputed Data
// ============================================================================
// ML-DSA keys are large (2560-4896 bytes depending on mode) and require
// NTT-precomputed values for efficient signing. These are stored in dedicated
// keyslots separate from the main P256/AES keyslot array.
//
// Key blob format (NIST FIPS 204 simple secret key):
//   rho (32) || rhoprime (32) || s1 (l*n*4) || s2 (k*n*4) || t0 (k*n*4)
// Sizes vary by mode:
//   ML-DSA-44: k=4, l=4 -> 32+32+4096+4096+4096 = 12,352 bytes (or 2560 packed)
//   ML-DSA-65: k=6, l=5 -> 32+32+5120+6144+6144 = 17,472 bytes (or 4032 packed)
//   ML-DSA-87: k=8, l=7 -> 32+32+7168+8192+8192 = 23,616 bytes (or 4896 packed)
//
// Card 27.18: Keyslot-first design - precomputed NTT data stored in device memory
// on key load for fast signing (no per-call ExpandA overhead).

constexpr uint32_t MLDSA_MAX_KEYSLOTS = 4;        // Limited due to large key size
constexpr uint32_t MLDSA_MAX_SK_BYTES = 23680;    // Card 27.31: ML-DSA-87 expanded with tr: 128 + l*n*4 + 2*k*n*4

// Mode identifiers (matches op_schema.cuh)
constexpr uint8_t MLDSA_MODE_44_VAL = 0;          // NIST Level 2: k=4, l=4
constexpr uint8_t MLDSA_MODE_65_VAL = 1;          // NIST Level 3: k=6, l=5
constexpr uint8_t MLDSA_MODE_87_VAL = 2;          // NIST Level 5: k=8, l=7

// Card 27.20: Key blob format identifiers
enum MLDSAKeyFormat : uint8_t {
    MLDSA_SK_FORMAT_PACKED   = 0,  // NIST packed format (2560/4032/4896 bytes)
    MLDSA_SK_FORMAT_EXPANDED = 1,  // Card 27.22: rho || rhoprime || tr || s1 || s2 || t0
};

// Mode parameters for buffer sizing
struct MLDSAModeParams {
    uint8_t k;           // Matrix rows
    uint8_t l;           // Matrix cols
    uint16_t n;          // Polynomial degree (always 256)
    uint32_t sk_len;     // Packed secret key length
    uint32_t sk_simple_len; // Card 27.22: Expanded secret key length (includes 64-byte tr)
    uint32_t sig_len;    // Signature length
};

// Helper function to get mode parameters
inline MLDSAModeParams mldsa_get_params(uint8_t mode) {
    MLDSAModeParams p;
    p.n = 256;  // Common across all modes
    switch (mode) {
        case MLDSA_MODE_44_VAL:
            p.k = 4; p.l = 4;
            p.sk_len = 2560;
            // Card 27.22: rho(32) + rhoprime(32) + tr(64) + s1 + s2 + t0
            p.sk_simple_len = 32 + 32 + 64 + 4*256*4 + 4*256*4 + 4*256*4;  // 12416 bytes
            p.sig_len = 2420;
            break;
        case MLDSA_MODE_65_VAL:
            p.k = 6; p.l = 5;
            p.sk_len = 4032;
            // Card 27.22: rho(32) + rhoprime(32) + tr(64) + s1 + s2 + t0
            p.sk_simple_len = 32 + 32 + 64 + 5*256*4 + 6*256*4 + 6*256*4;  // 17536 bytes
            p.sig_len = 3309;
            break;
        case MLDSA_MODE_87_VAL:
            p.k = 8; p.l = 7;
            p.sk_len = 4896;
            // Card 27.22: rho(32) + rhoprime(32) + tr(64) + s1 + s2 + t0
            p.sk_simple_len = 32 + 32 + 64 + 7*256*4 + 8*256*4 + 8*256*4;  // 23680 bytes
            p.sig_len = 4627;
            break;
        default:
            p.k = 0; p.l = 0; p.sk_len = 0; p.sk_simple_len = 0; p.sig_len = 0;
            break;
    }
    return p;
}

// Card 27.20: Detect key format from length
inline MLDSAKeyFormat mldsa_detect_key_format(uint8_t mode, uint32_t key_len) {
    MLDSAModeParams p = mldsa_get_params(mode);
    if (key_len == p.sk_simple_len) {
        return MLDSA_SK_FORMAT_EXPANDED;
    }
    return MLDSA_SK_FORMAT_PACKED;
}

// Card 27.18: ML-DSA Keyslot with device-resident precomputed data
// The precomputed device pointers enable fast signing without per-call overhead.
struct MLDSAKeyslot {
    uint32_t epoch;              // Monotonically increasing epoch ID
    uint32_t key_loaded;         // 1 if key is loaded, 0 otherwise
    uint8_t  mode;               // MLDSA_MODE_44/65/87
    uint8_t  precomputed;        // Card 27.18: 1 if device pointers are valid
    uint8_t  reserved[2];        // Padding for alignment

    // Raw secret key blob (packed format for archival/recovery)
    uint8_t  sk_blob[MLDSA_MAX_SK_BYTES];
    uint32_t sk_len;             // Actual secret key length

    // Integrity protection
    uint32_t fault_flags;        // KeyslotFaultFlags
    uint32_t reserved2;          // Padding
};
// sizeof(MLDSAKeyslot) = 16 + 4896 + 8 = 4920 bytes

// Card 27.18: Host-side tracking of device-resident precomputed data
// This structure is NOT stored on device - it's used by engine_host.cpp
// to track device memory allocations for each ML-DSA keyslot.
struct MLDSAKeyslotPrecomputed {
    bool     valid;              // True if device pointers are allocated and valid
    uint8_t  mode;               // Mode this was precomputed for
    uint8_t  reserved[2];

    // Card 27.20: Dedicated CUDA stream for ML-DSA operations
    // CRITICAL: Cannot use stream 0 because persistent kernel runs on it forever
    cudaStream_t sign_stream;    // Created on key load, used for all signing ops

    // Device pointers (allocated once on key load, freed on unload/shutdown)
    uint32_t* d_A_hat;           // NTT(A): k*l*n uint32 values
    uint32_t* d_s1;              // s1 vector: l*n uint32 values (NTT domain)
    uint32_t* d_s2;              // s2 vector: k*n uint32 values (NTT domain)
    uint32_t* d_t0;              // t0 vector: k*n uint32 values (NTT domain)
    uint8_t*  d_rhoprime;        // 32-byte seed for y sampling

    // Workspace buffers for signing (pre-allocated for batch size B)
    uint32_t  workspace_batch_size; // B for which workspace is sized
    uint32_t* d_z;               // z output: B * l*n uint32 values
    uint8_t*  d_h;               // hints output: B * k*n bytes
    int8_t*   d_c;               // challenge: B * n int8 values
    uint8_t*  d_ctilde;          // Card 27.22: c_tilde output: B * 32 bytes (FIPS 204)
    uint32_t* d_attempts;        // attempts per lane: B uint32 values
    uint8_t*  d_converged;       // convergence flags: B bytes
    uint8_t*  d_mu;              // message hashes: B * 64 bytes
    uint8_t*  d_sig;             // packed signatures: B * sig_len bytes

    // Card 27.37: Pre-allocated buffer for SHAKE256 input (tr||msg)
    // Avoids cudaStreamSynchronize deadlock with persistent kernel
    uint8_t*  d_mu_in;           // SHAKE256 input buffer: B * (64 + max_msg_len) bytes
    uint32_t  d_mu_in_capacity;  // Total capacity of d_mu_in buffer

    // Card 27.42: Deterministic rhoprime buffer for reproducible signing
    // rhoprime_det[i] = SHAKE256(tr || msg[i] || "SMOKE_DET_MLDSA", 32)
    uint8_t*  d_rhoprime_det;    // Per-lane deterministic rhoprime: B * 32 bytes
    uint32_t  d_rhoprime_det_capacity;  // Capacity (= workspace_batch_size * 32)
};
// This struct is stored in FastPathHandle, indexed by slot

// ============================================================================
// Card SLH-002 P2: SLH-DSA-SHA2-128s Keyslot
// ============================================================================
// SLH-DSA-SHA2-128s private key = (SK.seed, SK.prf, PK.seed, PK.root), four
// 16-byte fields (FIPS 205 §9.1). Stored in a DEDICATED array — never in the
// generic Keyslot array (D-081 keyslot-collision lesson: slot 0 holds exactly
// one key; PQ keys get their own address space like RSA/ML-DSA do).
//
// Key residency: SK.prf lives here so the PRF_msg / H_msg sign prologue can
// run on device — the host never holds SK.prf after load.
//
// Integrity follows the standard Keyslot pattern: redundant mirror copy of
// the 64 key bytes (rowhammer defense) + host-computed integrity tag +
// fault_flags. The fused signer checks key vs mirror before signing and
// fail-closes (FAULT_MIRROR_MISMATCH) on divergence.

constexpr uint32_t SLH_MAX_KEYSLOTS = 4;
constexpr uint32_t SLH_KEY_BYTES    = 64;   // sk_seed||sk_prf||pk_seed||pk_root

struct SLHKeyslot {
    uint32_t epoch;              // Monotonically increasing epoch ID
    uint32_t key_loaded;         // 1 if key is loaded, 0 otherwise

    // Key material. LAYOUT IS LOAD-BEARING: the fused signer kernel receives
    // &sk_seed as a contiguous 128-byte block (key[64] || mirror[64]) — the
    // static_asserts below pin it.
    uint8_t  sk_seed[16];
    uint8_t  sk_prf[16];
    uint8_t  pk_seed[16];
    uint8_t  pk_root[16];
    uint8_t  key_mirror[SLH_KEY_BYTES];  // redundant copy of the 64 key bytes

    uint8_t  integrity_tag[32];  // host-computed tag over key material + slot id
    uint32_t integrity_version;  // increments per key install
    uint32_t fault_flags;        // KeyslotFaultFlags
};
static_assert(offsetof(SLHKeyslot, sk_prf)     == offsetof(SLHKeyslot, sk_seed) + 16, "SLH key block must be contiguous");
static_assert(offsetof(SLHKeyslot, pk_seed)    == offsetof(SLHKeyslot, sk_seed) + 32, "SLH key block must be contiguous");
static_assert(offsetof(SLHKeyslot, pk_root)    == offsetof(SLHKeyslot, sk_seed) + 48, "SLH key block must be contiguous");
static_assert(offsetof(SLHKeyslot, key_mirror) == offsetof(SLHKeyslot, sk_seed) + 64, "SLH mirror must follow the key block");

// Card 17: Per-slot integrity status (for host visibility)
// Card 19: Added self_heal_count for tracking HKDF recoveries
struct KeyslotIntegrityStatus {
    uint32_t integrity_version;
    uint32_t fault_flags;
    uint32_t self_heal_count;  // Card 19: Number of times slot was re-derived from root
    uint32_t reserved_status;  // Padding for alignment
};

// Card 17: Engine-wide integrity status
struct EngineIntegrityStatus {
    KeyslotIntegrityStatus slots[ENGINE_MAX_KEY_SLOTS];
    uint64_t last_scan_op_count;
    uint64_t total_sign_ops;
};

// Card 10: Debug counters for adaptive microbatch telemetry
// Tracks batch size distribution to verify adaptive policy is working
struct EngineMicrobatchDebug {
    uint64_t total_batches;   // Total number of batches processed
    uint64_t total_requests;  // Total number of requests processed
    uint64_t small_batches;   // Batches with size <= 4 (low latency mode)
    uint64_t medium_batches;  // Batches with size 5-32 (balanced mode)
    uint64_t large_batches;   // Batches with size >= 33 (high throughput mode)
};

// Card 14: Nonce generation mode
// Card 15: RANDOM now uses real GPU CSPRNG (ChaCha20-based)
enum class NonceMode : uint32_t {
    DETERMINISTIC = 0,  // RFC 6979 (default, safe, deterministic)
    RANDOM        = 1,  // GPU CSPRNG (Card 15: ChaCha20-based random nonces)
};

// Card 16: Key rotation policy
// Controls how keyslots and epochs behave
enum class RotationPolicy : uint32_t {
    DISABLED  = 0,  // Always use slot 0, no epoch flipping (single-slot mode)
    MANUAL    = 1,  // Host calls rotate; engine flips epochs (DEFAULT)
    AUTOMATIC = 2,  // Future: engine auto-rotates based on counters
};

// Card 11: Runtime-configurable tuning parameters
// Allows switching between latency-first, balanced, and throughput-first modes
// Card 14: Added nonce_mode and low_s_enabled for signature canonicalization
// Card 15: Added rng_seed for CSPRNG-based random nonces
// Card 16: Added rotation_policy for configurable key rotation behavior
// Card 17: Added integrity_mode and scan parameters
struct EngineTuning {
    uint32_t microbatch_min;    // Minimum batch size
    uint32_t microbatch_max;    // Maximum batch size
    uint32_t depth_low_water;   // Queue depth threshold for small batches
    uint32_t depth_high_water;  // Queue depth threshold for max batches

    // Card 14: Nonce and signature normalization settings
    NonceMode nonce_mode;       // DETERMINISTIC (RFC 6979) or RANDOM (CSPRNG)
    uint32_t  low_s_enabled;    // 1 = canonical low-s (s <= n/2), 0 = raw s

    // Card 15: CSPRNG seed for random nonce generation
    uint8_t   rng_seed[32];     // 256-bit seed from host (set once at engine_create)

    // Card 16: Key rotation policy
    RotationPolicy rotation_policy;      // DISABLED, MANUAL (default), or AUTOMATIC
    uint32_t auto_rotate_interval_ops;   // Signatures between auto-rotations (for AUTOMATIC)
    uint32_t auto_rotate_idle_only;      // 0 = rotate anytime, 1 = only when queue empty

    // Card 17: Keyslot integrity protection settings
    IntegrityMode integrity_mode;              // DISABLED, MONITOR, or STRICT
    uint32_t integrity_scan_interval_ops;      // Ops between background HMAC scans
    uint32_t integrity_scan_idle_only;         // 1 = only scan when queue empty
    uint32_t integrity_max_repair_attempts;    // Max repair attempts per slot

    // Card 19: Self-healing via HKDF derivation
    uint32_t self_healing_enabled;             // 1 = re-derive from root seed on corruption
};

// Card 11: Predefined engine profiles
enum EngineProfile : uint32_t {
    ENGINE_PROFILE_LATENCY_FIRST = 0,  // Favor low latency (small batches)
    ENGINE_PROFILE_BALANCED      = 1,  // Default balanced mode
    ENGINE_PROFILE_THROUGHPUT    = 2,  // Favor high throughput (large batches)
};

// Device-resident engine state.
// All fields are intended to be accessed from both host (via copies/mapping)
// and device (persistent kernel).
struct EngineState {
    // --- Keyslot rotation (for later cards) ---
    Keyslot keyslots[ENGINE_MAX_KEY_SLOTS];
    volatile uint32_t active_epoch;  // 0 or 1

    // --- Request / response ring buffers ---
    // Queue invariants:
    // - Host only advances req_head and resp_tail
    // - Device only advances req_tail and resp_head
    // - This avoids write-write races for each index
    volatile uint32_t req_head;      // producer index (host writes here)
    volatile uint32_t req_tail;      // consumer index (device reads here)
    volatile uint32_t resp_head;     // producer index (device writes here)
    volatile uint32_t resp_tail;     // consumer index (host reads here)

    Request        requests[ENGINE_QUEUE_SIZE];
    EngineResponse responses[ENGINE_QUEUE_SIZE];

    // Card 24.5: Completion flags - can be either inline or mapped
    // If mapped_ready_flags != nullptr, use that (host-mapped memory for zero-copy poll)
    // Otherwise fall back to response_ready[] (legacy/fallback mode)
    volatile uint8_t* mapped_ready_flags;  // Device pointer to host-mapped completion flags

    // 0 = not ready, 1 = ready (status stored in responses[idx].status)
    // Card 24.5: Fallback array - used only if mapped_ready_flags is nullptr
    volatile uint8_t response_ready[ENGINE_QUEUE_SIZE];

    // --- Control flags ---
    volatile uint8_t shutdown_flag;

    // Card 15: Global RNG counter for unique stream IDs
    // Atomically incremented per request in RANDOM mode
    volatile uint64_t rng_counter;

    // Card 11: Runtime-configurable tuning
    EngineTuning tuning;

    // Card 17: Integrity protection
    uint8_t integrity_key[32];           // HMAC key for integrity tags (device-only)
    volatile uint64_t total_sign_ops;    // Total successful sign operations
    EngineIntegrityStatus integrity_status;  // Per-slot integrity status

    // Card 19: Root seed for HKDF self-healing
    uint8_t root_seed[32];               // 256-bit root seed (never leaves GPU after upload)
    uint8_t root_seed_initialized;       // 1 = root seed has been set, 0 = not set
    uint8_t reserved_card19[3];          // Padding for alignment

#if ENGINE_MICROBATCH_DEBUG
    // Card 10: Debug counters for adaptive microbatch telemetry
    EngineMicrobatchDebug microbatch_debug;
#endif
};

// ============================================================================
// Card 25.1: FAST PATH Data Structures
// ============================================================================
// These structures enable device-side batch formation without per-batch
// cudaMemcpy. Host writes to mapped memory, kernel reads directly.

// FAST PATH constants
// Card 26.4B: Increased from 1024 to 8192 to support 60 CTAs × 2 batches × 64 reqs
constexpr uint32_t FAST_PATH_RING_SIZE = 8192;      // Power of 2 (Submission Queue)
constexpr uint32_t FAST_PATH_BATCH_MAX = 128;       // C4.6.14: raised from 64 (matches ENGINE_MICROBATCH_MAX)
// Card 26.11: Increased CQ from 256 to 32768 to prevent overflow at high throughput
// At 111k sig/sec, 32K entries = ~290ms before wrap (plenty of time for host to drain)
// Memory: 32K * 112 bytes = 3.6MB pinned memory
constexpr uint32_t FAST_PATH_RESPONSE_BUFFER = 32768; // Completion Queue size

// Card 26.24: Payload slab for extended multi-message (Phase 2)
// Allows N > 2 messages per request by providing larger input buffer
// Memory: 4MB pinned/mapped slab, shared across all requests
// Max messages per request: slab_size / msg_len (e.g., 4MB / 32 = 128K messages)
// Usage: req.payload_offset = offset into slab, req.payload_len = total bytes
constexpr uint32_t FAST_PATH_PAYLOAD_SLAB_SIZE = 4 * 1024 * 1024;  // 4MB

// Card 26.25: Response slab for extended multi-message outputs (Phase 3).
// The original 64 MiB region retains its 2 KiB per-response layout so existing
// signature/decrypt consumers keep the same ABI. SEAL-SLAB-01 appends a
// disjoint 64 MiB region with segments large enough for the maximum accepted
// plaintext plus its AEAD tag. Keeping the regions disjoint prevents a large
// seal from overwriting an unrelated short slab result.
constexpr uint32_t FAST_PATH_OUTPUT_SEGMENT_BYTES = 2048;  // Per-slot segment (64 digests max)
constexpr uint32_t FAST_PATH_STANDARD_OUTPUT_SLAB_SIZE =
    FAST_PATH_RESPONSE_BUFFER * FAST_PATH_OUTPUT_SEGMENT_BYTES;  // 32768 * 2048 = 64MB
constexpr uint32_t FAST_PATH_SEAL_MAX_PLAINTEXT = 65535;
constexpr uint32_t FAST_PATH_SEAL_TAG_BYTES = 16;
constexpr uint32_t FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES = 65552;  // align16(65535 + 16)
constexpr uint32_t FAST_PATH_SEAL_OUTPUT_REGION_SIZE = 64 * 1024 * 1024;
constexpr uint32_t FAST_PATH_SEAL_OUTPUT_SEGMENTS =
    FAST_PATH_SEAL_OUTPUT_REGION_SIZE / FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES;
constexpr uint32_t FAST_PATH_SEAL_OUTPUT_REGION_OFFSET =
    FAST_PATH_STANDARD_OUTPUT_SLAB_SIZE;
// Card SLH-002 P2: PQ signature output region — third disjoint slab region.
// PQ signatures (SLH-DSA-SHA2-128s 7,856 B; ML-DSA 2,420/3,309/4,627 B) do
// NOT fit a 2,048-byte standard segment, and the pre-P2 ML-DSA scheme wrote
// raw `(request_id % 64) * sig_len` offsets into the standard region —
// violating its per-response-slot 2,048-byte framing (the slab-overrun audit
// force-fails such responses as SLAB_OVERRUN; see fast_path_poll). This
// region gives every PQ signature its own 8,192-byte slot:
//   slot size 8,192  >= 7,856 (SLH-128s) and >= 4,627 (ML-DSA-87)
//   slot count 2,048 -> 16 MiB. Sizing: SLH-128s runs ~6.6K signs/s
//   (D-097 L40S); host dispatch flushes <= SLH_BATCH_MAX_SIZE (512) per
//   batch and the synchronous ABI drains before the next flush, so
//   in-flight slots stay far below 2,048 (~310 ms of full-rate output
//   before wrap). Allocation is a monotonic per-handle counter
//   (next_pq_sig_slot % segments); recycling is audited fail-closed at
//   drain like the other regions.
constexpr uint32_t FAST_PATH_PQ_SIG_OUTPUT_SEGMENT_BYTES = 8192;
constexpr uint32_t FAST_PATH_PQ_SIG_OUTPUT_SEGMENTS = 2048;
constexpr uint32_t FAST_PATH_PQ_SIG_OUTPUT_REGION_SIZE =
    FAST_PATH_PQ_SIG_OUTPUT_SEGMENTS * FAST_PATH_PQ_SIG_OUTPUT_SEGMENT_BYTES;  // 16MB
constexpr uint32_t FAST_PATH_PQ_SIG_OUTPUT_REGION_OFFSET =
    FAST_PATH_STANDARD_OUTPUT_SLAB_SIZE + FAST_PATH_SEAL_OUTPUT_REGION_SIZE;
constexpr uint32_t FAST_PATH_OUTPUT_SLAB_SIZE =
    FAST_PATH_STANDARD_OUTPUT_SLAB_SIZE + FAST_PATH_SEAL_OUTPUT_REGION_SIZE
    + FAST_PATH_PQ_SIG_OUTPUT_REGION_SIZE;  // 144MB total
constexpr uint32_t FAST_PATH_OUTPUT_OWNER_SLOTS =
    FAST_PATH_RESPONSE_BUFFER + FAST_PATH_SEAL_OUTPUT_SEGMENTS
    + FAST_PATH_PQ_SIG_OUTPUT_SEGMENTS;
static_assert(FAST_PATH_PQ_SIG_OUTPUT_SEGMENT_BYTES >= 7856,
    "PQ signature slot must hold an SLH-DSA-SHA2-128s signature");
static_assert(FAST_PATH_PQ_SIG_OUTPUT_SEGMENT_BYTES >= 4627,
    "PQ signature slot must hold an ML-DSA-87 signature");

// Card 25.1: Request descriptor for FAST PATH (64 bytes, cache-aligned)
// Card 25.2C: Added t_submit_us for timestamp truth
// Card 26.16: Added client_id for multi-client CQ partitioning
// Card 26.18: Added opcode + input_len for multi-op support
// Card 26.24: Added payload_offset/payload_len for extended payloads
struct alignas(64) FastPathRequest {
    uint64_t request_id;     // For correlation with response
    uint64_t t_submit_us;    // Card 25.2C: Host timestamp when submitted (microseconds)
    uint8_t  input[32];      // Card 26.18: Generic input (hash for P256_SIGN, data for hashes)
    uint8_t  flags;          // FAST_PATH_FLAG_* options
    uint8_t  key_slot;       // Which keyslot to use
    uint8_t  client_id;      // Card 26.16: Client identifier for CQ partitioning (0-255)
    uint16_t opcode;         // Card 26.18: Operation code (0 = P256_SIGN for backward compat)
    uint16_t input_len;      // Card 26.18: Actual input length (for variable-length ops)
    // Card 26.24: Extended payload support (Phase 2 multi-msg)
    // If payload_len > 0, read from payload_slab[payload_offset..] instead of input[32]
    uint32_t payload_offset; // Card 26.24: Offset into payload slab (0 if not used)
    uint32_t payload_len;    // Card 26.24: Length of payload (0 = use inline input[32])
    // Note: Struct is exactly 64 bytes with natural alignment padding:
    // 8 + 8 + 32 + 1 + 1 + 1 + (1 pad) + 2 + 2 + 4 + 4 = 64
};

// FAST PATH request flags
constexpr uint8_t FAST_PATH_FLAG_LOW_S = 0x01;    // Canonical low-s signature
constexpr uint8_t FAST_PATH_FLAG_MULTIMSG = 0x02; // Card 26.22: Multi-message mode (N digests per request)

// Card 25.2B: Request outcome taxonomy
// Every request must end in EXACTLY ONE of these categories
enum class FastPathOutcome : uint8_t {
    OK = 0,                      // Completed successfully
    GPU_ERROR = 1,               // Kernel returned error status
    RING_FULL = 2,               // Submit failed - ring was full (host-side)
    TIMEOUT_WAITING_RESPONSE = 3, // Poll timed out (host-side)
    KERNEL_EXITED = 4,           // Kernel exited before completion (host-side)
};

// Card 26.1: Signer model enum (shape truth)
// Proves which signing algorithm the kernel is actually using
enum class SignerModel : uint32_t {
    THREAD_ONLY = 0,   // Each thread signs independently (double-and-add, ~3.7k sig/sec)
    WARP_COOP = 1,     // 32 threads cooperate on 1 signature (comb table, ~80k+ sig/sec)
};

// Card 25.1: Response descriptor for FAST PATH (was 120, B-319: 192 bytes)
// Card 25.2C: Added timestamps for latency breakdown
// Card 26.10: Added seq for overflow detection
// Card 26.16: Added client_id for multi-client CQ partitioning
// Card 26.18: Added opcode + generic output for multi-op support
// Card 26.25: Added output_offset/output_bytes for response slab (N > 2 digests)
//
// B-319 (2026-05-13): request_id (commit signal) MUST live on its own
// 64-byte cache line, isolated from all payload fields. The commit
// protocol (engine_loop.cu commit_response) writes payload fields,
// then __threadfence_system(), then request_id LAST. With the previous
// 128-byte layout that placed request_id (offset 0) and client_id
// (offset 41) on the same 64-byte cache line, PCIe write coalescing
// on response-slot wraparound could publish the new request_id to
// host visibility BEFORE the new client_id was visible — host's
// commit-protocol read saw req_id != 0 (commit signal) but then
// memcpy'd a stale client_id from the previous occupancy. See D-052
// for the differential bench-cgo-pipeline measurement that confirmed
// this with rotating clientIDs.
//
// Layout:
//   cache line 0 (offsets   0..63):   request_id + padding   ← commit signal ALONE
//   cache line 1 (offsets  64..127):  all metadata fields (seq, timestamps,
//                                     status, client_id, opcode, etc.) + padding
//   cache line 2 (offsets 128..191):  output[64] / r,s union
//
// The kernel-side __threadfence_system() before the request_id write
// now separates two DIFFERENT cache-line transactions on PCIe, which
// the PCIe controller respects at line granularity. Host sees
// payload-line committed before commit-signal-line.
//
// Size grows 128 -> 192 bytes (50%). FAST_PATH_RESPONSE_BUFFER=32768
// per shard * 60 shards * 64 extra bytes = ~120 MB extra host-mapped
// memory. Acceptable for a correctness fix that resolves the
// scheduler deadlock chain B-310..B-318.
//
// B-319b (2026-07-11): the "payload line lands before commit line"
// assumption above proved FALSE under load on datacenter parts. The
// host can observe the committed request_id (line 0) while the
// output_len/status write (line 1) is not yet visible — a torn read.
// Measured: A100 SXM4 dropped 1.24-5.76% of P-256 responses under
// sustained load; L40S ~1/1500 at single-op poll cadence (status=0,
// output_len=0, valid r/s — the write reached the GPU fence in order
// but the fabric delivered line 0 first). Host-side acquire fence
// had no effect (rules out compiler reordering); bounded re-read was
// inconclusive. See D-091.
//
// Fix: the fields the host must trust BEFORE touching the payload
// (output_len, status) are replicated into `commit_info` on cache
// line 0, and the commit is now a SINGLE 16-byte store publishing
// {request_id, commit_info} as one visibility unit (one PCIe
// transaction) — the host can never see a committed request_id
// without the matching output_len/status. The line-1 copies remain
// for layout compatibility; fast_path_poll overrides them from
// commit_info after the commit check.
struct alignas(64) FastPathResponse {
    // ── B-319 cache line 0 (offsets 0..63): commit signal + commit_info ──
    uint64_t request_id;          // offset 0  — kernel writes LAST; host commit-protocol check
    uint64_t commit_info;         // offset 8  — B-319b: output_len/status, published in the
                                  //             SAME 16-byte store as request_id
    uint64_t _pad_cl0[6];         // offsets 16..63 — pad rest of cache line

    // ── B-319 cache line 1 (offsets 64..127): payload metadata ──
    uint64_t seq;                 // offset 64 — Card 26.10: Monotonic sequence
    uint64_t t_submit_us;         // offset 72 — Card 25.2C
    uint64_t t_dequeue_us;        // offset 80
    uint64_t t_complete_us;       // offset 88
    uint8_t  status;              // offset 96 — 0 = OK
    uint8_t  client_id;           // offset 97 — Card 26.16: echoed from request
    uint16_t opcode;              // offset 98 — Card 26.18
    uint16_t output_len;          // offset 100
    uint8_t  epoch_used = 0xFF;   // offset 102 — Card 66
    uint8_t  slot_used  = 0xFF;   // offset 103 — Card 66
    uint32_t output_offset;       // offset 104 — Card 26.25
    uint32_t output_bytes;        // offset 108
    uint32_t _pad_cl1[4];         // offsets 112..127 — pad rest of cache line

    // ── B-319 cache line 2 (offsets 128..191): output payload ──
    union {
        struct {                  // P256_SIGN output
            uint8_t r[32];        // ECDSA signature r
            uint8_t s[32];        // ECDSA signature s
        };
        uint8_t output[64];       // Card 26.18: Generic output buffer (first 64 bytes)
    };
};

// B-319b: commit_info word layout. Every response writer (kernel commit
// sites AND host-side posters like the ML-DSA dispatcher) must publish
// this; fast_path_poll treats it as the authoritative output_len/status.
//   bits  0..15  output_len
//   bits 16..23  status
//   bits 24..63  reserved (0)
__host__ __device__ inline uint64_t fast_path_commit_info(uint8_t status, uint16_t output_len) {
    return static_cast<uint64_t>(output_len) | (static_cast<uint64_t>(status) << 16);
}
__host__ __device__ inline uint16_t fast_path_commit_output_len(uint64_t commit_info) {
    return static_cast<uint16_t>(commit_info & 0xFFFFu);
}
__host__ __device__ inline uint8_t fast_path_commit_status(uint64_t commit_info) {
    return static_cast<uint8_t>((commit_info >> 16) & 0xFFu);
}

// B-319b: the single-store commit protocol depends on this exact layout —
// request_id and commit_info adjacent and 16-byte aligned at offset 0, and
// the three-cache-line isolation (B-319) preserved.
static_assert(offsetof(FastPathResponse, request_id) == 0,   "B-319b: request_id must sit at offset 0");
static_assert(offsetof(FastPathResponse, commit_info) == 8,  "B-319b: commit_info must be adjacent to request_id (one 16B store)");
static_assert(offsetof(FastPathResponse, seq) == 64,         "B-319: metadata must start on cache line 1");
static_assert(offsetof(FastPathResponse, output) == 128,     "B-319: output payload must start on cache line 2");
static_assert(sizeof(FastPathResponse) == 192,               "B-319: FastPathResponse must stay 3 cache lines");

// Card 25.1: FAST PATH ring buffer (host-mapped, device-readable)
// Host writes requests, kernel reads and processes
struct FastPathRing {
    // Indices - host writes head, kernel writes tail
    // Using separate cache lines to avoid false sharing
    alignas(64) volatile uint32_t head;      // Host produces (atomic store)
    alignas(64) volatile uint32_t tail;      // Kernel consumes (atomic store)
    uint32_t capacity;                       // Always FAST_PATH_RING_SIZE
    uint32_t _pad;

    // Inline request descriptors
    FastPathRequest requests[FAST_PATH_RING_SIZE];
};

// Card 25.1: Response buffer (mapped memory, proper ring buffer)
// Uses head/tail indices to avoid race condition between kernel and host.
// Kernel (producer): atomicAdd(&write_idx, 1) to claim slot, then write
// Host (consumer): read up to write_idx, then advance read_idx
struct FastPathResponseBuffer {
    // Separate cache lines to avoid false sharing
    alignas(64) volatile uint32_t write_idx;  // Kernel producer (atomicAdd)
    alignas(64) volatile uint32_t read_idx;   // Host consumer (only host writes)
    uint32_t _pad[2];                         // Alignment

    FastPathResponse responses[FAST_PATH_RESPONSE_BUFFER];
};

// Card 25.1: Telemetry for FAST PATH (proves kernel is persistent)
// Card 26.1: Added shape truth fields to prove warp-coop engagement
struct FastPathTelemetry {
    volatile uint64_t total_requests_seen;       // Total dequeued by kernel
    volatile uint64_t total_batches_processed;   // Number of batch iterations
    volatile uint64_t total_batch_size_sum;      // For avg_batch_size calculation
    volatile uint64_t kernel_idle_cycles;        // Iterations with no work
    volatile uint64_t kernel_busy_cycles;        // Iterations with work
    volatile uint64_t kernel_launches;           // Should be 1 for persistent!
    volatile uint64_t kernel_start_time_us;      // Timestamp when kernel started
    volatile uint64_t debug_responses_written;   // DEBUG: Count of responses actually written
    volatile uint64_t debug_last_batch_size;     // DEBUG: Size of last batch processed

    // Card 25.2B: Outcome counters (kernel-side)
    // Sum of outcome_ok + outcome_gpu_error must equal total_requests_seen
    volatile uint64_t outcome_ok;                // Requests completed successfully
    volatile uint64_t outcome_gpu_error;         // Requests with GPU error status

    // Card 26.1: Shape truth fields (written once at kernel start)
    // These prove which signer model is ACTUALLY engaged, not just compiled in
    volatile uint32_t signer_model;              // SignerModel enum (THREAD_ONLY=0, WARP_COOP=1)
    volatile uint32_t threads_per_signature;     // 1 for THREAD_ONLY, 32 for WARP_COOP
    volatile uint32_t service_lanes;             // Number of CTAs (currently 1)
    volatile uint32_t comb_table_loaded;         // 1 if comb table uploaded, 0 otherwise

    // Card 26.1B: Per-CTA distribution telemetry (starvation proof)
    // Tracks how work is distributed across service lanes
    static constexpr uint32_t MAX_SERVICE_LANES = 64;  // Card 26.2: Increased for 60 CTAs
    volatile uint64_t per_cta_batches[MAX_SERVICE_LANES];   // Batches processed by each CTA
    volatile uint64_t per_cta_requests[MAX_SERVICE_LANES];  // Requests processed by each CTA
    volatile uint64_t per_cta_attempts[MAX_SERVICE_LANES];  // Card 26.2: Dequeue attempts per CTA

    // Card 26.2: Debug beacons for hang localization
    // These counters help pinpoint exactly where a kernel hang occurs
    volatile uint64_t dbg_enter_kernel;      // Incremented once at kernel start
    volatile uint64_t dbg_dequeue_attempts;  // Incremented on each dequeue attempt
    volatile uint64_t dbg_claim_success;     // Incremented when CTA claims work
    volatile uint64_t dbg_batch_begin;       // Incremented when batch begins
    volatile uint64_t dbg_sign_begin;        // Before calling signer
    volatile uint64_t dbg_sign_done;         // After signer returns
    volatile uint64_t dbg_write_response;    // After response written
    volatile uint64_t dbg_loop_heartbeat;    // Every 1000 iterations

    // Card 26.5: Inversion census - proves whether batch or per-sig inversion
    volatile uint64_t dbg_inv_fermat_calls;    // Per-sig Fermat/exp inversions (bad if high)
    volatile uint64_t dbg_inv_card14_batches;  // Card14 batch inversions (good if ~batches)
    volatile uint64_t dbg_j2a_calls;           // Jacobian→Affine conversion events
    volatile uint64_t dbg_scalar_mul_calls;    // k*G scalar multiplication calls

    // Card 26.5: Stage cycle breakdown (raw clock64 cycles)
    // Reveals which stage dominates compute time
    volatile uint64_t cyc_dequeue;        // Time in dequeue/claim phase
    volatile uint64_t cyc_scalar_mul;     // Time in k*G scalar multiplication
    volatile uint64_t cyc_affine;         // Time in Jacobian→Affine conversion
    volatile uint64_t cyc_sign_finalize;  // Time computing r,s from affine point
    volatile uint64_t cyc_enqueue_resp;   // Time writing response to ring

    // Card 26.10: Response buffer overflow detection
    // Kernel checks (write_idx - read_idx >= BUFFER_SIZE) before writing
    volatile uint64_t response_overwrites;     // Count of potential overwrites
    volatile uint64_t response_seq_counter;    // Current sequence number (monotonic)

    // Card 26.12: CQ sizing configuration
    // Allows runtime tuning of completion queue size without recompile
    volatile uint32_t cq_capacity_active;      // Current active CQ size (for modulo)
    volatile uint32_t cq_capacity_max;         // Maximum CQ size (compile-time)

    // Card 26.14: Poll mode indicator (proves mapped memory is active)
    // 0 = UNKNOWN, 1 = MAPPED (cudaHostAllocMapped), 2 = COPY (cudaMemcpy fallback)
    volatile uint32_t poll_mode;               // Always 1 for FastPath
    volatile uint32_t _pad_poll;               // Alignment padding

    // Card 26.12: Fast-path loop state counters (ITERATION-BASED - diagnostic only)
    // Each tid==0 iteration is classified into exactly one bucket:
    // - IDLE_EMPTY: resp_space > 0 AND available == 0 (GPU waiting for host to publish)
    // - IDLE_RESPFULL: resp_space == 0 (GPU blocked by host not draining)
    // - BUSY: resp_space > 0 AND available > 0 (GPU has work to do)
    // WARNING: These count iterations, NOT time. Use cycle counters for decisions.
    volatile uint64_t state_idle_empty;        // Iterations waiting for work (ring empty)
    volatile uint64_t state_idle_respfull;     // Iterations blocked (response buffer full)
    volatile uint64_t state_busy;              // Iterations with work to process

    // Card 26.15: CYCLE-BASED state counters (AUTHORITATIVE for GO/NO-GO decisions)
    // These measure actual GPU cycles spent in each state using clock64().
    // Unlike iteration counters, these reflect real time spent doing work.
    volatile uint64_t cycles_idle_empty;       // Cycles waiting for work (ring empty)
    volatile uint64_t cycles_idle_respfull;    // Cycles blocked (response buffer full)
    volatile uint64_t cycles_compute;          // Cycles executing ECDSA compute
    volatile uint64_t cycles_total;            // Total cycles measured (for sanity check)

    // Card 26.22: Multi-message work amplification telemetry
    // Tracks logical units (digests) completed, not just ABI requests
    // For multi-msg SHA-256: dbg_units_completed > debug_responses_written
    volatile uint64_t dbg_units_completed;     // Total digests/signatures completed

    // DECRYPT-ABI-01: Decrypt telemetry counters
    volatile uint64_t total_decrypt_aes_gcm;         // Total AES-GCM decrypt ops completed
    volatile uint64_t total_decrypt_chacha20;         // Total ChaCha20-Poly1305 decrypt ops completed
    volatile uint64_t total_auth_failures;            // Total auth tag verification failures (critical safety counter)
    volatile uint64_t plaintext_written_when_auth_fail; // MUST STAY 0 — invariant: no plaintext on auth fail

    // Card 26.34 DEBUG: Keyslot diagnostics (temporary)
    // Proves which slot kernel reads, whether key data is present
    volatile uint32_t dbg_slot_idx_used;       // Slot index kernel selected
    volatile uint32_t dbg_active_epoch_val;    // Value of *active_epoch when read
    volatile uint32_t dbg_key0_first_u32;      // First 4 bytes of keyslots[0].private_d
    volatile uint32_t dbg_key1_first_u32;      // First 4 bytes of keyslots[1].private_d
    volatile uint64_t dbg_keyslots_ptr;        // Kernel's keyslots pointer address
    volatile uint64_t dbg_key0_addr;           // Address kernel reads key0.private_d from

    // N5-A (2026-05-14): split P-256 sharded warp-coop CRYPTO_ERROR returns
    // by failing phase. phase1_fail: state.valid==false (RFC6979 k-gen).
    // phase3_fail: state.valid==true but ecdsa_sign_from_r_warp returned
    // false. Incremented once per op by lane 0 in the P-256 sharded branch.
    volatile uint64_t dbg_p256_phase1_fail;
    volatile uint64_t dbg_p256_phase3_fail;

    // N5 r_zero (2026-05-15): count ops where ecdsa_sign_from_r_warp's
    // `reduce_x_mod_n(r, x_full)` produced r == 0. Splits phase3_fail
    // into (r==0: upstream affine/Zinv corruption) vs (s==0: Montgomery
    // chain bug). Lane 0 of warp only — once per op.
    volatile uint64_t dbg_p256_r_zero;

    // N5 x_full_zero (2026-05-16): count ops where the gathered x_full
    // (affine x-coordinate entering reduce_x_mod_n) is all-zero. Splits
    // r_zero into (x_full == 0: jacobian_to_affine/Zinv emitted zero)
    // vs (x_full == n: physically impossible without bug). Lane 0 of
    // warp only — once per op.
    volatile uint64_t dbg_p256_x_full_zero;

    // N5 zinv_zero (2026-05-16): count ops whose reconstructed 8-limb
    // Zinv (post phase4a `p256_batch_inv_block_montgomery_v2`, pre
    // phase3 jacobian_to_affine_with_zinv_warp) is all-zero. Splits
    // x_full_zero into (zinv == 0: batch-inversion broken) vs
    // (zinv != 0 but x_full == 0: jacobian_to_affine_with_zinv_warp
    // broken). Lane 0 of warp only — once per op.
    volatile uint64_t dbg_p256_zinv_zero;
};

// Card 25.2B: Host-side outcome tracking
// These outcomes can only be detected by the host, not the kernel
struct FastPathHostOutcomes {
    uint64_t ring_full;              // Submit failed - ring was full
    uint64_t timeout;                // Poll timed out waiting for response
    uint64_t kernel_exited;          // Kernel exited before completion
};

// Card 25.1: FAST PATH engine state (separate from legacy EngineState)
struct FastPathEngineState {
    // Ring buffer (mapped memory - host pointer and device pointer are different)
    FastPathRing* d_ring;              // Device pointer to ring (mapped)
    FastPathResponseBuffer* d_responses;  // Device pointer to response buffer

    // Telemetry (device memory)
    FastPathTelemetry* d_telemetry;

    // Keyslot reference (shares with legacy EngineState)
    Keyslot* d_keyslots;               // Device pointer to keyslots

    // Card 26.70: RSA keyslots (separate due to large key size)
    RSAKeyslot* d_rsa_keyslots;        // Device pointer to RSA keyslots

    // Card 27.13: ML-DSA keyslots (separate due to large key size)
    MLDSAKeyslot* d_mldsa_keyslots;    // Device pointer to ML-DSA keyslots

    // Control
    volatile bool* d_shutdown;         // Device pointer to shutdown flag
    volatile uint32_t* d_active_epoch; // Device pointer to active epoch

    // Host-side copies for API
    FastPathRing* h_ring;              // Host pointer to ring (pinned)
    FastPathTelemetry h_telemetry_snapshot; // Latest telemetry copy
};

} // namespace engine
} // namespace smoke
