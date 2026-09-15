#pragma once

// Card 26.18: Multi-Op Schema Header
// Single source of truth for all operation codes, request/response layouts.
// This header is used by: engine loop, ABI marshaling, Python bindings.

#include <stdint.h>

namespace smoke {
namespace engine {

// ============================================================================
// OPERATION CODES
// ============================================================================
// OpCode is stored in FastPathRequest.opcode (2 bytes, allows 65536 ops)
// OpCode 0 = P256_SIGN for backward compatibility with existing code.

enum class OpCode : uint16_t {
    // Signing operations
    P256_SIGN = 0,          // ECDSA P-256 sign (default, backward compatible)
    RSA_2048_SIGN = 1,      // Card 26.70: RSA-2048 CRT sign

    // Hash operations (10-19)
    SHA256 = 10,            // SHA-256 hash
    SHA512 = 11,            // SHA-512 hash
    SHA3_256 = 12,          // SHA3-256 (Keccak)
    SHA3_512 = 13,          // SHA3-512 (Keccak)
    BLAKE2B_256 = 14,       // BLAKE2b-256
    BLAKE2B_512 = 15,       // BLAKE2b-512

    // Symmetric encryption (20-29)
    AES256_GCM = 20,        // AES-256-GCM authenticated encryption (seal)
    CHACHA20_POLY1305 = 21, // ChaCha20-Poly1305 AEAD (encrypt)
    AES256_CTR = 22,        // Card 26.54: AES-256-CTR stream cipher
    CHACHA20 = 23,          // Card 26.66: ChaCha20 stream cipher
    AES256_GCM_OPEN = 24,   // DECRYPT-ABI-01: AES-256-GCM authenticated decryption (open)
    CHACHA20_POLY1305_OPEN = 25, // DECRYPT-ABI-01: ChaCha20-Poly1305 AEAD decrypt (open)

    // Key derivation (30-39)
    HKDF_SHA256 = 30,       // HKDF with SHA-256

    // MAC operations (40-49)
    HMAC_SHA256 = 40,       // Card 26.29: HMAC-SHA256 (keyed hash), inline ≤64B msg
    HMAC_SHA512 = 41,       // Card 26.51: HMAC-SHA512 (keyed hash)
    HMAC_SHA384 = 42,       // Card 26.52: HMAC-SHA384 (keyed hash)
    POLY1305 = 43,          // Card 26.67: Poly1305 MAC
    HMAC_SHA256_VARLEN = 44,// ATTEST-NATIVE-01: variable-length HMAC-SHA256 over
                            // payload slab (one digest/request). Distinct from 40
                            // (whose ≤64B uniform-stride multimsg contract is
                            // load-bearing); fail-closed by construction — old
                            // callers can never hit the varlen slab path.

    // Post-quantum signing (50-59) - Card 27.11
    MLDSA_SIGN = 50,        // ML-DSA sign (msg in slab, sig in slab)
    MLDSA_KEYGEN = 51,      // ML-DSA keygen (reserved)
    MLDSA_VERIFY = 52,      // ML-DSA verify (reserved)
    SLH_DSA_SIGN = 53,      // Card SLH-002 P2: SLH-DSA-SHA2-128s sign
                            // (msg raw, sig 7856 B in PQ signature slab region).
                            // Host-mediated dispatch ONLY (fast_path_submit_slh);
                            // the persistent kernel refuses it inline.

    // Reserved for future use
    RESERVED_MAX = 0xFFFF
};

// ============================================================================
// OPERATION STATUS CODES
// ============================================================================
// Common status codes across all operations

enum class OpStatus : uint8_t {
    OK = 0,
    INVALID_OPCODE = 1,
    INVALID_INPUT_LEN = 2,
    INVALID_KEY_SLOT = 3,
    KEY_NOT_LOADED = 4,
    CRYPTO_ERROR = 5,
    BUFFER_TOO_SMALL = 6,
    INVALID_MULTIMSG = 7,        // Card 26.22: Multi-message validation failed
    SLAB_OVERRUN = 8,            // Slab-fix: output segment may be recycled (drained too late)
    INTERNAL_ERROR = 255
};

// ============================================================================
// MULTI-MESSAGE HEADER (Card 26.22 + 26.24)
// ============================================================================
// When FAST_PATH_FLAG_MULTIMSG is set, input[0..7] contains this header:
//
// Layout:
//   [0-1] n_msgs   (uint16_t): Number of messages
//   [2-3] msg_len  (uint16_t): Bytes per message (must be <= 32)
//   [4-7] flags    (uint32_t): Reserved (must be 0)
//
// ---- Phase 1 (Card 26.22): Inline Payload ----
// Messages follow at input[8..8+n_msgs*msg_len-1]
//
// Constraints (Phase 1):
//   - payload_len == 0 (use inline input buffer)
//   - n_msgs * msg_len + 8 <= 32 (must fit in input buffer)
//   - n_msgs * 32 <= 64 (output digests must fit in output buffer)
//   - Therefore: n_msgs <= 2
//
// Example: n_msgs=2, msg_len=12
//   Header: input[0..7]
//   Msg 0:  input[8..19]
//   Msg 1:  input[20..31]
//   Output: output[0..31] = digest 0, output[32..63] = digest 1
//
// ---- Phase 2 (Card 26.24): Extended Payload Slab ----
// Messages are stored in payload_slab[payload_offset..payload_offset+payload_len]
//
// Constraints (Phase 2):
//   - payload_len > 0 (use payload slab)
//   - n_msgs * msg_len <= payload_len
//   - n_msgs * 32 <= 64 (output buffer still capped at 64 bytes)
//   - Therefore: n_msgs <= 2 (until response slab is extended)
//
// Example: n_msgs=2, msg_len=64, payload_len=128
//   Header: input[0..7]
//   Msg 0:  payload_slab[payload_offset..payload_offset+63]
//   Msg 1:  payload_slab[payload_offset+64..payload_offset+127]
//   Output: output[0..31] = digest 0, output[32..63] = digest 1
//
// Note: Phase 2 infrastructure is in place. To support N > 2:
//   1. Extend response output buffer (or add response slab)
//   2. Update MULTIMSG_MAX_OUTPUT accordingly

struct MultiMsgHeader {
    uint16_t n_msgs;    // Number of messages packed (1-2 with 64-byte output)
    uint16_t msg_len;   // Bytes per message (<=32 for SHA-256 short)
    uint32_t flags;     // Reserved (must be 0)
};

// Multi-message limits
// Phase 1: inline input buffer (32 bytes max), output capped at 64 bytes (N <= 2)
// Phase 2: payload slab allows larger inputs, but output still capped at 64 bytes
// Phase 3 (Card 26.25): response slab allows N up to 64 digests (2048 bytes per slot)
constexpr uint16_t MULTIMSG_MAX_N = 64;             // Card 26.25: Max messages with response slab
constexpr uint16_t MULTIMSG_MAX_N_INLINE = 2;       // Max messages without slab (legacy, output[64])
constexpr uint16_t MULTIMSG_MAX_MSG_LEN = 32;       // Max bytes per message
constexpr uint16_t MULTIMSG_HEADER_SIZE = 8;        // sizeof(MultiMsgHeader)
constexpr uint16_t MULTIMSG_MAX_INPUT = 32;         // Phase 1: inline input buffer size
constexpr uint16_t MULTIMSG_MAX_OUTPUT = 64;        // Inline output buffer size (2 digests)
constexpr uint16_t MULTIMSG_MAX_OUTPUT_SLAB = 2048; // Card 26.25: Output segment size (64 digests)

// Card 26.29: HMAC-SHA256 constraints
constexpr uint16_t HMAC_SHA256_MAX_KEY_LEN = 64;    // HMAC key limit (fits in keyslot)
constexpr uint16_t HMAC_SHA256_MAX_MSG_LEN = 64;    // Single message limit
constexpr uint16_t HMAC_SHA256_OUTPUT_LEN = 32;     // SHA-256 digest output

// Card 26.51: HMAC-SHA512 constraints
constexpr uint16_t HMAC_SHA512_MAX_KEY_LEN = 128;   // HMAC-SHA512 key limit (block size)
constexpr uint16_t HMAC_SHA512_MAX_MSG_LEN = 64;    // Single message limit (same as SHA256)
constexpr uint16_t HMAC_SHA512_OUTPUT_LEN = 64;     // SHA-512 digest output

// Card 26.52: HMAC-SHA384 constraints
constexpr uint16_t HMAC_SHA384_MAX_KEY_LEN = 128;   // HMAC-SHA384 key limit (block size)
constexpr uint16_t HMAC_SHA384_MAX_MSG_LEN = 64;    // Single message limit
constexpr uint16_t HMAC_SHA384_OUTPUT_LEN = 48;     // SHA-384 digest output

// Card 26.54: AES-256-CTR constraints
constexpr uint16_t AES256_CTR_KEY_LEN = 32;         // AES-256 key size
constexpr uint16_t AES256_CTR_NONCE_LEN = 12;       // Standard CTR nonce (96 bits)
constexpr uint16_t AES256_CTR_MAX_INLINE_PT = 20;   // Max plaintext in inline mode (32 - 12)
constexpr uint16_t AES256_CTR_MIN_INPUT = 12;       // At least nonce required

// DECRYPT-ABI-01: AES-256-GCM Open (decrypt) constraints
// Input layout: nonce[12] || ciphertext[N] || tag[16]
// Inline mode: max 4 bytes ciphertext (32 - 12 - 16 = 4)
// Slab mode: variable ciphertext length
constexpr uint16_t AES256_GCM_OPEN_NONCE_LEN = 12;       // Nonce size
constexpr uint16_t AES256_GCM_OPEN_TAG_LEN = 16;         // Auth tag size
constexpr uint16_t AES256_GCM_OPEN_MIN_INPUT = 28;       // nonce[12] + tag[16], 0 bytes ct
constexpr uint16_t AES256_GCM_OPEN_MAX_INLINE_CT = 4;    // Max ciphertext in inline (32 - 12 - 16)
constexpr uint16_t AES256_GCM_OPEN_MAX_INLINE_INPUT = 32; // 12 + 4 + 16

// DECRYPT-ABI-01: ChaCha20-Poly1305 Open (decrypt) constraints
constexpr uint16_t CHACHA_OPEN_NONCE_LEN = 12;
constexpr uint16_t CHACHA_OPEN_TAG_LEN = 16;
constexpr uint16_t CHACHA_OPEN_MIN_INPUT = 28;            // nonce[12] + tag[16], 0 bytes ct
constexpr uint16_t CHACHA_OPEN_MAX_INLINE_CT = 4;         // Max ciphertext in inline (32 - 12 - 16)

// Card 26.66: ChaCha20 constraints
constexpr uint16_t CHACHA20_KEY_LEN = 32;           // ChaCha20 key size (256 bits)
constexpr uint16_t CHACHA20_NONCE_LEN = 12;         // RFC 8439 nonce (96 bits)
constexpr uint16_t CHACHA20_MAX_INLINE_PT = 20;     // Max plaintext in inline mode (32 - 12)
constexpr uint16_t CHACHA20_MIN_INPUT = 12;         // At least nonce required

// Card 26.67: Poly1305 constraints
constexpr uint16_t POLY1305_KEY_LEN = 32;           // Poly1305 key size (r[16] + s[16])
constexpr uint16_t POLY1305_TAG_LEN = 16;           // Output tag size
constexpr uint16_t POLY1305_MAX_INLINE_MSG = 32;    // Max message in inline mode
constexpr uint16_t POLY1305_MIN_INPUT = 0;          // Empty message is valid

// Card 26.34: P256 Multi-Sign (work amplification)
// Uses FAST_PATH_FLAG_MULTIMSG with opcode P256_SIGN
// Header: n_items (u16), item_len (u16, must be 32), flags (u32)
// Input: payload_slab[payload_offset..] contains n_items * 32 bytes of hashes
// Output: output_slab contains n_items * 64 bytes of signatures (r||s)
// Max n_items = floor(OUTPUT_SEGMENT_BYTES / 64) = 32
constexpr uint16_t P256_MULTISIGN_ITEM_LEN = 32;    // Each hash is 32 bytes
constexpr uint16_t P256_MULTISIGN_SIG_LEN = 64;     // Each signature is 64 bytes (r||s)
constexpr uint16_t P256_MULTISIGN_MAX_N = 32;       // Max signatures per request (2048/64)

// Card 26.70: RSA-2048 CRT constraints
// Input: 256 bytes (PKCS#1 v1.5 padded message) via payload slab
// Output: 256 bytes (signature) via output slab
// Key: RSA-2048 CRT key in dedicated RSA keyslot (~904 bytes)
constexpr uint16_t RSA2048_MESSAGE_LEN = 256;       // Input message size (must be PKCS#1 padded)
constexpr uint16_t RSA2048_SIGNATURE_LEN = 256;     // Output signature size
constexpr uint8_t RSA2048_MAX_KEYSLOTS = 4;         // Max RSA keyslots (large keys = fewer slots)

// Card 27.11: ML-DSA (Dilithium) constraints
// Uses payload slab for variable-length messages, output slab for signatures.
// Mode is specified in request.input[0]: 0=ML-DSA-44, 1=ML-DSA-65 (default), 2=ML-DSA-87
// Key material stored in MLDSA keyslots (precomputed NTT values for efficiency)

// ML-DSA signature sizes (NIST FIPS 204)
constexpr uint16_t MLDSA44_SIG_BYTES = 2420;        // ML-DSA-44 signature
constexpr uint16_t MLDSA65_SIG_BYTES = 3309;        // ML-DSA-65 signature (default)
constexpr uint16_t MLDSA87_SIG_BYTES = 4627;        // ML-DSA-87 signature

// ML-DSA private key sizes (expanded form)
constexpr uint16_t MLDSA44_SK_BYTES = 2560;         // ML-DSA-44 secret key
constexpr uint16_t MLDSA65_SK_BYTES = 4032;         // ML-DSA-65 secret key
constexpr uint16_t MLDSA87_SK_BYTES = 4896;         // ML-DSA-87 secret key

// ML-DSA public key sizes
constexpr uint16_t MLDSA44_PK_BYTES = 1312;         // ML-DSA-44 public key
constexpr uint16_t MLDSA65_PK_BYTES = 1952;         // ML-DSA-65 public key
constexpr uint16_t MLDSA87_PK_BYTES = 2592;         // ML-DSA-87 public key

// ML-DSA mode identifiers (in request.input[0])
constexpr uint8_t MLDSA_MODE_44 = 0;                // NIST Level 2
constexpr uint8_t MLDSA_MODE_65 = 1;                // NIST Level 3 (default)
constexpr uint8_t MLDSA_MODE_87 = 2;                // NIST Level 5

// Note: MLDSA_MAX_KEYSLOTS is defined in engine_state.cuh with keyslot structures

// Card SLH-002 P2: SLH-DSA-SHA2-128s (FIPS 205) constraints.
// Sign only; the raw message (NOT a digest) is the input — the PRF_msg /
// H_msg prologue runs ON DEVICE from the SLHKeyslot (key residency: SK.prf
// never leaves the GPU after load). Signature is fixed 7,856 bytes and does
// NOT fit a 2,048-byte standard output segment — it returns via the dedicated
// PQ signature output region (engine_state.cuh, 8,192-byte slots).
constexpr uint16_t SLH_DSA_128S_SIG_BYTES = 7856;   // FIPS 205 Table 2
constexpr uint16_t SLH_DSA_MAX_MSG_BYTES  = 65535;  // uint16 schema ceiling

// ============================================================================
// OPERATION CONSTRAINTS
// ============================================================================
// Per-op constraints for validation

struct OpConstraints {
    uint16_t opcode;
    uint16_t min_input_len;
    uint16_t max_input_len;
    uint16_t output_len;       // Fixed output length (0 = variable)
    bool     requires_key;     // Needs keyslot
};

// Constraint table (can be used for validation)
// Note: For GPU, keep this small or move to constant memory
constexpr OpConstraints OP_CONSTRAINTS[] = {
    // opcode          min_in  max_in  out_len  requires_key
    { (uint16_t)OpCode::P256_SIGN,     32,     32,     64,      true  },  // hash[32] -> r[32]||s[32]
    { (uint16_t)OpCode::RSA_2048_SIGN, 256,    256,    256,     true  },  // Card 26.70: msg[256] -> sig[256]
    { (uint16_t)OpCode::SHA256,         0,  65535,     32,      false },  // any -> digest[32]
    { (uint16_t)OpCode::SHA512,         0,  65535,     64,      false },  // any -> digest[64]
    { (uint16_t)OpCode::SHA3_256,       0,  65535,     32,      false },  // any -> digest[32]
    { (uint16_t)OpCode::SHA3_512,       0,  65535,     64,      false },  // any -> digest[64]
    { (uint16_t)OpCode::BLAKE2B_256,    0,  65535,     32,      false },  // any -> digest[32]
    { (uint16_t)OpCode::BLAKE2B_512,    0,  65535,     64,      false },  // any -> digest[64]
    // SEAL-SLAB-01: inline carries nonce[12]+pt[0-16]; the slab path carries
    // nonce[12] inline and plaintext[1..65535] in the payload slab. The
    // uint16 schema ceiling records the maximum plaintext payload. Dispatch
    // validates the exact inline/slab framing and refuses oversize input.
    { (uint16_t)OpCode::AES256_GCM,    12,  65535,      0,      true  },  // nonce inline + pt slab -> ct+tag
    { (uint16_t)OpCode::CHACHA20_POLY1305, 12, 65535,   0,      true  },  // nonce inline + pt slab -> ct+tag
    { (uint16_t)OpCode::AES256_CTR,    12,     32,      0,      true  },  // Card 26.54: nonce[12]+pt[0-20] -> ct[0-20]
    { (uint16_t)OpCode::CHACHA20,      12,     32,      0,      true  },  // Card 26.66: nonce[12]+pt[0-20] -> ct[0-20]
    { (uint16_t)OpCode::AES256_GCM_OPEN, 28, 65535,     0,      true  },  // DECRYPT-ABI-01: nonce[12]+ct[N]+tag[16] -> pt[N]
    { (uint16_t)OpCode::CHACHA20_POLY1305_OPEN, 28, 65535, 0,   true  },  // DECRYPT-ABI-01: nonce[12]+ct[N]+tag[16] -> pt[N]
    { (uint16_t)OpCode::HKDF_SHA256,   32,     96,     32,      true  },  // seed -> derived[32]
    { (uint16_t)OpCode::HMAC_SHA256,    0,     64,     32,      true  },  // Card 26.29: key + msg -> digest[32]
    { (uint16_t)OpCode::HMAC_SHA512,    0,     64,     64,      true  },  // Card 26.51: key + msg -> digest[64]
    { (uint16_t)OpCode::HMAC_SHA384,    0,     64,     48,      true  },  // Card 26.52: key + msg -> digest[48]
    // ATTEST-NATIVE-01: variable-length HMAC over payload slab. 65535 is the
    // uint16 schema-field ceiling (advisory); the real bound is payload-slab
    // capacity, enforced fail-closed in the engine_loop.cu branch — same
    // convention as AES256_GCM_OPEN. The streaming device HMAC (no 1024B cap)
    // makes the varlen input honest.
    { (uint16_t)OpCode::HMAC_SHA256_VARLEN, 0, 65535,  32,      true  },  // slab msg -> digest[32]
    // Card 27.11: ML-DSA (slab-based, variable output based on mode)
    { (uint16_t)OpCode::MLDSA_SIGN,     8,  65535,      0,      true  },  // msg -> sig (mode-dependent size)
    // Card SLH-002 P2: SLH-DSA-SHA2-128s. Raw message (0..65535 B) in, fixed
    // 7,856-byte signature out via the PQ signature slab region. Host-mediated
    // dispatch only — the persistent kernel refuses this opcode inline.
    { (uint16_t)OpCode::SLH_DSA_SIGN,   0,  SLH_DSA_MAX_MSG_BYTES, SLH_DSA_128S_SIG_BYTES, true },
};
constexpr size_t OP_CONSTRAINTS_COUNT = sizeof(OP_CONSTRAINTS) / sizeof(OP_CONSTRAINTS[0]);

// SEAL-SLAB-01 schema lock: the AEAD seal opcodes advertise the slab ceiling.
// The dispatch sites independently validate exact payload bounds and ownership.
constexpr uint16_t op_declared_max_input(OpCode op) {
    for (size_t i = 0; i < OP_CONSTRAINTS_COUNT; i++) {
        if (OP_CONSTRAINTS[i].opcode == (uint16_t)op) return OP_CONSTRAINTS[i].max_input_len;
    }
    return 0;
}
static_assert(op_declared_max_input(OpCode::CHACHA20_POLY1305) == 65535,
    "CHACHA20_POLY1305 schema must expose the SEAL-SLAB-01 ceiling");
static_assert(op_declared_max_input(OpCode::AES256_GCM) == 65535,
    "AES256_GCM schema must expose the SEAL-SLAB-01 ceiling");

// ============================================================================
// REQUEST LAYOUT (Card 26.18 Multi-Op Extension)
// ============================================================================
// Extended FastPathRequest with opcode support.
// Backward compatible: opcode=0 means P256_SIGN (existing behavior).
//
// Layout (64 bytes, cache-aligned):
//   [0-7]   request_id   (8 bytes)
//   [8-15]  t_submit_us  (8 bytes)
//   [16-47] input[32]    (32 bytes) - hash for P256, data for hashes
//   [48]    flags        (1 byte)
//   [49]    key_slot     (1 byte)
//   [50]    client_id    (1 byte)
//   [51-52] opcode       (2 bytes) - NEW: operation code
//   [53-54] input_len    (2 bytes) - NEW: actual input length (for variable ops)
//   [55-63] _reserved    (9 bytes)

// Note: The actual struct is in engine_state.cuh (FastPathRequest).
// This header documents the expected layout for multi-op.

// ============================================================================
// RESPONSE LAYOUT
// ============================================================================
// Response structure remains compatible.
// output[64] is interpreted based on opcode:
//   - P256_SIGN: r[32] || s[32]
//   - SHA256:    digest[32] || zeros[32]
//   - SHA512:    digest[64]
//   - etc.

// ============================================================================
// HELPER: Get output length for an opcode (CUDA-only)
// ============================================================================
#ifdef __CUDACC__
__host__ __device__ inline uint16_t get_output_len(OpCode op) {
    switch (op) {
        case OpCode::P256_SIGN:      return 64;  // r[32] + s[32]
        case OpCode::RSA_2048_SIGN:  return 256; // Card 26.70: 2048-bit signature
        case OpCode::SHA256:         return 32;
        case OpCode::SHA512:         return 64;
        case OpCode::SHA3_256:       return 32;
        case OpCode::SHA3_512:       return 64;
        case OpCode::BLAKE2B_256:    return 32;
        case OpCode::BLAKE2B_512:    return 64;
        case OpCode::HKDF_SHA256:    return 32;
        case OpCode::HMAC_SHA256:    return 32;  // Card 26.29
        case OpCode::HMAC_SHA512:    return 64;  // Card 26.51
        case OpCode::HMAC_SHA384:    return 48;  // Card 26.52
        case OpCode::POLY1305:       return 16;  // Card 26.67: Fixed 16-byte tag
        case OpCode::AES256_CTR:     return 0;   // Card 26.54: Variable (ciphertext = plaintext len)
        case OpCode::CHACHA20:       return 0;   // Card 26.66: Variable (ciphertext = plaintext len)
        case OpCode::AES256_GCM_OPEN:        return 0;  // DECRYPT-ABI-01: Variable (plaintext = ciphertext len)
        case OpCode::CHACHA20_POLY1305_OPEN: return 0;  // DECRYPT-ABI-01: Variable (plaintext = ciphertext len)
        case OpCode::MLDSA_SIGN:     return 0;   // Card 27.11: Variable (mode-dependent, 2420/3309/4627)
        case OpCode::SLH_DSA_SIGN:   return SLH_DSA_128S_SIG_BYTES;  // Card SLH-002: fixed 7856 (slab)
        default:                     return 0;   // Variable or unknown
    }
}

// Card 27.11: Get MLDSA signature size by mode
__host__ __device__ inline uint16_t get_mldsa_sig_len(uint8_t mode) {
    switch (mode) {
        case MLDSA_MODE_44: return MLDSA44_SIG_BYTES;
        case MLDSA_MODE_65: return MLDSA65_SIG_BYTES;
        case MLDSA_MODE_87: return MLDSA87_SIG_BYTES;
        default:            return 0;  // Invalid mode
    }
}

// ============================================================================
// HELPER: Check if opcode requires a key (CUDA-only)
// ============================================================================
__host__ __device__ inline bool op_requires_key(OpCode op) {
    switch (op) {
        case OpCode::P256_SIGN:
        case OpCode::RSA_2048_SIGN:   // Card 26.70
        case OpCode::AES256_GCM:
        case OpCode::CHACHA20_POLY1305:
        case OpCode::AES256_CTR:    // Card 26.54
        case OpCode::CHACHA20:      // Card 26.66
        case OpCode::AES256_GCM_OPEN:        // DECRYPT-ABI-01
        case OpCode::CHACHA20_POLY1305_OPEN: // DECRYPT-ABI-01
        case OpCode::HKDF_SHA256:
        case OpCode::HMAC_SHA256:   // Card 26.29
        case OpCode::HMAC_SHA512:   // Card 26.51
        case OpCode::HMAC_SHA384:   // Card 26.52
        case OpCode::POLY1305:      // Card 26.67
        case OpCode::MLDSA_SIGN:    // Card 27.11
        case OpCode::SLH_DSA_SIGN:  // Card SLH-002
            return true;
        default:
            return false;
    }
}
#endif // __CUDACC__

// ============================================================================
// HELPER: Human-readable opcode name
// ============================================================================
inline const char* op_name(OpCode op) {
    switch (op) {
        case OpCode::P256_SIGN:         return "P256_SIGN";
        case OpCode::RSA_2048_SIGN:     return "RSA_2048_SIGN";  // Card 26.70
        case OpCode::SHA256:            return "SHA256";
        case OpCode::SHA512:            return "SHA512";
        case OpCode::SHA3_256:          return "SHA3_256";
        case OpCode::SHA3_512:          return "SHA3_512";
        case OpCode::BLAKE2B_256:       return "BLAKE2B_256";
        case OpCode::BLAKE2B_512:       return "BLAKE2B_512";
        case OpCode::AES256_GCM:        return "AES256_GCM";
        case OpCode::CHACHA20_POLY1305: return "CHACHA20_POLY1305";
        case OpCode::AES256_CTR:        return "AES256_CTR";  // Card 26.54
        case OpCode::CHACHA20:          return "CHACHA20";    // Card 26.66
        case OpCode::AES256_GCM_OPEN:        return "AES256_GCM_OPEN";        // DECRYPT-ABI-01
        case OpCode::CHACHA20_POLY1305_OPEN: return "CHACHA20_POLY1305_OPEN"; // DECRYPT-ABI-01
        case OpCode::HKDF_SHA256:       return "HKDF_SHA256";
        case OpCode::HMAC_SHA256:       return "HMAC_SHA256";  // Card 26.29
        case OpCode::HMAC_SHA512:       return "HMAC_SHA512";  // Card 26.51
        case OpCode::HMAC_SHA384:       return "HMAC_SHA384";  // Card 26.52
        case OpCode::POLY1305:          return "POLY1305";     // Card 26.67
        case OpCode::MLDSA_SIGN:        return "MLDSA_SIGN";   // Card 27.11
        case OpCode::MLDSA_KEYGEN:      return "MLDSA_KEYGEN"; // Card 27.11
        case OpCode::MLDSA_VERIFY:      return "MLDSA_VERIFY"; // Card 27.11
        case OpCode::SLH_DSA_SIGN:      return "SLH_DSA_SIGN"; // Card SLH-002
        default:                        return "UNKNOWN";
    }
}

} // namespace engine
} // namespace smoke
