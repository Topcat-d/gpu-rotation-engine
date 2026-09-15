// dilithium_params.h - Dilithium/ML-DSA parameter definitions
// FIPS 204 compliant: ML-DSA-44, ML-DSA-65, ML-DSA-87

#pragma once
#include <cstdint>

namespace smoke {
namespace dilithium {

// Mode enum matching FIPS 204 / Dilithium2/3/5
enum class Mode : uint8_t {
    ML_DSA_44 = 0,  // Dilithium2, NIST Level 2
    ML_DSA_65 = 1,  // Dilithium3, NIST Level 3
    ML_DSA_87 = 2   // Dilithium5, NIST Level 5
};

// Parameter struct holding all mode-specific values
struct Params {
    // Ring parameters (shared across all modes)
    uint32_t n;      // 256 (polynomial degree)
    uint32_t q;      // 8380417 = 2^23 - 2^13 + 1

    // Matrix dimensions: A is k x l over R_q
    uint8_t k;
    uint8_t l;

    // FIPS 204 Table 2 parameters
    uint8_t d;       // Dropped bits from t
    uint8_t tau;     // Number of +/-1 coefficients in challenge c
    uint8_t eta;     // Secret coefficient bound
    uint8_t omega;   // Max number of 1s in hint
    uint32_t gamma1; // y coefficient range (2^17 or 2^19)
    uint32_t gamma2; // Low-order rounding range
    uint16_t beta;   // tau * eta (signing bound)

    // Sizes for allocation
    uint32_t pk_size;  // Public key bytes
    uint32_t sk_size;  // Secret key bytes
    uint32_t sig_size; // Signature bytes

    // Mode identifier for debugging/telemetry
    Mode mode;
};

// Get singleton parameter struct for a mode
const Params& get_params(Mode mode);

// Helper: convert mode to string
const char* mode_to_string(Mode mode);

// NTT-related constants
constexpr uint32_t DILITHIUM_N = 256;
constexpr uint32_t DILITHIUM_Q = 8380417;
constexpr uint32_t DILITHIUM_ROOT = 1753;  // 512-th root of unity

// Montgomery constants for q = 8380417
// R = 2^32
// R mod q = 4193792
// R^2 mod q = 2365951
// q^-1 mod R = 4236238847 (actually -q^-1 mod 2^32)
constexpr uint32_t MONT_R_MOD_Q = 4193792;
constexpr uint32_t MONT_R2_MOD_Q = 2365951;
constexpr uint32_t MONT_QINV = 4236238847u;

// Barrett reduction constant
// v = floor(2^26 / q) + 1 = 8
constexpr uint32_t BARRETT_V = 8;
constexpr uint32_t BARRETT_SHIFT = 26;

} // namespace dilithium
} // namespace smoke
