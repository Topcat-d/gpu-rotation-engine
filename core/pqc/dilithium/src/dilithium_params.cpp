// dilithium_params.cpp - Parameter definitions for ML-DSA modes
// Values from FIPS 204 Table 2

#include "../include/dilithium_params.h"

namespace smoke {
namespace dilithium {

namespace {

// ML-DSA-44 (Dilithium2) - NIST Level 2
constexpr Params MLDSA44 = {
    /* n         */ 256,
    /* q         */ 8380417,
    /* k         */ 4,
    /* l         */ 4,
    /* d         */ 13,
    /* tau       */ 39,
    /* eta       */ 2,
    /* omega     */ 80,
    /* gamma1    */ (1u << 17),           // 2^17 = 131072
    /* gamma2    */ (8380417 - 1) / 88,   // 95232
    /* beta      */ 78,                   // tau * eta
    /* pk_size   */ 1312,
    /* sk_size   */ 2560,
    /* sig_size  */ 2420,
    /* mode      */ Mode::ML_DSA_44
};

// ML-DSA-65 (Dilithium3) - NIST Level 3
constexpr Params MLDSA65 = {
    /* n         */ 256,
    /* q         */ 8380417,
    /* k         */ 6,
    /* l         */ 5,
    /* d         */ 13,
    /* tau       */ 49,
    /* eta       */ 4,
    /* omega     */ 55,
    /* gamma1    */ (1u << 19),           // 2^19 = 524288
    /* gamma2    */ (8380417 - 1) / 32,   // 261888
    /* beta      */ 196,                  // tau * eta
    /* pk_size   */ 1952,
    /* sk_size   */ 4032,
    /* sig_size  */ 3309,
    /* mode      */ Mode::ML_DSA_65
};

// ML-DSA-87 (Dilithium5) - NIST Level 5
constexpr Params MLDSA87 = {
    /* n         */ 256,
    /* q         */ 8380417,
    /* k         */ 8,
    /* l         */ 7,
    /* d         */ 13,
    /* tau       */ 60,
    /* eta       */ 2,
    /* omega     */ 75,
    /* gamma1    */ (1u << 19),           // 2^19 = 524288
    /* gamma2    */ (8380417 - 1) / 32,   // 261888
    /* beta      */ 120,                  // tau * eta
    /* pk_size   */ 2592,
    /* sk_size   */ 4896,
    /* sig_size  */ 4627,
    /* mode      */ Mode::ML_DSA_87
};

} // namespace

const Params& get_params(Mode mode) {
    switch (mode) {
        case Mode::ML_DSA_44: return MLDSA44;
        case Mode::ML_DSA_65: return MLDSA65;
        case Mode::ML_DSA_87: return MLDSA87;
        default:              return MLDSA44;  // Safe default
    }
}

const char* mode_to_string(Mode mode) {
    switch (mode) {
        case Mode::ML_DSA_44: return "ML-DSA-44";
        case Mode::ML_DSA_65: return "ML-DSA-65";
        case Mode::ML_DSA_87: return "ML-DSA-87";
        default:              return "UNKNOWN";
    }
}

} // namespace dilithium
} // namespace smoke
