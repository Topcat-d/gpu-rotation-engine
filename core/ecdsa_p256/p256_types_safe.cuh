// ==============================================================================
// P-256 TYPE-SAFE WRAPPERS
// ==============================================================================
// Purpose: Prevent domain confusion bugs with strong typing
// Created: 2025-11-27
//
// PROBLEM: P-256 operations involve multiple domains:
//   - Standard domain (normal field elements)
//   - Montgomery domain (elements multiplied by R)
//   - Jacobian coordinates (projective representation)
//   - Affine coordinates (standard x,y representation)
//
// Mixing these domains causes subtle bugs (e.g., "Montgomery multiplication bug")
//
// SOLUTION: Use C++ structs to enforce type safety at compile time
// ==============================================================================

#ifndef P256_TYPES_SAFE_CUH
#define P256_TYPES_SAFE_CUH

#include <stdint.h>

// ==============================================================================
// FIELD ELEMENT TYPES - Domain Separation
// ==============================================================================

// Field element in STANDARD domain (normal representation)
struct FE_Standard {
    uint32_t limb[8];

    __device__ __host__ FE_Standard() {
        #pragma unroll
        for (int i = 0; i < 8; ++i) limb[i] = 0;
    }

    __device__ __host__ explicit FE_Standard(uint32_t v0) {
        limb[0] = v0;
        #pragma unroll
        for (int i = 1; i < 8; ++i) limb[i] = 0;
    }
};

// Field element in MONTGOMERY domain (multiplied by R = 2^256 mod p)
struct FE_Montgomery {
    uint32_t limb[8];

    __device__ __host__ FE_Montgomery() {
        #pragma unroll
        for (int i = 0; i < 8; ++i) limb[i] = 0;
    }
};

// Field element in NORMAL domain (alias for clarity in some contexts)
using FE_Normal = FE_Standard;

// ==============================================================================
// POINT TYPES - Coordinate System Separation
// ==============================================================================

// Point in AFFINE coordinates (x, y) - Standard domain
// Represents point (x, y) on curve: y^2 = x^3 - 3x + b
struct Point_Affine_Standard {
    FE_Standard x;
    FE_Standard y;
    bool is_infinity;  // Special flag for point at infinity

    __device__ __host__ Point_Affine_Standard() : is_infinity(true) {}
};

// Point in AFFINE coordinates (x, y) - Montgomery domain
// Represents point (x*R, y*R) on curve
struct Point_Affine_Montgomery {
    FE_Montgomery x;
    FE_Montgomery y;
    bool is_infinity;

    __device__ __host__ Point_Affine_Montgomery() : is_infinity(true) {}
};

// Point in JACOBIAN coordinates (X, Y, Z) - Montgomery domain
// Represents point (X/Z^2, Y/Z^3) on curve
// All coordinates stored in Montgomery domain for efficiency
struct Point_Jacobian_Montgomery {
    FE_Montgomery X;
    FE_Montgomery Y;
    FE_Montgomery Z;

    __device__ __host__ Point_Jacobian_Montgomery() {}
};

// ==============================================================================
// CONVERSION HELPERS - Type-Safe Accessors
// ==============================================================================

// Extract raw limbs from typed field element (read-only)
__device__ __forceinline__ const uint32_t* fe_limbs(const FE_Standard &x) {
    return x.limb;
}

__device__ __forceinline__ const uint32_t* fe_limbs(const FE_Montgomery &x) {
    return x.limb;
}

// Extract raw limbs from typed field element (mutable)
__device__ __forceinline__ uint32_t* fe_limbs_mut(FE_Standard &x) {
    return x.limb;
}

__device__ __forceinline__ uint32_t* fe_limbs_mut(FE_Montgomery &x) {
    return x.limb;
}

// ==============================================================================
// TYPED WRAPPER FUNCTIONS - Field Operations
// ==============================================================================
// These wrappers call the golden reference functions with compile-time
// type checking to prevent domain confusion
// ==============================================================================

#ifdef P256_SCALAR_REF_GOLDEN_CUH

// ----------------------------------------------------------------------------
// Standard Domain Operations
// ----------------------------------------------------------------------------

__device__ __forceinline__ void fe_add_safe(FE_Standard &r, const FE_Standard &a, const FE_Standard &b) {
    fe_add_golden(r.limb, a.limb, b.limb);
}

__device__ __forceinline__ void fe_sub_safe(FE_Standard &r, const FE_Standard &a, const FE_Standard &b) {
    fe_sub_golden(r.limb, a.limb, b.limb);
}

__device__ __forceinline__ void fe_copy_safe(FE_Standard &r, const FE_Standard &a) {
    fe_copy_golden(r.limb, a.limb);
}

__device__ __forceinline__ int fe_cmp_safe(const FE_Standard &a, const FE_Standard &b) {
    return fe_cmp_golden(a.limb, b.limb);
}

__device__ __forceinline__ bool fe_is_zero_safe(const FE_Standard &a) {
    return fe_is_zero_golden(a.limb);
}

// ----------------------------------------------------------------------------
// Montgomery Domain Operations
// ----------------------------------------------------------------------------

__device__ __forceinline__ void fe_mont_add_safe(FE_Montgomery &r, const FE_Montgomery &a, const FE_Montgomery &b) {
    fe_mont_add_golden(r.limb, a.limb, b.limb);
}

__device__ __forceinline__ void fe_mont_sub_safe(FE_Montgomery &r, const FE_Montgomery &a, const FE_Montgomery &b) {
    fe_mont_sub_golden(r.limb, a.limb, b.limb);
}

__device__ __forceinline__ void fe_mont_mul_safe(FE_Montgomery &r, const FE_Montgomery &a, const FE_Montgomery &b) {
    fe_mont_mul_golden(r.limb, a.limb, b.limb);
}

__device__ __forceinline__ void fe_mont_sqr_safe(FE_Montgomery &r, const FE_Montgomery &a) {
    fe_mont_sqr_golden(r.limb, a.limb);
}

__device__ __forceinline__ void fe_mont_copy_safe(FE_Montgomery &r, const FE_Montgomery &a) {
    fe_copy_golden(r.limb, a.limb);
}

__device__ __forceinline__ bool fe_mont_is_zero_safe(const FE_Montgomery &a) {
    return fe_is_zero_golden(a.limb);
}

// ----------------------------------------------------------------------------
// Domain Conversions (CRITICAL - These change representation!)
// ----------------------------------------------------------------------------

// Convert Standard -> Montgomery: r_mont = a_std * R mod p
__device__ __forceinline__ void to_montgomery_safe(FE_Montgomery &r, const FE_Standard &a) {
    to_mont_golden(r.limb, a.limb);
}

// Convert Montgomery -> Standard: r_std = a_mont * R^{-1} mod p
__device__ __forceinline__ void from_montgomery_safe(FE_Standard &r, const FE_Montgomery &a) {
    from_mont_golden(r.limb, a.limb);
}

// ----------------------------------------------------------------------------
// Point Operations - Jacobian Montgomery
// ----------------------------------------------------------------------------

__device__ __forceinline__ void point_double_safe(Point_Jacobian_Montgomery &R, const Point_Jacobian_Montgomery &P) {
    P256_Jacobian_Golden P_raw, R_raw;

    // Copy input
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        P_raw.X[i] = P.X.limb[i];
        P_raw.Y[i] = P.Y.limb[i];
        P_raw.Z[i] = P.Z.limb[i];
    }

    // Call golden reference
    point_double_golden(R_raw, P_raw);

    // Copy output
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        R.X.limb[i] = R_raw.X[i];
        R.Y.limb[i] = R_raw.Y[i];
        R.Z.limb[i] = R_raw.Z[i];
    }
}

__device__ __forceinline__ void point_add_mixed_safe(Point_Jacobian_Montgomery &R,
                                                      const Point_Jacobian_Montgomery &P,
                                                      const Point_Affine_Montgomery &Q) {
    P256_Jacobian_Golden P_raw, R_raw;

    // Copy Jacobian input
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        P_raw.X[i] = P.X.limb[i];
        P_raw.Y[i] = P.Y.limb[i];
        P_raw.Z[i] = P.Z.limb[i];
    }

    // Call golden reference with affine point
    point_add_mixed_golden(R_raw, P_raw, Q.x.limb, Q.y.limb);

    // Copy output
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        R.X.limb[i] = R_raw.X[i];
        R.Y.limb[i] = R_raw.Y[i];
        R.Z.limb[i] = R_raw.Z[i];
    }
}

__device__ __forceinline__ void set_infinity_safe(Point_Jacobian_Montgomery &P) {
    P256_Jacobian_Golden P_raw;
    set_infinity_golden(P_raw);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        P.X.limb[i] = P_raw.X[i];
        P.Y.limb[i] = P_raw.Y[i];
        P.Z.limb[i] = P_raw.Z[i];
    }
}

__device__ __forceinline__ void set_basepoint_safe(Point_Jacobian_Montgomery &P) {
    P256_Jacobian_Golden P_raw;
    set_basepoint_golden(P_raw);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        P.X.limb[i] = P_raw.X[i];
        P.Y.limb[i] = P_raw.Y[i];
        P.Z.limb[i] = P_raw.Z[i];
    }
}

// ----------------------------------------------------------------------------
// Point Conversions - Affine <-> Jacobian
// ----------------------------------------------------------------------------

// Convert Affine Standard -> Affine Montgomery
__device__ __forceinline__ void point_affine_to_mont_safe(Point_Affine_Montgomery &r, const Point_Affine_Standard &a) {
    if (a.is_infinity) {
        r.is_infinity = true;
        return;
    }

    to_montgomery_safe(r.x, a.x);
    to_montgomery_safe(r.y, a.y);
    r.is_infinity = false;
}

// Convert Affine Montgomery -> Affine Standard
__device__ __forceinline__ void point_affine_from_mont_safe(Point_Affine_Standard &r, const Point_Affine_Montgomery &a) {
    if (a.is_infinity) {
        r.is_infinity = true;
        return;
    }

    from_montgomery_safe(r.x, a.x);
    from_montgomery_safe(r.y, a.y);
    r.is_infinity = false;
}

// Convert Affine Montgomery -> Jacobian Montgomery
__device__ __forceinline__ void point_affine_to_jacobian_safe(Point_Jacobian_Montgomery &r, const Point_Affine_Montgomery &a) {
    if (a.is_infinity) {
        set_infinity_safe(r);
        return;
    }

    fe_mont_copy_safe(r.X, a.x);
    fe_mont_copy_safe(r.Y, a.y);

    // Z = 1 in Montgomery domain = R mod p
    FE_Standard one_std(1);
    to_montgomery_safe(r.Z, one_std);
}

#endif // P256_SCALAR_REF_GOLDEN_CUH

// ==============================================================================
// USAGE EXAMPLES
// ==============================================================================
//
// GOOD - Type-safe operations:
// ```
// FE_Standard a_std(42);
// FE_Montgomery a_mont, b_mont, c_mont;
//
// to_montgomery_safe(a_mont, a_std);  // OK: explicit conversion
// fe_mont_mul_safe(c_mont, a_mont, b_mont);  // OK: both Montgomery
// ```
//
// BAD - Would fail at compile time:
// ```
// fe_mont_mul_safe(c_mont, a_std, b_mont);  // ERROR: type mismatch!
// fe_add_safe(c_std, a_mont, b_std);  // ERROR: domain confusion!
// ```
//
// POINT OPERATIONS:
// ```
// Point_Jacobian_Montgomery P, Q, R;
// set_basepoint_safe(P);
// point_double_safe(Q, P);  // OK: 2G
// point_add_mixed_safe(R, Q, affine_pt);  // OK: mixed addition
// ```
//
// ==============================================================================

// ==============================================================================
// MIGRATION GUIDE
// ==============================================================================
//
// To wrap existing code with type-safe wrappers:
//
// 1. Replace raw uint32_t[8] with typed structs:
//    OLD: uint32_t x[8], y[8];
//    NEW: FE_Montgomery x, y;
//
// 2. Replace function calls with _safe versions:
//    OLD: fe_mont_mul_golden(r, a, b);
//    NEW: fe_mont_mul_safe(r, a, b);
//
// 3. Add explicit domain conversions:
//    OLD: mont_mul_golden(r, a, R2);  // implicit domain change
//    NEW: to_montgomery_safe(r_mont, a_std);  // explicit conversion
//
// 4. Use type-safe point structs:
//    OLD: P256_Jacobian_Golden pt;
//    NEW: Point_Jacobian_Montgomery pt;
//
// 5. Compiler will catch domain confusion bugs!
//
// ==============================================================================

#endif // P256_TYPES_SAFE_CUH
