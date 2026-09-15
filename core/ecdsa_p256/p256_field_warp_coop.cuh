





































































// ==============================================================================
// Includes
// ==============================================================================
#include <cstdio>  // For printf in device code (debug statements)

// ==============================================================================
// Card 26.5: Telemetry instrumentation for inversion census
// ==============================================================================
// When built as part of the persistent engine, we can increment telemetry counters
// to prove whether per-sig or batch inversion is being used.
// The g_telemetry_ptr is defined in engine_loop.cu BEFORE including this header.
// We just use the symbol directly (it's already in scope via the using declaration).
#ifdef CARD26_TELEMETRY_ENABLED
#define CARD26_INV_FERMAT_INCREMENT() do { \
    if (g_telemetry_ptr != nullptr && lane_id == 0) { \
        atomicAdd((unsigned long long*)&g_telemetry_ptr->dbg_inv_fermat_calls, 1ull); \
    } \
} while(0)
#define CARD26_J2A_INCREMENT() do { \
    if (g_telemetry_ptr != nullptr && lane_id == 0) { \
        atomicAdd((unsigned long long*)&g_telemetry_ptr->dbg_j2a_calls, 1ull); \
    } \
} while(0)
#define CARD26_SCALAR_MUL_INCREMENT() do { \
    if (g_telemetry_ptr != nullptr && lane_id == 0) { \
        atomicAdd((unsigned long long*)&g_telemetry_ptr->dbg_scalar_mul_calls, 1ull); \
    } \
} while(0)
// Card 26.6: Batch inversion counter (proves Card14 is wired)
// Note: This macro takes batch_size and doesn't require lane_id since batch inv runs on lane0 only
#define CARD26_INV_BATCH_INCREMENT(batch_size) do { \
    if (g_telemetry_ptr != nullptr) { \
        atomicAdd((unsigned long long*)&g_telemetry_ptr->dbg_inv_card14_batches, 1ull); \
    } \
} while(0)
#else
#define CARD26_INV_FERMAT_INCREMENT() do {} while(0)
#define CARD26_J2A_INCREMENT() do {} while(0)
#define CARD26_SCALAR_MUL_INCREMENT() do {} while(0)
#define CARD26_INV_BATCH_INCREMENT(batch_size) do {} while(0)
#endif

// ==============================================================================
// Debug Flags
// ==============================================================================

// Enable deep CIOS carry debugging (lane-by-lane drift detector)
// Activated with: -DDEBUG_CIOS_DEEP
#ifdef DEBUG_CIOS_DEEP
#define CIOS_DEEP_DEBUG 1
#else
#define CIOS_DEEP_DEBUG 0
#endif

// Enable/disable serial prefix for Step A in fast warp mont
// KERNEL-P256-CIOS-01: Parallel path repaired (function-composition prefix scan)
#ifndef USE_SERIAL_PREFIX_A
#define USE_SERIAL_PREFIX_A 1
#endif

// CARD 7: Enable/disable conditional subtraction debug
#ifdef DEBUG_COND_SUB
#define COND_SUB_DEBUG 1
#else
#define COND_SUB_DEBUG 0
#endif

// CARD 8: Enable/disable serial conditional subtraction
// KERNEL-P256-CIOS-01: Parallel path repaired (borrow-lookahead prefix scan)
#ifndef USE_SERIAL_COND_SUB
#define USE_SERIAL_COND_SUB 1
#endif

// ==============================================================================
// P-256 Constants - Shared guard with p256_scalar_ref.cuh
// ==============================================================================
// Only define if not already defined by other headers
#ifndef P256_SCALAR_CONSTANTS_DEFINED
#define P256_SCALAR_CONSTANTS_DEFINED

// Prime P (used by both warp-coop and scalar ref)
__constant__ uint32_t P256_P_LE_CONST[8] = {
    0xFFFFFFFFu,
    0xFFFFFFFFu,
    0xFFFFFFFFu,
    0x00000000u,
    0x00000000u,
    0x00000000u,
    0x00000001u,
    0xFFFFFFFFu
};

// Alias for scalar_ref compatibility
__constant__ uint32_t P256_P[8] = {
    0xFFFFFFFFu,
    0xFFFFFFFFu,
    0xFFFFFFFFu,
    0x00000000u,
    0x00000000u,
    0x00000000u,
    0x00000001u,
    0xFFFFFFFFu
};


__constant__ uint32_t P256_R_LE_CONST[8] = {
    0x00000001, 0x00000000, 0x00000000, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0x00000000
};


__constant__ uint32_t P256_MONT_ONE[8] = {
    0x00000001, 0x00000000, 0x00000000, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0x00000000
};


__constant__ uint32_t P256_R2[8] = {
    0x00000003, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFB,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFD, 0x00000004
};


__constant__ uint32_t P256_ONE_LITERAL[8] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};



// Base point G coordinates (used by both)
__constant__ uint32_t P256_Gx_NORMAL[8] = {
    0xd898c296, 0xf4a13945, 0x2deb33a0, 0x77037d81,
    0x63a440f2, 0xf8bce6e5, 0xe12c4247, 0x6b17d1f2
};

// Alias for scalar_ref compatibility
__constant__ uint32_t P256_GX[8] = {
    0xd898c296, 0xf4a13945, 0x2deb33a0, 0x77037d81,
    0x63a440f2, 0xf8bce6e5, 0xe12c4247, 0x6b17d1f2
};

__constant__ uint32_t P256_Gy_NORMAL[8] = {
    0x37bf51f5, 0xcbb64068, 0x6b315ece, 0x2bce3357,
    0x7c0f9e16, 0x8ee7eb4a, 0xfe1a7f9b, 0x4fe342e2
};

// Alias for scalar_ref compatibility
__constant__ uint32_t P256_GY[8] = {
    0x37bf51f5, 0xcbb64068, 0x6b315ece, 0x2bce3357,
    0x7c0f9e16, 0x8ee7eb4a, 0xfe1a7f9b, 0x4fe342e2
};







__constant__ uint32_t P256_P_MINUS_2_LE[8] = {
    0xFFFFFFFDu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000001u, 0xFFFFFFFFu
};

#endif // P256_SCALAR_CONSTANTS_DEFINED

// ===================================================================
// CARD 9: Buffers for affine diagnostics
// ===================================================================
// These buffers capture the affine conversion pipeline to identify
// where ±1 errors first appear during Z-inversion and normalization.

// PRE-affine state (Jacobian coordinates)
__device__ uint32_t dbg_Z[8];
__device__ uint32_t dbg_X_before[8];
__device__ uint32_t dbg_Y_before[8];

// Z-inversion pipeline
__device__ uint32_t dbg_Zinv[8];    // Z^-1
__device__ uint32_t dbg_Zinv2[8];   // Z^-2
__device__ uint32_t dbg_Zinv3[8];   // Z^-3

// POST-affine state (affine coordinates)
__device__ uint32_t dbg_X_after[8];
__device__ uint32_t dbg_Y_after[8];

// ===================================================================
// CARD 11: Double de-Montgomery / Shuffle Diagnostics
// ===================================================================
// These buffers capture the three stages of double de-Montgomery:
// 1. After 1st de-Montgomery (R² → R domain)
// 2. After shuffle gather (reassemble full coordinates)
// 3. After 2nd de-Montgomery (R → normal domain)

__device__ uint32_t dbg_X_demont1[8];
__device__ uint32_t dbg_Y_demont1[8];

__device__ uint32_t dbg_X_shuffle[8];
__device__ uint32_t dbg_Y_shuffle[8];

__device__ uint32_t dbg_X_demont2[8];
__device__ uint32_t dbg_Y_demont2[8];







__device__ __forceinline__
uint32_t warp_mont_mul_comba(const uint32_t a[8], const uint32_t b[8], int lane_id);





__device__ __noinline__
void mont_mul_oracle_lane0(
    const uint32_t A[8],
    const uint32_t B[8],
    uint32_t OUT[8]
)
{
    
    const uint32_t P[8] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000,
        0x00000000, 0x00000000, 0x00000001, 0xFFFFFFFF
    };
    const uint32_t n0 = 0x00000001;  

    
    unsigned long long T[9] = {0};

    for (int i = 0; i < 8; ++i) {
        
        unsigned long long carry = 0;
        for (int j = 0; j < 8; ++j) {
            unsigned long long prod = (unsigned long long)A[i] * B[j];
            unsigned long long sum  = T[j] + (prod & 0xFFFFFFFFull) + carry;
            T[j] = (uint32_t)sum;
            carry = (sum >> 32) + (prod >> 32);
        }
        T[8] += carry;

        
        uint32_t m = (uint32_t)T[0] * n0;
        carry = 0;
        for (int j = 0; j < 8; ++j) {
            unsigned long long prod = (unsigned long long)m * P[j];
            unsigned long long sum  = T[j] + (prod & 0xFFFFFFFFull) + carry;
            T[j] = (uint32_t)sum;
            carry = (sum >> 32) + (prod >> 32);
        }
        T[8] += carry;

        
        for (int k = 0; k < 8; ++k) T[k] = T[k + 1];
        T[8] = 0;
    }


    // FIX (Card 4): Check T[8] for final overflow! CIOS can produce results up to 2*P.
    bool ge = (T[8] != 0);  // If T[8] != 0, definitely >= P

    if (!ge) {
        for (int i = 7; i >= 0; --i) {
            if (T[i] > P[i]) { ge = true; break; }
            if (T[i] < P[i]) { ge = false; break; }
        }
    }

    unsigned long long borrow = 0;
    for (int i = 0; i < 8; ++i) {
        unsigned long long xi = T[i];
        unsigned long long yi = ge ? P[i] : 0ull;
        unsigned long long diff = xi - yi - borrow;
        OUT[i] = (uint32_t)diff;
        borrow = (diff >> 63);
    }
}

// ==============================================================================
// Card 26.6: Lane-0-only Fermat inversion for batch Z-inverse (Card 14)
// ==============================================================================
// This is a scalar version of Fermat inversion using mont_mul_oracle_lane0.
// It's called by p256_batch_inv_block_montgomery_v2 for the single expensive
// inversion step. All Montgomery multiplications in the batch use lane0 oracle.
//
// CALLING CONVENTION:
// - ONLY lane 0 should call this function
// - Other lanes should NOT be executing this code path
// - This uses the same p-2 exponent as warp_fermat_inverse_p256
// ==============================================================================
__device__ __noinline__
void warp_fermat_inverse_p256_lane0(const uint32_t Z[8], uint32_t out[8])
{
    // P-2 constant in little-endian for Fermat's little theorem: Z^(p-2) = Z^(-1) mod p
    // Same constant as P256_P_MINUS_2_LE but inline to avoid constant memory access issues
    const uint32_t P_MINUS_2[8] = {
        0xFFFFFFFDu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000001u, 0xFFFFFFFFu
    };

    // Montgomery representation of 1 (R mod p)
    const uint32_t MONT_ONE[8] = {
        0x00000001, 0x00000000, 0x00000000, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0x00000000
    };

    // Check for zero input
    bool is_zero = true;
    for (int i = 0; i < 8; i++) {
        if (Z[i] != 0) { is_zero = false; break; }
    }
    if (is_zero) {
        for (int i = 0; i < 8; i++) out[i] = 0;
        return;
    }

    // Check for Montgomery 1 (inverse of 1 is 1)
    bool is_mont_one = true;
    for (int i = 0; i < 8; i++) {
        if (Z[i] != MONT_ONE[i]) { is_mont_one = false; break; }
    }
    if (is_mont_one) {
        for (int i = 0; i < 8; i++) out[i] = MONT_ONE[i];
        return;
    }

    // Initialize: r = 1 (Montgomery form), base = Z
    uint32_t r[8], base[8];
    for (int i = 0; i < 8; i++) {
        r[i] = MONT_ONE[i];
        base[i] = Z[i];
    }

    // Square-and-multiply ladder over 256 bits
    // Process from high bit (255) down to low bit (0)
    for (int bit = 255; bit >= 0; --bit) {
        // Square: r = r * r
        uint32_t r_sqr[8];
        mont_mul_oracle_lane0(r, r, r_sqr);
        for (int i = 0; i < 8; i++) r[i] = r_sqr[i];

        // Get bit from exponent (p-2)
        int limb = bit >> 5;
        int offset = bit & 31;
        uint32_t b = (P_MINUS_2[limb] >> offset) & 1u;

        // Conditional multiply: if bit is 1, r = r * base
        if (b) {
            uint32_t r_mul[8];
            mont_mul_oracle_lane0(r, base, r_mul);
            for (int i = 0; i < 8; i++) r[i] = r_mul[i];
        }
    }

    // Output result
    for (int i = 0; i < 8; i++) out[i] = r[i];
}



__device__ __forceinline__
void warp_mont_mul_ab_compare(
    const uint32_t A[8],
    const uint32_t B[8],
    uint32_t out_comba[8],
    uint32_t out_oracle[8],
    int lane_id
)
{
    const unsigned L8 = 0xFFu;

    
    out_comba[lane_id] = warp_mont_mul_comba(A, B, lane_id);

    
    uint32_t o_lane0[8];
    if (lane_id == 0) {
        mont_mul_oracle_lane0(A, B, o_lane0);
    }

    
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t limb_i = (lane_id == 0) ? o_lane0[i] : 0u;
        uint32_t bcast = __shfl_sync(L8, limb_i, 0);
        if (lane_id == i) {
            out_oracle[i] = bcast;
        }
    }

    

    
    
    bool eq = (out_comba[lane_id] == out_oracle[lane_id]);
    uint32_t ballot = __ballot_sync(L8, !eq);

    if (ballot && lane_id == __ffs(ballot) - 1) {
        printf("[AB DIFF] limb=%d  comba=%08x  oracle=%08x\n",
               lane_id, out_comba[lane_id], out_oracle[lane_id]);
    }
}





__device__ __forceinline__
uint32_t warp_demont_with_ab(const uint32_t a[8], int lane_id)
{

    const unsigned L8 = 0xFFu;
    uint32_t out_comba[8];
    uint32_t out_oracle[8];

    
    warp_mont_mul_ab_compare(a, P256_ONE_LITERAL, out_comba, out_oracle, lane_id);

    
    return (lane_id < 8) ? out_comba[lane_id] : 0u;



}












struct CiosOracle {
  uint32_t T[8];     
  uint32_t P[8];     
  uint32_t C_hi;     

  __device__ __forceinline__ void reset() {
    #pragma unroll
    for (int i = 0; i < 8; ++i) T[i] = 0;
    C_hi = 0;
  }

  __device__ __forceinline__ void set_modp_le(const uint32_t p_le[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) P[i] = p_le[i];
  }

  
  __device__ __forceinline__ void stepA(uint32_t a_i, const uint32_t b[8]) {
    unsigned long long carry = 0ull;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      unsigned long long s =
          (unsigned long long)T[k] +
          (unsigned long long)a_i * (unsigned long long)b[k] +
          carry;
      T[k] = (uint32_t)s;
      carry = s >> 32;
    }
    
  }

  
  
  __device__ __forceinline__ void stepB() {
    const uint32_t m = T[0];               
    unsigned long long carry = 0ull;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      unsigned long long s =
          (unsigned long long)T[k] +
          (unsigned long long)m * (unsigned long long)P[k] +
          carry;
      T[k] = (uint32_t)s;
      carry = s >> 32;
    }
    C_hi = (uint32_t)carry;                
    
  }

  
  __device__ __forceinline__ void stepC() {
    #pragma unroll
    for (int k = 0; k < 7; ++k) {
      T[k] = T[k+1];
    }
    T[7] = C_hi;
  }

  
  __device__ __forceinline__ void stepABC(uint32_t a_i, const uint32_t b[8]) {
    stepA(a_i, b);
    stepB();
    stepC();
  }
};




// Helper function to get lane ID within 8-lane group (0-7)
static __forceinline__ __device__ unsigned lane8() { return threadIdx.x & 7; }

__device__ __forceinline__ uint32_t scan_excl_8(uint32_t x, unsigned sub) {
    
    uint32_t v = x, y;
    y = __shfl_up_sync(sub, v, 1, 8);  if ((threadIdx.x & 7) >= 1) v += y;
    y = __shfl_up_sync(sub, v, 2, 8);  if ((threadIdx.x & 7) >= 2) v += y;
    y = __shfl_up_sync(sub, v, 4, 8);  if ((threadIdx.x & 7) >= 4) v += y;
    return v - x;  
}





















// ==============================================================================
// CIOS Prefix Oracle - Compute partial CIOS up to iteration i_max
// ==============================================================================
// Computes Montgomery multiplication T = (A * B * R^-1) mod P
// but only for iterations 0..i_max (not full 0..7)
// Used for per-iteration debugging to find where GPU diverges from oracle
__device__ __noinline__
void mont_mul_oracle_prefix(
    const uint32_t A[8],
    const uint32_t B[8],
    int i_max,  // Run CIOS iterations 0..i_max (inclusive)
    uint32_t T_out[8]  // Output: partial T array after iteration i_max
)
{
    CiosOracle oracle;
    oracle.reset();
    oracle.set_modp_le(P256_P_LE_CONST);

    // Run CIOS iterations 0 through i_max (inclusive)
    for (int i = 0; i <= i_max; ++i) {
        oracle.stepA(A[i], B);
        oracle.stepB();
        oracle.stepC();
    }

    // Copy partial result to output
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        T_out[k] = oracle.T[k];
    }
}

// ==============================================================================
// Warp-Cooperative Oracle Montgomery Multiply (Lane-0 Serial, Array Version)
// ==============================================================================
// Warp-level wrapper around the validated lane-0 oracle montgomery multiply.
// Uses lane 0 to compute the full multiply, then broadcasts result to all lanes.
// This is intentionally simple/safe to isolate bugs in warp-cooperative code.
//
// Args:
//   a[8], b[8]: Input operands (per-lane register arrays)
//   out[8]: Output result (per-lane register arrays)
//   lane: Thread's lane ID (0..31)
// ==============================================================================
__device__ __forceinline__
void warp_mont_mul_oracle_broadcast(
    const uint32_t a[8],
    const uint32_t b[8],
    uint32_t out[8],
    int lane
) {
    // CRITICAL: a[8] and b[8] are per-lane arrays where lane i holds limb i!
    // We need to GATHER all limbs to lane 0 first, then compute, then broadcast result

    uint32_t a_gathered[8], b_gathered[8];

    // Gather all limbs to lane 0 (each lane i contributes its limb to index i)
    for (int i = 0; i < 8; i++) {
        // Lane i has the value for position i in a[lane] and b[lane]
        uint32_t a_val = (lane == i && lane < 8) ? a[lane] : 0u;
        uint32_t b_val = (lane == i && lane < 8) ? b[lane] : 0u;

        // Broadcast from lane i to everyone
        a_val = __shfl_sync(0xFFFFFFFF, a_val, i);
        b_val = __shfl_sync(0xFFFFFFFF, b_val, i);

        a_gathered[i] = a_val;
        b_gathered[i] = b_val;
    }

    // Lane 0 does the entire multiply using validated oracle
    uint32_t tmp[8];
    if (lane == 0) {
        mont_mul_oracle_lane0(a_gathered, b_gathered, tmp);
    }
    // CRITICAL: ALL 32 lanes must participate in __syncwarp!
    __syncwarp(0xFFFFFFFF);

    // Broadcast each result limb from lane 0 to ALL lanes
    for (int i = 0; i < 8; i++) {
        uint32_t val = 0u;
        if (lane == 0) {
            val = tmp[i];  // Lane 0 has the computed limb
        }
        // ALL lanes (0-31) participate in shuffle
        val = __shfl_sync(0xFFFFFFFF, val, 0);  // Broadcast from lane 0
        out[i] = val;  // All lanes store same value
    }
    // CRITICAL: ALL 32 lanes participate
    __syncwarp(0xFFFFFFFF);
}

// ==============================================================================
// Oracle-Backed Compatibility Wrapper (Drop-In Replacement)
// ==============================================================================
// Same signature as warp_mont_mul_comba but uses oracle internally.
// This allows testing the ladder/doubles/adds without changing call sites.
// ==============================================================================
__device__ __forceinline__
uint32_t warp_mont_mul_oracle_compat(
    const uint32_t a[8],
    const uint32_t b[8],
    int lane_id
) {
    uint32_t result[8];
    warp_mont_mul_oracle_broadcast(a, b, result, lane_id);
    return (lane_id < 8) ? result[lane_id] : 0u;
}

// ==============================================================================
// Phase 3c: Group-local oracle multiply (width=8 variant)
// ==============================================================================
// Identical to warp_mont_mul_oracle_broadcast but uses width=8 shuffles
// so each 8-lane group operates independently for batch-4 dispatch.
// ==============================================================================
__device__ __forceinline__
void warp_mont_mul_oracle_broadcast_group(
    const uint32_t a[8],
    const uint32_t b[8],
    uint32_t out[8],
    int lane
) {
    // Group-local mask: only the 8 lanes in this group participate.
    // Required because inactive groups (i >= batch) skip this call entirely.
    const unsigned gmask = 0xFFu << (lane & 0x18);  // 0x18 = 24, masks to group base

    uint32_t a_gathered[8], b_gathered[8];

    // Gather: each lane contributes its limb within its 8-lane group
    for (int i = 0; i < 8; i++) {
        uint32_t a_val = ((lane & 7) == i) ? a[lane & 7] : 0u;
        uint32_t b_val = ((lane & 7) == i) ? b[lane & 7] : 0u;
        a_val = __shfl_sync(gmask, a_val, i, 8);  // width=8 + group mask
        b_val = __shfl_sync(gmask, b_val, i, 8);
        a_gathered[i] = a_val;
        b_gathered[i] = b_val;
    }

    // Group leader computes
    uint32_t tmp[8];
    if ((lane & 7) == 0) {
        mont_mul_oracle_lane0(a_gathered, b_gathered, tmp);
    }

    // Broadcast result within group
    for (int i = 0; i < 8; i++) {
        uint32_t val = 0u;
        if ((lane & 7) == 0) val = tmp[i];
        val = __shfl_sync(gmask, val, 0, 8);  // width=8 + group mask
        out[i] = val;
    }
}

__device__ __forceinline__
uint32_t warp_mont_mul_oracle_compat_group(
    const uint32_t a[8],
    const uint32_t b[8],
    int lane_id
) {
    uint32_t result[8];
    warp_mont_mul_oracle_broadcast_group(a, b, result, lane_id);
    return result[lane_id & 7];
}

// ==============================================================================
// Conditional Subtraction Helper: Lane-0 Serial, Reference-Style
// ==============================================================================
// Each warp cooperatively performs: T = T mod p
// - scratch_base: pointer to shared memory buffer (kernel owns this)
// - Tj: reference to limb value (updated in-place)
// - warp_in_block: which warp in this block (0..warps_per_block-1)
// - lane: 0..31
// ==============================================================================

// CARD 7: Simple scalar oracle for conditional subtraction (for debug comparison)
__device__ __forceinline__
void p256_cond_sub_scalar(uint32_t out[8], const uint32_t in[8], uint32_t final_carry) {
    const uint32_t *P = P256_P_LE_CONST;

    // Determine if we need to subtract: check final_carry or T >= P
    int ge = (final_carry != 0) ? 1 : 0;

    if (!ge) {
        // Compare T >= P (big-endian comparison: limb 7 down to 0)
        for (int i = 7; i >= 0; --i) {
            if (in[i] > P[i]) { ge = 1; break; }
            if (in[i] < P[i]) { ge = 0; break; }
        }
    }

    if (ge) {
        // Compute T - P with borrow
        unsigned long long borrow = 0ull;
        for (int k = 0; k < 8; ++k) {
            unsigned long long s = (unsigned long long)in[k]
                                 - (unsigned long long)P[k]
                                 - borrow;
            out[k] = (uint32_t)s;
            borrow = (s >> 63) & 1ull;
        }
    } else {
        // No subtraction needed, copy input to output
        for (int k = 0; k < 8; ++k) {
            out[k] = in[k];
        }
    }
}

// KERNEL-P256-CIOS-01: Truly warp-parallel conditional subtraction.
// Uses shuffles only — no shared memory needed.
// Part 1: warp-uniform comparison (T >= P or final_carry).
// Part 2: borrow-lookahead parallel prefix scan for T - P.
__device__ __forceinline__
void p256_cond_sub_warp_parallel(uint32_t &Tj, int lane_id, uint32_t final_carry) {
    const unsigned FULL = 0xFFFFFFFFu;

    // ====================================================================
    // Part 1: Warp-uniform comparison — determine if T >= P or final_carry
    // ====================================================================
    uint32_t ge;
    {
        uint32_t eq_acc = 1u, gt_acc = 0u;
        #pragma unroll
        for (int k = 7; k >= 0; --k) {
            uint32_t tk = __shfl_sync(FULL, Tj, k, 8);
            uint32_t pk = P256_P_LE_CONST[k];
            uint32_t gt_k = (tk > pk) ? 1u : 0u;
            uint32_t eq_k = (tk == pk) ? 1u : 0u;
            gt_acc = gt_acc | (eq_acc & gt_k);
            eq_acc = eq_acc & eq_k;
        }
        ge = gt_acc | eq_acc;
    }
    ge = ge | (final_carry != 0u ? 1u : 0u);

    if (!ge) return;  // T < P and no carry — nothing to do

    // ====================================================================
    // Part 2: Warp-parallel subtraction T = T - P using borrow-lookahead
    // ====================================================================
    {
        uint32_t my_p = P256_P_LE_CONST[lane_id & 7];

        // Generate/propagate flags for borrow chain
        uint32_t g = (Tj < my_p) ? 1u : 0u;      // borrow generated
        uint32_t p = (Tj == my_p) ? 1u : 0u;      // borrow propagated

        // Inclusive parallel prefix scan on (generate, propagate) pairs
        #pragma unroll
        for (int ofs = 1; ofs < 8; ofs <<= 1) {
            uint32_t g_lo = __shfl_up_sync(FULL, g, ofs, 8);
            uint32_t p_lo = __shfl_up_sync(FULL, p, ofs, 8);
            if ((lane_id & 7) >= ofs) {
                g = g | (p & g_lo);
                p = p & p_lo;
            }
        }

        // Convert inclusive scan to exclusive (borrow INTO lane k)
        uint32_t borrow_in = __shfl_up_sync(FULL, g, 1, 8);
        if ((lane_id & 7) == 0) borrow_in = 0u;

        // Final subtraction: T[k] - P[k] - borrow_in
        uint64_t diff = (uint64_t)Tj - (uint64_t)my_p - (uint64_t)borrow_in;
        Tj = (uint32_t)diff;
    }
}

// Legacy shared-memory version kept for reference (no longer called).
__device__ __forceinline__
void p256_cond_sub_warp(uint32_t* scratch_base, uint32_t &Tj, int warp_in_block, int lane, uint32_t final_carry) {
    const int li        = lane & 7;
    const int warp_base = warp_in_block * 8;
    if (lane < 8) { scratch_base[warp_base + li] = Tj; }
    __syncwarp(0xFFFFFFFF);
    if (lane == 0) {
        uint32_t t[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) t[i] = scratch_base[warp_base + i];
        p256_cond_sub_scalar(t, t, final_carry);
        #pragma unroll
        for (int i = 0; i < 8; ++i) scratch_base[warp_base + i] = t[i];
    }
    __syncwarp(0xFFFFFFFF);
    if (lane < 8) { Tj = scratch_base[warp_base + li]; }
}

// ==============================================================================
// 2-TIER MONTGOMERY MULTIPLICATION SYSTEM
// ==============================================================================
// Free/Default Tier: warp_mont_mul_comba_safe (scalar oracle, 100% correct)
// Pro/Fast Tier:     warp_mont_mul_comba_fast (warp CIOS, 8-32x faster)
//
// Build flags:
//   Default:                    Safe tier only (scalar oracle)
//   -DP256_USE_FAST_WARP_MONT:  Fast tier (warp CIOS parallelism)
//   -DDEBUG_WARP_MONT_MISMATCH: Dev mode (compare fast vs oracle)
// ==============================================================================

// ------------------------------------------------------------------------------
// SAFE TIER (FREE/DEFAULT): Scalar Oracle Implementation
// ------------------------------------------------------------------------------
// Properties:
//   - 100% mathematically correct (uses validated CPU CIOS)
//   - Stable, no crashes, no mismatches
//   - Lane 0 computes, broadcasts to all lanes
//   - Performance: Adequate for correctness-first workloads
// ------------------------------------------------------------------------------
__device__ __forceinline__
uint32_t warp_mont_mul_comba_safe(
    const uint32_t a[8],
    const uint32_t b[8],
    int lane_id
)
{
    const unsigned L8 = 0xFFu;  // lanes 0..7

    // Local buffer for oracle result (lane 0 only)
    uint32_t oracle_result[8];

    // Lane 0 runs the scalar oracle
    if (lane_id == 0) {
        mont_mul_oracle_lane0(a, b, oracle_result);
    }

    // Broadcast each limb from lane 0 to all lanes via warp shuffle
    // (no shared memory needed — avoids dynamic shmem allocation requirement)
    uint32_t Tj = 0;
    if (lane_id < 8) {
        uint32_t limb = (lane_id == 0) ? oracle_result[lane_id] : 0u;
        // Each lane needs its own limb: lane 0 broadcasts oracle_result[i] for each i
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t val = (lane_id == 0) ? oracle_result[i] : 0u;
            val = __shfl_sync(L8, val, 0);
            if (lane_id == i) Tj = val;
        }
    }

    return Tj;
}

// ------------------------------------------------------------------------------
// FAST TIER (PRO/PAID): Warp CIOS Implementation
// ------------------------------------------------------------------------------
// Properties:
//   - 8-32x faster than safe tier (full warp parallelism)
//   - All 8 lanes process limbs simultaneously
//   - Requires debugging to fix carry propagation bug at iteration i=6
//   - When ready: restore from p256_field_warp_coop.cuh.BEFORE_CARD1_FIX
// ------------------------------------------------------------------------------
__device__ __forceinline__
uint32_t warp_mont_mul_comba_fast(
    const uint32_t a[8],
    const uint32_t b[8],
    int lane_id
)
{
    // Phase 4a.1: Removed extern __shared__ p256_shared[] and warp_in_block.
    // Both were declared but never referenced in any active code path
    // (USE_SERIAL_COND_SUB=0 makes the shared-memory consumer dead code).
    // The extern __shared__ declaration conflicted with the kernel's static
    // shared arrays (s_Z_workspace etc.), causing deadlocks when this function
    // was called from the batch Z-inversion path.

    // KERNEL-P256-JACOBIAN-01: Use hardcoded full-warp mask instead of __activemask().
    // __activemask() is fragile to lane desynchronization — if ANY prior operation
    // caused divergence, it returns a partial mask, cascading into deadlock.
    // In the persistent kernel, all 32 warp lanes are always active.
    const unsigned full = 0xFFFFFFFFu;
    // KERNEL-P256-JACOBIAN-01: Use full-warp mask for ALL shuffles.
    // Phase 3: 4-instance mode — each group of 8 lanes runs independently via width=8 shuffles.
    const unsigned sub = full;  // 0xFFFFFFFF — all lanes participate


#ifdef DEBUG_CIOS_ORACLE
    do { unsigned mask = __activemask(); __syncwarp(mask); if ((threadIdx.x % 32) == 0) { printf("[HEARTBEAT-OK] %s (mask=%08x)\n", "MUL-ENTRY", mask); } } while(0);
    do { if ((threadIdx.x % 32) == 0 && sub != 0xFFu) { printf("[MASK-ERROR] %s: expected 0xFF, got %08x\n", "MUL-ENTRY-MASK", sub); } } while(0);
#endif


    uint32_t a_lane = a[lane_id & 7];
    uint32_t b_lane = b[lane_id & 7];


#ifdef DEBUG_CIOS_ORACLE
    if (lane_id < 8 && b_lane == 1 && lane_id == 1) {
        bool is_demont = (b[0] == 1);
        for (int k = 1; k < 8 && is_demont; k++) {
            if (b[k] != 0) is_demont = false;
        }
        if (is_demont) {
            printf("[DEMONT-INPUT lane1] a[1]=%08x b[1]=%08x\n", a[1], b[1]);
        }
    }
#endif


    uint32_t Tj = 0;
    uint32_t carry_out = 0;
    uint32_t C_hi = 0;

    uint32_t my_b = b_lane;
    uint32_t my_p = P256_P_LE_CONST[lane_id & 7];



    CiosOracle oracle;
    if ((threadIdx.x & 31) == 0) {
        oracle.reset();
        oracle.set_modp_le(P256_P_LE_CONST);
    }


    uint32_t b_arr[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        b_arr[k] = __shfl_sync(sub, my_b, k, 8);
    }






    #pragma unroll
    for (int i = 0; i < 8; ++i) {









#ifdef DEBUG_CIOS_ORACLE
        if (i == 0) {
            do { unsigned mask = __activemask(); __syncwarp(mask); if ((threadIdx.x % 32) == 0) { printf("[HEARTBEAT-OK] %s (mask=%08x)\n", "CIOS-A-0", mask); } } while(0);
        }
#endif

        uint32_t a_i = __shfl_sync(sub, a_lane, i, 8);



        uint32_t lo = a_i * my_b;
        uint32_t hi = __umulhi(a_i, my_b);


        uint64_t s = (uint64_t)Tj + (uint64_t)lo;
        uint32_t tl = (uint32_t)s;
        uint32_t c32 = (uint32_t)(s >> 32);


        uint32_t hi_local = hi + c32;

        // ================================================================
        // CARD 6: Step A - FULL SERIAL ORACLE MATCH (Correctness-First)
        // ================================================================
        // Lane 0 computes entire Step A serially to exactly match oracle

#if USE_SERIAL_PREFIX_A
        // ---- FULL SERIAL VERSION (exact oracle match) ----

        // 1. Gather all inputs — shuffles OUTSIDE guard (all 32 lanes participate)
        uint32_t all_Tj[8], all_lo[8], all_hi[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            all_Tj[k] = __shfl_sync(sub, Tj, k, 8);
            all_lo[k] = __shfl_sync(sub, lo, k, 8);
            all_hi[k] = __shfl_sync(sub, hi, k, 8);
        }

        // 2. Group-leader lane computes Step A exactly like oracle
        uint32_t result_arr[8];
        if ((lane_id & 7) == 0) {
            unsigned long long carry = 0ull;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                // Reconstruct 64-bit product from lo/hi
                unsigned long long prod = ((unsigned long long)all_hi[k] << 32) | all_lo[k];

                // Oracle's computation: s = T[k] + a[i]*b[k] + carry
                unsigned long long s = (unsigned long long)all_Tj[k] + prod + carry;

                result_arr[k] = (uint32_t)s;
                carry = s >> 32;
            }
            // Carry out of lane 7 is discarded (matches oracle)
        }

        // 3. Broadcast result to each lane
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t res_k = 0u;
            if ((lane_id & 7) == 0) {
                res_k = result_arr[k];
            }
            res_k = __shfl_sync(sub, res_k, 0, 8);
            if ((lane_id & 7) == k) {
                Tj = res_k;
            }
        }

#else
        // ---- KERNEL-P256-CIOS-01: FUNCTION-COMPOSITION PARALLEL PREFIX SCAN ----
        // Each lane k defines a carry function f_k : carry_in -> carry_out
        //   f_k(c) = hi_local[k] + ( (tl[k] != 0) && (c >= (0u - tl[k])) ? 1 : 0 )
        // This is a step function: (base, can_propagate, threshold).
        // Composition of step functions is a step function.
        // 3-stage Kogge-Stone inclusive prefix on composed functions.

        // Initialize per-lane function representation
        uint32_t fn_base  = hi_local;                      // output when input < threshold
        uint32_t fn_cprop = (tl != 0u) ? 1u : 0u;         // can this lane propagate?
        uint32_t fn_thr   = (uint32_t)(0u - tl);           // threshold (valid only when fn_cprop)

        // 3-stage Kogge-Stone inclusive prefix on composed functions
        #pragma unroll
        for (int ofs = 1; ofs < 8; ofs <<= 1) {
            // Shuffle upstream function from lane (lane_id - ofs)
            uint32_t up_base  = __shfl_up_sync(sub, fn_base,  ofs, 8);
            uint32_t up_cprop = __shfl_up_sync(sub, fn_cprop, ofs, 8);
            uint32_t up_thr   = __shfl_up_sync(sub, fn_thr,   ofs, 8);

            if ((lane_id & 7) >= ofs) {
                // Compose: self( upstream( c ) )
                // Does upstream's "lo" output (up_base) trigger self?
                uint32_t lo_triggers = (fn_cprop && (up_base >= fn_thr)) ? 1u : 0u;
                // Does upstream's "hi" output (up_base + 1) trigger self?
                // Use uint64_t to handle up_base+1 overflow past 32 bits
                uint32_t hi_triggers = (fn_cprop && ((uint64_t)up_base + 1u >= (uint64_t)fn_thr)) ? 1u : 0u;

                // New base = self.base + lo_triggers
                uint32_t new_base = fn_base + lo_triggers;

                if (lo_triggers == hi_triggers) {
                    // Self is constant regardless of upstream's input
                    fn_base  = new_base;
                    fn_cprop = 0u;
                } else {
                    // Self output differs — inherit upstream's gate
                    fn_base  = new_base;
                    fn_cprop = up_cprop;
                    fn_thr   = up_thr;
                }
            }
        }

        // After inclusive prefix, fn_base[k] = composed_f_{0..k}(0),
        // i.e. carry OUT of lane k when initial carry is 0.
        // carry_into[0] = 0, carry_into[k] = fn_base[k-1] for k >= 1.
        // KERNEL-P256-JACOBIAN-01: shuffle OUTSIDE guard — all 32 lanes must participate
        uint32_t carry_into_k = 0u;
        uint32_t shifted_fn = __shfl_up_sync(sub, fn_base, 1, 8);
        if ((lane_id & 7) > 0) {
            carry_into_k = shifted_fn;
        }

        // Apply carry to tl to get final Tj
        Tj = (uint32_t)((uint64_t)tl + (uint64_t)carry_into_k);

#endif  // USE_SERIAL_PREFIX_A



        uint32_t a_i_all = __shfl_sync(sub, a_i, 0, 8);
        if ((threadIdx.x & 31) == 0) {
            oracle.stepA(a_i_all, b_arr);
        }



        uint32_t T_after_A[8];
        uint32_t v_a[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v_a[k] = __shfl_sync(sub, Tj, k, 8);
            T_after_A[k] = v_a[k];
        }
#ifdef DEBUG_CIOS_ORACLE
        if (i == 0 && (threadIdx.x & 31) == 0) {
            printf("[A END i=0] COMBA: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                   v_a[0], v_a[1], v_a[2], v_a[3], v_a[4], v_a[5], v_a[6], v_a[7]);
            printf("[A END i=0] ORA  : %08x %08x %08x %08x %08x %08x %08x %08x\n",
                   oracle.T[0], oracle.T[1], oracle.T[2], oracle.T[3],
                   oracle.T[4], oracle.T[5], oracle.T[6], oracle.T[7]);
        }
#endif

        // ================================================================
        // CARD 5: Deep Debug - Check After Step A
        // ================================================================
        #ifdef DEBUG_CIOS_DEEP
        {
            // Gather GPU T after Step A
            uint32_t T_gpu_A[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                T_gpu_A[k] = __shfl_sync(sub, Tj, k, 8);
            }

            if (lane_id == 0) {
                // Use inline oracle.T[] (already updated by oracle.stepA above)
                // Check for mismatch
                bool mismatch = false;
                for (int k = 0; k < 8; ++k) {
                    if (T_gpu_A[k] != oracle.T[k]) {
                        mismatch = true;
                        break;
                    }
                }

                if (mismatch) {
                    printf("[CIOS-DEEP] i=%d AFTER STEP A:\n", i);
                    printf("  GPU:    ");
                    for (int k = 7; k >= 0; --k) printf("%08x ", T_gpu_A[k]);
                    printf("\n  Oracle: ");
                    for (int k = 7; k >= 0; --k) printf("%08x ", oracle.T[k]);
                    printf("\n  Diffs:  ");
                    for (int k = 7; k >= 0; --k) {
                        int diff = (int)T_gpu_A[k] - (int)oracle.T[k];
                        printf("%+d ", diff);
                    }
                    printf("\n");
                }
            }
            __syncwarp(sub);
        }
        #endif






        uint32_t m  = __shfl_sync(sub, Tj, 0, 8);
        uint32_t pj = my_p;


        lo = m * pj;
        tl = Tj + lo;
        uint32_t c0 = (tl < Tj);


        uint32_t carry_lo = scan_excl_8(c0, sub);
        uint32_t tl2      = tl + carry_lo;
        uint32_t c1       = (tl2 < tl);


        hi = __umulhi(m, pj);
        hi_local = hi + c1;


        // ---- Step B: Parallel CIOS (function-composition prefix scan) ----
        // Same pattern as Step A parallel prefix (lines 1062-1118).
        // tl2 and hi_local already computed above (lines 1193-1208):
        //   tl2 = Tj + m*P[k]_lo + lo-carry-scan
        //   hi_local = m*P[k]_hi + re-carry from lo addition
        // Now propagate hi carries across lanes using Kogge-Stone.
        {
            // Function-composition prefix: carry_out_k = f_k(carry_in_k)
            // where f_k is a step function parameterized by (base, cprop, threshold)
            uint32_t fn_base_b  = hi_local;
            uint32_t fn_cprop_b = (tl2 != 0u) ? 1u : 0u;
            uint32_t fn_thr_b   = (uint32_t)(0u - tl2);

            #pragma unroll
            for (int ofs = 1; ofs < 8; ofs <<= 1) {
                uint32_t up_base  = __shfl_up_sync(sub, fn_base_b,  ofs, 8);
                uint32_t up_cprop = __shfl_up_sync(sub, fn_cprop_b, ofs, 8);
                uint32_t up_thr   = __shfl_up_sync(sub, fn_thr_b,   ofs, 8);

                if ((lane_id & 7) >= ofs) {
                    uint32_t lo_triggers = (fn_cprop_b && (up_base >= fn_thr_b)) ? 1u : 0u;
                    uint32_t hi_triggers = (fn_cprop_b && ((uint64_t)up_base + 1u >= (uint64_t)fn_thr_b)) ? 1u : 0u;

                    uint32_t new_base = fn_base_b + lo_triggers;

                    if (lo_triggers == hi_triggers) {
                        fn_base_b  = new_base;
                        fn_cprop_b = 0u;
                    } else {
                        fn_base_b  = new_base;
                        fn_cprop_b = up_cprop;
                        fn_thr_b   = up_thr;
                    }
                }
            }

            // carry_into[0] = 0, carry_into[k] = fn_base_b[k-1] for k >= 1
            uint32_t carry_into_b = 0u;
            uint32_t shifted_b = __shfl_up_sync(sub, fn_base_b, 1, 8);
            if ((lane_id & 7) > 0) {
                carry_into_b = shifted_b;
            }

            // Apply carry to get final Step B result
            Tj = (uint32_t)((uint64_t)tl2 + (uint64_t)carry_into_b);

            // C_hi = carry out of lane 7 = fn_base_b[7]
            C_hi = __shfl_sync(sub, fn_base_b, 7, 8);
            carry_out = ((lane_id & 7) == 7) ? C_hi : 0;
        }



        if ((threadIdx.x & 31) == 0) {
            oracle.stepB();
        }



        uint32_t v_b[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v_b[k] = __shfl_sync(sub, Tj, k, 8);
        }
#ifdef DEBUG_CIOS_ORACLE
        if (i == 0 && (threadIdx.x & 31) == 0) {
            printf("[B END i=0] COMBA: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                   v_b[0], v_b[1], v_b[2], v_b[3], v_b[4], v_b[5], v_b[6], v_b[7]);
            printf("[B END i=0] ORA  : %08x %08x %08x %08x %08x %08x %08x %08x\n",
                   oracle.T[0], oracle.T[1], oracle.T[2], oracle.T[3],
                   oracle.T[4], oracle.T[5], oracle.T[6], oracle.T[7]);
        }

        uint32_t T0_after_B = __shfl_sync(sub, Tj, 0, 8);
        if (i == 0 && (threadIdx.x & 31) == 0) {
            if (T0_after_B == 0) {
                printf("[INV-PASS i=0] B END: T[0]==0 (CIOS invariant)\n");
            } else {
                printf("[INV-FAIL i=0] B END: T[0]=%08x (expected 0!)\n", T0_after_B);
            }
        }

        uint32_t gpu_B[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            gpu_B[k] = __shfl_sync(sub, Tj, k, 8);
        }

        if (i == 0 && (threadIdx.x & 31) == 0) {
            uint32_t T_ser[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) T_ser[k] = T_after_A[k];

            const uint32_t m = T_ser[0];
            unsigned long long carry = 0ull;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                unsigned long long s =
                    (unsigned long long)T_ser[k] +
                    (unsigned long long)m * (unsigned long long)P256_P_LE_CONST[k] +
                    carry;
                T_ser[k] = (uint32_t)s;
                carry = s >> 32;
            }

            printf("[S3 SERIAL i=0] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                   T_ser[0], T_ser[1], T_ser[2], T_ser[3],
                   T_ser[4], T_ser[5], T_ser[6], T_ser[7]);

            bool all_match = true;
            for (int k = 0; k < 8; k++) {
                if (T_ser[k] != gpu_B[k]) {
                    printf("[S3 MISMATCH] Lane %d: serial=%08x gpu=%08x\n", k, T_ser[k], gpu_B[k]);
                    all_match = false;
                }
            }
            if (all_match) {
                printf("[S3 PASS] Serial Step B matches GPU Step B\n");
            } else {
                printf("[S3 FAIL] Serial Step B differs from GPU Step B\n");
            }
        }
#endif

        // ================================================================
        // CARD 5: Deep Debug - Check After Step B
        // ================================================================
        #ifdef DEBUG_CIOS_DEEP
        {
            // Gather GPU T after Step B
            uint32_t T_gpu_B[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                T_gpu_B[k] = __shfl_sync(sub, Tj, k, 8);
            }

            if (lane_id == 0) {
                // Use inline oracle.T[] (already updated by oracle.stepB above)
                // Check for mismatch
                bool mismatch = false;
                for (int k = 0; k < 8; ++k) {
                    if (T_gpu_B[k] != oracle.T[k]) {
                        mismatch = true;
                        break;
                    }
                }

                if (mismatch) {
                    printf("[CIOS-DEEP] i=%d AFTER STEP B:\n", i);
                    printf("  GPU:    ");
                    for (int k = 7; k >= 0; --k) printf("%08x ", T_gpu_B[k]);
                    printf("\n  Oracle: ");
                    for (int k = 7; k >= 0; --k) printf("%08x ", oracle.T[k]);
                    printf("\n  Diffs:  ");
                    for (int k = 7; k >= 0; --k) {
                        int diff = (int)T_gpu_B[k] - (int)oracle.T[k];
                        printf("%+d ", diff);
                    }
                    printf("\n");
                }
            }
            __syncwarp(sub);
        }
        #endif









        C_hi = __shfl_sync(sub, carry_out, 7, 8);  // FIX: Don't redeclare, update outer variable


        uint32_t next = __shfl_down_sync(sub, Tj, 1, 8);


        uint32_t Tj_new = ((lane_id & 7) < 7) ? next : C_hi;


        Tj = Tj_new;

        carry_out = 0;



        if ((threadIdx.x & 31) == 0) {
            oracle.stepC();
        }

        // CARD B3 PER-ITERATION DIAGNOSTIC
        #ifdef DEBUG_CIOS_ITERATION
        {
            // Gather GPU T using __shfl_sync (no shared memory needed)
            uint32_t T_gpu[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                T_gpu[k] = __shfl_sync(0xFFFFFFFF, Tj, k);
            }

            if (lane_id == 0) {
                uint32_t T_oracle[8];
                mont_mul_oracle_prefix(a, b, i, T_oracle);

                bool mismatch = false;
                for (int k = 0; k < 8; ++k) {
                    if (T_gpu[k] != T_oracle[k]) {
                        mismatch = true;
                        break;
                    }
                }

                if (mismatch) {
                    printf("[CIOS-PREFIX MISMATCH] i=%d\n", i);
                    printf("  GPU:    ");
                    for (int k = 7; k >= 0; --k) printf("%08x ", T_gpu[k]);
                    printf("\n  Oracle: ");
                    for (int k = 7; k >= 0; --k) printf("%08x ", T_oracle[k]);
                    printf("\n  Diffs:  ");
                    for (int k = 7; k >= 0; --k) {
                        int diff = (int)T_gpu[k] - (int)T_oracle[k];
                        printf("%+d ", diff);
                    }
                    printf("\n");
                }
            }
            __syncwarp(0xFFFFFFFF);  // Card 26.2 FIX: Full mask
        }
        #endif


        uint32_t v_c[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v_c[k] = __shfl_sync(sub, Tj, k, 8);
        }
#ifdef DEBUG_CIOS_ORACLE
        if (i == 0 && (threadIdx.x & 31) == 0) {
            printf("[C END i=0] COMBA: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                   v_c[0], v_c[1], v_c[2], v_c[3], v_c[4], v_c[5], v_c[6], v_c[7]);
            printf("[C END i=0] ORA  : %08x %08x %08x %08x %08x %08x %08x %08x\n",
                   oracle.T[0], oracle.T[1], oracle.T[2], oracle.T[3],
                   oracle.T[4], oracle.T[5], oracle.T[6], oracle.T[7]);
        }
#endif

        // ================================================================
        // CARD 5: Deep Debug - Check After Step C (End of Iteration)
        // ================================================================
        #ifdef DEBUG_CIOS_DEEP
        {
            // Gather GPU T after Step C (complete iteration)
            uint32_t T_gpu_C[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                T_gpu_C[k] = __shfl_sync(sub, Tj, k, 8);
            }

            if (lane_id == 0) {
                // Use inline oracle.T[] (already updated by oracle.stepC above)
                // Check for mismatch
                bool mismatch = false;
                for (int k = 0; k < 8; ++k) {
                    if (T_gpu_C[k] != oracle.T[k]) {
                        mismatch = true;
                        break;
                    }
                }

                if (mismatch) {
                    printf("[CIOS-DEEP] i=%d AFTER STEP C (END OF ITERATION):\n", i);
                    printf("  GPU:    ");
                    for (int k = 7; k >= 0; --k) printf("%08x ", T_gpu_C[k]);
                    printf("\n  Oracle: ");
                    for (int k = 7; k >= 0; --k) printf("%08x ", oracle.T[k]);
                    printf("\n  Diffs:  ");
                    for (int k = 7; k >= 0; --k) {
                        int diff = (int)T_gpu_C[k] - (int)oracle.T[k];
                        printf("%+d ", diff);
                    }
                    printf("\n");
                }
            }
            __syncwarp(sub);
        }
        #endif






    }


#ifdef DEBUG_CIOS_ORACLE
    do { unsigned mask = __activemask(); __syncwarp(mask); if ((threadIdx.x % 32) == 0) { printf("[HEARTBEAT-OK] %s (mask=%08x)\n", "CIOS-LOOP-DONE", mask); } } while(0);
#endif



















#ifdef DEBUG_CIOS_ORACLE
    do { unsigned mask = __activemask(); __syncwarp(mask); if ((threadIdx.x % 32) == 0) { printf("[HEARTBEAT-OK] %s (mask=%08x)\n", "BEFORE-RETURN", mask); } } while(0);
#endif

    // Shared constant for shuffle masks
    const unsigned L8 = 0xFFu;

#ifdef DEBUG_CIOS_ORACLE
    // CARD B3 DIAGNOSTIC: Check if +2 bug exists BEFORE conditional subtraction
    {
        uint32_t Tj_before_sub = Tj;
        uint32_t oracle_before[8];

        if (lane_id == 0) {
            mont_mul_oracle_lane0(a, b, oracle_before);
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t oracle_before_limb = (lane_id == 0) ? oracle_before[i] : 0u;
            oracle_before_limb = __shfl_sync(L8, oracle_before_limb, 0);

            if (lane_id == i && lane_id < 8) {
                if (Tj_before_sub != oracle_before_limb) {
                    printf("[PRE-SUB MISMATCH] lane=%d comba=%08x oracle=%08x diff=%d\n",
                           lane_id, Tj_before_sub, oracle_before_limb,
                           (int)Tj_before_sub - (int)oracle_before_limb);
                }
            }
        }
    }
#endif

    // ======================================================================
    // Conditional Subtraction (KERNEL-P256-CIOS-01)
    // ======================================================================
    // FIX (Card 4): Pass C_hi as final_carry (t[16] overflow bit)
#if USE_SERIAL_COND_SUB
    // ---- CARD 8: SERIAL VERSION (exact scalar oracle match) ----
    {
        uint32_t all_T[8];
        {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                all_T[k] = __shfl_sync(0xFFFFFFFF, Tj, k, 8);
            }
        }
        uint32_t result_arr[8];
        if ((lane_id & 7) == 0) {
            p256_cond_sub_scalar(result_arr, all_T, C_hi);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t res_k = 0u;
            if ((lane_id & 7) == 0) { res_k = result_arr[k]; }
            res_k = __shfl_sync(0xFFFFFFFF, res_k, 0, 8);
            if ((lane_id & 7) == k) { Tj = res_k; }
        }
    }
#else
    // ---- CIOS-01: WARP-PARALLEL (borrow-lookahead, no shared memory) ----
    p256_cond_sub_warp_parallel(Tj, lane_id, C_hi);
#endif

#ifdef DEBUG_CIOS_ORACLE
    do { unsigned mask = __activemask(); __syncwarp(mask); if ((threadIdx.x % 32) == 0) { printf("[HEARTBEAT-OK] %s (mask=%08x)\n", "COND-SUB-DONE", mask); } } while(0);

    // POST-SUB oracle check
    {
        uint32_t oracle_result[8];
        if (lane_id == 0) {
            mont_mul_oracle_lane0(a, b, oracle_result);
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t oracle_limb = (lane_id == 0) ? oracle_result[i] : 0u;
            oracle_limb = __shfl_sync(L8, oracle_limb, 0);

            if (lane_id == i && lane_id < 8) {
                if (Tj != oracle_limb) {
                    printf("[A/B MISMATCH] Final: lane=%d comba=%08x oracle=%08x\n",
                           lane_id, Tj, oracle_limb);
                    printf("[A/B INPUT] a[%d]=%08x b[%d]=%08x\n", lane_id, a[lane_id], lane_id, b[lane_id]);
                }
            }
        }
    }
#endif

    // ======================================================================
    // CARD 3: Debug Comparison - Fast Tier vs Oracle (Non-Fatal)
    // ======================================================================
    // Enabled with: -DDEBUG_WARP_MONT_MISMATCH
    // Purpose: Identify mismatches between fast tier and oracle without crashing
    #ifdef DEBUG_WARP_MONT_MISMATCH
    {
        // Run oracle for comparison
        uint32_t oracle_result_final[8];
        if (lane_id == 0) {
            mont_mul_oracle_lane0(a, b, oracle_result_final);
        }

        // Broadcast oracle limbs to all lanes
        const unsigned L8_final = 0xFF;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t oracle_limb = (lane_id == 0) ? oracle_result_final[i] : 0u;
            oracle_limb = __shfl_sync(L8_final, oracle_limb, 0);

            // Each lane checks its own limb (non-fatal)
            if (lane_id == i && lane_id < 8) {
                if (Tj != oracle_limb) {
                    printf("[FAST-ORACLE MISMATCH] lane=%d fast=%08x oracle=%08x diff=%d\n",
                           lane_id, Tj, oracle_limb,
                           (int)Tj - (int)oracle_limb);
                }
            }
        }
    }
    #endif

    return Tj;
}

// ------------------------------------------------------------------------------
// PUBLIC API: Build-Time Tier Selection
// ------------------------------------------------------------------------------
// This wrapper preserves ABI/API compatibility.
// All existing call sites continue to work without changes.
//
// Tier selection via compile-time flags:
//   - Default (no flags):         Safe tier (scalar oracle)
//   - -DP256_USE_FAST_WARP_MONT:  Fast tier (warp CIOS)
// ------------------------------------------------------------------------------
__device__ __forceinline__
uint32_t warp_mont_mul_comba(
    const uint32_t a[8],
    const uint32_t b[8],
    int lane_id
)
{
#ifdef P256_USE_FAST_WARP_MONT
    return warp_mont_mul_comba_fast(a, b, lane_id);
#else
    return warp_mont_mul_comba_safe(a, b, lane_id);
#endif
}








__device__ __forceinline__
uint32_t warp_add_mod_p_u32(uint32_t a_j, uint32_t b_j, int lane_id)
{
    // KERNEL-P256-JACOBIAN-01: Hardcoded full-warp mask — see warp_mont_mul_comba_fast comment.
    const unsigned full = 0xFFFFFFFFu;
    const unsigned sub = full;  // KERNEL-P256-JACOBIAN-01: full-warp mask for all shuffles

    uint32_t out = 0;
    uint32_t carry = 0;

    {
        uint64_t s = (uint64_t)a_j + (uint64_t)b_j;
        out = (uint32_t)s;
        carry = (uint32_t)(s >> 32);
    }


    for (int ofs = 1; ofs < 8; ofs <<= 1) {
        uint32_t c_up = __shfl_up_sync(sub, carry, ofs, 8);
        if ((lane_id & 7) >= ofs) {
            uint64_t s2 = (uint64_t)out + (uint64_t)c_up;
            out = (uint32_t)s2;
            carry = (uint32_t)(s2 >> 32);
        }
    }


    uint32_t ge;
    {
        uint32_t eq_acc = 1u, gt_acc = 0u;
        #pragma unroll
        for (int k = 7; k >= 0; --k) {
            uint32_t tk = __shfl_sync(sub, out, k, 8);
            uint32_t pk = __shfl_sync(sub, P256_P_LE_CONST[lane_id & 7], k, 8);
            uint32_t gt_k = (tk > pk) ? 1u : 0u;
            uint32_t eq_k = (tk == pk) ? 1u : 0u;
            gt_acc = gt_acc | (eq_acc & gt_k);
            eq_acc = eq_acc & eq_k;
        }
        ge = gt_acc | eq_acc;
    }

    if (ge) {

        uint32_t borrow = 0;
        uint64_t d = (uint64_t)out - (uint64_t)P256_P_LE_CONST[lane_id & 7] - (uint64_t)borrow;
        uint32_t r = (uint32_t)d;
        borrow = (uint32_t)(d >> 63);
        for (int ofs = 1; ofs < 8; ofs <<= 1) {
            uint32_t nb = __shfl_up_sync(sub, borrow, ofs, 8);
            if ((lane_id & 7) >= ofs) {
                uint64_t d2 = (uint64_t)r - (uint64_t)nb;
                r = (uint32_t)d2;
                borrow = (uint32_t)(d2 >> 63);
            }
        }
        out = r;
    }
    return out;
}







__device__ __forceinline__
uint32_t warp_sub_mod_p_u32(uint32_t a_j, uint32_t b_j, int lane_id)
{
    // KERNEL-P256-JACOBIAN-01: Hardcoded full-warp mask — see warp_mont_mul_comba_fast comment.
    const unsigned full = 0xFFFFFFFFu;
    const unsigned sub = full;  // KERNEL-P256-JACOBIAN-01: full-warp mask for all shuffles

    uint32_t out = 0;
    uint32_t borrow = 0;

    {
        uint64_t d = (uint64_t)a_j - (uint64_t)b_j;
        out = (uint32_t)d;
        borrow = (uint32_t)(d >> 63);
    }


    for (int ofs = 1; ofs < 8; ofs <<= 1) {
        uint32_t nb = __shfl_up_sync(sub, borrow, ofs, 8);
        if ((lane_id & 7) >= ofs) {
            uint64_t d2 = (uint64_t)out - (uint64_t)nb;
            out = (uint32_t)d2;
            borrow = (uint32_t)(d2 >> 63);
        }
    }


    uint32_t any_borrow = __shfl_sync(sub, borrow, 7, 8);
    if (any_borrow) {
        uint64_t s = (uint64_t)out + (uint64_t)P256_P_LE_CONST[lane_id & 7];
        out = (uint32_t)s;
        uint32_t carry = (uint32_t)(s >> 32);
        for (int ofs = 1; ofs < 8; ofs <<= 1) {
            uint32_t c_up = __shfl_up_sync(sub, carry, ofs, 8);
            if ((lane_id & 7) >= ofs) {
                uint64_t s2 = (uint64_t)out + (uint64_t)c_up;
                out = (uint32_t)s2;
                carry = (uint32_t)(s2 >> 32);
            }
        }
    }
    return out;
}





__device__ __forceinline__
uint32_t warp_add2(uint32_t a_j, int lane_id) {
    return warp_add_mod_p_u32(a_j, a_j, lane_id); 
}

__device__ __forceinline__
uint32_t warp_add3(uint32_t a_j, int lane_id) {
    uint32_t t2 = warp_add_mod_p_u32(a_j, a_j, lane_id); 
    return warp_add_mod_p_u32(t2, a_j, lane_id);         
}

__device__ __forceinline__
uint32_t warp_add4(uint32_t a_j, int lane_id) {
    uint32_t t2 = warp_add_mod_p_u32(a_j, a_j, lane_id); 
    return warp_add_mod_p_u32(t2, t2, lane_id);          
}

__device__ __forceinline__
uint32_t warp_add8(uint32_t a_j, int lane_id) {
    uint32_t t4 = warp_add4(a_j, lane_id);
    return warp_add_mod_p_u32(t4, t4, lane_id);          
}




__device__ __forceinline__
uint32_t warp_mont_sqr_u32(const uint32_t a[8], int lane_id) {
    return warp_mont_mul_comba(a, a, lane_id);
}












__device__ __forceinline__
uint32_t warp_mont_sqr_lane(uint32_t a_lane, int lane_id) {

    const unsigned FULL = 0xFFFFFFFFu;
    uint32_t a[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = __shfl_sync(FULL, a_lane, i, 8);
    }

    // ALL 32 lanes call warp_mont_mul_comba — each 8-lane group runs independently
    return warp_mont_mul_comba(a, a, lane_id);
}













__device__ __forceinline__
uint32_t warp_mont_mul_array_lane(const uint32_t a[8], uint32_t b_lane, int lane_id) {

    const unsigned FULL = 0xFFFFFFFFu;
    uint32_t b[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        b[i] = __shfl_sync(FULL, b_lane, i, 8);
    }

    // KERNEL-P256-JACOBIAN-01: Call warp_mont_mul_comba directly.
    // warp_mont_mul_ab_compare uses L8-masked shuffles after the multiply,
    // which desynchronizes lanes 8-31 from 0-7 and causes __activemask()
    // fragmentation on the NEXT multiply call — deadlock in fast tier.
    return warp_mont_mul_comba(a, b, lane_id);
}





__device__ __forceinline__
uint32_t warp_negate_u32(uint32_t y_j, int lane_id) {
    return warp_sub_mod_p_u32(P256_P_LE_CONST[lane_id & 7], y_j, lane_id);
}




__device__ __forceinline__
void gather8(const uint32_t v[8], uint32_t out[8], int lane_id) {
    uint32_t lane = (lane_id < 8) ? v[lane_id] : 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = __shfl_sync(0xFF, lane, i);
}

__device__ __forceinline__
void dbg_print_vec(const char* tag, const uint32_t v[8], int lane_id) {
    uint32_t g[8];
    gather8(v, g, lane_id);
    if (lane_id == 0) {
        printf("%s = %08x %08x %08x %08x %08x %08x %08x %08x\n",
               tag, g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]);
    }
}





__device__ __forceinline__
bool vec_eq8(const uint32_t a[8], const uint32_t b[8], int lane_id) {
    bool local_match = (a[lane_id & 7] == b[lane_id & 7]);

    unsigned mask = __ballot_sync(0xFFFFFFFF, local_match);

    // Each 8-lane group checks its own slice
    unsigned shift = (lane_id & ~7);
    return (((mask >> shift) & 0xFFu) == 0xFFu);
}






__device__ __forceinline__
bool warp_equals_mont_one(const uint32_t a[8], int lane_id) {
    const unsigned FULL = 0xFFFFFFFFu;

    
    const bool limb_ok = (a[lane_id & 7] == P256_MONT_ONE[lane_id & 7]);


    const unsigned m = __ballot_sync(FULL, limb_ok);

    // Each 8-lane group checks its own slice
    unsigned shift = (lane_id & ~7);
    bool ok = (((m >> shift) & 0xFFu) == 0xFFu);


    ok = __shfl_sync(FULL, ok, 0, 8);
    return ok;
}






__device__ __forceinline__
bool warp_is_zero(const uint32_t a[8], int lane_id) {
    const unsigned FULL = 0xFFFFFFFFu;

    
    const bool limb_zero = (a[lane_id & 7] == 0);


    const unsigned m = __ballot_sync(FULL, limb_zero);

    // Each 8-lane group checks its own slice
    unsigned shift = (lane_id & ~7);
    bool ok = (((m >> shift) & 0xFFu) == 0xFFu);


    ok = __shfl_sync(FULL, ok, 0, 8);
    return ok;
}







__device__ __forceinline__
uint32_t dbl_vec(const uint32_t a[8], int lane_id) {
    return warp_add_mod_p_u32(a[lane_id & 7], a[lane_id & 7], lane_id);
}


__device__ __forceinline__
uint32_t quad_vec(const uint32_t a[8], int lane_id) {
    uint32_t t = dbl_vec(a, lane_id);
    return warp_add_mod_p_u32(t, t, lane_id);
}


__device__ __forceinline__
uint32_t oct_vec(const uint32_t a[8], int lane_id) {
    uint32_t t = quad_vec(a, lane_id);
    return warp_add_mod_p_u32(t, t, lane_id);
}


__device__ __forceinline__
uint32_t triple_vec(const uint32_t a[8], int lane_id) {
    uint32_t t = dbl_vec(a, lane_id);
    return warp_add_mod_p_u32(t, a[lane_id & 7], lane_id);
}


__device__ __forceinline__
void broadcast_lane_scalar_to_vec8(uint32_t scalar_lane, uint32_t out[8], int lane_id) {
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        out[i] = __shfl_sync(0xFFFFFFFF, scalar_lane, i, 8);
}

__device__ __forceinline__
uint32_t dbl_vec_lane(uint32_t a_lane, int lane_id) {
    return warp_add_mod_p_u32(a_lane, a_lane, lane_id);
}

__device__ __forceinline__
uint32_t triple_vec_lane(uint32_t a_lane, int lane_id) {
    uint32_t two_x = warp_add_mod_p_u32(a_lane, a_lane, lane_id);
    return warp_add_mod_p_u32(two_x, a_lane, lane_id);
}

__device__ __forceinline__
uint32_t quad_vec_lane(uint32_t a_lane, int lane_id) {
    uint32_t two_x = warp_add_mod_p_u32(a_lane, a_lane, lane_id);
    return warp_add_mod_p_u32(two_x, two_x, lane_id);
}

__device__ __forceinline__
uint32_t oct_vec_lane(uint32_t a_lane, int lane_id) {
    uint32_t two_x  = warp_add_mod_p_u32(a_lane, a_lane, lane_id);
    uint32_t four_x = warp_add_mod_p_u32(two_x, two_x, lane_id);
    return warp_add_mod_p_u32(four_x, four_x, lane_id);
}










__device__ __forceinline__
bool jacobian_is_infinity(const uint32_t Z[8], int lane_id)
{
    // KERNEL-P256-JACOBIAN-01: Hardcoded full-warp mask.
    const unsigned full = 0xFFFFFFFFu;
    uint32_t z = Z[lane_id & 7];
    unsigned mask = __ballot_sync(full, z == 0);
    // Each 8-lane group checks its own Z: extract the relevant 8-bit slice
    unsigned shift = (lane_id & ~7);  // 0, 8, 16, or 24
    return ((mask >> shift) & 0xFFu) == 0xFFu;
}






__device__ __forceinline__
void jacobian_set_infinity(uint32_t X[8], uint32_t Y[8], uint32_t Z[8], int lane_id)
{
    X[lane_id & 7] = 0;
    Y[lane_id & 7] = 0;
    Z[lane_id & 7] = 0;
}







__device__ __forceinline__
void jacobian_set_affine(uint32_t X[8], uint32_t Y[8], uint32_t Z[8],
                         const uint32_t Ax[8], const uint32_t Ay[8],
                         int lane_id)
{
    X[lane_id & 7] = Ax[lane_id & 7];
    Y[lane_id & 7] = Ay[lane_id & 7];
    Z[lane_id & 7] = P256_MONT_ONE[lane_id & 7];
}















// WARP-COOP-JACOBIAN-RECOVERY-01: Rewritten to eliminate partial-array staging.
// All intermediates are lane-scalars. Distributed vec8 arrays are materialized
// only via broadcast_lane_scalar_to_vec8 when a callee truly needs one.
__device__ __forceinline__
void jacobian_double_warp(uint32_t X[8], uint32_t Y[8], uint32_t Z[8], int lane_id)
{
    const unsigned FULL = 0xFFFFFFFFu;
    bool z_is_zero_lane = (Z[lane_id & 7] == 0);
    unsigned ballot = __ballot_sync(FULL, z_is_zero_lane);
    // Each 8-lane group checks its own Z slice
    unsigned shift = (lane_id & ~7);
    if (((ballot >> shift) & 0xFFu) == 0xFFu) return;

    // Phase 1: squares (input arrays X/Y/Z are valid distributed vec8)
    uint32_t delta_lane  = warp_mont_sqr_u32(Z, lane_id);      // Z^2
    uint32_t gamma_lane  = warp_mont_sqr_u32(Y, lane_id);      // Y^2
    uint32_t gamma2_lane = warp_mont_sqr_lane(gamma_lane, lane_id); // Y^4
    uint32_t beta_lane   = warp_mont_mul_array_lane(X, gamma_lane, lane_id); // X * Y^2

    // Phase 2: alpha = 3*(X^2 - Z^4) via (X-Z^2)(X+Z^2)
    uint32_t tmp1_lane = warp_sub_mod_p_u32(X[lane_id & 7], delta_lane, lane_id); // X - Z^2
    uint32_t tmp2_lane = warp_add_mod_p_u32(X[lane_id & 7], delta_lane, lane_id); // X + Z^2

    // tmp1 * tmp2: two scalar inputs, need vec8 for one side
    uint32_t tmp1_vec[8], tmp2_vec[8];
    broadcast_lane_scalar_to_vec8(tmp1_lane, tmp1_vec, lane_id);
    broadcast_lane_scalar_to_vec8(tmp2_lane, tmp2_vec, lane_id);
    uint32_t t_lane     = warp_mont_mul_comba(tmp1_vec, tmp2_vec, lane_id); // (X-Z^2)(X+Z^2)
    uint32_t alpha_lane = triple_vec_lane(t_lane, lane_id);                 // 3 * t

    // Phase 3: X3 = alpha^2 - 8*beta
    uint32_t alpha2_lane     = warp_mont_sqr_lane(alpha_lane, lane_id);
    uint32_t eight_beta_lane = oct_vec_lane(beta_lane, lane_id);
    uint32_t X3_lane = warp_sub_mod_p_u32(alpha2_lane, eight_beta_lane, lane_id);

    // Phase 4: Z3 = (Y+Z)^2 - Y^2 - Z^2
    uint32_t YplusZ  = warp_add_mod_p_u32(Y[lane_id & 7], Z[lane_id & 7], lane_id);
    uint32_t YZ2     = warp_mont_sqr_lane(YplusZ, lane_id);
    uint32_t t_z     = warp_sub_mod_p_u32(YZ2, gamma_lane, lane_id);
    uint32_t Z3_lane = warp_sub_mod_p_u32(t_z, delta_lane, lane_id);

    // Phase 5: Y3 = alpha*(4*beta - X3) - 8*Y^4
    uint32_t four_beta_lane    = quad_vec_lane(beta_lane, lane_id);
    uint32_t fb_minus_X3       = warp_sub_mod_p_u32(four_beta_lane, X3_lane, lane_id);
    uint32_t alpha_vec[8];
    broadcast_lane_scalar_to_vec8(alpha_lane, alpha_vec, lane_id);
    uint32_t alpha_times       = warp_mont_mul_array_lane(alpha_vec, fb_minus_X3, lane_id);
    uint32_t eight_gamma2_lane = oct_vec_lane(gamma2_lane, lane_id);
    uint32_t Y3_lane = warp_sub_mod_p_u32(alpha_times, eight_gamma2_lane, lane_id);

    // Write back: materialize final scalars into output distributed arrays
    broadcast_lane_scalar_to_vec8(X3_lane, X, lane_id);
    broadcast_lane_scalar_to_vec8(Y3_lane, Y, lane_id);
    broadcast_lane_scalar_to_vec8(Z3_lane, Z, lane_id);
}























__device__ __forceinline__
void jacobian_add_affine_warp(
    uint32_t X1[8],
    uint32_t Y1[8],
    uint32_t Z1[8],
    const uint32_t X2[8],
    const uint32_t Y2[8],
    int lane_id
)
{
    const unsigned FULL = 0xFFFFFFFFu;
    bool z_is_zero_lane = (Z1[lane_id & 7] == 0);
    unsigned ballot = __ballot_sync(FULL, z_is_zero_lane);
    // Each 8-lane group checks its own Z1 slice
    unsigned shift = (lane_id & ~7);
    if (((ballot >> shift) & 0xFFu) == 0xFFu) {
        X1[lane_id & 7] = X2[lane_id & 7];
        Y1[lane_id & 7] = Y2[lane_id & 7];
        Z1[lane_id & 7] = P256_MONT_ONE[lane_id & 7];
        return;
    }

    // Z1Z1 = Z1^2 (input Z1 is a valid distributed vec8)
    uint32_t Z1Z1_lane = warp_mont_sqr_u32(Z1, lane_id);

    // U2 = X2 * Z1Z1
    uint32_t U2_lane = warp_mont_mul_array_lane(X2, Z1Z1_lane, lane_id);

    // Z1_cu = Z1 * Z1Z1 (Z1 is valid vec8)
    uint32_t Z1_cu_lane = warp_mont_mul_array_lane(Z1, Z1Z1_lane, lane_id);

    // S2 = Y2 * Z1^3
    uint32_t S2_lane = warp_mont_mul_array_lane(Y2, Z1_cu_lane, lane_id);

    // H = U2 - X1 (both scalars)
    uint32_t H_lane = warp_sub_mod_p_u32(U2_lane, X1[lane_id & 7], lane_id);

    // HH = H^2
    uint32_t HH_lane = warp_mont_sqr_lane(H_lane, lane_id);

    // I = 4 * HH
    uint32_t I_lane = quad_vec_lane(HH_lane, lane_id);

    // J = H * I
    uint32_t H_vec[8];
    broadcast_lane_scalar_to_vec8(H_lane, H_vec, lane_id);
    uint32_t J_lane = warp_mont_mul_array_lane(H_vec, I_lane, lane_id);

    // r = 2 * (S2 - Y1)
    uint32_t S2_minus_Y1 = warp_sub_mod_p_u32(S2_lane, Y1[lane_id & 7], lane_id);
    uint32_t r_lane = dbl_vec_lane(S2_minus_Y1, lane_id);

    // V = X1 * I
    uint32_t V_lane = warp_mont_mul_array_lane(X1, I_lane, lane_id);

    // X3 = r^2 - J - 2*V
    uint32_t r2_lane      = warp_mont_sqr_lane(r_lane, lane_id);
    uint32_t r2_minus_J   = warp_sub_mod_p_u32(r2_lane, J_lane, lane_id);
    uint32_t twoV_lane    = dbl_vec_lane(V_lane, lane_id);
    uint32_t X3_lane      = warp_sub_mod_p_u32(r2_minus_J, twoV_lane, lane_id);

    // Y3 = r*(V - X3) - 2*Y1*J
    uint32_t V_minus_X3   = warp_sub_mod_p_u32(V_lane, X3_lane, lane_id);
    uint32_t r_vec[8];
    broadcast_lane_scalar_to_vec8(r_lane, r_vec, lane_id);
    uint32_t r_times      = warp_mont_mul_array_lane(r_vec, V_minus_X3, lane_id);
    uint32_t twoY1_lane   = dbl_vec_lane(Y1[lane_id & 7], lane_id);
    uint32_t J_vec[8];
    broadcast_lane_scalar_to_vec8(J_lane, J_vec, lane_id);
    uint32_t twoY1J_lane  = warp_mont_mul_array_lane(J_vec, twoY1_lane, lane_id);
    uint32_t Y3_lane      = warp_sub_mod_p_u32(r_times, twoY1J_lane, lane_id);

    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    uint32_t Z1_plus_H    = warp_add_mod_p_u32(Z1[lane_id & 7], H_lane, lane_id);
    uint32_t Z1H2_lane    = warp_mont_sqr_lane(Z1_plus_H, lane_id);
    uint32_t t_lane        = warp_sub_mod_p_u32(Z1H2_lane, Z1Z1_lane, lane_id);
    uint32_t Z3_lane      = warp_sub_mod_p_u32(t_lane, HH_lane, lane_id);

    // Write back final scalars into distributed output arrays
    broadcast_lane_scalar_to_vec8(X3_lane, X1, lane_id);
    broadcast_lane_scalar_to_vec8(Y3_lane, Y1, lane_id);
    broadcast_lane_scalar_to_vec8(Z3_lane, Z1, lane_id);
}









__device__ __forceinline__
uint32_t warp_mod_add(
    const uint32_t a[8],
    const uint32_t b[8],
    int lane_id
)
{
    // KERNEL-P256-JACOBIAN-01: Hardcoded full-warp mask.
    const unsigned full = 0xFFFFFFFFu;
    const unsigned sub = full;  // KERNEL-P256-JACOBIAN-01: full-warp mask for all shuffles


    uint32_t a_lane = a[lane_id & 7];
    uint32_t b_lane = b[lane_id & 7];

    uint64_t sum = (uint64_t)a_lane + (uint64_t)b_lane;
    uint32_t result = (uint32_t)sum;
    uint32_t carry = (sum >> 32);


    for (int offset = 1; offset < 8; offset *= 2) {
        uint32_t neighbor_carry = __shfl_up_sync(sub, carry, offset, 8);
        if ((lane_id & 7) >= offset) {
            sum = (uint64_t)result + (uint64_t)neighbor_carry;
            result = (uint32_t)sum;
            carry = (sum >> 32);
        }
    }


    uint32_t p_lane = P256_P_LE_CONST[lane_id & 7];


    bool need_reduce = false;
    if ((lane_id & 7) == 7) {
        need_reduce = (result >= p_lane);
    }
    need_reduce = __shfl_sync(sub, need_reduce, 7, 8);

    if (need_reduce) {
        uint32_t borrow = 0;
        uint64_t diff = (uint64_t)result - (uint64_t)p_lane - borrow;
        result = (uint32_t)diff;
        borrow = (diff >> 63) & 1u;


        for (int offset = 1; offset < 8; offset *= 2) {
            uint32_t neighbor_borrow = __shfl_up_sync(sub, borrow, offset, 8);
            if ((lane_id & 7) >= offset) {
                diff = (uint64_t)result - neighbor_borrow;
                result = (uint32_t)diff;
                borrow = (diff >> 63) & 1u;
            }
        }
    }

    return result;
}








__device__ __forceinline__
uint32_t warp_mod_sub(
    const uint32_t a[8],
    const uint32_t b[8],
    int lane_id
)
{
    // KERNEL-P256-JACOBIAN-01: Hardcoded full-warp mask.
    const unsigned full = 0xFFFFFFFFu;
    const unsigned sub = full;  // KERNEL-P256-JACOBIAN-01: full-warp mask for all shuffles


    bool need_add_p = false;

    {
        #pragma unroll
        for (int i = 7; i >= 0; --i) {
            uint32_t a_i = a[i];
            uint32_t b_i = b[i];

            if (a_i < b_i) {
                need_add_p = true;
                break;
            } else if (a_i > b_i) {
                break;
            }

        }
    }


    need_add_p = __any_sync(sub, need_add_p);

    uint32_t a_lane = a[lane_id & 7];
    uint32_t b_lane = b[lane_id & 7];
    uint32_t p_lane = P256_P_LE_CONST[lane_id & 7];


    uint32_t result;
    if (need_add_p) {

        uint64_t sum = (uint64_t)a_lane + (uint64_t)p_lane;
        result = (uint32_t)sum;
        uint32_t carry = (sum >> 32);


        for (int offset = 1; offset < 8; offset *= 2) {
            uint32_t neighbor_carry = __shfl_up_sync(sub, carry, offset, 8);
            if ((lane_id & 7) >= offset) {
                sum = (uint64_t)result + (uint64_t)neighbor_carry;
                result = (uint32_t)sum;
                carry = (sum >> 32);
            }
        }


        uint32_t borrow = 0;
        uint64_t diff = (uint64_t)result - (uint64_t)b_lane - borrow;
        result = (uint32_t)diff;
        borrow = (diff >> 63) & 1u;


        for (int offset = 1; offset < 8; offset *= 2) {
            uint32_t neighbor_borrow = __shfl_up_sync(sub, borrow, offset, 8);
            if ((lane_id & 7) >= offset) {
                diff = (uint64_t)result - neighbor_borrow;
                result = (uint32_t)diff;
                borrow = (diff >> 63) & 1u;
            }
        }
    } else {

        uint32_t borrow = 0;
        uint64_t diff = (uint64_t)a_lane - (uint64_t)b_lane - borrow;
        result = (uint32_t)diff;
        borrow = (diff >> 63) & 1u;


        for (int offset = 1; offset < 8; offset *= 2) {
            uint32_t neighbor_borrow = __shfl_up_sync(sub, borrow, offset, 8);
            if ((lane_id & 7) >= offset) {
                diff = (uint64_t)result - neighbor_borrow;
                result = (uint32_t)diff;
                borrow = (diff >> 63) & 1u;
            }
        }
    }

    return result;
}










__device__ __forceinline__
uint32_t warp_add_lazy(
    const uint32_t a[8],
    const uint32_t b[8],
    int lane_id
)
{
    if (lane_id >= 8) return 0;

    uint32_t a_lane = a[lane_id];
    uint32_t b_lane = b[lane_id];

    uint64_t sum = (uint64_t)a_lane + (uint64_t)b_lane;
    uint32_t result = (uint32_t)sum;
    uint32_t carry = (sum >> 32);

    
    for (int offset = 1; offset < 8; offset *= 2) {
        uint32_t neighbor_carry = __shfl_up_sync(0xFF, carry, offset);
        if (lane_id >= offset && lane_id < 8) {
            sum = (uint64_t)result + (uint64_t)neighbor_carry;
            result = (uint32_t)sum;
            carry = (sum >> 32);
        }
    }

    
    return result;
}







__device__ __forceinline__
uint32_t warp_cond_reduce(
    const uint32_t r[8],
    int lane_id
)
{
    if (lane_id >= 8) return 0;

    uint32_t my_limb = r[lane_id];
    uint32_t p_lane = P256_P_LE_CONST[lane_id];

    
    bool need_reduce = false;
    if (lane_id == 7) {
        need_reduce = (my_limb >= p_lane);
    }
    need_reduce = __shfl_sync(0xFF, need_reduce, 7);

    if (need_reduce) {
        uint32_t borrow = 0;
        uint64_t diff = (uint64_t)my_limb - (uint64_t)p_lane - borrow;
        my_limb = (uint32_t)diff;
        borrow = (diff >> 63) & 1u;

        for (int offset = 1; offset < 8; offset *= 2) {
            uint32_t neighbor_borrow = __shfl_up_sync(0xFF, borrow, offset);
            if (lane_id >= offset && lane_id < 8) {
                diff = (uint64_t)my_limb - neighbor_borrow;
                my_limb = (uint32_t)diff;
                borrow = (diff >> 63) & 1u;
            }
        }
    }

    return my_limb;
}








__device__ __constant__ int P256_FIXED_W = 8;  
__device__ __constant__ int P256_ENTRIES = 255;  // All multiples: 2^w - 1 for w=8


__device__ __constant__ uint32_t P256_G_TABLE_X_SOA[8 * 255];
__device__ __constant__ uint32_t P256_G_TABLE_Y_SOA[8 * 255];

#if P256_FASTPATH_FULLWINDOW
// SPEED-H2-01: full-window fixed-base table for zero-doubling k*G.
// 33 window positions x 128 signed multiples of 2^(8i)*G, Montgomery-domain
// affine, AoS: per entry = X[8 u32 LE] || Y[8 u32 LE] (16 u32 = 64 B).
// index = window*128 + (|digit|-1). 4224 entries = 264 KB.
// 33 (not 32) windows: the signed W=8 recode can emit a carry digit at
// window 32 (weight 2^256); without that window, carrying scalars sign wrong.
// GLOBAL (NOT __constant__): exceeds the 64 KB constant limit and per-warp
// lookups are digit-divergent — constant-memory broadcast would serialize
// them (measured in the llm.c sidecar). L2-resident on every target GPU.
#define P256_FULLWINDOW_WINDOWS 33
#define P256_FULLWINDOW_DIGITS  128
#define P256_FULLWINDOW_ENTRIES (P256_FULLWINDOW_WINDOWS * P256_FULLWINDOW_DIGITS)
__device__ uint32_t P256_G_FULLWINDOW[P256_FULLWINDOW_ENTRIES * 16];

// Gather affine (X,Y) for window w, absolute digit d in [1,128] into the
// SAME per-lane SoA layout soa_lookup_bucket_warp uses: each of the 8 lanes
// in a group holds ONE limb (Px[local_lane], Py[local_lane]), so the reused
// jacobian_add_affine_oracle_compat / warp_negate_u32 primitives see the
// exact operand shape they expect. Plain global loads — no shuffle, no
// cg::tile, no __syncthreads (D-018/D-020 hang class avoided, condition C6).
__device__ __forceinline__
void fullwindow_lookup_warp(int w, int abs_digit, uint32_t Px[8], uint32_t Py[8], int lane_id) {
    const int local_lane = lane_id & 7;
    const int base = (w * P256_FULLWINDOW_DIGITS + (abs_digit - 1)) * 16;
    Px[local_lane] = P256_G_FULLWINDOW[base + local_lane];
    Py[local_lane] = P256_G_FULLWINDOW[base + 8 + local_lane];
}
#endif // P256_FASTPATH_FULLWINDOW




__device__ __forceinline__
uint32_t get_bit_le256(const uint32_t k[8], int i)
{
    int limb = i >> 5;
    int off = i & 31;
    return (k[limb] >> off) & 1u;
}

















template<int W, int MAXL>
__device__ __forceinline__
int recode_signed_fixed_window(const uint32_t k[8], int8_t digits[MAXL])
{
    // t[0..7] = scalar limbs (LE), t[8] = extra limb for carry
    uint32_t t[9];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        t[i] = k[i];
    }
    t[8] = 0u;

    int pos  = 0;   // bit position in scalar
    int widx = 0;   // digit index

    while (pos < 256 && widx < MAXL) {
        // 1) Skip leading zero windows
        while (pos < 256) {
            int limb = pos >> 5;        // which 32-bit limb
            int off  = pos & 31;        // bit offset in limb

            uint32_t window = t[limb] >> off;
            if (off + W > 32 && limb + 1 < 9) {
                window |= t[limb + 1] << (32 - off);
            }
            window &= ((1u << W) - 1u);

            if (window != 0u) {
                break;                  // found a non-zero W-bit window
            }

            digits[widx++] = 0;
            pos += W;
        }

        if (pos >= 256) {
            break;
        }

        // 2) Extract actual window at this pos
        int limb = pos >> 5;
        int off  = pos & 31;
        uint32_t window = t[limb] >> off;
        if (off + W > 32 && limb + 1 < 9) {
            window |= t[limb + 1] << (32 - off);
        }
        window &= ((1u << W) - 1u);

        int32_t d = static_cast<int32_t>(window);

        // 3) Signed recentering into [-2^(W-1), 2^(W-1)-1]
        if (d & (1 << (W - 1))) {
            d -= (1 << W);

            // Add carry of +1 to next W-bit window
            int cpos = pos + W;
            int cL   = cpos >> 5;
            int cO   = cpos & 31;
            if (cL <= 8) {
                uint32_t add = (1u << cO);
                uint64_t s   = static_cast<uint64_t>(t[cL]) + add;
                t[cL]        = static_cast<uint32_t>(s);
                uint32_t carry = static_cast<uint32_t>(s >> 32);

                for (int j = cL + 1; carry && j <= 8; ++j) {
                    uint64_t s2 = static_cast<uint64_t>(t[j]) + 1u;
                    t[j]        = static_cast<uint32_t>(s2);
                    carry       = static_cast<uint32_t>(s2 >> 32);
                }
            }
        }

        digits[widx++] = static_cast<int8_t>(d);
        pos += W;
    }

    // Zero-pad remaining digits
    while (widx < MAXL) {
        digits[widx++] = 0;
    }

    int L = (256 + W - 1) / W;
    return L;
}













__device__ __forceinline__
void soa_lookup_bucket(int entry, uint32_t Xo[8], uint32_t Yo[8], int lane_id)
{
    
    
    
    {
        int li = lane_id & 7;
        Xo[li] = P256_G_TABLE_X_SOA[entry * 8 + li];
        Yo[li] = P256_G_TABLE_Y_SOA[entry * 8 + li];
    }
    
}








__device__ __forceinline__
uint32_t get_bit_le256_u32(const uint32_t e[8], int i) {
    int limb = i >> 5;           
    int off  = i & 31;           
    return (e[limb] >> off) & 1u;
}


__device__ __forceinline__
void vec_copy8(uint32_t dst[8], const uint32_t src[8], int lane_id) {
    dst[lane_id & 7] = src[lane_id & 7];
}
















__device__ __forceinline__
void warp_fermat_inverse_p256(const uint32_t Z[8], uint32_t out[8], int lane_id)
{
    // Card 26.5: Count per-signature Fermat inversions
    CARD26_INV_FERMAT_INCREMENT();

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[fermat] Entry\n");
#endif

    // KERNEL-P256-JACOBIAN-01: Hardcoded full-warp mask.
    const unsigned full = 0xFFFFFFFFu;
    const unsigned sub = full;  // KERNEL-P256-JACOBIAN-01: full-warp mask for all shuffles

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[fermat] full=0x%08x sub=0x%08x\n", full, sub);
#endif

    int li = lane_id & 7;
    uint32_t z_limb = Z[li];
    unsigned zero_mask = __ballot_sync(sub, z_limb == 0u);
    unsigned shift = (lane_id & ~7);
    bool z_is_zero = ((zero_mask >> shift) & 0xFFu) == 0xFFu;
    if (z_is_zero) {
        out[li] = 0u;
#if P256_WARP_COOP_DEBUG
        if (lane_id == 0) printf("[fermat] Z is zero, early return\n");
#endif
        return;
    }



    uint32_t is_lit_one = (li == 0) ? (Z[0] == 1u) : (Z[li] == 0u);
    unsigned lit_one_mask = __ballot_sync(sub, is_lit_one);
    if (((lit_one_mask >> shift) & 0xFFu) == 0xFFu) {
        out[li] = P256_MONT_ONE[li];
#if P256_WARP_COOP_DEBUG
        if (lane_id == 0) printf("[fermat] Z is literal one, early return\n");
#endif
        return;
    }


    uint32_t is_m_one = (Z[li] == P256_MONT_ONE[li]);
    unsigned mont_one_mask = __ballot_sync(sub, is_m_one);
    if (((mont_one_mask >> shift) & 0xFFu) == 0xFFu) {
        out[li] = P256_MONT_ONE[li];
#if P256_WARP_COOP_DEBUG
        if (lane_id == 0) printf("[fermat] Z is mont one, early return\n");
#endif
        return;
    }

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[fermat] Starting 256-bit square-multiply ladder...\n");
#endif


    uint32_t r[8], base[8];
    vec_copy8(r,    P256_MONT_ONE, lane_id);
    vec_copy8(base, Z,             lane_id);




    #pragma unroll
    for (int bit = 255; bit >= 0; --bit) {
#if P256_WARP_COOP_DEBUG
        if (lane_id == 0 && (bit == 255 || bit == 254 || bit == 253)) {
            printf("[fermat] bit=%d starting sqr...\n", bit);
        }
#endif

        uint32_t r_sqr = warp_mont_sqr_u32(r, lane_id);
        r[lane_id & 7] = r_sqr;

#if P256_WARP_COOP_DEBUG
        if (lane_id == 0 && (bit == 255 || bit == 254 || bit == 253)) {
            printf("[fermat] bit=%d sqr done\n", bit);
        }
#endif


        uint32_t b = get_bit_le256_u32(P256_P_MINUS_2_LE, bit);
        // Card 26.2 FIX: Use full mask - ALL 32 lanes need the broadcasted bit value
        // to avoid divergence at the if(b) branch below
        b = __shfl_sync(0xFFFFFFFF, b, 0, 8);

        if (b) {
            uint32_t r_mul = warp_mont_mul_comba(r, base, lane_id);
            r[lane_id & 7] = r_mul;
        }
        // Card 26.2 FIX: Use full mask since all 32 lanes execute this loop
        __syncwarp(0xFFFFFFFF);
    }

#if P256_WARP_COOP_DEBUG
    if (lane_id == 0) printf("[fermat] Ladder complete!\n");
#endif

    vec_copy8(out, r, lane_id);
}

