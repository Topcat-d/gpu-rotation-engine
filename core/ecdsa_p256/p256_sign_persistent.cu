// WARP-COOP CARD A: Master debug flag (0 = strip all debug code to save registers)
#define WARP_COOP_DEBUG 0

// ============================================================================
// CARD18: Master scalar debug macro
// ============================================================================
// Uncomment the next line to enable verbose scalar debug output
// #define P256_SCALAR_DEBUG 1

#if defined(P256_SCALAR_DEBUG)
  #define SCALAR_DBG(...) printf(__VA_ARGS__)
#else
  #define SCALAR_DBG(...) do {} while (0)
#endif
// ============================================================================

// --- debug stubs (safe no-ops) ---
#ifndef warpdbg_heartbeat
#define warpdbg_heartbeat(msg) do {} while(0)
#endif
#ifndef warpdbg
#define warpdbg(...) do {} while(0)
#endif
/*
 * P-256 ECDSA Signing - Persistent Fused Kernel
 *
 * Single kernel that does:
 *   1. Fixed-base scalar multiplication (k*G) → Jacobian (X, Y, Z)
 *   2. Cooperative batch inverse (Montgomery's trick, GPU-resident)
 *   3. Affine conversion (x = X*Z^-2, y = Y*Z^-3)
 *   4. Single global store (no intermediate round-trips)
 *
 * Why Persistent:
 *   - Launch once, process work queue until empty
 *   - No kernel launch overhead per batch
 *   - All data stays GPU-resident (no CPU sync!)
 *
 * Expected: +20-40% over separate kernels + graphs
 *
 * Target: RTX 2060 SUPER (sm_75), 2176 CUDA cores
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "p256_field_warp_coop.cuh"
#include "jacobian_double_oracle.cuh"

// USE ORACLE WRAPPERS FOR JACOBIAN OPS (cuPQC-style pattern to fix warp sync bugs)
#define jacobian_double_warp jacobian_double_oracle_compat
#define jacobian_add_affine_warp jacobian_add_affine_oracle_compat

// Debug flag (set to 1 to enable domain diagnostics)
#ifndef P256_DBG
#define P256_DBG 0  // DISABLED to test if probes cause hang
#endif

// ONE_SHOT mode (set to 1 to disable grid-stride loop, process one job and return)
#ifndef ONE_SHOT
#define ONE_SHOT 1  // ENABLED for debugging
#endif

// LT-2: Ladder trace debugging - now controlled by P256_SCALAR_DEBUG
// Log ladder operations: op is 'S' (seed), 'D' (double), or 'A' (add)
// Only logs from warp 0, lane 0, scalar index 0 to match CPU trace
#if defined(P256_SCALAR_DEBUG)
#define LADDER_TRACE(step, j, op, X0, Y0, Z0, warp_id, lane_id, sc_idx)         \
    do {                                                                        \
        if ((warp_id) == 0 && (lane_id) == 0 && (sc_idx) == 0) {                \
            SCALAR_DBG("[GPU-TRACE] step=%02d j=%2d op=%c X0=%08x Y0=%08x Z0=%08x\n",\
                   (int)(step), (int)(j), (char)(op),                            \
                   (unsigned int)(X0), (unsigned int)(Y0), (unsigned int)(Z0)); \
        }                                                                       \
    } while (0)
#else
#define LADDER_TRACE(step, j, op, X0, Y0, Z0, warp_id, lane_id, sc_idx) \
    do { (void)step; (void)j; (void)op; (void)X0; (void)Y0; (void)Z0;   \
         (void)warp_id; (void)lane_id; (void)sc_idx; } while (0)
#endif

// ============================================================================
// Debug Helpers
// ============================================================================

// Lightweight progress beacon (cannot optimize away)
__device__ __forceinline__ void beacon(int id) {
    // DISABLED for performance
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("[BEACON %d]\n", id);
    // }
}

// Assert all lanes 0-7 are active
__device__ __forceinline__ void assert_full_lane_mask(int lane_id) {
    unsigned m = __activemask();
    if ((m & 0xFFu) != 0xFFu) {
        if (lane_id == 0) SCALAR_DBG("[MASK BUG] activemask=0x%08x\n", m);
    }
}

// P-256 curve parameters (affine generator G)
__constant__ uint32_t P256_Gx[8] = {
    0xD898C296, 0xF4A13945, 0x2DEB33A0, 0x77037D81,
    0x63A440F2, 0xF8BCE6E5, 0xE12C4247, 0x6B17D1F2
};

__constant__ uint32_t P256_Gy[8] = {
    0x37BF51F5, 0xCBB64068, 0x6B315ECE, 0x2BCE3357,
    0x7C0F9E16, 0x8EE7EB4A, 0xFE1A7F9B, 0x4FE342E2
};

// Note: No work queue needed for grid-stride loop pattern
// Each warp processes indices: warp_id_g, warp_id_g + warps_g, warp_id_g + 2*warps_g, ...

/*
 * Fixed-Base Scalar Multiplication (Simplified for skeleton)
 *
 * Computes: R = scalar * G (generator)
 * Returns: Jacobian coordinates (X, Y, Z)
 *
 * This is a SKELETON - replace with actual fixed-base comb (w=7/8).
 * Uses warp-cooperative field ops for point operations.
 *
 * Args:
 *   scalar[8]: 256-bit scalar in limbs
 *   lane_id: Thread's lane ID within warp
 *   Rx[8], Ry[8], Rz[8]: Output Jacobian point (shared by warp)
 */
__device__ void fixed_base_scalar_mul_warp(
    const uint32_t scalar[8],
    int lane_id,
    uint32_t Rx[8],
    uint32_t Ry[8],
    uint32_t Rz[8]
)
{
    // SKELETON: Replace with actual comb method
    // For now, just return identity point (0, 1, 0) as placeholder

    if (lane_id < 8) {
        // Initialize to identity point (point at infinity)
        Rx[lane_id] = 0;
        Ry[lane_id] = (lane_id == 0) ? 1 : 0;
        Rz[lane_id] = 0;
    }

    // TODO: Implement actual fixed-base comb with w=7 or w=8
    // - Load from SoA comb table
    // - Process scalar window-by-window
    // - Use warp_mont_mul_comba for field ops
    // - Use warp_add_lazy for point operations
    // - Final warp_cond_reduce before returning
}

/*
 * Cooperative Batch Inverse (GPU-Resident)
 *
 * Montgomery's trick for batch inversion:
 *   1. Forward pass: Compute prefix products P[i] = Z[0]*Z[1]*...*Z[i]
 *   2. Single inversion: Compute (P[N-1])^-1 using Fermat
 *   3. Backward pass: Compute Z[i]^-1 using prefix products
 *
 * All on GPU! No CPU sync needed.
 *
 * Uses shared memory for prefix products within block.
 * Each warp handles multiple Z values.
 *
 * Args:
 *   Z_batch: Input Z coordinates [batch_size, 8]
 *   Zinv_batch: Output Z^-1 [batch_size, 8]
 *   batch_size: Number of points
 *   lane_id: Thread's lane ID
 */
__device__ void cooperative_batch_inverse(
    const uint32_t* Z_batch,    // [batch_size, 8]
    uint32_t* Zinv_batch,       // [batch_size, 8]
    int batch_size,
    int lane_id
)
{
    // Use shared memory for conditional subtraction (per block)
    extern __shared__ uint32_t p256_cond_sub_scratch[];

    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    // SKELETON: Implement Montgomery's trick with warp-cooperative ops

    // Phase 1: Forward pass - compute prefix products
    // Each warp handles subset of batch
    // Store intermediate products in shared memory

    // Phase 2: Single inverse using Fermat's little theorem
    // z^-1 = z^(p-2) mod p
    // Use warp_mont_mul_comba for modular exponentiation

    // Phase 3: Backward pass - distribute inverse to all elements
    // Z[i]^-1 = (P[i-1] * P[N-1]^-1 * Z[i+1] * ... * Z[N-1])

    // TODO: Full implementation
    // For now, just copy input to output (placeholder)
    if (lane_id < 8) {
        int point_idx = blockIdx.x * num_warps + warp_id;
        if (point_idx < batch_size) {
            Zinv_batch[point_idx * 8 + lane_id] = Z_batch[point_idx * 8 + lane_id];
        }
    }
}

/*
 * Affine Recovery (Warp-Level)
 *
 * Computes: x = X * Z^-2 mod p
 *           y = Y * Z^-3 mod p
 *
 * Uses warp-cooperative field ops.
 *
 * Args:
 *   X[8], Y[8], Zinv[8]: Input Jacobian + inverse
 *   lane_id: Thread's lane ID
 *   x[8], y[8]: Output affine coordinates
 */
__device__ void affine_recover_warp(
    const uint32_t X[8],
    const uint32_t Y[8],
    const uint32_t Zinv[8],
    int lane_id,
    uint32_t x[8],
    uint32_t y[8]
)
{
    // Compute Zinv^2
    uint32_t Zinv2[8];
    if (lane_id < 8) {
        Zinv2[lane_id] = warp_mont_mul_comba(Zinv, Zinv, lane_id);
    }
    __syncwarp(0xFF);

    // Compute Zinv^3 = Zinv^2 * Zinv
    uint32_t Zinv3[8];
    if (lane_id < 8) {
        Zinv3[lane_id] = warp_mont_mul_comba(Zinv2, Zinv, lane_id);
    }
    __syncwarp(0xFF);

    // x = X * Zinv^2
    if (lane_id < 8) {
        x[lane_id] = warp_mont_mul_comba(X, Zinv2, lane_id);
    }

    // y = Y * Zinv^3
    if (lane_id < 8) {
        y[lane_id] = warp_mont_mul_comba(Y, Zinv3, lane_id);
    }
}

/*
 * Persistent Fused Kernel (Grid-Stride Loop)
 *
 * Each warp processes work items in a grid-stride pattern:
 *   my_work_idx = warp_id_g, warp_id_g + warps_g, warp_id_g + 2*warps_g, ...
 *
 * This deterministic pattern avoids atomic contention and always terminates.
 *
 * Flow:
 *   1. Calculate warp's global ID and total warps
 *   2. Grid-stride loop over work items
 *   3. Load scalar
 *   4. Fixed-base scalar mul → Jacobian (X, Y, Z)
 *   5. Affine conversion (placeholder - returns deterministic XOR pattern)
 *   6. Single global store
 *
 * Launch config:
 *   - Grid: Many blocks (e.g., 2x-4x SM count)
 *   - Block: 128 threads (4 warps) - must be multiple of 32
 *   - Each warp = 1 signature
 */
// CARD 13.I.2: Debug de-Montgomery helper (exact copy of CARD11 logic)
// Converts Montgomery domain per-lane limbs to normal domain limb 0
__device__ __forceinline__
void debug_demont_from_warp(
    uint32_t x_lane,      // This lane's limb of X
    uint32_t y_lane,      // This lane's limb of Y
    int lane_id,
    uint32_t &X0_norm,    // Output: limb 0 of de-Mont'd X
    uint32_t &Y0_norm)    // Output: limb 0 of de-Mont'd Y
{
    // Step 1: Gather all 8 limbs so each lane has the full value
    // (Exact copy of CARD11 lines 1207-1214)
    uint32_t x_R[8], y_R[8];  // Full values in R domain
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        x_R[i] = __shfl_sync(0xFFFFFFFF, x_lane, i);  // Broadcast lane i's limb to all
        y_R[i] = __shfl_sync(0xFFFFFFFF, y_lane, i);
    }

    // Step 2: Use shared memory (exact copy of CARD11 pattern)
    __shared__ uint32_t shared_x[32][8];  // [warp_id][limb]
    __shared__ uint32_t shared_y[32][8];

    const int warp_id = (threadIdx.x >> 5);

    // Step 3: Lane 0 performs de-Montgomery via scalar oracle
    if (lane_id == 0) {
        uint32_t x_out_scalar[8];
        uint32_t y_out_scalar[8];

        // De-Montgomery: multiply by 1 to convert R -> normal
        mont_mul_oracle_lane0(x_R, P256_ONE_LITERAL, x_out_scalar);
        mont_mul_oracle_lane0(y_R, P256_ONE_LITERAL, y_out_scalar);

        // Store to shared memory for distribution
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            shared_x[warp_id][i] = x_out_scalar[i];
            shared_y[warp_id][i] = y_out_scalar[i];
        }
    }

    // Step 4: Synchronize warp
    __syncwarp(0xFFu);

    // Step 5: All lanes read limb 0 (same as CARD11)
    X0_norm = shared_x[warp_id][0];
    Y0_norm = shared_y[warp_id][0];
}

// CARD 6 DEBUG: Temporarily remove launch bounds to test
__global__ void
persistent_ecdsa_sign_kernel(
    const int32_t* scalars_in,     // [total_work, 8] - input scalars (torch.int32)
    int32_t* pubkeys_out,          // [total_work, 2, 8] - output (x, y)
    int total_work                 // Total number of signatures to generate
)
{
#if WARP_COOP_DEBUG
    // CARD 6 DEBUG: First instruction in kernel
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        SCALAR_DBG("[CARD6-DEBUG] Kernel entry reached\n");
    }
#endif

    // Warp configuration
    int lane_id = threadIdx.x & 31;           // 0..31
    int warp_id_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // Global warp ID
    int warps_g = (gridDim.x * blockDim.x) >> 5;                   // Total warps

#if WARP_COOP_DEBUG && defined(PROD_PROBE)
    // Probe 1: CFG - Verify launch math & stride (one-shot, warp 0 only)
    if ((threadIdx.x & 31) == 0 && blockIdx.x == 0) {
        int warp_id_b = threadIdx.x >> 5;
        int warps_b   = blockDim.x >> 5;
        SCALAR_DBG("CFG: grid=%d block=%d warps_b=%d warps_g=%d warp_id_g=%d total_work=%d\n",
               gridDim.x, blockDim.x, warps_b, warps_g, warp_id_g, total_work);
    }
#endif

    // Per-warp storage (registers)
    uint32_t scalar[8];
    uint32_t x[8], y[8];           // Affine result (placeholder)

#if WARP_COOP_DEBUG && defined(PROD_PROBE)
    // Probe 3: PROG - Progress watchdog
    int iter = 0;
#endif

    // Grid-stride loop: each warp processes multiple work items
#if ONE_SHOT
    // ONE_SHOT mode: process exactly one job (my_work_idx = warp_id_g) and return
    if (warp_id_g < total_work) {
        int my_work_idx = warp_id_g;
#else
    for (int my_work_idx = warp_id_g; my_work_idx < total_work; my_work_idx += warps_g) {
#endif

        beacon(1);  // After work dispatch

#if WARP_COOP_DEBUG && defined(PROD_PROBE)
        // Probe 2: IN - Prove inputs are non-zero before load
        if ((threadIdx.x & 31) == 0 && blockIdx.x == 0) {
            int idx0 = my_work_idx;
            int idx1 = (idx0 + 1 < total_work) ? idx0 + 1 : idx0;

            // Read first 2 work items' scalars exactly how kernel does
            uint32_t k0_0 = ((const uint32_t*)scalars_in)[idx0*8 + 0];
            uint32_t k0_1 = ((const uint32_t*)scalars_in)[idx0*8 + 1];
            uint32_t k1_0 = ((const uint32_t*)scalars_in)[idx1*8 + 0];

            SCALAR_DBG("IN: idx0=%d k[0..1]=%08x %08x, idx1=%d k[0]=%08x\n",
                   idx0, k0_0, k0_1, idx1, k1_0);
        }
#endif

        // Load scalar (only lanes 0-7 load one limb each)
        uint32_t my_limb = 0;
        if (lane_id < 8) {
            // Cast from int32_t to uint32_t (reinterpret bits)
            my_limb = (uint32_t)scalars_in[my_work_idx * 8 + lane_id];
        }
        __syncwarp(0xFF);

        // CARD19 FIX: Gather all 8 limbs to ALL lanes (each lane needs full scalar)
        // Previously only lane_id's own limb was set, leaving limbs 1-7 as garbage
        // This caused 256-bit scalar failures (k >= 2^32)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scalar[i] = __shfl_sync(0xFF, my_limb, i);
        }

        beacon(2);  // After scalar load

#if P256_DBG
        // ====================================================================
        // DOMAIN DIAGNOSTIC PROBES (run once for work_idx==0)
        // ====================================================================
        if (my_work_idx == 0 && lane_id < 8) {
            // Probe 1: Montgomery identity operations
            // decode(1_mont) should equal 1_literal
            // 1_mont * 1_mont should equal 1_mont
            uint32_t t1_lane = warp_mont_mul_comba(P256_MONT_ONE, P256_ONE_LITERAL, lane_id);
            uint32_t t2_lane = warp_mont_mul_comba(P256_MONT_ONE, P256_MONT_ONE, lane_id);

            // Gather all 8 limbs via shuffles (inline gather for correctness)
            uint32_t t1[8], t2[8];
            if (lane_id < 8) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    t1[i] = __shfl_sync(0xFF, t1_lane, i);
                    t2[i] = __shfl_sync(0xFF, t2_lane, i);
                }
            }

            bool probe1a = vec_eq8(t1, P256_ONE_LITERAL, lane_id);
            bool probe1b = vec_eq8(t2, P256_MONT_ONE, lane_id);

            if (lane_id == 0) {
                SCALAR_DBG("[PROBE 1] Montgomery identity: decode(1_mont)==1_literal? %s, 1_mont*1_mont==1_mont? %s\n",
                       probe1a ? "YES" : "NO", probe1b ? "YES" : "NO");
                SCALAR_DBG("  t1 (decode(1_mont)) = %08x %08x %08x %08x %08x %08x %08x %08x\n",
                       t1[0], t1[1], t1[2], t1[3], t1[4], t1[5], t1[6], t1[7]);
                SCALAR_DBG("  Expected (1_literal) = %08x %08x %08x %08x %08x %08x %08x %08x\n",
                       P256_ONE_LITERAL[0], P256_ONE_LITERAL[1], P256_ONE_LITERAL[2], P256_ONE_LITERAL[3],
                       P256_ONE_LITERAL[4], P256_ONE_LITERAL[5], P256_ONE_LITERAL[6], P256_ONE_LITERAL[7]);
            }

            // Probe 2: Encode/decode roundtrip for Gx
            // enc = encode(Gx_norm) = mont_mul(Gx_norm, R2)
            // dec = decode(enc) = mont_mul(enc, 1_literal)
            // dec should equal Gx_norm
            uint32_t enc_lane = warp_mont_mul_comba(P256_Gx_NORMAL, P256_R2, lane_id);

            // Gather enc via shuffles
            uint32_t enc[8];
            if (lane_id < 8) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    enc[i] = __shfl_sync(0xFF, enc_lane, i);
                }
            }

            uint32_t dec_lane = warp_mont_mul_comba(enc, P256_ONE_LITERAL, lane_id);

            // Gather dec via shuffles
            uint32_t dec[8];
            if (lane_id < 8) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dec[i] = __shfl_sync(0xFF, dec_lane, i);
                }
            }

            bool probe2 = vec_eq8(dec, P256_Gx_NORMAL, lane_id);

            if (lane_id == 0) {
                SCALAR_DBG("[PROBE 2] Gx roundtrip: encode→decode==Gx_norm? %s\n",
                       probe2 ? "YES" : "NO");
            }

            // Probe 3: Table entry 0 domain check
            // Load table entry 0 (should be G)
            // Decode it: Px_dec = mont_mul(Px, 1_literal)
            // If Px_dec == Gx_norm → table is in Montgomery domain
            // If Px == Gx_norm → table is in normal domain
            uint32_t Px[8], Py[8];
            soa_lookup_bucket(0, Px, Py, lane_id);

            uint32_t Px_dec_lane = warp_mont_mul_comba(Px, P256_ONE_LITERAL, lane_id);

            // Gather Px_dec via shuffles
            uint32_t Px_dec[8];
            if (lane_id < 8) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    Px_dec[i] = __shfl_sync(0xFF, Px_dec_lane, i);
                }
            }

            bool table_is_montgomery = vec_eq8(Px_dec, P256_Gx_NORMAL, lane_id);
            bool table_is_normal = vec_eq8(Px, P256_Gx_NORMAL, lane_id);

            if (lane_id == 0) {
                SCALAR_DBG("[PROBE 3] Table entry[0]: decode(Px)==Gx_norm? %s, Px==Gx_norm? %s\n",
                       table_is_montgomery ? "YES (table is Mont)" : "NO",
                       table_is_normal ? "YES (table is Normal)" : "NO");
            }
        }
#endif

        // ====================================================================
        // Scalar Multiplication with Oracle-Backed Montgomery Multiply
        // ====================================================================
        // Re-enabled with warp_mont_mul_comba replaced by oracle-backed version
        // to isolate whether hang is in warp-cooperative multiply or elsewhere
        //
        // TEMPORARY: Using warp_mont_mul_oracle_compat for all field multiplies
        // This uses the validated lane-0 oracle internally, avoiding warp-coop bugs
        // ====================================================================
        #define warp_mont_mul_comba warp_mont_mul_oracle_compat

        // ====================================================================
        // BISECTION GUARD #0: Verify kernel launch (return immediately)
        // ====================================================================
        #if 0  // DISABLED - Guard #0 passed, moving to Guard #1
        return;
        #endif

        constexpr int W = 7;  // Window width (must match table generation)
        constexpr int MAXL = (256 + W - 1) / W;  // 37 windows for W=7
        const unsigned FULL = 0xFFFFFFFFu;
        const unsigned LANES8 = 0xFFu;

        // Lane 0 does recoding into local array
        int8_t digits_loc[MAXL];
        int L = 0;
        if (lane_id == 0) {
            L = recode_signed_fixed_window<W, MAXL>(scalar, digits_loc);
        }
        // Broadcast L to all lanes
        L = __shfl_sync(FULL, L, 0);

        // ====================================================================
        // CARD B2: Dump window digits for k=1,2,3 (debug only)
        // ====================================================================
        if (my_work_idx == 0 && lane_id == 0) {
            SCALAR_DBG("[WARP DIGITS] k[7..0] =");
            for (int i = 7; i >= 0; --i) {
                SCALAR_DBG(" %08x", scalar[i]);
            }
            SCALAR_DBG(" (L=%d windows)\n", L);

            for (int j = L - 1; j >= 0; --j) {
                SCALAR_DBG("  j=%2d digit=%3d\n", j, (int)digits_loc[j]);
            }
        }

        // ====================================================================
        // BISECTION GUARD #1: After scalar recoding
        // ====================================================================
        #if 0  // DISABLED - Guard #1 passed, moving to Guard #2
        return;
        #endif

        // Initialize accumulator to point at infinity
        // CRITICAL: All 32 lanes must initialize their local arrays!
        // Each lane has its own stack copy, and lanes 8-31 participate in warp ops
        uint32_t X[8], Y[8], Z[8];
        for (int i = 0; i < 8; ++i) {
            X[i] = 0;
            Y[i] = 0;
            Z[i] = 0;
        }

        beacon(3);  // Before window loop

        // ====================================================================
        // DEBUG: TABLE DOMAIN PASS-THROUGH TEST
        // Emit G directly from table (entry 0) to verify Montgomery domain
        // ====================================================================
        #if 0  // Set to 0 to disable (CONFIRMED: Table is Montgomery domain!)
        if (my_work_idx == 0) {
            // Load entry 0 from table (should be G)
            uint32_t table_x[8], table_y[8];
            soa_lookup_bucket(0, table_x, table_y, lane_id);

            // Write directly to affine result arrays (no EC math, no conversion)
            if (lane_id < 8) {
                x[lane_id] = table_x[lane_id];
                y[lane_id] = table_y[lane_id];
            }

            if (lane_id == 0) {
                SCALAR_DBG("[TABLE-TEST] Emitting raw table entry 0 (should be G)\n");
            }

            // Skip all EC math and affine conversion - go straight to writeback
            goto writeback_output;
        }
        #endif

        // ====================================================================
        // BISECTION STEP 3: TABLE LOOKUP + DE-MONTGOMERY ONLY
        // ====================================================================
        // Test minimal path: table entry 0 → de-Montgomery → return
        // This isolates table lookup and oracle multiply without ladder/EC ops
        // Expected: Should return G in affine domain if both work correctly
        // If hangs: Bug is in table lookup or oracle multiply calls
        // If returns: Bug is in ladder loop / jacobian operations
        // ====================================================================
        #if 0  // DISABLED - gather/broadcast fix verified, now test full scalar mul
        {
            // Load table entry 0 (should be G in Montgomery domain)
            uint32_t table_x[8], table_y[8];
            soa_lookup_bucket(0, table_x, table_y, lane_id);

            if (lane_id == 0) {
                SCALAR_DBG("[BISECT-3] Loaded table entry 0, de-Montgomerying...\n");
                SCALAR_DBG("[TABLE-X] ");
                for (int i = 0; i < 8; i++) SCALAR_DBG("%08x ", table_x[i]);
                SCALAR_DBG("\n[TABLE-Y] ");
                for (int i = 0; i < 8; i++) SCALAR_DBG("%08x ", table_y[i]);
                SCALAR_DBG("\n");
            }

            // De-Montgomery X using oracle multiply (X * 1 mod p)
            // CRITICAL: All lanes must call for __syncwarp() inside oracle
            uint32_t x_affine = warp_mont_mul_comba(table_x, P256_ONE_LITERAL, lane_id);

            if (lane_id < 8) {
                SCALAR_DBG("[X-DEMONT lane%d] in=0x%08x out=0x%08x\n", lane_id, table_x[lane_id], x_affine);
            }

            // De-Montgomery Y using oracle multiply (Y * 1 mod p)
            uint32_t y_affine = warp_mont_mul_comba(table_y, P256_ONE_LITERAL, lane_id);

            if (lane_id < 8) {
                SCALAR_DBG("[Y-DEMONT lane%d] in=0x%08x out=0x%08x\n", lane_id, table_y[lane_id], y_affine);
            }

            // Write to output
            if (lane_id < 8) {
                x[lane_id] = x_affine;
                y[lane_id] = y_affine;
            }

            if (lane_id == 0) {
                SCALAR_DBG("[BISECT-3] Output written, jumping to writeback\n");
            }

            // Skip all ladder/EC math - go straight to writeback
            goto writeback_output;
        }
        #endif

        // ====================================================================
        // BISECTION STEP 5: TEST SINGLE WINDOW OF LADDER
        // ====================================================================
        // Run only the first window (j = L-1): 7 doublings + 1 add
        // This isolates whether hang is in jacobian_double or jacobian_add
        // ====================================================================
        #if 0  // DISABLED - Single window works! Now testing FULL ladder
        {
            int j = L - 1;  // Topmost window

            // Get digit for this window
            int d_j = 0;
            if (lane_id == 0) {
                d_j = (int)digits_loc[j];
            }
            d_j = __shfl_sync(0xFFFFFFFF, d_j, 0);  // Broadcast to all lanes

            // ====================================================================
            // BISECTION GUARD #2: After entering single-window block + getting d_j
            // ====================================================================
            #if 0  // DISABLED - Guard #2 passed, moving to Guard #3
            return;
            #endif

            // 7 doublings (W=7) - BUT TEST JUST THE FIRST ONE
            jacobian_double_warp(X, Y, Z, lane_id);

            // ====================================================================
            // BISECTION GUARD #3: After FIRST jacobian_double_warp call
            // ====================================================================
            #if 0  // DISABLED - Guard #3 passed, moving to Guard #4
            return;
            #endif

            // Remaining 6 doublings - BUT TEST ONLY THE SECOND ONE
            jacobian_double_warp(X, Y, Z, lane_id);  // k=1 (second doubling)

            // ====================================================================
            // BISECTION GUARD #4b: After SECOND jacobian_double_warp (total 2 doublings)
            // ====================================================================
            #if 0  // DISABLED - Hang is fixed! Now testing full ladder
            return;
            #endif

            for (int k = 2; k < 7; ++k) {
                jacobian_double_warp(X, Y, Z, lane_id);
            }

            // ====================================================================
            // BISECTION GUARD #4: After ALL 7 doublings complete
            // ====================================================================
            #if 0  // DISABLED - 7 doublings work! Now testing add
            return;
            #endif

            if (lane_id == 0) {
                SCALAR_DBG("[BISECT-5] All 7 doublings complete. Testing for add...\n");
            }

            // Conditional add (if d_j != 0)
            const bool z_is_zero = warp_is_zero(Z, lane_id);

            if (d_j != 0) {
                if (lane_id == 0) {
                    SCALAR_DBG("[BISECT-5] d_j=%d, loading table entry and calling add...\n", d_j);
                }

                // Load table entry (same as real ladder)
                int entry = (d_j > 0) ? d_j : -d_j;
                uint32_t Px_mont[8], Py_mont[8];
                soa_lookup_bucket(entry, Px_mont, Py_mont, lane_id);

                if (z_is_zero) {
                    // Seed accumulator
                    for (int i = 0; i < 8; ++i) {
                        X[i] = Px_mont[i];
                        Y[i] = Py_mont[i];
                        Z[i] = P256_MONT_ONE[i];
                    }
                    if (lane_id == 0) {
                        SCALAR_DBG("[BISECT-5] Seeded accumulator (Z was zero)\n");
                    }
                } else {
                    // Normal add
                    if (lane_id == 0) {
                        SCALAR_DBG("[BISECT-5] Calling jacobian_add_affine_warp...\n");
                    }
                    jacobian_add_affine_warp(X, Y, Z, Px_mont, Py_mont, lane_id);
                    if (lane_id == 0) {
                        SCALAR_DBG("[BISECT-5] jacobian_add_affine_warp complete\n");
                    }
                }
            } else {
                if (lane_id == 0) {
                    SCALAR_DBG("[BISECT-5] d_j=0, skipping add\n");
                }
            }

            if (lane_id == 0) {
                SCALAR_DBG("[BISECT-5] Single-window ladder complete, jumping to affine\n");
            }

            // Jump to affine conversion
            goto test_affine_conversion;
        }
        #endif

        // Process windows from high to low
        int iter_guard = 0;

        // Accumulator starts at infinity -> not initialized
        bool initialized = false;

        // LT-2: Ladder trace step counter (only meaningful for warp 0, lane 0, sc_idx 0)
        int trace_step = 0;

        for (int j = L - 1; j >= 0; --j) {
            // Safety: Guard against infinite loops
            if (++iter_guard > 64) {
                // DISABLED for performance
                // if (lane_id == 0) printf("[GUARD HIT] iter_guard exceeded 64\n");
                break;
            }

            beacon(4);  // Start of window loop iteration
            // Broadcast digit d_j from lane 0
            int d_j = 0;
            if (lane_id == 0) {
                d_j = (int)digits_loc[j];
            }
            d_j = __shfl_sync(FULL, d_j, 0);

            // ====================================================================
            // Correct window schedule:
            //   - Only do W doublings AFTER accumulator has been initialized
            //   - First non-zero window seeds WITHOUT prior doublings
            // ====================================================================
            if (initialized) {
                #pragma unroll
                for (int k = 0; k < W; ++k) {
                    assert_full_lane_mask(lane_id);
                    jacobian_double_warp(X, Y, Z, lane_id);
                }

                // LT-2: Log doubling operation (after W doublings)
                // CARD13J: Print raw Montgomery domain (no de-Mont conversion)
                LADDER_TRACE(trace_step, j, 'D', X[0], Y[0], Z[0], warp_id_g, lane_id, my_work_idx);
                trace_step++;
            }
            // If !initialized, we intentionally skip doublings here

            beacon(5);  // After doublings (or no-op if not initialized yet)

            // ====================================================================
            // Warp-Uniform Constant-Time Bucket Selection
            // ====================================================================
            // CRITICAL: All 32 lanes must execute the same path to avoid divergence
            // Use constant-time selection instead of branching

            // Compute bucket index for all-multiples table
            // entry = |d_j| - 1, where table[0] = 1G, table[1] = 2G, etc.
            // For d_j = 0, entry = -1 (no bucket)
            int entry = (d_j == 0) ? -1 : (abs(d_j) - 1);

            // Look up point from comb table (Montgomery domain)
            // ALL lanes call this (warp-uniform)
            uint32_t Px_mont[8], Py_mont[8];
            soa_lookup_bucket(entry, Px_mont, Py_mont, lane_id);

            // GPU-Y-0-002: Inspect bucket output BEFORE any operations
            if (my_work_idx == 0 && j == (L - 1) && lane_id == 0) {
                SCALAR_DBG("[GPU-Y-0-002 BUCKET] Px_mont[0]=%08x Py_mont[0]=%08x entry=%d d_j=%d\n",
                       Px_mont[0], Py_mont[0], entry, d_j);
            }

            // Negate Y if digit is negative (in Montgomery domain)
            // CRITICAL: All 32 lanes call warp_negate_u32 (warp-uniform execution)
            uint32_t Py_selected = Py_mont[lane_id];
            if (d_j < 0) {
                // All lanes participate in negate
                uint32_t neg_y = warp_negate_u32(Py_mont[lane_id], lane_id);
                if (lane_id < 8) {
                    Py_selected = neg_y;
                }
            }

            // Update Py array with selected value
            if (lane_id < 8) {
                Py_mont[lane_id] = Py_selected;
            }

            // ====================================================================
            // First non-zero window seeds accumulator; later non-zero windows add.
            // ====================================================================
            if (d_j != 0) {
                if (!initialized) {
                    // First non-zero digit: seed accumulator with table point
                    if (my_work_idx == 0 && lane_id == 0) {
                        SCALAR_DBG("[SEED] j=%d d_j=%d entry=%d seeding from table\n",
                               j, d_j, entry);
                    }

                    // Seed accumulator with table point (Montgomery domain)
                    // CRITICAL: Each thread must initialize ALL 8 limbs of its local arrays!
                    for (int i = 0; i < 8; ++i) {
                        X[i] = Px_mont[i];
                        Y[i] = Py_mont[i];
                        Z[i] = P256_MONT_ONE[i];  // Z = 1 in Montgomery
                    }

                    // GPU-Y-0-001: Verify all limbs written correctly
                    if (my_work_idx == 0 && lane_id == 0) {
                        SCALAR_DBG("[GPU-Y-0-001 SEED] X[0]=%08x Y[0]=%08x Z[0]=%08x Z[7]=%08x\n",
                               X[0], Y[0], Z[0], Z[7]);
                        SCALAR_DBG("[GPU-Y-0-001 SEED] Expected: Y[0] != 0, Z[0] = 0x00000001, Z[7] = 0x00000000\n");
                    }

                    initialized = true;

                    // LT-2: Log seed operation
                    // CARD13J: Print raw Montgomery domain (no de-Mont conversion)
                    LADDER_TRACE(trace_step, j, 'S', X[0], Y[0], Z[0], warp_id_g, lane_id, my_work_idx);
                    trace_step++;
                } else {
                    // Normal case: Add table point to accumulator
                    // Both are in Montgomery domain, jacobian_add_affine_warp expects this
                    assert_full_lane_mask(lane_id);
                    jacobian_add_affine_warp(X, Y, Z, Px_mont, Py_mont, lane_id);

                    // BREADCRUMB 2: After bucket selection/addition
                    if (my_work_idx == 0 && lane_id == 0) {
                        SCALAR_DBG("[BUCKET] y0=%08x\n", Y[0]);
                    }

                    // LT-2: Log add operation
                    // CARD13J: Print raw Montgomery domain (no de-Mont conversion)
                    LADDER_TRACE(trace_step, j, 'A', X[0], Y[0], Z[0], warp_id_g, lane_id, my_work_idx);
                    trace_step++;
                }
            }
            // else: d_j == 0, nothing to add (constant-time-ish no-op; table lookup still ran)

            beacon(6);  // After point addition / seeding
        }

        beacon(7);  // After window loop

        // ====================================================================
        // Affine Conversion with Fast Path and De-Montgomery
        // ====================================================================

        test_affine_conversion:  // Label for bisection test

        beacon(8);  // Before affine conversion

        // ====================================================================
        // SCALAR AFFINE FALLBACK (for debugging)
        // Use scalar reference for affine conversion to isolate if bug is
        // in Jacobian coords vs warp CIOS affine conversion
        // ====================================================================
        #define USE_SCALAR_AFFINE_FALLBACK 1

        #if USE_SCALAR_AFFINE_FALLBACK
        {
            // Gather Jacobian coordinates to lane 0
            uint32_t X_jac[8], Y_jac[8], Z_jac[8];
            uint32_t x_local = (lane_id < 8) ? X[lane_id] : 0u;
            uint32_t y_local = (lane_id < 8) ? Y[lane_id] : 0u;
            uint32_t z_local = (lane_id < 8) ? Z[lane_id] : 0u;

            for (int i = 0; i < 8; i++) {
                uint32_t vx = (lane_id == i && lane_id < 8) ? x_local : 0u;
                uint32_t vy = (lane_id == i && lane_id < 8) ? y_local : 0u;
                uint32_t vz = (lane_id == i && lane_id < 8) ? z_local : 0u;
                vx = __shfl_sync(0xFFFFFFFF, vx, i);
                vy = __shfl_sync(0xFFFFFFFF, vy, i);
                vz = __shfl_sync(0xFFFFFFFF, vz, i);
                if (lane_id == 0) {
                    X_jac[i] = vx;
                    Y_jac[i] = vy;
                    Z_jac[i] = vz;
                }
            }

            __syncwarp(0xFFFFFFFF);

            // Lane 0 does scalar affine conversion
            uint32_t x_aff[8], y_aff[8];
            if (lane_id == 0) {
                // Check for Z = MONT_ONE fast path
                bool is_mont_one_scalar = true;
                for (int i = 0; i < 8; i++) {
                    if (Z_jac[i] != P256_MONT_ONE[i]) {
                        is_mont_one_scalar = false;
                        break;
                    }
                }

                if (is_mont_one_scalar) {
                    // Fast path: just de-Montgomery X and Y
                    uint32_t one_lit[8] = {1,0,0,0,0,0,0,0};
                    mont_mul_scalar(x_aff, X_jac, one_lit);
                    mont_mul_scalar(y_aff, Y_jac, one_lit);
                } else {
                    // Full path: Fermat inverse + affine recovery
                    // Compute Z^-1 = Z^(p-2) mod p using Fermat's Little Theorem
                    uint32_t Zinv[8], Z2[8], Z3[8], x_mont[8], y_mont[8];

                    // Fermat inverse: Z^-1 = Z^(p-2) mod p using binary exponentiation
                    // For P-256, p-2 = 0xffffffff00000001000000000000000000000000fffffffffffffffffffffffd
                    uint32_t base[8], result[8];
                    for (int i = 0; i < 8; i++) base[i] = Z_jac[i];
                    for (int i = 0; i < 8; i++) result[i] = P256_MONT_ONE[i];  // result = 1 (Montgomery)

                    // p-2 in little-endian limbs
                    const uint32_t p_minus_2[8] = {
                        0xfffffffd, 0xffffffff, 0xffffffff, 0x00000000,
                        0x00000000, 0x00000000, 0x00000001, 0xffffffff
                    };

                    // Binary exponentiation: scan bits of p-2 from LSB to MSB
                    for (int limb_idx = 0; limb_idx < 8; limb_idx++) {
                        uint32_t exponent_limb = p_minus_2[limb_idx];
                        for (int bit = 0; bit < 32; bit++) {
                            // If bit is set, multiply result by base
                            if (exponent_limb & (1u << bit)) {
                                uint32_t temp[8];
                                mont_mul_scalar(temp, result, base);
                                for (int i = 0; i < 8; i++) result[i] = temp[i];
                            }
                            // Square base for next bit
                            if (limb_idx < 7 || bit < 31) {  // Don't square on last iteration
                                uint32_t temp[8];
                                mont_mul_scalar(temp, base, base);
                                for (int i = 0; i < 8; i++) base[i] = temp[i];
                            }
                        }
                    }

                    // Copy result to Zinv
                    for (int i = 0; i < 8; i++) Zinv[i] = result[i];

                    // Z2 = Zinv^2
                    mont_mul_scalar(Z2, Zinv, Zinv);

                    // Z3 = Z2 * Zinv
                    mont_mul_scalar(Z3, Z2, Zinv);

                    // x_mont = X * Z^-2
                    mont_mul_scalar(x_mont, X_jac, Z2);

                    // y_mont = Y * Z^-3
                    mont_mul_scalar(y_mont, Y_jac, Z3);

                    // De-Montgomery
                    uint32_t one_lit[8] = {1,0,0,0,0,0,0,0};
                    mont_mul_scalar(x_aff, x_mont, one_lit);
                    mont_mul_scalar(y_aff, y_mont, one_lit);
                }

                SCALAR_DBG("[SCALAR-AFFINE] x[0]=%08x y[0]=%08x (Z_is_one=%d)\n",
                       x_aff[0], y_aff[0], is_mont_one_scalar);
            }

            __syncwarp(0xFFFFFFFF);

            // Broadcast results to all lanes
            uint32_t x_result_lane = 0, y_result_lane = 0;
            for (int i = 0; i < 8; i++) {
                uint32_t vx = 0, vy = 0;
                if (lane_id == 0) {
                    vx = x_aff[i];
                    vy = y_aff[i];
                }
                vx = __shfl_sync(0xFFFFFFFF, vx, 0);
                vy = __shfl_sync(0xFFFFFFFF, vy, 0);
                if (i == lane_id && lane_id < 8) {
                    x_result_lane = vx;
                    y_result_lane = vy;
                }
            }

            if (lane_id < 8) {
                x[lane_id] = x_result_lane;
                y[lane_id] = y_result_lane;
            }

            goto writeback_output;  // Skip warp affine conversion
        }
        #endif

        // Mask sanity check: ensure lanes 0-7 are all active before affine
        unsigned m = __activemask();
        // DISABLED for performance
        // if (threadIdx.x % 32 == 0 && (m & 0xFFu) != 0xFFu) {
        //     printf("[MASK BUG] activemask=0x%08x before affine\n", m);
        // }

        // CARD 9: Capture PRE-affine state (Jacobian coordinates)
        if (lane_id < 8) {
            if (lane_id == 0) {
                SCALAR_DBG("[CARD9-PRE] work_idx=%d Z[0]=%08x X[0]=%08x Y[0]=%08x\n",
                       my_work_idx, Z[0], X[0], Y[0]);
            }
            dbg_Z[lane_id] = Z[lane_id];
            dbg_X_before[lane_id] = X[lane_id];
            dbg_Y_before[lane_id] = Y[lane_id];
        }

        // Fast path: If Z == P256_MONT_ONE, then Z^{-1} == P256_MONT_ONE
        // So x_affine = de_montgomery(X), y_affine = de_montgomery(Y)
        const bool is_mont_one = warp_equals_mont_one(Z, lane_id);

        if (is_mont_one) {
            // Fast path: Just de-Montgomery X and Y
            // GPU-Y-0-003: Trace de-Montgomery step by step with ALL limbs
            if (my_work_idx == 0 && lane_id < 8) {
                SCALAR_DBG("[PRE-DEMONT lane%d] X[%d]=%08x Y[%d]=%08x\n",
                       lane_id, lane_id, X[lane_id], lane_id, Y[lane_id]);
            }

            // CARD 12: Log 1st de-Mont inputs
            if (my_work_idx == 0 && lane_id == 0) {
                SCALAR_DBG("[CARD12-1ST-DEMONT-PRE] X[0]=%08x P256_ONE_LITERAL[0]=%08x\n",
                       X[0], P256_ONE_LITERAL[0]);
            }

                        // CRITICAL: All lanes (0-31) must call warp_mont_mul_comba() for __syncwarp() inside
            uint32_t x_result = warp_mont_mul_comba(X, P256_ONE_LITERAL, lane_id);

            // OPTION A TEST: Comment out printf to test if crash is printf-related
            // if (my_work_idx == 0 && lane_id < 8) {
            //     printf("[POST-X-DEMONT lane%d] x_result=%08x Y[%d]=%08x\n",
            //            lane_id, x_result, lane_id, Y[lane_id]);
            // }

            // Phase 3 instrumentation: Heartbeats to verify warp is intact after X
            warpdbg_heartbeat("POST-X-DEMONT-SAFE");
            warpdbg_heartbeat("PRE-Y-DEMONT");

            uint32_t y_result = warp_mont_mul_comba(Y, P256_ONE_LITERAL, lane_id);

            if (my_work_idx == 0 && lane_id < 8) {
                SCALAR_DBG("[POST-Y-DEMONT lane%d] x_result=%08x y_result=%08x\n",
                       lane_id, x_result, y_result);
            }
            if (lane_id < 8) {
                x[lane_id] = x_result;
                y[lane_id] = y_result;
            }

            // CARD17: Capture after 1st de-Montgomery (now NORMAL domain with fixed table)
            if (lane_id < 8) {
                dbg_X_demont1[lane_id] = x_result;
                dbg_Y_demont1[lane_id] = y_result;
            }
            if (lane_id == 0) SCALAR_DBG("[CARD11-1ST-DEMONT] x[0]=%08x y[0]=%08x\n", x[0], y[0]);

            if (lane_id == 0) SCALAR_DBG("[AFF FAST] stored\n");
            beacon(9);  // After fast path
        } else {
            // Full path: Compute Z^{-1}, then affine, then de-Montgomery
            // DISABLED for performance
            // if (lane_id == 0) printf("[AFFINE FULL PATH] Z != MONT_ONE\n");
            beacon(9);  // Before Fermat inverse

            // Compute Z^{-1} using Fermat's Little Theorem: Z^{-1} = Z^{p-2} mod p
            uint32_t Zinv[8];
            assert_full_lane_mask(lane_id);
            warp_fermat_inverse_p256(Z, Zinv, lane_id);

            // CARD 9: Capture Z^-1
            if (lane_id < 8) {
                if (lane_id == 0) SCALAR_DBG("[CARD9-ZINV] work_idx=%d Zinv[0]=%08x\n", my_work_idx, Zinv[0]);
                dbg_Zinv[lane_id] = Zinv[lane_id];
            }

            beacon(10);  // After Fermat inverse

            // Compute Z2 = Zinv^2
            // CRITICAL: All lanes call warp_mont_mul_comba, only lanes 0-7 store
            uint32_t z2_tmp = warp_mont_mul_comba(Zinv, Zinv, lane_id);
            uint32_t Z2[8];
            if (lane_id < 8) {
                Z2[lane_id] = z2_tmp;
            }

            // CARD 9: Capture Z^-2
            if (lane_id < 8) {
                if (lane_id == 0) SCALAR_DBG("[CARD9-ZINV2] work_idx=%d Zinv2[0]=%08x\n", my_work_idx, Z2[0]);
                dbg_Zinv2[lane_id] = Z2[lane_id];
            }

            // Compute Z3 = Z2 * Zinv
            // CRITICAL: All lanes call warp_mont_mul_comba, only lanes 0-7 store
            uint32_t z3_tmp = warp_mont_mul_comba(Z2, Zinv, lane_id);
            uint32_t Z3[8];
            if (lane_id < 8) {
                Z3[lane_id] = z3_tmp;
            }

            // CARD 9: Capture Z^-3
            if (lane_id < 8) {
                if (lane_id == 0) SCALAR_DBG("[CARD9-ZINV3] work_idx=%d Zinv3[0]=%08x\n", my_work_idx, Z3[0]);
                dbg_Zinv3[lane_id] = Z3[lane_id];
            }

            // Affine recovery (still in Montgomery): x_mont = X * Z^{-2}, y_mont = Y * Z^{-3}
            // CRITICAL: All lanes call warp_mont_mul_comba, only lanes 0-7 store
            uint32_t x_mont_tmp = warp_mont_mul_comba(X, Z2, lane_id);
            uint32_t y_mont_tmp = warp_mont_mul_comba(Y, Z3, lane_id);
            uint32_t x_mont[8], y_mont[8];
            if (lane_id < 8) {
                x_mont[lane_id] = x_mont_tmp;
                y_mont[lane_id] = y_mont_tmp;
            }

            // CARD 9: Capture POST-affine state (before de-Montgomery)
            if (lane_id < 8) {
                if (lane_id == 0) SCALAR_DBG("[CARD9-POST] work_idx=%d X_after[0]=%08x Y_after[0]=%08x\n",
                                        my_work_idx, x_mont[0], y_mont[0]);
                dbg_X_after[lane_id] = x_mont[lane_id];
                dbg_Y_after[lane_id] = y_mont[lane_id];
            }

            // De-Montgomery: convert to normal field elements
            if (lane_id == 0) SCALAR_DBG("[AFF FULL] about to de-Montgomery\n");
            // CRITICAL: All lanes call warp_mont_mul_comba, only lanes 0-7 store
            uint32_t x_tmp = warp_mont_mul_comba(x_mont, P256_ONE_LITERAL, lane_id);
            uint32_t y_tmp = warp_mont_mul_comba(y_mont, P256_ONE_LITERAL, lane_id);
            if (lane_id == 0) SCALAR_DBG("[AFF FULL] x0=%08x y0=%08x\n", x_tmp, y_tmp);
            if (lane_id < 8) {
                x[lane_id] = x_tmp;
                y[lane_id] = y_tmp;
            }
            if (lane_id == 0) SCALAR_DBG("[AFF FULL] stored\n");
        }

        #undef warp_mont_mul_comba  // End of oracle-backed multiply override

        // ====================================================================
        // Store Output (Normal Field Elements - De-Montgomerized)
        // ====================================================================

    writeback_output:  // Label for table domain pass-through test

#if WARP_COOP_DEBUG && defined(PROD_PROBE)
        // Probe 4: OUT - Verify outputs before writeback
        if ((threadIdx.x & 31) == 0 && blockIdx.x == 0 && my_work_idx == 0) {
            SCALAR_DBG("OUT: idx0 x[0..1]=%08x %08x y[0..1]=%08x %08x\n",
                   x[0], x[1], y[0], y[1]);
        }
#endif

        // ====================================================================
        // CARD17 FIX: Table is now in R domain (single Montgomery)
        // ====================================================================
        // OLD (BUGGY):
        //   Table stored points in R^2 domain (a*R^2 mod p)
        //   First demont gave R domain, needed second demont
        // NEW (FIXED):
        //   Table stores points in R domain (a*R mod p)
        //   First demont gives NORMAL domain directly - no second demont needed
        //
        // After first de-Montgomery, x[lane_id] and y[lane_id] are already
        // in normal (standard) field element representation.


        // CARD17: Debug capture (x[] and y[] are now in normal domain directly)
        if (lane_id < 8) {
            dbg_X_demont2[lane_id] = x[lane_id];  // CARD17: Now same as 1st de-Mont result
            dbg_Y_demont2[lane_id] = y[lane_id];
        }
        if (lane_id == 0) SCALAR_DBG("[CARD17-OUTPUT] x[0]=%08x y[0]=%08x (normal domain)\n",
                                x[0], y[0]);

        if (lane_id < 8) {
            // Cast from uint32_t to int32_t (reinterpret bits)
            // CARD17: x[lane_id] is already in normal domain after first de-Mont
            pubkeys_out[my_work_idx * 16 + lane_id] = (int32_t)x[lane_id];         // x
            pubkeys_out[my_work_idx * 16 + 8 + lane_id] = (int32_t)y[lane_id];     // y
        }

        beacon(11);  // After writeback

#if WARP_COOP_DEBUG && defined(PROD_PROBE)
        // Probe 3: PROG - Progress and watchdog
        if ((threadIdx.x & 31) == 0 && blockIdx.x == 0) {
            if ((iter & 0x3FF) == 0) {  // Every 1024 iterations
                SCALAR_DBG("PROG: warp0 idx=%d/%d iter=%d\n", my_work_idx, total_work, iter);
            }
            iter++;
            if (iter > (total_work + warps_g) * 4) {
                SCALAR_DBG("WDOG: abort warp0 idx=%d iter=%d (runaway)\n", my_work_idx, iter);
                return;  // Abort kernel (works in both ONE_SHOT and loop modes)
            }
        }
#endif

#if ONE_SHOT
    }  // Close ONE_SHOT if block
#else
    }  // Close grid-stride for loop
#endif
}

/*
 * Load Comb Table from .npy Files
 *
 * Reads precomputed comb table and uploads to GPU constant memory.
 * Call this once at initialization.
 *
 * Args:
 *   x_path: Path to X coordinates .npy file
 *   y_path: Path to Y coordinates .npy file
 *   w: Window width (must match table generation)
 */
void load_comb_table_from_tensors(torch::Tensor x_table, torch::Tensor y_table, int w)
{
    TORCH_CHECK(x_table.dim() == 2, "X table must be 2D [8, entries]");
    TORCH_CHECK(y_table.dim() == 2, "Y table must be 2D [8, entries]");
    TORCH_CHECK(x_table.size(0) == 8, "X table must have 8 limbs");
    TORCH_CHECK(y_table.size(0) == 8, "Y table must have 8 limbs");

    int entries = x_table.size(1);
    TORCH_CHECK(entries == ((1 << w) - 1), "Entries must match 2^w - 1 (all multiples)");

    // Ensure tables are on CPU and contiguous
    auto x_cpu = x_table.cpu().contiguous();
    auto y_cpu = y_table.cpu().contiguous();

    // Flatten for cudaMemcpyToSymbol (expects 1D array)
    auto x_flat = x_cpu.flatten();
    auto y_flat = y_cpu.flatten();

    // Upload to GPU constant memory
    cudaError_t err;
    err = cudaMemcpyToSymbol(P256_G_TABLE_X_SOA, x_flat.data_ptr<uint32_t>(),
                             8 * entries * sizeof(uint32_t));
    TORCH_CHECK(err == cudaSuccess, "Failed to upload X table: ", cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(P256_G_TABLE_Y_SOA, y_flat.data_ptr<uint32_t>(),
                             8 * entries * sizeof(uint32_t));
    TORCH_CHECK(err == cudaSuccess, "Failed to upload Y table: ", cudaGetErrorString(err));

    // Set window parameters
    int fixed_w = w;
    err = cudaMemcpyToSymbol(P256_FIXED_W, &fixed_w, sizeof(int));
    TORCH_CHECK(err == cudaSuccess, "Failed to set P256_FIXED_W: ", cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(P256_ENTRIES, &entries, sizeof(int));
    TORCH_CHECK(err == cudaSuccess, "Failed to set P256_ENTRIES: ", cudaGetErrorString(err));
}

/*
 * Host-side launcher
 *
 * Sets up persistent kernel with appropriate grid/block config.
 */
torch::Tensor persistent_ecdsa_sign_batch(
    torch::Tensor scalars  // [batch_size, 8] int32
)
{
    TORCH_CHECK(scalars.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(scalars.dtype() == torch::kInt32, "Input must be int32");
    TORCH_CHECK(scalars.dim() == 2, "Input must be 2D [batch_size, 8]");
    TORCH_CHECK(scalars.size(1) == 8, "Input must have 8 limbs");

    int batch_size = scalars.size(0);

    // Allocate output: [batch_size, 2, 8] for (x, y)
    auto output = torch::empty({batch_size, 2, 8}, scalars.options());

    // Grid/block configuration
    int threads_per_block = 128;  // 4 warps per block (must be multiple of 32)
    int warps_per_block = threads_per_block / 32;

    // CARD20 FIX: Launch enough blocks to cover all work items
    // With ONE_SHOT=1, each warp processes exactly one item (no grid-stride loop)
    // So we need ceil(batch_size / warps_per_block) blocks minimum
    int num_blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    // Calculate shared memory needed for conditional subtraction
    // Each warp needs 8 limbs * 4 bytes = 32 bytes
    // 128 threads = 4 warps per block → 4 * 32 = 128 bytes
    int shared_mem_bytes = warps_per_block * 8 * sizeof(uint32_t);  // 128 bytes

    // Launch persistent kernel with shared memory for conditional subtraction
    persistent_ecdsa_sign_kernel<<<num_blocks, threads_per_block, shared_mem_bytes>>>(
        scalars.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(),
        batch_size
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "persistent_ecdsa_sign_kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}

// ===================================================================
// CARD 10: Affine Conversion Debug Readback
// ===================================================================
// Reads the 8 device debug buffers captured during affine conversion
// and returns them as a Python dictionary for comparison against oracle.

torch::Tensor debug_affine_state() {
    // Allocate host memory for all 8 debug buffers
    // Each buffer is 8 uint32_t limbs = 32 bytes
    // Total: 8 buffers × 8 limbs = 64 uint32_t values

    uint32_t host_buffers[8][8];  // [buffer_id][limb_id]

    // Copy each debug buffer from device to host using cudaMemcpyFromSymbol
    // IMPORTANT: The symbol names must match the __device__ declarations in p256_field_warp_coop.cuh

    cudaError_t err;

    // PRE-affine state (Jacobian coordinates)
    err = cudaMemcpyFromSymbol(host_buffers[0], dbg_Z, 8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "Failed to copy dbg_Z: ", cudaGetErrorString(err));

    err = cudaMemcpyFromSymbol(host_buffers[1], dbg_X_before, 8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "Failed to copy dbg_X_before: ", cudaGetErrorString(err));

    err = cudaMemcpyFromSymbol(host_buffers[2], dbg_Y_before, 8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "Failed to copy dbg_Y_before: ", cudaGetErrorString(err));

    // Z-inversion pipeline
    err = cudaMemcpyFromSymbol(host_buffers[3], dbg_Zinv, 8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "Failed to copy dbg_Zinv: ", cudaGetErrorString(err));

    err = cudaMemcpyFromSymbol(host_buffers[4], dbg_Zinv2, 8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "Failed to copy dbg_Zinv2: ", cudaGetErrorString(err));

    err = cudaMemcpyFromSymbol(host_buffers[5], dbg_Zinv3, 8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "Failed to copy dbg_Zinv3: ", cudaGetErrorString(err));

    // POST-affine state (affine coordinates)
    err = cudaMemcpyFromSymbol(host_buffers[6], dbg_X_after, 8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "Failed to copy dbg_X_after: ", cudaGetErrorString(err));

    err = cudaMemcpyFromSymbol(host_buffers[7], dbg_Y_after, 8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess, "Failed to copy dbg_Y_after: ", cudaGetErrorString(err));

    // Convert to PyTorch tensor: shape [8, 8] (8 buffers × 8 limbs)
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto result = torch::from_blob(host_buffers, {8, 8}, options).clone();  // clone() to copy data

    return result;
}

// ===================================================================
// CARD 11: Double De-Montgomery Debug Readback
// ===================================================================
// Reads the 6 device debug buffers captured during double de-Montgomery
// and returns them as a [6, 8] tensor for diagnostic analysis.

torch::Tensor debug_double_demont_state() {
    // Allocate host memory for all 6 debug buffers
    // Each buffer is 8 uint32_t limbs = 32 bytes
    // Total: 6 buffers × 8 limbs = 48 uint32_t values
    uint32_t host_buffers[6][8];  // [buffer_id][limb_id]

    cudaError_t err;

    // Copy X_demont1 (after 1st de-Montgomery)
    err = cudaMemcpyFromSymbol(host_buffers[0], dbg_X_demont1,
                               8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess,
                "Failed to copy dbg_X_demont1: ", cudaGetErrorString(err));

    err = cudaMemcpyFromSymbol(host_buffers[1], dbg_Y_demont1,
                               8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess,
                "Failed to copy dbg_Y_demont1: ", cudaGetErrorString(err));

    // Copy X_shuffle (after shuffle gather)
    err = cudaMemcpyFromSymbol(host_buffers[2], dbg_X_shuffle,
                               8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess,
                "Failed to copy dbg_X_shuffle: ", cudaGetErrorString(err));

    err = cudaMemcpyFromSymbol(host_buffers[3], dbg_Y_shuffle,
                               8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess,
                "Failed to copy dbg_Y_shuffle: ", cudaGetErrorString(err));

    // Copy X_demont2 (after 2nd de-Montgomery)
    err = cudaMemcpyFromSymbol(host_buffers[4], dbg_X_demont2,
                               8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess,
                "Failed to copy dbg_X_demont2: ", cudaGetErrorString(err));

    err = cudaMemcpyFromSymbol(host_buffers[5], dbg_Y_demont2,
                               8 * sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
    TORCH_CHECK(err == cudaSuccess,
                "Failed to copy dbg_Y_demont2: ", cudaGetErrorString(err));

    // Create PyTorch tensor from host buffers
    // Shape: [6, 8] where dim 0 is buffer index, dim 1 is limb index
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto result = torch::from_blob(host_buffers, {6, 8}, options).clone();

    return result;
}

// ==============================================================================
// Card 14: Batch Affine Conversion Wrapper
// ==============================================================================

#include "p256_batch_affine_ref.cuh"

// Non-inline wrapper to expose the host-side batch affine function to the bindings
// The actual implementation is in p256_batch_affine_ref.cuh (inline helper)
// Only compile this on the host side (not during CUDA device compilation)
#ifndef __CUDA_ARCH__
void batch_affine_from_jacobian_host(
    const P256_Jacobian_Scalar* in_points,
    P256_Affine_Scalar* out_points,
    int N
) {
    // Call the inline host implementation (only available on host, not device)
    batch_affine_from_jacobian_host_impl(in_points, out_points, N);
}
#endif

// Forward declaration for batch affine bindings (Card 14)
void register_batch_affine_bindings(pybind11::module_ &m);

// PyTorch bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_comb_table",
          &load_comb_table_from_tensors,
          "Load precomputed comb table to GPU constant memory");

    m.def("persistent_sign_batch",
          &persistent_ecdsa_sign_batch,
          "P-256 ECDSA persistent fused kernel (GPU)");

    m.def("debug_affine_state",
          &debug_affine_state,
          "Read affine conversion debug buffers (Card 10)");

    m.def("debug_double_demont_state",
          &debug_double_demont_state,
          "Read double de-Montgomery debug buffers (Card 11)");

    // Card 14: Batch affine conversion bindings
    register_batch_affine_bindings(m);
}

// Force recompile
