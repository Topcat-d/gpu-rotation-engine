// ==============================================================================
// Jacobian Doubling Oracle Wrapper (cuPQC-Style Pattern)
// ==============================================================================
// This file implements a warp-safe Jacobian doubling using the gather/broadcast
// pattern, similar to the Montgomery multiply oracle wrapper.
//
// Key insight from cuPQC: "A second invocation of a warp-cooperative Jacobian
// doubling will deadlock if any internal helper performs __syncwarp() under a
// lane-dependent predicate."
//
// Solution: Use a scalar reference implementation wrapped with proper gather/
// broadcast to ensure all lanes participate in warp collectives.
//
// Card 26: Copied from pytorch-crypto-legacy/cuda/ with include path fix
// ==============================================================================

#ifndef JACOBIAN_DOUBLE_ORACLE_CUH
#define JACOBIAN_DOUBLE_ORACLE_CUH

// Card 26 FIX: Use _golden version in cuda-crypto
#include "p256_scalar_ref_golden.cuh"

#ifndef FULL_MASK
#define FULL_MASK 0xFFFFFFFFu
#endif

// ==============================================================================
// Gather/Broadcast Helpers for Distributed Limbs
// ==============================================================================

// Gather a distributed limb (each lane i has its limb in local_limb)
// into a lane-0 array out[8].
__device__ __forceinline__
void gather_fe_from_lanes(uint32_t local_limb, uint32_t out[8], int lane) {
    // Each lane contributes its own limb index = lane (0..7)
    // Lane 0 collects all via __shfl_sync.
    for (int i = 0; i < 8; i++) {
        // Only lane i has the "real" value for position i
        uint32_t v = (lane == i && lane < 8) ? local_limb : 0u;
        // Broadcast from lane i to all lanes
        v = __shfl_sync(FULL_MASK, v, i);
        if (lane == 0) {
            out[i] = v;
        }
    }
}

// Broadcast a lane-0 array in[8] back into distributed limbs:
// lane i gets limb i in local_limb.
__device__ __forceinline__
uint32_t broadcast_fe_to_lanes(const uint32_t in[8], int lane) {
    // Lane 0 has the full array, we need to broadcast in[i] to lane i
    uint32_t result = 0u;
    for (int i = 0; i < 8; i++) {
        uint32_t val = 0u;
        if (lane == 0) {
            val = in[i];  // Lane 0 reads in[i]
        }
        val = __shfl_sync(FULL_MASK, val, 0);  // Broadcast in[i] to all lanes
        if (i == lane && lane < 8) {
            result = val;  // Lane i stores it
        }
    }
    return result;
}

// ==============================================================================
// Scalar Jacobian Doubling Reference (Lane-0 Only)
// ==============================================================================
// Pure scalar implementation using proven point_double_scalar from p256_scalar_ref_golden.cuh
// NO warp collectives, NO __syncwarp - safe to call from lane 0 only
// ==============================================================================

__device__ void jacobian_double_ref_scalar(
    const uint32_t X_in[8],
    const uint32_t Y_in[8],
    const uint32_t Z_in[8],
    uint32_t X_out[8],
    uint32_t Y_out[8],
    uint32_t Z_out[8]
) {
    // Check for point at infinity (Z = 0)
    bool Z_is_zero = true;
    for (int i = 0; i < 8; i++) {
        if (Z_in[i] != 0) {
            Z_is_zero = false;
            break;
        }
    }

    if (Z_is_zero) {
        // Doubling infinity returns infinity
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            X_out[i] = 0;
            Y_out[i] = 0;
            Z_out[i] = 0;
        }
        return;
    }

    // Use the proven scalar reference implementation
    P256_Jacobian_Scalar P, R;

    // Copy inputs
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        P.X[i] = X_in[i];
        P.Y[i] = Y_in[i];
        P.Z[i] = Z_in[i];
    }

    // Call proven scalar reference
    point_double_scalar(R, P);

    // Copy outputs
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X_out[i] = R.X[i];
        Y_out[i] = R.Y[i];
        Z_out[i] = R.Z[i];
    }
}

// ==============================================================================
// Warp-Compatible Jacobian Doubling Oracle Wrapper
// ==============================================================================
// This wrapper ensures all 32 lanes participate in warp collectives,
// preventing the "second call hangs" bug.
// ==============================================================================

__device__ __forceinline__
void jacobian_double_oracle_compat(
    uint32_t X[8],
    uint32_t Y[8],
    uint32_t Z[8],
    int lane
) {
    // 1) Gather from distributed limbs into lane-0 arrays
    uint32_t X_in[8], Y_in[8], Z_in[8];

    // Each lane uses its *own* limb as local input for gather
    uint32_t x_local = (lane < 8) ? X[lane] : 0u;
    uint32_t y_local = (lane < 8) ? Y[lane] : 0u;
    uint32_t z_local = (lane < 8) ? Z[lane] : 0u;

    gather_fe_from_lanes(x_local, X_in, lane);
    gather_fe_from_lanes(y_local, Y_in, lane);
    gather_fe_from_lanes(z_local, Z_in, lane);

    __syncwarp(FULL_MASK);

    // 2) Lane 0 calls scalar reference double
    uint32_t X_out[8], Y_out[8], Z_out[8];
    if (lane == 0) {
        jacobian_double_ref_scalar(X_in, Y_in, Z_in, X_out, Y_out, Z_out);
    }

    __syncwarp(FULL_MASK);

    // 3) Broadcast results back out to distributed limbs
    // ALL lanes must participate in broadcast (has __shfl_sync inside)
    uint32_t x_limb = broadcast_fe_to_lanes(X_out, lane);
    uint32_t y_limb = broadcast_fe_to_lanes(Y_out, lane);
    uint32_t z_limb = broadcast_fe_to_lanes(Z_out, lane);

    // Only lanes < 8 store the results
    if (lane < 8) {
        X[lane] = x_limb;
        Y[lane] = y_limb;
        Z[lane] = z_limb;
    }

    __syncwarp(FULL_MASK);
}

// ==============================================================================
// Scalar Jacobian Add Affine Reference (Lane-0 Only)
// ==============================================================================

__device__ void jacobian_add_affine_ref_scalar(
    const uint32_t X1_in[8],
    const uint32_t Y1_in[8],
    const uint32_t Z1_in[8],
    const uint32_t X2_in[8],
    const uint32_t Y2_in[8],
    uint32_t X_out[8],
    uint32_t Y_out[8],
    uint32_t Z_out[8]
) {
    // Pure wrapper around the scalar reference - no special cases!
    // Let the reference handle infinity the same way it always has.
    P256_Jacobian_Scalar P1, R;

    // Copy Jacobian input
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        P1.X[i] = X1_in[i];
        P1.Y[i] = Y1_in[i];
        P1.Z[i] = Z1_in[i];
    }

    // Call proven scalar mixed addition
    point_add_mixed_scalar(R, P1, X2_in, Y2_in);

    // Copy outputs
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X_out[i] = R.X[i];
        Y_out[i] = R.Y[i];
        Z_out[i] = R.Z[i];
    }
}

// ==============================================================================
// Warp-Compatible Jacobian Add Affine Oracle Wrapper
// ==============================================================================

__device__ __forceinline__
void jacobian_add_affine_oracle_compat(
    uint32_t X1[8],
    uint32_t Y1[8],
    uint32_t Z1[8],
    const uint32_t X2[8],
    const uint32_t Y2[8],
    int lane
) {
    // 1) Gather from distributed limbs
    uint32_t X1_in[8], Y1_in[8], Z1_in[8], X2_in[8], Y2_in[8];

    uint32_t x1_local = (lane < 8) ? X1[lane] : 0u;
    uint32_t y1_local = (lane < 8) ? Y1[lane] : 0u;
    uint32_t z1_local = (lane < 8) ? Z1[lane] : 0u;
    uint32_t x2_local = (lane < 8) ? X2[lane] : 0u;
    uint32_t y2_local = (lane < 8) ? Y2[lane] : 0u;

    gather_fe_from_lanes(x1_local, X1_in, lane);
    gather_fe_from_lanes(y1_local, Y1_in, lane);
    gather_fe_from_lanes(z1_local, Z1_in, lane);
    gather_fe_from_lanes(x2_local, X2_in, lane);
    gather_fe_from_lanes(y2_local, Y2_in, lane);

    __syncwarp(FULL_MASK);

    // 2) Lane 0 calls scalar reference add
    uint32_t X_out[8], Y_out[8], Z_out[8];
    if (lane == 0) {
        jacobian_add_affine_ref_scalar(X1_in, Y1_in, Z1_in, X2_in, Y2_in,
                                        X_out, Y_out, Z_out);
    }

    __syncwarp(FULL_MASK);

    // 3) Broadcast results back (ALL lanes participate)
    uint32_t x_limb = broadcast_fe_to_lanes(X_out, lane);
    uint32_t y_limb = broadcast_fe_to_lanes(Y_out, lane);
    uint32_t z_limb = broadcast_fe_to_lanes(Z_out, lane);

    // Only lanes < 8 store results
    if (lane < 8) {
        X1[lane] = x_limb;
        Y1[lane] = y_limb;
        Z1[lane] = z_limb;
    }

    __syncwarp(FULL_MASK);
}

#endif // JACOBIAN_DOUBLE_ORACLE_CUH
