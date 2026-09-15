// dilithium_challenge.h - GPU Challenge Polynomial Sampling
// Card 14.3: GPU-side challenge computation for signing
//
// Implements H(mu || w1) -> sparse +/-1 challenge polynomial
// Per FIPS 204 Section 6.2 (ML-DSA.Sign)
//
// This moves challenge computation to GPU, eliminating the per-attempt
// host round-trip that was blocking true GPU-resident signing.

#ifndef SMOKE_DILITHIUM_CHALLENGE_H
#define SMOKE_DILITHIUM_CHALLENGE_H

#include "dilithium_params.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace smoke {
namespace dilithium {

// =============================================================================
// GPU Challenge Sampling
// =============================================================================

// Sample challenge polynomial c from (mu, w1) using GPU SHAKE256.
//
// FIPS 204 SampleInBall:
// - tau distinct positions in [0, n-1] are non-zero
// - Value at each position is +1 or -1 with equal probability
// - All other positions are 0
//
// Inputs:
//   params: mode parameters
//   d_mu: [B, 32] mu = H(tr || msg) for each message
//   d_w1: [B, k, n] high bits of w from RoundA
//   d_active: [B] mask (1=active, 0=skip)
//   B: batch size
//
// Outputs:
//   d_c: [B, n] challenge polynomials, dtype int8 in {-1, 0, 1}
//
// Workspace:
//   d_work: temporary buffer for serialized w1 + SHAKE state
//   Required size: challenge_workspace_size(params, B)
//
void sample_challenge_batch_gpu(
    const Params& params,
    const uint8_t* d_mu,           // [B, 32] mu per message
    const uint32_t* d_w1,          // [B, k, n] high bits of w
    int8_t* d_c,                   // [B, n] output challenges
    const uint8_t* d_active,       // [B] mask
    uint8_t* d_work,               // workspace
    int B,
    cudaStream_t stream = 0);

// Workspace size for challenge sampling
inline size_t challenge_workspace_size(const Params& params, int B) {
    // Serialized w1 per message: k * n * 4 bytes (little-endian uint32 to match CPU)
    // Plus SHAKE output buffer for c_tilde computation: 512 bytes per message
    // Plus SHAKE output buffer for FIPS 204 SampleInBall: 512 bytes per message
    // Card 27.28: Added second SHAKE buffer for proper FIPS 204 compliance
    size_t w1_bytes = static_cast<size_t>(params.k) * params.n * 4;
    size_t shake_out_ctilde = 512;  // SHAKE256(mu||w1) for c_tilde
    size_t shake_out_sample = 512;  // SHAKE256(c_tilde) for sampling
    return B * (w1_bytes + shake_out_ctilde + shake_out_sample);
}

// =============================================================================
// Unified GPU Signing - Full Pipeline (Card 14.3)
// =============================================================================

// Full GPU signing with internal rejection loop.
//
// This function runs the complete signing pipeline on GPU:
// 1. Pre-compute: NTT(s1), NTT(s2), NTT(t0) once per batch
// 2. Loop (internal, GPU-resident):
//    a. RoundA: sample_y, NTT(y), A*y, INTT(w), decompose
//    b. Challenge: H(mu || w1) on GPU
//    c. RoundB: z, r, norms, hints
//    d. Update active mask
//    e. Break when all lanes complete
// 3. Return final z, h, c
//
// CPU involvement: NONE during rejection loop (only final D->H copy)
//
// Inputs:
//   params: mode parameters
//   d_A_hat: [k, l, n] matrix in NTT domain (shared across batch)
//   d_rho: [32] public seed (for A expansion, already expanded)
//   d_rhoprime: [32] secret seed (for y sampling)
//   d_s1: [l, n] secret s1 in time domain
//   d_s2: [k, n] secret s2 in time domain
//   d_t0: [k, n] t0 in time domain
//   d_mu: [B, 32] mu = H(tr || msg) per message (computed by caller)
//   B: batch size
//   max_attempts: maximum rejection attempts per lane
//
// Outputs:
//   d_z: [B, l, n] signature component z
//   d_h: [B, k, n] hint bits (dense, packed later)
//   d_c: [B, n] challenge polynomials
//   d_attempts: [B] final attempt count per lane
//   d_converged: [B] 1 if converged, 0 if max_attempts reached
//
// Returns: number of rounds executed (for telemetry)
//
// Card 27.22: Added d_ctilde output for FIPS 204 signature packing.
// c_tilde is the first 32 bytes of SHAKE256(mu || w1_encoded) used to derive c.
//
int sign_full_batch_gpu(
    const Params& params,
    const uint32_t* d_A_hat,       // [k, l, n] shared
    const uint8_t* d_rhoprime,     // [32] seed for y
    const uint32_t* d_s1,          // [l, n] time domain
    const uint32_t* d_s2,          // [k, n] time domain
    const uint32_t* d_t0,          // [k, n] time domain
    const uint8_t* d_mu,           // [B, 32] mu per message
    uint32_t* d_z,                 // [B, l, n] output
    uint8_t* d_h,                  // [B, k, n] output hints (dense)
    int8_t* d_c,                   // [B, n] output challenges
    uint8_t* d_ctilde,             // [B, 32] output c_tilde (FIPS 204) - Card 27.22
    uint16_t* d_attempts,          // [B] output attempt counts
    uint8_t* d_converged,          // [B] output convergence flags
    int B,
    uint16_t max_attempts,
    cudaStream_t stream = 0);

// Workspace size for full batch signing
inline size_t sign_full_workspace_size(const Params& params, int B) {
    // NTT versions of secrets: (l + 2*k) * n * 4 bytes (shared)
    size_t ntt_secrets = (params.l + 2 * params.k) * params.n * sizeof(uint32_t);

    // cs buffer needs max(l, k) since cs1 uses l polys, cs2 uses k polys
    int cs_polys = (params.l > params.k) ? params.l : params.k;

    // Per-round buffers:
    // - y: B * l * n
    // - y_hat: B * l * n (NTT copy of y, used for matvec)
    // - w: B * k * n
    // - w1: B * k * n
    // - w0: B * k * n (int32)
    // - c_uint32: B * n (for NTT)
    // - cs: B * max(l,k) * n (shared for cs1 and cs2)
    // - r: B * k * n
    // - r0: B * k * n (int32) - for proper r0 norm check
    // - ct0: B * k * n
    // - neg_ct0: B * k * n
    // - r_hint: B * k * n
    size_t per_round = static_cast<size_t>(B) * (
        params.l * params.n +      // y
        params.l * params.n +      // y_hat (for NTT)
        params.k * params.n +      // w
        params.k * params.n +      // w1
        params.k * params.n +      // w0
        params.n +                 // c_uint32
        cs_polys * params.n +      // cs (max(l,k) for cs1/cs2)
        params.k * params.n +      // r
        params.k * params.n +      // r0 (for decomposed r)
        params.k * params.n +      // ct0
        params.k * params.n +      // neg_ct0
        params.k * params.n        // r_hint
    ) * sizeof(uint32_t);

    // Challenge workspace
    size_t challenge = challenge_workspace_size(params, B);

    // Sample Y workspace: B * l * (34 + 768) bytes
    // This is pre-allocated to avoid cudaMalloc/cudaFree per iteration
    constexpr size_t BYTES_PER_Y_POLY = 3 * 256;  // 768 bytes
    size_t sample_y_work = static_cast<size_t>(B) * params.l * (34 + BYTES_PER_Y_POLY);

    // State buffers: active[B], pass[B], (padding for alignment), hint_count[B], active_count
    // Align hint_count to 4 bytes by rounding up active+pass size
    size_t active_pass = static_cast<size_t>(B) * 2;  // active[B] + pass[B]
    size_t active_pass_aligned = (active_pass + 3) & ~3;  // Round up to 4-byte boundary
    size_t state = active_pass_aligned + static_cast<size_t>(B) * sizeof(uint32_t) + sizeof(uint32_t);

    return ntt_secrets + per_round + challenge + sample_y_work + state;
}

// =============================================================================
// Telemetry Structure
// =============================================================================

struct SignFullTelemetry {
    int batch_size;
    int rounds_executed;
    int total_converged;
    int total_max_attempts;

    // Per-lane (optional, filled if attempts array provided)
    // uint16_t* attempts;
};

} // namespace dilithium
} // namespace smoke

#endif // SMOKE_DILITHIUM_CHALLENGE_H
