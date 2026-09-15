// dilithium_sign_batch.h - Batch-dimension CUDA kernels for Dilithium signing
// Card 19B: True batch-dimension kernels for launch amortization
//
// These functions process B signatures in parallel with a single kernel launch
// per stage (or constant number of launches).

#ifndef SMOKE_DILITHIUM_SIGN_BATCH_H
#define SMOKE_DILITHIUM_SIGN_BATCH_H

#include "dilithium_params.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace smoke {
namespace dilithium {

// =============================================================================
// Batch Sample Y
// =============================================================================
// Sample y vectors for all B lanes from gamma1 distribution.
// Grid: (B, l), single launch.
//
void sample_y_gamma1_batch_gpu(
    const Params& params,
    const uint8_t* d_rhoprime,     // [32] seed on device (shared)
    const uint16_t* d_attempts,    // [B] per-lane attempt counter
    const uint8_t* d_active,       // [B] mask (1=active, 0=done)
    uint32_t* d_y,                 // [B, l, n] output
    int B,
    cudaStream_t stream = 0);

// =============================================================================
// Batch Matvec
// =============================================================================
// Compute w_hat = A_hat * y_hat for all B lanes.
// Grid: (B, k), single launch.
//
void matvec_batch_gpu(
    const Params& params,
    const uint32_t* d_A_hat,       // [k, l, n] shared matrix in NTT domain
    const uint32_t* d_y_hat,       // [B, l, n] y in NTT domain
    uint32_t* d_w_hat,             // [B, k, n] output in NTT domain
    const uint8_t* d_active,       // [B] mask
    int B,
    cudaStream_t stream = 0);

// =============================================================================
// Batch Decompose
// =============================================================================
// Decompose w -> (w1, w0) for all B lanes.
// Single launch over B*k*n coefficients.
//
void decompose_batch_gpu(
    const Params& params,
    const uint32_t* d_w,           // [B, k, n] input
    uint32_t* d_w1,                // [B, k, n] high bits output
    int32_t* d_w0,                 // [B, k, n] low bits (signed for norm check)
    const uint8_t* d_active,       // [B] mask
    int B,
    cudaStream_t stream = 0);

// =============================================================================
// Batch Compute Z
// =============================================================================
// Compute z = y + c*s1 for all B lanes.
// Uses NTT multiplication with batched operations.
//
void compute_z_batch_gpu(
    const Params& params,
    const uint32_t* d_y,           // [B, l, n] y values
    const int8_t* d_c,             // [B, n] challenge polynomials (int8)
    const uint32_t* d_s1_ntt,      // [l, n] s1 in NTT domain (shared across batch)
    uint32_t* d_z,                 // [B, l, n] output
    const uint8_t* d_active,       // [B] mask
    int B,
    uint32_t* d_work,              // Workspace: size >= B*n + B*l*n
    cudaStream_t stream = 0);

// =============================================================================
// Batch Check Z Norm
// =============================================================================
// Check ||z||_inf < gamma1 - beta for all B lanes.
// Single launch with per-block reduction.
//
void check_z_norm_batch_gpu(
    const Params& params,
    const uint32_t* d_z,           // [B, l, n]
    uint8_t* d_pass,               // [B] output flags (1=pass, 0=fail)
    const uint8_t* d_active,       // [B] mask
    int B,
    cudaStream_t stream = 0);

// =============================================================================
// Batch Update Active Mask
// =============================================================================
// Update active mask based on pass flags and increment attempts.
//
void update_active_batch_gpu(
    uint8_t* d_active,             // [B] mask (modified in place)
    uint16_t* d_attempts,          // [B] attempt counters (modified)
    const uint8_t* d_pass,         // [B] pass flags from current round
    int B,
    uint16_t max_attempts,
    cudaStream_t stream = 0);

// =============================================================================
// Count Active Lanes
// =============================================================================
// Returns count of lanes still active.
//
uint32_t count_active_batch_gpu(
    const uint8_t* d_active,       // [B] mask
    int B,
    cudaStream_t stream = 0);

// =============================================================================
// MAIN ENTRYPOINTS: RoundA and RoundB (Card 19B)
// =============================================================================
// These are the ONLY functions Python should call per round.
// Each is ONE composite operation = constant kernel launches.
//
// Round structure:
//   RoundA: sample_y -> NTT(y) -> matvec -> INTT(w) -> decompose -> produce w1
//   [CPU: compute c from w1]
//   RoundB: consume c -> compute z,r -> check norms -> make hints -> update mask
//

// -----------------------------------------------------------------------------
// RoundA: Produces w1 for all active lanes
// -----------------------------------------------------------------------------
// Launches: ~5 kernels (sample_y, NTT, matvec, INTT, decompose)
// Output: d_w1 ready for CPU challenge computation
//
void sign_roundA_w1_batch_gpu(
    const Params& params,
    // Shared inputs (constant across batch)
    const uint8_t* d_rhoprime,     // [32] seed
    const uint32_t* d_A_hat,       // [k, l, n] matrix in NTT domain
    // Per-lane state
    const uint8_t* d_active,       // [B] mask
    const uint16_t* d_attempts,    // [B] attempt counters
    // Per-lane outputs
    uint32_t* d_y,                 // [B, l, n] sampled y (kept for z computation)
    uint32_t* d_w,                 // [B, k, n] w = A*y (kept for r computation)
    uint32_t* d_w1,                // [B, k, n] high bits of w (output for CPU)
    int32_t* d_w0,                 // [B, k, n] low bits (kept for norm check)
    // Workspace
    uint32_t* d_work,              // Size: B * max(l,k) * n * 4
    int B,
    cudaStream_t stream = 0);

// -----------------------------------------------------------------------------
// RoundB: Consumes c, produces z/h/pass, updates mask
// -----------------------------------------------------------------------------
// Launches: ~8 kernels (c_convert, NTT, mul, INTT, z_norm, r_compute, hints, mask_update)
// Modifies: d_active, d_attempts, d_pass, d_active_count
//
void sign_roundB_zh_batch_gpu(
    const Params& params,
    // Shared inputs
    const uint32_t* d_s1_ntt,      // [l, n] s1 in NTT domain
    const uint32_t* d_s2_ntt,      // [k, n] s2 in NTT domain
    const uint32_t* d_t0_ntt,      // [k, n] t0 in NTT domain
    // Per-lane inputs from RoundA
    const uint32_t* d_y,           // [B, l, n]
    const uint32_t* d_w,           // [B, k, n]
    const int32_t* d_w0,           // [B, k, n] low bits for r0 norm
    const int8_t* d_c,             // [B, n] challenge from CPU
    // Per-lane state (modified)
    uint8_t* d_active,             // [B] mask (updated on success)
    uint16_t* d_attempts,          // [B] attempt counters (incremented on fail)
    // Per-lane outputs
    uint32_t* d_z,                 // [B, l, n] signature component
    uint8_t* d_h,                  // [B, k, n] hints (dense)
    uint8_t* d_pass,               // [B] pass flags
    uint32_t* d_hint_count,        // [B] hint count per lane
    uint32_t* d_active_count,      // [1] total active after this round
    // Config
    uint16_t max_attempts,
    // Workspace
    uint32_t* d_work,              // Size: B * max(l,k) * n * 4 + B * n
    int B,
    cudaStream_t stream = 0);

// =============================================================================
// Telemetry Structures (Card 19B)
// =============================================================================

struct BatchSignTelemetry {
    int batch_size;
    int rounds_total;
    int* attempts_per_sig;         // [B] attempts for each signature
    int active_count_per_round[256]; // Active count at each round (up to 256 rounds)

    // Statistics (computed after signing)
    float avg_attempts;
    int p50_attempts;
    int p90_attempts;
    int p99_attempts;
    int max_attempts;

    // Launch accounting
    static constexpr int LAUNCHES_PER_ROUND = 2;  // RoundA + RoundB
    int total_launches;
    float launches_per_sig;
};

} // namespace dilithium
} // namespace smoke

#endif // SMOKE_DILITHIUM_SIGN_BATCH_H
