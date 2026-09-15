// dilithium_shake_batch.h - Batched SHAKE-256 XOF for Dilithium y sampling
// Card 19C: GPU Batch SHAKE for y Sampling (Eliminate Per-Lane Iteration)
//
// Specialized batched SHAKE-256 for fixed 34-byte inputs (rhoprime || nonce).
// Single kernel launch processes all (B × l) jobs in parallel.

#ifndef SMOKE_DILITHIUM_SHAKE_BATCH_H
#define SMOKE_DILITHIUM_SHAKE_BATCH_H

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

namespace smoke {
namespace dilithium {

// =============================================================================
// Batched SHAKE-256 XOF for 34-byte inputs
// =============================================================================
//
// Specialized for Dilithium y sampling where:
// - Input: rhoprime (32 bytes) || nonce (2 bytes LE) = 34 bytes
// - Output: 3 × n bytes (768 bytes for n=256)
//
// Launch: 1 warp per job (efficient for many small jobs)
// Grid: ceil(count / WARPS_PER_BLOCK) blocks
//

// Bytes per y polynomial: 3 bytes per coefficient × 256 coefficients
constexpr size_t BYTES_PER_Y_POLY = 3 * 256;  // 768 bytes

// SHAKE-256 rate in bytes: (1600 - 2×256) / 8 = 136
constexpr int SHAKE256_RATE = 136;

// =============================================================================
// Prepare Seeds Kernel
// =============================================================================
// Builds contiguous input buffer for batched SHAKE:
//   d_seeds[(b * l + j) * 34 ... + 33] = rhoprime || LE16(attempt[b] * l + j)
//
// For inactive lanes (d_active[b] == 0), seed is still written but will
// produce deterministic zero output.
//
void prepare_y_seeds_batch_gpu(
    const uint8_t* d_rhoprime,     // [32] shared seed
    const uint16_t* d_attempts,    // [B] per-lane attempt counter
    const uint8_t* d_active,       // [B] mask
    uint8_t* d_seeds,              // [B * l, 34] output buffer
    int B,
    int l,
    cudaStream_t stream = 0);

// =============================================================================
// Batched SHAKE-256 XOF (34-byte input specialization)
// =============================================================================
// Processes `count` independent SHAKE-256 jobs:
//   Input: d_in[job * 34 ... + 33] (34 bytes per job)
//   Output: d_out[job * out_bytes ... + out_bytes - 1]
//
// Parameters:
//   d_in       - Contiguous input buffer [count, 34]
//   d_out      - Contiguous output buffer [count, out_bytes]
//   count      - Number of jobs (typically B * l)
//   out_bytes  - Output bytes per job (typically BYTES_PER_Y_POLY = 768)
//   stream     - CUDA stream
//
void shake256_xof_batch_34b_gpu(
    const uint8_t* d_in,           // [count, 34] input seeds
    uint8_t* d_out,                // [count, out_bytes] output
    int count,
    int out_bytes,
    cudaStream_t stream = 0);

// =============================================================================
// Convert XOF bytes to y coefficients
// =============================================================================
// Converts XOF output bytes to polynomial coefficients in [-gamma1+1, gamma1-1].
// Coefficients are encoded in [0, q) for storage.
//
// This is separate from SHAKE to allow reuse and clear separation of concerns.
//
void xof_bytes_to_y_batch_gpu(
    const uint8_t* d_xof_bytes,    // [B * l, BYTES_PER_Y_POLY] XOF output
    uint32_t* d_y,                 // [B, l, n] output coefficients
    const uint8_t* d_active,       // [B] mask (inactive lanes get zeros)
    int B,
    int l,
    int n,
    uint32_t gamma1,
    cudaStream_t stream = 0);

// =============================================================================
// Combined: Sample y from gamma1 distribution (Card 19C replacement)
// =============================================================================
// Replaces the per-lane iteration in sample_y_gamma1_batch_gpu with:
//   1. prepare_y_seeds_batch_gpu (1 launch)
//   2. shake256_xof_batch_34b_gpu (1 launch)
//   3. xof_bytes_to_y_batch_gpu (1 launch)
//
// Total: 3 constant launches regardless of B or l
//
void sample_y_gamma1_batch_v2_gpu(
    const uint8_t* d_rhoprime,     // [32] seed
    const uint16_t* d_attempts,    // [B] per-lane attempt counter
    const uint8_t* d_active,       // [B] mask
    uint32_t* d_y,                 // [B, l, n] output
    uint8_t* d_work,               // Workspace: size >= B * l * (34 + BYTES_PER_Y_POLY)
    int B,
    int l,
    int n,
    uint32_t gamma1,
    cudaStream_t stream = 0);

// =============================================================================
// Workspace size calculation
// =============================================================================
inline size_t sample_y_workspace_size(int B, int l) {
    // Seeds: B * l * 34 bytes
    // XOF output: B * l * BYTES_PER_Y_POLY bytes
    return static_cast<size_t>(B) * l * (34 + BYTES_PER_Y_POLY);
}

} // namespace dilithium
} // namespace smoke

#endif // SMOKE_DILITHIUM_SHAKE_BATCH_H
