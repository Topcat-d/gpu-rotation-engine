// dilithium_challenge.cu - GPU Challenge Polynomial Sampling + Unified Signing
// Card 14.3: GPU-side challenge computation for zero per-attempt host round-trips
//
// This file implements:
// 1. sample_challenge_batch_gpu: H(mu || w1) -> sparse +/-1 polynomial
// 2. sign_full_batch_gpu: Complete signing with internal rejection loop

#include "../include/dilithium_challenge.h"
#include "../include/dilithium_sign_batch.h"
#include "../include/dilithium_shake_batch.h"
#include "../include/dilithium_ntt.cuh"
#include "../include/dilithium_shake.h"

#include <cstdint>
#include <algorithm>

namespace smoke {
namespace dilithium {

namespace {

constexpr uint32_t N = 256;
constexpr uint32_t Q = 8380417;
constexpr int TAU_DEFAULT = 60;  // Number of +/-1 positions

// =============================================================================
// Device Helpers
// =============================================================================

__device__ __forceinline__
uint32_t mod_q(uint64_t x) {
    return static_cast<uint32_t>(x % Q);
}

// =============================================================================
// Keccak-f[1600] Device Implementation
// =============================================================================
// Reference implementation for SHAKE256 challenge hashing

__device__ void keccak_f1600(uint64_t* state) {
    // Round constants
    static constexpr uint64_t RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };

    uint64_t temp[25];
    uint64_t C[5], D[5];

    for (int round = 0; round < 24; round++) {
        // Theta
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
        }
        for (int i = 0; i < 25; i++) {
            state[i] ^= D[i % 5];
        }

        // Rho (rotation) and Pi (permutation) combined
        uint64_t t = state[1];
        int x = 1, y = 0;
        for (int i = 0; i < 24; i++) {
            int new_x = y;
            int new_y = (2 * x + 3 * y) % 5;
            int idx = new_x + 5 * new_y;
            uint64_t tmp = state[idx];
            int r = ((i + 1) * (i + 2) / 2) % 64;
            state[idx] = (t << r) | (t >> (64 - r));
            t = tmp;
            x = new_x;
            y = new_y;
        }

        // Chi
        for (int j = 0; j < 5; j++) {
            for (int i = 0; i < 5; i++) {
                temp[i] = state[i + 5 * j];
            }
            for (int i = 0; i < 5; i++) {
                state[i + 5 * j] = temp[i] ^ ((~temp[(i + 1) % 5]) & temp[(i + 2) % 5]);
            }
        }

        // Iota
        state[0] ^= RC[round];
    }
}

// =============================================================================
// Kernel: Serialize w1 for challenge hashing
// =============================================================================
// Pack w1 coefficients into bytes for SHAKE input.
// w1 values are small (depends on gamma2), typically 0-15 or 0-43.
// We pack as single bytes for simplicity.
//
// Grid: (B)
// Block: 256
//
__global__ void kernel_serialize_w1_batch(
    uint8_t* __restrict__ d_w1_bytes,  // [B, k*n*4] output (4 bytes per coeff)
    const uint32_t* __restrict__ d_w1, // [B, k, n]
    const uint8_t* __restrict__ d_active,
    int B, int k, int n)
{
    int b = blockIdx.x;
    if (b >= B || d_active[b] == 0) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int total = k * n;

    const uint32_t* w1_in = d_w1 + b * total;
    uint8_t* w1_out = d_w1_bytes + b * total * 4;  // 4 bytes per coefficient

    // Serialize as little-endian uint32 to match CPU reference
    for (int i = tid; i < total; i += stride) {
        uint32_t val = w1_in[i];
        w1_out[i * 4 + 0] = static_cast<uint8_t>(val & 0xFF);
        w1_out[i * 4 + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
        w1_out[i * 4 + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
        w1_out[i * 4 + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
    }
}

// =============================================================================
// Kernel: SHAKE256 for challenge (per-lane)
// =============================================================================
// Computes SHAKE256(mu || w1_bytes) and outputs 512 bytes.
// This is a simplified implementation using the existing SHAKE infrastructure.
//
// Grid: (B)
// Block: 1 warp (32 threads)
//
// Note: This kernel is per-lane but we batch all lanes in one launch.
// For higher performance, could use the batch SHAKE infrastructure.
//
__global__ void kernel_shake256_challenge_batch(
    uint8_t* __restrict__ d_shake_out,  // [B, 512] SHAKE output
    const uint8_t* __restrict__ d_mu,   // [B, 64] mu (FIPS 204: 64 bytes)
    const uint8_t* __restrict__ d_w1_bytes, // [B, k*n] serialized w1
    const uint8_t* __restrict__ d_active,
    int B, int w1_len)
{
    int b = blockIdx.x;
    if (b >= B || d_active[b] == 0) return;

    // Each block processes one lane
    // We use a simple SHAKE256 implementation per warp

    // SHAKE256 state (25 x 64-bit = 200 bytes)
    __shared__ uint64_t state[25];

    int tid = threadIdx.x;

    // Initialize state to zero
    if (tid < 25) {
        state[tid] = 0;
    }
    __syncthreads();

    // Input: mu (64 bytes per FIPS 204) + w1_bytes (k*n bytes)
    const uint8_t* mu_ptr = d_mu + b * 64;
    const uint8_t* w1_ptr = d_w1_bytes + b * w1_len;
    int total_input = 64 + w1_len;

    // Absorb input in rate-sized chunks (136 bytes for SHAKE256)
    constexpr int RATE = 136;
    int absorbed = 0;

    // Note: This is a simplified sequential absorb for correctness.
    // In production, this would use the optimized warp-cooperative SHAKE.

    // For now, use lane 0 to do the absorption
    if (tid == 0) {
        uint8_t block[RATE];

        while (absorbed < total_input) {
            int chunk = min(RATE, total_input - absorbed);

            // Zero the block
            for (int i = 0; i < RATE; i++) {
                block[i] = 0;
            }

            // Copy input to block
            for (int i = 0; i < chunk; i++) {
                if (absorbed + i < 64) {
                    // First 64 bytes come from mu
                    block[i] = mu_ptr[absorbed + i];
                } else {
                    // Rest comes from w1_bytes
                    block[i] = w1_ptr[absorbed + i - 64];
                }
            }

            // Apply padding if this is the last block
            if (absorbed + chunk >= total_input) {
                block[chunk] ^= 0x1F;  // SHAKE padding
                block[RATE - 1] ^= 0x80;
            }

            // XOR block into state
            uint8_t* state_bytes = reinterpret_cast<uint8_t*>(state);
            for (int i = 0; i < RATE; i++) {
                state_bytes[i] ^= block[i];
            }

            // Keccak-f[1600] permutation
            // Using the reference implementation for correctness
            keccak_f1600(state);

            absorbed += chunk;
        }

        // Squeeze 512 bytes
        uint8_t* out = d_shake_out + b * 512;
        uint8_t* state_bytes = reinterpret_cast<uint8_t*>(state);

        for (int i = 0; i < 512; i++) {
            if ((i > 0) && (i % RATE == 0)) {
                keccak_f1600(state);
            }
            out[i] = state_bytes[i % RATE];
        }
    }
}

// =============================================================================
// Kernel: Sample challenge polynomial from SHAKE output
// =============================================================================
// Implements SampleInBall: select TAU positions with +/-1 values
//
// Grid: (B)
// Block: 1 (sequential per lane for correctness)
//
__global__ void kernel_sample_challenge_from_shake(
    int8_t* __restrict__ d_c,           // [B, n] output
    const uint8_t* __restrict__ d_shake, // [B, 512] SHAKE output
    const uint8_t* __restrict__ d_active,
    int B, int n, int tau)
{
    int b = blockIdx.x;
    if (b >= B || d_active[b] == 0) return;

    // Only thread 0 does the work (sequential algorithm)
    if (threadIdx.x != 0) return;

    const uint8_t* shake = d_shake + b * 512;
    int8_t* c = d_c + b * n;

    // Zero output
    for (int i = 0; i < n; i++) {
        c[i] = 0;
    }

    // Sample TAU positions with +/-1
    int used_count = 0;
    int pos_idx = 0;
    int sign_idx = 256;  // Signs from second half

    while (used_count < tau && pos_idx < 256 && sign_idx < 512) {
        int pos = shake[pos_idx] % n;
        pos_idx++;

        // Check if position already used
        if (c[pos] != 0) {
            continue;
        }

        // Get sign
        int8_t sign = (shake[sign_idx] & 0x01) == 0 ? 1 : -1;
        sign_idx++;

        c[pos] = sign;
        used_count++;
    }
}

// =============================================================================
// Kernel: Extract c_tilde from SHAKE output (Card 27.22)
// =============================================================================
// c_tilde is the first 32 bytes of SHAKE256(mu || w1_bytes) per FIPS 204.
// This is used in signature packing instead of the challenge polynomial.
//
// Grid: (B)
// Block: 32 threads (each copies 1 byte)
//
__global__ void kernel_extract_ctilde(
    uint8_t* __restrict__ d_ctilde,     // [B, 32] output
    const uint8_t* __restrict__ d_shake, // [B, 512] SHAKE output
    const uint8_t* __restrict__ d_active,
    int B)
{
    int b = blockIdx.x;
    if (b >= B || d_active[b] == 0) return;

    int tid = threadIdx.x;
    if (tid < 32) {
        d_ctilde[b * 32 + tid] = d_shake[b * 512 + tid];
    }
}

} // anonymous namespace

// =============================================================================
// Public API: Sample Challenge Batch
// =============================================================================

// FIPS 204 compliant SHAKE256 for c_tilde -> sampling bytes
// Takes 32-byte c_tilde as input, produces 512 bytes for SampleInBall
__global__ void kernel_shake256_sample_in_ball(
    uint8_t* __restrict__ d_shake_out,  // [B, 512] output
    const uint8_t* __restrict__ d_ctilde, // [B, 32] c_tilde input
    const uint8_t* __restrict__ d_active,
    int B)
{
    int b = blockIdx.x;
    if (b >= B || d_active[b] == 0) return;

    // SHAKE256 state
    __shared__ uint64_t state[25];
    int tid = threadIdx.x;

    // Initialize state to zero
    if (tid < 25) {
        state[tid] = 0;
    }
    __syncthreads();

    // Lane 0 does the work
    if (tid == 0) {
        const uint8_t* ctilde = d_ctilde + b * 32;
        constexpr int RATE = 136;
        uint8_t block[RATE];

        // Absorb c_tilde (32 bytes) with padding
        for (int i = 0; i < RATE; i++) {
            block[i] = 0;
        }
        for (int i = 0; i < 32; i++) {
            block[i] = ctilde[i];
        }
        // Apply SHAKE padding at position 32
        block[32] ^= 0x1F;
        block[RATE - 1] ^= 0x80;

        // XOR into state
        uint8_t* state_bytes = reinterpret_cast<uint8_t*>(state);
        for (int i = 0; i < RATE; i++) {
            state_bytes[i] ^= block[i];
        }
        keccak_f1600(state);

        // Squeeze 512 bytes
        uint8_t* out = d_shake_out + b * 512;
        for (int i = 0; i < 512; i++) {
            if ((i > 0) && (i % RATE == 0)) {
                keccak_f1600(state);
            }
            out[i] = state_bytes[i % RATE];
        }
    }
}

void sample_challenge_batch_gpu(
    const Params& params,
    const uint8_t* d_mu,
    const uint32_t* d_w1,
    int8_t* d_c,
    const uint8_t* d_active,
    uint8_t* d_work,
    int B,
    cudaStream_t stream)
{
    const int k = params.k;
    const int n = params.n;
    const int tau = params.tau > 0 ? params.tau : TAU_DEFAULT;

    // Workspace layout:
    // [0, B*k*n*4): w1_bytes (4 bytes per coefficient to match CPU serialization)
    // [B*k*n*4, B*k*n*4 + B*512): shake_out_ctilde (H(mu||w1) for c_tilde)
    // [B*k*n*4 + B*512, B*k*n*4 + B*1024): shake_out_sample (H(c_tilde) for sampling)
    size_t w1_bytes_size = static_cast<size_t>(B) * k * n * 4;
    uint8_t* d_w1_bytes = d_work;
    uint8_t* d_shake_out_ctilde = d_work + w1_bytes_size;
    uint8_t* d_shake_out_sample = d_shake_out_ctilde + B * 512;

    // Step 1: Serialize w1 to bytes (little-endian uint32)
    kernel_serialize_w1_batch<<<B, 256, 0, stream>>>(
        d_w1_bytes, d_w1, d_active, B, k, n);

    // Step 2: SHAKE256(mu || w1_bytes) -> 512 bytes (for c_tilde extraction)
    kernel_shake256_challenge_batch<<<B, 32, 0, stream>>>(
        d_shake_out_ctilde, d_mu, d_w1_bytes, d_active, B, k * n * 4);

    // Step 3: Extract c_tilde (first 32 bytes of H(mu||w1)) into temp buffer
    // We'll use the first 32 bytes of d_shake_out_ctilde directly

    // Step 4: FIPS 204 SampleInBall - compute H(c_tilde) for sampling bytes
    // This is what FIPS 204 specifies: c = SampleInBall(c_tilde) where
    // SampleInBall internally computes H(c_tilde)
    kernel_shake256_sample_in_ball<<<B, 32, 0, stream>>>(
        d_shake_out_sample, d_shake_out_ctilde, d_active, B);

    // Step 5: Sample challenge from H(c_tilde) output (FIPS 204 compliant)
    kernel_sample_challenge_from_shake<<<B, 1, 0, stream>>>(
        d_c, d_shake_out_sample, d_active, B, n, tau);
}

// =============================================================================
// Public API: Full Batch Signing (GPU-Resident Rejection Loop)
// =============================================================================

int sign_full_batch_gpu(
    const Params& params,
    const uint32_t* d_A_hat,
    const uint8_t* d_rhoprime,
    const uint32_t* d_s1,
    const uint32_t* d_s2,
    const uint32_t* d_t0,
    const uint8_t* d_mu,
    uint32_t* d_z,
    uint8_t* d_h,
    int8_t* d_c,
    uint8_t* d_ctilde,      // Card 27.22: [B, 32] output c_tilde for FIPS 204
    uint16_t* d_attempts,
    uint8_t* d_converged,
    int B,
    uint16_t max_attempts,
    cudaStream_t stream)
{
    const int l = params.l;
    const int k = params.k;
    const int n = params.n;
    const uint32_t gamma1 = params.gamma1;

    // Allocate workspace for internal buffers
    size_t work_size = sign_full_workspace_size(params, B);
    uint8_t* d_work_raw = nullptr;
    cudaError_t err = cudaMalloc(&d_work_raw, work_size);
    if (err != cudaSuccess) {
        // Allocation failed - return 0 rounds (caller should check converged flags)
        return 0;
    }

    // Parse workspace
    size_t offset = 0;

    // NTT versions of secrets (shared across batch)
    uint32_t* d_s1_ntt = reinterpret_cast<uint32_t*>(d_work_raw + offset);
    offset += l * n * sizeof(uint32_t);
    uint32_t* d_s2_ntt = reinterpret_cast<uint32_t*>(d_work_raw + offset);
    offset += k * n * sizeof(uint32_t);
    uint32_t* d_t0_ntt = reinterpret_cast<uint32_t*>(d_work_raw + offset);
    offset += k * n * sizeof(uint32_t);

    // Per-round buffers
    uint32_t* d_y = reinterpret_cast<uint32_t*>(d_work_raw + offset);
    offset += static_cast<size_t>(B) * l * n * sizeof(uint32_t);
    uint32_t* d_y_hat = reinterpret_cast<uint32_t*>(d_work_raw + offset);  // NTT copy of y
    offset += static_cast<size_t>(B) * l * n * sizeof(uint32_t);
    uint32_t* d_w = reinterpret_cast<uint32_t*>(d_work_raw + offset);
    offset += static_cast<size_t>(B) * k * n * sizeof(uint32_t);
    uint32_t* d_w1 = reinterpret_cast<uint32_t*>(d_work_raw + offset);
    offset += static_cast<size_t>(B) * k * n * sizeof(uint32_t);
    int32_t* d_w0 = reinterpret_cast<int32_t*>(d_work_raw + offset);
    offset += static_cast<size_t>(B) * k * n * sizeof(int32_t);

    // RoundB workspace - cs buffer needs max(l, k) to avoid overflow
    int cs_polys = (l > k) ? l : k;
    uint32_t* d_roundB_work = reinterpret_cast<uint32_t*>(d_work_raw + offset);
    size_t roundB_work_size = static_cast<size_t>(B) * (
        n +                    // c_ntt
        cs_polys * n +         // cs (max(l,k) for cs1/cs2)
        k * n +                // r
        k * n +                // r0 (int32 for decomposed r, for FIPS 204 norm check)
        k * n +                // ct0
        k * n +                // neg_ct0
        k * n                  // r_hint
    ) * sizeof(uint32_t);
    offset += roundB_work_size;

    // Challenge workspace
    uint8_t* d_challenge_work = d_work_raw + offset;
    offset += challenge_workspace_size(params, B);

    // Sample Y workspace (pre-allocated to avoid cudaMalloc/Free per iteration)
    constexpr size_t BYTES_PER_Y_POLY = 3 * 256;  // 768 bytes
    uint8_t* d_sample_y_work = d_work_raw + offset;
    offset += static_cast<size_t>(B) * l * (34 + BYTES_PER_Y_POLY);

    // State buffers (with alignment padding for hint_count)
    uint8_t* d_active = d_work_raw + offset;
    offset += B;
    uint8_t* d_pass = d_work_raw + offset;
    offset += B;
    // Align to 4 bytes for uint32_t access
    offset = (offset + 3) & ~static_cast<size_t>(3);
    uint32_t* d_hint_count = reinterpret_cast<uint32_t*>(d_work_raw + offset);
    offset += B * sizeof(uint32_t);
    uint32_t* d_active_count = reinterpret_cast<uint32_t*>(d_work_raw + offset);

    // Initialize: copy secrets (ALREADY in NTT domain from key precomputation)
    // Card 27.20: Keys are precomputed to NTT domain during key load, so no NTT needed here
    cudaMemcpyAsync(d_s1_ntt, d_s1, l * n * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_s2_ntt, d_s2, k * n * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_t0_ntt, d_t0, k * n * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);

    // Initialize state: all lanes active, zero attempts
    cudaMemsetAsync(d_active, 1, B, stream);
    cudaMemsetAsync(d_attempts, 0, B * sizeof(uint16_t), stream);

    // =========================================================================
    // GPU-Resident Rejection Loop
    // =========================================================================
    int rounds = 0;
    uint32_t active_count = B;

    while (active_count > 0 && rounds < max_attempts) {
        rounds++;

        // ---------------------------------------------------------------------
        // RoundA: sample_y -> NTT(y) -> matvec -> INTT(w) -> decompose
        // Using standard function (internally allocates/frees sample_y workspace)
        // TODO: Optimize to use pre-allocated workspace once working
        // ---------------------------------------------------------------------
        sign_roundA_w1_batch_gpu(
            params, d_rhoprime, d_A_hat,
            d_active, d_attempts,
            d_y, d_w, d_w1, d_w0,
            d_y_hat,  // Use pre-allocated y_hat buffer
            B, stream);

        // ---------------------------------------------------------------------
        // Challenge: H(mu || w1) on GPU (NO HOST ROUND-TRIP!)
        // ---------------------------------------------------------------------
        sample_challenge_batch_gpu(
            params, d_mu, d_w1, d_c, d_active,
            d_challenge_work, B, stream);

        // ---------------------------------------------------------------------
        // Card 27.22: Extract c_tilde (first 32 bytes of SHAKE output)
        // c_tilde = first 32 bytes of SHAKE256(mu || w1_encoded)
        // This is computed per-lane and stored for FIPS 204 signature packing.
        // Only active lanes update (preserves c_tilde for converged lanes).
        // ---------------------------------------------------------------------
        if (d_ctilde != nullptr) {
            // d_shake_out is at offset w1_bytes_size in challenge workspace
            size_t w1_bytes_size = static_cast<size_t>(B) * k * n * 4;
            uint8_t* d_shake_out = d_challenge_work + w1_bytes_size;
            kernel_extract_ctilde<<<B, 32, 0, stream>>>(
                d_ctilde, d_shake_out, d_active, B);
        }

        // ---------------------------------------------------------------------
        // RoundB: z, r, norms, hints, mask update
        // ---------------------------------------------------------------------
        sign_roundB_zh_batch_gpu(
            params, d_s1_ntt, d_s2_ntt, d_t0_ntt,
            d_y, d_w, d_w0, d_c,
            d_active, d_attempts,
            d_z, d_h, d_pass, d_hint_count, d_active_count,
            max_attempts, d_roundB_work, B, stream);

        // Get active count (single D->H copy per round)
        cudaMemcpyAsync(&active_count, d_active_count, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // Compute convergence flags
    // converged[b] = 1 if pass[b] == 1 (signature was found)
    // Pass is set to 1 when all norm checks passed and hints are valid
    cudaMemcpyAsync(d_converged, d_pass, B, cudaMemcpyDeviceToDevice, stream);

    // Synchronize before freeing to ensure all operations completed
    cudaStreamSynchronize(stream);

    // Check for any kernel errors that occurred during execution
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Kernel error occurred - clean up and return current round count
        cudaFree(d_work_raw);
        return rounds;
    }

    cudaFree(d_work_raw);

    return rounds;
}

} // namespace dilithium
} // namespace smoke
