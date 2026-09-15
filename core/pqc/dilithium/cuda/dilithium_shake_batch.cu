// dilithium_shake_batch.cu - Batched SHAKE-256 XOF for Dilithium y sampling
// Card 19C: GPU Batch SHAKE for y Sampling (Eliminate Per-Lane Iteration)
//
// Specialized batched SHAKE-256 for fixed 34-byte inputs (rhoprime || nonce).
// Uses 1 warp per job for efficient parallel processing of many small jobs.

#include "../include/dilithium_shake_batch.h"
#include <cstdio>

namespace smoke {
namespace dilithium {

namespace {

// ============================================================================
// Keccak-f[1600] Constants (same as dilithium_shake.cu)
// ============================================================================

constexpr int KECCAK_LANES = 25;
constexpr int KECCAK_STATE_BYTES = 200;

__constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ int KECCAK_RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant__ int KECCAK_PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

// Host copies for initialization
static const uint64_t HOST_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};
static const int HOST_RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};
static const int HOST_PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

static bool g_batch_shake_initialized = false;

// ============================================================================
// Keccak-f[1600] Device Implementation (per-thread state)
// ============================================================================

__device__ __forceinline__
uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccak_round(uint64_t* state, int round) {
    uint64_t C[5], D[5], B[25];

    // Theta
    for (int x = 0; x < 5; ++x) {
        C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    for (int x = 0; x < 5; ++x) {
        D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
    }
    for (int i = 0; i < 25; ++i) {
        state[i] ^= D[i % 5];
    }

    // Rho + Pi
    B[0] = state[0];
    uint64_t t = state[1];
    for (int i = 0; i < 24; ++i) {
        int j = KECCAK_PI[i];
        B[j] = rotl64(t, KECCAK_RHO[i]);
        t = state[j];
    }

    // Chi
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            int idx = y * 5 + x;
            state[idx] = B[idx] ^ ((~B[y * 5 + (x + 1) % 5]) & B[y * 5 + (x + 2) % 5]);
        }
    }

    // Iota
    state[0] ^= KECCAK_RC[round];
}

__device__ void keccak_f1600(uint64_t* state) {
    for (int round = 0; round < 24; ++round) {
        keccak_round(state, round);
    }
}

// ============================================================================
// Kernel: Prepare y sampling seeds
// ============================================================================
// Each thread handles one (b, j) pair
// Output: d_seeds[(b * l + j) * 34 ... + 33] = rhoprime || LE16(attempt[b] * l + j)
//
__global__ void kernel_prepare_y_seeds(
    uint8_t* __restrict__ d_seeds,          // [B * l, 34] output
    const uint8_t* __restrict__ d_rhoprime, // [32] shared seed
    const uint16_t* __restrict__ d_attempts,// [B]
    const uint8_t* __restrict__ d_active,   // [B] (unused but kept for consistency)
    int B, int l)
{
    int job = blockIdx.x * blockDim.x + threadIdx.x;
    int total_jobs = B * l;
    if (job >= total_jobs) return;

    int b = job / l;
    int j = job % l;

    // Compute nonce: attempt[b] * l + j
    uint16_t attempt = d_attempts[b];
    uint16_t nonce = attempt * l + j;

    // Write to output: rhoprime (32 bytes) || nonce (2 bytes LE)
    uint8_t* out = d_seeds + job * 34;

    // Copy rhoprime (32 bytes)
    for (int i = 0; i < 32; ++i) {
        out[i] = d_rhoprime[i];
    }

    // Append nonce as little-endian 16-bit
    out[32] = static_cast<uint8_t>(nonce & 0xFF);
    out[33] = static_cast<uint8_t>((nonce >> 8) & 0xFF);
}

// ============================================================================
// Kernel: Batched SHAKE-256 XOF for 34-byte inputs
// ============================================================================
// 1 thread per job (baseline, efficient for many jobs)
// Each thread: absorb 34 bytes + pad, squeeze out_bytes
//
__global__ void kernel_shake256_xof_batch_34b(
    uint8_t* __restrict__ d_out,      // [count, out_bytes]
    const uint8_t* __restrict__ d_in, // [count, 34]
    int count,
    int out_bytes)
{
    int job = blockIdx.x * blockDim.x + threadIdx.x;
    if (job >= count) return;

    const uint8_t* in = d_in + job * 34;
    uint8_t* out = d_out + job * out_bytes;

    // SHAKE-256 rate = 136 bytes
    constexpr int rate = 136;

    // Initialize state
    uint64_t state[25];
    for (int i = 0; i < 25; ++i) {
        state[i] = 0;
    }
    uint8_t* state_bytes = reinterpret_cast<uint8_t*>(state);

    // Absorb: 34 bytes fits entirely in first rate block
    // XOR input into state
    for (int i = 0; i < 34; ++i) {
        state_bytes[i] ^= in[i];
    }

    // SHAKE-256 domain separator: 0x1F at byte 34
    state_bytes[34] ^= 0x1F;

    // Padding: 0x80 at byte (rate - 1) = 135
    state_bytes[rate - 1] ^= 0x80;

    // Permute
    keccak_f1600(state);

    // Squeeze: extract out_bytes
    int squeezed = 0;
    while (squeezed < out_bytes) {
        int to_copy = out_bytes - squeezed;
        if (to_copy > rate) to_copy = rate;

        for (int i = 0; i < to_copy; ++i) {
            out[squeezed + i] = state_bytes[i];
        }
        squeezed += to_copy;

        if (squeezed < out_bytes) {
            keccak_f1600(state);
        }
    }
}

// ============================================================================
// Kernel: Convert XOF bytes to y coefficients
// ============================================================================
// Grid: (B, l)
// Each block processes one polynomial
//
constexpr uint32_t Q = 8380417;

__global__ void kernel_xof_bytes_to_y(
    uint32_t* __restrict__ d_y,           // [B, l, n]
    const uint8_t* __restrict__ d_xof,    // [B * l, BYTES_PER_Y_POLY]
    const uint8_t* __restrict__ d_active, // [B]
    int B, int l, int n,
    uint32_t gamma1)
{
    int b = blockIdx.x;
    int j = blockIdx.y;

    if (b >= B || j >= l) return;

    // Zero output for inactive lanes
    uint32_t* y_out = d_y + (b * l + j) * n;
    if (d_active[b] == 0) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            y_out[i] = 0;
        }
        return;
    }

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // XOF output for this (b, j)
    const uint8_t* xof = d_xof + (b * l + j) * BYTES_PER_Y_POLY;

    // Bit mask depends on gamma1
    // gamma1 = 2^17 (ML-DSA-44) or 2^19 (ML-DSA-65/87)
    uint32_t mask = (gamma1 == (1u << 17)) ? 0x3FFFF : 0xFFFFF;

    for (int i = tid; i < n; i += stride) {
        // Extract 3 bytes
        uint32_t val = xof[3*i] | (xof[3*i + 1] << 8) | (xof[3*i + 2] << 16);
        val &= mask;

        // Reduce to [0, 2*gamma1 - 2]
        val = val % (2 * gamma1 - 1);

        // Map to [-(gamma1-1), gamma1-1]
        int32_t coeff = static_cast<int32_t>(val) - static_cast<int32_t>(gamma1 - 1);

        // Encode in [0, q)
        uint32_t encoded;
        if (coeff >= 0) {
            encoded = static_cast<uint32_t>(coeff);
        } else {
            encoded = static_cast<uint32_t>(static_cast<int64_t>(Q) + coeff);
        }

        y_out[i] = encoded;
    }
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

static void ensure_initialized() {
    if (g_batch_shake_initialized) return;

    cudaMemcpyToSymbol(KECCAK_RC, HOST_RC, sizeof(HOST_RC));
    cudaMemcpyToSymbol(KECCAK_RHO, HOST_RHO, sizeof(HOST_RHO));
    cudaMemcpyToSymbol(KECCAK_PI, HOST_PI, sizeof(HOST_PI));

    g_batch_shake_initialized = true;
}

void prepare_y_seeds_batch_gpu(
    const uint8_t* d_rhoprime,
    const uint16_t* d_attempts,
    const uint8_t* d_active,
    uint8_t* d_seeds,
    int B,
    int l,
    cudaStream_t stream)
{
    int total_jobs = B * l;
    dim3 block(256);
    dim3 grid((total_jobs + block.x - 1) / block.x);

    kernel_prepare_y_seeds<<<grid, block, 0, stream>>>(
        d_seeds, d_rhoprime, d_attempts, d_active, B, l);
}

void shake256_xof_batch_34b_gpu(
    const uint8_t* d_in,
    uint8_t* d_out,
    int count,
    int out_bytes,
    cudaStream_t stream)
{
    if (count == 0) return;

    ensure_initialized();

    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);

    kernel_shake256_xof_batch_34b<<<grid, block, 0, stream>>>(
        d_out, d_in, count, out_bytes);
}

void xof_bytes_to_y_batch_gpu(
    const uint8_t* d_xof_bytes,
    uint32_t* d_y,
    const uint8_t* d_active,
    int B,
    int l,
    int n,
    uint32_t gamma1,
    cudaStream_t stream)
{
    dim3 grid(B, l);
    dim3 block(256);

    kernel_xof_bytes_to_y<<<grid, block, 0, stream>>>(
        d_y, d_xof_bytes, d_active, B, l, n, gamma1);
}

void sample_y_gamma1_batch_v2_gpu(
    const uint8_t* d_rhoprime,
    const uint16_t* d_attempts,
    const uint8_t* d_active,
    uint32_t* d_y,
    uint8_t* d_work,
    int B,
    int l,
    int n,
    uint32_t gamma1,
    cudaStream_t stream)
{
    // Workspace layout:
    // [0, B*l*34): seeds
    // [B*l*34, B*l*34 + B*l*BYTES_PER_Y_POLY): XOF output
    uint8_t* d_seeds = d_work;
    uint8_t* d_xof = d_work + B * l * 34;

    int total_jobs = B * l;

    // Step 1: Prepare seeds (1 launch)
    prepare_y_seeds_batch_gpu(d_rhoprime, d_attempts, d_active, d_seeds, B, l, stream);

    // Step 2: Batched SHAKE-256 (1 launch)
    shake256_xof_batch_34b_gpu(d_seeds, d_xof, total_jobs, BYTES_PER_Y_POLY, stream);

    // Step 3: Convert XOF bytes to y coefficients (1 launch)
    xof_bytes_to_y_batch_gpu(d_xof, d_y, d_active, B, l, n, gamma1, stream);
}

} // namespace dilithium
} // namespace smoke
