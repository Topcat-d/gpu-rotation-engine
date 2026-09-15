// dilithium_shake.cu - SHAKE XOF GPU implementation for Dilithium
// Card 11: GPU Keccak-f[1600] + SHAKE-128/256 XOF
//
// Implements SHAKE directly using Keccak-f[1600] permutation.
// This is a self-contained implementation for Dilithium, not relying on external libs.

#include "../include/dilithium_shake.h"
#include <cstdio>
#include <vector>

namespace smoke {
namespace dilithium {

namespace {

// ============================================================================
// Keccak-f[1600] Constants
// ============================================================================

// Keccak state: 5x5 matrix of 64-bit lanes = 1600 bits = 200 bytes
constexpr int KECCAK_LANES = 25;
constexpr int KECCAK_STATE_BYTES = 200;

// Round constants for Keccak-f[1600] (24 rounds)
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

// Rotation offsets for rho step
__constant__ int KECCAK_RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

// Pi step permutation indices
__constant__ int KECCAK_PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

// Host copies for initialization
static const uint64_t HOST_KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static const int HOST_KECCAK_RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

static const int HOST_KECCAK_PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

static bool g_shake_initialized = false;

// ============================================================================
// Keccak-f[1600] Permutation (Device)
// ============================================================================

__device__ __forceinline__
uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Single round of Keccak-f[1600]
__device__ void keccak_round(uint64_t* state, int round) {
    uint64_t C[5], D[5], B[25];

    // Theta step
    for (int x = 0; x < 5; ++x) {
        C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    for (int x = 0; x < 5; ++x) {
        D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
    }
    for (int i = 0; i < 25; ++i) {
        state[i] ^= D[i % 5];
    }

    // Rho and Pi steps combined
    B[0] = state[0];
    uint64_t t = state[1];
    for (int i = 0; i < 24; ++i) {
        int j = KECCAK_PI[i];
        B[j] = rotl64(t, KECCAK_RHO[i]);
        t = state[j];
    }

    // Chi step
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            int idx = y * 5 + x;
            state[idx] = B[idx] ^ ((~B[y * 5 + (x + 1) % 5]) & B[y * 5 + (x + 2) % 5]);
        }
    }

    // Iota step
    state[0] ^= KECCAK_RC[round];
}

// Full Keccak-f[1600] permutation (24 rounds)
__device__ void keccak_f1600(uint64_t* state) {
    for (int round = 0; round < 24; ++round) {
        keccak_round(state, round);
    }
}

// ============================================================================
// SHAKE XOF Kernel (1 thread per job, baseline)
// ============================================================================

// SHAKE padding: domain separator is 0x1F (vs 0x06 for SHA-3)
// Padding: input || 0x1F || 0x00...0x00 || 0x80

__global__ void kernel_shake_xof_baseline(
    const uint8_t* const* __restrict__ d_inputs,
    const std::size_t* __restrict__ in_lens,
    uint8_t* const* __restrict__ d_outputs,
    const std::size_t* __restrict__ out_lens,
    std::size_t batch_size,
    int rate_bytes  // 168 for SHAKE128, 136 for SHAKE256
) {
    std::size_t job_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (job_idx >= batch_size) return;

    const uint8_t* input = d_inputs[job_idx];
    std::size_t in_len = in_lens[job_idx];
    uint8_t* output = d_outputs[job_idx];
    std::size_t out_len = out_lens[job_idx];

    // Initialize state to zero
    uint64_t state[KECCAK_LANES];
    for (int i = 0; i < KECCAK_LANES; ++i) {
        state[i] = 0;
    }

    uint8_t* state_bytes = reinterpret_cast<uint8_t*>(state);

    // Absorb phase: process input in rate-sized blocks
    std::size_t absorbed = 0;
    while (absorbed + rate_bytes <= in_len) {
        // XOR rate_bytes into state
        for (int i = 0; i < rate_bytes; ++i) {
            state_bytes[i] ^= input[absorbed + i];
        }
        keccak_f1600(state);
        absorbed += rate_bytes;
    }

    // Absorb remaining bytes + padding
    std::size_t remaining = in_len - absorbed;
    for (std::size_t i = 0; i < remaining; ++i) {
        state_bytes[i] ^= input[absorbed + i];
    }

    // SHAKE domain separator: 0x1F
    state_bytes[remaining] ^= 0x1F;

    // Final bit of padding: 0x80 at end of rate
    state_bytes[rate_bytes - 1] ^= 0x80;

    keccak_f1600(state);

    // Squeeze phase: extract out_len bytes
    std::size_t squeezed = 0;
    while (squeezed < out_len) {
        std::size_t to_copy = out_len - squeezed;
        if (to_copy > static_cast<std::size_t>(rate_bytes)) {
            to_copy = rate_bytes;
        }

        for (std::size_t i = 0; i < to_copy; ++i) {
            output[squeezed + i] = state_bytes[i];
        }
        squeezed += to_copy;

        if (squeezed < out_len) {
            keccak_f1600(state);
        }
    }
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

void shake_init() {
    if (g_shake_initialized) return;

    cudaMemcpyToSymbol(KECCAK_RC, HOST_KECCAK_RC, sizeof(HOST_KECCAK_RC));
    cudaMemcpyToSymbol(KECCAK_RHO, HOST_KECCAK_RHO, sizeof(HOST_KECCAK_RHO));
    cudaMemcpyToSymbol(KECCAK_PI, HOST_KECCAK_PI, sizeof(HOST_KECCAK_PI));

    g_shake_initialized = true;
}

void shake_xof_batch(SHAKEKind kind,
                     const SHAKEJob* jobs,
                     std::size_t batch_size,
                     cudaStream_t stream) {
    if (batch_size == 0) return;

    if (!g_shake_initialized) {
        shake_init();
    }

    // Rate in bytes: SHAKE128 = 168 (1600 - 2*128)/8, SHAKE256 = 136 (1600 - 2*256)/8
    int rate_bytes = (kind == SHAKEKind::SHAKE128) ? 168 : 136;

    // Prepare device arrays of pointers and lengths
    std::vector<const uint8_t*> h_inputs(batch_size);
    std::vector<std::size_t> h_in_lens(batch_size);
    std::vector<uint8_t*> h_outputs(batch_size);
    std::vector<std::size_t> h_out_lens(batch_size);

    for (std::size_t i = 0; i < batch_size; ++i) {
        h_inputs[i] = jobs[i].d_in;
        h_in_lens[i] = jobs[i].in_len;
        h_outputs[i] = jobs[i].d_out;
        h_out_lens[i] = jobs[i].out_len;
    }

    // Allocate device arrays
    const uint8_t** d_inputs;
    std::size_t* d_in_lens;
    uint8_t** d_outputs;
    std::size_t* d_out_lens;

    cudaMalloc(&d_inputs, batch_size * sizeof(const uint8_t*));
    cudaMalloc(&d_in_lens, batch_size * sizeof(std::size_t));
    cudaMalloc(&d_outputs, batch_size * sizeof(uint8_t*));
    cudaMalloc(&d_out_lens, batch_size * sizeof(std::size_t));

    cudaMemcpyAsync(d_inputs, h_inputs.data(), batch_size * sizeof(const uint8_t*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_in_lens, h_in_lens.data(), batch_size * sizeof(std::size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_outputs, h_outputs.data(), batch_size * sizeof(uint8_t*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_lens, h_out_lens.data(), batch_size * sizeof(std::size_t),
                    cudaMemcpyHostToDevice, stream);

    // Launch kernel: 1 thread per job (baseline, can optimize later)
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    kernel_shake_xof_baseline<<<grid, block, 0, stream>>>(
        d_inputs, d_in_lens, d_outputs, d_out_lens, batch_size, rate_bytes);

    // Cleanup (sync first to ensure kernel is done)
    cudaStreamSynchronize(stream);
    cudaFree(d_inputs);
    cudaFree(d_in_lens);
    cudaFree(d_outputs);
    cudaFree(d_out_lens);
}

} // namespace dilithium
} // namespace smoke
