// dilithium_expandA.cpp - ExpandA implementation for Dilithium
// Card 12: Generate matrix A in NTT domain from seed rho
//
// This is a skeleton implementation for Card 12.
// The actual SHAKE/sampling is done on CPU for now;
// full GPU implementation comes in a follow-up card.

#include "../include/dilithium_expandA.h"
#include "../include/dilithium_ntt.cuh"
#include "../include/dilithium_shake.h"

#include <vector>
#include <cstring>
#include <stdexcept>

namespace smoke {
namespace dilithium {

namespace {

// ============================================================================
// Placeholder Rejection Sampler
// ============================================================================

// NOTE: This is a placeholder sampler that interprets SHAKE output as
// 3-byte values and rejects those >= q. This matches the structure of
// Dilithium's poly_uniform but may not be 100% spec-accurate yet.
//
// The placeholder allows us to exercise the ExpandA pipeline; we can
// refine the sampler in a follow-up card.

void sample_poly_uniform_placeholder(uint32_t* poly,
                                     const uint8_t* buf,
                                     std::size_t buf_len,
                                     const Params& params) {
    const uint32_t q = params.q;
    const std::size_t n = params.n;

    std::size_t pos = 0;
    std::size_t coeff_idx = 0;

    // Dilithium uses 3-byte samples with rejection
    while (coeff_idx < n && pos + 3 <= buf_len) {
        // Extract 3 bytes as little-endian 24-bit value
        uint32_t t = static_cast<uint32_t>(buf[pos]) |
                     (static_cast<uint32_t>(buf[pos + 1]) << 8) |
                     (static_cast<uint32_t>(buf[pos + 2]) << 16);

        // Mask to 23 bits (Dilithium uses t & 0x7FFFFF)
        t &= 0x7FFFFF;

        // Reject if >= q
        if (t < q) {
            poly[coeff_idx++] = t;
        }

        pos += 3;
    }

    // If we didn't fill all coefficients, fill rest with zeros
    // (shouldn't happen with enough SHAKE output)
    while (coeff_idx < n) {
        poly[coeff_idx++] = 0;
    }
}

// ============================================================================
// Host-Side SHAKE (placeholder until GPU SHAKE is wired)
// ============================================================================

// For Card 12, we compute SHAKE on CPU using a simple implementation.
// This can be replaced with GPU SHAKE in a follow-up card.

// Keccak-f[1600] constants
static const uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static const int KECCAK_RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

static const int KECCAK_PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f1600_host(uint64_t* state) {
    for (int round = 0; round < 24; ++round) {
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

        // Rho and Pi
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
}

void shake128_xof_host(const uint8_t* input, std::size_t in_len,
                       uint8_t* output, std::size_t out_len) {
    const int rate_bytes = 168;  // SHAKE-128 rate

    uint64_t state[25] = {0};
    uint8_t* state_bytes = reinterpret_cast<uint8_t*>(state);

    // Absorb
    std::size_t absorbed = 0;
    while (absorbed + rate_bytes <= in_len) {
        for (int i = 0; i < rate_bytes; ++i) {
            state_bytes[i] ^= input[absorbed + i];
        }
        keccak_f1600_host(state);
        absorbed += rate_bytes;
    }

    // Absorb remaining + padding
    std::size_t remaining = in_len - absorbed;
    for (std::size_t i = 0; i < remaining; ++i) {
        state_bytes[i] ^= input[absorbed + i];
    }
    state_bytes[remaining] ^= 0x1F;  // SHAKE domain separator
    state_bytes[rate_bytes - 1] ^= 0x80;  // Final padding bit
    keccak_f1600_host(state);

    // Squeeze
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
            keccak_f1600_host(state);
        }
    }
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

void expand_A_ntt_gpu(const Params& params,
                      const uint8_t rho[SEEDBYTES],
                      uint32_t* d_A_hat,
                      cudaStream_t stream) {
    if (!d_A_hat) {
        throw std::invalid_argument("expand_A_ntt_gpu: d_A_hat is null");
    }

    const std::size_t n = params.n;  // 256
    const std::size_t poly_count = expand_A_poly_count(params);
    const std::size_t coeff_count = expand_A_coeff_count(params);

    // Allocate host workspace for A in coefficient domain
    std::vector<uint32_t> h_A(coeff_count, 0);

    // For each (i, j), generate polynomial using SHAKE-128
    const std::size_t input_len = 34;  // rho (32) + j (1) + i (1)
    const std::size_t shake_out_len = 3 * n + 128;  // Enough for rejection sampling

    std::vector<uint8_t> shake_in(input_len);
    std::vector<uint8_t> shake_out(shake_out_len);

    for (int i = 0; i < static_cast<int>(params.k); ++i) {
        for (int j = 0; j < static_cast<int>(params.l); ++j) {
            // Build SHAKE input: rho || j || i
            make_expandA_input(shake_in.data(), rho,
                               static_cast<uint8_t>(i),
                               static_cast<uint8_t>(j));

            // SHAKE-128 XOF
            shake128_xof_host(shake_in.data(), input_len,
                              shake_out.data(), shake_out_len);

            // Sample polynomial with rejection
            uint32_t* poly_ptr = h_A.data() + ((i * params.l) + j) * n;
            sample_poly_uniform_placeholder(poly_ptr,
                                            shake_out.data(), shake_out_len,
                                            params);
        }
    }

    // Copy A (coefficient domain) to device
    cudaMemcpyAsync(d_A_hat, h_A.data(),
                    coeff_count * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);

    // Apply NTT to all k*l polynomials to get A-hat (NTT domain)
    ntt_forward_batch(d_A_hat, poly_count, params, stream, NTTImpl::AUTO);
}

} // namespace dilithium
} // namespace smoke
