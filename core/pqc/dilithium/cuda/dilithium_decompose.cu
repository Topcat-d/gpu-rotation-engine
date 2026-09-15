// dilithium_decompose.cu - CUDA power2round decomposition for Dilithium
// Card 14.2: t -> (t1, t0) decomposition
//
// FIPS 204 Power2Round:
//   a1 = (a + 2^(d-1)) >> d
//   a0 = a - a1 * 2^d
//
// We encode a0 in [0, q) for storage: negative values become q + a0.

#include "../include/dilithium_decompose.h"
#include <stdexcept>

namespace smoke {
namespace dilithium {

namespace {

// =============================================================================
// Device Functions
// =============================================================================

// Power2Round for a single coefficient.
// Input a is in [0, q).
// Output: a1 (high bits), a0 (low bits encoded in [0, q))
__device__ __forceinline__
void power2round_coeff_device(uint32_t a, uint32_t q, uint32_t d,
                              uint32_t& a1_out, uint32_t& a0_out)
{
    // Ensure a is in [0, q) - should already be, but be safe
    a = a % q;

    uint32_t two_d = 1u << d;
    uint32_t half = two_d >> 1;  // 2^(d-1)

    // Round a to nearest multiple of 2^d (round-to-nearest)
    uint32_t a1 = (a + half) >> d;

    // Compute a0 = a - a1 * 2^d
    // This can be negative (centered around 0)
    int32_t a0 = static_cast<int32_t>(a) - static_cast<int32_t>(a1 << d);

    // Encode a0 in [0, q): negative values become q + a0
    uint32_t a0_enc;
    if (a0 >= 0) {
        a0_enc = static_cast<uint32_t>(a0);
    } else {
        a0_enc = static_cast<uint32_t>(static_cast<int64_t>(q) + a0);
    }

    a1_out = a1;
    a0_out = a0_enc;
}

// =============================================================================
// Kernel: Power2Round on a vector of coefficients
// =============================================================================

__global__ void kernel_power2round_vec(
    const uint32_t* __restrict__ d_t,
    uint32_t* __restrict__ d_t1,
    uint32_t* __restrict__ d_t0,
    std::size_t coeff_count,
    uint32_t q,
    uint32_t d)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coeff_count) return;

    uint32_t a = d_t[idx];
    uint32_t a1, a0;
    power2round_coeff_device(a, q, d, a1, a0);

    d_t1[idx] = a1;
    d_t0[idx] = a0;
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

void power2round_vec_gpu(const Params& params,
                         const uint32_t* d_t,
                         uint32_t* d_t1,
                         uint32_t* d_t0,
                         cudaStream_t stream)
{
    if (!d_t || !d_t1 || !d_t0) {
        throw std::invalid_argument("power2round_vec_gpu: null pointer");
    }

    std::size_t coeff_count = t_vec_coeff_count(params);
    uint32_t q = params.q;
    uint32_t d = params.d;

    const int threads = 256;
    const int blocks = static_cast<int>((coeff_count + threads - 1) / threads);

    kernel_power2round_vec<<<blocks, threads, 0, stream>>>(
        d_t, d_t1, d_t0, coeff_count, q, d);
}

void power2round_poly_gpu(const Params& params,
                          const uint32_t* d_poly_in,
                          uint32_t* d_poly_t1,
                          uint32_t* d_poly_t0,
                          cudaStream_t stream)
{
    if (!d_poly_in || !d_poly_t1 || !d_poly_t0) {
        throw std::invalid_argument("power2round_poly_gpu: null pointer");
    }

    uint32_t q = params.q;
    uint32_t d = params.d;
    std::size_t n = params.n;

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    kernel_power2round_vec<<<blocks, threads, 0, stream>>>(
        d_poly_in, d_poly_t1, d_poly_t0, n, q, d);
}

} // namespace dilithium
} // namespace smoke
