// dilithium_pack.cu - CUDA packing for Dilithium keys
// Card 14.2: Pack pk and sk on GPU for zero host round-trips
//
// Uses async memcpy for seeds (rho, rhoprime) and reinterpret-cast for
// coefficient arrays (already little-endian uint32 on both CPU and GPU).

#include "../include/dilithium_pack.h"
#include <stdexcept>

namespace smoke {
namespace dilithium {

// =============================================================================
// Single-Key Packing
// =============================================================================

void pack_pk_gpu(const Params& params,
                 const uint8_t* d_rho,
                 const uint32_t* d_t1,
                 uint8_t* d_pk,
                 cudaStream_t stream)
{
    if (!d_rho || !d_t1 || !d_pk) {
        throw std::invalid_argument("pack_pk_gpu: null pointer");
    }

    const std::size_t rho_size = 32;
    const std::size_t t1_size = static_cast<std::size_t>(params.k) * params.n * sizeof(uint32_t);

    // pk = rho || t1
    // Copy rho (32 bytes)
    cudaMemcpyAsync(d_pk, d_rho, rho_size, cudaMemcpyDeviceToDevice, stream);

    // Copy t1 as bytes (already little-endian)
    cudaMemcpyAsync(d_pk + rho_size,
                    reinterpret_cast<const uint8_t*>(d_t1),
                    t1_size,
                    cudaMemcpyDeviceToDevice,
                    stream);
}

void pack_sk_gpu(const Params& params,
                 const uint8_t* d_rho,
                 const uint8_t* d_rhoprime,
                 const uint32_t* d_s1,
                 const uint32_t* d_s2,
                 const uint32_t* d_t0,
                 uint8_t* d_sk,
                 cudaStream_t stream)
{
    if (!d_rho || !d_rhoprime || !d_s1 || !d_s2 || !d_t0 || !d_sk) {
        throw std::invalid_argument("pack_sk_gpu: null pointer");
    }

    const std::size_t seed_size = 32;
    const std::size_t s1_size = static_cast<std::size_t>(params.l) * params.n * sizeof(uint32_t);
    const std::size_t s2_size = static_cast<std::size_t>(params.k) * params.n * sizeof(uint32_t);
    const std::size_t t0_size = static_cast<std::size_t>(params.k) * params.n * sizeof(uint32_t);

    std::size_t offset = 0;

    // sk = rho || rhoprime || s1 || s2 || t0

    // Copy rho
    cudaMemcpyAsync(d_sk + offset, d_rho, seed_size, cudaMemcpyDeviceToDevice, stream);
    offset += seed_size;

    // Copy rhoprime
    cudaMemcpyAsync(d_sk + offset, d_rhoprime, seed_size, cudaMemcpyDeviceToDevice, stream);
    offset += seed_size;

    // Copy s1
    cudaMemcpyAsync(d_sk + offset,
                    reinterpret_cast<const uint8_t*>(d_s1),
                    s1_size,
                    cudaMemcpyDeviceToDevice,
                    stream);
    offset += s1_size;

    // Copy s2
    cudaMemcpyAsync(d_sk + offset,
                    reinterpret_cast<const uint8_t*>(d_s2),
                    s2_size,
                    cudaMemcpyDeviceToDevice,
                    stream);
    offset += s2_size;

    // Copy t0
    cudaMemcpyAsync(d_sk + offset,
                    reinterpret_cast<const uint8_t*>(d_t0),
                    t0_size,
                    cudaMemcpyDeviceToDevice,
                    stream);
}

// =============================================================================
// Batch Packing
// =============================================================================

void pack_pk_batch_gpu(const Params& params,
                       const uint8_t* d_rho,
                       const uint32_t* d_t1,
                       uint8_t* d_pk,
                       int batch_size,
                       cudaStream_t stream)
{
    if (!d_rho || !d_t1 || !d_pk) {
        throw std::invalid_argument("pack_pk_batch_gpu: null pointer");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("pack_pk_batch_gpu: batch_size must be > 0");
    }

    const std::size_t rho_size = 32;
    const std::size_t t1_coeffs = static_cast<std::size_t>(params.k) * params.n;
    const std::size_t t1_size = t1_coeffs * sizeof(uint32_t);
    const std::size_t pk_size = pk_simple_size(params);

    for (int b = 0; b < batch_size; ++b) {
        const uint8_t* src_rho = d_rho + b * rho_size;
        const uint32_t* src_t1 = d_t1 + b * t1_coeffs;
        uint8_t* dst_pk = d_pk + b * pk_size;

        // pk[b] = rho[b] || t1[b]
        cudaMemcpyAsync(dst_pk, src_rho, rho_size, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(dst_pk + rho_size,
                        reinterpret_cast<const uint8_t*>(src_t1),
                        t1_size,
                        cudaMemcpyDeviceToDevice,
                        stream);
    }
}

void pack_sk_batch_gpu(const Params& params,
                       const uint8_t* d_rho,
                       const uint8_t* d_rhoprime,
                       const uint32_t* d_s1,
                       const uint32_t* d_s2,
                       const uint32_t* d_t0,
                       uint8_t* d_sk,
                       int batch_size,
                       cudaStream_t stream)
{
    if (!d_rho || !d_rhoprime || !d_s1 || !d_s2 || !d_t0 || !d_sk) {
        throw std::invalid_argument("pack_sk_batch_gpu: null pointer");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("pack_sk_batch_gpu: batch_size must be > 0");
    }

    const std::size_t seed_size = 32;
    const std::size_t s1_coeffs = static_cast<std::size_t>(params.l) * params.n;
    const std::size_t s2_coeffs = static_cast<std::size_t>(params.k) * params.n;
    const std::size_t t0_coeffs = static_cast<std::size_t>(params.k) * params.n;
    const std::size_t s1_size = s1_coeffs * sizeof(uint32_t);
    const std::size_t s2_size = s2_coeffs * sizeof(uint32_t);
    const std::size_t t0_size = t0_coeffs * sizeof(uint32_t);
    const std::size_t sk_size = sk_simple_size(params);

    for (int b = 0; b < batch_size; ++b) {
        const uint8_t* src_rho = d_rho + b * seed_size;
        const uint8_t* src_rhoprime = d_rhoprime + b * seed_size;
        const uint32_t* src_s1 = d_s1 + b * s1_coeffs;
        const uint32_t* src_s2 = d_s2 + b * s2_coeffs;
        const uint32_t* src_t0 = d_t0 + b * t0_coeffs;
        uint8_t* dst_sk = d_sk + b * sk_size;

        std::size_t offset = 0;

        // sk[b] = rho[b] || rhoprime[b] || s1[b] || s2[b] || t0[b]
        cudaMemcpyAsync(dst_sk + offset, src_rho, seed_size, cudaMemcpyDeviceToDevice, stream);
        offset += seed_size;

        cudaMemcpyAsync(dst_sk + offset, src_rhoprime, seed_size, cudaMemcpyDeviceToDevice, stream);
        offset += seed_size;

        cudaMemcpyAsync(dst_sk + offset,
                        reinterpret_cast<const uint8_t*>(src_s1),
                        s1_size,
                        cudaMemcpyDeviceToDevice,
                        stream);
        offset += s1_size;

        cudaMemcpyAsync(dst_sk + offset,
                        reinterpret_cast<const uint8_t*>(src_s2),
                        s2_size,
                        cudaMemcpyDeviceToDevice,
                        stream);
        offset += s2_size;

        cudaMemcpyAsync(dst_sk + offset,
                        reinterpret_cast<const uint8_t*>(src_t0),
                        t0_size,
                        cudaMemcpyDeviceToDevice,
                        stream);
    }
}

} // namespace dilithium
} // namespace smoke
