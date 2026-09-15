// dilithium_cuda_bindings.cpp - Pybind11 bindings for Dilithium CUDA functions
// Card 14.1: Wire all Dilithium CUDA kernels to Python
//
// This file exports:
//   - CUDA memory management (cuda_malloc, cuda_free, cuda_memcpy_htod/dtoh)
//   - Parameter access (dilithium_get_params)
//   - NTT functions (ntt_forward_batch_gpu, ntt_inverse_batch_gpu)
//   - SHAKE XOF functions (shake_xof_batch_gpu)
//   - ExpandA (expand_A_ntt_gpu)
//   - Secret sampling (sample_s1_gpu, sample_s2_gpu)
//   - Keygen core (keygen_core_gpu)
//   - Decomposition (power2round_vec_gpu)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "../include/dilithium_params.h"
#include "../include/dilithium_ntt.cuh"
#include "../include/dilithium_shake.h"
#include "../include/dilithium_expandA.h"
#include "../include/dilithium_sampling.h"
#include "../include/dilithium_keygen.h"
#include "../include/dilithium_decompose.h"
#include "../include/dilithium_pack.h"  // Card 14.2: GPU packing
#include "../include/dilithium_w1.h"  // Card 16.1: w = A*y and w1 = highbits(w)
#include "../include/dilithium_sign.h"  // Card 17: GPU sign/verify
#include "../include/dilithium_sign_batch.h"  // Card 19B: Batch-dimension kernels
#include "../include/dilithium_shake_batch.h"  // Card 19C: Batched SHAKE for y sampling
#include "../include/dilithium_challenge.h"  // Card 14.3: GPU challenge + unified signing

namespace py = pybind11;

using namespace smoke::dilithium;

// =============================================================================
// Mode Parsing Helper
// =============================================================================

static Mode parse_mode(const std::string& mode_str) {
    if (mode_str == "ML_DSA_44" || mode_str == "ML-DSA-44") return Mode::ML_DSA_44;
    if (mode_str == "ML_DSA_65" || mode_str == "ML-DSA-65") return Mode::ML_DSA_65;
    if (mode_str == "ML_DSA_87" || mode_str == "ML-DSA-87") return Mode::ML_DSA_87;
    throw std::runtime_error("Unknown mode: " + mode_str);
}

// =============================================================================
// CUDA Memory Management Wrappers
// =============================================================================

static std::uintptr_t py_cuda_malloc(std::size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                 cudaGetErrorString(err));
    }
    return reinterpret_cast<std::uintptr_t>(ptr);
}

static void py_cuda_free(std::uintptr_t ptr) {
    cudaError_t err = cudaFree(reinterpret_cast<void*>(ptr));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaFree failed: ") +
                                 cudaGetErrorString(err));
    }
}

static void py_cuda_memcpy_htod(std::uintptr_t dst,
                                 py::array_t<uint32_t> src,
                                 std::size_t bytes) {
    auto buf = src.request();
    if (buf.size * sizeof(uint32_t) < bytes) {
        throw std::runtime_error("Source array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(dst),
                                  buf.ptr,
                                  bytes,
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyHtoD failed: ") +
                                 cudaGetErrorString(err));
    }
}

static void py_cuda_memcpy_dtoh(py::array_t<uint32_t> dst,
                                 std::uintptr_t src,
                                 std::size_t bytes) {
    auto buf = dst.request();
    if (buf.size * sizeof(uint32_t) < bytes) {
        throw std::runtime_error("Destination array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(buf.ptr,
                                  reinterpret_cast<void*>(src),
                                  bytes,
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyDtoH failed: ") +
                                 cudaGetErrorString(err));
    }
}

// =============================================================================
// Parameter Access
// =============================================================================

static py::dict py_dilithium_get_params(const std::string& mode_str) {
    Mode mode = parse_mode(mode_str);
    const Params& p = get_params(mode);

    py::dict result;
    result["n"] = p.n;
    result["q"] = p.q;
    result["k"] = p.k;
    result["l"] = p.l;
    result["d"] = p.d;
    result["tau"] = p.tau;
    result["eta"] = p.eta;
    result["omega"] = p.omega;
    result["gamma1"] = p.gamma1;
    result["gamma2"] = p.gamma2;
    result["beta"] = p.beta;
    result["pk_size"] = p.pk_size;
    result["sk_size"] = p.sk_size;
    result["sig_size"] = p.sig_size;
    return result;
}

// =============================================================================
// NTT Functions
// =============================================================================

static void py_ntt_init() {
    ntt_init();
}

static bool py_ntt_is_initialized() {
    return ntt_is_initialized();
}

static void py_ntt_forward_batch_gpu(const std::string& mode_str,
                                      std::uintptr_t d_coeffs,
                                      std::size_t poly_count) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);
    auto* ptr = reinterpret_cast<uint32_t*>(d_coeffs);
    ntt_forward_batch(ptr, poly_count, params, 0, NTTImpl::AUTO);
}

static void py_ntt_inverse_batch_gpu(const std::string& mode_str,
                                      std::uintptr_t d_coeffs,
                                      std::size_t poly_count) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);
    auto* ptr = reinterpret_cast<uint32_t*>(d_coeffs);
    ntt_inverse_batch(ptr, poly_count, params, 0, NTTImpl::AUTO);
}

static void py_pointwise_multiply_batch_gpu(const std::string& mode_str,
                                             std::uintptr_t d_out,
                                             std::uintptr_t d_a,
                                             std::uintptr_t d_b,
                                             std::size_t batch_size) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);
    pointwise_multiply_batch(reinterpret_cast<uint32_t*>(d_out),
                             reinterpret_cast<const uint32_t*>(d_a),
                             reinterpret_cast<const uint32_t*>(d_b),
                             batch_size, params, 0);
}

static void py_pointwise_add_batch_gpu(const std::string& mode_str,
                                        std::uintptr_t d_out,
                                        std::uintptr_t d_a,
                                        std::uintptr_t d_b,
                                        std::size_t batch_size) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);
    pointwise_add_batch(reinterpret_cast<uint32_t*>(d_out),
                        reinterpret_cast<const uint32_t*>(d_a),
                        reinterpret_cast<const uint32_t*>(d_b),
                        batch_size, params, 0);
}

// =============================================================================
// SHAKE XOF Functions
// =============================================================================

// Single SHAKE128 call
static void py_shake128_xof_gpu(std::uintptr_t d_in,
                                 std::size_t in_len,
                                 std::uintptr_t d_out,
                                 std::size_t out_len) {
    shake128_xof(reinterpret_cast<const uint8_t*>(d_in), in_len,
                 reinterpret_cast<uint8_t*>(d_out), out_len, 0);
}

// Single SHAKE256 call
static void py_shake256_xof_gpu(std::uintptr_t d_in,
                                 std::size_t in_len,
                                 std::uintptr_t d_out,
                                 std::size_t out_len) {
    shake256_xof(reinterpret_cast<const uint8_t*>(d_in), in_len,
                 reinterpret_cast<uint8_t*>(d_out), out_len, 0);
}

// =============================================================================
// ExpandA Functions
// =============================================================================

static void py_expand_A_ntt_gpu(const std::string& mode_str,
                                 py::bytes rho_bytes,
                                 std::uintptr_t d_A_hat) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    std::string rho_str = rho_bytes;
    if (rho_str.size() != 32) {
        throw std::runtime_error("rho must be 32 bytes");
    }
    const uint8_t* rho = reinterpret_cast<const uint8_t*>(rho_str.data());

    expand_A_ntt_gpu(params, rho, reinterpret_cast<uint32_t*>(d_A_hat), 0);
}

// =============================================================================
// Secret Sampling Functions
// =============================================================================

static void py_sample_s1_gpu(const std::string& mode_str,
                              py::bytes rhoprime_bytes,
                              std::uintptr_t d_s1) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    std::string rhoprime_str = rhoprime_bytes;
    if (rhoprime_str.size() != 32) {
        throw std::runtime_error("rhoprime must be 32 bytes");
    }
    const uint8_t* rhoprime = reinterpret_cast<const uint8_t*>(rhoprime_str.data());

    sample_s1_gpu(params, rhoprime, reinterpret_cast<uint32_t*>(d_s1), 0);
}

static void py_sample_s2_gpu(const std::string& mode_str,
                              py::bytes rhoprime_bytes,
                              std::uintptr_t d_s2) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    std::string rhoprime_str = rhoprime_bytes;
    if (rhoprime_str.size() != 32) {
        throw std::runtime_error("rhoprime must be 32 bytes");
    }
    const uint8_t* rhoprime = reinterpret_cast<const uint8_t*>(rhoprime_str.data());

    sample_s2_gpu(params, rhoprime, reinterpret_cast<uint32_t*>(d_s2), 0);
}

static void py_sample_poly_eta_gpu(const std::string& mode_str,
                                    py::bytes rhoprime_bytes,
                                    uint16_t nonce,
                                    std::uintptr_t d_poly) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    std::string rhoprime_str = rhoprime_bytes;
    if (rhoprime_str.size() != 32) {
        throw std::runtime_error("rhoprime must be 32 bytes");
    }
    const uint8_t* rhoprime = reinterpret_cast<const uint8_t*>(rhoprime_str.data());

    sample_poly_eta_gpu(params, rhoprime, nonce, reinterpret_cast<uint32_t*>(d_poly), 0);
}

// =============================================================================
// Keygen Core Functions
// =============================================================================

static void py_keygen_core_gpu(const std::string& mode_str,
                                py::bytes rho_bytes,
                                py::bytes rhoprime_bytes,
                                std::uintptr_t d_A_hat,
                                std::uintptr_t d_s1,
                                std::uintptr_t d_s2,
                                std::uintptr_t d_t) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    std::string rho_str = rho_bytes;
    std::string rhoprime_str = rhoprime_bytes;
    if (rho_str.size() != 32 || rhoprime_str.size() != 32) {
        throw std::runtime_error("rho and rhoprime must be 32 bytes");
    }

    const uint8_t* rho = reinterpret_cast<const uint8_t*>(rho_str.data());
    const uint8_t* rhoprime = reinterpret_cast<const uint8_t*>(rhoprime_str.data());

    keygen_core_gpu(params, rho, rhoprime,
                    reinterpret_cast<uint32_t*>(d_A_hat),
                    reinterpret_cast<uint32_t*>(d_s1),
                    reinterpret_cast<uint32_t*>(d_s2),
                    reinterpret_cast<uint32_t*>(d_t), 0);
}

static void py_compute_t_gpu(const std::string& mode_str,
                              std::uintptr_t d_A_hat,
                              std::uintptr_t d_s1,
                              std::uintptr_t d_s2,
                              std::uintptr_t d_t) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    compute_t_As1_plus_s2_ntt_gpu(params,
                                   reinterpret_cast<const uint32_t*>(d_A_hat),
                                   reinterpret_cast<const uint32_t*>(d_s1),
                                   reinterpret_cast<const uint32_t*>(d_s2),
                                   reinterpret_cast<uint32_t*>(d_t), 0);
}

// =============================================================================
// Decomposition Functions
// =============================================================================

static void py_power2round_vec_gpu(const std::string& mode_str,
                                    std::uintptr_t d_t,
                                    std::uintptr_t d_t1,
                                    std::uintptr_t d_t0) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    power2round_vec_gpu(params,
                        reinterpret_cast<const uint32_t*>(d_t),
                        reinterpret_cast<uint32_t*>(d_t1),
                        reinterpret_cast<uint32_t*>(d_t0), 0);
}

// =============================================================================
// Card 14.2: Full Keygen with GPU Power2Round and Packing
// =============================================================================

static void py_keygen_full_gpu_v2(const std::string& mode_str,
                                   py::bytes rho_bytes,
                                   py::bytes rhoprime_bytes,
                                   std::uintptr_t d_pk,
                                   std::uintptr_t d_sk) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    std::string rho_str = rho_bytes;
    std::string rhoprime_str = rhoprime_bytes;
    if (rho_str.size() != 32 || rhoprime_str.size() != 32) {
        throw std::runtime_error("rho and rhoprime must be 32 bytes");
    }

    const uint8_t* rho = reinterpret_cast<const uint8_t*>(rho_str.data());
    const uint8_t* rhoprime = reinterpret_cast<const uint8_t*>(rhoprime_str.data());

    keygen_full_gpu(params, rho, rhoprime,
                    reinterpret_cast<uint8_t*>(d_pk),
                    reinterpret_cast<uint8_t*>(d_sk), 0);
}

static void py_keygen_full_batch_gpu(const std::string& mode_str,
                                      py::bytes h_rho_bytes,
                                      py::bytes h_rhoprime_bytes,
                                      std::uintptr_t d_pk,
                                      std::uintptr_t d_sk,
                                      int batch_size) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    std::string rho_str = h_rho_bytes;
    std::string rhoprime_str = h_rhoprime_bytes;
    if (rho_str.size() != static_cast<std::size_t>(batch_size) * 32) {
        throw std::runtime_error("h_rho must be batch_size * 32 bytes");
    }
    if (rhoprime_str.size() != static_cast<std::size_t>(batch_size) * 32) {
        throw std::runtime_error("h_rhoprime must be batch_size * 32 bytes");
    }

    const uint8_t* h_rho = reinterpret_cast<const uint8_t*>(rho_str.data());
    const uint8_t* h_rhoprime = reinterpret_cast<const uint8_t*>(rhoprime_str.data());

    keygen_full_batch_gpu(params, h_rho, h_rhoprime,
                          reinterpret_cast<uint8_t*>(d_pk),
                          reinterpret_cast<uint8_t*>(d_sk),
                          batch_size, 0);
}

// Helper: copy bytes from device to host and return as py::bytes
static py::bytes py_cuda_memcpy_dtoh_bytes_ret(std::uintptr_t src, std::size_t bytes) {
    std::string result(bytes, '\0');
    cudaError_t err = cudaMemcpy(&result[0],
                                  reinterpret_cast<void*>(src),
                                  bytes,
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyDtoH bytes failed: ") +
                                 cudaGetErrorString(err));
    }
    return py::bytes(result);
}

// =============================================================================
// W1 Functions (Card 16.1: w = A*y and w1 = highbits(w))
// =============================================================================

static void py_compute_w_gpu(const std::string& mode_str,
                              std::uintptr_t d_A_hat,
                              std::uintptr_t d_y,
                              std::uintptr_t d_w) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    compute_w_gpu(params,
                  reinterpret_cast<const uint32_t*>(d_A_hat),
                  reinterpret_cast<const uint32_t*>(d_y),
                  reinterpret_cast<uint32_t*>(d_w), 0);
}

static void py_decompose_w_gpu(const std::string& mode_str,
                                std::uintptr_t d_w,
                                std::uintptr_t d_w1,
                                std::uintptr_t d_w0) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    decompose_w_gpu(params,
                    reinterpret_cast<const uint32_t*>(d_w),
                    reinterpret_cast<uint32_t*>(d_w1),
                    reinterpret_cast<uint32_t*>(d_w0), 0);
}

// =============================================================================
// Sign Functions (Card 17)
// =============================================================================

// Helper: memcpy bytes host->device
static void py_cuda_memcpy_htod_bytes(std::uintptr_t dst,
                                       py::array_t<uint8_t> src,
                                       std::size_t bytes) {
    auto buf = src.request();
    if (buf.size < static_cast<py::ssize_t>(bytes)) {
        throw std::runtime_error("Source array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(dst),
                                  buf.ptr,
                                  bytes,
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyHtoD bytes failed: ") +
                                 cudaGetErrorString(err));
    }
}

// Helper: memcpy bytes device->host
static void py_cuda_memcpy_dtoh_bytes(py::array_t<uint8_t> dst,
                                       std::uintptr_t src,
                                       std::size_t bytes) {
    auto buf = dst.request();
    if (buf.size < static_cast<py::ssize_t>(bytes)) {
        throw std::runtime_error("Destination array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(buf.ptr,
                                  reinterpret_cast<void*>(src),
                                  bytes,
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyDtoH bytes failed: ") +
                                 cudaGetErrorString(err));
    }
}

// Helper: memcpy int8 host->device
static void py_cuda_memcpy_htod_int8(std::uintptr_t dst,
                                      py::array_t<int8_t> src,
                                      std::size_t count) {
    auto buf = src.request();
    if (buf.size < static_cast<py::ssize_t>(count)) {
        throw std::runtime_error("Source array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(dst),
                                  buf.ptr,
                                  count,
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyHtoD int8 failed: ") +
                                 cudaGetErrorString(err));
    }
}

static void py_sample_y_gamma1_gpu(const std::string& mode_str,
                                    std::uintptr_t d_rhoprime,
                                    uint32_t kappa,
                                    std::uintptr_t d_y) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    sample_y_gamma1_gpu(params,
                        reinterpret_cast<const uint8_t*>(d_rhoprime),
                        kappa,
                        reinterpret_cast<uint32_t*>(d_y), 0);
}

static void py_compute_z_gpu(const std::string& mode_str,
                              std::uintptr_t d_y,
                              std::uintptr_t d_c,
                              std::uintptr_t d_s1,
                              std::uintptr_t d_z) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    compute_z_gpu(params,
                  reinterpret_cast<const uint32_t*>(d_y),
                  reinterpret_cast<const int8_t*>(d_c),
                  reinterpret_cast<const uint32_t*>(d_s1),
                  reinterpret_cast<uint32_t*>(d_z), 0);
}

static void py_compute_r_gpu(const std::string& mode_str,
                              std::uintptr_t d_w,
                              std::uintptr_t d_c,
                              std::uintptr_t d_s2,
                              std::uintptr_t d_r) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    compute_r_gpu(params,
                  reinterpret_cast<const uint32_t*>(d_w),
                  reinterpret_cast<const int8_t*>(d_c),
                  reinterpret_cast<const uint32_t*>(d_s2),
                  reinterpret_cast<uint32_t*>(d_r), 0);
}

static void py_compute_ct0_gpu(const std::string& mode_str,
                                std::uintptr_t d_c,
                                std::uintptr_t d_t0,
                                std::uintptr_t d_ct0) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    compute_ct0_gpu(params,
                    reinterpret_cast<const int8_t*>(d_c),
                    reinterpret_cast<const uint32_t*>(d_t0),
                    reinterpret_cast<uint32_t*>(d_ct0), 0);
}

static void py_make_hint_gpu(const std::string& mode_str,
                              std::uintptr_t d_neg_ct0,
                              std::uintptr_t d_r_hint,
                              std::uintptr_t d_h,
                              std::uintptr_t d_hint_count) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    make_hint_gpu(params,
                  reinterpret_cast<const uint32_t*>(d_neg_ct0),
                  reinterpret_cast<const uint32_t*>(d_r_hint),
                  reinterpret_cast<uint8_t*>(d_h),
                  reinterpret_cast<uint32_t*>(d_hint_count), 0);
}

static void py_use_hint_gpu(const std::string& mode_str,
                             std::uintptr_t d_h,
                             std::uintptr_t d_w_prime,
                             std::uintptr_t d_w1_recon) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    use_hint_gpu(params,
                 reinterpret_cast<const uint8_t*>(d_h),
                 reinterpret_cast<const uint32_t*>(d_w_prime),
                 reinterpret_cast<uint32_t*>(d_w1_recon), 0);
}

static void py_compute_ct_gpu(const std::string& mode_str,
                               std::uintptr_t d_c,
                               std::uintptr_t d_t1,
                               std::uintptr_t d_ct) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    compute_ct_gpu(params,
                   reinterpret_cast<const int8_t*>(d_c),
                   reinterpret_cast<const uint32_t*>(d_t1),
                   reinterpret_cast<uint32_t*>(d_ct), 0);
}

static void py_compute_w_prime_gpu(const std::string& mode_str,
                                    std::uintptr_t d_Az,
                                    std::uintptr_t d_ct,
                                    std::uintptr_t d_w_prime) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    compute_w_prime_gpu(params,
                        reinterpret_cast<const uint32_t*>(d_Az),
                        reinterpret_cast<const uint32_t*>(d_ct),
                        reinterpret_cast<uint32_t*>(d_w_prime), 0);
}

// =============================================================================
// Batch Sign Functions (Card 19B)
// =============================================================================

static void py_sign_roundA_w1_batch_gpu(const std::string& mode_str,
                                         std::uintptr_t d_rhoprime,
                                         std::uintptr_t d_A_hat,
                                         std::uintptr_t d_active,
                                         std::uintptr_t d_attempts,
                                         std::uintptr_t d_y,
                                         std::uintptr_t d_w,
                                         std::uintptr_t d_w1,
                                         std::uintptr_t d_w0,
                                         std::uintptr_t d_work,
                                         int B) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    sign_roundA_w1_batch_gpu(
        params,
        reinterpret_cast<const uint8_t*>(d_rhoprime),
        reinterpret_cast<const uint32_t*>(d_A_hat),
        reinterpret_cast<const uint8_t*>(d_active),
        reinterpret_cast<const uint16_t*>(d_attempts),
        reinterpret_cast<uint32_t*>(d_y),
        reinterpret_cast<uint32_t*>(d_w),
        reinterpret_cast<uint32_t*>(d_w1),
        reinterpret_cast<int32_t*>(d_w0),
        reinterpret_cast<uint32_t*>(d_work),
        B, 0);
}

static void py_sign_roundB_zh_batch_gpu(const std::string& mode_str,
                                         std::uintptr_t d_s1_ntt,
                                         std::uintptr_t d_s2_ntt,
                                         std::uintptr_t d_t0_ntt,
                                         std::uintptr_t d_y,
                                         std::uintptr_t d_w,
                                         std::uintptr_t d_w0,
                                         std::uintptr_t d_c,
                                         std::uintptr_t d_active,
                                         std::uintptr_t d_attempts,
                                         std::uintptr_t d_z,
                                         std::uintptr_t d_h,
                                         std::uintptr_t d_pass,
                                         std::uintptr_t d_hint_count,
                                         std::uintptr_t d_active_count,
                                         uint16_t max_attempts,
                                         std::uintptr_t d_work,
                                         int B) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    sign_roundB_zh_batch_gpu(
        params,
        reinterpret_cast<const uint32_t*>(d_s1_ntt),
        reinterpret_cast<const uint32_t*>(d_s2_ntt),
        reinterpret_cast<const uint32_t*>(d_t0_ntt),
        reinterpret_cast<const uint32_t*>(d_y),
        reinterpret_cast<const uint32_t*>(d_w),
        reinterpret_cast<const int32_t*>(d_w0),
        reinterpret_cast<const int8_t*>(d_c),
        reinterpret_cast<uint8_t*>(d_active),
        reinterpret_cast<uint16_t*>(d_attempts),
        reinterpret_cast<uint32_t*>(d_z),
        reinterpret_cast<uint8_t*>(d_h),
        reinterpret_cast<uint8_t*>(d_pass),
        reinterpret_cast<uint32_t*>(d_hint_count),
        reinterpret_cast<uint32_t*>(d_active_count),
        max_attempts,
        reinterpret_cast<uint32_t*>(d_work),
        B, 0);
}

// Helper: memcpy uint16 host->device
static void py_cuda_memcpy_htod_uint16(std::uintptr_t dst,
                                        py::array_t<uint16_t> src,
                                        std::size_t count) {
    auto buf = src.request();
    if (buf.size < static_cast<py::ssize_t>(count)) {
        throw std::runtime_error("Source array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(dst),
                                  buf.ptr,
                                  count * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyHtoD uint16 failed: ") +
                                 cudaGetErrorString(err));
    }
}

// Helper: memcpy uint16 device->host
static void py_cuda_memcpy_dtoh_uint16(py::array_t<uint16_t> dst,
                                        std::uintptr_t src,
                                        std::size_t count) {
    auto buf = dst.request();
    if (buf.size < static_cast<py::ssize_t>(count)) {
        throw std::runtime_error("Destination array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(buf.ptr,
                                  reinterpret_cast<void*>(src),
                                  count * sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyDtoH uint16 failed: ") +
                                 cudaGetErrorString(err));
    }
}

// Helper: memcpy int32 host->device
static void py_cuda_memcpy_htod_int32(std::uintptr_t dst,
                                       py::array_t<int32_t> src,
                                       std::size_t count) {
    auto buf = src.request();
    if (buf.size < static_cast<py::ssize_t>(count)) {
        throw std::runtime_error("Source array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(dst),
                                  buf.ptr,
                                  count * sizeof(int32_t),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyHtoD int32 failed: ") +
                                 cudaGetErrorString(err));
    }
}

// Helper: memcpy int32 device->host
static void py_cuda_memcpy_dtoh_int32(py::array_t<int32_t> dst,
                                       std::uintptr_t src,
                                       std::size_t count) {
    auto buf = dst.request();
    if (buf.size < static_cast<py::ssize_t>(count)) {
        throw std::runtime_error("Destination array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(buf.ptr,
                                  reinterpret_cast<void*>(src),
                                  count * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyDtoH int32 failed: ") +
                                 cudaGetErrorString(err));
    }
}

// =============================================================================
// Batched SHAKE Functions (Card 19C)
// =============================================================================

static void py_shake256_xof_batch_34b_gpu(std::uintptr_t d_in,
                                           std::uintptr_t d_out,
                                           int count,
                                           int out_bytes) {
    shake256_xof_batch_34b_gpu(
        reinterpret_cast<const uint8_t*>(d_in),
        reinterpret_cast<uint8_t*>(d_out),
        count, out_bytes, 0);
}

static std::size_t py_sample_y_workspace_size(int B, int l) {
    return sample_y_workspace_size(B, l);
}

static std::size_t py_bytes_per_y_poly() {
    return BYTES_PER_Y_POLY;
}

// =============================================================================
// Card 14.3: GPU Challenge Sampling + Unified Signing
// =============================================================================

// Helper: memcpy int8 device->host
static void py_cuda_memcpy_dtoh_int8(py::array_t<int8_t> dst,
                                      std::uintptr_t src,
                                      std::size_t count) {
    auto buf = dst.request();
    if (buf.size < static_cast<py::ssize_t>(count)) {
        throw std::runtime_error("Destination array too small for memcpy");
    }
    cudaError_t err = cudaMemcpy(buf.ptr,
                                  reinterpret_cast<void*>(src),
                                  count,
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyDtoH int8 failed: ") +
                                 cudaGetErrorString(err));
    }
}

// Workspace size for challenge sampling
static std::size_t py_challenge_workspace_size(const std::string& mode_str, int B) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);
    return challenge_workspace_size(params, B);
}

// Workspace size for full batch signing
static std::size_t py_sign_full_workspace_size(const std::string& mode_str, int B) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);
    return sign_full_workspace_size(params, B);
}

// GPU challenge sampling: H(mu || w1) -> c
static void py_sample_challenge_batch_gpu(const std::string& mode_str,
                                           std::uintptr_t d_mu,
                                           std::uintptr_t d_w1,
                                           std::uintptr_t d_c,
                                           std::uintptr_t d_active,
                                           std::uintptr_t d_work,
                                           int B) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    sample_challenge_batch_gpu(
        params,
        reinterpret_cast<const uint8_t*>(d_mu),
        reinterpret_cast<const uint32_t*>(d_w1),
        reinterpret_cast<int8_t*>(d_c),
        reinterpret_cast<const uint8_t*>(d_active),
        reinterpret_cast<uint8_t*>(d_work),
        B, 0);
}

// Full GPU signing with internal rejection loop (Card 14.3)
static int py_sign_full_batch_gpu(const std::string& mode_str,
                                   std::uintptr_t d_A_hat,
                                   std::uintptr_t d_rhoprime,
                                   std::uintptr_t d_s1,
                                   std::uintptr_t d_s2,
                                   std::uintptr_t d_t0,
                                   std::uintptr_t d_mu,
                                   std::uintptr_t d_z,
                                   std::uintptr_t d_h,
                                   std::uintptr_t d_c,
                                   std::uintptr_t d_attempts,
                                   std::uintptr_t d_converged,
                                   int B,
                                   uint16_t max_attempts) {
    Mode mode = parse_mode(mode_str);
    const Params& params = get_params(mode);

    return sign_full_batch_gpu(
        params,
        reinterpret_cast<const uint32_t*>(d_A_hat),
        reinterpret_cast<const uint8_t*>(d_rhoprime),
        reinterpret_cast<const uint32_t*>(d_s1),
        reinterpret_cast<const uint32_t*>(d_s2),
        reinterpret_cast<const uint32_t*>(d_t0),
        reinterpret_cast<const uint8_t*>(d_mu),
        reinterpret_cast<uint32_t*>(d_z),
        reinterpret_cast<uint8_t*>(d_h),
        reinterpret_cast<int8_t*>(d_c),
        reinterpret_cast<uint16_t*>(d_attempts),
        reinterpret_cast<uint8_t*>(d_converged),
        B,
        max_attempts,
        0);
}

// =============================================================================
// Module Definition
// =============================================================================

PYBIND11_MODULE(dilithium_cuda, m) {
    m.doc() = "Dilithium CUDA extension for GPU-accelerated post-quantum signatures";

    // CUDA memory management
    m.def("cuda_malloc", &py_cuda_malloc, "Allocate GPU memory");
    m.def("cuda_free", &py_cuda_free, "Free GPU memory");
    m.def("cuda_memcpy_htod", &py_cuda_memcpy_htod, "Copy host to device");
    m.def("cuda_memcpy_dtoh", &py_cuda_memcpy_dtoh, "Copy device to host");

    // Parameters
    m.def("dilithium_get_params", &py_dilithium_get_params,
          "Get Dilithium parameters for a mode");

    // NTT functions
    m.def("ntt_init", &py_ntt_init, "Initialize NTT twiddle factors");
    m.def("ntt_is_initialized", &py_ntt_is_initialized, "Check if NTT is initialized");
    m.def("ntt_forward_batch_gpu", &py_ntt_forward_batch_gpu,
          "Forward NTT on GPU (in-place)");
    m.def("ntt_inverse_batch_gpu", &py_ntt_inverse_batch_gpu,
          "Inverse NTT on GPU (in-place)");
    m.def("pointwise_multiply_batch_gpu", &py_pointwise_multiply_batch_gpu,
          "Pointwise multiply on GPU");
    m.def("pointwise_add_batch_gpu", &py_pointwise_add_batch_gpu,
          "Pointwise add on GPU");

    // SHAKE XOF
    m.def("shake128_xof_gpu", &py_shake128_xof_gpu,
          "SHAKE-128 XOF on GPU");
    m.def("shake256_xof_gpu", &py_shake256_xof_gpu,
          "SHAKE-256 XOF on GPU");

    // ExpandA
    m.def("expand_A_ntt_gpu", &py_expand_A_ntt_gpu,
          "Expand A matrix in NTT domain on GPU");

    // Secret sampling
    m.def("sample_s1_gpu", &py_sample_s1_gpu, "Sample s1 vector on GPU");
    m.def("sample_s2_gpu", &py_sample_s2_gpu, "Sample s2 vector on GPU");
    m.def("sample_poly_eta_gpu", &py_sample_poly_eta_gpu,
          "Sample poly_eta on GPU");

    // Keygen core
    m.def("keygen_core_gpu", &py_keygen_core_gpu,
          "Full keygen core: expand A, sample s1/s2, compute t");
    m.def("compute_t_gpu", &py_compute_t_gpu,
          "Compute t = A*s1 + s2 from pre-expanded inputs");

    // Decomposition
    m.def("power2round_vec_gpu", &py_power2round_vec_gpu,
          "Power2Round decomposition: t -> (t1, t0)");

    // Card 14.2: Full keygen with GPU power2round and packing
    m.def("keygen_full_gpu_v2", &py_keygen_full_gpu_v2,
          "Full keygen with GPU power2round and packing (Card 14.2)");
    m.def("keygen_full_batch_gpu", &py_keygen_full_batch_gpu,
          "Batched keygen for B > 1 keypairs (Card 14.2)");
    m.def("cuda_memcpy_dtoh_bytes", &py_cuda_memcpy_dtoh_bytes_ret,
          "Copy bytes from device to host and return as bytes");

    // W1 functions (Card 16.1)
    m.def("compute_w_gpu", &py_compute_w_gpu,
          "Compute w = A*y via NTT-domain multiplication");
    m.def("decompose_w_gpu", &py_decompose_w_gpu,
          "HighBits/LowBits decomposition: w -> (w1, w0)");

    // Sign functions (Card 17)
    m.def("cuda_memcpy_htod_bytes", &py_cuda_memcpy_htod_bytes,
          "Copy uint8 bytes host to device");
    m.def("cuda_memcpy_dtoh_bytes", &py_cuda_memcpy_dtoh_bytes,
          "Copy uint8 bytes device to host");
    m.def("cuda_memcpy_htod_int8", &py_cuda_memcpy_htod_int8,
          "Copy int8 array host to device");
    m.def("sample_y_gamma1_gpu", &py_sample_y_gamma1_gpu,
          "Sample y from gamma1 distribution on GPU");
    m.def("compute_z_gpu", &py_compute_z_gpu,
          "Compute z = y + c*s1 on GPU");
    m.def("compute_r_gpu", &py_compute_r_gpu,
          "Compute r = w - c*s2 on GPU");
    m.def("compute_ct0_gpu", &py_compute_ct0_gpu,
          "Compute c*t0 on GPU");
    m.def("make_hint_gpu", &py_make_hint_gpu,
          "MakeHint on GPU");
    m.def("use_hint_gpu", &py_use_hint_gpu,
          "UseHint on GPU");
    m.def("compute_ct_gpu", &py_compute_ct_gpu,
          "Compute c*t (t = t1 * 2^d) on GPU");
    m.def("compute_w_prime_gpu", &py_compute_w_prime_gpu,
          "Compute w' = Az - ct on GPU");

    // Batch sign functions (Card 19B)
    m.def("cuda_memcpy_htod_uint16", &py_cuda_memcpy_htod_uint16,
          "Copy uint16 array host to device");
    m.def("cuda_memcpy_dtoh_uint16", &py_cuda_memcpy_dtoh_uint16,
          "Copy uint16 array device to host");
    m.def("cuda_memcpy_htod_int32", &py_cuda_memcpy_htod_int32,
          "Copy int32 array host to device");
    m.def("cuda_memcpy_dtoh_int32", &py_cuda_memcpy_dtoh_int32,
          "Copy int32 array device to host");
    m.def("sign_roundA_w1_batch_gpu", &py_sign_roundA_w1_batch_gpu,
          "Round A: sample y, compute w = A*y, decompose to w1/w0 (batch)");
    m.def("sign_roundB_zh_batch_gpu", &py_sign_roundB_zh_batch_gpu,
          "Round B: consume c, compute z/r, check norms, make hints, update mask (batch)");

    // Card 19C: Batched SHAKE functions
    m.def("shake256_xof_batch_34b_gpu", &py_shake256_xof_batch_34b_gpu,
          "Batched SHAKE-256 XOF for 34-byte inputs (Card 19C)");
    m.def("sample_y_workspace_size", &py_sample_y_workspace_size,
          "Get workspace size for batched y sampling");
    m.def("bytes_per_y_poly", &py_bytes_per_y_poly,
          "Get bytes per y polynomial (768)");

    // Card 14.3: GPU challenge sampling + unified signing
    m.def("cuda_memcpy_dtoh_int8", &py_cuda_memcpy_dtoh_int8,
          "Copy int8 array device to host");
    m.def("challenge_workspace_size", &py_challenge_workspace_size,
          "Get workspace size for GPU challenge sampling");
    m.def("sign_full_workspace_size", &py_sign_full_workspace_size,
          "Get workspace size for full batch signing");
    m.def("sample_challenge_batch_gpu", &py_sample_challenge_batch_gpu,
          "Sample challenge polynomial c = H(mu || w1) on GPU (Card 14.3)");
    m.def("sign_full_batch_gpu", &py_sign_full_batch_gpu,
          "Full GPU signing with internal rejection loop (Card 14.3)");
}
