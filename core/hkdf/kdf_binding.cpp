// Smoke KDF — PyTorch binding for HKDF kernels with CUDA events
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type()==at::kInt, #x " must be int32")
#define CHECK_UINT8(x) TORCH_CHECK(x.scalar_type()==at::kByte, #x " must be uint8")

// Kernel forward declarations (CUDA __global__ functions)
extern "C" void hkdf_extract_kernel(
    const uint8_t*, const uint8_t*, const int32_t*, const int32_t*,
    const int32_t*, const int32_t*, uint8_t*, int32_t, int32_t, int32_t);
extern "C" void hkdf_expand_kernel(
    const uint8_t*, const uint8_t*, const int32_t*, const int32_t*,
    uint8_t*, const int32_t*, const int32_t*, int32_t, int32_t, int32_t);
extern "C" void hkdf_extract_expand_kernel(
    const uint8_t*, const uint8_t*, const uint8_t*, const int32_t*, const int32_t*,
    const int32_t*, const int32_t*, const int32_t*, const int32_t*, uint8_t*,
    const int32_t*, const int32_t*, int32_t, int32_t);

static void launch_grid(int n_msgs, void* func, void** args, cudaStream_t stream, size_t shmem_bytes) {
    dim3 grid(n_msgs);
    dim3 block(32); // one warp
    cudaError_t st = cudaLaunchKernel(func, grid, block, args, shmem_bytes, stream);
    TORCH_CHECK(st == cudaSuccess, "CUDA launch failed: ", cudaGetErrorString(st));
}

torch::Tensor hkdf_extract(
    torch::Tensor salt, torch::Tensor salt_off, torch::Tensor salt_len,
    torch::Tensor ikm,  torch::Tensor ikm_off,  torch::Tensor ikm_len,
    int64_t hash_len)
{
    CHECK_CUDA(salt); CHECK_CUDA(ikm);
    CHECK_CUDA(salt_off); CHECK_CUDA(salt_len);
    CHECK_CUDA(ikm_off);  CHECK_CUDA(ikm_len);
    CHECK_CONTIG(salt); CHECK_CONTIG(ikm);
    CHECK_CONTIG(salt_off); CHECK_CONTIG(salt_len);
    CHECK_CONTIG(ikm_off);  CHECK_CONTIG(ikm_len);
    CHECK_UINT8(salt); CHECK_UINT8(ikm);
    CHECK_INT32(salt_off); CHECK_INT32(salt_len);
    CHECK_INT32(ikm_off);  CHECK_INT32(ikm_len);
    TORCH_CHECK(hash_len==32 || hash_len==64, "hash_len must be 32 or 64");

    const int n = salt_off.numel();
    auto prk = torch::empty({n, hash_len}, salt.options());
    auto prk_stride = prk.stride(0) * prk.element_size();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    void* args[] = {
        (void*)salt.data_ptr<uint8_t>(),
        (void*)ikm.data_ptr<uint8_t>(),
        (void*)salt_off.data_ptr<int32_t>(),
        (void*)salt_len.data_ptr<int32_t>(),
        (void*)ikm_off.data_ptr<int32_t>(),
        (void*)ikm_len.data_ptr<int32_t>(),
        (void*)prk.data_ptr<uint8_t>(),
        (void*)&n,
        (void*)&hash_len,
        (void*)&prk_stride
    };

    // timing (optional)
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, stream);

    const size_t shmem = 448; // 64+64+256+64 bytes for temp buffers
    launch_grid(n, (void*)&hkdf_extract_kernel, args, stream, shmem);

    cudaEventRecord(ev_stop, stream);
    cudaEventSynchronize(ev_stop);
    float ms=0; cudaEventElapsedTime(&ms, ev_start, ev_stop);
    // printf("hkdf_extract: %d msgs, %.3f ms\n", n, ms);

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    return prk;
}

torch::Tensor hkdf_expand(
    torch::Tensor prk,
    torch::Tensor info, torch::Tensor info_off, torch::Tensor info_len,
    torch::Tensor okm, torch::Tensor okm_off, torch::Tensor L_out,
    int64_t hash_len)
{
    CHECK_CUDA(prk); CHECK_CUDA(info); CHECK_CUDA(okm);
    CHECK_CUDA(info_off); CHECK_CUDA(info_len); CHECK_CUDA(okm_off); CHECK_CUDA(L_out);
    CHECK_CONTIG(prk); CHECK_CONTIG(info); CHECK_CONTIG(okm);
    CHECK_UINT8(prk); CHECK_UINT8(info); CHECK_UINT8(okm);
    CHECK_INT32(info_off); CHECK_INT32(info_len); CHECK_INT32(okm_off); CHECK_INT32(L_out);
    TORCH_CHECK(hash_len==32 || hash_len==64, "hash_len must be 32 or 64");

    const int n = info_off.numel();
    const int prk_stride = prk.stride(0) * prk.element_size();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    void* args[] = {
        (void*)prk.data_ptr<uint8_t>(),
        (void*)info.data_ptr<uint8_t>(),
        (void*)info_off.data_ptr<int32_t>(),
        (void*)info_len.data_ptr<int32_t>(),
        (void*)okm.data_ptr<uint8_t>(),
        (void*)okm_off.data_ptr<int32_t>(),
        (void*)L_out.data_ptr<int32_t>(),
        (void*)&n,
        (void*)&hash_len,
        (void*)&prk_stride
    };

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, stream);

    const size_t shmem = 448; // 64+64+256+64 bytes for temp buffers
    launch_grid(n, (void*)&hkdf_expand_kernel, args, stream, shmem);

    cudaEventRecord(ev_stop, stream);
    cudaEventSynchronize(ev_stop);
    float ms=0; cudaEventElapsedTime(&ms, ev_start, ev_stop);
    // printf("hkdf_expand: %d msgs, %.3f ms\n", n, ms);

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    return okm;
}

torch::Tensor hkdf_extract_expand(
    torch::Tensor salt, torch::Tensor salt_off, torch::Tensor salt_len,
    torch::Tensor ikm,  torch::Tensor ikm_off,  torch::Tensor ikm_len,
    torch::Tensor info, torch::Tensor info_off, torch::Tensor info_len,
    torch::Tensor okm,  torch::Tensor okm_off,  torch::Tensor L_out,
    int64_t hash_len)
{
    CHECK_CUDA(salt); CHECK_CUDA(ikm); CHECK_CUDA(info); CHECK_CUDA(okm);
    CHECK_CUDA(salt_off); CHECK_CUDA(salt_len);
    CHECK_CUDA(ikm_off);  CHECK_CUDA(ikm_len);
    CHECK_CUDA(info_off); CHECK_CUDA(info_len);
    CHECK_CUDA(okm_off);  CHECK_CUDA(L_out);
    CHECK_CONTIG(salt); CHECK_CONTIG(ikm); CHECK_CONTIG(info); CHECK_CONTIG(okm);
    CHECK_UINT8(salt); CHECK_UINT8(ikm); CHECK_UINT8(info); CHECK_UINT8(okm);
    CHECK_INT32(salt_off); CHECK_INT32(salt_len);
    CHECK_INT32(ikm_off);  CHECK_INT32(ikm_len);
    CHECK_INT32(info_off); CHECK_INT32(info_len);
    CHECK_INT32(okm_off);  CHECK_INT32(L_out);
    TORCH_CHECK(hash_len==32 || hash_len==64, "hash_len must be 32 or 64");

    const int n = salt_off.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    void* args[] = {
        (void*)salt.data_ptr<uint8_t>(),
        (void*)ikm.data_ptr<uint8_t>(),
        (void*)info.data_ptr<uint8_t>(),
        (void*)salt_off.data_ptr<int32_t>(),
        (void*)salt_len.data_ptr<int32_t>(),
        (void*)ikm_off.data_ptr<int32_t>(),
        (void*)ikm_len.data_ptr<int32_t>(),
        (void*)info_off.data_ptr<int32_t>(),
        (void*)info_len.data_ptr<int32_t>(),
        (void*)okm.data_ptr<uint8_t>(),
        (void*)okm_off.data_ptr<int32_t>(),
        (void*)L_out.data_ptr<int32_t>(),
        (void*)&n,
        (void*)&hash_len
    };

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, stream);

    const size_t shmem = 448; // 64+64+256+64 bytes for temp buffers
    launch_grid(n, (void*)&hkdf_extract_expand_kernel, args, stream, shmem);

    cudaEventRecord(ev_stop, stream);
    cudaEventSynchronize(ev_stop);
    float ms=0; cudaEventElapsedTime(&ms, ev_start, ev_stop);
    // printf("hkdf_extract_expand: %d msgs, %.3f ms\n", n, ms);

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    return okm;
}

TORCH_LIBRARY(smoke_kdf, m) {
    m.def("hkdf_extract(Tensor salt, Tensor salt_off, Tensor salt_len, "
          "Tensor ikm, Tensor ikm_off, Tensor ikm_len, int hash_len) -> Tensor");
    m.def("hkdf_expand(Tensor prk, Tensor info, Tensor info_off, Tensor info_len, "
          "Tensor okm, Tensor okm_off, Tensor L_out, int hash_len) -> Tensor");
    m.def("hkdf_extract_expand(Tensor salt, Tensor salt_off, Tensor salt_len, "
          "Tensor ikm, Tensor ikm_off, Tensor ikm_len, Tensor info, Tensor info_off, Tensor info_len, "
          "Tensor okm, Tensor okm_off, Tensor L_out, int hash_len) -> Tensor");
}

TORCH_LIBRARY_IMPL(smoke_kdf, CUDA, m) {
    m.impl("hkdf_extract", hkdf_extract);
    m.impl("hkdf_expand",  hkdf_expand);
    m.impl("hkdf_extract_expand", hkdf_extract_expand);
}

// Python module initialization
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // The TORCH_LIBRARY macros above register the operators
    // This just creates the Python module
}
