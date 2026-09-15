// SPDX-License-Identifier: Apache-2.0
// PyTorch binding for fast HKDF-SHA256 kernel

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_UINT8(x) TORCH_CHECK((x).scalar_type()==at::kByte, #x " must be uint8")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type()==at::kInt,  #x " must be int32")

extern "C" __global__ void hkdf_extract_fast_256_kernel(
    const uint8_t*, const int32_t*, const int32_t*,
    const uint8_t*, const uint8_t*, const int32_t,
    uint8_t*, int32_t);

torch::Tensor hkdf_extract_fast_256(
    torch::Tensor inner_blocks,  // flat bytes
    torch::Tensor inner_off,     // int32 per msg (byte offsets)
    torch::Tensor inner_nblocks, // int32 per msg
    torch::Tensor opad_block)    // [n, 64] uint8
{
    CHECK_CUDA(inner_blocks); CHECK_CONTIG(inner_blocks); CHECK_UINT8(inner_blocks);
    CHECK_CUDA(inner_off);    CHECK_CONTIG(inner_off);    CHECK_INT32(inner_off);
    CHECK_CUDA(inner_nblocks);CHECK_CONTIG(inner_nblocks);CHECK_INT32(inner_nblocks);
    CHECK_CUDA(opad_block);   CHECK_CONTIG(opad_block);   CHECK_UINT8(opad_block);

    const int32_t n = static_cast<int32_t>(inner_off.numel());
    TORCH_CHECK(opad_block.size(0) == n && opad_block.size(1) == 64, "opad must be [n,64]");

    // Allocate output
    auto prk_out = torch::empty({n, 32}, torch::TensorOptions().dtype(torch::kUInt8).device(inner_blocks.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    dim3 grid(n);
    dim3 block(32);

    // Get pointers
    const uint8_t* inner_blocks_ptr = inner_blocks.data_ptr<uint8_t>();
    const int32_t* inner_off_ptr = inner_off.data_ptr<int32_t>();
    const int32_t* inner_nblocks_ptr = inner_nblocks.data_ptr<int32_t>();
    const uint8_t* opad_block_ptr = opad_block.data_ptr<uint8_t>();
    const uint8_t* opad2_ptr = nullptr;  // unused for SHA-256
    int32_t opad_has_2blocks = 0;
    uint8_t* prk_out_ptr = prk_out.data_ptr<uint8_t>();

    void* args[] = {
        (void*)&inner_blocks_ptr,
        (void*)&inner_off_ptr,
        (void*)&inner_nblocks_ptr,
        (void*)&opad_block_ptr,
        (void*)&opad2_ptr,
        (void*)&opad_has_2blocks,
        (void*)&prk_out_ptr,
        (void*)&n
    };

    auto st = cudaLaunchKernel((void*)hkdf_extract_fast_256_kernel, grid, block, args, 0, stream);
    TORCH_CHECK(st == cudaSuccess, "CUDA launch failed: ", cudaGetErrorString(st));

    // Sync
    auto sync_st = cudaStreamSynchronize(stream);
    TORCH_CHECK(sync_st == cudaSuccess, "Kernel execution failed: ", cudaGetErrorString(sync_st));

    return prk_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hkdf_extract_fast_256", &hkdf_extract_fast_256, "Fast HKDF Extract SHA-256");
}
