// GPU epoch Merkle aggregation (smoke-merkle-v0) — production host entry.
// Domain-tagged: leaf=SHA256(0x00||leaf[32]), node=SHA256(0x01||L[32]||R[32]),
// odd count -> last node promoted. log(N) fully-parallel levels on-device.
// Byte-for-byte parity with trust merkle_root_v0 (oracle-gated at the ABI).
#include <cstdint>
#include <cuda_runtime.h>
#include "sha256_device.cuh"

namespace smoke {
namespace merkle {

using namespace smoke::hash;  // SHA256_IV, sha256_compress_block, sha256_hash_short

// 2-block SHA-256 over 0x01 || left[32] || right[32] (65 bytes = 520 bits).
__device__ __forceinline__ void sha256_node(const uint8_t* left, const uint8_t* right, uint8_t out[32]) {
    uint32_t H[8];
#pragma unroll
    for (int i = 0; i < 8; i++) H[i] = SHA256_IV[i];
    uint8_t b0[64];
    b0[0] = 0x01;
#pragma unroll
    for (int i = 0; i < 32; i++) b0[1 + i] = left[i];
#pragma unroll
    for (int i = 0; i < 31; i++) b0[33 + i] = right[i];   // right[0..30]
    sha256_compress_block(H, b0);
    uint8_t b1[64];
#pragma unroll
    for (int i = 0; i < 64; i++) b1[i] = 0;
    b1[0] = right[31];
    b1[1] = 0x80;
    b1[62] = 0x02; b1[63] = 0x08;
    sha256_compress_block(H, b1);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4+0] = (uint8_t)(H[i] >> 24);
        out[i*4+1] = (uint8_t)(H[i] >> 16);
        out[i*4+2] = (uint8_t)(H[i] >> 8);
        out[i*4+3] = (uint8_t)(H[i]);
    }
}

__global__ void k_leaves(const uint8_t* leaves, uint8_t* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t buf[33];
    buf[0] = 0x00;
#pragma unroll
    for (int j = 0; j < 32; j++) buf[1 + j] = leaves[i*32 + j];
    sha256_hash_short(buf, 33, out + i*32);
}

__global__ void k_level(const uint8_t* in, uint8_t* out, uint32_t pairs) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    sha256_node(in + (2*i)*32, in + (2*i+1)*32, out + i*32);
}

// Host entry: computes the smoke-merkle-v0 root over n 32-byte leaves.
// Returns 0 on success, a cudaError_t (>0) on failure. Caller must ensure n>=1
// (an empty root is meaningless / fail-closed at the Python layer).
extern "C" int smoke_merkle_root_v0_gpu(const uint8_t* h_leaves, uint32_t n, uint8_t out_root[32]) {
    if (n == 0 || h_leaves == nullptr || out_root == nullptr) return (int)cudaErrorInvalidValue;
    uint8_t *d_a = nullptr, *d_b = nullptr;
    cudaError_t err;
    const size_t bytes = (size_t)n * 32;

    // Run on a PRIVATE non-blocking stream rather than the legacy default (NULL)
    // stream. This is hygiene — the NULL stream implicitly barriers against every
    // blocking stream in the context, which is worth avoiding on principle.
    //
    // *** IT DOES NOT FIX THE ENGINE-RESIDENT HANG. *** This block was originally
    // added on the theory that NULL-stream serialization behind the persistent
    // engine kernel caused `merkle_root_v0_gpu` to hang once the engine was
    // initialized. That theory was GPU-DISPROVEN on an L40S (2026-08-17, D-107):
    // with this private stream compiled in, the guarded repro
    // `tests/repro_d107_merkle_engine_resident.py` still reports FAIL: HANG.
    // Adding the stream-ordered allocator (cudaMallocAsync/cudaFreeAsync) on top
    // was also disproven. Do not re-attempt a stream/allocator fix.
    //
    // The real cause is OCCUPANCY STARVATION, documented since July in
    // `trust/smoke_trust/attestation/test_attest_gpu_endtoend.py`: the persistent
    // engine kernel holds ALL SMs and never exits, so this one-shot Merkle kernel
    // is never scheduled. Bisect: standalone = 0.225 s OK; engine-resident = hang.
    // Resolutions are architectural (D-107): run Merkle with no engine resident,
    // or make Merkle an engine opcode, or leave SMs free (fewer CTAs / MPS slice).
    cudaStream_t stream = nullptr;
    if ((err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)) != cudaSuccess)
        return (int)err;

    if ((err = cudaMalloc(&d_a, bytes)) != cudaSuccess) { cudaStreamDestroy(stream); return (int)err; }
    if ((err = cudaMalloc(&d_b, bytes)) != cudaSuccess) { cudaFree(d_a); cudaStreamDestroy(stream); return (int)err; }
    if ((err = cudaMemcpyAsync(d_a, h_leaves, bytes, cudaMemcpyHostToDevice, stream)) != cudaSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaStreamDestroy(stream); return (int)err;
    }
    const int TPB = 256;
    uint32_t count = n;
    k_leaves<<<(count + TPB - 1)/TPB, TPB, 0, stream>>>(d_a, d_b, count);
    uint8_t *cur = d_b, *nxt = d_a;
    while (count > 1) {
        uint32_t pairs = count / 2;
        k_level<<<(pairs + TPB - 1)/TPB, TPB, 0, stream>>>(cur, nxt, pairs);
        if (count & 1) {
            cudaMemcpyAsync(nxt + (size_t)pairs*32, cur + (size_t)(count-1)*32, 32, cudaMemcpyDeviceToDevice, stream);
            count = pairs + 1;
        } else {
            count = pairs;
        }
        uint8_t* tmp = cur; cur = nxt; nxt = tmp;
    }
    err = cudaStreamSynchronize(stream);
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaMemcpyAsync(out_root, cur, 32, cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    cudaFree(d_a); cudaFree(d_b);
    cudaStreamDestroy(stream);
    return (int)err;
}

}  // namespace merkle
}  // namespace smoke
