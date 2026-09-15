// SPDX-License-Identifier: Apache-2.0
// SHA-256 warp-cooperative compress block
// V1: Warp cooperatively builds W[0..63]; lane0 executes 64 rounds.
// V2 scaffold (commented) included for full cooperative rounds.
// Reg budget target: <= 40 regs/thread (V1 meets easily).

// Debug flag: set to 1 to enable verbose debug prints, 0 for production
#define DEBUG_SHA256_WARPCOOP 0

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#if DEBUG_SHA256_WARPCOOP
#include <cstdio>
#endif

#ifndef SMOKE_WARP
#define SMOKE_WARP 32
#endif

// ---- low-level helpers -------------------------------------------------

// Pure C rotate (no PTX inline asm) - for correctness validation on Windows/MSVC
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int r){
    return (x >> r) | (x << (32 - r));
}

__device__ __forceinline__ uint32_t Ch(uint32_t x,uint32_t y,uint32_t z){ return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t Maj(uint32_t x,uint32_t y,uint32_t z){ return (x & y) ^ (x & z) ^ (y & z); }

__device__ __forceinline__ uint32_t SIG0(uint32_t x){ return rotr32(x,2) ^ rotr32(x,13) ^ rotr32(x,22); }
__device__ __forceinline__ uint32_t SIG1(uint32_t x){ return rotr32(x,6) ^ rotr32(x,11) ^ rotr32(x,25); }
__device__ __forceinline__ uint32_t sig0(uint32_t x){ return rotr32(x,7) ^ rotr32(x,18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t sig1(uint32_t x){ return rotr32(x,17)^ rotr32(x,19)^ (x >> 10); }

__device__ __forceinline__ uint32_t lane_id(){ return threadIdx.x & (SMOKE_WARP - 1); }

// V3 helpers: subgroup (8-lane groups within warp)
__device__ __forceinline__ int subgroup_id(){ return (lane_id() >> 3); }   // 0..3
__device__ __forceinline__ int lid8(){ return (lane_id() & 7); }           // 0..7 within subgroup
__device__ __forceinline__ unsigned subgroup_mask(int sg){ return (0xFFu << (sg * 8)); }

// ---- K constants (kept in const memory) --------------------------------

__constant__ uint32_t K256[64] = {
  0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
  0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
  0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
  0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
  0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
  0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
  0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
  0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

// ---- public IV/init & digest store (reuse your existing ones if present) --

extern "C" __device__ void sha256_init_state(uint32_t H[8]){
    H[0]=0x6a09e667u; H[1]=0xbb67ae85u; H[2]=0x3c6ef372u; H[3]=0xa54ff53au;
    H[4]=0x510e527fu; H[5]=0x9b05688cu; H[6]=0x1f83d9abu; H[7]=0x5be0cd19u;
}

extern "C" __device__ void sha256_store_digest(const uint32_t H[8], uint8_t out[32]){
    // big-endian store
    #pragma unroll
    for (int i=0;i<8;i++){
        uint32_t v = H[i];
        out[i*4+0] = (uint8_t)(v >> 24);
        out[i*4+1] = (uint8_t)(v >> 16);
        out[i*4+2] = (uint8_t)(v >>  8);
        out[i*4+3] = (uint8_t)(v >>  0);
    }
}

// ---- load big-endian 32-bit from block ---------------------------------

__device__ __forceinline__ uint32_t be32_load(const uint8_t* p){
    // fully coalesced when lanes read consecutive words
    return (uint32_t(p[0])<<24) | (uint32_t(p[1])<<16) | (uint32_t(p[2])<<8) | uint32_t(p[3]);
}

// ---- Scalar reference rounds (for debugging) -------

__device__ void sha256_rounds_scalar(uint32_t Hs[8], const uint32_t* W){
    uint32_t a=Hs[0],b=Hs[1],c=Hs[2],d=Hs[3],e=Hs[4],f=Hs[5],g=Hs[6],h=Hs[7];
    #pragma unroll 8
    for(int t=0;t<64;++t){
        uint32_t T1 = h + SIG1(e) + Ch(e,f,g) + K256[t] + W[t];
        uint32_t T2 = SIG0(a) + Maj(a,b,c);
        h=g; g=f; f=e; e=d + T1; d=c; c=b; b=a; a=T1 + T2;
    }
    Hs[0]=a; Hs[1]=b; Hs[2]=c; Hs[3]=d; Hs[4]=e; Hs[5]=f; Hs[6]=g; Hs[7]=h;
}

// ---- V2: warp-cooperative rounds (8-lane) - operates on shared Hs[] -------

// NOTE: Hs[] MUST be warp-shared memory. All 32 lanes participate.
// Lanes 0..7 perform the round updates; lanes 8..31 idle.
extern "C" __device__
void sha256_rounds_warp8(uint32_t* __restrict__ Hs, const uint32_t* __restrict__ Wpub){
    const unsigned WM = 0xFFFFFFFFu;
    const int lid = threadIdx.x & 31;

    // 64 rounds
    for (int t=0; t<64; ++t){
        // All 32 lanes read current state snapshot
        uint32_t a = Hs[0], b = Hs[1], c = Hs[2], d = Hs[3];
        uint32_t e = Hs[4], f = Hs[5], g = Hs[6], h = Hs[7];

        // Compute T1 and T2 at lane 0
        uint32_t T1 = 0, T2 = 0;
        if (lid == 0){
            T1 = h + SIG1(e) + Ch(e,f,g) + K256[t] + Wpub[t];
            T2 = SIG0(a) + Maj(a,b,c);
        }

        // Broadcast T1 and T2 to all lanes
        T1 = __shfl_sync(WM, T1, 0);
        T2 = __shfl_sync(WM, T2, 0);

        // Write updated state: (a,b,c,d,e,f,g,h) -> (a',a,b,c,e',e,f,g)
        // Each lane updates its corresponding index in shared Hs[]
        if (lid == 0){
            Hs[0] = T1 + T2;           // a' = T1 + T2
        } else if (lid == 1){
            Hs[1] = a;                 // b' = a
        } else if (lid == 2){
            Hs[2] = b;                 // c' = b
        } else if (lid == 3){
            Hs[3] = c;                 // d' = c
        } else if (lid == 4){
            Hs[4] = d + T1;            // e' = d + T1
        } else if (lid == 5){
            Hs[5] = e;                 // f' = e
        } else if (lid == 6){
            Hs[6] = f;                 // g' = f
        } else if (lid == 7){
            Hs[7] = g;                 // h' = g
        }
        // Lanes 8..31 idle (no writes)

        __syncwarp(WM);  // Ensure all writes complete before next iteration
    }
}

// ---- V1: cooperative W schedule + lane0 rounds (DETERMINISTIC) ----------

extern "C" __device__
void sha256_compress_block_warpcoop(uint32_t* H_global, const uint8_t* block){
    const unsigned M = 0xFFFFFFFFu;
    const int lid = lane_id();

    // Safety tripwire: ensure ALL 32 lanes are active (no divergence)
    // This catches bugs where some lanes exited early before calling this function
    unsigned int active_mask = __activemask();
#if DEBUG_SHA256_WARPCOOP
    if (lid == 0 && active_mask != 0xFFFFFFFFu) {
        #if DEBUG_SHA256_WARPCOOP

        printf("[SHA256 TRIPWIRE] activemask=%08x (expected 0xFFFFFFFF) - DIVERGED!\n", active_mask);

        #endif
    }
#endif
    if (active_mask != 0xFFFFFFFFu) {
        return;  // Abort if warp is diverged
    }

    // Shared memory for warp-cooperative state
    __shared__ uint32_t Hs[8];         // Working variables (32 bytes)
    __shared__ uint32_t H_init[8];     // Initial H for feed-forward (32 bytes)
    __shared__ uint32_t Wpub[64];      // W schedule (256 bytes)

    // Lane 0 copies H_global into shared arrays
    if (lid == 0){
        H_init[0] = H_global[0];  // Save initial for feed-forward
        H_init[1] = H_global[1];
        H_init[2] = H_global[2];
        H_init[3] = H_global[3];
        H_init[4] = H_global[4];
        H_init[5] = H_global[5];
        H_init[6] = H_global[6];
        H_init[7] = H_global[7];

        Hs[0] = H_global[0];      // Working variables
        Hs[1] = H_global[1];
        Hs[2] = H_global[2];
        Hs[3] = H_global[3];
        Hs[4] = H_global[4];
        Hs[5] = H_global[5];
        Hs[6] = H_global[6];
        Hs[7] = H_global[7];
    }
    __syncwarp(M);

    // Preload W[0..15] from block
    if (lid < 16){
        Wpub[lid] = be32_load(block + lid*4);
    }
    __syncwarp(M);

    // DEBUG: Print W[0..3] and W[14..15] for first message
    if (lid == 0) {
        #if DEBUG_SHA256_WARPCOOP

        printf("[W SCHEDULE] W[0]=%08x W[1]=%08x W[2]=%08x W[3]=%08x\n",
               Wpub[0], Wpub[1], Wpub[2], Wpub[3]);

        #endif
        #if DEBUG_SHA256_WARPCOOP

        printf("[W SCHEDULE] W[14]=%08x W[15]=%08x (expect: 00000000 00000018)\n",
               Wpub[14], Wpub[15]);

        #endif
        #if DEBUG_SHA256_WARPCOOP

        printf("[INITIAL STATE] H_init: %08x %08x %08x %08x %08x %08x %08x %08x\n",
               H_init[0], H_init[1], H_init[2], H_init[3],
               H_init[4], H_init[5], H_init[6], H_init[7]);

        #endif
    }

    // Build W[16..63] in three ordered tiles (V1 deterministic schedule)
    // Tile 1: W[16..31]
    for (int t = 16; t <= 31; ++t){
        if (lid == (t & 31)){
            const uint32_t wtm2  = Wpub[t-2];
            const uint32_t wtm7  = Wpub[t-7];
            const uint32_t wtm15 = Wpub[t-15];
            const uint32_t wtm16 = Wpub[t-16];
            Wpub[t] = sig1(wtm2) + wtm7 + sig0(wtm15) + wtm16;
        }
        __syncwarp(M);
    }

    // Tile 2: W[32..47]
    for (int t = 32; t <= 47; ++t){
        if (lid == (t & 31)){
            const uint32_t wtm2  = Wpub[t-2];
            const uint32_t wtm7  = Wpub[t-7];
            const uint32_t wtm15 = Wpub[t-15];
            const uint32_t wtm16 = Wpub[t-16];
            Wpub[t] = sig1(wtm2) + wtm7 + sig0(wtm15) + wtm16;
        }
        __syncwarp(M);
    }

    // Tile 3: W[48..63]
    for (int t = 48; t <= 63; ++t){
        if (lid == (t & 31)){
            const uint32_t wtm2  = Wpub[t-2];
            const uint32_t wtm7  = Wpub[t-7];
            const uint32_t wtm15 = Wpub[t-15];
            const uint32_t wtm16 = Wpub[t-16];
            Wpub[t] = sig1(wtm2) + wtm7 + sig0(wtm15) + wtm16;
        }
        __syncwarp(M);
    }

    // DEBUG: Run scalar reference on lane 0 for comparison
    uint32_t scalar_result[8];
    if (lid == 0) {
        for (int i = 0; i < 8; i++) scalar_result[i] = Hs[i];
        sha256_rounds_scalar(scalar_result, Wpub);
        #if DEBUG_SHA256_WARPCOOP

        printf("[SCALAR ROUNDS] After rounds (no feed-forward): %08x %08x %08x %08x %08x %08x %08x %08x\n",
               scalar_result[0], scalar_result[1], scalar_result[2], scalar_result[3],
               scalar_result[4], scalar_result[5], scalar_result[6], scalar_result[7]);

        #endif
        // Feed-forward for scalar
        for (int i = 0; i < 8; i++) scalar_result[i] += H_init[i];
        #if DEBUG_SHA256_WARPCOOP

        printf("[SCALAR FINAL] With feed-forward: %08x %08x %08x %08x %08x %08x %08x %08x\n",
               scalar_result[0], scalar_result[1], scalar_result[2], scalar_result[3],
               scalar_result[4], scalar_result[5], scalar_result[6], scalar_result[7]);

        #endif
    }
    __syncwarp(M);

    // V2: 8-lane cooperative rounds (operates on shared Hs[])
    sha256_rounds_warp8(Hs, Wpub);

    // DEBUG: Print V2 result for comparison
    if (lid == 0) {
        #if DEBUG_SHA256_WARPCOOP

        printf("[V2 ROUNDS] After rounds (no feed-forward): %08x %08x %08x %08x %08x %08x %08x %08x\n",
               Hs[0], Hs[1], Hs[2], Hs[3], Hs[4], Hs[5], Hs[6], Hs[7]);

        #endif
    }

    // Lane 0: feed-forward (add initial H to final working variables)
    __syncwarp(M);
    if (lid == 0){
        H_global[0] = H_init[0] + Hs[0];
        H_global[1] = H_init[1] + Hs[1];
        H_global[2] = H_init[2] + Hs[2];
        H_global[3] = H_init[3] + Hs[3];
        H_global[4] = H_init[4] + Hs[4];
        H_global[5] = H_init[5] + Hs[5];
        H_global[6] = H_init[6] + Hs[6];
        H_global[7] = H_init[7] + Hs[7];
        #if DEBUG_SHA256_WARPCOOP

        printf("[V2 FINAL] With feed-forward: %08x %08x %08x %08x %08x %08x %08x %08x\n",
               H_global[0], H_global[1], H_global[2], H_global[3],
               H_global[4], H_global[5], H_global[6], H_global[7]);

        #endif
    }
    __syncwarp(M);
}

// ---- Public wrapper with same symbol name used by your fast path --------
// Replace calls to sha256_compress_block(...) with this function name in fast kernels.

extern "C" __device__
void sha256_compress_block(uint32_t* H, const uint8_t* block){
    sha256_compress_block_warpcoop(H, block);
}

// ---- Batch kernel for PyTorch integration --------------------------------

extern "C" __global__ __launch_bounds__(32, 8)
void sha256_compress_batch_warpcoop(
    const int32_t* blocks,      // Input blocks (int32 words, big-endian)
    const int32_t* num_blocks,  // Number of blocks per message
    const int32_t* block_offsets, // Block offset per message
    int32_t* state_out,         // Output state (8 words per message)
    int batch_size)
{
    const unsigned M = 0xFFFFFFFFu;
    int msg_id = blockIdx.x;

    // DEBUG: Print thread info for first block
    if (msg_id == 0 && threadIdx.x < 4) {
        #if DEBUG_SHA256_WARPCOOP

        printf("[KERNEL START] blockIdx.x=%d threadIdx.x=%d blockDim.x=%d\n",
               blockIdx.x, threadIdx.x, blockDim.x);

        #endif
    }

    // Uniform early exit for whole warp (no divergence)
    if (msg_id >= batch_size) return;

    // Shared memory for block bytes (shared across warp)
    __shared__ uint8_t block_bytes[64];

    // Each block processes one message with one warp
    int lid = lane_id();

    // DEBUG: Capture activemask BEFORE any divergent branches
    unsigned int mask_at_start = __activemask();
    if (msg_id == 0 && lid == 0) {
        #if DEBUG_SHA256_WARPCOOP

        printf("[BATCH KERNEL] msg=%d activemask=%08x (all 32 lanes: %s)\n",
               msg_id, mask_at_start,
               (mask_at_start == 0xFFFFFFFFu) ? "YES" : "NO");

        #endif
    }

    // Get message info (msg_id < batch_size guaranteed by early exit)
    int num_blks = num_blocks[msg_id];
    int blk_offset = block_offsets[msg_id];

    // Initialize state
    uint32_t H[8];
    sha256_init_state(H);

    // Process all blocks for this message
    for (int b = 0; b < num_blks; b++){
        // Convert int32 words to bytes for this block
        const int32_t* block_words = blocks + (blk_offset + b) * 16;

        // Parallel load: each lane loads its word(s)
        if (lid < 16){
            uint32_t w = (uint32_t)block_words[lid];
            block_bytes[lid*4 + 0] = (w >> 24) & 0xFF;
            block_bytes[lid*4 + 1] = (w >> 16) & 0xFF;
            block_bytes[lid*4 + 2] = (w >> 8) & 0xFF;
            block_bytes[lid*4 + 3] = (w >> 0) & 0xFF;
        }
        __syncwarp(M);

        // DEBUG: Capture activemask BEFORE compress
        unsigned int mask_before = __activemask();
        if (msg_id == 0 && b == 0 && lid == 0) {
            #if DEBUG_SHA256_WARPCOOP

            printf("[BEFORE COMPRESS] msg=%d block=%d activemask=%08x\n",
                   msg_id, b, mask_before);

            #endif
        }

        // ALL 32 lanes call compress unconditionally (no divergence!)
        sha256_compress_block(H, block_bytes);

        // DEBUG: Verify after compress
        unsigned int mask_after = __activemask();
        if (msg_id == 0 && b == 0 && lid == 0) {
            #if DEBUG_SHA256_WARPCOOP

            printf("[AFTER COMPRESS] msg=%d block=%d activemask=%08x\n",
                   msg_id, b, mask_after);

            #endif
            #if DEBUG_SHA256_WARPCOOP

            printf("[AFTER COMPRESS] H[0]=%08x H[4]=%08x\n", H[0], H[4]);

            #endif
        }
    }

    // Write final state (only lane 0)
    if (lid == 0){
        int32_t* out = state_out + msg_id * 8;
        for (int i = 0; i < 8; i++){
            // Convert unsigned uint32 to signed int32 for PyTorch
            uint32_t w = H[i];
            if (w >= 0x80000000u){
                out[i] = (int32_t)(w - 0x100000000ULL);
            } else {
                out[i] = (int32_t)w;
            }
        }
    }

    // Final barrier before return
    __syncwarp(M);
}

/* =======================
   V3: REGISTER-ONLY ROUNDS (8-lane subgroup, 4× messages per warp)

   Key improvements over V2:
   - No SMEM traffic in 64-round loop (register-only)
   - No per-round __syncwarp() barriers
   - 8 shuffles per round (instead of 8 loads + 8 stores + 1 barrier)
   - 4× messages per warp (lanes 0-7, 8-15, 16-23, 24-31)
   - Expected: 3-6x speedup over Phase A/V2
======================= */

// V3: Register-only rounds for 8-lane subgroup
// Input: Hs[8] working state, Wpub[64] message schedule
// Output: Hs[8] updated in place
// No SMEM in loop, no barriers, only 8 shuffles per round
__device__ __forceinline__
void sha256_rounds_register_only_group8(uint32_t Hs[8], const uint32_t* __restrict__ Wpub,
                                        unsigned msk /* subgroup mask */, int sg /*0..3*/){
    const int l8 = lid8();     // 0..7 within subgroup

    // Each active lane holds its own state variable in a register
    uint32_t v = Hs[l8];       // lanes 0..7 == a..h respectively

    #pragma unroll
    for (int t = 0; t < 64; ++t){
        // Snapshot a..h via shuffles (register-only, warp-synchronous)
        uint32_t a = __shfl_sync(msk, v, (sg*8 + 0));   // source lane index in full warp
        uint32_t b = __shfl_sync(msk, v, (sg*8 + 1));
        uint32_t c = __shfl_sync(msk, v, (sg*8 + 2));
        uint32_t d = __shfl_sync(msk, v, (sg*8 + 3));
        uint32_t e = __shfl_sync(msk, v, (sg*8 + 4));
        uint32_t f = __shfl_sync(msk, v, (sg*8 + 5));
        uint32_t g = __shfl_sync(msk, v, (sg*8 + 6));
        uint32_t h = __shfl_sync(msk, v, (sg*8 + 7));

        // Compute locally: no broadcasts needed (redundant computation is cheap)
        uint32_t T1 = h + SIG1(e) + Ch(e,f,g) + K256[t] + Wpub[t];
        uint32_t T2 = SIG0(a) + Maj(a,b,c);

        // Rotate from the *snapshot* (not from updated values)
        uint32_t next;
        switch (l8){
            case 0: next = T1 + T2;   break;        // a'
            case 1: next = a;         break;        // b' = a
            case 2: next = b;         break;        // c' = b
            case 3: next = c;         break;        // d' = c
            case 4: next = d + T1;    break;        // e' = d + T1
            case 5: next = e;         break;        // f' = e
            case 6: next = f;         break;        // g' = f
            case 7: next = g;         break;        // h' = g
        }
        v = next;   // register update, no barrier needed
    }
    // Write back to Hs (one store per lane)
    Hs[l8] = v;
}

// V3: 4× messages per warp kernel (single-block per message)
// One warp processes 4 messages (subgroups 0-7, 8-15, 16-23, 24-31)
// blocks: [N, 64] flat uint8 array (one 64-byte block per message)
// H_in: [N, 8] initial state (SHA-256 IV or ongoing state)
// H_out: [N, 8] output state (feed-forward applied)
extern "C" __global__
void sha256_compress_warpcoop_v3_4x(const uint8_t* __restrict__ blocks,   // [N, 64]
                                    const uint32_t* __restrict__ H_in,    // [N, 8]
                                    uint32_t* __restrict__ H_out,         // [N, 8]
                                    int N)
{
    const int warp = blockIdx.x;        // one warp per up-to-4 msgs
    const int sg   = subgroup_id();     // 0..3
    const int l8   = lid8();            // 0..7
    const unsigned msk = subgroup_mask(sg);

    // Message index for this subgroup
    const int msg = warp*4 + sg;

    // Build per-subgroup pointers (valid or safe-dummy)
    const uint8_t*  block_ptr = (msg < N) ? (blocks + msg*64) : nullptr;
    const uint32_t* Hin_ptr   = (msg < N) ? (H_in   + msg*8)  : nullptr;
    uint32_t*       Hout_ptr  = (msg < N) ? (H_out  + msg*8)  : nullptr;

    // Stage H_init and working Hs in registers (as an 8-lane array)
    uint32_t H_init[8];
    uint32_t Hs[8];

    // Lane l8 loads its corresponding H_init word; then broadcast within subgroup
    uint32_t tmp = 0;
    if (l8 < 8 && msg < N){
        tmp = Hin_ptr[l8];
    }
    // Materialize H_init and Hs by subgroup broadcast from each owner lane
    #pragma unroll
    for (int i=0;i<8;++i){
        H_init[i] = __shfl_sync(msk, tmp, (sg*8 + i));
        Hs[i]     = H_init[i];
    }

    // Build Wpub[64] for this subgroup (shared across 8-lane subgroup)
    // Use shared memory with 4 subgroups × 64 words = 256 words = 1KB
    __shared__ uint32_t Wpub_shared[4][64];
    uint32_t* Wpub = Wpub_shared[sg];  // This subgroup's message schedule

    // Load W[0..15] (big-endian uint32 from 64-byte block)
    // Lanes process strided indices: l8, l8+8
    for (int i = l8; i < 16; i += 8){
        uint32_t w = 0;
        if (msg < N){
            const uint8_t* bp = block_ptr + (i*4);
            w = (uint32_t(bp[0])<<24) | (uint32_t(bp[1])<<16) |
                (uint32_t(bp[2])<< 8) | (uint32_t(bp[3])<< 0);
        }
        Wpub[i] = w;
    }
    __syncwarp(msk);  // Ensure all lanes see W[0..15] before expansion

    // Expand W[16..63] in tiles of 8 (need sync between tiles for dependencies)
    // Tile 1: W[16..23]
    if (true){
        int t = 16 + l8;
        uint32_t w15 = Wpub[t-15];
        uint32_t w2  = Wpub[t-2];
        Wpub[t] = sig1(w2) + Wpub[t-7] + sig0(w15) + Wpub[t-16];
    }
    __syncwarp(msk);  // Ensure all lanes see W[16..23] before computing W[24..31]

    // Tile 2: W[24..31]
    if (true){
        int t = 24 + l8;
        uint32_t w15 = Wpub[t-15];
        uint32_t w2  = Wpub[t-2];
        Wpub[t] = sig1(w2) + Wpub[t-7] + sig0(w15) + Wpub[t-16];
    }
    __syncwarp(msk);

    // Tile 3: W[32..39]
    if (true){
        int t = 32 + l8;
        uint32_t w15 = Wpub[t-15];
        uint32_t w2  = Wpub[t-2];
        Wpub[t] = sig1(w2) + Wpub[t-7] + sig0(w15) + Wpub[t-16];
    }
    __syncwarp(msk);

    // Tile 4: W[40..47]
    if (true){
        int t = 40 + l8;
        uint32_t w15 = Wpub[t-15];
        uint32_t w2  = Wpub[t-2];
        Wpub[t] = sig1(w2) + Wpub[t-7] + sig0(w15) + Wpub[t-16];
    }
    __syncwarp(msk);

    // Tile 5: W[48..55]
    if (true){
        int t = 48 + l8;
        uint32_t w15 = Wpub[t-15];
        uint32_t w2  = Wpub[t-2];
        Wpub[t] = sig1(w2) + Wpub[t-7] + sig0(w15) + Wpub[t-16];
    }
    __syncwarp(msk);

    // Tile 6: W[56..63]
    if (true){
        int t = 56 + l8;
        uint32_t w15 = Wpub[t-15];
        uint32_t w2  = Wpub[t-2];
        Wpub[t] = sig1(w2) + Wpub[t-7] + sig0(w15) + Wpub[t-16];
    }
    __syncwarp(msk);  // Final sync before rounds

    // ---- REGISTER-ONLY ROUNDS (8-lane subgroup) ----
    sha256_rounds_register_only_group8(Hs, Wpub, msk, sg);

    // Feed-forward and store (each lane stores its word)
    if (msg < N && l8 < 8){
        uint32_t outv = H_init[l8] + Hs[l8];
        Hout_ptr[l8] = outv;
    }
}

/* =======================
   V2 (FULL WARP COOP ROUNDS) – Scaffold (optional, enable after V1 passes)

Idea:
- map lanes to state words (e.g., lanes 0..7 hold a..h clones)
- compute round T1/T2 in parallel with shuffles to rotate roles
- or, 16 lanes do two rounds each per loop (t and t+16), using __shfl_xor

Due to space & risk, keep this as a TODO scaffold you can expand:
- keep per-lane (a,b,c,d,e,f,g,h) fragments
- at each step, use __shfl_sync to get needed neighbors
- reduce back into lane0 at the end with XOR-free shuffles.

======================= */
