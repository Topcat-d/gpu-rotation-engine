// SPDX-License-Identifier: MIT
// Smoke – AES CUDA Baseline (ECB + CTR)
// Goal: correctness-first, reasonable perf; optimize after baseline validates.
// Design: warp-per-message; uint4 (16B) I/O; roundkeys pre-expanded on host.

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ===== Utility =====
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Load/store 16B as uint4
struct u128 { uint4 v; };

static __device__ __forceinline__ uint4 ld_u128(const uint4* p){ return __ldg(p); }
static __device__ __forceinline__ void st_u128(uint4* p, uint4 v){ *p = v; }

// ===== AES constants =====
// S-box in constant memory (shared by AES-128/256)
__constant__ __align__(16) uint8_t AES_SBOX[256] = {
  0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
  0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
  0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
  0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
  0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
  0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
  0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
  0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
  0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
  0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
  0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
  0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
  0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
  0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
  0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
  0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

// Helpers
static __device__ __forceinline__ uint8_t sbox(uint8_t x){ return AES_SBOX[x]; }

static __device__ __forceinline__ uint8_t xtime(uint8_t x){
    return (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1b : 0x00));
}

static __device__ void mix_single_col(uint8_t* a){
    // MixColumns on 4 bytes (one column)
    uint8_t t = a[0] ^ a[1] ^ a[2] ^ a[3];
    uint8_t u = a[0];
    a[0] ^= t ^ xtime((uint8_t)(a[0] ^ a[1]));
    a[1] ^= t ^ xtime((uint8_t)(a[1] ^ a[2]));
    a[2] ^= t ^ xtime((uint8_t)(a[2] ^ a[3]));
    a[3] ^= t ^ xtime((uint8_t)(a[3] ^ u));
}

static __device__ void subbytes_shiftrows(uint8_t st[16]){
    uint8_t t[16];
    // SubBytes + ShiftRows combined
    t[0] = sbox(st[0]);   t[4] = sbox(st[4]);   t[8]  = sbox(st[8]);   t[12] = sbox(st[12]);
    t[1] = sbox(st[5]);   t[5] = sbox(st[9]);   t[9]  = sbox(st[13]);  t[13] = sbox(st[1]);
    t[2] = sbox(st[10]);  t[6] = sbox(st[14]);  t[10] = sbox(st[2]);   t[14] = sbox(st[6]);
    t[3] = sbox(st[15]);  t[7] = sbox(st[3]);   t[11] = sbox(st[7]);   t[15] = sbox(st[11]);
    #pragma unroll
    for(int i=0;i<16;i++) st[i]=t[i];
}

static __device__ __forceinline__ uint32_t pack4(uint8_t a0,uint8_t a1,uint8_t a2,uint8_t a3){
    return (uint32_t)a0 | ((uint32_t)a1<<8) | ((uint32_t)a2<<16) | ((uint32_t)a3<<24);
}

static __device__ __forceinline__ void unpack4(uint32_t w, uint8_t &a0,uint8_t &a1,uint8_t &a2,uint8_t &a3){
    // Big-endian unpacking to match FIPS 197 byte order
    a0 = (uint8_t)((w>>24)&0xFF); a1=(uint8_t)((w>>16)&0xFF); a2=(uint8_t)((w>>8)&0xFF); a3=(uint8_t)(w & 0xFF);
}

// ===== AES core (device) =====
// round_keys: pointer to round key schedule (Nr+1) * 4 words, for each message (can be shared across msgs)
// Nr = 10 (AES-128) or 14 (AES-256)

static __device__ void aes_encrypt_block(const uint32_t* __restrict__ rk, int Nr, const uint8_t in[16], uint8_t out[16]){
    // State as 4 words (column-major)
    uint8_t st[16];
    #pragma unroll
    for (int i=0;i<16;i++) st[i] = in[i];

    // AddRoundKey (round 0)
    uint8_t a0,a1,a2,a3; uint32_t w;
    // col 0
    w = rk[0]; unpack4(w,a0,a1,a2,a3); st[0]^=a0; st[1]^=a1; st[2]^=a2; st[3]^=a3;
    w = rk[1]; unpack4(w,a0,a1,a2,a3); st[4]^=a0; st[5]^=a1; st[6]^=a2; st[7]^=a3;
    w = rk[2]; unpack4(w,a0,a1,a2,a3); st[8]^=a0; st[9]^=a1; st[10]^=a2; st[11]^=a3;
    w = rk[3]; unpack4(w,a0,a1,a2,a3); st[12]^=a0; st[13]^=a1; st[14]^=a2; st[15]^=a3;

    // Rounds 1..Nr-1
    for(int r=1;r<Nr;r++){
        subbytes_shiftrows(st);
        // MixColumns (per column)
        mix_single_col(&st[0]);
        mix_single_col(&st[4]);
        mix_single_col(&st[8]);
        mix_single_col(&st[12]);
        // AddRoundKey
        const uint32_t* rkr = rk + 4*r;
        w = rkr[0]; unpack4(w,a0,a1,a2,a3); st[0]^=a0; st[1]^=a1; st[2]^=a2; st[3]^=a3;
        w = rkr[1]; unpack4(w,a0,a1,a2,a3); st[4]^=a0; st[5]^=a1; st[6]^=a2; st[7]^=a3;
        w = rkr[2]; unpack4(w,a0,a1,a2,a3); st[8]^=a0; st[9]^=a1; st[10]^=a2; st[11]^=a3;
        w = rkr[3]; unpack4(w,a0,a1,a2,a3); st[12]^=a0; st[13]^=a1; st[14]^=a2; st[15]^=a3;
    }
    // Final round
    subbytes_shiftrows(st);
    // AddRoundKey (no MixColumns)
    const uint32_t* rkl = rk + 4*Nr;
    w = rkl[0]; unpack4(w,a0,a1,a2,a3); st[0]^=a0; st[1]^=a1; st[2]^=a2; st[3]^=a3;
    w = rkl[1]; unpack4(w,a0,a1,a2,a3); st[4]^=a0; st[5]^=a1; st[6]^=a2; st[7]^=a3;
    w = rkl[2]; unpack4(w,a0,a1,a2,a3); st[8]^=a0; st[9]^=a1; st[10]^=a2; st[11]^=a3;
    w = rkl[3]; unpack4(w,a0,a1,a2,a3); st[12]^=a0; st[13]^=a1; st[14]^=a2; st[15]^=a3;

    #pragma unroll
    for(int i=0;i<16;i++) out[i]=st[i];
}

// Pack counter block: little-endian counter + 64-bit nonce (CTR layout can be adjusted)
static __device__ __forceinline__ void make_ctr_block(uint64_t nonce_hi, uint64_t ctr_lo, uint8_t out[16]){
    // [0..7] = ctr_lo (LE), [8..15] = nonce_hi (LE) – matches Smoke's other kernels convention
    *(uint64_t*)&out[0] = ctr_lo;
    *(uint64_t*)&out[8] = nonce_hi;
}

// Kernel parameters:
// in/out: flat buffers of N blocks (16*N bytes). Warp-per-message with uniform blocks_per_msg per warp.
// rk: roundkeys (Nr+1)*4 words for AES-128 or AES-256. Shared across messages in this launch (bucketed by key size).
// Nr: 10 or 14.
// blocks_per_msg: number of 16B blocks per message.
// num_msgs: number of messages (warps) in this launch.

extern "C" __global__ void aes_ecb_encrypt_kernel(
    const uint4* __restrict__ in,
    uint4* __restrict__ out,
    const uint32_t* __restrict__ rk,
    int Nr,
    int blocks_per_msg,
    int num_msgs)
{
    const int lane = threadIdx.x & (WARP_SIZE-1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_global = (blockIdx.x * warps_per_block) + warp_in_block;
    if (warp_global >= num_msgs) return;

    const int base_block = warp_global * blocks_per_msg;

    // Each lane processes strided blocks of its message
    for (int b = lane; b < blocks_per_msg; b += WARP_SIZE){
        const int idx = base_block + b;
        uint4 x = ld_u128(in + idx);
        uint8_t inb[16];
        *(uint4*)&inb[0] = x; // aliased store
        uint8_t outb[16];
        aes_encrypt_block(rk, Nr, inb, outb);
        uint4 y = *(uint4*)&outb[0];
        st_u128(out + idx, y);
    }
}

extern "C" __global__ void aes_ctr_encrypt_kernel(
    const uint4* __restrict__ in,  // plaintext
    uint4* __restrict__ out,       // ciphertext
    const uint32_t* __restrict__ rk,
    int Nr,
    uint64_t nonce_hi,
    uint64_t base_counter, // starting counter for message 0
    int blocks_per_msg,
    int num_msgs)
{
    const int lane = threadIdx.x & (WARP_SIZE-1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_global = (blockIdx.x * warps_per_block) + warp_in_block;
    if (warp_global >= num_msgs) return;

    const int base_block = warp_global * blocks_per_msg;
    const uint64_t msg_ctr0 = base_counter + (uint64_t)base_block;

    for (int b = lane; b < blocks_per_msg; b += WARP_SIZE){
        const int idx = base_block + b;
        // form counter block and encrypt to keystream
        uint8_t ctrblk[16];
        make_ctr_block(nonce_hi, (uint64_t)idx + base_counter, ctrblk);
        uint8_t ks[16];
        aes_encrypt_block(rk, Nr, ctrblk, ks);
        // XOR keystream with plaintext
        uint4 p = ld_u128(in + idx);
        uint4 k = *(uint4*)&ks[0];
        uint4 c; c.x = p.x ^ k.x; c.y = p.y ^ k.y; c.z = p.z ^ k.z; c.w = p.w ^ k.w;
        st_u128(out + idx, c);
    }
}
