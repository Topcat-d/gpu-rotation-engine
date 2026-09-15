// cuda/poly1305_26.cuh
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

struct Poly1305Acc {
    // 5 limbs, 26 bits each
    uint32_t h0, h1, h2, h3, h4;
};

struct Poly1305Key26 {
    // r limbs (clamped), and 5*r precomputed for lazy reduction
    uint32_t r0, r1, r2, r3, r4;
    uint32_t s1, s2, s3, s4; // r1*5..r4*5
    uint32_t pad0, pad1, pad2, pad3; // "s" (the second half of one-time key)
};

__device__ __forceinline__ uint32_t ld_le32(const uint8_t* p){
    uint32_t v; memcpy(&v, p, 4); return v;
}

__device__ __forceinline__ void poly1305_clamp(Poly1305Key26& k, const uint8_t* raw_r){
    // Caller provides raw 16B r; we clamp r here.
    // r &= 0x0ffffffc0ffffffc0ffffffc0fffffff (per RFC)
    // We then split into 26-bit limbs.
    uint32_t r0 = ld_le32(raw_r + 0) & 0x0fffffff;
    uint32_t r1 = (ld_le32(raw_r + 3) >> 2) & 0x0fffffff;
    uint32_t r2 = (ld_le32(raw_r + 6) >> 4) & 0x0fffffff;
    uint32_t r3 = (ld_le32(raw_r + 9) >> 6) & 0x0fffffff;
    uint32_t r4 = (ld_le32(raw_r +12) >> 8) & 0x0fffffff;

    // Mask per 26-bit limbs
    k.r0 = r0 & 0x3ffffff;
    k.r1 = r1 & 0x3ffff03; // clears top 2 bits equivalent to & ~0x3
    k.r2 = r2 & 0x3ffc0ff;
    k.r3 = r3 & 0x3f03fff;
    k.r4 = r4 & 0x00fffff;

    k.s1 = k.r1 * 5u;
    k.s2 = k.r2 * 5u;
    k.s3 = k.r3 * 5u;
    k.s4 = k.r4 * 5u;
}

// Split a 16-byte message block + 1 marker into 26-bit limbs
__device__ __forceinline__ void poly1305_load_block26(const uint8_t* m, bool add_1bit,
                                                      uint32_t& t0,uint32_t& t1,uint32_t& t2,uint32_t& t3,uint32_t& t4){
    uint64_t t01 = ((uint64_t)ld_le32(m)) | ((uint64_t)ld_le32(m+4) << 32);
    uint64_t t23 = ((uint64_t)ld_le32(m+8)) | ((uint64_t)ld_le32(m+12) << 32);
    // 26-bit slices
    t0 =  (uint32_t)( t01        & 0x3ffffff);
    t1 =  (uint32_t)((t01 >> 26) & 0x3ffffff);
    uint64_t mid = ((t01 >> 52) | (t23 << 12));
    t2 =  (uint32_t)( mid        & 0x3ffffff);
    t3 =  (uint32_t)((mid >> 26) & 0x3ffffff);
    t4 =  (uint32_t)((t23 >> 40) & 0x3ffffff);
    if (add_1bit) t4 += (1u << 24); // add the 1 bit (represents the 17th byte 0x01)
}

// Multiply-accumulate one 16B block (lazy reduction)
__device__ __forceinline__ void poly1305_acc_1block(Poly1305Acc& h, const Poly1305Key26& k,
                                                    uint32_t t0,uint32_t t1,uint32_t t2,uint32_t t3,uint32_t t4)
{
    // h += t
    uint64_t x0 = (uint64_t)h.h0 + t0;
    uint64_t x1 = (uint64_t)h.h1 + t1;
    uint64_t x2 = (uint64_t)h.h2 + t2;
    uint64_t x3 = (uint64_t)h.h3 + t3;
    uint64_t x4 = (uint64_t)h.h4 + t4;

    // h = (h * r) mod (2^130-5), fully using lazy reduction
    uint64_t d0 = x0 * k.r0 + x1 * k.s4 + x2 * k.s3 + x3 * k.s2 + x4 * k.s1;
    uint64_t d1 = x0 * k.r1 + x1 * k.r0 + x2 * k.s4 + x3 * k.s3 + x4 * k.s2;
    uint64_t d2 = x0 * k.r2 + x1 * k.r1 + x2 * k.r0 + x3 * k.s4 + x4 * k.s3;
    uint64_t d3 = x0 * k.r3 + x1 * k.r2 + x2 * k.r1 + x3 * k.r0 + x4 * k.s4;
    uint64_t d4 = x0 * k.r4 + x1 * k.r3 + x2 * k.r2 + x3 * k.r1 + x4 * k.r0;

    // carry propagate to 26-bit limbs
    uint64_t c;

    c = (d0 >> 26); h.h0 = (uint32_t)(d0 & 0x3ffffff);
    d1 += c;
    c = (d1 >> 26); h.h1 = (uint32_t)(d1 & 0x3ffffff);
    d2 += c;
    c = (d2 >> 26); h.h2 = (uint32_t)(d2 & 0x3ffffff);
    d3 += c;
    c = (d3 >> 26); h.h3 = (uint32_t)(d3 & 0x3ffffff);
    d4 += c;
    c = (d4 >> 26); h.h4 = (uint32_t)(d4 & 0x3ffffff);
    h.h0 += (uint32_t)(c * 5u);
    // one more carry into h1 if h0 overflowed
    h.h1 += (h.h0 >> 26); h.h0 &= 0x3ffffff;
}

// Absorb 64 bytes as 4 blocks of 16B (each with the 1-bit)
__device__ __forceinline__ void poly1305_absorb_4x16B(Poly1305Acc& h, const Poly1305Key26& k, const uint8_t* p64){
    uint32_t t0,t1,t2,t3,t4;

    poly1305_load_block26(p64 +  0, true, t0,t1,t2,t3,t4);
    poly1305_acc_1block(h,k,t0,t1,t2,t3,t4);

    poly1305_load_block26(p64 + 16, true, t0,t1,t2,t3,t4);
    poly1305_acc_1block(h,k,t0,t1,t2,t3,t4);

    poly1305_load_block26(p64 + 32, true, t0,t1,t2,t3,t4);
    poly1305_acc_1block(h,k,t0,t1,t2,t3,t4);

    poly1305_load_block26(p64 + 48, true, t0,t1,t2,t3,t4);
    poly1305_acc_1block(h,k,t0,t1,t2,t3,t4);
}

// Finalize: fold carries, add pad (s), serialize 16-byte tag
__device__ __forceinline__ void poly1305_finish(Poly1305Acc& h, const Poly1305Key26& k, uint8_t tag[16]){
    // full carry reduction
    uint32_t c;
    h.h1 += (h.h0 >> 26); h.h0 &= 0x3ffffff;
    h.h2 += (h.h1 >> 26); h.h1 &= 0x3ffffff;
    h.h3 += (h.h2 >> 26); h.h2 &= 0x3ffffff;
    h.h4 += (h.h3 >> 26); h.h3 &= 0x3ffffff;
    h.h0 += (h.h4 >> 26) * 5u; h.h4 &= 0x3ffffff;
    h.h1 += (h.h0 >> 26); h.h0 &= 0x3ffffff;

    // compute h + -p (conditional subtract)
    uint64_t g0 = (uint64_t)h.h0 + 5u; // add 5, then subtract p
    uint64_t g1 = (uint64_t)h.h1 + (g0 >> 26); g0 &= 0x3ffffff;
    uint64_t g2 = (uint64_t)h.h2 + (g1 >> 26); g1 &= 0x3ffffff;
    uint64_t g3 = (uint64_t)h.h3 + (g2 >> 26); g2 &= 0x3ffffff;
    uint64_t g4 = (uint64_t)h.h4 + (g3 >> 26) - (1ull<<26); g3 &= 0x3ffffff;

    // select h if no underflow, else original
    uint64_t mask = (g4 >> 63) - 1; // all 1s if g4 did not underflow
    uint32_t f0 = (uint32_t)((h.h0 & ~mask) | ((uint32_t)g0 & mask));
    uint32_t f1 = (uint32_t)((h.h1 & ~mask) | ((uint32_t)g1 & mask));
    uint32_t f2 = (uint32_t)((h.h2 & ~mask) | ((uint32_t)g2 & mask));
    uint32_t f3 = (uint32_t)((h.h3 & ~mask) | ((uint32_t)g3 & mask));
    uint32_t f4 = (uint32_t)((h.h4 & ~mask) | ((uint32_t)g4 & mask));

    // serialize to 128-bit little-endian, then add pad s
    uint64_t f01 = (uint64_t)f0 | ((uint64_t)f1 << 26);
    uint64_t f23 = (uint64_t)f2 | ((uint64_t)f3 << 26);
    uint64_t f4w = (uint64_t)f4;

    uint64_t g_lo = (f01        ) | ((f23 & 0xfff) << 52);
    uint64_t g_hi = (f23 >> 12) | (f4w  << 40);

    // add s (pad)
    uint64_t s0 = ((uint64_t)k.pad0) | ((uint64_t)k.pad1 << 32);
    uint64_t s1 = ((uint64_t)k.pad2) | ((uint64_t)k.pad3 << 32);
    g_lo += s0;
    g_hi += s1 + (g_lo < s0);

    memcpy(tag + 0, &g_lo, 8);
    memcpy(tag + 8, &g_hi, 8);
}
