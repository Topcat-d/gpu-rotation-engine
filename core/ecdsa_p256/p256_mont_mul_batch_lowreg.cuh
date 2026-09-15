#ifndef P256_MONT_MUL_BATCH_LOWREG_CUH
#define P256_MONT_MUL_BATCH_LOWREG_CUH

// =============================================================================
// Phase 5: Low-Register Montgomery Multiply for Batch Z-Inversion
// =============================================================================
// Parallel products + serial carry on group leader + broadcast.
// KEY FIX: tracks T[8] overflow across both Step A and Step B.
// The original lowreg omitted Step A's carry→T[8], causing divergence at i=0.
// =============================================================================

__device__ __forceinline__
uint32_t warp_mont_mul_batch_lowreg(
    const uint32_t a[8],
    const uint32_t b[8],
    int lane_id
) {
    const int ll = lane_id & 7;
    const unsigned FULL = 0xFFFFFFFF;

    uint32_t my_a = a[ll];
    uint32_t my_b = b[ll];

    uint32_t Tj = 0;
    uint32_t T8 = 0;  // 9th limb overflow — MUST be tracked across Step A + B

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t a_i = __shfl_sync(FULL, my_a, i, 8);

        // ---- Step A: T += a[i] * b (parallel products, serial carry) ----
        uint32_t lo_a = a_i * my_b;
        uint32_t hi_a = __umulhi(a_i, my_b);

        // Gather to leader
        uint32_t all_Tj[8], all_lo[8], all_hi[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            all_Tj[k] = __shfl_sync(FULL, Tj, k, 8);
            all_lo[k] = __shfl_sync(FULL, lo_a, k, 8);
            all_hi[k] = __shfl_sync(FULL, hi_a, k, 8);
        }

        // Leader: serial Step A + accumulate carry into T8
        uint32_t result_a[8];
        uint32_t T8_after_a = T8;
        if (ll == 0) {
            unsigned long long carry = 0ull;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                unsigned long long prod = ((unsigned long long)all_hi[k] << 32) | all_lo[k];
                unsigned long long s = (unsigned long long)all_Tj[k] + prod + carry;
                result_a[k] = (uint32_t)s;
                carry = s >> 32;
            }
            T8_after_a = T8 + (uint32_t)carry;  // FIX: accumulate into T8
        }

        // Broadcast Step A result + T8
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t val = (ll == 0) ? result_a[k] : 0u;
            val = __shfl_sync(FULL, val, 0, 8);
            if (ll == k) Tj = val;
        }
        T8 = __shfl_sync(FULL, T8_after_a, 0, 8);

        // ---- Step B: T += m * P (Montgomery reduction, serial carry) ----
        uint32_t m = __shfl_sync(FULL, Tj, 0, 8);
        uint32_t my_p = P256_P_LE_CONST[ll];
        uint32_t lo_b = m * my_p;
        uint32_t hi_b = __umulhi(m, my_p);

        uint32_t all_Tj_b[8], all_lo_b[8], all_hi_b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            all_Tj_b[k] = __shfl_sync(FULL, Tj, k, 8);
            all_lo_b[k] = __shfl_sync(FULL, lo_b, k, 8);
            all_hi_b[k] = __shfl_sync(FULL, hi_b, k, 8);
        }

        // Leader: serial Step B + accumulate carry into T8
        uint32_t result_b[8];
        uint32_t T8_after_b = T8;
        if (ll == 0) {
            unsigned long long carry = 0ull;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                unsigned long long prod = ((unsigned long long)all_hi_b[k] << 32) | all_lo_b[k];
                unsigned long long s = (unsigned long long)all_Tj_b[k] + prod + carry;
                result_b[k] = (uint32_t)s;
                carry = s >> 32;
            }
            T8_after_b = T8 + (uint32_t)carry;  // FIX: accumulate into T8
        }

        // Broadcast Step B result + T8
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t val = (ll == 0) ? result_b[k] : 0u;
            val = __shfl_sync(FULL, val, 0, 8);
            if (ll == k) Tj = val;
        }
        T8 = __shfl_sync(FULL, T8_after_b, 0, 8);

        // ---- Step C: Shift T[k] = T[k+1], T[7] = T8, reset T8 ----
        uint32_t next = __shfl_down_sync(FULL, Tj, 1, 8);
        Tj = (ll < 7) ? next : T8;
        T8 = 0;
    }

    // ---- Conditional subtraction: BRANCHLESS ----
    uint32_t all_T[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        all_T[k] = __shfl_sync(FULL, Tj, k, 8);
    }

    uint32_t result_final = Tj;
    if (ll == 0) {
        bool is_ge = false;
        for (int k = 7; k >= 0; --k) {
            if (all_T[k] > P256_P_LE_CONST[k]) { is_ge = true; break; }
            if (all_T[k] < P256_P_LE_CONST[k]) { is_ge = false; break; }
        }
        if (is_ge) {
            unsigned long long borrow = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                unsigned long long d = (unsigned long long)all_T[k]
                    - (unsigned long long)P256_P_LE_CONST[k] - borrow;
                all_T[k] = (uint32_t)d;
                borrow = (d >> 63) & 1;
            }
        }
    }

    // Broadcast final result from leader
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t val = (ll == 0) ? all_T[k] : 0u;
        val = __shfl_sync(FULL, val, 0, 8);
        if (ll == k) result_final = val;
    }

    return result_final;
}

#endif // P256_MONT_MUL_BATCH_LOWREG_CUH
