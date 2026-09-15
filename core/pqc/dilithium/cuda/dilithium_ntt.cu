// dilithium_ntt.cu - CUDA NTT implementation for Dilithium (ML-DSA)
// Card 09: Baseline single-thread-per-polynomial implementation
// Card 10: Warp-cooperative optimized implementation (32 threads per poly)
//
// Baseline is "boring, obviously correct" for validation.
// Warp kernel is faster but must remain bit-identical to baseline.

#include "../include/dilithium_ntt.cuh"
#include <stdexcept>
#include <cstdio>  // Card 27.20: Debug output

namespace smoke {
namespace dilithium {

namespace {

constexpr uint32_t N = 256;
constexpr uint32_t Q = 8380417;

// ============================================================================
// Precomputed Constants
// ============================================================================

// Primitive 512-th root of unity: zeta = 1753
// zeta^256 = -1 mod q, zeta^512 = 1 mod q
constexpr uint32_t ZETA = 1753;

// FIPS 204 style pre-bit-reversed zetas for negacyclic NTT
// ZETAS[k] = zeta^bitrev8(k) for k = 1..255, ZETAS[0] unused
__constant__ uint32_t ZETAS[256] = {
    0, 4808194, 3765607, 3761513, 5178923, 5496691, 5234739, 5178987,
    7778734, 3542485, 2682288, 2129892, 3764867, 7375178, 557458, 7159240,
    5010068, 4317364, 2663378, 6705802, 4855975, 7946292, 676590, 7044481,
    5152541, 1714295, 2453983, 1460718, 7737789, 4795319, 2815639, 2283733,
    3602218, 3182878, 2740543, 4793971, 5269599, 2101410, 3704823, 1159875,
    394148, 928749, 1095468, 4874037, 2071829, 4361428, 3241972, 2156050,
    3415069, 1759347, 7562881, 4805951, 3756790, 6444618, 6663429, 4430364,
    5483103, 3192354, 556856, 3870317, 2917338, 1853806, 3345963, 1858416,
    3073009, 1277625, 5744944, 3852015, 4183372, 5157610, 5258977, 8106357,
    2508980, 2028118, 1937570, 4564692, 2811291, 5396636, 7270901, 4158088,
    1528066, 482649, 1148858, 5418153, 7814814, 169688, 2462444, 5046034,
    4213992, 4892034, 1987814, 5183169, 1736313, 235407, 5130263, 3258457,
    5801164, 1787943, 5989328, 6125690, 3482206, 4197502, 7080401, 6018354,
    7062739, 2461387, 3035980, 621164, 3901472, 7153756, 2925816, 3374250,
    1356448, 5604662, 2683270, 5601629, 4912752, 2312838, 7727142, 7921254,
    348812, 8052569, 1011223, 6026202, 4561790, 6458164, 6143691, 1744507,
    1753, 6444997, 5720892, 6924527, 2660408, 6600190, 8321269, 2772600,
    1182243, 87208, 636927, 4415111, 4423672, 6084020, 5095502, 4663471,
    8352605, 822541, 1009365, 5926272, 6400920, 1596822, 4423473, 4620952,
    6695264, 4969849, 2678278, 4611469, 4829411, 635956, 8129971, 5925040,
    4234153, 6607829, 2192938, 6653329, 2387513, 4768667, 8111961, 5199961,
    3747250, 2296099, 1239911, 4541938, 3195676, 2642980, 1254190, 8368000,
    2998219, 141835, 8291116, 2513018, 7025525, 613238, 7070156, 6161950,
    7921677, 6458423, 4040196, 4908348, 2039144, 6500539, 7561656, 6201452,
    6757063, 2105286, 6006015, 6346610, 586241, 7200804, 527981, 5637006,
    6903432, 1994046, 2491325, 6987258, 507927, 7192532, 7655613, 6545891,
    5346675, 8041997, 2647994, 3009748, 5767564, 4148469, 749577, 4357667,
    3980599, 2569011, 6764887, 1723229, 1665318, 2028038, 1163598, 5011144,
    3994671, 8368538, 7009900, 3020393, 3363542, 214880, 545376, 7609976,
    3105558, 7277073, 508145, 7826699, 860144, 3430436, 140244, 6866265,
    6195333, 3123762, 2358373, 6187330, 5365997, 6663603, 2926054, 7987710,
    8077412, 3531229, 4405932, 4606686, 1900052, 7598542, 1054478, 7648983,
};

// Negative zetas for INTT: ZETAS_INV[k] = -ZETAS[k] mod q
__constant__ uint32_t ZETAS_INV[256] = {
    0, 3572223, 4614810, 4618904, 3201494, 2883726, 3145678, 3201430,
    601683, 4837932, 5698129, 6250525, 4615550, 1005239, 7822959, 1221177,
    3370349, 4063053, 5717039, 1674615, 3524442, 434125, 7703827, 1335936,
    3227876, 6666122, 5926434, 6919699, 642628, 3585098, 5564778, 6096684,
    4778199, 5197539, 5639874, 3586446, 3110818, 6279007, 4675594, 7220542,
    7986269, 7451668, 7284949, 3506380, 6308588, 4018989, 5138445, 6224367,
    4965348, 6621070, 817536, 3574466, 4623627, 1935799, 1716988, 3950053,
    2897314, 5188063, 7823561, 4510100, 5463079, 6526611, 5034454, 6522001,
    5307408, 7102792, 2635473, 4528402, 4197045, 3222807, 3121440, 274060,
    5871437, 6352299, 6442847, 3815725, 5569126, 2983781, 1109516, 4222329,
    6852351, 7897768, 7231559, 2962264, 565603, 8210729, 5917973, 3334383,
    4166425, 3488383, 6392603, 3197248, 6644104, 8145010, 3250154, 5121960,
    2579253, 6592474, 2391089, 2254727, 4898211, 4182915, 1300016, 2362063,
    1317678, 5919030, 5344437, 7759253, 4478945, 1226661, 5454601, 5006167,
    7023969, 2775755, 5697147, 2778788, 3467665, 6067579, 653275, 459163,
    8031605, 327848, 7369194, 2354215, 3818627, 1922253, 2236726, 6635910,
    8378664, 1935420, 2659525, 1455890, 5720009, 1780227, 59148, 5607817,
    7198174, 8293209, 7743490, 3965306, 3956745, 2296397, 3284915, 3716946,
    27812, 7557876, 7371052, 2454145, 1979497, 6783595, 3956944, 3759465,
    1685153, 3410568, 5702139, 3768948, 3551006, 7744461, 250446, 2455377,
    4146264, 1772588, 6187479, 1727088, 5992904, 3611750, 268456, 3180456,
    4633167, 6084318, 7140506, 3838479, 5184741, 5737437, 7126227, 12417,
    5382198, 8238582, 89301, 5867399, 1354892, 7767179, 1310261, 2218467,
    458740, 1921994, 4340221, 3472069, 6341273, 1879878, 818761, 2178965,
    1623354, 6275131, 2374402, 2033807, 7794176, 1179613, 7852436, 2743411,
    1476985, 6386371, 5889092, 1393159, 7872490, 1187885, 724804, 1834526,
    3033742, 338420, 5732423, 5370669, 2612853, 4231948, 7630840, 4022750,
    4399818, 5811406, 1615530, 6657188, 6715099, 6352379, 7216819, 3369273,
    4385746, 11879, 1370517, 5360024, 5016875, 8165537, 7835041, 770441,
    5274859, 1103344, 7872272, 553718, 7520273, 4949981, 8240173, 1514152,
    2185084, 5256655, 6022044, 2193087, 3014420, 1716814, 5454363, 392707,
    303005, 4849188, 3974485, 3773731, 6480365, 781875, 7325939, 731434,
};

// N^-1 mod q for final INTT scaling
constexpr uint32_t N_INV = 8347681;

// Bit-reversal table for N=256 (8 bits)
__constant__ uint8_t BIT_REV[256] = {
    0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 48, 176, 112, 240,
    8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
    4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244,
    12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
    2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242,
    10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
    6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246,
    14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
    1, 129, 65, 193, 33, 161, 97, 225, 17, 145, 81, 209, 49, 177, 113, 241,
    9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
    5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
    13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
    3, 131, 67, 195, 35, 163, 99, 227, 19, 147, 83, 211, 51, 179, 115, 243,
    11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
    7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247,
    15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255
};

static bool g_ntt_initialized = false;

// ============================================================================
// Modular Arithmetic Device Functions
// ============================================================================

// Modular multiplication: (a * b) mod q
__device__ __forceinline__
uint32_t mul_mod_q(uint32_t a, uint32_t b) {
    uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    return static_cast<uint32_t>(prod % Q);
}

// Modular addition: (a + b) mod q
__device__ __forceinline__
uint32_t add_mod_q(uint32_t a, uint32_t b) {
    uint32_t r = a + b;
    return (r >= Q) ? (r - Q) : r;
}

// Modular subtraction: (a - b) mod q, result in [0, q)
__device__ __forceinline__
uint32_t sub_mod_q(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (Q + a - b);
}

// Modular power (device-side, for computing omega powers on the fly)
__device__ __forceinline__
uint32_t pow_mod_q(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    base = base % Q;
    while (exp > 0) {
        if (exp & 1) {
            result = mul_mod_q(result, base);
        }
        exp >>= 1;
        base = mul_mod_q(base, base);
    }
    return result;
}

// ============================================================================
// Baseline NTT Device Functions (single-thread per polynomial)
// FIPS 204 Algorithm 41/42 - Negacyclic NTT with pre-bit-reversed zetas
// ============================================================================

// Forward NTT (FIPS 204 Algorithm 41)
// Input: polynomial coefficients a[0..255] in normal order
// Output: NTT(a) in NTT domain
__device__ void device_ntt_forward_baseline(uint32_t* a) {
    int k = 0;
    for (int len = 128; len >= 1; len >>= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            k++;
            uint32_t zeta = ZETAS[k];
            for (int j = start; j < start + len; j++) {
                uint32_t t = mul_mod_q(zeta, a[j + len]);
                a[j + len] = sub_mod_q(a[j], t);
                a[j] = add_mod_q(a[j], t);
            }
        }
    }
}

// Inverse NTT (FIPS 204 Algorithm 42)
// Input: NTT coefficients a[0..255]
// Output: INTT(a) in normal domain (scaled by N^-1)
__device__ void device_ntt_inverse_baseline(uint32_t* a) {
    int k = 256;
    for (int len = 1; len <= 128; len <<= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            k--;
            uint32_t zeta = ZETAS_INV[k];
            for (int j = start; j < start + len; j++) {
                uint32_t t = a[j];
                a[j] = add_mod_q(t, a[j + len]);
                // Note: (t - a[j+len]), not (a[j+len] - t)
                a[j + len] = mul_mod_q(zeta, sub_mod_q(t, a[j + len]));
            }
        }
    }

    // Final scaling by N^-1
    for (int i = 0; i < N; ++i) {
        a[i] = mul_mod_q(a[i], N_INV);
    }
}

// ============================================================================
// Baseline Kernels (1 thread per polynomial)
// ============================================================================

__global__ void kernel_ntt_forward_baseline(uint32_t* __restrict__ coeffs,
                                            std::size_t batch_size) {
    std::size_t poly_idx = blockIdx.x;
    if (poly_idx >= batch_size) return;
    if (threadIdx.x != 0) return;  // Only thread 0 does work

    uint32_t* poly = coeffs + poly_idx * N;
    device_ntt_forward_baseline(poly);
}

__global__ void kernel_ntt_inverse_baseline(uint32_t* __restrict__ coeffs,
                                            std::size_t batch_size) {
    std::size_t poly_idx = blockIdx.x;
    if (poly_idx >= batch_size) return;
    if (threadIdx.x != 0) return;

    uint32_t* poly = coeffs + poly_idx * N;
    device_ntt_inverse_baseline(poly);
}

// ============================================================================
// Card 10: Warp-Cooperative NTT (32 threads per polynomial)
// FIPS 204 Algorithm 41/42 - Negacyclic NTT with pre-bit-reversed zetas
// ============================================================================

constexpr int WARP_SIZE = 32;

// Forward NTT using shared memory, 32 threads cooperating (FIPS 204)
__device__ void device_ntt_forward_warp(uint32_t* s_poly) {
    int lane = threadIdx.x;  // 0..31

    // FIPS 204 forward NTT structure
    // k increments from 1 to 255, len halves from 128 to 1
    int k = 0;
    for (int len = 128; len >= 1; len >>= 1) {
        int num_groups = N / (2 * len);  // Number of butterfly groups

        // Each group uses one zeta value
        // Total butterflies per len = 128 (len butterflies per group * num_groups)
        for (int group = 0; group < num_groups; group++) {
            k++;
            uint32_t zeta = ZETAS[k];
            int start = group * 2 * len;

            // Parallelize the butterflies within this group
            for (int j = lane; j < len; j += WARP_SIZE) {
                int idx0 = start + j;
                int idx1 = start + j + len;

                uint32_t t = mul_mod_q(zeta, s_poly[idx1]);
                s_poly[idx1] = sub_mod_q(s_poly[idx0], t);
                s_poly[idx0] = add_mod_q(s_poly[idx0], t);
            }
        }
        __syncthreads();  // Sync before next len
    }
}

// Inverse NTT using shared memory, 32 threads cooperating (FIPS 204)
__device__ void device_ntt_inverse_warp(uint32_t* s_poly) {
    int lane = threadIdx.x;  // 0..31

    // FIPS 204 inverse NTT structure
    // k decrements from 255 to 1, len doubles from 1 to 128
    int k = 256;
    for (int len = 1; len <= 128; len <<= 1) {
        int num_groups = N / (2 * len);

        for (int group = 0; group < num_groups; group++) {
            k--;
            uint32_t zeta = ZETAS_INV[k];
            int start = group * 2 * len;

            // Parallelize the butterflies within this group
            for (int j = lane; j < len; j += WARP_SIZE) {
                int idx0 = start + j;
                int idx1 = start + j + len;

                uint32_t t = s_poly[idx0];
                s_poly[idx0] = add_mod_q(t, s_poly[idx1]);
                // Note: (t - a[j+len]), not (a[j+len] - t)
                s_poly[idx1] = mul_mod_q(zeta, sub_mod_q(t, s_poly[idx1]));
            }
        }
        __syncthreads();  // Sync before next len
    }

    // Final scaling by N^-1
    for (int i = lane; i < N; i += WARP_SIZE) {
        s_poly[i] = mul_mod_q(s_poly[i], N_INV);
    }
    __syncthreads();
}

// Warp forward NTT kernel: 1 block per polynomial, 32 threads per block
__global__ void kernel_ntt_forward_warp(uint32_t* __restrict__ coeffs,
                                        std::size_t batch_size) {
    std::size_t poly_idx = blockIdx.x;
    if (poly_idx >= batch_size) return;
    int lane = threadIdx.x;
    if (lane >= WARP_SIZE) return;

    __shared__ uint32_t s_poly[N];

    // Load polynomial from global to shared memory
    for (int i = lane; i < N; i += WARP_SIZE) {
        s_poly[i] = coeffs[poly_idx * N + i];
    }
    __syncthreads();

    device_ntt_forward_warp(s_poly);

    // Store back to global memory
    for (int i = lane; i < N; i += WARP_SIZE) {
        coeffs[poly_idx * N + i] = s_poly[i];
    }
}

// Warp inverse NTT kernel: 1 block per polynomial, 32 threads per block
__global__ void kernel_ntt_inverse_warp(uint32_t* __restrict__ coeffs,
                                        std::size_t batch_size) {
    std::size_t poly_idx = blockIdx.x;
    if (poly_idx >= batch_size) return;
    int lane = threadIdx.x;
    if (lane >= WARP_SIZE) return;

    __shared__ uint32_t s_poly[N];

    // Load polynomial from global to shared memory
    for (int i = lane; i < N; i += WARP_SIZE) {
        s_poly[i] = coeffs[poly_idx * N + i];
    }
    __syncthreads();

    device_ntt_inverse_warp(s_poly);

    // Store back to global memory
    for (int i = lane; i < N; i += WARP_SIZE) {
        coeffs[poly_idx * N + i] = s_poly[i];
    }
}

// ============================================================================
// Pointwise Operations
// ============================================================================

__global__ void kernel_pointwise_mul(uint32_t* __restrict__ out,
                                     const uint32_t* __restrict__ a,
                                     const uint32_t* __restrict__ b,
                                     std::size_t total_coeffs) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) return;
    out[idx] = mul_mod_q(a[idx], b[idx]);
}

__global__ void kernel_pointwise_add(uint32_t* __restrict__ out,
                                     const uint32_t* __restrict__ a,
                                     const uint32_t* __restrict__ b,
                                     std::size_t total_coeffs) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) return;
    out[idx] = add_mod_q(a[idx], b[idx]);
}

__global__ void kernel_pointwise_sub(uint32_t* __restrict__ out,
                                     const uint32_t* __restrict__ a,
                                     const uint32_t* __restrict__ b,
                                     std::size_t total_coeffs) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) return;
    out[idx] = sub_mod_q(a[idx], b[idx]);
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

void ntt_init() {
    if (g_ntt_initialized) return;
    // ZETAS, ZETAS_INV, and BIT_REV are already in __constant__ memory
    // No runtime initialization needed
    g_ntt_initialized = true;
}

bool ntt_is_initialized() {
    return g_ntt_initialized;
}

// ============================================================================
// Explicit Baseline Entrypoints (for testing/validation)
// ============================================================================

void ntt_forward_batch_baseline(uint32_t* d_coeffs,
                                std::size_t batch_size,
                                const Params& params,
                                cudaStream_t stream) {
    if (!g_ntt_initialized) {
        ntt_init();
    }

    dim3 grid(batch_size);
    dim3 block(1);
    kernel_ntt_forward_baseline<<<grid, block, 0, stream>>>(d_coeffs, batch_size);
}

void ntt_inverse_batch_baseline(uint32_t* d_coeffs,
                                std::size_t batch_size,
                                const Params& params,
                                cudaStream_t stream) {
    if (!g_ntt_initialized) {
        ntt_init();
    }

    dim3 grid(batch_size);
    dim3 block(1);
    kernel_ntt_inverse_baseline<<<grid, block, 0, stream>>>(d_coeffs, batch_size);
}

// ============================================================================
// Primary NTT API with Implementation Selection
// ============================================================================

void ntt_forward_batch(uint32_t* d_coeffs,
                       std::size_t batch_size,
                       const Params& params,
                       cudaStream_t stream,
                       NTTImpl impl) {
    if (!g_ntt_initialized) {
        ntt_init();
    }

    switch (impl) {
        case NTTImpl::BASELINE:
            {
                dim3 grid(batch_size);
                dim3 block(1);
                kernel_ntt_forward_baseline<<<grid, block, 0, stream>>>(d_coeffs, batch_size);
            }
            break;

        case NTTImpl::WARP:
        case NTTImpl::AUTO:
        default:
            // WARP/AUTO: warp-cooperative kernel (faster, must be bit-identical to baseline)
            {
                dim3 grid(batch_size);
                dim3 block(WARP_SIZE);
                kernel_ntt_forward_warp<<<grid, block, 0, stream>>>(d_coeffs, batch_size);
            }
            break;
    }
}

void ntt_inverse_batch(uint32_t* d_coeffs,
                       std::size_t batch_size,
                       const Params& params,
                       cudaStream_t stream,
                       NTTImpl impl) {
    if (!g_ntt_initialized) {
        ntt_init();
    }

    switch (impl) {
        case NTTImpl::BASELINE:
            {
                dim3 grid(batch_size);
                dim3 block(1);
                kernel_ntt_inverse_baseline<<<grid, block, 0, stream>>>(d_coeffs, batch_size);
            }
            break;

        case NTTImpl::WARP:
            {
                dim3 grid(batch_size);
                dim3 block(WARP_SIZE);
                kernel_ntt_inverse_warp<<<grid, block, 0, stream>>>(d_coeffs, batch_size);
            }
            break;

        case NTTImpl::AUTO:
        default:
            // AUTO defaults to WARP
            {
                dim3 grid(batch_size);
                dim3 block(WARP_SIZE);
                kernel_ntt_inverse_warp<<<grid, block, 0, stream>>>(d_coeffs, batch_size);
            }
            break;
    }
}

void pointwise_multiply_batch(uint32_t* d_out,
                              const uint32_t* d_a,
                              const uint32_t* d_b,
                              std::size_t batch_size,
                              const Params& params,
                              cudaStream_t stream) {
    const std::size_t total = batch_size * N;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);

    kernel_pointwise_mul<<<grid, block, 0, stream>>>(d_out, d_a, d_b, total);
}

void pointwise_add_batch(uint32_t* d_out,
                         const uint32_t* d_a,
                         const uint32_t* d_b,
                         std::size_t batch_size,
                         const Params& params,
                         cudaStream_t stream) {
    const std::size_t total = batch_size * N;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);

    kernel_pointwise_add<<<grid, block, 0, stream>>>(d_out, d_a, d_b, total);
}

void pointwise_sub_batch(uint32_t* d_out,
                         const uint32_t* d_a,
                         const uint32_t* d_b,
                         std::size_t batch_size,
                         const Params& params,
                         cudaStream_t stream) {
    const std::size_t total = batch_size * N;
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);

    kernel_pointwise_sub<<<grid, block, 0, stream>>>(d_out, d_a, d_b, total);
}

#ifdef DEBUG_DILITHIUM
void ntt_debug_dump_stage(const uint32_t* d_coeffs,
                          std::size_t batch_size,
                          std::size_t stage,
                          uint32_t* d_debug_out,
                          cudaStream_t stream) {
    // TODO: Implement debug stage dumping
    (void)d_coeffs;
    (void)batch_size;
    (void)stage;
    (void)d_debug_out;
    (void)stream;
}
#endif

} // namespace dilithium
} // namespace smoke
