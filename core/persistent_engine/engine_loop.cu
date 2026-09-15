#include <cuda_runtime.h>
#include <cstdio>  // Optimize-6.4: For fprintf diagnostics in launch_fast_path
#include "engine_state.cuh"
#include "engine_constants.cuh"

// Card 26.5: Define telemetry pointer BEFORE including signer headers
// This must be defined here so that the macros in p256_field_warp_coop.cuh can use it
namespace smoke { namespace engine {
__device__ FastPathTelemetry* g_telemetry_ptr = nullptr;
}}
using smoke::engine::g_telemetry_ptr;

// Card 08: Include real P-256 signing implementation
#include "../p256_persistent/p256_persistent.cuh"

// Card 26: Include warp-cooperative P-256 signer (disabled by default)
// Enable with -DP256_WARP_COOP_ENABLED=1 when multi-CTA is ready
// Trade-off: Warp-coop is faster per-signature but only 1 sig at a time per warp
//            Thread-only is slower per-signature but 32 concurrent signatures per warp
// With single CTA, thread-only wins. With multi-CTA, warp-coop wins.
#include "../p256_persistent/p256_sign_warp_coop.cuh"

// Card 26.6: Include Card14 batch Z-inversion for batched signing
// This enables 1 inversion per batch instead of 1 per signature
// NOTE: Requires warp-coop functions, so only include when WARP_COOP is enabled
#if P256_WARP_COOP_ENABLED
#include "../ecdsa_p256/p256_batch_zinv_card14.cuh"
#endif

// Card 18: Include integrity tag computation and verification
#include "integrity_device.cuh"

// Card 19: Include HKDF self-healing support
#include "self_healing_device.cuh"

// Card 26.19: Include SHA-256 device implementation and opcode schema
#include "sha256_device.cuh"
// Card 26.31: Include SHA-512 device implementation
#include "sha512_device.cuh"
// Card 26.32: Include SHA3-256 device implementation
#include "sha3_256_device.cuh"
// Card 26.33: Include BLAKE2b-256 device implementation
#include "blake2b_256_device.cuh"
// Card 140: Include SHA3-512 device implementation
#include "sha3_512_device.cuh"
// Card 141: Include BLAKE2b-512 device implementation
#include "blake2b_512_device.cuh"
#include "op_schema.cuh"

// Card 26.29: Include HMAC-SHA256 device implementation
#include "../hmac_gpu/hmac_sha2_device.cuh"

// Card 26.54: Include AES-256-CTR device implementation
#include "../aes/aes256_ctr_device.cuh"

// Card 26.65: Include AES-256-GCM device implementation
#include "../aes/aes256_gcm_device.cuh"

// Card 26.66: Include ChaCha20 device implementation
#include "../chacha/chacha20_device.cuh"

// Card 26.67: Include Poly1305 device implementation
#include "../poly1305/poly1305_device.cuh"

// Card 26.68: Include ChaCha20-Poly1305 AEAD device implementation
#include "../chacha/chacha20_poly1305_device.cuh"

// Card 26.69: Include HKDF-SHA256 device implementation
#include "../hkdf/hkdf_sha256_device.cuh"

// Card 26.70: Include RSA-2048 CRT device implementation
#include "rsa2048_crt_device.cuh"

// Compile-time switch: 0 = stub mode, 1 = real P-256 ECDSA
// Set via -DENGINE_USE_REAL_P256=1 in nvcc flags, or default to real
#ifndef ENGINE_USE_REAL_P256
#define ENGINE_USE_REAL_P256 1
#endif

// Card 26.3: Lite telemetry mode - skip per-iteration debug atomics
// Set via -DBENCH_LITE_TELEMETRY=1 for performance benchmarks
#ifndef BENCH_LITE_TELEMETRY
#define BENCH_LITE_TELEMETRY 0
#endif

namespace smoke {
namespace engine {

static __device__ __forceinline__ void engine_device_yield() {
    __nanosleep(1000);
}

// Nonce counter for deterministic nonce generation (Card 08)
// Persists across requests within a kernel launch
__device__ uint32_t g_nonce_counter = 0;

// Card 16: Select active keyslot based on rotation policy
// - DISABLED: Always slot 0 (single-slot mode)
// - MANUAL/AUTOMATIC: Use active_epoch to select slot
__device__ __forceinline__
uint32_t engine_select_active_slot(const EngineState* state) {
    const EngineTuning& tuning = state->tuning;
    switch (tuning.rotation_policy) {
    case RotationPolicy::DISABLED:
        return 0;  // Always slot 0, ignore active_epoch
    case RotationPolicy::MANUAL:
    case RotationPolicy::AUTOMATIC:
    default:
        return (state->active_epoch & 1u);  // Existing behavior
    }
}

// SEAL-SLAB-01: validation shared by both AEAD seal opcodes. These checks live
// in the kernel even though host clients also validate, because untrusted or
// stale descriptors must never produce a successful seal under an empty,
// wrong-type, or faulted keyslot.
__device__ __forceinline__
uint8_t validate_aead_keyslot(const Keyslot* keyslots, uint8_t key_slot) {
    if (key_slot >= ENGINE_MAX_KEY_SLOTS) {
        return static_cast<uint8_t>(OpStatus::INVALID_KEY_SLOT);
    }
    const Keyslot& ks = keyslots[key_slot];
    if (ks.key_type != static_cast<uint32_t>(KeyslotType::KEYSLOT_AES256) ||
        ks.key_len != KEYSLOT_AES256_KEY_LEN ||
        (ks.fault_flags & FAULT_HARD_ZEROIZED) != 0) {
        return static_cast<uint8_t>(OpStatus::KEY_NOT_LOADED);
    }
    return static_cast<uint8_t>(OpStatus::OK);
}

__device__ __forceinline__
bool valid_payload_slab_range(uint32_t offset, uint32_t length) {
    return length > 0 &&
           length <= FAST_PATH_SEAL_MAX_PLAINTEXT &&
           offset <= FAST_PATH_PAYLOAD_SLAB_SIZE &&
           length <= FAST_PATH_PAYLOAD_SLAB_SIZE - offset;
}

// Open shares the 64 KiB seal output geometry. The input includes the tag, so
// it may be 16 bytes larger than the plaintext while the authenticated output
// is still capped at FAST_PATH_SEAL_MAX_PLAINTEXT.
__device__ __forceinline__
bool valid_open_payload_slab_range(uint32_t offset, uint32_t length) {
    return length > FAST_PATH_SEAL_TAG_BYTES &&
           length <=
               FAST_PATH_SEAL_MAX_PLAINTEXT + FAST_PATH_SEAL_TAG_BYTES &&
           offset <= FAST_PATH_PAYLOAD_SLAB_SIZE &&
           length <= FAST_PATH_PAYLOAD_SLAB_SIZE - offset;
}

// Card 24.5: Write completion flag to mapped memory if available, else inline array
// Uses mapped_ready_flags pointer if set, otherwise falls back to response_ready[]
// NOTE: For mapped memory, we use __threadfence_system() to ensure CPU visibility
__device__ __forceinline__
void engine_set_response_ready(EngineState* state, uint32_t idx) {
    if (state->mapped_ready_flags != nullptr) {
        // Card 24.5: Write to host-mapped memory
        // Use __threadfence_system() to ensure CPU can see the write
        state->mapped_ready_flags[idx] = 1;
        __threadfence_system();  // Ensure visibility to CPU
    } else {
        // Legacy: write to device memory (requires cudaMemcpy to read)
        state->response_ready[idx] = 1;
    }
}

// =============================================================================
// Card 26.72: Response Commit Protocol
// =============================================================================
// Problem: Host poll can observe response slots before kernel fully writes them.
// Solution: Write request_id LAST as commit signal. Host skips slots with request_id==0.
//
// Protocol:
// 1. Kernel reserves slot by incrementing write_idx
// 2. Kernel writes all response data EXCEPT request_id
// 3. __threadfence_system() ensures writes visible to host (system scope)
// 4. Kernel writes request_id (non-zero) as commit signal, packaged with
//    commit_info (output_len + status) in ONE 16-byte store (B-319b)
// 5. Host poll checks request_id != 0 before consuming, then takes
//    output_len/status from commit_info — never from cache line 1
//
// B-319b (2026-07-11): step 3's fence orders the writes at the GPU, but the
// PCIe fabric delivered the line-0 commit before the line-1 payload on
// A100/L40S under load (see engine_state.cuh B-319b comment, D-091). The
// commit signal and the fields the host must trust before touching payload
// therefore travel in a single 16-byte transaction.

// Helper: Write response data (all fields except request_id)
// Call this FIRST, then write any output-specific data, then call commit_response()
__device__ __forceinline__
void write_response_base(volatile FastPathResponse* dst, const FastPathResponse& resp) {
    // Write all fields EXCEPT request_id
    dst->seq = resp.seq;
    dst->t_submit_us = resp.t_submit_us;
    dst->t_dequeue_us = resp.t_dequeue_us;
    dst->t_complete_us = resp.t_complete_us;
    dst->status = resp.status;
    dst->client_id = resp.client_id;
    dst->epoch_used = resp.epoch_used;    // Card 66
    dst->slot_used = resp.slot_used;      // Card 66
    dst->opcode = resp.opcode;
    dst->output_len = resp.output_len;
    dst->output_offset = resp.output_offset;
    dst->output_bytes = resp.output_bytes;
}

// B-319b: the bare commit store — publishes {request_id, commit_info} as a
// single 16-byte vector store. One PTX instruction, 16B-aligned (slot offset
// 0, slots are alignas(64)), so it reaches the host as one PCIe transaction:
// the host can never observe the commit signal without the matching
// output_len/status. Caller must issue __threadfence_system() FIRST to order
// the cache-line-1/2 payload writes before this store.
__device__ __forceinline__
void commit_store(volatile FastPathResponse* dst,
                  uint64_t request_id, uint8_t status, uint16_t output_len) {
    const uint64_t info = fast_path_commit_info(status, output_len);
    asm volatile("st.volatile.v2.u64 [%0], {%1, %2};"
                 :: "l"(const_cast<FastPathResponse*>(dst)),
                    "l"(request_id), "l"(info)
                 : "memory");
}

// Helper: Commit response by writing request_id LAST
// Uses system fence for host-mapped memory visibility
__device__ __forceinline__
void commit_response(volatile FastPathResponse* dst,
                     uint64_t request_id, uint8_t status, uint16_t output_len) {
    __threadfence_system();  // System scope fence for host visibility
    // Commit signal - written LAST (never 0 for valid requests), fused with
    // commit_info (B-319b)
    commit_store(dst, request_id, status, output_len);
}

// =============================================================================

// Card 10: Adaptive microbatch size computation
// Returns optimal batch size based on queue depth:
// - Low depth (<= low_water): small batches for low latency
// - High depth (>= high_water): max batches for throughput
// - In between: linear interpolation
__device__ __forceinline__
uint32_t engine_compute_microbatch_size(
    uint32_t available,
    uint32_t min_batch,
    uint32_t max_batch,
    uint32_t low_water,
    uint32_t high_water)
{
    if (available == 0) {
        return 0;
    }

    if (available <= low_water) {
        // Low load: prefer small batches for low latency
        return (available < min_batch) ? available : min_batch;
    }

    if (available >= high_water) {
        // High load: max batches for throughput
        return max_batch;
    }

    // Linear interpolation between min_batch and max_batch
    uint32_t span   = high_water - low_water;
    uint32_t offset = available - low_water;
    uint32_t range  = max_batch - min_batch;

    uint32_t scaled    = (range * offset) / (span > 0 ? span : 1);
    uint32_t candidate = min_batch + scaled;

    // Clamp to valid range
    if (candidate < min_batch) candidate = min_batch;
    if (candidate > max_batch) candidate = max_batch;

    return candidate;
}

// Persistent kernel entry point (P-256 engine for Phase 2).
// Card 06-07: Stub signing with keyslot rotation
// Card 08+: Real P-256 ECDSA signing
// Card 09+: Microbatch scheduling (process up to ENGINE_MICROBATCH_MAX per loop)
// Card 10+: Adaptive microbatch sizing based on queue depth
static __global__ void p256_engine_loop(EngineState* state) {
    if (!state) {
        return;
    }

    const uint32_t lane = threadIdx.x;

    while (true) {
        if (state->shutdown_flag == ENGINE_SHUTDOWN_TRUE) {
            return;
        }

        // Only lane 0 manages the host<->device queue.
        if (lane == 0) {
            // Card 09: Snapshot queue bounds once per loop iteration
            uint32_t head = state->req_head;
            uint32_t tail = state->req_tail;

            // Compute how many requests are available
            uint32_t available = (head > tail) ? (head - tail) : 0;

            if (available > 0) {
                // Card 11: Read runtime tuning from state
                EngineTuning tuning = state->tuning;

                // Card 10: Adaptive batch size based on queue depth
                // Card 11: Now uses runtime-configurable tuning instead of constants
                uint32_t batch = engine_compute_microbatch_size(
                    available,
                    tuning.microbatch_min,
                    tuning.microbatch_max,
                    tuning.depth_low_water,
                    tuning.depth_high_water
                );

                // Sanity clamp: never process more than available
                if (batch > available) {
                    batch = available;
                }
                if (batch == 0) {
                    // Nothing to process this iteration
                    continue;
                }

                // Card 16: Select active slot based on rotation policy
                // For DISABLED: always slot 0; for MANUAL/AUTOMATIC: use active_epoch
                // Read ONCE for entire batch to preserve rotation semantics
                uint32_t slot_idx = engine_select_active_slot(state);
                Keyslot& slot = state->keyslots[slot_idx];

                // Card 18+19: Pre-signing integrity check with self-healing
                // If slot is corrupted and we're in STRICT mode, try self-heal first
                bool skip_batch = false;
                if (!is_slot_usable(&slot, tuning.integrity_mode)) {
                    // In STRICT mode, slot has blocking fault flags
                    // Card 19: Try self-healing if enabled
                    if (check_repair_or_heal_keyslot(state, slot_idx)) {
                        // Self-heal succeeded, slot is now usable
                        skip_batch = false;
                    } else {
                        // Self-heal failed or not available
                        skip_batch = true;
                    }
                }

                // Process all requests in this batch (or skip if integrity failed)
                for (uint32_t i = 0; i < batch; ++i) {
                    uint32_t idx = (tail + i) % ENGINE_QUEUE_SIZE;

                    // Consume the request at idx
                    Request req = state->requests[idx];

                    // Prepare response
                    EngineResponse resp{};
                    resp.request_id = req.request_id;
                    resp.reserved[0] = 0;
                    resp.reserved[1] = 0;
                    resp.reserved[2] = 0;

                    // Card 18: If integrity check failed, return error for all requests
                    if (skip_batch) {
                        resp.status = ENGINE_RESP_INTEGRITY_ERROR;
                        state->responses[idx] = resp;
                        __threadfence_system();  // System fence for host-mapped memory
                        // Card 24.5: Use helper for mapped/legacy write
                        engine_set_response_ready(state, idx);
                        continue;
                    }

#if ENGINE_USE_REAL_P256
                    // Card 08+12+14+15: Real P-256 ECDSA signing
                    // Card 12: Use RFC 6979 nonce if provided, else fall back to counter
                    // Card 14: Pass low_s_enabled from tuning for canonical signatures
                    // Card 15: Support random nonce mode with CSPRNG
                    uint32_t nonce_counter = atomicAdd(&g_nonce_counter, 1);

                    const uint8_t* ext_nonce = nullptr;
                    if (req.has_nonce) {
                        ext_nonce = req.nonce_k;  // Use RFC 6979 nonce from host
                    }

                    // Card 14: Convert tuning.low_s_enabled (uint32_t) to bool
                    bool low_s = (tuning.low_s_enabled != 0);

                    // Card 15: Determine if using random nonce mode
                    bool use_random = (tuning.nonce_mode == NonceMode::RANDOM);

                    // Card 15: Get unique stream ID for CSPRNG
                    // Combines global counter with request index for uniqueness
                    uint64_t rng_stream_id = 0;
                    if (use_random) {
                        // Atomically increment the global RNG counter
                        rng_stream_id = atomicAdd(
                            const_cast<unsigned long long*>(
                                reinterpret_cast<volatile unsigned long long*>(&state->rng_counter)
                            ),
                            1ull
                        );
                    }

                    bool valid = smoke::p256::p256_sign_persistent(
                        resp.sig_r,
                        resp.sig_s,
                        req.input,
                        slot.private_d,
                        nonce_counter,
                        ext_nonce,
                        low_s,
                        use_random,
                        tuning.rng_seed,
                        rng_stream_id
                    );

                    resp.status = valid ? ENGINE_RESP_OK : ENGINE_RESP_ERROR;
#else
                    // Stub mode (Cards 05-07): XOR-based deterministic signature
                    // sig_r = input XOR private_d
                    // sig_s = private_d
                    for (int j = 0; j < 32; ++j) {
                        resp.sig_r[j] = req.input[j] ^ slot.private_d[j];
                    }
                    for (int j = 0; j < 32; ++j) {
                        resp.sig_s[j] = slot.private_d[j];
                    }
                    resp.status = ENGINE_RESP_OK;
#endif

                    // Write response and mark ready
                    state->responses[idx] = resp;
                    __threadfence_system();  // System fence for host-mapped memory
                    // Card 24.5: Use helper for mapped/legacy write
                    engine_set_response_ready(state, idx);
                }

                // After batch done: advance consumer and response producer indices
                state->req_tail  = tail + batch;
                state->resp_head = state->resp_head + batch;

#if ENGINE_MICROBATCH_DEBUG
                // Card 10: Update telemetry counters. atomicAdd has no
                // uint64_t overload on LP64 Linux (uint64_t = unsigned
                // long there); cast to unsigned long long — identical
                // type on Windows, same 64-bit representation on both.
                atomicAdd(reinterpret_cast<unsigned long long*>(&state->microbatch_debug.total_batches), 1ull);
                atomicAdd(reinterpret_cast<unsigned long long*>(&state->microbatch_debug.total_requests), (unsigned long long)batch);

                if (batch <= 4u) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(&state->microbatch_debug.small_batches), 1ull);
                } else if (batch <= 32u) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(&state->microbatch_debug.medium_batches), 1ull);
                } else {
                    atomicAdd(reinterpret_cast<unsigned long long*>(&state->microbatch_debug.large_batches), 1ull);
                }
#endif

                // Card 17: Update total sign ops counter
                atomicAdd((unsigned long long*)&state->total_sign_ops, (unsigned long long)batch);
                // Also update integrity_status copy for host visibility
                atomicAdd((unsigned long long*)&state->integrity_status.total_sign_ops, (unsigned long long)batch);

                // Card 18+19: Background integrity scan trigger with self-healing
                // Run scan every scan_interval_ops if enabled
                uint32_t scan_interval = tuning.integrity_scan_interval_ops;
                if (tuning.integrity_mode != IntegrityMode::INTEGRITY_DISABLED &&
                    scan_interval > 0)
                {
                    uint64_t ops = state->total_sign_ops;
                    uint64_t last_scan = state->integrity_status.last_scan_op_count;

                    // Check if enough ops have passed since last scan
                    if (ops >= last_scan + scan_interval) {
                        // Card 19: Use self-healing-aware scan instead of basic scan
                        // Scan all slots with repair+heal capability
                        for (uint32_t scan_slot = 0; scan_slot < ENGINE_MAX_KEY_SLOTS; ++scan_slot) {
                            check_repair_or_heal_keyslot(state, scan_slot);
                        }
                        // Update last scan count
                        state->integrity_status.last_scan_op_count = state->total_sign_ops;
                    }
                }
            }
        }

        __syncthreads();  // keep warp aligned for now
        engine_device_yield();
    }
}

// Host-callable wrapper to launch the persistent kernel.
// Called from engine_host.cpp via forward declaration in smoke::engine namespace.
// Card 26.77: Minimal implementation - no debug output to avoid stack usage from fprintf.
cudaError_t launch_p256_engine_loop(EngineState* d_state, cudaStream_t stream) {
    // Launch persistent kernel with 1 block of 32 threads (1 warp)
    p256_engine_loop<<<1, 32, 0, stream>>>(d_state);
    return cudaGetLastError();
}

// ============================================================================
// Card 25.1: FAST PATH Persistent Kernel
// ============================================================================
// This kernel reads from host-mapped memory (no per-batch cudaMemcpy).
// Key differences from legacy p256_engine_loop:
// - Reads from FastPathRing (mapped memory, host writes directly)
// - Writes to FastPathResponseBuffer (device memory, host batch-fetches)
// - Kernel decides batch size (device-driven batching)
// - Updates FastPathTelemetry to prove persistence

// Optimize-6.4: __launch_bounds__ helps compiler optimize register allocation
// Card 26.4 REVERT: Back to 256 threads (128 caused slowdown)
// Card 26.2: Multi-shard version - each CTA owns its shard (zero contention)
// Card 26.24: Added payload_slab for extended multi-message
// Card 26.25: Added output_slab for extended multi-message outputs
__global__ __launch_bounds__(256) void fast_path_engine_loop_sharded(
    FastPathRing** ring_ptrs,           // Card 26.2: Array of ring pointers (one per shard)
    FastPathResponseBuffer** resp_ptrs, // Card 26.2: Array of response buffer pointers
    FastPathTelemetry* telemetry,
    Keyslot* keyslots,
    volatile uint32_t* active_epoch,
    volatile bool* shutdown,
    uint32_t num_shards,                 // Card 26.2: Number of shards
    const uint8_t* payload_slab,         // Card 26.24: Extended payload buffer (mapped)
    uint8_t* output_slab,                // Card 26.25: Extended output buffer (mapped, writable)
    RSAKeyslot* rsa_keyslots,            // Card 26.70: RSA-2048 CRT keyslots
    MLDSAKeyslot* mldsa_keyslots         // Card 27.14: ML-DSA keyslots
);

#ifdef ENABLE_LEGACY_KERNEL  // Dead code — sharded kernel is production. Removing saves ~2500 lines of register pressure.
// Optimize-6.4: __launch_bounds__ helps compiler optimize register allocation
// Card 26.4 REVERT: Back to 256 threads (128 caused slowdown)
// Card 26.24: Added payload_slab for extended multi-message
// Card 26.25: Added output_slab for extended multi-message outputs
__global__ __launch_bounds__(256) void fast_path_engine_loop(
    volatile FastPathRing* ring,
    volatile FastPathResponseBuffer* responses,
    FastPathTelemetry* telemetry,
    Keyslot* keyslots,
    volatile uint32_t* active_epoch,
    volatile bool* shutdown,
    const uint8_t* payload_slab,  // Card 26.24: Extended payload buffer (mapped)
    uint8_t* output_slab,         // Card 26.25: Extended output buffer (mapped, writable)
    RSAKeyslot* rsa_keyslots,     // Card 26.70: RSA-2048 CRT keyslots
    MLDSAKeyslot* mldsa_keyslots  // Card 27.14: ML-DSA keyslots
) {
    // DEBUG: Write magic marker immediately to prove kernel started
    // This happens before ANY other code to catch early crashes
    if (threadIdx.x == 0) {
        telemetry->kernel_start_time_us = 0xDEADBEEF;  // Magic marker
        __threadfence_system();  // Ensure visible to host
    }
    __syncthreads();

    // Shared memory for batch processing
    __shared__ uint32_t s_batch_count;
    __shared__ uint32_t s_tail;
    __shared__ uint64_t s_dequeue_cycles;  // Card 25.2C: Dequeue timestamp for batch
    __shared__ FastPathRequest s_batch[FAST_PATH_BATCH_MAX];

    // Card 26.15: Shared cycle timing for authoritative metrics (legacy kernel stub)
    __shared__ uint64_t s_iter_start_cycles;
    __shared__ uint64_t s_compute_start_cycles;
    (void)s_iter_start_cycles;   // Suppress unused warnings in legacy kernel
    (void)s_compute_start_cycles;

#if P256_WARP_COOP_ENABLED
    // Card 26.6: Shared memory for batched Z-inversion
    // State arrays hold intermediate signing state between phases
    __shared__ smoke::p256::warp_coop::WarpCoopSignState s_sign_state[FAST_PATH_BATCH_MAX];
    // Workspace arrays for Card14 batch inversion (8 limbs per element)
    __shared__ uint32_t s_Z_workspace[FAST_PATH_BATCH_MAX * 8];     // Z coordinates
    __shared__ uint32_t s_Zinv_workspace[FAST_PATH_BATCH_MAX * 8];  // Inverted Z
    __shared__ uint32_t s_P_workspace[FAST_PATH_BATCH_MAX * 8];     // Prefix products
#endif

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % 32;
    const uint32_t warp_id = tid / 32;    // Card 26.7: Warp index within CTA
    const uint32_t num_warps = blockDim.x / 32;  // Card 26.7: Number of warps in CTA
    const uint32_t cta_id = blockIdx.x;  // Card 26.1B: CTA index for multi-CTA

    // Initialize telemetry on kernel start (CTA 0, lane 0 only)
    // Card 26.1B: Only first CTA initializes telemetry to avoid races
    if (cta_id == 0 && lane == 0) {
        telemetry->kernel_launches = 1;
        telemetry->total_requests_seen = 0;
        telemetry->total_batches_processed = 0;
        telemetry->total_batch_size_sum = 0;
        telemetry->kernel_idle_cycles = 0;
        telemetry->kernel_busy_cycles = 0;
        telemetry->kernel_start_time_us = 0;  // Can't get time on GPU easily

        // Card 26.1: Shape truth fields - prove which signer is ACTUALLY engaged
#if P256_WARP_COOP_ENABLED
        telemetry->signer_model = static_cast<uint32_t>(SignerModel::WARP_COOP);
        telemetry->threads_per_signature = 32;  // Warp-cooperative: 32 threads per sig
#else
        telemetry->signer_model = static_cast<uint32_t>(SignerModel::THREAD_ONLY);
        telemetry->threads_per_signature = 1;   // Thread-only: 1 thread per sig
#endif
        // Card 26.1B: service_lanes = number of CTAs (gridDim.x)
        telemetry->service_lanes = gridDim.x;
        // NOTE: comb_table_loaded is set by host via fast_path_set_comb_table_loaded() - don't reset here

        // Card 26.1B: Zero out per-CTA distribution counters
        for (uint32_t i = 0; i < FastPathTelemetry::MAX_SERVICE_LANES; i++) {
            telemetry->per_cta_batches[i] = 0;
            telemetry->per_cta_requests[i] = 0;
            telemetry->per_cta_attempts[i] = 0;  // Card 26.2: Per-CTA dequeue attempts
        }

        // Card 26.2: Initialize debug beacons to 0
        telemetry->dbg_enter_kernel = 0;
        telemetry->dbg_dequeue_attempts = 0;
        telemetry->dbg_claim_success = 0;
        telemetry->dbg_batch_begin = 0;
        telemetry->dbg_sign_begin = 0;
        telemetry->dbg_sign_done = 0;
        telemetry->dbg_write_response = 0;
        telemetry->dbg_loop_heartbeat = 0;

        // Card 26.5: Initialize inversion census counters
        telemetry->dbg_inv_fermat_calls = 0;
        telemetry->dbg_inv_card14_batches = 0;
        telemetry->dbg_j2a_calls = 0;
        telemetry->dbg_scalar_mul_calls = 0;

        // Card 26.5: Initialize stage cycle counters
        telemetry->cyc_dequeue = 0;
        telemetry->cyc_scalar_mul = 0;
        telemetry->cyc_affine = 0;
        telemetry->cyc_sign_finalize = 0;
        telemetry->cyc_enqueue_resp = 0;

        // Card 26.10: Initialize overflow detection counters
        telemetry->response_overwrites = 0;
        telemetry->response_seq_counter = 0;

        // Card 26.12: Initialize CQ sizing (default to max if not set by host)
        // cq_capacity_active may have been set by fast_path_set_cq_capacity() before start
        // Only initialize if it's 0 (not set by host)
        if (telemetry->cq_capacity_active == 0) {
            telemetry->cq_capacity_active = FAST_PATH_RESPONSE_BUFFER;
        }
        telemetry->cq_capacity_max = FAST_PATH_RESPONSE_BUFFER;

        // Card 26.5: Set global telemetry pointer for nested function instrumentation
        g_telemetry_ptr = telemetry;
    }

    // Card 26.2: Signal kernel entry (after telemetry init)
    if (cta_id == 0 && lane == 0) {
        telemetry->dbg_enter_kernel = 1;
        __threadfence_system();
    }
    // Card 26.1B: All CTAs must wait for CTA 0 to initialize telemetry
    __syncthreads();
    if (cta_id == 0) {
        __threadfence();  // Ensure telemetry writes visible to all CTAs
    }

    // Main persistent loop
    while (true) {
        // Check shutdown flag
        if (*shutdown) {
            return;
        }

        // Card 26.2 FIX: Only thread 0 of each CTA dequeues work from the ring
        // BUG FIX: "lane == 0" matches one per warp (4 threads with 128/block, 8 with 256)
        // This caused a race on s_batch_count. Now only tid == 0 handles dequeue.
        if (tid == 0) {
            s_batch_count = 0;  // Default: no work

            // Card 26.2: Heartbeat every 1000 iterations (CTA 0 only)
            if (cta_id == 0) {
                static __shared__ uint32_t heartbeat_counter;
                heartbeat_counter = (heartbeat_counter + 1) % 1000;
                if (heartbeat_counter == 0) {
                    atomicAdd((unsigned long long*)&telemetry->dbg_loop_heartbeat, 1ull);
                }
            }

            // Card 26.2: Count dequeue attempts (global and per-CTA)
            atomicAdd((unsigned long long*)&telemetry->dbg_dequeue_attempts, 1ull);
            if (cta_id < FastPathTelemetry::MAX_SERVICE_LANES) {
                atomicAdd((unsigned long long*)&telemetry->per_cta_attempts[cta_id], 1ull);
            }

            // Check response buffer space first
            // Card 26.12: Use runtime CQ capacity instead of compile-time constant
            uint32_t cq_capacity = telemetry->cq_capacity_active;
            uint32_t resp_used = responses->write_idx - responses->read_idx;
            uint32_t resp_space = cq_capacity - resp_used;
            if (resp_space == 0) {
                atomicAdd((unsigned long long*)&telemetry->kernel_idle_cycles, 1ull);
            } else {
                // Try to atomically claim work from the ring
                // Use CAS loop to handle contention from other CTAs
                // Card 26.2 FIX: Volatile reads (pointers are volatile)
                // Card 26.34 FIX: Ring protocol - host writes to tail, kernel reads from head
                // available = tail - head (producer - consumer), NOT head - tail
                uint32_t old_head = ring->head;
                uint32_t tail = ring->tail;
                uint32_t available = tail - old_head;

                // Card 26.2 FIX: Conditional fence - only when appears idle
                if (available == 0) {
                    __threadfence_system();
                    old_head = ring->head;
                    tail = ring->tail;
                    available = tail - old_head;
                    if (available == 0) {
                        __nanosleep(64);
                    }
                }

                if (available > 0) {
                    // Calculate batch size (limited by available, max batch, and response space)
                    uint32_t batch = min(available, (uint32_t)FAST_PATH_BATCH_MAX);
                    batch = min(batch, resp_space);

                    // Card 26.1B: Atomically claim [old_head, old_head + batch)
                    // Kernel (consumer) advances head, host (producer) advances tail
                    uint32_t expected = old_head;
                    uint32_t desired = old_head + batch;
                    uint32_t actual = atomicCAS((uint32_t*)&ring->head, expected, desired);

                    if (actual == expected) {
                        // Successfully claimed the batch
                        s_batch_count = batch;
                        s_tail = old_head;  // Note: s_tail is poorly named, this is the start read position

                        // Card 26.2: Signal successful claim
                        atomicAdd((unsigned long long*)&telemetry->dbg_claim_success, 1ull);

                        // Card 25.2C: Record dequeue timestamp (clock64 cycles)
                        s_dequeue_cycles = clock64();

                        // Pull requests into shared memory
                        // Card 26.2 FIX: Explicit copy from volatile ring to shared mem
                        // Card 26.34 FIX: Read from old_head (consumer position), not tail
                        #if ((FAST_PATH_RING_SIZE & (FAST_PATH_RING_SIZE - 1)) == 0)
                        constexpr uint32_t ring_mask = FAST_PATH_RING_SIZE - 1;
                        for (uint32_t i = 0; i < batch; i++) {
                            uint32_t slot = (old_head + i) & ring_mask;
                            const volatile FastPathRequest* src = &ring->requests[slot];
                            s_batch[i].request_id = src->request_id;
                            s_batch[i].t_submit_us = src->t_submit_us;
                            for (int j = 0; j < 32; j++) s_batch[i].input[j] = src->input[j];
                            s_batch[i].key_slot = src->key_slot;
                            s_batch[i].flags = src->flags;
                            s_batch[i].client_id = src->client_id;
                            s_batch[i].opcode = src->opcode;         // Card 26.19: Multi-op support
                            s_batch[i].input_len = src->input_len;   // Card 26.19: Variable input length
                            s_batch[i].payload_offset = src->payload_offset;  // Card 26.24: Extended payload
                            s_batch[i].payload_len = src->payload_len;        // Card 26.24: Extended payload
                        }
                        #else
                        for (uint32_t i = 0; i < batch; i++) {
                            uint32_t slot = (old_head + i) % FAST_PATH_RING_SIZE;
                            const volatile FastPathRequest* src = &ring->requests[slot];
                            s_batch[i].request_id = src->request_id;
                            s_batch[i].t_submit_us = src->t_submit_us;
                            for (int j = 0; j < 32; j++) s_batch[i].input[j] = src->input[j];
                            s_batch[i].key_slot = src->key_slot;
                            s_batch[i].flags = src->flags;
                            s_batch[i].client_id = src->client_id;
                            s_batch[i].opcode = src->opcode;         // Card 26.19: Multi-op support
                            s_batch[i].input_len = src->input_len;   // Card 26.19: Variable input length
                            s_batch[i].payload_offset = src->payload_offset;  // Card 26.24: Extended payload
                            s_batch[i].payload_len = src->payload_len;        // Card 26.24: Extended payload
                        }
                        #endif

                        // Update telemetry
                        atomicAdd((unsigned long long*)&telemetry->total_requests_seen, (unsigned long long)batch);
                        atomicAdd((unsigned long long*)&telemetry->total_batches_processed, 1ull);
                        atomicAdd((unsigned long long*)&telemetry->total_batch_size_sum, (unsigned long long)batch);
                        atomicAdd((unsigned long long*)&telemetry->kernel_busy_cycles, 1ull);

                        // Card 26.1B: Per-CTA distribution tracking
                        if (cta_id < FastPathTelemetry::MAX_SERVICE_LANES) {
                            atomicAdd((unsigned long long*)&telemetry->per_cta_batches[cta_id], 1ull);
                            atomicAdd((unsigned long long*)&telemetry->per_cta_requests[cta_id], (unsigned long long)batch);
                        }
                    } else {
                        // Another CTA claimed the work, spin to next iteration
                        atomicAdd((unsigned long long*)&telemetry->kernel_idle_cycles, 1ull);
                    }
                } else {
                    atomicAdd((unsigned long long*)&telemetry->kernel_idle_cycles, 1ull);
                }
            }
        }
        __syncthreads();

        uint32_t batch = s_batch_count;
        uint32_t tail = s_tail;

        if (batch > 0) {
            // Card 26.2: Signal batch begin (lane 0 only)
            if (lane == 0) {
                atomicAdd((unsigned long long*)&telemetry->dbg_batch_begin, 1ull);
            }
            __syncthreads();

            // Get active keyslot
            uint32_t slot_idx = (*active_epoch) & 1u;
            Keyslot& slot = keyslots[slot_idx];

#if P256_WARP_COOP_ENABLED
            // ========================================================================
            // Card 26.7: MULTI-WARP BATCHED SIGNING PATH (with multi-op dispatch)
            // ========================================================================

            // Card 26.19: Check if batch contains non-P256 opcodes
            __shared__ uint16_t s_batch_opcode_nonshard;
            if (tid == 0) {
                s_batch_opcode_nonshard = s_batch[0].opcode;
            }
            __syncthreads();

            const uint16_t batch_opcode_nonshard = s_batch_opcode_nonshard;

            // Card 26.19/26.22: Fast path for hash operations (SHA-256) with multi-msg support
            if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::SHA256)) {
                // Batch slot allocation for hash ops
                __shared__ uint32_t s_base_write_idx_hash;
                __shared__ uint64_t s_base_seq_hash;
                __shared__ uint32_t s_total_digests;
                if (tid == 0) {
                    s_base_seq_hash = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_hash = atomicAdd((uint32_t*)&responses->write_idx, batch);
                    s_total_digests = 0;
                }
                __syncthreads();

                // Per-thread SHA-256 hashing (with Card 26.22 multi-msg support)
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;

                    // Card 26.22: Check for multi-message mode
                    const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                    uint32_t digests_this_req = 0;

                    // Card 26.25: Pre-compute response slot for potential slab write
                    uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                    uint32_t write_idx_pre = s_base_write_idx_hash + i;
                    uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                    if (is_multimsg) {
                        // Parse MultiMsgHeader from input[0..7]
                        const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                        const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                        const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                                   ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                        // Card 26.24: Determine payload source
                        // Phase 1 (payload_len == 0): Messages in input[8..31]
                        // Phase 2 (payload_len > 0): Messages in payload_slab[payload_offset..]
                        const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                        const uint8_t* payload_data = use_extended_payload
                            ? (payload_slab + req.payload_offset)
                            : (req.input + MULTIMSG_HEADER_SIZE);
                        const uint32_t payload_avail = use_extended_payload
                            ? req.payload_len
                            : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                        // Card 26.25: Calculate bytes needed for output
                        const uint32_t bytes_needed = n_msgs * 32;

                        // Validate multi-message constraints
                        bool valid = true;
                        if (n_msgs == 0) valid = false;
                        if (msg_len > MULTIMSG_MAX_MSG_LEN) valid = false;
                        if (hdr_flags != 0) valid = false;
                        if (n_msgs * msg_len > payload_avail) valid = false;
                        // Card 26.25: Check against output slab segment size (2048 bytes = 64 digests)
                        if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                        if (!valid) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                            resp.output_offset = 0;
                            resp.output_bytes = 0;
                        } else if (bytes_needed <= 64) {
                            // Card 26.25: Inline mode (N <= 2, backward compatible)
                            for (uint16_t m = 0; m < n_msgs; m++) {
                                smoke::hash::sha256_hash_short(
                                    payload_data + m * msg_len,
                                    msg_len,
                                    resp.output + m * 32
                                );
                            }
                            resp.output_len = bytes_needed;
                            resp.output_offset = 0;
                            resp.output_bytes = 0;  // Signal: use inline output[64]
                            digests_this_req = n_msgs;
                        } else {
                            // Card 26.25: Slab mode (N > 2)
                            // Each response slot owns a fixed segment in the output slab
                            // Single shared ring in this kernel: write_idx-derived slots are
                            // globally unique among in-flight responses, so slot-keyed slab
                            // segments cannot collide here (unlike the sharded kernel).
                            uint32_t slab_offset = resp_slot_pre * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                            uint8_t* slab_ptr = output_slab + slab_offset;

                            for (uint16_t m = 0; m < n_msgs; m++) {
                                smoke::hash::sha256_hash_short(
                                    payload_data + m * msg_len,
                                    msg_len,
                                    slab_ptr + m * 32
                                );
                            }

                            // Also copy first 64 bytes to inline output (convenience)
                            for (int j = 0; j < 64 && j < (int)bytes_needed; j++) {
                                resp.output[j] = slab_ptr[j];
                            }

                            resp.output_len = (bytes_needed < 64) ? bytes_needed : 64;  // Inline portion
                            resp.output_offset = slab_offset;
                            resp.output_bytes = bytes_needed;  // Signal: use slab
                            digests_this_req = n_msgs;
                        }
                    } else {
                        // Single-message mode (existing behavior)
                        if (req.input_len > 32) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            resp.output_offset = 0;
                            resp.output_bytes = 0;
                        } else {
                            smoke::hash::sha256_hash_short(req.input, req.input_len, resp.output);
                            resp.output_len = 32;
                            resp.output_offset = 0;  // Card 26.25: Inline mode
                            resp.output_bytes = 0;
                            digests_this_req = 1;
                        }
                    }

                    // Track digests for telemetry
                    if (digests_this_req > 0) {
                        atomicAdd(&s_total_digests, digests_this_req);
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_hash + i;
                    uint32_t write_idx = s_base_write_idx_hash + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Use commit protocol - write request_id LAST
                    write_response_base(dst, resp);
                    // Copy all 64 bytes for multi-message output (inline portion)
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }
                __syncthreads();

                // Telemetry updates for hash ops (Card 26.19B + 26.22)
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests);
                }
                __syncthreads();
            }
            // Card 26.31: Fast path for SHA-512 hash operations (single-message only for now)
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::SHA512)) {
                // Batch slot allocation for hash ops
                __shared__ uint32_t s_base_write_idx_sha512;
                __shared__ uint64_t s_base_seq_sha512;
                if (tid == 0) {
                    s_base_seq_sha512 = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_sha512 = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread SHA-512 hashing (single-message mode)
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    // Single-message mode only (no multi-msg support for SHA-512 yet)
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::sha512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                    }

                    // Record completion timestamp
                    resp.t_complete_us = clock64();

                    // Write to response buffer
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_sha512 + i;
                    uint32_t write_idx = s_base_write_idx_sha512 + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Use commit protocol - write request_id LAST
                    write_response_base(dst, resp);
                    // Copy 64-byte output
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }

                // Telemetry updates for SHA-512 ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                }
                __syncthreads();
            }
            // Card 26.32: Fast path for SHA3-256 hash operations (single-message only)
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::SHA3_256)) {
                // Batch slot allocation for hash ops
                __shared__ uint32_t s_base_write_idx_sha3;
                __shared__ uint64_t s_base_seq_sha3;
                if (tid == 0) {
                    s_base_seq_sha3 = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_sha3 = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread SHA3-256 hashing (single-message mode)
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    // Single-message mode only
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::sha3_256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                    }

                    // Record completion timestamp
                    resp.t_complete_us = clock64();

                    // Write to response buffer
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_sha3 + i;
                    uint32_t write_idx = s_base_write_idx_sha3 + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Use commit protocol - write request_id LAST
                    write_response_base(dst, resp);
                    // Copy 32-byte output
                    for (int j = 0; j < 32; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }

                // Telemetry updates for SHA3-256 ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                }
                __syncthreads();
            }
            // Card 26.33: Fast path for BLAKE2b-256 hash operations (single-message only)
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::BLAKE2B_256)) {
                // Batch slot allocation for hash ops
                __shared__ uint32_t s_base_write_idx_blake2b;
                __shared__ uint64_t s_base_seq_blake2b;
                if (tid == 0) {
                    s_base_seq_blake2b = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_blake2b = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread BLAKE2b-256 hashing (single-message mode)
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    // Single-message mode only
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::blake2b_256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                    }

                    // Record completion timestamp
                    resp.t_complete_us = clock64();

                    // Write to response buffer
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_blake2b + i;
                    uint32_t write_idx = s_base_write_idx_blake2b + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Use commit protocol - write request_id LAST
                    write_response_base(dst, resp);
                    // Copy 32-byte output
                    for (int j = 0; j < 32; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }

                // Telemetry updates for BLAKE2b-256 ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                }
                __syncthreads();
            }
            // Card 140: Fast path for SHA3-512 hash operations (single-message only)
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::SHA3_512)) {
                __shared__ uint32_t s_base_write_idx_sha3_512;
                __shared__ uint64_t s_base_seq_sha3_512;
                if (tid == 0) {
                    s_base_seq_sha3_512 = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_sha3_512 = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::sha3_512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                    }

                    resp.t_complete_us = clock64();

                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_sha3_512 + i;
                    uint32_t write_idx = s_base_write_idx_sha3_512 + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    write_response_base(dst, resp);
                    // Copy 64-byte output
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }

                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                }
                __syncthreads();
            }
            // Card 141: Fast path for BLAKE2b-512 hash operations (single-message only)
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::BLAKE2B_512)) {
                __shared__ uint32_t s_base_write_idx_blake2b_512;
                __shared__ uint64_t s_base_seq_blake2b_512;
                if (tid == 0) {
                    s_base_seq_blake2b_512 = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_blake2b_512 = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::blake2b_512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                    }

                    resp.t_complete_us = clock64();

                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_blake2b_512 + i;
                    uint32_t write_idx = s_base_write_idx_blake2b_512 + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    write_response_base(dst, resp);
                    // Copy 64-byte output
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }

                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard != 0
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::P256_SIGN)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::RSA_2048_SIGN)  // Card 26.70
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::SHA256)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::SHA512)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::SHA3_256)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::BLAKE2B_256)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::SHA3_512)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::BLAKE2B_512)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::HMAC_SHA256)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::HMAC_SHA512)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::HMAC_SHA384)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::AES256_CTR)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::AES256_GCM)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::CHACHA20)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::POLY1305)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::CHACHA20_POLY1305)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::HKDF_SHA256)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::AES256_GCM_OPEN)
                     && batch_opcode_nonshard != static_cast<uint16_t>(OpCode::CHACHA20_POLY1305_OPEN)) {
                // Unknown opcode - return INVALID_OPCODE for all
                __shared__ uint32_t s_base_write_idx_err;
                __shared__ uint64_t s_base_seq_err;
                if (tid == 0) {
                    s_base_seq_err = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_err = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.t_complete_us = clock64();
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_OPCODE);
                    resp.client_id = req.client_id;

                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_err + i;
                    uint32_t write_idx = s_base_write_idx_err + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write base fields first (commit protocol)
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->epoch_used = resp.epoch_used;    // Card 66
                    dst->slot_used = resp.slot_used;      // Card 66
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }

                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->outcome_gpu_error, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::HMAC_SHA256)) {
                // Card 26.29: Fast path for HMAC-SHA256 keyed hash operations (with multi-msg support)
                __shared__ uint32_t s_base_write_idx_hmac;
                __shared__ uint64_t s_base_seq_hmac;
                __shared__ uint32_t s_total_digests_hmac;
                if (tid == 0) {
                    s_base_seq_hmac = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_hmac = atomicAdd((uint32_t*)&responses->write_idx, batch);
                    s_total_digests_hmac = 0;
                }
                __syncthreads();

                // Per-thread HMAC-SHA256 hashing (with multi-msg support)
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;

                    // Check for multi-message mode
                    const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                    uint32_t digests_this_req = 0;

                    // Pre-compute response slot for potential slab write
                    uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                    uint32_t write_idx_pre = s_base_write_idx_hmac + i;
                    uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                    if (is_multimsg) {
                        // Parse MultiMsgHeader from input[0..7]
                        const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                        const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                        const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                                   ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                        // Determine payload source
                        const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                        const uint8_t* payload_data = use_extended_payload
                            ? (payload_slab + req.payload_offset)
                            : (req.input + MULTIMSG_HEADER_SIZE);
                        const uint32_t payload_avail = use_extended_payload
                            ? req.payload_len
                            : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                        // Calculate bytes needed for output
                        const uint32_t bytes_needed = n_msgs * 32;

                        // Validate multi-message constraints
                        bool valid = true;
                        if (n_msgs == 0) valid = false;
                        if (msg_len > HMAC_SHA256_MAX_MSG_LEN) valid = false;
                        if (hdr_flags != 0) valid = false;
                        if (n_msgs * msg_len > payload_avail) valid = false;
                        if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                        if (!valid) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                            resp.output_offset = 0;
                            resp.output_bytes = 0;
                        } else if (bytes_needed <= 64) {
                            // Inline mode (N <= 2)
                            for (uint16_t m = 0; m < n_msgs; m++) {
                                hmac_sha256_device(
                                    slot.private_d,            // shared key from keyslot
                                    32,                        // key_len
                                    payload_data + m * msg_len,
                                    msg_len,
                                    resp.output + m * 32
                                );
                            }
                            resp.output_len = bytes_needed;
                            resp.output_offset = 0;
                            resp.output_bytes = 0;
                            digests_this_req = n_msgs;
                        } else {
                            // Slab mode (N > 2)
                            // Single shared ring in this kernel: write_idx-derived slots are
                            // globally unique among in-flight responses, so slot-keyed slab
                            // segments cannot collide here (unlike the sharded kernel).
                            uint32_t slab_offset = resp_slot_pre * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                            uint8_t* slab_ptr = output_slab + slab_offset;

                            for (uint16_t m = 0; m < n_msgs; m++) {
                                hmac_sha256_device(
                                    slot.private_d,
                                    32,
                                    payload_data + m * msg_len,
                                    msg_len,
                                    slab_ptr + m * 32
                                );
                            }

                            // Copy first 64 bytes to inline output
                            for (int j = 0; j < 64 && j < (int)bytes_needed; j++) {
                                resp.output[j] = slab_ptr[j];
                            }

                            resp.output_len = (bytes_needed < 64) ? bytes_needed : 64;
                            resp.output_offset = slab_offset;
                            resp.output_bytes = bytes_needed;
                            digests_this_req = n_msgs;
                        }
                    } else {
                        // Single-message mode
                        if (req.input_len > HMAC_SHA256_MAX_MSG_LEN) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            resp.output_offset = 0;
                            resp.output_bytes = 0;
                        } else {
                            hmac_sha256_device(
                                slot.private_d,
                                32,
                                req.input,
                                req.input_len,
                                resp.output
                            );
                            resp.output_len = 32;
                            resp.output_offset = 0;
                            resp.output_bytes = 0;
                            digests_this_req = 1;
                        }
                    }

                    // Track digests for telemetry
                    if (digests_this_req > 0) {
                        atomicAdd(&s_total_digests_hmac, digests_this_req);
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_hmac + i;
                    uint32_t write_idx = s_base_write_idx_hmac + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write all fields except request_id first
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->epoch_used = resp.epoch_used;    // Card 66
                    dst->slot_used = resp.slot_used;      // Card 66
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                // Telemetry updates for HMAC ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_hmac);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::HMAC_SHA512)) {
                // Card 26.51: Fast path for HMAC-SHA512 keyed hash operations
                __shared__ uint32_t s_base_write_idx_hmac512;
                __shared__ uint64_t s_base_seq_hmac512;
                if (tid == 0) {
                    s_base_seq_hmac512 = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_hmac512 = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread HMAC-SHA512 hashing (single message mode only for now)
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    // Single-message mode (no multi-msg support yet for HMAC-512)
                    if (req.input_len > HMAC_SHA512_MAX_MSG_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        // Use shared slot.private_d (same as HMAC-SHA256)
                        hmac_sha512_device(
                            slot.private_d,     // key (32 bytes from keyslot, will be padded)
                            32,                 // key_len
                            req.input,          // message
                            req.input_len,      // message length
                            resp.output         // 64-byte output
                        );
                        resp.output_len = 64;
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_hmac512 + i;
                    uint32_t write_idx = s_base_write_idx_hmac512 + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write all fields except request_id first
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->epoch_used = resp.epoch_used;    // Card 66
                    dst->slot_used = resp.slot_used;      // Card 66
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                // Telemetry updates for HMAC-512 ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::HMAC_SHA384)) {
                // Card 26.52: Fast path for HMAC-SHA384 keyed hash operations
                __shared__ uint32_t s_base_write_idx_hmac384;
                __shared__ uint64_t s_base_seq_hmac384;
                if (tid == 0) {
                    s_base_seq_hmac384 = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_hmac384 = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread HMAC-SHA384 hashing (single message mode only)
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    // Single-message mode
                    if (req.input_len > HMAC_SHA384_MAX_MSG_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        // Use shared slot.private_d (same pattern as HMAC-SHA256/512)
                        hmac_sha384_device(
                            slot.private_d,     // key (32 bytes from keyslot, will be padded)
                            32,                 // key_len
                            req.input,          // message
                            req.input_len,      // message length
                            resp.output         // 48-byte output
                        );
                        resp.output_len = 48;
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_hmac384 + i;
                    uint32_t write_idx = s_base_write_idx_hmac384 + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Use commit protocol - write request_id LAST
                    write_response_base(dst, resp);
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }
                __syncthreads();

                // Telemetry updates for HMAC-384 ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::AES256_CTR)) {
                // Card 26.54: Fast path for AES-256-CTR stream cipher
                __shared__ uint32_t s_base_write_idx_aes;
                __shared__ uint64_t s_base_seq_aes;
                if (tid == 0) {
                    s_base_seq_aes = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_aes = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread AES-256-CTR encryption
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    // Validate input: must have at least nonce (12 bytes)
                    if (req.input_len < AES256_CTR_NONCE_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (req.input_len > 32) {
                        // Max inline input is 32 bytes (nonce + up to 20 bytes plaintext)
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        // Card 26.64: Use precomputed round keys from keyslot
                        smoke::aes::aes256_ctr_encrypt_short_cached(
                            slot.aes_roundkeys, // precomputed round keys
                            req.input,          // nonce[12] + plaintext[N]
                            req.input_len,      // total input length
                            resp.output         // ciphertext output
                        );
                        resp.output_len = req.input_len - AES256_CTR_NONCE_LEN;  // Ciphertext length = plaintext length
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_aes + i;
                    uint32_t write_idx = s_base_write_idx_aes + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write all fields except request_id first
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->epoch_used = resp.epoch_used;    // Card 66
                    dst->slot_used = resp.slot_used;      // Card 66
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                // Telemetry updates for AES-256-CTR ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::AES256_GCM)) {
                // Card 26.65: Fast path for AES-256-GCM authenticated encryption
                __shared__ uint32_t s_base_write_idx_gcm;
                __shared__ uint64_t s_base_seq_gcm;
                if (tid == 0) {
                    s_base_seq_gcm = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_gcm = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread AES-256-GCM encryption
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.opcode = req.opcode;
                    resp.epoch_used = 0xFF;
                    resp.slot_used = req.key_slot;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    resp.status = validate_aead_keyslot(keyslots, req.key_slot);
                    if (resp.status == 0 && req.payload_len > 0) {
                        // Slab input contains plaintext; inline input is nonce only.
                        if (payload_slab == nullptr || output_slab == nullptr) {
                            resp.status = static_cast<uint8_t>(OpStatus::INTERNAL_ERROR);
                        } else if (req.input_len != 12 ||
                                   !valid_payload_slab_range(
                                       req.payload_offset, req.payload_len)) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        } else {
                            const uint32_t write_pos = s_base_write_idx_gcm + i;
                            const uint32_t lag = write_pos - responses->read_idx;
                            if (lag >= FAST_PATH_SEAL_OUTPUT_SEGMENTS) {
                                // Do not write into a segment that may still be
                                // owned by an unread completion.
                                resp.status = static_cast<uint8_t>(OpStatus::SLAB_OVERRUN);
                            } else {
                                const uint32_t seal_seg =
                                    write_pos % FAST_PATH_SEAL_OUTPUT_SEGMENTS;
                                const uint32_t out_offset =
                                    FAST_PATH_SEAL_OUTPUT_REGION_OFFSET +
                                    seal_seg * FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES;
                                uint32_t out_len = 0;
                                const Keyslot& ks = keyslots[req.key_slot];
                                smoke::aes::aes256_gcm_seal_slab(
                                    ks.aes_roundkeys, req.input,
                                    payload_slab, req.payload_offset, req.payload_len,
                                    output_slab, out_offset, &out_len);
                                if (out_len != req.payload_len + FAST_PATH_SEAL_TAG_BYTES) {
                                    resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                                } else {
                                    resp.output_offset = out_offset;
                                    resp.output_bytes = out_len;
                                }
                            }
                        }
                    } else if (resp.status == 0 &&
                               (req.input_len < 12 || req.input_len > 28)) {
                        // Nonce must be present; oversized inline plaintext fails.
                        // input_len == 12 is a 0-byte plaintext seal (output = tag
                        // only) — valid per D-089 and op_schema min_input_len 12.
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (resp.status == 0) {
                        uint16_t out_len = 0;
                        const Keyslot& ks = keyslots[req.key_slot];
                        smoke::aes::aes256_gcm_seal_short(
                            ks.aes_roundkeys,
                            req.input,
                            req.input_len,
                            resp.output,
                            &out_len
                        );
                        resp.output_len = out_len;
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_gcm + i;
                    uint32_t write_idx = s_base_write_idx_gcm + i;
                    uint32_t resp_slot = write_idx % cq_cap;

                    resp.seq = seq;
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write all fields except request_id first
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->opcode = resp.opcode;
                    dst->epoch_used = resp.epoch_used;
                    dst->slot_used = resp.slot_used;
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                // Telemetry updates for AES-256-GCM ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                }
                __syncthreads();
            }
            // DECRYPT-ABI-01: AES-256-GCM Open (decrypt) — non-shard path
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::AES256_GCM_OPEN)) {
                __shared__ uint32_t s_base_write_idx_gcm_open;
                __shared__ uint64_t s_base_seq_gcm_open;
                if (tid == 0) {
                    s_base_seq_gcm_open = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_gcm_open = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;
                    resp.status = validate_aead_keyslot(keyslots, req.key_slot);

                    // DECRYPT-ABI-01 + DECRYPT-SLAB-01: AES-256-GCM Open dispatch
                    // Inline: nonce[12] || ciphertext[N] || tag[16] in input[32]
                    // Slab: nonce[12] in input, ciphertext[N] || tag[16] in payload_slab
                    if (resp.status == 0 && req.payload_len > 0) {
                        // SLAB PATH: large payloads
                        if (payload_slab == nullptr || output_slab == nullptr) {
                            resp.status = static_cast<uint8_t>(OpStatus::INTERNAL_ERROR);
                        } else if (req.input_len != 12 ||
                                   !valid_open_payload_slab_range(
                                       req.payload_offset, req.payload_len)) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        } else {
                            const uint32_t write_pos = s_base_write_idx_gcm_open + i;
                            const uint32_t lag = write_pos - responses->read_idx;
                            if (lag >= FAST_PATH_SEAL_OUTPUT_SEGMENTS) {
                                resp.status = static_cast<uint8_t>(OpStatus::SLAB_OVERRUN);
                            } else {
                                uint32_t out_len = 0;
                                uint8_t auth_ok = 0;
                                const uint32_t resp_slot_idx =
                                    write_pos % FAST_PATH_SEAL_OUTPUT_SEGMENTS;
                                const uint32_t out_offset =
                                    FAST_PATH_SEAL_OUTPUT_REGION_OFFSET +
                                    resp_slot_idx * FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES;
                                const Keyslot& slot = keyslots[req.key_slot];
                                smoke::aes::aes256_gcm_open_slab(
                                    slot.aes_roundkeys,
                                    req.input,
                                    payload_slab,
                                    req.payload_offset,
                                    req.payload_len,
                                    output_slab,
                                    out_offset,
                                    &out_len,
                                    &auth_ok
                                );
                                if (auth_ok &&
                                    out_len <= FAST_PATH_SEAL_MAX_PLAINTEXT) {
                                    resp.output_offset = out_offset;
                                    resp.output_bytes = out_len;
                                } else {
                                    resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                                    atomicAdd((unsigned long long*)&telemetry->total_auth_failures, 1ULL);
                                }
                            }
                        }
                    } else if (resp.status == 0 && req.input_len < 28) {
                        // Empty plaintext is not a valid Capsule record.
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (resp.status == 0 && req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (resp.status == 0) {
                        // INLINE PATH: short payloads (max 4 bytes ciphertext)
                        uint16_t out_len = 0;
                        uint8_t auth_ok = 0;
                        const Keyslot& slot = keyslots[req.key_slot];
                        smoke::aes::aes256_gcm_open_short(
                            slot.aes_roundkeys,
                            req.input,
                            req.input_len,
                            resp.output,
                            &out_len,
                            &auth_ok
                        );
                        if (auth_ok) {
                            resp.status = 0;
                            resp.output_len = out_len;
                        } else {
                            resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                            resp.output_len = 0;
                            for (int z = 0; z < 64; z++) resp.output[z] = 0;
                            atomicAdd((unsigned long long*)&telemetry->total_auth_failures, 1ULL);
                        }
                    }

                    resp.t_complete_us = clock64();

                    const uint32_t cq_cap = telemetry->cq_capacity_active;
                    const uint64_t seq = s_base_seq_gcm_open + (uint64_t)i;
                    const uint32_t write_idx = s_base_write_idx_gcm_open + i;
                    const uint32_t resp_slot = write_idx % cq_cap;
                    resp.seq = seq;

                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->opcode = req.opcode;  // VERIFY-DECRYPT-STACK-01: set opcode to avoid zero-sig invariant false positive
                    dst->epoch_used = 0xFF;
                    dst->slot_used = req.key_slot;
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->total_decrypt_aes_gcm, (unsigned long long)batch);
                }
                __syncthreads();
            }
            // DECRYPT-ABI-01: ChaCha20-Poly1305 Open (decrypt) — non-shard path
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::CHACHA20_POLY1305_OPEN)) {
                __shared__ uint32_t s_base_write_idx_chacha_open;
                __shared__ uint64_t s_base_seq_chacha_open;
                if (tid == 0) {
                    s_base_seq_chacha_open = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_chacha_open = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;
                    resp.status = validate_aead_keyslot(keyslots, req.key_slot);

                    // CHACHA-OPEN-01: ChaCha20-Poly1305 decrypt dispatch

                    if (resp.status == 0 && req.payload_len > 0) {
                        // SLAB PATH
                        if (payload_slab == nullptr || output_slab == nullptr) {
                            resp.status = static_cast<uint8_t>(OpStatus::INTERNAL_ERROR);
                        } else if (req.input_len != 12 ||
                                   !valid_open_payload_slab_range(
                                       req.payload_offset, req.payload_len)) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        } else {
                            const uint32_t write_pos = s_base_write_idx_chacha_open + i;
                            const uint32_t lag = write_pos - responses->read_idx;
                            if (lag >= FAST_PATH_SEAL_OUTPUT_SEGMENTS) {
                                resp.status = static_cast<uint8_t>(OpStatus::SLAB_OVERRUN);
                            } else {
                                uint32_t out_len = 0;
                                uint8_t auth_ok = 0;
                                const uint32_t resp_slot_idx =
                                    write_pos % FAST_PATH_SEAL_OUTPUT_SEGMENTS;
                                const uint32_t out_offset =
                                    FAST_PATH_SEAL_OUTPUT_REGION_OFFSET +
                                    resp_slot_idx * FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES;
                                const Keyslot& slot_ch = keyslots[req.key_slot];
                                smoke::aead::chacha20_poly1305_decrypt_slab(
                                    slot_ch.private_d, req.input,
                                    payload_slab, req.payload_offset, req.payload_len,
                                    output_slab, out_offset, &out_len, &auth_ok
                                );
                                if (auth_ok &&
                                    out_len <= FAST_PATH_SEAL_MAX_PLAINTEXT) {
                                    resp.output_offset = out_offset;
                                    resp.output_bytes = out_len;
                                } else {
                                    resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                                    atomicAdd((unsigned long long*)&telemetry->total_auth_failures, 1ULL);
                                }
                            }
                        }
                    } else if (resp.status == 0 && req.input_len < 28) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (resp.status == 0 && req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (resp.status == 0) {
                        // INLINE PATH
                        uint16_t out_len = 0;
                        uint8_t auth_ok = 0;
                        const Keyslot& slot_ch = keyslots[req.key_slot];
                        smoke::aead::chacha20_poly1305_decrypt_short_keyslot(
                            slot_ch.private_d, req.input, req.input_len,
                            resp.output, &out_len, &auth_ok
                        );
                        if (auth_ok) {
                            resp.status = 0;
                            resp.output_len = out_len;
                        } else {
                            resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                            resp.output_len = 0;
                            for (int z = 0; z < 64; z++) resp.output[z] = 0;
                            atomicAdd((unsigned long long*)&telemetry->total_auth_failures, 1ULL);
                        }
                    }

                    resp.t_complete_us = clock64();

                    const uint32_t cq_cap = telemetry->cq_capacity_active;
                    const uint64_t seq = s_base_seq_chacha_open + (uint64_t)i;
                    const uint32_t write_idx = s_base_write_idx_chacha_open + i;
                    const uint32_t resp_slot = write_idx % cq_cap;
                    resp.seq = seq;

                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->opcode = req.opcode;  // VERIFY-DECRYPT-STACK-01: set opcode
                    dst->epoch_used = 0xFF;
                    dst->slot_used = req.key_slot;
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->total_decrypt_chacha20, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::CHACHA20)) {
                // Card 26.66: Fast path for ChaCha20 stream cipher
                __shared__ uint32_t s_base_write_idx_chacha;
                __shared__ uint64_t s_base_seq_chacha;
                if (tid == 0) {
                    s_base_seq_chacha = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_chacha = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread ChaCha20 encryption
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    // Validate input: must have at least nonce (12 bytes)
                    if (req.input_len < CHACHA20_NONCE_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (req.input_len > 32) {
                        // Max inline input is 32 bytes (nonce + up to 20 bytes plaintext)
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        // Card 26.66: ChaCha20 encrypt using key from keyslot
                        smoke::chacha::chacha20_encrypt_short_keyslot(
                            slot.private_d,     // 32-byte key
                            req.input,          // nonce[12] + plaintext[N]
                            req.input_len,      // total input length
                            resp.output,        // ciphertext output
                            &resp.output_len    // output length
                        );
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_chacha + (uint64_t)i;
                    uint32_t write_idx = s_base_write_idx_chacha + i;
                    uint32_t resp_slot = write_idx % cq_cap;
                    resp.seq = seq;

                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write all fields except request_id first
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->epoch_used = resp.epoch_used;    // Card 66
                    dst->slot_used = resp.slot_used;      // Card 66
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                // Telemetry updates for ChaCha20 ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::POLY1305)) {
                // Card 26.67: Fast path for Poly1305 MAC
                __shared__ uint32_t s_base_write_idx_poly;
                __shared__ uint64_t s_base_seq_poly;
                if (tid == 0) {
                    s_base_seq_poly = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_poly = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread Poly1305 MAC computation
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = clock64();
                    resp.client_id = req.client_id;
                    resp.status = 0;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    // Validate input: max 32 bytes message
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        // Card 26.67: Poly1305 MAC using key from keyslot
                        smoke::poly1305::poly1305_mac_short_keyslot(
                            slot.private_d,     // 32-byte key (r[16] + s[16])
                            req.input,          // message
                            req.input_len,      // message length
                            resp.output,        // 16-byte tag output
                            &resp.output_len    // output length
                        );
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_poly + (uint64_t)i;
                    uint32_t write_idx = s_base_write_idx_poly + i;
                    uint32_t resp_slot = write_idx % cq_cap;
                    resp.seq = seq;

                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write all fields except request_id first
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->epoch_used = resp.epoch_used;    // Card 66
                    dst->slot_used = resp.slot_used;      // Card 66
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                // Telemetry updates for Poly1305 ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::CHACHA20_POLY1305)) {
                // Card 26.68: Fast path for ChaCha20-Poly1305 AEAD
                __shared__ uint32_t s_base_write_idx_aead;
                __shared__ uint64_t s_base_seq_aead;
                if (tid == 0) {
                    s_base_seq_aead = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_aead = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread ChaCha20-Poly1305 AEAD computation
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = s_dequeue_cycles;
                    resp.status = 0;
                    resp.client_id = req.client_id;
                    resp.opcode = req.opcode;
                    resp.epoch_used = 0xFF;
                    resp.slot_used = req.key_slot;
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    uint16_t input_len = req.input_len;

                    resp.status = validate_aead_keyslot(keyslots, req.key_slot);
                    if (resp.status == 0 && req.payload_len > 0) {
                        if (payload_slab == nullptr || output_slab == nullptr) {
                            resp.status = static_cast<uint8_t>(OpStatus::INTERNAL_ERROR);
                        } else if (input_len != 12 ||
                                   !valid_payload_slab_range(
                                       req.payload_offset, req.payload_len)) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        } else {
                            const uint32_t write_pos = s_base_write_idx_aead + i;
                            const uint32_t lag = write_pos - responses->read_idx;
                            if (lag >= FAST_PATH_SEAL_OUTPUT_SEGMENTS) {
                                resp.status = static_cast<uint8_t>(OpStatus::SLAB_OVERRUN);
                            } else {
                                const uint32_t seal_seg =
                                    write_pos % FAST_PATH_SEAL_OUTPUT_SEGMENTS;
                                const uint32_t out_offset =
                                    FAST_PATH_SEAL_OUTPUT_REGION_OFFSET +
                                    seal_seg * FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES;
                                uint32_t out_len = 0;
                                const Keyslot& ks = keyslots[req.key_slot];
                                smoke::aead::chacha20_poly1305_encrypt_slab(
                                    ks.private_d, req.input,
                                    payload_slab, req.payload_offset, req.payload_len,
                                    output_slab, out_offset, &out_len);
                                if (out_len != req.payload_len + FAST_PATH_SEAL_TAG_BYTES) {
                                    resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                                } else {
                                    resp.output_offset = out_offset;
                                    resp.output_bytes = out_len;
                                }
                            }
                        }
                    } else if (resp.status == 0 &&
                               (input_len < 12 || input_len > 28)) {
                        // input_len == 12 is a valid 0-byte plaintext seal (D-096).
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (resp.status == 0) {
                        const Keyslot& ks = keyslots[req.key_slot];
                        smoke::aead::chacha20_poly1305_encrypt_short_keyslot(
                            ks.private_d,
                            req.input,          // nonce + plaintext
                            input_len,          // input length
                            resp.output,        // ciphertext + tag
                            &resp.output_len    // output length
                        );
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_aead + (uint64_t)i;
                    uint32_t write_idx = s_base_write_idx_aead + i;
                    uint32_t resp_slot = write_idx % cq_cap;
                    resp.seq = seq;

                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write all fields except request_id first
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->opcode = resp.opcode;
                    dst->epoch_used = resp.epoch_used;
                    dst->slot_used = resp.slot_used;
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                // Telemetry updates for ChaCha20-Poly1305 ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::HKDF_SHA256)) {
                // Card 26.69: Fast path for HKDF-SHA256 key derivation
                __shared__ uint32_t s_base_write_idx_hkdf;
                __shared__ uint64_t s_base_seq_hkdf;
                if (tid == 0) {
                    s_base_seq_hkdf = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_hkdf = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // Per-thread HKDF-SHA256 key derivation
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    const Keyslot ks = keyslots[req.key_slot];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = s_dequeue_cycles;
                    resp.status = 0;
                    resp.client_id = req.client_id;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;

                    uint16_t input_len = req.input_len;

                    // HKDF-SHA256: input = IKM (32-64 bytes), keyslot = salt
                    if (input_len < 32 || input_len > 64) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_len = 0;
                    } else {
                        // Derive 32-byte key using HKDF-SHA256
                        smoke::kdf::hkdf_sha256_keyslot(
                            ks.private_d,       // 32-byte salt
                            req.input,          // IKM
                            input_len,          // IKM length
                            resp.output,        // OKM (32 bytes)
                            &resp.output_len    // output length
                        );
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    uint32_t cq_cap = telemetry->cq_capacity_active;
                    uint64_t seq = s_base_seq_hkdf + (uint64_t)i;
                    uint32_t write_idx = s_base_write_idx_hkdf + i;
                    uint32_t resp_slot = write_idx % cq_cap;
                    resp.seq = seq;

                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write all fields except request_id first
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->epoch_used = resp.epoch_used;    // Card 66
                    dst->slot_used = resp.slot_used;      // Card 66
                    dst->output_len = resp.output_len;
                    dst->output_offset = resp.output_offset;
                    dst->output_bytes = resp.output_bytes;
                    for (int j = 0; j < 64; j++) {
                        dst->output[j] = resp.output[j];
                    }
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
                __syncthreads();

                // Telemetry updates for HKDF-SHA256 ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::RSA_2048_SIGN)) {
                // Card 26.70: RSA-2048 CRT signing
                // Uses payload_slab for 256-byte input, output_slab for 256-byte output
                __shared__ uint32_t s_base_write_idx_rsa;
                __shared__ uint64_t s_base_seq_rsa;
                if (tid == 0) {
                    s_base_seq_rsa = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_rsa = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // RSA signing is heavy - process one request per warp to allow parallelism
                const uint32_t warp_idx = tid / 32;
                const uint32_t num_warps_cta = blockDim.x / 32;
                const uint32_t lane = tid % 32;

                for (uint32_t i = warp_idx; i < batch; i += num_warps_cta) {
                    const FastPathRequest req = s_batch[i];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = s_dequeue_cycles;
                    resp.status = 0;
                    resp.client_id = req.client_id;
                    resp.opcode = req.opcode;

                    // RSA-2048 uses payload_slab for 256-byte input
                    const uint16_t input_len = req.input_len;
                    const uint32_t cq_cap_rsa = telemetry->cq_capacity_active;
                    const uint32_t write_idx_rsa = s_base_write_idx_rsa + i;
                    const uint32_t resp_slot_rsa = write_idx_rsa % cq_cap_rsa;

                    // Check if RSA keyslots are available and key is valid
                    const uint8_t rsa_slot = req.key_slot;
                    bool key_valid = (rsa_keyslots != nullptr && rsa_slot < RSA_MAX_KEYSLOTS
                                      && smoke::rsa::rsa_keyslot_valid(&rsa_keyslots[rsa_slot]));

                    if (input_len != 256 || req.payload_len != 256 || payload_slab == nullptr) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_len = 0;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (!key_valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::KEY_NOT_LOADED);
                        resp.output_len = 0;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        // Get input from payload_slab
                        const uint8_t* message = payload_slab + req.payload_offset;

                        // Get output slot in output_slab (256 bytes per signature)
                        const uint32_t slab_offset = resp_slot_rsa * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* signature = output_slab + slab_offset;

                        // Perform RSA-2048 CRT signing (warp-cooperative internally)
                        smoke::rsa::rsa2048_sign_keyslot(
                            &rsa_keyslots[rsa_slot],
                            message,
                            signature
                        );

                        resp.output_len = 0;  // Output is in slab, not inline
                        resp.output_offset = slab_offset;
                        resp.output_bytes = 256;
                    }

                    resp.t_complete_us = clock64();

                    // Only lane 0 of each warp writes the response
                    if (lane == 0) {
                        resp.seq = s_base_seq_rsa + (uint64_t)i;

                        volatile FastPathResponse* dst = &responses->responses[resp_slot_rsa];
                        // Card 26.72: Use commit protocol - write request_id LAST
                        write_response_base(dst, resp);
                        // RSA has no inline output data (uses slab)
                        commit_response(dst, resp.request_id, resp.status, resp.output_len);
                    }
                    __syncwarp();
                }
                __syncthreads();

                // Telemetry updates for RSA ops
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                    __threadfence_system();  // Card 26.71: Ensure response writes visible to host
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::MLDSA_SIGN)) {
                // Card 27.14: ML-DSA signing
                // Uses payload_slab for variable-length message, output_slab for signature
                // TODO: Wire to dilithium v3 batch kernel once key loading is validated
                __shared__ uint32_t s_base_write_idx_mldsa;
                __shared__ uint64_t s_base_seq_mldsa;
                if (tid == 0) {
                    s_base_seq_mldsa = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_mldsa = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                // MLDSA signing - process one request per warp
                const uint32_t warp_idx = tid / 32;
                const uint32_t num_warps_cta = blockDim.x / 32;
                const uint32_t lane = tid % 32;

                for (uint32_t i = warp_idx; i < batch; i += num_warps_cta) {
                    const FastPathRequest req = s_batch[i];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = s_dequeue_cycles;
                    resp.status = 0;
                    resp.client_id = req.client_id;
                    resp.opcode = req.opcode;

                    const uint32_t cq_cap_mldsa = telemetry->cq_capacity_active;
                    const uint32_t write_idx_mldsa = s_base_write_idx_mldsa + i;
                    const uint32_t resp_slot_mldsa = write_idx_mldsa % cq_cap_mldsa;

                    // Validate keyslot
                    const uint8_t mldsa_slot = req.key_slot;
                    const uint8_t mldsa_mode = req.input[0];  // Mode in first byte of input

                    bool key_valid = (mldsa_keyslots != nullptr
                                      && mldsa_slot < MLDSA_MAX_KEYSLOTS
                                      && mldsa_keyslots[mldsa_slot].key_loaded != 0
                                      && mldsa_keyslots[mldsa_slot].mode == mldsa_mode);

                    if (payload_slab == nullptr || output_slab == nullptr) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_len = 0;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (!key_valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::KEY_NOT_LOADED);
                        resp.output_len = 0;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        // Card 27.19: MLDSA_SIGN is handled via host-mediated dispatch
                        // (see engine_host.cpp:fast_path_submit_mldsa).
                        // sign_full_batch_gpu uses CUDA Dynamic Parallelism which cannot
                        // be called from inside a persistent kernel.
                        // If a request reaches here, something is wrong - caller should
                        // use fast_path_submit_mldsa() instead of fast_path_submit_batch().
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_OPCODE);
                        resp.output_len = 0;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    }

                    resp.t_complete_us = clock64();

                    // Only lane 0 of each warp writes the response
                    if (lane == 0) {
                        resp.seq = s_base_seq_mldsa + (uint64_t)i;

                        volatile FastPathResponse* dst = &responses->responses[resp_slot_mldsa];
                        write_response_base(dst, resp);
                        commit_response(dst, resp.request_id, resp.status, resp.output_len);
                    }
                    __syncwarp();
                }
                __syncthreads();

                // Telemetry updates
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    // Note: outcome_ok not incremented since signing not implemented yet
                    __threadfence_system();
                }
                __syncthreads();
            }
            else if (batch_opcode_nonshard == static_cast<uint16_t>(OpCode::SLH_DSA_SIGN)) {
                // Card SLH-002 P2: SLH_DSA_SIGN is handled via host-mediated
                // dispatch (see engine_host.cpp:fast_path_submit_slh), the same
                // Card 27.19 pattern as MLDSA_SIGN. The fused SLH signer is a
                // full-GPU batch kernel (~2.2M SHA-256 compressions/signature)
                // that cannot run inside a persistent kernel occupying all SMs.
                // If a request reaches here, the caller used the ring instead of
                // fast_path_submit_slh() — refuse fail-closed.
                __shared__ uint32_t s_base_write_idx_slh;
                __shared__ uint64_t s_base_seq_slh;
                if (tid == 0) {
                    s_base_seq_slh = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_slh = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = s_dequeue_cycles;
                    resp.client_id = req.client_id;
                    resp.opcode = req.opcode;
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_OPCODE);
                    resp.output_len = 0;
                    resp.output_offset = 0;
                    resp.output_bytes = 0;
                    resp.t_complete_us = clock64();

                    const uint32_t cq_cap_slh = telemetry->cq_capacity_active;
                    const uint32_t write_idx_slh = s_base_write_idx_slh + i;
                    const uint32_t resp_slot_slh = write_idx_slh % cq_cap_slh;
                    resp.seq = s_base_seq_slh + (uint64_t)i;

                    volatile FastPathResponse* dst = &responses->responses[resp_slot_slh];
                    write_response_base(dst, resp);
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }
                __syncthreads();

                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                    __threadfence_system();
                }
                __syncthreads();
            }
            else {
            // Card 26.34: Check if this batch uses multi-sign mode
            // Check first request - assume homogeneous batch
            const bool batch_is_multisign = (s_batch[0].flags & FAST_PATH_FLAG_MULTIMSG) != 0
                                            && payload_slab != nullptr && output_slab != nullptr;

            __syncthreads();

            if (batch_is_multisign) {
                // Card 26.34: P256 Multi-Sign path (work amplification)
                // Uses simple per-request signing, not warp-coop, because we need to loop
                // over N hashes per request.

                // Allocate response slots for multi-sign batch
                __shared__ uint32_t s_base_write_idx_ms;
                __shared__ uint64_t s_base_seq_ms;
                if (tid == 0) {
                    s_base_seq_ms = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                    s_base_write_idx_ms = atomicAdd((uint32_t*)&responses->write_idx, batch);
                }
                __syncthreads();

                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    FastPathRequest& req = s_batch[i];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = s_dequeue_cycles;
                    resp.status = 0;
                    resp.client_id = req.client_id;

                    // Parse header from input[0..7]
                    const MultiMsgHeader* hdr = reinterpret_cast<const MultiMsgHeader*>(req.input);
                    const uint16_t n_sigs = hdr->n_msgs;
                    const uint16_t item_len = hdr->msg_len;

                    // Validate: item_len must be 32, n_sigs in [1, P256_MULTISIGN_MAX_N]
                    if (item_len != P256_MULTISIGN_ITEM_LEN || n_sigs == 0 || n_sigs > P256_MULTISIGN_MAX_N) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                    }
                    else if (req.payload_len < n_sigs * P256_MULTISIGN_ITEM_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    }
                    else {
                        // Read hashes from payload_slab
                        const uint8_t* hash_data = payload_slab + req.payload_offset;

                        // Compute resp_slot for output slab access
                        const uint32_t cq_cap_ms = telemetry->cq_capacity_active;
                        const uint32_t write_idx_ms = s_base_write_idx_ms + i;
                        const uint32_t resp_slot_ms = write_idx_ms % cq_cap_ms;

                        // Get output slab segment for this response slot
                        const uint32_t slab_offset = resp_slot_ms * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* sig_output = output_slab + slab_offset;

                        const bool low_s = (req.flags & FAST_PATH_FLAG_LOW_S) != 0;
                        bool all_valid = true;

                        // Sign each hash
                        for (uint16_t m = 0; m < n_sigs; m++) {
                            const uint8_t* hash_m = hash_data + m * P256_MULTISIGN_ITEM_LEN;
                            uint8_t* sig_r = sig_output + m * P256_MULTISIGN_SIG_LEN;
                            uint8_t* sig_s = sig_r + 32;

                            const uint32_t nonce_ctr = atomicAdd(&g_nonce_counter, 1u);

#if ENGINE_USE_REAL_P256
                            const bool valid = smoke::p256::p256_sign_persistent(
                                sig_r, sig_s, hash_m, slot.private_d, nonce_ctr,
                                nullptr, low_s, false, nullptr, 0);
                            if (!valid) all_valid = false;
#else
                            // Stub mode
                            for (int j = 0; j < 32; j++) {
                                sig_r[j] = hash_m[j] ^ slot.private_d[j];
                                sig_s[j] = slot.private_d[j];
                            }
#endif
                        }

                        // Response metadata
                        resp.output_offset = slab_offset;
                        resp.output_bytes = n_sigs * P256_MULTISIGN_SIG_LEN;
                        resp.output_len = 64;  // First sig in inline output

                        // Copy first signature to inline output for convenience
                        for (int j = 0; j < 64; j++) {
                            resp.output[j] = sig_output[j];
                        }

                        if (!all_valid) {
                            resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                        }

                        // Telemetry: count units (signatures) not just requests
                        atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)n_sigs);
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    const uint32_t cq_cap = telemetry->cq_capacity_active;
                    const uint64_t seq = s_base_seq_ms + (uint64_t)i;
                    const uint32_t write_idx = s_base_write_idx_ms + i;
                    const uint32_t resp_slot = write_idx % cq_cap;
                    resp.seq = seq;

                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Use commit protocol - write request_id LAST
                    write_response_base(dst, resp);
                    for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }
                __syncthreads();

                // Telemetry updates for multi-sign
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                    __threadfence_system();
                }
            }
            else {
            // P256_SIGN: Original warp-coop path
            // ========================================================================
            // Multiple warps work in parallel on different signatures.
            // Phase 1: Each warp computes k*G for its subset of signatures
            // Phase 2: Batch inversion (1 Fermat inv for entire batch!) - lane 0 only
            // Phase 3: Each warp applies its Zinv and computes signatures
            //
            // With 4 warps, we get 4x parallelism per CTA!

            // Card 26.2: Signal batch begin (tid 0 only)
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->dbg_sign_begin, (unsigned long long)batch);
            }
            __syncthreads();

            // Card 26.5: Capture clock before phase 1
            uint64_t t_phase1_start = clock64();

            // ================================================================
            // PHASE 1: Multi-warp parallel k*G computation
            // ================================================================
            // Phase 3b: Each warp handles 4 signatures per round (batch-4)
            // Lane groups: 0-7=instance0, 8-15=instance1, 16-23=instance2, 24-31=instance3
            for (uint32_t base = 0; base < batch; base += num_warps * 4) {
                uint32_t sig_base = base + warp_id * 4;
                uint32_t instance = (lane >> 3) & 3;
                uint32_t i = sig_base + instance;

                if (i < batch) {
                    FastPathRequest& req = s_batch[i];

                    // Phase 1: each group processes its own signature
                    smoke::p256::warp_coop::p256_sign_warp_coop_phase1(
                        req.input,
                        slot.private_d,
                        s_sign_state[i],
                        lane
                    );

                    // Group leader collects Z for batch inversion
                    if ((lane & 7) == 0 && s_sign_state[i].valid) {
                        for (int limb = 0; limb < 8; limb++) {
                            s_Z_workspace[i * 8 + limb] = s_sign_state[i].Rz[limb];
                        }
                    }
                }
                __syncthreads();
            }

            // Card 26.5: Phase 1 timing
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->cyc_scalar_mul, clock64() - t_phase1_start);
            }

            // ================================================================
            // PHASE 2: Batch inversion (Card 14) - 1 Fermat inv for all!
            // ================================================================
            // Phase 4a: Warp 0 runs parallel batch inversion (all 32 lanes).
            // Guard is HERE in the caller, NOT inside the function.
            // Inlined return inside the function causes __shfl_sync divergence deadlock.
            uint64_t t_inv_start = clock64();
            if (warp_id == 0) {
                p256_batch_inv_block_montgomery_v2(
                    s_Z_workspace,
                    s_Zinv_workspace,
                    s_P_workspace,
                    batch
                );
            }
            __syncthreads();  // All threads wait for batch inv to complete

            // Card 26.5: Affine phase timing (batch inv is the main cost here)
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->cyc_affine, clock64() - t_inv_start);
            }

            // ================================================================
            // PHASE 3: Multi-warp parallel signature finalization
            // ================================================================
            // Optimize-6.5: Batch slot allocation for WARP_COOP (same as THREAD_ONLY)
            __shared__ uint32_t s_base_write_idx;
            __shared__ uint64_t s_base_seq;
            if (tid == 0) {
                s_base_seq = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                s_base_write_idx = atomicAdd((uint32_t*)&responses->write_idx, batch);
            }
            __syncthreads();

            uint64_t t_phase3_start = clock64();
            // Phase 3b: Each warp finalizes 4 signatures per round
            for (uint32_t base = 0; base < batch; base += num_warps * 4) {
                uint32_t sig_base = base + warp_id * 4;
                uint32_t instance = (lane >> 3) & 3;
                uint32_t i = sig_base + instance;

                if (i < batch) {
                    FastPathRequest& req = s_batch[i];
                    FastPathResponse resp;

                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = s_dequeue_cycles;
                    resp.status = 0;
                    resp.client_id = req.client_id;
                    resp.epoch_used = static_cast<uint8_t>((*active_epoch) & 0xFF);
                    resp.slot_used  = static_cast<uint8_t>(slot_idx);
                    // B-319b: this batch-4 variant never published output_len
                    // (latent — the live warp-coop path is the single-instance
                    // one; see N3-fix). The commit store now publishes it, so
                    // it must be set: r[32]||s[32] = 64 bytes.
                    resp.output_len = 64;

                    bool low_s = (req.flags & FAST_PATH_FLAG_LOW_S) != 0;

                    // Each group loads its own Zinv (width=8 broadcast)
                    uint32_t my_zinv_limb = s_Zinv_workspace[i * 8 + (lane & 7)];
                    uint32_t Zinv_local[8];
                    #pragma unroll
                    for (int limb = 0; limb < 8; limb++) {
                        Zinv_local[limb] = __shfl_sync(0xFFFFFFFF, my_zinv_limb, limb, 8);
                    }

                    bool valid = smoke::p256::warp_coop::p256_sign_warp_coop_phase3(
                        s_sign_state[i],
                        Zinv_local,
                        resp.r,
                        resp.s,
                        lane,
                        low_s
                    );

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                    }

                    resp.t_complete_us = clock64();

                    // Each group leader writes its response
                    if ((lane & 7) == 0) {
                        uint32_t cq_cap = telemetry->cq_capacity_active;
                        uint64_t seq = s_base_seq + i;
                        uint32_t write_idx = s_base_write_idx + i;
                        uint32_t resp_slot = write_idx % cq_cap;

                        resp.seq = seq;
                        volatile FastPathResponse* dst = &responses->responses[resp_slot];
                        dst->seq = resp.seq;
                        dst->t_submit_us = resp.t_submit_us;
                        dst->t_dequeue_us = resp.t_dequeue_us;
                        dst->t_complete_us = resp.t_complete_us;
                        dst->status = resp.status;
                        dst->client_id = resp.client_id;
                        dst->epoch_used = resp.epoch_used;
                        dst->slot_used = resp.slot_used;
                        dst->output_len = resp.output_len;  // B-319b: keep line-1 copy consistent
                        for (int j = 0; j < 32; j++) dst->r[j] = resp.r[j];
                        for (int j = 0; j < 32; j++) dst->s[j] = resp.s[j];
                        __threadfence_system();
                        commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                    }
                }
                __syncthreads();
            }

            // Card 26.10: Overflow detection - check once per batch (thread 0)
            if (tid == 0) {
                uint32_t cq_cap = telemetry->cq_capacity_active;
                uint32_t read_idx = responses->read_idx;
                uint32_t end_idx = s_base_write_idx + batch;
                if (end_idx >= read_idx + cq_cap) {
                    uint32_t overwrites = end_idx - (read_idx + cq_cap);
                    if (overwrites > batch) overwrites = batch;
                    atomicAdd((unsigned long long*)&telemetry->response_overwrites, (unsigned long long)overwrites);
                }
            }

            // Card 26.5: Phase 3 timing (sign finalize + enqueue)
            // Optimize-6.5: Batched telemetry updates (single atomic per batch)
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->cyc_sign_finalize, clock64() - t_phase3_start);
                atomicAdd((unsigned long long*)&telemetry->dbg_sign_done, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_write_response, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                // Note: Assumes all outcomes OK (common case). For accurate error tracking,
                // would need to count errors in the signing loop and update here.
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
            }

            __syncthreads();
            } // end warp-coop single-sign else block
            } // end else P256_SIGN path (Card 26.34: includes multi-sign)
#else
            // ========================================================================
            // THREAD-ONLY SIGNING PATH (Default)
            // ========================================================================
            // Each thread handles independent requests (32 concurrent signatures).
            // Optimal for single-CTA persistent kernel.

            // Optimize-6.5: Batch slot allocation - thread 0 reserves all slots at once
            // This reduces atomic contention from O(batch) to O(1) atomicAdds
            __shared__ uint32_t s_base_write_idx;
            __shared__ uint64_t s_base_seq;
            if (tid == 0) {
                s_base_seq = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (unsigned long long)batch);
                s_base_write_idx = atomicAdd((uint32_t*)&responses->write_idx, batch);
            }
            __syncthreads();

            // Process batch - each thread handles multiple requests
            for (uint32_t i = tid; i < batch; i += blockDim.x) {
                FastPathRequest& req = s_batch[i];
                FastPathResponse resp;

                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;    // Card 25.2C: Echo submit timestamp
                resp.t_dequeue_us = s_dequeue_cycles;  // Card 25.2C: Dequeue cycles (shared)
                resp.status = 0;  // OK
                resp.client_id = req.client_id;        // Card 26.16: Multi-client

// Card 26.19: Multi-op dispatch based on opcode
                uint16_t opcode = req.opcode;

                switch (opcode) {
                    case 0:  // P256_SIGN (backward compat: opcode=0)
                    case static_cast<uint16_t>(OpCode::P256_SIGN): {
                        // Card 66: Record epoch/slot used for P256 signing
                        resp.epoch_used = static_cast<uint8_t>((*active_epoch) & 0xFF);
                        resp.slot_used  = static_cast<uint8_t>(slot_idx);
                        // Card 26.34: Check for multi-sign mode (work amplification)
                        const bool is_multisign = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;

                        if (is_multisign && payload_slab != nullptr && output_slab != nullptr) {
                            // Card 26.34: P256 Multi-Sign (work amplification)
                            // Parse header from input[0..7]
                            const MultiMsgHeader* hdr = reinterpret_cast<const MultiMsgHeader*>(req.input);
                            const uint16_t n_sigs = hdr->n_msgs;
                            const uint16_t item_len = hdr->msg_len;

                            // Validate: item_len must be 32, n_sigs in [1, P256_MULTISIGN_MAX_N]
                            if (item_len != P256_MULTISIGN_ITEM_LEN || n_sigs == 0 || n_sigs > P256_MULTISIGN_MAX_N) {
                                resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                                break;
                            }

                            // Validate payload_len covers all hashes
                            if (req.payload_len < n_sigs * P256_MULTISIGN_ITEM_LEN) {
                                resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                                break;
                            }

                            // Read hashes from payload_slab
                            const uint8_t* hash_data = payload_slab + req.payload_offset;

                            // Compute resp_slot early for output slab access
                            const uint32_t cq_cap_ms = telemetry->cq_capacity_active;
                            const uint32_t write_idx_ms = s_base_write_idx + i;
                            const uint32_t resp_slot_ms = write_idx_ms % cq_cap_ms;

                            // Get output slab segment for this response slot
                            const uint32_t slab_offset = resp_slot_ms * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                            uint8_t* sig_output = output_slab + slab_offset;

                            const bool low_s = (req.flags & FAST_PATH_FLAG_LOW_S) != 0;
                            bool all_valid = true;

                            // Sign each hash
                            for (uint16_t m = 0; m < n_sigs; m++) {
                                const uint8_t* hash_m = hash_data + m * P256_MULTISIGN_ITEM_LEN;
                                uint8_t* sig_r = sig_output + m * P256_MULTISIGN_SIG_LEN;
                                uint8_t* sig_s = sig_r + 32;

                                const uint32_t nonce_ctr = atomicAdd(&g_nonce_counter, 1u);

#if ENGINE_USE_REAL_P256
                                const bool valid = smoke::p256::p256_sign_persistent(
                                    sig_r, sig_s, hash_m, slot.private_d, nonce_ctr,
                                    nullptr, low_s, false, nullptr, 0);
                                if (!valid) all_valid = false;
#else
                                // Stub mode
                                for (int j = 0; j < 32; j++) {
                                    sig_r[j] = hash_m[j] ^ slot.private_d[j];
                                    sig_s[j] = slot.private_d[j];
                                }
#endif
                            }

                            // Response metadata
                            resp.output_offset = slab_offset;
                            resp.output_bytes = n_sigs * P256_MULTISIGN_SIG_LEN;
                            resp.output_len = 64;  // First sig in inline output

                            // Copy first signature to inline output for convenience
                            for (int j = 0; j < 64; j++) {
                                resp.output[j] = sig_output[j];
                            }

                            if (!all_valid) {
                                resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                            }

                            // Telemetry: count units (signatures) not just requests
                            atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)n_sigs);
                            break;
                        }

                        // Single-sign path (original)
#if ENGINE_USE_REAL_P256
                        // Real P-256 signing
                        bool low_s = (req.flags & FAST_PATH_FLAG_LOW_S) != 0;

                        // Simple counter-based nonce for deterministic mode
                        uint32_t nonce_ctr = atomicAdd(&g_nonce_counter, 1u);

                        // Card 26.2: Signal sign begin (per thread - first thread only to avoid spam)
                        if (tid == 0) {
                            atomicAdd((unsigned long long*)&telemetry->dbg_sign_begin, 1ull);
                        }

                        // p256_sign_persistent(out_r, out_s, hash, priv_d, nonce_counter,
                        //                       ext_nonce, low_s_enabled, use_random, rng_seed, stream_id)
                        bool valid = smoke::p256::p256_sign_persistent(
                            resp.r,           // out_r
                            resp.s,           // out_s
                            req.input,        // input (hash for P256_SIGN)
                            slot.private_d,   // priv_d
                            nonce_ctr,        // nonce_counter
                            nullptr,          // ext_nonce (use internal)
                            low_s,            // low_s_enabled
                            false,            // use_random_nonce
                            nullptr,          // rng_seed
                            0                 // rng_stream_id
                        );

                        // Card 26.2: Signal sign done (per thread - first thread only)
                        if (tid == 0) {
                            atomicAdd((unsigned long long*)&telemetry->dbg_sign_done, 1ull);
                        }

                        if (!valid) {
                            resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                        }
#else
                        // Stub mode: XOR-based signature
                        for (int j = 0; j < 32; j++) {
                            resp.r[j] = req.input[j] ^ slot.private_d[j];
                            resp.s[j] = slot.private_d[j];
                        }
#endif
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::SHA256): {
                        // Card 26.19: SHA-256 hashing
                        // Validate input_len ∈ [0, 32]
                        if (req.input_len > 32) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        // Hash using single-block SHA-256
                        smoke::hash::sha256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::SHA512): {
                        // Card 26.31: SHA-512 hashing
                        // Validate input_len ∈ [0, 32]
                        if (req.input_len > 32) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        // Hash using single-block SHA-512
                        smoke::hash::sha512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::SHA3_256): {
                        // Card 26.32: SHA3-256 (Keccak) hashing
                        // Validate input_len ∈ [0, 32]
                        if (req.input_len > 32) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        // Hash using single-block SHA3-256
                        smoke::hash::sha3_256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::BLAKE2B_256): {
                        // Card 26.33: BLAKE2b-256 hashing
                        // Validate input_len ∈ [0, 32]
                        if (req.input_len > 32) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        // Hash using single-block BLAKE2b-256
                        smoke::hash::blake2b_256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::SHA3_512): {
                        // Card 140: SHA3-512 (Keccak) hashing
                        // Validate input_len ∈ [0, 32]
                        if (req.input_len > 32) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        smoke::hash::sha3_512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::BLAKE2B_512): {
                        // Card 141: BLAKE2b-512 hashing
                        // Validate input_len ∈ [0, 32]
                        if (req.input_len > 32) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        smoke::hash::blake2b_512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::HMAC_SHA256): {
                        // Card 66: Record epoch/slot used for keyed op
                        resp.epoch_used = static_cast<uint8_t>((*active_epoch) & 0xFF);
                        resp.slot_used  = static_cast<uint8_t>(slot_idx);
                        // Card 26.29: HMAC-SHA256 keyed hash
                        // Validate input_len ∈ [0, HMAC_SHA256_MAX_MSG_LEN]
                        if (req.input_len > HMAC_SHA256_MAX_MSG_LEN) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        // Use key from active keyslot (slot.private_d is 32 bytes)
                        // HMAC-SHA256 key is the keyslot's private_d field
                        hmac_sha256_device(
                            slot.private_d,    // key (32 bytes from keyslot)
                            32,                // key_len (P-256 keys are 32 bytes)
                            req.input,         // message
                            req.input_len,     // message length
                            resp.output        // output (32 bytes)
                        );
                        resp.output_len = 32;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::HMAC_SHA512): {
                        resp.epoch_used = static_cast<uint8_t>((*active_epoch) & 0xFF);  // Card 66
                        resp.slot_used  = static_cast<uint8_t>(slot_idx);                // Card 66
                        // Card 26.51: HMAC-SHA512 keyed hash
                        if (req.input_len > HMAC_SHA512_MAX_MSG_LEN) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        hmac_sha512_device(
                            slot.private_d,    // key (32 bytes from keyslot)
                            32,                // key_len
                            req.input,         // message
                            req.input_len,     // message length
                            resp.output        // output (64 bytes)
                        );
                        resp.output_len = 64;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::HMAC_SHA384): {
                        resp.epoch_used = static_cast<uint8_t>((*active_epoch) & 0xFF);  // Card 66
                        resp.slot_used  = static_cast<uint8_t>(slot_idx);                // Card 66
                        // Card 26.52: HMAC-SHA384 keyed hash
                        if (req.input_len > HMAC_SHA384_MAX_MSG_LEN) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        hmac_sha384_device(
                            slot.private_d,    // key (32 bytes from keyslot)
                            32,                // key_len
                            req.input,         // message
                            req.input_len,     // message length
                            resp.output        // output (48 bytes)
                        );
                        resp.output_len = 48;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::AES256_CTR): {
                        resp.epoch_used = static_cast<uint8_t>((*active_epoch) & 0xFF);  // Card 66
                        resp.slot_used  = static_cast<uint8_t>(slot_idx);                // Card 66
                        // Card 26.54: AES-256-CTR stream cipher
                        if (req.input_len < AES256_CTR_NONCE_LEN) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }
                        if (req.input_len > 32) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        // Card 26.64: Use precomputed round keys from keyslot
                        smoke::aes::aes256_ctr_encrypt_short_cached(
                            slot.aes_roundkeys, // precomputed round keys
                            req.input,          // nonce[12] + plaintext[N]
                            req.input_len,      // total input length
                            resp.output         // ciphertext output
                        );
                        resp.output_len = req.input_len - AES256_CTR_NONCE_LEN;
                        resp.status = 0;  // OK
                        break;
                    }

                    case static_cast<uint16_t>(OpCode::CHACHA20): {
                        resp.epoch_used = static_cast<uint8_t>((*active_epoch) & 0xFF);  // Card 66
                        resp.slot_used  = static_cast<uint8_t>(slot_idx);                // Card 66
                        // Card 26.66: ChaCha20 stream cipher
                        if (req.input_len < CHACHA20_NONCE_LEN) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }
                        if (req.input_len > 32) {
                            resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                            break;
                        }

                        smoke::chacha::chacha20_encrypt_short_keyslot(
                            slot.private_d,     // key (32 bytes from keyslot)
                            req.input,          // nonce[12] + plaintext[N]
                            req.input_len,      // total input length
                            resp.output,        // ciphertext output
                            &resp.output_len    // output length
                        );
                        resp.status = 0;  // OK
                        break;
                    }

                    default: {
                        // Unknown opcode
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_OPCODE);
                        break;
                    }
                }

                // Card 25.2C: Record completion timestamp
                resp.t_complete_us = clock64();

                // Optimize-6.5: Use pre-allocated slot from batch allocation
                // Each thread calculates its slot offset within the batch
                uint32_t cq_cap = telemetry->cq_capacity_active;
                uint64_t seq = s_base_seq + i;  // Pre-allocated sequence
                uint32_t write_idx = s_base_write_idx + i;  // Pre-allocated slot
                uint32_t resp_slot = write_idx % cq_cap;

                // Set sequence number and write response
                resp.seq = seq;
                // Card 26.2 FIX: Explicit copy to volatile response buffer
                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                for (int j = 0; j < 32; j++) dst->r[j] = resp.r[j];
                for (int j = 0; j < 32; j++) dst->s[j] = resp.s[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b

            }
            __syncthreads();

            // Card 26.10: Overflow detection - check once per batch (thread 0)
            if (tid == 0) {
                uint32_t cq_cap = telemetry->cq_capacity_active;
                uint32_t read_idx = responses->read_idx;
                uint32_t end_idx = s_base_write_idx + batch;
                if (end_idx >= read_idx + cq_cap) {
                    // Count how many overwrites in this batch
                    uint32_t overwrites = end_idx - (read_idx + cq_cap);
                    if (overwrites > batch) overwrites = batch;
                    atomicAdd((unsigned long long*)&telemetry->response_overwrites, (unsigned long long)overwrites);
                }
            }

            // Optimize-6.2: Batch telemetry updates using warp-level reduction
            // Count outcomes per-warp, then have warp 0 do atomic adds
            {
                // Each thread reports 1 if it processed a request (i < batch means valid)
                // Using ballot to count successes/failures per warp
                uint32_t my_ok = 0, my_err = 0;
                for (uint32_t i = tid; i < batch; i += blockDim.x) {
                    // Note: status was set during processing above
                    // We don't have access to it here since resp is local
                    // For now, assume all OK (the common case) and fix if needed
                    my_ok++;
                }

                // Warp-level reduction of counts
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    my_ok += __shfl_down_sync(0xFFFFFFFF, my_ok, offset);
                }

                // Lane 0 of each warp has the warp total
                if (lane == 0) {
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)my_ok);
                }
            }

            // Debug counters - single update per batch by thread 0
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->dbg_write_response, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
            }
#endif // P256_WARP_COOP_ENABLED

            // Card 26.1B: Tail was already updated atomically during claiming
            // Just need threadfence to ensure response writes are visible to host
            if (lane == 0) {
                __threadfence_system();  // Ensure response writes visible to host
            }
        } else {
            // No work - yield to avoid spinning
            engine_device_yield();
        }

        __syncthreads();
    }
}
#endif // ENABLE_LEGACY_KERNEL

// DEBUG: Minimal test kernel to verify basic launch works
__global__ void fast_path_debug_kernel(volatile uint64_t* marker) {
    if (threadIdx.x == 0) {
        *marker = 0xDEADC0DEUL;
        __threadfence_system();
    }
}

// ============================================================================
// Card 26.2: SHARDED Fast Path Kernel - Zero Contention
// ============================================================================
// Each CTA owns its own shard (ring + response buffer).
// No atomicCAS contention since there's no sharing between CTAs.
// This eliminates the 99.99% contention waste seen with multi-CTA single-ring.
// Card 26.4 REVERT: Back to 256 threads (128 caused slowdown)
// Card 26.24: Added payload_slab for extended multi-message
// Card 26.25: Added output_slab for extended multi-message outputs
__global__ __launch_bounds__(256) void fast_path_engine_loop_sharded(
    FastPathRing** ring_ptrs,           // Card 26.2: Array of ring pointers (one per shard)
    FastPathResponseBuffer** resp_ptrs, // Card 26.2: Array of response buffer pointers
    FastPathTelemetry* telemetry,
    Keyslot* keyslots,
    volatile uint32_t* active_epoch,
    volatile bool* shutdown,
    uint32_t num_shards,                 // Card 26.2: Number of shards
    const uint8_t* payload_slab,         // Card 26.24: Extended payload buffer (mapped)
    uint8_t* output_slab,                // Card 26.25: Extended output buffer (mapped, writable)
    RSAKeyslot* rsa_keyslots,            // Card 26.70: RSA-2048 CRT keyslots
    MLDSAKeyslot* mldsa_keyslots         // Card 27.14: ML-DSA keyslots
) {
    // Card 26.2 FIX: Each CTA owns exactly one shard - DIRECT 1:1 MAPPING
    // No modulo! The hard gate ensures num_ctas == num_shards.
    const uint32_t cta_id = blockIdx.x;
    const uint32_t my_shard = cta_id;  // Direct mapping (no modulo)

    // Safety: If somehow we have more CTAs than shards, exit early
    // This should never happen if the hard gate in engine_host.cpp is working
    if (my_shard >= num_shards) {
        return;
    }

    // Slab-collision fix: the output slab is SHARED across all shards, but
    // slab segment indices were previously derived from the PER-SHARD response
    // slot — two shards holding equal slot values overwrote each other's
    // segments (silent output corruption under concurrent slab-mode ops).
    // Partition the slab: each shard owns a disjoint span of
    // slab_segs_per_shard segments; within a shard a segment is recycled every
    // slab_segs_per_shard responses. The host drain path audits that bound and
    // fails any response drained too late (OpStatus::SLAB_OVERRUN).
    const uint32_t slab_segs_per_shard =
        (FAST_PATH_STANDARD_OUTPUT_SLAB_SIZE / FAST_PATH_OUTPUT_SEGMENT_BYTES) / num_shards;
    const uint32_t slab_seg_base = my_shard * slab_segs_per_shard;
    const uint32_t seal_segs_per_shard =
        FAST_PATH_SEAL_OUTPUT_SEGMENTS / num_shards;
    const uint32_t seal_seg_base = my_shard * seal_segs_per_shard;

    // Get this CTA's ring and response buffer (exclusive ownership)
    // Card 26.2 FIX: Use volatile pointers to ensure every access goes to memory
    // (host writes head, GPU must see latest value)
    volatile FastPathRing* ring = ring_ptrs[my_shard];
    volatile FastPathResponseBuffer* responses = resp_ptrs[my_shard];

    // DEBUG: Write magic marker immediately to prove kernel started
    if (threadIdx.x == 0) {
        telemetry->kernel_start_time_us = 0xDEADBEEF;  // Magic marker
        __threadfence_system();
    }
    __syncthreads();

    // Shared memory for batch processing
    __shared__ uint32_t s_batch_count;
    __shared__ uint32_t s_tail;
    __shared__ uint64_t s_dequeue_cycles;
    __shared__ FastPathRequest s_batch[FAST_PATH_BATCH_MAX];

    // Response reservation (batch-wide, declared once)
    __shared__ uint32_t s_base_write_idx;
    __shared__ uint64_t s_base_seq;

    // Card 26.15: Shared cycle timing for authoritative metrics
    __shared__ uint64_t s_iter_start_cycles;
    __shared__ uint64_t s_compute_start_cycles;

#if P256_WARP_COOP_ENABLED
    __shared__ smoke::p256::warp_coop::WarpCoopSignState s_sign_state[FAST_PATH_BATCH_MAX];
    __shared__ uint32_t s_Z_workspace[FAST_PATH_BATCH_MAX * 8];
    __shared__ uint32_t s_Zinv_workspace[FAST_PATH_BATCH_MAX * 8];
    __shared__ uint32_t s_P_workspace[FAST_PATH_BATCH_MAX * 8];
#endif

    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp_id = tid >> 5;
    const uint32_t num_warps = (uint32_t)blockDim.x >> 5;  // 8 warps @ 256

    // Initialize telemetry on kernel start (CTA 0 only)
    if (cta_id == 0 && lane == 0) {
        telemetry->kernel_launches = 1;
        telemetry->total_requests_seen = 0;
        telemetry->total_batches_processed = 0;
        telemetry->total_batch_size_sum = 0;
        telemetry->kernel_idle_cycles = 0;
        telemetry->kernel_busy_cycles = 0;
        telemetry->kernel_start_time_us = 0;

#if P256_WARP_COOP_ENABLED
        telemetry->signer_model = static_cast<uint32_t>(SignerModel::WARP_COOP);
        telemetry->threads_per_signature = 32;
#else
        telemetry->signer_model = static_cast<uint32_t>(SignerModel::THREAD_ONLY);
        telemetry->threads_per_signature = 1;
#endif
        telemetry->service_lanes = gridDim.x;

        // Zero out per-CTA distribution counters
        for (uint32_t i = 0; i < FastPathTelemetry::MAX_SERVICE_LANES; i++) {
            telemetry->per_cta_batches[i] = 0;
            telemetry->per_cta_requests[i] = 0;
            telemetry->per_cta_attempts[i] = 0;
        }

        // Initialize debug beacons
        telemetry->dbg_enter_kernel = 0;
        telemetry->dbg_dequeue_attempts = 0;
        telemetry->dbg_claim_success = 0;
        telemetry->dbg_batch_begin = 0;
        telemetry->dbg_sign_begin = 0;
        telemetry->dbg_sign_done = 0;
        telemetry->dbg_write_response = 0;
        telemetry->dbg_loop_heartbeat = 0;

        // Card 26.5: Initialize inversion census counters
        telemetry->dbg_inv_fermat_calls = 0;
        telemetry->dbg_inv_card14_batches = 0;
        telemetry->dbg_j2a_calls = 0;
        telemetry->dbg_scalar_mul_calls = 0;

        telemetry->cyc_dequeue = 0;
        telemetry->cyc_scalar_mul = 0;
        telemetry->cyc_affine = 0;
        telemetry->cyc_sign_finalize = 0;
        telemetry->cyc_enqueue_resp = 0;

        telemetry->response_overwrites = 0;
        telemetry->response_seq_counter = 0;

        // Card 26.12: Initialize state counters (feed/drain/compute limiter proof)
        // WARNING: These are ITERATION-based, not time-based. Use cycle counters for decisions.
        telemetry->state_idle_empty = 0;
        telemetry->state_idle_respfull = 0;
        telemetry->state_busy = 0;

        // Card 26.15: Initialize CYCLE-BASED state counters (AUTHORITATIVE)
        telemetry->cycles_idle_empty = 0;
        telemetry->cycles_idle_respfull = 0;
        telemetry->cycles_compute = 0;
        telemetry->cycles_total = 0;

        if (telemetry->cq_capacity_active == 0) {
            telemetry->cq_capacity_active = FAST_PATH_RESPONSE_BUFFER;
        }
        telemetry->cq_capacity_max = FAST_PATH_RESPONSE_BUFFER;

        g_telemetry_ptr = telemetry;
    }

    if (cta_id == 0 && lane == 0) {
        telemetry->dbg_enter_kernel = 1;
        __threadfence_system();
    }
    __syncthreads();
    if (cta_id == 0) {
        __threadfence();
    }

    // Main persistent loop
    while (true) {
        if (*shutdown) {
            return;
        }

        // Card 26.15: Record iteration start time for cycle-accurate metrics
        if (tid == 0) {
            s_iter_start_cycles = clock64();
        }

        // Card 26.2: Only thread 0 dequeues work - NO atomicCAS needed!
        // This CTA owns its shard exclusively, so simple read/write suffices.
        if (tid == 0) {
            s_batch_count = 0;

#if !BENCH_LITE_TELEMETRY
            // Heartbeat every 1000 iterations (CTA 0 only)
            if (cta_id == 0) {
                static __shared__ uint32_t heartbeat_counter;
                heartbeat_counter = (heartbeat_counter + 1) % 1000;
                if (heartbeat_counter == 0) {
                    atomicAdd((unsigned long long*)&telemetry->dbg_loop_heartbeat, 1ull);
                }
            }

            // Count dequeue attempts
            atomicAdd((unsigned long long*)&telemetry->dbg_dequeue_attempts, 1ull);
            if (cta_id < FastPathTelemetry::MAX_SERVICE_LANES) {
                atomicAdd((unsigned long long*)&telemetry->per_cta_attempts[cta_id], 1ull);
            }
#endif

            // Check response buffer space
            // Card 26.2 FIX: responses pointer is volatile, so all field accesses are volatile
            uint32_t cq_capacity = telemetry->cq_capacity_active;
            uint32_t write_idx_val = responses->write_idx;  // Volatile read (we update this)
            uint32_t read_idx_val = responses->read_idx;    // Volatile read (host updates this)
            uint32_t resp_used = write_idx_val - read_idx_val;
            uint32_t resp_space = cq_capacity - resp_used;

            if (resp_space == 0) {
                // Card 26.12: IDLE_RESPFULL - GPU blocked by host not draining responses
                atomicAdd((unsigned long long*)&telemetry->state_idle_respfull, 1ull);
                // Card 26.15: AUTHORITATIVE cycle tracking for IDLE_RESPFULL
                atomicAdd((unsigned long long*)&telemetry->cycles_idle_respfull, clock64() - s_iter_start_cycles);
#if !BENCH_LITE_TELEMETRY
                atomicAdd((unsigned long long*)&telemetry->kernel_idle_cycles, 1ull);
#endif
            } else {
                // Card 26.2: Direct read - no atomicCAS needed!
                // This CTA owns this shard exclusively.

                // Cheap volatile read first (no fence)
                // Card 26.34 FIX: Standard ring buffer semantics
                // Producer (host) writes at tail, advances tail
                // Consumer (kernel) reads at head, advances head
                uint32_t head = ring->head;
                uint32_t tail = ring->tail;
                uint32_t available = tail - head;  // Card 26.59 FIX: tail - head (producer - consumer)

                // Only pay __threadfence_system() when we *appear idle*.
                // This preserves host->GPU visibility for mapped memory
                // while avoiding the fence cost when work is already visible.
                if (available == 0) {
                    __threadfence_system();
                    head = ring->head;   // re-read after fence
                    tail = ring->tail;   // keep consistent
                    available = tail - head;  // Card 26.59 FIX: tail - head

                    if (available == 0) {
                        // Card 26.12: IDLE_EMPTY - GPU waiting for host to publish work
                        atomicAdd((unsigned long long*)&telemetry->state_idle_empty, 1ull);
                        // Card 26.15: AUTHORITATIVE cycle tracking for IDLE_EMPTY
                        atomicAdd((unsigned long long*)&telemetry->cycles_idle_empty, clock64() - s_iter_start_cycles);
                        // Truly idle - small backoff to reduce PCIe polling
                        __nanosleep(64);
                    }
                }

                if (available > 0) {
                    // Card 26.12: BUSY - GPU has work to process
                    atomicAdd((unsigned long long*)&telemetry->state_busy, 1ull);
                    // Card 26.15: Record compute start time for AUTHORITATIVE cycle tracking
                    s_compute_start_cycles = clock64();
                    // Calculate batch size
                    uint32_t batch = min(available, (uint32_t)FAST_PATH_BATCH_MAX);
                    batch = min(batch, resp_space);

                    s_batch_count = batch;
                    s_tail = tail;

                    // Signal successful claim
                    atomicAdd((unsigned long long*)&telemetry->dbg_claim_success, 1ull);

                    s_dequeue_cycles = clock64();

                    // Card 26.59 FIX: Consumer reads from HEAD position, not tail
                    // Pull requests into shared memory BEFORE advancing head.
                    // This prevents host from overwriting slots we're still reading.
                    // FAST_PATH_RING_SIZE is power-of-two (8192), so & is faster than %
                    // Card 26.2 FIX: Explicit copy from volatile ring to non-volatile shared mem
                    #if ((FAST_PATH_RING_SIZE & (FAST_PATH_RING_SIZE - 1)) == 0)
                    constexpr uint32_t ring_mask = FAST_PATH_RING_SIZE - 1;
                    for (uint32_t i = 0; i < batch; i++) {
                        uint32_t slot = (head + i) & ring_mask;  // Card 26.59 FIX: read from head
                        // Copy from volatile to local (volatile struct copy)
                        const volatile FastPathRequest* src = &ring->requests[slot];
                        s_batch[i].request_id = src->request_id;
                        s_batch[i].t_submit_us = src->t_submit_us;
                        for (int j = 0; j < 32; j++) s_batch[i].input[j] = src->input[j];
                        s_batch[i].key_slot = src->key_slot;
                        s_batch[i].flags = src->flags;
                        s_batch[i].client_id = src->client_id;
                        s_batch[i].opcode = src->opcode;         // Card 26.19: Multi-op support
                        s_batch[i].input_len = src->input_len;   // Card 26.19: Variable input length
                        s_batch[i].payload_offset = src->payload_offset;  // Card 26.24: Extended payload
                        s_batch[i].payload_len = src->payload_len;        // Card 26.24: Extended payload
                    }
                    #else
                    for (uint32_t i = 0; i < batch; i++) {
                        uint32_t slot = (head + i) % FAST_PATH_RING_SIZE;  // Card 26.59 FIX: read from head
                        const volatile FastPathRequest* src = &ring->requests[slot];
                        s_batch[i].request_id = src->request_id;
                        s_batch[i].t_submit_us = src->t_submit_us;
                        for (int j = 0; j < 32; j++) s_batch[i].input[j] = src->input[j];
                        s_batch[i].key_slot = src->key_slot;
                        s_batch[i].flags = src->flags;
                        s_batch[i].client_id = src->client_id;
                        s_batch[i].opcode = src->opcode;         // Card 26.19: Multi-op support
                        s_batch[i].input_len = src->input_len;   // Card 26.19: Variable input length
                        s_batch[i].payload_offset = src->payload_offset;  // Card 26.24: Extended payload
                        s_batch[i].payload_len = src->payload_len;        // Card 26.24: Extended payload
                    }
                    #endif

                    // Card 26.59 FIX: Advance HEAD (consumer position) AFTER copy completes.
                    // Fence ensures all reads complete before we publish head.
                    // This prevents reordering around the mapped memory write.
                    __threadfence_system();
                    ring->head = head + batch;  // Card 26.59 FIX: advance head (consumer position)

                    // Update telemetry (essential for accounting)
                    atomicAdd((unsigned long long*)&telemetry->total_requests_seen, (unsigned long long)batch);
#if !BENCH_LITE_TELEMETRY
                    atomicAdd((unsigned long long*)&telemetry->total_batches_processed, 1ull);
                    atomicAdd((unsigned long long*)&telemetry->total_batch_size_sum, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->kernel_busy_cycles, 1ull);

                    if (cta_id < FastPathTelemetry::MAX_SERVICE_LANES) {
                        atomicAdd((unsigned long long*)&telemetry->per_cta_batches[cta_id], 1ull);
                        atomicAdd((unsigned long long*)&telemetry->per_cta_requests[cta_id], (unsigned long long)batch);
                    }
#endif
                } else {
#if !BENCH_LITE_TELEMETRY
                    atomicAdd((unsigned long long*)&telemetry->kernel_idle_cycles, 1ull);
#endif
                }
            }
        }
        __syncthreads();

        const uint32_t batch = s_batch_count;

        // Early continue if no work - keeps all threads in lockstep
        if (batch == 0) {
            __syncthreads();
            continue;
        }

#if !BENCH_LITE_TELEMETRY
        // Got work - signal batch begin (tid 0 only, not all lane 0s)
        if (tid == 0) {
            atomicAdd((unsigned long long*)&telemetry->dbg_batch_begin, 1ull);
        }
#endif

        // Reserve response slots + seq range ONCE per batch (tid 0)
        if (tid == 0) {
            s_base_seq = atomicAdd((unsigned long long*)&telemetry->response_seq_counter, (uint64_t)batch);
            s_base_write_idx = atomicAdd((uint32_t*)&responses->write_idx, batch);
        }
        __syncthreads();

#if P256_WARP_COOP_ENABLED
        // ========== WARP-COOP SIGNING PATH (with Card 26.19 Multi-Op Dispatch) ==========

        // Card 26.19: Check if batch contains non-P256 opcodes
        // Sample first request's opcode to determine path
        // Note: Under SHARD_LOCAL policy, batches are typically homogeneous
        __shared__ uint16_t s_batch_opcode;
        if (tid == 0) {
            s_batch_opcode = s_batch[0].opcode;
        }
        __syncthreads();

        const uint16_t batch_opcode = s_batch_opcode;

        // Card 26.19: Fast path for hash operations (SHA-256)
        // Card 26.22: Multi-message support for work amplification
        // These don't need warp-coop or batch inversion - just per-thread hashing
        if (batch_opcode == static_cast<uint16_t>(OpCode::SHA256)) {
            // Track total digests for multi-message telemetry
            __shared__ uint32_t s_total_digests;
            if (tid == 0) s_total_digests = 0;
            __syncthreads();

            // Per-thread SHA-256 hashing
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;

                // Card 26.22: Check for multi-message mode
                const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                uint32_t digests_this_req = 0;

                // Card 26.25: Pre-compute response slot for potential slab write
                const uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                const uint32_t write_idx_pre = s_base_write_idx + i;
                const uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                if (is_multimsg) {
                    // Parse MultiMsgHeader from input[0..7]
                    // Little-endian: n_msgs at [0-1], msg_len at [2-3], flags at [4-7]
                    const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                    const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                    const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                               ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                    // Card 26.24: Determine payload source
                    // Phase 1 (payload_len == 0): Messages in input[8..31]
                    // Phase 2 (payload_len > 0): Messages in payload_slab[payload_offset..]
                    const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                    const uint8_t* payload_data = use_extended_payload
                        ? (payload_slab + req.payload_offset)
                        : (req.input + MULTIMSG_HEADER_SIZE);
                    const uint32_t payload_avail = use_extended_payload
                        ? req.payload_len
                        : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                    // Card 26.25: Calculate bytes needed for output
                    const uint32_t bytes_needed = n_msgs * 32;

                    // Validate multi-message constraints
                    bool valid = true;
                    if (n_msgs == 0) valid = false;
                    if (msg_len > MULTIMSG_MAX_MSG_LEN) valid = false;
                    if (hdr_flags != 0) valid = false;  // Reserved must be 0
                    if (n_msgs * msg_len > payload_avail) valid = false;
                    // Card 26.25: Check against output slab segment size (2048 bytes = 64 digests)
                    if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (bytes_needed <= 64) {
                        // Card 26.25: Inline mode (N <= 2, backward compatible)
                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::sha256_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                resp.output + m * 32
                            );
                        }
                        resp.output_len = bytes_needed;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;  // Signal: use inline output[64]
                        digests_this_req = n_msgs;
                    } else {
                        // Card 26.25: Slab mode (N > 2)
                        // Each response slot owns a fixed segment in the output slab
                        uint32_t slab_offset = (slab_seg_base + (write_idx_pre % slab_segs_per_shard))
                                               * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* slab_ptr = output_slab + slab_offset;

                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::sha256_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                slab_ptr + m * 32
                            );
                        }

                        // Also copy first 64 bytes to inline output (convenience)
                        for (int j = 0; j < 64 && j < (int)bytes_needed; j++) {
                            resp.output[j] = slab_ptr[j];
                        }

                        resp.output_len = (bytes_needed < 64) ? bytes_needed : 64;  // Inline portion
                        resp.output_offset = slab_offset;
                        resp.output_bytes = bytes_needed;  // Signal: use slab
                        digests_this_req = n_msgs;
                    }
                } else {
                    // Single-message mode (existing behavior)
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        smoke::hash::sha256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                        resp.output_offset = 0;  // Card 26.25: Inline mode
                        resp.output_bytes = 0;
                        digests_this_req = 1;
                    }
                }

                // Track digests for telemetry
                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests, digests_this_req);
                }

                resp.t_complete_us = clock64();

                // Write response
                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;  // Card 26.25
                dst->output_bytes = resp.output_bytes;    // Card 26.25
                // Copy all 64 bytes for multi-message output (inline portion)
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            // Telemetry updates for hash ops (Card 26.19B: parity with P-256)
            // Card 26.22: Also track logical units (digests) completed
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests);
                __threadfence_system();
            }
        }
        // Card 26.31 + Card 26.59: Fast path for SHA-512 hash operations with multi-msg support
        else if (batch_opcode == static_cast<uint16_t>(OpCode::SHA512)) {
            // Track total digests for multi-message telemetry
            __shared__ uint32_t s_total_digests_sha512;
            if (tid == 0) s_total_digests_sha512 = 0;
            __syncthreads();

            // Per-thread SHA-512 hashing
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;

                // Card 26.59: Check for multi-message mode
                const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                uint32_t digests_this_req = 0;

                // Pre-compute response slot for potential slab write
                const uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                const uint32_t write_idx_pre = s_base_write_idx + i;
                const uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                if (is_multimsg) {
                    // Parse MultiMsgHeader from input[0..7]
                    const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                    const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                    const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                               ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                    // Determine payload source
                    const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                    const uint8_t* payload_data = use_extended_payload
                        ? (payload_slab + req.payload_offset)
                        : (req.input + MULTIMSG_HEADER_SIZE);
                    const uint32_t payload_avail = use_extended_payload
                        ? req.payload_len
                        : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                    // SHA-512: 64 bytes per digest
                    const uint32_t bytes_needed = n_msgs * 64;

                    // Validate multi-message constraints
                    bool valid = true;
                    if (n_msgs == 0) valid = false;
                    if (msg_len > MULTIMSG_MAX_MSG_LEN) valid = false;
                    if (hdr_flags != 0) valid = false;
                    if (n_msgs * msg_len > payload_avail) valid = false;
                    if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (bytes_needed <= 64) {
                        // Inline mode (N == 1 for SHA-512)
                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::sha512_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                resp.output + m * 64
                            );
                        }
                        resp.output_len = bytes_needed;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = n_msgs;
                    } else {
                        // Slab mode (N > 1)
                        uint32_t slab_offset = (slab_seg_base + (write_idx_pre % slab_segs_per_shard))
                                               * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* slab_ptr = output_slab + slab_offset;

                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::sha512_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                slab_ptr + m * 64
                            );
                        }

                        // Copy first 64 bytes to inline output
                        for (int j = 0; j < 64; j++) {
                            resp.output[j] = slab_ptr[j];
                        }

                        resp.output_len = 64;
                        resp.output_offset = slab_offset;
                        resp.output_bytes = bytes_needed;
                        digests_this_req = n_msgs;
                    }
                } else {
                    // Single-message mode
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        smoke::hash::sha512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = 1;
                    }
                }

                // Track digests for telemetry
                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests_sha512, digests_this_req);
                }

                resp.t_complete_us = clock64();

                // Write response
                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // N2-fix (2026-05-13): Card 26.72 / B-322 — write request_id LAST.
                // This branch was missed by the B-322 application pass; D-063
                // measured this as the root cause of cid_mismatches + partial
                // oracle pass under the sharded path. Pattern below copies
                // SHA-256's correct tail at lines 3733-3750.
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;

                // Copy 64-byte output
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            // Telemetry updates for SHA-512 ops
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_sha512);
                __threadfence_system();
            }
        }
        // Card 26.32 + Card 26.59: Fast path for SHA3-256 (Keccak) hash operations with multi-msg support
        else if (batch_opcode == static_cast<uint16_t>(OpCode::SHA3_256)) {
            // Track total digests for multi-message telemetry
            __shared__ uint32_t s_total_digests_sha3;
            if (tid == 0) s_total_digests_sha3 = 0;
            __syncthreads();

            // Per-thread SHA3-256 hashing
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;

                // Card 26.59: Check for multi-message mode
                const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                uint32_t digests_this_req = 0;

                // Pre-compute response slot for potential slab write
                const uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                const uint32_t write_idx_pre = s_base_write_idx + i;
                const uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                if (is_multimsg) {
                    // Parse MultiMsgHeader from input[0..7]
                    const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                    const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                    const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                               ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                    // Determine payload source
                    const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                    const uint8_t* payload_data = use_extended_payload
                        ? (payload_slab + req.payload_offset)
                        : (req.input + MULTIMSG_HEADER_SIZE);
                    const uint32_t payload_avail = use_extended_payload
                        ? req.payload_len
                        : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                    // SHA3-256: 32 bytes per digest
                    const uint32_t bytes_needed = n_msgs * 32;

                    // Validate multi-message constraints
                    bool valid = true;
                    if (n_msgs == 0) valid = false;
                    if (msg_len > MULTIMSG_MAX_MSG_LEN) valid = false;
                    if (hdr_flags != 0) valid = false;
                    if (n_msgs * msg_len > payload_avail) valid = false;
                    if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (bytes_needed <= 64) {
                        // Inline mode (N <= 2 for 32-byte digests)
                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::sha3_256_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                resp.output + m * 32
                            );
                        }
                        resp.output_len = bytes_needed;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = n_msgs;
                    } else {
                        // Slab mode (N > 2)
                        uint32_t slab_offset = (slab_seg_base + (write_idx_pre % slab_segs_per_shard))
                                               * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* slab_ptr = output_slab + slab_offset;

                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::sha3_256_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                slab_ptr + m * 32
                            );
                        }

                        // Copy first 64 bytes to inline output
                        for (int j = 0; j < 64 && j < (int)bytes_needed; j++) {
                            resp.output[j] = slab_ptr[j];
                        }

                        resp.output_len = (bytes_needed < 64) ? bytes_needed : 64;
                        resp.output_offset = slab_offset;
                        resp.output_bytes = bytes_needed;
                        digests_this_req = n_msgs;
                    }
                } else {
                    // Single-message mode
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        smoke::hash::sha3_256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = 1;
                    }
                }

                // Track digests for telemetry
                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests_sha3, digests_this_req);
                }

                resp.t_complete_us = clock64();

                // Write response
                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // N2-fix (2026-05-13): Card 26.72 / B-322 — write request_id LAST.
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;

                // Copy 64-byte output (for multi-msg inline portion)
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            // Telemetry updates for SHA3-256 ops
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_sha3);
                __threadfence_system();
            }
        }
        // Card 26.33 + Card 26.59: Fast path for BLAKE2b-256 hash operations with multi-msg support
        else if (batch_opcode == static_cast<uint16_t>(OpCode::BLAKE2B_256)) {
            // Track total digests for multi-message telemetry
            __shared__ uint32_t s_total_digests_blake2b;
            if (tid == 0) s_total_digests_blake2b = 0;
            __syncthreads();

            // Per-thread BLAKE2b-256 hashing
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;

                // Card 26.59: Check for multi-message mode
                const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                uint32_t digests_this_req = 0;

                // Pre-compute response slot for potential slab write
                const uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                const uint32_t write_idx_pre = s_base_write_idx + i;
                const uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                if (is_multimsg) {
                    // Parse MultiMsgHeader from input[0..7]
                    const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                    const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                    const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                               ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                    // Determine payload source
                    const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                    const uint8_t* payload_data = use_extended_payload
                        ? (payload_slab + req.payload_offset)
                        : (req.input + MULTIMSG_HEADER_SIZE);
                    const uint32_t payload_avail = use_extended_payload
                        ? req.payload_len
                        : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                    // BLAKE2b-256: 32 bytes per digest
                    const uint32_t bytes_needed = n_msgs * 32;

                    // Validate multi-message constraints
                    bool valid = true;
                    if (n_msgs == 0) valid = false;
                    if (msg_len > MULTIMSG_MAX_MSG_LEN) valid = false;
                    if (hdr_flags != 0) valid = false;
                    if (n_msgs * msg_len > payload_avail) valid = false;
                    if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (bytes_needed <= 64) {
                        // Inline mode (N <= 2 for 32-byte digests)
                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::blake2b_256_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                resp.output + m * 32
                            );
                        }
                        resp.output_len = bytes_needed;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = n_msgs;
                    } else {
                        // Slab mode (N > 2)
                        uint32_t slab_offset = (slab_seg_base + (write_idx_pre % slab_segs_per_shard))
                                               * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* slab_ptr = output_slab + slab_offset;

                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::blake2b_256_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                slab_ptr + m * 32
                            );
                        }

                        // Copy first 64 bytes to inline output
                        for (int j = 0; j < 64 && j < (int)bytes_needed; j++) {
                            resp.output[j] = slab_ptr[j];
                        }

                        resp.output_len = (bytes_needed < 64) ? bytes_needed : 64;
                        resp.output_offset = slab_offset;
                        resp.output_bytes = bytes_needed;
                        digests_this_req = n_msgs;
                    }
                } else {
                    // Single-message mode
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        smoke::hash::blake2b_256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = 1;
                    }
                }

                // Track digests for telemetry
                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests_blake2b, digests_this_req);
                }

                resp.t_complete_us = clock64();

                // Write response
                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // N2-fix (2026-05-13): Card 26.72 / B-322 — write request_id LAST.
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;

                // Copy 64-byte output (for multi-msg inline portion)
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            // Telemetry updates for BLAKE2b-256 ops
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_blake2b);
                __threadfence_system();
            }
        }
        // Card 140: Fast path for SHA3-512 hash operations with multi-msg support
        else if (batch_opcode == static_cast<uint16_t>(OpCode::SHA3_512)) {
            __shared__ uint32_t s_total_digests_sha3_512;
            if (tid == 0) s_total_digests_sha3_512 = 0;
            __syncthreads();

            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;

                const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                uint32_t digests_this_req = 0;

                const uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                const uint32_t write_idx_pre = s_base_write_idx + i;
                const uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                if (is_multimsg) {
                    const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                    const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                    const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                               ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                    const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                    const uint8_t* payload_data = use_extended_payload
                        ? (payload_slab + req.payload_offset)
                        : (req.input + MULTIMSG_HEADER_SIZE);
                    const uint32_t payload_avail = use_extended_payload
                        ? req.payload_len
                        : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                    // SHA3-512: 64 bytes per digest
                    const uint32_t bytes_needed = n_msgs * 64;

                    bool valid = true;
                    if (n_msgs == 0) valid = false;
                    if (msg_len > MULTIMSG_MAX_MSG_LEN) valid = false;
                    if (hdr_flags != 0) valid = false;
                    if (n_msgs * msg_len > payload_avail) valid = false;
                    if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (bytes_needed <= 64) {
                        // Inline mode (N == 1 for 64-byte digests)
                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::sha3_512_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                resp.output + m * 64
                            );
                        }
                        resp.output_len = bytes_needed;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = n_msgs;
                    } else {
                        // Slab mode (N > 1)
                        uint32_t slab_offset = (slab_seg_base + (write_idx_pre % slab_segs_per_shard))
                                               * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* slab_ptr = output_slab + slab_offset;

                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::sha3_512_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                slab_ptr + m * 64
                            );
                        }

                        for (int j = 0; j < 64 && j < (int)bytes_needed; j++) {
                            resp.output[j] = slab_ptr[j];
                        }

                        resp.output_len = (bytes_needed < 64) ? bytes_needed : 64;
                        resp.output_offset = slab_offset;
                        resp.output_bytes = bytes_needed;
                        digests_this_req = n_msgs;
                    }
                } else {
                    // Single-message mode
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        smoke::hash::sha3_512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = 1;
                    }
                }

                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests_sha3_512, digests_this_req);
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // N2-fix (2026-05-13): Card 26.72 / B-322 — write request_id LAST.
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;

                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_sha3_512);
                __threadfence_system();
            }
        }
        // Card 141: Fast path for BLAKE2b-512 hash operations with multi-msg support
        else if (batch_opcode == static_cast<uint16_t>(OpCode::BLAKE2B_512)) {
            __shared__ uint32_t s_total_digests_blake2b_512;
            if (tid == 0) s_total_digests_blake2b_512 = 0;
            __syncthreads();

            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;

                const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                uint32_t digests_this_req = 0;

                const uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                const uint32_t write_idx_pre = s_base_write_idx + i;
                const uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                if (is_multimsg) {
                    const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                    const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                    const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                               ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                    const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                    const uint8_t* payload_data = use_extended_payload
                        ? (payload_slab + req.payload_offset)
                        : (req.input + MULTIMSG_HEADER_SIZE);
                    const uint32_t payload_avail = use_extended_payload
                        ? req.payload_len
                        : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                    // BLAKE2b-512: 64 bytes per digest
                    const uint32_t bytes_needed = n_msgs * 64;

                    bool valid = true;
                    if (n_msgs == 0) valid = false;
                    if (msg_len > MULTIMSG_MAX_MSG_LEN) valid = false;
                    if (hdr_flags != 0) valid = false;
                    if (n_msgs * msg_len > payload_avail) valid = false;
                    if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (bytes_needed <= 64) {
                        // Inline mode (N == 1 for 64-byte digests)
                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::blake2b_512_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                resp.output + m * 64
                            );
                        }
                        resp.output_len = bytes_needed;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = n_msgs;
                    } else {
                        // Slab mode (N > 1)
                        uint32_t slab_offset = (slab_seg_base + (write_idx_pre % slab_segs_per_shard))
                                               * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* slab_ptr = output_slab + slab_offset;

                        for (uint16_t m = 0; m < n_msgs; m++) {
                            smoke::hash::blake2b_512_hash_short(
                                payload_data + m * msg_len,
                                msg_len,
                                slab_ptr + m * 64
                            );
                        }

                        for (int j = 0; j < 64 && j < (int)bytes_needed; j++) {
                            resp.output[j] = slab_ptr[j];
                        }

                        resp.output_len = (bytes_needed < 64) ? bytes_needed : 64;
                        resp.output_offset = slab_offset;
                        resp.output_bytes = bytes_needed;
                        digests_this_req = n_msgs;
                    }
                } else {
                    // Single-message mode
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        smoke::hash::blake2b_512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = 1;
                    }
                }

                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests_blake2b_512, digests_this_req);
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // N2-fix (2026-05-13): Card 26.72 / B-322 — write request_id LAST.
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;

                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_blake2b_512);
                __threadfence_system();
            }
        }
        // Card 26.29: Fast path for HMAC-SHA256 keyed hash operations
        else if (batch_opcode == static_cast<uint16_t>(OpCode::HMAC_SHA256)) {
            // Track total digests for multi-message telemetry
            __shared__ uint32_t s_total_digests_hmac;
            if (tid == 0) s_total_digests_hmac = 0;
            __syncthreads();

            // Per-thread HMAC-SHA256 hashing
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];
                const Keyslot ks = keyslots[req.key_slot];  // Use per-request keyslot

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;

                // Check for multi-message mode
                const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                uint32_t digests_this_req = 0;

                // Pre-compute response slot for slab write
                uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                uint32_t write_idx_pre = s_base_write_idx + i;
                uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                if (is_multimsg) {
                    // Parse MultiMsgHeader from input[0..7]
                    const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                    const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                    const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                               ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                    // Determine payload source
                    const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                    const uint8_t* payload_data = use_extended_payload
                        ? (payload_slab + req.payload_offset)
                        : (req.input + MULTIMSG_HEADER_SIZE);
                    const uint32_t payload_avail = use_extended_payload
                        ? req.payload_len
                        : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                    const uint32_t bytes_needed = n_msgs * 32;

                    // Validate constraints
                    bool valid = true;
                    if (n_msgs == 0) valid = false;
                    if (msg_len > HMAC_SHA256_MAX_MSG_LEN) valid = false;
                    if (hdr_flags != 0) valid = false;
                    if (n_msgs * msg_len > payload_avail) valid = false;
                    if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (bytes_needed <= 64) {
                        // Inline mode (N <= 2)
                        for (uint16_t m = 0; m < n_msgs; m++) {
                            hmac_sha256_device(
                                ks.private_d,
                                32,
                                payload_data + m * msg_len,
                                msg_len,
                                resp.output + m * 32
                            );
                        }
                        resp.output_len = bytes_needed;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = n_msgs;
                    } else {
                        // Slab mode (N > 2)
                        uint32_t slab_offset = (slab_seg_base + (write_idx_pre % slab_segs_per_shard))
                                               * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* slab_ptr = output_slab + slab_offset;

                        for (uint16_t m = 0; m < n_msgs; m++) {
                            hmac_sha256_device(
                                ks.private_d,
                                32,
                                payload_data + m * msg_len,
                                msg_len,
                                slab_ptr + m * 32
                            );
                        }

                        for (int j = 0; j < 64 && j < (int)bytes_needed; j++) {
                            resp.output[j] = slab_ptr[j];
                        }

                        resp.output_len = (bytes_needed < 64) ? bytes_needed : 64;
                        resp.output_offset = slab_offset;
                        resp.output_bytes = bytes_needed;
                        digests_this_req = n_msgs;
                    }
                } else {
                    // Single-message mode
                    if (req.input_len > HMAC_SHA256_MAX_MSG_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        hmac_sha256_device(
                            ks.private_d,
                            32,
                            req.input,
                            req.input_len,
                            resp.output
                        );
                        resp.output_len = 32;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = 1;
                    }
                }

                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests_hmac, digests_this_req);
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            // Telemetry updates for HMAC ops
            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_hmac);
                __threadfence_system();
            }
        }
        // ATTEST-NATIVE-01: variable-length HMAC-SHA256 over the payload slab.
        // One digest per request; 32-byte digest written INLINE to resp.output
        // (≤64 → no output slab, no output-collision surface). The streaming
        // device HMAC (hmac_sha256_device, no 1024B cap) handles KB-scale tool
        // payloads; fail-closed on a missing slab or an out-of-bounds range.
        else if (batch_opcode == static_cast<uint16_t>(OpCode::HMAC_SHA256_VARLEN)) {
            __shared__ uint32_t s_total_digests_hmacv;
            if (tid == 0) s_total_digests_hmacv = 0;
            __syncthreads();

            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];
                const Keyslot ks = keyslots[req.key_slot];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;
                resp.output_offset = 0;
                resp.output_bytes = 0;
                // Fail-closed: error responses return zeroed output, never stale
                // stack bytes; the valid path fills [0..31], leaving [32..63]=0.
                #pragma unroll
                for (int j = 0; j < 64; j++) resp.output[j] = 0;

                uint32_t digests_this_req = 0;

                // Empty message (payload_len == 0) is a VALID HMAC input; only a
                // non-empty message must have a slab and an in-bounds range.
                // Fail-closed otherwise — never read past the slab.
                const uint64_t end = (uint64_t)req.payload_offset + (uint64_t)req.payload_len;
                const bool bad = (req.payload_len > 0) &&
                                 (payload_slab == nullptr ||
                                  end > (uint64_t)FAST_PATH_PAYLOAD_SLAB_SIZE);
                if (bad) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else {
                    // For payload_len == 0 the streaming HMAC reads zero message
                    // blocks, so msg_ptr is never dereferenced.
                    const uint8_t* msg_ptr = (payload_slab != nullptr)
                        ? (payload_slab + req.payload_offset) : payload_slab;
                    hmac_sha256_device(
                        ks.private_d, 32, msg_ptr, req.payload_len, resp.output);
                    resp.output_len = 32;
                    digests_this_req = 1;
                }

                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests_hmacv, digests_this_req);
                }

                resp.t_complete_us = clock64();
                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;
                dst->slot_used = resp.slot_used;
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_hmacv);
                __threadfence_system();
            }
        }
        // Card 26.51 + Card 26.59: HMAC-SHA512 batch processing with multi-msg support
        else if (batch_opcode == static_cast<uint16_t>(OpCode::HMAC_SHA512)) {
            // Track total digests for multi-message telemetry
            __shared__ uint32_t s_total_digests_hmac512;
            if (tid == 0) s_total_digests_hmac512 = 0;
            __syncthreads();

            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];
                const Keyslot ks = keyslots[req.key_slot];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;

                // Card 26.59: Check for multi-message mode
                const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                uint32_t digests_this_req = 0;

                // Pre-compute response slot for slab write
                uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                uint32_t write_idx_pre = s_base_write_idx + i;
                uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                if (is_multimsg) {
                    // Parse MultiMsgHeader from input[0..7]
                    const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                    const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                    const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                               ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                    // Determine payload source
                    const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                    const uint8_t* payload_data = use_extended_payload
                        ? (payload_slab + req.payload_offset)
                        : (req.input + MULTIMSG_HEADER_SIZE);
                    const uint32_t payload_avail = use_extended_payload
                        ? req.payload_len
                        : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                    // HMAC-SHA512: 64 bytes per digest
                    const uint32_t bytes_needed = n_msgs * 64;

                    // Validate constraints
                    bool valid = true;
                    if (n_msgs == 0) valid = false;
                    if (msg_len > HMAC_SHA512_MAX_MSG_LEN) valid = false;
                    if (hdr_flags != 0) valid = false;
                    if (n_msgs * msg_len > payload_avail) valid = false;
                    if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (bytes_needed <= 64) {
                        // Inline mode (N == 1 for 64-byte digests)
                        for (uint16_t m = 0; m < n_msgs; m++) {
                            hmac_sha512_device(
                                ks.private_d,
                                32,
                                payload_data + m * msg_len,
                                msg_len,
                                resp.output + m * 64
                            );
                        }
                        resp.output_len = bytes_needed;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = n_msgs;
                    } else {
                        // Slab mode (N > 1)
                        uint32_t slab_offset = (slab_seg_base + (write_idx_pre % slab_segs_per_shard))
                                               * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* slab_ptr = output_slab + slab_offset;

                        for (uint16_t m = 0; m < n_msgs; m++) {
                            hmac_sha512_device(
                                ks.private_d,
                                32,
                                payload_data + m * msg_len,
                                msg_len,
                                slab_ptr + m * 64
                            );
                        }

                        // Copy first 64 bytes to inline output
                        for (int j = 0; j < 64; j++) {
                            resp.output[j] = slab_ptr[j];
                        }

                        resp.output_len = 64;
                        resp.output_offset = slab_offset;
                        resp.output_bytes = bytes_needed;
                        digests_this_req = n_msgs;
                    }
                } else {
                    // Single-message mode
                    if (req.input_len > HMAC_SHA512_MAX_MSG_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        hmac_sha512_device(
                            ks.private_d,
                            32,
                            req.input,
                            req.input_len,
                            resp.output
                        );
                        resp.output_len = 64;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = 1;
                    }
                }

                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests_hmac512, digests_this_req);
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_hmac512);
                __threadfence_system();
            }
        }
        // Card 26.52 + Card 26.59: HMAC-SHA384 batch processing with multi-msg support
        else if (batch_opcode == static_cast<uint16_t>(OpCode::HMAC_SHA384)) {
            // Track total digests for multi-message telemetry
            __shared__ uint32_t s_total_digests_hmac384;
            if (tid == 0) s_total_digests_hmac384 = 0;
            __syncthreads();

            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];
                const Keyslot ks = keyslots[req.key_slot];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;

                // Card 26.59: Check for multi-message mode
                const bool is_multimsg = (req.flags & FAST_PATH_FLAG_MULTIMSG) != 0;
                uint32_t digests_this_req = 0;

                // Pre-compute response slot for slab write
                uint32_t cq_cap_pre = telemetry->cq_capacity_active;
                uint32_t write_idx_pre = s_base_write_idx + i;
                uint32_t resp_slot_pre = write_idx_pre % cq_cap_pre;

                if (is_multimsg) {
                    // Parse MultiMsgHeader from input[0..7]
                    const uint16_t n_msgs = (uint16_t)req.input[0] | ((uint16_t)req.input[1] << 8);
                    const uint16_t msg_len = (uint16_t)req.input[2] | ((uint16_t)req.input[3] << 8);
                    const uint32_t hdr_flags = (uint32_t)req.input[4] | ((uint32_t)req.input[5] << 8) |
                                               ((uint32_t)req.input[6] << 16) | ((uint32_t)req.input[7] << 24);

                    // Determine payload source
                    const bool use_extended_payload = (req.payload_len > 0) && (payload_slab != nullptr);
                    const uint8_t* payload_data = use_extended_payload
                        ? (payload_slab + req.payload_offset)
                        : (req.input + MULTIMSG_HEADER_SIZE);
                    const uint32_t payload_avail = use_extended_payload
                        ? req.payload_len
                        : (req.input_len > MULTIMSG_HEADER_SIZE ? req.input_len - MULTIMSG_HEADER_SIZE : 0);

                    // HMAC-SHA384: 48 bytes per digest
                    const uint32_t bytes_needed = n_msgs * 48;

                    // Validate constraints
                    bool valid = true;
                    if (n_msgs == 0) valid = false;
                    if (msg_len > HMAC_SHA384_MAX_MSG_LEN) valid = false;
                    if (hdr_flags != 0) valid = false;
                    if (n_msgs * msg_len > payload_avail) valid = false;
                    if (bytes_needed > FAST_PATH_OUTPUT_SEGMENT_BYTES) valid = false;

                    if (!valid) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else if (bytes_needed <= 64) {
                        // Inline mode (N == 1 for 48-byte digests)
                        for (uint16_t m = 0; m < n_msgs; m++) {
                            hmac_sha384_device(
                                ks.private_d,
                                32,
                                payload_data + m * msg_len,
                                msg_len,
                                resp.output + m * 48
                            );
                        }
                        resp.output_len = bytes_needed;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = n_msgs;
                    } else {
                        // Slab mode (N > 1)
                        uint32_t slab_offset = (slab_seg_base + (write_idx_pre % slab_segs_per_shard))
                                               * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* slab_ptr = output_slab + slab_offset;

                        for (uint16_t m = 0; m < n_msgs; m++) {
                            hmac_sha384_device(
                                ks.private_d,
                                32,
                                payload_data + m * msg_len,
                                msg_len,
                                slab_ptr + m * 48
                            );
                        }

                        // Copy first 64 bytes to inline output
                        for (int j = 0; j < 64 && j < (int)bytes_needed; j++) {
                            resp.output[j] = slab_ptr[j];
                        }

                        resp.output_len = (bytes_needed < 64) ? bytes_needed : 64;
                        resp.output_offset = slab_offset;
                        resp.output_bytes = bytes_needed;
                        digests_this_req = n_msgs;
                    }
                } else {
                    // Single-message mode
                    if (req.input_len > HMAC_SHA384_MAX_MSG_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                    } else {
                        hmac_sha384_device(
                            ks.private_d,
                            32,
                            req.input,
                            req.input_len,
                            resp.output
                        );
                        resp.output_len = 48;
                        resp.output_offset = 0;
                        resp.output_bytes = 0;
                        digests_this_req = 1;
                    }
                }

                if (digests_this_req > 0) {
                    atomicAdd(&s_total_digests_hmac384, digests_this_req);
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)s_total_digests_hmac384);
                __threadfence_system();
            }
        }
        // Card 26.54: AES-256-CTR batch processing
        else if (batch_opcode == static_cast<uint16_t>(OpCode::AES256_CTR)) {
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];
                const Keyslot ks = keyslots[req.key_slot];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;
                resp.output_offset = 0;
                resp.output_bytes = 0;

                // Validate input: must have at least nonce (12 bytes), max 32 bytes
                if (req.input_len < AES256_CTR_NONCE_LEN) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else if (req.input_len > 32) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else {
                    // Card 26.64: Use precomputed round keys from keyslot
                    smoke::aes::aes256_ctr_encrypt_short_cached(
                        ks.aes_roundkeys,
                        req.input,
                        req.input_len,
                        resp.output
                    );
                    resp.output_len = req.input_len - AES256_CTR_NONCE_LEN;
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                __threadfence_system();
            }
        }
        // Card 26.65: AES-256-GCM authenticated encryption batch processing
        else if (batch_opcode == static_cast<uint16_t>(OpCode::AES256_GCM)) {
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.opcode = req.opcode;
                resp.epoch_used = 0xFF;
                resp.slot_used = req.key_slot;
                resp.output_len = 0;
                resp.output_offset = 0;
                resp.output_bytes = 0;

                resp.status = validate_aead_keyslot(keyslots, req.key_slot);
                if (resp.status == 0 && req.payload_len > 0) {
                    if (payload_slab == nullptr || output_slab == nullptr) {
                        resp.status = static_cast<uint8_t>(OpStatus::INTERNAL_ERROR);
                    } else if (req.input_len != 12 ||
                               !valid_payload_slab_range(
                                   req.payload_offset, req.payload_len)) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        const uint32_t write_pos = s_base_write_idx + i;
                        const uint32_t lag = write_pos - responses->read_idx;
                        if (seal_segs_per_shard == 0 ||
                            lag >= seal_segs_per_shard) {
                            resp.status = static_cast<uint8_t>(OpStatus::SLAB_OVERRUN);
                        } else {
                            const uint32_t seal_seg =
                                seal_seg_base + (write_pos % seal_segs_per_shard);
                            const uint32_t out_offset =
                                FAST_PATH_SEAL_OUTPUT_REGION_OFFSET +
                                seal_seg * FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES;
                            uint32_t out_len = 0;
                            const Keyslot& ks = keyslots[req.key_slot];
                            smoke::aes::aes256_gcm_seal_slab(
                                ks.aes_roundkeys, req.input,
                                payload_slab, req.payload_offset, req.payload_len,
                                output_slab, out_offset, &out_len);
                            if (out_len != req.payload_len + FAST_PATH_SEAL_TAG_BYTES) {
                                resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                            } else {
                                resp.output_offset = out_offset;
                                resp.output_bytes = out_len;
                            }
                        }
                    }
                } else if (resp.status == 0 &&
                           (req.input_len < 12 || req.input_len > 28)) {
                    // input_len == 12 is a valid 0-byte plaintext seal (D-096).
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else if (resp.status == 0) {
                    uint16_t out_len = 0;
                    const Keyslot& ks = keyslots[req.key_slot];
                    smoke::aes::aes256_gcm_seal_short(
                        ks.aes_roundkeys,
                        req.input,
                        req.input_len,
                        resp.output,
                        &out_len
                    );
                    resp.output_len = out_len;
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->opcode = resp.opcode;
                dst->epoch_used = resp.epoch_used;
                dst->slot_used = resp.slot_used;
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                __threadfence_system();
            }
        }
        // DECRYPT-ABI-01: AES-256-GCM Open (decrypt) — shard path
        else if (batch_opcode == static_cast<uint16_t>(OpCode::AES256_GCM_OPEN)) {
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;
                resp.output_offset = 0;
                resp.output_bytes = 0;
                resp.status = validate_aead_keyslot(keyslots, req.key_slot);

                // DECRYPT-SLAB-01: AES-256-GCM Open — shard path (inline + slab)
                if (resp.status == 0 && req.payload_len > 0) {
                    // SLAB PATH
                    if (payload_slab == nullptr || output_slab == nullptr) {
                        resp.status = static_cast<uint8_t>(OpStatus::INTERNAL_ERROR);
                    } else if (req.input_len != 12 ||
                               !valid_open_payload_slab_range(
                                   req.payload_offset, req.payload_len)) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        const uint32_t write_pos = s_base_write_idx + i;
                        const uint32_t lag = write_pos - responses->read_idx;
                        if (seal_segs_per_shard == 0 || lag >= seal_segs_per_shard) {
                            resp.status = static_cast<uint8_t>(OpStatus::SLAB_OVERRUN);
                        } else {
                            uint32_t out_len = 0;
                            uint8_t auth_ok = 0;
                            const uint32_t out_offset =
                                FAST_PATH_SEAL_OUTPUT_REGION_OFFSET +
                                (seal_seg_base + (write_pos % seal_segs_per_shard))
                                * FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES;
                            const Keyslot& ks = keyslots[req.key_slot];
                            smoke::aes::aes256_gcm_open_slab(
                                ks.aes_roundkeys,
                                req.input,
                                payload_slab, req.payload_offset, req.payload_len,
                                output_slab, out_offset, &out_len, &auth_ok
                            );
                            if (auth_ok &&
                                out_len <= FAST_PATH_SEAL_MAX_PLAINTEXT) {
                                resp.output_offset = out_offset;
                                resp.output_bytes = out_len;
                            } else {
                                resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                                atomicAdd((unsigned long long*)&telemetry->total_auth_failures, 1ULL);
                            }
                        }
                    }
                } else if (resp.status == 0 && req.input_len < 28) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else if (resp.status == 0 && req.input_len > 32) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else if (resp.status == 0) {
                    // INLINE PATH
                    uint16_t out_len = 0;
                    uint8_t auth_ok = 0;
                    const Keyslot& ks = keyslots[req.key_slot];
                    smoke::aes::aes256_gcm_open_short(
                        ks.aes_roundkeys, req.input, req.input_len,
                        resp.output, &out_len, &auth_ok
                    );
                    if (auth_ok) {
                        resp.status = 0;
                        resp.output_len = out_len;
                    } else {
                        resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                        resp.output_len = 0;
                        for (int z = 0; z < 64; z++) resp.output[z] = 0;
                        atomicAdd((unsigned long long*)&telemetry->total_auth_failures, 1ULL);
                    }
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->opcode = req.opcode;  // VERIFY-DECRYPT-STACK-01: set opcode
                dst->epoch_used = 0xFF;
                dst->slot_used = req.key_slot;
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->total_decrypt_aes_gcm, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                __threadfence_system();
            }
        }
        // CHACHA-OPEN-01: ChaCha20-Poly1305 Open (decrypt) — shard path
        else if (batch_opcode == static_cast<uint16_t>(OpCode::CHACHA20_POLY1305_OPEN)) {
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;
                resp.output_offset = 0;
                resp.output_bytes = 0;
                resp.status = validate_aead_keyslot(keyslots, req.key_slot);

                if (resp.status == 0 && req.payload_len > 0) {
                    if (payload_slab == nullptr || output_slab == nullptr) {
                        resp.status = static_cast<uint8_t>(OpStatus::INTERNAL_ERROR);
                    } else if (req.input_len != 12 ||
                               !valid_open_payload_slab_range(
                                   req.payload_offset, req.payload_len)) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        const uint32_t write_pos = s_base_write_idx + i;
                        const uint32_t lag = write_pos - responses->read_idx;
                        if (seal_segs_per_shard == 0 || lag >= seal_segs_per_shard) {
                            resp.status = static_cast<uint8_t>(OpStatus::SLAB_OVERRUN);
                        } else {
                            uint32_t out_len = 0;
                            uint8_t auth_ok = 0;
                            const uint32_t out_offset =
                                FAST_PATH_SEAL_OUTPUT_REGION_OFFSET +
                                (seal_seg_base + (write_pos % seal_segs_per_shard))
                                * FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES;
                            const Keyslot& ks = keyslots[req.key_slot];
                            smoke::aead::chacha20_poly1305_decrypt_slab(
                                ks.private_d, req.input,
                                payload_slab, req.payload_offset, req.payload_len,
                                output_slab, out_offset, &out_len, &auth_ok
                            );
                            if (auth_ok &&
                                out_len <= FAST_PATH_SEAL_MAX_PLAINTEXT) {
                                resp.output_offset = out_offset;
                                resp.output_bytes = out_len;
                            } else {
                                resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                                atomicAdd((unsigned long long*)&telemetry->total_auth_failures, 1ULL);
                            }
                        }
                    }
                } else if (resp.status == 0 && req.input_len < 28) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else if (resp.status == 0 && req.input_len > 32) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else if (resp.status == 0) {
                    uint16_t out_len = 0;
                    uint8_t auth_ok = 0;
                    const Keyslot& ks = keyslots[req.key_slot];
                    smoke::aead::chacha20_poly1305_decrypt_short_keyslot(
                        ks.private_d, req.input, req.input_len,
                        resp.output, &out_len, &auth_ok
                    );
                    if (auth_ok) {
                        resp.status = 0;
                        resp.output_len = out_len;
                    } else {
                        resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                        resp.output_len = 0;
                        for (int z = 0; z < 64; z++) resp.output[z] = 0;
                        atomicAdd((unsigned long long*)&telemetry->total_auth_failures, 1ULL);
                    }
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->opcode = req.opcode;  // VERIFY-DECRYPT-STACK-01: set opcode
                dst->epoch_used = 0xFF;
                dst->slot_used = req.key_slot;
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->total_decrypt_chacha20, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                __threadfence_system();
            }
        }
        // Card 26.66: ChaCha20 stream cipher batch processing
        else if (batch_opcode == static_cast<uint16_t>(OpCode::CHACHA20)) {
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];
                const Keyslot ks = keyslots[req.key_slot];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;
                resp.output_offset = 0;
                resp.output_bytes = 0;

                // Validate input: must have at least nonce (12 bytes), max 32 bytes
                if (req.input_len < CHACHA20_NONCE_LEN) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else if (req.input_len > 32) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else {
                    // Card 26.66: ChaCha20 encrypt using key from keyslot
                    smoke::chacha::chacha20_encrypt_short_keyslot(
                        ks.private_d,       // key (32 bytes from keyslot)
                        req.input,          // nonce[12] + plaintext[N]
                        req.input_len,      // total input length
                        resp.output,        // ciphertext output
                        &resp.output_len    // output length
                    );
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                __threadfence_system();
            }
        }
        // Card 26.67: Poly1305 MAC batch processing
        else if (batch_opcode == static_cast<uint16_t>(OpCode::POLY1305)) {
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];
                const Keyslot ks = keyslots[req.key_slot];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;
                resp.output_offset = 0;
                resp.output_bytes = 0;

                // Validate input: max 32 bytes message
                if (req.input_len > 32) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else {
                    // Card 26.67: Poly1305 MAC using key from keyslot
                    smoke::poly1305::poly1305_mac_short_keyslot(
                        ks.private_d,       // key (32 bytes from keyslot)
                        req.input,          // message
                        req.input_len,      // message length
                        resp.output,        // 16-byte tag output
                        &resp.output_len    // output length
                    );
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                __threadfence_system();
            }
        }
        // Card 26.68: ChaCha20-Poly1305 AEAD batch processing
        else if (batch_opcode == static_cast<uint16_t>(OpCode::CHACHA20_POLY1305)) {
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.opcode = req.opcode;
                resp.epoch_used = 0xFF;
                resp.slot_used = req.key_slot;
                resp.output_len = 0;
                resp.output_offset = 0;
                resp.output_bytes = 0;

                uint16_t input_len = req.input_len;

                resp.status = validate_aead_keyslot(keyslots, req.key_slot);
                if (resp.status == 0 && req.payload_len > 0) {
                    if (payload_slab == nullptr || output_slab == nullptr) {
                        resp.status = static_cast<uint8_t>(OpStatus::INTERNAL_ERROR);
                    } else if (input_len != 12 ||
                               !valid_payload_slab_range(
                                   req.payload_offset, req.payload_len)) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        const uint32_t write_pos = s_base_write_idx + i;
                        const uint32_t lag = write_pos - responses->read_idx;
                        if (seal_segs_per_shard == 0 ||
                            lag >= seal_segs_per_shard) {
                            resp.status = static_cast<uint8_t>(OpStatus::SLAB_OVERRUN);
                        } else {
                            const uint32_t seal_seg =
                                seal_seg_base + (write_pos % seal_segs_per_shard);
                            const uint32_t out_offset =
                                FAST_PATH_SEAL_OUTPUT_REGION_OFFSET +
                                seal_seg * FAST_PATH_SEAL_OUTPUT_SEGMENT_BYTES;
                            uint32_t out_len = 0;
                            const Keyslot& ks = keyslots[req.key_slot];
                            smoke::aead::chacha20_poly1305_encrypt_slab(
                                ks.private_d, req.input,
                                payload_slab, req.payload_offset, req.payload_len,
                                output_slab, out_offset, &out_len);
                            if (out_len != req.payload_len + FAST_PATH_SEAL_TAG_BYTES) {
                                resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                            } else {
                                resp.output_offset = out_offset;
                                resp.output_bytes = out_len;
                            }
                        }
                    }
                } else if (resp.status == 0 &&
                           (input_len < 12 || input_len > 28)) {
                    // input_len == 12 is a valid 0-byte plaintext seal (D-096).
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                } else if (resp.status == 0) {
                    const Keyslot& ks = keyslots[req.key_slot];
                    smoke::aead::chacha20_poly1305_encrypt_short_keyslot(
                        ks.private_d,       // 32-byte key
                        req.input,          // nonce + plaintext
                        input_len,          // input length
                        resp.output,        // ciphertext + tag
                        &resp.output_len    // output length
                    );
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->opcode = resp.opcode;
                dst->epoch_used = resp.epoch_used;
                dst->slot_used = resp.slot_used;
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                __threadfence_system();
            }
        }
        // Card 26.69: HKDF-SHA256 key derivation batch processing
        else if (batch_opcode == static_cast<uint16_t>(OpCode::HKDF_SHA256)) {
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];
                const Keyslot ks = keyslots[req.key_slot];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                resp.output_len = 0;
                resp.output_offset = 0;
                resp.output_bytes = 0;

                uint16_t input_len = req.input_len;

                // HKDF-SHA256: input = IKM (32-64 bytes), keyslot = salt
                if (input_len < 32 || input_len > 64) {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    resp.output_len = 0;
                } else {
                    // Derive 32-byte key using HKDF-SHA256
                    smoke::kdf::hkdf_sha256_keyslot(
                        ks.private_d,       // 32-byte salt
                        req.input,          // IKM
                        input_len,          // IKM length
                        resp.output,        // OKM (32 bytes)
                        &resp.output_len    // output length
                    );
                }

                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                // Card 26.72: Write all fields except request_id first
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                dst->output_len = resp.output_len;
                dst->output_offset = resp.output_offset;
                dst->output_bytes = resp.output_bytes;
                for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                __threadfence_system();
                commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
            }
            __syncthreads();

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                __threadfence_system();
            }
        }
        // Card 26.19: Unknown opcode handling.
        // NOTE: the host-mediated PQ signing opcodes — MLDSA_SIGN (50,
        // Card 27.19) and SLH_DSA_SIGN (53, Card SLH-002 P2) — are
        // deliberately NOT in the known list: their batch kernels cannot run
        // inside a persistent kernel occupying all SMs, so ring submissions
        // of those opcodes are refused here with INVALID_OPCODE (fail-closed).
        // Use fast_path_submit_mldsa / fast_path_submit_slh instead.
        //
        // RRP-A06b: RSA_2048_SIGN (1) is refused here too. The real RSA
        // handler exists only in the ENABLE_LEGACY_KERNEL dead code — this
        // production kernel had a KEY_NOT_LOADED stub that reserved its own
        // response slot with a second atomicAdd(write_idx) AFTER the batch
        // path had already reserved s_base_write_idx for the whole batch, so
        // its response landed one batch ahead of the drain cursor and the
        // pre-reserved slot never committed: the host drain stalled forever
        // on the uncommitted slot ("kernel never responds", L40S 2026-08-13).
        // The stub is deleted; RSA falls into this branch, which uses the
        // batch-reserved slots correctly. The ABI additionally refuses RSA
        // at smoke_generic_submit / smoke_load_key_rsa2048_crt (earliest
        // gate); this is the kernel-side backstop.
        else if (batch_opcode != 0 && batch_opcode != static_cast<uint16_t>(OpCode::P256_SIGN)
                 && batch_opcode != static_cast<uint16_t>(OpCode::SHA256)
                 && batch_opcode != static_cast<uint16_t>(OpCode::SHA512)
                 && batch_opcode != static_cast<uint16_t>(OpCode::SHA3_256)
                 && batch_opcode != static_cast<uint16_t>(OpCode::BLAKE2B_256)
                 && batch_opcode != static_cast<uint16_t>(OpCode::SHA3_512)
                 && batch_opcode != static_cast<uint16_t>(OpCode::BLAKE2B_512)
                 && batch_opcode != static_cast<uint16_t>(OpCode::HMAC_SHA256)
                 && batch_opcode != static_cast<uint16_t>(OpCode::HMAC_SHA512)
                 && batch_opcode != static_cast<uint16_t>(OpCode::HMAC_SHA384)
                 && batch_opcode != static_cast<uint16_t>(OpCode::AES256_CTR)
                 && batch_opcode != static_cast<uint16_t>(OpCode::AES256_GCM)
                 && batch_opcode != static_cast<uint16_t>(OpCode::CHACHA20)
                 && batch_opcode != static_cast<uint16_t>(OpCode::POLY1305)
                 && batch_opcode != static_cast<uint16_t>(OpCode::CHACHA20_POLY1305)
                 && batch_opcode != static_cast<uint16_t>(OpCode::HKDF_SHA256)
                 && batch_opcode != static_cast<uint16_t>(OpCode::AES256_GCM_OPEN)
                 && batch_opcode != static_cast<uint16_t>(OpCode::CHACHA20_POLY1305_OPEN)) {
            // Unknown opcode - return error for all requests in batch
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = static_cast<uint8_t>(OpStatus::INVALID_OPCODE);
                resp.client_id = req.client_id;
                resp.output_len = 0;  // B-319b: was never set on this error path (stack garbage)
                resp.t_complete_us = clock64();

                const uint32_t cq_cap = telemetry->cq_capacity_active;
                const uint64_t seq = s_base_seq + (uint64_t)i;
                const uint32_t write_idx = s_base_write_idx + i;
                const uint32_t resp_slot = write_idx % cq_cap;
                resp.seq = seq;

                volatile FastPathResponse* dst = &responses->responses[resp_slot];
                dst->seq = resp.seq;
                dst->t_submit_us = resp.t_submit_us;
                dst->t_dequeue_us = resp.t_dequeue_us;
                dst->t_complete_us = resp.t_complete_us;
                dst->status = resp.status;
                dst->client_id = resp.client_id;
                dst->epoch_used = resp.epoch_used;    // Card 66
                dst->slot_used = resp.slot_used;      // Card 66
                // B-319b: this site previously wrote request_id FIRST (before
                // the payload, no fence) — a pre-existing Card 26.72 protocol
                // violation on this legacy error path. Commit is now last.
                commit_response(dst, resp.request_id, resp.status, resp.output_len);
            }
            __syncthreads();

            if (tid == 0) {
                // RRP-A06b: this branch never bumped debug_responses_written /
                // dbg_units_completed, so (a) the B-311 fast-empty-poll exit
                // (written_now == last_seen) could short-circuit BEFORE the
                // host ever drained an INVALID_OPCODE response — a refusal the
                // caller never sees is not fail-closed — and (b) accounting
                // audits under-counted completions for refused ops. Mirror the
                // other dispatch branches.
                atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                atomicAdd((unsigned long long*)&telemetry->outcome_gpu_error, (unsigned long long)batch);
                atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)batch);
                __threadfence_system();
            }
        }
        // P256_SIGN (opcode 0): Use existing warp-coop signing path
        else {
            // Card 26.34: Check if this batch uses multi-sign mode
            // Check first request - assume homogeneous batch
            const bool batch_is_multisign = (s_batch[0].flags & FAST_PATH_FLAG_MULTIMSG) != 0
                                            && payload_slab != nullptr && output_slab != nullptr;

            if (batch_is_multisign) {
                // Card 26.34: P256 Multi-Sign path (work amplification)
                // Uses simple per-request signing, not warp-coop, because we need to loop
                // over N hashes per request.
                for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                    const FastPathRequest req = s_batch[i];
                    const Keyslot ks = keyslots[req.key_slot];

                    FastPathResponse resp;
                    resp.request_id = req.request_id;
                    resp.t_submit_us = req.t_submit_us;
                    resp.t_dequeue_us = s_dequeue_cycles;
                    resp.status = 0;
                    resp.client_id = req.client_id;
                    // Card 66: Record epoch/slot used for P256 multi-sign
                    resp.epoch_used = static_cast<uint8_t>((*active_epoch) & 0xFF);
                    resp.slot_used  = static_cast<uint8_t>(req.key_slot);

                    // Parse header from input[0..7]
                    const MultiMsgHeader* hdr = reinterpret_cast<const MultiMsgHeader*>(req.input);
                    const uint16_t n_sigs = hdr->n_msgs;
                    const uint16_t item_len = hdr->msg_len;

                    // Validate: item_len must be 32, n_sigs in [1, P256_MULTISIGN_MAX_N]
                    if (item_len != P256_MULTISIGN_ITEM_LEN || n_sigs == 0 || n_sigs > P256_MULTISIGN_MAX_N) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_MULTIMSG);
                    }
                    else if (req.payload_len < n_sigs * P256_MULTISIGN_ITEM_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    }
                    else {
                        // Read hashes from payload_slab
                        const uint8_t* hash_data = payload_slab + req.payload_offset;

                        // Get shard-partitioned output slab segment for this response
                        const uint32_t write_idx_ms = s_base_write_idx + i;
                        const uint32_t slab_offset = (slab_seg_base + (write_idx_ms % slab_segs_per_shard))
                                                     * FAST_PATH_OUTPUT_SEGMENT_BYTES;
                        uint8_t* sig_output = output_slab + slab_offset;

                        const bool low_s = (req.flags & FAST_PATH_FLAG_LOW_S) != 0;
                        bool all_valid = true;

                        // Sign each hash
                        for (uint16_t m = 0; m < n_sigs; m++) {
                            const uint8_t* hash_m = hash_data + m * P256_MULTISIGN_ITEM_LEN;
                            uint8_t* sig_r = sig_output + m * P256_MULTISIGN_SIG_LEN;
                            uint8_t* sig_s = sig_r + 32;

                            const uint32_t nonce_ctr = atomicAdd(&g_nonce_counter, 1u);

#if ENGINE_USE_REAL_P256
                            const bool valid = smoke::p256::p256_sign_persistent(
                                sig_r, sig_s, hash_m, ks.private_d, nonce_ctr,
                                nullptr, low_s, false, nullptr, 0);
                            if (!valid) all_valid = false;
#else
                            // Stub mode
                            for (int j = 0; j < 32; j++) {
                                sig_r[j] = hash_m[j] ^ ks.private_d[j];
                                sig_s[j] = ks.private_d[j];
                            }
#endif
                        }

                        // Response metadata
                        resp.output_offset = slab_offset;
                        resp.output_bytes = n_sigs * P256_MULTISIGN_SIG_LEN;
                        resp.output_len = 64;  // First sig in inline output

                        // Copy first signature to inline output for convenience
                        for (int j = 0; j < 64; j++) {
                            resp.output[j] = sig_output[j];
                        }

                        if (!all_valid) {
                            resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                        }

                        // Telemetry: count units (signatures) not just requests
                        atomicAdd((unsigned long long*)&telemetry->dbg_units_completed, (unsigned long long)n_sigs);
                    }

                    resp.t_complete_us = clock64();

                    // Write response
                    const uint32_t cq_cap = telemetry->cq_capacity_active;
                    const uint64_t seq = s_base_seq + (uint64_t)i;
                    const uint32_t write_idx = s_base_write_idx + i;
                    const uint32_t resp_slot = write_idx % cq_cap;
                    resp.seq = seq;

                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Use commit protocol - write request_id LAST
                    write_response_base(dst, resp);
                    for (int j = 0; j < 64; j++) dst->output[j] = resp.output[j];
                    commit_response(dst, resp.request_id, resp.status, resp.output_len);
                }
                __syncthreads();

                // Telemetry updates for multi-sign
                if (tid == 0) {
                    atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
                    atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
                    atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
                    __threadfence_system();
                }
            }
            else {
            // Original warp-coop path for single-sign mode

#if !BENCH_LITE_TELEMETRY
        if (tid == 0) {
            atomicAdd((unsigned long long*)&telemetry->dbg_sign_begin, (unsigned long long)batch);
        }
#endif

        uint64_t t_phase1_start = clock64();

        // PHASE 1: Multi-warp parallel k*G
        // Use base+offset loop so ALL threads iterate same number of times (required for __syncthreads)
        for (uint32_t base = 0; base < batch; base += num_warps) {
            const uint32_t i = base + warp_id;
            if (i < batch) {
                const FastPathRequest req = s_batch[i];
                const Keyslot ks = keyslots[req.key_slot];  // Use per-request keyslot

                smoke::p256::warp_coop::p256_sign_warp_coop_phase1(
                    req.input, ks.private_d, s_sign_state[i], lane);

                // Collect Z for batch inversion (lane 0 only)
                if (lane == 0 && s_sign_state[i].valid) {
                    #pragma unroll
                    for (int limb = 0; limb < 8; limb++) {
                        s_Z_workspace[i * 8 + limb] = s_sign_state[i].Rz[limb];
                    }
                }
            }
            __syncthreads();  // All threads sync every iteration
        }

#if !BENCH_LITE_TELEMETRY
        if (tid == 0) {
            atomicAdd((unsigned long long*)&telemetry->cyc_scalar_mul, clock64() - t_phase1_start);
        }
#endif

        // Phase 4a: Warp 0 runs parallel batch inversion (all 32 lanes).
        // Guard in CALLER � inlined return inside function deadlocks __shfl_sync.
        uint64_t t_inv_start = clock64();
        if (warp_id == 0) {
            p256_batch_inv_block_montgomery_v2(
                s_Z_workspace, s_Zinv_workspace, s_P_workspace, batch);
        }
        __syncthreads();

#if !BENCH_LITE_TELEMETRY
        if (tid == 0) {
            atomicAdd((unsigned long long*)&telemetry->cyc_affine, clock64() - t_inv_start);
        }
#endif

        // PHASE 3: Multi-warp parallel signature finalization
        // Use base+offset loop so ALL threads iterate same number of times (required for __syncthreads)
        uint64_t t_phase3_start = clock64();
        for (uint32_t base = 0; base < batch; base += num_warps) {
            const uint32_t i = base + warp_id;
            if (i < batch) {
                const FastPathRequest req = s_batch[i];

                FastPathResponse resp;
                resp.request_id = req.request_id;
                resp.t_submit_us = req.t_submit_us;
                resp.t_dequeue_us = s_dequeue_cycles;
                resp.status = 0;
                resp.client_id = req.client_id;
                // Card 66: Record epoch/slot used for P256 warp-coop signing
                resp.epoch_used = static_cast<uint8_t>((*active_epoch) & 0xFF);
                resp.slot_used  = static_cast<uint8_t>(req.key_slot);
                // N3-fix (2026-05-14): P-256 sign writes r[32]||s[32] into the
                // FastPathResponse output-cache-line via the union with output[64].
                // smoke_generic_poll's translator (smoke_engine_abi.cpp:1907-1911)
                // zero-fills dst->output beyond src.output_len, so without this
                // field set the translator's memset clobbers the signature bytes.
                resp.output_len = 64;

                const bool low_s = (req.flags & FAST_PATH_FLAG_LOW_S) != 0;

                // N5 zinv_zero (2026-05-16): lane 0 of warp checks the
                // 8-limb post-phase4a Zinv for this op in shared memory
                // immediately before the broadcast/affine path. Splits
                // x_full_zero into batch-inversion-broken (zinv == 0)
                // vs jacobian_to_affine_with_zinv_warp-broken (zinv != 0
                // but downstream x_full == 0). Once per op.
                if (lane == 0) {
                    bool zinv_is_zero = true;
                    #pragma unroll
                    for (int limb = 0; limb < 8; limb++) {
                        if (s_Zinv_workspace[i * 8 + limb] != 0u) {
                            zinv_is_zero = false;
                            break;
                        }
                    }
                    if (zinv_is_zero) {
                        atomicAdd((unsigned long long*)&telemetry->dbg_p256_zinv_zero,
                                  (unsigned long long)1);
                    }
                }

                // Load Zinv into registers via shfl
                uint32_t my_zinv_limb = (lane < 8) ? s_Zinv_workspace[i * 8 + lane] : 0;
                __syncwarp(0xFFFFFFFF);

                uint32_t Zinv_local[8];
                #pragma unroll
                for (int limb = 0; limb < 8; limb++) {
                    Zinv_local[limb] = __shfl_sync(0xFFFFFFFF, my_zinv_limb, limb);
                }

                bool valid = smoke::p256::warp_coop::p256_sign_warp_coop_phase3(
                    s_sign_state[i], Zinv_local, resp.r, resp.s, lane, low_s);

                if (!valid) resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
                resp.t_complete_us = clock64();

                // Write response (lane 0 only)
                if (lane == 0) {
                    // N5-A (2026-05-14): split phase1 vs phase3 failures.
                    // Lane 0 only — one increment per op. state.valid set
                    // by phase 1 (RFC6979 k-gen). If state.valid==true but
                    // phase3 returned false, ecdsa_sign_from_r_warp failed.
                    if (!valid) {
                        if (!s_sign_state[i].valid) {
                            atomicAdd((unsigned long long*)&telemetry->dbg_p256_phase1_fail,
                                      (unsigned long long)1);
                        } else {
                            atomicAdd((unsigned long long*)&telemetry->dbg_p256_phase3_fail,
                                      (unsigned long long)1);
                        }
                    }
                    const uint32_t cq_cap = telemetry->cq_capacity_active;
                    const uint64_t seq = s_base_seq + (uint64_t)i;
                    const uint32_t write_idx = s_base_write_idx + i;
                    const uint32_t resp_slot = write_idx % cq_cap;
                    resp.seq = seq;
                    // Card 26.2 FIX: Explicit copy to volatile response buffer
                    volatile FastPathResponse* dst = &responses->responses[resp_slot];
                    // Card 26.72: Write all fields except request_id first
                    dst->seq = resp.seq;
                    dst->t_submit_us = resp.t_submit_us;
                    dst->t_dequeue_us = resp.t_dequeue_us;
                    dst->t_complete_us = resp.t_complete_us;
                    dst->status = resp.status;
                    dst->client_id = resp.client_id;
                    dst->epoch_used = resp.epoch_used;    // Card 66
                    dst->slot_used = resp.slot_used;      // Card 66
                    // N3-fix completion (2026-05-14): publish output_len so the
                    // smoke_generic_poll translator (smoke_engine_abi.cpp:1907-1911)
                    // copies r||s rather than zero-filling. Matches publication
                    // pattern used by SHA-256 (3743), SHA-512, AES-GCM, etc.
                    dst->output_len = resp.output_len;
                    for (int j = 0; j < 32; j++) dst->r[j] = resp.r[j];
                    for (int j = 0; j < 32; j++) dst->s[j] = resp.s[j];
                    // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
                    __threadfence_system();
                    commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
                }
            }
            __syncthreads();  // All threads sync every iteration
        }

        // Overflow detection
#if !BENCH_LITE_TELEMETRY
        if (tid == 0) {
            uint32_t cq_cap = telemetry->cq_capacity_active;
            uint32_t read_idx = responses->read_idx;
            uint32_t end_idx = s_base_write_idx + batch;
            if (end_idx >= read_idx + cq_cap) {
                uint32_t overwrites = end_idx - (read_idx + cq_cap);
                if (overwrites > batch) overwrites = batch;
                atomicAdd((unsigned long long*)&telemetry->response_overwrites, (unsigned long long)overwrites);
            }
        }
#endif

        if (tid == 0) {
#if !BENCH_LITE_TELEMETRY
            atomicAdd((unsigned long long*)&telemetry->cyc_sign_finalize, clock64() - t_phase3_start);
            atomicAdd((unsigned long long*)&telemetry->dbg_sign_done, (unsigned long long)batch);
            atomicAdd((unsigned long long*)&telemetry->dbg_write_response, (unsigned long long)batch);
            atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
#endif
            // Card 26.15: AUTHORITATIVE cycle tracking for COMPUTE (work was done)
            atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
            // Essential for accounting - always keep
            atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)batch);
            // Card 26.2 FIX: Ensure telemetry updates visible to host
            __threadfence_system();
        }

        } // Close warp-coop single-sign else block
        } // Card 26.19: Close P256_SIGN else block (Card 26.34: includes multi-sign)

#else
        // ========== THREAD-ONLY SIGNING PATH (with Card 26.19 Multi-Op Dispatch) ==========
        // s_base_write_idx and s_base_seq already reserved above

        // Card 26.34 DEBUG: Keyslot diagnostic probe (once per batch, tid==0 only)
        if (tid == 0) {
            uint32_t epoch_val = *active_epoch;
            telemetry->dbg_active_epoch_val = epoch_val;
            telemetry->dbg_slot_idx_used = 0;  // Thread-only uses req.key_slot=0

            // Store pointer addresses for debugging
            telemetry->dbg_keyslots_ptr = reinterpret_cast<uint64_t>(keyslots);
            telemetry->dbg_key0_addr = reinterpret_cast<uint64_t>(keyslots[0].private_d);

            // Read first 4 bytes of each keyslot's private_d (big-endian)
            const uint8_t* k0 = keyslots[0].private_d;
            const uint8_t* k1 = keyslots[1].private_d;
            telemetry->dbg_key0_first_u32 = ((uint32_t)k0[0] << 24) | ((uint32_t)k0[1] << 16) |
                                            ((uint32_t)k0[2] << 8) | (uint32_t)k0[3];
            telemetry->dbg_key1_first_u32 = ((uint32_t)k1[0] << 24) | ((uint32_t)k1[1] << 16) |
                                            ((uint32_t)k1[2] << 8) | (uint32_t)k1[3];
            __threadfence_system();
        }

        for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
            const FastPathRequest req = s_batch[i];
            const Keyslot ks = keyslots[req.key_slot];  // Use per-request keyslot

            FastPathResponse resp;
            resp.request_id = req.request_id;
            resp.t_submit_us = req.t_submit_us;
            resp.t_dequeue_us = s_dequeue_cycles;
            resp.status = 0;
            resp.client_id = req.client_id;

            // Card 26.19: Multi-op dispatch based on opcode
            const uint16_t opcode = req.opcode;

            switch (opcode) {
                case 0:  // P256_SIGN (backward compat: opcode=0)
                case static_cast<uint16_t>(OpCode::P256_SIGN): {
#if ENGINE_USE_REAL_P256
            const bool low_s = (req.flags & FAST_PATH_FLAG_LOW_S) != 0;
            uint32_t nonce_ctr = atomicAdd(&g_nonce_counter, 1u);

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->dbg_sign_begin, 1ull);
            }

            bool valid = smoke::p256::p256_sign_persistent(
                resp.r, resp.s, req.input, ks.private_d, nonce_ctr,
                nullptr, low_s, false, nullptr, 0);

            if (tid == 0) {
                atomicAdd((unsigned long long*)&telemetry->dbg_sign_done, 1ull);
            }

            if (!valid) resp.status = static_cast<uint8_t>(OpStatus::CRYPTO_ERROR);
#else
            for (int j = 0; j < 32; j++) {
                resp.r[j] = req.input[j] ^ ks.private_d[j];
                resp.s[j] = ks.private_d[j];
            }
#endif
                    break;
                }

                case static_cast<uint16_t>(OpCode::SHA256): {
                    // Validate input_len ∈ [0, 32]
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::sha256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::SHA512): {
                    // Card 26.31: SHA-512 hashing
                    // Validate input_len ∈ [0, 32]
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::sha512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::SHA3_256): {
                    // Card 26.32: SHA3-256 (Keccak) hashing
                    // Validate input_len ∈ [0, 32]
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::sha3_256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::BLAKE2B_256): {
                    // Card 26.33: BLAKE2b-256 hashing
                    // Validate input_len ∈ [0, 32]
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::blake2b_256_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 32;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::SHA3_512): {
                    // Card 140: SHA3-512 (Keccak) hashing
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::sha3_512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::BLAKE2B_512): {
                    // Card 141: BLAKE2b-512 hashing
                    if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::hash::blake2b_512_hash_short(req.input, req.input_len, resp.output);
                        resp.output_len = 64;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::HMAC_SHA256): {
                    // Card 26.29: HMAC-SHA256 keyed hash
                    // Validate input_len ∈ [0, HMAC_SHA256_MAX_MSG_LEN]
                    if (req.input_len > HMAC_SHA256_MAX_MSG_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        // Use key from per-request keyslot (ks.private_d is 32 bytes)
                        hmac_sha256_device(
                            ks.private_d,      // key (32 bytes from keyslot)
                            32,                // key_len (P-256 keys are 32 bytes)
                            req.input,         // message
                            req.input_len,     // message length
                            resp.output        // output (32 bytes)
                        );
                        resp.output_len = 32;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::HMAC_SHA512): {
                    // Card 26.51: HMAC-SHA512 keyed hash
                    if (req.input_len > HMAC_SHA512_MAX_MSG_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        hmac_sha512_device(
                            ks.private_d,      // key (32 bytes from keyslot)
                            32,                // key_len
                            req.input,         // message
                            req.input_len,     // message length
                            resp.output        // output (64 bytes)
                        );
                        resp.output_len = 64;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::HMAC_SHA384): {
                    // Card 26.52: HMAC-SHA384 keyed hash
                    if (req.input_len > HMAC_SHA384_MAX_MSG_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        hmac_sha384_device(
                            ks.private_d,      // key (32 bytes from keyslot)
                            32,                // key_len
                            req.input,         // message
                            req.input_len,     // message length
                            resp.output        // output (48 bytes)
                        );
                        resp.output_len = 48;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::AES256_CTR): {
                    // Card 26.54: AES-256-CTR stream cipher
                    // Card 26.64: Use precomputed round keys from keyslot
                    if (req.input_len < AES256_CTR_NONCE_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::aes::aes256_ctr_encrypt_short_cached(
                            ks.aes_roundkeys,  // precomputed round keys
                            req.input,         // nonce[12] + plaintext[N]
                            req.input_len,     // total input length
                            resp.output        // ciphertext output
                        );
                        resp.output_len = req.input_len - AES256_CTR_NONCE_LEN;
                    }
                    break;
                }

                case static_cast<uint16_t>(OpCode::CHACHA20): {
                    // Card 26.66: ChaCha20 stream cipher
                    if (req.input_len < CHACHA20_NONCE_LEN) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else if (req.input_len > 32) {
                        resp.status = static_cast<uint8_t>(OpStatus::INVALID_INPUT_LEN);
                    } else {
                        smoke::chacha::chacha20_encrypt_short_keyslot(
                            ks.private_d,      // key (32 bytes from keyslot)
                            req.input,         // nonce[12] + plaintext[N]
                            req.input_len,     // total input length
                            resp.output,       // ciphertext output
                            &resp.output_len   // output length
                        );
                    }
                    break;
                }

                default: {
                    resp.status = static_cast<uint8_t>(OpStatus::INVALID_OPCODE);
                    break;
                }
            }

            resp.t_complete_us = clock64();

            const uint32_t cq_cap = telemetry->cq_capacity_active;
            const uint64_t seq = s_base_seq + (uint64_t)i;
            const uint32_t write_idx = s_base_write_idx + i;
            const uint32_t resp_slot = write_idx % cq_cap;
            resp.seq = seq;

            // Card 26.2 FIX: Explicit volatile field copies (can't struct-assign volatile)
            volatile FastPathResponse* dst = &responses->responses[resp_slot];
            // Card 26.72: Write all fields except request_id first
            dst->seq = resp.seq;
            dst->t_submit_us = resp.t_submit_us;
            dst->t_dequeue_us = resp.t_dequeue_us;
            dst->t_complete_us = resp.t_complete_us;
            dst->status = resp.status;
            dst->client_id = resp.client_id;
            dst->epoch_used = resp.epoch_used;    // Card 66
            dst->slot_used = resp.slot_used;      // Card 66
            for (int j = 0; j < 32; j++) dst->r[j] = resp.r[j];
            for (int j = 0; j < 32; j++) dst->s[j] = resp.s[j];
            // Card 26.72: Commit - write request_id LAST (system fence for host visibility)
            __threadfence_system();
            commit_store(dst, resp.request_id, resp.status, resp.output_len);  // B-319b
        }
        __syncthreads();

        // Overflow detection
        if (tid == 0) {
            uint32_t cq_cap = telemetry->cq_capacity_active;
            uint32_t read_idx = responses->read_idx;
            uint32_t end_idx = s_base_write_idx + batch;
            if (end_idx >= read_idx + cq_cap) {
                uint32_t overwrites = end_idx - (read_idx + cq_cap);
                if (overwrites > batch) overwrites = batch;
                atomicAdd((unsigned long long*)&telemetry->response_overwrites, (unsigned long long)overwrites);
            }
        }

        // Warp-level reduction for outcome count
        {
            uint32_t my_ok = 0;
            for (uint32_t i = tid; i < batch; i += (uint32_t)blockDim.x) {
                my_ok++;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                my_ok += __shfl_down_sync(0xFFFFFFFF, my_ok, offset);
            }
            if (lane == 0) {
                atomicAdd((unsigned long long*)&telemetry->outcome_ok, (unsigned long long)my_ok);
            }
        }

        if (tid == 0) {
            atomicAdd((unsigned long long*)&telemetry->dbg_write_response, (unsigned long long)batch);
            atomicAdd((unsigned long long*)&telemetry->debug_responses_written, (unsigned long long)batch);
            // Card 26.15: AUTHORITATIVE cycle tracking for COMPUTE (THREAD_ONLY path)
            atomicAdd((unsigned long long*)&telemetry->cycles_compute, clock64() - s_compute_start_cycles);
        }
#endif // P256_WARP_COOP_ENABLED

        // Memory fence to ensure responses visible to host
        if (lane == 0) {
            __threadfence_system();
        }
        __syncthreads();
    }
}

// Host-callable wrapper to launch the SHARDED FAST PATH kernel
// Card 26.2: Zero contention - each CTA owns its shard
// Card 26.25: Added d_output_slab parameter for extended multi-message outputs
// Card 26.70: Added d_rsa_keyslots parameter for RSA-2048 signing
// Card 27.14: Added d_mldsa_keyslots parameter for ML-DSA signing
cudaError_t launch_fast_path_engine_loop_sharded(
    FastPathRing** d_ring_ptrs,
    FastPathResponseBuffer** d_resp_ptrs,
    FastPathTelemetry* d_telemetry,
    Keyslot* d_keyslots,
    volatile uint32_t* d_active_epoch,
    volatile bool* d_shutdown,
    cudaStream_t stream,
    uint32_t num_ctas,
    uint32_t num_shards,
    const uint8_t* d_payload_slab,  // Card 26.24: Extended payload buffer
    uint8_t* d_output_slab,         // Card 26.25: Extended output buffer
    RSAKeyslot* d_rsa_keyslots,     // Card 26.70: RSA-2048 CRT keyslots
    MLDSAKeyslot* d_mldsa_keyslots  // Card 27.14: ML-DSA keyslots
) {
    // Card 26.4 REVERT: Back to 256 threads for testing
    dim3 block(256);

    // Occupancy query
    int min_grid_size = 0, opt_block_size = 0, num_blocks = 0;
    int actual_block_size = block.x;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &opt_block_size, fast_path_engine_loop_sharded, 0, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, fast_path_engine_loop_sharded, actual_block_size, 0);

    int device_id = 0;
    cudaGetDevice(&device_id);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);

    uint32_t max_ctas = static_cast<uint32_t>(num_blocks * sm_count);
    bool auto_tuned = false;
    if (num_ctas == 0) {
        num_ctas = max_ctas;
        auto_tuned = true;
    }
    if (num_ctas > max_ctas) num_ctas = max_ctas;
    if (num_ctas < 1) num_ctas = 1;

    dim3 grid(num_ctas);

    fprintf(stderr, "[launch_fast_path_sharded] DIAGNOSTICS:\n");
    fprintf(stderr, "  SHARDED MODE:       %u shards (zero contention)\n", num_shards);
    if (auto_tuned) {
        fprintf(stderr, "  CTAs (auto-tuned):  %u\n", num_ctas);
    } else {
        fprintf(stderr, "  Requested CTAs:     %u\n", num_ctas);
    }
    fprintf(stderr, "  Block size:         %d threads\n", actual_block_size);
    fprintf(stderr, "  Max blocks/SM:      %d\n", num_blocks);
    fprintf(stderr, "  SM count:           %d\n", sm_count);
    fprintf(stderr, "  Max total blocks:   %d\n", num_blocks * sm_count);
    fprintf(stderr, "  CTAs per shard:     %.2f\n", (float)num_ctas / num_shards);
    fflush(stderr);

    fast_path_engine_loop_sharded<<<grid, block, 0, stream>>>(
        d_ring_ptrs,
        d_resp_ptrs,
        d_telemetry,
        d_keyslots,
        d_active_epoch,
        d_shutdown,
        num_shards,
        d_payload_slab,  // Card 26.24
        d_output_slab,   // Card 26.25
        d_rsa_keyslots,  // Card 26.70
        d_mldsa_keyslots // Card 27.14
    );

    return cudaGetLastError();
}

// Host-callable wrapper to launch the FAST PATH kernel
// Card 26.1B: Added num_ctas parameter for multi-CTA dequeue
#ifdef ENABLE_LEGACY_KERNEL
// Card 26.24: Added d_payload_slab parameter for extended multi-message
// Card 26.25: Added d_output_slab parameter for extended multi-message outputs
// Card 26.70: Added d_rsa_keyslots parameter for RSA-2048 signing
// Card 27.14: Added d_mldsa_keyslots parameter for ML-DSA signing
cudaError_t launch_fast_path_engine_loop(
    FastPathRing* d_ring,
    FastPathResponseBuffer* d_responses,
    FastPathTelemetry* d_telemetry,
    Keyslot* d_keyslots,
    volatile uint32_t* d_active_epoch,
    volatile bool* d_shutdown,
    cudaStream_t stream,
    uint32_t num_ctas,  // Card 26.1B: Number of CTAs (service lanes)
    const uint8_t* d_payload_slab,  // Card 26.24: Extended payload buffer
    uint8_t* d_output_slab,         // Card 26.25: Extended output buffer
    RSAKeyslot* d_rsa_keyslots,     // Card 26.70: RSA-2048 CRT keyslots
    MLDSAKeyslot* d_mldsa_keyslots  // Card 27.14: ML-DSA keyslots
) {
    // Card 26.4 REVERT: Back to 256 threads
    dim3 block(256);

    // Optimize-6.4: Query occupancy FIRST to enable auto-tuning
    int min_grid_size = 0;
    int opt_block_size = 0;
    int num_blocks = 0;
    int actual_block_size = block.x;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &opt_block_size, fast_path_engine_loop, 0, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, fast_path_engine_loop, actual_block_size, 0);

    int device_id = 0;
    cudaGetDevice(&device_id);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);

    // Optimize-6.4: Auto-tune CTA count if num_ctas == 0
    uint32_t max_ctas = static_cast<uint32_t>(num_blocks * sm_count);
    bool auto_tuned = false;
    if (num_ctas == 0) {
        // Auto-select: use all available blocks for maximum GPU saturation
        num_ctas = max_ctas;
        auto_tuned = true;
    }
    // Card 26.2: Cap at max possible (no point requesting more than can run)
    if (num_ctas > max_ctas) {
        num_ctas = max_ctas;
    }
    // Safety floor
    if (num_ctas < 1) num_ctas = 1;

    dim3 grid(num_ctas);

    fprintf(stderr, "[launch_fast_path] DIAGNOSTICS:\n");
    if (auto_tuned) {
        fprintf(stderr, "  CTAs (auto-tuned):  %u\n", num_ctas);
    } else {
        fprintf(stderr, "  Requested CTAs:     %u\n", num_ctas);
    }
    fprintf(stderr, "  Block size:         %d threads\n", actual_block_size);
    fprintf(stderr, "  Max blocks/SM:      %d\n", num_blocks);
    fprintf(stderr, "  SM count:           %d\n", sm_count);
    fprintf(stderr, "  Max total blocks:   %d (blocks/SM * SMs)\n", num_blocks * sm_count);
    fprintf(stderr, "  Occupancy optimal:  grid=%d, block=%d\n", min_grid_size, opt_block_size);
    if (!auto_tuned && num_ctas < max_ctas) {
        fprintf(stderr, "  [HINT] Using %u/%u available blocks. Pass num_ctas=0 for auto-tune.\n",
                num_ctas, max_ctas);
    }
    fflush(stderr);

    // Launch persistent kernel on provided stream (should be non-blocking)
    // The kernel runs forever until d_shutdown is set
    fast_path_engine_loop<<<grid, block, 0, stream>>>(
        d_ring,
        d_responses,
        d_telemetry,
        d_keyslots,
        d_active_epoch,
        d_shutdown,
        d_payload_slab,  // Card 26.24
        d_output_slab,   // Card 26.25
        d_rsa_keyslots,  // Card 26.70
        d_mldsa_keyslots // Card 27.14
    );

    // DON'T sync - kernel is persistent, just return launch status
    return cudaGetLastError();
}
#endif // ENABLE_LEGACY_KERNEL

// ============================================================================
// Card 26: P-256 Comb Table Loading for Warp-Coop Mode
// ============================================================================
// Load pre-computed comb table into constant memory before launching kernel.
// Table must be generated by scripts/tools/gen_p256_comb_table.py with w=7.

#if P256_WARP_COOP_ENABLED

cudaError_t load_p256_comb_table(
    const uint32_t* h_x_table,  // Host pointer to X coordinates [8 * 127]
    const uint32_t* h_y_table,  // Host pointer to Y coordinates [8 * 127]
    size_t table_size           // Expected: 8 * 255 = 2040 elements (w=8)
) {
    // Validate table size
    constexpr size_t EXPECTED_SIZE = 8 * 255;
    if (table_size != EXPECTED_SIZE) {
        fprintf(stderr, "[load_p256_comb_table] ERROR: Expected %zu elements, got %zu\n",
                EXPECTED_SIZE, table_size);
        return cudaErrorInvalidValue;
    }

    // Transpose from SOA [limb * ENTRIES + entry] to AOS [entry * 8 + limb]
    // AOS layout puts all 8 limbs of one point in one constant cache line,
    // reducing 8 serialized constant reads to 1 per warp-group lookup.
    constexpr int ENTRIES = EXPECTED_SIZE / 8;  // 255 for w=8
    uint32_t x_aos[EXPECTED_SIZE];
    uint32_t y_aos[EXPECTED_SIZE];
    for (int entry = 0; entry < ENTRIES; entry++) {
        for (int limb = 0; limb < 8; limb++) {
            x_aos[entry * 8 + limb] = h_x_table[limb * ENTRIES + entry];
            y_aos[entry * 8 + limb] = h_y_table[limb * ENTRIES + entry];
        }
    }

    // Upload X table to constant memory (AOS layout)
    cudaError_t err = cudaMemcpyToSymbol(
        P256_G_TABLE_X_SOA,
        x_aos,
        EXPECTED_SIZE * sizeof(uint32_t),
        0,  // offset
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_p256_comb_table] X table upload failed: %s\n",
                cudaGetErrorString(err));
        return err;
    }

    // Upload Y table to constant memory (AOS layout)
    err = cudaMemcpyToSymbol(
        P256_G_TABLE_Y_SOA,
        y_aos,
        EXPECTED_SIZE * sizeof(uint32_t),
        0,  // offset
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_p256_comb_table] Y table upload failed: %s\n",
                cudaGetErrorString(err));
        return err;
    }

    fprintf(stderr, "[load_p256_comb_table] Loaded %zu entries to constant memory (AOS layout)\n",
            EXPECTED_SIZE);
    return cudaSuccess;
}

#if P256_FASTPATH_FULLWINDOW
// SPEED-H2-01: upload the full-window fixed-base table to the __device__
// global P256_G_FULLWINDOW. The payload is already AoS (entry=X[8]||Y[8]),
// byte-identical to the device layout, so this is a straight copy — no
// transpose. `payload` is the .bin WITHOUT its 32-byte header (the ABI
// layer strips + validates the header and fingerprint before calling here).
cudaError_t load_p256_fullwindow_table(
    const uint32_t* h_payload,   // AoS X||Y, ENTRIES*16 u32
    size_t          n_u32        // must equal P256_FULLWINDOW_ENTRIES*16
) {
    constexpr size_t EXPECTED = (size_t)P256_FULLWINDOW_ENTRIES * 16;
    if (n_u32 != EXPECTED) {
        fprintf(stderr, "[load_p256_fullwindow_table] ERROR: expected %zu u32, got %zu\n",
                EXPECTED, n_u32);
        return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaMemcpyToSymbol(
        P256_G_FULLWINDOW, h_payload, EXPECTED * sizeof(uint32_t),
        0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_p256_fullwindow_table] upload failed: %s\n",
                cudaGetErrorString(err));
        return err;
    }
    fprintf(stderr, "[load_p256_fullwindow_table] Loaded %d entries (%zu KB) to device global\n",
            P256_FULLWINDOW_ENTRIES, (EXPECTED * sizeof(uint32_t)) / 1024);
    return cudaSuccess;
}

// SPEED-H2-01 GAP 4 (2026-07-22): read back one 16-u32 entry of the
// __device__ global P256_G_FULLWINDOW so the ABI loader can verify the
// device actually holds the validated host bytes (device-side sentinel,
// independent of host bookkeeping). Must live in THIS TU: the symbol is
// defined in p256_field_warp_coop.cuh without extern linkage, so it is
// TU-local — cudaMemcpyFromSymbol from any other TU would read a different
// (never-uploaded) instance than the one the kernel reads.
cudaError_t readback_p256_fullwindow_entry(
    size_t   entry_index,
    uint32_t out_entry[16]
) {
    if (!out_entry || entry_index >= (size_t)P256_FULLWINDOW_ENTRIES) {
        return cudaErrorInvalidValue;
    }
    return cudaMemcpyFromSymbol(
        out_entry,
        P256_G_FULLWINDOW,
        16 * sizeof(uint32_t),
        entry_index * 16 * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);
}
#endif // P256_FASTPATH_FULLWINDOW

#endif // P256_WARP_COOP_ENABLED

} // namespace engine
} // namespace smoke
