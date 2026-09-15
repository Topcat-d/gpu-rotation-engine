/**
 * Smoke Engine ABI v0 - Real Implementation
 *
 * Card 24.1: Wired to persistent P-256 engine (not a stub anymore).
 * Uses engine_host.cpp APIs to submit/poll real GPU signatures.
 *
 * CARD: CARD-L3-ENGINE-ABI-02 + CARD-ABI-PERSISTENT-WIRING-V1
 */

// MUST be before any Windows headers (including those from torch)
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "../../include/smoke_engine.h"
#include "../../include/smoke_abi.h"
#include "../../include/smoke_generic_abi.h"
#include "../../include/smoke_rotation_probe.h"

// Persistent engine headers
#include "engine_state.cuh"
#include "engine_constants.cuh"
#include "op_schema.cuh"  // kernel-truth OP_CONSTRAINTS; s_op_schema is pinned to it below

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>

// ============================================================================
// Card 26.78: Fail-Loud CPU Fallback Policy Helpers
// ============================================================================

static bool smoke_cpu_fallback_enabled() {
    const char* v = std::getenv("SMOKE_ALLOW_CPU_FALLBACK");
    if (!v) return false;
#ifdef _WIN32
    return (std::strcmp(v, "1") == 0 ||
            _stricmp(v, "true") == 0 ||
            _stricmp(v, "yes") == 0);
#else
    return (std::strcmp(v, "1") == 0 ||
            strcasecmp(v, "true") == 0 ||
            strcasecmp(v, "yes") == 0);
#endif
}

static void smoke_fail_loud_once(const char* api, const char* msg) {
    static std::atomic<int> once{0};
    if (once.fetch_add(1, std::memory_order_relaxed) == 0) {
        std::fprintf(stderr, "[SMOKE][FAIL-LOUD] %s: %s\n", api, msg);
        std::fflush(stderr);
    }
}

// pybind11 includes (conditional for C ABI-only build)
#ifndef SMOKE_ABI_NO_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // For py::array_t (Card 26: comb table loading)
namespace py = pybind11;
#endif

#ifndef _WIN32
#include <unistd.h>
#endif

// Forward declarations from engine_host.cpp (in smoke::engine namespace)
// EngineResponse is defined in engine_state.cuh (already included)
namespace smoke {
namespace engine {
    struct EngineHandle;  // Opaque handle (defined in engine_host.cpp)

    EngineHandle* engine_create();
    void engine_destroy(EngineHandle* handle);
    bool engine_start(EngineHandle* handle);
    bool engine_shutdown(EngineHandle* handle);
    bool engine_set_key(EngineHandle* handle, const uint8_t private_key[32]);
    // Card 26.57: Typed key loaders
    bool engine_set_key_typed(EngineHandle* handle, uint32_t slot_idx, KeyslotType key_type,
                              const uint8_t* key_data, uint16_t key_len);
    bool engine_set_key_p256(EngineHandle* handle, const uint8_t private_key[32]);
    bool engine_set_key_aes256(EngineHandle* handle, uint32_t slot_idx, const uint8_t key[32]);
    bool engine_set_key_hmac(EngineHandle* handle, uint32_t slot_idx, const uint8_t* key, uint16_t key_len);
    KeyslotType engine_get_key_type(EngineHandle* handle, uint32_t slot_idx);
    bool engine_submit_request(EngineHandle* handle, const uint8_t hash[32], uint32_t request_id);
    bool engine_poll_response(EngineHandle* handle, uint32_t request_id, EngineResponse* out);

    // Card 24.2: Batch APIs for sync-collapse optimization
    uint32_t engine_submit_batch(EngineHandle* handle,
                                  const uint8_t* hashes,      // N * 32 bytes, contiguous
                                  const uint32_t* request_ids,
                                  uint32_t count);
    uint32_t engine_poll_batch(EngineHandle* handle,
                                const uint32_t* request_ids,
                                EngineResponse* out,
                                uint8_t* out_ready,
                                uint32_t count);

    // Card 24.4: Per-thread stream support for multi-threaded batch operations
    cudaStream_t engine_create_thread_stream();
    void engine_destroy_thread_stream(cudaStream_t stream);
    uint32_t engine_submit_batch_with_stream(EngineHandle* handle,
                                              const uint8_t* hashes,
                                              const uint32_t* request_ids,
                                              uint32_t count,
                                              cudaStream_t stream);
    uint32_t engine_poll_batch_with_stream(EngineHandle* handle,
                                            const uint32_t* request_ids,
                                            EngineResponse* out,
                                            uint8_t* out_ready,
                                            uint32_t count,
                                            cudaStream_t stream);

    // Card 24.5: Read GPU stats (total_sign_ops counter from device memory)
    uint64_t engine_get_total_sign_ops(EngineHandle* handle);

    // Card 25.1: FAST PATH APIs - Host-mapped memory for zero-copy submits
    struct FastPathHandle;  // Opaque handle

    FastPathHandle* fast_path_create(EngineHandle* legacy_handle);
    void fast_path_destroy(FastPathHandle* handle);
    bool fast_path_start(FastPathHandle* handle);
    bool fast_path_submit(FastPathHandle* handle, const uint8_t hash[32], uint8_t flags);
    // Card 26.19: Added opcode and input_len for multi-op support
    // D5b-keyslot: Added key_slot (shared by the batch; default 0). Before
    // this the host writers hardcoded ring key_slot=0, so inline requests
    // could never select a non-default keyslot.
    uint32_t fast_path_submit_batch(FastPathHandle* handle, const uint8_t* inputs,
                                     uint32_t count, uint8_t flags, uint8_t client_id,
                                     uint16_t opcode, uint16_t input_len,
                                     uint8_t key_slot = 0);
    // D5b-keyslot: keyslot-array size (ENGINE_MAX_KEY_SLOTS) for fail-closed
    // bounds checks at the ABI truth boundary.
    uint32_t fast_path_max_key_slots();
    uint32_t fast_path_poll(FastPathHandle* handle, FastPathResponse* out, uint32_t max_count);
    bool fast_path_get_telemetry(FastPathHandle* handle, FastPathTelemetry* out);
    void fast_path_get_ring_status(FastPathHandle* handle, uint32_t* head, uint32_t* tail, uint32_t* pending);
    // Card 26.2: Aggregate status across ALL shards
    void fast_path_get_ring_status_all(FastPathHandle* handle, uint64_t* total_head, uint64_t* total_tail,
                                        uint64_t* total_pending, uint32_t* max_pending_shard);
    void fast_path_get_response_status_all(FastPathHandle* handle, uint64_t* total_write_idx,
                                            uint64_t* total_read_idx, uint64_t* total_pending);
    void fast_path_debug_pending_shards(FastPathHandle* handle, uint32_t* count,
                                         uint32_t shard_ids[16], uint32_t pending_counts[16]);
    // B-317 (2026-05-12): Per-shard read-only diagnostic snapshot. Fills five
    // parallel uint32_t[max_shards] arrays with each active shard's request-
    // ring head/tail and response-buffer write_idx/read_idx. Read-only.
    uint32_t fast_path_get_shard_diag_all(FastPathHandle* handle,
                                          uint32_t* out_shard_id,
                                          uint32_t* out_req_head,
                                          uint32_t* out_req_tail,
                                          uint32_t* out_resp_write_idx,
                                          uint32_t* out_resp_read_idx,
                                          uint32_t max_shards);
    // B-321 (2026-05-13): Read-only getter for handle->next_request_id.
    // See implementation in engine_host.cpp.
    uint64_t fast_path_get_next_request_id(FastPathHandle* handle);
    bool fast_path_stop(FastPathHandle* handle);  // Card 26.2: Stop kernel, keep resources
    void fast_path_set_quick_restart(bool enable);  // Card 27.28: Quick restart mode
    bool fast_path_set_comb_table_loaded(FastPathHandle* handle, uint32_t loaded);  // Card 26.1
    bool fast_path_set_num_ctas(FastPathHandle* handle, uint32_t num_ctas);  // Card 26.2
    bool fast_path_set_num_shards(FastPathHandle* handle, uint32_t num_shards);  // Card 26.2: Multi-shard zero contention
    bool fast_path_set_cq_capacity(FastPathHandle* handle, uint32_t cq_capacity);  // Card 26.12
    bool fast_path_set_submit_policy(FastPathHandle* handle, uint32_t policy);  // Card 26.14
    bool fast_path_set_shards_per_batch(FastPathHandle* handle, uint32_t k);  // Card 26.14

    // Card 26.25: Response slab for extended multi-message outputs
    bool fast_path_get_output_slab(FastPathHandle* handle, uint8_t** out_ptr, uint32_t* out_size);
    // Slab-fix: count of responses forced to SLAB_OVERRUN at drain (fail-loud audit)
    uint64_t fast_path_slab_overrun_count(FastPathHandle* handle);
    // Slab-fix: owner-verified, recency-bracketed slab read (TOCTOU-safe)
    int fast_path_read_output_segment_checked(FastPathHandle* handle, uint32_t offset,
                                              uint32_t length, uint64_t expected_request_id,
                                              uint8_t* out_buf);

    // Card 26.34: Payload slab for multi-sign inputs
    bool fast_path_get_payload_slab(FastPathHandle* handle, uint8_t** out_ptr, uint32_t* out_size);

    // Card 26.71: Load RSA-2048 CRT key into fast path RSA keyslot
    bool fast_path_load_rsa_key(
        FastPathHandle* handle,
        uint32_t slot_idx,
        const uint8_t p[128],
        const uint8_t q[128],
        const uint8_t dp[128],
        const uint8_t dq[128],
        const uint8_t qinv[128],
        const uint8_t R2_p[128],
        const uint8_t R2_q[128],
        uint32_t p_prime,
        uint32_t q_prime
    );

    // Card 27.13: Load ML-DSA private key into fast path ML-DSA keyslot
    bool fast_path_load_mldsa_key(
        FastPathHandle* handle,
        uint32_t slot_idx,
        uint8_t mode,               // 0=ML-DSA-44, 1=ML-DSA-65, 2=ML-DSA-87
        const uint8_t* sk_blob,
        uint32_t sk_len
    );

    // Card 27.19: Submit ML-DSA sign request (host-mediated dispatch)
    // Card 27.42: Added flags parameter for deterministic mode
    // Returns: 1 on success, 0 on failure
    uint32_t fast_path_submit_mldsa(
        FastPathHandle* handle,
        const uint8_t* msg,
        uint32_t msg_len,
        uint8_t mode,
        uint8_t key_slot,
        uint8_t client_id,
        uint8_t flags = 0  // Card 27.42: bit 0 = deterministic mode
    );

    // Card 27.24: Flush ML-DSA batch queue
    // Returns: number of successfully signed requests
    uint32_t fast_path_mldsa_flush_batch(FastPathHandle* handle);

    // Card 27.23: ML-DSA signature verification (host-mediated dispatch)
    // Returns: true if signature is valid, false otherwise
    bool mldsa_verify_host_dispatch(
        uint8_t mode,
        const uint8_t* pk,
        uint32_t pk_len,
        const uint8_t* msg,
        uint32_t msg_len,
        const uint8_t* sig,
        uint32_t sig_len
    );

    // Card 231: ML-DSA keygen (host-mediated dispatch)
    // Returns: true on success, fills pk_out/sk_out with simple format keys
    bool mldsa_keygen_host_dispatch(
        FastPathHandle* handle,
        uint8_t mode,
        const uint8_t rho[32],
        const uint8_t rhoprime[32],
        uint8_t* pk_out, uint32_t pk_max, uint32_t* pk_len_out,
        uint8_t* sk_out, uint32_t sk_max, uint32_t* sk_len_out
    );

    // Card 26.71: Submit RSA-2048 CRT sign request
    uint32_t fast_path_submit_rsa2048(
        FastPathHandle* handle,
        const uint8_t message[256],
        uint8_t key_slot,
        uint8_t client_id
    );

#ifdef SLH_HAVE_CUDA
    // Card SLH-002 P2: SLH-DSA-SHA2-128s host-mediated dispatch
    bool fast_path_load_slh_key(
        FastPathHandle* handle,
        uint32_t slot_idx,
        const uint8_t sk_seed[16],
        const uint8_t sk_prf[16],
        const uint8_t pk_seed[16],
        const uint8_t pk_root[16]
    );
    uint32_t fast_path_submit_slh(
        FastPathHandle* handle,
        const uint8_t* msg,
        uint32_t msg_len,
        const uint8_t* addrnd,      // 16 B (hedged) or NULL (deterministic)
        uint8_t key_slot,
        uint8_t client_id,
        uint64_t* out_request_id
    );
    uint32_t fast_path_slh_flush_batch(FastPathHandle* handle);
    // §5d item 5 debug/test hook (fault-injection; never product code)
    bool fast_path_slh_corrupt_key_mirror(FastPathHandle* handle, uint32_t slot_idx);
#endif

    // Card 26.34: Submit single multi-sign request with payload slab reference
    uint32_t fast_path_submit_multisign(
        FastPathHandle* handle,
        const uint8_t* input,       // 32-byte input buffer (header in [0..7])
        uint8_t flags,              // Should include FAST_PATH_FLAG_MULTIMSG
        uint8_t client_id,
        uint8_t key_slot,
        uint32_t payload_offset,    // Offset in payload slab where hashes start
        uint32_t payload_len        // Total bytes in payload (n_sigs * 32)
    );

    // Card 26.85: Generic submit with full control over all request fields
    uint32_t fast_path_submit_generic(
        FastPathHandle* handle,
        const uint8_t* input,       // 32-byte inline input buffer
        uint16_t input_len,         // Actual input length
        uint8_t flags,
        uint8_t key_slot,
        uint8_t client_id,
        uint16_t opcode,
        uint32_t payload_offset,    // Slab offset (0=inline)
        uint32_t payload_len        // Slab length (0=inline)
    );

    // Card 26.13: Native C++ benchmark submitter result struct
    struct NativeBenchResult {
        double   duration_sec;
        uint64_t ops_submitted;
        uint64_t ops_completed;
        double   throughput_sig_sec;
        uint64_t state_idle_empty;
        uint64_t state_idle_respfull;
        uint64_t state_busy;
        double   pct_busy;
        // Card 26.15: Cycle-based metrics (AUTHORITATIVE for GO/NO-GO decisions)
        uint64_t cycles_idle_empty;
        uint64_t cycles_idle_respfull;
        uint64_t cycles_compute;
        double   pct_busy_cycles;  // cycles_compute / (cycles_compute + cycles_idle_empty + cycles_idle_respfull)
        bool     accounting_valid;
        // Card 26.23: Multi-message work amplification metrics
        uint64_t units_completed;     // Total digests/signatures (from dbg_units_completed)
        double   units_per_sec;       // units_completed / duration_sec
        uint16_t multi_msg_n;         // N messages per request (0 = disabled)
    };
    // Card 26.23: Extended with multi_msg_n and msg_len for multi-message support
    // Card 26.27: Extended with tuning knobs for GO gate (pct_busy_cycles >= 80%)
    NativeBenchResult fast_path_run_native_bench(FastPathHandle* handle, double duration_sec, uint32_t batch_size,
                                                  uint16_t opcode = 0, uint16_t input_len = 32,
                                                  uint16_t multi_msg_n = 0, uint16_t msg_len = 8,
                                                  // Card 26.27: Tuning knobs
                                                  uint32_t high_water = 0, uint32_t low_water = 0,
                                                  uint32_t poll_burst = 1, uint32_t poll_max = 1024,
                                                  uint32_t yield_us = 0);

    // Native submit-and-drain: eliminate Python from batch hot loop
    struct NativeDrainResult {
        int64_t  submitted;
        int64_t  received;
        int64_t  errors;
        double   wall_ms;
        double   submit_ms;
        double   poll_ms;
    };
    NativeDrainResult fast_path_submit_and_drain(
        FastPathHandle* handle,
        const uint8_t*  hash_blob,
        int64_t         total_ops,
        int32_t         chunk_size,
        uint8_t         flags,
        uint16_t        opcode = 0,
        uint16_t        input_len = 32);

    // ATTEST-NATIVE-01: outputs-returning variable-length hash/HMAC feeder.
    struct HashDrainResult {
        int64_t  submitted;
        int64_t  received;
        int64_t  errors;
        uint32_t completed_count;
        uint32_t accounting_ok;
        double   wall_ms;
        double   submit_ms;
        double   poll_ms;
    };
    HashDrainResult fast_path_submit_and_drain_hashes(
        FastPathHandle* handle,
        const uint8_t*  payloads,
        const uint32_t* offsets,
        uint32_t        item_count,
        uint16_t        opcode,
        uint8_t         key_slot,
        uint8_t*        out_digests);

    // C4.6: Native ABI Continuous Feeder
    struct ContinuousFeederResult {
        double   duration_sec;
        uint64_t ops_submitted, ops_completed, ops_errors;
        double   throughput_sig_sec;
        bool     accounting_valid;
        uint64_t submit_calls, drain_calls;
        double   submit_time_us, drain_time_us;
        uint32_t target_depth, refill_threshold, batch_cap;
        double   avg_queue_depth;
        uint32_t min_queue_depth, max_queue_depth;
        double   avg_batch_size;
        uint64_t cycles_idle_empty, cycles_compute, cycles_idle_respfull;
        double   pct_busy_cycles, starvation_pct, backpressure_pct;
        int      bottleneck_class;
        // C4.6.19: Research sprint instrumentation
        uint64_t ring_full_count;
        uint64_t inline_drain_calls, inline_drain_items;
        uint64_t outer_poll_calls, outer_poll_items;
        uint64_t quiesce_items;
        double   time_submit_ns, time_inline_drain_ns, time_outer_poll_ns, quiesce_time_ns;
    };
    ContinuousFeederResult fast_path_run_continuous_feeder(
        FastPathHandle* handle,
        double duration_sec,
        uint32_t target_depth,
        uint32_t refill_threshold,
        uint32_t batch_cap,
        uint32_t poll_max,
        uint16_t opcode,
        uint16_t input_len,
        uint8_t  flags,
        int      source_mode,
        uint32_t source_seed,
        uint32_t poll_burst = 0);

    // Flywheel-25: C++ flywheel scheduler
    struct FlywheelCompletionRecord {
        uint64_t request_id;
        uint8_t  success;
        uint8_t  status;
        uint8_t  r[32];
        uint8_t  s[32];
        uint8_t  _pad[6];
    };
    struct FlywheelState;  // opaque to pybind11

    FlywheelState* fast_path_flywheel_create(
        FastPathHandle* engine, uint32_t target_in_flight, uint32_t refill_threshold,
        uint32_t submit_quantum, uint32_t max_submit_rounds, uint32_t max_drain_rounds,
        uint32_t poll_max, uint8_t flags);
    void fast_path_flywheel_destroy(FlywheelState* fw);
    uint64_t fast_path_flywheel_submit_ids(FlywheelState* fw, const uint8_t* payload_blob, uint32_t count);
    uint32_t fast_path_flywheel_submit(FlywheelState* fw, const uint8_t* packed_blob, uint32_t count);
    uint32_t fast_path_flywheel_tick(FlywheelState* fw, FlywheelCompletionRecord* out, uint32_t out_max);
    // Accessor functions (FlywheelState is opaque here)
    uint64_t fast_path_flywheel_get_tick_count(FlywheelState* fw);
    uint64_t fast_path_flywheel_get_submitted(FlywheelState* fw);
    uint64_t fast_path_flywheel_get_completed(FlywheelState* fw);
    uint64_t fast_path_flywheel_get_errors(FlywheelState* fw);
    uint64_t fast_path_flywheel_get_submit_calls(FlywheelState* fw);
    uint64_t fast_path_flywheel_get_drain_calls(FlywheelState* fw);
    uint64_t fast_path_flywheel_get_refill_events(FlywheelState* fw);
    uint64_t fast_path_flywheel_get_ring_full(FlywheelState* fw);
    uint32_t fast_path_flywheel_get_in_flight(FlywheelState* fw);
    uint32_t fast_path_flywheel_get_pending(FlywheelState* fw);
    uint32_t fast_path_flywheel_submit_raw(FlywheelState* fw, const uint8_t* payload_blob, uint32_t count);
    uint32_t fast_path_flywheel_run_raw(FlywheelState* fw, const uint8_t* payload_blob, uint32_t count, double timeout_sec);
    uint32_t fast_path_flywheel_run(FlywheelState* fw, FlywheelCompletionRecord* out,
        uint32_t out_max, uint32_t target_completions, double timeout_sec);

    struct ContinuousRunResult {
        uint32_t completed;
        uint32_t submitted;
        uint32_t errors;
        double elapsed_sec;
        uint32_t ring_full_count;
        uint32_t avg_inflight;
        uint32_t target_depth;
        uint32_t refill_threshold;
    };
    ContinuousRunResult fast_path_flywheel_run_continuous(FlywheelState* fw, double timeout_sec);

    // Card 26: Comb table loading for warp-coop mode (only when P256_WARP_COOP_ENABLED=1)
#if P256_WARP_COOP_ENABLED
    cudaError_t load_p256_comb_table(const uint32_t* h_x_table, const uint32_t* h_y_table, size_t table_size);
#if P256_FASTPATH_FULLWINDOW
    // SPEED-H2-01: full-window fixed-base table (payload only, AoS, no header)
    cudaError_t load_p256_fullwindow_table(const uint32_t* h_payload, size_t n_u32);
    // SPEED-H2-01 GAP 4: read back one 16-u32 entry of the device global
    // P256_G_FULLWINDOW (device-side sentinel verification after upload).
    // Lives in engine_loop.cu — same TU as the symbol the kernel reads.
    cudaError_t readback_p256_fullwindow_entry(size_t entry_index, uint32_t out_entry[16]);
#endif
#endif

    // Card 29: Rotation probe + staged rotation ops
    bool engine_rotation_probe(EngineHandle* handle,
                               Keyslot out_slots[2],
                               uint32_t* out_active_epoch);
    bool engine_rotate_key(EngineHandle* handle, const uint8_t private_key[32]);
    bool engine_prepare_next_key(EngineHandle* handle, const uint8_t private_key[32]);
    bool engine_commit_rotation(EngineHandle* handle);
    bool engine_rollback_rotation(EngineHandle* handle);

    // Card 78: Typed rotation — prepare next key with arbitrary key type
    bool engine_prepare_next_key_typed(EngineHandle* handle,
                                        KeyslotType key_type,
                                        const uint8_t* key_data,
                                        uint16_t key_len);
    bool engine_prepare_next_key_typed_async(EngineHandle* handle,
                                              KeyslotType key_type,
                                              const uint8_t* key_data,
                                              uint16_t key_len);

    // Card 32B: Async rotation (zero-sync batch primitives)
    bool engine_prepare_next_key_async(EngineHandle* handle, const uint8_t private_key[32]);
    bool engine_commit_rotation_async(EngineHandle* handle);
    bool engine_rotation_fence(EngineHandle* handle);

    // Card 32C: Query pending fence state
    bool engine_rotation_pending_fence(EngineHandle* handle);

    // Card 64: Force recovery from stuck fence state
    bool engine_rotation_force_recovery(EngineHandle* handle);
}
}

// ============================================================================
// Internal Engine State - Wraps Real Persistent Engine
// ============================================================================

struct SmokeEngine {
    // ABI version this engine was created with
    uint32_t abi_version;

    // Device index
    int device_index;

    // Statistics
    uint64_t total_sign_p256;
    uint64_t total_hash_sha256;
    uint64_t total_errors;
    uint64_t start_time_ms;

    // Card 24.1: Real persistent engine handle
    smoke::engine::EngineHandle* persistent_engine;
    bool engine_running;

    // Card 26.82: FAST PATH handle for generic submit/poll C ABI
    smoke::engine::FastPathHandle* fast_path;
    bool fast_path_running;

    // Request ID counter for sign operations
    std::atomic<uint32_t> next_request_id;

    // Key tracking (for ABI key_id mapping)
    bool key_loaded;
    uint32_t loaded_key_id;

    // Card 32C: Async rotation fence latch (ABI-level mirror of engine state)
    // Set by async rotation calls, cleared by fence. Guards signing APIs.
    bool rotation_pending_fence;

    // N5 fail-closed (2026-05-16): P-256 warp-coop signing requires the
    // generator-point comb table to be uploaded to __constant__ memory
    // via smoke_engine_load_p256_comb_table(). Without it, scalar_mul_g
    // silently produces point-at-infinity (Rz=0), the batch inversion
    // returns all zeros, and every signature emerges as r=0 / CRYPTO_ERROR.
    // smoke_fast_path_start refuses to launch the kernel if
    // has_p256_key && !p256_comb_loaded — fail-closed init.
    bool has_p256_key;
    bool p256_comb_loaded;
    // SPEED-H2-01: full-window fixed-base table (zero-doubling scalar-mul),
    // only meaningful when the engine is built P256_FASTPATH_FULLWINDOW.
    // Same fail-closed contract as the comb table: if the compiled path is
    // fullwindow and this is false, smoke_fast_path_start refuses to launch.
    bool p256_fullwindow_loaded;
};

// ============================================================================
// Helper: Get current time in milliseconds
// ============================================================================

static uint64_t get_time_ms() {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
    return static_cast<uint64_t>(ms);
}

// ============================================================================
// D-080: bind the calling thread to the engine's CUDA device.
//
// cudaSetDevice is thread-local. ABI entry points may be invoked from
// different host threads over an engine's lifetime — e.g. a Go caller's
// goroutine migrates across OS threads between cgo calls (D-080), so the
// device selected in smoke_engine_init does not necessarily hold on the
// thread that later runs create / start / a key load. Device-sensitive
// operations (allocation, kernel launch, H2D memcpy, free) must therefore
// re-assert the device, or they hit the default device and fail with
// "invalid resource handle" / cross-device errors.
//
// This is defense-in-depth: the caller-side runtime.LockOSThread fix
// (smoke-ipc-go, D-080) already pins setup, but this makes the ABI robust
// for ANY FFI caller regardless of its threading. cudaSetDevice is cheap
// and idempotent when the device is already current, so it is safe at
// setup/teardown entry points. It is deliberately NOT added to the
// submit/poll hot paths, which operate on host-mapped memory and do not
// need a device context once the persistent kernel is running.
static inline void smoke_bind_device(const SmokeEngine* engine) {
    if (engine && engine->device_index >= 0) {
        cudaSetDevice(engine->device_index);
    }
}

// ============================================================================
// Engine Lifecycle
// ============================================================================

extern "C"
SmokeStatus smoke_engine_init(
    const SmokeEngineConfig* config,
    SmokeEngineHandle* out_engine,
    SmokeErrorInfo* out_error
) {
    // Validate args
    if (!config || !out_engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "config and out_engine must not be NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Check ABI version
    if (config->abi_version != SMOKE_ENGINE_ABI_VERSION) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "ABI version mismatch: got %u, expected %u",
                     config->abi_version, SMOKE_ENGINE_ABI_VERSION);
        }
        return SMOKE_STATUS_ERROR_UNSUPPORTED;
    }

    // Check CUDA device
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = static_cast<int32_t>(cuda_err);
            snprintf(out_error->message, sizeof(out_error->message),
                     "No CUDA devices available");
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    int device_idx = config->device_index;
    if (device_idx < 0) {
        device_idx = 0;  // Default to first device
    }
    if (device_idx >= device_count) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Device index %d out of range (0..%d)",
                     device_idx, device_count - 1);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Set device
    cuda_err = cudaSetDevice(device_idx);
    if (cuda_err != cudaSuccess) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = static_cast<int32_t>(cuda_err);
            snprintf(out_error->message, sizeof(out_error->message),
                     "Failed to set CUDA device %d", device_idx);
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    // Allocate engine
    SmokeEngine* engine = new (std::nothrow) SmokeEngine();
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_UNKNOWN;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Failed to allocate engine");
        }
        return SMOKE_STATUS_ERROR_UNKNOWN;
    }

    // Initialize engine state
    engine->abi_version = config->abi_version;
    engine->device_index = device_idx;
    engine->total_sign_p256 = 0;
    engine->total_hash_sha256 = 0;
    engine->total_errors = 0;
    engine->start_time_ms = get_time_ms();
    engine->key_loaded = false;
    engine->loaded_key_id = 0;
    engine->next_request_id.store(0);
    engine->persistent_engine = nullptr;
    engine->engine_running = false;
    engine->fast_path = nullptr;
    engine->fast_path_running = false;
    engine->p256_fullwindow_loaded = false;  // SPEED-H2-01 fail-closed init

    // Card 24.1: Create and start the real persistent P-256 engine
    // Card 26.77: Removed debug fprintf to minimize stack usage.
    engine->persistent_engine = smoke::engine::engine_create();
    if (!engine->persistent_engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 10;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Failed to create persistent P-256 engine");
        }
        delete engine;
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    // Start the persistent kernel
    if (!smoke::engine::engine_start(engine->persistent_engine)) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 11;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Failed to start persistent P-256 engine kernel");
        }
        smoke::engine::engine_destroy(engine->persistent_engine);
        delete engine;
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    engine->engine_running = true;

    *out_engine = engine;
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_shutdown(SmokeEngineHandle engine) {
    if (!engine) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    smoke_bind_device(engine); // D-080: free on the engine's device

    // Card 26.82: Clean up FAST PATH first
    if (engine->fast_path) {
        if (engine->fast_path_running) {
            smoke::engine::fast_path_stop(engine->fast_path);
            engine->fast_path_running = false;
        }
        smoke::engine::fast_path_destroy(engine->fast_path);
        engine->fast_path = nullptr;
    }

    // Card 24.1: Shutdown and destroy the persistent engine
    if (engine->persistent_engine) {
        if (engine->engine_running) {
            smoke::engine::engine_shutdown(engine->persistent_engine);
            engine->engine_running = false;
        }
        smoke::engine::engine_destroy(engine->persistent_engine);
        engine->persistent_engine = nullptr;
    }

    delete engine;
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_get_stats(
    SmokeEngineHandle engine,
    SmokeEngineStats* out_stats
) {
    if (!engine || !out_stats) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Card 24.5: Read total_sign_p256 from GPU if persistent engine is running
    // The GPU kernel updates its own counter, so we read from device memory
    // to get the authoritative count. Fall back to local counter if no engine.
    if (engine->persistent_engine && engine->engine_running) {
        out_stats->total_sign_p256 = smoke::engine::engine_get_total_sign_ops(
            engine->persistent_engine
        );
    } else {
        out_stats->total_sign_p256 = engine->total_sign_p256;
    }

    out_stats->total_hash_sha256 = engine->total_hash_sha256;
    out_stats->total_errors = engine->total_errors;
    out_stats->uptime_ms = get_time_ms() - engine->start_time_ms;
    memset(out_stats->reserved_u64, 0, sizeof(out_stats->reserved_u64));

    return SMOKE_STATUS_OK;
}

// ============================================================================
// Key Management
// ============================================================================

extern "C"
SmokeStatus smoke_engine_load_key_p256(
    SmokeEngineHandle engine,
    const SmokeP256KeyId* key_id,
    const SmokeP256PrivateKey* key
) {
    if (!engine || !key_id || !key) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    smoke_bind_device(engine); // D-080: H2D key copy on the engine's device

    // FAST PATH creation stops the legacy kernel but keeps its keyslot state.
    // Key loading remains valid before and after FAST PATH start; the table
    // checks below protect loading into a running signer.
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    // For v0: only support key_id 0 (maps to active keyslot)
    if (key_id->id != 0) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // SPEED-H2-01 GAP 3 (2026-07-22): key-after-start ordering. The N5/C4
    // gates in smoke_fast_path_start only protect keys loaded BEFORE launch.
    // Loading a P-256 key into an already-RUNNING fast path whose signer
    // table(s) are absent would arm the exact silent point-at-infinity
    // failure those gates exist to prevent (r=0 / CRYPTO_ERROR on every
    // sign, no fail-closed signal). Refuse fail-closed: structured status,
    // no partial state (nothing has been written yet). Required tables for
    // the compiled signer path: comb always; full-window additionally on
    // P256_FASTPATH_FULLWINDOW builds. Mirrors the smoke_fast_path_start
    // gate conditions exactly.
    if (engine->fast_path && engine->fast_path_running) {
        if (!engine->p256_comb_loaded) {
            return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
        }
#if P256_FASTPATH_FULLWINDOW
        if (!engine->p256_fullwindow_loaded) {
            return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
        }
#endif
    }

    // Card 29: Use engine_set_key_typed to set key_type and key_len (required for rotation probe)
    if (!smoke::engine::engine_set_key_typed(engine->persistent_engine, 0,
            smoke::engine::KeyslotType::KEYSLOT_P256, key->d, 32)) {
        engine->total_errors++;
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    engine->loaded_key_id = key_id->id;
    engine->key_loaded = true;
    // N5 fail-closed: record that a P-256 keyslot exists so that
    // smoke_fast_path_start can refuse to launch without the comb table.
    engine->has_p256_key = true;

    return SMOKE_STATUS_OK;
}

// sha256_hash is defined later in this TU (~line 1364); forward-declared
// here so smoke_engine_get_measurement (below) can use it. Same function
// smoke_engine_rotation_probe's integrity-tag helper uses.
static void sha256_hash(const uint8_t* data, size_t len, uint8_t digest[32]);

// ============================================================================
// SBE-002 (2026-08-17): Engine Measurement Export
// ============================================================================
// Exports a measurement of the engine's RUNTIME identity — which ABI build,
// which public key(s) are loaded (by hash, never private material), and the
// active rotation epoch. See the doc comment in smoke_engine.h for the exact
// hash layout and the honest scope of what this export does and does not
// prove (it does NOT measure the calling process/daemon — that is a
// deployment-level TDX RTMR3 job; see
// engine/memory-bank/architecture/attested_key_custody.md).
//
// Fail-closed: reads the same raw keyslot snapshot smoke_engine_rotation_probe
// uses (smoke::engine::engine_rotation_probe), zeroizes the local copy before
// returning, and never reads or hashes a private scalar.
extern "C"
SmokeStatus smoke_engine_get_measurement(
    SmokeEngineHandle        engine,
    SmokeEngineMeasurement*  out_measurement
) {
    if (!out_measurement) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    memset(out_measurement, 0, sizeof(SmokeEngineMeasurement));

    if (!engine) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine || !engine->engine_running) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    smoke::engine::Keyslot slots[2];
    uint32_t active_epoch = 0;
    if (!smoke::engine::engine_rotation_probe(engine->persistent_engine, slots, &active_epoch)) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    // Build the stable identity message. Public-key bytes only — never
    // private_d / private_d_mirror. Non-P-256 slot types (or empty slots)
    // contribute a 32-zero-byte placeholder since there is no ABI-level
    // public-key export for symmetric/HMAC/RSA keyslots in v1.
    std::vector<uint8_t> msg;
    const char* domain = "smoke-engine-measurement-v1";
    msg.insert(msg.end(), domain, domain + strlen(domain) + 1 /* incl NUL */);

    auto append_u32_le = [&msg](uint32_t v) {
        msg.push_back(static_cast<uint8_t>(v & 0xFF));
        msg.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
        msg.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
        msg.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    };

    append_u32_le(SMOKE_ENGINE_ABI_VERSION);
    append_u32_le(2u); // num_key_slots considered (fixed at 2 in v1 rotation design)

    uint8_t num_loaded = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        append_u32_le(i);
        append_u32_le(slots[i].key_type);
        uint8_t pk_hash[32] = {0};
        const uint32_t EMPTY = static_cast<uint32_t>(smoke::engine::KeyslotType::KEYSLOT_EMPTY);
        const uint32_t P256 = static_cast<uint32_t>(smoke::engine::KeyslotType::KEYSLOT_P256);
        if (slots[i].key_type == P256) {
            uint8_t pub[64];
            memcpy(pub, slots[i].pub_x, 32);
            memcpy(pub + 32, slots[i].pub_y, 32);
            sha256_hash(pub, sizeof(pub), pk_hash);
            memset(pub, 0, sizeof(pub));
        }
        if (slots[i].key_type != EMPTY) {
            num_loaded++;
        }
        msg.insert(msg.end(), pk_hash, pk_hash + 32);
    }
    append_u32_le(active_epoch);

    uint8_t digest[32];
    sha256_hash(msg.data(), msg.size(), digest);
    memset(msg.data(), 0, msg.size()); // no secrets in msg, but wipe on general principle

    // Zeroize the local keyslot snapshot (contains private_d / mirror even
    // though we never read them) before returning — same discipline as
    // smoke_engine_rotation_probe.
    memset(slots, 0, sizeof(slots));

    memcpy(out_measurement->engine_measurement, digest, 32);
    out_measurement->active_epoch = active_epoch;
    out_measurement->num_key_slots = num_loaded;
    out_measurement->abi_version = SMOKE_ENGINE_ABI_VERSION;

    return SMOKE_STATUS_OK;
}

// ============================================================================
// N5 P-256 Comb Table Loader (2026-05-16)
// ============================================================================
// Upload the precomputed generator-point comb table (w=8, 8 limbs × 255
// entries × 2 coordinates) to GPU __constant__ memory. REQUIRED before
// smoke_fast_path_start when any P-256 keyslot is loaded — otherwise the
// warp-cooperative scalar multiplication produces point-at-infinity for
// every nonce, and every signature emerges as r=0 / CRYPTO_ERROR
// (silently, no fail-closed signal from the kernel).
//
// Expected layout per table: SOA, [limb * 255 + entry], uint32. The
// underlying smoke::engine::load_p256_comb_table validates table_size ==
// 8*255 and transposes to AOS internally before cudaMemcpyToSymbol.
//
// This was previously exposed only via pybind11; the Go cgo path had no
// way to populate it, which is what caused the 100% P-256 failure rate
// in bench-cgo-pipeline and bench-inproc (telemetry: zinv_zero ≈
// x_full_zero ≈ r_zero ≈ phase3_fail ≈ 100%).
extern "C" SMOKE_API
SmokeStatus smoke_engine_load_p256_comb_table(
    SmokeEngineHandle engine,
    const uint32_t*   x_table,
    const uint32_t*   y_table,
    size_t            table_size
) {
    if (!engine || !x_table || !y_table) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (table_size != 8u * 255u) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    // Symbol copies use the default stream. Refuse while either persistent
    // kernel runs rather than waiting forever for its stream to complete.
    // FAST PATH creation stops the legacy kernel; load tables before start.
    if (engine->engine_running || engine->fast_path_running) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    smoke_bind_device(engine); // D-080: __constant__ upload on the engine's device

#if P256_WARP_COOP_ENABLED
    cudaError_t err = smoke::engine::load_p256_comb_table(x_table, y_table, table_size);
    if (err != cudaSuccess) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    engine->p256_comb_loaded = true;
    // Mirror into FAST PATH telemetry if the fast path already exists.
    // The telemetry flag is informational; the host-side bool above is
    // what gates smoke_fast_path_start.
    if (engine->fast_path) {
        smoke::engine::fast_path_set_comb_table_loaded(engine->fast_path, 1);
    }
    return SMOKE_STATUS_OK;
#else
    (void)x_table; (void)y_table; (void)table_size;
    return SMOKE_STATUS_ERROR_UNSUPPORTED;
#endif
}

// ============================================================================
// SPEED-H2-01 (2026-07-20): full-window fixed-base table loader.
//
// Loads the zero-doubling k*G table (32 windows x 128 signed multiples,
// 256 KB, AoS Montgomery) into the __device__ global P256_G_FULLWINDOW.
//
// Fail-closed integrity stack (architect conditions C1-C5):
//   C1  returns SmokeStatus, namespaced codes, no partial device state on error.
//   C2  SHA-256 of the WHOLE blob (header+payload) is compared against a
//       BUILD-TIME COMPILED golden constant. The caller-supplied fp is only a
//       secondary cross-check — the compiled golden is the trust anchor.
//   C3  the 32-byte typed header is parsed and validated (magic + table_type +
//       dims); a comb blob (or any other) is refused here, not misread.
//   C5  on any mismatch the device global is left untouched and a structured
//       error is returned (never warn-and-continue).
//
// Only present when built P256_FASTPATH_FULLWINDOW; otherwise UNSUPPORTED.
// ============================================================================
#if P256_WARP_COOP_ENABLED && P256_FASTPATH_FULLWINDOW
// Golden fingerprint of engine/data/p256_fullwindow/p256_G_fullwindow_w8.bin
// (v2, typed header). Regenerate the table => update this constant in the
// same commit. This is the trust anchor (condition C2).
static const char* kP256FullwindowGoldenSha256 =
    "dbe88875e99034dc395b8101fcc42d0462bd8cd080fab5dd8e40ee7c49142b79";

// Forward decl: sha256_hash is defined later in this TU (line ~1255).
static void sha256_hash(const uint8_t* data, size_t len, uint8_t digest[32]);

// Typed header (must match gen_p256_fullwindow_table.py build_header()).
#pragma pack(push, 1)
struct P256FullwindowHeader {
    char     magic[8];        // "SMKTBL01"
    uint32_t table_type;      // 2 = fullwindow_w8
    uint32_t version;         // 1
    uint32_t windows;         // 32
    uint32_t digits;          // 128
    uint32_t entry_bytes;     // 64
    uint32_t reserved;        // 0
};
#pragma pack(pop)
static_assert(sizeof(P256FullwindowHeader) == 32, "fullwindow header must be 32 bytes");

static void bytes_to_hex(const uint8_t* b, size_t n, char* out /* 2n+1 */) {
    static const char* h = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) { out[2*i] = h[b[i] >> 4]; out[2*i+1] = h[b[i] & 0xF]; }
    out[2*n] = '\0';
}

extern "C" SMOKE_API
SmokeStatus smoke_engine_load_p256_fullwindow_table(
    SmokeEngineHandle engine,
    const uint8_t*    blob,           // entire .bin: 32B header + 256KB payload
    size_t            blob_size,
    const char*       caller_fp_hex   // optional secondary cross-check (may be NULL)
) {
    if (!engine || !blob) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    constexpr size_t kHeaderBytes  = 32;
    constexpr size_t kEntries      = 33u * 128u;   // 33rd window = 2^256 carry
    constexpr size_t kPayloadBytes = kEntries * 16u * sizeof(uint32_t); // 270336
    if (blob_size != kHeaderBytes + kPayloadBytes) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (engine->engine_running || engine->fast_path_running) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // --- C3: typed header validation (refuse a comb/wrong/other blob) ---
    P256FullwindowHeader hdr;
    memcpy(&hdr, blob, sizeof(hdr));
    if (memcmp(hdr.magic, "SMKTBL01", 8) != 0 ||
        hdr.table_type != 2u || hdr.version != 1u ||
        hdr.windows != 33u || hdr.digits != 128u || hdr.entry_bytes != 64u) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // --- C2: fingerprint over the WHOLE blob vs the compiled golden ---
    uint8_t digest[32];
    sha256_hash(blob, blob_size, digest);
    char digest_hex[65];
    bytes_to_hex(digest, 32, digest_hex);
    if (strcmp(digest_hex, kP256FullwindowGoldenSha256) != 0) {
        // Tampered / wrong-vintage table. Device global untouched (C5).
        return SMOKE_STATUS_ERROR_VERIFY_FAILED;
    }
    // Secondary cross-check: if the caller passed a fingerprint, it must agree.
    if (caller_fp_hex && strcmp(digest_hex, caller_fp_hex) != 0) {
        return SMOKE_STATUS_ERROR_VERIFY_FAILED;
    }

    smoke_bind_device(engine); // D-080: upload on the engine's device
    const uint32_t* payload = reinterpret_cast<const uint32_t*>(blob + kHeaderBytes);
    cudaError_t err = smoke::engine::load_p256_fullwindow_table(payload, kEntries * 16u);
    if (err != cudaSuccess) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE; // no partial state (C1)
    }

    // --- GAP 4 (2026-07-22): device-side sentinel verification. Read back
    // the LAST table entry from the __device__ global P256_G_FULLWINDOW and
    // compare it against the same bytes of the validated host blob. This
    // makes the loaded-guarantee independent of host bookkeeping: a copy the
    // driver mis-applied, a wrong-symbol upload, or a truncated transfer
    // (last entry catches truncation) is detected here, not as silent r=0
    // signatures later. On any failure the loaded flag is NOT set, so
    // smoke_fast_path_start / GAP 3 keep refusing — fail closed. (The device
    // global may hold partial bytes at that point, but nothing ever reads it
    // while the flag is false.)
    {
        constexpr size_t kSentinelEntry = kEntries - 1;
        uint32_t dev_entry[16] = {};
        cudaError_t rb = smoke::engine::readback_p256_fullwindow_entry(
            kSentinelEntry, dev_entry);
        if (rb != cudaSuccess) {
            return SMOKE_STATUS_ERROR_DEVICE_FAILURE; // loaded flag NOT set
        }
        const uint32_t* host_entry = payload + kSentinelEntry * 16u;
        bool all_zero = true;
        for (int i = 0; i < 16; ++i) {
            if (dev_entry[i] != 0u) { all_zero = false; break; }
        }
        if (all_zero || memcmp(dev_entry, host_entry, sizeof(dev_entry)) != 0) {
            // All-zero = the upload never landed (fresh device global);
            // mismatch = the global holds bytes we did not validate.
            return SMOKE_STATUS_ERROR_VERIFY_FAILED; // loaded flag NOT set
        }
    }

    engine->p256_fullwindow_loaded = true;
    return SMOKE_STATUS_OK;
}
#else
extern "C" SMOKE_API
SmokeStatus smoke_engine_load_p256_fullwindow_table(
    SmokeEngineHandle, const uint8_t*, size_t, const char*) {
    return SMOKE_STATUS_ERROR_UNSUPPORTED;
}
#endif

// ============================================================================
// Card N5c (2026-06-01): Device SM-count query for caller-side CTA
// auto-scaling. Returns the multiprocessor (SM) count of `device`
// (pass -1 for the current/default device, which is what the engine
// selected during init). Callers clamp num_ctas/num_shards to this
// value to avoid the persistent-kernel CTA-oversubscription deadlock
// that smoke_fast_path_start now fails closed on (D-077). Pure device
// query — no engine handle required, safe to call any time after the
// CUDA device is selected.
// ============================================================================
extern "C" SMOKE_API
SmokeStatus smoke_engine_query_sm_count(int32_t device, int32_t* out_sm_count) {
    if (!out_sm_count) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    int dev = device;
    if (dev < 0) {
        cudaError_t ge = cudaGetDevice(&dev);
        if (ge != cudaSuccess) {
            return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
        }
    }
    int sm = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &sm, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    *out_sm_count = sm;
    return SMOKE_STATUS_OK;
}

// ============================================================================
// RRP-02 (2026-07-10): Device-name query for the daemon's validated-GPU
// boot gate. Returns the CUDA device name of `device` (pass -1 for the
// current/default device) as a NUL-terminated string in out_name. Every
// new GPU class so far has broken P-256 in a new way (D-082 kernel
// miscompile on L40S; A100 response drops), so the daemon refuses to
// boot on models it cannot identify — this export is what identifies
// them. Same cudaGetDeviceProperties source as the Python-only
// engine_get_device_name binding (engine_device_info.cpp). Pure device
// query — no engine handle required. out_name is written only on
// SMOKE_STATUS_OK; a name that does not fit max_len (including the NUL)
// is an error, never a truncation — a truncated model name could alias
// a different, unvalidated GPU in the caller's allowlist.
// ============================================================================
extern "C" SMOKE_API
SmokeStatus smoke_engine_query_device_name(int32_t device, char* out_name,
                                           int32_t max_len) {
    if (!out_name || max_len <= 0) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    int dev = device;
    if (dev < 0) {
        cudaError_t ge = cudaGetDevice(&dev);
        if (ge != cudaSuccess) {
            return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
        }
    }
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    size_t name_len = strnlen(prop.name, sizeof(prop.name));
    if (name_len + 1 > static_cast<size_t>(max_len)) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    memcpy(out_name, prop.name, name_len);
    out_name[name_len] = '\0';
    return SMOKE_STATUS_OK;
}

// ============================================================================
// P-256 Sign - Card 24.1: Real Implementation via Persistent Engine
// ============================================================================

extern "C"
SmokeStatus smoke_engine_sign_p256(
    SmokeEngineHandle engine,
    const SmokeP256SignRequest* req,
    SmokeP256SignResponse* resp,
    SmokeErrorInfo* out_error
) {
    if (!engine || !req || !resp) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "engine, req, and resp must not be NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Check engine is running
    if (!engine->persistent_engine || !engine->engine_running) {
        engine->total_errors++;
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Persistent engine not running");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    // Card 32C: Refuse to sign while async rotation is pending fence
    if (engine->rotation_pending_fence) {
        engine->total_errors++;
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_FENCE_REQUIRED;
            out_error->internal_code = 10;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Async rotation pending. Call rotation_fence() before signing.");
        }
        return SMOKE_STATUS_ERROR_FENCE_REQUIRED;
    }

    // Check key is loaded
    if (!engine->key_loaded) {
        engine->total_errors++;
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_KEY_MISSING;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "No P-256 key loaded");
        }
        return SMOKE_STATUS_ERROR_KEY_MISSING;
    }

    // Check key ID matches (v0: only support key_id 0)
    if (req->key_id.id != engine->loaded_key_id) {
        engine->total_errors++;
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_KEY_MISSING;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Key ID %u not found (have %u)",
                     req->key_id.id, engine->loaded_key_id);
        }
        return SMOKE_STATUS_ERROR_KEY_MISSING;
    }

    // Validate message hash
    if (!req->msg.data || req->msg.len != 32) {
        engine->total_errors++;
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 5;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Message hash must be exactly 32 bytes");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Card 24.1: Submit request to persistent engine
    uint32_t request_id = engine->next_request_id.fetch_add(1);

    if (!smoke::engine::engine_submit_request(
            engine->persistent_engine,
            req->msg.data,
            request_id)) {
        engine->total_errors++;
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 10;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Failed to submit sign request to persistent engine");
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    // Poll for response (with timeout)
    smoke::engine::EngineResponse engine_resp{};
    const int MAX_POLL_ATTEMPTS = 100000;  // 5 seconds at ~50µs per attempt
    const int YIELD_THRESHOLD = 100;       // Yield for first 100 iterations
    const int SPIN_THRESHOLD = 1000;       // Then 10µs sleeps up to this point
    bool got_response = false;

    for (int i = 0; i < MAX_POLL_ATTEMPTS; ++i) {
        if (smoke::engine::engine_poll_response(
                engine->persistent_engine,
                request_id,
                &engine_resp)) {
            got_response = true;
            break;
        }
        // Tiered backoff
        if (i < YIELD_THRESHOLD) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        } else if (i < SPIN_THRESHOLD) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    if (!got_response) {
        engine->total_errors++;
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_TIMEOUT;
            out_error->internal_code = 20;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Timeout waiting for sign response (request_id=%u)", request_id);
        }
        return SMOKE_STATUS_ERROR_TIMEOUT;
    }

    // Card 24.1: Map engine response status to ABI status
    // Engine status: 0=empty, 1=ok, 2=error, 3=integrity_error
    SmokeStatus status = SMOKE_STATUS_OK;

    switch (engine_resp.status) {
        case 1:  // OK
            status = SMOKE_STATUS_OK;
            memcpy(resp->r, engine_resp.sig_r, 32);
            memcpy(resp->s, engine_resp.sig_s, 32);
            memset(resp->reserved, 0, 32);
            engine->total_sign_p256++;
            break;

        case 3:  // INTEGRITY_ERROR (Card 18)
            status = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            memset(resp->r, 0, 32);
            memset(resp->s, 0, 32);
            memset(resp->reserved, 0, 32);
            engine->total_errors++;
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
                out_error->internal_code = 30;
                snprintf(out_error->message, sizeof(out_error->message),
                         "Keyslot integrity check failed");
            }
            break;

        case 2:  // ERROR
        default:
            status = SMOKE_STATUS_ERROR_UNKNOWN;
            memset(resp->r, 0, 32);
            memset(resp->s, 0, 32);
            memset(resp->reserved, 0, 32);
            engine->total_errors++;
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_UNKNOWN;
                out_error->internal_code = 31;
                snprintf(out_error->message, sizeof(out_error->message),
                         "Sign failed with engine status=%u", engine_resp.status);
            }
            break;
    }

    return status;
}

// ============================================================================
// SHA-256 Implementation (CPU for v0, GPU later)
// ============================================================================

// SHA-256 constants
static const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static const uint32_t SHA256_IV[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_compress_block(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];

    // Load block as big-endian words
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }

    // Expand
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(W[i-15], 7) ^ rotr32(W[i-15], 18) ^ (W[i-15] >> 3);
        uint32_t s1 = rotr32(W[i-2], 17) ^ rotr32(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    // Working variables
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    // 64 rounds
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + SHA256_K[i] + W[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    // Add to state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void sha256_hash(const uint8_t* data, size_t len, uint8_t digest[32]) {
    uint32_t state[8];
    memcpy(state, SHA256_IV, sizeof(state));

    // Process complete blocks
    size_t num_blocks = len / 64;
    for (size_t i = 0; i < num_blocks; ++i) {
        sha256_compress_block(state, data + i * 64);
    }

    // Final block with padding
    uint8_t final_block[128];
    size_t remaining = len % 64;
    memcpy(final_block, data + num_blocks * 64, remaining);

    // Append 0x80
    final_block[remaining] = 0x80;

    // Check if we need one or two blocks
    size_t pad_len;
    if (remaining < 56) {
        // Single block
        pad_len = 56 - remaining - 1;
        memset(final_block + remaining + 1, 0, pad_len);
        // Append length in bits (big-endian)
        uint64_t bit_len = (uint64_t)len * 8;
        for (int i = 0; i < 8; ++i) {
            final_block[56 + i] = (bit_len >> (56 - i * 8)) & 0xFF;
        }
        sha256_compress_block(state, final_block);
    } else {
        // Two blocks
        memset(final_block + remaining + 1, 0, 64 - remaining - 1);
        sha256_compress_block(state, final_block);

        memset(final_block, 0, 56);
        uint64_t bit_len = (uint64_t)len * 8;
        for (int i = 0; i < 8; ++i) {
            final_block[56 + i] = (bit_len >> (56 - i * 8)) & 0xFF;
        }
        sha256_compress_block(state, final_block);
    }

    // Output digest (big-endian)
    for (int i = 0; i < 8; ++i) {
        digest[i*4] = (state[i] >> 24) & 0xFF;
        digest[i*4+1] = (state[i] >> 16) & 0xFF;
        digest[i*4+2] = (state[i] >> 8) & 0xFF;
        digest[i*4+3] = state[i] & 0xFF;
    }
}

extern "C"
SmokeStatus smoke_engine_hash_sha256(
    SmokeEngineHandle engine,
    const SmokeSHA256HashRequest* req,
    SmokeSHA256HashResponse* resp,
    SmokeErrorInfo* out_error
) {
    if (!engine || !req || !resp) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "engine, req, and resp must not be NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Card 26.78: FAIL-LOUD — CPU fallback disabled by default
    if (!smoke_cpu_fallback_enabled()) {
        smoke_fail_loud_once("smoke_engine_hash_sha256",
            "CPU fallback DISABLED. Use fast_path_submit_batch(opcode=10) for GPU SHA-256, "
            "or set SMOKE_ALLOW_CPU_FALLBACK=1 for debug.");
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_IMPLEMENTED;
            out_error->internal_code = 100;
            snprintf(out_error->message, sizeof(out_error->message),
                     "CPU fallback disabled. Use FAST PATH GPU for SHA-256.");
        }
        return SMOKE_STATUS_ERROR_NOT_IMPLEMENTED;
    }

    smoke_fail_loud_once("smoke_engine_hash_sha256",
        "CPU fallback ENABLED (SMOKE_ALLOW_CPU_FALLBACK=1). Running SHA256 on CPU.");

    // Compute SHA-256 hash (CPU implementation for v0)
    sha256_hash(req->input.data, req->input.len, resp->digest);
    memset(resp->reserved, 0, 32);

    // Update stats
    engine->total_hash_sha256++;

    return SMOKE_STATUS_OK;
}

// ============================================================================
// Card 29: Rotation Probe + Staged Rotation (C ABI)
// ============================================================================

// P-256 order N as big-endian bytes for scalar range check
static const uint8_t PROBE_P256_N[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xBC, 0xE6, 0xFA, 0xAD, 0xA7, 0x17, 0x9E, 0x84,
    0xF3, 0xB9, 0xCA, 0xC2, 0xFC, 0x63, 0x25, 0x51
};

static int probe_bigendian_cmp32(const uint8_t a[32], const uint8_t b[32]) {
    for (int i = 0; i < 32; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static bool probe_is_zero_32(const uint8_t v[32]) {
    for (int i = 0; i < 32; ++i) {
        if (v[i] != 0) return false;
    }
    return true;
}

// Card 29: Recompute integrity tag for probe verification.
// MUST exactly mirror compute_integrity_tag_host() in engine_host.cpp:194.
// Construction: tag = SHA-256("smoke:keyslot-tag" || slot_index || integrity_version || private_d)
// This is a plain SHA-256 hash (NOT HMAC). If the engine ever changes to HMAC
// or a keyed construction, this function MUST be updated in lockstep.
static void probe_compute_integrity_tag(
    uint8_t tag_out[32],
    uint32_t slot_index,
    uint32_t integrity_version,
    const uint8_t private_d[32])
{
    // Build message: prefix (17) + slot_index (4) + version (4) + private_d (32) = 57 bytes
    uint8_t msg[57];

    const char* prefix = "smoke:keyslot-tag";
    for (int i = 0; i < 17; ++i) {
        msg[i] = static_cast<uint8_t>(prefix[i]);
    }

    // slot_index big-endian
    msg[17] = (slot_index >> 24) & 0xFF;
    msg[18] = (slot_index >> 16) & 0xFF;
    msg[19] = (slot_index >> 8) & 0xFF;
    msg[20] = slot_index & 0xFF;

    // integrity_version big-endian
    msg[21] = (integrity_version >> 24) & 0xFF;
    msg[22] = (integrity_version >> 16) & 0xFF;
    msg[23] = (integrity_version >> 8) & 0xFF;
    msg[24] = integrity_version & 0xFF;

    // private_d
    for (int i = 0; i < 32; ++i) {
        msg[25 + i] = private_d[i];
    }

    sha256_hash(msg, 57, tag_out);
}

extern "C"
SmokeStatus smoke_engine_rotation_probe(
    SmokeEngineHandle engine,
    SmokeRotationProbeResult* out
) {
    if (!engine || !out) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    memset(out, 0, sizeof(SmokeRotationProbeResult));

    if (!engine->persistent_engine) {
        out->probe_error = 1;
        snprintf(out->error_message, sizeof(out->error_message),
                 "Persistent engine not initialized");
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    // Read raw keyslot data from device
    smoke::engine::Keyslot slots[2];
    uint32_t active_epoch = 0;
    if (!smoke::engine::engine_rotation_probe(engine->persistent_engine, slots, &active_epoch)) {
        out->probe_error = 2;
        snprintf(out->error_message, sizeof(out->error_message),
                 "Device read failed");
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    out->active_epoch = active_epoch;
    out->num_slots = 2;

    // Run inline checks on each slot
    for (uint32_t i = 0; i < 2; ++i) {
        SmokeSlotProbeResult& sp = out->slots[i];
        const smoke::engine::Keyslot& slot = slots[i];

        sp.slot_index = i;
        sp.epoch = slot.epoch;
        sp.key_type = slot.key_type;
        sp.key_len = slot.key_len;
        sp.fault_flags = slot.fault_flags;
        sp.integrity_version = slot.integrity_version;
        sp.d_top_byte = slot.private_d[0];
        sp.pub_x_top_byte = slot.pub_x[0];

        // Default Python-side checks to unchecked
        sp.check_on_curve = SMOKE_PROBE_UNCHECKED;
        sp.check_not_infinity = SMOKE_PROBE_UNCHECKED;

        // If slot is empty, mark all checks as 0 (pass/n-a) and continue
        if (slot.key_type == static_cast<uint32_t>(smoke::engine::KeyslotType::KEYSLOT_EMPTY)) {
            sp.healthy = 1;
            continue;
        }

        // --- Mirror match (all key types) ---
        sp.check_mirror_match = (memcmp(slot.private_d, slot.private_d_mirror, 32) == 0) ? 0 : 1;

        // --- Fault flags (all key types) ---
        sp.check_no_fault_flags = (slot.fault_flags == smoke::engine::FAULT_NONE) ? 0 : 1;

        // --- Version monotonic (all key types) ---
        sp.check_version_monotonic = (slot.integrity_version >= 1) ? 0 : 1;

        // --- Key length valid (all key types) ---
        uint32_t expected_key_len = 0;
        switch (static_cast<smoke::engine::KeyslotType>(slot.key_type)) {
            case smoke::engine::KeyslotType::KEYSLOT_P256:    expected_key_len = 32; break;
            case smoke::engine::KeyslotType::KEYSLOT_AES256:  expected_key_len = 32; break;
            case smoke::engine::KeyslotType::KEYSLOT_HMAC:    expected_key_len = 0; break;
            case smoke::engine::KeyslotType::KEYSLOT_RSA2048: expected_key_len = 0; break;
            default: expected_key_len = 0; break;
        }
        if (expected_key_len > 0) {
            sp.check_key_len_valid = (slot.key_len == expected_key_len) ? 0 : 1;
        } else {
            sp.check_key_len_valid = 0;
        }

        // --- Integrity tag (all key types) ---
        // Uses probe_compute_integrity_tag() which mirrors engine_host.cpp:compute_integrity_tag_host()
        {
            uint8_t recomputed_tag[32];
            probe_compute_integrity_tag(recomputed_tag, i, slot.integrity_version, slot.private_d);
            sp.check_integrity_tag = (memcmp(recomputed_tag, slot.integrity_tag, 32) == 0) ? 0 : 1;
        }

        // --- P-256 specific checks ---
        if (slot.key_type == static_cast<uint32_t>(smoke::engine::KeyslotType::KEYSLOT_P256)) {
            bool d_is_zero = probe_is_zero_32(slot.private_d);
            int cmp_n = probe_bigendian_cmp32(slot.private_d, PROBE_P256_N);
            sp.check_scalar_range = (!d_is_zero && cmp_n < 0) ? 0 : 1;

            bool top4_zero = (slot.private_d[0] == 0 && slot.private_d[1] == 0 &&
                              slot.private_d[2] == 0 && slot.private_d[3] == 0);
            sp.check_packing_sanity = top4_zero ? 1 : 0;
        } else {
            sp.check_scalar_range = 0;
            sp.check_packing_sanity = 0;
        }

        // Healthy = correctness invariants only
        sp.healthy = (sp.check_scalar_range == 0 &&
                      sp.check_mirror_match == 0 &&
                      sp.check_integrity_tag == 0 &&
                      sp.check_no_fault_flags == 0 &&
                      sp.check_version_monotonic == 0 &&
                      sp.check_key_len_valid == 0) ? 1 : 0;
    }

    // Zeroize local copies of key material
    memset(slots, 0, sizeof(slots));

    // Cross-slot checks
    out->check_epoch_in_range = (active_epoch < 2) ? 0 : 1;
    out->check_active_slot_loaded =
        (out->slots[active_epoch].key_type != static_cast<uint32_t>(smoke::engine::KeyslotType::KEYSLOT_EMPTY)) ? 0 : 1;

    const uint32_t EMPTY = static_cast<uint32_t>(smoke::engine::KeyslotType::KEYSLOT_EMPTY);
    bool both_loaded = (out->slots[0].key_type != EMPTY && out->slots[1].key_type != EMPTY);
    if (both_loaded) {
        out->check_epochs_distinct = (out->slots[0].epoch != out->slots[1].epoch) ? 0 : 1;
    } else {
        out->check_epochs_distinct = 0;
    }

    out->overall_healthy = (out->slots[0].healthy &&
                            out->slots[1].healthy &&
                            out->check_epoch_in_range == 0 &&
                            out->check_active_slot_loaded == 0 &&
                            out->check_epochs_distinct == 0) ? 1 : 0;

    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_rotate_key(
    SmokeEngineHandle engine,
    const uint8_t* private_key,
    uint32_t key_len
) {
    if (!engine || !private_key || key_len != 32) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!smoke::engine::engine_rotate_key(engine->persistent_engine, private_key)) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_prepare_next_key(
    SmokeEngineHandle engine,
    const uint8_t* private_key,
    uint32_t key_len
) {
    if (!engine || !private_key || key_len != 32) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!smoke::engine::engine_prepare_next_key(engine->persistent_engine, private_key)) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_commit_rotation(SmokeEngineHandle engine) {
    if (!engine) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!smoke::engine::engine_commit_rotation(engine->persistent_engine)) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_rollback_rotation(SmokeEngineHandle engine) {
    if (!engine) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!smoke::engine::engine_rollback_rotation(engine->persistent_engine)) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    return SMOKE_STATUS_OK;
}

// ============================================================================
// Card 78: Typed Rotation (C ABI) — prepare next key with arbitrary type
// ============================================================================

extern "C"
SmokeStatus smoke_engine_prepare_next_key_typed(
    SmokeEngineHandle engine,
    uint32_t key_type,
    const uint8_t* key_data,
    uint32_t key_len
) {
    if (!engine || !key_data || key_len == 0 || key_len > UINT16_MAX) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!smoke::engine::engine_prepare_next_key_typed(
            engine->persistent_engine,
            static_cast<smoke::engine::KeyslotType>(key_type),
            key_data,
            static_cast<uint16_t>(key_len))) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_prepare_next_key_typed_async(
    SmokeEngineHandle engine,
    uint32_t key_type,
    const uint8_t* key_data,
    uint32_t key_len
) {
    if (!engine || !key_data || key_len == 0 || key_len > UINT16_MAX) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    // Match the untyped async ABI: generic submit and signing check this
    // outer latch, not the inner EngineHandle flag. Set it before enqueuing
    // so a partial CUDA failure also stays closed until an explicit fence.
    engine->rotation_pending_fence = true;
    if (!smoke::engine::engine_prepare_next_key_typed_async(
            engine->persistent_engine,
            static_cast<smoke::engine::KeyslotType>(key_type),
            key_data,
            static_cast<uint16_t>(key_len))) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    return SMOKE_STATUS_OK;
}

// ============================================================================
// Card 32B: Async Rotation (C ABI)
// ============================================================================

extern "C"
SmokeStatus smoke_engine_prepare_next_key_async(
    SmokeEngineHandle engine,
    const uint8_t* private_key,
    uint32_t key_len
) {
    if (!engine || !private_key || key_len != 32) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!smoke::engine::engine_prepare_next_key_async(engine->persistent_engine, private_key)) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    // Card 32C: Set fence latch — signing blocked until fence
    engine->rotation_pending_fence = true;
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_commit_rotation_async(SmokeEngineHandle engine) {
    if (!engine) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!smoke::engine::engine_commit_rotation_async(engine->persistent_engine)) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    // Card 32C: Set fence latch — signing blocked until fence
    engine->rotation_pending_fence = true;
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_rotation_fence(SmokeEngineHandle engine) {
    if (!engine) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!smoke::engine::engine_rotation_fence(engine->persistent_engine)) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    // Card 32C: Clear fence latch — signing is safe again
    engine->rotation_pending_fence = false;
    return SMOKE_STATUS_OK;
}

// Card 64: Force recovery from stuck fence state
SmokeStatus smoke_engine_rotation_force_recovery(SmokeEngineHandle engine) {
    if (!engine) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->persistent_engine) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    if (!smoke::engine::engine_rotation_force_recovery(engine->persistent_engine)) {
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    engine->rotation_pending_fence = false;
    return SMOKE_STATUS_OK;
}

// ============================================================================
// Card 26.82: FAST PATH Lifecycle (C ABI)
// ============================================================================

extern "C"
SmokeStatus smoke_fast_path_create(
    SmokeEngineHandle engine,
    SmokeErrorInfo*   out_error
) {
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "engine must not be NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    smoke_bind_device(engine); // D-080: ring/slab allocation on the engine's device

    if (!engine->persistent_engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Persistent engine not initialized");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    // If FAST PATH already exists, return OK
    if (engine->fast_path) {
        return SMOKE_STATUS_OK;
    }

    // Shut down legacy kernel to free GPU for FAST PATH
    if (engine->engine_running) {
        smoke::engine::engine_shutdown(engine->persistent_engine);
        engine->engine_running = false;
    }

    // Create FAST PATH handle
    engine->fast_path = smoke::engine::fast_path_create(engine->persistent_engine);
    if (!engine->fast_path) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Failed to create FAST PATH handle");
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_fast_path_start(
    SmokeEngineHandle engine,
    SmokeErrorInfo*   out_error
) {
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "engine must not be NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    smoke_bind_device(engine); // D-080: kernel launch on the engine's device

    if (!engine->fast_path) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "FAST PATH not created. Call smoke_fast_path_create() first.");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    // N5 fail-closed (2026-05-16): refuse to launch the warp-coop signer
    // kernel if a P-256 keyslot is loaded but the generator-point comb
    // table has not been uploaded. Silent symptom otherwise: every sign
    // returns CRYPTO_ERROR with r=0 because scalar_mul_g produces ∞ from
    // an all-zero __constant__ comb table. See smoke_engine_load_p256_comb_table.
    if (engine->has_p256_key && !engine->p256_comb_loaded) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "P-256 keyslot loaded but comb table missing. "
                     "Call smoke_engine_load_p256_comb_table() before smoke_fast_path_start().");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

#if P256_FASTPATH_FULLWINDOW
    // SPEED-H2-01 condition C4: this build compiles the fullwindow scalar-mul
    // path, which reads P256_G_FULLWINDOW. Refuse to launch if a P-256 key is
    // loaded but that table has not been uploaded — same silent-∞ failure
    // mode as the comb gate above, no silent fallback to the comb table.
    if (engine->has_p256_key && !engine->p256_fullwindow_loaded) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "P-256 keyslot loaded but full-window table missing. "
                     "Call smoke_engine_load_p256_fullwindow_table() before smoke_fast_path_start().");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
#endif

    // If already running, return OK
    if (engine->fast_path_running) {
        return SMOKE_STATUS_OK;
    }

    // Start the kernel
    if (!smoke::engine::fast_path_start(engine->fast_path)) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Failed to start FAST PATH kernel");
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    engine->fast_path_running = true;
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_fast_path_stop(
    SmokeEngineHandle engine
) {
    if (!engine) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    smoke_bind_device(engine); // D-080: kernel teardown on the engine's device

    if (engine->fast_path && engine->fast_path_running) {
        smoke::engine::fast_path_stop(engine->fast_path);
        engine->fast_path_running = false;
    }

    return SMOKE_STATUS_OK;
}

extern "C"
int smoke_fast_path_is_running(
    SmokeEngineHandle engine
) {
    if (!engine) return 0;
    return engine->fast_path_running ? 1 : 0;
}

// ============================================================================
// B-306 (2026-05-10): FAST PATH Configuration Controls (C ABI)
// ============================================================================
// Thin extern "C" wrappers around the smoke::engine::fast_path_set_num_*
// namespace functions. Without these, native callers (Go cgo / Rust FFI)
// can't bring up FAST PATH at non-default shard counts -- the engine
// rejects 1 shard ("Legacy single-ring kernel disabled"). The
// pybind11 layer already had these as lambdas (search the file for
// m.def("fast_path_set_num_ctas", ...)); these wrappers expose the
// SAME functions through C linkage with no new behavior.

extern "C" SMOKE_API
SmokeStatus smoke_fast_path_set_num_ctas(
    SmokeEngineHandle engine,
    uint32_t          num_ctas
) {
    if (!engine) return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (!engine->fast_path) return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    if (engine->fast_path_running) {
        // Can't reconfigure a running kernel; the namespace function
        // would silently fail anyway. Make it explicit at the ABI.
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    bool ok = smoke::engine::fast_path_set_num_ctas(engine->fast_path, num_ctas);
    return ok ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_INVALID_ARG;
}

extern "C" SMOKE_API
SmokeStatus smoke_fast_path_set_num_shards(
    SmokeEngineHandle engine,
    uint32_t          num_shards
) {
    if (!engine) return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (!engine->fast_path) return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    if (engine->fast_path_running) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    bool ok = smoke::engine::fast_path_set_num_shards(engine->fast_path, num_shards);
    return ok ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_INVALID_ARG;
}

// ============================================================================
// B-306 (2026-05-10): AES-256 Key Load (C ABI)
// ============================================================================
// Counterpart to smoke_engine_load_key_p256. Wraps the same
// smoke::engine::engine_set_key_typed(... KEYSLOT_AES256 ...) path that
// py_load_key_aes256 already uses, just without the pybind11 indirection.

// ============================================================================
// B-307 (2026-05-10): Coalesced inline-input batched submit
// ============================================================================
// Thin extern "C" wrapper around the existing
// smoke::engine::fast_path_submit_batch. Callers (Go IPC bridge,
// other native drivers) avoid the per-request loop in
// smoke_generic_submit -- that loop calls fast_path_submit_batch
// with count=1 per request, paying N memory fences per shard.
// Batched submit coalesces to ~1 fence per shard regardless of N.

extern "C" SMOKE_API
uint32_t smoke_fast_path_submit_inline_batch(
    SmokeEngineHandle engine,
    const uint8_t*    inputs,
    uint32_t          count,
    uint8_t           flags,
    uint8_t           client_id,
    uint16_t          opcode,
    uint16_t          input_len
) {
    if (!engine || !engine->fast_path || !inputs || count == 0) {
        return 0;
    }
    if (!engine->fast_path_running) {
        return 0;
    }
    return smoke::engine::fast_path_submit_batch(
        engine->fast_path, inputs, count, flags, client_id, opcode, input_len
    );
}

// ============================================================================
// D5b-keyslot: keyed coalesced inline-batch submit
// ============================================================================
// smoke_fast_path_submit_inline_batch always submitted with ring key_slot=0
// (fast_path_submit_batch hardcoded it), so the coalesced path could not
// address per-tenant keyslots. This variant carries the batch's shared
// key_slot. The legacy symbol is kept byte-compatible (slot 0); callers that
// need a non-zero slot MUST use this one — and its presence is the loader's
// capability probe that this build forwards key_slot on inline submits at
// all (smoke_generic_submit's inline branch was fixed in the same change).
//
// Same return contract as the sibling: number of requests submitted. An
// out-of-range key_slot submits 0 — fast_path_submit_batch refuses the whole
// batch loudly rather than run under a slot the caller did not ask for.

extern "C" SMOKE_API
uint32_t smoke_fast_path_submit_inline_batch_slot(
    SmokeEngineHandle engine,
    const uint8_t*    inputs,
    uint32_t          count,
    uint8_t           flags,
    uint8_t           client_id,
    uint16_t          opcode,
    uint16_t          input_len,
    uint8_t           key_slot
) {
    if (!engine || !engine->fast_path || !inputs || count == 0) {
        return 0;
    }
    if (!engine->fast_path_running) {
        return 0;
    }
    return smoke::engine::fast_path_submit_batch(
        engine->fast_path, inputs, count, flags, client_id, opcode, input_len,
        key_slot
    );
}

extern "C" SMOKE_API
SmokeStatus smoke_fast_path_set_submit_policy(
    SmokeEngineHandle engine,
    uint32_t          policy
) {
    if (!engine) return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (!engine->fast_path) return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    if (engine->fast_path_running) return SMOKE_STATUS_ERROR_INVALID_ARG;
    bool ok = smoke::engine::fast_path_set_submit_policy(engine->fast_path, policy);
    return ok ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_INVALID_ARG;
}

extern "C" SMOKE_API
SmokeStatus smoke_fast_path_set_shards_per_batch(
    SmokeEngineHandle engine,
    uint32_t          k
) {
    if (!engine) return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (!engine->fast_path) return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    if (engine->fast_path_running) return SMOKE_STATUS_ERROR_INVALID_ARG;
    bool ok = smoke::engine::fast_path_set_shards_per_batch(engine->fast_path, k);
    return ok ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_INVALID_ARG;
}

extern "C" SMOKE_API
SmokeStatus smoke_engine_load_key_aes256(
    SmokeEngineHandle engine,
    uint32_t          slot,
    const uint8_t     key[32],
    SmokeErrorInfo*   out_error
) {
    if (!engine || !key) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "engine and key must not be NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    smoke_bind_device(engine); // D-080: H2D key copy on the engine's device
    if (!engine->persistent_engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "persistent engine not initialized");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    bool ok = smoke::engine::engine_set_key_typed(
        engine->persistent_engine,
        slot,
        smoke::engine::KeyslotType::KEYSLOT_AES256,
        key,
        32
    );
    if (!ok) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "engine_set_key_typed(AES256, slot=%u) failed", slot);
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    return SMOKE_STATUS_OK;
}

// ============================================================================
// ATTEST-NATIVE-01: C ABI for HMAC key load + the native varlen hash feeder.
// The stable, language-neutral surface (Go/Rust/etc.) over the same C++ core
// the pybind uses. Thin wrappers: resolve the handle, validate the contract,
// call the core, branch on status. No logic duplicated here.
// ============================================================================
extern "C" SMOKE_API
SmokeStatus smoke_engine_load_key_hmac(
    SmokeEngineHandle engine,
    uint32_t          slot,
    const uint8_t*    key,
    uint32_t          key_len
) {
    if (!engine || !key) return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (key_len == 0 || key_len > 64) return SMOKE_STATUS_ERROR_INVALID_ARG;
    smoke_bind_device(engine); // D-080: H2D key copy on the engine's device
    if (!engine->persistent_engine) return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    bool ok = smoke::engine::engine_set_key_typed(
        engine->persistent_engine, slot,
        smoke::engine::KeyslotType::KEYSLOT_HMAC, key,
        static_cast<uint16_t>(key_len));
    return ok ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_DEVICE_FAILURE;
}

extern "C" SMOKE_API
SmokeStatus smoke_engine_submit_and_drain(
    SmokeEngineHandle       engine,
    const SmokeSubmitBatch* submit,
    SmokeDrainBatch*        drain
) {
    if (!engine || !submit || !drain) return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (!submit->payloads || !submit->offsets || !drain->digests)
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (submit->item_count == 0) return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (drain->capacity < submit->item_count) return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (submit->offsets[0] != 0 ||
        submit->offsets[submit->item_count] != submit->payload_bytes)
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    smoke_bind_device(engine);
    if (!engine->fast_path || !engine->fast_path_running)
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;

    drain->completed_count = 0;
    drain->accounting_ok = 0;

    smoke::engine::HashDrainResult res =
        smoke::engine::fast_path_submit_and_drain_hashes(
            engine->fast_path,
            submit->payloads, submit->offsets, submit->item_count,
            submit->opcode, static_cast<uint8_t>(submit->key_slot),
            drain->digests);

    drain->completed_count = res.completed_count;
    drain->accounting_ok = res.accounting_ok;
    // Fail-closed: the core already zeroed any failed/undrained digests.
    return (res.accounting_ok == 1u) ? SMOKE_STATUS_OK
                                     : SMOKE_STATUS_ERROR_DEVICE_FAILURE;
}

// ============================================================================
// Card 289: Per-Engine FAST PATH Handle Registration (C ABI)
// ============================================================================

extern "C"
SmokeStatus smoke_engine_set_fast_path_handle(
    SmokeEngineHandle engine,
    void*             fp_handle
) {
    if (!engine) return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    engine->fast_path = static_cast<smoke::engine::FastPathHandle*>(fp_handle);
    engine->fast_path_running = (fp_handle != nullptr);
    return SMOKE_STATUS_OK;
}

extern "C"
SmokeStatus smoke_engine_clear_fast_path_handle(
    SmokeEngineHandle engine
) {
    if (!engine) return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    engine->fast_path = nullptr;
    engine->fast_path_running = false;
    return SMOKE_STATUS_OK;
}

// ============================================================================
// Card 26.82: Generic Submit/Poll (C ABI)
// ============================================================================

extern "C"
SmokeStatus smoke_generic_submit(
    SmokeEngineHandle             engine,
    const SmokeGenericRequest*    reqs,
    uint32_t                      count,
    SmokeErrorInfo*               out_error
) {
    if (!engine || !reqs || count == 0) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 0;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Invalid arguments: engine=%p, reqs=%p, count=%u",
                     (void*)engine, (void*)reqs, count);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!engine->fast_path || !engine->fast_path_running) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "FAST PATH not running. Call smoke_fast_path_start() first.");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    // Card 32C: Refuse to submit while async rotation is pending fence
    if (engine->rotation_pending_fence) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_FENCE_REQUIRED;
            out_error->internal_code = 10;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Async rotation pending. Call rotation_fence() before submitting.");
        }
        return SMOKE_STATUS_ERROR_FENCE_REQUIRED;
    }

    // Validate ABI version and submit each request
    uint32_t submitted = 0;
    for (uint32_t i = 0; i < count; i++) {
        const SmokeGenericRequest* req = &reqs[i];

        // Validate ABI version
        if (req->abi_version != SMOKE_GENERIC_ABI_VERSION) {
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
                out_error->internal_code = static_cast<int32_t>(submitted);
                snprintf(out_error->message, sizeof(out_error->message),
                         "ABI version mismatch at request %u: got %u, expected %u",
                         i, req->abi_version, SMOKE_GENERIC_ABI_VERSION);
            }
            return submitted > 0 ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_UNSUPPORTED;
        }

        // RRP-A06b fail-closed: RSA_2048_SIGN has no handler in the
        // production sharded kernel (the real one is ENABLE_LEGACY_KERNEL
        // dead code). The kernel's old stub for it violated the response
        // commit protocol (double slot reservation), so a ring submission
        // never produced a drainable response and the caller hung. Refuse at
        // the truth boundary instead of forwarding an op that cannot
        // complete; the kernel's unknown-opcode INVALID_OPCODE path is the
        // backstop for callers that bypass this ABI.
        if (req->opcode ==
            static_cast<uint16_t>(smoke::engine::OpCode::RSA_2048_SIGN)) {
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
                out_error->internal_code = static_cast<int32_t>(submitted);
                snprintf(out_error->message, sizeof(out_error->message),
                         "Request %u: RSA_2048_SIGN is not available in the "
                         "production sharded kernel (legacy-kernel-only "
                         "handler); ring submission refused fail-closed",
                         i);
            }
            return submitted > 0 ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_UNSUPPORTED;
        }

        // D-089 fail-closed: an inline request (payload_len == 0) physically
        // carries at most 32 input bytes — the memcpy below caps there. A
        // larger input_len used to be forwarded anyway, so the kernel operated
        // on a silent 32-byte prefix and reported success (§4 violation:
        // partial output). Reject at the truth boundary instead. Slab callers
        // (payload_len > 0, e.g. RSA/ML-DSA) are unaffected.
        if (req->payload_len == 0 && req->input_len > 32) {
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
                out_error->internal_code = static_cast<int32_t>(submitted);
                snprintf(out_error->message, sizeof(out_error->message),
                         "Inline request %u: input_len %u exceeds the 32-byte "
                         "inline slot (opcode %u); use the payload slab or "
                         "split the input",
                         i, req->input_len, req->opcode);
            }
            return submitted > 0 ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_INVALID_ARG;
        }

        // Prepare input buffer (pad to 32 bytes)
        uint8_t input_buf[32] = {0};
        uint16_t copy_len = req->input_len < 32 ? req->input_len : 32;
        memcpy(input_buf, req->input, copy_len);

        // B-306 (2026-05-10): route through fast_path_submit_batch so
        // requests are distributed round-robin across the configured
        // shards. fast_path_submit_generic only writes to shard 0,
        // which limits the engine to 1/N of its capacity when N shards
        // are active (e.g. 1/60 at the production 60/60 config). The
        // pybind11 generic_submit (line 4254) already uses _batch with
        // count=1; this brings the C ABI to the same shape.
        //
        // Payload-slab mode (payload_len > 0) still requires per-call
        // routing through _submit_generic since _submit_batch only
        // handles inline inputs. Slab-mode callers should be batching
        // through the slab API directly anyway -- the inline path is
        // the hot one for B-306 throughput.
        // D-093: the transport slot (FastPathRequest.client_id, frozen
        // 64-byte ring layout, Card 26.16) is uint8_t — only the low 8
        // bits of the caller's 64-bit client_id reach the kernel, and
        // the kernel echoes that 8-bit value back. Bits 8..63 never
        // round-trip. This is the documented contract (see
        // SmokeGenericRequest.client_id in smoke_generic_abi.h), not a
        // defect to widen away: the request slot has no room without
        // breaking the frozen ring layout, the response layout is
        // load-bearing (B-319/B-319b commit protocol), and responses
        // must be routed by request_id regardless because the client_id
        // echo can mis-tag under load (B-318..B-320, B-321). Callers
        // that assumed a full 64-bit round-trip have been burned before
        // (suite governed coalescer, suite commit 5a07869) — say it
        // once per process, loudly.
        if (req->client_id > 0xFF) {
            static std::atomic<int> cid_notice_once{0};
            if (cid_notice_once.fetch_add(1, std::memory_order_relaxed) == 0) {
                std::fprintf(stderr,
                             "[SMOKE][ABI-CONTRACT] generic_submit: client_id 0x%llx "
                             "has bits set above the low 8. The engine transports "
                             "client_id & 0xFF and echoes that 8-bit value in "
                             "responses; bits 8..63 do not round-trip (D-093). "
                             "Route responses by request_id (B-321), not client_id. "
                             "(printed once per process)\n",
                             static_cast<unsigned long long>(req->client_id));
                std::fflush(stderr);
            }
        }
        uint8_t client_id = static_cast<uint8_t>(req->client_id & 0xFF);
        // D5b-keyslot fail-closed: the kernel indexes keyslots[req.key_slot]
        // with no device-side bounds check — an out-of-range slot would read
        // adjacent EngineState memory as key material and return plausible
        // wrong output (§4). Reject at the truth boundary, like D-089.
        // Inline branch only: slab opcodes (RSA/ML-DSA) index their own
        // per-primitive keyslot arrays with different bounds, validated in
        // their dedicated submit paths.
        if (req->payload_len == 0 &&
            req->key_slot >= smoke::engine::fast_path_max_key_slots()) {
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
                out_error->internal_code = static_cast<int32_t>(submitted);
                snprintf(out_error->message, sizeof(out_error->message),
                         "Inline request %u: key_slot %u >= engine keyslot "
                         "count %u (D5b-keyslot fail-closed)",
                         i, req->key_slot, smoke::engine::fast_path_max_key_slots());
            }
            return submitted > 0 ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_INVALID_ARG;
        }
        uint32_t ok;
        if (req->payload_len == 0) {
            // D5b-keyslot: forward req->key_slot. This branch used to call
            // fast_path_submit_batch WITHOUT it, so every inline request
            // (AEAD seal/open, keyed hash) ran under slot 0 no matter what
            // the caller set — the engine-side break behind the D5b
            // multi-tenant pause. GPU isolation validation tracked on the
            // D5b card; responses echo slot_used (Card 66) for the probe.
            ok = smoke::engine::fast_path_submit_batch(
                engine->fast_path,
                input_buf,
                1,
                req->flags,
                client_id,
                req->opcode,
                req->input_len,
                req->key_slot
            );
        } else {
            ok = smoke::engine::fast_path_submit_generic(
                engine->fast_path,
                input_buf,
                req->input_len,
                req->flags,
                req->key_slot,
                client_id,
                req->opcode,
                req->payload_offset,
                req->payload_len
            );
        }

        if (ok > 0) {
            submitted++;
        }
    }

    // Report number submitted in internal_code
    if (out_error) {
        out_error->internal_code = static_cast<int32_t>(submitted);
        if (submitted < count) {
            out_error->code = SMOKE_STATUS_OK;  // Partial success
            snprintf(out_error->message, sizeof(out_error->message),
                     "Submitted %u of %u requests (ring may be full)", submitted, count);
        }
    }

    return SMOKE_STATUS_OK;
}

extern "C"
uint32_t smoke_generic_poll(
    SmokeEngineHandle       engine,
    SmokeGenericResponse*   out,
    uint32_t                max_out
) {
    if (!engine || !out || max_out == 0) {
        return 0;
    }

    if (!engine->fast_path || !engine->fast_path_running) {
        return 0;
    }

    // Poll from FAST PATH
    std::vector<smoke::engine::FastPathResponse> responses(max_out);
    uint32_t got = smoke::engine::fast_path_poll(engine->fast_path, responses.data(), max_out);

    // Convert to generic response format
    for (uint32_t i = 0; i < got; i++) {
        const smoke::engine::FastPathResponse& src = responses[i];
        SmokeGenericResponse* dst = &out[i];

        dst->client_id = src.client_id;
        dst->seq = src.seq;
        dst->status = src.status;
        dst->backend = SMOKE_BACKEND_GPU_FASTPATH;  // Always GPU for FAST PATH
        dst->opcode = src.opcode;
        dst->output_len = src.output_len;
        dst->reserved_u16 = 0;
        dst->output_offset = src.output_offset;
        dst->output_bytes = src.output_bytes;
        dst->t_submit_us = src.t_submit_us;
        dst->t_dequeue_us = src.t_dequeue_us;
        dst->t_complete_us = src.t_complete_us;
        dst->request_id = src.request_id;  // B-321: route by this, not client_id

        // Copy output (inline portion, up to 64 bytes)
        uint16_t copy_len = src.output_len < 64 ? src.output_len : 64;
        memcpy(dst->output, src.output, copy_len);
        if (copy_len < 64) {
            memset(dst->output + copy_len, 0, 64 - copy_len);
        }
    }

    return got;
}

// ============================================================================
// B-308 (2026-05-11): Blocking poll variant
// ============================================================================
// Wait inside the engine layer (one syscall from the host's POV)
// until at least one response is available or the timeout expires.
//
// Rationale: B-307 profile showed ~330 syscalls per batch at
// pack=1500 with ~328 of them empty -- each costing ~7us on Windows.
// That overhead (2.3ms / batch) is now the dominant cost above the
// kernel's actual processing time. Moving the spin INTO C cuts the
// host-side syscall count from ~330 to ~1-5 per batch.
//
// Internal loop: tight `_mm_pause` between fast_path_poll checks,
// deadline checked every kSpinsBeforeClockCheck iterations to
// amortize the clock-read cost. No sleeps -- caller picks an
// appropriate timeout_us.

#if defined(_WIN32) && (defined(_M_X64) || defined(_M_IX86))
#include <emmintrin.h>  // _mm_pause
#define SMOKE_CPU_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#define SMOKE_CPU_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define SMOKE_CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#define SMOKE_CPU_PAUSE() ((void)0)
#endif

extern "C" SMOKE_API
uint32_t smoke_generic_poll_wait(
    SmokeEngineHandle       engine,
    SmokeGenericResponse*   out,
    uint32_t                max_out,
    uint32_t                timeout_us
) {
    if (!engine || !out || max_out == 0) {
        return 0;
    }
    if (!engine->fast_path || !engine->fast_path_running) {
        return 0;
    }

    std::vector<smoke::engine::FastPathResponse> responses(max_out);

    // Fast path: one nonblocking drain. If responses are already
    // ready, return immediately without ever reading the clock.
    uint32_t got = smoke::engine::fast_path_poll(
        engine->fast_path, responses.data(), max_out);

    if (got == 0 && timeout_us > 0) {
        const auto start = std::chrono::steady_clock::now();
        const auto deadline = start + std::chrono::microseconds(timeout_us);

        // kSpinsBeforeClockCheck: tight spin count before reading
        // the clock to check the deadline. Each fast_path_poll +
        // _mm_pause is ~100 ns on a fast kernel — but under heavy
        // kernel work (warp-coop P-256: phase1/4a/3 sequencing) a
        // single fast_path_poll can cost ~1-6 us due to cache-line
        // snoop on the response ring written by the device. With the
        // original 64-spin block, that pushed the per-deadline-check
        // window from the intended ~6 us up to multiple hundred ms,
        // inflating GenericPollWait avg from ~5 ms (configured
        // timeout) to ~386 ms (measured P-256 inproc, Card N6).
        // 8 spins keeps the clock-amortization intent (8 fast_path_poll
        // calls per std::chrono::steady_clock::now()) while bounding
        // the worst-case deadline overrun to ~1/8 of the original.
        constexpr uint32_t kSpinsBeforeClockCheck = 8;

        while (true) {
            for (uint32_t i = 0; i < kSpinsBeforeClockCheck; i++) {
                got = smoke::engine::fast_path_poll(
                    engine->fast_path, responses.data(), max_out);
                if (got != 0) goto have_responses;
                SMOKE_CPU_PAUSE();
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }
        }
    }

have_responses:
    // Convert FastPathResponse -> SmokeGenericResponse. Mirrors
    // smoke_generic_poll byte-for-byte.
    for (uint32_t i = 0; i < got; i++) {
        const smoke::engine::FastPathResponse& src = responses[i];
        SmokeGenericResponse* dst = &out[i];

        dst->client_id = src.client_id;
        dst->seq = src.seq;
        dst->status = src.status;
        dst->backend = SMOKE_BACKEND_GPU_FASTPATH;
        dst->opcode = src.opcode;
        dst->output_len = src.output_len;
        dst->reserved_u16 = 0;
        dst->output_offset = src.output_offset;
        dst->output_bytes = src.output_bytes;
        dst->t_submit_us = src.t_submit_us;
        dst->t_dequeue_us = src.t_dequeue_us;
        dst->t_complete_us = src.t_complete_us;
        dst->request_id = src.request_id;  // B-321: route by this, not client_id

        uint16_t copy_len = src.output_len < 64 ? src.output_len : 64;
        memcpy(dst->output, src.output, copy_len);
        if (copy_len < 64) {
            memset(dst->output + copy_len, 0, 64 - copy_len);
        }
    }

    return got;
}

// ============================================================================
// B-312 (2026-05-11): Read-only debug probes (C ABI)
// ============================================================================
// Read-only accessors for fields the host already maps but doesn't
// expose through the C ABI. Used by B-312 cross-thread visibility
// audit; safe to leave shipped (engine-side state, no side effects).

extern "C" SMOKE_API
uint64_t smoke_debug_responses_written(SmokeEngineHandle engine) {
    if (!engine || !engine->fast_path) return 0;
    smoke::engine::FastPathTelemetry telem{};
    smoke::engine::fast_path_get_telemetry(engine->fast_path, &telem);
    return telem.debug_responses_written;
}

// N5-A (2026-05-14): P-256 sharded warp-coop failure-phase split.
// phase1_fail counts ops where state.valid==false (RFC6979 k-gen).
// phase3_fail counts ops where state.valid==true but phase3 returned
// false (ecdsa_sign_from_r_warp failure). Per-op increment by lane 0
// only. Both read via fast_path_get_telemetry.
extern "C" SMOKE_API
uint64_t smoke_debug_p256_phase1_fail(SmokeEngineHandle engine) {
    if (!engine || !engine->fast_path) return 0;
    smoke::engine::FastPathTelemetry telem{};
    smoke::engine::fast_path_get_telemetry(engine->fast_path, &telem);
    return telem.dbg_p256_phase1_fail;
}

extern "C" SMOKE_API
uint64_t smoke_debug_p256_phase3_fail(SmokeEngineHandle engine) {
    if (!engine || !engine->fast_path) return 0;
    smoke::engine::FastPathTelemetry telem{};
    smoke::engine::fast_path_get_telemetry(engine->fast_path, &telem);
    return telem.dbg_p256_phase3_fail;
}

// N5 r_zero (2026-05-15): subset of dbg_p256_phase3_fail where failure
// originated in `is_zero_scalar(r)` after reduce_x_mod_n inside
// ecdsa_sign_from_r_warp. Splits phase3_fail into r==0 (upstream
// affine/Zinv path) vs s==0 (scalar-order Montgomery chain).
extern "C" SMOKE_API
uint64_t smoke_debug_p256_r_zero(SmokeEngineHandle engine) {
    if (!engine || !engine->fast_path) return 0;
    smoke::engine::FastPathTelemetry telem{};
    smoke::engine::fast_path_get_telemetry(engine->fast_path, &telem);
    return telem.dbg_p256_r_zero;
}

// N5 x_full_zero (2026-05-16): subset of dbg_p256_r_zero where the
// gathered affine x-coordinate (x_full) is literally all-zero BEFORE
// the mod-n reduction. Splits r_zero into: x_full == 0 (upstream
// jacobian_to_affine_with_zinv_warp / batch-inv produced zero affine)
// vs x_full != 0 but reduced r == 0 (physically impossible without a
// gather/reduction bug — r==0 requires x_full == n which is outside
// field range).
extern "C" SMOKE_API
uint64_t smoke_debug_p256_x_full_zero(SmokeEngineHandle engine) {
    if (!engine || !engine->fast_path) return 0;
    smoke::engine::FastPathTelemetry telem{};
    smoke::engine::fast_path_get_telemetry(engine->fast_path, &telem);
    return telem.dbg_p256_x_full_zero;
}

// N5 zinv_zero (2026-05-16): count of ops whose reconstructed 8-limb
// Zinv (post phase4a `p256_batch_inv_block_montgomery_v2`, pre phase3
// affine conversion) is literally all-zero. Splits x_full_zero into:
// zinv == 0 (phase 4a batch inversion is broken — produces zero) vs
// zinv != 0 but downstream x_full == 0 (jacobian_to_affine_with_zinv_warp
// is broken — emits zero affine x despite valid Zinv).
extern "C" SMOKE_API
uint64_t smoke_debug_p256_zinv_zero(SmokeEngineHandle engine) {
    if (!engine || !engine->fast_path) return 0;
    smoke::engine::FastPathTelemetry telem{};
    smoke::engine::fast_path_get_telemetry(engine->fast_path, &telem);
    return telem.dbg_p256_zinv_zero;
}

// B-321 (2026-05-13): Read-only getter for `handle->next_request_id`. The
// engine assigns this monotonic counter to every request submitted via
// `fast_path_submit_batch` / `fast_path_submit_batch_shard_local` (one
// increment per request). The scheduler captures the value BEFORE a
// submit; together with the count actually submitted, this yields the
// request_id RANGE [rid_base, rid_base + submitted) owned by that
// submit. That range is then used to route responses back to the
// originating chunk via request_id instead of client_id (which has been
// observed to be mis-tagged under residual kernel race conditions;
// see D-052..D-054).
//
// Single-caller assumption: this is meant to be called from the
// scheduler goroutine immediately before its own SubmitInlineBatch
// call. Concurrent submits from other callers would invalidate the
// captured base. Smoke's engine handle is SPSC for submit anyway.
extern "C" SMOKE_API
uint64_t smoke_debug_next_request_id(SmokeEngineHandle engine) {
    if (!engine || !engine->fast_path) return 0;
    return smoke::engine::fast_path_get_next_request_id(engine->fast_path);
}

// B-317 (2026-05-12): Per-shard ring + response-buffer diagnostic snapshot.
// Fills out_array[0..min(num_shards, max_shards)] with each active shard's
// request-ring head/tail and response-buffer write_idx/read_idx, plus the
// pre-computed pending counts. Returns the number of entries written.
//
// `req_pending = req_tail - req_head` follows the convention established
// by `fast_path_get_ring_status_all` (Card 26.34 fix). `resp_pending =
// resp_write_idx - resp_read_idx` follows the standard producer-consumer
// convention for FastPathResponseBuffer (kernel produces, host consumes).
extern "C" SMOKE_API
uint32_t smoke_debug_shard_diag(SmokeEngineHandle engine,
                                SmokeShardDiag*   out_array,
                                uint32_t          max_shards) {
    if (!engine || !engine->fast_path || !out_array || max_shards == 0) {
        return 0;
    }

    // Stack-allocated parallel arrays sized to FAST_PATH_MAX_SHARDS (160
    // in engine_host.cpp since D-083 raised it from 64). This clamp was
    // missed by D-083: at 64 it silently truncated the diag view to the
    // first 64 shards, blinding per-shard occupancy consumers (RRP-03
    // adaptive drain watermark) on >64-shard datacenter shapes (A100
    // 108/108, L40S 142). 5 x 160 x 4B = 3.2KB stack — fine.
    static constexpr uint32_t MAX = 160;
    if (max_shards > MAX) max_shards = MAX;

    uint32_t shard_id[MAX] = {0};
    uint32_t req_head[MAX] = {0};
    uint32_t req_tail[MAX] = {0};
    uint32_t resp_w[MAX]   = {0};
    uint32_t resp_r[MAX]   = {0};

    uint32_t n = smoke::engine::fast_path_get_shard_diag_all(
        engine->fast_path,
        shard_id, req_head, req_tail, resp_w, resp_r,
        max_shards);

    for (uint32_t i = 0; i < n; ++i) {
        out_array[i].shard_id       = shard_id[i];
        out_array[i].req_head       = req_head[i];
        out_array[i].req_tail       = req_tail[i];
        out_array[i].req_pending    = req_tail[i] - req_head[i];
        out_array[i].resp_write_idx = resp_w[i];
        out_array[i].resp_read_idx  = resp_r[i];
        out_array[i].resp_pending   = resp_w[i] - resp_r[i];
        out_array[i].reserved       = 0;
    }
    return n;
}

// ============================================================================
// Card 26.85: Payload/Output Slab Accessors (C ABI)
// ============================================================================

extern "C"
SmokeStatus smoke_engine_get_payload_slab(
    SmokeEngineHandle engine,
    uint8_t**         out_ptr,
    uint32_t*         out_size
) {
    if (!engine || !out_ptr || !out_size) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!engine->fast_path) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    bool ok = smoke::engine::fast_path_get_payload_slab(
        engine->fast_path, out_ptr, out_size);

    return ok ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_DEVICE_FAILURE;
}

extern "C"
SmokeStatus smoke_engine_get_output_slab(
    SmokeEngineHandle engine,
    uint8_t**         out_ptr,
    uint32_t*         out_size
) {
    if (!engine || !out_ptr || !out_size) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!engine->fast_path) {
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    bool ok = smoke::engine::fast_path_get_output_slab(
        engine->fast_path, out_ptr, out_size);

    return ok ? SMOKE_STATUS_OK : SMOKE_STATUS_ERROR_DEVICE_FAILURE;
}

// ============================================================================
// Card 26.91: Zero-Copy Read Helpers (C ABI)
// ============================================================================

extern "C" SMOKE_API
SmokeStatus smoke_response_get_output(
    SmokeEngineHandle             engine,
    const SmokeGenericResponse*   response,
    const uint8_t**               out_ptr,
    uint32_t*                     out_len
) {
    if (!engine || !response || !out_ptr || !out_len) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Check if output is in slab (output_bytes > 0) or inline
    if (response->output_bytes > 0) {
        // Output is in slab
        if (!engine->fast_path) {
            return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
        }

        uint8_t* slab_ptr = nullptr;
        uint32_t slab_size = 0;
        if (!smoke::engine::fast_path_get_output_slab(
                engine->fast_path, &slab_ptr, &slab_size)) {
            return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
        }

        // Validate offset
        if (response->output_offset + response->output_bytes > slab_size) {
            return SMOKE_STATUS_ERROR_INVALID_ARG;
        }

        *out_ptr = slab_ptr + response->output_offset;
        *out_len = response->output_bytes;
    } else {
        // Output is inline
        *out_ptr = response->output;
        *out_len = response->output_len;
    }

    return SMOKE_STATUS_OK;
}

extern "C" SMOKE_API
SmokeStatus smoke_response_copy_output(
    SmokeEngineHandle             engine,
    const SmokeGenericResponse*   response,
    uint8_t*                      dst,
    uint32_t*                     dst_len
) {
    if (!engine || !response || !dst || !dst_len) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Get source pointer
    const uint8_t* src_ptr = nullptr;
    uint32_t src_len = 0;
    SmokeStatus status = smoke_response_get_output(engine, response, &src_ptr, &src_len);
    if (status != SMOKE_STATUS_OK) {
        return status;
    }

    // Check buffer size
    if (*dst_len < src_len) {
        *dst_len = src_len;  // Tell caller how much space is needed
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Copy data
    memcpy(dst, src_ptr, src_len);
    *dst_len = src_len;

    return SMOKE_STATUS_OK;
}

// ============================================================================
// Card 26.94: RSA-2048 CRT Key Loading (C ABI)
// ============================================================================

extern "C" SMOKE_API
SmokeStatus smoke_load_key_rsa2048_crt(
    SmokeEngineHandle engine,
    uint8_t           slot,
    const uint8_t*    key_blob,
    uint32_t          key_len,
    SmokeErrorInfo*   out_error
) {
    // RRP-A06b fail-closed: refuse at the EARLIEST gate. The production
    // sharded kernel cannot execute RSA_2048_SIGN — the real handler is
    // ENABLE_LEGACY_KERNEL dead code — so a loaded RSA key can never be
    // used. This function used to parse the blob and cudaMemcpy it into
    // d_rsa_keyslots, reporting success for a capability that does not
    // exist; the failure then surfaced only at submit time (which, before
    // the stub was removed from engine_loop.cu, hung the caller forever).
    // The export stays (frozen ABI surface, test_abi_exports) but the
    // status is a structured refusal. Restore the loader (git history has
    // the full body) only together with a real sharded-kernel RSA handler.
    (void)slot;
    (void)key_blob;
    (void)key_len;
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Engine handle is NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (out_error) {
        out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
        out_error->internal_code = 7;
        snprintf(out_error->message, sizeof(out_error->message),
                 "RSA-2048 CRT is not available in the production sharded "
                 "kernel (legacy-kernel-only handler); key load refused "
                 "fail-closed (RRP-A06b)");
    }
    return SMOKE_STATUS_ERROR_UNSUPPORTED;
}

// ============================================================================
// Card 27.13: ML-DSA Private Key Loading (C ABI)
// ============================================================================

extern "C" SMOKE_API
SmokeStatus smoke_load_key_mldsa_private(
    SmokeEngineHandle engine,
    uint8_t           slot,
    uint8_t           mode,
    const uint8_t*    key_blob,
    uint32_t          key_len,
    SmokeErrorInfo*   out_error
) {
    // Validate parameters
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Engine handle is NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!key_blob) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Key blob is NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Card 27.22: Accept both packed and expanded key formats
    // Packed: NIST format (2560/4032/4896 bytes)
    // Expanded: rho(32) || rhoprime(32) || tr(64) || s1 || s2 || t0 (12416/17536/23680 bytes)
    uint32_t packed_len = 0;
    uint32_t expanded_len = 0;
    const char* mode_name = nullptr;
    switch (mode) {
        case SMOKE_MLDSA_MODE_44:
            packed_len = SMOKE_MLDSA44_SK_BYTES;    // 2560
            expanded_len = 32 + 32 + 64 + 4*256*4 + 4*256*4 + 4*256*4;  // 12416
            mode_name = "ML-DSA-44";
            break;
        case SMOKE_MLDSA_MODE_65:
            packed_len = SMOKE_MLDSA65_SK_BYTES;    // 4032
            expanded_len = 32 + 32 + 64 + 5*256*4 + 6*256*4 + 6*256*4;  // 17536
            mode_name = "ML-DSA-65";
            break;
        case SMOKE_MLDSA_MODE_87:
            packed_len = SMOKE_MLDSA87_SK_BYTES;    // 4896
            expanded_len = 32 + 32 + 64 + 7*256*4 + 8*256*4 + 8*256*4;  // 23680
            mode_name = "ML-DSA-87";
            break;
        default:
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
                out_error->internal_code = 3;
                snprintf(out_error->message, sizeof(out_error->message),
                         "Invalid mode %u (must be 0=44, 1=65, or 2=87)", mode);
            }
            return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Accept either packed or expanded format
    if (key_len != packed_len && key_len != expanded_len) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "%s key must be %u (packed) or %u (expanded) bytes, got %u",
                     mode_name, packed_len, expanded_len, key_len);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // ML-DSA has limited slots (0-3) due to key size
    if (slot >= SMOKE_MLDSA_MAX_KEYSLOTS) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 5;
            snprintf(out_error->message, sizeof(out_error->message),
                     "ML-DSA slot index %u out of range (max %u)",
                     slot, SMOKE_MLDSA_MAX_KEYSLOTS - 1);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!engine->fast_path) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 6;
            snprintf(out_error->message, sizeof(out_error->message),
                     "FAST PATH not created - call smoke_fast_path_create() first");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    // Call the underlying fast path key loader
    bool ok = smoke::engine::fast_path_load_mldsa_key(
        engine->fast_path,
        slot,
        mode,
        key_blob,
        key_len
    );

    if (!ok) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 7;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Failed to load %s key to GPU (check stderr; a build without "
                     "DILITHIUM_HAVE_CUDA has no ML-DSA support)", mode_name);
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    // Success
    if (out_error) {
        out_error->code = SMOKE_STATUS_OK;
        out_error->internal_code = 0;
        out_error->message[0] = '\0';
    }

    return SMOKE_STATUS_OK;
}

// ============================================================================
// Card 27.19: ML-DSA Sign (Host-Mediated Dispatch)
// ============================================================================

extern "C" SMOKE_API
SmokeStatus smoke_mldsa_sign(
    SmokeEngineHandle engine,
    uint8_t           mode,
    uint8_t           key_slot,
    const uint8_t*    msg,
    uint32_t          msg_len,
    uint8_t*          sig_out,
    uint32_t*         sig_len_out,
    SmokeErrorInfo*   out_error
) {
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Engine handle is null");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!msg || msg_len == 0) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Message is null or empty");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!sig_out || !sig_len_out) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Output buffer or length pointer is null");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (mode > 2) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Invalid ML-DSA mode %u (must be 0-2)", mode);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (key_slot >= SMOKE_MLDSA_MAX_KEYSLOTS) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 5;
            snprintf(out_error->message, sizeof(out_error->message),
                     "ML-DSA key slot %u out of range (max %u)",
                     key_slot, SMOKE_MLDSA_MAX_KEYSLOTS - 1);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!engine->fast_path) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 6;
            snprintf(out_error->message, sizeof(out_error->message),
                     "FAST PATH not created - call smoke_fast_path_create() first");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

    // Card 32C / D-098: refuse to sign while an async rotation is pending
    // fence — same engine-global latch as smoke_generic_submit /
    // smoke_engine_sign_p256 / smoke_engine_submit_slh_dsa_sign
    // (FENCE_REQUIRED gates ALL signing surfaces, including the separate
    // MLDSAKeyslot array).
    if (engine->rotation_pending_fence) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_FENCE_REQUIRED;
            out_error->internal_code = 11;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Async rotation pending. Call rotation_fence() before signing.");
        }
        return SMOKE_STATUS_ERROR_FENCE_REQUIRED;
    }

#ifdef DILITHIUM_HAVE_CUDA
    // Call the host-mediated signing function
    uint32_t submitted = smoke::engine::fast_path_submit_mldsa(
        engine->fast_path,
        msg,
        msg_len,
        mode,
        key_slot,
        0  // client_id
    );

    if (submitted == 0) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 7;
            snprintf(out_error->message, sizeof(out_error->message),
                     "ML-DSA signing failed (see stderr for details)");
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    // Card 27.24: Flush batch queue to ensure request is processed
    // The simple ABI is synchronous, so we flush immediately after submit.
    // For high-throughput batch signing, use the batch API directly.
    smoke::engine::fast_path_mldsa_flush_batch(engine->fast_path);

    // Get signature from output slab
    // The signature was written to the output slab by fast_path_submit_mldsa
    // We need to poll the response to get the output_offset and output_bytes

    // Poll for response
    smoke::engine::FastPathResponse resp;
    uint32_t count = smoke::engine::fast_path_poll(engine->fast_path, &resp, 1);

    if (count == 0) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_TIMEOUT;
            out_error->internal_code = 8;
            snprintf(out_error->message, sizeof(out_error->message),
                     "No response received from ML-DSA signing");
        }
        return SMOKE_STATUS_ERROR_TIMEOUT;
    }

    if (resp.status != 0) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 9;
            snprintf(out_error->message, sizeof(out_error->message),
                     "ML-DSA signing returned error status %u", resp.status);
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    // Copy signature from output slab to caller's buffer
    uint8_t* output_slab = nullptr;
    uint32_t output_slab_size = 0;
    smoke::engine::fast_path_get_output_slab(engine->fast_path, &output_slab, &output_slab_size);

    if (!output_slab || resp.output_offset + resp.output_bytes > output_slab_size) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 10;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Output slab access error");
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    memcpy(sig_out, output_slab + resp.output_offset, resp.output_bytes);
    *sig_len_out = resp.output_bytes;

    // Success
    if (out_error) {
        out_error->code = SMOKE_STATUS_OK;
        out_error->internal_code = 0;
        out_error->message[0] = '\0';
    }

    return SMOKE_STATUS_OK;
#else
    // Dilithium not available at compile time
    if (out_error) {
        out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
        out_error->internal_code = 99;
        snprintf(out_error->message, sizeof(out_error->message),
                 "ML-DSA not available (DILITHIUM_HAVE_CUDA not defined)");
    }
    return SMOKE_STATUS_ERROR_UNSUPPORTED;
#endif
}

// ============================================================================
// Card 27.25: ML-DSA Batch Sign (True B>1 GPU Batching)
// ============================================================================
// Signs multiple messages in a single GPU batch call.
// All messages use the same mode and key_slot.
// Returns the number of successfully signed messages.

extern "C" SMOKE_API
uint32_t smoke_mldsa_sign_batch(
    SmokeEngineHandle engine,
    uint8_t           mode,
    uint8_t           key_slot,
    const uint8_t* const* msgs,      // Array of message pointers
    const uint32_t*   msg_lens,      // Array of message lengths
    uint32_t          num_msgs,      // Number of messages
    uint8_t**         sigs_out,      // Array of signature output buffers
    uint32_t*         sig_lens_out,  // Array of signature lengths (output)
    SmokeErrorInfo*   out_error
) {
    if (!engine || !engine->fast_path) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Engine or fast_path is null");
        }
        return 0;
    }

    if (num_msgs == 0 || !msgs || !msg_lens || !sigs_out || !sig_lens_out) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Invalid batch parameters");
        }
        return 0;
    }

    // Card 32C / D-098: same engine-global rotation-fence latch as all other
    // signing surfaces — refuse the whole batch while pending (no partial
    // output, §4b).
    if (engine->rotation_pending_fence) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_FENCE_REQUIRED;
            out_error->internal_code = 10;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Async rotation pending. Call rotation_fence() before signing.");
        }
        return 0;
    }

#ifdef DILITHIUM_HAVE_CUDA
    // Submit all messages to batch queue without flushing
    for (uint32_t i = 0; i < num_msgs; i++) {
        if (!msgs[i] || msg_lens[i] == 0) continue;

        uint32_t submitted = smoke::engine::fast_path_submit_mldsa(
            engine->fast_path,
            msgs[i],
            msg_lens[i],
            mode,
            key_slot,
            0  // client_id
        );

        if (submitted == 0) {
            // Submit failed - flush what we have and return partial
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
                snprintf(out_error->message, sizeof(out_error->message),
                         "Failed to submit message %u", i);
            }
            break;
        }
    }

    // Flush the entire batch in one GPU call
    uint32_t flushed = smoke::engine::fast_path_mldsa_flush_batch(engine->fast_path);

    // Poll all responses
    uint32_t success_count = 0;
    uint8_t* output_slab = nullptr;
    uint32_t output_slab_size = 0;
    smoke::engine::fast_path_get_output_slab(engine->fast_path, &output_slab, &output_slab_size);

    for (uint32_t i = 0; i < flushed; i++) {
        smoke::engine::FastPathResponse resp;
        uint32_t count = smoke::engine::fast_path_poll(engine->fast_path, &resp, 1);

        if (count == 0) continue;

        if (resp.status == 0 && output_slab &&
            resp.output_offset + resp.output_bytes <= output_slab_size &&
            sigs_out[i] != nullptr) {
            memcpy(sigs_out[i], output_slab + resp.output_offset, resp.output_bytes);
            sig_lens_out[i] = resp.output_bytes;
            success_count++;
        } else {
            sig_lens_out[i] = 0;
        }
    }

    if (out_error) {
        out_error->code = SMOKE_STATUS_OK;
        out_error->internal_code = 0;
        out_error->message[0] = '\0';
    }

    return success_count;
#else
    if (out_error) {
        out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
        snprintf(out_error->message, sizeof(out_error->message),
                 "ML-DSA not available");
    }
    return 0;
#endif
}

// ============================================================================
// Card 27.42: ML-DSA Deterministic Batch Sign
// ============================================================================
// Signs multiple messages in deterministic mode (reproducible signatures).
// Uses rhoprime_det = SHAKE256(tr || msg || "SMOKE_DET_MLDSA", 32) instead
// of the random rhoprime from the keyslot.
//
// Key property: signing the same message with the same key produces identical
// signature bytes every time.

extern "C" SMOKE_API
uint32_t smoke_mldsa_sign_batch_deterministic(
    SmokeEngineHandle engine,
    uint8_t           mode,
    uint8_t           key_slot,
    const uint8_t* const* msgs,      // Array of message pointers
    const uint32_t*   msg_lens,      // Array of message lengths
    uint32_t          num_msgs,      // Number of messages
    uint8_t**         sigs_out,      // Array of signature output buffers
    uint32_t*         sig_lens_out,  // Array of signature lengths (output)
    SmokeErrorInfo*   out_error
) {
    if (!engine || !engine->fast_path) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Engine or fast_path is null");
        }
        return 0;
    }

    if (num_msgs == 0 || !msgs || !msg_lens || !sigs_out || !sig_lens_out) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Invalid batch parameters");
        }
        return 0;
    }

    // Card 32C / D-098: same engine-global rotation-fence latch as all other
    // signing surfaces — refuse the whole batch while pending (no partial
    // output, §4b).
    if (engine->rotation_pending_fence) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_FENCE_REQUIRED;
            out_error->internal_code = 10;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Async rotation pending. Call rotation_fence() before signing.");
        }
        return 0;
    }

#ifdef DILITHIUM_HAVE_CUDA
    // Card 27.42: MLDSA_FLAG_DETERMINISTIC = 0x01
    constexpr uint8_t MLDSA_FLAG_DETERMINISTIC = 0x01;

    // Submit all messages to batch queue with deterministic flag
    for (uint32_t i = 0; i < num_msgs; i++) {
        if (!msgs[i] || msg_lens[i] == 0) continue;

        uint32_t submitted = smoke::engine::fast_path_submit_mldsa(
            engine->fast_path,
            msgs[i],
            msg_lens[i],
            mode,
            key_slot,
            0,  // client_id
            MLDSA_FLAG_DETERMINISTIC  // Card 27.42: Deterministic mode
        );

        if (submitted == 0) {
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
                snprintf(out_error->message, sizeof(out_error->message),
                         "Failed to submit message %u", i);
            }
            break;
        }
    }

    // Flush the entire batch in one GPU call
    uint32_t flushed = smoke::engine::fast_path_mldsa_flush_batch(engine->fast_path);

    // Poll all responses
    uint32_t success_count = 0;
    uint8_t* output_slab = nullptr;
    uint32_t output_slab_size = 0;
    smoke::engine::fast_path_get_output_slab(engine->fast_path, &output_slab, &output_slab_size);

    for (uint32_t i = 0; i < flushed; i++) {
        smoke::engine::FastPathResponse resp;
        uint32_t count = smoke::engine::fast_path_poll(engine->fast_path, &resp, 1);

        if (count == 0) continue;

        if (resp.status == 0 && output_slab &&
            resp.output_offset + resp.output_bytes <= output_slab_size &&
            sigs_out[i] != nullptr) {
            memcpy(sigs_out[i], output_slab + resp.output_offset, resp.output_bytes);
            sig_lens_out[i] = resp.output_bytes;
            success_count++;
        } else {
            sig_lens_out[i] = 0;
        }
    }

    if (out_error) {
        out_error->code = SMOKE_STATUS_OK;
        out_error->internal_code = 0;
        out_error->message[0] = '\0';
    }

    return success_count;
#else
    if (out_error) {
        out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
        snprintf(out_error->message, sizeof(out_error->message),
                 "ML-DSA not available");
    }
    return 0;
#endif
}

// ============================================================================
// Card SLH-002 P2: SLH-DSA-SHA2-128s Key Load + Batch Sign (C ABI)
// ============================================================================
// Host-mediated dispatch mirroring the ML-DSA pattern (Card 27.19/27.24):
// the persistent kernel refuses opcode 53 inline; these entries queue and
// flush through the fused batch signer with the DEVICE-side sign prologue
// (SK.prf never resident in host RAM after load — the key-residency line PQ
// keys must hold). Exports exist in every build; without SLH_HAVE_CUDA they
// fail closed with SMOKE_STATUS_ERROR_UNSUPPORTED.
// ============================================================================

extern "C" SMOKE_API
SmokeStatus smoke_engine_load_key_slh_dsa(
    SmokeEngineHandle             engine,
    uint32_t                      slot,
    const SmokeSlhDsaPrivateKey*  key,
    SmokeErrorInfo*               out_error
) {
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Engine handle is NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!key) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Key struct is NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (slot >= SMOKE_SLH_DSA_MAX_KEYSLOTS) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "SLH-DSA slot index %u out of range (max %u)",
                     slot, SMOKE_SLH_DSA_MAX_KEYSLOTS - 1);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->fast_path) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "FAST PATH not created - call smoke_fast_path_create() first");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }

#ifdef SLH_HAVE_CUDA
    bool ok = smoke::engine::fast_path_load_slh_key(
        engine->fast_path, slot,
        key->sk_seed, key->sk_prf, key->pk_seed, key->pk_root);
    if (!ok) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 5;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Failed to load SLH-DSA-SHA2-128s key to GPU (see stderr)");
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }
    if (out_error) {
        out_error->code = SMOKE_STATUS_OK;
        out_error->internal_code = 0;
        out_error->message[0] = '\0';
    }
    return SMOKE_STATUS_OK;
#else
    if (out_error) {
        out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
        out_error->internal_code = 99;
        snprintf(out_error->message, sizeof(out_error->message),
                 "SLH-DSA not available (SLH_HAVE_CUDA not defined)");
    }
    return SMOKE_STATUS_ERROR_UNSUPPORTED;
#endif
}

extern "C" SMOKE_API
SmokeStatus smoke_engine_submit_slh_dsa_sign(
    SmokeEngineHandle     engine,
    uint8_t               key_slot,
    const uint8_t* const* msgs,
    const uint32_t*       msg_lens,
    const uint8_t*        addrnds,
    uint32_t              num_msgs,
    uint8_t*              sigs_out,
    SmokeErrorInfo*       out_error
) {
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Engine handle is NULL");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!msgs || !msg_lens || !sigs_out || num_msgs == 0) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Invalid batch parameters (msgs/msg_lens/sigs_out/num_msgs)");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (key_slot >= SMOKE_SLH_DSA_MAX_KEYSLOTS) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "SLH-DSA key slot %u out of range (max %u)",
                     key_slot, SMOKE_SLH_DSA_MAX_KEYSLOTS - 1);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    if (!engine->fast_path || !engine->fast_path_running) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_NOT_INITIALIZED;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "FAST PATH not running. Call smoke_fast_path_start() first.");
        }
        return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
    }
    // Card 32C / Card SLH-002: refuse to sign while an async rotation is
    // pending fence — identical latch to smoke_generic_submit /
    // smoke_engine_sign_p256 (FENCE_REQUIRED gates ALL signing surfaces).
    if (engine->rotation_pending_fence) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_FENCE_REQUIRED;
            out_error->internal_code = 10;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Async rotation pending. Call rotation_fence() before signing.");
        }
        return SMOKE_STATUS_ERROR_FENCE_REQUIRED;
    }

#ifdef SLH_HAVE_CUDA
    // Queue the whole batch (routing by request_id — D-052..D-054/B-321:
    // never route by client_id), then flush once.
    std::vector<uint64_t> request_ids(num_msgs, 0);
    for (uint32_t i = 0; i < num_msgs; i++) {
        if (!msgs[i] && msg_lens[i] > 0) {
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
                out_error->internal_code = 5;
                snprintf(out_error->message, sizeof(out_error->message),
                         "Message %u is NULL with nonzero length", i);
            }
            return SMOKE_STATUS_ERROR_INVALID_ARG;
        }
        if (msg_lens[i] > SMOKE_SLH_DSA_MAX_MSG_BYTES) {
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
                out_error->internal_code = 6;
                snprintf(out_error->message, sizeof(out_error->message),
                         "Message %u length %u exceeds %u",
                         i, msg_lens[i], SMOKE_SLH_DSA_MAX_MSG_BYTES);
            }
            return SMOKE_STATUS_ERROR_INVALID_ARG;
        }
        uint32_t submitted = smoke::engine::fast_path_submit_slh(
            engine->fast_path,
            msgs[i], msg_lens[i],
            addrnds ? addrnds + (size_t)i * 16 : nullptr,
            key_slot,
            0,  // client_id (transport tag only; routing is by request_id)
            &request_ids[i]);
        if (submitted == 0) {
            if (out_error) {
                out_error->code = SMOKE_STATUS_ERROR_KEY_MISSING;
                out_error->internal_code = 7;
                snprintf(out_error->message, sizeof(out_error->message),
                         "SLH-DSA submit refused at message %u (key slot %u "
                         "unloaded/faulted, or invalid input — see stderr)",
                         i, key_slot);
            }
            return SMOKE_STATUS_ERROR_KEY_MISSING;
        }
    }
    smoke::engine::fast_path_slh_flush_batch(engine->fast_path);

    // Drain responses and route them by request_id. NOTE (same contract as
    // smoke_mldsa_sign_batch): this synchronous entry assumes it owns the
    // completion stream for the duration of the call — interleaved foreign
    // responses drained here are counted and reported as a failure rather
    // than silently dropped.
    uint8_t* output_slab = nullptr;
    uint32_t output_slab_size = 0;
    smoke::engine::fast_path_get_output_slab(engine->fast_path, &output_slab,
                                             &output_slab_size);

    uint32_t matched = 0;
    uint32_t failed = 0;
    uint32_t foreign = 0;
    // One flush produces exactly num_msgs responses; poll with bounded
    // headroom (foreign responses must not spin this loop forever).
    const uint32_t poll_budget = num_msgs + 4096;
    for (uint32_t polled_total = 0;
         matched + failed < num_msgs && polled_total < poll_budget;
         polled_total++) {
        smoke::engine::FastPathResponse resp;
        uint32_t count = smoke::engine::fast_path_poll(engine->fast_path, &resp, 1);
        if (count == 0) break;   // dispatch posted everything already; gone = error
        // Find the batch index owning this request_id.
        uint32_t idx = num_msgs;
        for (uint32_t i = 0; i < num_msgs; i++) {
            if (request_ids[i] == resp.request_id) { idx = i; break; }
        }
        if (idx == num_msgs) { foreign++; continue; }
        if (resp.status != 0 || resp.output_bytes != SMOKE_SLH_DSA_128S_SIG_BYTES ||
            !output_slab ||
            resp.output_offset + resp.output_bytes > output_slab_size) {
            failed++;
            continue;
        }
        memcpy(sigs_out + (size_t)idx * SMOKE_SLH_DSA_128S_SIG_BYTES,
               output_slab + resp.output_offset, SMOKE_SLH_DSA_128S_SIG_BYTES);
        matched++;
    }

    if (matched != num_msgs) {
        // No partial output: zero the caller's buffer before failing.
        memset(sigs_out, 0, (size_t)num_msgs * SMOKE_SLH_DSA_128S_SIG_BYTES);
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
            out_error->internal_code = 8;
            snprintf(out_error->message, sizeof(out_error->message),
                     "SLH-DSA sign incomplete: %u/%u signatures (failed=%u, "
                     "foreign_responses=%u) — output zeroed",
                     matched, num_msgs, failed, foreign);
        }
        return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
    }

    if (out_error) {
        out_error->code = SMOKE_STATUS_OK;
        out_error->internal_code = 0;
        out_error->message[0] = '\0';
    }
    return SMOKE_STATUS_OK;
#else
    if (out_error) {
        out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
        out_error->internal_code = 99;
        snprintf(out_error->message, sizeof(out_error->message),
                 "SLH-DSA not available (SLH_HAVE_CUDA not defined)");
    }
    return SMOKE_STATUS_ERROR_UNSUPPORTED;
#endif
}

// Card SLH-002 §5d item 5 — DEBUG/TEST HOOK. Flips one device-mirror byte in
// SLH keyslot `key_slot` so the fused signer's key==mirror check trips on the
// next sign (fault-injection validation of the fail-closed path). Same
// debug-export family as smoke_debug_p256_*; never called by product code.
extern "C" SMOKE_API
SmokeStatus smoke_debug_slh_corrupt_key_mirror(
    SmokeEngineHandle engine,
    uint8_t           key_slot
) {
    if (!engine) return SMOKE_STATUS_ERROR_INVALID_ARG;
    if (!engine->fast_path) return SMOKE_STATUS_ERROR_NOT_INITIALIZED;
#ifdef SLH_HAVE_CUDA
    if (!smoke::engine::fast_path_slh_corrupt_key_mirror(engine->fast_path,
                                                         key_slot)) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }
    return SMOKE_STATUS_OK;
#else
    return SMOKE_STATUS_ERROR_UNSUPPORTED;
#endif
}

// ============================================================================
// Card 27.23: ML-DSA Verify (Host-Mediated)
// ============================================================================

extern "C" SMOKE_API
SmokeStatus smoke_mldsa_verify(
    SmokeEngineHandle engine,
    uint8_t           mode,
    const uint8_t*    pk,
    uint32_t          pk_len,
    const uint8_t*    msg,
    uint32_t          msg_len,
    const uint8_t*    sig,
    uint32_t          sig_len,
    SmokeErrorInfo*   out_error
) {
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Engine handle is null");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!pk || pk_len == 0) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Public key is null or empty");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!msg) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Message is null");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!sig || sig_len == 0) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Signature is null or empty");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (mode > 2) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 5;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Invalid ML-DSA mode %u (must be 0-2)", mode);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    // Validate public key size (simple format: rho || t1)
    uint32_t expected_pk_len = 0;
    uint32_t expected_sig_len = 0;
    const char* mode_name = nullptr;
    switch (mode) {
        case SMOKE_MLDSA_MODE_44:
            expected_pk_len = SMOKE_MLDSA44_PK_SIMPLE_BYTES;
            expected_sig_len = SMOKE_MLDSA44_SIG_BYTES;
            mode_name = "ML-DSA-44";
            break;
        case SMOKE_MLDSA_MODE_65:
            expected_pk_len = SMOKE_MLDSA65_PK_SIMPLE_BYTES;
            expected_sig_len = SMOKE_MLDSA65_SIG_BYTES;
            mode_name = "ML-DSA-65";
            break;
        case SMOKE_MLDSA_MODE_87:
            expected_pk_len = SMOKE_MLDSA87_PK_SIMPLE_BYTES;
            expected_sig_len = SMOKE_MLDSA87_SIG_BYTES;
            mode_name = "ML-DSA-87";
            break;
        default:
            break;
    }

    if (pk_len != expected_pk_len) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 6;
            snprintf(out_error->message, sizeof(out_error->message),
                     "%s public key must be %u bytes (simple format), got %u",
                     mode_name, expected_pk_len, pk_len);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (sig_len != expected_sig_len) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 7;
            snprintf(out_error->message, sizeof(out_error->message),
                     "%s signature must be %u bytes (FIPS 204), got %u",
                     mode_name, expected_sig_len, sig_len);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

#ifdef DILITHIUM_HAVE_CUDA
    // Card 27.28: Stop persistent kernel before GPU verification
    // (GPU verification uses kernels that conflict with the persistent kernel)
    bool was_running = (engine->fast_path != nullptr &&
                        smoke_fast_path_is_running(engine) != 0);
    if (was_running) {
        smoke::engine::fast_path_stop(engine->fast_path);
    }

    // Call the host-mediated verification function
    bool valid = smoke::engine::mldsa_verify_host_dispatch(
        mode,
        pk, pk_len,
        msg, msg_len,
        sig, sig_len
    );

    // Restart persistent kernel with quick restart mode
    if (was_running) {
        smoke::engine::fast_path_set_quick_restart(true);
        smoke::engine::fast_path_start(engine->fast_path);
        smoke::engine::fast_path_set_quick_restart(false);
    }

    if (valid) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_OK;
            out_error->internal_code = 0;
            out_error->message[0] = '\0';
        }
        return SMOKE_STATUS_OK;
    } else {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_VERIFY_FAILED;
            out_error->internal_code = 0;
            snprintf(out_error->message, sizeof(out_error->message),
                     "%s signature verification failed", mode_name);
        }
        return SMOKE_STATUS_ERROR_VERIFY_FAILED;
    }
#else
    // Dilithium not available at compile time
    if (out_error) {
        out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
        out_error->internal_code = 99;
        snprintf(out_error->message, sizeof(out_error->message),
                 "ML-DSA not available (DILITHIUM_HAVE_CUDA not defined)");
    }
    return SMOKE_STATUS_ERROR_UNSUPPORTED;
#endif
}

// ============================================================================
// Card 231: ML-DSA Keygen (C ABI)
// ============================================================================

SmokeStatus smoke_mldsa_keygen(
    SmokeEngineHandle engine,
    uint8_t           mode,
    const uint8_t*    rho,         // 32 bytes
    const uint8_t*    rhoprime,    // 32 bytes
    uint8_t*          pk_out,
    uint32_t          pk_max,
    uint32_t*         pk_len_out,
    uint8_t*          sk_out,
    uint32_t          sk_max,
    uint32_t*         sk_len_out,
    SmokeErrorInfo*   out_error
) {
    if (!engine) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 1;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Engine handle is null");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!rho || !rhoprime) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 2;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Seed (rho/rhoprime) is null");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (!pk_out || !sk_out || !pk_len_out || !sk_len_out) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 3;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Output buffer or length pointer is null");
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    if (mode > 2) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_ERROR_INVALID_ARG;
            out_error->internal_code = 4;
            snprintf(out_error->message, sizeof(out_error->message),
                     "Invalid ML-DSA mode %u (must be 0-2)", mode);
        }
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

#ifdef DILITHIUM_HAVE_CUDA
    // Card 283: Stop persistent kernel before GPU keygen
    // (keygen GPU kernels conflict with the persistent kernel)
    bool was_running = (engine->fast_path != nullptr &&
                        smoke_fast_path_is_running(engine) != 0);
    if (was_running) {
        smoke::engine::fast_path_stop(engine->fast_path);
    }

    bool ok = smoke::engine::mldsa_keygen_host_dispatch(
        engine->fast_path,  // may be NULL if no kernel was running
        mode,
        rho,
        rhoprime,
        pk_out, pk_max, pk_len_out,
        sk_out, sk_max, sk_len_out
    );

    // Restart persistent kernel with quick restart mode
    if (was_running) {
        smoke::engine::fast_path_set_quick_restart(true);
        smoke::engine::fast_path_start(engine->fast_path);
        smoke::engine::fast_path_set_quick_restart(false);
    }

    if (ok) {
        if (out_error) {
            out_error->code = SMOKE_STATUS_OK;
            out_error->internal_code = 0;
            out_error->message[0] = '\0';
        }
        return SMOKE_STATUS_OK;
    }

    if (out_error) {
        out_error->code = SMOKE_STATUS_ERROR_DEVICE_FAILURE;
        out_error->internal_code = 10;
        snprintf(out_error->message, sizeof(out_error->message),
                 "ML-DSA keygen GPU dispatch failed");
    }
    return SMOKE_STATUS_ERROR_DEVICE_FAILURE;
#else
    if (out_error) {
        out_error->code = SMOKE_STATUS_ERROR_UNSUPPORTED;
        out_error->internal_code = 99;
        snprintf(out_error->message, sizeof(out_error->message),
                 "ML-DSA not available (DILITHIUM_HAVE_CUDA not defined)");
    }
    return SMOKE_STATUS_ERROR_UNSUPPORTED;
#endif
}

// ============================================================================
// Card 26.86: Operation Schema Table (C ABI)
// ============================================================================

// Static table of all supported operations
// From op_schema.cuh: OpCode enum values (MUST MATCH — the AEAD rows are
// enforced by the static_asserts after the table)
static constexpr SmokeOpInfo s_op_schema[] = {
    // opcode, min_in, max_in, out_len, requires_key, reserved, name
    {  0, 32,     32,     64, 1, {0}, "P256_SIGN"          },  // 32-byte hash -> 64-byte sig
    // RRP-A06b: RSA2048_CRT_SIGN (1) removed — inverse of the A-06 rows
    // below (advertised here but NOT dispatched by the production sharded
    // kernel; the real handler is ENABLE_LEGACY_KERNEL dead code). Advertising
    // it was a fail-open capability lie: a probing caller enabled an RSA path
    // whose ring submission could never complete. Ring submission is refused
    // at smoke_generic_submit and the kernel returns INVALID_OPCODE; re-add
    // the row only when a sharded-kernel RSA handler actually lands.
    { 10,  1,     32,     32, 0, {0}, "SHA256"             },  // 1-32 bytes -> 32-byte digest
    { 11,  1,     32,     64, 0, {0}, "SHA512"             },  // 1-32 bytes -> 64-byte digest
    { 12,  1,     32,     32, 0, {0}, "SHA3_256"           },  // 1-32 bytes -> 32-byte digest
    { 14,  1,     32,     32, 0, {0}, "BLAKE2B_256"        },  // 1-32 bytes -> 32-byte digest
    // SEAL-SLAB-01: inline carries nonce[12]+pt[0-16]; larger plaintext is
    // carried in the payload slab through 65535 bytes. max_input_len is the
    // uint16 slab payload ceiling; dispatch validates the exact framing.
    { 20, 12,  65535,      0, 1, {0}, "AES256_GCM"         },  // nonce inline + pt slab -> ct+tag
    { 21, 12,  65535,      0, 1, {0}, "CHACHA20_POLY1305"  },  // nonce inline + pt slab -> ct+tag
    { 22, 16,     32,      0, 1, {0}, "AES256_CTR"         },  // 16-byte IV + ciphertext (in-place)
    { 23, 16,     32,      0, 1, {0}, "CHACHA20"           },  // 16-byte nonce + ciphertext (in-place)
    { 30, 32,     32,     32, 1, {0}, "HKDF_SHA256"        },  // 32-byte IKM -> 32-byte OKM
    { 40,  1,     32,     32, 1, {0}, "HMAC_SHA256"        },  // message -> 32-byte MAC
    { 41,  1,     32,     64, 1, {0}, "HMAC_SHA512"        },  // message -> 64-byte MAC
    { 42,  1,     32,     48, 1, {0}, "HMAC_SHA384"        },  // message -> 48-byte MAC
    { 43, 32,     32,     16, 1, {0}, "POLY1305"           },  // message + key -> 16-byte tag
    // A-06: SHA3-512 (13) and BLAKE2B-512 (15) are dispatched by the kernel
    // (engine_loop.cu) but were missing from this introspection table, so
    // capability-probing clients wrongly saw them as unsupported.
    { 13,  0,  65535,     64, 1, {0}, "SHA3_512"           },  // any -> digest[64]
    { 15,  0,  65535,     64, 1, {0}, "BLAKE2B_512"        },  // any -> digest[64]
    // D-089 chip (same A-06 capability-lie class as the rows above): the OPEN
    // (decrypt) opcodes are dispatched by the kernel — engine_loop.cu has
    // inline AND decrypt-slab branches on both dispatch paths — but were
    // missing from this table, so capability-probing clients wrongly saw AEAD
    // decrypt as unsupported. Bounds describe the logical AEAD input
    // nonce[12] || ct[N] || tag[16] and mirror op_schema.cuh OP_CONSTRAINTS:
    // min 28 = nonce + tag (0-byte ct). The inline slot carries at most 32
    // (ct 0-4; smoke_generic_submit fail-closes inline input_len > 32);
    // larger ciphertexts go via the payload slab (input = nonce[12],
    // payload = ct[N] || tag[16]). SmokeOpInfo has no slab-capability field,
    // so max_input_len describes the slab path; a reserved[] "slab-capable"
    // flag would be a new ABI contract for existing probing clients and is
    // deliberately not invented here. output_len=0 means variable: pt[N].
    { 24, 28,  65535,      0, 1, {0}, "AES256_GCM_OPEN"        },  // nonce[12]+ct[N]+tag[16] -> pt[N]
    { 25, 28,  65535,      0, 1, {0}, "CHACHA20_POLY1305_OPEN" },  // nonce[12]+ct[N]+tag[16] -> pt[N]
#ifdef DILITHIUM_HAVE_CUDA
    // Card 27.11: Post-quantum (ML-DSA). Advertised ONLY when compiled with
    // DILITHIUM_HAVE_CUDA — otherwise every MLDSA call returns UNSUPPORTED, so
    // advertising it here would be a fail-open capability lie (a probing caller
    // would enable a PQC path that cannot work).
    { 50,  8,  65535,      0, 1, {0}, "MLDSA_SIGN"         },  // msg (slab) -> sig (slab, mode-dependent)
#endif
#ifdef SLH_HAVE_CUDA
    // Card SLH-002 P2: SLH-DSA-SHA2-128s sign. Same gating rationale as the
    // MLDSA row: advertised ONLY when the fused signer is compiled in
    // (SLH_HAVE_CUDA — deliberately NOT piggybacked on DILITHIUM_HAVE_CUDA);
    // without it every SLH call fail-closes with UNSUPPORTED. Raw message in
    // (0..65535 B), fixed 7,856-byte signature out via the PQ signature slab
    // region. Host-mediated dispatch only (smoke_engine_submit_slh_dsa_sign);
    // ring submissions of opcode 53 are refused INVALID_OPCODE by the kernel.
    { 53,  0,  65535,   7856, 1, {0}, "SLH_DSA_SIGN"       },  // msg -> sig[7856] (PQ slab)
#endif
};

static constexpr uint32_t s_op_schema_count = sizeof(s_op_schema) / sizeof(s_op_schema[0]);

// AEAD schema lock (ABI side): this table and op_schema.cuh OP_CONSTRAINTS
// declare the same kernel truth. Pin the family at compile time so capability
// probing can never silently diverge from the slab implementation.
// Extending the pin to the hash rows first requires reconciling their bounds
// (this table declares inline-only 10/11/12/14; OP_CONSTRAINTS says 65535).
static constexpr bool abi_aead_row_matches_kernel(smoke::engine::OpCode op) {
    for (uint32_t i = 0; i < s_op_schema_count; i++) {
        if (s_op_schema[i].opcode != (uint16_t)op) continue;
        for (size_t j = 0; j < smoke::engine::OP_CONSTRAINTS_COUNT; j++) {
            if (smoke::engine::OP_CONSTRAINTS[j].opcode != (uint16_t)op) continue;
            return s_op_schema[i].min_input_len == smoke::engine::OP_CONSTRAINTS[j].min_input_len
                && s_op_schema[i].max_input_len == smoke::engine::OP_CONSTRAINTS[j].max_input_len
                && s_op_schema[i].output_len   == smoke::engine::OP_CONSTRAINTS[j].output_len
                && (s_op_schema[i].requires_key != 0) == smoke::engine::OP_CONSTRAINTS[j].requires_key;
        }
        return false;  // present here, missing from OP_CONSTRAINTS
    }
    return false;      // missing from this table
}
static_assert(abi_aead_row_matches_kernel(smoke::engine::OpCode::AES256_GCM),
    "ABI s_op_schema row 20 disagrees with op_schema.cuh OP_CONSTRAINTS (D-089)");
static_assert(abi_aead_row_matches_kernel(smoke::engine::OpCode::CHACHA20_POLY1305),
    "ABI s_op_schema row 21 disagrees with op_schema.cuh OP_CONSTRAINTS (D-089)");
static_assert(abi_aead_row_matches_kernel(smoke::engine::OpCode::AES256_GCM_OPEN),
    "ABI s_op_schema row 24 disagrees with op_schema.cuh OP_CONSTRAINTS (D-089)");
static_assert(abi_aead_row_matches_kernel(smoke::engine::OpCode::CHACHA20_POLY1305_OPEN),
    "ABI s_op_schema row 25 disagrees with op_schema.cuh OP_CONSTRAINTS (D-089)");
#ifdef SLH_HAVE_CUDA
// Card SLH-002 P2: extend the D-090 schema lock to the SLH row — the matcher
// is generic (name predates the PQ rows); one kernel truth, two tables,
// pinned at compile time.
static_assert(abi_aead_row_matches_kernel(smoke::engine::OpCode::SLH_DSA_SIGN),
    "ABI s_op_schema row 53 disagrees with op_schema.cuh OP_CONSTRAINTS (D-090)");
#endif

extern "C"
SmokeStatus smoke_get_op_info(
    uint16_t      opcode,
    SmokeOpInfo*  out_info
) {
    if (!out_info) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    for (uint32_t i = 0; i < s_op_schema_count; i++) {
        if (s_op_schema[i].opcode == opcode) {
            *out_info = s_op_schema[i];
            return SMOKE_STATUS_OK;
        }
    }

    return SMOKE_STATUS_ERROR_UNSUPPORTED;
}

extern "C"
uint32_t smoke_get_op_count(void) {
    return s_op_schema_count;
}

extern "C"
SmokeStatus smoke_get_op_info_by_index(
    uint32_t      index,
    SmokeOpInfo*  out_info
) {
    if (!out_info || index >= s_op_schema_count) {
        return SMOKE_STATUS_ERROR_INVALID_ARG;
    }

    *out_info = s_op_schema[index];
    return SMOKE_STATUS_OK;
}

// ============================================================================
// Card 24.2: Batch APIs for Sync-Collapse Optimization
//
// These internal functions expose batch submit/poll to the ABI bridge,
// allowing O(1) GPU syncs per batch instead of O(N) per request.
// ============================================================================

// Submit multiple sign requests in one batch (single GPU sync)
extern "C"
uint32_t smoke_engine_submit_batch(
    SmokeEngineHandle engine,
    const uint8_t* hashes,          // count * 32 bytes, contiguous
    const uint32_t* request_ids,    // count request IDs
    uint32_t count
) {
    if (!engine || !hashes || !request_ids || count == 0) {
        return 0;
    }
    if (!engine->persistent_engine || !engine->engine_running) {
        return 0;
    }

    return smoke::engine::engine_submit_batch(
        engine->persistent_engine,
        hashes,
        request_ids,
        count
    );
}

// Poll multiple responses in one batch (single GPU sync)
// Returns number of responses that were ready
extern "C"
uint32_t smoke_engine_poll_batch(
    SmokeEngineHandle engine,
    const uint32_t* request_ids,
    smoke::engine::EngineResponse* out,   // count responses
    uint8_t* out_ready,                   // count ready flags (1=ready, 0=not ready)
    uint32_t count
) {
    if (!engine || !request_ids || !out || !out_ready || count == 0) {
        return 0;
    }
    if (!engine->persistent_engine || !engine->engine_running) {
        return 0;
    }

    return smoke::engine::engine_poll_batch(
        engine->persistent_engine,
        request_ids,
        out,
        out_ready,
        count
    );
}

// Get next request ID (thread-safe)
extern "C"
uint32_t smoke_engine_next_request_id(SmokeEngineHandle engine) {
    if (!engine) return 0;
    return engine->next_request_id.fetch_add(1);
}

// Check if key is loaded
extern "C"
bool smoke_engine_key_loaded(SmokeEngineHandle engine) {
    return engine && engine->key_loaded;
}

// ============================================================================
// Card 24.4: Per-Thread Stream Support for Multi-Threaded Batch Operations
// ============================================================================

// Create a CUDA stream for the calling thread
extern "C"
void* smoke_engine_create_thread_stream() {
    cudaStream_t stream = smoke::engine::engine_create_thread_stream();
    return static_cast<void*>(stream);
}

// Destroy a thread stream
extern "C"
void smoke_engine_destroy_thread_stream(void* stream) {
    smoke::engine::engine_destroy_thread_stream(static_cast<cudaStream_t>(stream));
}

// Submit batch with explicit stream (thread-safe)
extern "C"
uint32_t smoke_engine_submit_batch_with_stream(
    SmokeEngineHandle engine,
    const uint8_t* hashes,
    const uint32_t* request_ids,
    uint32_t count,
    void* stream
) {
    if (!engine || !hashes || !request_ids || count == 0 || !stream) {
        return 0;
    }
    if (!engine->persistent_engine || !engine->engine_running) {
        return 0;
    }

    return smoke::engine::engine_submit_batch_with_stream(
        engine->persistent_engine,
        hashes,
        request_ids,
        count,
        static_cast<cudaStream_t>(stream)
    );
}

// Poll batch with explicit stream (thread-safe)
extern "C"
uint32_t smoke_engine_poll_batch_with_stream(
    SmokeEngineHandle engine,
    const uint32_t* request_ids,
    smoke::engine::EngineResponse* out,
    uint8_t* out_ready,
    uint32_t count,
    void* stream
) {
    if (!engine || !request_ids || !out || !out_ready || count == 0 || !stream) {
        return 0;
    }
    if (!engine->persistent_engine || !engine->engine_running) {
        return 0;
    }

    return smoke::engine::engine_poll_batch_with_stream(
        engine->persistent_engine,
        request_ids,
        out,
        out_ready,
        count,
        static_cast<cudaStream_t>(stream)
    );
}

// ============================================================================
// Python Bindings (pybind11) - Card 26.87: Conditional for C ABI-only build
// ============================================================================

#ifndef SMOKE_ABI_NO_PYTHON

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Global engine handle for Python (single instance for simplicity)
static SmokeEngineHandle g_engine = nullptr;

// =========================================================================
// Card 210: C ABI Audit Hooks (callback registration)
// =========================================================================

static SmokeAuditCallback g_audit_callback = nullptr;
static void* g_audit_user_data = nullptr;

extern "C" SMOKE_API
int smoke_engine_set_audit_callback(SmokeAuditCallback cb, void* user_data) {
    g_audit_callback = cb;
    g_audit_user_data = user_data;
    return 0;
}

extern "C" SMOKE_API
int smoke_engine_clear_audit_callback(void) {
    g_audit_callback = nullptr;
    g_audit_user_data = nullptr;
    return 0;
}

static inline void fire_audit_event(
    const char* op_name, uint16_t opcode, uint8_t status,
    uint8_t key_slot, uint32_t input_len, uint32_t output_len,
    uint64_t duration_us, uint64_t request_id = 0
) {
    if (g_audit_callback) {
        g_audit_callback(
            op_name, opcode, status, key_slot,
            input_len, output_len, duration_us, request_id,
            g_audit_user_data
        );
    }
}

// Python-friendly wrappers
static int py_init(int device_index) {
    if (g_engine) {
        return static_cast<int>(SMOKE_STATUS_OK);  // Already initialized
    }

    SmokeEngineConfig config = {};
    config.abi_version = SMOKE_ENGINE_ABI_VERSION;
    config.device_index = device_index;
    config.queue_depth = 1024;

    SmokeErrorInfo error = {};
    SmokeStatus status = smoke_engine_init(&config, &g_engine, &error);

    if (status != SMOKE_STATUS_OK) {
        throw std::runtime_error(std::string("Engine init failed: ") + error.message);
    }

    return static_cast<int>(status);
}

static int py_shutdown() {
    if (!g_engine) {
        return static_cast<int>(SMOKE_STATUS_OK);
    }

    SmokeStatus status = smoke_engine_shutdown(g_engine);
    g_engine = nullptr;
    return static_cast<int>(status);
}

static py::dict py_get_stats() {
    if (!g_engine) {
        throw std::runtime_error("Engine not initialized");
    }

    SmokeEngineStats stats = {};
    smoke_engine_get_stats(g_engine, &stats);

    py::dict result;
    result["total_sign_p256"] = stats.total_sign_p256;
    result["total_hash_sha256"] = stats.total_hash_sha256;
    result["total_errors"] = stats.total_errors;
    result["uptime_ms"] = stats.uptime_ms;
    return result;
}

static int py_load_key_p256(uint32_t key_id, py::bytes private_key) {
    if (!g_engine) {
        throw std::runtime_error("Engine not initialized");
    }

    std::string key_str = private_key;
    if (key_str.size() != 32) {
        throw std::invalid_argument("Private key must be 32 bytes");
    }

    SmokeP256KeyId kid = { key_id, 0 };
    SmokeP256PrivateKey pkey = {};
    memcpy(pkey.d, key_str.data(), 32);

    SmokeStatus status = smoke_engine_load_key_p256(g_engine, &kid, &pkey);

    // Clear key from stack
    memset(&pkey, 0, sizeof(pkey));

    return static_cast<int>(status);
}

// Card 26.57: Load AES-256 key (32 bytes) into specified slot
static int py_load_key_aes256(uint32_t slot_idx, py::bytes key) {
    if (!g_engine) {
        throw std::runtime_error("Engine not initialized");
    }

    std::string key_str = key;
    if (key_str.size() != 32) {
        throw std::invalid_argument("AES-256 key must be 32 bytes");
    }

    if (!g_engine->persistent_engine || !g_engine->engine_running) {
        return static_cast<int>(SMOKE_STATUS_ERROR_NOT_INITIALIZED);
    }

    // Use the typed key loader
    bool ok = smoke::engine::engine_set_key_typed(
        g_engine->persistent_engine,
        slot_idx,
        smoke::engine::KeyslotType::KEYSLOT_AES256,
        reinterpret_cast<const uint8_t*>(key_str.data()),
        32
    );

    return ok ? static_cast<int>(SMOKE_STATUS_OK) : static_cast<int>(SMOKE_STATUS_ERROR_DEVICE_FAILURE);
}

// Card 26.57: Load HMAC key (variable length, up to 64 bytes) into specified slot
static int py_load_key_hmac(uint32_t slot_idx, py::bytes key) {
    if (!g_engine) {
        throw std::runtime_error("Engine not initialized");
    }

    std::string key_str = key;
    if (key_str.size() == 0 || key_str.size() > 64) {
        throw std::invalid_argument("HMAC key must be 1-64 bytes");
    }

    if (!g_engine->persistent_engine || !g_engine->engine_running) {
        return static_cast<int>(SMOKE_STATUS_ERROR_NOT_INITIALIZED);
    }

    // Use the typed key loader
    bool ok = smoke::engine::engine_set_key_typed(
        g_engine->persistent_engine,
        slot_idx,
        smoke::engine::KeyslotType::KEYSLOT_HMAC,
        reinterpret_cast<const uint8_t*>(key_str.data()),
        static_cast<uint16_t>(key_str.size())
    );

    return ok ? static_cast<int>(SMOKE_STATUS_OK) : static_cast<int>(SMOKE_STATUS_ERROR_DEVICE_FAILURE);
}

// Card 26.57: Get key type in specified slot
static int py_get_key_type(uint32_t slot_idx) {
    if (!g_engine) {
        throw std::runtime_error("Engine not initialized");
    }

    if (!g_engine->persistent_engine || !g_engine->engine_running) {
        return static_cast<int>(smoke::engine::KeyslotType::KEYSLOT_EMPTY);
    }

    smoke::engine::KeyslotType key_type = smoke::engine::engine_get_key_type(
        g_engine->persistent_engine,
        slot_idx
    );

    return static_cast<int>(key_type);
}

static py::tuple py_sign_p256(uint32_t key_id, py::bytes msg_hash) {
    if (!g_engine) {
        throw std::runtime_error("Engine not initialized");
    }

    std::string hash_str = msg_hash;
    if (hash_str.size() != 32) {
        throw std::invalid_argument("Message hash must be 32 bytes");
    }

    SmokeP256SignRequest req = {};
    req.key_id.id = key_id;
    req.msg.data = reinterpret_cast<const uint8_t*>(hash_str.data());
    req.msg.len = 32;
    req.hash_precomputed = 1;

    SmokeP256SignResponse resp = {};
    SmokeErrorInfo error = {};

    SmokeStatus status = smoke_engine_sign_p256(g_engine, &req, &resp, &error);

    py::bytes r_bytes(reinterpret_cast<const char*>(resp.r), 32);
    py::bytes s_bytes(reinterpret_cast<const char*>(resp.s), 32);

    return py::make_tuple(static_cast<int>(status), r_bytes, s_bytes, error.message);
}

static py::tuple py_hash_sha256(py::bytes data) {
    if (!g_engine) {
        throw std::runtime_error("Engine not initialized");
    }

    std::string data_str = data;

    SmokeSHA256HashRequest req = {};
    req.input.data = reinterpret_cast<const uint8_t*>(data_str.data());
    req.input.len = data_str.size();

    SmokeSHA256HashResponse resp = {};
    SmokeErrorInfo error = {};

    SmokeStatus status = smoke_engine_hash_sha256(g_engine, &req, &resp, &error);

    py::bytes digest(reinterpret_cast<const char*>(resp.digest), 32);

    return py::make_tuple(static_cast<int>(status), digest, error.message);
}

// GPU epoch Merkle aggregation (smoke-merkle-v0) — defined in merkle_gpu.cu.
// C linkage: symbol is the plain name regardless of its defining namespace.
extern "C" int smoke_merkle_root_v0_gpu(const uint8_t* h_leaves, uint32_t n, uint8_t out_root[32]);

PYBIND11_MODULE(smoke_engine, m) {
    m.doc() = "Smoke Engine ABI v1 - GPU Cryptographic Engine";

    m.attr("ABI_VERSION") = SMOKE_ENGINE_ABI_VERSION;

    // Status codes
    m.attr("STATUS_OK") = static_cast<int>(SMOKE_STATUS_OK);
    m.attr("STATUS_ERROR_UNKNOWN") = static_cast<int>(SMOKE_STATUS_ERROR_UNKNOWN);
    m.attr("STATUS_ERROR_INVALID_ARG") = static_cast<int>(SMOKE_STATUS_ERROR_INVALID_ARG);
    m.attr("STATUS_ERROR_NOT_INITIALIZED") = static_cast<int>(SMOKE_STATUS_ERROR_NOT_INITIALIZED);
    m.attr("STATUS_ERROR_DEVICE_FAILURE") = static_cast<int>(SMOKE_STATUS_ERROR_DEVICE_FAILURE);
    m.attr("STATUS_ERROR_TIMEOUT") = static_cast<int>(SMOKE_STATUS_ERROR_TIMEOUT);
    m.attr("STATUS_ERROR_KEY_MISSING") = static_cast<int>(SMOKE_STATUS_ERROR_KEY_MISSING);
    m.attr("STATUS_ERROR_UNSUPPORTED") = static_cast<int>(SMOKE_STATUS_ERROR_UNSUPPORTED);
    m.attr("STATUS_ERROR_NOT_IMPLEMENTED") = static_cast<int>(SMOKE_STATUS_ERROR_NOT_IMPLEMENTED);
    m.attr("STATUS_ERROR_FENCE_REQUIRED") = static_cast<int>(SMOKE_STATUS_ERROR_FENCE_REQUIRED);

    // Card 26.22: FAST PATH flags for multi-message mode
    m.attr("FAST_PATH_FLAG_LOW_S") = static_cast<int>(smoke::engine::FAST_PATH_FLAG_LOW_S);
    m.attr("FAST_PATH_FLAG_MULTIMSG") = static_cast<int>(smoke::engine::FAST_PATH_FLAG_MULTIMSG);

    // Card 26.25: Response slab segment size for N > 2 multi-message outputs
    m.attr("FAST_PATH_OUTPUT_SEGMENT_BYTES") = static_cast<int>(smoke::engine::FAST_PATH_OUTPUT_SEGMENT_BYTES);

    // Card 26.57: Keyslot type constants
    m.attr("KEYSLOT_EMPTY") = static_cast<int>(smoke::engine::KeyslotType::KEYSLOT_EMPTY);
    m.attr("KEYSLOT_P256") = static_cast<int>(smoke::engine::KeyslotType::KEYSLOT_P256);
    m.attr("KEYSLOT_AES256") = static_cast<int>(smoke::engine::KeyslotType::KEYSLOT_AES256);
    m.attr("KEYSLOT_HMAC") = static_cast<int>(smoke::engine::KeyslotType::KEYSLOT_HMAC);
    m.attr("KEYSLOT_RSA2048") = static_cast<int>(smoke::engine::KeyslotType::KEYSLOT_RSA2048);

    // Functions
    m.def("init", &py_init, py::arg("device_index") = -1,
          "Initialize the engine on specified GPU device (-1 = default)");
    m.def("shutdown", &py_shutdown,
          "Shutdown the engine and free resources");
    m.def("get_stats", &py_get_stats,
          "Get engine statistics");
    m.def("load_key_p256", &py_load_key_p256,
          py::arg("key_id"), py::arg("private_key"),
          "Load a P-256 private key (32 bytes, big-endian)");

    // Card 26.57: Typed key loaders
    m.def("load_key_aes256", &py_load_key_aes256,
          py::arg("slot_idx"), py::arg("key"),
          "Load an AES-256 key (32 bytes) into specified slot");
    m.def("load_key_hmac", &py_load_key_hmac,
          py::arg("slot_idx"), py::arg("key"),
          "Load an HMAC key (1-64 bytes) into specified slot");
    m.def("get_key_type", &py_get_key_type,
          py::arg("slot_idx"),
          "Get key type in specified slot (returns KEYSLOT_* constant)");
    m.def("sign_p256", &py_sign_p256,
          py::arg("key_id"), py::arg("msg_hash"),
          "Sign a 32-byte message hash. Returns (status, r, s, error_msg)");
    m.def("hash_sha256", &py_hash_sha256,
          py::arg("data"),
          "Compute SHA-256 hash. Returns (status, digest, error_msg)");

    // ABI Bridge functions (Card 22)
    m.def("abi_start", [](py::object shm_name) {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        const char* name = shm_name.is_none() ? nullptr : py::cast<std::string>(shm_name).c_str();
        return smoke_abi_start(g_engine, name);
    }, py::arg("shm_name") = py::none(),
       "Start ABI bridge with native worker thread");

    m.def("abi_stop", []() {
        smoke_abi_stop();
    }, "Stop ABI bridge");

    m.def("abi_is_running", []() {
        return smoke_abi_is_running() != 0;
    }, "Check if ABI bridge is running");

    // Card 24: Multi-client ABI functions
    m.def("abi_start_multi", [](uint32_t num_clients, py::object shm_name) {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        const char* name = shm_name.is_none() ? nullptr : py::cast<std::string>(shm_name).c_str();
        return smoke_abi_start_multi(g_engine, num_clients, name);
    }, py::arg("num_clients"), py::arg("shm_name") = py::none(),
       "Start multi-client ABI bridge with N client slots");

    m.def("abi_stop_multi", []() {
        smoke_abi_stop_multi();
    }, "Stop multi-client ABI bridge");

    m.def("abi_multi_is_running", []() {
        return smoke_abi_multi_is_running() != 0;
    }, "Check if multi-client ABI bridge is running");

    m.def("abi_get_num_clients", []() {
        return smoke_abi_get_num_clients();
    }, "Get number of configured client slots");

    // =========================================================================
    // Card 25.1: FAST PATH APIs
    // =========================================================================

    m.def("fast_path_create", [](int64_t legacy_handle) -> int64_t {
        auto* engine = reinterpret_cast<smoke::engine::EngineHandle*>(legacy_handle);
        auto* fp = smoke::engine::fast_path_create(engine);
        return reinterpret_cast<int64_t>(fp);
    }, py::arg("legacy_handle"),
       "Create FAST PATH handle from legacy engine handle");

    m.def("fast_path_destroy", [](int64_t handle) {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        smoke::engine::fast_path_destroy(fp);
    }, py::arg("handle"),
       "Destroy FAST PATH handle");

    m.def("fast_path_start", [](int64_t handle) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        // SPEED-H2-01 GAP 2a (2026-07-22): this lambda calls the namespace
        // fast_path_start directly, BYPASSING smoke_fast_path_start and its
        // N5 comb / C4 full-window fail-closed gates. The Python flow loads
        // keys via load_key_p256 -> smoke_engine_load_key_p256(g_engine,...),
        // so g_engine carries the authoritative has_p256_key state, and the
        // pybind table loaders below record their loaded-state on g_engine.
        // Replicate the gates here: refuse to start, structured error, no
        // fallback. Same conditions, same order as smoke_fast_path_start.
        if (g_engine && g_engine->has_p256_key) {
            if (!g_engine->p256_comb_loaded) {
                throw std::runtime_error(
                    "fast_path_start: P-256 keyslot loaded but comb table missing. "
                    "Call load_p256_comb_table() before fast_path_start() "
                    "(N5 fail-closed gate).");
            }
#if P256_FASTPATH_FULLWINDOW
            if (!g_engine->p256_fullwindow_loaded) {
                throw std::runtime_error(
                    "fast_path_start: P-256 keyslot loaded but full-window table "
                    "missing. Call load_p256_fullwindow_table() before "
                    "fast_path_start() (SPEED-H2-01 C4 fail-closed gate).");
            }
#endif
        }
        return smoke::engine::fast_path_start(fp);
    }, py::arg("handle"),
       "Start FAST PATH persistent kernel");

    // Card 289: Per-engine FAST PATH handle registration
    // pybind11 variant uses g_engine (Python doesn't hold the SmokeEngineHandle directly).
    // The pure C ABI variant (smoke_engine_set_fast_path_handle) is for ctypes/foreign callers.
    m.def("engine_set_fast_path_handle", [](int64_t /*legacy_handle*/, int64_t fp_handle) -> int {
        if (!g_engine) {
            throw std::runtime_error("engine_set_fast_path_handle: engine not initialized");
        }
        g_engine->fast_path = reinterpret_cast<smoke::engine::FastPathHandle*>(fp_handle);
        g_engine->fast_path_running = (fp_handle != 0);
        return SMOKE_STATUS_OK;
    }, py::arg("engine_handle"), py::arg("fp_handle"),
       "Register FAST PATH handle on engine (Card 289)");

    m.def("engine_clear_fast_path_handle", [](int64_t /*legacy_handle*/) -> int {
        if (!g_engine) {
            throw std::runtime_error("engine_clear_fast_path_handle: engine not initialized");
        }
        g_engine->fast_path = nullptr;
        g_engine->fast_path_running = false;
        return SMOKE_STATUS_OK;
    }, py::arg("engine_handle"),
       "Clear FAST PATH handle from engine (Card 289)");

    // Card 280: Global register/unregister (DEPRECATED — use per-engine above)
    m.def("register_fast_path", [](int64_t handle) {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized — call init() first");
        }
        g_engine->fast_path = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        g_engine->fast_path_running = (handle != 0);
    }, py::arg("handle"),
       "[DEPRECATED: use engine_set_fast_path_handle] Register FAST PATH handle into global g_engine");

    m.def("unregister_fast_path", []() {
        if (g_engine) {
            g_engine->fast_path = nullptr;
            g_engine->fast_path_running = false;
        }
    }, "[DEPRECATED: use engine_clear_fast_path_handle] Clear FAST PATH handle from global g_engine");

    // Card 26.2: Set number of CTAs (service lanes) before start
    m.def("fast_path_set_num_ctas", [](int64_t handle, uint32_t num_ctas) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_num_ctas(fp, num_ctas);
    }, py::arg("handle"), py::arg("num_ctas"),
       "Set number of CTAs (service lanes) for FAST PATH. Must be called before fast_path_start().");

    // Card 26.2: Set number of shards (zero contention mode)
    m.def("fast_path_set_num_shards", [](int64_t handle, uint32_t num_shards) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_num_shards(fp, num_shards);
    }, py::arg("handle"), py::arg("num_shards"),
       "Set number of ring shards for FAST PATH. Must be called before fast_path_start(). "
       "Requires num_shards == num_ctas for exclusive shard ownership (zero contention).");

    // Card 26.12: Set CQ capacity before start
    m.def("fast_path_set_cq_capacity", [](int64_t handle, uint32_t cq_capacity) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_cq_capacity(fp, cq_capacity);
    }, py::arg("handle"), py::arg("cq_capacity"),
       "Set CQ capacity for FAST PATH. Must be called before fast_path_start(). Range: 256-32768.");

    // Card 26.14: Set submit policy (0=RR_PER_REQ, 1=SHARD_LOCAL)
    m.def("fast_path_set_submit_policy", [](int64_t handle, uint32_t policy) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_submit_policy(fp, policy);
    }, py::arg("handle"), py::arg("policy"),
       "Set submit policy: 0=RR_PER_REQ (round-robin), 1=SHARD_LOCAL (pack into K shards). "
       "SHARD_LOCAL reduces memory fences per batch. Call before fast_path_start() (Card 26.14).");

    // Card 26.14: Set shards per batch for SHARD_LOCAL policy
    m.def("fast_path_set_shards_per_batch", [](int64_t handle, uint32_t k) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_shards_per_batch(fp, k);
    }, py::arg("handle"), py::arg("k"),
       "Set number of shards (K) to touch per batch for SHARD_LOCAL policy. "
       "Default: 1 (minimum fences). Call before fast_path_start() (Card 26.14).");

    m.def("fast_path_submit", [](int64_t handle, py::bytes hash_bytes, uint8_t flags) -> bool {
        // Card 32C: Guard against pending async rotation
        if (g_engine && g_engine->rotation_pending_fence) {
            throw std::runtime_error("Async rotation pending. Call rotation_fence() before submitting.");
        }
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::string hash_str = hash_bytes;
        if (hash_str.size() != 32) {
            throw std::runtime_error("Hash must be exactly 32 bytes");
        }
        return smoke::engine::fast_path_submit(fp,
            reinterpret_cast<const uint8_t*>(hash_str.data()), flags);
    }, py::arg("handle"), py::arg("hash"), py::arg("flags"),
       "Submit a single hash through FAST PATH");

    // Card 26.16: Added client_id for multi-client CQ partitioning
    // Card 26.19: Added opcode and input_len for multi-op support
    // D5b-keyslot: Added key_slot (shared by the batch; default 0)
    m.def("fast_path_submit_batch", [](int64_t handle, py::bytes inputs_bytes, uint8_t flags,
                                        uint8_t client_id, uint16_t opcode, uint16_t input_len,
                                        uint8_t key_slot) -> uint32_t {
        // Card 32C: Guard against pending async rotation
        if (g_engine && g_engine->rotation_pending_fence) {
            throw std::runtime_error("Async rotation pending. Call rotation_fence() before submitting.");
        }
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::string inputs_str = inputs_bytes;
        if (inputs_str.size() % 32 != 0) {
            throw std::runtime_error("Input buffer must be a multiple of 32 bytes");
        }
        uint32_t count = static_cast<uint32_t>(inputs_str.size() / 32);
        return smoke::engine::fast_path_submit_batch(fp,
            reinterpret_cast<const uint8_t*>(inputs_str.data()), count, flags, client_id, opcode, input_len,
            key_slot);
    }, py::arg("handle"), py::arg("inputs"), py::arg("flags") = 0,
       py::arg("client_id") = 0, py::arg("opcode") = 0, py::arg("input_len") = 32,
       py::arg("key_slot") = 0,
       "Submit batch through FAST PATH. opcode: 0=P256_SIGN, 10=SHA256 (Card 26.19)");

    m.def("fast_path_poll", [](int64_t handle, uint32_t max_count) -> py::list {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        std::vector<smoke::engine::FastPathResponse> responses(max_count);
        uint32_t got = smoke::engine::fast_path_poll(fp, responses.data(), max_count);

        py::list result;
        for (uint32_t i = 0; i < got; i++) {
            py::dict resp;
            resp["request_id"] = responses[i].request_id;
            resp["status"] = responses[i].status;
            resp["client_id"] = responses[i].client_id;  // Card 26.16: Multi-client
            resp["r"] = py::bytes(reinterpret_cast<const char*>(responses[i].r), 32);
            resp["s"] = py::bytes(reinterpret_cast<const char*>(responses[i].s), 32);
            // Card 25.2C: Timestamp truth - expose timing data
            resp["t_submit_us"] = responses[i].t_submit_us;
            resp["t_dequeue_us"] = responses[i].t_dequeue_us;
            resp["t_complete_us"] = responses[i].t_complete_us;
            resp["epoch_used"] = responses[i].epoch_used;    // Card 66
            resp["slot_used"] = responses[i].slot_used;      // Card 66
            result.append(resp);
        }
        return result;
    }, py::arg("handle"), py::arg("max_count"),
       "Poll for completed responses from FAST PATH");

    // Card 26.25: Extended poll that returns slab metadata (output_offset, output_bytes)
    m.def("fast_path_poll_extended", [](int64_t handle, uint32_t max_count) -> py::list {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        std::vector<smoke::engine::FastPathResponse> responses(max_count);
        uint32_t got = smoke::engine::fast_path_poll(fp, responses.data(), max_count);

        py::list result;
        for (uint32_t i = 0; i < got; i++) {
            py::dict resp;
            resp["request_id"] = responses[i].request_id;
            resp["status"] = responses[i].status;
            resp["client_id"] = responses[i].client_id;
            resp["r"] = py::bytes(reinterpret_cast<const char*>(responses[i].r), 32);
            resp["s"] = py::bytes(reinterpret_cast<const char*>(responses[i].s), 32);
            // Standard timing fields
            resp["t_submit_us"] = responses[i].t_submit_us;
            resp["t_dequeue_us"] = responses[i].t_dequeue_us;
            resp["t_complete_us"] = responses[i].t_complete_us;
            resp["epoch_used"] = responses[i].epoch_used;    // Card 66
            resp["slot_used"] = responses[i].slot_used;      // Card 66
            // Card 26.25: Extended output for slab mode
            resp["output_len"] = responses[i].output_len;
            resp["output_offset"] = responses[i].output_offset;
            resp["output_bytes"] = responses[i].output_bytes;
            // For inline mode, provide output as well
            resp["output"] = py::bytes(reinterpret_cast<const char*>(responses[i].output), 64);
            result.append(resp);
        }
        return result;
    }, py::arg("handle"), py::arg("max_count"),
       "Poll for completed responses with extended slab metadata (Card 26.25)");

    // Card 26.25: Get output slab for reading extended multi-message results
    m.def("fast_path_get_output_slab", [](int64_t handle) -> py::bytes {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        uint8_t* slab_ptr = nullptr;
        uint32_t slab_size = 0;
        if (!smoke::engine::fast_path_get_output_slab(fp, &slab_ptr, &slab_size) ||
            !slab_ptr || slab_size == 0) {
            return py::bytes();  // Empty bytes on failure
        }

        return py::bytes(reinterpret_cast<const char*>(slab_ptr), slab_size);
    }, py::arg("handle"),
       "Get output slab for reading extended multi-message results (Card 26.25)");

    // Slab-fix: TOCTOU-safe slab read. The caller MUST pass the response's
    // request_id; the read is owner-verified and recency-bracketed (checked
    // before AND after the copy) so a segment recycled between poll and read —
    // or overwritten mid-copy — throws instead of returning stale bytes.
    m.def("fast_path_read_output_segment", [](int64_t handle, uint32_t offset,
                                              uint32_t length, uint64_t request_id) -> py::bytes {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::vector<uint8_t> buf(length);
        int rc = smoke::engine::fast_path_read_output_segment_checked(
            fp, offset, length, request_id, buf.data());
        if (rc != 0) {
            static const char* why[] = {
                "", "bad handle/args", "out of slab bounds", "read spans segments",
                "shards not configured", "segment outside shard span",
                "not the segment's current owner (recycled or never drained)",
                "segment aged out before the read (drain-to-read gap too long)",
                "segment recycled DURING the read"};
            int idx = (rc >= -8 && rc <= -1) ? -rc : 1;
            throw std::runtime_error(
                std::string("slab segment read failed (fail-closed): ") + why[idx]);
        }
        return py::bytes(reinterpret_cast<const char*>(buf.data()), length);
    }, py::arg("handle"), py::arg("offset"), py::arg("length"), py::arg("request_id"),
       "Owner-verified, recency-bracketed slab read; throws if the segment may be recycled");

    // Slab-fix: fail-loud audit counter — responses forced to SLAB_OVERRUN (status 8)
    // at drain because their slab segment may have been recycled before the host read it.
    m.def("fast_path_slab_overrun_count", [](int64_t handle) -> uint64_t {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_slab_overrun_count(fp);
    }, py::arg("handle"),
       "Count of slab-mode responses forced to SLAB_OVERRUN at drain (must be 0 for a clean run)");

    // Card 26.34: Get payload slab for writing multi-sign inputs
    m.def("fast_path_get_payload_slab", [](int64_t handle) -> py::tuple {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        uint8_t* slab_ptr = nullptr;
        uint32_t slab_size = 0;
        if (!smoke::engine::fast_path_get_payload_slab(fp, &slab_ptr, &slab_size) ||
            !slab_ptr || slab_size == 0) {
            return py::make_tuple(py::none(), 0);  // None on failure
        }

        // Return memoryview for zero-copy writes
        return py::make_tuple(
            py::memoryview::from_memory(slab_ptr, slab_size),
            slab_size
        );
    }, py::arg("handle"),
       "Get payload slab (memoryview, size) for writing multi-sign inputs (Card 26.34)");

    // Card 26.34: Submit multi-sign request with payload slab
    // n_sigs: number of signatures to compute
    // hashes: bytes containing n_sigs * 32 bytes of message hashes
    // Returns: number of requests submitted (1 on success, 0 on failure)
    m.def("fast_path_submit_multisign", [](int64_t handle, uint16_t n_sigs, py::bytes hashes_bytes,
                                            uint8_t flags, uint8_t client_id, uint8_t key_slot) -> uint32_t {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::string hashes_str = hashes_bytes;

        // Validate
        if (n_sigs == 0 || n_sigs > 32) {
            throw std::runtime_error("n_sigs must be in [1, 32]");
        }
        if (hashes_str.size() != n_sigs * 32) {
            throw std::runtime_error("hashes must be n_sigs * 32 bytes");
        }

        // Get payload slab
        uint8_t* payload_ptr = nullptr;
        uint32_t payload_size = 0;
        if (!smoke::engine::fast_path_get_payload_slab(fp, &payload_ptr, &payload_size) ||
            !payload_ptr || payload_size == 0) {
            throw std::runtime_error("Payload slab not available");
        }

        // Write hashes to payload slab (at offset 0 for simplicity)
        uint32_t payload_offset = 0;
        uint32_t payload_len = n_sigs * 32;
        if (payload_len > payload_size) {
            throw std::runtime_error("Payload too large for slab");
        }
        memcpy(payload_ptr + payload_offset, hashes_str.data(), payload_len);

        // Build header in input[0..7]
        uint8_t input[32] = {0};
        // MultiMsgHeader: n_msgs (u16), msg_len (u16), flags (u32)
        input[0] = n_sigs & 0xFF;
        input[1] = (n_sigs >> 8) & 0xFF;
        input[2] = 32;  // msg_len = 32
        input[3] = 0;
        // flags = 0 (reserved)
        input[4] = input[5] = input[6] = input[7] = 0;

        // Submit single request with MULTIMSG flag
        uint8_t req_flags = flags | 0x02;  // FAST_PATH_FLAG_MULTIMSG = 0x02

        return smoke::engine::fast_path_submit_multisign(fp, input, req_flags, client_id,
            key_slot, payload_offset, payload_len);
    }, py::arg("handle"), py::arg("n_sigs"), py::arg("hashes"),
       py::arg("flags") = 0, py::arg("client_id") = 0, py::arg("key_slot") = 0,
       "Submit multi-sign request (n_sigs signatures from hashes via payload slab) (Card 26.34)");

    // DECRYPT-SLAB-01: Generic submit with full control (payload slab support)
    m.def("fast_path_submit_generic", [](int64_t handle, py::bytes input_bytes,
                                          uint16_t input_len, uint8_t flags,
                                          uint8_t key_slot, uint8_t client_id,
                                          uint16_t opcode,
                                          uint32_t payload_offset,
                                          uint32_t payload_len) -> uint32_t {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::string input_str = input_bytes;

        uint8_t input_buf[32] = {0};
        size_t copy_len = std::min<size_t>(32, input_str.size());
        memcpy(input_buf, input_str.data(), copy_len);

        return smoke::engine::fast_path_submit_generic(
            fp, input_buf, input_len, flags, key_slot, client_id,
            opcode, payload_offset, payload_len);
    }, py::arg("handle"), py::arg("input"), py::arg("input_len"),
       py::arg("flags"), py::arg("key_slot"), py::arg("client_id"),
       py::arg("opcode"), py::arg("payload_offset"), py::arg("payload_len"),
       "Submit request with full control over all fields including payload slab (DECRYPT-SLAB-01)");

    m.def("fast_path_get_telemetry", [](int64_t handle) -> py::dict {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        smoke::engine::FastPathTelemetry telem;
        smoke::engine::fast_path_get_telemetry(fp, &telem);

        py::dict result;
        result["total_requests_seen"] = telem.total_requests_seen;
        result["total_batches_processed"] = telem.total_batches_processed;
        result["total_batch_size_sum"] = telem.total_batch_size_sum;
        result["kernel_idle_cycles"] = telem.kernel_idle_cycles;
        result["kernel_busy_cycles"] = telem.kernel_busy_cycles;
        result["kernel_launches"] = telem.kernel_launches;
        result["kernel_start_time_us"] = telem.kernel_start_time_us;
        result["debug_responses_written"] = telem.debug_responses_written;
        result["debug_last_batch_size"] = telem.debug_last_batch_size;
        // Card 25.2B: Outcome counters
        result["outcome_ok"] = telem.outcome_ok;
        result["outcome_gpu_error"] = telem.outcome_gpu_error;

        // Card 26.1: Shape truth fields - proves which signer is ACTUALLY engaged
        result["signer_model"] = telem.signer_model;  // 0=THREAD_ONLY, 1=WARP_COOP
        result["signer_model_name"] = (telem.signer_model == 1) ? "WARP_COOP" : "THREAD_ONLY";
        result["threads_per_signature"] = telem.threads_per_signature;
        result["service_lanes"] = telem.service_lanes;
        result["comb_table_loaded"] = telem.comb_table_loaded;

        // Card 26.1B: Per-CTA distribution (starvation proof)
        py::list per_cta_batches;
        py::list per_cta_requests;
        py::list per_cta_attempts;  // Card 26.2: Dequeue attempts per CTA
        uint32_t num_lanes = telem.service_lanes;
        if (num_lanes == 0) num_lanes = 1;  // At least 1 lane
        if (num_lanes > 64) num_lanes = 64;  // Card 26.2: Increased limit
        for (uint32_t i = 0; i < num_lanes; i++) {
            per_cta_batches.append(telem.per_cta_batches[i]);
            per_cta_requests.append(telem.per_cta_requests[i]);
            per_cta_attempts.append(telem.per_cta_attempts[i]);
        }
        result["per_cta_batches"] = per_cta_batches;
        result["per_cta_requests"] = per_cta_requests;
        result["per_cta_attempts"] = per_cta_attempts;

        // Card 26.2: Debug beacons for hang localization
        result["dbg_enter_kernel"] = telem.dbg_enter_kernel;
        result["dbg_dequeue_attempts"] = telem.dbg_dequeue_attempts;
        result["dbg_claim_success"] = telem.dbg_claim_success;
        result["dbg_batch_begin"] = telem.dbg_batch_begin;
        result["dbg_sign_begin"] = telem.dbg_sign_begin;
        result["dbg_sign_done"] = telem.dbg_sign_done;
        result["dbg_write_response"] = telem.dbg_write_response;
        result["dbg_loop_heartbeat"] = telem.dbg_loop_heartbeat;

        // Card 26.5: Inversion census counters
        result["dbg_inv_fermat_calls"] = telem.dbg_inv_fermat_calls;
        result["dbg_inv_card14_batches"] = telem.dbg_inv_card14_batches;
        result["dbg_j2a_calls"] = telem.dbg_j2a_calls;
        result["dbg_scalar_mul_calls"] = telem.dbg_scalar_mul_calls;

        // Card 26.5: Stage cycle counters
        result["cyc_dequeue"] = telem.cyc_dequeue;
        result["cyc_scalar_mul"] = telem.cyc_scalar_mul;
        result["cyc_affine"] = telem.cyc_affine;
        result["cyc_sign_finalize"] = telem.cyc_sign_finalize;
        result["cyc_enqueue_resp"] = telem.cyc_enqueue_resp;

        // Card 26.10: Response buffer overflow detection
        result["response_overwrites"] = telem.response_overwrites;
        result["response_seq_counter"] = telem.response_seq_counter;

        // Card 26.12: CQ capacity
        result["cq_capacity_active"] = telem.cq_capacity_active;
        result["cq_capacity_max"] = telem.cq_capacity_max;

        // Card 26.14: Poll mode (0=UNKNOWN, 1=MAPPED, 2=COPY)
        result["poll_mode"] = telem.poll_mode;

        // Card 26.12: State counters for feed/drain/compute limiter proof
        result["state_idle_empty"] = telem.state_idle_empty;
        result["state_idle_respfull"] = telem.state_idle_respfull;
        result["state_busy"] = telem.state_busy;

        // Card 26.15: Cycle-based metrics (AUTHORITATIVE for GO/NO-GO decisions)
        result["cycles_idle_empty"] = telem.cycles_idle_empty;
        result["cycles_idle_respfull"] = telem.cycles_idle_respfull;
        result["cycles_compute"] = telem.cycles_compute;
        result["cycles_total"] = telem.cycles_total;

        // Card 26.22: Multi-message work amplification telemetry
        result["dbg_units_completed"] = telem.dbg_units_completed;

        // Card 26.34 DEBUG: Keyslot diagnostic fields
        result["dbg_slot_idx_used"] = telem.dbg_slot_idx_used;
        result["dbg_active_epoch_val"] = telem.dbg_active_epoch_val;
        result["dbg_key0_first_u32"] = telem.dbg_key0_first_u32;
        result["dbg_key1_first_u32"] = telem.dbg_key1_first_u32;
        result["dbg_keyslots_ptr"] = telem.dbg_keyslots_ptr;
        result["dbg_key0_addr"] = telem.dbg_key0_addr;

        // DECRYPT-ABI-01: Decrypt telemetry counters
        result["total_decrypt_aes_gcm"] = telem.total_decrypt_aes_gcm;
        result["total_decrypt_chacha20"] = telem.total_decrypt_chacha20;
        result["total_auth_failures"] = telem.total_auth_failures;
        result["plaintext_written_when_auth_fail"] = telem.plaintext_written_when_auth_fail;

        return result;
    }, py::arg("handle"),
       "Get FAST PATH telemetry (Card 25.1, 26.1, 26.12, 26.14, 26.15, 26.22)");

    m.def("fast_path_get_ring_status", [](int64_t handle) -> py::tuple {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        uint32_t head = 0, tail = 0, pending = 0;
        smoke::engine::fast_path_get_ring_status(fp, &head, &tail, &pending);

        return py::make_tuple(head, tail);
    }, py::arg("handle"),
       "Get FAST PATH ring buffer head/tail positions (LEGACY: shard 0 only)");

    // Card 26.2: Get aggregate status across ALL shards
    m.def("fast_path_get_ring_status_all", [](int64_t handle) -> py::dict {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        uint64_t total_head = 0, total_tail = 0, total_pending = 0;
        uint32_t max_pending_shard = 0;
        smoke::engine::fast_path_get_ring_status_all(fp, &total_head, &total_tail,
                                                      &total_pending, &max_pending_shard);

        py::dict result;
        result["total_head"] = total_head;
        result["total_tail"] = total_tail;
        result["total_pending"] = total_pending;
        result["max_pending_shard"] = max_pending_shard;
        return result;
    }, py::arg("handle"),
       "Get aggregate ring status across ALL shards (Card 26.2)");

    m.def("fast_path_get_response_status_all", [](int64_t handle) -> py::dict {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        uint64_t total_write = 0, total_read = 0, total_pending = 0;
        smoke::engine::fast_path_get_response_status_all(fp, &total_write, &total_read, &total_pending);

        py::dict result;
        result["total_write_idx"] = total_write;
        result["total_read_idx"] = total_read;
        result["total_pending"] = total_pending;
        return result;
    }, py::arg("handle"),
       "Get aggregate response buffer status across ALL shards (Card 26.2)");

    m.def("fast_path_debug_pending_shards", [](int64_t handle) -> py::list {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        uint32_t count = 0;
        uint32_t shard_ids[16] = {};
        uint32_t pending_counts[16] = {};
        smoke::engine::fast_path_debug_pending_shards(fp, &count, shard_ids, pending_counts);

        py::list result;
        for (uint32_t i = 0; i < count; i++) {
            py::dict item;
            item["shard"] = shard_ids[i];
            item["pending"] = pending_counts[i];
            result.append(item);
        }
        return result;
    }, py::arg("handle"),
       "Debug: Get list of shards with pending ring work (Card 26.2)");

    // Card 26.2: Stop kernel but keep resources allocated (for telemetry reading)
    m.def("fast_path_stop", [](int64_t handle) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_stop(fp);
    }, py::arg("handle"),
       "Stop FAST PATH kernel but keep resources (allows telemetry read after shutdown)");

    // Card 26.1: Set comb_table_loaded flag in telemetry
    m.def("fast_path_set_comb_table_loaded", [](int64_t handle, bool loaded) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_comb_table_loaded(fp, loaded ? 1 : 0);
    }, py::arg("handle"), py::arg("loaded"),
       "Set comb_table_loaded flag in FAST PATH telemetry (Card 26.1)");

    // Card 26.71: Load RSA-2048 CRT key into fast path RSA keyslot
    m.def("fast_path_load_rsa_key", [](int64_t handle, uint32_t slot_idx,
                                        py::bytes p_bytes, py::bytes q_bytes,
                                        py::bytes dp_bytes, py::bytes dq_bytes,
                                        py::bytes qinv_bytes,
                                        py::bytes R2_p_bytes, py::bytes R2_q_bytes,
                                        uint32_t p_prime, uint32_t q_prime) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        if (!fp) {
            throw std::runtime_error("Invalid handle");
        }

        // Validate all byte strings are 128 bytes (1024 bits)
        std::string p = p_bytes;
        std::string q = q_bytes;
        std::string dp = dp_bytes;
        std::string dq = dq_bytes;
        std::string qinv = qinv_bytes;
        std::string R2_p = R2_p_bytes;
        std::string R2_q = R2_q_bytes;

        if (p.size() != 128 || q.size() != 128 || dp.size() != 128 ||
            dq.size() != 128 || qinv.size() != 128 ||
            R2_p.size() != 128 || R2_q.size() != 128) {
            throw std::invalid_argument("All key components must be 128 bytes (1024 bits)");
        }

        return smoke::engine::fast_path_load_rsa_key(
            fp, slot_idx,
            reinterpret_cast<const uint8_t*>(p.data()),
            reinterpret_cast<const uint8_t*>(q.data()),
            reinterpret_cast<const uint8_t*>(dp.data()),
            reinterpret_cast<const uint8_t*>(dq.data()),
            reinterpret_cast<const uint8_t*>(qinv.data()),
            reinterpret_cast<const uint8_t*>(R2_p.data()),
            reinterpret_cast<const uint8_t*>(R2_q.data()),
            p_prime, q_prime
        );
    }, py::arg("handle"), py::arg("slot_idx"),
       py::arg("p"), py::arg("q"), py::arg("dp"), py::arg("dq"), py::arg("qinv"),
       py::arg("R2_p"), py::arg("R2_q"), py::arg("p_prime"), py::arg("q_prime"),
       "Load RSA-2048 CRT key into fast path RSA keyslot (Card 26.71). All components are 128-byte big-endian.");

    // Card 26.71: Submit RSA-2048 CRT sign request
    m.def("fast_path_submit_rsa2048", [](int64_t handle, py::bytes message_bytes,
                                          uint8_t key_slot, uint8_t client_id) -> uint32_t {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        if (!fp) {
            throw std::runtime_error("Invalid handle");
        }

        std::string message = message_bytes;
        if (message.size() != 256) {
            throw std::runtime_error("Message must be exactly 256 bytes");
        }

        return smoke::engine::fast_path_submit_rsa2048(
            fp,
            reinterpret_cast<const uint8_t*>(message.data()),
            key_slot,
            client_id
        );
    }, py::arg("handle"), py::arg("message"),
       py::arg("key_slot") = 0, py::arg("client_id") = 0,
       "Submit RSA-2048 CRT sign request (Card 26.71). Message must be 256-byte PKCS#1 padded.");

    // Expose legacy engine handle for FAST PATH setup
    m.def("engine_create", []() -> int64_t {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }
        return reinterpret_cast<int64_t>(g_engine->persistent_engine);
    }, "Get legacy engine handle for FAST PATH creation");

    // Card 25.1: Shut down legacy kernel (keeps memory allocated)
    // This allows FAST PATH to run without competing with legacy kernel
    m.def("engine_shutdown", [](int64_t handle) -> bool {
        auto* engine_handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle);
        if (!engine_handle) {
            return false;
        }
        return smoke::engine::engine_shutdown(engine_handle);
    }, py::arg("handle"),
       "Shut down legacy persistent kernel (keeps memory for FAST PATH)");

    // Card 26: Warp-coop mode APIs
#if P256_WARP_COOP_ENABLED
    m.def("load_p256_comb_table", [](py::array_t<uint32_t> x_table, py::array_t<uint32_t> y_table) -> bool {
        // Validate array shapes
        auto x_buf = x_table.request();
        auto y_buf = y_table.request();

        if (x_buf.ndim != 1 || y_buf.ndim != 1) {
            throw std::runtime_error("Tables must be 1D arrays");
        }
        if (x_buf.size != y_buf.size) {
            throw std::runtime_error("X and Y tables must have same size");
        }
        if (x_buf.size != 8 * 255) {
            throw std::runtime_error("Tables must have 8*255 = 2040 elements (w=8 comb table)");
        }

        cudaError_t err = smoke::engine::load_p256_comb_table(
            static_cast<const uint32_t*>(x_buf.ptr),
            static_cast<const uint32_t*>(y_buf.ptr),
            x_buf.size
        );
        if (err != cudaSuccess) {
            return false;
        }
        // SPEED-H2-01 GAP 2a (2026-07-22): record the loaded-state on the
        // ABI engine handle so the pybind fast_path_start gate above and the
        // C ABI gates (smoke_fast_path_start, smoke_engine_load_key_p256)
        // see it. Mirrors smoke_engine_load_p256_comb_table, including the
        // fast-path telemetry flag.
        if (g_engine) {
            g_engine->p256_comb_loaded = true;
            if (g_engine->fast_path) {
                smoke::engine::fast_path_set_comb_table_loaded(g_engine->fast_path, 1);
            }
        }
        return true;
    }, py::arg("x_table"), py::arg("y_table"),
       "Load P-256 comb table for warp-coop mode. Tables must be flat uint32 arrays of 1016 elements (Card 26).");

    m.def("is_warp_coop_enabled", []() -> bool { return true; },
       "Returns True if warp-coop mode is compiled in (Card 26).");

    // GPU epoch Merkle aggregation (smoke-merkle-v0). Input = n concatenated
    // 32-byte leaves; output = 32-byte root. Byte-for-byte parity with the trust
    // merkle_root_v0 oracle (the Python layer fail-closed cross-checks below the
    // GPU/CPU crossover). This is the embarrassingly-parallel GPU win: ~500M
    // leaves/s vs ~660K/s/CPU-core.
    m.def("merkle_root_v0_gpu", [](py::buffer leaves_flat, uint32_t n) -> py::bytes {
        // Zero-copy: read the caller buffer (bytes/bytearray/memoryview/numpy)
        // in place. The old py::bytes -> std::string round-trip copied the
        // ENTIRE leaf array on the CPU (512MB at 16M leaves) before the H2D
        // transfer — that copy, not the kernel, was the merkle product-path
        // bottleneck (card D). The kernel does 510M/1402M leaves/s; this makes
        // the pybind entry stop throwing away that speed.
        py::buffer_info info = leaves_flat.request();
        if (info.ndim != 1 || info.itemsize != 1)
            throw std::runtime_error("leaves_flat must be a flat byte buffer");
        if (static_cast<size_t>(info.size) != static_cast<size_t>(n) * 32)
            throw std::runtime_error("leaves_flat must be exactly n*32 bytes");
        if (n == 0)
            throw std::runtime_error("refusing merkle root over zero leaves (fail-closed)");
        uint8_t root[32];
        int rc = smoke_merkle_root_v0_gpu(static_cast<const uint8_t*>(info.ptr), n, root);
        if (rc != 0)
            throw std::runtime_error("merkle_root_v0_gpu failed: cuda rc=" + std::to_string(rc));
        return py::bytes(reinterpret_cast<const char*>(root), 32);
    }, py::arg("leaves_flat"), py::arg("n"),
       "GPU smoke-merkle-v0 root over n concatenated 32-byte leaves -> 32B root (zero-copy).");

#if P256_FASTPATH_FULLWINDOW
    // SPEED-H2-01: Python mirror of the full-window loader (the real Python
    // bindings live in THIS file, not engine_bindings.cpp — build_engine.py
    // compiles smoke_engine_abi.cpp).
    //
    // GAP 2b (2026-07-22): this loader previously took the header-stripped
    // payload and did a RAW device upload with no integrity check — bypassing
    // the C1-C5 fail-closed stack the C ABI enforces for Go/FFI callers. It
    // now takes the ENTIRE .bin blob (32-byte typed header + 256 KB payload)
    // and routes through smoke_engine_load_p256_fullwindow_table: typed
    // header validation, whole-blob SHA-256 vs the compiled golden constant,
    // device-side sentinel readback (GAP 4), and the same loaded-state
    // (g_engine->p256_fullwindow_loaded) the C ABI sets — which is what the
    // pybind fast_path_start gate (GAP 2a) checks. Throws on any failure:
    // no raw-upload path survives.
    m.def("load_p256_fullwindow_table", [](py::bytes blob) -> bool {
        if (!g_engine) {
            throw std::runtime_error(
                "load_p256_fullwindow_table: engine not initialized — call init() first");
        }
        std::string buf = blob;
        SmokeStatus st = smoke_engine_load_p256_fullwindow_table(
            g_engine,
            reinterpret_cast<const uint8_t*>(buf.data()),
            buf.size(),
            nullptr /* caller_fp_hex: the compiled golden is the trust anchor */);
        if (st != SMOKE_STATUS_OK) {
            throw std::runtime_error(
                "load_p256_fullwindow_table failed fail-closed (status=" +
                std::to_string(static_cast<int>(st)) +
                "): blob rejected (size/header/golden-SHA-256) or device "
                "upload/sentinel-readback failed");
        }
        return true;
    }, py::arg("blob"),
       "SPEED-H2-01: load the full-window fixed-base table from the ENTIRE .bin blob "
       "(32-byte typed header + payload). Validated against the compiled golden "
       "SHA-256 with device-side sentinel readback; throws fail-closed on any failure.");
    m.def("is_fullwindow_enabled", []() -> bool { return true; },
       "Returns True if the full-window fixed-base path is compiled in (SPEED-H2-01).");
#else
    m.def("is_fullwindow_enabled", []() -> bool { return false; },
       "Returns True if the full-window fixed-base path is compiled in (SPEED-H2-01).");
#endif
#else
    m.def("is_warp_coop_enabled", []() -> bool { return false; },
       "Returns True if warp-coop mode is compiled in (Card 26).");
    m.def("is_fullwindow_enabled", []() -> bool { return false; },
       "Returns True if the full-window fixed-base path is compiled in (SPEED-H2-01).");
#endif

    // Card 26.11: Parity print functions for regression tracking
    m.def("get_build_info", []() -> py::dict {
        py::dict info;

        // Compile-time flags
#if P256_WARP_COOP_ENABLED
        info["P256_WARP_COOP_ENABLED"] = true;
#else
        info["P256_WARP_COOP_ENABLED"] = false;
#endif

#if ENGINE_USE_REAL_P256
        info["ENGINE_USE_REAL_P256"] = true;
#else
        info["ENGINE_USE_REAL_P256"] = false;
#endif

#if BENCH_LITE_TELEMETRY
        info["BENCH_LITE_TELEMETRY"] = true;
#else
        info["BENCH_LITE_TELEMETRY"] = false;
#endif

#if CARD26_TELEMETRY_ENABLED
        info["CARD26_TELEMETRY_ENABLED"] = true;
#else
        info["CARD26_TELEMETRY_ENABLED"] = false;
#endif

        // Runtime constants
        info["FAST_PATH_BATCH_MAX"] = static_cast<int>(smoke::engine::FAST_PATH_BATCH_MAX);
        info["FAST_PATH_RING_SIZE"] = static_cast<int>(smoke::engine::FAST_PATH_RING_SIZE);
        info["FAST_PATH_RESPONSE_BUFFER"] = static_cast<int>(smoke::engine::FAST_PATH_RESPONSE_BUFFER);

        return info;
    }, "Returns build configuration for regression tracking (Card 26.11).");

    // Card 26.13: Native C++ benchmark submitter (bypass Python/GIL overhead)
    // Card 26.21: Extended with opcode + input_len for multi-op benchmarks
    // Card 26.23: Extended with multi_msg_n + msg_len for multi-message hash support
    // Card 26.27: Extended with tuning knobs for GO gate (pct_busy_cycles >= 80%)
    m.def("fast_path_run_native_bench", [](int64_t handle, double duration_sec, uint32_t batch_size,
                                           uint16_t opcode, uint16_t input_len,
                                           uint16_t multi_msg_n, uint16_t msg_len,
                                           // Card 26.27: Tuning knobs
                                           uint32_t high_water, uint32_t low_water,
                                           uint32_t poll_burst, uint32_t poll_max,
                                           uint32_t yield_us) -> py::dict {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

        auto result = smoke::engine::fast_path_run_native_bench(
            fp, duration_sec, batch_size, opcode, input_len,
            multi_msg_n, msg_len,
            high_water, low_water, poll_burst, poll_max, yield_us
        );

        py::dict out;
        out["duration_sec"] = result.duration_sec;
        out["ops_submitted"] = result.ops_submitted;
        out["ops_completed"] = result.ops_completed;
        out["throughput_sig_sec"] = result.throughput_sig_sec;
        out["state_idle_empty"] = result.state_idle_empty;
        out["state_idle_respfull"] = result.state_idle_respfull;
        out["state_busy"] = result.state_busy;
        out["pct_busy"] = result.pct_busy;
        // Card 26.15: Cycle-based metrics (AUTHORITATIVE for GO/NO-GO decisions)
        out["cycles_idle_empty"] = result.cycles_idle_empty;
        out["cycles_idle_respfull"] = result.cycles_idle_respfull;
        out["cycles_compute"] = result.cycles_compute;
        out["pct_busy_cycles"] = result.pct_busy_cycles;
        out["accounting_valid"] = result.accounting_valid;
        // Card 26.23: Multi-message metrics
        out["units_completed"] = result.units_completed;
        out["units_per_sec"] = result.units_per_sec;
        out["multi_msg_n"] = result.multi_msg_n;
        return out;
    }, py::arg("handle"), py::arg("duration_sec"), py::arg("batch_size"),
       py::arg("opcode") = 0, py::arg("input_len") = 32,
       py::arg("multi_msg_n") = 0, py::arg("msg_len") = 8,
       // Card 26.27: Tuning knobs with defaults
       py::arg("high_water") = 0, py::arg("low_water") = 0,
       py::arg("poll_burst") = 1, py::arg("poll_max") = 1024, py::arg("yield_us") = 0,
       "Run native C++ benchmark submitter (Card 26.13 + 26.21 + 26.23 + 26.27: multi-op/multi-msg/tuning support)");

    // =========================================================================
    // Native submit-and-drain: eliminate Python from batch hot loop
    // =========================================================================
    m.def("fast_path_submit_and_drain", [](int64_t handle, py::bytes hash_bytes,
                                           uint8_t flags, int32_t chunk_size,
                                           uint16_t opcode, uint16_t input_len) -> py::dict {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::string blob_str = hash_bytes;
        int64_t total_ops = static_cast<int64_t>(blob_str.size()) / 32;
        if (total_ops <= 0) {
            throw std::runtime_error("hash_bytes must be N*32 bytes (one 32-byte hash per op)");
        }

        auto result = smoke::engine::fast_path_submit_and_drain(
            fp,
            reinterpret_cast<const uint8_t*>(blob_str.data()),
            total_ops,
            chunk_size,
            flags,
            opcode,
            input_len
        );

        py::dict out;
        out["submitted"] = result.submitted;
        out["received"]  = result.received;
        out["errors"]    = result.errors;
        out["wall_ms"]   = result.wall_ms;
        out["submit_ms"] = result.submit_ms;
        out["poll_ms"]   = result.poll_ms;
        return out;
    }, py::arg("handle"), py::arg("hash_bytes"), py::arg("flags") = 1,
       py::arg("chunk_size") = 4096, py::arg("opcode") = 0, py::arg("input_len") = 32,
       "Submit a batch of pre-built hashes and drain all responses in C++ (no Python in hot loop). "
       "hash_bytes: N*32 bytes. Returns dict with submitted/received/errors/wall_ms/submit_ms/poll_ms.");

    // ATTEST-NATIVE-01: variable-length hash/HMAC feeder, outputs returned in
    // submit order. Thin adapter over the native core: zero-copy buffers in,
    // GIL released across the whole native loop, ONE contiguous digest buffer
    // out (never per-digest Python objects). See fast_path_submit_and_drain_hashes.
    m.def("fast_path_submit_and_drain_hashes",
        [](int64_t handle, py::buffer payloads_buf, py::buffer offsets_buf,
           uint32_t item_count, uint16_t opcode, uint8_t key_slot) -> py::tuple {
            auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
            if (item_count == 0) throw std::runtime_error("item_count must be > 0");
            py::buffer_info pinfo = payloads_buf.request();
            py::buffer_info oinfo = offsets_buf.request();
            if (pinfo.ndim != 1 || pinfo.itemsize != 1)
                throw std::runtime_error("payloads must be a flat byte buffer");
            if (oinfo.ndim != 1 || oinfo.itemsize != 4 ||
                static_cast<uint32_t>(oinfo.size) != item_count + 1)
                throw std::runtime_error("offsets must be item_count+1 uint32 entries");
            const uint8_t* payloads = static_cast<const uint8_t*>(pinfo.ptr);
            const uint32_t* offsets = static_cast<const uint32_t*>(oinfo.ptr);
            if (offsets[item_count] != static_cast<uint32_t>(pinfo.size))
                throw std::runtime_error("offsets[item_count] must equal len(payloads)");

            std::string digests;
            digests.resize(static_cast<size_t>(item_count) * 32);

            smoke::engine::HashDrainResult res;
            {
                py::gil_scoped_release nogil;  // native loop runs GIL-free
                res = smoke::engine::fast_path_submit_and_drain_hashes(
                    fp, payloads, offsets, item_count, opcode, key_slot,
                    reinterpret_cast<uint8_t*>(&digests[0]));
            }

            py::dict acct;
            acct["submitted"] = res.submitted;
            acct["received"] = res.received;
            acct["errors"] = res.errors;
            acct["completed_count"] = res.completed_count;
            acct["accounting_ok"] = res.accounting_ok;
            acct["wall_ms"] = res.wall_ms;
            acct["submit_ms"] = res.submit_ms;
            acct["poll_ms"] = res.poll_ms;
            return py::make_tuple(py::bytes(digests), acct);
        }, py::arg("handle"), py::arg("payloads"), py::arg("offsets"),
           py::arg("item_count"), py::arg("opcode"), py::arg("key_slot") = 0,
           "ATTEST-NATIVE-01 native varlen hash/HMAC feeder. payloads=concatenated "
           "bytes, offsets=uint32[item_count+1] CSR. Returns (digests bytes "
           "item_count*32 in submit order, accounting dict). GIL released across "
           "the native loop; fail-closed (accounting_ok=0 on any error).");

    // =========================================================================
    // C4.6: Native ABI Continuous Feeder
    // Production-shape ring driver with target-depth management, source modes,
    // queue telemetry, and bottleneck classification. No Python in hot path.
    // =========================================================================
    m.def("fast_path_run_continuous_feeder",
        [](int64_t handle,
           double duration_sec,
           uint32_t target_depth,
           uint32_t refill_threshold,
           uint32_t batch_cap,
           uint32_t poll_max,
           uint16_t opcode,
           uint16_t input_len,
           uint8_t flags,
           int source_mode,
           uint32_t source_seed,
           uint32_t poll_burst) -> py::dict {
            auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);

            auto result = smoke::engine::fast_path_run_continuous_feeder(
                fp, duration_sec, target_depth, refill_threshold,
                batch_cap, poll_max, opcode, input_len, flags,
                source_mode, source_seed, poll_burst
            );

            py::dict out;
            out["duration_sec"] = result.duration_sec;
            out["ops_submitted"] = result.ops_submitted;
            out["ops_completed"] = result.ops_completed;
            out["ops_errors"] = result.ops_errors;
            out["throughput_sig_sec"] = result.throughput_sig_sec;
            out["accounting_valid"] = result.accounting_valid;
            out["submit_calls"] = result.submit_calls;
            out["drain_calls"] = result.drain_calls;
            out["submit_time_us"] = result.submit_time_us;
            out["drain_time_us"] = result.drain_time_us;
            out["target_depth"] = result.target_depth;
            out["refill_threshold"] = result.refill_threshold;
            out["batch_cap"] = result.batch_cap;
            out["avg_queue_depth"] = result.avg_queue_depth;
            out["min_queue_depth"] = result.min_queue_depth;
            out["max_queue_depth"] = result.max_queue_depth;
            out["avg_batch_size"] = result.avg_batch_size;
            out["cycles_idle_empty"] = result.cycles_idle_empty;
            out["cycles_compute"] = result.cycles_compute;
            out["cycles_idle_respfull"] = result.cycles_idle_respfull;
            out["pct_busy_cycles"] = result.pct_busy_cycles;
            out["starvation_pct"] = result.starvation_pct;
            out["backpressure_pct"] = result.backpressure_pct;
            out["bottleneck_class"] = result.bottleneck_class;
            // C4.6.19: Research sprint instrumentation
            out["ring_full_count"] = result.ring_full_count;
            out["inline_drain_calls"] = result.inline_drain_calls;
            out["inline_drain_items"] = result.inline_drain_items;
            out["outer_poll_calls"] = result.outer_poll_calls;
            out["outer_poll_items"] = result.outer_poll_items;
            out["quiesce_items"] = result.quiesce_items;
            out["time_submit_ns"] = result.time_submit_ns;
            out["time_inline_drain_ns"] = result.time_inline_drain_ns;
            out["time_outer_poll_ns"] = result.time_outer_poll_ns;
            out["quiesce_time_ns"] = result.quiesce_time_ns;
            return out;
        },
        py::arg("handle"),
        py::arg("duration_sec"),
        py::arg("target_depth") = 300000,
        py::arg("refill_threshold") = 150000,
        py::arg("batch_cap") = 4096,
        py::arg("poll_max") = 4096,
        py::arg("opcode") = 0,
        py::arg("input_len") = 32,
        py::arg("flags") = 0x01,
        py::arg("source_mode") = 0,
        py::arg("source_seed") = 42,
        py::arg("poll_burst") = 0,
        "C4.6.7: Run continuous native feeder with target-depth management and poll burst. "
        "source_mode: 0=fixed, 1=random, 2=bursty, 3=sparse. poll_burst: 0=auto(4). "
        "Returns dict with throughput, accounting, queue stats, cycle breakdown, bottleneck class.");

    // =========================================================================
    // Card 26.79: Generic Submit/Poll ABI
    // =========================================================================
    // Unified interface to submit/poll any opcode through FAST PATH.
    // Returns backend tag to prove GPU execution.

    m.attr("GENERIC_ABI_VERSION") = static_cast<int>(SMOKE_GENERIC_ABI_VERSION);

    // Backend constants (Card 26.79)
    m.attr("BACKEND_UNKNOWN") = static_cast<int>(SMOKE_BACKEND_UNKNOWN);
    m.attr("BACKEND_GPU_FASTPATH") = static_cast<int>(SMOKE_BACKEND_GPU_FASTPATH);
    m.attr("BACKEND_GPU_LEGACY") = static_cast<int>(SMOKE_BACKEND_GPU_LEGACY);
    m.attr("BACKEND_CPU_FALLBACK") = static_cast<int>(SMOKE_BACKEND_CPU_FALLBACK);

    // Card 26.79: Generic submit - wraps fast_path_submit_batch
    m.def("generic_submit", [](int64_t handle, py::list requests) -> uint32_t {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        if (!fp || requests.size() == 0) return 0;

        // Validate all requests have correct ABI version
        for (size_t i = 0; i < requests.size(); i++) {
            py::dict r = requests[i].cast<py::dict>();
            if (r.contains("abi_version")) {
                uint32_t ver = r["abi_version"].cast<uint32_t>();
                if (ver != SMOKE_GENERIC_ABI_VERSION) {
                    std::fprintf(stderr, "[SMOKE][FAIL-LOUD] generic_submit: "
                                "ABI version mismatch (got %u, expected %u)\n",
                                ver, SMOKE_GENERIC_ABI_VERSION);
                    return 0;
                }
            }
        }

        // Submit each request via fast_path_submit_batch (one at a time for simplicity)
        uint32_t submitted = 0;
        for (size_t i = 0; i < requests.size(); i++) {
            py::dict r = requests[i].cast<py::dict>();

            uint16_t opcode = r["opcode"].cast<uint16_t>();
            uint8_t flags = r.contains("flags") ? r["flags"].cast<uint8_t>() : 0;
            uint8_t key_slot = r.contains("key_slot") ? r["key_slot"].cast<uint8_t>() : 0;
            uint8_t client_id = r.contains("client_id") ? static_cast<uint8_t>(r["client_id"].cast<uint64_t>() & 0xFF) : 0;

            py::bytes input_bytes = r["input"].cast<py::bytes>();
            std::string input_str = input_bytes;
            uint16_t input_len = static_cast<uint16_t>(input_str.size());

            // Pad input to 32 bytes for submit_batch
            uint8_t input_buf[32] = {0};
            size_t copy_len = std::min<size_t>(32, input_str.size());
            memcpy(input_buf, input_str.data(), copy_len);

            // Submit via existing fast_path_submit_batch infrastructure.
            // D5b-keyslot: forward key_slot (was parsed above and dropped).
            uint32_t count = smoke::engine::fast_path_submit_batch(
                fp, input_buf, 1, flags, client_id, opcode, input_len, key_slot);
            if (count > 0) submitted++;
        }
        return submitted;
    }, py::arg("handle"), py::arg("requests"),
       "Submit batch of generic requests (Card 26.79). Each request: {opcode, input, key_slot?, flags?, client_id?}");

    // Card 26.79: Generic poll - wraps fast_path_poll with backend tag
    m.def("generic_poll", [](int64_t handle, uint32_t max_count) -> py::list {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        if (!fp || max_count == 0) return py::list();

        std::vector<smoke::engine::FastPathResponse> responses(max_count);
        uint32_t got = smoke::engine::fast_path_poll(fp, responses.data(), max_count);

        py::list result;
        for (uint32_t i = 0; i < got; i++) {
            py::dict d;
            d["client_id"] = static_cast<uint64_t>(responses[i].client_id);
            d["seq"] = responses[i].seq;
            d["status"] = responses[i].status;
            d["backend"] = static_cast<int>(SMOKE_BACKEND_GPU_FASTPATH);  // Always GPU for FAST PATH
            d["opcode"] = responses[i].opcode;
            d["output_len"] = responses[i].output_len;
            d["t_submit_us"] = responses[i].t_submit_us;
            d["t_dequeue_us"] = responses[i].t_dequeue_us;
            d["t_complete_us"] = responses[i].t_complete_us;
            d["epoch_used"] = responses[i].epoch_used;    // Card 66
            d["slot_used"] = responses[i].slot_used;      // Card 66
            // Output bytes (inline)
            d["output"] = py::bytes(reinterpret_cast<const char*>(responses[i].output),
                                    std::min<size_t>(64, responses[i].output_len));
            result.append(d);
        }
        return result;
    }, py::arg("handle"), py::arg("max_count"),
       "Poll completed responses with backend tag (Card 26.79). Returns: [{client_id, status, backend, opcode, output, ...}]");

    // =========================================================================
    // Card 26.82: True C ABI Generic Submit/Poll (Python wrappers)
    // =========================================================================
    // These use the global g_engine handle, matching the C ABI behavior.

    m.def("c_fast_path_create", []() -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }
        SmokeErrorInfo error = {};
        SmokeStatus status = smoke_fast_path_create(g_engine, &error);
        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error(std::string("fast_path_create failed: ") + error.message);
        }
        return static_cast<int>(status);
    }, "Create FAST PATH via C ABI (Card 26.82)");

    m.def("c_fast_path_start", []() -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }
        SmokeErrorInfo error = {};
        SmokeStatus status = smoke_fast_path_start(g_engine, &error);
        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error(std::string("fast_path_start failed: ") + error.message);
        }
        return static_cast<int>(status);
    }, "Start FAST PATH via C ABI (Card 26.82)");

    m.def("c_fast_path_stop", []() -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }
        return static_cast<int>(smoke_fast_path_stop(g_engine));
    }, "Stop FAST PATH via C ABI (Card 26.82)");

    m.def("c_fast_path_is_running", []() -> bool {
        if (!g_engine) return false;
        return smoke_fast_path_is_running(g_engine) != 0;
    }, "Check if FAST PATH is running via C ABI (Card 26.82)");

    m.def("c_generic_submit", [](py::list requests) -> uint32_t {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }

        // Convert Python requests to C structs
        std::vector<SmokeGenericRequest> c_reqs(requests.size());
        for (size_t i = 0; i < requests.size(); i++) {
            py::dict r = requests[i].cast<py::dict>();
            SmokeGenericRequest* req = &c_reqs[i];

            req->abi_version = SMOKE_GENERIC_ABI_VERSION;
            req->opcode = r["opcode"].cast<uint16_t>();
            req->flags = r.contains("flags") ? r["flags"].cast<uint8_t>() : 0;
            req->key_slot = r.contains("key_slot") ? r["key_slot"].cast<uint8_t>() : 0;
            req->client_id = r.contains("client_id") ? r["client_id"].cast<uint64_t>() : 0;
            // Card 26.85: Support payload slab fields
            req->payload_offset = r.contains("payload_offset") ? r["payload_offset"].cast<uint32_t>() : 0;
            req->payload_len = r.contains("payload_len") ? r["payload_len"].cast<uint32_t>() : 0;
            req->reserved_u16 = 0;
            req->reserved_u32 = 0;

            py::bytes input_bytes = r["input"].cast<py::bytes>();
            std::string input_str = input_bytes;
            req->input_len = static_cast<uint16_t>(std::min<size_t>(32, input_str.size()));
            memset(req->input, 0, 32);
            memcpy(req->input, input_str.data(), req->input_len);
        }

        SmokeErrorInfo error = {};
        SmokeStatus status = smoke_generic_submit(g_engine, c_reqs.data(),
                                                   static_cast<uint32_t>(c_reqs.size()), &error);
        if (status != SMOKE_STATUS_OK && error.internal_code == 0) {
            throw std::runtime_error(std::string("generic_submit failed: ") + error.message);
        }

        // Return number submitted (stored in internal_code)
        return static_cast<uint32_t>(error.internal_code);
    }, py::arg("requests"),
       "Submit requests via C ABI (Card 26.82/26.85). Each request: {opcode, input, key_slot?, flags?, client_id?, payload_offset?, payload_len?}");

    m.def("c_generic_poll", [](uint32_t max_count) -> py::list {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }

        std::vector<SmokeGenericResponse> responses(max_count);
        uint32_t got = smoke_generic_poll(g_engine, responses.data(), max_count);

        py::list result;
        for (uint32_t i = 0; i < got; i++) {
            py::dict d;
            d["client_id"] = responses[i].client_id;
            d["seq"] = responses[i].seq;
            d["status"] = responses[i].status;
            d["backend"] = static_cast<int>(responses[i].backend);
            d["opcode"] = responses[i].opcode;
            d["output_len"] = responses[i].output_len;
            d["output_offset"] = responses[i].output_offset;
            d["output_bytes"] = responses[i].output_bytes;
            d["t_submit_us"] = responses[i].t_submit_us;
            d["t_dequeue_us"] = responses[i].t_dequeue_us;
            d["t_complete_us"] = responses[i].t_complete_us;
            d["output"] = py::bytes(reinterpret_cast<const char*>(responses[i].output),
                                    std::min<size_t>(64, responses[i].output_len));
            result.append(d);
        }
        return result;
    }, py::arg("max_count"),
       "Poll responses via C ABI (Card 26.82). Returns: [{client_id, status, backend, opcode, output, output_offset, output_bytes, ...}]");

    // =========================================================================
    // Card 26.85: Payload/Output Slab Accessors (Python bindings)
    // =========================================================================

    m.def("c_get_payload_slab", []() -> py::tuple {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }
        uint8_t* ptr = nullptr;
        uint32_t size = 0;
        SmokeStatus status = smoke_engine_get_payload_slab(g_engine, &ptr, &size);
        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error("Failed to get payload slab");
        }
        // Return (address, size) - caller can use ctypes or memoryview
        return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
    }, "Get payload slab (address, size) for writing large inputs");

    m.def("c_get_output_slab", []() -> py::tuple {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }
        uint8_t* ptr = nullptr;
        uint32_t size = 0;
        SmokeStatus status = smoke_engine_get_output_slab(g_engine, &ptr, &size);
        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error("Failed to get output slab");
        }
        // Return (address, size) - caller can use ctypes or memoryview
        return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
    }, "Get output slab (address, size) for reading large outputs");

    // Return slabs as numpy arrays for direct Python access
    m.def("c_get_payload_slab_array", []() -> py::array_t<uint8_t> {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }
        uint8_t* ptr = nullptr;
        uint32_t size = 0;
        SmokeStatus status = smoke_engine_get_payload_slab(g_engine, &ptr, &size);
        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error("Failed to get payload slab");
        }
        // Return as numpy array (no copy, shares memory)
        return py::array_t<uint8_t>({static_cast<size_t>(size)}, {1}, ptr, py::none());
    }, "Get payload slab as numpy array (direct memory access)");

    m.def("c_get_output_slab_array", []() -> py::array_t<uint8_t> {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized - call init() first");
        }
        uint8_t* ptr = nullptr;
        uint32_t size = 0;
        SmokeStatus status = smoke_engine_get_output_slab(g_engine, &ptr, &size);
        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error("Failed to get output slab");
        }
        // Return as numpy array (no copy, shares memory)
        return py::array_t<uint8_t>({static_cast<size_t>(size)}, {1}, ptr, py::none());
    }, "Get output slab as numpy array (direct memory access)");

    // =========================================================================
    // Card 26.86: Operation Schema Query (Python bindings)
    // =========================================================================

    m.def("c_get_op_count", []() -> uint32_t {
        return smoke_get_op_count();
    }, "Get number of supported operations");

    m.def("c_get_op_info", [](uint16_t opcode) -> py::dict {
        SmokeOpInfo info = {};
        SmokeStatus status = smoke_get_op_info(opcode, &info);
        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error("Unknown opcode");
        }
        py::dict d;
        d["opcode"] = info.opcode;
        d["name"] = info.name ? info.name : "";
        d["min_input_len"] = info.min_input_len;
        d["max_input_len"] = info.max_input_len;
        d["output_len"] = info.output_len;
        d["requires_key"] = info.requires_key != 0;
        return d;
    }, py::arg("opcode"), "Get operation info for an opcode");

    m.def("c_get_op_info_by_index", [](uint32_t index) -> py::dict {
        SmokeOpInfo info = {};
        SmokeStatus status = smoke_get_op_info_by_index(index, &info);
        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error("Invalid index");
        }
        py::dict d;
        d["opcode"] = info.opcode;
        d["name"] = info.name ? info.name : "";
        d["min_input_len"] = info.min_input_len;
        d["max_input_len"] = info.max_input_len;
        d["output_len"] = info.output_len;
        d["requires_key"] = info.requires_key != 0;
        return d;
    }, py::arg("index"), "Get operation info by index (for enumeration)");

    m.def("c_list_all_ops", []() -> py::list {
        py::list result;
        uint32_t count = smoke_get_op_count();
        for (uint32_t i = 0; i < count; i++) {
            SmokeOpInfo info = {};
            if (smoke_get_op_info_by_index(i, &info) == SMOKE_STATUS_OK) {
                py::dict d;
                d["opcode"] = info.opcode;
                d["name"] = info.name ? info.name : "";
                d["min_input_len"] = info.min_input_len;
                d["max_input_len"] = info.max_input_len;
                d["output_len"] = info.output_len;
                d["requires_key"] = info.requires_key != 0;
                result.append(d);
            }
        }
        return result;
    }, "List all supported operations with their constraints");

    // =========================================================================
    // Card 29: Rotation Probe + Staged Rotation
    // =========================================================================

    m.def("rotation_probe", [](bool include_pubkey_bytes) -> py::dict {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }

        SmokeRotationProbeResult result;
        SmokeStatus status = smoke_engine_rotation_probe(g_engine, &result);
        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error(std::string("rotation_probe failed: ") + result.error_message);
        }

        // Build Python dict for each slot
        py::list slots;
        for (int i = 0; i < 2; ++i) {
            const SmokeSlotProbeResult& sp = result.slots[i];
            py::dict sd;
            sd["slot_index"] = sp.slot_index;
            sd["epoch"] = sp.epoch;
            sd["key_type"] = sp.key_type;
            sd["key_len"] = sp.key_len;
            sd["check_scalar_range"] = sp.check_scalar_range;
            sd["check_mirror_match"] = sp.check_mirror_match;
            sd["check_integrity_tag"] = sp.check_integrity_tag;
            sd["check_no_fault_flags"] = sp.check_no_fault_flags;
            sd["check_version_monotonic"] = sp.check_version_monotonic;
            sd["check_key_len_valid"] = sp.check_key_len_valid;
            sd["check_packing_sanity"] = sp.check_packing_sanity;
            sd["check_on_curve"] = sp.check_on_curve;
            sd["check_not_infinity"] = sp.check_not_infinity;
            sd["d_top_byte"] = sp.d_top_byte;
            sd["pub_x_top_byte"] = sp.pub_x_top_byte;
            sd["fault_flags"] = sp.fault_flags;
            sd["integrity_version"] = sp.integrity_version;
            sd["healthy"] = sp.healthy;

            // If include_pubkey_bytes, read pub_x/pub_y from device
            // (requires re-reading slot data — only for test/audit)
            if (include_pubkey_bytes && g_engine->persistent_engine &&
                sp.key_type == static_cast<uint32_t>(smoke::engine::KeyslotType::KEYSLOT_P256)) {
                smoke::engine::Keyslot raw_slots[2];
                uint32_t ae;
                if (!smoke::engine::engine_rotation_probe(g_engine->persistent_engine, raw_slots, &ae)) {
                    throw std::runtime_error("Failed to retrieve pubkey bytes for on-curve check");
                }
                sd["pub_x"] = py::bytes(reinterpret_cast<const char*>(raw_slots[i].pub_x), 32);
                sd["pub_y"] = py::bytes(reinterpret_cast<const char*>(raw_slots[i].pub_y), 32);
                memset(raw_slots, 0, sizeof(raw_slots));
            }

            slots.append(sd);
        }

        py::dict out;
        out["active_epoch"] = result.active_epoch;
        out["num_slots"] = result.num_slots;
        out["overall_healthy"] = result.overall_healthy;
        out["probe_error"] = result.probe_error;
        out["slots"] = slots;
        out["check_active_slot_loaded"] = result.check_active_slot_loaded;
        out["check_epoch_in_range"] = result.check_epoch_in_range;
        out["check_epochs_distinct"] = result.check_epochs_distinct;
        return out;
    }, py::arg("include_pubkey_bytes") = false,
       "Run rotation probe: diagnostic keyslot inspection (Card 29). "
       "Set include_pubkey_bytes=True to return pub_x/pub_y for on-curve checks.");

    m.def("rotate_key", [](py::bytes private_key) -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        std::string key_str = private_key;
        if (key_str.size() != 32) {
            throw std::runtime_error("private_key must be 32 bytes");
        }
        SmokeStatus status = smoke_engine_rotate_key(
            g_engine,
            reinterpret_cast<const uint8_t*>(key_str.data()),
            32);
        return static_cast<int>(status);
    }, py::arg("private_key"),
       "Rotate key: load into inactive slot and flip epoch (Card 29)");

    m.def("prepare_next_key", [](py::bytes private_key) -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        std::string key_str = private_key;
        if (key_str.size() != 32) {
            throw std::runtime_error("private_key must be 32 bytes");
        }
        SmokeStatus status = smoke_engine_prepare_next_key(
            g_engine,
            reinterpret_cast<const uint8_t*>(key_str.data()),
            32);
        return static_cast<int>(status);
    }, py::arg("private_key"),
       "Stage next key into inactive slot WITHOUT flipping epoch (Card 29)");

    m.def("commit_rotation", []() -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        return static_cast<int>(smoke_engine_commit_rotation(g_engine));
    }, "Flip active epoch to staged key (Card 29)");

    m.def("rollback_rotation", []() -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        return static_cast<int>(smoke_engine_rollback_rotation(g_engine));
    }, "Abandon staged key: zeroize inactive slot, leave epoch unchanged (Card 29)");

    // Card 78: Typed rotation — prepare next key with arbitrary type
    m.def("prepare_next_key_typed", [](uint32_t key_type, py::bytes key_data) -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        std::string key_str = key_data;
        SmokeStatus status = smoke_engine_prepare_next_key_typed(
            g_engine,
            key_type,
            reinterpret_cast<const uint8_t*>(key_str.data()),
            static_cast<uint32_t>(key_str.size()));
        return static_cast<int>(status);
    }, py::arg("key_type"), py::arg("key_data"),
       "Stage next key with specified type into inactive slot (Card 78)");

    m.def("prepare_next_key_typed_async", [](uint32_t key_type, py::bytes key_data) -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        std::string key_str = key_data;
        SmokeStatus status = smoke_engine_prepare_next_key_typed_async(
            g_engine,
            key_type,
            reinterpret_cast<const uint8_t*>(key_str.data()),
            static_cast<uint32_t>(key_str.size()));
        return static_cast<int>(status);
    }, py::arg("key_type"), py::arg("key_data"),
       "Stage next key with specified type, NO sync (Card 78). Fence after batch.");

    // Card 32B: Async rotation (zero-sync batch primitives)
    m.def("prepare_next_key_async", [](py::bytes private_key) -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        std::string key_str = private_key;
        if (key_str.size() != 32) {
            throw std::runtime_error("private_key must be 32 bytes");
        }
        SmokeStatus status = smoke_engine_prepare_next_key_async(
            g_engine,
            reinterpret_cast<const uint8_t*>(key_str.data()),
            32);
        return static_cast<int>(status);
    }, py::arg("private_key"),
       "Stage next key into inactive slot, NO sync (Card 32B). Fence after batch.");

    m.def("commit_rotation_async", []() -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        return static_cast<int>(smoke_engine_commit_rotation_async(g_engine));
    }, "Flip active epoch, NO sync (Card 32B). Fence after batch.");

    m.def("rotation_fence", []() -> int {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        return static_cast<int>(smoke_engine_rotation_fence(g_engine));
    }, "Wait for all enqueued rotation ops to complete (Card 32B).");

    // Card 32C: Query pending fence state
    m.def("rotation_pending_fence", []() -> bool {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        return g_engine->rotation_pending_fence;
    }, "True if async rotation ops are pending fence (Card 32C).");

    // Card 64: Force recovery from stuck fence state
    m.def("rotation_force_recovery", []() -> bool {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        return smoke_engine_rotation_force_recovery(g_engine) == SMOKE_STATUS_OK;
    }, "Force recovery from stuck fence: destroy/recreate comm_stream, re-sync epoch, clear fence (Card 64).");

    // =========================================================================
    // Card 100: ML-DSA Key Loading
    // =========================================================================

    // Load ML-DSA private key into fast path ML-DSA keyslot
    m.def("fast_path_load_mldsa_key", [](int64_t handle, uint32_t slot_idx,
                                          uint8_t mode, py::bytes sk_blob) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        if (!fp) {
            throw std::runtime_error("Invalid handle");
        }
        std::string sk = sk_blob;
        return smoke::engine::fast_path_load_mldsa_key(
            fp, slot_idx, mode,
            reinterpret_cast<const uint8_t*>(sk.data()),
            static_cast<uint32_t>(sk.size()));
    }, py::arg("handle"), py::arg("slot_idx"), py::arg("mode"), py::arg("sk_blob"),
       "Load ML-DSA private key into fast path keyslot. Mode: 0=44, 1=65, 2=87");

    // =========================================================================
    // Card 27.24: ML-DSA Batch Control
    // =========================================================================

    // Flush queued ML-DSA sign requests
    m.def("fast_path_mldsa_flush_batch", [](int64_t handle) -> uint32_t {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        if (!fp) {
            throw std::runtime_error("Invalid handle");
        }
        return smoke::engine::fast_path_mldsa_flush_batch(fp);
    }, py::arg("handle"),
       "Flush queued ML-DSA sign requests (Card 27.24). Returns: number signed.");

    // Card 234: Submit ML-DSA sign request with flags parameter
    // flags bit 0 = MLDSA_FLAG_DETERMINISTIC (deterministic rhoprime)
    m.def("fast_path_submit_mldsa", [](int64_t handle, py::bytes msg_bytes,
                                        uint8_t mode, uint8_t key_slot,
                                        uint8_t client_id, uint8_t flags) -> uint32_t {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        if (!fp) {
            throw std::runtime_error("Invalid handle");
        }
        std::string msg_str = msg_bytes;
        return smoke::engine::fast_path_submit_mldsa(
            fp,
            reinterpret_cast<const uint8_t*>(msg_str.data()),
            static_cast<uint32_t>(msg_str.size()),
            mode,
            key_slot,
            client_id,
            flags
        );
    }, py::arg("handle"), py::arg("msg"), py::arg("mode") = 1,
       py::arg("key_slot") = 0, py::arg("client_id") = 0, py::arg("flags") = 0,
       "Submit ML-DSA sign request (Card 234). flags bit 0 = deterministic mode.");

    // =========================================================================
    // Card 210: Audit Hooks (pybind11 binding)
    // =========================================================================

    // Set audit callback — accepts a Python callable
    // Signature: callback(op_name: str, opcode: int, status: int, key_slot: int,
    //                     input_len: int, output_len: int, duration_us: int, request_id: int)
    m.def("set_audit_callback", [](py::function callback) {
        // Store the Python callback in a static so it persists
        static py::function stored_callback;
        stored_callback = callback;

        // Register a C callback that invokes the Python function
        smoke_engine_set_audit_callback(
            [](const char* op_name, uint16_t opcode, uint8_t status,
               uint8_t key_slot, uint32_t input_len, uint32_t output_len,
               uint64_t duration_us, uint64_t request_id, void* user_data) {
                py::gil_scoped_acquire gil;
                try {
                    auto* cb = static_cast<py::function*>(user_data);
                    (*cb)(op_name, opcode, status, key_slot,
                          input_len, output_len, duration_us, request_id);
                } catch (const py::error_already_set& e) {
                    // Don't propagate Python exceptions from callback
                    fprintf(stderr, "[audit_callback] Python error: %s\n", e.what());
                }
            },
            &stored_callback
        );
    }, py::arg("callback"),
       "Register Python audit callback (Card 210). Key material is NEVER passed.");

    m.def("clear_audit_callback", []() {
        smoke_engine_clear_audit_callback();
    }, "Clear audit callback (Card 210).");

    // =========================================================================
    // Card 230: ML-DSA Verification (pybind11 binding)
    // =========================================================================

    // Verify ML-DSA signature using host-mediated GPU dispatch
    m.def("mldsa_verify", [](uint8_t mode, py::bytes pk_bytes,
                              py::bytes msg_bytes, py::bytes sig_bytes) -> bool {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        std::string pk_str = pk_bytes;
        std::string msg_str = msg_bytes;
        std::string sig_str = sig_bytes;

        SmokeErrorInfo err_info = {};
        SmokeStatus status = smoke_mldsa_verify(
            g_engine,
            mode,
            reinterpret_cast<const uint8_t*>(pk_str.data()),
            static_cast<uint32_t>(pk_str.size()),
            reinterpret_cast<const uint8_t*>(msg_str.data()),
            static_cast<uint32_t>(msg_str.size()),
            reinterpret_cast<const uint8_t*>(sig_str.data()),
            static_cast<uint32_t>(sig_str.size()),
            &err_info
        );

        if (status == SMOKE_STATUS_OK) {
            return true;   // Signature valid
        }
        if (status == SMOKE_STATUS_ERROR_VERIFY_FAILED) {
            return false;  // Signature invalid
        }
        // Any other status is an argument/engine error — raise
        throw std::runtime_error(
            std::string("mldsa_verify failed: status=") + std::to_string(status) +
            " " + err_info.message
        );
    }, py::arg("mode"), py::arg("pk"), py::arg("msg"), py::arg("sig"),
       "Verify ML-DSA signature (Card 230). Returns True if valid, False if invalid. "
       "Raises on argument errors. Mode: 0=ML-DSA-44, 1=ML-DSA-65, 2=ML-DSA-87");

    // Card 231: ML-DSA Keygen
    m.def("mldsa_keygen", [](uint8_t mode, py::bytes rho_bytes,
                              py::bytes rhoprime_bytes) -> py::tuple {
        if (!g_engine) {
            throw std::runtime_error("Engine not initialized");
        }
        std::string rho_str = rho_bytes;
        std::string rhoprime_str = rhoprime_bytes;

        if (rho_str.size() != 32) {
            throw std::runtime_error("rho must be 32 bytes");
        }
        if (rhoprime_str.size() != 32) {
            throw std::runtime_error("rhoprime must be 32 bytes");
        }

        // Allocate output buffers (max simple format sizes for ML-DSA-87)
        // pk: 32 + 8*256*4 = 8224, sk: 32+32 + 7*256*4 + 8*256*4 + 8*256*4 = 23616
        constexpr uint32_t MAX_PK = 8224;
        constexpr uint32_t MAX_SK = 23616;
        std::vector<uint8_t> pk_buf(MAX_PK);
        std::vector<uint8_t> sk_buf(MAX_SK);
        uint32_t pk_len = 0, sk_len = 0;

        SmokeErrorInfo err_info = {};
        SmokeStatus status = smoke_mldsa_keygen(
            g_engine,
            mode,
            reinterpret_cast<const uint8_t*>(rho_str.data()),
            reinterpret_cast<const uint8_t*>(rhoprime_str.data()),
            pk_buf.data(), MAX_PK, &pk_len,
            sk_buf.data(), MAX_SK, &sk_len,
            &err_info
        );

        if (status != SMOKE_STATUS_OK) {
            throw std::runtime_error(
                std::string("mldsa_keygen failed: status=") + std::to_string(status) +
                " " + err_info.message
            );
        }

        return py::make_tuple(
            py::bytes(reinterpret_cast<const char*>(pk_buf.data()), pk_len),
            py::bytes(reinterpret_cast<const char*>(sk_buf.data()), sk_len)
        );
    }, py::arg("mode"), py::arg("rho"), py::arg("rhoprime"),
       "Generate ML-DSA keypair (Card 231). Returns (pk, sk) in simple format. "
       "Mode: 0=ML-DSA-44, 1=ML-DSA-65, 2=ML-DSA-87");

    // =========================================================================
    // Flywheel-25: C++ Externally-Fed Flywheel Scheduler
    // =========================================================================
    // Bulk ingest packed work records, run hysteresis loop in C++,
    // return completions with identity mapping in bulk.

    m.def("fast_path_flywheel_create",
        [](int64_t engine_handle,
           uint32_t target_in_flight,
           uint32_t refill_threshold,
           uint32_t submit_quantum,
           uint32_t max_submit_rounds,
           uint32_t max_drain_rounds,
           uint32_t poll_max,
           uint8_t flags) -> int64_t {
            auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(engine_handle);
            auto* fw = smoke::engine::fast_path_flywheel_create(
                fp, target_in_flight, refill_threshold, submit_quantum,
                max_submit_rounds, max_drain_rounds, poll_max, flags
            );
            return reinterpret_cast<int64_t>(fw);
        },
        py::arg("engine_handle"),
        py::arg("target_in_flight") = 4096,
        py::arg("refill_threshold") = 2048,
        py::arg("submit_quantum") = 64,
        py::arg("max_submit_rounds") = 16,
        py::arg("max_drain_rounds") = 8,
        py::arg("poll_max") = 4096,
        py::arg("flags") = 0x01,
        "Flywheel-25: Create C++ flywheel scheduler state.");

    // Card 28: Submit with request_id assignment (identity decoupled)
    m.def("fast_path_flywheel_submit_ids",
        [](int64_t fw_handle, py::bytes payload_blob, uint32_t count) -> uint64_t {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            std::string blob_str(payload_blob);
            if (blob_str.size() != count * 32u) {
                throw std::runtime_error("payload_blob size != count * 32");
            }
            return smoke::engine::fast_path_flywheel_submit_ids(
                fw, reinterpret_cast<const uint8_t*>(blob_str.data()), count
            );
        },
        py::arg("fw_handle"), py::arg("payload_blob"), py::arg("count"),
        "Card 28: Submit payloads, get first request_id. IDs are [first, first+count).");

    m.def("fast_path_flywheel_submit",
        [](int64_t fw_handle, py::bytes packed_blob, uint32_t count) -> uint32_t {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            std::string blob_str(packed_blob);
            // Validate: count * 140 == blob size
            if (blob_str.size() != count * 140u) {
                throw std::runtime_error(
                    "packed_blob size (" + std::to_string(blob_str.size()) +
                    ") != count (" + std::to_string(count) + ") * 140"
                );
            }
            return smoke::engine::fast_path_flywheel_submit(
                fw, reinterpret_cast<const uint8_t*>(blob_str.data()), count
            );
        },
        py::arg("fw_handle"), py::arg("packed_blob"), py::arg("count"),
        "Flywheel-25: Bulk submit packed work records (140 bytes each: payload+identity).");

    m.def("fast_path_flywheel_tick",
        [](int64_t fw_handle, uint32_t max_completions) -> py::list {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            // Use pre-allocated completion buffer
            // Allocate completion buffer locally (FlywheelState is opaque)
            std::vector<smoke::engine::FlywheelCompletionRecord> cbuf(max_completions);
            uint32_t got = smoke::engine::fast_path_flywheel_tick(
                fw, cbuf.data(), max_completions
            );
            py::list result;
            for (uint32_t i = 0; i < got; i++) {
                auto& c = cbuf[i];
                py::dict d;
                d["request_id"] = c.request_id;
                d["success"] = (c.success != 0);
                d["status"] = c.status;
                d["r"] = py::bytes(reinterpret_cast<const char*>(c.r), 32);
                d["s"] = py::bytes(reinterpret_cast<const char*>(c.s), 32);
                result.append(d);
            }
            return result;
        },
        py::arg("fw_handle"), py::arg("max_completions") = 4096,
        "Flywheel-25: Run one hysteresis tick (drain+refill in C++). "
        "Returns list of completion dicts with identity mapping.");

    // Flywheel-25 stats: accessor functions (FlywheelState is opaque)
    // These are defined in engine_host.cpp alongside FlywheelState
    // Flywheel-25: Raw bytes tick � returns packed completion blob
    // Eliminates per-item Python dict creation in the hot path.
    // Each completion: 176 bytes (work_id[36] + actor_id[36] + audit_ref[36] +
    //   success[1] + status[1] + r[32] + s[32] + pad[2])
    m.def("fast_path_flywheel_tick_raw",
        [](int64_t fw_handle, uint32_t max_completions) -> py::tuple {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            std::vector<smoke::engine::FlywheelCompletionRecord> cbuf(max_completions);
            uint32_t got = smoke::engine::fast_path_flywheel_tick(
                fw, cbuf.data(), max_completions
            );
            if (got == 0) {
                return py::make_tuple(0, py::bytes("", 0));
            }
            // Return as raw bytes blob � ONE py::bytes allocation for all completions
            return py::make_tuple(
                got,
                py::bytes(reinterpret_cast<const char*>(cbuf.data()),
                          got * sizeof(smoke::engine::FlywheelCompletionRecord))
            );
        },
        py::arg("fw_handle"), py::arg("max_completions") = 4096,
        "Flywheel-25: Raw bytes tick � returns (count, packed_blob) instead of list of dicts.");

        // Flywheel-25: Continuous run � tick in tight C++ loop until target completions
    m.def("fast_path_flywheel_run",
        [](int64_t fw_handle, uint32_t target_completions, double timeout_sec,
           uint32_t max_completions) -> py::tuple {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            std::vector<smoke::engine::FlywheelCompletionRecord> cbuf(max_completions);
            uint32_t got = smoke::engine::fast_path_flywheel_run(
                fw, cbuf.data(), max_completions, target_completions, timeout_sec
            );
            if (got == 0) {
                return py::make_tuple(0, py::bytes("", 0));
            }
            return py::make_tuple(
                got,
                py::bytes(reinterpret_cast<const char*>(cbuf.data()),
                          got * sizeof(smoke::engine::FlywheelCompletionRecord))
            );
        },
        py::arg("fw_handle"), py::arg("target_completions"),
        py::arg("timeout_sec") = 60.0, py::arg("max_completions") = 100000,
        "Flywheel-25: Run C++ tick loop until target completions or timeout. "
        "Returns (count, packed_completions_blob).");

    // Flywheel-25 Phase 3: Raw mode � payloads only, integer IDs
    m.def("fast_path_flywheel_submit_raw",
        [](int64_t fw_handle, py::bytes payload_blob, uint32_t count) -> uint32_t {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            std::string blob_str(payload_blob);
            if (blob_str.size() != count * 32u) {
                throw std::runtime_error(
                    "payload_blob size (" + std::to_string(blob_str.size()) +
                    ") != count (" + std::to_string(count) + ") * 32"
                );
            }
            return smoke::engine::fast_path_flywheel_submit_raw(
                fw, reinterpret_cast<const uint8_t*>(blob_str.data()), count
            );
        },
        py::arg("fw_handle"), py::arg("payload_blob"), py::arg("count"),
        "Flywheel-25: Raw bulk submit � payloads only (32 bytes each), no identity strings.");

    m.def("fast_path_flywheel_run_raw",
        [](int64_t fw_handle, py::bytes payload_blob, uint32_t count,
           double timeout_sec) -> py::dict {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            std::string blob_str(payload_blob);
            if (blob_str.size() != count * 32u) {
                throw std::runtime_error(
                    "payload_blob size (" + std::to_string(blob_str.size()) +
                    ") != count (" + std::to_string(count) + ") * 32"
                );
            }
            auto t0 = std::chrono::steady_clock::now();
            uint32_t completed = smoke::engine::fast_path_flywheel_run_raw(
                fw, reinterpret_cast<const uint8_t*>(blob_str.data()), count, timeout_sec
            );
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            py::dict out;
            out["completed"] = completed;
            out["elapsed_sec"] = elapsed;
            out["throughput_sig_sec"] = (elapsed > 0) ? completed / elapsed : 0.0;
            out["accounting_valid"] = (completed == count);
            return out;
        },
        py::arg("fw_handle"), py::arg("payload_blob"), py::arg("count"),
        py::arg("timeout_sec") = 60.0,
        "Flywheel-25: Raw run � submit payloads + run C++ loop until complete. "
        "Returns throughput dict. No per-item Python objects.");

    // Continuous scheduler: NativeFeeder-style loop, sole poll owner
    m.def("fast_path_flywheel_run_continuous",
        [](int64_t fw_handle, double timeout_sec) -> py::dict {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            auto result = smoke::engine::fast_path_flywheel_run_continuous(fw, timeout_sec);
            py::dict out;
            out["completed"] = result.completed;
            out["submitted"] = result.submitted;
            out["errors"] = result.errors;
            out["elapsed_sec"] = result.elapsed_sec;
            out["ring_full_count"] = result.ring_full_count;
            out["avg_inflight"] = result.avg_inflight;
            out["throughput_sig_sec"] = (result.elapsed_sec > 0)
                ? result.completed / result.elapsed_sec : 0.0;
            out["accounting_valid"] = (result.completed == result.submitted && result.errors == 0);
            return out;
        },
        py::arg("fw_handle"), py::arg("timeout_sec") = 30.0,
        "Accelerated continuous scheduler: NativeFeeder-style pressure loop with "
        "hysteresis. Pauses dispatcher, becomes sole poll owner, generates work "
        "internally, returns throughput dict.");

    m.def("fast_path_flywheel_stats",
        [](int64_t fw_handle) -> py::dict {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            py::dict out;
            // Use extern accessor functions since FlywheelState is opaque here
            out["tick_count"] = smoke::engine::fast_path_flywheel_get_tick_count(fw);
            out["total_submitted"] = smoke::engine::fast_path_flywheel_get_submitted(fw);
            out["total_completed"] = smoke::engine::fast_path_flywheel_get_completed(fw);
            out["total_errors"] = smoke::engine::fast_path_flywheel_get_errors(fw);
            out["submit_calls"] = smoke::engine::fast_path_flywheel_get_submit_calls(fw);
            out["drain_calls"] = smoke::engine::fast_path_flywheel_get_drain_calls(fw);
            out["refill_events"] = smoke::engine::fast_path_flywheel_get_refill_events(fw);
            out["ring_full_count"] = smoke::engine::fast_path_flywheel_get_ring_full(fw);
            out["in_flight_count"] = smoke::engine::fast_path_flywheel_get_in_flight(fw);
            out["pending_count"] = smoke::engine::fast_path_flywheel_get_pending(fw);
            return out;
        },
        py::arg("fw_handle"),
        "Flywheel-25: Get flywheel scheduler statistics.");

    m.def("fast_path_flywheel_destroy",
        [](int64_t fw_handle) {
            auto* fw = reinterpret_cast<smoke::engine::FlywheelState*>(fw_handle);
            smoke::engine::fast_path_flywheel_destroy(fw);
        },
        py::arg("fw_handle"),
        "Flywheel-25: Destroy flywheel scheduler state.");
}

#endif // SMOKE_ABI_NO_PYTHON
