/**
 * Smoke ABI Bridge - Native Worker Thread Implementation
 *
 * Card 24.4 STRICT: No-Fallback Batch Path + CUDA Thread Ownership
 *
 * The bridge runs in its own thread, completely independent of Python.
 * Python can start/stop the bridge but MUST NOT be the pump.
 *
 * Architecture:
 *   - smoke_abi_start() creates shared memory + spawns worker thread
 *   - Worker thread loops: drain requests → batch submit → batch poll → write responses
 *   - smoke_abi_stop() signals shutdown + joins thread
 *
 * STRICT RULES (Card 24.4):
 *   - Bridge NEVER calls smoke_engine_sign_p256() in steady state
 *   - Uses ONLY batch submit/poll APIs
 *   - If CUDA init fails, start returns hard error (does NOT start)
 *   - Idempotent: start/start OK, stop/stop OK
 *   - On batch failure: transition to FAULTED, stop processing
 *
 * CARD: Card 24.4 - ABI Bridge Strict Mode
 */

#include "../../include/smoke_abi.h"
#include "../../include/smoke_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* Include CUDA runtime for thread initialization */
#include <cuda_runtime.h>

/* ============================================================================
 * Card 24.4: Bridge State Machine
 * ============================================================================ */

enum class BridgeState : uint32_t {
    STOPPED = 0,    /* Not running */
    RUNNING = 1,    /* Active and processing */
    FAULTED = 2     /* Error occurred, not processing */
};

/* Card 24.4: Last error info for diagnostics */
struct BridgeError {
    int32_t code;
    char message[256];
};

/* ============================================================================
 * Global Bridge State (singleton for V1)
 * ============================================================================ */

static struct {
    void*               region;
    uint32_t            region_size;
    SmokeEngineHandle   engine;
    std::thread*        worker;
    std::atomic<BridgeState> state{BridgeState::STOPPED};
    std::atomic<bool>   shutdown_requested{false};
    BridgeError         last_error;
#ifdef _WIN32
    HANDLE              shm_handle;
#else
    int                 shm_fd;
    char                shm_name[256];
#endif
} g_bridge;

/* Card 24: Multi-client bridge state */
static struct {
    void*               region;
    uint32_t            region_size;
    uint32_t            num_clients;
    SmokeEngineHandle   engine;
    std::thread*        worker;
    std::atomic<BridgeState> state{BridgeState::STOPPED};
    std::atomic<bool>   shutdown_requested{false};
    BridgeError         last_error;
    void*               worker_stream;  /* Card 24.4: Thread-local CUDA stream */
#ifdef _WIN32
    HANDLE              shm_handle;
#else
    int                 shm_fd;
    char                shm_name[256];
#endif
} g_multi_bridge;

/* ============================================================================
 * Error handling helpers
 * ============================================================================ */

static void set_bridge_error(BridgeError* err, int32_t code, const char* msg) {
    if (err) {
        err->code = code;
        strncpy(err->message, msg, sizeof(err->message) - 1);
        err->message[sizeof(err->message) - 1] = '\0';
    }
}

/* ============================================================================
 * Platform Shared Memory
 * ============================================================================ */

#ifdef _WIN32

static void* shm_create(const char* name, uint32_t size, HANDLE* out) {
    HANDLE h = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, name);
    if (!h) {
        fprintf(stderr, "[abi] CreateFileMapping failed: %lu\n", GetLastError());
        return NULL;
    }
    void* p = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!p) {
        fprintf(stderr, "[abi] MapViewOfFile failed: %lu\n", GetLastError());
        CloseHandle(h);
        return NULL;
    }
    *out = h;
    return p;
}

static void shm_destroy(void* p, HANDLE h) {
    if (p) UnmapViewOfFile(p);
    if (h) CloseHandle(h);
}

#else

static void* shm_create(const char* name, uint32_t size, int* out) {
    shm_unlink(name);  /* Remove stale */
    int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        perror("[abi] shm_open");
        return NULL;
    }
    if (ftruncate(fd, size) < 0) {
        perror("[abi] ftruncate");
        close(fd);
        shm_unlink(name);
        return NULL;
    }
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        perror("[abi] mmap");
        close(fd);
        shm_unlink(name);
        return NULL;
    }
    *out = fd;
    return p;
}

static void shm_destroy(const char* name, void* p, uint32_t size, int fd) {
    if (p) munmap(p, size);
    if (fd >= 0) close(fd);
    if (name) shm_unlink(name);
}

#endif

/* ============================================================================
 * Card 24.4: Batch API Forward Declarations
 *
 * The bridge uses ONLY these batch APIs - no per-request fallback.
 * ============================================================================ */

extern "C" {
    /* Card 24.4: Per-thread stream support */
    void* smoke_engine_create_thread_stream();
    void smoke_engine_destroy_thread_stream(void* stream);

    /* Card 24.4: Stream-aware batch APIs (used by bridge worker) */
    uint32_t smoke_engine_submit_batch_with_stream(
        SmokeEngineHandle engine,
        const uint8_t* hashes,
        const uint32_t* request_ids,
        uint32_t count,
        void* stream
    );
    uint32_t smoke_engine_poll_batch_with_stream(
        SmokeEngineHandle engine,
        const uint32_t* request_ids,
        void* out,           /* EngineResponse* */
        uint8_t* out_ready,
        uint32_t count,
        void* stream
    );

    uint32_t smoke_engine_next_request_id(SmokeEngineHandle engine);
    bool smoke_engine_key_loaded(SmokeEngineHandle engine);
}

/* Card 24.4: Maximum batch size for sync-collapse */
#define ABI_BATCH_MAX 64

/* Card 24.8: Get current time in microseconds */
static inline uint64_t get_time_us() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

/* Card 24.4: In-flight request tracking */
struct InFlightRequest {
    uint32_t client_id;
    uint64_t abi_req_id;    /* Original ABI request ID */
    uint32_t engine_req_id; /* Engine-assigned request ID */
    uint32_t resp_slot;     /* Response ring slot to write to */
    bool     completed;     /* Already processed */
    /* Card 24.8: Timestamps for latency decomposition */
    uint64_t t_submit_us;   /* When client submitted (copied from request) */
    uint64_t t_dequeue_us;  /* When bridge picked up this request */
};

/* Card 24.5: Per-client response slot tracking for deferred head advance */
struct ClientPendingResp {
    uint32_t start_head;    /* resp_ring head before this batch */
    uint32_t pending_count; /* Number of slots reserved in this batch */
};

/* Card 24.4: EngineResponse struct layout for batch poll
 * Must match smoke::engine::EngineResponse from engine_state.cuh */
struct EngineResponseLayout {
    uint32_t request_id;
    uint8_t  status;      /* 0=empty, 1=ok, 2=error, 3=integrity_error */
    uint8_t  reserved[3];
    uint8_t  sig_r[32];
    uint8_t  sig_s[32];
};

/* ============================================================================
 * Card 24.4: Multi-Client Worker Thread - Strict Batch Mode
 *
 * ALGORITHM (sync-collapse, no fallback):
 * 1. Initialize CUDA on worker thread (hard fail if fails)
 * 2. Drain pending requests from all clients into batch buffer
 * 3. Submit batch to engine (single GPU sync)
 * 4. Poll batch for completions (bounded loop, no per-request API)
 * 5. Fan-out results to client response rings
 * 6. On any failure: transition to FAULTED, stop processing
 *
 * FORBIDDEN: calling smoke_engine_sign_p256() inside the worker.
 * ============================================================================ */

static void multi_bridge_worker_loop() {
    void* region = g_multi_bridge.region;
    SmokeEngineHandle engine = g_multi_bridge.engine;
    uint32_t num_clients = g_multi_bridge.num_clients;

    /* =========================================================
     * Card 24.4: Initialize CUDA on worker thread
     * If this fails, transition to FAULTED immediately.
     * ========================================================= */
    void* worker_stream = smoke_engine_create_thread_stream();
    if (!worker_stream) {
        fprintf(stderr, "[bridge-FATAL] Failed to create worker thread CUDA stream\n");
        set_bridge_error(&g_multi_bridge.last_error, -100, "CUDA stream creation failed on worker thread");
        g_multi_bridge.state.store(BridgeState::FAULTED, std::memory_order_release);
        return;
    }
    g_multi_bridge.worker_stream = worker_stream;
    fprintf(stderr, "[bridge] CUDA stream initialized: %p\n", worker_stream);

    SmokeABIMultiHeader* header = (SmokeABIMultiHeader*)region;
    SmokeABITelemetry* global_telem = smoke_abi_multi_global_telemetry(region);

    /* Mark bridge as running */
    SMOKE_ATOMIC_STORE(&header->bridge_running, 1, SMOKE_MO_RELEASE);
    SMOKE_ATOMIC_STORE(&global_telem->bridge_running, 1, SMOKE_MO_RELEASE);

    uint32_t idle_count = 0;
    const uint32_t IDLE_THRESHOLD = 100;

    /* Batch buffers */
    uint8_t batch_hashes[ABI_BATCH_MAX * 32];
    uint32_t batch_request_ids[ABI_BATCH_MAX];
    InFlightRequest in_flight[ABI_BATCH_MAX];

    /* Response buffers for batch poll */
    EngineResponseLayout poll_responses[ABI_BATCH_MAX];
    uint8_t poll_ready[ABI_BATCH_MAX];

    /* =========================================================
     * Main processing loop
     * ========================================================= */
    while (!g_multi_bridge.shutdown_requested.load(std::memory_order_acquire)) {
        /* Check if we're faulted */
        if (g_multi_bridge.state.load(std::memory_order_acquire) == BridgeState::FAULTED) {
            break;
        }

        /* Check for shutdown signal from shared memory */
        if (SMOKE_ATOMIC_LOAD(&header->shutdown_flag, SMOKE_MO_ACQUIRE)) {
            break;
        }

        uint32_t total_processed = 0;
        uint32_t batch_count = 0;

        /* Card 24.8: Per-batch timing for bottleneck analysis */
        uint64_t t_batch_start = get_time_us();
        uint64_t t_drain_end = 0;
        uint64_t t_gpu_end = 0;
        uint64_t t_fanout_end = 0;

        /* Card 24.5: Track pending response slots per client (deferred head advance) */
        ClientPendingResp pending_resp[SMOKE_ABI_MAX_CLIENTS];
        for (uint32_t c = 0; c < num_clients; ++c) {
            SmokeABIRing* resp_ring = smoke_abi_multi_resp_ring(region, c);
            pending_resp[c].start_head = SMOKE_ATOMIC_LOAD(&resp_ring->head, SMOKE_MO_RELAXED);
            pending_resp[c].pending_count = 0;
        }

        /* =========================================================
         * PHASE 1: Drain pending requests from all clients
         * Card 24.5: Don't advance resp_ring->head here - defer until responses written
         * ========================================================= */
        for (uint32_t client_id = 0; client_id < num_clients && batch_count < ABI_BATCH_MAX; ++client_id) {
            SmokeABIRing* req_ring = smoke_abi_multi_req_ring(region, client_id);
            SmokeABIRing* resp_ring = smoke_abi_multi_resp_ring(region, client_id);
            SmokeABIRequest* req_data = smoke_abi_multi_req_data(region, client_id);
            SmokeABIClientTelemetry* client_telem = smoke_abi_multi_client_telemetry(region, client_id);

            /* Drain up to DRAIN_PER_CLIENT requests from this client */
            for (uint32_t i = 0; i < SMOKE_ABI_DRAIN_PER_CLIENT && batch_count < ABI_BATCH_MAX; ++i) {
                /* Check request available */
                uint32_t req_head = SMOKE_ATOMIC_LOAD(&req_ring->head, SMOKE_MO_ACQUIRE);
                uint32_t req_tail = SMOKE_ATOMIC_LOAD(&req_ring->tail, SMOKE_MO_RELAXED);
                if (req_head == req_tail) break;

                /* Check response space (use local pending count for accurate space check) */
                uint32_t resp_head_local = (pending_resp[client_id].start_head + pending_resp[client_id].pending_count) & SMOKE_ABI_RING_MASK;
                uint32_t resp_tail = SMOKE_ATOMIC_LOAD(&resp_ring->tail, SMOKE_MO_ACQUIRE);
                if (((resp_head_local + 1) & SMOKE_ABI_RING_MASK) == resp_tail) break;

                /* Read request */
                SmokeABIRequest* req = &req_data[req_tail & SMOKE_ABI_RING_MASK];
                uint64_t abi_req_id = req->id;
                uint8_t op = req->op;

                /* Consume request from ring */
                SMOKE_ATOMIC_STORE(&req_ring->tail, (req_tail + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);

                /* Handle non-sign ops immediately (no GPU needed) */
                if (op == SMOKE_ABI_OP_SHUTDOWN) {
                    SmokeABIResponse* resp = smoke_abi_multi_resp_at(region, client_id, resp_head_local);
                    resp->id = abi_req_id;
                    resp->status = SMOKE_ABI_OK;
                    resp->fault = 0;
                    resp->epoch = 0;
                    memset(resp->r, 0, 32);
                    memset(resp->s, 0, 32);
                    /* Advance head immediately for non-batched responses */
                    SMOKE_ATOMIC_STORE(&resp_ring->head, (resp_head_local + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
                    SMOKE_ATOMIC_STORE(&header->shutdown_flag, 1, SMOKE_MO_RELEASE);
                    /* Update local tracking to reflect advanced head */
                    pending_resp[client_id].start_head = (resp_head_local + 1) & SMOKE_ABI_RING_MASK;
                    continue;
                }

                if (op != SMOKE_ABI_OP_SIGN) {
                    /* NOP or unknown - just ack */
                    SmokeABIResponse* resp = smoke_abi_multi_resp_at(region, client_id, resp_head_local);
                    resp->id = abi_req_id;
                    resp->status = SMOKE_ABI_OK;
                    resp->fault = 0;
                    resp->epoch = 0;
                    memset(resp->r, 0, 32);
                    memset(resp->s, 0, 32);
                    /* Advance head immediately for non-batched responses */
                    SMOKE_ATOMIC_STORE(&resp_ring->head, (resp_head_local + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
                    /* Update local tracking to reflect advanced head */
                    pending_resp[client_id].start_head = (resp_head_local + 1) & SMOKE_ABI_RING_MASK;
                    SMOKE_ATOMIC_FETCH_ADD(&client_telem->requests, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&client_telem->responses, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&global_telem->requests, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&global_telem->responses, 1, SMOKE_MO_RELAXED);
                    total_processed++;
                    continue;
                }

                /* SIGN op: add to batch for GPU processing */
                uint32_t engine_req_id = smoke_engine_next_request_id(engine);

                /* Card 24.8: Record dequeue timestamp */
                uint64_t t_dequeue = get_time_us();

                /* Copy hash to batch buffer */
                memcpy(&batch_hashes[batch_count * 32], req->hash, 32);
                batch_request_ids[batch_count] = engine_req_id;

                /* Track in-flight for response fan-out */
                in_flight[batch_count].client_id = client_id;
                in_flight[batch_count].abi_req_id = abi_req_id;
                in_flight[batch_count].engine_req_id = engine_req_id;
                in_flight[batch_count].resp_slot = resp_head_local;  /* Local slot, not yet visible */
                in_flight[batch_count].completed = false;
                /* Card 24.8: Copy timestamps */
                in_flight[batch_count].t_submit_us = req->t_submit_us;
                in_flight[batch_count].t_dequeue_us = t_dequeue;

                /* Card 24.5: Don't advance resp_ring->head here - defer until response written */
                /* Just track that we've reserved a slot locally */
                pending_resp[client_id].pending_count++;

                /* Update per-client telemetry */
                SMOKE_ATOMIC_FETCH_ADD(&client_telem->requests, 1, SMOKE_MO_RELAXED);
                SMOKE_ATOMIC_FETCH_ADD(&global_telem->requests, 1, SMOKE_MO_RELAXED);

                batch_count++;
            }

            /* Update per-client queue depths */
            SMOKE_ATOMIC_STORE(&client_telem->req_depth, smoke_ring_count(req_ring), SMOKE_MO_RELAXED);
            SMOKE_ATOMIC_STORE(&client_telem->resp_depth, smoke_ring_count(resp_ring), SMOKE_MO_RELAXED);
        }

        /* Card 24.8: Mark end of drain phase */
        t_drain_end = get_time_us();

        /* =========================================================
         * PHASE 2: Submit batch to persistent engine (single GPU sync)
         * Card 24.4: Using thread-local stream, no per-request fallback
         * ========================================================= */
        if (batch_count > 0) {
            uint32_t submitted = smoke_engine_submit_batch_with_stream(
                engine,
                batch_hashes,
                batch_request_ids,
                batch_count,
                worker_stream
            );

            if (submitted != batch_count) {
                /* Card 24.4 STRICT: Batch submit partial failure -> FAULTED */
                fprintf(stderr, "[bridge-FAULT] Batch submit failed: %u/%u\n", submitted, batch_count);
                set_bridge_error(&g_multi_bridge.last_error, -101, "Batch submit partial failure");
                g_multi_bridge.state.store(BridgeState::FAULTED, std::memory_order_release);

                /* Mark unsubmitted requests as errors */
                for (uint32_t i = submitted; i < batch_count; ++i) {
                    uint32_t client_id = in_flight[i].client_id;
                    SmokeABIResponse* resp = smoke_abi_multi_resp_at(region, client_id, in_flight[i].resp_slot);
                    SmokeABIClientTelemetry* client_telem = smoke_abi_multi_client_telemetry(region, client_id);

                    resp->id = in_flight[i].abi_req_id;
                    resp->status = SMOKE_ABI_ERR_INTERNAL;
                    resp->fault = 1;
                    resp->epoch = 0;
                    memset(resp->r, 0, 32);
                    memset(resp->s, 0, 32);

                    SMOKE_ATOMIC_FETCH_ADD(&client_telem->errors, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&client_telem->responses, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&global_telem->errors, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&global_telem->responses, 1, SMOKE_MO_RELAXED);
                }
                break;  /* Exit main loop */
            }

            /* =========================================================
             * PHASE 3: Poll for completions (bounded loop)
             * Card 24.5B: Tiered backoff - yield first, then microsecond sleeps
             * ========================================================= */
            const int MAX_POLL_ATTEMPTS = 100000;  /* 5 seconds at ~50µs per attempt */
            const int YIELD_THRESHOLD = 100;       /* Yield for first 100 iterations */
            const int SPIN_THRESHOLD = 1000;       /* Then 10µs sleeps up to this point */
            uint32_t remaining = batch_count;
            int poll_attempts = 0;

            while (remaining > 0 && poll_attempts < MAX_POLL_ATTEMPTS) {
                memset(poll_ready, 0, batch_count);

                smoke_engine_poll_batch_with_stream(
                    engine,
                    batch_request_ids,
                    poll_responses,
                    poll_ready,
                    batch_count,
                    worker_stream
                );

                /* =========================================================
                 * PHASE 4: Fan-out ready responses to clients
                 * ========================================================= */
                for (uint32_t i = 0; i < batch_count; ++i) {
                    if (!poll_ready[i]) continue;
                    if (in_flight[i].completed) continue;  /* Already processed */

                    /* Card 24.8: Record completion timestamp */
                    uint64_t t_complete = get_time_us();

                    uint32_t client_id = in_flight[i].client_id;
                    SmokeABIResponse* resp = smoke_abi_multi_resp_at(region, client_id, in_flight[i].resp_slot);
                    SmokeABIClientTelemetry* client_telem = smoke_abi_multi_client_telemetry(region, client_id);

                    resp->id = in_flight[i].abi_req_id;
                    resp->fault = 0;
                    resp->epoch = 0;

                    /* Card 24.8: Write timestamps to response */
                    resp->t_submit_us = in_flight[i].t_submit_us;
                    resp->t_dequeue_us = in_flight[i].t_dequeue_us;
                    resp->t_complete_us = t_complete;

                    EngineResponseLayout* engine_resp = &poll_responses[i];

                    if (engine_resp->status == 1) {  /* OK */
                        resp->status = SMOKE_ABI_OK;
                        memcpy(resp->r, engine_resp->sig_r, 32);
                        memcpy(resp->s, engine_resp->sig_s, 32);
                    } else {
                        resp->status = SMOKE_ABI_ERR_INTERNAL;
                        memset(resp->r, 0, 32);
                        memset(resp->s, 0, 32);
                        SMOKE_ATOMIC_FETCH_ADD(&client_telem->errors, 1, SMOKE_MO_RELAXED);
                        SMOKE_ATOMIC_FETCH_ADD(&global_telem->errors, 1, SMOKE_MO_RELAXED);
                    }

                    SMOKE_ATOMIC_FETCH_ADD(&client_telem->responses, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&global_telem->responses, 1, SMOKE_MO_RELAXED);
                    total_processed++;

                    /* Mark completed */
                    in_flight[i].completed = true;
                    remaining--;
                }

                if (remaining > 0) {
                    poll_attempts++;
                    /* Card 24.5B: Tiered backoff - use small sleeps to avoid CPU contention
                     * On Windows, yield() may not actually yield, so use microsecond sleeps */
                    if (poll_attempts < YIELD_THRESHOLD) {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    } else if (poll_attempts < SPIN_THRESHOLD) {
                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                }
            }

            if (remaining > 0) {
                /* Card 24.4 STRICT: Timeout -> FAULTED */
                fprintf(stderr, "[bridge-FAULT] Poll timeout: %u/%u requests not completed\n",
                        remaining, batch_count);
                set_bridge_error(&g_multi_bridge.last_error, -102, "Poll timeout - requests not completed");
                g_multi_bridge.state.store(BridgeState::FAULTED, std::memory_order_release);

                /* Mark remaining as errors */
                for (uint32_t i = 0; i < batch_count; ++i) {
                    if (in_flight[i].completed) continue;

                    uint32_t client_id = in_flight[i].client_id;
                    SmokeABIResponse* resp = smoke_abi_multi_resp_at(region, client_id, in_flight[i].resp_slot);
                    SmokeABIClientTelemetry* client_telem = smoke_abi_multi_client_telemetry(region, client_id);

                    resp->id = in_flight[i].abi_req_id;
                    resp->status = SMOKE_ABI_ERR_INTERNAL;
                    resp->fault = 1;
                    resp->epoch = 0;
                    memset(resp->r, 0, 32);
                    memset(resp->s, 0, 32);

                    SMOKE_ATOMIC_FETCH_ADD(&client_telem->errors, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&client_telem->responses, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&global_telem->errors, 1, SMOKE_MO_RELAXED);
                    SMOKE_ATOMIC_FETCH_ADD(&global_telem->responses, 1, SMOKE_MO_RELAXED);
                }
                break;  /* Exit main loop */
            }

            /* Card 24.8: Mark end of GPU+poll phase (all responses collected) */
            t_gpu_end = get_time_us();

            /* Card 24.5: Advance resp_ring->head for all clients after responses written */
            /* This makes the responses visible to clients atomically after data is ready */
            for (uint32_t c = 0; c < num_clients; ++c) {
                if (pending_resp[c].pending_count > 0) {
                    SmokeABIRing* resp_ring = smoke_abi_multi_resp_ring(region, c);
                    uint32_t new_head = (pending_resp[c].start_head + pending_resp[c].pending_count) & SMOKE_ABI_RING_MASK;
                    SMOKE_ATOMIC_STORE(&resp_ring->head, new_head, SMOKE_MO_RELEASE);
                }
            }

            /* Card 24.8: Mark end of fan-out phase */
            t_fanout_end = get_time_us();

            /* Card 24.6: Update batch statistics */
            SMOKE_ATOMIC_FETCH_ADD(&global_telem->batch_count, 1, SMOKE_MO_RELAXED);
            SMOKE_ATOMIC_FETCH_ADD(&global_telem->batch_sum, batch_count, SMOKE_MO_RELAXED);
            uint32_t prev_max = SMOKE_ATOMIC_LOAD(&global_telem->max_batch, SMOKE_MO_RELAXED);
            while (batch_count > prev_max) {
                if (SMOKE_ATOMIC_CAS(&global_telem->max_batch, &prev_max, batch_count, SMOKE_MO_RELAXED, SMOKE_MO_RELAXED)) {
                    break;
                }
            }

            /* Card 24.8: Update per-batch timing telemetry */
            if (t_drain_end > t_batch_start) {
                SMOKE_ATOMIC_FETCH_ADD(&global_telem->batch_drain_us_sum, t_drain_end - t_batch_start, SMOKE_MO_RELAXED);
            }
            if (t_gpu_end > t_drain_end) {
                SMOKE_ATOMIC_FETCH_ADD(&global_telem->batch_gpu_us_sum, t_gpu_end - t_drain_end, SMOKE_MO_RELAXED);
            }
            if (t_fanout_end > t_gpu_end) {
                SMOKE_ATOMIC_FETCH_ADD(&global_telem->batch_fanout_us_sum, t_fanout_end - t_gpu_end, SMOKE_MO_RELAXED);
            }
        }

        /* Adaptive sleep: if idle, don't burn CPU */
        if (total_processed == 0) {
            idle_count++;
            if (idle_count > IDLE_THRESHOLD) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        } else {
            idle_count = 0;
        }
    }

    /* Mark bridge as stopped */
    SMOKE_ATOMIC_STORE(&header->bridge_running, 0, SMOKE_MO_RELEASE);
    SMOKE_ATOMIC_STORE(&global_telem->bridge_running, 0, SMOKE_MO_RELEASE);

    /* Cleanup thread stream */
    if (worker_stream) {
        smoke_engine_destroy_thread_stream(worker_stream);
        g_multi_bridge.worker_stream = nullptr;
        fprintf(stderr, "[bridge] CUDA stream destroyed\n");
    }
}

/* ============================================================================
 * Single-client Worker Thread - Strict Batch Mode
 * (Same strict rules as multi-client)
 * ============================================================================ */

static void bridge_worker_loop() {
    void* region = g_bridge.region;
    SmokeEngineHandle engine = g_bridge.engine;

    /* Card 24.4: Initialize CUDA on worker thread */
    void* worker_stream = smoke_engine_create_thread_stream();
    if (!worker_stream) {
        fprintf(stderr, "[bridge-FATAL] Failed to create worker thread CUDA stream\n");
        set_bridge_error(&g_bridge.last_error, -100, "CUDA stream creation failed");
        g_bridge.state.store(BridgeState::FAULTED, std::memory_order_release);
        return;
    }

    SmokeABIRing* req_ring = smoke_abi_req_ring(region);
    SmokeABIRing* resp_ring = smoke_abi_resp_ring(region);
    SmokeABIRequest* req_data = smoke_abi_req_data(region);
    SmokeABITelemetry* telem = smoke_abi_telemetry(region);

    /* Mark bridge as running */
    SMOKE_ATOMIC_STORE(&telem->bridge_running, 1, SMOKE_MO_RELEASE);

    uint32_t idle_count = 0;
    const uint32_t IDLE_THRESHOLD = 100;

    /* Batch buffers */
    uint8_t batch_hashes[ABI_BATCH_MAX * 32];
    uint32_t batch_request_ids[ABI_BATCH_MAX];
    uint32_t batch_resp_slots[ABI_BATCH_MAX];
    uint64_t batch_abi_ids[ABI_BATCH_MAX];
    bool batch_completed[ABI_BATCH_MAX];
    EngineResponseLayout poll_responses[ABI_BATCH_MAX];
    uint8_t poll_ready[ABI_BATCH_MAX];

    while (!g_bridge.shutdown_requested.load(std::memory_order_acquire)) {
        if (g_bridge.state.load(std::memory_order_acquire) == BridgeState::FAULTED) {
            break;
        }

        if (SMOKE_ATOMIC_LOAD(&telem->shutdown_flag, SMOKE_MO_ACQUIRE)) {
            break;
        }

        uint32_t batch_count = 0;

        /* Drain up to ABI_BATCH_MAX requests */
        while (batch_count < ABI_BATCH_MAX) {
            uint32_t req_head = SMOKE_ATOMIC_LOAD(&req_ring->head, SMOKE_MO_ACQUIRE);
            uint32_t req_tail = SMOKE_ATOMIC_LOAD(&req_ring->tail, SMOKE_MO_RELAXED);
            if (req_head == req_tail) break;

            uint32_t resp_head = SMOKE_ATOMIC_LOAD(&resp_ring->head, SMOKE_MO_RELAXED);
            uint32_t resp_tail = SMOKE_ATOMIC_LOAD(&resp_ring->tail, SMOKE_MO_ACQUIRE);
            if (((resp_head + 1) & SMOKE_ABI_RING_MASK) == resp_tail) break;

            SmokeABIRequest* req = &req_data[req_tail & SMOKE_ABI_RING_MASK];
            uint64_t req_id = req->id;
            uint8_t op = req->op;

            SMOKE_ATOMIC_STORE(&req_ring->tail, (req_tail + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);

            SmokeABIResponse* resp = smoke_abi_resp_at(region, resp_head & SMOKE_ABI_RING_MASK);

            if (op == SMOKE_ABI_OP_SHUTDOWN) {
                resp->id = req_id;
                resp->status = SMOKE_ABI_OK;
                resp->fault = 0;
                resp->epoch = 0;
                memset(resp->r, 0, 32);
                memset(resp->s, 0, 32);
                SMOKE_ATOMIC_STORE(&resp_ring->head, (resp_head + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
                SMOKE_ATOMIC_STORE(&telem->shutdown_flag, 1, SMOKE_MO_RELEASE);
                continue;
            }

            if (op != SMOKE_ABI_OP_SIGN) {
                resp->id = req_id;
                resp->status = SMOKE_ABI_OK;
                resp->fault = 0;
                resp->epoch = 0;
                memset(resp->r, 0, 32);
                memset(resp->s, 0, 32);
                SMOKE_ATOMIC_STORE(&resp_ring->head, (resp_head + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
                SMOKE_ATOMIC_FETCH_ADD(&telem->requests, 1, SMOKE_MO_RELAXED);
                SMOKE_ATOMIC_FETCH_ADD(&telem->responses, 1, SMOKE_MO_RELAXED);
                continue;
            }

            /* SIGN: add to batch */
            uint32_t engine_req_id = smoke_engine_next_request_id(engine);
            memcpy(&batch_hashes[batch_count * 32], req->hash, 32);
            batch_request_ids[batch_count] = engine_req_id;
            batch_resp_slots[batch_count] = resp_head & SMOKE_ABI_RING_MASK;
            batch_abi_ids[batch_count] = req_id;
            batch_completed[batch_count] = false;

            SMOKE_ATOMIC_STORE(&resp_ring->head, (resp_head + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
            SMOKE_ATOMIC_FETCH_ADD(&telem->requests, 1, SMOKE_MO_RELAXED);
            batch_count++;
        }

        if (batch_count > 0) {
            /* Submit batch */
            uint32_t submitted = smoke_engine_submit_batch_with_stream(
                engine, batch_hashes, batch_request_ids, batch_count, worker_stream);

            if (submitted != batch_count) {
                fprintf(stderr, "[bridge-FAULT] Batch submit failed: %u/%u\n", submitted, batch_count);
                g_bridge.state.store(BridgeState::FAULTED, std::memory_order_release);
                break;
            }

            /* Poll for completions
             * Card 24.5B: Tiered backoff - yield first, then microsecond sleeps */
            uint32_t remaining = batch_count;
            int poll_attempts = 0;
            const int MAX_POLL = 100000;       /* 5 seconds at ~50µs per attempt */
            const int YIELD_THRESHOLD = 100;   /* Yield for first 100 iterations */
            const int SPIN_THRESHOLD = 1000;   /* Then 10µs sleeps up to this point */

            while (remaining > 0 && poll_attempts < MAX_POLL) {
                memset(poll_ready, 0, batch_count);
                smoke_engine_poll_batch_with_stream(
                    engine, batch_request_ids, poll_responses, poll_ready, batch_count, worker_stream);

                for (uint32_t i = 0; i < batch_count; ++i) {
                    if (!poll_ready[i] || batch_completed[i]) continue;

                    SmokeABIResponse* resp = smoke_abi_resp_at(region, batch_resp_slots[i]);
                    resp->id = batch_abi_ids[i];
                    resp->fault = 0;
                    resp->epoch = 0;

                    if (poll_responses[i].status == 1) {
                        resp->status = SMOKE_ABI_OK;
                        memcpy(resp->r, poll_responses[i].sig_r, 32);
                        memcpy(resp->s, poll_responses[i].sig_s, 32);
                    } else {
                        resp->status = SMOKE_ABI_ERR_INTERNAL;
                        memset(resp->r, 0, 32);
                        memset(resp->s, 0, 32);
                        SMOKE_ATOMIC_FETCH_ADD(&telem->errors, 1, SMOKE_MO_RELAXED);
                    }

                    SMOKE_ATOMIC_FETCH_ADD(&telem->responses, 1, SMOKE_MO_RELAXED);
                    batch_completed[i] = true;
                    remaining--;
                }

                if (remaining > 0) {
                    poll_attempts++;
                    /* Card 24.5B: Tiered backoff - use small sleeps to avoid CPU contention */
                    if (poll_attempts < YIELD_THRESHOLD) {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    } else if (poll_attempts < SPIN_THRESHOLD) {
                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                }
            }

            if (remaining > 0) {
                fprintf(stderr, "[bridge-FAULT] Poll timeout\n");
                g_bridge.state.store(BridgeState::FAULTED, std::memory_order_release);
                break;
            }
        }

        /* Adaptive sleep */
        if (batch_count == 0) {
            idle_count++;
            if (idle_count > IDLE_THRESHOLD) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        } else {
            idle_count = 0;
        }

        SMOKE_ATOMIC_STORE(&telem->req_depth, smoke_ring_count(req_ring), SMOKE_MO_RELAXED);
        SMOKE_ATOMIC_STORE(&telem->resp_depth, smoke_ring_count(resp_ring), SMOKE_MO_RELAXED);
    }

    SMOKE_ATOMIC_STORE(&telem->bridge_running, 0, SMOKE_MO_RELEASE);

    if (worker_stream) {
        smoke_engine_destroy_thread_stream(worker_stream);
    }
}

/* ============================================================================
 * Public API - Card 24.4: Idempotent + Strict Error Handling
 * ============================================================================ */

extern "C" {

int smoke_abi_start(void* engine, const char* shm_name) {
    BridgeState current = g_bridge.state.load(std::memory_order_acquire);

    /* Card 24.4: Idempotent - if already running, return OK */
    if (current == BridgeState::RUNNING) {
        return 0;
    }

    /* If faulted, must stop first */
    if (current == BridgeState::FAULTED) {
        fprintf(stderr, "[abi] Bridge is faulted, call stop first\n");
        return -3;
    }

    if (!engine) {
        fprintf(stderr, "[abi] engine is NULL\n");
        return -1;
    }

    const char* name = shm_name ? shm_name : SMOKE_ABI_SHM_NAME;
    uint32_t size = SMOKE_ABI_REGION_SIZE;

#ifdef _WIN32
    g_bridge.region = shm_create(name, size, &g_bridge.shm_handle);
#else
    g_bridge.region = shm_create(name, size, &g_bridge.shm_fd);
    strncpy(g_bridge.shm_name, name, sizeof(g_bridge.shm_name) - 1);
#endif

    if (!g_bridge.region) {
        return -1;
    }

    g_bridge.region_size = size;
    g_bridge.engine = (SmokeEngineHandle)engine;

    if (smoke_abi_init_region(g_bridge.region, size) != 0) {
        fprintf(stderr, "[abi] Failed to init region\n");
        smoke_abi_stop();
        return -1;
    }

    /* Start worker thread */
    g_bridge.shutdown_requested.store(false, std::memory_order_release);
    g_bridge.state.store(BridgeState::RUNNING, std::memory_order_release);
    g_bridge.worker = new std::thread(bridge_worker_loop);

    /* Wait briefly to check if worker faulted during init */
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (g_bridge.state.load(std::memory_order_acquire) == BridgeState::FAULTED) {
        fprintf(stderr, "[abi] Worker thread faulted during init: %s\n", g_bridge.last_error.message);
        smoke_abi_stop();
        return -2;  /* Card 24.4: Hard fail on CUDA init error */
    }

    fprintf(stderr, "[abi] Bridge started on '%s'\n", name);
    return 0;
}

void smoke_abi_stop(void) {
    BridgeState current = g_bridge.state.load(std::memory_order_acquire);

    /* Card 24.4: Idempotent - if already stopped, return */
    if (current == BridgeState::STOPPED) {
        return;
    }

    /* Signal shutdown */
    g_bridge.shutdown_requested.store(true, std::memory_order_release);
    g_bridge.state.store(BridgeState::STOPPED, std::memory_order_release);

    if (g_bridge.worker) {
        if (g_bridge.worker->joinable()) {
            g_bridge.worker->join();
        }
        delete g_bridge.worker;
        g_bridge.worker = nullptr;
    }

#ifdef _WIN32
    shm_destroy(g_bridge.region, g_bridge.shm_handle);
    g_bridge.shm_handle = NULL;
#else
    shm_destroy(g_bridge.shm_name, g_bridge.region, g_bridge.region_size, g_bridge.shm_fd);
    g_bridge.shm_fd = -1;
#endif

    g_bridge.region = nullptr;
    g_bridge.engine = nullptr;
    memset(&g_bridge.last_error, 0, sizeof(g_bridge.last_error));

    fprintf(stderr, "[abi] Bridge stopped\n");
}

int smoke_abi_is_running(void) {
    return g_bridge.state.load(std::memory_order_acquire) == BridgeState::RUNNING ? 1 : 0;
}

int smoke_abi_is_faulted(void) {
    return g_bridge.state.load(std::memory_order_acquire) == BridgeState::FAULTED ? 1 : 0;
}

void* smoke_abi_get_region(void) {
    return g_bridge.region;
}

const char* smoke_abi_get_last_error(void) {
    return g_bridge.last_error.message;
}

/* ============================================================================
 * Multi-Client API - Card 24.4: Idempotent + Strict Error Handling
 * ============================================================================ */

int smoke_abi_start_multi(void* engine, uint32_t num_clients, const char* shm_name) {
    BridgeState current = g_multi_bridge.state.load(std::memory_order_acquire);

    /* Card 24.4: Idempotent - if already running, return OK */
    if (current == BridgeState::RUNNING) {
        return 0;
    }

    /* If faulted, must stop first */
    if (current == BridgeState::FAULTED) {
        fprintf(stderr, "[abi-multi] Bridge is faulted, call stop first\n");
        return -3;
    }

    if (!engine) {
        fprintf(stderr, "[abi-multi] engine is NULL\n");
        return -1;
    }

    if (num_clients == 0 || num_clients > SMOKE_ABI_MAX_CLIENTS) {
        fprintf(stderr, "[abi-multi] Invalid num_clients: %u (max %u)\n", num_clients, SMOKE_ABI_MAX_CLIENTS);
        return -1;
    }

    const char* name = shm_name ? shm_name : SMOKE_ABI_MULTI_SHM_NAME;
    uint32_t size = SMOKE_ABI_MULTI_REGION_SIZE(num_clients);

#ifdef _WIN32
    g_multi_bridge.region = shm_create(name, size, &g_multi_bridge.shm_handle);
#else
    g_multi_bridge.region = shm_create(name, size, &g_multi_bridge.shm_fd);
    strncpy(g_multi_bridge.shm_name, name, sizeof(g_multi_bridge.shm_name) - 1);
#endif

    if (!g_multi_bridge.region) {
        return -1;
    }

    g_multi_bridge.region_size = size;
    g_multi_bridge.num_clients = num_clients;
    g_multi_bridge.engine = (SmokeEngineHandle)engine;

    if (smoke_abi_multi_init_region(g_multi_bridge.region, size, num_clients) != 0) {
        fprintf(stderr, "[abi-multi] Failed to init region\n");
#ifdef _WIN32
        shm_destroy(g_multi_bridge.region, g_multi_bridge.shm_handle);
        g_multi_bridge.shm_handle = NULL;
#else
        shm_destroy(g_multi_bridge.shm_name, g_multi_bridge.region, g_multi_bridge.region_size, g_multi_bridge.shm_fd);
        g_multi_bridge.shm_fd = -1;
#endif
        g_multi_bridge.region = nullptr;
        return -1;
    }

    /* Start worker thread */
    g_multi_bridge.shutdown_requested.store(false, std::memory_order_release);
    g_multi_bridge.state.store(BridgeState::RUNNING, std::memory_order_release);
    g_multi_bridge.worker = new std::thread(multi_bridge_worker_loop);

    /* Wait briefly to check if worker faulted during CUDA init */
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (g_multi_bridge.state.load(std::memory_order_acquire) == BridgeState::FAULTED) {
        fprintf(stderr, "[abi-multi] Worker thread faulted during init: %s\n", g_multi_bridge.last_error.message);
        smoke_abi_stop_multi();
        return -2;  /* Card 24.4: Hard fail on CUDA init error */
    }

    fprintf(stderr, "[abi-multi] Bridge started on '%s' with %u clients\n", name, num_clients);
    return 0;
}

void smoke_abi_stop_multi(void) {
    BridgeState current = g_multi_bridge.state.load(std::memory_order_acquire);

    /* Card 24.4: Idempotent - if already stopped, return */
    if (current == BridgeState::STOPPED) {
        return;
    }

    /* Signal shutdown */
    g_multi_bridge.shutdown_requested.store(true, std::memory_order_release);
    g_multi_bridge.state.store(BridgeState::STOPPED, std::memory_order_release);

    if (g_multi_bridge.worker) {
        if (g_multi_bridge.worker->joinable()) {
            g_multi_bridge.worker->join();
        }
        delete g_multi_bridge.worker;
        g_multi_bridge.worker = nullptr;
    }

#ifdef _WIN32
    shm_destroy(g_multi_bridge.region, g_multi_bridge.shm_handle);
    g_multi_bridge.shm_handle = NULL;
#else
    shm_destroy(g_multi_bridge.shm_name, g_multi_bridge.region, g_multi_bridge.region_size, g_multi_bridge.shm_fd);
    g_multi_bridge.shm_fd = -1;
#endif

    g_multi_bridge.region = nullptr;
    g_multi_bridge.engine = nullptr;
    g_multi_bridge.num_clients = 0;
    g_multi_bridge.worker_stream = nullptr;
    memset(&g_multi_bridge.last_error, 0, sizeof(g_multi_bridge.last_error));

    fprintf(stderr, "[abi-multi] Bridge stopped\n");
}

void* smoke_abi_get_multi_region(void) {
    return g_multi_bridge.region;
}

uint32_t smoke_abi_get_num_clients(void) {
    return g_multi_bridge.num_clients;
}

int smoke_abi_multi_is_running(void) {
    return g_multi_bridge.state.load(std::memory_order_acquire) == BridgeState::RUNNING ? 1 : 0;
}

int smoke_abi_multi_is_faulted(void) {
    return g_multi_bridge.state.load(std::memory_order_acquire) == BridgeState::FAULTED ? 1 : 0;
}

const char* smoke_abi_multi_get_last_error(void) {
    return g_multi_bridge.last_error.message;
}

}  /* extern "C" */
