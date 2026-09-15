/**
 * Smoke ABI v1 - Async Ring Buffer Interface
 *
 * Low-latency interface for P-256 signing via shared memory.
 * Bridge runs in native thread - NO Python in critical path.
 *
 * Interprocess Safety:
 *   - C11 atomics with acquire/release semantics
 *   - Naturally aligned indices (4-byte)
 *   - Power-of-2 ring size for mask math
 *
 * Usage:
 *   Server: smoke_abi_start() spawns native worker thread
 *   Client: smoke_abi_submit() / smoke_abi_read() via shared memory
 *
 * CARD: Card 21/22 - ABI Channel V1
 *
 * ============================================================================
 * ABI FREEZE NOTICE (Card 27.05)
 * ============================================================================
 * The struct layouts in this header are FROZEN as of v1. Changes to struct
 * sizes, field offsets, or field types are BREAKING CHANGES that require:
 *   1. Incrementing SMOKE_ABI_VERSION
 *   2. Maintaining backward compatibility or explicit migration path
 *
 * Frozen struct sizes (v1):
 *   - SmokeABIRequest:   64 bytes (cache-aligned)
 *   - SmokeABIResponse:  128 bytes
 *   - SmokeABIHeader:    64 bytes
 *   - SmokeABIRing:      128 bytes (head + tail with cache-line padding)
 *
 * Ring constants frozen:
 *   - SMOKE_ABI_RING_SIZE = 256 (max 255 in-flight)
 *   - SMOKE_ABI_MAX_CLIENTS = 16
 *
 * Struct layouts verified by compile-time assertions in tests/c_abi/test_headers.c
 * ============================================================================
 */

#ifndef SMOKE_ABI_H
#define SMOKE_ABI_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "smoke_engine.h"  /* For SMOKE_API export macro (Card 26.98) */
/* Cross-platform atomic support */
#ifdef __cplusplus
#include <atomic>
#define SMOKE_ATOMIC(T) std::atomic<T>
#define SMOKE_ATOMIC_LOAD(ptr, order) (ptr)->load(order)
#define SMOKE_ATOMIC_STORE(ptr, val, order) (ptr)->store(val, order)
#define SMOKE_ATOMIC_FETCH_ADD(ptr, val, order) (ptr)->fetch_add(val, order)
#define SMOKE_ATOMIC_CAS(ptr, expected, desired, succ, fail) (ptr)->compare_exchange_weak(*(expected), desired, succ, fail)
#define SMOKE_MO_ACQUIRE std::memory_order_acquire
#define SMOKE_MO_RELEASE std::memory_order_release
#define SMOKE_MO_RELAXED std::memory_order_relaxed
extern "C" {
#else
#include <stdatomic.h>
#define SMOKE_ATOMIC(T) _Atomic T
#define SMOKE_ATOMIC_LOAD(ptr, order) atomic_load_explicit(ptr, order)
#define SMOKE_ATOMIC_STORE(ptr, val, order) atomic_store_explicit(ptr, val, order)
#define SMOKE_ATOMIC_FETCH_ADD(ptr, val, order) atomic_fetch_add_explicit(ptr, val, order)
#define SMOKE_ATOMIC_CAS(ptr, expected, desired, succ, fail) atomic_compare_exchange_weak_explicit(ptr, expected, desired, succ, fail)
#define SMOKE_MO_ACQUIRE memory_order_acquire
#define SMOKE_MO_RELEASE memory_order_release
#define SMOKE_MO_RELAXED memory_order_relaxed
#endif

/* Cross-platform alignment */
#ifdef _MSC_VER
#define SMOKE_ALIGN(n) __declspec(align(n))
#else
#define SMOKE_ALIGN(n) __attribute__((aligned(n)))
#endif

/* ============================================================================
 * ABI Constants
 * ============================================================================ */

#define SMOKE_ABI_MAGIC         0x534D4F4B  /* "SMOK" */
#define SMOKE_ABI_VERSION       1
#define SMOKE_ABI_RING_SIZE     256         /* Power of 2, max 255 in-flight */
#define SMOKE_ABI_RING_MASK     (SMOKE_ABI_RING_SIZE - 1)
#define SMOKE_ABI_MICROBATCH    32          /* Max requests per bridge tick */
#define SMOKE_ABI_MAX_CLIENTS   16          /* Card 24: Max concurrent clients */
#define SMOKE_ABI_DRAIN_PER_CLIENT 8        /* Card 24: Max drain per client per scheduler tick */

#ifdef _WIN32
#define SMOKE_ABI_SHM_NAME      "SmokeP256ABI_V1"
#define SMOKE_ABI_MULTI_SHM_NAME "SmokeP256ABI_Multi_V1"
#else
#define SMOKE_ABI_SHM_NAME      "/smoke_p256_abi_v1"
#define SMOKE_ABI_MULTI_SHM_NAME "/smoke_p256_abi_multi_v1"
#endif

/* ============================================================================
 * Operation & Status Codes
 * ============================================================================ */

typedef enum SmokeABIOp {
    SMOKE_ABI_OP_NOP      = 0,
    SMOKE_ABI_OP_SIGN     = 1,
    SMOKE_ABI_OP_SHUTDOWN = 255
} SmokeABIOp;

typedef enum SmokeABIStatus {
    SMOKE_ABI_OK              = 0,
    SMOKE_ABI_PENDING         = 1,
    SMOKE_ABI_ERR_INPUT       = 2,
    SMOKE_ABI_ERR_BUSY        = 3,
    SMOKE_ABI_ERR_NO_KEY      = 4,
    SMOKE_ABI_ERR_INTEGRITY   = 5,
    SMOKE_ABI_ERR_INTERNAL    = 255
} SmokeABIStatus;

/* ============================================================================
 * Request Flags
 * ============================================================================ */

#define SMOKE_ABI_NONCE_RFC6979   0x00
#define SMOKE_ABI_NONCE_RANDOM    0x01
#define SMOKE_ABI_LOW_S           0x04

/* ============================================================================
 * Request / Response (cache-line aligned)
 * ============================================================================ */

typedef struct SMOKE_ALIGN(64) SmokeABIRequest {
    uint64_t    id;             /* Request ID (client-assigned) */
    uint8_t     op;             /* SmokeABIOp */
    uint8_t     flags;          /* Nonce mode, low-S */
    uint8_t     key_slot;       /* Key slot (0-15) */
    uint8_t     _pad[5];
    uint8_t     hash[32];       /* SHA-256 to sign */
    /* Card 24.8: Submit timestamp (client writes) */
    uint64_t    t_submit_us;    /* Timestamp when client submitted (usec) */
    uint8_t     _reserved[8];
} SmokeABIRequest;  /* 64 bytes */

typedef struct SMOKE_ALIGN(64) SmokeABIResponse {
    uint64_t    id;             /* Echo request ID */
    uint8_t     status;         /* SmokeABIStatus */
    uint8_t     fault;          /* Fault flags */
    uint16_t    epoch;          /* Key epoch */
    uint8_t     _pad[4];
    uint8_t     r[32];          /* Signature R (big-endian) */
    uint8_t     s[32];          /* Signature S (big-endian) */
    /* Card 24.8: Timing timestamps (bridge writes) */
    uint64_t    t_submit_us;    /* Echo from request (for client convenience) */
    uint64_t    t_dequeue_us;   /* When bridge picked up request */
    uint64_t    t_complete_us;  /* When bridge wrote response */
    uint8_t     _reserved24_8[24]; /* Pad to 128 bytes total */
} SmokeABIResponse; /* 128 bytes */

/* ============================================================================
 * Ring Indices (atomic, cache-line padded)
 *
 * CRITICAL: These use C11 _Atomic for interprocess safety.
 * - head: written by producer, read by consumer (acquire on read)
 * - tail: written by consumer, read by producer (acquire on read)
 * ============================================================================ */

typedef struct SMOKE_ALIGN(64) SmokeABIRing {
    SMOKE_ATOMIC(uint32_t) head;      /* Producer writes, consumer reads */
    uint32_t _pad1[15];         /* Pad to separate cache line */
    SMOKE_ATOMIC(uint32_t) tail;      /* Consumer writes, producer reads */
    uint32_t _pad2[15];         /* Pad to 128 bytes total */
} SmokeABIRing;

/* ============================================================================
 * Telemetry (atomic counters)
 * ============================================================================ */

typedef struct SMOKE_ALIGN(64) SmokeABITelemetry {
    SMOKE_ATOMIC(uint64_t) requests;      /* Total requests processed */
    SMOKE_ATOMIC(uint64_t) responses;     /* Total responses written */
    SMOKE_ATOMIC(uint64_t) dropped;       /* Requests dropped (ring full) */
    SMOKE_ATOMIC(uint64_t) errors;        /* Processing errors */
    SMOKE_ATOMIC(uint32_t) req_depth;     /* Current request queue depth */
    SMOKE_ATOMIC(uint32_t) resp_depth;    /* Current response queue depth */
    SMOKE_ATOMIC(uint32_t) bridge_running;/* 1 if native bridge thread running */
    SMOKE_ATOMIC(uint32_t) shutdown_flag; /* 1 to signal bridge shutdown */
    /* Card 24.6: Batch statistics */
    SMOKE_ATOMIC(uint64_t) batch_count;   /* Number of batches processed */
    SMOKE_ATOMIC(uint64_t) batch_sum;     /* Sum of batch sizes (for avg) */
    SMOKE_ATOMIC(uint32_t) max_batch;     /* Largest batch ever */
    SMOKE_ATOMIC(uint32_t) _pad24_6;      /* Padding for alignment */
    /* Card 24.8: Per-batch bottleneck timing (cumulative, in microseconds) */
    SMOKE_ATOMIC(uint64_t) batch_drain_us_sum;  /* Time draining requests from rings */
    SMOKE_ATOMIC(uint64_t) batch_gpu_us_sum;    /* Time in GPU batch submit+poll */
    SMOKE_ATOMIC(uint64_t) batch_fanout_us_sum; /* Time writing responses back to rings */
    SMOKE_ATOMIC(uint64_t) batch_poll_us_sum;   /* Time spent polling for completion */
} SmokeABITelemetry;

/* ============================================================================
 * Shared Memory Header
 * ============================================================================ */

typedef struct SMOKE_ALIGN(64) SmokeABIHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t ring_size;
    uint32_t req_size;
    uint32_t resp_size;
    uint32_t req_ring_off;
    uint32_t resp_ring_off;
    uint32_t req_data_off;
    uint32_t resp_data_off;
    uint32_t telemetry_off;
    uint32_t total_size;
    uint32_t _reserved[5];
} SmokeABIHeader;   /* 64 bytes */

/* Computed region size */
#define SMOKE_ABI_REGION_SIZE ( \
    sizeof(SmokeABIHeader) + \
    sizeof(SmokeABIRing) * 2 + \
    sizeof(SmokeABIRequest) * SMOKE_ABI_RING_SIZE + \
    128 * SMOKE_ABI_RING_SIZE + /* Response padded to 128 */ \
    sizeof(SmokeABITelemetry) \
)

/* ============================================================================
 * Ring Buffer Helpers (inline, interprocess-safe)
 * ============================================================================ */

static inline uint32_t smoke_ring_count(const SmokeABIRing* r) {
    uint32_t h = SMOKE_ATOMIC_LOAD(&r->head, SMOKE_MO_ACQUIRE);
    uint32_t t = SMOKE_ATOMIC_LOAD(&r->tail, SMOKE_MO_ACQUIRE);
    return (h - t) & SMOKE_ABI_RING_MASK;
}

static inline int smoke_ring_full(const SmokeABIRing* r) {
    return smoke_ring_count(r) == (SMOKE_ABI_RING_SIZE - 1);
}

static inline int smoke_ring_empty(const SmokeABIRing* r) {
    uint32_t h = SMOKE_ATOMIC_LOAD(&r->head, SMOKE_MO_ACQUIRE);
    uint32_t t = SMOKE_ATOMIC_LOAD(&r->tail, SMOKE_MO_ACQUIRE);
    return h == t;
}

/* ============================================================================
 * Region Init / Validate
 * ============================================================================ */

static inline int smoke_abi_init_region(void* region, uint32_t size) {
    if (!region || size < SMOKE_ABI_REGION_SIZE) return -1;

    memset(region, 0, size);
    SmokeABIHeader* h = (SmokeABIHeader*)region;
    uint32_t off = sizeof(SmokeABIHeader);

    h->magic = SMOKE_ABI_MAGIC;
    h->version = SMOKE_ABI_VERSION;
    h->ring_size = SMOKE_ABI_RING_SIZE;
    h->req_size = sizeof(SmokeABIRequest);
    h->resp_size = 128;  /* Padded response size */

    h->req_ring_off = off; off += sizeof(SmokeABIRing);
    h->resp_ring_off = off; off += sizeof(SmokeABIRing);
    h->req_data_off = off; off += sizeof(SmokeABIRequest) * SMOKE_ABI_RING_SIZE;
    h->resp_data_off = off; off += 128 * SMOKE_ABI_RING_SIZE;
    h->telemetry_off = off; off += sizeof(SmokeABITelemetry);
    h->total_size = off;

    return 0;
}

static inline int smoke_abi_validate(const void* region) {
    if (!region) return -1;
    const SmokeABIHeader* h = (const SmokeABIHeader*)region;
    if (h->magic != SMOKE_ABI_MAGIC) return -1;
    if (h->version != SMOKE_ABI_VERSION) return -1;
    return 0;
}

/* ============================================================================
 * Accessors
 * ============================================================================ */

static inline SmokeABIRing* smoke_abi_req_ring(void* region) {
    SmokeABIHeader* h = (SmokeABIHeader*)region;
    return (SmokeABIRing*)((uint8_t*)region + h->req_ring_off);
}

static inline SmokeABIRing* smoke_abi_resp_ring(void* region) {
    SmokeABIHeader* h = (SmokeABIHeader*)region;
    return (SmokeABIRing*)((uint8_t*)region + h->resp_ring_off);
}

static inline SmokeABIRequest* smoke_abi_req_data(void* region) {
    SmokeABIHeader* h = (SmokeABIHeader*)region;
    return (SmokeABIRequest*)((uint8_t*)region + h->req_data_off);
}

static inline SmokeABIResponse* smoke_abi_resp_at(void* region, uint32_t idx) {
    SmokeABIHeader* h = (SmokeABIHeader*)region;
    return (SmokeABIResponse*)((uint8_t*)region + h->resp_data_off + idx * 128);
}

static inline SmokeABITelemetry* smoke_abi_telemetry(void* region) {
    SmokeABIHeader* h = (SmokeABIHeader*)region;
    return (SmokeABITelemetry*)((uint8_t*)region + h->telemetry_off);
}

/* ============================================================================
 * Client API: Submit & Read (lock-free, interprocess-safe)
 * ============================================================================ */

/**
 * Submit a sign request to the ring buffer.
 * Returns 0 on success, -1 if ring full.
 */
static inline int smoke_abi_submit(void* region, uint64_t id,
                                   const uint8_t hash[32], uint8_t flags, uint8_t key_slot) {
    SmokeABIRing* r = smoke_abi_req_ring(region);

    /* Check space (read tail with acquire) */
    uint32_t head = SMOKE_ATOMIC_LOAD(&r->head, SMOKE_MO_RELAXED);
    uint32_t tail = SMOKE_ATOMIC_LOAD(&r->tail, SMOKE_MO_ACQUIRE);
    if (((head + 1) & SMOKE_ABI_RING_MASK) == tail) {
        return -1;  /* Full */
    }

    /* Write request data */
    SmokeABIRequest* req = &smoke_abi_req_data(region)[head & SMOKE_ABI_RING_MASK];
    req->id = id;
    req->op = SMOKE_ABI_OP_SIGN;
    req->flags = flags;
    req->key_slot = key_slot;
    memcpy(req->hash, hash, 32);

    /* Publish with release (makes request visible to consumer) */
    SMOKE_ATOMIC_STORE(&r->head, (head + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
    return 0;
}

/**
 * Read a response from the ring buffer.
 * Returns 0 on success, -1 if ring empty.
 */
static inline int smoke_abi_read(void* region, SmokeABIResponse* out) {
    SmokeABIRing* r = smoke_abi_resp_ring(region);

    /* Check data available (read head with acquire) */
    uint32_t head = SMOKE_ATOMIC_LOAD(&r->head, SMOKE_MO_ACQUIRE);
    uint32_t tail = SMOKE_ATOMIC_LOAD(&r->tail, SMOKE_MO_RELAXED);
    if (head == tail) {
        return -1;  /* Empty */
    }

    /* Read response data */
    SmokeABIResponse* resp = smoke_abi_resp_at(region, tail & SMOKE_ABI_RING_MASK);
    memcpy(out, resp, sizeof(SmokeABIResponse));

    /* Publish consumption with release */
    SMOKE_ATOMIC_STORE(&r->tail, (tail + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
    return 0;
}

/* ============================================================================
 * Bridge Lifecycle API (implemented in smoke_abi_bridge.cpp)
 *
 * These start/stop the native worker thread that pumps the ring buffer.
 * Python can call these but MUST NOT be the pump itself.
 * ============================================================================ */

/**
 * Start the ABI bridge with a native worker thread.
 *
 * @param engine    SmokeEngineHandle from smoke_engine_init()
 * @param shm_name  Shared memory name (NULL for default)
 * @return          0 on success, -1 on error
 */
SMOKE_API int smoke_abi_start(void* engine, const char* shm_name);

/**
 * Stop the ABI bridge and clean up.
 * Blocks until worker thread exits.
 */
SMOKE_API void smoke_abi_stop(void);

/**
 * Check if bridge is running.
 */
SMOKE_API int smoke_abi_is_running(void);

/**
 * Get pointer to shared memory region (for telemetry inspection).
 */
SMOKE_API void* smoke_abi_get_region(void);

/**
 * Check if bridge has faulted (Card 26.99).
 */
SMOKE_API int smoke_abi_is_faulted(void);

/**
 * Get last error message from bridge (Card 26.99).
 */
SMOKE_API const char* smoke_abi_get_last_error(void);

/* ============================================================================
 * Card 24: Multi-Client ABI Support
 *
 * Each client gets its own SPSC ring pair (request + response).
 * The bridge scheduler drains clients in round-robin fashion.
 * ============================================================================ */

/* Per-client telemetry */
typedef struct SMOKE_ALIGN(64) SmokeABIClientTelemetry {
    SMOKE_ATOMIC(uint64_t) requests;      /* Requests submitted by this client */
    SMOKE_ATOMIC(uint64_t) responses;     /* Responses read by this client */
    SMOKE_ATOMIC(uint64_t) dropped;       /* Requests dropped (ring full) */
    SMOKE_ATOMIC(uint64_t) errors;        /* Processing errors for this client */
    SMOKE_ATOMIC(uint64_t) latency_sum_ns;/* Sum of latencies (for mean calc) */
    SMOKE_ATOMIC(uint32_t) req_depth;     /* Current request queue depth */
    SMOKE_ATOMIC(uint32_t) resp_depth;    /* Current response queue depth */
    SMOKE_ATOMIC(uint32_t) client_active; /* 1 if client connected */
    uint32_t _pad[5];
} SmokeABIClientTelemetry;

/* Multi-client shared memory header */
typedef struct SMOKE_ALIGN(64) SmokeABIMultiHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_clients;           /* Number of client slots configured */
    uint32_t ring_size;
    uint32_t req_size;
    uint32_t resp_size;
    uint32_t global_telemetry_off;  /* Offset to global SmokeABITelemetry */
    uint32_t client_slots_off;      /* Offset to array of client slot offsets */
    uint32_t total_size;
    SMOKE_ATOMIC(uint32_t) bridge_running;/* 1 if bridge thread running */
    SMOKE_ATOMIC(uint32_t) shutdown_flag; /* 1 to signal shutdown */
    uint32_t _reserved[5];
} SmokeABIMultiHeader;   /* 64 bytes */

/* Per-client slot layout (offsets relative to region base) */
typedef struct SmokeABIClientSlot {
    uint32_t req_ring_off;
    uint32_t resp_ring_off;
    uint32_t req_data_off;
    uint32_t resp_data_off;
    uint32_t telemetry_off;
    uint32_t _pad[3];
} SmokeABIClientSlot;

/* Size of one client slot's data */
#define SMOKE_ABI_CLIENT_SLOT_SIZE ( \
    sizeof(SmokeABIRing) * 2 + \
    sizeof(SmokeABIRequest) * SMOKE_ABI_RING_SIZE + \
    128 * SMOKE_ABI_RING_SIZE + \
    sizeof(SmokeABIClientTelemetry) \
)

/* Total multi-client region size */
#define SMOKE_ABI_MULTI_REGION_SIZE(n) ( \
    sizeof(SmokeABIMultiHeader) + \
    sizeof(SmokeABITelemetry) + \
    sizeof(SmokeABIClientSlot) * (n) + \
    SMOKE_ABI_CLIENT_SLOT_SIZE * (n) \
)

/* ============================================================================
 * Multi-Client Region Init / Validate
 * ============================================================================ */

static inline int smoke_abi_multi_init_region(void* region, uint32_t size, uint32_t num_clients) {
    if (!region || num_clients == 0 || num_clients > SMOKE_ABI_MAX_CLIENTS) return -1;
    if (size < SMOKE_ABI_MULTI_REGION_SIZE(num_clients)) return -1;

    memset(region, 0, size);
    SmokeABIMultiHeader* h = (SmokeABIMultiHeader*)region;
    uint32_t off = sizeof(SmokeABIMultiHeader);

    h->magic = SMOKE_ABI_MAGIC;
    h->version = SMOKE_ABI_VERSION;
    h->num_clients = num_clients;
    h->ring_size = SMOKE_ABI_RING_SIZE;
    h->req_size = sizeof(SmokeABIRequest);
    h->resp_size = 128;

    /* Global telemetry */
    h->global_telemetry_off = off;
    off += sizeof(SmokeABITelemetry);

    /* Client slot array */
    h->client_slots_off = off;
    off += sizeof(SmokeABIClientSlot) * num_clients;

    /* Initialize each client slot */
    SmokeABIClientSlot* slots = (SmokeABIClientSlot*)((uint8_t*)region + h->client_slots_off);
    for (uint32_t i = 0; i < num_clients; ++i) {
        slots[i].req_ring_off = off;
        off += sizeof(SmokeABIRing);
        slots[i].resp_ring_off = off;
        off += sizeof(SmokeABIRing);
        slots[i].req_data_off = off;
        off += sizeof(SmokeABIRequest) * SMOKE_ABI_RING_SIZE;
        slots[i].resp_data_off = off;
        off += 128 * SMOKE_ABI_RING_SIZE;
        slots[i].telemetry_off = off;
        off += sizeof(SmokeABIClientTelemetry);
    }

    h->total_size = off;
    return 0;
}

static inline int smoke_abi_multi_validate(const void* region) {
    if (!region) return -1;
    const SmokeABIMultiHeader* h = (const SmokeABIMultiHeader*)region;
    if (h->magic != SMOKE_ABI_MAGIC) return -1;
    if (h->version != SMOKE_ABI_VERSION) return -1;
    if (h->num_clients == 0 || h->num_clients > SMOKE_ABI_MAX_CLIENTS) return -1;
    return 0;
}

/* ============================================================================
 * Multi-Client Accessors
 * ============================================================================ */

static inline SmokeABITelemetry* smoke_abi_multi_global_telemetry(void* region) {
    SmokeABIMultiHeader* h = (SmokeABIMultiHeader*)region;
    return (SmokeABITelemetry*)((uint8_t*)region + h->global_telemetry_off);
}

static inline SmokeABIClientSlot* smoke_abi_multi_client_slots(void* region) {
    SmokeABIMultiHeader* h = (SmokeABIMultiHeader*)region;
    return (SmokeABIClientSlot*)((uint8_t*)region + h->client_slots_off);
}

static inline SmokeABIRing* smoke_abi_multi_req_ring(void* region, uint32_t client_id) {
    SmokeABIClientSlot* slots = smoke_abi_multi_client_slots(region);
    return (SmokeABIRing*)((uint8_t*)region + slots[client_id].req_ring_off);
}

static inline SmokeABIRing* smoke_abi_multi_resp_ring(void* region, uint32_t client_id) {
    SmokeABIClientSlot* slots = smoke_abi_multi_client_slots(region);
    return (SmokeABIRing*)((uint8_t*)region + slots[client_id].resp_ring_off);
}

static inline SmokeABIRequest* smoke_abi_multi_req_data(void* region, uint32_t client_id) {
    SmokeABIClientSlot* slots = smoke_abi_multi_client_slots(region);
    return (SmokeABIRequest*)((uint8_t*)region + slots[client_id].req_data_off);
}

static inline SmokeABIResponse* smoke_abi_multi_resp_at(void* region, uint32_t client_id, uint32_t idx) {
    SmokeABIClientSlot* slots = smoke_abi_multi_client_slots(region);
    return (SmokeABIResponse*)((uint8_t*)region + slots[client_id].resp_data_off + idx * 128);
}

static inline SmokeABIClientTelemetry* smoke_abi_multi_client_telemetry(void* region, uint32_t client_id) {
    SmokeABIClientSlot* slots = smoke_abi_multi_client_slots(region);
    return (SmokeABIClientTelemetry*)((uint8_t*)region + slots[client_id].telemetry_off);
}

static inline uint32_t smoke_abi_multi_num_clients(const void* region) {
    const SmokeABIMultiHeader* h = (const SmokeABIMultiHeader*)region;
    return h->num_clients;
}

/* ============================================================================
 * Multi-Client Submit & Read (per-client SPSC)
 * ============================================================================ */

static inline int smoke_abi_multi_submit(void* region, uint32_t client_id, uint64_t id,
                                          const uint8_t hash[32], uint8_t flags, uint8_t key_slot) {
    SmokeABIRing* r = smoke_abi_multi_req_ring(region, client_id);
    SmokeABIRequest* req_data = smoke_abi_multi_req_data(region, client_id);

    uint32_t head = SMOKE_ATOMIC_LOAD(&r->head, SMOKE_MO_RELAXED);
    uint32_t tail = SMOKE_ATOMIC_LOAD(&r->tail, SMOKE_MO_ACQUIRE);
    if (((head + 1) & SMOKE_ABI_RING_MASK) == tail) {
        return -1;  /* Full */
    }

    SmokeABIRequest* req = &req_data[head & SMOKE_ABI_RING_MASK];
    req->id = id;
    req->op = SMOKE_ABI_OP_SIGN;
    req->flags = flags;
    req->key_slot = key_slot;
    memcpy(req->hash, hash, 32);
    req->t_submit_us = 0;  /* Will be set by caller if timestamp tracking is needed */

    SMOKE_ATOMIC_STORE(&r->head, (head + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
    return 0;
}

/* Card 24.8: Submit with timestamp for latency decomposition */
static inline int smoke_abi_multi_submit_timed(void* region, uint32_t client_id, uint64_t id,
                                                const uint8_t hash[32], uint8_t flags, uint8_t key_slot,
                                                uint64_t t_submit_us) {
    SmokeABIRing* r = smoke_abi_multi_req_ring(region, client_id);
    SmokeABIRequest* req_data = smoke_abi_multi_req_data(region, client_id);

    uint32_t head = SMOKE_ATOMIC_LOAD(&r->head, SMOKE_MO_RELAXED);
    uint32_t tail = SMOKE_ATOMIC_LOAD(&r->tail, SMOKE_MO_ACQUIRE);
    if (((head + 1) & SMOKE_ABI_RING_MASK) == tail) {
        return -1;  /* Full */
    }

    SmokeABIRequest* req = &req_data[head & SMOKE_ABI_RING_MASK];
    req->id = id;
    req->op = SMOKE_ABI_OP_SIGN;
    req->flags = flags;
    req->key_slot = key_slot;
    memcpy(req->hash, hash, 32);
    req->t_submit_us = t_submit_us;  /* Card 24.8: Timestamp for latency decomposition */

    SMOKE_ATOMIC_STORE(&r->head, (head + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
    return 0;
}

static inline int smoke_abi_multi_read(void* region, uint32_t client_id, SmokeABIResponse* out) {
    SmokeABIRing* r = smoke_abi_multi_resp_ring(region, client_id);

    uint32_t head = SMOKE_ATOMIC_LOAD(&r->head, SMOKE_MO_ACQUIRE);
    uint32_t tail = SMOKE_ATOMIC_LOAD(&r->tail, SMOKE_MO_RELAXED);
    if (head == tail) {
        return -1;  /* Empty */
    }

    SmokeABIResponse* resp = smoke_abi_multi_resp_at(region, client_id, tail & SMOKE_ABI_RING_MASK);
    memcpy(out, resp, sizeof(SmokeABIResponse));

    SMOKE_ATOMIC_STORE(&r->tail, (tail + 1) & SMOKE_ABI_RING_MASK, SMOKE_MO_RELEASE);
    return 0;
}

/* ============================================================================
 * Multi-Client Bridge API (implemented in smoke_abi_bridge.cpp)
 * ============================================================================ */

/**
 * Start multi-client ABI bridge with N client slots.
 *
 * @param engine        SmokeEngineHandle
 * @param num_clients   Number of client slots (1-16)
 * @param shm_name      Shared memory name (NULL for default)
 * @return              0 on success, -1 on error
 */
SMOKE_API int smoke_abi_start_multi(void* engine, uint32_t num_clients, const char* shm_name);

/**
 * Get multi-client region pointer.
 */
SMOKE_API void* smoke_abi_get_multi_region(void);

/**
 * Get number of active clients.
 */
SMOKE_API uint32_t smoke_abi_get_num_clients(void);

/**
 * Stop multi-client ABI bridge and clean up.
 */
SMOKE_API void smoke_abi_stop_multi(void);

/**
 * Check if multi-client bridge is running.
 */
SMOKE_API int smoke_abi_multi_is_running(void);

/**
 * Check if multi-client bridge has faulted (Card 26.99).
 */
SMOKE_API int smoke_abi_multi_is_faulted(void);

/**
 * Get last error message from multi-client bridge (Card 26.99).
 */
SMOKE_API const char* smoke_abi_multi_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* SMOKE_ABI_H */
