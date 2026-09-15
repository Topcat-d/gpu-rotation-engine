#pragma once

#include <stdint.h>

namespace smoke {
namespace engine {

// Queue size (must be power of 2 for efficient modulo)
static constexpr uint32_t ENGINE_QUEUE_SIZE = 1024;

// Card 09: Microbatch scheduling - process multiple requests per loop iteration
// Card 10: Made adaptive based on queue depth
static constexpr uint32_t ENGINE_MICROBATCH_MIN = 1;
static constexpr uint32_t ENGINE_MICROBATCH_MAX = 128;  // C4.6.14: raised from 64 (shared mem ~41KB, fits 48KB/SM)

// Card 10: Depth thresholds for adaptive microbatch sizing
// Queue depth <= LOW_WATER  -> use MICROBATCH_MIN (low latency)
// Queue depth >= HIGH_WATER -> use MICROBATCH_MAX (high throughput)
// In between: linear interpolation
static constexpr uint32_t ENGINE_DEPTH_LOW_WATER  = 4;   // <= 4 requests -> tiny batches
static constexpr uint32_t ENGINE_DEPTH_HIGH_WATER = 96;  // C4.6.14: scaled with MICROBATCH_MAX (was 48 for max=64)

// Card 10: Enable debug counters for microbatch telemetry
// Set to 0 for release builds to avoid overhead
#define ENGINE_MICROBATCH_DEBUG 1

// Maximum keyslots for double-buffered rotation
static constexpr uint32_t ENGINE_MAX_KEY_SLOTS = 2;

// Shutdown flags
static constexpr uint8_t ENGINE_SHUTDOWN_FALSE = 0;
static constexpr uint8_t ENGINE_SHUTDOWN_TRUE  = 1;

// Response status codes for engine responses
enum EngineResponseStatus : uint8_t {
    ENGINE_RESP_EMPTY           = 0,
    ENGINE_RESP_OK              = 1,
    ENGINE_RESP_ERROR           = 2,
    ENGINE_RESP_INTEGRITY_ERROR = 3,  // Card 18: Keyslot integrity check failed
};

} // namespace engine
} // namespace smoke
