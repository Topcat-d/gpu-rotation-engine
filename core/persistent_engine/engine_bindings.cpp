// engine_bindings.cpp
//
// Pybind11 bindings for the Persistent P-256 Engine.
//
// CARD 06: Double-buffered keyslots with epoch rotation.
// CARD 08B: Device info helper for benchmark reporting.
// CARD 10: Adaptive microbatch telemetry.
// CARD 11: Engine profiles and tuning.
// CARD 14: Nonce modes and canonical low-s signatures.
// CARD 26: Warp-coop mode support (P256_WARP_COOP_ENABLED)

// Card 26: Warp-coop mode flag (default: disabled)
// Enable with -DP256_WARP_COOP_ENABLED=1 at compile time
#ifndef P256_WARP_COOP_ENABLED
#define P256_WARP_COOP_ENABLED 0
#endif

#include <cstdint>
#include <cstring>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "engine_constants.cuh"
#include "engine_state.cuh"

namespace py = pybind11;

namespace smoke {
namespace engine {

struct EngineHandle;
struct EngineResponse;

EngineHandle* engine_create();
void engine_destroy(EngineHandle* handle);
bool engine_start(EngineHandle* handle);
bool engine_shutdown(EngineHandle* handle);
bool engine_set_key(EngineHandle* handle, const uint8_t private_key[32]);
bool engine_rotate_key(EngineHandle* handle, const uint8_t private_key[32]);
bool engine_submit_request(EngineHandle* handle, const uint8_t hash[32], uint32_t request_id);
bool engine_poll_response(EngineHandle* handle, uint32_t request_id, EngineResponse* out);

// Card 08B: Device info
std::string engine_get_device_name_raw();

// Card 10: Microbatch debug accessor
#if ENGINE_MICROBATCH_DEBUG
bool engine_get_microbatch_debug(EngineHandle* handle, EngineMicrobatchDebug* out_debug);
#endif

// Card 11: Profile and tuning APIs
bool engine_set_profile(EngineHandle* handle, EngineProfile profile);
bool engine_set_tuning(EngineHandle* handle, const EngineTuning& tuning);

// Card 14: Nonce mode and low-s APIs
bool engine_set_nonce_mode(EngineHandle* handle, NonceMode mode);
bool engine_set_low_s_enabled(EngineHandle* handle, bool enabled);

// Card 15: CSPRNG reseed API
bool engine_reseed_rng(EngineHandle* handle, const uint8_t seed[32]);

// Card 16: Rotation policy APIs
bool engine_set_rotation_policy(EngineHandle* handle,
                                RotationPolicy policy,
                                uint32_t auto_rotate_interval_ops,
                                uint32_t auto_rotate_idle_only);
RotationPolicy engine_get_rotation_policy(EngineHandle* handle);

// Card 17: Integrity APIs
// Note: EngineIntegrityStatus and IntegrityMode are already defined in engine_state.cuh
bool engine_set_integrity_tuning(EngineHandle* handle,
                                 IntegrityMode mode,
                                 uint32_t scan_interval_ops,
                                 uint32_t scan_idle_only,
                                 uint32_t max_repair_attempts);
bool engine_get_integrity_status(EngineHandle* handle, struct EngineIntegrityStatus* out_status);
IntegrityMode engine_get_integrity_mode(EngineHandle* handle);

// Card 19: Root seed + self-healing APIs
bool engine_set_root_seed(EngineHandle* handle, const uint8_t seed[32]);
bool engine_clear_root_seed(EngineHandle* handle);
bool engine_set_self_healing_enabled(EngineHandle* handle, bool enabled);
bool engine_is_root_seed_set(EngineHandle* handle);
bool engine_is_self_healing_enabled(EngineHandle* handle);

// Card 20.5: Batch submit/poll APIs for high-throughput path
uint32_t engine_submit_batch(EngineHandle* handle, const uint8_t* hashes, const uint32_t* request_ids, uint32_t count);
uint32_t engine_poll_batch(EngineHandle* handle, const uint32_t* request_ids, EngineResponse* out, uint8_t* out_ready, uint32_t count);

// Card 25.1: FAST PATH APIs
struct FastPathHandle;
FastPathHandle* fast_path_create(EngineHandle* legacy_handle);
void fast_path_destroy(FastPathHandle* handle);
bool fast_path_set_num_ctas(FastPathHandle* handle, uint32_t num_ctas);  // Card 26.1B
bool fast_path_set_num_shards(FastPathHandle* handle, uint32_t num_shards);  // Card 26.2
bool fast_path_set_submit_policy(FastPathHandle* handle, uint32_t policy);  // Card 26.14
bool fast_path_set_shards_per_batch(FastPathHandle* handle, uint32_t k);  // Card 26.14
bool fast_path_start(FastPathHandle* handle);
bool fast_path_submit(FastPathHandle* handle, const uint8_t hash[32], uint8_t flags);
// Card 26.19: Added opcode and input_len for multi-op support
// D5b-keyslot: Added key_slot (shared by the batch; default 0)
uint32_t fast_path_submit_batch(FastPathHandle* handle, const uint8_t* inputs, uint32_t count, uint8_t flags, uint8_t client_id, uint16_t opcode, uint16_t input_len, uint8_t key_slot = 0);
uint32_t fast_path_poll(FastPathHandle* handle, FastPathResponse* out, uint32_t max_count);
bool fast_path_get_telemetry(FastPathHandle* handle, FastPathTelemetry* out);
bool fast_path_set_comb_table_loaded(FastPathHandle* handle, uint32_t loaded);  // Card 26.1
void fast_path_get_ring_status(FastPathHandle* handle, uint32_t* out_head, uint32_t* out_tail, uint32_t* out_pending);
// Card 26.25: Get output slab pointer for extended multi-message outputs
bool fast_path_get_output_slab(FastPathHandle* handle, uint8_t** out_ptr, uint32_t* out_size);

// Card 26: Comb table loading for warp-coop mode (only available when P256_WARP_COOP_ENABLED=1)
#if P256_WARP_COOP_ENABLED
cudaError_t load_p256_comb_table(const uint32_t* h_x_table, const uint32_t* h_y_table, size_t table_size);
#if P256_FASTPATH_FULLWINDOW
cudaError_t load_p256_fullwindow_table(const uint32_t* h_payload, size_t n_u32); // SPEED-H2-01
#endif
#endif

} // namespace engine
} // namespace smoke

namespace {

int64_t py_engine_create() {
    auto* handle = smoke::engine::engine_create();
    return reinterpret_cast<int64_t>(handle);
}

void py_engine_destroy(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    smoke::engine::engine_destroy(handle);
}

bool py_engine_start(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    return smoke::engine::engine_start(handle);
}

bool py_engine_shutdown(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    return smoke::engine::engine_shutdown(handle);
}

bool py_engine_set_key(int64_t handle_int, py::bytes private_key_bytes) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    std::string key_str = private_key_bytes;
    if (key_str.size() != 32) {
        throw std::runtime_error("private_key_bytes must be exactly 32 bytes");
    }
    return smoke::engine::engine_set_key(
        handle,
        reinterpret_cast<const uint8_t*>(key_str.data())
    );
}

bool py_engine_rotate_key(int64_t handle_int, py::bytes private_key_bytes) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    std::string key_str = private_key_bytes;
    if (key_str.size() != 32) {
        throw std::runtime_error("private_key_bytes must be exactly 32 bytes");
    }
    return smoke::engine::engine_rotate_key(
        handle,
        reinterpret_cast<const uint8_t*>(key_str.data())
    );
}

bool py_engine_submit_request(int64_t handle_int, py::bytes hash_bytes, uint32_t request_id) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    std::string hash_str = hash_bytes;
    if (hash_str.size() != 32) {
        throw std::runtime_error("hash_bytes must be exactly 32 bytes");
    }
    return smoke::engine::engine_submit_request(
        handle,
        reinterpret_cast<const uint8_t*>(hash_str.data()),
        request_id
    );
}

py::object py_engine_poll_response(int64_t handle_int, uint32_t request_id) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);

    struct EngineResponsePy {
        uint32_t request_id;
        uint8_t  status;
        uint8_t  reserved[3];
        uint8_t  sig_r[32];
        uint8_t  sig_s[32];
    };

    EngineResponsePy resp{};
    bool ok = smoke::engine::engine_poll_response(
        handle,
        request_id,
        reinterpret_cast<smoke::engine::EngineResponse*>(&resp)
    );

    if (!ok) {
        return py::none();
    }

    py::bytes sig_r(reinterpret_cast<const char*>(resp.sig_r), 32);
    py::bytes sig_s(reinterpret_cast<const char*>(resp.sig_s), 32);

    return py::make_tuple(resp.request_id, resp.status, sig_r, sig_s);
}

} // anonymous namespace

// Card 08B: Device info wrapper
std::string py_engine_get_device_name() {
    try {
        return smoke::engine::engine_get_device_name_raw();
    } catch (const std::exception& e) {
        return std::string("UNKNOWN (") + e.what() + ")";
    }
}

#if ENGINE_MICROBATCH_DEBUG
// Card 10: Get microbatch debug stats wrapper
py::dict py_engine_get_microbatch_debug(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);

    smoke::engine::EngineMicrobatchDebug dbg{};
    bool ok = smoke::engine::engine_get_microbatch_debug(handle, &dbg);

    if (!ok) {
        throw std::runtime_error("Failed to get microbatch debug stats");
    }

    py::dict result;
    result["total_batches"] = dbg.total_batches;
    result["total_requests"] = dbg.total_requests;
    result["small_batches"] = dbg.small_batches;
    result["medium_batches"] = dbg.medium_batches;
    result["large_batches"] = dbg.large_batches;
    return result;
}
#endif

// Card 11: Set engine profile wrapper
bool py_engine_set_profile(int64_t handle_int, int profile_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    auto profile = static_cast<smoke::engine::EngineProfile>(profile_int);
    return smoke::engine::engine_set_profile(handle, profile);
}

// Card 11: Set custom tuning wrapper
bool py_engine_set_tuning(int64_t handle_int,
                          uint32_t microbatch_min,
                          uint32_t microbatch_max,
                          uint32_t depth_low_water,
                          uint32_t depth_high_water) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    smoke::engine::EngineTuning tuning;
    tuning.microbatch_min = microbatch_min;
    tuning.microbatch_max = microbatch_max;
    tuning.depth_low_water = depth_low_water;
    tuning.depth_high_water = depth_high_water;
    // Card 14: Preserve existing nonce_mode and low_s_enabled (use safe defaults)
    tuning.nonce_mode = smoke::engine::NonceMode::DETERMINISTIC;
    tuning.low_s_enabled = 1;
    return smoke::engine::engine_set_tuning(handle, tuning);
}

// Card 14: Set nonce mode wrapper
bool py_engine_set_nonce_mode(int64_t handle_int, int nonce_mode_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    auto mode = static_cast<smoke::engine::NonceMode>(nonce_mode_int);
    return smoke::engine::engine_set_nonce_mode(handle, mode);
}

// Card 14: Set low-s enabled wrapper
bool py_engine_set_low_s_enabled(int64_t handle_int, bool enabled) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    return smoke::engine::engine_set_low_s_enabled(handle, enabled);
}

// Card 15: Reseed CSPRNG wrapper
bool py_engine_reseed_rng(int64_t handle_int, py::bytes seed_bytes) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    std::string seed_str = seed_bytes;
    if (seed_str.size() != 32) {
        throw std::runtime_error("seed_bytes must be exactly 32 bytes");
    }
    return smoke::engine::engine_reseed_rng(
        handle,
        reinterpret_cast<const uint8_t*>(seed_str.data())
    );
}

// Card 16: Set rotation policy wrapper
bool py_engine_set_rotation_policy(int64_t handle_int,
                                   int policy_int,
                                   uint32_t auto_rotate_interval_ops,
                                   uint32_t auto_rotate_idle_only) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    auto policy = static_cast<smoke::engine::RotationPolicy>(policy_int);
    return smoke::engine::engine_set_rotation_policy(
        handle, policy, auto_rotate_interval_ops, auto_rotate_idle_only
    );
}

// Card 16: Get rotation policy wrapper
int py_engine_get_rotation_policy(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    return static_cast<int>(smoke::engine::engine_get_rotation_policy(handle));
}

// Card 17: Set integrity tuning wrapper
bool py_engine_set_integrity_tuning(int64_t handle_int,
                                    int mode_int,
                                    uint32_t scan_interval_ops,
                                    uint32_t scan_idle_only,
                                    uint32_t max_repair_attempts) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    auto mode = static_cast<smoke::engine::IntegrityMode>(mode_int);
    return smoke::engine::engine_set_integrity_tuning(
        handle, mode, scan_interval_ops, scan_idle_only, max_repair_attempts
    );
}

// Card 17: Get integrity status wrapper - returns dict
py::dict py_engine_get_integrity_status(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    smoke::engine::EngineIntegrityStatus status;

    if (!smoke::engine::engine_get_integrity_status(handle, &status)) {
        throw std::runtime_error("Failed to get integrity status from device");
    }

    py::dict result;
    result["last_scan_op_count"] = status.last_scan_op_count;
    result["total_sign_ops"] = status.total_sign_ops;

    py::list slots_list;
    for (int i = 0; i < static_cast<int>(smoke::engine::ENGINE_MAX_KEY_SLOTS); ++i) {
        py::dict slot_dict;
        slot_dict["integrity_version"] = status.slots[i].integrity_version;
        slot_dict["fault_flags"] = status.slots[i].fault_flags;
        slot_dict["self_heal_count"] = status.slots[i].self_heal_count;  // Card 19
        slots_list.append(slot_dict);
    }
    result["slots"] = slots_list;

    return result;
}

// Card 17: Get integrity mode wrapper
int py_engine_get_integrity_mode(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    return static_cast<int>(smoke::engine::engine_get_integrity_mode(handle));
}

// Card 19: Set root seed wrapper
bool py_engine_set_root_seed(int64_t handle_int, py::bytes seed_bytes) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    std::string seed_str = seed_bytes;
    if (seed_str.size() != 32) {
        throw std::invalid_argument("Root seed must be exactly 32 bytes");
    }
    return smoke::engine::engine_set_root_seed(
        handle, reinterpret_cast<const uint8_t*>(seed_str.data())
    );
}

// Card 19: Clear root seed wrapper
bool py_engine_clear_root_seed(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    return smoke::engine::engine_clear_root_seed(handle);
}

// Card 19: Set self-healing enabled wrapper
bool py_engine_set_self_healing_enabled(int64_t handle_int, bool enabled) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    return smoke::engine::engine_set_self_healing_enabled(handle, enabled);
}

// Card 19: Is root seed set wrapper
bool py_engine_is_root_seed_set(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    return smoke::engine::engine_is_root_seed_set(handle);
}

// Card 19: Is self-healing enabled wrapper
bool py_engine_is_self_healing_enabled(int64_t handle_int) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    return smoke::engine::engine_is_self_healing_enabled(handle);
}

// Card 20.5: Batch submit wrapper
// Takes list of hashes (each 32 bytes) and list of request_ids
// Returns number of successfully submitted requests
uint32_t py_engine_submit_batch(int64_t handle_int, py::bytes hashes_bytes, py::list request_ids_list) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    std::string hashes_str = hashes_bytes;
    
    uint32_t count = static_cast<uint32_t>(request_ids_list.size());
    if (count == 0 || hashes_str.size() != count * 32) {
        throw std::runtime_error("hashes must be count * 32 bytes");
    }
    
    std::vector<uint32_t> request_ids(count);
    for (uint32_t i = 0; i < count; ++i) {
        request_ids[i] = request_ids_list[i].cast<uint32_t>();
    }
    
    return smoke::engine::engine_submit_batch(
        handle,
        reinterpret_cast<const uint8_t*>(hashes_str.data()),
        request_ids.data(),
        count
    );
}

// Card 20.5: Batch poll wrapper
// Returns tuple of (ready_count, list of (request_id, status, sig_r, sig_s) for ready responses)
py::tuple py_engine_poll_batch(int64_t handle_int, py::list request_ids_list) {
    auto* handle = reinterpret_cast<smoke::engine::EngineHandle*>(handle_int);
    
    uint32_t count = static_cast<uint32_t>(request_ids_list.size());
    if (count == 0) {
        return py::make_tuple(0, py::list());
    }
    
    std::vector<uint32_t> request_ids(count);
    for (uint32_t i = 0; i < count; ++i) {
        request_ids[i] = request_ids_list[i].cast<uint32_t>();
    }
    
    std::vector<smoke::engine::EngineResponse> responses(count);
    std::vector<uint8_t> ready_flags(count);
    
    uint32_t ready_count = smoke::engine::engine_poll_batch(
        handle,
        request_ids.data(),
        responses.data(),
        ready_flags.data(),
        count
    );
    
    py::list result_list;
    for (uint32_t i = 0; i < count; ++i) {
        if (ready_flags[i]) {
            auto& resp = responses[i];
            result_list.append(py::make_tuple(
                resp.request_id,
                static_cast<int>(resp.status),
                py::bytes(reinterpret_cast<const char*>(resp.sig_r), 32),
                py::bytes(reinterpret_cast<const char*>(resp.sig_s), 32)
            ));
        }
    }
    
    return py::make_tuple(ready_count, result_list);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Persistent P-256 Engine bindings (CARD 06 + 08B + 10 + 11 + 14 + 15 + 16 + 17 + 18 + 19 + 20.5)";

    m.def("engine_create", &py_engine_create, "Create engine handle.");
    m.def("engine_destroy", &py_engine_destroy, py::arg("handle"), "Destroy engine.");
    m.def("engine_start", &py_engine_start, py::arg("handle"), "Start kernel.");
    m.def("engine_shutdown", &py_engine_shutdown, py::arg("handle"), "Shutdown kernel.");
    m.def("engine_set_key", &py_engine_set_key, py::arg("handle"), py::arg("private_key_bytes"), "Set initial key (epoch 0).");
    m.def("engine_rotate_key", &py_engine_rotate_key, py::arg("handle"), py::arg("private_key_bytes"), "Rotate key (flip epoch).");
    m.def("engine_submit_request", &py_engine_submit_request, py::arg("handle"), py::arg("hash_bytes"), py::arg("request_id"), "Submit hash for signing.");
    m.def("engine_poll_response", &py_engine_poll_response, py::arg("handle"), py::arg("request_id"), "Poll for response.");
    m.def("engine_get_device_name", &py_engine_get_device_name, "Get GPU device name (Card 08B).");

#if ENGINE_MICROBATCH_DEBUG
    m.def("engine_get_microbatch_debug", &py_engine_get_microbatch_debug, py::arg("handle"),
          "Get microbatch debug telemetry (Card 10).");
#endif

    // Card 11: Engine profiles
    py::enum_<smoke::engine::EngineProfile>(m, "EngineProfile")
        .value("LATENCY_FIRST", smoke::engine::ENGINE_PROFILE_LATENCY_FIRST)
        .value("BALANCED", smoke::engine::ENGINE_PROFILE_BALANCED)
        .value("THROUGHPUT", smoke::engine::ENGINE_PROFILE_THROUGHPUT)
        .export_values();

    m.def("engine_set_profile", &py_engine_set_profile,
          py::arg("handle"), py::arg("profile"),
          "Set engine profile (Card 11).");
    m.def("engine_set_tuning", &py_engine_set_tuning,
          py::arg("handle"),
          py::arg("microbatch_min"),
          py::arg("microbatch_max"),
          py::arg("depth_low_water"),
          py::arg("depth_high_water"),
          "Set custom engine tuning (Card 11).");

    // Card 14: Nonce modes
    py::enum_<smoke::engine::NonceMode>(m, "NonceMode")
        .value("DETERMINISTIC", smoke::engine::NonceMode::DETERMINISTIC)
        .value("RANDOM", smoke::engine::NonceMode::RANDOM)
        .export_values();

    m.def("engine_set_nonce_mode", &py_engine_set_nonce_mode,
          py::arg("handle"), py::arg("nonce_mode"),
          "Set nonce generation mode: DETERMINISTIC (RFC 6979) or RANDOM (Card 14+15).");
    m.def("engine_set_low_s_enabled", &py_engine_set_low_s_enabled,
          py::arg("handle"), py::arg("enabled"),
          "Enable/disable canonical low-s normalization (Card 14).");

    // Card 15: CSPRNG reseed
    m.def("engine_reseed_rng", &py_engine_reseed_rng,
          py::arg("handle"), py::arg("seed_bytes"),
          "Reseed the CSPRNG with new 32-byte seed (Card 15).");

    // Card 16: Rotation policy
    py::enum_<smoke::engine::RotationPolicy>(m, "RotationPolicy")
        .value("DISABLED", smoke::engine::RotationPolicy::DISABLED)
        .value("MANUAL", smoke::engine::RotationPolicy::MANUAL)
        .value("AUTOMATIC", smoke::engine::RotationPolicy::AUTOMATIC)
        .export_values();

    m.def("engine_set_rotation_policy", &py_engine_set_rotation_policy,
          py::arg("handle"),
          py::arg("policy"),
          py::arg("auto_rotate_interval_ops") = 0,
          py::arg("auto_rotate_idle_only") = 1,
          "Set rotation policy: DISABLED, MANUAL (default), or AUTOMATIC (Card 16).");
    m.def("engine_get_rotation_policy", &py_engine_get_rotation_policy,
          py::arg("handle"),
          "Get current rotation policy (Card 16).");

    // Card 17: Integrity mode enum
    py::enum_<smoke::engine::IntegrityMode>(m, "IntegrityMode")
        .value("DISABLED", smoke::engine::IntegrityMode::INTEGRITY_DISABLED)
        .value("MONITOR", smoke::engine::IntegrityMode::INTEGRITY_MONITOR)
        .value("STRICT", smoke::engine::IntegrityMode::INTEGRITY_STRICT)
        .export_values();

    // Card 17: Keyslot fault flags (bitmask values)
    m.attr("FAULT_NONE") = static_cast<int>(smoke::engine::FAULT_NONE);
    m.attr("FAULT_MIRROR_MISMATCH") = static_cast<int>(smoke::engine::FAULT_MIRROR_MISMATCH);
    m.attr("FAULT_TAG_MISMATCH") = static_cast<int>(smoke::engine::FAULT_TAG_MISMATCH);
    m.attr("FAULT_REPAIRED_FROM_MIRROR") = static_cast<int>(smoke::engine::FAULT_REPAIRED_FROM_MIRROR);
    m.attr("FAULT_HARD_ZEROIZED") = static_cast<int>(smoke::engine::FAULT_HARD_ZEROIZED);
    // Card 19: Self-heal fault flag
    m.attr("FAULT_SELF_HEALED") = static_cast<int>(smoke::engine::FAULT_SELF_HEALED);

    m.def("engine_set_integrity_tuning", &py_engine_set_integrity_tuning,
          py::arg("handle"),
          py::arg("mode"),
          py::arg("scan_interval_ops") = 10000,
          py::arg("scan_idle_only") = 1,
          py::arg("max_repair_attempts") = 1,
          "Set integrity tuning: mode, scan interval, idle-only, max repair attempts (Card 17).");
    m.def("engine_get_integrity_status", &py_engine_get_integrity_status,
          py::arg("handle"),
          "Get integrity status from device: slots, fault_flags, versions (Card 17).");
    m.def("engine_get_integrity_mode", &py_engine_get_integrity_mode,
          py::arg("handle"),
          "Get current integrity mode (Card 17).");

    // Card 19: Root seed + self-healing APIs
    m.def("engine_set_root_seed", &py_engine_set_root_seed,
          py::arg("handle"), py::arg("seed_bytes"),
          "Set 32-byte root seed for HKDF key derivation (Card 19).");
    m.def("engine_clear_root_seed", &py_engine_clear_root_seed,
          py::arg("handle"),
          "Clear root seed (zeroize, disable self-healing) (Card 19).");
    m.def("engine_set_self_healing_enabled", &py_engine_set_self_healing_enabled,
          py::arg("handle"), py::arg("enabled"),
          "Enable/disable self-healing of corrupted keyslots (Card 19).");
    m.def("engine_is_root_seed_set", &py_engine_is_root_seed_set,
          py::arg("handle"),
          "Check if root seed has been set (Card 19).");
    m.def("engine_is_self_healing_enabled", &py_engine_is_self_healing_enabled,
          py::arg("handle"),
          "Check if self-healing is enabled (Card 19).");

    // Card 20.5: High-throughput batch APIs
    m.def("engine_submit_batch", &py_engine_submit_batch,
          py::arg("handle"), py::arg("hashes_bytes"), py::arg("request_ids"),
          "Submit batch of hashes for signing. Returns count submitted (Card 20.5).");
    m.def("engine_poll_batch", &py_engine_poll_batch,
          py::arg("handle"), py::arg("request_ids"),
          "Poll batch of responses. Returns (ready_count, list of responses) (Card 20.5).");

    // Card 25.1: FAST PATH APIs
    m.def("fast_path_create", [](int64_t legacy_handle) -> int64_t {
        auto* legacy = reinterpret_cast<smoke::engine::EngineHandle*>(legacy_handle);
        auto* fp = smoke::engine::fast_path_create(legacy);
        return reinterpret_cast<int64_t>(fp);
    }, py::arg("legacy_handle"), "Create FAST PATH engine from legacy handle (Card 25.1).");

    m.def("fast_path_destroy", [](int64_t handle) {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        smoke::engine::fast_path_destroy(fp);
    }, py::arg("handle"), "Destroy FAST PATH engine (Card 25.1).");

    // Card 26.1B: Set number of CTAs (service lanes) before starting
    m.def("fast_path_set_num_ctas", [](int64_t handle, uint32_t num_ctas) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_num_ctas(fp, num_ctas);
    }, py::arg("handle"), py::arg("num_ctas"),
    "Set number of CTAs (service lanes) for multi-CTA mode. Call BEFORE fast_path_start() (Card 26.1B).");

    // Card 26.2: Set number of shards before starting (zero contention mode)
    m.def("fast_path_set_num_shards", [](int64_t handle, uint32_t num_shards) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_num_shards(fp, num_shards);
    }, py::arg("handle"), py::arg("num_shards"),
    "Set number of shards for zero-contention mode. Call BEFORE fast_path_start() (Card 26.2).");

    // Card 26.14: Set submit policy (0=RR_PER_REQ, 1=SHARD_LOCAL)
    m.def("fast_path_set_submit_policy", [](int64_t handle, uint32_t policy) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_submit_policy(fp, policy);
    }, py::arg("handle"), py::arg("policy"),
    "Set submit policy: 0=RR_PER_REQ (round-robin, many fences), 1=SHARD_LOCAL (pack into K shards). Call BEFORE fast_path_start() (Card 26.14).");

    // Card 26.14: Set shards per batch for SHARD_LOCAL policy
    m.def("fast_path_set_shards_per_batch", [](int64_t handle, uint32_t k) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_shards_per_batch(fp, k);
    }, py::arg("handle"), py::arg("k"),
    "Set number of shards to touch per batch (K) for SHARD_LOCAL policy. Call BEFORE fast_path_start() (Card 26.14).");

    m.def("fast_path_start", [](int64_t handle) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_start(fp);
    }, py::arg("handle"), "Start FAST PATH persistent kernel (Card 25.1).");

    m.def("fast_path_submit", [](int64_t handle, py::bytes hash_bytes, uint8_t flags) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::string hash_str = hash_bytes;
        if (hash_str.size() != 32) return false;
        return smoke::engine::fast_path_submit(fp, reinterpret_cast<const uint8_t*>(hash_str.data()), flags);
    }, py::arg("handle"), py::arg("hash_bytes"), py::arg("flags") = 0,
    "Submit single request to FAST PATH (no cudaMemcpy!) (Card 25.1).");

    // Card 26.19: Added opcode and input_len parameters for multi-op support
    // D5b-keyslot: Added key_slot (shared by the batch; default 0)
    m.def("fast_path_submit_batch", [](int64_t handle, py::bytes inputs_bytes, uint8_t flags,
                                        uint8_t client_id, uint16_t opcode, uint16_t input_len,
                                        uint8_t key_slot) -> uint32_t {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::string inputs_str = inputs_bytes;
        if (inputs_str.size() % 32 != 0) return 0;
        uint32_t count = static_cast<uint32_t>(inputs_str.size() / 32);
        return smoke::engine::fast_path_submit_batch(fp, reinterpret_cast<const uint8_t*>(inputs_str.data()),
                                                      count, flags, client_id, opcode, input_len, key_slot);
    }, py::arg("handle"), py::arg("inputs_bytes"), py::arg("flags") = 0,
       py::arg("client_id") = 0, py::arg("opcode") = 0, py::arg("input_len") = 32,
       py::arg("key_slot") = 0,
    "Submit batch to FAST PATH. Returns count submitted. opcode: 0=P256_SIGN, 10=SHA256 (Card 26.19).");

    m.def("fast_path_poll", [](int64_t handle, uint32_t max_count) -> py::list {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::vector<smoke::engine::FastPathResponse> responses(max_count);
        uint32_t count = smoke::engine::fast_path_poll(fp, responses.data(), max_count);

        py::list result;
        for (uint32_t i = 0; i < count; i++) {
            auto& resp = responses[i];
            result.append(py::make_tuple(
                resp.request_id,
                static_cast<int>(resp.status),
                py::bytes(reinterpret_cast<const char*>(resp.r), 32),
                py::bytes(reinterpret_cast<const char*>(resp.s), 32)
            ));
        }
        return result;
    }, py::arg("handle"), py::arg("max_count") = 256,
    "Poll FAST PATH responses. Returns list of (req_id, status, r, s) (Card 25.1).");

    // Card 26.25: Extended poll that includes slab metadata for N > 2 multi-message
    m.def("fast_path_poll_extended", [](int64_t handle, uint32_t max_count) -> py::list {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        std::vector<smoke::engine::FastPathResponse> responses(max_count);
        uint32_t count = smoke::engine::fast_path_poll(fp, responses.data(), max_count);

        py::list result;
        for (uint32_t i = 0; i < count; i++) {
            auto& resp = responses[i];
            // Return dict with all fields including slab metadata
            py::dict resp_dict;
            resp_dict["request_id"] = resp.request_id;
            resp_dict["status"] = static_cast<int>(resp.status);
            resp_dict["output_len"] = resp.output_len;
            resp_dict["output_offset"] = resp.output_offset;
            resp_dict["output_bytes"] = resp.output_bytes;
            resp_dict["output"] = py::bytes(reinterpret_cast<const char*>(resp.output), 64);
            result.append(resp_dict);
        }
        return result;
    }, py::arg("handle"), py::arg("max_count") = 256,
    "Poll FAST PATH responses with slab metadata (Card 26.25). Returns list of dicts with output_offset, output_bytes.");

    m.def("fast_path_get_telemetry", [](int64_t handle) -> py::dict {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        smoke::engine::FastPathTelemetry telem{};
        if (!smoke::engine::fast_path_get_telemetry(fp, &telem)) {
            return py::dict();
        }
        py::dict result;
        result["total_requests_seen"] = telem.total_requests_seen;
        result["total_batches_processed"] = telem.total_batches_processed;
        result["total_batch_size_sum"] = telem.total_batch_size_sum;
        result["kernel_idle_cycles"] = telem.kernel_idle_cycles;
        result["kernel_busy_cycles"] = telem.kernel_busy_cycles;
        result["kernel_launches"] = telem.kernel_launches;
        result["kernel_start_time_us"] = telem.kernel_start_time_us;
        if (telem.total_batches_processed > 0) {
            result["avg_batch_size"] = static_cast<double>(telem.total_batch_size_sum) / telem.total_batches_processed;
        } else {
            result["avg_batch_size"] = 0.0;
        }

        // Card 26.1: Shape truth fields - proves which signer is ACTUALLY engaged
        result["signer_model"] = telem.signer_model;  // 0=THREAD_ONLY, 1=WARP_COOP
        result["signer_model_name"] = (telem.signer_model == 1) ? "WARP_COOP" : "THREAD_ONLY";
        result["threads_per_signature"] = telem.threads_per_signature;
        result["service_lanes"] = telem.service_lanes;
        result["comb_table_loaded"] = telem.comb_table_loaded;

        // Card 26.1B: Per-CTA distribution (starvation proof)
        py::list per_cta_batches;
        py::list per_cta_requests;
        uint32_t num_lanes = telem.service_lanes;
        if (num_lanes > 32) num_lanes = 32;
        for (uint32_t i = 0; i < num_lanes; i++) {
            per_cta_batches.append(telem.per_cta_batches[i]);
            per_cta_requests.append(telem.per_cta_requests[i]);
        }
        result["per_cta_batches"] = per_cta_batches;
        result["per_cta_requests"] = per_cta_requests;

        // Card 26.22: Multi-msg telemetry
        result["dbg_units_completed"] = telem.dbg_units_completed;

        return result;
    }, py::arg("handle"), "Get FAST PATH telemetry (proves persistence!) (Card 25.1, 26.1).");

    m.def("fast_path_get_ring_status", [](int64_t handle) -> py::tuple {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        uint32_t head = 0, tail = 0, pending = 0;
        smoke::engine::fast_path_get_ring_status(fp, &head, &tail, &pending);
        return py::make_tuple(head, tail, pending);
    }, py::arg("handle"), "Get FAST PATH ring status (head, tail, pending) (Card 25.1).");

    // Card 26.1: Set comb_table_loaded flag in telemetry
    m.def("fast_path_set_comb_table_loaded", [](int64_t handle, bool loaded) -> bool {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        return smoke::engine::fast_path_set_comb_table_loaded(fp, loaded ? 1 : 0);
    }, py::arg("handle"), py::arg("loaded"),
    "Set comb_table_loaded flag in FAST PATH telemetry (Card 26.1).");

    // Card 26.25: Get output slab for extended multi-message outputs
    // Returns bytes object containing the full output slab (zero-copy via memoryview would be better)
    m.def("fast_path_get_output_slab", [](int64_t handle) -> py::bytes {
        auto* fp = reinterpret_cast<smoke::engine::FastPathHandle*>(handle);
        uint8_t* ptr = nullptr;
        uint32_t size = 0;
        if (!smoke::engine::fast_path_get_output_slab(fp, &ptr, &size) || !ptr || size == 0) {
            return py::bytes();  // Return empty bytes if not available
        }
        return py::bytes(reinterpret_cast<char*>(ptr), size);
    }, py::arg("handle"),
    "Get FAST PATH output slab as bytes (Card 26.25). Use with resp.output_offset and resp.output_bytes.");

    // Card 26.25: Get output slab segment size constant
    m.attr("FAST_PATH_OUTPUT_SEGMENT_BYTES") = smoke::engine::FAST_PATH_OUTPUT_SEGMENT_BYTES;

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
        return err == cudaSuccess;
    }, py::arg("x_table"), py::arg("y_table"),
    "Load P-256 comb table for warp-coop mode. Tables must be flat uint32 arrays of 1016 elements (Card 26).");

    m.def("is_warp_coop_enabled", []() -> bool { return true; },
    "Returns True if warp-coop mode is compiled in (Card 26).");

#if P256_FASTPATH_FULLWINDOW
    // SPEED-H2-01: Python pybind mirror of the full-window loader. Takes the
    // AoS Montgomery payload (ENTRIES*16 = 65536 u32) WITHOUT the file
    // header — the Python auto-loader strips the 32-byte header. Header type
    // + whole-file fingerprint validation is enforced on the C ABI / Go path
    // (smoke_engine_load_p256_fullwindow_table); the pybind path is for
    // in-process Python benches on trusted local artifacts.
    m.def("load_p256_fullwindow_table", [](py::array_t<uint32_t> payload) -> bool {
        auto buf = payload.request();
        if (buf.ndim != 1) throw std::runtime_error("payload must be 1D");
        if (buf.size != 33 * 128 * 16)
            throw std::runtime_error("payload must be 33*128*16 = 67584 u32 (fullwindow AoS, 33 windows)");
        cudaError_t err = smoke::engine::load_p256_fullwindow_table(
            static_cast<const uint32_t*>(buf.ptr), buf.size);
        return err == cudaSuccess;
    }, py::arg("payload"),
    "SPEED-H2-01: load the full-window fixed-base table payload (AoS, 65536 u32, header stripped).");

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
}
