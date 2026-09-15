/**
 * Card 26.84/26.87: C ABI Header Validation Test
 *
 * This file validates that the C ABI headers compile cleanly from C.
 * It does NOT link against the library - just checks header correctness.
 *
 * Compile: cl /c /nologo /W3 /WX /I../../include test_headers.c
 */

#include <stdint.h>
#include <stddef.h>

/* Card 26.87: Test with DLL import macros */
#define SMOKE_ABI_USING_DLL

/* Include all public C ABI headers */
#include "smoke_engine.h"
#include "smoke_generic_abi.h"
#include "smoke_abi.h"          /* Card 27.00B: Shared-memory ABI */

/* Compile-time size assertions */
#ifdef __cplusplus
#define STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define STATIC_ASSERT(cond, msg) typedef char static_assertion_##msg[(cond)?1:-1]
#endif

/* Verify struct sizes are as expected (ABI stability) */
STATIC_ASSERT(sizeof(SmokeGenericRequest) == 64, request_size);
STATIC_ASSERT(sizeof(SmokeGenericResponse) == 128, response_size);
STATIC_ASSERT(sizeof(SmokeErrorInfo) == 264, error_info_size);

/* Card 27.00B: Verify smoke_abi.h struct sizes */
STATIC_ASSERT(sizeof(SmokeABIRequest) == 64, abi_request_size);
STATIC_ASSERT(sizeof(SmokeABIResponse) == 128, abi_response_size);
STATIC_ASSERT(sizeof(SmokeABIHeader) == 64, abi_header_size);
STATIC_ASSERT(sizeof(SmokeABIRing) == 128, abi_ring_size);

/* SBE-002 (2026-08-17): Verify SmokeEngineMeasurement layout is stable —
 * this struct crosses the C ABI (smoke_engine_get_measurement) and Go
 * enginecgo mirrors it field-for-field, so an accidental size/offset
 * change here would silently desync the two. No private key material is
 * exposed by this struct (see smoke_engine.h doc comment above the
 * declaration) — this assertion is layout-only, not a residency claim. */
STATIC_ASSERT(sizeof(SmokeEngineMeasurement) == 60, engine_measurement_size);
STATIC_ASSERT(offsetof(SmokeEngineMeasurement, engine_measurement) == 0, meas_engine_measurement);
STATIC_ASSERT(offsetof(SmokeEngineMeasurement, active_epoch) == 32, meas_active_epoch);
STATIC_ASSERT(offsetof(SmokeEngineMeasurement, num_key_slots) == 36, meas_num_key_slots);
STATIC_ASSERT(offsetof(SmokeEngineMeasurement, abi_version) == 40, meas_abi_version);

/* Verify enum values are stable */
STATIC_ASSERT(SMOKE_STATUS_OK == 0, status_ok);
STATIC_ASSERT(SMOKE_BACKEND_GPU_FASTPATH == 1, backend_fastpath);

/* Verify request struct layout */
STATIC_ASSERT(offsetof(SmokeGenericRequest, abi_version) == 0, req_abi_version);
STATIC_ASSERT(offsetof(SmokeGenericRequest, opcode) == 4, req_opcode);
STATIC_ASSERT(offsetof(SmokeGenericRequest, flags) == 6, req_flags);
STATIC_ASSERT(offsetof(SmokeGenericRequest, key_slot) == 7, req_key_slot);
STATIC_ASSERT(offsetof(SmokeGenericRequest, client_id) == 8, req_client_id);
STATIC_ASSERT(offsetof(SmokeGenericRequest, input_len) == 16, req_input_len);
STATIC_ASSERT(offsetof(SmokeGenericRequest, payload_offset) == 20, req_payload_offset);
STATIC_ASSERT(offsetof(SmokeGenericRequest, payload_len) == 24, req_payload_len);
STATIC_ASSERT(offsetof(SmokeGenericRequest, input) == 32, req_input);

/* Verify response struct layout */
STATIC_ASSERT(offsetof(SmokeGenericResponse, client_id) == 0, resp_client_id);
STATIC_ASSERT(offsetof(SmokeGenericResponse, seq) == 8, resp_seq);
STATIC_ASSERT(offsetof(SmokeGenericResponse, status) == 16, resp_status);
STATIC_ASSERT(offsetof(SmokeGenericResponse, backend) == 17, resp_backend);
STATIC_ASSERT(offsetof(SmokeGenericResponse, opcode) == 18, resp_opcode);
STATIC_ASSERT(offsetof(SmokeGenericResponse, output_len) == 20, resp_output_len);
STATIC_ASSERT(offsetof(SmokeGenericResponse, output_offset) == 24, resp_output_offset);
STATIC_ASSERT(offsetof(SmokeGenericResponse, output_bytes) == 28, resp_output_bytes);
STATIC_ASSERT(offsetof(SmokeGenericResponse, output) == 64, resp_output);

/* Dummy main to satisfy linker if compiled as executable */
int main(void) {
    /* All checks are compile-time - if we get here, headers are valid */
    return 0;
}
