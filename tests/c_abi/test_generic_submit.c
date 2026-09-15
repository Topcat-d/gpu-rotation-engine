/**
 * Card 26.82/26.87: Minimal C ABI Smoke Test
 *
 * This test verifies the generic submit/poll C ABI without Python/PyTorch.
 * Compile: cmake --build . (see CMakeLists.txt)
 *
 * Test sequence:
 *   1. smoke_engine_init()
 *   2. smoke_fast_path_create()
 *   3. smoke_fast_path_start()
 *   4. smoke_generic_submit() - SHA256 request
 *   5. smoke_generic_poll() - verify backend=GPU_FASTPATH
 *   6. smoke_engine_shutdown()
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Card 26.87: Define SMOKE_ABI_USING_DLL when linking against the DLL */
#ifndef SMOKE_ABI_USING_DLL
#define SMOKE_ABI_USING_DLL
#endif

#include "smoke_engine.h"
#include "smoke_generic_abi.h"

#ifdef _WIN32
#include <windows.h>
#define SLEEP_MS(ms) Sleep(ms)
#else
#include <unistd.h>
#define SLEEP_MS(ms) usleep((ms) * 1000)
#endif

/* Test result tracking */
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s: %s\n", __func__, msg); \
        tests_failed++; \
        return -1; \
    } else { \
        printf("[PASS] %s\n", msg); \
        tests_passed++; \
    } \
} while (0)

/* Test: Engine initialization */
static int test_engine_init(SmokeEngineHandle* out_engine) {
    SmokeEngineConfig config = {0};
    config.abi_version = SMOKE_ENGINE_ABI_VERSION;
    config.device_index = 0;  /* First GPU */
    config.queue_depth = 1024;

    SmokeErrorInfo error = {0};
    SmokeStatus status = smoke_engine_init(&config, out_engine, &error);

    TEST_ASSERT(status == SMOKE_STATUS_OK,
                "smoke_engine_init() returns OK");
    TEST_ASSERT(*out_engine != NULL,
                "Engine handle is non-NULL");

    return 0;
}

/* Test: FAST PATH lifecycle */
static int test_fast_path_lifecycle(SmokeEngineHandle engine) {
    SmokeErrorInfo error = {0};

    /* Create FAST PATH */
    SmokeStatus status = smoke_fast_path_create(engine, &error);
    TEST_ASSERT(status == SMOKE_STATUS_OK,
                "smoke_fast_path_create() returns OK");

    /* The supported persistent path requires matching service lanes/shards. */
    status = smoke_fast_path_set_num_ctas(engine, 2);
    TEST_ASSERT(status == SMOKE_STATUS_OK, "set two service lanes");
    status = smoke_fast_path_set_num_shards(engine, 2);
    TEST_ASSERT(status == SMOKE_STATUS_OK, "set two ring shards");

    /* Start FAST PATH */
    status = smoke_fast_path_start(engine, &error);
    TEST_ASSERT(status == SMOKE_STATUS_OK,
                "smoke_fast_path_start() returns OK");

    /* Verify running */
    int running = smoke_fast_path_is_running(engine);
    TEST_ASSERT(running != 0,
                "smoke_fast_path_is_running() returns true");

    /* Wait for kernel to initialize */
    SLEEP_MS(300);

    return 0;
}

/* Test: Generic submit/poll with SHA-256 */
static int test_generic_sha256(SmokeEngineHandle engine) {
    SmokeErrorInfo error = {0};

    /* Prepare SHA-256 request */
    SmokeGenericRequest req = {0};
    req.abi_version = SMOKE_GENERIC_ABI_VERSION;
    req.opcode = 10;  /* SHA256 */
    req.flags = 0;
    req.key_slot = 0;
    req.client_id = 42;
    req.input_len = 4;
    memcpy(req.input, "test", 4);

    /* Submit */
    SmokeStatus status = smoke_generic_submit(engine, &req, 1, &error);
    TEST_ASSERT(status == SMOKE_STATUS_OK,
                "smoke_generic_submit() returns OK");

    /* Poll for response (with timeout) */
    SmokeGenericResponse resp = {0};
    int got = 0;
    for (int i = 0; i < 200; i++) {  /* 2 second timeout */
        uint32_t count = smoke_generic_poll(engine, &resp, 1);
        if (count > 0) {
            got = 1;
            break;
        }
        SLEEP_MS(10);
    }

    TEST_ASSERT(got != 0,
                "smoke_generic_poll() returns response");

    /* Verify backend tag */
    TEST_ASSERT(resp.backend == SMOKE_BACKEND_GPU_FASTPATH,
                "Response backend is GPU_FASTPATH");

    /* Verify status */
    TEST_ASSERT(resp.status == 0,
                "Response status is OK");

    /* Verify output length (SHA-256 = 32 bytes) */
    TEST_ASSERT(resp.output_len == 32,
                "Response output_len is 32 (SHA-256)");

    char digest_hex[65];
    for (int i = 0; i < 32; ++i) {
        snprintf(digest_hex + 2 * i, 3, "%02x", resp.output[i]);
    }
    TEST_ASSERT(strcmp(digest_hex,
                "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08") == 0,
                "GPU SHA-256 matches the known digest of test");

    printf("  SHA-256 output (first 8 bytes): ");
    for (int i = 0; i < 8 && i < resp.output_len; i++) {
        printf("%02x", resp.output[i]);
    }
    printf("...\n");

    return 0;
}

/* Test: FAST PATH stop and shutdown */
static int test_shutdown(SmokeEngineHandle engine) {
    /* Stop FAST PATH */
    SmokeStatus status = smoke_fast_path_stop(engine);
    TEST_ASSERT(status == SMOKE_STATUS_OK,
                "smoke_fast_path_stop() returns OK");

    /* Verify stopped */
    int running = smoke_fast_path_is_running(engine);
    TEST_ASSERT(running == 0,
                "smoke_fast_path_is_running() returns false after stop");

    /* Shutdown engine */
    status = smoke_engine_shutdown(engine);
    TEST_ASSERT(status == SMOKE_STATUS_OK,
                "smoke_engine_shutdown() returns OK");

    return 0;
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("Card 26.82: C ABI Smoke Test\n");
    printf("============================================================\n\n");

    SmokeEngineHandle engine = NULL;
    int result = 0;

    /* Run tests */
    if (test_engine_init(&engine) != 0) {
        result = 1;
        goto cleanup;
    }

    if (test_fast_path_lifecycle(engine) != 0) {
        result = 1;
        goto cleanup;
    }

    if (test_generic_sha256(engine) != 0) {
        result = 1;
        goto cleanup;
    }

cleanup:
    if (engine) {
        test_shutdown(engine);
    }

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return (tests_failed > 0) ? 1 : 0;
}
