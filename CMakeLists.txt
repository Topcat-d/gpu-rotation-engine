# Card 26.87 + 26.89 + 26.92: Smoke Engine C ABI Library Build
#
# Builds smoke_engine_abi.dll (Windows) / libsmoke_engine_abi.so (Linux)
# This is the pure C ABI library without Python/PyTorch dependencies.
#
# ABI Version: 1 (Frozen per Card 26.89)
#
# Usage:
#   mkdir build_abi && cd build_abi
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --config Release
#
# On Windows with Visual Studio:
#   cmake -G "Visual Studio 17 2022" -A x64 ..
#   cmake --build . --config Release

cmake_minimum_required(VERSION 3.18)
project(smoke_engine_abi LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Hopper acceptance must contain native sm_90 code. Keep the list configurable
# so a rental rehearsal can build only its named architecture while release
# artifacts retain the supported compatibility set.
set(
    SMOKE_CUDA_ARCHITECTURES
    "70;75;80;86;89;90"
    CACHE STRING
    "CUDA architectures compiled into the Smoke Engine ABI"
)

# MSVC requires explicit /std:c11 for C atomics
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /std:c11")
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler definitions
#
# P-256 signing-path flags MUST match engine/build_engine.py (the validated
# .pyd build) exactly — they select which Jacobian and Montgomery code paths
# compile. Omitting them silently selects the never-validated warp-parallel
# Jacobian (jacobian_double_warp), which produces off-curve/non-verifying
# signatures. P256_USE_ORACLE_JACOBIAN is #ifdef-gated, so defining it to 0
# still ENABLES the scalar-oracle Jacobian (the correct path) — the value is
# irrelevant, only the definition matters. See engine/build_engine.py:89-102.
add_compile_definitions(
    ENGINE_USE_REAL_P256=1
    P256_WARP_COOP_ENABLED=1
    P256_USE_ORACLE_JACOBIAN=0   # #ifdef-gated → selects scalar-oracle Jacobian (correct)
    P256_USE_FAST_WARP_MONT=1    # fast-tier warp CIOS Montgomery multiply
    USE_SERIAL_PREFIX_A=0        # parallel Step-A prefix scan (matches validated)
    USE_SERIAL_COND_SUB=0        # parallel conditional-subtract (matches validated)
    CARD26_TELEMETRY_ENABLED=1
    DILITHIUM_HAVE_CUDA=1        # ML-DSA (PQC) — parity with engine/build_engine.py:96
    SLH_HAVE_CUDA=1              # Card SLH-002 P2: fused SLH-DSA-SHA2-128s signer
    SMOKE_ABI_BUILDING_DLL=1      # For dllexport
    SMOKE_ABI_NO_PYTHON=1         # Exclude pybind11
)

# SPEED-H2-01: opt-in full-window fixed-base signer (zero-doubling k*G).
# Env-gated so the default Linux CMake build is byte-for-byte unchanged.
# Parity with engine/build_engine.py's SMOKE_FASTPATH_FULLWINDOW gate.
# Enable: SMOKE_FASTPATH_FULLWINDOW=1 cmake ... (or -DP256_FASTPATH_FULLWINDOW=1).
set(_smoke_fullwindow_default OFF)
if("$ENV{SMOKE_FASTPATH_FULLWINDOW}" STREQUAL "1")
    set(_smoke_fullwindow_default ON)
endif()
option(SMOKE_FASTPATH_FULLWINDOW "Enable the full-window P-256 signer" ${_smoke_fullwindow_default})
if(SMOKE_FASTPATH_FULLWINDOW)
    add_compile_definitions(P256_FASTPATH_FULLWINDOW=1)
    message(STATUS "SPEED-H2-01: full-window fixed-base signer ENABLED (zero-doubling)")
endif()

# Include directories
set(SMOKE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core/persistent_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/core/p256_persistent
    # ML-DSA / Dilithium (PQC) — mirror engine/build_engine.py include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}/core/pqc/dilithium/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core/pqc/dilithium/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/core/pqc/dilithium/src
    # Card SLH-002 P2: SLH-DSA fused signer header (device prologue + entries)
    ${CMAKE_CURRENT_SOURCE_DIR}/core/pqc/slh_dsa/cuda
)

# Source files for C ABI library
set(ABI_SOURCES
    engine/abi/smoke_engine_abi.cpp
    engine/abi/smoke_abi_bridge.cpp
    core/persistent_engine/engine_host.cpp
    core/persistent_engine/engine_device_info.cpp
    # ML-DSA host C++ (parity with engine/build_engine.py dilithium_cpp_sources)
    core/pqc/dilithium/src/dilithium_params.cpp
    core/pqc/dilithium/src/dilithium_gpu.cpp
    core/pqc/dilithium/src/dilithium_expandA.cpp
)

# RRP-A06b note: there are deliberately NO core/rsa2048_crt sources here. The
# RSA device code is header-only (engine_loop.cu includes
# rsa2048_crt_device.cuh -> core/rsa2048_crt/rsa2048_crt.cuh), so it compiles
# with engine_loop.cu — but the production sharded kernel has no RSA dispatch
# handler (legacy-kernel-only), and the ABI refuses RSA ring ops fail-closed.
# Adding sources here would change nothing; a sharded-kernel handler is what
# re-enables RSA.
set(CUDA_SOURCES
    core/persistent_engine/engine_loop.cu
    # ML-DSA CUDA (parity with engine/build_engine.py dilithium_cuda_sources)
    core/pqc/dilithium/cuda/dilithium_ntt.cu
    core/pqc/dilithium/cuda/dilithium_shake.cu
    core/pqc/dilithium/cuda/dilithium_shake_batch.cu
    core/pqc/dilithium/cuda/dilithium_sampling.cu
    core/pqc/dilithium/cuda/dilithium_decompose.cu
    core/pqc/dilithium/cuda/dilithium_pack.cu
    core/pqc/dilithium/cuda/dilithium_w1.cu
    core/pqc/dilithium/cuda/dilithium_sign.cu
    core/pqc/dilithium/cuda/dilithium_sign_batch.cu
    core/pqc/dilithium/cuda/dilithium_challenge.cu
    core/pqc/dilithium/cuda/dilithium_keygen.cu
    # Card SLH-002 P2: fused SLH-DSA-SHA2-128s signer (both kernel entries;
    # the P1 harness slh_p1_harness.cu stays OUT of the engine build)
    core/pqc/slh_dsa/cuda/slh_sha2_128s_sign.cu
)

# Build CUDA sources as a static library first (avoids NVCC command-line issues)
add_library(smoke_engine_cuda STATIC
    ${CUDA_SOURCES}
)

target_include_directories(smoke_engine_cuda PRIVATE
    ${SMOKE_INCLUDE_DIRS}
)

set_target_properties(smoke_engine_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${SMOKE_CUDA_ARCHITECTURES}"
)

target_compile_options(smoke_engine_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)

# Create shared library (C++ only, links CUDA library)
add_library(smoke_engine_abi SHARED
    ${ABI_SOURCES}
)

target_include_directories(smoke_engine_abi PRIVATE
    ${SMOKE_INCLUDE_DIRS}
)

# Link CUDA runtime and our CUDA static lib
target_link_libraries(smoke_engine_abi PRIVATE
    smoke_engine_cuda
    CUDA::cudart
)

# Windows-specific settings
if(WIN32)
    # Generate import library
    set_target_properties(smoke_engine_abi PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF  # We use explicit exports
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
    # Suppress some warnings for C++ sources
    target_compile_options(smoke_engine_abi PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /wd4996 /wd4251 /wd4275>
    )
endif()

# Install targets
install(FILES LICENSE NOTICE DESTINATION share/licenses/smoke_engine)
install(FILES LICENSES/MIT.txt DESTINATION share/licenses/smoke_engine/LICENSES)

# Public precomputation assets are part of the standalone SDK, not private keys.
install(DIRECTORY data/p256_comb data/p256_fullwindow DESTINATION share/smoke_engine/data)
install(FILES scripts/tools/gen_p256_comb_table.py scripts/tools/gen_p256_fullwindow_table.py
    DESTINATION share/smoke_engine/tools)
install(FILES examples/standalone_rotation.py DESTINATION share/smoke_engine/examples)
install(FILES requirements/standalone.txt DESTINATION share/smoke_engine/requirements)

install(TARGETS smoke_engine_abi
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    include/smoke_engine.h
    include/smoke_generic_abi.h
    include/smoke_abi.h
    include/smoke_rotation_probe.h
    DESTINATION include
)

# ============================================================================
# C ABI Tests
# ============================================================================

# Test executable
add_executable(test_c_abi
    tests/c_abi/test_generic_submit.c
)

target_include_directories(test_c_abi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_c_abi PRIVATE
    smoke_engine_abi
    CUDA::cudart
)

# Header validation test (compile-only)
add_executable(test_c_abi_headers
    tests/c_abi/test_headers.c
)

target_include_directories(test_c_abi_headers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# MSVC needs /std:c11 and /experimental:c11atomics for C atomics
if(MSVC)
    target_compile_options(test_c_abi_headers PRIVATE /std:c11 /experimental:c11atomics)
    target_compile_options(test_c_abi PRIVATE /std:c11 /experimental:c11atomics)
endif()

# CTest integration
enable_testing()
add_test(NAME c_abi_headers COMMAND test_c_abi_headers)
add_test(NAME c_abi_smoke COMMAND test_c_abi)

# Print configuration
message(STATUS "===========================================")
message(STATUS "Smoke Engine C ABI Build Configuration")
message(STATUS "===========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "Include dirs: ${SMOKE_INCLUDE_DIRS}")
message(STATUS "===========================================")
