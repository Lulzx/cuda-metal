cmake_minimum_required(VERSION 3.28)

project(cumetal VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

option(CUMETAL_BUILD_TESTS "Build test targets" ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CUMETAL_ENABLE_BINARY_SHIM_DEFAULT OFF)
else()
    set(CUMETAL_ENABLE_BINARY_SHIM_DEFAULT ON)
endif()
option(CUMETAL_ENABLE_BINARY_SHIM
    "Enable CUDA binary-shim library alias (build/install libcuda.dylib -> libcumetal.dylib)"
    ${CUMETAL_ENABLE_BINARY_SHIM_DEFAULT}
)

add_library(cumetal_common STATIC
    compiler/common/src/metallib.cpp
)

target_include_directories(cumetal_common
    PUBLIC
        compiler/common/include
)

target_compile_options(cumetal_common
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_ptx STATIC
    compiler/ptx/src/parser.cpp
)

target_include_directories(cumetal_ptx
    PUBLIC
        compiler/ptx/include
)

target_compile_options(cumetal_ptx
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_passes STATIC
    compiler/passes/src/addrspace.cpp
    compiler/passes/src/intrinsic_lower.cpp
    compiler/passes/src/metadata.cpp
    compiler/passes/src/phase1_pipeline.cpp
    compiler/passes/src/printf_lower.cpp
    compiler/passes/src/threadgroup_tiling.cpp
)

target_include_directories(cumetal_passes
    PUBLIC
        compiler/passes/include
)

target_link_libraries(cumetal_passes
    PUBLIC
        cumetal_ptx
)

target_compile_options(cumetal_passes
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_ptx_codegen STATIC
    compiler/ptx/src/lower_to_llvm.cpp
    compiler/ptx/src/lower_to_metal.cpp
)

target_include_directories(cumetal_ptx_codegen
    PUBLIC
        compiler/ptx/include
)

target_link_libraries(cumetal_ptx_codegen
    PUBLIC
        cumetal_passes
)

target_compile_options(cumetal_ptx_codegen
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(CUMETAL_ENABLE_BINARY_SHIM)
    set(CUMETAL_REGISTRATION_SOURCE runtime/registration/registration.cpp)
else()
    set(CUMETAL_REGISTRATION_SOURCE runtime/registration/registration_stub.cpp)
endif()

add_library(cumetal_runtime SHARED
    runtime/cache/module_cache.cpp
    runtime/driver/cuda_driver.cpp
    runtime/error/library_conflict.cpp
    ${CUMETAL_REGISTRATION_SOURCE}
    runtime/rt/allocation_table.cpp
    runtime/rt/cublas.cpp
    runtime/rt/cufft.cpp
    runtime/rt/curand.cpp
    runtime/rt/cuda_runtime.cpp
    runtime/metal_backend/metal_backend.mm
)

set_target_properties(cumetal_runtime
    PROPERTIES
        OUTPUT_NAME cumetal
)

set_source_files_properties(
    runtime/metal_backend/metal_backend.mm
    PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
)

target_include_directories(cumetal_runtime
    PUBLIC
        compiler/air_emitter/include
        compiler/common/include
        compiler/ptx/include
        runtime/api
        runtime/cache
        runtime/error
        runtime/registration
        runtime/rt
        runtime/metal_backend
)

target_link_libraries(cumetal_runtime
    PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
)

target_compile_options(cumetal_runtime
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(APPLE)
    add_custom_command(
        TARGET cumetal_runtime
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcublas.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcurand.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcufft.dylib
    )

    if(CUMETAL_ENABLE_BINARY_SHIM)
        add_custom_command(
            TARGET cumetal_runtime
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                    $<TARGET_FILE_NAME:cumetal_runtime>
                    $<TARGET_FILE_DIR:cumetal_runtime>/libcuda.dylib
        )
    endif()
endif()

add_library(cumetal_air_validate STATIC
    compiler/air_validate/src/validator.cpp
)

target_include_directories(cumetal_air_validate
    PUBLIC
        compiler/air_validate/include
)

target_link_libraries(cumetal_air_validate
    PUBLIC
        cumetal_common
)

target_compile_options(cumetal_air_validate
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(air_validate
    compiler/air_validate/src/main.cpp
)

target_link_libraries(air_validate
    PRIVATE
        cumetal_air_validate
)

add_library(cumetal_air_emitter STATIC
    compiler/air_emitter/src/emitter.cpp
)

target_include_directories(cumetal_air_emitter
    PUBLIC
        compiler/air_emitter/include
)

target_link_libraries(cumetal_air_emitter
    PUBLIC
        cumetal_air_validate
)

target_compile_options(cumetal_air_emitter
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(cumetal-air-emitter
    compiler/air_emitter/src/main.cpp
)

target_link_libraries(cumetal-air-emitter
    PRIVATE
        cumetal_air_emitter
)

target_link_libraries(cumetal_runtime
    PRIVATE
        cumetal_air_emitter
        cumetal_ptx_codegen
)

add_executable(cumetalc
    compiler/cumetalc/main.cpp
)

target_link_libraries(cumetalc
    PRIVATE
        cumetal_air_emitter
        cumetal_ptx_codegen
)

target_compile_options(cumetalc
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_definitions(cumetalc
    PRIVATE
        CUMETAL_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_executable(cumetal-ptx2llvm
    compiler/ptx/src/main.cpp
)

target_link_libraries(cumetal-ptx2llvm
    PRIVATE
        cumetal_common
        cumetal_ptx_codegen
)

target_compile_options(cumetal-ptx2llvm
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(air_inspect
    tools/air_inspect/main.cpp
)

target_link_libraries(air_inspect
    PRIVATE
        cumetal_common
)

target_compile_options(air_inspect
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(ptx_diff
    tools/ptx_diff/main.cpp
)

target_link_libraries(ptx_diff
    PRIVATE
        cumetal_ptx_codegen
)

target_compile_options(ptx_diff
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(cumetal_bench
    tools/cumetal_bench/main.cpp
)

target_link_libraries(cumetal_bench
    PRIVATE
        cumetal_runtime
)

target_compile_options(cumetal_bench
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(CUMETAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/air_abi)
    add_subdirectory(tests/conformance)
    add_subdirectory(tests/functional)
    add_subdirectory(tests/ptx_sweep)
    add_subdirectory(tests/unit)

    # Phase 5 performance benchmark tests (skipped if xcrun is unavailable).
    # Uses --max-ratio=2.0 to enforce the spec §5.7 / §10.6 gate: CuMetal GPU time must
    # be within 2x of native Metal GPU time for all memory-bound kernels.
    if(APPLE)
        add_test(
            NAME bench_phase5_all_kernels
            COMMAND bash
                ${CMAKE_SOURCE_DIR}/scripts/run_bench_phase5.sh
                $<TARGET_FILE:cumetal_bench>
                ${CMAKE_SOURCE_DIR}/scripts/generate_bench_metallib.sh
                ${CMAKE_CURRENT_BINARY_DIR}/bench_phase5
        )
        set_tests_properties(bench_phase5_all_kernels PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

install(TARGETS
    cumetal_runtime
    air_inspect
    air_validate
    cumetal-air-emitter
    cumetalc
    cumetal-ptx2llvm
    ptx_diff
    cumetal_bench
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(APPLE)
    install(CODE [[
        file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/lib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcublas.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcurand.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcufft.dylib")
    ]])

    if(CUMETAL_ENABLE_BINARY_SHIM)
        install(CODE [[
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E create_symlink
                        "libcumetal.dylib"
                        "${CMAKE_INSTALL_PREFIX}/lib/libcuda.dylib")
        ]])
    endif()
endif()

install(FILES
    runtime/api/cublas_v2.h
    runtime/api/cuda.h
    runtime/api/cuda_fp16.h
    runtime/api/cuda_runtime.h
    runtime/api/cuda_runtime_api.h
    runtime/api/cufft.h
    runtime/api/curand.h
    runtime/api/cooperative_groups.h
    DESTINATION include
)

install(FILES
    runtime/api/cooperative_groups/reduce.h
    DESTINATION include/cooperative_groups
)
