cmake_minimum_required(VERSION 3.28)

project(cumetal VERSION 1.0.0 LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

option(CUMETAL_BUILD_TESTS "Build test targets" ON)

# Clang launches ptxas/fatbinary directly while compiling CUDA source. Native
# shims avoid macOS provenance policy killing interpreter scripts when they are
# spawned as compiler subprocesses.
add_executable(cumetal_ptxas_shim
    tools/cuda_toolchain_shims/ptxas.cpp
)
set_target_properties(cumetal_ptxas_shim PROPERTIES
    OUTPUT_NAME ptxas
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda_toolchain
)

add_executable(cumetal_fatbinary_shim
    tools/cuda_toolchain_shims/fatbinary.cpp
)
set_target_properties(cumetal_fatbinary_shim PROPERTIES
    OUTPUT_NAME fatbinary
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda_toolchain
)

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

add_library(cumetal_ir STATIC
    compiler/ir/src/ir.cpp
    compiler/ir/src/nvvm_importer.cpp
    compiler/ir/src/ptx_importer.cpp
)

target_include_directories(cumetal_ir
    PUBLIC
        compiler/ir/include
)

target_link_libraries(cumetal_ir
    PUBLIC
        cumetal_ptx
)

find_package(LLVM CONFIG QUIET
    HINTS
        /opt/homebrew/opt/llvm/lib/cmake/llvm
        /usr/local/opt/llvm/lib/cmake/llvm
)

if(LLVM_FOUND AND LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL 18)
    message(STATUS "CuMetal NVVM importer enabled with LLVM ${LLVM_PACKAGE_VERSION}")
    llvm_map_components_to_libnames(CUMETAL_LLVM_LIBS
        Core
        IRReader
        Support
    )
    target_include_directories(cumetal_ir SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(cumetal_ir PUBLIC ${CUMETAL_LLVM_LIBS})
    target_compile_definitions(cumetal_ir PUBLIC CUMETAL_HAVE_LLVM=1)
else()
    message(WARNING "LLVM 18+ not found; genuine CUDA/NVVM import will be unavailable")
    target_compile_definitions(cumetal_ir PUBLIC CUMETAL_HAVE_LLVM=0)
endif()

target_compile_options(cumetal_ir
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_metal_codegen STATIC
    compiler/metal/src/msl_ast.cpp
    compiler/metal/src/lower_to_msl.cpp
)

target_include_directories(cumetal_metal_codegen
    PUBLIC
        compiler/metal/include
)

target_link_libraries(cumetal_metal_codegen
    PUBLIC
        cumetal_ir
)

target_compile_options(cumetal_metal_codegen
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
        cumetal_metal_codegen
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
    runtime/registration/native_registration.cpp
    ${CUMETAL_REGISTRATION_SOURCE}
    runtime/rt/allocation_table.cpp
    runtime/rt/cublas.cpp
    runtime/rt/cufft.cpp
    runtime/rt/curand.cpp
    runtime/rt/cusparse.cpp
    runtime/rt/cusolver.cpp
    runtime/rt/cublaslt.cpp
    runtime/rt/cudnn.cpp
    runtime/rt/nvml.cpp
    runtime/rt/nccl.cpp
    runtime/rt/cuda_runtime.cpp
    runtime/rt/hash_shim.cpp
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
        runtime/common
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
        # Files produced inside a quarantined/provenance-marked checkout can
        # inherit com.apple.provenance. Re-sign after removing it so dyld does
        # not reject an otherwise valid local development dylib.
        COMMAND ${CMAKE_COMMAND}
                -DCUMETAL_BINARY=$<TARGET_FILE:cumetal_runtime>
                -P
                ${CMAKE_SOURCE_DIR}/cmake/prepare_macos_binary.cmake
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcublas.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcurand.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcufft.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcusparse.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcusolver.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcublasLt.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcudnn.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libnvidia-ml.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libnccl.dylib
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

if(APPLE)
    foreach(cumetal_tool_target IN ITEMS
            cumetal_ptxas_shim
            cumetal_fatbinary_shim
            air_validate
            cumetal-air-emitter
            cumetalc
            cumetal-ptx2llvm
            air_inspect
            ptx_diff
            cumetal_bench)
        add_custom_command(
            TARGET ${cumetal_tool_target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                    -DCUMETAL_BINARY=$<TARGET_FILE:${cumetal_tool_target}>
                    -P
                    ${CMAKE_SOURCE_DIR}/cmake/prepare_macos_binary.cmake
        )
    endforeach()
endif()

if(CUMETAL_BUILD_TESTS)
    enable_testing()
    include(cmake/CuMetalToolchain.cmake)
    cumetal_discover_metal_toolchain(CUMETAL_TEST_METAL_TOOLCHAIN)
    if(CUMETAL_TEST_METAL_TOOLCHAIN)
        message(STATUS
            "CTest Metal toolchain: ${CUMETAL_TEST_METAL_TOOLCHAIN}")
    endif()
    add_subdirectory(tests/air_abi)
    add_subdirectory(tests/conformance)
    add_subdirectory(tests/functional)
    add_subdirectory(tests/cuda_projects)
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

    cumetal_apply_metal_toolchain_to_tests(
        "${CUMETAL_TEST_METAL_TOOLCHAIN}"
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/tests/air_abi"
        "${CMAKE_SOURCE_DIR}/tests/conformance"
        "${CMAKE_SOURCE_DIR}/tests/functional"
        "${CMAKE_SOURCE_DIR}/tests/cuda_projects"
        "${CMAKE_SOURCE_DIR}/tests/ptx_sweep"
        "${CMAKE_SOURCE_DIR}/tests/unit"
    )
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
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcusparse.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcusolver.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcublasLt.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcudnn.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libnvidia-ml.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libnccl.dylib")
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
    runtime/api/cumetal_native.h
    runtime/api/cublas_v2.h
    runtime/api/cuda.h
    runtime/api/cuda_fp16.h
    runtime/api/cuda_runtime.h
    runtime/api/cuda_runtime_api.h
    runtime/api/cufft.h
    runtime/api/curand.h
    runtime/api/cusparse.h
    runtime/api/cusolver_common.h
    runtime/api/cusolverDn.h
    runtime/api/cusolverSp.h
    runtime/api/cublasLt.h
    runtime/api/cudnn.h
    runtime/api/nvml.h
    runtime/api/nccl.h
    runtime/api/cooperative_groups.h
    runtime/api/cuComplex.h
    runtime/api/device_launch_parameters.h
    runtime/api/driver_types.h
    runtime/api/library_types.h
    runtime/api/channel_descriptor.h
    runtime/api/device_atomic_functions.h
    runtime/api/math_functions.h
    runtime/api/cuda_profiler_api.h
    runtime/api/cuda_occupancy.h
    runtime/api/nvToolsExt.h
    runtime/api/sm_20_intrinsics.h
    runtime/api/sm_30_intrinsics.h
    runtime/api/sm_60_intrinsics.h
    runtime/api/sm_70_intrinsics.h
    runtime/api/sm_80_intrinsics.h
    runtime/api/cublas_api.h
    runtime/api/cuda_pipeline.h
    runtime/api/cuda_device_runtime_api.h
    runtime/api/curand_kernel.h
    runtime/api/cusparse_v2.h
    DESTINATION include
)

install(FILES
    runtime/api/nvtx3/nvToolsExt.h
    DESTINATION include/nvtx3
)

install(FILES
    runtime/api/cooperative_groups/reduce.h
    DESTINATION include/cooperative_groups
)

install(FILES
    runtime/api/thrust/thrust.h
    runtime/api/thrust/device_ptr.h
    runtime/api/thrust/device_vector.h
    runtime/api/thrust/host_vector.h
    runtime/api/thrust/sort.h
    runtime/api/thrust/reduce.h
    runtime/api/thrust/scan.h
    runtime/api/thrust/transform.h
    runtime/api/thrust/functional.h
    runtime/api/thrust/unique.h
    runtime/api/thrust/pair.h
    runtime/api/thrust/execution_policy.h
    DESTINATION include/thrust
)

install(FILES
    runtime/api/thrust/iterator/counting_iterator.h
    runtime/api/thrust/iterator/zip_iterator.h
    runtime/api/thrust/iterator/transform_iterator.h
    runtime/api/thrust/iterator/constant_iterator.h
    runtime/api/thrust/iterator/discard_iterator.h
    runtime/api/thrust/iterator/permutation_iterator.h
    DESTINATION include/thrust/iterator
)

install(FILES
    runtime/api/thrust/system/cuda/vector.h
    runtime/api/thrust/system/cuda/execution_policy.h
    DESTINATION include/thrust/system/cuda
)

install(FILES
    runtime/api/thrust/system/cuda/detail/par.h
    DESTINATION include/thrust/system/cuda/detail
)

install(FILES
    runtime/api/cub/cub.h
    DESTINATION include/cub
)

install(FILES
    runtime/api/cub/block/block_reduce.h
    runtime/api/cub/block/block_scan.h
    runtime/api/cub/block/block_exchange.h
    runtime/api/cub/block/block_load.h
    runtime/api/cub/block/block_store.h
    DESTINATION include/cub/block
)

install(FILES
    runtime/api/cub/warp/warp_reduce.h
    runtime/api/cub/warp/warp_scan.h
    DESTINATION include/cub/warp
)

install(FILES
    runtime/api/cub/device/device_reduce.h
    runtime/api/cub/device/device_scan.h
    runtime/api/cub/device/device_radix_sort.h
    runtime/api/cub/device/device_select.h
    runtime/api/cub/device/device_histogram.h
    runtime/api/cub/device/device_run_length_encode.h
    runtime/api/cub/device/device_partition.h
    runtime/api/cub/device/device_merge_sort.h
    DESTINATION include/cub/device
)
