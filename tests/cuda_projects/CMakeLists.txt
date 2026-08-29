if(NOT APPLE)
    return()
endif()

function(cumetal_add_cuda_project_test test_name project_subdir source_cu binary_name)
    add_test(
        NAME ${test_name}
        COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_standalone_cu.sh
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${project_subdir}
            ${source_cu}
            ${binary_name}
    )
    set_tests_properties(
        ${test_name}
        PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
endfunction()

add_test(
    NAME functional_cuda_source_gpu_vector_add
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_cuda_source_gpu.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(
    functional_cuda_source_gpu_vector_add
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)

# Full upstream cuda-samples sweep, gated against a recorded baseline. Skips when
# no cuda-samples checkout is present. See cuda_samples_sweep_manifest.txt.
add_test(
    NAME conformance_cuda_samples_sweep
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_cuda_samples_sweep.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(
    conformance_cuda_samples_sweep
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        TIMEOUT 2400
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)

add_test(
    NAME unit_cuda_samples_sweep_status
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_sweep_status.sh
)

add_test(
    NAME conformance_compiler_backend_matrix
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_backend_matrix.sh
        ${CMAKE_SOURCE_DIR}
        $<TARGET_FILE:cumetalc>
        ${CMAKE_CURRENT_SOURCE_DIR}/backend_matrix_manifest.txt
)
set_tests_properties(
    conformance_compiler_backend_matrix
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        TIMEOUT 300
)

add_test(
    NAME conformance_compiler_backend_matrix_versions
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_backend_matrix_versions.sh
        ${CMAKE_SOURCE_DIR}
        $<TARGET_FILE:cumetalc>
        ${CMAKE_CURRENT_SOURCE_DIR}/backend_matrix_manifest.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/run_backend_matrix.sh
)
set_tests_properties(
    conformance_compiler_backend_matrix_versions
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        TIMEOUT 1200
)

add_test(
    NAME conformance_cuda_samples_vectoradd_gpu
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_cuda_samples_vectoradd_gpu.sh
        ${CMAKE_SOURCE_DIR}
)
set_tests_properties(
    conformance_cuda_samples_vectoradd_gpu
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)

# Clang does not pass device printf arguments directly in the PTX call tuple: it
# emits an initialized module-global format and a packed local vprintf tuple.
# Verify that exact source-frontend ABI, including a multidimensional launch.
add_test(
    NAME functional_cuda_projects_device_printf_clang
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_device_printf_clang.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(
    functional_cuda_projects_device_printf_clang
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        TIMEOUT 60
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)

add_test(
    NAME functional_cuda_projects_device_printf_clang_typed_ptx
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_device_printf_clang.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        cumetal-ir
)
set_tests_properties(
    functional_cuda_projects_device_printf_clang_typed_ptx
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        TIMEOUT 60
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)

# Regression: LLVM `fence` crashed Apple's AIR backend at pipeline creation (so every
# atomicCAS-bearing kernel silently did nothing), and scalar __shared__ objects were
# counted as zero bytes when sizing threadgroup memory.
cumetal_add_cuda_project_test(
    functional_cuda_projects_device_sync_primitives
    device_sync_primitives device_sync_primitives.cu device_sync_primitives
)

# Regression: __cudaRegisterVar used to map a __device__/__constant__ symbol onto
# its device-side name string, and external PTX symbols were not bound into
# kernels. The project checks symbol APIs, a >4 KB constant-buffer read, and
# persistent writable device-global state across two launches.
cumetal_add_cuda_project_test(
    functional_cuda_projects_constant_symbol
    constant_symbol constant_symbol.cu constant_symbol
)

# libdevice binary64 frexp is implemented with IEEE bit decomposition so it
# remains available when native Metal double ALU is unavailable.
cumetal_add_cuda_project_test(
    functional_cuda_projects_libdevice_frexp
    libdevice_frexp libdevice_frexp.cu libdevice_frexp
)

cumetal_add_cuda_project_test(
    functional_cuda_projects_system_atomics
    system_atomics system_atomics.cu system_atomics
)

# 64-bit atomics. Metal has none at any width, so each one is serialized behind
# a bank of 32-bit locks. The contended cases are the point: a GPU lock either
# hands over or hangs the device, and nothing short of running it says which.
cumetal_add_cuda_project_test(
    functional_cuda_projects_wide_atomics
    wide_atomics wide_atomics.cu wide_atomics
)

add_test(
    NAME functional_cuda_projects_wide_atomics_typed_ptx
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_standalone_cu.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        wide_atomics
        wide_atomics.cu
        wide_atomics_typed_ptx
        cumetal-ir
)
set_tests_properties(
    functional_cuda_projects_wide_atomics_typed_ptx
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        TIMEOUT 120
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)

cumetal_add_cuda_project_test(
    functional_cuda_projects_cooperative_thread_group
    cooperative_thread_group cooperative_thread_group.cu cooperative_thread_group
)

cumetal_add_cuda_project_test(
    functional_cuda_projects_dynamic_cooperative_groups
    dynamic_cooperative_groups dynamic_cooperative_groups.cu dynamic_cooperative_groups
)

# CUDA warp code commonly initializes a scalar only in the lanes later selected
# by a shuffle.  Clang must not use the other lanes' undefined scalar values to
# infer a false threadIdx.x range and erase a subsequent guarded store.
cumetal_add_cuda_project_test(
    functional_cuda_projects_partial_lane_shuffle_guard
    partial_lane_shuffle_guard partial_lane_shuffle_guard.cu partial_lane_shuffle_guard
)

cumetal_add_cuda_project_test(
    functional_cuda_projects_sgemm_naive
    sgemm sgemm_naive.cu sgemm_naive
)
cumetal_add_cuda_project_test(
    functional_cuda_projects_sgemm_shmem
    sgemm sgemm_shmem.cu sgemm_shmem
)
cumetal_add_cuda_project_test(
    functional_cuda_projects_sgemm_2d
    sgemm sgemm_2d.cu sgemm_2d
)
cumetal_add_cuda_project_test(
    functional_cuda_projects_reduction
    reduction reduction_standalone.cu reduction_standalone
)
cumetal_add_cuda_project_test(
    functional_cuda_projects_transpose
    transpose transpose_standalone.cu transpose_standalone
)

# A blit and a compute dispatch in one command buffer are ordered by Metal only
# for hazard-tracked resources, and heap allocations are not tracked. This is
# the shape that caught it: copy in, increment, copy out, above the heap
# threshold.
cumetal_add_cuda_project_test(
    functional_cuda_projects_async_memcpy_hazard
    async_memcpy async_memcpy_hazard.cu async_memcpy_hazard
)

# Measures every mapped libdevice math function against the host libm. Strict:
# a mis-mapped builtin computes the wrong function silently, and a newly
# unsupported one means a lowering regression.
add_test(
    NAME functional_cuda_projects_libdevice_math
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_strict_standalone_cu.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        libdevice
        libdevice_math.cu
        libdevice_math
        "PASS: libdevice float surface probe"
)
set_tests_properties(
    functional_cuda_projects_libdevice_math
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)

# Ray Tracing in One Weekend. Strict runner on purpose: this kernel is known to
# lower, so a future "registered kernel missing metallib" is a regression, not a
# coverage gap, and must not be downgraded to a skip.
add_test(
    NAME functional_cuda_projects_raytracer
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_strict_standalone_cu.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        raytracer
        rtiow.cu
        rtiow
        "OK: GPU render matches CPU reference."
)
set_tests_properties(
    functional_cuda_projects_raytracer
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)

# FP64 emulation contract. Metal has no `double`, so FP64 arithmetic runs on
# Dekker FP32 pairs while storage stays IEEE-754 binary64. Strict runner: this
# pins the measured ~48-bit significand, the split/pack idempotence that keeps
# error from compounding, and special-value behaviour. It exists because "the
# kernel compiled and ran" says nothing about how many significand bits
# survived -- the previous implementation dropped the low limb at every register
# write and ran at ~24 bits, which compiles and runs perfectly while quietly
# turning an LP solver's residuals into noise.
add_test(
    NAME functional_cuda_projects_fp64_precision
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_strict_standalone_cu.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        fp64
        fp64_precision.cu
        fp64_precision
        "PASS: fp64 emulation meets the ~48-bit significand contract"
)
set_tests_properties(
    functional_cuda_projects_fp64_precision
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)

add_test(
    NAME functional_cuda_projects_ggml_output_ops
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_strict_standalone_cu.sh
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ggml_output_ops
        ggml_output_ops.cu
        ggml_output_ops
)
set_tests_properties(
    functional_cuda_projects_ggml_output_ops
    PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR};CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=1"
)

# Canonical typed-PTX numerical closure gate. The manifest is exact over every
# in-tree CUDA project, each project gets a fresh runtime cache, unsupported
# lowering is a hard failure, and name-selected workload specializations stay
# disabled so this measures the compiler rather than compatibility shortcuts.
find_program(CUMETAL_CUDA_PROJECTS_PYTHON3 python3)
if(CUMETAL_CUDA_PROJECTS_PYTHON3)
    add_test(
        NAME conformance_cuda_projects_typed_ptx_corpus
        COMMAND ${CUMETAL_CUDA_PROJECTS_PYTHON3}
            ${CMAKE_CURRENT_SOURCE_DIR}/sweep_cuda_projects.py
            --manifest ${CMAKE_CURRENT_SOURCE_DIR}/coverage_manifest.json
            --build-dir ${CMAKE_CURRENT_BINARY_DIR}/typed-ptx-corpus
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/typed-ptx-results
            --backend cumetal-ir
            --require-all-pass
            --timeout 300
    )
    set_tests_properties(
        conformance_cuda_projects_typed_ptx_corpus
        PROPERTIES
            RUN_SERIAL TRUE
            TIMEOUT 3600
            ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR};CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0"
    )

    if(CUMETAL_ENABLE_CUDA_REGISTRATION)
        # Canonical direct-source proof: every reviewed fixture is compiled to
        # a self-contained native executable, executes its CPU oracle on Apple
        # GPU, and is rejected if it falls back to first-launch PTX JIT.
        add_test(
            NAME conformance_cuda_projects_native_aot_corpus
            COMMAND ${CUMETAL_CUDA_PROJECTS_PYTHON3}
                ${CMAKE_CURRENT_SOURCE_DIR}/sweep_cuda_projects.py
                --manifest ${CMAKE_CURRENT_SOURCE_DIR}/coverage_manifest.json
                --build-dir ${CMAKE_CURRENT_BINARY_DIR}/native-aot-corpus
                --output-dir ${CMAKE_CURRENT_BINARY_DIR}/native-aot-results
                --backend native-aot
                --cumetalc $<TARGET_FILE:cumetalc>
                --require-all-pass
                --timeout 300
        )
        set_tests_properties(
            conformance_cuda_projects_native_aot_corpus
            PROPERTIES
                RUN_SERIAL TRUE
                TIMEOUT 3600
                ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR};CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0"
        )

        add_test(
            NAME functional_cumetalc_native_aot_device_printf
            COMMAND bash
                ${CMAKE_SOURCE_DIR}/tests/functional/run_cumetalc_native_device_printf.sh
                $<TARGET_FILE:cumetalc>
                ${CMAKE_CURRENT_SOURCE_DIR}/device_printf_clang/device_printf_clang.cu
                ${CMAKE_CURRENT_BINARY_DIR}/native-aot-device-printf
        )
        set_tests_properties(
            functional_cumetalc_native_aot_device_printf
            PROPERTIES
                RUN_SERIAL TRUE
                TIMEOUT 120
        )
    endif()
endif()
