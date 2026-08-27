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

cumetal_add_cuda_project_test(
    functional_cuda_projects_cooperative_thread_group
    cooperative_thread_group cooperative_thread_group.cu cooperative_thread_group
)

cumetal_add_cuda_project_test(
    functional_cuda_projects_dynamic_cooperative_groups
    dynamic_cooperative_groups dynamic_cooperative_groups.cu dynamic_cooperative_groups
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
