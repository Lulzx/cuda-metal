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
        ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
)
