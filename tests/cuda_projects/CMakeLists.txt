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
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

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
