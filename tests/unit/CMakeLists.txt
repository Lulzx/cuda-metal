add_executable(cumetal_allocation_table_test
    allocation_table_test.cpp
    ${CMAKE_SOURCE_DIR}/runtime/rt/allocation_table.cpp
)

add_executable(cumetal_module_cache_test
    module_cache_test.cpp
    ${CMAKE_SOURCE_DIR}/runtime/cache/module_cache.cpp
)

add_executable(cumetal_library_conflict_test
    library_conflict_test.cpp
    ${CMAKE_SOURCE_DIR}/runtime/error/library_conflict.cpp
)

add_executable(cumetal_metallib_parser_test
    metallib_parser_test.cpp
)

add_executable(cumetal_ptx_parser_test
    ptx_parser_test.cpp
)

add_executable(cumetal_ir_test
    ir_test.cpp
)

add_executable(cumetal_msl_ast_test
    msl_ast_test.cpp
)

add_executable(cumetal_ptx_ir_msl_test
    ptx_ir_msl_test.cpp
)

add_executable(cumetal_nvvm_ir_msl_test
    nvvm_ir_msl_test.cpp
)

add_executable(cumetal_native_registration_test
    native_registration_test.cpp
)

add_executable(cumetal_registration_arg_index_test
    registration_arg_index_test.cpp
)

add_executable(cumetal_intrinsic_lower_test
    intrinsic_lower_test.cpp
)

add_executable(cumetal_printf_lower_test
    printf_lower_test.cpp
)

add_executable(cumetal_addrspace_pass_test
    addrspace_pass_test.cpp
)

add_executable(cumetal_metadata_pass_test
    metadata_pass_test.cpp
)

add_executable(cumetal_phase1_pipeline_test
    phase1_pipeline_test.cpp
)

add_executable(cumetal_ptx_lower_to_llvm_test
    ptx_lower_to_llvm_test.cpp
)

add_executable(cumetal_ptx_lower_to_metal_test
    ptx_lower_to_metal_test.cpp
)

add_executable(cumetal_cuda_fp16_host_test
    cuda_fp16_host_test.cpp
)

add_executable(cumetal_cuda_vector_types_test
    cuda_vector_types_test.cpp
)

add_executable(cumetal_cuda_host_qualifiers_test
    cuda_host_qualifiers_test.cpp
)

add_executable(cumetal_threadgroup_tiling_test
    threadgroup_tiling_test.cpp
)

target_include_directories(cumetal_allocation_table_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/api
        ${CMAKE_SOURCE_DIR}/runtime/rt
        ${CMAKE_SOURCE_DIR}/runtime/metal_backend
)

target_include_directories(cumetal_module_cache_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/cache
)

target_include_directories(cumetal_library_conflict_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/error
)

target_link_libraries(cumetal_metallib_parser_test
    PRIVATE
        cumetal_common
)

target_link_libraries(cumetal_ptx_parser_test
    PRIVATE
        cumetal_ptx
)

target_link_libraries(cumetal_ir_test
    PRIVATE
        cumetal_ir
)

target_link_libraries(cumetal_msl_ast_test
    PRIVATE
        cumetal_metal_codegen
)

target_link_libraries(cumetal_ptx_ir_msl_test
    PRIVATE
        cumetal_metal_codegen
)

target_link_libraries(cumetal_nvvm_ir_msl_test
    PRIVATE
        cumetal_metal_codegen
)

target_link_libraries(cumetal_native_registration_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_registration_arg_index_test
    PRIVATE
        cumetal_runtime
)

target_include_directories(cumetal_registration_arg_index_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/registration
)

target_link_libraries(cumetal_intrinsic_lower_test
    PRIVATE
        cumetal_passes
)

target_link_libraries(cumetal_printf_lower_test
    PRIVATE
        cumetal_passes
)

target_link_libraries(cumetal_addrspace_pass_test
    PRIVATE
        cumetal_passes
)

target_link_libraries(cumetal_metadata_pass_test
    PRIVATE
        cumetal_passes
)

target_link_libraries(cumetal_phase1_pipeline_test
    PRIVATE
        cumetal_passes
)

target_link_libraries(cumetal_ptx_lower_to_llvm_test
    PRIVATE
        cumetal_ptx_codegen
)

target_link_libraries(cumetal_ptx_lower_to_metal_test
    PRIVATE
        cumetal_ptx_codegen
)

target_include_directories(cumetal_cuda_fp16_host_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/api
)

target_include_directories(cumetal_cuda_vector_types_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/api
)

target_include_directories(cumetal_cuda_host_qualifiers_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/api
)

target_compile_options(cumetal_allocation_table_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_module_cache_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_library_conflict_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_metallib_parser_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_ptx_parser_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_ir_test PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(cumetal_msl_ast_test PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(cumetal_ptx_ir_msl_test PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(cumetal_nvvm_ir_msl_test PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(cumetal_native_registration_test PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(cumetal_registration_arg_index_test PRIVATE -Wall -Wextra -Wpedantic)

target_compile_options(cumetal_intrinsic_lower_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_printf_lower_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_addrspace_pass_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_metadata_pass_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_phase1_pipeline_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_ptx_lower_to_llvm_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_ptx_lower_to_metal_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_cuda_fp16_host_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_cuda_vector_types_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_link_libraries(cumetal_threadgroup_tiling_test
    PRIVATE
        cumetal_passes
)

target_compile_options(cumetal_threadgroup_tiling_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_test(
    NAME unit_allocation_table
    COMMAND cumetal_allocation_table_test
)

add_test(
    NAME unit_module_cache
    COMMAND cumetal_module_cache_test
)

add_test(
    NAME unit_library_conflict
    COMMAND cumetal_library_conflict_test
)

# Generate the experimental metallib reference fixture from the .ll source.
# This uses the cumetal-air-emitter tool built as part of the project.
set(_exp_metallib "${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.experimental.metallib")
set(_air_ll "${CMAKE_SOURCE_DIR}/tests/air_abi/reference/vector_add_air.ll")

add_custom_command(
    OUTPUT  "${_exp_metallib}"
    COMMAND $<TARGET_FILE:cumetal-air-emitter>
            --mode experimental
            --input  "${_air_ll}"
            --output "${_exp_metallib}"
            --overwrite
    DEPENDS cumetal-air-emitter "${_air_ll}"
    COMMENT "Generating reference.experimental.metallib"
    VERBATIM
)
add_custom_target(generate_experimental_metallib ALL DEPENDS "${_exp_metallib}")

add_test(
    NAME unit_metallib_parser
    COMMAND cumetal_metallib_parser_test
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
            ${_exp_metallib}
)

add_test(
    NAME unit_ptx_parser
    COMMAND cumetal_ptx_parser_test
)

add_test(NAME unit_cumetal_ir COMMAND cumetal_ir_test)
add_test(NAME unit_msl_ast COMMAND cumetal_msl_ast_test)
add_test(NAME unit_ptx_ir_msl COMMAND cumetal_ptx_ir_msl_test)
add_test(NAME unit_nvvm_ir_msl COMMAND cumetal_nvvm_ir_msl_test)
add_test(NAME unit_native_registration COMMAND cumetal_native_registration_test)
add_test(NAME unit_registration_arg_index COMMAND cumetal_registration_arg_index_test)
add_test(
    NAME unit_cumetalc_shared_ir
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_shared_ir_test.sh
            $<TARGET_FILE:cumetalc>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/vector_add.ptx
            ${CMAKE_SOURCE_DIR}/tests/cuda_projects/vector_add/vector_add.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/unsupported_shared_ir.ptx
)

add_test(
    NAME unit_intrinsic_lower
    COMMAND cumetal_intrinsic_lower_test
)

add_test(
    NAME unit_printf_lower
    COMMAND cumetal_printf_lower_test
)

add_test(
    NAME unit_addrspace_pass
    COMMAND cumetal_addrspace_pass_test
)

add_test(
    NAME unit_metadata_pass
    COMMAND cumetal_metadata_pass_test
)

add_test(
    NAME unit_phase1_pipeline
    COMMAND cumetal_phase1_pipeline_test
)

add_test(
    NAME unit_ptx_lower_to_llvm
    COMMAND cumetal_ptx_lower_to_llvm_test
)

add_test(
    NAME unit_ptx_lower_to_metal
    COMMAND cumetal_ptx_lower_to_metal_test
)

add_test(
    NAME unit_cuda_fp16_host
    COMMAND cumetal_cuda_fp16_host_test
)

add_test(
    NAME unit_cuda_vector_types
    COMMAND cumetal_cuda_vector_types_test
)

add_test(
    NAME unit_cuda_host_qualifiers
    COMMAND cumetal_cuda_host_qualifiers_test
)

add_test(
    NAME unit_threadgroup_tiling
    COMMAND cumetal_threadgroup_tiling_test
)

add_test(
    NAME unit_cumetal_bench_help
    COMMAND $<TARGET_FILE:cumetal_bench> --help
)

add_test(
    NAME unit_cumetal_bench_invalid_arg
    COMMAND $<TARGET_FILE:cumetal_bench> --definitely-invalid-flag
)
set_tests_properties(unit_cumetal_bench_invalid_arg PROPERTIES WILL_FAIL TRUE)

add_test(
    NAME unit_cumetal_bench_ratio_gate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_bench_ratio_gate_test.sh
            ${CMAKE_BINARY_DIR}
            $<TARGET_FILE:cumetal_bench>
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)
set_tests_properties(unit_cumetal_bench_ratio_gate PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME unit_install_uninstall_scripts
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_install_uninstall_test.sh
            ${CMAKE_SOURCE_DIR}/install/install.sh
            ${CMAKE_SOURCE_DIR}/install/uninstall.sh
            ${CMAKE_BINARY_DIR}
)

add_test(
    NAME unit_runtime_library_aliases
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_library_aliases_test.sh
            $<TARGET_FILE:cumetal_runtime>
            $<TARGET_FILE_DIR:cumetal_runtime>/libcublas.dylib
            $<TARGET_FILE_DIR:cumetal_runtime>/libcurand.dylib
)

if(CUMETAL_ENABLE_BINARY_SHIM)
    add_test(
        NAME unit_binary_shim_library_alias
        COMMAND bash
                ${CMAKE_CURRENT_SOURCE_DIR}/run_binary_shim_library_alias_test.sh
                $<TARGET_FILE:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcuda.dylib
    )

    add_test(
        NAME unit_binary_shim_link_alias
        COMMAND bash
                ${CMAKE_CURRENT_SOURCE_DIR}/run_cuda_link_alias_test.sh
                ${CMAKE_CXX_COMPILER}
                ${CMAKE_SOURCE_DIR}/runtime/api
                $<TARGET_FILE_DIR:cumetal_runtime>
                ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(unit_binary_shim_link_alias PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME unit_binary_shim_symbol_exports
        COMMAND bash
                ${CMAKE_CURRENT_SOURCE_DIR}/run_binary_shim_symbol_exports_test.sh
                $<TARGET_FILE:cumetal_runtime>
    )
    set_tests_properties(unit_binary_shim_symbol_exports PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_test(
    NAME unit_library_link_aliases
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_library_link_aliases_test.sh
            ${CMAKE_CXX_COMPILER}
            ${CMAKE_SOURCE_DIR}/runtime/api
            $<TARGET_FILE_DIR:cumetal_runtime>
            ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(unit_library_link_aliases PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME unit_cusparse_cusolver_symlinks
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/cusolver_link_test.sh
            $<TARGET_FILE_DIR:cumetal_runtime>
)
