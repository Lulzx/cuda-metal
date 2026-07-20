if(NOT APPLE)
    message(STATUS "Skipping air_abi tests: Apple platform required")
    return()
endif()

add_executable(cumetal_metal_load_test
    metallib_load_test.mm
)

target_link_libraries(cumetal_metal_load_test
    PRIVATE
        "-framework Foundation"
        "-framework Metal"
)

target_compile_options(cumetal_metal_load_test
    PRIVATE
        -fobjc-arc
        -Wall
        -Wextra
)

add_test(
    NAME air_abi_metal_load
    COMMAND cumetal_metal_load_test ${CMAKE_CURRENT_SOURCE_DIR}/reference/reference.metallib
)

set_tests_properties(air_abi_metal_load PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME air_abi_emit_validate_experimental
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_experimental_validation.sh
            $<TARGET_FILE:cumetal-air-emitter>
            $<TARGET_FILE:air_validate>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add_air.ll
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.experimental.metallib
)

add_test(
    NAME air_abi_validate_reference
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_reference_validation.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:air_validate>
            $<TARGET_FILE:air_inspect>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/reference.metallib
)

add_test(
    NAME air_abi_emit_validate_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_xcrun_emitter_validation.sh
            $<TARGET_FILE:cumetal-air-emitter>
            $<TARGET_FILE:air_validate>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.xcrun.metallib
)

add_test(
    NAME air_abi_emit_load_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_xcrun_emit_and_load.sh
            $<TARGET_FILE:cumetal-air-emitter>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.xcrun.load.metallib
)

add_test(
    NAME air_abi_cumetalc_emit_load_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_xcrun_emit_and_load.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cumetalc.xcrun.load.metallib
)

add_test(
    NAME air_abi_cumetalc_positional_emit_load_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_positional_emit_and_load.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cumetalc.positional.xcrun.load.metallib
)

add_test(
    NAME air_abi_cumetalc_default_output_emit_load_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_default_output_emit_and_load.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
    NAME air_abi_multikernel_emit_validate_load_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_multikernel_emit_validate_and_load.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:air_validate>
            $<TARGET_FILE:air_inspect>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/multi_kernel.metal
            ${CMAKE_CURRENT_BINARY_DIR}/multi_kernel.cumetalc.xcrun.load.metallib
)

add_test(
    NAME air_abi_validate_negative
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_negative_validation.sh
            $<TARGET_FILE:air_validate>
            ${CMAKE_CURRENT_BINARY_DIR}/negative
)

add_test(
    NAME air_abi_ptx_to_experimental_validate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_ptx_to_experimental_validation.sh
            $<TARGET_FILE:cumetal-ptx2llvm>
            $<TARGET_FILE:cumetal-air-emitter>
            $<TARGET_FILE:air_validate>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.ptx
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.from_ptx.ll
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.from_ptx.experimental.metallib
)

add_test(
    NAME air_abi_matrix_ptx_to_experimental_validate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_matrix_ptx_to_experimental_validation.sh
            $<TARGET_FILE:cumetal-ptx2llvm>
            $<TARGET_FILE:cumetal-air-emitter>
            $<TARGET_FILE:air_validate>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul.ptx
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul.from_ptx.ll
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul.from_ptx.experimental.metallib
)

add_test(
    NAME air_abi_cumetalc_ptx_experimental_validate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_ptx_experimental_validate.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:air_validate>
            $<TARGET_FILE:air_inspect>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.ptx
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cumetalc.from_ptx.experimental.metallib
)

add_test(
    NAME air_abi_cumetalc_matrix_ptx_experimental_validate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_matrix_ptx_experimental_validate.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:air_validate>
            $<TARGET_FILE:air_inspect>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul.ptx
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul.cumetalc.from_ptx.experimental.metallib
)

add_test(
    NAME air_abi_cumetalc_cu_experimental_validate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_cu_experimental_validate.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:air_validate>
            $<TARGET_FILE:air_inspect>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.cu
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cumetalc.from_cu.experimental.metallib
)

add_test(
    NAME air_abi_cumetalc_cuda_device_frontend_validate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_cuda_device_frontend_validate.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:air_validate>
            $<TARGET_FILE:air_inspect>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/cuda_device_frontend.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/reference
            ${CMAKE_CURRENT_BINARY_DIR}/cuda_device_frontend.experimental.metallib
)

add_test(
    NAME air_abi_cumetalc_cu_default_output_validate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_cu_default_output_validate.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:air_validate>
            $<TARGET_FILE:air_inspect>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.cu
            ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
    NAME air_abi_cumetalc_cu_emit_load_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_cu_emit_and_load_xcrun.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.cu
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cumetalc.from_cu.xcrun.load.metallib
)

add_test(
    NAME air_abi_cumetalc_ptx_default_output_validate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_ptx_default_output_validate.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:air_validate>
            $<TARGET_FILE:air_inspect>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.ptx
            ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
    NAME air_abi_cumetalc_ptx_emit_load_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_ptx_emit_and_load_xcrun.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.ptx
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cumetalc.from_ptx.xcrun.load.metallib
)

add_test(
    NAME air_abi_cumetalc_matrix_ptx_emit_load_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_ptx_emit_and_load_xcrun.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul.ptx
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul.cumetalc.from_ptx.xcrun.load.metallib
            matrix_mul
)

add_test(
    NAME air_abi_ptx2llvm_positional_default_output
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_ptx2llvm_positional_default_output.sh
            $<TARGET_FILE:cumetal-ptx2llvm>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.ptx
            ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
    NAME air_abi_xcode_matrix_regression
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_xcode_matrix_regression.sh
            $<TARGET_FILE:air_inspect>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}/xcode-matrix
)

set_tests_properties(
    air_abi_validate_reference
    air_abi_emit_validate_xcrun
    air_abi_emit_load_xcrun
    air_abi_cumetalc_emit_load_xcrun
    air_abi_cumetalc_positional_emit_load_xcrun
    air_abi_cumetalc_default_output_emit_load_xcrun
    air_abi_multikernel_emit_validate_load_xcrun
    air_abi_cumetalc_ptx_emit_load_xcrun
    air_abi_cumetalc_matrix_ptx_emit_load_xcrun
    air_abi_cumetalc_cu_experimental_validate
    air_abi_cumetalc_cuda_device_frontend_validate
    air_abi_cumetalc_cu_default_output_validate
    air_abi_cumetalc_cu_emit_load_xcrun
    air_abi_xcode_matrix_regression
    PROPERTIES
        SKIP_RETURN_CODE 77
)
