add_test(
    NAME ptx_sweep_supported_ops
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_supported_ops.sh
            $<TARGET_FILE:cumetal-ptx2llvm>
            ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
    NAME ptx_sweep_unsupported_ops
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_unsupported_ops.sh
            $<TARGET_FILE:cumetal-ptx2llvm>
            ${CMAKE_CURRENT_BINARY_DIR}
)

# Numerical per-opcode sweep (spec §10.2). Unlike the lowering-only sweeps above, this executes
# each opcode on the GPU and compares bit-for-bit against a hand-derived PTX ISA oracle.
add_executable(cumetal_ptx_sweep_numeric_test
    ptx_sweep_numeric_test.cpp
)

target_link_libraries(cumetal_ptx_sweep_numeric_test
    PRIVATE
        cumetal_runtime
)

target_include_directories(cumetal_ptx_sweep_numeric_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/api
)

target_compile_options(cumetal_ptx_sweep_numeric_test PRIVATE -Wall -Wextra -Wpedantic)

add_test(
    NAME ptx_sweep_numeric
    COMMAND cumetal_ptx_sweep_numeric_test
            $<TARGET_FILE:cumetalc>
            ${CMAKE_CURRENT_BINARY_DIR}/ptx_sweep_numeric
)

set_tests_properties(ptx_sweep_numeric PROPERTIES SKIP_RETURN_CODE 77)
