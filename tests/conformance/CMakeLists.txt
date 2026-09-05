if(NOT APPLE)
    message(STATUS "Skipping conformance tests: Apple platform required")
    return()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    find_program(CUMETAL_XCRUN_EXECUTABLE xcrun)
    if(CUMETAL_XCRUN_EXECUTABLE)
        add_test(
            NAME conformance_physx_grb
            COMMAND bash
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_physx_grb.sh
        )
        set_tests_properties(conformance_physx_grb PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 1800
            ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
        )
        add_test(
            NAME conformance_physx_grb_friction
            COMMAND bash
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_physx_grb_friction.sh
        )
        set_tests_properties(conformance_physx_grb_friction PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 1800
            ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
        )
        add_test(
            NAME conformance_physx_grb_multibody
            COMMAND bash
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_physx_grb_multibody.sh
        )
        set_tests_properties(conformance_physx_grb_multibody PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 1800
            ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
        )
        add_test(
            NAME conformance_physx_grb_stacked
            COMMAND bash
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_physx_grb_stacked.sh
        )
        set_tests_properties(conformance_physx_grb_stacked PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 1800
            ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
        )
        add_test(
            NAME conformance_physx_grb_box
            COMMAND bash
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_physx_grb_box.sh
        )
        set_tests_properties(conformance_physx_grb_box PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 1800
            ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
        )
        add_test(
            NAME conformance_physx_grb_trimesh
            COMMAND bash
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_physx_grb_trimesh.sh
        )
        set_tests_properties(conformance_physx_grb_trimesh PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 1800
            ENVIRONMENT "CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
        )
    else()
        message(STATUS "Skipping PhysX GRB conformance registration (xcrun unavailable)")
    endif()
else()
    message(STATUS "Skipping PhysX GRB conformance registration (Apple Silicon required)")
endif()

add_test(
    NAME conformance_phase4_functional
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_conformance_suite.sh
            ${CMAKE_BINARY_DIR}
            90
            ${CMAKE_CURRENT_SOURCE_DIR}/phase4_functional_manifest.txt
)

add_test(
    NAME conformance_llama_cpp_output_gate
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/check_llama_cpp_output_test.sh
)

set_tests_properties(conformance_phase4_functional PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
)

add_test(
    NAME unit_phase4_gate_policy
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/check_phase4_gate_policy.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/run_conformance_suite.sh
)
set_tests_properties(unit_phase4_gate_policy PROPERTIES LABELS "hosted;unit")

# ── llama.cpp conformance ─────────────────────────────────────────────────────
# Auto-discover an existing llama.cpp checkout or build directory.
set(CUMETAL_LLAMA_DIR_ENV "$ENV{CUMETAL_LLAMA_DIR}")
set(CUMETAL_LLAMA_DIR_AUTO "")
foreach(_candidate
        "${CMAKE_SOURCE_DIR}/../llama.cpp"
        "${CMAKE_SOURCE_DIR}/llama.cpp")
    if(EXISTS "${_candidate}")
        set(CUMETAL_LLAMA_DIR_AUTO "${_candidate}")
        break()
    endif()
endforeach()

set(CUMETAL_LLAMA_DIR_ACTIVE "${CUMETAL_LLAMA_DIR_ENV}")
if(NOT CUMETAL_LLAMA_DIR_ACTIVE)
    set(CUMETAL_LLAMA_DIR_ACTIVE "${CUMETAL_LLAMA_DIR_AUTO}")
endif()

if((CUMETAL_LLAMA_DIR_ENV AND EXISTS "${CUMETAL_LLAMA_DIR_ENV}") OR CUMETAL_LLAMA_DIR_AUTO)
    add_test(
        NAME conformance_llama_cpp
        COMMAND bash
                ${CMAKE_CURRENT_SOURCE_DIR}/run_llama_cpp_cumetal.sh
    )

    set_tests_properties(conformance_llama_cpp PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        ENVIRONMENT
            "CUMETAL_LLAMA_DIR=${CUMETAL_LLAMA_DIR_ACTIVE};CUMETAL_LLAMA_BUILD_CMD=${CMAKE_SOURCE_DIR}/scripts/build_llama_cpp_cumetal.sh;CUMETAL_LLAMA_NGL=4;CUMETAL_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
else()
    message(STATUS "Skipping llama.cpp conformance registration (set CUMETAL_LLAMA_DIR to enable)")
endif()
