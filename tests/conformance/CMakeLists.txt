if(NOT APPLE)
    message(STATUS "Skipping conformance tests: Apple platform required")
    return()
endif()

add_test(
    NAME conformance_phase4_functional
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_conformance_suite.sh
            ${CMAKE_BINARY_DIR}
            90
            ^functional_
)

set_tests_properties(conformance_phase4_functional PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
)

set(CUMETAL_LLMC_DIR_ENV "$ENV{CUMETAL_LLMC_DIR}")
set(CUMETAL_LLMC_DIR_AUTO "")
if(EXISTS "${CMAKE_SOURCE_DIR}/../llm.c")
    set(CUMETAL_LLMC_DIR_AUTO "${CMAKE_SOURCE_DIR}/../llm.c")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/llm.c")
    set(CUMETAL_LLMC_DIR_AUTO "${CMAKE_SOURCE_DIR}/llm.c")
endif()

set(CUMETAL_LLMC_DIR_ACTIVE "${CUMETAL_LLMC_DIR_ENV}")
if(NOT CUMETAL_LLMC_DIR_ACTIVE)
    set(CUMETAL_LLMC_DIR_ACTIVE "${CUMETAL_LLMC_DIR_AUTO}")
endif()

if((CUMETAL_LLMC_DIR_ENV AND EXISTS "${CUMETAL_LLMC_DIR_ENV}") OR CUMETAL_LLMC_DIR_AUTO)
    add_test(
        NAME conformance_llmc_gpt2fp32cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_llmc_gpt2fp32cu.sh
    )

    set_tests_properties(conformance_llmc_gpt2fp32cu PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1800
        ENVIRONMENT
            "CUMETAL_LLMC_DIR=${CUMETAL_LLMC_DIR_ACTIVE};CUMETAL_LLMC_BUILD_CMD=${CMAKE_SOURCE_DIR}/scripts/build_llmc_test_gpt2fp32cu.sh;CUMETAL_LLMC_TEST_CMD=${CMAKE_SOURCE_DIR}/scripts/run_llmc_test_gpt2fp32cu.sh"
    )
else()
    message(STATUS "Skipping llm.c conformance registration (set CUMETAL_LLMC_DIR to enable)")
endif()

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
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_llama_cpp_cumetal.sh
    )

    set_tests_properties(conformance_llama_cpp PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600
        ENVIRONMENT
            "CUMETAL_LLAMA_DIR=${CUMETAL_LLAMA_DIR_ACTIVE};CUMETAL_LLAMA_BUILD_CMD=${CMAKE_SOURCE_DIR}/scripts/build_llama_cpp_cumetal.sh"
    )
else()
    message(STATUS "Skipping llama.cpp conformance registration (set CUMETAL_LLAMA_DIR to enable)")
endif()
