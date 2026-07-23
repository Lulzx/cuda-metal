include_guard(GLOBAL)

# Resolve Apple's separately installed Metal toolchain once at configure time.
# Modern Xcode may make `xcrun metal` work without TOOLCHAINS; older/downloaded
# component layouts require the identifier reported by xcodebuild. The result is
# applied to every CTest test through ENVIRONMENT_MODIFICATION.
function(cumetal_discover_metal_toolchain out_identifier)
    set(${out_identifier} "" PARENT_SCOPE)
    if(NOT APPLE)
        return()
    endif()

    if(DEFINED CUMETAL_METAL_TOOLCHAIN_IDENTIFIER AND
       NOT CUMETAL_METAL_TOOLCHAIN_IDENTIFIER STREQUAL "")
        set(${out_identifier} "${CUMETAL_METAL_TOOLCHAIN_IDENTIFIER}" PARENT_SCOPE)
        return()
    endif()

    if(DEFINED ENV{TOOLCHAINS} AND NOT "$ENV{TOOLCHAINS}" STREQUAL "")
        set(CUMETAL_METAL_TOOLCHAIN_IDENTIFIER "$ENV{TOOLCHAINS}" CACHE STRING
            "Apple toolchain identifier used for Metal compiler discovery")
        set(${out_identifier} "$ENV{TOOLCHAINS}" PARENT_SCOPE)
        return()
    endif()

    find_program(CUMETAL_XCODEBUILD_EXECUTABLE xcodebuild)
    if(NOT CUMETAL_XCODEBUILD_EXECUTABLE)
        return()
    endif()

    execute_process(
        COMMAND "${CUMETAL_XCODEBUILD_EXECUTABLE}"
                -showComponent MetalToolchain -json
        RESULT_VARIABLE _component_status
        OUTPUT_VARIABLE _component_json
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _component_status EQUAL 0 OR _component_json STREQUAL "")
        return()
    endif()

    string(JSON _component_state ERROR_VARIABLE _state_error
           GET "${_component_json}" status)
    string(JSON _identifier ERROR_VARIABLE _identifier_error
           GET "${_component_json}" toolchainIdentifier)
    if(_state_error OR _identifier_error OR
       NOT _component_state STREQUAL "installed" OR _identifier STREQUAL "")
        return()
    endif()

    find_program(CUMETAL_XCRUN_EXECUTABLE xcrun)
    if(NOT CUMETAL_XCRUN_EXECUTABLE)
        return()
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "TOOLCHAINS=${_identifier}"
                "${CUMETAL_XCRUN_EXECUTABLE}" -f metal
        RESULT_VARIABLE _metal_status
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT _metal_status EQUAL 0)
        return()
    endif()

    set(CUMETAL_METAL_TOOLCHAIN_IDENTIFIER "${_identifier}" CACHE STRING
        "Apple toolchain identifier used for Metal compiler discovery")
    set(${out_identifier} "${_identifier}" PARENT_SCOPE)
endfunction()

function(cumetal_apply_metal_toolchain_to_tests identifier)
    if(identifier STREQUAL "")
        return()
    endif()
    foreach(_directory IN LISTS ARGN)
        get_property(_tests DIRECTORY "${_directory}" PROPERTY TESTS)
        if(_tests)
            set_tests_properties(
                ${_tests}
                DIRECTORY "${_directory}"
                PROPERTIES
                    ENVIRONMENT_MODIFICATION "TOOLCHAINS=set:${identifier}"
            )
        endif()
    endforeach()
endfunction()
