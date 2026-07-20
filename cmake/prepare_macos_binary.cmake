if(NOT DEFINED CUMETAL_BINARY)
    message(FATAL_ERROR "CUMETAL_BINARY was not provided")
endif()

# Ignore a missing attribute: locally created files outside a quarantined
# checkout normally do not carry this metadata.
execute_process(
    COMMAND /usr/bin/xattr -d com.apple.provenance "${CUMETAL_BINARY}"
    ERROR_QUIET
)

execute_process(
    COMMAND /usr/bin/codesign --force --sign - "${CUMETAL_BINARY}"
    RESULT_VARIABLE codesign_result
    ERROR_VARIABLE codesign_error
)
if(NOT codesign_result EQUAL 0)
    message(FATAL_ERROR "Could not ad-hoc sign ${CUMETAL_BINARY}: ${codesign_error}")
endif()
