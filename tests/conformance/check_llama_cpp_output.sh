#!/usr/bin/env bash
# Validate a captured llama.cpp/CuMetal run. Treat the capture as bytes: llama.cpp
# writes UI text while CuMetal writes provenance to the same pipe, so concurrent
# output can split a multibyte UTF-8 sequence even when all searched markers are
# intact ASCII.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <output-file> <llama-exit-code> <ngl> <expected-text>" >&2
    exit 2
fi

OUTPUT_FILE="$1"
EXIT_CODE="$2"
NGL="$3"
EXPECT="$4"
FAIL_REASON=""

[[ "${EXIT_CODE}" -ne 0 ]] &&
    FAIL_REASON="llama-cli exited with code ${EXIT_CODE}"

OUTPUT_LEN="$(wc -c < "${OUTPUT_FILE}")"
if [[ -z "${FAIL_REASON}" && "${OUTPUT_LEN}" -lt 50 ]]; then
    FAIL_REASON="output too short (${OUTPUT_LEN} bytes) — no tokens generated"
fi

if [[ -z "${FAIL_REASON}" ]] &&
    LC_ALL=C grep -aqiE "Segmentation fault|Bus error|Illegal instruction" \
        "${OUTPUT_FILE}"; then
    FAIL_REASON="fatal signal in llama-cli output"
fi

if [[ -z "${FAIL_REASON}" ]] &&
    LC_ALL=C grep -aqiE "CUDA error|cudaError|ggml_cuda.*failed" \
        "${OUTPUT_FILE}"; then
    FAIL_REASON="CUDA error reported by GGML backend"
fi

if [[ -z "${FAIL_REASON}" && "${NGL}" -gt 0 ]]; then
    if ! LC_ALL=C grep -aqE \
        'CUMETAL_PROVENANCE .*source=(generic_ptx|specialized_msl|metallib) device=apple_gpu .*launch_success=true' \
        "${OUTPUT_FILE}"; then
        FAIL_REASON="no successful Apple-GPU kernel provenance was recorded"
    elif LC_ALL=C grep -aqE \
        'CUMETAL_PROVENANCE .*source=(cpu_fallback|stub)' \
        "${OUTPUT_FILE}"; then
        FAIL_REASON="CPU fallback or stub provenance was recorded during GPU inference"
    fi
fi

if [[ -z "${FAIL_REASON}" && -n "${EXPECT}" ]]; then
    # A generated token fragment can be written immediately before a provenance
    # record. Removing that record and its transport newline reconstructs the UI
    # text the terminal displayed (for example, "Par" + record + "is").
    NORMALIZED_OUTPUT="$(
        LC_ALL=C tr -cd '\11\12\15\40-\176' < "${OUTPUT_FILE}" |
            sed -E 's/CUMETAL_PROVENANCE[^[:cntrl:]]*//g' |
            tr -d '\r\n'
    )"
    if ! printf '%s' "${NORMALIZED_OUTPUT}" |
        LC_ALL=C grep -aiF "${EXPECT}" >/dev/null; then
        FAIL_REASON="incoherent output — generation did not contain '${EXPECT}'. \
CuMetal ran llama.cpp to completion but produced text that does not match a \
correct (stock CPU) run: numerically wrong output, not a crash. See \
docs/known-gaps.md for status."
    fi
fi

if [[ -n "${FAIL_REASON}" ]]; then
    printf '%s\n' "${FAIL_REASON}"
    exit 1
fi
