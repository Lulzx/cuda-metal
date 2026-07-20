#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CUMETAL_BUILD_DIR="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}"

BIN_NAME="${CUMETAL_LLMC_TEST_BINARY:-test_gpt2fp32cu}"
if [[ ! -x "./${BIN_NAME}" ]]; then
    echo "missing llm.c binary: ./${BIN_NAME}" >&2
    exit 2
fi

export DYLD_LIBRARY_PATH="${CUMETAL_BUILD_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export CUMETAL_DISABLE_LLMC_EMULATION="${CUMETAL_DISABLE_LLMC_EMULATION:-1}"
export CUMETAL_ENABLE_LLMC_CPU_EMULATION=0
exec "./${BIN_NAME}"
