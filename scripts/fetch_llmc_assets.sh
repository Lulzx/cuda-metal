#!/usr/bin/env bash
# Fetch llm.c GPT-2 124M checkpoint + debug state required by test_gpt2_fp32.cu.
#
# Usage:
#   bash scripts/fetch_llmc_assets.sh [llm.c-dir]
#
# Uses upstream dev/download_starter_pack.sh when present; otherwise prints hints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLMC_DIR="${CUMETAL_LLMC_DIR:-${1:-${ROOT_DIR}/../llm.c}}"
if [[ ! -d "${LLMC_DIR}" ]]; then
    echo "llm.c directory not found: ${LLMC_DIR}" >&2
    echo "Clone https://github.com/karpathy/llm.c and set CUMETAL_LLMC_DIR" >&2
    exit 2
fi

need=(
    gpt2_124M.bin
    gpt2_124M_debug_state.bin
)

have_all=1
for f in "${need[@]}"; do
    if [[ -f "${LLMC_DIR}/${f}" ]]; then
        echo "OK: ${LLMC_DIR}/${f}"
    elif [[ -f "${LLMC_DIR}/dev/data/${f}" ]]; then
        echo "OK: ${LLMC_DIR}/dev/data/${f} (will symlink to checkout root)"
        ln -sf "dev/data/${f}" "${LLMC_DIR}/${f}"
    else
        echo "MISSING: ${f}"
        have_all=0
    fi
done

if [[ "${have_all}" -eq 1 ]]; then
    echo "llm.c assets ready under ${LLMC_DIR}"
    exit 0
fi

DOWNLOAD="${LLMC_DIR}/dev/download_starter_pack.sh"
if [[ -x "${DOWNLOAD}" ]]; then
    echo "Running ${DOWNLOAD}..."
    (cd "${LLMC_DIR}" && "${DOWNLOAD}")
else
    echo "Run llm.c's starter pack download from ${LLMC_DIR}:" >&2
    echo "  chmod u+x ./dev/download_starter_pack.sh && ./dev/download_starter_pack.sh" >&2
    exit 2
fi

for f in "${need[@]}"; do
    if [[ -f "${LLMC_DIR}/dev/data/${f}" && ! -f "${LLMC_DIR}/${f}" ]]; then
        ln -sf "dev/data/${f}" "${LLMC_DIR}/${f}"
    fi
done

for f in "${need[@]}"; do
    [[ -f "${LLMC_DIR}/${f}" ]] || { echo "still missing ${f}" >&2; exit 1; }
done

echo "llm.c assets ready under ${LLMC_DIR}"
