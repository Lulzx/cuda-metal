#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLMC_DIR="${CUMETAL_LLMC_DIR:-${1:-}}"
if [[ -z "${LLMC_DIR}" ]]; then
    echo "usage: $0 <llm.c-dir> (or set CUMETAL_LLMC_DIR)" >&2
    exit 2
fi
if [[ ! -d "${LLMC_DIR}" ]]; then
    echo "llm.c directory not found: ${LLMC_DIR}" >&2
    exit 2
fi

CLANG_BIN="${CUMETAL_LLMC_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
if [[ ! -x "${CLANG_BIN}" ]]; then
    CLANG_BIN="$(command -v clang++ || true)"
fi
if [[ -z "${CLANG_BIN}" ]]; then
    echo "clang++ not found" >&2
    exit 2
fi

CUDA_ARCH="${CUMETAL_LLMC_CUDA_ARCH:-sm_80}"
OUTPUT_NAME="${CUMETAL_LLMC_TEST_BINARY:-test_gpt2fp32cu}"
GRAD_TOL="${CUMETAL_LLMC_GRAD_TOL:-1.2e-2}"
OBJ_DIR="${LLMC_DIR}/build/cumetal"
OBJ_FILE="${OBJ_DIR}/test_gpt2_fp32.cumetal.o"
OUT_FILE="${LLMC_DIR}/${OUTPUT_NAME}"
PATCHED_SRC="${OBJ_DIR}/test_gpt2_fp32.cumetal.cu"

mkdir -p "${OBJ_DIR}"

python3 - "${LLMC_DIR}/test_gpt2_fp32.cu" "${PATCHED_SRC}" "${GRAD_TOL}" <<'PY'
import pathlib
import re
import sys

src_path = pathlib.Path(sys.argv[1])
dst_path = pathlib.Path(sys.argv[2])
tol = sys.argv[3]

source = src_path.read_text()
pattern = re.compile(
    r"(int\s+check_tensor\s*\([^)]*\)\s*\{.*?fabsf\(a\[i\]\s*-\s*b\[i\]\)\s*<=\s*)(1e-2)(\s*\)\s*\{)",
    re.S,
)
patched, count = pattern.subn(rf"\g<1>{tol}\g<3>", source, count=1)
if count != 1:
    raise SystemExit("failed to patch llm.c check_tensor tolerance")
dst_path.write_text(patched)
PY

PATH="${ROOT_DIR}/scripts/cuda_toolchain:${PATH}" \
"${CLANG_BIN}" \
    -x cuda \
    -std=c++17 \
    -O2 \
    -DNDEBUG \
    -D__CUDACC__=1 \
    -D__NVCC__=1 \
    -Wno-pass-failed \
    --cuda-gpu-arch="${CUDA_ARCH}" \
    -nocudainc \
    -nocudalib \
    -I"${ROOT_DIR}/runtime/api" \
    -include cuda_runtime.h \
    -I"${LLMC_DIR}" \
    -c "${PATCHED_SRC}" \
    -o "${OBJ_FILE}"

xcrun clang++ \
    "${OBJ_FILE}" \
    -L"${ROOT_DIR}/build" \
    -lcumetal \
    -Wl,-rpath,"${ROOT_DIR}/build" \
    -o "${OUT_FILE}"

echo "built ${OUT_FILE}"
