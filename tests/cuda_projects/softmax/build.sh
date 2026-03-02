#!/usr/bin/env bash
# Build softmax.cu against CuMetal using the clang CUDA pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CLANG_BIN="${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
if [[ ! -x "${CLANG_BIN}" ]]; then
    CLANG_BIN="$(command -v clang++ || true)"
fi
if [[ -z "${CLANG_BIN}" ]]; then
    echo "clang++ not found" >&2
    exit 2
fi

CUDA_ARCH="${CUMETAL_CUDA_ARCH:-sm_80}"
OUT_DIR="${SCRIPT_DIR}/build"
mkdir -p "${OUT_DIR}"

OBJ="${OUT_DIR}/softmax.o"
BIN="${OUT_DIR}/softmax"

echo "Compiling softmax.cu → ${OBJ}"
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
    -c "${SCRIPT_DIR}/softmax.cu" \
    -o "${OBJ}"

echo "Linking → ${BIN}"
xcrun clang++ \
    "${OBJ}" \
    -L"${ROOT_DIR}/build" \
    -lcumetal \
    -Wl,-rpath,"${ROOT_DIR}/build" \
    -o "${BIN}"

echo "Built: ${BIN}"
