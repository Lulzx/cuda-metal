#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CLANG_BIN="${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
[[ -x "${CLANG_BIN}" ]] || CLANG_BIN="$(command -v clang++)"
CUDA_ARCH="${CUMETAL_CUDA_ARCH:-sm_80}"
OUT_DIR="${SCRIPT_DIR}/build"; mkdir -p "${OUT_DIR}"

echo "Compiling reduction_standalone.cu..."
PATH="${ROOT_DIR}/scripts/cuda_toolchain:${PATH}" \
"${CLANG_BIN}" -x cuda -std=c++17 -O2 -DNDEBUG \
    -D__CUDACC__=1 -D__NVCC__=1 -Wno-pass-failed \
    --cuda-gpu-arch="${CUDA_ARCH}" -nocudainc -nocudalib \
    -I"${ROOT_DIR}/runtime/api" -include cuda_runtime.h \
    -c "${SCRIPT_DIR}/reduction_standalone.cu" -o "${OUT_DIR}/reduction.o"
echo "Linking..."
xcrun clang++ "${OUT_DIR}/reduction.o" \
    -L"${ROOT_DIR}/build" -lcumetal -Wl,-rpath,"${ROOT_DIR}/build" \
    -o "${OUT_DIR}/reduction_standalone"
echo "Built: ${OUT_DIR}/reduction_standalone"
