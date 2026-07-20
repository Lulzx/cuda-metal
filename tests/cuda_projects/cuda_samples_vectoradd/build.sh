#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CUMETAL_BUILD_DIR="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}"
CUDA_SAMPLES_DIR="${1:-/Users/lulzx/work/cuda-samples}"
CLANG_BIN="${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
[[ -x "${CLANG_BIN}" ]] || CLANG_BIN="$(command -v clang++)"
OUT_DIR="${SCRIPT_DIR}/build"; mkdir -p "${OUT_DIR}"
# shellcheck source=scripts/cumetal_cuda_flags.sh
source "${ROOT_DIR}/scripts/cumetal_cuda_flags.sh"
cumetal_cuda_device_flags
echo "Compiling vectorAdd.cu..."
PATH="${CUMETAL_BUILD_DIR}/cuda_toolchain:${ROOT_DIR}/scripts/cuda_toolchain:${PATH}" \
"${CLANG_BIN}" -x cuda -std=c++17 -O2 -DNDEBUG \
    -D__CUDACC__=1 -D__NVCC__=1 -Wno-pass-failed \
    "${CUMETAL_CUDA_DEVICE_FLAGS[@]}" -nocudainc -nocudalib \
    -I"${ROOT_DIR}/runtime/api" -include cuda_runtime.h \
    -I"${CUDA_SAMPLES_DIR}/Common" \
    -c "${CUDA_SAMPLES_DIR}/Samples/0_Introduction/vectorAdd/vectorAdd.cu" \
    -o "${OUT_DIR}/vectorAdd.o"
echo "Linking..."
xcrun clang++ "${OUT_DIR}/vectorAdd.o" \
    -L"${CUMETAL_BUILD_DIR}" -lcumetal -Wl,-rpath,"${CUMETAL_BUILD_DIR}" \
    -o "${OUT_DIR}/vectorAdd"
echo "Built: ${OUT_DIR}/vectorAdd"
