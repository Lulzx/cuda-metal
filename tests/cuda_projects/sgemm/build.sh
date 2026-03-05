#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CLANG_BIN="${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
[[ -x "${CLANG_BIN}" ]] || CLANG_BIN="$(command -v clang++)"
CUDA_ARCH="${CUMETAL_CUDA_ARCH:-sm_80}"
OUT_DIR="${SCRIPT_DIR}/build"; mkdir -p "${OUT_DIR}"

compile_and_link() {
    local src="$1" out="$2"
    echo "Compiling ${src}..."
    PATH="${ROOT_DIR}/scripts/cuda_toolchain:${PATH}" \
    "${CLANG_BIN}" -x cuda -std=c++17 -O2 -DNDEBUG \
        -D__CUDACC__=1 -D__NVCC__=1 -Wno-pass-failed \
        --cuda-gpu-arch="${CUDA_ARCH}" -nocudainc -nocudalib \
        -I"${ROOT_DIR}/runtime/api" -include cuda_runtime.h \
        -c "${SCRIPT_DIR}/${src}" -o "${OUT_DIR}/${src%.cu}.o"
    echo "Linking -> ${out}"
    xcrun clang++ "${OUT_DIR}/${src%.cu}.o" \
        -L"${ROOT_DIR}/build" -lcumetal -Wl,-rpath,"${ROOT_DIR}/build" \
        -o "${OUT_DIR}/${out}"
    echo "Built: ${OUT_DIR}/${out}"
}

compile_and_link sgemm_naive.cu sgemm_naive
compile_and_link sgemm_shmem.cu sgemm_shmem
compile_and_link sgemm_2d.cu    sgemm_2d
