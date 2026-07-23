#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <build_llama_cpp_cumetal.sh>" >&2
    exit 2
fi

BUILD_SCRIPT="$1"
if [[ ! -f "${BUILD_SCRIPT}" ]]; then
    echo "FAIL: llama.cpp build script not found: ${BUILD_SCRIPT}" >&2
    exit 1
fi

if ! grep -q -- '-DGGML_CUDA_FA=OFF' "${BUILD_SCRIPT}"; then
    echo "FAIL: CuMetal llama.cpp build must report FlashAttention unsupported" >&2
    exit 1
fi

if grep -q -- '-DGGML_CUDA_FA=ON' "${BUILD_SCRIPT}"; then
    echo "FAIL: CuMetal llama.cpp build enables unsupported FlashAttention" >&2
    exit 1
fi

echo "PASS: llama.cpp build disables unsupported CUDA FlashAttention"
