#!/usr/bin/env bash
# Shared helpers for cuda_projects CTest drivers. Source, do not execute directly.

cumetal_cuda_projects_check_prereqs() {
    local root_dir="$1"

    if ! command -v xcrun >/dev/null 2>&1; then
        echo "SKIP: xcrun not installed"
        return 77
    fi

    if ! xcrun --find metal >/dev/null 2>&1; then
        echo "SKIP: xcrun metal not available (install Xcode command-line tools)"
        return 77
    fi

    CLANG_BIN="${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
    if [[ ! -x "${CLANG_BIN}" ]]; then
        CLANG_BIN="$(command -v clang++ || true)"
    fi
    if [[ -z "${CLANG_BIN}" ]]; then
        echo "SKIP: clang++ not found"
        return 77
    fi

    if [[ ! -f "${root_dir}/build/libcumetal.dylib" ]]; then
        echo "SKIP: libcumetal not built at ${root_dir}/build"
        return 77
    fi

    return 0
}

cumetal_cuda_projects_compile_link() {
    local root_dir="$1"
    local src_dir="$2"
    local out_dir="$3"
    local src_cu="$4"
    local out_bin="$5"

    # shellcheck source=scripts/cumetal_cuda_flags.sh
    source "${root_dir}/scripts/cumetal_cuda_flags.sh"
    cumetal_cuda_device_flags

    export PATH="${root_dir}/scripts/cuda_toolchain:${PATH}"

    echo "Compiling ${src_cu}..."
    "${CLANG_BIN}" -x cuda -std=c++17 -O2 -DNDEBUG \
        -D__CUDACC__=1 -D__NVCC__=1 -Wno-pass-failed \
        "${CUMETAL_CUDA_DEVICE_FLAGS[@]}" -nocudainc -nocudalib \
        -I"${root_dir}/runtime/api" -include cuda_runtime.h \
        -c "${src_dir}/${src_cu}" -o "${out_dir}/${src_cu%.cu}.o"
    xcrun clang++ "${out_dir}/${src_cu%.cu}.o" \
        -L"${root_dir}/build" -lcumetal -Wl,-rpath,"${root_dir}/build" \
        -o "${out_dir}/${out_bin}"
}
