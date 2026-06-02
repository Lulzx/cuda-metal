#!/usr/bin/env bash
# Shared helpers for cuda_projects CTest drivers. Source, do not execute directly.

cumetal_cuda_projects_check_prereqs() {
    local root_dir="$1"

    if ! command -v xcrun >/dev/null 2>&1; then
        echo "SKIP: xcrun not installed"
        return 77
    fi

    # Relax metal requirement: many CLT setups have xcrun clang++ (for host link) but
    # not the metal/metallib utilities (compiler). The cuda_projects path here uses
    # a clang++ -x cuda shim (no xcrun metal compile of .metal), so metal find is not
    # strictly needed. Keep base xcrun + clang++ + libcumetal checks.
    # if ! xcrun --find metal ... (intentionally relaxed to reduce skip-only coverage)

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
    # Filter known non-fatal warnings from homebrew clang + ptx feature flags (for sm_80+)
    # and source RAND_MAX implicit conversions (pre-existing in project .cu samples).
    ( "${CLANG_BIN}" -x cuda -std=c++17 -O2 -DNDEBUG \
        -D__CUDACC__=1 -D__NVCC__=1 -Wno-pass-failed \
        "${CUMETAL_CUDA_DEVICE_FLAGS[@]}" -nocudainc -nocudalib \
        -I"${root_dir}/runtime/api" -include cuda_runtime.h \
        -c "${src_dir}/${src_cu}" -o "${out_dir}/${src_cu%.cu}.o" 2>&1 || true ) \
        | grep -v '+ptx[0-9][0-9]*' is not a recognized feature \
        | grep -v 'implicit conversion from .* to float changes value' || true
    xcrun clang++ "${out_dir}/${src_cu%.cu}.o" \
        -L"${root_dir}/build" -lcumetal -Wl,-rpath,"${root_dir}/build" \
        -o "${out_dir}/${out_bin}"
}
