#!/usr/bin/env bash
# Shared helpers for cuda_projects CTest drivers. Source, do not execute directly.

cumetal_cuda_projects_check_prereqs() {
    local root_dir="$1"
    local cumetal_build_dir="${CUMETAL_BUILD_DIR:-${root_dir}/build}"

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

    if [[ ! -f "${cumetal_build_dir}/libcumetal.dylib" ]]; then
        echo "SKIP: libcumetal not built at ${cumetal_build_dir}"
        return 77
    fi

    # The standalone .cu harnesses emit host code that references the CUDA
    # runtime registration symbols (__cudaRegisterFatBinary et al.). These are
    # only present when libcumetal is built with the binary shim enabled
    # (CUMETAL_ENABLE_BINARY_SHIM=ON, the default for non-Release builds). A
    # Release build ships the registration stub instead, so linking/loading
    # would fail with a cryptic dyld error. Detect that and skip cleanly.
    # Capture into a variable (no pipe): `nm | grep -q` under `set -o pipefail`
    # can report failure when grep closes the pipe early (SIGPIPE on nm).
    if command -v nm >/dev/null 2>&1; then
        local cumetal_syms
        cumetal_syms="$(nm -gU "${cumetal_build_dir}/libcumetal.dylib" 2>/dev/null || true)"
        if [[ "${cumetal_syms}" != *cudaRegisterFatBinary* ]]; then
            echo "SKIP: libcumetal built without the binary shim (CUMETAL_ENABLE_BINARY_SHIM=OFF);"
            echo "      CUDA registration symbols unavailable — rebuild with -DCUMETAL_ENABLE_BINARY_SHIM=ON"
            return 77
        fi
    fi

    return 0
}

cumetal_cuda_projects_compile_link() {
    local root_dir="$1"
    local src_dir="$2"
    local out_dir="$3"
    local src_cu="$4"
    local out_bin="$5"
    local cumetal_build_dir="${CUMETAL_BUILD_DIR:-${root_dir}/build}"

    # shellcheck source=scripts/cumetal_cuda_flags.sh
    source "${root_dir}/scripts/cumetal_cuda_flags.sh"
    cumetal_cuda_device_flags

    # Prefer native compiler subprocess shims. macOS may SIGKILL interpreter
    # scripts carrying downloaded-file provenance when Clang execs them
    # directly, even though `bash script.sh` is allowed.
    export PATH="${cumetal_build_dir}/cuda_toolchain:${root_dir}/scripts/cuda_toolchain:${PATH}"

    echo "Compiling ${src_cu}..."
    # Filter known non-fatal warnings from homebrew clang + ptx feature flags (for sm_80+)
    # and source RAND_MAX implicit conversions (pre-existing in project .cu samples).
    ( "${CLANG_BIN}" -x cuda -std=c++17 -O2 -DNDEBUG \
        -D__CUDACC__=1 -D__NVCC__=1 -Wno-pass-failed \
        "${CUMETAL_CUDA_DEVICE_FLAGS[@]}" -nocudainc -nocudalib \
        -I"${root_dir}/runtime/api" -include cuda_runtime.h \
        -c "${src_dir}/${src_cu}" -o "${out_dir}/${src_cu%.cu}.o" 2>&1 || true ) \
        | grep -v -E 'ptx[0-9]+ is not a recognized feature|\+ptx[0-9]+|Wimplicit-const-int-float-conversion|warnings generated when compiling for' || true
    xcrun clang++ "${out_dir}/${src_cu%.cu}.o" \
        -L"${cumetal_build_dir}" -lcumetal -Wl,-rpath,"${cumetal_build_dir}" \
        -o "${out_dir}/${out_bin}"
}
