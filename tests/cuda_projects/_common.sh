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
    # runtime registration symbols (__cudaRegisterFatBinary et al.). Those are
    # part of the source-recompilation path and are built unconditionally
    # (CUMETAL_ENABLE_CUDA_REGISTRATION defaults ON in every build type), so
    # their absence is a build defect rather than an expected configuration.
    # Fail instead of skipping: this check previously keyed off the binary shim
    # and silently skipped the entire source path in Release builds.
    # Capture into a variable (no pipe): `nm | grep -q` under `set -o pipefail`
    # can report failure when grep closes the pipe early (SIGPIPE on nm).
    if command -v nm >/dev/null 2>&1; then
        local cumetal_syms
        cumetal_syms="$(nm -gU "${cumetal_build_dir}/libcumetal.dylib" 2>/dev/null || true)"
        if [[ "${cumetal_syms}" != *cudaRegisterFatBinary* ]]; then
            echo "FAIL: libcumetal exports no CUDA registration symbols."
            echo "      The source path requires CUMETAL_ENABLE_CUDA_REGISTRATION=ON (the default)."
            return 1
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

    local obj_file="${out_dir}/${src_cu%.cu}.o"
    local compile_log="${out_dir}/${src_cu%.cu}.compile.log"

    echo "Compiling ${src_cu}..."
    # The compiler's exit status must be propagated, and any object file from an
    # earlier build removed first. Discarding the status let a failed compile link
    # a stale object instead, so the harness reported PASS while verifying code
    # that no longer existed.
    #
    # RAND_MAX implicit conversions are pre-existing in project .cu samples and
    # are filtered out of the printed log only — never out of the exit status.
    rm -f "${obj_file}" "${out_dir}/${out_bin}"
    local compile_status=0
    "${CLANG_BIN}" -x cuda -std=c++17 -O2 -DNDEBUG \
        -D__CUDACC__=1 -D__NVCC__=1 -Wno-pass-failed \
        "${CUMETAL_CUDA_DEVICE_FLAGS[@]}" -nocudainc -nocudalib \
        -I"${root_dir}/runtime/api" -include cuda_runtime.h \
        -c "${src_dir}/${src_cu}" -o "${obj_file}" \
        >"${compile_log}" 2>&1 || compile_status=$?
    grep -v -E 'Wimplicit-const-int-float-conversion|warnings generated when compiling for' \
        "${compile_log}" || true

    if [[ ${compile_status} -ne 0 ]]; then
        echo "FAIL: compiling ${src_cu} failed (clang exit ${compile_status})"
        return 1
    fi
    if grep -q -E 'ptx[0-9]+.*is not a recognized feature for this target' "${compile_log}"; then
        echo "FAIL: the CUDA PTX feature leaked into the Apple host compilation"
        return 1
    fi
    if [[ ! -f "${obj_file}" ]]; then
        echo "FAIL: compiling ${src_cu} produced no object file"
        return 1
    fi

    xcrun clang++ "${obj_file}" \
        -L"${cumetal_build_dir}" -lcumetal -Wl,-rpath,"${cumetal_build_dir}" \
        -o "${out_dir}/${out_bin}"
}
