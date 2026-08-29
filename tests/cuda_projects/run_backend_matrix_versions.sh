#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <source-root> <cumetalc> <manifest> <matrix-runner>" >&2
    exit 2
fi

source_root="$1"
cumetalc="$2"
manifest="$3"
matrix_runner="$4"

if [[ ! -x "${cumetalc}" ]]; then
    echo "SKIP: cumetalc is not built: ${cumetalc}" >&2
    exit 77
fi
if [[ ! -x "${matrix_runner}" ]]; then
    echo "FAIL: backend matrix runner is not executable: ${matrix_runner}" >&2
    exit 1
fi

resolve_clang() {
    local major="$1"
    local override_name="CUMETAL_CUDA_CLANG_${major}"
    local override="${!override_name:-}"
    local candidate
    local -a candidates=()

    if [[ -n "${override}" ]]; then
        candidates+=("${override}")
    else
        candidates+=("/opt/homebrew/opt/llvm@${major}/bin/clang++")
        candidates+=("/usr/local/opt/llvm@${major}/bin/clang++")
        if [[ "${major}" == 23 ]]; then
            candidates+=("/opt/homebrew/opt/llvm/bin/clang++")
            candidates+=("/usr/local/opt/llvm/bin/clang++")
        fi
    fi

    for candidate in "${candidates[@]}"; do
        [[ -x "${candidate}" ]] || continue
        if "${candidate}" --version 2>/dev/null | head -1 |
            grep -Eq "clang version ${major}\\."; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    if [[ -n "${override}" ]]; then
        echo "FAIL: ${override_name} does not name an executable CUDA Clang ${major}" >&2
        return 1
    fi
    return 2
}

declare -a versions=(21 22 23)
declare -a compilers=()
declare -a missing=()

for version in "${versions[@]}"; do
    if compiler="$(resolve_clang "${version}")"; then
        compilers+=("${compiler}")
    else
        rc=$?
        if [[ ${rc} -eq 1 ]]; then
            exit 1
        fi
        compilers+=("")
        missing+=("${version}")
    fi
done

if [[ ${#missing[@]} -ne 0 ]]; then
    echo "SKIP: required CUDA Clang version(s) unavailable: ${missing[*]}" >&2
    echo "Set CUMETAL_CUDA_CLANG_21, _22, and _23 to explicit compiler paths." >&2
    exit 77
fi

for index in "${!versions[@]}"; do
    version="${versions[index]}"
    compiler="${compilers[index]}"
    identity="$("${compiler}" --version | head -1)"
    echo "=== CUDA Clang ${version}: ${compiler} ==="
    echo "identity=${identity}"
    CUMETAL_CUDA_CLANG="${compiler}" \
        "${matrix_runner}" "${source_root}" "${cumetalc}" "${manifest}"
done

echo "PASS: compiler backend matrix matches with CUDA Clang 21, 22, and 23"
