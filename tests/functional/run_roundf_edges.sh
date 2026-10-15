#!/usr/bin/env bash
set -euo pipefail

root_dir="${1:?}"
build_dir="${2:?}"
project_subdir="${3:?}"
source_file="${4:?}"
output_name="${5:?}"
expected="${6:?}"
source "${root_dir}/tests/cuda_projects/_common.sh"

# Preserve genuine prerequisite skips and build defects as distinct outcomes.
if cumetal_cuda_projects_check_prereqs "${root_dir}"; then
    :
else
    exit "$?"
fi

# The registration cache does not include the MSL support module's contents.
# Require a new cache and freshly compiled input even for repeated CTest runs.
run_dir="$(mktemp -d "${build_dir}/roundf-${output_name}.XXXXXX")"
trap 'rm -rf "${run_dir}"' EXIT
export CUMETAL_CACHE_DIR="${run_dir}/cache"
cumetal_cuda_projects_compile_link \
    "${root_dir}" "${root_dir}/tests/cuda_projects/${project_subdir}" \
    "${run_dir}" "${source_file}" "${output_name}"

if output="$(CUMETAL_TRACE_GPU=1 "${run_dir}/${output_name}" 2>&1)"; then
    printf '%s\n' "${output}"
else
    status=$?
    printf '%s\n' "${output}"
    exit "${status}"
fi

if ! grep -Fq "${expected}" <<<"${output}"; then
    echo "FAIL: expected roundf numerical result was not reported"
    exit 1
fi
if ! grep -Eq '^CUMETAL_PROVENANCE .*device=apple_gpu .*launch_success=true duration_ns=[1-9][0-9]*' <<<"${output}"; then
    echo "FAIL: roundf requires a successful Apple GPU launch with positive duration"
    exit 1
fi
if grep -q 'source=approximate_stub' <<<"${output}"; then
    echo "FAIL: roundf used an approximate stub"
    exit 1
fi
