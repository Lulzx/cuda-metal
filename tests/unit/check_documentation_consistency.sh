#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <source-root>" >&2
    exit 2
fi

root="$1"
sample_manifest="${root}/tests/cuda_projects/cuda_samples_sweep_manifest.txt"
backend_manifest="${root}/tests/cuda_projects/backend_matrix_manifest.txt"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

expect_fragment() {
    local file="$1"
    local fragment="$2"
    grep -Fq -- "${fragment}" "${root}/${file}" ||
        fail "${file} is missing current fragment: ${fragment}"
}

read -r sample_pass sample_waive sample_total < <(
    awk '
        NF >= 2 && $1 !~ /^#/ {
            total++
            if ($1 == "pass") pass_count++
            if ($1 == "waive") waive_count++
        }
        END { print pass_count + 0, waive_count + 0, total + 0 }
    ' "${sample_manifest}"
)
sample_nonpassing=$((sample_total - sample_pass - sample_waive))

[[ "${sample_pass}" -eq "${sample_total}" ]] ||
    fail "current roadmap requires an all-pass enrolled sample snapshot"
[[ "${sample_waive}" -eq 0 && "${sample_nonpassing}" -eq 0 ]] ||
    fail "current roadmap requires zero sample waivers and nonpassing entries"

expect_fragment README.md "**${sample_pass}/${sample_total} pass**"
expect_fragment docs/status.md "**${sample_pass}/${sample_total} pass**"
expect_fragment docs/known-gaps/verification.md "all ${sample_pass} pass"
expect_fragment docs/verified-results.md "classifies all ${sample_total} enrolled"
expect_fragment docs/spec-closure-roadmap.md "contains ${sample_total} enrolled headless samples"

read -r matrix_total direct_legacy direct_typed ptx_legacy ptx_typed < <(
    awk '
        NF >= 5 && $1 !~ /^#/ {
            total++
            if ($2 == "pass") direct_legacy++
            if ($3 == "pass") direct_typed++
            if ($4 == "pass") ptx_legacy++
            if ($5 == "pass") ptx_typed++
        }
        END {
            print total + 0, direct_legacy + 0, direct_typed + 0,
                  ptx_legacy + 0, ptx_typed + 0
        }
    ' "${backend_manifest}"
)

for file in \
    README.md \
    docs/status.md \
    docs/status/compiler.md \
    docs/known-gaps/compiler.md \
    docs/compiler-architecture.md \
    docs/spec-closure-roadmap.md
do
    expect_fragment "${file}" "${direct_typed}/${matrix_total}"
    expect_fragment "${file}" "${ptx_legacy}/${matrix_total}"
    expect_fragment "${file}" "${ptx_typed}/${matrix_total}"
done

for file in \
    README.md \
    docs/status.md \
    docs/status/compiler.md \
    docs/known-gaps/compiler.md \
    docs/compiler-architecture.md \
    docs/spec-closure-roadmap.md
do
    expect_fragment "${file}" "CUDA Clang 21-23"
done
expect_fragment docs/spec-closure-roadmap.md "conformance_compiler_backend_matrix{,_versions}"
expect_fragment docs/known-gaps/compiler.md "conformance_compiler_backend_matrix_versions"

# High-risk boundaries must remain visible in the current indexes and roadmap.
expect_fragment README.md "SIMD/warp width is fixed at 32"
expect_fragment README.md "no private Apple"
expect_fragment README.md "SASS execution is unsupported"
expect_fragment docs/known-gaps/platform.md "No private Apple APIs"
expect_fragment docs/known-gaps/platform.md "No SASS execution"
expect_fragment docs/spec-closure-roadmap.md "no first-launch PTX JIT"
expect_fragment docs/spec-closure-roadmap.md "CPU/UMA fallback is reported as GPU execution"
expect_fragment docs/spec-closure-roadmap.md "Big-endian and SASS-only inputs remain explicit non-goals"
expect_fragment docs/spec-closure-roadmap.md 'observable `ieee64` exception status'

# Keep source-first packaging policy tied to its actual CMake default.
expect_fragment CMakeLists.txt 'set(CUMETAL_ENABLE_BINARY_SHIM_DEFAULT OFF)'
expect_fragment README.md "disabled in Release builds unless explicitly"
expect_fragment docs/spec-closure-roadmap.md 'Release packaging must still omit the `libcuda.dylib` alias by default'

echo "PASS: documentation matches manifest counts and high-risk boundaries"
