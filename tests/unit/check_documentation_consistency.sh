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

# Every enrolled sample passes except the documented known failures below;
# any other nonpassing entry, or a waiver, fails this check.
known_nonpassing=(
    "3_CUDA_Features/cdpAdvancedQuicksort"
    "3_CUDA_Features/cdpQuadtree"
)
actual_nonpassing="$(awk 'NF >= 2 && $1 !~ /^#/ && $1 != "pass" {print $2}' "${sample_manifest}" | sort)"
expected_nonpassing="$(printf '%s\n' "${known_nonpassing[@]}" | sort)"
[[ "${actual_nonpassing}" == "${expected_nonpassing}" ]] ||
    fail "nonpassing samples differ from the documented known failures: ${actual_nonpassing//$'\n'/, }"
[[ "${sample_waive}" -eq 0 ]] ||
    fail "current roadmap requires zero sample waivers"
for sample in "${known_nonpassing[@]}"; do
    expect_fragment docs/known-gaps/verification.md "\`${sample##*/}\`"
done

# README stays a concise entry point; detailed evidence lives in linked pages.
expect_fragment README.md "[verified results](docs/verified-results.md)"
expect_fragment README.md "[known gaps](docs/known-gaps.md)"
expect_fragment README.md "[compiler architecture](docs/compiler-architecture.md)"
expect_fragment docs/status.md "**${sample_pass}/${sample_total} pass**"
expect_fragment docs/known-gaps/verification.md "samples; ${sample_pass} pass."
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
expect_fragment docs/legal-notice.md "This is engineering policy, not legal advice"
expect_fragment docs/legal-notice.md "production compiler emits typed Metal Shading Language"
expect_fragment docs/legal-notice.md "research and regression tooling only"
expect_fragment docs/legal-notice.md "does not claim a private AIR ABI as a supported production interface"
expect_fragment docs/known-gaps/compiler.md "LZ4/Zstd-compressed PTX entries"
expect_fragment docs/status/runtime.md "Decompression is capped at 64 MiB"
expect_fragment docs/spec-closure-roadmap.md "LZ4/Zstd PTX entry support"

# Keep source-first packaging policy tied to its actual CMake default.
expect_fragment CMakeLists.txt 'set(CUMETAL_ENABLE_BINARY_SHIM_DEFAULT OFF)'
expect_fragment README.md "disabled in Release builds unless explicitly"
expect_fragment docs/spec-closure-roadmap.md 'Release packaging must still omit the `libcuda.dylib` alias by default'

echo "PASS: documentation matches manifest counts and high-risk boundaries"
