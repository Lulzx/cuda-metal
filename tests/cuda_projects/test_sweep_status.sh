#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/cuda_projects/sweep_status.sh
source "${SCRIPT_DIR}/sweep_status.sh"

expect_transition() {
    local expected="$1" actual="$2" wanted="$3"
    local observed
    observed="$(cumetal_sweep_transition "${expected}" "${actual}")"
    if [[ "${observed}" != "${wanted}" ]]; then
        echo "FAIL: ${expected} -> ${actual}: expected ${wanted}, got ${observed}" >&2
        exit 1
    fi
}

expect_transition pass pass match
expect_transition waive run-fail regression
expect_transition compile-fail pass improvement
expect_transition run-unverified pass evidence-update
expect_transition run-unverified run-fail evidence-update
expect_transition run-unverified no-lowering evidence-update
expect_transition compile-fail run-fail unsupported-drift

MANIFEST="${SCRIPT_DIR}/cuda_samples_sweep_manifest.txt"
KNOWN_GAPS="${SCRIPT_DIR}/../../docs/known-gaps/verification.md"
README="${SCRIPT_DIR}/../../README.md"
VERIFIED_RESULTS="${SCRIPT_DIR}/../../docs/verified-results.md"
read -r pass_count waive_count total_count < <(
    awk '
        NF >= 2 && $1 !~ /^#/ {
            total++;
            if ($1 == "pass") pass_count++;
            if ($1 == "waive") waive_count++;
        }
        END { print pass_count + 0, waive_count + 0, total + 0 }
    ' "${MANIFEST}"
)
nonpassing_count=$((total_count - pass_count - waive_count))
headline="all ${pass_count} pass"
if ! grep -Fq "${headline}" "${KNOWN_GAPS}"; then
    echo "FAIL: docs/known-gaps/verification.md sweep headline does not match the manifest" >&2
    echo "Expected fragment: ${headline}" >&2
    exit 1
fi

readme_headline="**${pass_count}/${total_count} pass**"
if ! grep -Fq "${readme_headline}" "${README}"; then
    echo "FAIL: README cuda-samples headline does not match the manifest" >&2
    echo "Expected fragment: ${readme_headline}" >&2
    exit 1
fi

verified_headline="classifies all ${total_count} enrolled"
if ! grep -Fq "${verified_headline}" "${VERIFIED_RESULTS}"; then
    echo "FAIL: docs/verified-results.md cuda-samples headline does not match the manifest" >&2
    echo "Expected fragment: ${verified_headline}" >&2
    exit 1
fi

echo "PASS: cuda-samples manifest transition and documentation policy"
