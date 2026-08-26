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
KNOWN_GAPS="${SCRIPT_DIR}/../../docs/known-gaps.md"
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
headline="**${pass_count} pass, ${waive_count} waive cleanly, ${nonpassing_count} do not yet have a passing runtime result**."
if ! grep -Fq "${headline}" "${KNOWN_GAPS}"; then
    echo "FAIL: docs/known-gaps.md sweep headline does not match the manifest" >&2
    echo "Expected: ${headline}" >&2
    exit 1
fi

echo "PASS: cuda-samples manifest transition policy"
