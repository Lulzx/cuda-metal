#!/usr/bin/env bash
# Run CTest and report passed / skipped / failed as three separate numbers.
#
# docs/status.md is explicit that a CTest registration count is not a pass count: the suite
# contains environment-dependent skips and external-project gates. A CI job that prints only
# "N% tests passed" hides how much of the suite never ran, so this script always surfaces the
# skip count and lists the skipped tests by name.
#
# Usage: ci_report.sh <build-dir> [extra ctest args...]
set -uo pipefail

BUILD_DIR="${1:?usage: ci_report.sh <build-dir> [ctest args...]}"
shift || true

LOG="$(mktemp -t cumetal-ctest)"
trap 'rm -f "${LOG}"' EXIT

ctest --test-dir "${BUILD_DIR}" --output-on-failure "$@" 2>&1 | tee "${LOG}"
CTEST_STATUS="${PIPESTATUS[0]}"

TOTAL=$(grep -cE '^[[:space:]]*[0-9]+/[0-9]+ Test' "${LOG}" || true)
PASSED=$(grep -cE '\.\.\.[[:space:]]+Passed' "${LOG}" || true)
SKIPPED=$(grep -cE '\.\.\.\*+Skipped' "${LOG}" || true)
FAILED=$(grep -cE '\.\.\.\*+Failed|\.\.\.\*+Exception' "${LOG}" || true)

{
    echo ""
    echo "## CuMetal test summary"
    echo ""
    echo "| Result | Count |"
    echo "|--------|-------|"
    echo "| Passed | ${PASSED} |"
    echo "| Skipped | ${SKIPPED} |"
    echo "| Failed | ${FAILED} |"
    echo "| Registered | ${TOTAL} |"
    echo ""
    echo "_A registration count is not a pass count; skips are environment-dependent._"
    echo ""

    if [[ "${SKIPPED}" -gt 0 ]]; then
        echo "### Skipped tests"
        echo ""
        grep -E '\.\.\.\*+Skipped' "${LOG}" |
            sed -E 's/^[[:space:]]*[0-9]+\/[0-9]+ Test +#[0-9]+: +//; s/ *\.+\*+Skipped.*//' |
            sed 's/^/- /'
        echo ""
    fi

    if [[ "${FAILED}" -gt 0 ]]; then
        echo "### Failed tests"
        echo ""
        grep -E '\.\.\.\*+Failed|\.\.\.\*+Exception' "${LOG}" |
            sed -E 's/^[[:space:]]*[0-9]+\/[0-9]+ Test +#[0-9]+: +//; s/ *\.+\*+(Failed|Exception).*//' |
            sed 's/^/- /'
        echo ""
    fi
} | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"

exit "${CTEST_STATUS}"
