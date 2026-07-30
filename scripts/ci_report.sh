#!/usr/bin/env bash
# Run CTest and report passed / skipped / failed as three separate numbers.
#
# docs/status.md is explicit that a CTest registration count is not a pass count: the suite
# contains environment-dependent skips and external-project gates. A CI job that prints only
# "N% tests passed" hides how much of the suite never ran, so this script always surfaces the
# skip count and lists the skipped tests by name.
#
# Usage:
#   ci_report.sh <build-dir> [--require-tests] [--require-no-skips] [ctest args...]
#
# --require-tests fails when the CTest selection is empty. This catches stale
# label/regex selections that would otherwise make a CI job pass without
# exercising anything.
#
# --require-no-skips fails when any selected test skips. Use it for narrow
# hardware proof gates whose prerequisites are part of the gate. Do not use it
# for the full suite, which intentionally contains external-project skips.
set -uo pipefail

BUILD_DIR="${1:?usage: ci_report.sh <build-dir> [ctest args...]}"
shift || true

REQUIRE_TESTS=0
REQUIRE_NO_SKIPS=0
CTEST_ARGS=()
for argument in "$@"; do
    case "${argument}" in
        --require-tests)
            REQUIRE_TESTS=1
            ;;
        --require-no-skips)
            REQUIRE_NO_SKIPS=1
            ;;
        *)
            CTEST_ARGS+=("${argument}")
            ;;
    esac
done

LOG="$(mktemp -t cumetal-ctest)"
trap 'rm -f "${LOG}"' EXIT

ctest --test-dir "${BUILD_DIR}" --output-on-failure "${CTEST_ARGS[@]}" 2>&1 | tee "${LOG}"
CTEST_STATUS="${PIPESTATUS[0]}"

TOTAL=$(grep -cE '^[[:space:]]*[0-9]+/[0-9]+ Test' "${LOG}" || true)
PASSED=$(grep -cE '\.\.\.[[:space:]]+Passed' "${LOG}" || true)
SKIPPED=$(grep -cE '\.\.\.\*+Skipped' "${LOG}" || true)
FAILED=$(grep -cE '\.\.\.\*+Failed|\.\.\.\*+Exception' "${LOG}" || true)

POLICY_FAILURES=()
if [[ "${REQUIRE_TESTS}" -eq 1 && "${TOTAL}" -eq 0 ]]; then
    POLICY_FAILURES+=("CTest selection was empty (--require-tests).")
fi
if [[ "${REQUIRE_NO_SKIPS}" -eq 1 && "${SKIPPED}" -gt 0 ]]; then
    POLICY_FAILURES+=("${SKIPPED} selected test(s) skipped (--require-no-skips).")
fi

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

    if [[ "${#POLICY_FAILURES[@]}" -gt 0 ]]; then
        echo "### CI policy failures"
        echo ""
        for failure in "${POLICY_FAILURES[@]}"; do
            echo "- ${failure}"
        done
        echo ""
    fi
} | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"

if [[ "${CTEST_STATUS}" -eq 0 && "${#POLICY_FAILURES[@]}" -gt 0 ]]; then
    exit 1
fi

exit "${CTEST_STATUS}"
