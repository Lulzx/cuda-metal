#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <ci-report-script>" >&2
    exit 2
fi

REPORT_SCRIPT="$1"
TEST_ROOT="$(mktemp -d -t cumetal-ci-report-test)"
trap 'rm -rf "${TEST_ROOT}"' EXIT

mkdir -p "${TEST_ROOT}/bin"
cat >"${TEST_ROOT}/bin/ctest" <<'EOF'
#!/usr/bin/env bash
set -u

case "${FAKE_CTEST_RESULT:-pass}" in
    pass)
        cat <<'OUTPUT'
Test project /tmp/cumetal
    Start 1: fixture_pass
1/1 Test #1: fixture_pass .....................   Passed    0.01 sec
100% tests passed, 0 tests failed out of 1
OUTPUT
        exit 0
        ;;
    skip)
        cat <<'OUTPUT'
Test project /tmp/cumetal
    Start 1: fixture_skip
1/1 Test #1: fixture_skip .....................***Skipped   0.01 sec
100% tests passed, 0 tests failed out of 1
OUTPUT
        exit 0
        ;;
    empty)
        cat <<'OUTPUT'
Test project /tmp/cumetal
No tests were found!!!
OUTPUT
        exit 0
        ;;
    fail)
        cat <<'OUTPUT'
Test project /tmp/cumetal
    Start 1: fixture_fail
1/1 Test #1: fixture_fail .....................***Failed    0.01 sec
0% tests passed, 1 tests failed out of 1
OUTPUT
        exit 8
        ;;
    *)
        echo "unexpected fake result: ${FAKE_CTEST_RESULT}" >&2
        exit 2
        ;;
esac
EOF
chmod +x "${TEST_ROOT}/bin/ctest"

run_report() {
    PATH="${TEST_ROOT}/bin:${PATH}" \
        GITHUB_STEP_SUMMARY="${TEST_ROOT}/summary.md" \
        FAKE_CTEST_RESULT="$1" \
        bash "${REPORT_SCRIPT}" "${TEST_ROOT}/build" "${@:2}" \
        >"${TEST_ROOT}/output.log" 2>&1
}

run_report pass --require-tests --require-no-skips --label-regex '^hosted$'
run_report skip --require-tests

if run_report skip --require-tests --require-no-skips; then
    echo "FAIL: --require-no-skips accepted a skipped test" >&2
    exit 1
fi
if ! grep -q 'selected test(s) skipped' "${TEST_ROOT}/output.log"; then
    echo "FAIL: skipped-test policy failure was not reported" >&2
    exit 1
fi

if run_report empty --require-tests; then
    echo "FAIL: --require-tests accepted an empty selection" >&2
    exit 1
fi
if ! grep -q 'CTest selection was empty' "${TEST_ROOT}/output.log"; then
    echo "FAIL: empty-selection policy failure was not reported" >&2
    exit 1
fi

if run_report fail; then
    echo "FAIL: the underlying CTest failure status was lost" >&2
    exit 1
fi

echo "PASS: ci_report policy gates reject skips and empty selections"
