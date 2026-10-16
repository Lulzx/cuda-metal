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
    exception)
        cat <<'OUTPUT'
Test project /tmp/cumetal
    Start 1: fixture_exception
1/1 Test #1: fixture_exception ................***Exception: SegFault  0.01 sec
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
    : >"${TEST_ROOT}/summary.md"
    PATH="${TEST_ROOT}/bin:${PATH}" \
        GITHUB_STEP_SUMMARY="${TEST_ROOT}/summary.md" \
        FAKE_CTEST_RESULT="$1" \
        bash "${REPORT_SCRIPT}" "${TEST_ROOT}/build" "${@:2}" \
        >"${TEST_ROOT}/output.log" 2>&1
}

assert_counts() {
    local result
    for result in "Passed | $1" "Skipped | $2" "Failed | $3" "Registered | $4"; do
        if ! grep -Fqx "| ${result} |" "${TEST_ROOT}/summary.md"; then
            echo "FAIL: missing summary count: ${result}" >&2
            cat "${TEST_ROOT}/summary.md" >&2
            exit 1
        fi
    done
}

assert_names() {
    local heading="$1"
    shift
    printf '%s\n' "$@" >"${TEST_ROOT}/expected-names"
    awk -v heading="### ${heading} tests" '
        /^### / { active = ($0 == heading); next }
        active && /^- / { sub(/^- /, ""); print }
    ' "${TEST_ROOT}/summary.md" >"${TEST_ROOT}/actual-names"
    diff -u "${TEST_ROOT}/expected-names" "${TEST_ROOT}/actual-names"
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

status=0
run_report exception || status=$?
if [[ "${status}" -ne 8 ]]; then
    echo "FAIL: CTest exception exit status changed to ${status}" >&2
    exit 1
fi
assert_counts 0 0 1 1
assert_names Failed fixture_exception

# Exercise CTest's real Timeout, Not Run, and Disabled spellings without a build.
mkdir -p "${TEST_ROOT}/real-build"
CMAKE_COMMAND="$(command -v cmake)"
cat >"${TEST_ROOT}/fail.sh" <<'EOF'
#!/usr/bin/env bash
echo 'child output ...   Passed'
echo 'child output ...***Skipped'
echo 'child output ...***Failed'
echo 'child output ...***Exception'
exit 1
EOF
cat >"${TEST_ROOT}/real-build/CTestTestfile.cmake" <<EOF
add_test(fixture_pass "${CMAKE_COMMAND}" -E true)
add_test(fixture_fail "${BASH}" "${TEST_ROOT}/fail.sh")
add_test(fixture_timeout "${CMAKE_COMMAND}" -E sleep 5)
set_tests_properties(fixture_timeout PROPERTIES TIMEOUT 0.1)
add_test(fixture_skip "${BASH}" -c "exit 77")
set_tests_properties(fixture_skip PROPERTIES SKIP_RETURN_CODE 77)
add_test(fixture_missing "${TEST_ROOT}/missing-executable")
add_test(fixture_disabled "${CMAKE_COMMAND}" -E true)
set_tests_properties(fixture_disabled PROPERTIES DISABLED TRUE)
EOF

run_real_report() {
    : >"${TEST_ROOT}/summary.md"
    GITHUB_STEP_SUMMARY="${TEST_ROOT}/summary.md" \
        bash "${REPORT_SCRIPT}" "${TEST_ROOT}/real-build" "$@" \
        >"${TEST_ROOT}/output.log" 2>&1
}

status=0
run_real_report --require-tests -j 1 || status=$?
if [[ "${status}" -ne 8 ]]; then
    echo "FAIL: real CTest failure exit status changed to ${status}" >&2
    cat "${TEST_ROOT}/output.log" >&2
    exit 1
fi
assert_counts 1 2 3 6
assert_names Skipped fixture_skip fixture_disabled
assert_names Failed fixture_fail fixture_timeout fixture_missing
sed -n '/^## CuMetal test summary/,$p' "${TEST_ROOT}/output.log" >"${TEST_ROOT}/console-summary"
sed -n '/^## CuMetal test summary/,$p' "${TEST_ROOT}/summary.md" >"${TEST_ROOT}/github-summary"
diff -u "${TEST_ROOT}/console-summary" "${TEST_ROOT}/github-summary"

status=0
run_real_report --require-tests -R '^fixture_disabled$' || status=$?
if [[ "${status}" -ne 0 ]]; then
    echo "FAIL: disabled-only selection failed without the no-skips policy" >&2
    exit 1
fi
assert_counts 0 1 0 1
assert_names Skipped fixture_disabled

status=0
run_real_report --require-tests --require-no-skips -R '^fixture_disabled$' || status=$?
if [[ "${status}" -ne 1 ]]; then
    echo "FAIL: disabled-only strict policy exit status changed to ${status}" >&2
    exit 1
fi
if ! grep -Fq '1 selected test(s) skipped (--require-no-skips).' "${TEST_ROOT}/output.log"; then
    echo "FAIL: disabled-test policy failure was not reported" >&2
    exit 1
fi

echo "PASS: ci_report counts real CTest outcomes and preserves policy gates and exit status"
