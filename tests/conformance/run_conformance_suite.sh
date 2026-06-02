#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "usage: $0 <build_dir> [min_pass_rate_percent] [ctest_regex]"
    exit 2
fi

BUILD_DIR="$1"
MIN_PASS_RATE="${2:-90}"
TEST_REGEX="${3:-^functional_}"
SINGLE_TEST_TIMEOUT="${CUMETAL_CONFORMANCE_SINGLE_TEST_TIMEOUT:-120}"

if ! command -v ctest >/dev/null 2>&1; then
    echo "SKIP: ctest not available"
    exit 77
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: python3 not available for ctest JSON parsing"
    exit 77
fi

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "FAIL: build directory not found: $BUILD_DIR"
    exit 1
fi

if ! [[ "$SINGLE_TEST_TIMEOUT" =~ ^[0-9]+$ ]] || [[ "$SINGLE_TEST_TIMEOUT" -eq 0 ]]; then
    echo "FAIL: CUMETAL_CONFORMANCE_SINGLE_TEST_TIMEOUT must be a positive integer"
    exit 1
fi

JSON_FILE="$(mktemp)"
trap 'rm -f "$JSON_FILE"' EXIT

ctest --test-dir "$BUILD_DIR" --show-only=json-v1 >"$JSON_FILE"

TEST_NAMES=()
while IFS= read -r name; do
    if [ -n "$name" ]; then
        TEST_NAMES+=("$name")
    fi
done < <(
    python3 - "$JSON_FILE" "$TEST_REGEX" <<'PY'
import json
import re
import sys

path = sys.argv[1]
pattern = re.compile(sys.argv[2])

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for test in data.get("tests", []):
    name = test.get("name", "")
    if pattern.search(name):
        print(name)
PY
)

if [[ ${#TEST_NAMES[@]} -eq 0 ]]; then
    echo "SKIP: no tests matched regex '${TEST_REGEX}'"
    exit 77
fi

passed=0
failed=0
skipped=0
total="${#TEST_NAMES[@]}"
index=0

echo "Conformance run: matched ${total} tests (timeout=${SINGLE_TEST_TIMEOUT}s each)"

for test_name in "${TEST_NAMES[@]}"; do
    index=$((index + 1))
    echo "[${index}/${total}] ${test_name}"
    output="$(ctest --test-dir "$BUILD_DIR" --timeout "$SINGLE_TEST_TIMEOUT" -R "^${test_name}$" --output-on-failure 2>&1 || true)"
    if grep -Fq "Not Run (Disabled)" <<<"$output" || grep -Fq "***Skipped" <<<"$output"; then
        echo "  -> skipped"
        skipped=$((skipped + 1))
        continue
    fi
    if grep -Fq "100% tests passed, 0 tests failed out of 1" <<<"$output"; then
        echo "  -> pass"
        passed=$((passed + 1))
    else
        echo "$output"
        echo "  -> fail"
        failed=$((failed + 1))
    fi
done

executed=$((passed + failed))
if [[ $executed -eq 0 ]]; then
    echo "SKIP: all matched tests were skipped"
    exit 77
fi

pass_rate="$(python3 - "$passed" "$executed" <<'PY'
import sys
passed = int(sys.argv[1])
executed = int(sys.argv[2])
print(f"{(100.0 * passed / executed):.2f}")
PY
)"

echo "Conformance summary:"
echo "  matched:  ${#TEST_NAMES[@]}"
echo "  executed: ${executed}"
echo "  passed:   ${passed}"
echo "  failed:   ${failed}"
echo "  skipped:  ${skipped}"
echo "  pass_rate(executed): ${pass_rate}%"
echo "  threshold:           ${MIN_PASS_RATE}%"

if python3 - "$pass_rate" "$MIN_PASS_RATE" <<'PY'
import sys
pass_rate = float(sys.argv[1])
threshold = float(sys.argv[2])
sys.exit(0 if pass_rate + 1e-9 >= threshold else 1)
PY
then
    echo "PASS: conformance threshold met"
    exit 0
fi

echo "FAIL: conformance threshold not met"
exit 1
