#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <phase4-gate-script>"
    exit 2
fi

GATE="$1"
ROOT="$(mktemp -d -t cumetal-phase4-policy)"
trap 'rm -rf "${ROOT}"' EXIT
mkdir -p "${ROOT}/bin" "${ROOT}/build"

printf '%s\n' required_pass prerequisite_skip > "${ROOT}/manifest.txt"
cat > "${ROOT}/bin/ctest" <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *prerequisite_skip* ]]; then
    echo '1/1 Test #2: prerequisite_skip ........***Skipped 0.01 sec'
else
    echo '1/1 Test #1: required_pass .............   Passed 0.01 sec'
fi
echo '100% tests passed, 0 tests failed out of 1'
EOF
chmod +x "${ROOT}/bin/ctest"

set +e
OUTPUT="$(PATH="${ROOT}/bin:${PATH}" bash "${GATE}" \
    "${ROOT}/build" 90 "${ROOT}/manifest.txt" 2>&1)"
STATUS=$?
set -e

if [[ ${STATUS} -eq 0 ]]; then
    echo "FAIL: a prerequisite skip was excluded from the Phase 4 denominator"
    exit 1
fi
if [[ "${OUTPUT}" != *'pass_rate(required): 50.00%'* ]]; then
    echo "${OUTPUT}"
    echo "FAIL: Phase 4 gate did not report the manifest-denominator pass rate"
    exit 1
fi

echo "PASS: Phase 4 skips remain non-passing denominator entries"
