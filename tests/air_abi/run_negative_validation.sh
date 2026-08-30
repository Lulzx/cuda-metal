#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <air-validate-binary> <air-inspect-binary> <work-dir>" >&2
  exit 2
fi

AIR_VALIDATE="$1"
AIR_INSPECT="$2"
WORK_DIR="$3"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

run_expect_failure() {
  local label="$1"
  shift
  local logfile="$WORK_DIR/${label}.log"

  if "$@" >"$logfile" 2>&1; then
    echo "FAIL: expected command to fail for $label" >&2
    cat "$logfile" >&2
    exit 1
  fi
}

# Missing file should fail with I/O error.
run_expect_failure missing "$AIR_VALIDATE" "$WORK_DIR/does_not_exist.metallib"

# Empty file should fail explicit empty-file check.
: > "$WORK_DIR/empty.metallib"
run_expect_failure empty "$AIR_VALIDATE" "$WORK_DIR/empty.metallib"

# Non-metallib payload should fail strict magic/bitcode validation.
printf 'NOTAMETALLIB\x00\x01\x02\x03' > "$WORK_DIR/bad_magic.metallib"
run_expect_failure bad_magic "$AIR_VALIDATE" "$WORK_DIR/bad_magic.metallib"

# Missing function list requirement should fail for malformed input.
run_expect_failure require_function_list "$AIR_VALIDATE" \
  "$WORK_DIR/bad_magic.metallib" --require-function-list

# Missing metadata requirement should fail for malformed input.
run_expect_failure require_metadata "$AIR_VALIDATE" \
  "$WORK_DIR/bad_magic.metallib" --require-metadata

# Numeric options must reject malformed values instead of silently becoming 0.
run_expect_failure inspect_bad_count "$AIR_INSPECT" \
  "$WORK_DIR/bad_magic.metallib" --max-strings not-a-number

# Both JSON modes must escape every control byte in user-controlled paths.
CONTROL_PATH="${WORK_DIR}/control"$'\t'"path.metallib"
cp "$WORK_DIR/bad_magic.metallib" "$CONTROL_PATH"
"$AIR_INSPECT" "$CONTROL_PATH" --json >"$WORK_DIR/inspect-control.json"
python3 -c 'import json,sys; json.load(sys.stdin)' <"$WORK_DIR/inspect-control.json"
if "$AIR_VALIDATE" "$CONTROL_PATH" --json >"$WORK_DIR/validate-control.json"; then
  echo "FAIL: malformed control-path input unexpectedly validated" >&2
  exit 1
fi
python3 -c 'import json,sys; json.load(sys.stdin)' <"$WORK_DIR/validate-control.json"

echo "PASS: air_validate rejects malformed inputs as expected"
