#!/usr/bin/env bash
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"

CACHE_DIR="$(mktemp -d)"
FAST_LOG="$(mktemp)"
SAFE_LOG="$(mktemp)"
INVALID_LOG="$(mktemp)"
trap 'rm -rf "$CACHE_DIR"; rm -f "$FAST_LOG" "$SAFE_LOG" "$INVALID_LOG"' EXIT

run_mode() {
  local mode="$1"
  local log="$2"
  set +e
  CUMETAL_CACHE_DIR="$CACHE_DIR" \
  CUMETAL_MSL_MATH_MODE="$mode" \
  CUMETAL_TRACE_GPU=1 \
    "$TEST_BINARY" "$PTX_PATH" >"$log" 2>&1
  local status=$?
  set -e
  cat "$log"
  if [[ $status -eq 77 ]]; then
    exit 77
  fi
  if [[ $status -ne 0 ]]; then
    exit "$status"
  fi
}

run_mode fast "$FAST_LOG"
if ! grep -q 'CUMETAL_PROVENANCE .*math_mode=fast .*launch_success=true' "$FAST_LOG"; then
  echo "FAIL: fast mode did not produce successful Apple-GPU provenance"
  exit 1
fi

run_mode safe "$SAFE_LOG"
if ! grep -q 'CUMETAL_PROVENANCE .*math_mode=safe .*launch_success=true' "$SAFE_LOG"; then
  echo "FAIL: safe mode did not produce successful Apple-GPU provenance"
  exit 1
fi

artifact_count="$(
  find "$CACHE_DIR/registration-jit" -maxdepth 1 -type f \
    \( -name '*.metal' -o -name '*.metallib' \) -print 2>/dev/null |
    wc -l | tr -d ' '
)"
if [[ "$artifact_count" -ne 2 ]]; then
  echo "FAIL: fast and safe policies should produce two JIT cache artifacts, got $artifact_count"
  exit 1
fi

run_mode invalid-mode "$INVALID_LOG"
if ! grep -q "invalid CUMETAL_MSL_MATH_MODE='invalid-mode'" "$INVALID_LOG"; then
  echo "FAIL: invalid math mode did not emit a diagnostic"
  exit 1
fi
if ! grep -q 'CUMETAL_PROVENANCE .*math_mode=fast .*launch_success=true' "$INVALID_LOG"; then
  echo "FAIL: invalid math mode did not use the documented fast default"
  exit 1
fi

echo "PASS: fast/safe MSL math modes are selected, traced, and cache-isolated"
