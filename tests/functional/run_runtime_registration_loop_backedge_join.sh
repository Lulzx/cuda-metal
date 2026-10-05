#!/usr/bin/env bash
set -euo pipefail

TEST_BINARY="$1"

# Run the same registered PTX through both JIT backends. Correct sums are only
# meaningful if the kernel actually ran on the Apple GPU.
for backend in cumetal-ir legacy; do
  OUTPUT_FILE="$(mktemp)"
  CACHE_DIR="$(mktemp -d)"
  set +e
  CUMETAL_CACHE_DIR="$CACHE_DIR" CUMETAL_TRACE_GPU=1 CUMETAL_PTX_BACKEND="$backend" \
    "$TEST_BINARY" >"$OUTPUT_FILE" 2>&1
  STATUS=$?
  set -e
  echo "== CUMETAL_PTX_BACKEND=$backend"
  cat "$OUTPUT_FILE"
  OK=$(grep -Fxc "LOOP_BACKEDGE_JOIN_OK" "$OUTPUT_FILE" || true)
  LAUNCHES=$(grep -c 'device=apple_gpu .*launch_success=true' "$OUTPUT_FILE" || true)
  rm -f "$OUTPUT_FILE"
  rm -rf "$CACHE_DIR"
  if [[ $STATUS -eq 77 ]]; then
    exit 77
  fi
  if [[ $STATUS -ne 0 ]]; then
    exit "$STATUS"
  fi
  if [[ "$OK" -ne 1 ]]; then
    echo "FAIL: expected LOOP_BACKEDGE_JOIN_OK under $backend"
    exit 1
  fi
  if [[ "$LAUNCHES" -lt 4 ]]; then
    echo "FAIL: all four launches must report a successful Apple GPU dispatch under $backend"
    exit 1
  fi
done
