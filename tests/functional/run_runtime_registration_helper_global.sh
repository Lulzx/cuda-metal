#!/usr/bin/env bash
set -euo pipefail

TEST_BINARY="$1"

# A correct pair of results is only meaningful if the kernel actually ran on the
# Apple GPU: a CPU fallback would satisfy the arithmetic without proving the
# hidden helper binding was populated.
OUTPUT_FILE="$(mktemp)"
CACHE_DIR="$(mktemp -d)"
trap 'rm -f "$OUTPUT_FILE"; rm -rf "$CACHE_DIR"' EXIT

set +e
# The typed importer owns hidden promoted-global bindings; the legacy LLVM JIT
# path has no Metal toolchain on this host and cannot produce an executable
# metallib for a module with device helper calls.
CUMETAL_CACHE_DIR="$CACHE_DIR" CUMETAL_TRACE_GPU=1 CUMETAL_PTX_BACKEND=cumetal-ir \
  "$TEST_BINARY" >"$OUTPUT_FILE" 2>&1
STATUS=$?
set -e
cat "$OUTPUT_FILE"

if [[ $STATUS -eq 77 ]]; then
  exit 77
fi
if [[ $STATUS -ne 0 ]]; then
  exit "$STATUS"
fi
if ! grep -Fqx "HELPER_ONLY_GLOBAL_OK 12 19" "$OUTPUT_FILE"; then
  echo "FAIL: expected the helper-only global to read 12 then 19"
  exit 1
fi
if [[ "$(grep -c 'device=apple_gpu .*launch_success=true' "$OUTPUT_FILE")" -lt 2 ]]; then
  echo "FAIL: both launches must report a successful Apple GPU dispatch"
  exit 1
fi
