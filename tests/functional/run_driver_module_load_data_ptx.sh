#!/usr/bin/env bash
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"

# Tests cuModuleLoadData with PTX text + fatbin variants, plus launch of the loaded kernel.
# The launch verification requires being able to produce a usable metallib for the kernel;
# the direct lowering or packaging for 'vector_add' needs the metal compiler in this flow.
if ! command -v xcrun >/dev/null 2>&1; then
  echo "SKIP: xcrun not installed"
  exit 77
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "SKIP: xcrun metal not available (driver ptx load+launch test requires it for this kernel)"
  exit 77
fi

if ! xcrun --find metallib >/dev/null 2>&1; then
  echo "SKIP: xcrun metallib not available"
  exit 77
fi

OUTPUT_FILE="$(mktemp)"
CACHE_DIR="$(mktemp -d)"
trap 'rm -f "$OUTPUT_FILE"; rm -rf "$CACHE_DIR"' EXIT

set +e
CUMETAL_CACHE_DIR="$CACHE_DIR" CUMETAL_TRACE_GPU=1 \
  "$TEST_BINARY" "$PTX_PATH" >"$OUTPUT_FILE" 2>&1
STATUS=$?
set -e
cat "$OUTPUT_FILE"
if [[ $STATUS -ne 0 ]]; then
  exit "$STATUS"
fi
for name in \
  "LZ4 fatbin" \
  "Zstd fatbin" \
  "ELF LZ4 fatbin" \
  "ELF Zstd fatbin"
do
  if ! grep -Fqx "COMPRESSED_DRIVER_OK ${name}" "$OUTPUT_FILE"; then
    echo "FAIL: missing compressed Driver API execution marker for ${name}"
    exit 1
  fi
done
if ! grep -q 'CUMETAL_PROVENANCE .*device=apple_gpu .*launch_success=true' \
    "$OUTPUT_FILE"; then
  echo "FAIL: compressed Driver API output lacks Apple-GPU provenance"
  exit 1
fi
