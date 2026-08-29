#!/usr/bin/env bash
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"
FALLBACK_METALLIB="${3:-}"

# This exercises the real fatbin PTX -> direct MSL -> Metal compute path.  Direct
# MSL is compiled by MTLDevice::newLibraryWithSource, so modern Xcode installs do
# not need the removed standalone `metallib` command-line utility.
OUTPUT_FILE="$(mktemp)"
CACHE_DIR="$(mktemp -d)"
trap 'rm -f "$OUTPUT_FILE"; rm -rf "$CACHE_DIR"' EXIT

set +e
CUMETAL_CACHE_DIR="$CACHE_DIR" CUMETAL_TRACE_GPU=1 \
  "$TEST_BINARY" "$PTX_PATH" "$FALLBACK_METALLIB" >"$OUTPUT_FILE" 2>&1
STATUS=$?
set -e
cat "$OUTPUT_FILE"

if [[ $STATUS -eq 77 ]]; then
  exit 77
fi
if [[ $STATUS -ne 0 ]]; then
  exit "$STATUS"
fi
if ! grep -q 'CUMETAL_PROVENANCE .*source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu .*launch_success=true' "$OUTPUT_FILE"; then
  echo "FAIL: correct output was produced without evidence of a Metal GPU dispatch"
  exit 1
fi
for name in \
  "LZ4 fatbin" \
  "Zstd fatbin" \
  "ELF LZ4 fatbin" \
  "ELF Zstd fatbin"
do
  if ! grep -Fqx "COMPRESSED_REGISTRATION_OK ${name}" "$OUTPUT_FILE"; then
    echo "FAIL: missing compressed registration execution marker for ${name}"
    exit 1
  fi
done
if [[ -n "$FALLBACK_METALLIB" ]] &&
   ! grep -Fqx "UNSUPPORTED_FATBIN_FALLBACK_REFUSED" "$OUTPUT_FILE"; then
  echo "FAIL: unsupported fatbin environment-fallback refusal was not proved"
  exit 1
fi
