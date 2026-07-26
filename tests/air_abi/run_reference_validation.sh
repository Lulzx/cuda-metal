#!/usr/bin/env bash
set -euo pipefail

GENERATE_SCRIPT="$1"
VALIDATOR="$2"
INSPECTOR="$3"
REFERENCE_METALLIB="$4"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MLA_CMD="swift run --package-path $ROOT_DIR/tools/metal_library_archive_bridge cumetal-mla-validate"

# A green AIR ABI result is only interpretable alongside the toolchain that produced it.
# shellcheck source=tests/air_abi/_toolchain_provenance.sh
source "$(dirname "$0")/_toolchain_provenance.sh"
cumetal_print_toolchain_provenance "air_abi_validate_reference"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "SKIP: xcrun not installed"
  exit 77
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "SKIP: xcrun metal not available"
  exit 77
fi

if ! xcrun --find metallib >/dev/null 2>&1; then
  echo "SKIP: xcrun metallib not available"
  exit 77
fi

"$GENERATE_SCRIPT"

if [[ ! -s "$REFERENCE_METALLIB" ]]; then
  echo "FAIL: expected metallib at $REFERENCE_METALLIB"
  exit 1
fi

"$VALIDATOR" \
  "$REFERENCE_METALLIB" \
  --xcrun \
  --mla \
  --mla-cmd "$MLA_CMD" \
  --require-function-list \
  --require-metadata

INSPECT_JSON="$($INSPECTOR "$REFERENCE_METALLIB" --json)"
echo "$INSPECT_JSON" | rg '"magic": "MTLB"' >/dev/null

echo "PASS: reference metallib validation completed"
