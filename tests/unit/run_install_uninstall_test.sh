#!/usr/bin/env bash
set -euo pipefail

INSTALL_SCRIPT="$1"
UNINSTALL_SCRIPT="$2"
BUILD_DIR="$3"

TMP_ROOT="$(mktemp -d)"
PREFIX="$TMP_ROOT/prefix"
SHELL_RC="$TMP_ROOT/.zshrc"
touch "$SHELL_RC"

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

CUMETAL_SHELL_RC="$SHELL_RC" bash "$INSTALL_SCRIPT" "$BUILD_DIR" "$PREFIX"

test -x "$PREFIX/bin/air_inspect"
test -x "$PREFIX/bin/air_validate"
test -x "$PREFIX/bin/cumetal-air-emitter"
test -x "$PREFIX/bin/cumetalc"
test -f "$PREFIX/lib/libcumetal.dylib"
test -f "$PREFIX/include/cuda.h"
test -f "$PREFIX/include/cuda_runtime.h"
test -x "$PREFIX/uninstall.sh"

grep -qF "# >>> cumetal >>>" "$SHELL_RC"
grep -qF "# <<< cumetal <<<" "$SHELL_RC"
grep -qF "export PATH=\"$PREFIX/bin:\$PATH\"" "$SHELL_RC"
grep -qF "export DYLD_FALLBACK_LIBRARY_PATH=\"$PREFIX/lib:\${DYLD_FALLBACK_LIBRARY_PATH:-}\"" \
  "$SHELL_RC"

CUMETAL_SHELL_RC="$SHELL_RC" bash "$PREFIX/uninstall.sh" "$PREFIX"

if [[ -e "$PREFIX/bin/air_inspect" || -e "$PREFIX/lib/libcumetal.dylib" ]]; then
  echo "FAIL: expected installed files to be removed" >&2
  exit 1
fi

if grep -qF "# >>> cumetal >>>" "$SHELL_RC"; then
  echo "FAIL: expected shell marker to be removed" >&2
  exit 1
fi

echo "PASS: install/uninstall scripts manage files and shell config markers"
