#!/usr/bin/env bash
set -euo pipefail

INSTALL_SCRIPT="$1"
UNINSTALL_SCRIPT="$2"
BUILD_DIR="$3"

TMP_ROOT="$(mktemp -d)"
TMP_ROOT="$(cd "$TMP_ROOT" && pwd -P)"
PREFIX="$TMP_ROOT/prefix"
SHELL_RC="$TMP_ROOT/.zshrc"
HOME_DIR="$TMP_ROOT/home"
mkdir -p "$HOME_DIR"
touch "$SHELL_RC"

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

# Default installation must not mutate shell startup files.
env -u CUMETAL_SHELL_RC HOME="$HOME_DIR" SHELL=/bin/zsh \
  bash "$INSTALL_SCRIPT" "$BUILD_DIR" "$PREFIX"

test -x "$PREFIX/bin/air_inspect"
test -x "$PREFIX/bin/air_validate"
test -x "$PREFIX/bin/cumetal-air-emitter"
test -x "$PREFIX/bin/cumetal"
test -x "$PREFIX/bin/cumetalc"
test -f "$PREFIX/lib/libcumetal.dylib"
test -f "$PREFIX/include/cuda.h"
test -f "$PREFIX/include/cuda_runtime.h"
test -f "$PREFIX/share/cumetal/examples/vectorAdd.cu"
test -x "$PREFIX/uninstall.sh"
test -f "$PREFIX/share/cumetal/install_manifest.txt"

test ! -e "$HOME_DIR/.zshrc"
"$PREFIX/bin/cumetal" version | grep -q "^cumetal "
"$PREFIX/bin/cumetal" run /usr/bin/true

env -u CUMETAL_SHELL_RC HOME="$HOME_DIR" SHELL=/bin/zsh \
  bash "$PREFIX/uninstall.sh" "$PREFIX"

if [[ -e "$PREFIX/bin/air_inspect" || -e "$PREFIX/lib/libcumetal.dylib" ]]; then
  echo "FAIL: expected installed files to be removed" >&2
  exit 1
fi

# Shell configuration remains available as an explicit opt-in and is reversible.
CUMETAL_SHELL_RC="$SHELL_RC" bash "$INSTALL_SCRIPT" "$BUILD_DIR" "$PREFIX" --shell-config
grep -qF "# >>> cumetal >>>" "$SHELL_RC"
grep -qF "# <<< cumetal <<<" "$SHELL_RC"
grep -qF "export PATH=\"$PREFIX/bin:\$PATH\"" "$SHELL_RC"
if grep -qF "DYLD_" "$SHELL_RC"; then
  echo "FAIL: installer should not add global DYLD variables" >&2
  exit 1
fi

CUMETAL_SHELL_RC="$SHELL_RC" bash "$PREFIX/uninstall.sh" "$PREFIX"
if grep -qF "# >>> cumetal >>>" "$SHELL_RC"; then
  echo "FAIL: expected shell marker to be removed" >&2
  exit 1
fi

if bash "$INSTALL_SCRIPT" "$BUILD_DIR" "$PREFIX" --not-an-option >/dev/null 2>&1; then
  echo "FAIL: installer accepted an unknown option" >&2
  exit 1
fi
if bash "$INSTALL_SCRIPT" "$BUILD_DIR" / >/dev/null 2>&1; then
  echo "FAIL: installer accepted the filesystem root as its prefix" >&2
  exit 1
fi
if bash "$UNINSTALL_SCRIPT" / >/dev/null 2>&1; then
  echo "FAIL: uninstaller accepted the filesystem root as its prefix" >&2
  exit 1
fi

echo "PASS: install/uninstall scripts are isolated, manifest-backed, and reversible"
