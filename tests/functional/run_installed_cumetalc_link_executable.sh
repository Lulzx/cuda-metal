#!/usr/bin/env bash
set -euo pipefail

INSTALL_SCRIPT="${1:?usage: run_installed_cumetalc_link_executable.sh <install.sh> <build-dir> <source.cu> <link-test.sh>}"
BUILD_DIR="${2:?}"
SOURCE_CU="${3:?}"
LINK_TEST="${4:?}"

TMP_ROOT="$(mktemp -d)"
TMP_ROOT="$(cd "$TMP_ROOT" && pwd -P)"
PREFIX="$TMP_ROOT/prefix"
HOME_DIR="$TMP_ROOT/home"
mkdir -p "$HOME_DIR"

cleanup() {
  if [[ -x "$PREFIX/uninstall.sh" ]]; then
    env -u CUMETAL_SHELL_RC HOME="$HOME_DIR" SHELL=/bin/zsh \
      bash "$PREFIX/uninstall.sh" "$PREFIX" >/dev/null
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

env -u CUMETAL_SHELL_RC HOME="$HOME_DIR" SHELL=/bin/zsh \
  bash "$INSTALL_SCRIPT" "$BUILD_DIR" "$PREFIX"

if [[ -e "$HOME_DIR/.zshrc" ]]; then
  echo "FAIL: default installation modified a shell startup file" >&2
  exit 1
fi

"$PREFIX/bin/cumetal" doctor
INSTALLED_EXAMPLE="$PREFIX/share/cumetal/examples/vectorAdd.cu"
if [[ ! -f "$INSTALLED_EXAMPLE" ]]; then
  echo "FAIL: installed doctor example is missing: $INSTALLED_EXAMPLE" >&2
  exit 1
fi
cmp "$SOURCE_CU" "$INSTALLED_EXAMPLE"
if ! NO_COLOR=1 "$PREFIX/bin/cumetal" doctor | grep -qF \
    "cumetalc '$INSTALLED_EXAMPLE' -o /tmp/vectorAdd"; then
  echo "FAIL: doctor did not print a copy-paste command for the installed example" >&2
  exit 1
fi
bash "$LINK_TEST" \
  "$PREFIX/bin/cumetalc" \
  "$INSTALLED_EXAMPLE" \
  "$TMP_ROOT/compile-and-run"

echo "PASS: a fresh installed prefix compiled and ran unmodified CUDA source"
