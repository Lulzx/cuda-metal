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
bash "$LINK_TEST" \
  "$PREFIX/bin/cumetalc" \
  "$SOURCE_CU" \
  "$TMP_ROOT/compile-and-run"

echo "PASS: a fresh installed prefix compiled and ran unmodified CUDA source"
