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

FP64_SOURCE="${5:-}"
if [[ -n "$FP64_SOURCE" ]]; then
  # Issue #204: an installed cumetalc must compile software FP64 from its own
  # share/ tree, not the source checkout it was built from. Poison the
  # installed support source first and require the compile to see it.
  SUPPORT="$PREFIX/share/cumetal/metal/compiler/metal/support/cumetal_fp64_inline_support.metal"
  if [[ ! -f "$SUPPORT" ]]; then
    echo "FAIL: FP64 Metal support source was not installed" >&2
    exit 1
  fi
  cp "$SUPPORT" "$TMP_ROOT/support.orig"
  printf '#error "cumetal-installed-support-used"\n' >> "$SUPPORT"
  if "$PREFIX/bin/cumetalc" "$FP64_SOURCE" --fp64=ieee64 -o "$TMP_ROOT/fp64-poisoned" \
       >"$TMP_ROOT/poisoned.log" 2>&1; then
    echo "FAIL: installed cumetalc ignored its installed FP64 support source" >&2
    exit 1
  fi
  if ! grep -q "cumetal-installed-support-used" "$TMP_ROOT/poisoned.log"; then
    cat "$TMP_ROOT/poisoned.log" >&2
    echo "FAIL: FP64 compile failed for a reason other than the installed source" >&2
    exit 1
  fi
  cp "$TMP_ROOT/support.orig" "$SUPPORT"
  "$PREFIX/bin/cumetalc" "$FP64_SOURCE" --fp64=ieee64 -o "$TMP_ROOT/fp64"
  if [[ "$("$TMP_ROOT/fp64")" != PASS ]]; then
    echo "FAIL: installed FP64/bfloat16 program produced wrong results" >&2
    exit 1
  fi
  # The device pass must read the installed headers too, not the source tree
  # baked in at build time (Homebrew deletes it): poison them for device code.
  HEADER="$PREFIX/include/cuda_runtime.h"
  cp "$HEADER" "$TMP_ROOT/header.orig"
  printf '#ifdef __CUDA_ARCH__\n#error "cumetal-installed-header-used"\n#endif\n' >> "$HEADER"
  if "$PREFIX/bin/cumetalc" "$FP64_SOURCE" -o "$TMP_ROOT/header-poisoned" \
       >"$TMP_ROOT/header-poisoned.log" 2>&1 ||
     ! grep -q "cumetal-installed-header-used" "$TMP_ROOT/header-poisoned.log"; then
    cat "$TMP_ROOT/header-poisoned.log" >&2
    echo "FAIL: installed cumetalc's device pass ignored its installed headers" >&2
    exit 1
  fi
  cp "$TMP_ROOT/header.orig" "$HEADER"
  if otool -L "$PREFIX/bin/cumetalc" "$PREFIX/lib/libcumetal.dylib" | grep -q libz3; then
    echo "FAIL: installed binaries load libz3, which they never use" >&2
    exit 1
  fi
fi

echo "PASS: a fresh installed prefix compiled and ran unmodified CUDA source"
