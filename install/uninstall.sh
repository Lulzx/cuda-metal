#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-/opt/cumetal}"

MARKER_BEGIN="# >>> cumetal >>>"
MARKER_END="# <<< cumetal <<<"

if [[ -z "$PREFIX" || "$PREFIX" == "/" || ! -d "$PREFIX" ]]; then
  echo "refusing to uninstall from an empty, missing, or filesystem-root prefix" >&2
  exit 2
fi
PREFIX="$(cd "$PREFIX" && pwd -P)"

# Mirror the shell detection logic from install.sh.
if [[ -n "${CUMETAL_SHELL_RC:-}" ]]; then
  SHELL_RC="$CUMETAL_SHELL_RC"
elif [[ "${SHELL:-}" == */fish ]]; then
  SHELL_RC="${HOME}/.config/fish/config.fish"
else
  SHELL_RC="${HOME}/.zshrc"
fi

MANIFEST="$PREFIX/share/cumetal/install_manifest.txt"
if [[ -f "$MANIFEST" ]]; then
  DIRECTORY_LIST="$(mktemp)"
  while IFS= read -r installed_file; do
    [[ -z "$installed_file" ]] && continue
    case "$installed_file" in
      "$PREFIX"/*)
        rm -f "$installed_file"
        directory="$(dirname "$installed_file")"
        while [[ "$directory" == "$PREFIX"/* ]]; do
          printf '%s\n' "$directory" >> "$DIRECTORY_LIST"
          directory="$(dirname "$directory")"
        done
        ;;
      *)
        echo "Refusing to remove path outside prefix: $installed_file" >&2
        rm -f "$DIRECTORY_LIST"
        exit 1
        ;;
    esac
  done < "$MANIFEST"
  LC_ALL=C sort -ru "$DIRECTORY_LIST" | while IFS= read -r directory; do
    rmdir "$directory" 2>/dev/null || true
  done
  rm -f "$DIRECTORY_LIST"
else
  # Compatibility with installations made before manifests were recorded.
  rm -f "$PREFIX/bin/air_inspect"
  rm -f "$PREFIX/bin/air_validate"
  rm -f "$PREFIX/bin/cumetal-air-emitter"
  rm -f "$PREFIX/bin/cumetal"
  rm -f "$PREFIX/bin/cumetalc"
  rm -f "$PREFIX/lib/libcumetal.dylib"
  rm -f "$PREFIX/lib/libcublas.dylib"
  rm -f "$PREFIX/lib/libcufft.dylib"
  rm -f "$PREFIX/lib/libcurand.dylib"
  rm -f "$PREFIX/include/cuda.h"
  rm -f "$PREFIX/include/cuda_fp16.h"
  rm -f "$PREFIX/include/cuda_runtime.h"
  rm -f "$PREFIX/include/cufft.h"
  rm -f "$PREFIX/include/cublas_v2.h"
  rm -f "$PREFIX/include/curand.h"
  rm -f "$PREFIX/include/cooperative_groups.h"
  rm -f "$PREFIX/include/cooperative_groups/reduce.h"
fi
rm -f "$MANIFEST"
rm -f "$PREFIX/uninstall.sh"

if [[ -f "$SHELL_RC" ]] && grep -qF "$MARKER_BEGIN" "$SHELL_RC"; then
  tmp="$(mktemp)"
  awk -v begin="$MARKER_BEGIN" -v end="$MARKER_END" '
    $0 == begin {skip=1; next}
    $0 == end {skip=0; next}
    skip != 1 {print}
  ' "$SHELL_RC" > "$tmp"
  mv "$tmp" "$SHELL_RC"
fi

rmdir "$PREFIX/libexec/cumetal/cuda_toolchain" 2>/dev/null || true
rmdir "$PREFIX/libexec/cumetal" 2>/dev/null || true
rmdir "$PREFIX/libexec" 2>/dev/null || true
rmdir "$PREFIX/share/cumetal" 2>/dev/null || true
rmdir "$PREFIX/share" 2>/dev/null || true
rmdir "$PREFIX/bin" 2>/dev/null || true
rmdir "$PREFIX/lib" 2>/dev/null || true
rmdir "$PREFIX/include/cooperative_groups" 2>/dev/null || true
rmdir "$PREFIX/include" 2>/dev/null || true
rmdir "$PREFIX" 2>/dev/null || true

echo "Removed CuMetal from $PREFIX"
if [[ -f "$SHELL_RC" ]]; then
  echo "Removed CuMetal environment settings from $SHELL_RC"
fi
