#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
PREFIX="${2:-/opt/cumetal}"
SHELL_CONFIG="${3:-}"

MARKER_BEGIN="# >>> cumetal >>>"
MARKER_END="# <<< cumetal <<<"

if [[ -n "$SHELL_CONFIG" && "$SHELL_CONFIG" != "--shell-config" ]]; then
  echo "unknown option: $SHELL_CONFIG" >&2
  echo "usage: $0 [build-dir] [prefix] [--shell-config]" >&2
  exit 2
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "build directory not found: $BUILD_DIR" >&2
  exit 1
fi

if [[ -z "$PREFIX" || "$PREFIX" == "/" ]]; then
  echo "refusing to install into an empty prefix or filesystem root" >&2
  exit 2
fi
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd -P)"

cmake --install "$BUILD_DIR" --prefix "$PREFIX"

install -m 755 "$(dirname "$0")/uninstall.sh" "$PREFIX/uninstall.sh"
mkdir -p "$PREFIX/share/cumetal"
if [[ -f "$BUILD_DIR/install_manifest.txt" ]]; then
  install -m 644 "$BUILD_DIR/install_manifest.txt" "$PREFIX/share/cumetal/install_manifest.txt"
fi

if [[ "$SHELL_CONFIG" == "--shell-config" || -n "${CUMETAL_SHELL_RC:-}" ]]; then
  # Shell startup files are changed only by explicit request. CUMETAL_SHELL_RC
  # both opts in and selects a file, which keeps automated installs isolated.
  if [[ -n "${CUMETAL_SHELL_RC:-}" ]]; then
    SHELL_RC="$CUMETAL_SHELL_RC"
    IS_FISH=0
    if [[ "$SHELL_RC" == *config.fish ]]; then IS_FISH=1; fi
  elif [[ "${SHELL:-}" == */fish ]]; then
    SHELL_RC="${HOME}/.config/fish/config.fish"
    IS_FISH=1
  else
    SHELL_RC="${HOME}/.zshrc"
    IS_FISH=0
  fi

  mkdir -p "$(dirname "$SHELL_RC")"
  touch "$SHELL_RC"

  if grep -qF "$MARKER_BEGIN" "$SHELL_RC"; then
    tmp="$(mktemp)"
    awk -v begin="$MARKER_BEGIN" -v end="$MARKER_END" '
      $0 == begin {skip=1; next}
      $0 == end {skip=0; next}
      skip != 1 {print}
    ' "$SHELL_RC" > "$tmp"
    mv "$tmp" "$SHELL_RC"
  fi

  if [[ "$IS_FISH" -eq 1 ]]; then
    cat >> "$SHELL_RC" <<EOF
$MARKER_BEGIN
set -gx PATH "$PREFIX/bin" \$PATH
$MARKER_END
EOF
  else
    cat >> "$SHELL_RC" <<EOF
$MARKER_BEGIN
export PATH="$PREFIX/bin:\$PATH"
$MARKER_END
EOF
  fi

  # Record the exact file changed so uninstall only reverses shell integration
  # created by this installation. A default install must not affect an older or
  # unrelated CuMetal marker in the user's current shell configuration.
  printf '%s\n' "$SHELL_RC" > "$PREFIX/share/cumetal/shell_config_path"

  echo "Updated $SHELL_RC with CuMetal's bin directory"
fi

echo "Installed CuMetal to $PREFIX"
echo "Run $PREFIX/bin/cumetal doctor to verify the installation"
if [[ "$SHELL_CONFIG" != "--shell-config" && -z "${CUMETAL_SHELL_RC:-}" ]]; then
  echo "Add $PREFIX/bin to PATH, or rerun with --shell-config to do that automatically"
fi
