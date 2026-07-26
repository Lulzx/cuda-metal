#!/usr/bin/env bash
# spec.md §10.5: verify AIR ABI invariants hold across Xcode toolchains. §13 rates a silent AIR
# ABI change between Xcode releases as the project's highest risk, so this test only earns its
# name when it actually compiles with two *different* toolchains.
#
# It previously defaulted both slots to `xcode-select -p` when the override variables were unset,
# compiled the same source twice with the same compiler, and printed "Xcode matrix ABI regression
# checks succeeded" -- reporting cross-version coverage that had not happened. It also extracted a
# function-hash prefix from each run and then never compared them.
#
# Now: toolchains are identified by their actual metal compiler version rather than by directory
# path (two paths can point at one Xcode), a single-toolchain run says so plainly instead of
# claiming a matrix, and the hashes are compared with the comparison's meaning spelled out.
# Set CUMETAL_REQUIRE_XCODE_MATRIX=1 to make a single-toolchain environment a failure.
set -euo pipefail

AIR_INSPECT="$1"
SOURCE_METAL="$2"
WORKDIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/air_abi/_toolchain_provenance.sh
source "${SCRIPT_DIR}/_toolchain_provenance.sh"
cumetal_print_toolchain_provenance "air_abi_xcode_matrix"

DEFAULT_DEVELOPER_DIR="$(xcode-select -p 2>/dev/null || true)"
XCODE15_DIR="${CUMETAL_XCODE15_DEVELOPER_DIR:-$DEFAULT_DEVELOPER_DIR}"
XCODE16_DIR="${CUMETAL_XCODE16_DEVELOPER_DIR:-$DEFAULT_DEVELOPER_DIR}"

if [[ -z "$XCODE15_DIR" && -z "$XCODE16_DIR" ]]; then
  echo "SKIP: no Xcode developer directory found"
  echo "      set CUMETAL_XCODE15_DEVELOPER_DIR/CUMETAL_XCODE16_DEVELOPER_DIR or run xcode-select --switch"
  exit 77
fi

# Identify a toolchain by what actually determines its AIR output.
metal_version_for() {
  DEVELOPER_DIR="$1" xcrun metal --version 2>/dev/null | head -1 || true
}

declare -a LABELS=()
declare -a DIRS=()
declare -a VERSIONS=()

add_toolchain() {
  local label="$1" dir="$2"
  [[ -n "$dir" && -d "$dir" ]] || return 0
  if ! DEVELOPER_DIR="$dir" xcrun --find metal >/dev/null 2>&1 ||
     ! DEVELOPER_DIR="$dir" xcrun --find metallib >/dev/null 2>&1; then
    echo "INFO: skipping ${label}: no metal/metallib at ${dir}"
    return 0
  fi
  local version
  version="$(metal_version_for "$dir")"
  # Deduplicate by compiler version, so two paths pointing at one Xcode count once.
  local existing
  for existing in ${VERSIONS[@]+"${VERSIONS[@]}"}; do
    if [[ "$existing" == "$version" ]]; then
      echo "INFO: ${label} at ${dir} is the same toolchain as an earlier entry; not counted twice"
      return 0
    fi
  done
  LABELS+=("$label")
  DIRS+=("$dir")
  VERSIONS+=("$version")
}

add_toolchain "xcode15" "$XCODE15_DIR"
add_toolchain "xcode16" "$XCODE16_DIR"

TOOLCHAIN_COUNT="${#DIRS[@]}"
if [[ "$TOOLCHAIN_COUNT" -eq 0 ]]; then
  echo "SKIP: no usable Metal toolchain found"
  exit 77
fi

mkdir -p "$WORKDIR"

# The ABI invariants CuMetal depends on. These must hold for every toolchain we support; a
# change here is exactly the silent-ABI-drift event this test exists to catch.
run_for_toolchain() {
  local label="$1"
  local developer_dir="$2"
  local out_txt="$3"
  local out_air="$WORKDIR/${label}.air"
  local out_metallib="$WORKDIR/${label}.metallib"

  DEVELOPER_DIR="$developer_dir" xcrun metal -c "$SOURCE_METAL" -o "$out_air"
  DEVELOPER_DIR="$developer_dir" xcrun metallib "$out_air" -o "$out_metallib"
  "$AIR_INSPECT" "$out_metallib" > "$out_txt"

  local check
  for check in \
      "^Magic: MTLB" \
      "^Function count: 1$" \
      "\\[kernel 0\\] vector_add" \
      "air.version=2.8" \
      "language.version=4.0"; do
    if ! rg -q "$check" "$out_txt"; then
      echo "FAIL: ${label} failed ABI invariant check: ${check}"
      echo "      toolchain: $(metal_version_for "$developer_dir")"
      exit 1
    fi
  done
}

declare -a HASHES=()
for i in "${!DIRS[@]}"; do
  label="${LABELS[$i]}"
  out_txt="$WORKDIR/${label}.inspect.txt"
  echo "Compiling with ${label}: ${VERSIONS[$i]}"
  run_for_toolchain "$label" "${DIRS[$i]}" "$out_txt"
  hash_prefix="$(rg -o "function.hash.prefix=[0-9a-f]+" "$out_txt" | head -n1 || true)"
  if [[ -z "$hash_prefix" ]]; then
    echo "FAIL: could not extract a function hash prefix from ${label} output"
    exit 1
  fi
  HASHES+=("$hash_prefix")
  echo "  ${label}: ${hash_prefix}"
done

if [[ "$TOOLCHAIN_COUNT" -lt 2 ]]; then
  if [[ "${CUMETAL_REQUIRE_XCODE_MATRIX:-0}" == "1" ]]; then
    echo "FAIL: CUMETAL_REQUIRE_XCODE_MATRIX=1 but only one Metal toolchain is available."
    echo "      Install a second Xcode and point CUMETAL_XCODE15_DEVELOPER_DIR/"
    echo "      CUMETAL_XCODE16_DEVELOPER_DIR at the two developer directories."
    exit 1
  fi
  echo ""
  echo "PASS: AIR ABI invariants hold for the one available toolchain (${VERSIONS[0]})."
  echo "NOTE: this is NOT cross-version coverage. spec.md §10.5 asks for Xcode 15.0/15.4/16.0/16.2+;"
  echo "      only one toolchain is installed, so nothing here would catch an ABI change between"
  echo "      Xcode releases. Set CUMETAL_XCODE15_DEVELOPER_DIR and CUMETAL_XCODE16_DEVELOPER_DIR"
  echo "      to two different Xcodes, or rely on CI running across macOS runner images."
  exit 0
fi

# Two distinct toolchains agreed on every invariant. The hash prefixes are reported rather than
# required to match: they are derived from the compiled bitcode, so a legitimate codegen change
# between Xcode versions can move them without breaking the ABI CuMetal depends on. A change here
# is a signal worth reading, not an automatic failure.
if [[ "${HASHES[0]}" == "${HASHES[1]}" ]]; then
  echo ""
  echo "PASS: AIR ABI invariants hold across ${TOOLCHAIN_COUNT} toolchains; function hashes identical."
else
  echo ""
  echo "PASS: AIR ABI invariants hold across ${TOOLCHAIN_COUNT} toolchains."
  echo "NOTE: function hash prefixes differ between toolchains (${HASHES[0]} vs ${HASHES[1]})."
  echo "      Expected when codegen changes between Xcode versions; the ABI fields CuMetal reads"
  echo "      were verified on both."
fi
for i in "${!DIRS[@]}"; do
  echo "  ${LABELS[$i]}: ${VERSIONS[$i]}"
done
