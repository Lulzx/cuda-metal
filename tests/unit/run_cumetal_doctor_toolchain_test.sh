#!/usr/bin/env bash
# `cumetal doctor` and `cumetal version --json` against simulated Metal
# toolchain states (#217): a fake `xcrun`/`xcode-select` earlier on PATH stands
# in for a missing toolchain, Xcode 26/27's launcher stub, a toolchain too old
# for CuMetal's AIR, one newer than the running macOS, and a working one.
set -euo pipefail

CUMETAL="${1:?usage: run_cumetal_doctor_toolchain_test.sh <cumetal>}"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT
FAKE_BIN="$TMP_ROOT/bin"
mkdir -p "$FAKE_BIN"
HOST_MACOS_MAJOR="$(/usr/bin/sw_vers -productVersion | cut -d. -f1)"

cat > "$FAKE_BIN/xcrun" <<'EOF'
#!/usr/bin/env bash
# Modes: missing | stub | old | newer | ok
mode="${FAKE_XCRUN_MODE:?}"
args=" $* "
if [[ "$1" == "--find" ]]; then
  [[ "$mode" == "missing" ]] && exit 1
  echo "/fake/toolchain/usr/bin/$2"
  exit 0
fi
[[ "$mode" == "missing" ]] && { echo "xcrun: error: unable to find utility" >&2; exit 72; }
if [[ "$args" == *" --version "* ]]; then
  if [[ "$mode" == "stub" ]]; then
    echo "error: error: cannot execute tool 'metal' due to missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain" >&2
    exit 1
  fi
  echo "Apple metal version 32023.0 (fake)"
  exit 0
fi
if [[ "$args" == *" -### "* ]]; then
  case "$mode" in
    old) echo '"-triple" "air64_v26-apple-macosx14.0.0"' ;;
    newer) echo '"-triple" "air64_v29-apple-macosx99.0.0"' ;;
    *) echo "\"-triple\" \"air64_v28-apple-macosx${FAKE_HOST_MACOS}.0.0\"" ;;
  esac
  exit 0
fi
if [[ "$args" == *" -std=metal3.2 "* && "$mode" == "old" ]]; then
  echo "error: invalid value 'metal3.2' in '-std=metal3.2'" >&2
  exit 1
fi
# compile or metallib: create the -o output
out=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then out="$2"; shift; fi
  shift
done
[[ -n "$out" ]] && : > "$out"
exit 0
EOF
cat > "$FAKE_BIN/xcode-select" <<'EOF'
#!/usr/bin/env bash
if [[ "${FAKE_XCRUN_MODE:-}" == "missing" ]]; then
  echo "/Library/Developer/CommandLineTools"
else
  echo "/Applications/Xcode.app/Contents/Developer"
fi
EOF
chmod +x "$FAKE_BIN/xcrun" "$FAKE_BIN/xcode-select"

fail() { echo "FAIL: $*" >&2; exit 1; }

# run_doctor <mode> [clang]: sets DOCTOR_STATUS, DOCTOR_OUT (stdout+stderr).
run_doctor() {
  local mode="$1" clang="${2:-/usr/bin/true}"
  set +e
  DOCTOR_OUT="$(env NO_COLOR=1 PATH="$FAKE_BIN:$PATH" FAKE_XCRUN_MODE="$mode" \
      FAKE_HOST_MACOS="$HOST_MACOS_MAJOR" CUMETAL_CUDA_CLANG="$clang" \
      "$CUMETAL" doctor 2>&1)"
  DOCTOR_STATUS=$?
  set -e
}

expect() {
  local needle="$1"
  grep -qF -- "$needle" <<<"$DOCTOR_OUT" ||
    fail "doctor output lacks '$needle':"$'\n'"$DOCTOR_OUT"
}

# 1. Toolchain absent, Command Line Tools selected: JIT-only, with the fix.
run_doctor missing
[[ $DOCTOR_STATUS -eq 3 ]] || fail "missing toolchain: exit $DOCTOR_STATUS, want 3"$'\n'"$DOCTOR_OUT"
expect "[✗] Metal toolchain"
expect "JIT-only"
expect "Command Line Tools"
expect "sudo xcodebuild -runFirstLaunch"
expect "xcodebuild -downloadComponent MetalToolchain"

# 2. Xcode's launcher stub: `xcrun --find metal` succeeds but metal cannot run.
run_doctor stub
[[ $DOCTOR_STATUS -eq 3 ]] || fail "stub toolchain: exit $DOCTOR_STATUS, want 3"$'\n'"$DOCTOR_OUT"
expect "launcher stub"
expect "missing Metal Toolchain"
expect "xcodebuild -downloadComponent MetalToolchain"

# 3. Too old for AIR 2.7 / -std=metal3.2.
run_doctor old
[[ $DOCTOR_STATUS -eq 3 ]] || fail "old toolchain: exit $DOCTOR_STATUS, want 3"$'\n'"$DOCTOR_OUT"
expect "toolchain too old for AIR 2.7"
expect "AIR 2.6"

# 4. Targets a newer macOS than this one: its metallibs may not load.
run_doctor newer
[[ $DOCTOR_STATUS -eq 3 ]] || fail "newer toolchain: exit $DOCTOR_STATUS, want 3"$'\n'"$DOCTOR_OUT"
expect "toolchain targets macOS 99"

# 5. Working toolchain: ready.
run_doctor ok
[[ $DOCTOR_STATUS -eq 0 ]] || fail "working toolchain: exit $DOCTOR_STATUS, want 0"$'\n'"$DOCTOR_OUT"
expect "[✓] Metal toolchain"
expect "AIR 2.8"
expect "No issues found!"

# 6. A non-toolchain failure is not JIT-only: no CUDA clang means not ready.
run_doctor missing /nonexistent/clang++
[[ $DOCTOR_STATUS -eq 1 ]] || fail "missing clang: exit $DOCTOR_STATUS, want 1"$'\n'"$DOCTOR_OUT"
expect "CuMetal is not ready"
expect "xcodebuild -downloadComponent MetalToolchain"

# 7. version --json reflects the same probe and parses as JSON.
for mode in missing ok; do
  JSON="$(env PATH="$FAKE_BIN:$PATH" FAKE_XCRUN_MODE="$mode" FAKE_HOST_MACOS="$HOST_MACOS_MAJOR" \
      "$CUMETAL" version --json)"
  python3 - "$mode" "$JSON" <<'EOF'
import json, sys
mode, text = sys.argv[1], sys.argv[2]
data = json.loads(text)
assert data["name"] == "cumetal", data
major, minor, patch = (int(p) for p in data["version"].split("."))
assert data["version_number"] == major * 10000 + minor * 100 + patch, data
assert data["macros"]["__CUMETAL_VERSION__"] == data["version_number"], data
features = data["features"]
assert features["aot_metal_toolchain"] is (mode == "ok"), (mode, features)
assert features["air_version"] == ("2.8" if mode == "ok" else None), (mode, features)
for key in ("source_compiler", "runtime", "jit", "binary_shim"):
    assert isinstance(features[key], bool), (key, features)
EOF
done

echo "PASS: cumetal doctor distinguishes ready, JIT-only and not-ready toolchain states"
