#!/usr/bin/env bash
# CUB's warp reductions are inline asm: several statements on one line, a
# shuffle with a predicate output, and a guarded add. The typed backend must run
# them correctly; the legacy backend, which cannot honour the guard, must refuse
# rather than compile the kernel without it (it used to, silently).
set -euo pipefail

CUMETALC="$1"
RUNNER="$2"
PTX="$3"

if ! xcrun --find metal >/dev/null 2>&1; then
    echo "SKIP: xcrun metal not available"
    exit 77
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$CUMETALC" "$PTX" --backend=cumetal-ir -o "$WORK/warp.metallib" --overwrite >"$WORK/ir.log" 2>&1 || {
    echo "FAIL: typed backend refused the inline-asm warp reduction"
    cat "$WORK/ir.log"
    exit 1
}
"$RUNNER" "$WORK/warp.metallib"

if "$CUMETALC" "$PTX" --backend=legacy -o "$WORK/legacy.metallib" --overwrite >"$WORK/legacy.log" 2>&1; then
    echo "FAIL: legacy backend compiled a guarded add it cannot honour"
    exit 1
fi
grep -q "predicated non-branch" "$WORK/legacy.log" || {
    echo "FAIL: legacy refusal did not name the guarded instruction"
    cat "$WORK/legacy.log"
    exit 1
}
echo "PASS: legacy backend refuses the guarded add"
