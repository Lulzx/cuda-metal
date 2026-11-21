#!/usr/bin/env bash
# cumetalc on a translation unit whose only kernel is device-pass-only must
# succeed (the program is valid) and warn that no metallib is built for it.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <cumetalc> <source.cu> <work-dir>" >&2
    exit 2
fi
cumetalc="$1"
source_file="$2"
work_dir="$3"

if ! xcrun --find metal >/dev/null 2>&1; then
    echo "SKIP: Metal toolchain not available"
    exit 77
fi

rm -rf "$work_dir"
mkdir -p "$work_dir"
log="$work_dir/compile.log"
if ! "$cumetalc" "$source_file" -o "$work_dir/program" >"$log" 2>&1; then
    cat "$log" >&2
    echo "FAIL: cumetalc refused a valid translation unit" >&2
    exit 1
fi
if ! grep -q "defines device kernels with no host launch stub" "$log"; then
    cat "$log" >&2
    echo "FAIL: cumetalc did not warn about the device-only kernel" >&2
    exit 1
fi
output="$("$work_dir/program")"
if [[ "$output" != *"PASS: host program without launchable kernels"* ]]; then
    echo "FAIL: program output: $output" >&2
    exit 1
fi
echo "PASS: device-only kernel compiles with a warning"
