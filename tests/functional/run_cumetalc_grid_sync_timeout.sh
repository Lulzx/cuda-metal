#!/usr/bin/env bash
# A cooperative grid sync that a threadgroup never reaches must end in
# cudaErrorLaunchTimeout from the grid-sync watchdog, within the spin budget,
# instead of hanging the host (#175).
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <cumetalc> <source.cu> <workdir> [cumetalc args...]" >&2
    exit 2
fi
if ! command -v xcrun >/dev/null 2>&1 ||
   ! xcrun --find metal >/dev/null 2>&1 ||
   ! xcrun --find metallib >/dev/null 2>&1; then
    echo "SKIP: complete Metal toolchain unavailable"
    exit 77
fi

cumetalc="$1"
source_cu="$2"
work_dir="$3"
shift 3
mkdir -p "$work_dir"
exe="$work_dir/grid_sync_timeout"
rm -f "$exe"
"$cumetalc" "$source_cu" "$@" -o "$exe"

cache="$work_dir/runtime-cache"
rm -rf "$cache"
status=0
output="$(CUMETAL_CACHE_DIR="$cache" CUMETAL_TRACE_GPU=1 "$exe" 2>&1)" || status=$?
echo "$output"
if [[ $status -ne 0 ]]; then
    echo "FAIL: grid-sync timeout probe exited $status"
    exit 1
fi
if ! grep -q '^PASS: unbalanced cooperative grid sync reported cudaErrorLaunchTimeout' <<<"$output"; then
    echo "FAIL: expected the watchdog to report cudaErrorLaunchTimeout"
    exit 1
fi
if ! grep -q 'cooperative grid sync timed out' <<<"$output"; then
    echo "FAIL: missing the grid-sync timeout diagnostic"
    exit 1
fi
if ! grep -q 'event=kernel_launch .*device=apple_gpu .*launch_success=false' <<<"$output"; then
    echo "FAIL: provenance does not show the launch failing on the Apple GPU"
    exit 1
fi
