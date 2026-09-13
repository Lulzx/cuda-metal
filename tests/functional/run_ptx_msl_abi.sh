#!/usr/bin/env bash
set -euo pipefail
compiler="$1"
fixture="$2"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

for backend in cumetal-ir legacy; do
    output="$work/$backend.metal"
    "$compiler" "$fixture" --backend="$backend" --ptx-strict \
        --entry vector_add --emit=msl -o "$output"
    test -s "$output"
    cat > "$work/expected" <<'EOF'
CUMETAL_ABI_V2
kernel vector_add
shared 0
arg buffer 8
arg buffer 8
arg buffer 8
EOF
    diff -u "$work/expected" "$output.cumetal-abi"
    # A sidecar write failure must fail compilation, not report usable output.
    rm "$output" "$output.cumetal-abi"
    mkdir "$output.cumetal-abi"
    if "$compiler" "$fixture" --backend="$backend" --ptx-strict \
        --entry vector_add --emit=msl -o "$output" > "$work/error" 2>&1; then
        echo 'FAIL: unwritable ABI sidecar was accepted'
        exit 1
    fi
    test -d "$output.cumetal-abi"
done
echo 'PASS: PTX-to-MSL ABI sidecars and write failures'
