#!/usr/bin/env bash
set -euo pipefail

cumetalc=$1
ptx=$2
cu=$3
unsupported=$4
switch_source=$5

workdir=$(mktemp -d "${TMPDIR:-/tmp}/cumetalc-shared-ir.XXXXXX")
trap 'rm -rf "$workdir"' EXIT

"$cumetalc" "$ptx" --backend=cumetal-ir --emit=cumetal-ir \
    --overwrite -o "$workdir/vector.cmir"
grep -q 'kernel @vector_add' "$workdir/vector.cmir"
grep -q 'gpu.thread_id' "$workdir/vector.cmir"

"$cumetalc" "$ptx" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/vector.metal"
grep -q 'cumetal-provenance: generic_ptx_lowering' "$workdir/vector.metal"
grep -q 'cumetal-semantic-quality: exact' "$workdir/vector.metal"
grep -q 'kernel void vector_add' "$workdir/vector.metal"

"$cumetalc" "$cu" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/source.metal"
grep -q 'cumetal-provenance: generic_nvvm_lowering' "$workdir/source.metal"
grep -q 'kernel void vector_add' "$workdir/source.metal"

"$cumetalc" "$switch_source" --backend=cumetal-ir --emit=llvm \
    --overwrite -o "$workdir/switch.ll"
if grep -q ' switch ' "$workdir/switch.ll"; then
    echo "LLVM switch survived canonical CUDA normalization" >&2
    exit 1
fi
grep -q 'br i1' "$workdir/switch.ll"

if "$cumetalc" "$unsupported" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/unsupported.metal" \
    >"$workdir/unsupported.stdout" 2>"$workdir/unsupported.stderr"; then
    echo "unsupported PTX unexpectedly compiled" >&2
    exit 1
fi
grep -q 'unsupported opcode' "$workdir/unsupported.stderr"
test ! -e "$workdir/unsupported.metal"
