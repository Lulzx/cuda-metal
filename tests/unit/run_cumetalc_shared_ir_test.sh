#!/usr/bin/env bash
set -euo pipefail

cumetalc=$1
ptx=$2
cu=$3
unsupported=$4
switch_source=$5
float_abs_source=$6
float_math_source=$7
byval_aggregate_memcpy_source=$8
float_atomic_add_source=$9
nvcc=${10}

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

"$cumetalc" "$float_abs_source" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/float_abs.metal"
grep -q 'kernel void cuda_float_abs' "$workdir/float_abs.metal"
test "$(grep -o 'fabs(' "$workdir/float_abs.metal" | wc -l | tr -d ' ')" -ge 2

"$cumetalc" "$byval_aggregate_memcpy_source" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/byval_aggregate_memcpy.metal"
grep -q 'kernel void byval_aggregate_memcpy' "$workdir/byval_aggregate_memcpy.metal"
grep -q 'reinterpret_cast<device uint\*>' "$workdir/byval_aggregate_memcpy.metal"

"$cumetalc" "$float_atomic_add_source" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/float_atomic_add.metal"
grep -q 'kernel void cuda_float_atomic_add' "$workdir/float_atomic_add.metal"
test "$(grep -o 'reinterpret_cast<device atomic_float\*>' "$workdir/float_atomic_add.metal" | wc -l | tr -d ' ')" -ge 2
if grep -qE 'cm_atomic_cas|atomic_compare_exchange' "$workdir/float_atomic_add.metal"; then
    echo "device float atomicAdd regressed to a CAS loop on the source-first path" >&2
    exit 1
fi

# NVIDIA's C++ overlay selects binary32 for unsuffixed rsqrt/fma calls. A
# missing overload silently inserts f32->f64->f32 conversions and the double
# libdevice calls, which is especially expensive on Apple GPUs.
"$nvcc" -S --cuda-device-only -std=c++17 "$float_math_source" \
    -o "$workdir/float_math.ptx"
grep -q '__nv_rsqrtf' "$workdir/float_math.ptx"
grep -q '__nv_fmaf' "$workdir/float_math.ptx"
grep -q 'atom.global.add.f32' "$workdir/float_math.ptx"
if grep -q 'atom.*cas.b32' "$workdir/float_math.ptx"; then
    echo "float atomicAdd regressed to a CAS loop" >&2
    exit 1
fi
if grep -qE 'cvt\.f64\.f32|__nv_rsqrt([^fA-Za-z0-9_]|$)|__nv_fma([^fA-Za-z0-9_]|$)' \
    "$workdir/float_math.ptx"; then
    echo "FP32 CUDA math overloads promoted to FP64" >&2
    exit 1
fi

if "$cumetalc" "$unsupported" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/unsupported.metal" \
    >"$workdir/unsupported.stdout" 2>"$workdir/unsupported.stderr"; then
    echo "unsupported PTX unexpectedly compiled" >&2
    exit 1
fi
grep -q 'unsupported opcode' "$workdir/unsupported.stderr"
test ! -e "$workdir/unsupported.metal"
