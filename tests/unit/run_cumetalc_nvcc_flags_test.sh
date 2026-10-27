#!/usr/bin/env bash
set -euo pipefail

cumetalc=$1
cu=$2

workdir=$(mktemp -d "${TMPDIR:-/tmp}/cumetalc-nvcc-flags.XXXXXX")
trap 'rm -rf "$workdir"' EXIT

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

# nvcc-style arch selection plus accepted compatibility flags must compile.
"$cumetalc" "$cu" -gencode arch=compute_80,code=sm_80 \
    -Xcompiler -Wall --expt-relaxed-constexpr --expt-extended-lambda \
    -rdc=false --emit=cumetal-ir --no-link --overwrite \
    -o "$workdir/out.ir" || fail "nvcc compatibility flags were rejected"
[[ -s "$workdir/out.ir" ]] || fail "no IR produced under nvcc flags"

# -arch in equals form selects the same target.
"$cumetalc" "$cu" -arch=sm_80 --emit=cumetal-ir --no-link --overwrite \
    -o "$workdir/out2.ir" || fail "-arch=sm_80 was rejected"

# -c (compile-only) is accepted and does not ask for an executable link.
"$cumetalc" "$cu" -c -gencode arch=compute_80,code=sm_80 \
    --emit=cumetal-ir --overwrite -o "$workdir/out3.ir" \
    || fail "-c compile-only was rejected"

# Relocatable device code is refused explicitly, not silently ignored.
if "$cumetalc" "$cu" -rdc=true -o "$workdir/rdc" 2>"$workdir/rdc.log"; then
    fail "-rdc=true should be rejected"
fi
grep -q "relocatable device code" "$workdir/rdc.log" \
    || fail "-rdc=true rejection should explain itself"

# A gencode spec without an sm_ target is rejected.
if "$cumetalc" "$cu" -gencode arch=compute_80 --emit=cumetal-ir \
        -o "$workdir/bad.ir" 2>"$workdir/bad.log"; then
    fail "-gencode without sm_XX should be rejected"
fi

# A launch-bounded kernel must survive native AOT reconciliation: the
# [[max_total_threads_per_threadgroup]] attribute sits between the kernel
# name and its parameter list, and the driver matcher must still find it.
# (On machines without the Metal toolchain this stops later, at the xcrun
# stage; both outcomes are fine, the reconciliation error is not.)
cat > "$workdir/bounded.cu" <<'EOF'
#include <cuda_runtime.h>
__global__ void __launch_bounds__(128) bounded_kernel(const float* in, float* out) {
    out[threadIdx.x] = in[threadIdx.x] * 2.0f;
}
int main() { return 0; }
EOF
"$cumetalc" "$workdir/bounded.cu" -o "$workdir/bounded_exe" \
    >"$workdir/bounded.log" 2>&1 || true
if grep -q "has no typed Metal kernel" "$workdir/bounded.log"; then
    fail "launch-bounded kernel was not reconciled in native AOT mode"
fi

# Unknown flags are still rejected (exit code 2).
if "$cumetalc" "$cu" --definitely-not-an-nvcc-flag \
        -o "$workdir/unknown" 2>"$workdir/unknown.log"; then
    fail "unknown flag should be rejected"
fi

echo "PASS: cumetalc nvcc flag compatibility"
