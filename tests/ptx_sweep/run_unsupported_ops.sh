#!/usr/bin/env bash
set -euo pipefail

PTX2LLVM="$1"
WORKDIR="$2"
OUT_DIR="$WORKDIR/ptx_sweep_unsupported"
mkdir -p "$OUT_DIR"

run_case_expect_fail() {
  local case_name="$1"
  local instruction="$2"
  local ptx_file="$OUT_DIR/${case_name}.ptx"
  local ll_file="$OUT_DIR/${case_name}.ll"

  cat >"$ptx_file" <<EOF
.version 8.0
.target sm_90
.address_size 64
.visible .entry ${case_name}(
  .param .u64 p0
)
{
  ${instruction}
  ret;
}
EOF

  if "$PTX2LLVM" --input "$ptx_file" --output "$ll_file" --entry "$case_name" --strict --overwrite; then
    echo "FAIL: strict PTX sweep expected unsupported opcode to fail (${case_name})"
    exit 1
  fi
}

run_case_expect_fail "unsupported_foo" "foo.bar %r1, %r2;"

# Hopper cluster ops (spec §5.1.1: per-instruction compile-time error)
run_case_expect_fail "unsupported_cluster_sync"   "cluster.sync.aligned;"
run_case_expect_fail "unsupported_mbarrier_init"  "mbarrier.init.b64 [%rd1], 32;"
run_case_expect_fail "unsupported_mbarrier_arrive" "mbarrier.arrive.b64 %rd1, [%rd2];"

# TMA (Tensor Memory Accelerator) ops (spec §5.1.1: per-instruction compile-time error)
run_case_expect_fail "unsupported_tma_bulk_1d" \
    "cp.async.bulk.tensor.1d.global.shared::cluster.bulk_group [%rd1], [%rd2, {%r1}];"
run_case_expect_fail "unsupported_redux" \
    "redux.sync.add.s32 %r1, %r2, 0xffffffff;"

# FP8 / Transformer Engine (spec §5.1.1: per-instruction compile-time error)
run_case_expect_fail "unsupported_fp8_cvt" "cvt.rn.f8x2.e4m3x2.rn.satfinite.f32 %r1, %f1, %f2;"

# Tensor core ops (wmma/mma/ldmatrix: no Metal equivalent; use MPSMatrixMultiplication)
run_case_expect_fail "unsupported_wmma_load"   "wmma.load.a.sync.aligned.row.m16n16k16.f16 {%r1,%r2,%r3,%r4}, [%rd1], 16;"
run_case_expect_fail "unsupported_mma_sync"    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4}, {%r1,%r2}, {%r3}, {%f1,%f2,%f3,%f4};"
run_case_expect_fail "unsupported_ldmatrix"    "ldmatrix.sync.aligned.m8n8.x1.b16 {%r1}, [%rd1];"

# Texture/surface instructions (spec §2.2: deferred to v2; compile-time error)
run_case_expect_fail "unsupported_tex"   "tex.2d.v4.f32.f32 {%f1,%f2,%f3,%f4}, [%rd1], {%f5, %f6};"
run_case_expect_fail "unsupported_tld4"  "tld4.r.2d.v4.s32.f32 {%r1,%r2,%r3,%r4}, [%rd1], {%f1,%f2};"
run_case_expect_fail "unsupported_txq_attribute" "txq.filter_mode.b32 %r1, [%rd1];"
run_case_expect_fail "unsupported_txq_level" "txq.level.width.b32 %r1, [%rd1], %r2;"
run_case_expect_fail "unsupported_suld"  "suld.b.2d.b32.zero %r1, [%rd1, {%r2, %r3}];"
run_case_expect_fail "unsupported_sust"  "sust.b.2d.b32.zero [%rd1, {%r1, %r2}], %r3;"
run_case_expect_fail "unsupported_sured" "sured.b.2d.add.u32 [%rd1, {%r1, %r2}], %r3;"
run_case_expect_fail "unsupported_suq_attribute" "suq.memory_layout.b32 %r1, [%rd1];"

echo "PASS: PTX sweep unsupported-op strict rejection completed"
