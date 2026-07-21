#!/usr/bin/env bash
set -euo pipefail

PTX2LLVM="$1"
WORKDIR="$2"
OUT_DIR="$WORKDIR/ptx_sweep_supported"
mkdir -p "$OUT_DIR"

run_case() {
  local case_name="$1"
  local body="$2"
  local ptx_file="$OUT_DIR/${case_name}.ptx"
  local ll_file="$OUT_DIR/${case_name}.ll"

  cat >"$ptx_file" <<EOF
.version 8.0
.target sm_90
.address_size 64
.visible .entry ${case_name}(
  .param .u64 p0,
  .param .u64 p1
)
{
  ${body}
  ret;
}
EOF

  "$PTX2LLVM" \
    --input "$ptx_file" \
    --output "$ll_file" \
    --entry "$case_name" \
    --strict \
    --overwrite

  rg "define void @${case_name}" "$ll_file" >/dev/null
}

run_case "sweep_add_s32" "add.s32 %r1, %r2, %r3;"
run_case "sweep_sub_s32" "sub.s32 %r1, %r2, %r3;"
run_case "sweep_mul_lo_s32" "mul.lo.s32 %r1, %r2, %r3;"
run_case "sweep_div_s32" "div.s32 %r1, %r2, %r3;"
run_case "sweep_rem_s32" "rem.s32 %r1, %r2, %r3;"
run_case "sweep_mad_lo_s32" "mad.lo.s32 %r1, %r2, %r3, %r4;"
run_case "sweep_neg_f32" "neg.f32 %f1, %f2;"
run_case "sweep_shl_b64" "shl.b64 %rd1, %rd2, 2;"
run_case "sweep_not_b32" "not.b32 %r1, %r2;"
run_case "sweep_and_b32" "and.b32 %r1, %r2, %r3;"
run_case "sweep_or_b32" "or.b32 %r1, %r2, %r3;"
run_case "sweep_xor_b32" "xor.b32 %r1, %r2, %r3;"
run_case "sweep_selp_f32" "selp.f32 %f1, %f2, %f3, %p1;"
run_case "sweep_rcp_f32" "rcp.rn.f32 %f1, %f2;"
run_case "sweep_mov_tid_x" "mov.u32 %r1, %tid.x;"
run_case "sweep_mov_ctaid_y" "mov.u32 %r1, %ctaid.y;"
run_case "sweep_mov_ntid_z" "mov.u32 %r1, %ntid.z;"
run_case "sweep_mov_nctaid_x" "mov.u32 %r1, %nctaid.x;"
run_case "sweep_mov_laneid"        "mov.u32 %r1, %laneid;"
run_case "sweep_mov_warpsize"      "mov.u32 %r1, %warpsize;"
run_case "sweep_mov_lanemask_eq"   "mov.u32 %r1, %lanemask_eq;"
run_case "sweep_mov_lanemask_lt"   "mov.u32 %r1, %lanemask_lt;"
run_case "sweep_mov_lanemask_gt"   "mov.u32 %r1, %lanemask_gt;"
run_case "sweep_ld_shared_u32" "ld.shared.u32 %r1, [%rd1];"
run_case "sweep_st_shared_u32" "st.shared.u32 [%rd1], %r1;"
run_case "sweep_ld_global_f32" "ld.global.f32 %f1, [%rd1];"
run_case "sweep_st_global_f32" "st.global.f32 [%rd1], %f1;"
run_case "sweep_ld_local_u16" "ld.local.u16 %r1, [%rd1];"
run_case "sweep_st_local_u16" "st.local.u16 [%rd1], %r1;"
run_case "sweep_cvta_shared" "cvta.to.shared.u64 %rd2, %rd1;"
run_case "sweep_cvta_global" "cvta.to.global.u64 %rd2, %rd1;"
run_case "sweep_cvta_local" "cvta.to.local.u64 %rd2, %rd1;"
run_case "sweep_bar_sync" "bar.sync 0;"
run_case "sweep_setp_eq_s32" "setp.eq.s32 %p1, %r1, %r2;"
run_case "sweep_bra_label" $'bra L_done;\nL_done:'
run_case "sweep_call_vprintf" "call.uni (%r0), vprintf, (\"tid=%u\", %r1);"
run_case "sweep_atom_global_add_f32" "atom.global.add.f32 %f1, [%rd1], %f2;"

# Warp / SIMD-group primitives
run_case "sweep_shfl_sync_idx"    "shfl.sync.idx.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;"
run_case "sweep_shfl_sync_down"   "shfl.sync.down.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;"
run_case "sweep_shfl_sync_up"     "shfl.sync.up.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;"
run_case "sweep_shfl_sync_bfly"   "shfl.sync.bfly.b32 %r1, %r2, %r3, 0x1f, 0xffffffff;"
# f32 variants (common in CUDA reduction kernels — same lowering path as b32)
run_case "sweep_shfl_sync_idx_f32"  "shfl.sync.idx.f32 %f1, %f2, %r1, 0x1f, 0xffffffff;"
run_case "sweep_shfl_sync_down_f32" "shfl.sync.down.f32 %f1, %f2, %r1, 0x1f, 0xffffffff;"
run_case "sweep_vote_sync_ballot" "vote.sync.ballot.b32 %r1, %p1, 0xffffffff;"
run_case "sweep_vote_sync_any"   "vote.sync.any.pred %p1, %p2, 0xffffffff;"
run_case "sweep_vote_sync_all"   "vote.sync.all.pred %p1, %p2, 0xffffffff;"
run_case "sweep_bar_warp_sync"   "bar.warp.sync 0xffffffff;"

# Memory barriers
run_case "sweep_membar_gl"   "membar.gl;"
run_case "sweep_membar_cta"  "membar.cta;"
run_case "sweep_membar_sys"  "membar.sys;"

# Async copy
run_case "sweep_cp_async_ca"          "cp.async.ca.shared.global [%rd1], [%rd2], 16;"
run_case "sweep_cp_async_commit"      "cp.async.commit_group;"
run_case "sweep_cp_async_wait_all"    "cp.async.wait_all;"
run_case "sweep_cp_async_wait_group0" "cp.async.wait_group 0;"

# Warp reductions
run_case "sweep_redux_sync_add_s32" "redux.sync.add.s32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_add_f32" "redux.sync.add.f32 %f1, %f2, 0xffffffff;"
run_case "sweep_redux_sync_and_b32" "redux.sync.and.b32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_or_b32"  "redux.sync.or.b32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_xor_b32" "redux.sync.xor.b32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_min_s32" "redux.sync.min.s32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_max_s32" "redux.sync.max.s32 %r1, %r2, 0xffffffff;"
run_case "sweep_redux_sync_min_f32" "redux.sync.min.f32 %f1, %f2, 0xffffffff;"
run_case "sweep_redux_sync_max_f32" "redux.sync.max.f32 %f1, %f2, 0xffffffff;"

# Math intrinsics (extended)
run_case "sweep_sqrt_rn_f32"    "sqrt.rn.f32 %f1, %f2;"
run_case "sweep_rsqrt_approx"   "rsqrt.approx.f32 %f1, %f2;"
run_case "sweep_ex2_approx_f32" "ex2.approx.f32 %f1, %f2;"
run_case "sweep_lg2_approx_f32" "lg2.approx.f32 %f1, %f2;"
run_case "sweep_sin_approx_f32" "sin.approx.f32 %f1, %f2;"
run_case "sweep_cos_approx_f32" "cos.approx.f32 %f1, %f2;"
run_case "sweep_fma_rn_f32"     "fma.rn.f32 %f1, %f2, %f3, %f4;"
run_case "sweep_abs_f32"        "abs.f32 %f1, %f2;"
run_case "sweep_min_f32"        "min.f32 %f1, %f2, %f3;"
run_case "sweep_max_f32"        "max.f32 %f1, %f2, %f3;"

# Integer 64-bit arithmetic
run_case "sweep_add_s64"        "add.s64 %rd1, %rd2, %rd3;"
run_case "sweep_sub_s64"        "sub.s64 %rd1, %rd2, %rd3;"
run_case "sweep_mul_lo_s64"     "mul.lo.s64 %rd1, %rd2, %rd3;"
run_case "sweep_shl_b32"        "shl.b32 %r1, %r2, %r3;"
run_case "sweep_shr_u32"        "shr.u32 %r1, %r2, %r3;"
run_case "sweep_shr_s32"        "shr.s32 %r1, %r2, %r3;"

# Integer abs / bitfield
run_case "sweep_abs_s32"        "abs.s32 %r1, %r2;"
run_case "sweep_min_s32"        "min.s32 %r1, %r2, %r3;"
run_case "sweep_max_s32"        "max.s32 %r1, %r2, %r3;"
run_case "sweep_min_u32"        "min.u32 %r1, %r2, %r3;"
run_case "sweep_max_u32"        "max.u32 %r1, %r2, %r3;"
run_case "sweep_clz_b32"        "clz.b32 %r1, %r2;"
run_case "sweep_popc_b32"       "popc.b32 %r1, %r2;"

# Multiply-wide (produces 64-bit result from two 32-bit inputs)
run_case "sweep_mul_wide_u32"   "mul.wide.u32 %rd1, %r2, %r3;"
run_case "sweep_mul_wide_s32"   "mul.wide.s32 %rd1, %r2, %r3;"

# Type conversions
run_case "sweep_cvt_f32_s32"    "cvt.rn.f32.s32 %f1, %r1;"
run_case "sweep_cvt_f32_u32"    "cvt.rn.f32.u32 %f1, %r1;"
run_case "sweep_cvt_s32_f32"    "cvt.rzi.s32.f32 %r1, %f1;"
run_case "sweep_cvt_u32_f32"    "cvt.rzi.u32.f32 %r1, %f1;"
run_case "sweep_cvt_f64_f32"    "cvt.f64.f32 %fd1, %f1;"
run_case "sweep_cvt_f32_f64"    "cvt.rn.f32.f64 %f1, %fd1;"
run_case "sweep_cvt_u64_u32"    "cvt.u64.u32 %rd1, %r1;"
run_case "sweep_cvt_s64_s32"    "cvt.s64.s32 %rd1, %r1;"

# Predicate conversions
run_case "sweep_selp_s32"       "selp.s32 %r1, %r2, %r3, %p1;"
run_case "sweep_selp_s64"       "selp.s64 %rd1, %rd2, %rd3, %p1;"

# Setp variants (beyond eq.s32)
run_case "sweep_setp_lt_f32"    "setp.lt.f32 %p1, %f1, %f2;"
run_case "sweep_setp_gt_s32"    "setp.gt.s32 %p1, %r1, %r2;"
run_case "sweep_setp_ne_u32"    "setp.ne.u32 %p1, %r1, %r2;"
run_case "sweep_setp_ge_f32"    "setp.ge.f32 %p1, %f1, %f2;"
run_case "sweep_setp_le_u32"    "setp.le.u32 %p1, %r1, %r2;"

# FP64 compile-time (native mode: parsed + LLVM IR emitted; may not execute on Apple Silicon)
run_case "sweep_fma_rn_f64"     "fma.rn.f64 %fd1, %fd2, %fd3, %fd4;"
run_case "sweep_add_f64"        "add.rn.f64 %fd1, %fd2, %fd3;"
run_case "sweep_mul_f64"        "mul.rn.f64 %fd1, %fd2, %fd3;"

# Load types
run_case "sweep_ld_global_u32"  "ld.global.u32 %r1, [%rd1];"
run_case "sweep_ld_global_s32"  "ld.global.s32 %r1, [%rd1];"
run_case "sweep_ld_global_u64"  "ld.global.u64 %rd1, [%rd2];"
run_case "sweep_ld_global_f64"  "ld.global.f64 %fd1, [%rd1];"

# Atomic add (global, integer)
run_case "sweep_atom_global_add_s32" "atom.global.add.s32 %r1, [%rd1], %r2;"
run_case "sweep_atom_global_add_u32" "atom.global.add.u32 %r1, [%rd1], %r2;"

# Integer negation (neg maps to llvm.neg for integers, llvm.fneg for floats)
run_case "sweep_neg_s32"        "neg.s32 %r1, %r2;"
run_case "sweep_neg_s64"        "neg.s64 %rd1, %rd2;"

# Multiply-high: high 32 bits of 64-bit product
run_case "sweep_mul_hi_u32"     "mul.hi.u32 %r1, %r2, %r3;"
run_case "sweep_mul_hi_s32"     "mul.hi.s32 %r1, %r2, %r3;"

# FP64 atomic add (spec §5.1.1 Ampere atomics)
run_case "sweep_atom_global_add_f64" "atom.global.add.f64 %fd1, [%rd1], %fd2;"

# set: predicate-to-integer mask (set.CmpOp.dtype.stype d, a, b)
run_case "sweep_set_eq_u32"     "set.eq.u32.u32 %r1, %r2, %r3;"
run_case "sweep_set_lt_u32"     "set.lt.u32.s32 %r1, %r2, %r3;"
run_case "sweep_set_ne_f32"     "set.ne.u32.f32 %r1, %f1, %f2;"

# abs (integer and float)
run_case "sweep_abs_s32"        "abs.s32 %r1, %r2;"
run_case "sweep_abs_s64"        "abs.s64 %rd1, %rd2;"
run_case "sweep_abs_f32"        "abs.f32 %f1, %f2;"
run_case "sweep_abs_f64"        "abs.f64 %fd1, %fd2;"

# shr (logical and arithmetic shifts)
run_case "sweep_shr_b32"        "shr.b32 %r1, %r2, 3;"
run_case "sweep_shr_u32"        "shr.u32 %r1, %r2, 5;"
run_case "sweep_shr_s32"        "shr.s32 %r1, %r2, 2;"
run_case "sweep_shr_b64"        "shr.b64 %rd1, %rd2, 4;"
run_case "sweep_shr_u64"        "shr.u64 %rd1, %rd2, 7;"

# non-sync vote forms
run_case "sweep_vote_ballot_b32" "vote.ballot.b32 %r1, %p1;"
run_case "sweep_vote_any"        "vote.any.pred %p1, %p2;"
run_case "sweep_vote_all"        "vote.all.pred %p1, %p2;"

# additional st.global variants
run_case "sweep_st_global_u32"  "st.global.u32 [%rd1], %r1;"
run_case "sweep_st_global_u64"  "st.global.u64 [%rd1], %rd2;"
run_case "sweep_st_global_f64"  "st.global.f64 [%rd1], %fd1;"

# additional ld.global variants (byte/short widths)
run_case "sweep_ld_global_u8"   "ld.global.u8 %r1, [%rd1];"
run_case "sweep_ld_global_s8"   "ld.global.s8 %r1, [%rd1];"
run_case "sweep_ld_global_u16"  "ld.global.u16 %r1, [%rd1];"
run_case "sweep_ld_global_s16"  "ld.global.s16 %r1, [%rd1];"

# atom cas and bitwise
run_case "sweep_atom_global_cas_b32" "atom.global.cas.b32 %r1, [%rd1], %r2, %r3;"
run_case "sweep_atom_global_and_b32" "atom.global.and.b32 %r1, [%rd1], %r2;"
run_case "sweep_atom_global_or_b32"  "atom.global.or.b32 %r1, [%rd1], %r2;"
run_case "sweep_atom_global_xor_b32" "atom.global.xor.b32 %r1, [%rd1], %r2;"
run_case "sweep_atom_global_min_s32" "atom.global.min.s32 %r1, [%rd1], %r2;"
run_case "sweep_atom_global_max_s32" "atom.global.max.s32 %r1, [%rd1], %r2;"
run_case "sweep_atom_global_exch_b32" "atom.global.exch.b32 %r1, [%rd1], %r2;"

# brev (bit reverse)
run_case "sweep_brev_b32"       "brev.b32 %r1, %r2;"
run_case "sweep_brev_b64"       "brev.b64 %rd1, %rd2;"

# ld/st shared float (complement existing u32 variants)
run_case "sweep_ld_shared_f32"  "ld.shared.f32 %f1, [%rd1];"
run_case "sweep_st_shared_f32"  "st.shared.f32 [%rd1], %f1;"
run_case "sweep_ld_shared_s32"  "ld.shared.s32 %r1, [%rd1];"
run_case "sweep_ld_shared_u64"  "ld.shared.u64 %rd1, [%rd2];"

# mad.lo.u32 (unsigned GID computation pattern)
run_case "sweep_mad_lo_u32"     "mad.lo.u32 %r1, %r2, %r3, %r4;"

# clz / popc 64-bit variants (lowering exists; sweep coverage was missing)
run_case "sweep_clz_b64"        "clz.b64 %r1, %rd1;"
run_case "sweep_popc_b64"       "popc.b64 %r1, %rd1;"

# basic float arithmetic (f32)
run_case "sweep_add_f32"        "add.f32 %f1, %f2, %f3;"
run_case "sweep_sub_f32"        "sub.f32 %f1, %f2, %f3;"
run_case "sweep_mul_f32"        "mul.f32 %f1, %f2, %f3;"
run_case "sweep_div_f32"        "div.rn.f32 %f1, %f2, %f3;"

# float f64 unary/binary ops
run_case "sweep_neg_f64"        "neg.f64 %fd1, %fd2;"
run_case "sweep_abs_f64"        "abs.f64 %fd1, %fd2;"
run_case "sweep_min_f64"        "min.f64 %fd1, %fd2, %fd3;"
run_case "sweep_max_f64"        "max.f64 %fd1, %fd2, %fd3;"

# 64-bit bitwise ops (lowering maps root, so b64 should also work)
run_case "sweep_and_b64"        "and.b64 %rd1, %rd2, %rd3;"
run_case "sweep_or_b64"         "or.b64 %rd1, %rd2, %rd3;"
run_case "sweep_xor_b64"        "xor.b64 %rd1, %rd2, %rd3;"
run_case "sweep_not_b64"        "not.b64 %rd1, %rd2;"

# 64-bit multiply low
run_case "sweep_mul_lo_u64"     "mul.lo.u64 %rd1, %rd2, %rd3;"

# unsigned / 64-bit remainder
run_case "sweep_rem_u32"        "rem.u32 %r1, %r2, %r3;"
run_case "sweep_rem_s64"        "rem.s64 %rd1, %rd2, %rd3;"

# partial-mask warp primitives (mask != 0xFFFFFFFF; explicit vote/shuffle membership)
run_case "sweep_shfl_partial_mask"  "shfl.sync.idx.b32 %r1, %r2, %r3, 0x1f, 0x0000ffff;"
run_case "sweep_vote_partial_mask"  "vote.sync.ballot.b32 %r1, %p1, 0x0000ffff;"

# div additional types
run_case "sweep_div_u32"            "div.u32 %r1, %r2, %r3;"
run_case "sweep_div_s64"            "div.s64 %rd1, %rd2, %rd3;"
run_case "sweep_div_u64"            "div.u64 %rd1, %rd2, %rd3;"
run_case "sweep_div_f64"            "div.rn.f64 %fd1, %fd2, %fd3;"

# selp for unsigned and f64 types
run_case "sweep_selp_u32"           "selp.u32 %r1, %r2, %r3, %p1;"
run_case "sweep_selp_u64"           "selp.u64 %rd1, %rd2, %rd3, %p1;"
run_case "sweep_selp_f64"           "selp.f64 %fd1, %fd2, %fd3, %p1;"

# setp f64 comparisons
run_case "sweep_setp_lt_f64"        "setp.lt.f64 %p1, %fd1, %fd2;"
run_case "sweep_setp_eq_f64"        "setp.eq.f64 %p1, %fd1, %fd2;"

# additional cvt variants
run_case "sweep_cvt_f32_f16"        "cvt.f32.f16 %f1, %r1;"
run_case "sweep_cvt_rn_s32_f64"     "cvt.rzi.s32.f64 %r1, %fd1;"

# ld.global.nc: non-coherent (texture-cache) global load — __ldg() equivalent (spec §8)
# Lowered to plain global load (no texture cache on Apple Silicon UMA).
run_case "sweep_ld_global_nc_f32"   "ld.global.nc.f32 %f1, [%rd1];"
run_case "sweep_ld_global_nc_u32"   "ld.global.nc.u32 %r1, [%rd1];"

# 64-bit global atomics
run_case "sweep_atom_global_add_u64"  "atom.global.add.u64 %rd1, [%rd2], %rd3;"
run_case "sweep_atom_global_exch_b64" "atom.global.exch.b64 %rd1, [%rd2], %rd3;"
run_case "sweep_atom_global_cas_b64"  "atom.global.cas.b64 %rd1, [%rd2], %rd3, %rd4;"

# unsigned min/max atomics
run_case "sweep_atom_global_min_u32"  "atom.global.min.u32 %r1, [%rd1], %r2;"
run_case "sweep_atom_global_max_u32"  "atom.global.max.u32 %r1, [%rd1], %r2;"

# shared memory atomics
run_case "sweep_atom_shared_add_u32"  "atom.shared.add.u32 %r1, [%rd1], %r2;"
run_case "sweep_atom_shared_exch_b32" "atom.shared.exch.b32 %r1, [%rd1], %r2;"
run_case "sweep_atom_shared_cas_b32"  "atom.shared.cas.b32 %r1, [%rd1], %r2, %r3;"

# ld.const: constant address-space load (spec §5.4.1, __constant__ memory)
run_case "sweep_ld_const_f32"       "ld.const.f32 %f1, [%rd1];"
run_case "sweep_ld_const_u32"       "ld.const.u32 %r1, [%rd1];"

# lop3: 3-input LUT logic op (Turing/Ampere+)
run_case "sweep_lop3_b32"           "lop3.b32 %r1, %r2, %r3, %r4, 0xf0, 1;"

# sad: sum of absolute differences
run_case "sweep_sad_u32"            "sad.u32 %r1, %r2, %r3, %r4;"
run_case "sweep_sad_s32"            "sad.s32 %r1, %r2, %r3, %r4;"

# match.any.sync / match.all.sync (Ampere+, ISA 7.0+)
# Conservative lowering: air.match.any.sync / air.match.all.sync
run_case "sweep_match_any_sync_b32" "match.any.sync.b32 %r1, %r2, 0xffffffff;"
run_case "sweep_match_all_sync_b32" "match.all.sync.b32 %r1, %p1, %r2, 0xffffffff;"

# nanosleep: busy-wait (Pascal+, ISA 6.0+) → conservative no-op on Metal
run_case "sweep_nanosleep_u32"      "nanosleep.u32 64;"

# trap: raise hardware trap → llvm.trap()
run_case "sweep_trap"               "trap;"

# exit: terminate thread (same as ret in our model)
run_case "sweep_exit"               "exit;"

# bar.arrive: arrive at barrier without waiting (cooperative groups)
# Conservative: full threadgroup barrier
run_case "sweep_bar_arrive"         "bar.arrive 0, %r1;"

# bar.red: barrier-level predicate reductions (sync + reduce at named barrier)
# Conservative: map to simdgroup-level operations (spec §9.7, barrier instructions)
run_case "sweep_bar_red_and"        "bar.red.and.pred %p1, 0, %r1, %p2;"
run_case "sweep_bar_red_or"         "bar.red.or.pred  %p1, 0, %r1, %p2;"
run_case "sweep_bar_red_popc"       "bar.red.popc.u32 %r1, 0, %r2, %p1;"

# prefetch / prefetchu: cache prefetch hints → no-op on UMA (spec §8, __ldg() note)
run_case "sweep_prefetch_global_L1" "prefetch.global.L1 [%rd1];"
run_case "sweep_prefetch_local_L1"  "prefetch.local.L1 [%rd1];"
run_case "sweep_prefetchu_L1"       "prefetchu.L1 [%rd1];"

# vote.uni: predicate uniformity test → air.simdgroup.all (conservative)
run_case "sweep_vote_uni_pred"      "vote.uni.pred %p1, %p2;"

# red: write-only global/shared atomic reduction (no return value)
run_case "sweep_red_global_add_f32" "red.global.add.f32 [%rd1], %f1;"
run_case "sweep_red_global_add_u32" "red.global.add.u32 [%rd1], %r1;"
run_case "sweep_red_shared_add_u32" "red.shared.add.u32 [%rd1], %r1;"

# fence: Ampere+ fine-grained memory fence (ISA 7.0+) → air.mem.barrier
run_case "sweep_fence_sc_cta"       "fence.sc.cta;"
run_case "sweep_fence_sc_gpu"       "fence.sc.gpu;"
run_case "sweep_fence_sc_sys"       "fence.sc.sys;"

# activemask: bitmask of active threads in current warp (ISA 6.0+)
# Lowered to constant 0xFFFFFFFF (all lanes active; no SIMD divergence in our model).
run_case "sweep_activemask_b32"     "activemask.b32 %r1;"

# fns: find Nth set bit in a warp mask (Volta+)
run_case "sweep_fns_b32"            "fns.b32 %r1, %r2, %r3, %r4;"

# bfind: find most significant non-sign bit (ISA 5.0+)
run_case "sweep_bfind_u32"          "bfind.u32 %r1, %r2;"
run_case "sweep_bfind_s32"          "bfind.s32 %r1, %r2;"
run_case "sweep_bfind_shiftamt"     "bfind.shiftamt.u32 %r1, %r2;"

# mul.wide: widening multiply (produces 64-bit from 32-bit inputs)
run_case "sweep_mul_wide_u32"       "mul.wide.u32 %rd1, %r1, %r2;"
run_case "sweep_mul_wide_s32"       "mul.wide.s32 %rd1, %r1, %r2;"

# mul.hi: high word of 32×32-bit multiply
run_case "sweep_mul_hi_u32"         "mul.hi.u32 %r1, %r2, %r3;"
run_case "sweep_mul_hi_s32"         "mul.hi.s32 %r1, %r2, %r3;"

# f32 atomics (Ampere+: atom.global.{max,min}.f32)
run_case "sweep_atom_global_max_f32" "atom.global.max.f32 %f1, [%rd1], %f2;"
run_case "sweep_atom_global_min_f32" "atom.global.min.f32 %f1, [%rd1], %f2;"

# isspacep: pointer address-space predicate (spec §5.4)
# Conservative lowering: .global → true; .shared/.local → false on UMA model.
run_case "sweep_isspacep_global"  "isspacep.global %p1, %rd1;"
run_case "sweep_isspacep_shared"  "isspacep.shared %p1, %rd1;"
run_case "sweep_isspacep_local"   "isspacep.local  %p1, %rd1;"

# prmt: byte permutation from two 32-bit words (ISA 6.0+)
run_case "sweep_prmt_b32"         "prmt.b32 %r1, %r2, %r3, %r4;"
run_case "sweep_prmt_f4e"         "prmt.b32.f4e %r1, %r2, %r3, %r4;"

# bfe: bit field extract (ISA 6.0+)
run_case "sweep_bfe_u32"          "bfe.u32 %r1, %r2, %r3, %r4;"
run_case "sweep_bfe_s32"          "bfe.s32 %r1, %r2, %r3, %r4;"

# bfi: bit field insert (ISA 6.0+)
run_case "sweep_bfi_b32"          "bfi.b32 %r1, %r2, %r3, %r4, %r5;"

# testp: floating-point property test (ISA 6.0+) → llvm.is_fpclass
run_case "sweep_testp_finite_f32"    "testp.finite.f32 %p1, %f1;"
run_case "sweep_testp_infinite_f32"  "testp.infinite.f32 %p1, %f1;"
run_case "sweep_testp_nan_f32"       "testp.nan.f32 %p1, %f1;"
run_case "sweep_testp_normal_f32"    "testp.normal.f32 %p1, %f1;"
run_case "sweep_testp_number_f32"    "testp.number.f32 %p1, %f1;"
run_case "sweep_testp_subnormal_f32" "testp.subnormal.f32 %p1, %f1;"

echo "PASS: PTX sweep supported-op strict checks completed"
