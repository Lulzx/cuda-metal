# CuMetal Intrinsic Map

Complete CUDA/PTX → AIR intrinsic mapping table for the CuMetal intrinsic lowering pass.

---

## Thread Indexing

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `mov.u32 %r, %tid.x` | `air.thread_position_in_threadgroup.x` | Thread index within threadgroup |
| `mov.u32 %r, %tid.y` | `air.thread_position_in_threadgroup.y` | |
| `mov.u32 %r, %tid.z` | `air.thread_position_in_threadgroup.z` | |
| `mov.u32 %r, %ctaid.x` | `air.threadgroup_position_in_grid.x` | Block/threadgroup index |
| `mov.u32 %r, %ctaid.y` | `air.threadgroup_position_in_grid.y` | |
| `mov.u32 %r, %ctaid.z` | `air.threadgroup_position_in_grid.z` | |
| `mov.u32 %r, %ntid.x` | `air.threads_per_threadgroup.x` | Block dimension |
| `mov.u32 %r, %ntid.y` | `air.threads_per_threadgroup.y` | |
| `mov.u32 %r, %ntid.z` | `air.threads_per_threadgroup.z` | |
| `mov.u32 %r, %nctaid.x` | `air.threadgroups_per_grid.x` | Grid dimension |
| `mov.u32 %r, %nctaid.y` | `air.threadgroups_per_grid.y` | |
| `mov.u32 %r, %nctaid.z` | `air.threadgroups_per_grid.z` | |
| `mov.u32 %r, %laneid` | `air.thread_position_in_simdgroup` | Lane index within warp/SIMD-group (0–31) |
| `mov.u32 %r, %warpsize` | `air.constant.warp_size` | Architecturally fixed at 32 (spec §7) |
| `mov.u32 %r, %lanemask_eq` | `air.simdgroup.lanemask_eq` | Bitmask: only current lane bit set |
| `mov.u32 %r, %lanemask_lt` | `air.simdgroup.lanemask_lt` | Bitmask: lanes with index < laneid |
| `mov.u32 %r, %lanemask_le` | `air.simdgroup.lanemask_le` | Bitmask: lanes with index ≤ laneid |
| `mov.u32 %r, %lanemask_gt` | `air.simdgroup.lanemask_gt` | Bitmask: lanes with index > laneid |
| `mov.u32 %r, %lanemask_ge` | `air.simdgroup.lanemask_ge` | Bitmask: lanes with index ≥ laneid |

---

## Synchronization

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `bar.sync N` | `air.threadgroup_barrier` | `__syncthreads()` — threadgroup scope |
| `bar.warp.sync mask` | `air.simdgroup.barrier(i32 2, i32 4)` | `__syncwarp(mask)` — SIMD-group scope and threadgroup-memory visibility; divergent half-warp ordering GPU-tested |
| `bar.arrive N, count` | `air.threadgroup_barrier` | Cooperative-groups barrier arrive (conservative: full threadgroup barrier) |
| `bar.red.and.pred d, bar, cnt, pred` | `air.simdgroup.all` | Barrier predicate AND-reduction (conservative simdgroup lowering) |
| `bar.red.or.pred d, bar, cnt, pred` | `air.simdgroup.any` | Barrier predicate OR-reduction |
| `bar.red.popc.u32 d, bar, cnt, pred` | `air.simdgroup.reduce_add` | Barrier population-count reduction |
| `membar.gl` | `air.mem.barrier.device` | `__threadfence()` — device-wide memory fence |
| `membar.sys` | `air.mem.barrier.device` | `__threadfence_system()` — system-wide fence (lowered to device scope) |
| `membar.cta` | `air.mem.barrier.threadgroup` | `__threadfence_block()` — threadgroup memory fence |
| `fence.sc.cta` | `air.mem.barrier.threadgroup` | Ampere+ fine-grained CTA fence (ISA 7.0+) |
| `fence.sc.gpu` | `air.mem.barrier.device` | Ampere+ fine-grained GPU-scope fence |
| `fence.sc.sys` | `air.mem.barrier.device` | Ampere+ fine-grained system-scope fence |
| `nanosleep.u32 ns` | `air.nanosleep` | Thread sleep hint — no-op on Apple Silicon (UMA/scheduler handles it) |
| `trap` | `llvm.trap` | Abnormal kernel termination |
| `exit` | `air.exit` | Normal kernel thread exit |
| `prefetch.global.{L1,L2} [addr]` | `air.prefetch.noop` | Cache prefetch hint — no-op on UMA (prefetch is automatic) |
| `prefetchu.L1 [addr]` | `air.prefetch.noop` | Unified-address prefetch — no-op on UMA |

## Async Copy

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `cp.async.ca.shared.global [dst], [src], sz` | `air.cp_async` | Async copy global→shared; lowered to synchronous ld+st (functional, not performance-equivalent) |
| `cp.async.commit_group` | `air.threadgroup_barrier` | Commit outstanding async copies (barrier serializes) |
| `cp.async.wait_group N` | `air.threadgroup_barrier` | Wait until ≤N outstanding groups remain (barrier serializes) |
| `cp.async.wait_all` | `air.threadgroup_barrier` | Wait for all outstanding async copies |

---

## Warp / SIMD-group Reductions

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `redux.sync.add.s32 dst, src, mask` | `air.simdgroup.reduce_add` | `__redux_sync` warp-wide add (integer) |
| `redux.sync.add.f32 dst, src, mask` | `air.simdgroup.reduce_add.f32` | Warp-wide add (float) |
| `redux.sync.and.b32 dst, src, mask` | `air.simdgroup.reduce_and` | Warp-wide bitwise AND |
| `redux.sync.or.b32 dst, src, mask` | `air.simdgroup.reduce_or` | Warp-wide bitwise OR |
| `redux.sync.xor.b32 dst, src, mask` | `air.simdgroup.reduce_xor` | Warp-wide bitwise XOR |
| `redux.sync.min.s32 dst, src, mask` | `air.simdgroup.reduce_min` | Warp-wide min (integer) |
| `redux.sync.min.f32 dst, src, mask` | `air.simdgroup.reduce_min.f32` | Warp-wide min (float) |
| `redux.sync.max.s32 dst, src, mask` | `air.simdgroup.reduce_max` | Warp-wide max (integer) |
| `redux.sync.max.f32 dst, src, mask` | `air.simdgroup.reduce_max.f32` | Warp-wide max (float) |

## Warp / SIMD-group Primitives

Apple Silicon SIMD-group width is architecturally fixed at 32 (matching CUDA warp size).

| PTX Opcode | AIR / LLVM Intrinsic | Notes |
|------------|---------------------|-------|
| `shfl.sync.idx.b32 dst, src, lane, clamp, mask` | `air.simdgroup.shuffle` | `__shfl_sync` — index shuffle |
| `shfl.sync.down.b32 dst, src, delta, clamp, mask` | `air.simdgroup.shuffle_down` | `__shfl_down_sync` |
| `shfl.sync.up.b32 dst, src, delta, clamp, mask` | `air.simdgroup.shuffle_up` | `__shfl_up_sync` |
| `shfl.sync.bfly.b32 dst, src, lanexor, clamp, mask` | `air.simdgroup.shuffle_xor` | `__shfl_xor_sync` |
| `vote.sync.ballot.b32 dst, pred, mask` | `air.simd_ballot.i64` + truncate/mask | `__ballot_sync` — active predicate bits intersected with member mask |
| `vote.sync.any.pred dst, pred, mask` | masked `air.simd_ballot.i64` != 0 | `__any_sync` |
| `vote.sync.all.pred dst, pred, mask` | masked predicate ballot == masked active ballot | `__all_sync` |
| `vote.ballot.b32 dst, pred` | `air.simdgroup.ballot` | Non-sync form |
| `vote.any.pred dst, pred` | `air.simdgroup.any` | Non-sync form |
| `vote.all.pred dst, pred` | `air.simdgroup.all` | Non-sync form |
| `vote.uni.pred dst, pred` | `air.simdgroup.all` | Uniformity test (conservative: same as all) |
| `activemask.b32 dst` | `air.simd_ballot.i64(true)` + truncate | Returns the real active-lane mask |
| `fns.b32 dst, mask, base, offset` | `air.fns` | Find Nth set bit in mask |
| `match.any.sync.b32 dst, src, mask` | `air.match.any.sync` | Warp-wide value match (any lane) |
| `match.all.sync.b32 dst, pred, src, mask` | `air.match.all.sync` | Warp-wide value match (all lanes) |

**Mask semantics note**: vote operations honor the member mask and current AIR active lanes.
Shuffle callers outside the mask receive identity; CUDA leaves their result undefined. A
masked warp barrier uses the currently active SIMD group. AIR does not take CUDA's explicit
member mask, so additional active lanes can receive stronger ordering; divergent lower/upper
half-warp shared-memory ordering is covered by a production GPU test.

---

## Arithmetic and Logic

| PTX Opcode Root | LLVM IR Intrinsic | Notes |
|----------------|-------------------|-------|
| `add` | `llvm.add` | Integer add; `add.f32` → float add |
| `sub` | `llvm.sub` | |
| `mul` | `llvm.mul` | |
| `div` | `llvm.div` | |
| `rem` | `llvm.rem` | |
| `and` | `llvm.and` | |
| `or` | `llvm.or` | |
| `xor` | `llvm.xor` | |
| `not` | `llvm.not` | |
| `shl` | `llvm.shl` | |
| `shr` | `llvm.shr` | |
| `neg.f32` | `llvm.fneg` | Float negate |
| `neg.s32` | `llvm.neg` | Integer negate |
| `selp` | `llvm.select` | Conditional select |
| `rcp` | `llvm.rcp` | Reciprocal |
| `mad.f32` | `llvm.fma` | Fused multiply-add (float) |
| `mad.s32` | `llvm.mad` | Multiply-add (integer) |
| `fma` | `llvm.fma` | Fused multiply-add |
| `max.f32` | `llvm.fmax` | Float max |
| `max.s32` | `llvm.max` | Integer max |
| `min.f32` | `llvm.fmin` | Float min |
| `min.s32` | `llvm.min` | Integer min |
| `abs.f32` | `llvm.fabs` | Float absolute value |
| `abs.s32` | `llvm.abs` | Integer absolute value |

| `clz.b32` | `llvm.ctlz.i32` | Count leading zeros (32-bit) |
| `clz.b64` | `llvm.ctlz.i64` | Count leading zeros (64-bit) |
| `popc.b32` | `llvm.ctpop.i32` | Population count (32-bit) |
| `popc.b64` | `llvm.ctpop.i64` | Population count (64-bit) |
| `brev.b32` | `llvm.bitreverse.i32` | Bit reverse (32-bit) |
| `brev.b64` | `llvm.bitreverse.i64` | Bit reverse (64-bit) |
| `bfind.u32 d, a` | `air.bfind.i32` | Find most significant non-zero bit (MSB position); 0xffffffff if a==0 |
| `bfind.s32 d, a` | `air.bfind.i32` | Signed variant (MSB of non-sign bits) |
| `bfind.u64 d, a` | `air.bfind.i64` | 64-bit variant |
| `lop3.b32 d, a, b, c, immLut` | `air.lop3` | 3-input LUT logic op (Turing+); downstream emits and/or/xor/not sequence |
| `sad.u32 d, a, b, c` | `air.sad` | Sum of absolute differences: d = |a-b| + c |
| `sad.s32 d, a, b, c` | `air.sad` | Signed variant |
| `isspacep.global pred, ptr` | `air.isspacep.global` | Pointer space test: true for global (UMA flat model) |
| `isspacep.shared/local/const pred, ptr` | `air.isspacep.nonglobal` | Always false (conservative UMA lowering) |
| `bfe.u32 d, a, b, c` | `air.bfe.unsigned` | Bit field extract (unsigned) |
| `bfe.s32 d, a, b, c` | `air.bfe.signed` | Bit field extract (signed, sign-extended) |
| `bfi.b32 d, a, b, c, f` | `air.bfi` | Bit field insert |
| `prmt.b32 d, a, b, c` | `air.prmt` | Byte permutation from two 32-bit words |
| `testp.finite.{f32,f64} pred, src` | `air.testp.finite` | Floating-point property test: `isfinite(src)` |
| `testp.infinite.{f32,f64} pred, src` | `air.testp.infinite` | `isinf(src)` |
| `testp.nan.{f32,f64} pred, src` | `air.testp.nan` | `isnan(src)` |
| `testp.number.{f32,f64} pred, src` | `air.testp.number` | `!isnan(src)` |
| `testp.normal.{f32,f64} pred, src` | `air.testp.normal` | `isnormal(src)` |
| `testp.subnormal.{f32,f64} pred, src` | `air.testp.subnormal` | `fpclassify(src) == FP_SUBNORMAL` |
| `testp.notfinite.{f32,f64} pred, src` | `air.testp.notfinite` | `!isfinite(src)` |

---

## Math Intrinsics

| PTX Opcode | LLVM IR / AIR Intrinsic | Notes |
|------------|------------------------|-------|
| `sqrt.rn.f32` | `llvm.sqrt` | Square root |
| `rsqrt.approx.f32` | `llvm.rsqrt` | Reciprocal square root |
| `ex2.approx.f32` | `llvm.exp2` | 2^x (exp base 2) |
| `lg2.approx.f32` | `llvm.log2` | log₂(x) |
| `sin.approx.f32` | `llvm.sin` | Sine; precision may differ ≤1 ULP from NVIDIA |
| `cos.approx.f32` | `llvm.cos` | Cosine; precision may differ ≤1 ULP from NVIDIA |

---

## Memory

| PTX Opcode | Lowering | Notes |
|------------|----------|-------|
| `ld.param.*` | Param load — handled structurally | Mapped to kernel argument index |
| `ld.global.f32 dst, [addr]` | `device float* ptr; dst = ptr[gid]` | Direct Metal buffer access |
| `ld.global.nc.* dst, [addr]` | plain load (same as `ld.global.*`) | `.nc` (non-coherent/read-only cache) hint; no-op on UMA — same as plain load (spec §8 `__ldg` policy) |
| `ld.const.* dst, [addr]` | plain load (AIR AS 2) | CUDA constant memory; no read-only cache benefit on UMA |
| `st.global.f32 [addr], src` | `device float* ptr; ptr[gid] = src` | Direct Metal buffer store |
| `atom.global.add.f32 dst, [addr], src` | `atomic_fetch_add_explicit(...)` | Global atomic add |
| `atom.global.add.f64` | `llvm.atomic.add.f64` | FP64 atomics via LLVM IR |
| `red.global.add.f32 [addr], src` | `llvm.atomicrmw.noret.global` | Write-only global atomic add (no return value) |
| `red.global.add.u32 [addr], src` | `llvm.atomicrmw.noret.global` | Write-only global atomic add (unsigned) |
| `red.shared.add.u32 [addr], src` | `llvm.atomicrmw.noret.shared` | Write-only shared-memory atomic add |
| `ld.const.* dst, [addr]` | `llvm.load` (AS 2) | Load from `__constant__` memory (CUDA AS 4 → AIR AS 2) |

---

## Passthrough (no lowering needed)

The following PTX opcodes are passed through to subsequent pipeline stages unchanged:

`ret`, `ld` (non-global), `st` (non-global), `setp`, `bra`, `cvt`, `cvta`, `mov` (non-special-register), `call`, `atom`, `set`

---

## Unsupported / Compile-time Errors

| PTX Feature | Status |
|-------------|--------|
| `cluster.*`, `mbarrier.*` | Per-instruction compile-time error (Hopper distributed shared memory) |
| `cp.async.bulk.tensor.*` (TMA) | Per-instruction compile-time error |
| `cvt.rn.f8x2.*` (FP8) | Per-instruction compile-time error |
| `wmma.*`, `mma.sync.*`, `ldmatrix.*` | Per-instruction compile-time error (tensor cores — no Metal equivalent; use MPSMatrixMultiplication) |
| `tex.*`, `tld4.*`, `txq.*` | Per-instruction compile-time error (texture sampling — deferred to v2) |
| `sust.*`, `suld.*`, `sured.*`, `suq.*` | Per-instruction compile-time error (surface ops — deferred to v2) |
| Dynamic parallelism (`launch_cooperative_kernel`) | Compile-time error |
