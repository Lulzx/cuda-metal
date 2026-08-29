# Known Gaps

**Note:** This document tracks divergences from the v1 spec (spec.md) and areas of partial coverage.
See [docs/status.md](status.md) for comprehensive implemented coverage (post-Phase 5,
library-shim subsets, llama.cpp/llm.c conformance via PTX path, etc.). Many items formerly listed here
as gaps have been closed.

## Durable platform / legal limits

- CuMetal targets macOS on Apple Silicon. Windows, Linux ARM, discrete-GPU,
  and Thunderbolt eGPU execution are outside the supported platform.
- SASS-only binaries cannot run: CuMetal translates documented PTX and does not
  decompile NVIDIA machine code.
- CUDA graphics interop (OpenGL/Vulkan/DirectX) is outside the Metal compute ABI CuMetal exposes.
- Multi-GPU / peer-to-peer execution is outside the single-GPU Apple Silicon target.
- Native AIR `double` pipelines are rejected by the current public Metal toolchain;
  the supported subset uses reduced-precision FP32-pair emulation instead.
- CUDA persisting-L2 cache policy has no public Metal equivalent. CuMetal accepts,
  validates, and round-trips the CUDA limit and stream access-policy state using a
  conservative quarter-L2 budget, but cannot guarantee physical cache residency.
- Grid-wide synchronization uses a device-wide sense-reversing barrier and is
  limited to cooperative grids with at most one block per reported GPU core. This
  avoids deadlock without claiming support for arbitrary oversubscribed grids.

## Deferred but implementable compatibility work

- Dynamic parallelism (kernels launching kernels) requires a CPU launch trampoline and
  explicit scheduling/error semantics; current device-side launches fail compilation.
- Native Metal texture/sampler bindings are not yet used; the source path instead
  lowers texture/surface object access through a GPU-readable software descriptor.
  Lifecycle, array memcpy, and full resource/texture/view descriptor retention
  and query are supported with stale-handle and invalid-resource coverage;
  device-side
  `tex.*` / `suld.*` etc. error at compile for the PTX path (see intrinsic-map.md).
  Kernels that sidestep sampling by dereferencing a linear/pitch2D resource's
  `devPtr` directly need `CUMETAL_USE_METAL_DEVICE_ADDRESSES=1`; without it the
  loads read as zeros. `cudaCreateTextureObject` warns once in that case.
- CUDA graph kernel/memcpy/memset/host-node capture and replay, cloning, root
  introspection, kernel-node parameter updates, and topology-compatible
  executable updates exist. A focused runtime test covers graph allocation/free
  nodes, fixed returned addresses, linear memcpy, external synchronous/asynchronous
  free, cross-graph free, auto-free-on-launch, one-executable enforcement, memory
  attributes, trimming, and negative paths. Virtual-address/physical-allocation
  reuse, cross-stream event-capture topology, and the remaining advanced node types
  are incomplete.
- MLIR GPU-dialect kernel fusion / advanced scheduling remains an optional
  architecture direction, not a compatibility claim.

## Partial / conservative implementations
- Cooperative groups include a generic `thread_group` value that preserves
  block versus tile synchronization domains; tile-scoped shuffles/votes; and
  mask/rank-aware `coalesced_group`, `binary_partition`, and
  `labeled_partition`. Focused GPU tests cover generic block/tile reductions,
  independent 16-lane barriers across two SIMD groups, noncontiguous 24-lane
  active groups, two-way and three-label partitions, arbitrary-size reduction,
  generic conversion, subgroup votes, and invalid source-rank handling. The
  dynamic reduction is correctness-first and gathers every group rank at its
  leader rather than claiming tree-reduction performance. Multi-warp static
  tiles still use a conservative whole-block barrier, and `block_tile_memory`
  remains absent.
- Masked `__syncwarp(mask != 0xFFFFFFFF)` lowers to an AIR SIMD-group barrier with
  threadgroup-memory visibility. AIR does not consume CUDA's explicit member mask, so
  additional currently active lanes can receive stronger ordering. Divergent lower/upper
  half-warp ordering is GPU-tested, including static shared-memory visibility. Partial-mask
  ballot/any/all intersect the real AIR active-lane ballot with the CUDA member mask,
  `activemask` returns the real active lanes, and shuffle callers outside the member mask
  receive identity (their CUDA result is undefined). Broader irregular-mask coverage remains
  incomplete.
- Grid-wide cooperative sync (`this_grid().sync()`) uses a hidden atomic
  sense-reversing barrier. Cooperative occupancy is deliberately reported as one
  block per GPU core and larger grids fail with
  `cudaErrorCooperativeLaunchTooLarge`; arbitrary oversubscribed cooperative grids
  remain unsupported because Metal does not guarantee simultaneous residency.
- FP64: driver and registration JIT default to `--fp64=emulate` / `CUMETAL_FP64_MODE=emulate`.
  Entry names have no semantic effect. ALU uses Dekker FP32 pairs without native AIR `double`
  (which Metal rejects at pipeline creation).

  **Storage is always IEEE-754 binary64.** Register slots, `.local` spills, global memory and
  libdevice call slots all hold the ordinary CUDA-visible bit pattern; the FP32 pair exists only
  inside a single instruction's ALU. `mov.b64`, warp shuffles, `cudaMemcpy`, the CPU-backed
  cuBLAS/cuSPARSE paths, and aliasing the same eight bytes as a `uint64_t` therefore need no
  special handling. Only FP64 *arithmetic* is reduced precision, which is a narrower and more
  defensible contract than a private packed-pair format would be.

  The conversion contract:
  - `ld` splits binary64 into `hi = roundToNearestEven_f32(x)` and `lo = RNE_f32(x - hi)`, with
    the residual taken from the binary64 significand by integer arithmetic rather than derived
    from an already-rounded `hi`.
  - `st` packs both limbs back by aligning `lo` into `hi`'s significand and rounding to nearest
    even -- never via `hi + lo` in FP32, which is what previously collapsed every value to 24 bits.
  - Round-tripping is idempotent: a value the packer produced re-splits to the same pair, so
    error does not compound across a chain of instructions.

  **Accuracy: relative error <= 2^-48 (~48-bit significand), plus an absolute floor at binary32's
  minimum normal.** The low limb is itself a binary32, so a residual below `FLT_MIN` cannot be
  represented; precision degrades smoothly from 48 bits toward 24 as `|x|` approaches `FLT_MIN`,
  and the whole representation inherits binary32's exponent range (values above `FLT_MAX`
  overflow to infinity, and intermediates below `FLT_MIN` flush to zero). Binary64 subnormal
  inputs flush to signed zero. Signed zero, infinity and NaN are preserved.

  Explicit `ld.global.f64` / `st.global.f64` and rounded FP64 integer conversions
  (`cvt.rzi.s32.f64` and friends) still fail lowering explicitly rather than computing something
  wrong. A one-time `CUMETAL WARNING` notes the reduced precision; `CUMETAL_FP64_MODE=native`
  compiles true doubles (fails at launch on current hardware).
- **Legacy default-stream ordering is complete.** Every blocking user stream publishes a
  monotonically increasing `MTLSharedEvent` value and waits on the latest legacy-default value;
  each legacy-default submission waits on the latest value from every blocking stream. The
  resource-hazard and legacy-order reservations share one atomic submission transaction, so
  concurrent host submissions cannot create cross-queue dependency cycles. Stream creation flags
  are retained and reported by `cudaStreamGetFlags` / `cuStreamGetFlags`; streams created with
  `cudaStreamNonBlocking` / `CU_STREAM_NON_BLOCKING` and per-thread default streams are correctly
  excluded from legacy implicit synchronization. Runtime and Driver API tests cover both ordering
  directions with disjoint buffers and both non-blocking negative directions.
- Registered fatbinary launches are asynchronous by default. Per-stream
  `MTLSharedEvent` values now fence accesses to the same tracked Metal buffer
  across command queues, including MPS/cuBLAS commands, and duplicate waits for
  several aliases are coalesced. `CUMETAL_SYNC_REGISTERED_LAUNCH=1` restores the
  former host drain for diagnostics. This is conservative buffer-level hazard
  ordering; it does not yet infer disjoint byte ranges within one arena.
- Device printf: compiler-recognized printf calls use a 256-byte-format bounded ring buffer and
  post-launch drain. Direct literal PTX calls and Clang's initialized module-global format plus
  packed 32-bit local argument-tuple ABI are supported in both direct MSL and generic LLVM/AIR;
  the unmodified upstream `simplePrintf` and `simpleCooperativeGroups` samples emit their expected
  records. Persistent JIT-cache hits recover the host-side format table before binding the two
  hidden ring arguments. Wider tuple values are not yet representable by the current 32-bit ring
  record. An unrecognized `call ... vprintf` ABI is rejected, never a silent zero-return no-op.
- Raw PTX `redux.sync.*` reaches a semantic AIR mapping in the phase IR, but the generic LLVM
  emitter does not yet have a validated AIR reduction ABI. It now fails explicitly instead of
  returning the caller lane's input as if a warp reduction had occurred. CUDA header
  `__reduce_*_sync` helpers continue to use the tested shuffle implementation.
- cuDNN multi-head attention has one numerically implemented CPU/UMA subset: FP32,
  projection-free, dropout-free, canonical time/batch/beam/vector sequence descriptors with
  equal Q/K/V feature size and fixed sequence lengths. Projection weights, dropout, attention
  windows, variable sequence lengths, and incremental decoding return
  `CUDNN_STATUS_NOT_SUPPORTED` instead of copying Q to O.
- Binary-shim fatbinary support: CMTL envelopes, raw PTX, basic FatBinary/FatBinary2/3
  PTX wrappers, and little-endian ELF32/ELF64 objects with named `.nv_fatbin` or raw-PTX sections
  are supported by both registration and `cuModuleLoadData`. ELF extraction follows validated
  section-table ranges under the 64 MiB image cap, including extended section counts and
  string-table indexes carried by section header 0 for both classes; the former registration-only
  blind 1 MiB memory scan has been removed. Big-endian ELF, compressed fatbin payloads, complex
  symbol layouts, and SASS-only images remain unsupported (SASS never was supported; per spec).
- PhysX 5.6 reduced GRB coverage is limited to the 93-kernel selected-shape PGS
  manifest and selected rigid/static contacts. Patch 0008 removes
  the former body-per-thread `preIntegration` and serialized `updateBodiesLaunch`
  fallbacks; their upstream warp-cooperative paths pass twenty consecutive 30-step
  CPU/GPU resting conformance runs. Patches 0009 and 0010 add a selected
  one-anchor friction path: CPU and GPU agree through sliding and reach
  no-slip rolling near `vx=3.17, wz=-3.17` at step 60, while the
  friction-disabled control retains `vx=5` and zero spin. The selected path
  stages one previous patch on the host because generic device-side
  friction-patch correlation is still unsupported.
  Patch 0011 covers two independent dynamic spheres contacting the same plane
  by launching each contact pre-prep/prepare batch as its own 32-lane Metal
  SIMD group and indexing the static solve per island body and slab.
  Patch 0012 covers one selected dynamic/dynamic contact in a two-sphere stack.
  It directly resets and writes back Metal solver buffers without shared
  device-pointer staging, aggregates body slabs, and matches CPU for 30
  frictional and frictionless steps. Larger stacks, multiple simultaneous
  dynamic contacts per body, packed general batching, joints, articulations,
  user impulse limits, general falling-contact, and chaotic long-run solver
  conformance are not claimed. Patch 0013 adds a selected unit box/plane path
  through `convexPlaneNphase_Kernel`; its four contact points and 30-step
  frictionless CPU/GPU transforms agree after fixing entry-specific aligned
  static shared-memory layout. Patch 0014 adds `boxBoxNphase_Kernel`; a selected
  two-unit-box stack matches CPU body states for 30 frictionless steps after
  forcing viable CUDA device calls to inline. Patch 0016 adds one cooked
  six-vertex prism topology and verifies one selected dynamic convex/convex pair
  above the convex/plane contact over 30 frictionless steps. The gate requires
  both GJK/EPA stages, contact finalization, dynamic/static preparation and
  solve, writeback, and integration to dispatch as production Metal kernels;
  CPU/GPU states stay within 1% component-wise, with the largest transient in
  angular velocity at step 5. Arbitrary convex topology/orientation, multiple
  simultaneous convex pairs, heightfields, and SDF collisions remain unsupported
  or unverified. Patch 0017 adds one selected sphere/static-triangle-mesh path:
  a frictionless unit sphere moving across two coplanar faces matches CPU
  byte-for-byte for 30 steps. It carries a single separation value
  through the correlation index as a scoped workaround for an incoherent generic
  temporary-contact record. Patch 0018 masks PhysX's flagged adjacency indices
  and preserves plane separation across the selected coplanar seam. Patch 0019
  runs the existing one-anchor friction path over that seam and stays within the
  established 60-step `3e-3` CPU/GPU envelope. Non-coplanar seams, mesh contacts
  with other shapes, multiple bodies, generic friction correlation, and general
  mesh batching remain unsupported.
  The 60-step friction gate is repeatable, but its `3e-3` relative plus `1e-5`
  absolute tolerance is not evidence of general FP determinism. Runtime-compiled MSL now has
  an explicit `CUMETAL_MSL_MATH_MODE=fast|safe` policy: `fast` preserves the historical
  default, while `safe` selects `MTLMathModeSafe` on macOS 15+ and disables fast math through
  the legacy compile option on macOS 14. The normalized mode is isolated in the registration
  JIT cache and reported as `math_mode=` in GPU provenance. This does not retroactively change
  precompiled metallibs. Explicit FMA-contraction coverage and long chaotic-scene divergence
  remain unverified and still require a dedicated contraction matrix plus long-horizon
  conformance.
  Convex/convex stage 2 compiles from canonical non-inline CUDA/NVVM to a
  validated metallib through typed CuMetal IR. Stage 1 still uses the explicit
  legacy CUDA-to-PTX backend because typed generic-pointer legalization reports
  a conflicting concrete address space. The typed backend structures the
  nested natural loops, preserves values across loop exits, specializes helper
  calls selected by mixed CUDA address-space PHIs, threads static shared globals
  through reachable helpers, emits value-returning device calls, and performs
  byte offsets through byte pointers rather than scaled aggregate pointers. A
  selected six-vertex prism scene produces a finite three-point manifold and
  passes the committed 30-step two-body gate. Contact finalization currently
  uses legacy PTX lowering; its mixed static-shared/constant-symbol address bug
  is covered by positive and undeclared-symbol negative tests. General
  convex/convex runtime support is therefore still not claimed.

## .cu / cumetalc frontend limitations
- `cumetalc --cuda-device` is the real source frontend for project-scale CUDA:
  it requires a CUDA-capable Homebrew LLVM Clang (or
  `--cuda-clang`/`CUMETAL_CUDA_CLANG`) and forwards `-I`, `-D`,
  `--cuda-include`, and `--cuda-arch`. It still requests `-fno-jump-tables` to
  favor structured control flow, but PTX `brx.idx`/`.branchtargets` tables are
  now parsed and lowered to LLVM switches. This closes the aggregate-shuffle
  switch in NVIDIA's unmodified `shfl_scan`, whose sum and 1920x1080 integral
  image both validate exactly. CUDA source compilation can still succeed while later
  strict PTX lowering still rejects an unimplemented opcode or libdevice call.
  Standalone PTX `.func` bodies are not lowered; projects can request aggressive
  device inlining with `--cuda-inline-threshold`, which on LLVM 22+ also forces
  every viable reachable device call to inline. Recursion, indirect calls, and
  explicitly non-inlineable helpers remain unsupported. The reduced PhysX
  rigid-body subset uses this for helpers including `updateCacheAndBound` and
  `getIncidentPolygon4`.
- The older `.cu` mode without `--cuda-device` remains a qualifier-stripping
  host-LLVM prototype suitable only for simple patterns; it is not a general
  CUDA frontend.
- The backend default now follows the input rather than being one global setting, because the
  two backends are complementary rather than ranked. Measured over the 19-file in-tree `.cu`
  corpus (`tests/` + `samples/`):

  | frontend | `--backend=legacy` | `--backend=cumetal-ir` |
  |----------|-------------------|------------------------|
  | direct `.cu` | 0/19 | **10/19** |
  | `--cuda-device` (PTX) | **17/19** | 6/19 |

  So a direct `.cu` input defaults to `cumetal-ir` (legacy's direct-`.cu` mode is the
  qualifier-stripping prototype below and lowers nothing in this corpus), while `--cuda-device`
  and PTX inputs default to `legacy` — defaulting those to typed IR would regress the path
  llm.c, llama.cpp, and PhysX depend on. `--backend` overrides either way. Reproduce the table by
  running both backends over `find tests samples -name '*.cu'`.

  The typed `cumetal-ir` path supports selected-entry device-call closures, structured natural-loop CFG
  lowering with multiple exits and nested `continue`, dispatcher fallback for
  barrier-free residual CFGs, loop-carried PHIs, CUDA vector and named aggregate values,
  thread-local allocas, constant global tables, warp shuffle/vote operations,
  transitive Metal builtin threading, and common CUDA/libdevice math and bit
  intrinsics (including exp/log/trigonometric/rounding/pow operations). Mixed
  CUDA-generic pointers can dispatch
  supported loads, stores, offsets, and void helper calls across their concrete
  Metal address spaces; unsupported mixed-pointer operations, dynamic shared-memory
  emission, atomics, reductions, full FP64 handling, and the remaining intrinsic
  surface still fail explicitly. Static CUDA shared globals with compile-time sizes are
  emitted as kernel-local threadgroup storage and threaded through device calls.
- Stock Clang CUDA device IR import requires LLVM 18 or newer at build time.
  Unknown NVVM intrinsics, arbitrary pointer/integer round trips, indirect
  calls, recursion, unsupported atomics, and irreducible/unsupported CFG shapes
  are rejected. The new backend never falls back to qualifier stripping,
  legacy PTX lowering, substitutions, or CPU execution.
- The old source-pattern-specific vector-add AIR template has been removed.
  Direct AIR generation remains limited to explicit research/inspection tools;
  it is not a hidden production fallback.
- **Four further name-matched body templates were found and removed on 2026-07-26.** The claim
  immediately above was true of the AIR emitter but false of `lower_to_llvm.cpp`, which still
  carried `vector_add`, `matrix_mul`, `negate`, and `reduce_sum` templates. Each replaced a
  kernel's *actual* PTX body with a canned implementation whenever the entry name contained a
  matching substring and the parameters had roughly the right shape, and each was consulted
  **before** generic lowering was attempted, which also bypassed `--ptx-strict`.
  A kernel named `neg_but_actually_triples` whose PTX computed `x*3` was emitted as `fneg`.
  `neg.s32` came out as a float sign-bit flip: `-(7)` returned `0x80000007` instead of
  `0xFFFFFFF9`. The name match additionally mutated the ABI — retyping parameters and appending a
  thread-position builtin — which then caused generic lowering to fail, which selected the
  template that produced the wrong body. Removing all four required no compensating work: the
  generic path lowers every affected case, and the full suite stays green.
  The unit and AIR ABI fixtures that "covered" these paths were themselves artifacts — each had a
  stub body (`mov.u32 %r0, %tid.x; ret;`) while asserting a fully computed body, so they were
  verifying the template rather than the compiler and could never have caught the miscompile.
  They are now real kernels with assertions on real lowering, plus explicit negative tests that a
  matching entry name does not substitute semantics.
  Found by `ptx_sweep_numeric` on its first run.
- A full audit of every name-driven decision in the compiler and runtime is recorded in
  [name-match-audit-2026-07-26.md](name-match-audit-2026-07-26.md), including one further site
  removed from the runtime launch path and the categories that were cleared.
- **The MSL name-matched specialization table no longer pre-empts real translation
  (2026-07-26).** `lower_to_metal.cpp` consulted its hardcoded llm.c/GGML entry-name table
  *before* attempting generic PTX→MSL translation — the comments said so outright — so a kernel
  whose name merely *contained* one of those substrings had its real body replaced by a canned
  implementation even when generic translation could handle it. A kernel named
  `gelu_forward_kernel_mine` that doubled its input was emitted as GELU. Generic translation is
  now attempted first and the table is consulted only when it declines.
  This surface is less exposed than the LLVM-path templates were — the names are long and
  specific (`encoder_forward_kernel3`, `adamw_kernel2`) rather than generic words, and the
  result is at least labelled `specialized_msl` in provenance rather than claiming to be a real
  translation — but the failure mode was the same, and it would also have bitten on version
  skew: if llm.c or GGML changed what a kernel of a given name computes, CuMetal would have gone
  on silently computing the old definition.
  **Name collisions no longer select a body by default.** Where generic translation cannot lower
  a kernel, the table is consulted only when the caller explicitly enables
  `CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=1`. The llm.c and llama.cpp conformance launchers opt
  in because their currently verified paths depend on exact specialized kernels; ordinary CUDA
  projects do not. Selected bodies remain reported as `specialized_msl` /
  `workload_specialization` in `CUMETAL_TRACE_GPU=1` provenance. Ordering and the default-off
  policy are pinned by `unit_ptx_lower_to_metal`.
- **A kernel that cannot be lowered is now refused instead of emitted empty.** Non-strict
  lowering previously fell through to a bare `ret void` body, producing a kernel that loaded,
  launched, and wrote nothing — the caller read back whatever was already in the output buffer
  with no diagnostic. Both strict and tolerant modes now return an error.
- **`cumetal run` is an environment-scoped launcher, not a universal binary translator.** It
  prepends the selected installation's library directory only for the child process. Programs
  built by `cumetalc` already run directly and do not need it. Prebuilt applications require
  `CUMETAL_ENABLE_BINARY_SHIM=ON`, a load command that dyld can resolve through the shim, and
  embedded supported PTX. Absolute NVIDIA library paths and SASS-only binaries are not repaired.
- **`cumetalc foo.cu -o foo` produces a complete linked executable** (spec §11 Phase 2/3 exit
  criterion), covered by `functional_cumetalc_link_executable`. An unmodified CUDA source file
  using `<<<>>>` compiles and runs with no host/device split and no metallib path at runtime.
  The driver runs Clang over the whole translation unit: the host side compiles to the standard
  CUDA registration ABI, the device side goes through the in-tree `ptxas`/`fatbinary` shims that
  carry PTX into a fatbinary envelope, and `libcumetal` lowers that PTX to a metallib on first
  launch. It resolves headers, `libcumetal.dylib`, and the shims relative to its own location, so
  it works from both the build tree and an install prefix (`CUMETAL_ROOT` overrides).
  Note what this does *not* change: device-code coverage is still exactly the PTX lowering
  surface documented below. A program whose kernels fall outside it will link and then abort at
  launch with the usual "registered kernel missing metallib" — the driver makes the toolchain
  complete, not the lowering.
- The CuMetal-native registration ABI and runtime lookup path are implemented and versioned.
  Automated host-job rewriting and generated launch stubs for the *native* (`cumetalKernel_t`)
  ABI are still not wired into `cumetalc`; that path remains the explicit two-file flow shown in
  `samples/nativeLaunch`.
- The Clang-based `.cu`/PTX registration path supports many simple kernels and
  samples (vectorAdd etc.) and dispatches them through Metal on the Apple GPU.
  CUDA kernel CPU emulation is disabled by default. The legacy llm.c host
  implementation is diagnostic-only and requires
  `CUMETAL_ENABLE_LLMC_CPU_EMULATION=1`. GGML's `k_compute_batched_ptrs` is an
  exact runtime ABI helper rather than a kernel fallback: it synchronizes its
  input stream and writes native Metal GPU addresses into the cuBLAS tables.
  CPU kernel emulation continues to emit a warning.
  `CUMETAL_TRACE_GPU=1` provides positive dispatch evidence.
- Complex CUDA C++ sources exercise mixed coverage. The strict llm.c GPT-2 FP32
  conformance workload reaches numerical parity on Apple M4 Pro with CPU emulation
  disabled, using specialized MSL replacements. llama.cpp's much broader GGML
  CUDA kernel set remains incomplete.
- **The llm.c parity intermittency is fixed (2026-07-26).** `conformance_llmc_gpt2fp32cu` used to
  fail 2-4 runs in 15 with a real numerical divergence (`LOSS MISMATCH AT STEP 1: 3.752515
  4.059707`, sometimes a `-inf` loss) at a step that varied between runs. Two independent defects,
  both fixed; measured 0/75 afterwards against 2/25 with the race left in.

  1. **The JIT cache key did not describe the compiler that produced the entry.** The key was
     (hand-maintained schema string + policy + PTX + kernel name), so any change to lowering — an
     MSL template, an instruction handler, a legalization rule — produced the *same* key and the
     runtime silently reused a metallib compiled by the previous build. Correctness depended on a
     human remembering to bump a magic string. Demonstrated directly: editing the
     `fused_classifier_kernel3` template and re-running llm.c left the old kernel executing, and a
     cache populated across several builds holds kernels from different compiler versions at once.
     It also crosses build trees — a worktree at an older commit shares the cache and will consume
     entries a newer build wrote. The key now includes the libcumetal Mach-O `LC_UUID`, which the
     linker regenerates whenever the binary changes, so an entry cannot outlive its compiler. A
     rebuild invalidates automatically; a user who never rebuilds keeps full reuse.
  2. **A race in the specialized `fused_classifier_kernel3` MSL template.** Thread 0 read
     `row_logits[target]` to compute the loss while every thread overwrote `row_logits[]` with
     gradients immediately below, with no barrier between. Whichever thread owned index `target`
     could store first, so the loss was computed from a gradient instead of a logit. Upstream
     llm.c has the same shape but keeps its warps in step through a block-wide cooperative
     reduction; this port has thread 0 do the whole max/sum scan alone and more work after the
     barrier, widening the window enormously. Fixed with a device-memory threadgroup barrier.

- The binary-shim / PTX reg + lower path (plus special llm.c cases) gets further than pure
  generic emitter. Direct MSL name-matched cases (compiler/ptx/src/lower_to_metal.cpp) now cover
  common GGML kernels used by small models: k_bin_bcast (op_addff/op_mulff + f16 variants),
  rms_norm_f32 (with stride/mul/add support), Q8_0/Q6_K-to-f16 dequantization,
  typed float/half conversion, forward RoPE, gated SiLU, and typed float/half
  strided scalar copies.
  A fast negative filter skips heavy lowering for the bulk of GGML's 1000s of mul_mat_q* / flash
  / other dequants / cpy etc (they hit "registered kernel missing" and GGML typically falls back
  or aborts depending on NGL and op).
- **Approximate/passthru stubs are unconditionally refused (no silent wrong answers).** A handful of
  templates (unsupported `convert_unary`, `rope_norm`, and `rope_neox` variants,
  `dequantize_q5_0`/`_block_q5`, `k_set_rows`, and unsupported `cpy_`/`k_cpy`
  variants) exist only as passthru placeholders — they copy or zero data
  instead of computing the real quantized/rotary/copy result. Their output is numerically wrong,
  so lowering **always skips them** (the kernel falls through to the same clean "registered
  kernel missing metallib" abort as any unsupported op) and prints a one-time
  `CUMETAL WARNING: kernel '…' has only an approximate/passthru lowering and was skipped …`.
  `CUMETAL_ENABLE_APPROX_KERNELS` no longer overrides this safety rule.
- **The covered llama.cpp SmolLM2 path is numerically coherent.** Rechecked
  2026-07-23 on SmolLM2-135M-Instruct-Q4_K_M, greedy decode of
  "The capital of France is":
  - Stock CPU llama.cpp (no CuMetal): `Paris.` ✅
  - llama.cpp linked against libcumetal, **NGL=0**: `Paris.` ✅
  - llama.cpp linked against libcumetal, **NGL=1**: `The capital of France is
    Paris.` ✅ at 279.2 tokens/s median generation across five warm runs on
    Apple M4 Pro (223.2–307.7 tokens/s observed).
  - Registration resolves only the launched PTX entry signature, avoiding ABI
    metadata allocation for thousands of unused GGML kernels. This reduced the covered one-layer,
    one-token run from 8.20 s to 1.00 s on Apple M4 Pro; native FP16
    `cublasGemmEx` lowering further reduced the five-run warm median to 0.57 s
    and the 16-token gate to 0.61 s. Memoized streaming cache keys and targeted
    ABI resolution subsequently reduced the controlled 16-token warm median
    from 0.60 s to 0.575 s (about 4.2% versus `a41b4e5`). The earlier
    linear-scanner change had already reduced it from 290.24 s. Actual kernel
    lowering retains the full parser. Unannotated 64-bit PTX parameters remain
    conservatively pointer-classified, with the existing allocation-aware
    launch fallback for small scalar values.
  - The conformance harness (`run_llama_cpp_cumetal.sh`) now enforces a **coherence gate**: greedy
    decode must contain the expected answer (`CUMETAL_LLAMA_EXPECT`, default `Paris`) and an
    NGL>0 run must include completed Apple-GPU provenance, so the test correctly
    FAILS on garbage instead of passing on "some bytes were generated". Set
    `CUMETAL_LLAMA_EXPECT=""` to opt out explicitly.
  - The harness uses llama.cpp `--simple-io` so token output reaches its capture
    pipe even when a controlling terminal is present. The gate parses the
    combined token/provenance capture byte-wise and removes provenance records
    before coherence matching, including when a record splits token fragments.
  - This remains a focused SmolLM2 result, not a claim that arbitrary models are
    supported. Broader GGML kernel coverage is still required. Device-resident
    batched-GEMM pointer tables now preserve
    Metal GPU-address identity, and FP16-input/FP32-output `cublasGemmEx` now
    runs directly through MPS. CuMetal's llama.cpp build also disables
    `GGML_CUDA_FA`, making the backend select ordinary attention instead of
    fused kernels that CuMetal cannot lower. The former NGL=4 incoherence was
    isolated to GGML's strided `cpy_scalar<half, half>` value-cache
    materialization: the old LLVM path handled the first logical row but not the
    later strided rows. Exact MSL float/half scalar-copy variants preserve all
    four source and destination byte strides. The 2026-07-23 automated sweep
    passes every NGL from 1 through 99; values above the model's layer count are
    repeated full-offload saturation checks.

## Tooling / build notes
- CMake discovers the standalone Apple Metal toolchain for the entire CTest
  graph and preserves explicitly selected `TOOLCHAINS` values. Discovery is
  verified before it is cached, so installing the component after an earlier
  configure is recoverable.
- `tests/cuda_projects/sweep_cuda_projects.py` provides a manifest-complete
  strict sweep with classified TSV/JSON output and a fresh JIT cache per fixture.
  The 2026-07-27 local baseline is eleven passes, including the strict libdevice
  and ray-tracer projects; the earlier `sgemm_2d` numerical failure was a PTX
  `.local` stack-depot sizing bug, now fixed.
- Direct PTX→MSL pointer-base resolution was flow-insensitive until 2026-07-26:
  `lower_to_metal` kept one register→base map per entry, so a register reassigned
  from one pointer base to another resolved every use to the last assignment and
  emitted wrong code instead of declining to match. Address classification is now
  snapshotted at each global load/store/atomic while the forward pass is at that
  instruction. Covered by `unit_ptx_lower_to_metal` and end-to-end by
  `functional_runtime_ptx_lowering_regression`.
- `air_emitter` "experimental" mode produces test containers, not production metallib ABI (for validation/air_abi only; runtime execution requires real metallib from xcrun or prebuilt).
- AIR metadata validation relies on MetalLibraryArchive + xcrun where available; the
  bridge is optional at build time.
- Homebrew LLVM users targeting sm_70+ need the feature-flag shim
  (`scripts/cumetal_cuda_flags.sh`) because of PTX version defaults; the in-tree
- cuda_projects conformance harness now runs its compile step (clang -x cuda shim + fatbin registration setup) in environments without xcrun metal/metallib (only base xcrun + clang++ needed); runtime exec still limited by PTX lowering coverage for complex kernels (sgemm etc.) and falls back gracefully to SKIP (see run_standalone_cu.sh). This reduces skip-only coverage for the harness itself.
  `scripts/cuda_toolchain/fatbinary` accepts modern `--image3` args.
- The external llm.c stress gate now passes through specialized Metal kernels.
  llama.cpp builds, links, initializes, and executes a covered subset, but other
  GGML kernels (mul_mat variants, dequants, flash attention, conversions, and
  rotary operations) still hit lowering gaps or are refused placeholder paths. Exact forward,
  no-frequency-factor GPT-NeoX RoPE is covered for the concrete float-to-float and float-to-half
  GGML ABIs; backward, frequency-factor, and half-input `rope_neox` variants remain refused
  rather than using the passthrough.
  See the bin_bcast special case in compiler/ptx/src/lower_to_metal.cpp and Metal source path
  in runtime/metal_backend.
- Full AIR ABI reverse-engineering continues to be refined as Xcode releases change
  undocumented fields (regression tests in `tests/air_abi/` + `air_validate` catch breaks).
- **AIR ABI cross-version coverage is narrower than spec §10.5 asks for.** The spec wants
  regression across Xcode 15.0 / 15.4 / 16.0 / 16.2+; in practice a developer machine has one
  Xcode, and there is no CI to spread the matrix across machines. Every
  AIR ABI test now prints a `CUMETAL_AIR_ABI_PROVENANCE` line naming the macOS build, Xcode
  version and build, selected `TOOLCHAINS`, chip, and metal compiler version, so a result is
  attributable to a toolchain instead of floating free.
  `air_abi_xcode_matrix_regression` previously defaulted both of its Xcode slots to
  `xcode-select -p` when the override variables were unset, compiled the same source twice with
  the same compiler, and reported "Xcode matrix ABI regression checks succeeded" — claiming
  cross-version coverage it never had. It now identifies toolchains by metal compiler version
  (not directory path), deduplicates them, and says plainly when only one is present.
  `CUMETAL_REQUIRE_XCODE_MATRIX=1` turns a single-toolchain environment into a failure for
  machines that do have two Xcodes installed. Genuinely exercising the specific versions in the
  spec table still requires installing them side by side.

## External dependency for full stress conformance
- The llm.c GPT-2 FP32 stress harness remains available for explicit manual runs,
  but is not registered in CTest. The llama.cpp test requires an external source
  checkout plus model assets and auto-skips (77) when absent. When present these
  harnesses exercise real production kernels.

## AIR / metallib
- Production compilation uses typed MSL and Apple's `metal`/`metallib` tools.
  The emitter + validate + runtime loading continue to serve legacy paths and
  AIR ABI regression tests where Xcode toolchains are present.
- "Full" metadata RE is effectively complete for the kernels we emit; unknown future
  ABI changes will be caught by the xcode regression harness.

## NVIDIA cuda-samples sweep (2026-08-29)

83 gated headless samples from `NVIDIA/cuda-samples` (`cpp/0_Introduction`,
`2_Concepts_and_Techniques`, `3_CUDA_Features`, `4_CUDA_Libraries`,
`6_Performance`) are compiled and run against `libcumetal`. Result:
**83 pass, 0 waive cleanly, 0 do not yet have a passing runtime result**.

This runs as `conformance_cuda_samples_sweep`. The samples themselves are not
vendored -- the test skips (77) unless a `cuda-samples` checkout is present at
`../cuda-samples` or `CUMETAL_CUDA_SAMPLES_DIR`, and supports both the current
`cpp/` and the older `Samples/` layout. Each sample's outcome is compared against
`tests/cuda_projects/cuda_samples_sweep_manifest.txt`. Every listed sample must
build, run, and satisfy its available numerical/output checks; waivers and
unsupported outcomes fail the current all-pass baseline.

The sweep first found one defect that masked everything else: CuMetal's headers
used `#pragma once` and never defined CUDA's canonical include-guard macros
(`__DRIVER_TYPES_H__`, `CUBLAS_API_H_`, `_CUFFT_H_`, ...). NVIDIA's own
`Common/helper_cuda.h` -- and plenty of third-party CUDA code -- feature-detects
on those to decide whether to declare `checkCudaErrors()` and friends, so 82 of
88 samples failed with "use of undeclared identifier 'checkCudaErrors'" while
the headers themselves compiled clean. Fixed, along with a `cudaDataType`
collision (`cusparse.h` declared it `typedef int`, `cublas_v2.h` as an enum, so
including both was a hard typedef-redefinition error); `library_types.h` now
owns that type the way real CUDA does.

The unmodified `simpleCUFFT` sample also exposed two independent silent-wrong
paths. Generic PTX-to-MSL global memory emission discarded a derived pointer's
byte stride and literal displacement, so scalar loads for one `float2` both read
`ptr[gid]` and both stores targeted the same scalar. It now derives checked Metal
element indices from the PTX byte address. Separately, the synchronous CPU/vDSP
cuFFT shim did not wait for a preceding Metal kernel on its stream; every execute
now synchronizes its bound/default stream before accessing plan buffers.

The unmodified `dct8x8` sample likewise caught two independent generic-runtime
errors. Synchronous `cudaMemcpy2D` now waits for preceding Metal producers before
performing its host-side UMA row copy, which makes the float texture DCT/IDCT path
match its CPU oracle. The LLVM PTX backend now implements `mul.wide.s16` as a
signed 16-to-32-bit multiply rather than accidentally zero-extending through its
32-bit implementation. All three GPU variants pass: float matrix, float texture,
and fixed-point short.

The unmodified `interval` sample now passes all three Newton implementations and
its CPU comparison. Device functions use their own declared PTX `.local` depot
sizes instead of losing helper stack frames, and FP64-pair emulation lowers
`abs.f64` and `selp.f64` entirely through FP32/integer operations. This avoids
leaking native `double` instructions that Apple GPU pipeline creation rejects.

What the sweep says is still missing, in rough order of how many samples it
blocks:

- **Texture and surface fetch in device code** -- Texture/surface objects now own
  GPU-readable descriptors whose nested payload buffers are declared resident at
  Metal dispatch without consuming public kernel argument slots. Point and linear
  `tex2D`, layered and 3D fetches, cubemap face mapping, unnormalized-coordinate
  clamp semantics, normalized unsigned reads, and `surf2Dwrite` are implemented.
  NVIDIA's unmodified `simpleTexture`, `simplePitchLinearTexture`,
  `simpleLayeredTexture`, `simpleCubemapTexture`, `simpleSurfaceWrite`,
  `convolutionTexture`, and `lineOfSight` numerical checks pass. The headless
  `simpleTexture3D --file data/ref_texture3D.bin` reference path also passes;
  CUDA/OpenGL resource mapping remains explicitly unsupported even though the
  clean-room declarations let programs with a separate headless path link.
- **Remaining `libcu++` device headers** -- Clean-room `cuda/pipeline` and
  `cuda/barrier` provide synchronous shared-memory copies and block barriers;
  NVIDIA's default `globalToShmemAsyncCopy` 1280-by-1280 matrix run passes its
  numerical check. `cooperative_groups/memcpy_async.h` and broader variants
  remain incomplete.
- **Remaining cooperative-groups surface** -- Generic `thread_group`,
  `coalesced_group`, binary/labeled partitions,
  tile `shfl_up`, and dynamic-group reduction have focused GPU coverage; the
  unmodified `binaryPartitionCG` and `warpAggregatedAtomicsCG` translation units
  now compile and link. Their 100K/10M-element workloads were deliberately not
  run in this resource-bounded pass, so neither manifest entry is promoted.
  `simpleCooperativeGroups` now passes its unmodified 64-thread run: the LLVM/AIR
  path writes Clang's initialized-global / packed-local `vprintf` ABI into the
  bounded runtime ring, including vector tuple stores and immediate lanes. The
  sweep requires the one whole-block and four tiled-partition numerical records,
  so a zero-output launch cannot be counted as a pass. The remaining samples in
  this cluster retain independent runtime/header blockers.
- **Thrust header and ordering surface** -- Clean-room copy, sequence,
  generation, random distribution, keyed scan/unique, adjacent difference,
  allocation, and sortedness surfaces synchronize preceding GPU work before
  their CPU-on-UMA loops. NVIDIA's unmodified `radixSortThrust` passes its
  1,048,576-key/value run, and `segmentationTreeThrust` builds all ten expected
  image-pyramid levels and validates its two reference levels. The
  `cub::DeviceFind`, `cub::DeviceSegmentedScan`, and N-input/M-output
  `cub::DeviceTransform` surfaces now run on unified memory, reject invalid
  counts, and pass NVIDIA's three corresponding unmodified samples.
- **Broader device `printf` formats** -- the clean-room header and bounded ring-buffer backend
  handle direct literal PTX plus Clang's initialized module-global format / packed 32-bit local
  tuple form on both compiler backends. The unmodified `simplePrintf` and
  `simpleCooperativeGroups` samples pass. The ring record still stores one 32-bit word per
  argument, so binary64 and full 64-bit integer formatting remain explicit follow-on work.
- **WMMA** -- The clean-room `mma.h` preserves documented fragment storage and
  distributes tiles across one 32-lane SIMD group. FP16/FP32 accumulation,
  unsigned 8-bit/integer accumulation, and TF32 use public Metal ALU and shuffle
  operations. BF16 16-by-16-by-16 multiply-accumulate is decomposed into public
  Metal 8-by-8 SIMD-group matrix loads/MACs/stores, retaining user-visible
  fragment edits between operations. Focused GPU oracles validate every output
  element, including a nonzero modified BF16 accumulator. NVIDIA's unmodified
  `cudaTensorCoreGemm`, `immaTensorCoreGemm`, `tf32TensorCoreGemm`, and
  `bf16TensorCoreGemm` workloads finish successfully. The 8-by-8-by-4 DMMA shape
  stages FP64-emulated operands as FP32 and uses public Metal FP32 SIMD-group
  matrix operations; a focused oracle validates all 64 outputs and the
  unmodified `dmmaTensorCoreGemm` workload completes. Large FP32-matrix grids
  are split into watchdog-safe command buffers through a hidden block-Y offset,
  preserving CUDA `blockIdx.y`. This is not full IEEE-754 binary64 WMMA semantics.
- **Device C++ allocation and virtual dispatch** -- Device `malloc`/`free`,
  scalar/array `new` and `delete` (including sized delete) use a persistent,
  bounded Metal-shared heap with atomic bump allocation and free-list reuse.
  Initialized symbolic `.u64` tables carry stable function tokens, and indirect
  PTX calls dispatch those tokens to lowered device-function bodies while
  preserving local/global pointer address spaces. Focused GPU coverage checks
  cross-launch lifetime, exhaustion, reuse, virtual methods, local reference
  results, and heap-limit negative paths. NVIDIA's unmodified `newdelete` sample
  passes its container heap, shared-memory placement-new, and user-defined
  complex-type tests.
- **Driver virtual-memory allocation** -- The reserve/create/map/access/unmap
  lifecycle is backed by CuMetal's tracked Metal-shared allocations, including
  64 KiB granularity, primary-context interop, compression-property round-trip,
  and strict invalid-state checks. The generic-compression field is an accepted
  allocation hint: public Metal exposes no API to force or inspect the physical
  compression policy. NVIDIA's unmodified `cudaCompressibleMemory` sample
  completes its compressible-hint and ordinary 160 MiB SAXPY runs.
- **CUDA graph memory nodes** -- The clean-room CUDA 12 ABI and runtime now expose
  `cudaGraphAddMemAllocNode`, `cudaGraphAddMemFreeNode`,
  `cudaGraphAddMemcpyNode1D`, parameter queries, auto-free instantiation, device
  graph-memory attributes, and trimming. Focused runtime coverage validates fixed
  addresses; ordered same-graph free; allocation lifetime beyond graph destruction;
  `cudaFree`, `cudaFreeAsync`, and relaunch; free-only cross-graph execution;
  mutually exclusive free ownership; high-water counters; and invalid parameters.
  NVIDIA's unmodified `graphMemoryNodes` now passes its ordinary stream,
  event-linked multi-stream capture, explicit graph, external `cudaFree`,
  external `cudaFreeAsync`, and cross-graph free validations over 8 million
  elements. Captured async allocations/frees become graph memory nodes, and
  embedded device pointers in by-value kernel structs are relocated to Metal
  GPU addresses and their target buffers are declared resident for the dispatch.
  `graphMemoryFootprint` passes using CuMetal's monotonic device
  clock emulation. CuMetal does not yet reuse one freed graph virtual
  address for a later allocation or model CUDA's retained physical allocator cache,
  so `cudaDeviceGraphMemTrim` is a correct no-op only for the current no-cache
  implementation. `simpleCudaGraphs` now produces consistent FP64 reductions,
  and `jacobiCudaGraphs` now completes its Jacobi compute pipelines; both pass
  the unmodified sample sweep.
- **cuFFT advanced plans** -- Contiguous rank-1 `cufftXtMakePlanMany` maps matching
  FP32/FP64 C2C, R2C, and C2R types onto the existing vDSP-backed engine. A bounded
  direct DFT covers vDSP-unsupported lengths through 1,024 elements, and synchronous
  shim execution waits for the bound/default CUDA stream. This closes NVIDIA's
  unmodified `simpleCUFFT` sample, including its 56-point convolution and device
  pointwise-multiply kernel. Strided/embedded Xt layouts, multidimensional execution,
  mixed precision, half/bfloat16, and larger vDSP-unsupported lengths remain explicit
  `CUFFT_NOT_SUPPORTED` gaps.
- **Dynamic parallelism (CDP)** -- Device-side launch records are drained and
  submitted by the host runtime; the four gated CDP samples pass. This is a
  compatibility scheduler, not NVIDIA hardware dynamic-parallelism parity.
- **Library pointer modes** -- cuBLAS handle state supports host and device
  pointer modes. The synchronous S/D AXPY, SCAL, and DOT paths resolve tracked
  device scalar storage (including native Metal-address allocations), and reject
  a scalar from the wrong location; focused runtime coverage exercises device
  alpha and result values. The remaining cuBLAS operations have not all been
  audited for device scalar arguments, and graph capture still needs asynchronous
  library-node semantics. cuSPARSE exposes host pointer-mode state; selecting
  device mode returns `CUSPARSE_STATUS_NOT_SUPPORTED` instead of silently
  dereferencing the wrong address. NVIDIA's unmodified
  `conjugateGradientCudaGraphs` workload now passes the unmodified sweep.
- **cuSPARSE ILU(0) preconditioning** -- CSR ILU(0) analysis/factorization,
  sparse-matrix fill/diagonal attributes, and lower/upper SpSV substitution run
  over synchronized UMA storage with negative-path coverage. NVIDIA's
  unmodified `conjugateGradientPrecond` converges in 564 iterations without
  preconditioning and 188 with ILU(0), reporting zero errors.
- **HiGHS GPU PDLP integration** -- `cusparseDnVecSetValues`,
  `cusparseSpMV_preprocess`, and CUDA-graph capture/replay of cuSPARSE SpMV are
  implemented. Focused tests cover non-eager capture, captured pointer identity,
  changed device data across replays, and both forced-Metal and CPU sparse
  routes. An unmodified HiGHS 1.15.1 `CUPDLP_GPU=ON` build solves `afiro` with
  `--presolve off` to Optimal while traced kernels run on the Apple M4 Pro.
  This does not establish full HiGHS model coverage, complete graph capture for
  other CUDA libraries, or IEEE-754 binary64 arithmetic; the GPU path still uses
  CuMetal's reduced-precision FP64 emulation.
- Batched GEMM/TRSM and device-resident pointer tables are already covered; the
  broad former `cublas*Batched` label was stale. The 32-bit signed/unsigned
  `atomic{Add,Exch,Min,Max,CAS,And,Or,Xor,Inc,Dec}_system` surface now lowers
  with system scope and passes a focused GPU/host managed-memory test. NVIDIA's
  unmodified `systemWideAtomics` stress run passes. Arbitrary pageable `malloc`
  pointers remain unsupported as
  kernel arguments; the runtime reports both pageable-memory attributes as 0
  instead of steering applications onto that invalid path.
- **Unified-memory allocation modes** -- `cudaMallocManaged` accepts CUDA's
  global and host attachment modes. On Apple Silicon both map to the same
  permanently CPU/GPU-visible tracked allocation; invalid attachment flags are
  still rejected. NVIDIA's unmodified `UnifiedMemoryPerf` completes all eight
  allocation/copy strategies across its full 4 KiB through 16 MiB size sweep.

The scalar-to-packed-half `__float2half2_rn`, packed `__half2` constructor, and
`__hfma2` surface are implemented. Packed `add.f16x2` and `fma.rn.f16x2` lower
as two-lane half operations; focused lowering tests pass, and NVIDIA's
unmodified `fp16ScalarProduct` produces matching 653349 results through both
its native-operator and intrinsic kernels.
The unmodified sample's numerical comparison passes in the sweep.

`cudaDevAttrMemoryPoolsSupported` is now present and reports the existing
stream-ordered allocation/default-pool subset. NVIDIA's unmodified
`streamOrderedAllocation` million-element, 21-dispatch run passes; release-threshold
caching remains a performance hint rather than allocator-reuse parity.

`cublasSgetrfBatched` / `cublasDgetrfBatched` now factor tracked UMA matrices
from a device-resident pointer table. Focused runtime coverage checks single and
double precision, pivoted and no-pivot forms, singular per-batch `info`, and a
truncated-table negative path while native Metal addresses are enabled. NVIDIA's
unmodified `simpleCUBLAS_LU` 10,000-matrix run passes the sweep.

`cudaDeviceProp::cooperativeLaunch` is reported as **1** for resident grids.
`reductionMultiBlockCG`, `conjugateGradientMultiBlockCG`, and `simpleAWBarrier`
pass through the device-wide barrier path; oversized cooperative grids are
rejected explicitly.

Four samples formerly classified as runtime failures now pass after fixing three
compatibility contracts: legacy-stream event recording enqueues a real marker and
therefore waits for prior blocking-stream host operations (`simpleStreams`); host
cuRAND generators accept ordinary host output and execute synchronously
(`MersenneTwisterGP11213`); and `cusparseSpMV_bufferSize()` reports a usable nonzero
workspace size (`conjugateGradient`, `conjugateGradientUM`).

`simpleAtomicIntrinsics` and `scan` were in that list until two silent defects behind
them were fixed. Both are worth knowing about, because both produced zeros with
`cudaSuccess` reported everywhere the caller looked:

- **Apple's AIR backend cannot lower LLVM's `fence`.** Emitting one crashes the Metal
  compiler service when the pipeline state is created -- `XPC_ERROR_CONNECTION_INTERRUPTED`,
  "after multiple retries" -- long after the metallib has been written and validated.
  The kernel then never runs. This was not only about explicit `__threadfence()`: clang
  plants a membar next to `atomicCAS`, so every CAS-bearing kernel was affected, and
  `atomicInc`/`atomicDec` with it. `membar`/`fence` now lowers to a call to
  `air.atomic.fence(mem_flags, memory_order, scope)`, which is what Metal's own compiler
  emits for `atomic_thread_fence`.
- **Scalar `__shared__` objects were sized as zero bytes.**
  `compute_static_shared_bytes()` counted only `.bN name[N]` array declarations, while
  the layout pass that assigns offsets accepted the full type set. A scalar
  `__shared__ unsigned x`, which clang emits as `.shared .align 4 .u32 name;`, was given
  an offset by one pass and counted as nothing by the other, so the threadgroup
  allocation came out at length 0: every store dropped, every load zero. This is the
  same shape as the `.local` stack-depot bug -- two parsers for one thing, the stricter
  one silently winning.

`functional_cuda_projects_device_sync_primitives` covers both, verified failing against
each unfixed version.

For anyone debugging a kernel that "runs" but returns zeros, every
`cudaLaunchKernel` failure is retained as a pending launch error, including
pre-submission registration/JIT validation and Metal pipeline-creation failures.
`cudaDeviceSynchronize()` returns that error even when no command buffer was
enqueued, and it remains visible to `cudaGetLastError()`. The focused runtime
error test covers both a missing Metal function and an early registered-kernel
failure. `CUMETAL_DEBUG_LAUNCH=1` prints the detailed launch reason, and
`CUMETAL_DEBUG_DUMP_IR_DIR=<dir>` dumps the emitted LLVM IR.

- `%clock`, `%clock64`, and `%globaltimer` use a monotonic device-wide atomic
  counter because public Metal exposes no CUDA-compatible cycle counter. This
  preserves progress/timeout semantics but is not cycle-accurate profiling.
  `clock`, `simpleHyperQ`, and `eigenvalues` pass the unmodified sweep.

Three samples formerly appeared in that list because of missing CUDA libdevice
integer helpers. `scalarProd` needed signed/unsigned 24-bit multiply, which now
masks or sign-extends the low 24 bits before returning the low 32-bit product.
`mergeSort` needed unsigned min/max. Both it and `sortingNetworks` pass their
unmodified key/value validation; merge sort also passes its stability check.

The intermittent `simpleCallback` failure was a cold-cache race, not singleton
teardown. Eight host threads could all observe the same registration-JIT miss and
overwrite one `.metal`/`.metallib` cache path concurrently; a launch that consumed
a partial artifact silently left its workload unchanged. Cache lookup, compilation,
and publication are now serialized. A functional test starts eight simultaneous
first-use launches and requires exactly one miss, and the upstream sample passed
100/100 runs with a distinct empty cache for every run.

The last three of those used to SIGBUS instead, which turned out to be a
memory-safety bug in the runtime rather than anything to do with their kernels.
`__cudaRegisterVar` was registering each `__device__`/`__constant__` variable's
device-side *name string* as its address -- clang and nvcc both pass the name for
both the third and fourth argument -- so `cudaMemcpyToSymbol` memcpy'd the
caller's bytes over a string literal. 27 KB over `"excess_params"` faulted;
smaller writes would have quietly corrupted the binary's own data. Symbols now map
to the host shadow, `cudaGetSymbolAddress` resolves through the same table instead
of disagreeing with `cudaMemcpyToSymbol` about where a symbol lives, and referenced
external constants are flushed into the spec-defined aligned module buffer at
Metal binding 30 before launch. `functional_cuda_projects_constant_symbol` covers
the round trip and a GPU read beyond the 4 KB inline-argument limit. Both the fault
and the silent-corruption case are gated: the test anchors on the compile-time
address of the shadow, because a round trip through a wrongly registered address
is self-consistent and passes every naive check. The unmodified
`LargeKernelParameter` sample now passes both its 4 KB and 32,764-byte cases.
The same module-buffer path lets unmodified `convolutionSeparable` pass with
zero relative L2 error. Runtime-registered writable external `__device__`
globals now use persistent shared Metal buffers rather than per-launch
snapshots. Symbol APIs and kernels resolve the same bytes; two-launch persistence
is gated in `functional_cuda_projects_constant_symbol`, and unmodified
`threadFenceReduction` passes its single-pass 64-block reduction with exact
CPU/GPU agreement. Driver API `cuModuleGetGlobal` and initialized PTX globals
remain separate gaps.

Finding it also exposed that `run_standalone_cu.sh` discarded the harness exit
status (`$(cmd || true)`), so a sample that died on a signal before printing
anything fell through every content check and was reported PASS. Fixed; the
crash-before-output case is the loudest failure there is and it read as green.

See also: spec.md §8, [docs/status.md](status.md), [docs/air-abi.md](air-abi.md).
