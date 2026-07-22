# Known Gaps

**Note:** This document tracks divergences from the v1 spec (spec.md) and areas of partial coverage.
See [docs/status.md](status.md) for comprehensive implemented coverage (post-Phase 5, full
library shims, llama.cpp/llm.c conformance via PTX path, etc.). Many items formerly listed here
as gaps have been closed.

## Intentional non-goals (spec §2.2, deferred to v2+)
- Dynamic parallelism (kernels launching kernels) — compile-time error.
- CUDA graphics interop (OpenGL/Vulkan/DirectX).
- Multi-GPU / peer-to-peer (Apple Silicon is single-GPU die).
- Full texture/surface object GPU sampling (lifecycle + array memcpy supported; device-side
  `tex.*` / `suld.*` etc. error at compile for PTX path; see intrinsic-map.md).
- MLIR GPU-dialect kernel fusion / advanced scheduling (optional Phase 5 path not taken).

## Partial / conservative implementations
- Masked `__syncwarp(mask != 0xFFFFFFFF)` lowers to an AIR SIMD-group barrier with
  threadgroup-memory visibility. AIR does not consume CUDA's explicit member mask, so
  additional currently active lanes can receive stronger ordering. Divergent lower/upper
  half-warp ordering is GPU-tested, including static shared-memory visibility. Partial-mask
  ballot/any/all intersect the real AIR active-lane ballot with the CUDA member mask,
  `activemask` returns the real active lanes, and shuffle callers outside the member mask
  receive identity (their CUDA result is undefined). Broader irregular-mask coverage remains
  incomplete.
- Grid-wide cooperative sync (`this_grid().sync()`): a no-op on Metal (no cross-threadgroup
  barrier). A multi-block `cudaLaunchCooperativeKernel` / `cuLaunchCooperativeKernel` now prints
  a one-time `CUMETAL WARNING` so code that depends on grid-wide sync for correctness is not
  silently wrong. Single-block launches are safe (block-scoped CG works).
- FP64: `--fp64=emulate` (Dekker single-double via FP32 pairs, ~44-bit mantissa) is only
  activated for name-matched kernels (`*fp64*{mul,fma,add}*` etc.). Arbitrary `.f64` PTX
  streams fall back to native (which is rejected at Metal pipeline create time on current
  Apple Silicon; runtime forces emulate default). General lowering pass deferred. When a
  driver-JIT kernel actually contains `.f64` ops under the emulate default, the runtime prints a
  one-time `CUMETAL WARNING` noting the reduced (~44-bit) precision; `CUMETAL_FP64_MODE=native`
  compiles true doubles (which fail at launch on current hardware, useful only for testing).
- Null stream (legacy default): observable serialization correct via command-buffer ordering
  on default queue. The full spec §6.3.1 cross-stream "user streams wait for null" via
  MTLSharedEvent is not implemented; current approach suffices for single-context use.
- Registered fatbinary launches are synchronized before returning by default. This
  correctness-first policy avoids stale reads when frameworks use multiple CUDA
  streams over adjacent suballocations of one large Metal buffer. Direct/source-first
  launches remain asynchronous. `CUMETAL_ENABLE_ASYNC_REGISTERED_LAUNCH=1` restores
  the experimental asynchronous registration path, which is known to corrupt
  llama.cpp inference until cross-command-queue resource fencing is complete.
- Device printf: fully works for PTX registration + direct paths (256-byte format limit,
  ring buffer, post-launch drain). Reordering vs. CUDA possible (as on real CUDA too).
- Binary-shim fatbinary support: CMTL envelopes, raw PTX, basic FatBinary/FatBinary2/3
  PTX wrappers supported. Full NVCC fatbinary variants, complex symbol layouts, or SASS-only
  images not supported (SASS never was; per spec).
- PhysX 5.6 reduced GRB coverage is limited to the 84-kernel selected-shape PGS
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
  forcing viable CUDA device calls to inline. General convex meshes, general
  convex/convex pairs, triangle meshes, heightfields, and SDF collisions remain
  unsupported or unverified.
  The first upstream GJK/EPA convex/convex stage executes on Apple GPU. The
  canonical non-inline CUDA/NVVM path now imports the second stage and its
  reachable device-call closure, lowers its noncanonical CFG through typed MSL,
  and preserves its constant lookup tables. Legalization then rejects a real
  CUDA-generic PHI that selects either threadgroup scratch or device memory:
  Metal requires one static address space, so address-space monomorphization
  plus mixed-PHI dispatch is still required. Static threadgroup globals and one
  FP64 calculation remain subsequent MSL blockers. The older 624 KB
  forced-inline form can still abort Apple's pipeline compiler. Convex/convex
  runtime support is therefore not yet claimed.

## .cu / cumetalc frontend limitations
- `cumetalc --cuda-device` is the real source frontend for project-scale CUDA:
  it requires a CUDA-capable Homebrew LLVM Clang (or
  `--cuda-clang`/`CUMETAL_CUDA_CLANG`) and forwards `-I`, `-D`,
  `--cuda-include`, and `--cuda-arch`. It deliberately uses
  `-fno-jump-tables`; `brx.idx`/`.branchtargets` remain unsupported in the PTX
  lowering path. CUDA source compilation can therefore succeed while later
  strict PTX lowering still rejects an unimplemented opcode or libdevice call.
  Standalone PTX `.func` bodies are not lowered; projects can request aggressive
  device inlining with `--cuda-inline-threshold`, which also forces every viable
  reachable device call to inline. Recursion, indirect calls, and explicitly
  non-inlineable helpers remain unsupported. The reduced PhysX rigid-body subset
  uses this for helpers including `updateCacheAndBound` and
  `getIncidentPolygon4`.
- The older `.cu` mode without `--cuda-device` remains a qualifier-stripping
  host-LLVM prototype suitable only for simple patterns; it is not a general
  CUDA frontend.
- The verified `--backend=cumetal-ir` path is not yet the default. It now
  supports selected-entry device-call closures, structured and dispatcher-based
  CFG lowering, loop-carried PHIs, CUDA vector and named aggregate values,
  thread-local allocas, constant global tables, warp shuffle/vote operations,
  transitive Metal builtin threading, and the CUDA math/bit intrinsics exercised
  by the current PhysX convex path. Mixed CUDA-generic pointers whose runtime
  value can name different Metal address spaces, static/dynamic shared-memory
  emission, atomics, reductions, full FP64 handling, and the remaining
  intrinsic surface still fail explicitly.
- Stock Clang CUDA device IR import requires LLVM 18 or newer at build time.
  Unknown NVVM intrinsics, arbitrary pointer/integer round trips, indirect
  calls, recursion, unsupported atomics, and irreducible/unsupported CFG shapes
  are rejected. The new backend never falls back to qualifier stripping,
  legacy PTX lowering, substitutions, or CPU execution.
- The CuMetal-native registration ABI and runtime lookup path are implemented
  and versioned. Automated host-job rewriting, generated launch stubs, and
  embedding descriptors/metallib bytes into the final host object are not yet
  wired into `cumetalc`; source applications must not yet assume the new backend
  produces a complete linked executable.
- The Clang-based `.cu`/PTX registration path supports many simple kernels and
  samples (vectorAdd etc.) and dispatches them through Metal on the Apple GPU.
  CUDA kernel CPU emulation is disabled by default. The legacy llm.c host
  implementation is diagnostic-only and requires
  `CUMETAL_ENABLE_LLMC_CPU_EMULATION=1`; GGML's host helper fallback similarly
  requires `CUMETAL_ENABLE_HOST_KERNEL_FALLBACKS=1`. Both modes emit a warning.
  `CUMETAL_TRACE_GPU=1` provides positive dispatch evidence.
- Complex CUDA C++ sources exercise mixed coverage. The strict llm.c GPT-2 FP32
  conformance workload passes numerical parity on Apple M4 Pro with CPU emulation
  disabled, using specialized MSL replacements. llama.cpp's much broader GGML
  CUDA kernel set remains incomplete.
- The binary-shim / PTX reg + lower path (plus special llm.c cases) gets further than pure
  generic emitter. Direct MSL name-matched cases (compiler/ptx/src/lower_to_metal.cpp) now cover
  common GGML kernels used by small models: k_bin_bcast (op_addff/op_mulff + f16 variants),
  rms_norm_f32 (with stride/mul/add support), and Q8_0-to-f16 dequantization.
  A fast negative filter skips heavy lowering for the bulk of GGML's 1000s of mul_mat_q* / flash
  / other dequants / cpy etc (they hit "registered kernel missing" and GGML typically falls back
  or aborts depending on NGL and op).
- **Approximate/passthru stubs are refused by default (no silent wrong answers).** A handful of
  templates (`convert_unary`, `rope_norm`/`rope_neox`, `dequantize_q5_0`/`_block_q5`,
  `k_set_rows`, `cpy_`/`k_cpy`) exist only as passthru placeholders — they copy or zero data
  instead of computing the real quantized/rotary/copy result. Their output is numerically wrong,
  so the runtime **skips them by default** (the kernel falls through to the same clean "registered
  kernel missing metallib" abort as any unsupported op) and prints a one-time
  `CUMETAL WARNING: kernel '…' has only an approximate/passthru lowering and was skipped …`.
  Set `CUMETAL_ENABLE_APPROX_KERNELS=1` to run them anyway for experimentation — the run then
  completes but the output is not correct, and a warning says so. This trades "it launches but
  lies" for "it fails loudly," which is the safer default for a translation layer.
- **The covered llama.cpp SmolLM2 smoke path is numerically coherent.** Rechecked
  2026-07-22 on SmolLM2-135M-Instruct-Q4_K_M, greedy decode of
  "The capital of France is":
  - Stock CPU llama.cpp (no CuMetal): `Paris.` ✅
  - llama.cpp linked against libcumetal, **NGL=0**: `Paris.` ✅
  - llama.cpp linked against libcumetal, **NGL=1**: `The capital of France is
    Paris.` ✅ at 279.2 tokens/s median generation across five warm runs on
    Apple M4 Pro (223.2–307.7 tokens/s observed). Registered launches
    use the correctness-first synchronization policy described above; enabling
    experimental asynchronous registered launches reproduces incoherent output.
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
  - This is a focused NGL=1 smoke result, not a claim that arbitrary models or
    high layer-offload counts are supported. Broader GGML kernel coverage is
    still required for robust high-NGL inference.

## Tooling / build notes
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
  rotary operations) still hit lowering gaps or are refused placeholder paths.
  See the bin_bcast special case in compiler/ptx/src/lower_to_metal.cpp and Metal source path
  in runtime/metal_backend.
- Full AIR ABI reverse-engineering continues to be refined as Xcode releases change
  undocumented fields (regression tests in `tests/air_abi/` + `air_validate` catch breaks).

## External dependency for full stress conformance
- `conformance_llmc_gpt2fp32cu` and llama.cpp tests require external source checkouts
  (`../llm.c` or `CUMETAL_LLMC_DIR`, similarly for llama.cpp) + model assets. They
  auto-skip (77) when absent. When present they exercise real production kernels.

## AIR / metallib
- Production compilation uses typed MSL and Apple's `metal`/`metallib` tools.
  The emitter + validate + runtime loading continue to serve legacy paths and
  AIR ABI regression tests where Xcode toolchains are present.
- "Full" metadata RE is effectively complete for the kernels we emit; unknown future
  ABI changes will be caught by the xcode regression harness.

See also: spec.md §8, [docs/status.md](status.md), [docs/air-abi.md](air-abi.md).
