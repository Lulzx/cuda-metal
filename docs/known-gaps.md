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
- Warp primitives with partial masks (`mask != 0xFFFFFFFF`): conservative full-SIMD-group
  emulation (all lanes participate or get identity). Correct but may not match NVIDIA
  "inactive lane" semantics exactly. Kernels relying on partial masks should be validated.
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
- PhysX 5.6 reduced GRB coverage is limited to the 83-kernel sphere/plane PGS
  manifest and a single rigid/static, normal-only resting contact. The CuMetal
  patch serializes several partial-warp scans and omits friction-patch
  correlation, joints, articulations, multi-body batching, and user impulse
  limits. General falling-contact and chaotic long-run solver conformance are
  not claimed. The 30-step resting-contact gate matches CPU transforms within
  `1e-3` relative tolerance.
- PhysX's warp-swizzled `preIntegration` path requires exact partial-mask
  semantics. Patch 0005 selects a body-per-thread equivalent under
  `PX_CUMETAL`; upstream CUDA builds retain the original kernel.

## .cu / cumetalc frontend limitations
- `cumetalc --cuda-device` is the real source frontend for project-scale CUDA:
  it requires a CUDA-capable Homebrew LLVM Clang (or
  `--cuda-clang`/`CUMETAL_CUDA_CLANG`) and forwards `-I`, `-D`,
  `--cuda-include`, and `--cuda-arch`. It deliberately uses
  `-fno-jump-tables`; `brx.idx`/`.branchtargets` remain unsupported in the PTX
  lowering path. CUDA source compilation can therefore succeed while later
  strict PTX lowering still rejects an unimplemented opcode or libdevice call.
  Standalone PTX `.func` bodies are not lowered; projects can request aggressive
  device inlining with `--cuda-inline-threshold`. The reduced PhysX rigid-body
  subset uses this to inline `updateCacheAndBound`.
- The older `.cu` mode without `--cuda-device` remains a qualifier-stripping
  host-LLVM prototype suitable only for simple patterns; it is not a general
  CUDA frontend.
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
  2026-07-20 on SmolLM2-135M-Instruct-Q4_K_M, greedy decode of
  "The capital of France is":
  - Stock CPU llama.cpp (no CuMetal): `Paris.` ✅
  - llama.cpp linked against libcumetal, **NGL=0**: `Paris.` ✅
  - llama.cpp linked against libcumetal, **NGL=1**: `The capital of France is
    Paris.` ✅ at 8.4 tokens/s generation on Apple M4 Pro. Registered launches
    use the correctness-first synchronization policy described above; enabling
    experimental asynchronous registered launches reproduces incoherent output.
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
- The emitter + validate + runtime loading work for all supported paths and pass
  the AIR ABI matrix tests where Xcode toolchains are present.
- "Full" metadata RE is effectively complete for the kernels we emit; unknown future
  ABI changes will be caught by the xcode regression harness.

See also: spec.md §8, [docs/status.md](status.md), [docs/air-abi.md](air-abi.md).
