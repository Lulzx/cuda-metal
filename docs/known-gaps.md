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
- FP64: `--fp64=emulate` (Dekker single-double via FP32 pairs, ~44-bit mantissa) is only
  activated for name-matched kernels (`*fp64*{mul,fma,add}*` etc.). Arbitrary `.f64` PTX
  streams fall back to native (which is rejected at Metal pipeline create time on current
  Apple Silicon; runtime forces emulate default). General lowering pass deferred.
- Null stream (legacy default): observable serialization correct via command-buffer ordering
  on default queue. The full spec §6.3.1 cross-stream "user streams wait for null" via
  MTLSharedEvent is not implemented; current approach suffices for single-context use.
- Device printf: fully works for PTX registration + direct paths (256-byte format limit,
  ring buffer, post-launch drain). Reordering vs. CUDA possible (as on real CUDA too).
- Binary-shim fatbinary support: CMTL envelopes, raw PTX, basic FatBinary/FatBinary2/3
  PTX wrappers supported. Full NVCC fatbinary variants, complex symbol layouts, or SASS-only
  images not supported (SASS never was; per spec).

## .cu / cumetalc frontend limitations
- The Clang-based `.cu` → AIR path via `cumetalc` supports many simple kernels and
  samples (vectorAdd etc.).
- Complex CUDA C++ sources (e.g. full `llm.c/train_gpt2_fp32.cu` or GGML's 100+ kernels in
  llama.cpp) exercise only partial coverage: build succeeds (nvcc shim + clang -x cuda +
  fake CUDA toolkit), device init reports "Apple M4 Pro, compute capability 8.0", and
  fatbin/PTX registration succeeds for the kernels present in the objects.
- Execution hits gaps on first non-trivial kernel dispatch:
  - llama.cpp (GGML CUDA): aborts in ggml_cuda_compute_forward (ADD) with cudaErrorInvalidValue
    on templated k_bin_bcast (e.g. `_ZL11k_bin_bcastIXadL_ZL6op_addffEE...`); the metallib
    resolved via registration-jit was an "experimental container" (produced by air_emitter
    fallback when no `xcrun metal` in PATH) which Metal rejects as "Invalid library file".
  - llm.c: aborts with cudaErrorInvalidValue inside train (e.g. around encoder/forward paths)
    even with CUMETAL_DISABLE_LLMC_EMULATION; some of the 17 kernels rely on special cases in
    lower_to_metal.cpp or direct MSL emission, but not all GGML-style or full combinations are
    covered, and JIT/experimental path can still be hit depending on binary registration.
- The binary-shim / PTX reg + lower path (plus special llm.c cases) gets further than pure
  generic emitter. Direct MSL name-matched cases (compiler/ptx/src/lower_to_metal.cpp) now cover
  common GGML kernels used by small models: k_bin_bcast (op_addff/op_mulff + f16 variants),
  rms_norm_f32 (with stride/mul/add support), convert_unary, dequantize_block_q8_0_f16, plus
  passthru stubs for rope_norm/neox, dequant q5_0, k_set_rows to prevent immediate aborts.
  A fast negative filter skips heavy lowering for the bulk of GGML's 1000s of mul_mat_q* / flash
  / other dequants / cpy etc (they hit "registered kernel missing" and GGML typically falls back
  or aborts depending on NGL and op). 
  - NGL=0 (CPU) for SmolLM2-135M etc: harness run_llama_cpp_cumetal.sh completes, PASS, reaches
    generation + timings (text quality limited by tiny model, not CuMetal).
  - NGL>0: exercises .metal + newLibraryWithSource GPU path for covered kernels (rms, bcast
    residuals etc run on Apple GPU); still hits missing for cpy/set_rows/rope variants etc during
    load/decode when forcing many layers, leading to ggml_cuda abort. Use --n-gpu-layers 0 for
    reliable full run on small models; partial NGL may work if the offloaded layers only use
    covered ops. Broader coverage or better GGML fallback integration would be needed for robust
    high-NGL on full GGML CUDA models.

## Tooling / build notes
- `air_emitter` "experimental" mode produces test containers, not production metallib ABI (for validation/air_abi only; runtime execution requires real metallib from xcrun or prebuilt).
- AIR metadata validation relies on MetalLibraryArchive + xcrun where available; the
  bridge is optional at build time.
- Homebrew LLVM users targeting sm_70+ need the feature-flag shim
  (`scripts/cumetal_cuda_flags.sh`) because of PTX version defaults; the in-tree
- cuda_projects conformance harness now runs its compile step (clang -x cuda shim + fatbin registration setup) in environments without xcrun metal/metallib (only base xcrun + clang++ needed); runtime exec still limited by PTX lowering coverage for complex kernels (sgemm etc.) and falls back gracefully to SKIP (see run_standalone_cu.sh). This reduces skip-only coverage for the harness itself.
  `scripts/cuda_toolchain/fatbinary` accepts modern `--image3` args.
- "Bigger project" tries (llama.cpp GGML full CUDA backend, llm.c gpt2 train) via the dedicated
  build_*_cumetal.sh + run_*_cumetal.sh + fake toolkit succeed at compile+link+device init+reg;
  first kernel launch for complex ops fails as described above (experimental metallib or
  uncovered lowering). Direct MSL lowering + runtime newLibraryWithSource was added for the
  common k_bin_bcast<op_addff> (and f16) family to allow elementwise broadcast adds on GPU
  without needing CLI metal tools. Other GGML kernels (mul_mat variants, dequants, flash attn
  tiles, rms_norm etc.) still hit lowering gaps and will report clear "failed to find kernel
  function (lowering not supported)" or experimental hints suggesting n-gpu-layers=0.
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
