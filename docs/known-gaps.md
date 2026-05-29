# Known Gaps

- Full AIR metadata reverse-engineering is still in progress.
- `cumetal-air-emitter --mode experimental` emits a CuMetal test container, not Apple's metallib ABI.
- MetalLibraryArchive validation is optional and currently executed through an external bridge command.
- Binary-shim registration supports CuMetal envelopes (`CMTL`), direct PTX images, and basic CUDA
  fatbin PTX images (wrapper and direct blob) with `FatBinary/FatBinary2/FatBinary3` entry points, plus
  host-function and basic symbol-variable mapping, but not full NVCC fatbinary variants.
- Homebrew LLVM (PTX 4.2 default) needs `scripts/cumetal_cuda_flags.sh` feature flags when targeting
  `sm_70+`; the `scripts/cuda_toolchain/fatbinary` shim accepts LLVM 17+ `--image3=...` arguments.
- Warp-primitive intrinsic lowering (`shfl.sync.*`, `vote.sync.{ballot,any,all}`, `bar.warp.sync`)
  is implemented in the `intrinsic_lower` pass (LLVM→AIR path), mapping to
  `air.simdgroup.{shuffle,shuffle_down,shuffle_up,shuffle_xor,ballot,any,all,barrier}`.
  The generic PTX→Metal emitter also now emits Metal `simd_shuffle*`, `simd_ballot`, `simd_any/all`,
  and `simd_sum/and/or/xor/min/max` for `shfl.sync`, `vote.sync`, and `redux.sync` respectively.
  Mask emulation for non-0xFFFFFFFF membermasks is conservative (full-group) rather than
  lane-selective; kernels using partial masks should be tested carefully.
- `llm.c` stress coverage (`conformance_llmc_gpt2fp32cu`) now auto-builds/runs via CuMetal's local
  clang+fatbin shim scripts when an `llm.c` checkout is present, but this path still depends on an external
  checkout with model/debug-state assets.
- End-to-end GPU execution of full `llm.c` CUDA sources through `cumetalc` is not yet implemented:
  current `.cu` frontend lowering is partial and does not support full CUDA language/device-runtime
  semantics required by `train_gpt2_fp32.cu` (cooperative groups, CUDA builtins, and kernel-launch codegen).
- PTX registration path reaches full parity for `llm.c` `test_gpt2_fp32.cu` via direct Metal lowering
  for all 17 GPT-2 training kernels. `CUMETAL_LLMC_REQUIRE_NO_EMULATION=1` now passes without any
  emulation fallback. The lowering path (`compiler/ptx/src/lower_to_metal.cpp`) has two tiers:
  (1) hardcoded name-matched kernels for the 17 llm.c GPT-2 kernels, and (2) a generic
  PTX→Metal instruction-level translator that handles arbitrary element-wise kernels via two-pass
  register-provenance analysis. The generic emitter supports: `ld/st.global`, `atom.global.add.f32`,
  `setp`-based bounds guards, `mad.lo.u32` and `mul.lo.u32`+`add.u32` GID patterns, arithmetic
  (`add`, `sub`, `mul`, `div`, `rem`, `shl`, `shr`, `and`, `or`, `xor`, `not`, `selp`), `fma`/`mad`,
  `neg`/`abs`/`rcp`, `max`/`min`, and the unary math intrinsics `sqrt`, `rsqrt`, `ex2`→`exp2`,
  `lg2`→`log2`, `sin`, `cos`. Kernels with unsupported patterns fall back to PTX→LLVM lowering.
- `--fp64=emulate` mode (Dekker's algorithm FP32-pair decomposition, spec §8.1) is implemented
  for kernels matched by the `fp64_mul_add` pattern (kernels named `*fp64*{mul,fma,add}*`).
  The runtime defaults to `kEmulate` because Apple Silicon GPU hardware rejects native FP64
  arithmetic (`fmul double`, `@llvm.fma.f64`) at Metal pipeline-creation time even though
  `xcrun metal` compiles the LLVM IR successfully; `fpext`/`fptrunc` conversions do work.
  Set `CUMETAL_FP64_MODE=native` to force `kNative` mode (useful for testing the compilation
  path; will fail at launch on current Apple Silicon hardware).  General Dekker decomposition
  for arbitrary PTX `.f64` instruction streams requires a dedicated LLVM pass and is deferred.
- Null stream synchronization (spec §6.3.1) is implemented via command-buffer sequencing on the
  default `MTLCommandQueue` rather than the spec's described `MTLSharedEvent`-based approach.
  The observable CUDA semantics (null-stream serialization) are correct for the common single-context
  use case, but the explicit multi-stream "all user streams wait for null stream" guarantee is not
  fully enforced via events. `MTLSharedEvent` integration is deferred to a future milestone.
- Device printf (spec §5.3): fully implemented end-to-end for the PTX registration path.
  The `printf_lower` compiler pass extracts format strings; the generic Metal emitter injects
  ring-buffer write code; the runtime allocates a 1 MB ring buffer, binds it as hidden args,
  and drains recorded printf records to stderr after kernel completion (spec §6.5 step 10).
  Both the direct `cumetalKernel_t` path and the `__cudaRegisterFunction` registration path
  are covered by functional tests.
