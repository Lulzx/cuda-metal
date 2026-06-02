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
- Complex CUDA C++ sources (e.g. full `llm.c/train_gpt2_fp32.cu`) are not yet supported
  end-to-end through pure `cumetalc` because the frontend lowering does not yet cover all
  required CUDA language features, cooperative groups grid sync, certain builtins, and
  host-side launch glue. 
- For such workloads the supported path is the binary-shim / PTX registration path
  (`__cudaRegisterFatBinary` etc. or `cuModuleLoad*` with PTX/fatbin), which achieves full
  parity for llm.c (all 17 kernels lowered, `CUMETAL_LLMC_REQUIRE_NO_EMULATION=1` passes)
  and llama.cpp GGML CUDA backend.

## Tooling / build notes
- `air_emitter` "experimental" mode produces test containers, not production metallib ABI (for validation/air_abi only; runtime execution requires real metallib from xcrun or prebuilt).
- AIR metadata validation relies on MetalLibraryArchive + xcrun where available; the
  bridge is optional at build time.
- Homebrew LLVM users targeting sm_70+ need the feature-flag shim
  (`scripts/cumetal_cuda_flags.sh`) because of PTX version defaults; the in-tree
- cuda_projects conformance harness now runs its compile step (clang -x cuda shim + fatbin registration setup) in environments without xcrun metal/metallib (only base xcrun + clang++ needed); runtime exec still limited by PTX lowering coverage for complex kernels (sgemm etc.) and falls back gracefully to SKIP (see run_standalone_cu.sh). This reduces skip-only coverage for the harness itself.
  `scripts/cuda_toolchain/fatbinary` accepts modern `--image3` args.
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
