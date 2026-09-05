# Compiler and toolchain gaps

[Known-gaps index](../known-gaps.md) · [Compiler status](../status/compiler.md)

## Typed CuMetal IR migration

With CUDA Clang 21-23, the reviewed production-metallib matrix is:

| Frontend | Legacy | Typed CuMetal IR |
| --- | ---: | ---: |
| direct `.cu` | 0/30 | **30/30** |
| PTX / `--cuda-device` | **28/30** | **30/30** |

The manifest is `tests/cuda_projects/backend_matrix_manifest.txt`; the CTest
gate is `conformance_compiler_backend_matrix`. Counts are compilation evidence,
not runtime correctness. `conformance_compiler_backend_matrix_versions` records
and checks the CUDA Clang 21, 22, and 23 identities separately.

Remaining typed-path blockers include combinations of:

- CFG structurization for residual irreducible or non-reconvergent
  barrier-containing regions beyond the proven uniform multi-exit helper;
- compound shared-memory layouts beyond the proven static arrays and single
  runtime-sized `extern __shared__` binding;
- generic pointer provenance through call/aggregate/memory/merge combinations
  beyond the proven device and shared-memory helper arguments and constant-size
  aggregate copies between host-populated device-buffer descriptors;
- atomic scope/order/address-space combinations beyond the numerically proven
  32-bit direct/PTX family, lock-backed 64-bit typed-PTX family, and the
  32-bit float add/sub/exchange family (native `atomic_float` in device
  storage, a bit-pattern CAS loop in threadgroup storage; float min/max and
  CAS remain diagnostics);
- PTX call forms beyond the proven FP32 libdevice, constant-format `vprintf`,
  direct scalar-return/pointer-argument helpers, and flat 12-byte by-value/single
  aggregate-return ABI. The generic registration-JIT path additionally proves
  the bounded `cuda-samples/newdelete` virtual dispatch with one aligned
  16-byte by-value argument; nested/irregular aggregates, multi-result
  signatures, general indirect calls, and general double-signature calls remain
  open;
- aggregate insertion/extraction beyond the bounded NVVM reconstruction limit
  (depth 8, width 16, and 64 scalar leaves), plus irregularly padded nested
  device-call ABIs beyond the proven depth-two 12-byte fixture;
- initialized writable PTX `.global` forms beyond the proven visible numeric
  byte-array and translation-unit-private integer-scalar paths; module-private
  CUDA Clang `__const_$` aggregate literals and other write-free initialized
  byte arrays are embedded read-only, while unsupported initializer types fail
  explicitly;
- FP64 modes and operations beyond the numerically proven `fast48`
  arithmetic/storage/comparison/rounding corpus, including observable IEEE
  exception status;
- native-AOT symbol combinations beyond the constant/writable multi-kernel
  paths covered by typed NVVM and linked-executable tests.

The old direct legacy `.cu` path is textual qualifier stripping and fails this
corpus. It is not a correctness fallback.

The exact 27-project in-tree numerical corpus passes both typed PTX and direct
native AOT on Apple M4 Pro with workload specializations disabled. This closes
the reviewed corpus, not the residual combinations listed above.

## Source AOT architecture

The linked source flow uses native ABI version 3 with an embedded metallib and
no first-launch PTX lowering. Its descriptor carries per-kernel constant and
writable-global bindings plus the device-`printf` format table; focused tests
cover host symbol copies, constant offsets, persistent GPU writes across
launches, and exact 32-lane formatted output. ABI versions other than 3 are
rejected explicitly.

## PTX and fatbinary coverage

PTX support is per instruction form. Direct PTX indirect-object
`txq`/`suq` width, height, and depth queries are numerically tested; remaining
texture/surface forms, TMA/cluster operations, FP8, unrestricted device calls,
and other unsupported forms fail. The binary parser covers bounded raw PTX, CuMetal envelopes, common
fatbin PTX wrappers, version-`0x0101` LZ4/Zstd-compressed PTX entries, and
checked little-endian ELF32/ELF64 sections. Plausible framed entries with an
unknown kind cannot fall through to the legacy raw-PTX scanner or the
registration environment fallback. Other entry versions, codecs, and remaining
container variants are open; SASS-only and big-endian inputs are outside the
current target.

## Threadgroup float atomics

Metal has no threadgroup float atomic in any language version. CuMetal expands a
32-bit threadgroup float add into a compare-and-swap loop over the word's bit
pattern -- `cm_atomic_fadd_threadgroup` on the MSL path,
`air.atomic.local.cmpxchg.weak.i32` on the LLVM path -- which is correct but
serializes contending threads rather than using hardware. Other threadgroup
float operations (min, max, exchange, compare-and-swap) are refused with a
diagnostic. Device float add, subtract and exchange use Metal's native
`atomic_float`.

## Runtime compilation (NVRTC)

`nvrtcCompileProgram` compiles by spawning `cumetalc`, so runtime compilation
needs the compiler binary on disk and Xcode's Metal toolchain; a caller that
ships only `libcumetal.dylib` gets `NVRTC_ERROR_BUILTIN_OPERATION_FAILURE` with
the reason in the program log. Output is a metallib: `nvrtcGetPTX` and
`nvrtcGetLTOIR` fail, and a `compute_XX` architecture request is rejected at
compile time rather than served with bytes the caller would mis-handle. There is
no `-dlto` path. `nvrtcGetLoweredName` answers only for `extern "C"` entry
points, whose lowered name is the expression itself; template and namespace
expressions return `NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID` because the shim does
not recover the device compiler's mangling. `nvPTXCompiler` is a pass-through,
not a PTX compiler: it returns its input, which the module loader then compiles.

## AIR and Apple tools

Production output depends on Apple's public Metal compiler. `air_inspect`,
`air_validate`, and direct AIR container generation do not constitute a stable
private AIR compiler. Cross-Xcode evidence is incomplete without genuinely
distinct installations and runtime-load results.
