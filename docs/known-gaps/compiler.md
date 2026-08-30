# Compiler and toolchain gaps

[Known-gaps index](../known-gaps.md) · [Compiler status](../status/compiler.md)

## Typed CuMetal IR migration

With CUDA Clang 21-23, the reviewed production-metallib matrix is:

| Frontend | Legacy | Typed CuMetal IR |
| --- | ---: | ---: |
| direct `.cu` | 0/29 | **29/29** |
| PTX / `--cuda-device` | **27/29** | **29/29** |

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
  beyond the proven device and shared-memory helper arguments;
- atomic scope/order/address-space combinations beyond the numerically proven
  32-bit direct/PTX family and lock-backed 64-bit typed-PTX family;
- PTX call forms beyond the proven FP32 libdevice, constant-format `vprintf`,
  direct scalar-return/pointer-argument helpers, and flat 12-byte by-value/single
  aggregate-return ABI, including nested/irregular aggregates, multi-result
  signatures, indirect calls, and general double-signature calls;
- aggregate insertion/extraction beyond the bounded NVVM reconstruction limit
  (depth 8, width 16, and 64 scalar leaves), plus irregularly padded nested
  device-call ABIs beyond the proven depth-two 12-byte fixture;
- initialized writable PTX `.global` forms beyond the proven visible numeric
  byte-array registration path; module-private CUDA Clang `__const_$` aggregate
  literals are embedded read-only, while module-private initialized writable
  globals without a host registration symbol fail explicitly;
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

PTX support is per instruction form. Direct PTX texture/surface instructions,
TMA/cluster operations, FP8, unrestricted device calls, and other unsupported
forms fail. The binary parser covers bounded raw PTX, CuMetal envelopes, common
fatbin PTX wrappers, version-`0x0101` LZ4/Zstd-compressed PTX entries, and
checked little-endian ELF32/ELF64 sections. Other entry versions, codecs, and
remaining container variants are open; SASS-only and big-endian inputs are
outside the current target.

## AIR and Apple tools

Production output depends on Apple's public Metal compiler. `air_inspect`,
`air_validate`, and direct AIR container generation do not constitute a stable
private AIR compiler. Cross-Xcode evidence is incomplete without genuinely
distinct installations and runtime-load results.
