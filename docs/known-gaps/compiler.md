# Compiler and toolchain gaps

[Known-gaps index](../known-gaps.md) · [Compiler status](../status/compiler.md)

## Typed CuMetal IR migration

The reproducible production-metallib matrix is:

| Frontend | Legacy | Typed CuMetal IR |
| --- | ---: | ---: |
| direct `.cu` | 0/23 | **9/23** |
| PTX / `--cuda-device` | **23/23** | **6/23** |

The manifest is `tests/cuda_projects/backend_matrix_manifest.txt`; the CTest
gate is `conformance_compiler_backend_matrix`. Counts are compilation evidence,
not runtime correctness.

Remaining typed-path blockers include combinations of:

- CFG structurization for residual loops, joins, exits, and barrier-containing
  regions;
- static/dynamic threadgroup declarations and shared-memory ABI;
- generic pointer provenance through calls, aggregates, memory, and merges;
- LLVM atomics including `cmpxchg`, system-scope forms, and wide operations;
- PTX parameter stores/device calls and `vprintf` normalization;
- heterogeneous aggregate/vector insertion and extraction;
- broader libdevice and FP64 legality;
- constant/global address-space emission and Apple MSL legality.

The old direct legacy `.cu` path is textual qualifier stripping and fails this
corpus. It is not a correctness fallback.

## Source AOT architecture

The complete executable flow still uses CUDA-compatible source registration and
first-launch PTX lowering. The spec requires a versioned native registration and
launch-stub AOT path. Do not describe the current executable as having closed
that architectural requirement.

## PTX and fatbinary coverage

PTX support is per instruction form. Direct PTX texture/surface instructions,
TMA/cluster operations, FP8, unrestricted device calls, and other unsupported
forms fail. The binary parser covers bounded raw PTX, CuMetal envelopes, common
fatbin PTX wrappers, and checked little-endian ELF32/ELF64 sections. Compressed
PTX payloads and remaining variants are open; SASS-only and big-endian inputs
are outside the current target.

## AIR and Apple tools

Production output depends on Apple's public Metal compiler. `air_inspect`,
`air_validate`, and direct AIR container generation do not constitute a stable
private AIR compiler. Cross-Xcode evidence is incomplete without genuinely
distinct installations and runtime-load results.
