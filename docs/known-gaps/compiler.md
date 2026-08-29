# Compiler and toolchain gaps

[Known-gaps index](../known-gaps.md) · [Compiler status](../status/compiler.md)

## Typed CuMetal IR migration

With CUDA Clang 21-23, the reviewed production-metallib matrix is:

| Frontend | Legacy | Typed CuMetal IR |
| --- | ---: | ---: |
| direct `.cu` | 0/24 | **24/24** |
| PTX / `--cuda-device` | **24/24** | **24/24** |

The manifest is `tests/cuda_projects/backend_matrix_manifest.txt`; the CTest
gate is `conformance_compiler_backend_matrix`. Counts are compilation evidence,
not runtime correctness. `conformance_compiler_backend_matrix_versions` records
and checks the CUDA Clang 21, 22, and 23 identities separately.

Remaining typed-path blockers include combinations of:

- CFG structurization for residual loops, joins, exits, and barrier-containing
  regions;
- compound shared-memory layouts beyond the proven static arrays and single
  runtime-sized `extern __shared__` binding;
- generic pointer provenance through calls, aggregates, memory, and merges;
- atomic scope/order/address-space combinations beyond the numerically proven
  32-bit direct/PTX family and lock-backed 64-bit typed-PTX family;
- PTX call forms beyond the proven FP32 libdevice and constant-format `vprintf`
  parameter-slot ABIs, including device helpers and pointer-bearing or general
  double-signature calls;
- nested or partially initialized aggregate insertion and extraction beyond
  the proven flat typed-struct path;
- FP64 modes and operations beyond the numerically proven `fast48`
  arithmetic/storage/comparison/rounding corpus, including observable IEEE
  exception status;
- native-AOT metadata that lets the runtime populate hidden constant/global
  bindings without the current CUDA registration path, plus unproven symbol
  combinations beyond the directly and transitively referenced buffers covered
  by typed NVVM tests.

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
