# Build, verification, and release gates

[Specification index](../spec.md)

## Supported build contract

- macOS 14+ on Apple Silicon.
- CMake 3.28+, C++20, LLVM 18+ libraries, a CUDA-capable Clang 21, 22, or 23
  frontend, and Apple's public Metal tools.
- Release builds keep source registration enabled and the binary alias disabled
  unless `CUMETAL_ENABLE_BINARY_SHIM=ON` is explicitly selected.
- A shim-off build must compile and pass its applicable tests.

## Evidence classes

Keep these claims separate:

1. source/API presence;
2. host unit behavior;
3. successful MSL/metallib production;
4. runtime API success;
5. numerical Apple-GPU correctness with provenance;
6. performance against a stated baseline;
7. downstream unmodified workload success.

A stronger claim cannot be inferred from a weaker class. Skips, waivers, and
unsupported classifications are reported independently from passes.

## Required tests

- Unit tests for parsers, IR verification, transformations, ABI layout, runtime
  state, and every negative path introduced by a behavior change.
- Production compiler corpus tests for each frontend/backend cell.
- The production compiler corpus across every supported CUDA Clang major.
- Functional numerical tests on Apple GPU for accepted compiler/runtime paths.
- Conformance manifests with explicit denominator, expected outcome, and reason.
- Malformed input, stale handle, invalid pointer, overflow, ordering, and
  unsupported-mode tests.
- AIR/metallib validation and runtime load across genuinely distinct supported
  Xcode versions.
- Benchmarks with device/toolchain provenance and native Metal comparison.

## Phase gates

- The Phase 4 conformance denominator must be defined before claiming the 90%
  target. The CUDA-samples enrollment snapshot is useful but not automatically
  that denominator.
- The typed shared-IR path must match or exceed legacy generic correctness before
  replacing a broader default.
- Phase 5 requires selected memory-bound kernels at or below 2x native Metal;
  the selected set must be named and reproducible.
- External workloads such as llm.c, llama.cpp, PhysX, HiGHS, and VF64-metal need
  pinned revisions, exact commands, numerical checks, and device provenance.

## Release gate

Before a release:

- reconcile spec, README, status, gaps, and verified results;
- run Release and shim-off build/test gates;
- run the CUDA sample manifest and compiler matrix where dependencies exist;
- record skipped external/toolchain gates without converting them to passes;
- update version, changelog/release notes, packaging, and install smoke tests;
- publish only from a clean, reviewed commit and verify the remote tag/release.
