# Roadmap and closure criteria

[Specification index](../spec.md)

The repository is beyond bootstrap. Work is prioritized by correctness and
closure evidence, not historical phase numbering.

## P0 — Documentation integrity

Keep canonical requirements, current status, gaps, manifests, and measured
results consistent. Long pages must be split behind stable indexes. Historical
snapshots remain dated and must not masquerade as current state.

## P1 — Typed compiler path

- Close LLVM/NVVM and PTX importer gaps, pointer/address-space propagation,
  device calls, atomics, shared memory, FP64, and CFG structurization.
- Reach legacy production-compilation coverage and numerical correctness with no
  fallback.
- Complete the versioned native source registration/launch ABI so source AOT no
  longer depends on NVIDIA registration or first-launch PTX lowering.

Closure: committed corpus plus numerical Apple-GPU tests match or exceed the
legacy path, and a source executable uses the native ABI end to end.

## P1 — Correctness and CI

Define the Phase 4 denominator, preserve all-pass enrolled sample coverage, and
run Release/shim-off plus debug/shim-on lanes. M-series execution must report
pass, skip, and failure separately.

## P1 — AIR/toolchain stability

Validate genuinely distinct Xcode 15.0, 15.4, 16.0, and 16.2+ toolchains where
they remain support targets. Duplicate compiler identities do not count as a
cross-version matrix.

## P2 — Runtime, libraries, and binary compatibility

Expand graphs, dynamic launch, textures/surfaces, printf, atomics, library
semantics, and bounded fatbinary variants with positive and negative tests.
Never weaken source-first defaults or container validation.

## P3 — Performance

Expand native-Metal comparisons beyond the initial vector add, SAXPY, and
reduction measurements. Implement allocator/heap or compiler optimizations only
against measured bottlenecks.

The live, evidence-linked task table is maintained in
`docs/spec-closure-roadmap.md`.
