# Current status

CuMetal is an experimental source-first CUDA compatibility stack with real
Apple-GPU execution for covered kernels. This file is an index; detailed status
is split by subsystem.

## Subsystem status

- [Compiler and toolchain](status/compiler.md)
- [Runtime and CUDA APIs](status/runtime.md)
- [Library shims](status/libraries.md)
- [Verification and performance](status/verification.md)
- [Downstream workloads](status/workloads.md)

## Snapshot

- Enrolled headless `cuda-samples`: **83/83 pass**, zero waivers and zero
  nonpassing entries on the 2026-08-30 rerun. The cooperative CG entry requires
  both its device residual and independent host equation error.
- Production-metallib source corpus with CUDA Clang 21-23: direct typed IR
  **30/30**, typed PTX **30/30**, legacy PTX **28/30**.
- Exact in-tree numerical corpus: typed PTX **27/27** and direct native AOT
  **27/27** on Apple M4 Pro with workload specializations disabled.
- The named five-kernel Phase 5 release set—vector add, SAXPY, STREAM copy,
  STREAM triad, and reduction—meets the recorded 2x native-Metal ceiling on
  Apple M4 Pro.
- Backend defaults remain frontend-dependent while numerical coverage continues
  migrating; one reviewed compile-count advantage does not establish broader
  external-workload compatibility.

These numbers describe bounded gates, not a general CUDA compatibility
percentage. See [verified results](verified-results.md) for evidence and
[known gaps](known-gaps.md) for incomplete semantics.

## Evidence policy

A registered test is not a pass, a skip is not compatibility, successful
compilation is not numerical correctness, and a correct value without
`device=apple_gpu` provenance may be a fallback. Use:

```bash
bash scripts/ci_report.sh build --exclude-regex '^bench_'
```

The canonical requirements are in [the specification](../spec.md).
