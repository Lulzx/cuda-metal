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

- Enrolled headless `cuda-samples`: **83/83 pass**, zero waivers, zero
  nonpassing manifest entries.
- Production-metallib source corpus with CUDA Clang 21-23: direct typed IR
  **23/24**, typed PTX **23/24**, legacy PTX **24/24**.
- Phase 5 measured kernels: vector add, SAXPY, and reduction meet the recorded
  2x native-Metal ceiling on Apple M4 Pro.
- Current production PTX compatibility is broader than the typed path. This is
  why backend defaults remain frontend-dependent.

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
