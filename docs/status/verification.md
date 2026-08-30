# Verification and performance status

[Status index](../status.md) · [Verification gaps](../known-gaps/verification.md)

## Local suite

The CTest suite combines host unit tests, compiler production checks, runtime
functional tests, Apple-GPU numerical gates, benchmarks, and environment-bound
external workloads. `scripts/ci_report.sh` reports pass/skip/fail separately.

High-value executable gates include:

- 185/185 required Phase 4 functional tests with zero skips on the recorded
  2026-08-29 Apple M4 Pro run;
- 83-entry `cuda-samples` coverage manifest and sweep classifier;
- 27-file four-cell frontend/backend production-metallib matrix;
- numerical PTX instruction sweep with hand-derived oracles;
- runtime/API negative paths and GPU provenance checks;
- Release/shim-off and Debug/shim-on configuration contracts;
- AIR toolchain identity and metallib load/validation checks.

The repository intentionally contains no GitHub Actions workflows. Release,
shim-policy, and Apple-GPU gates are run explicitly on suitable machines; there
is no recurring hosted or self-hosted automation to claim.

## Performance

`cumetal_bench` compares equivalent CuMetal and native Metal kernels using GPU
and wall timing. The selected release set is vector add, SAXPY, STREAM copy,
STREAM triad, and FP32 reduction. The recorded Apple M4 Pro fastest-of-50
results from 2026-08-29 are:

| Kernel | CuMetal/native wall ratio |
| --- | ---: |
| vector add | 1.103x |
| SAXPY | 1.202x |
| STREAM copy | 1.154x |
| STREAM triad | 1.031x |
| reduction | 1.070x |

All five meet the current 2x gate. This closes the specification's named,
reproducible selected-set requirement; it is not a whole-suite performance
bound.

## Evidence records

- [Verified results](../verified-results.md)
- [Apple-GPU execution record](../apple-gpu-execution.md)
- [Correctness audit](../correctness-audit-2026-07-26.md)
- [Testing guide](../testing.md)
