# Verification and performance status

[Status index](../status.md) · [Verification gaps](../known-gaps/verification.md)

## Local suite

The CTest suite combines host unit tests, compiler production checks, runtime
functional tests, Apple-GPU numerical gates, benchmarks, and environment-bound
external workloads. `scripts/ci_report.sh` reports pass/skip/fail separately.

High-value executable gates include:

- 83-entry `cuda-samples` coverage manifest and sweep classifier;
- 24-file four-cell frontend/backend production-metallib matrix;
- numerical PTX instruction sweep with hand-derived oracles;
- runtime/API negative paths and GPU provenance checks;
- Release/shim-off and Debug/shim-on configuration contracts;
- AIR toolchain identity and metallib load/validation checks.

GitHub Actions definitions are enabled for the hosted Release/shim-off and
Debug/shim-on matrix. Their source presence does not prove a successful remote
run. The Apple-GPU workflow is also enabled, but push execution still requires
a commissioned runner and `CUMETAL_GPU_CI_ENABLED=true`; manual dispatch only
attempts the job.

## Performance

`cumetal_bench` compares equivalent CuMetal and native Metal kernels using GPU
and wall timing. The recorded Apple M4 Pro fastest-of-20 results are:

| Kernel | CuMetal/native wall ratio |
| --- | ---: |
| vector add | 1.063x |
| SAXPY | 1.036x |
| reduction | 1.008x |

All three meet the current 2x gate. They do not prove a whole-suite performance
bound; expanding the selected memory-bound release set remains work.

## Evidence records

- [Verified results](../verified-results.md)
- [Apple-GPU execution record](../apple-gpu-execution.md)
- [Correctness audit](../correctness-audit-2026-07-26.md)
- [Testing guide](../testing.md)
