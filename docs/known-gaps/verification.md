# Verification, CI, and downstream gaps

[Known-gaps index](../known-gaps.md) · [Verification status](../status/verification.md)

## Conformance denominator

The Phase 4 denominator is the reviewed 185-test manifest at
`tests/conformance/phase4_functional_manifest.txt`. Every entry has expected
outcome `pass`; skips and waivers are not removed from the denominator. The
separate NVIDIA `cuda-samples` manifest has 83 enrolled headless samples; all 83 pass.
Both are bounded snapshots, not general CUDA compatibility percentages;
tests and samples outside the enrollments are unclassified.

## CI

Layered GitHub Actions workflows are enabled. The repository still needs observed
successful hosted runs before recurring hosted CI is proven. The Apple-GPU lane
also requires a commissioned trusted runner and repository variable; workflow
source and local passes do not substitute for observing those remote runs.

## Toolchain matrix

Local AIR tests record toolchain identity and reject duplicate identities as
cross-version evidence. Required Xcode 15.0, 15.4, 16.0, and 16.2+ coverage is
not complete until distinct installations produce attributable validation and
runtime-load results.

## External workloads

llm.c, llama.cpp, PhysX, HiGHS, VF64-metal, and other third-party gates depend on
external revisions, assets, models, or build systems. Each result applies only
to its pinned revision and command. Focused success is not whole-project
compatibility.

The pinned VF64-metal integration passes all three CuMetal FP64 modes on the
recorded Apple M4 Pro. In the frozen HiGHS `afiro` comparison, `wide48` and
`ieee64` pass the residual gate; `fast48` reaches Optimal but misses the dual
residual threshold. The `ieee64` HiGHS path is still mixed because the current
cuSPARSE FP64 SpMV path reports reduced precision.

## Performance

Only vector add, SAXPY, and reduction currently form the measured native-Metal
comparison. The spec's every-functional-kernel language is not closed, and the
release set of memory-bound kernels needs an explicit decision before the 2x
target becomes a broad performance claim.
