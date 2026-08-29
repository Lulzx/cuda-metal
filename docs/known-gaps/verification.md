# Verification, CI, and downstream gaps

[Known-gaps index](../known-gaps.md) · [Verification status](../status/verification.md)

## Conformance denominator

The current NVIDIA `cuda-samples` manifest has 83 enrolled headless samples and
all 83 pass. This is an all-pass bounded snapshot, not the spec's undefined
"full conformance suite" denominator and not a general compatibility percentage.
Samples outside the enrollment are unclassified.

## CI

Layered GitHub Actions workflows exist but are disabled. Therefore the repository
does not currently have proven recurring hosted or self-hosted CI. The Apple-GPU
lane also requires a commissioned trusted runner and repository variable. Local
passes do not substitute for observing those remote schedules.

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

## Packaging

The published Homebrew tap currently references the older `cuda-metal` v0.1.3
archive and version test. It must be updated only after a new verified CuMetal
release exists, then audited and install-tested from the published bottle/source
formula.
