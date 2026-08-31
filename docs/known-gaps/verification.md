# Verification, CI, and downstream gaps

[Known-gaps index](../known-gaps.md) · [Verification status](../status/verification.md)

## Conformance denominator

The Phase 4 denominator is the reviewed 185-test manifest at
`tests/conformance/phase4_functional_manifest.txt`. Every entry has expected
outcome `pass`; skips and waivers are not removed from the denominator. The
recorded 2026-08-29 Apple M4 Pro run passed all 185 with zero skips. The
separate NVIDIA `cuda-samples` manifest has 83 enrolled headless samples; all 83 pass.
Both are bounded snapshots, not general CUDA compatibility percentages;
tests and samples outside the enrollments are unclassified.

## CI

The repository intentionally contains no GitHub Actions workflows. Therefore it
does not provide recurring hosted or self-hosted CI. Local and commissioned-host
results must be recorded with their configuration and cannot be generalized to
an automated schedule.

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

The recorded GROMACS native-Metal comparison is pinned to MR !6137 commit
`c7fc4ef64a23f2fe4795d6342af5bcb769d9ca9a`, one 96,000-atom water input, and a
common GPU-nonbonded/PME task mix. Rematched warm medians are 2.726 ms/step for
CuMetal and 2.990 ms/step for native Metal (8.8% lower CuMetal latency). That is
evidence for this configuration only, not a general ratio between CuMetal and
native Metal. The performance and every-step-energy correctness TPRs are kept
separate.

The recorded AdaptiveCpp comparison is separately pinned to the official
96,000-atom water corpus and the same GROMACS commit. It compares only GPU
nonbonded work: PME, FFT, bonded, update, and constraints run on the CPU for
both backends because GROMACS main does not support a GPU FFT for AdaptiveCpp's
generic/Metal target. Its 9.18x warm CuMetal throughput ratio is not evidence
for full-GPU SYCL/Metal performance or for other benchmark cases. The two
GROMACS builds also necessarily used different host Clang versions (20.1.8 for
AdaptiveCpp and 23.1.0 for CuMetal), which is recorded with the result.

These measured wins are not an all-cases performance guarantee. `ns/day` is
derived from `ms/step` and the simulation timestep; rows with different TPRs or
GPU/CPU task placement cannot be ranked against one another. Closing the
GROMACS performance target requires correctness and device-provenance gates
followed by paired warm medians against both native Metal and AdaptiveCpp on
every enrolled case. A Metal-capable AdaptiveCpp FFT/PME path and that complete
paired corpus are still missing.

The pinned VF64-metal integration passes all three CuMetal FP64 modes on the
recorded Apple M4 Pro. In the frozen HiGHS `afiro` comparison, `wide48` and
`ieee64` pass the residual gate; `fast48` reaches Optimal but misses the dual
residual threshold. The `ieee64` HiGHS path is still mixed because the current
cuSPARSE FP64 SpMV path reports reduced precision.

## Performance

The named Phase 5 release set is vector add, SAXPY, STREAM copy, STREAM triad,
and FP32 reduction. All five pass the reproducible 2x native-Metal gate on the
recorded Apple M4 Pro system. This selected set satisfies the specification's
Phase 5 criterion; it does not establish a whole-suite performance bound.
