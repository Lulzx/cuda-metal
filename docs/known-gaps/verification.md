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

The pinned LAMMPS `stable_30Sep2026` / Kokkos 5.2.1 GPU gate passed on 2026-10-04.
The unmodified CUDA application compiles and links, and completes the stock
32,000-atom, 100-step LJ benchmark in both full and half neighbour modes on
Apple M4 Pro. Maximum scaled CPU-double error is `5.91e-6`, cross-mode
difference `1.27e-6`, and analytic initial pair-energy error `1.26e-7`, all below
the unchanged `5e-4` tolerance. Launches use generic typed PTX lowering with
workload specializations disabled. FP64 emulation remains reduced precision;
cold half-neighbour pipeline compilation takes minutes, and performance
comparisons remain open. The single-precision CPU reference retains its failed
mode agreement (`0.000578169` maximum scaled difference). This is one pinned
workload, not general LAMMPS compatibility. See the
[LAMMPS experiment](../../demos/lammps/README.md) for the exact scope and pin.

LAMMPS' own force-style unit tests (`test_pair_style`, `test_bond_style`,
`test_angle_style`, `test_dihedral_style`, `test_improper_style`,
`test_fix_timestep`; MOLECULE, MANYBODY and KSPACE packages; `*kokkos*` cases)
pass 716 of 716 tests on the GPU build as of 2026-10-05 (the Kokkos/OpenMP
CPU control passes all 493 Kokkos cases). One of the 716, `KSpaceStyles`
(`test_kspace_styles`), runs no kernels: it excludes the `kk` suffix styles by
design, so it checks the CPU kspace styles in a CuMetal-linked binary and adds
no GPU coverage. Its 20 `*_omp` cases skip because the OPENMP package is not
built. Run the tests serially: two
test processes sharing one `CUMETAL_CACHE_DIR` produce spurious launch and
Thrust failures.

PPPM needs the build with `CUMETAL_LAMMPS_FFT_KOKKOS=CUFFT`
(`scripts/build_lammps_cumetal.sh`). LAMMPS defaults to the KISS FFT, whose
recursive `kf_work` Metal cannot run, so the four PPPM tests are refused in a
KISS build. Under cuFFT they exposed two runtime bugs: a CPU transform
dereferencing a Metal GPU address, and CPU-backed library calls on stream 0
that did not wait for Kokkos' blocking stream (forces off by up to 4e8).
`coul_long`, `coul_table` and `hybrid-scaled` passed once a register joining
the functor's stack copy and the device view (`itype < 13 ? m_params :
params`) was split per address space.

Before the fixes of 2026-10-05, 52 `FixTimestep` cases failed numerically because
loops whose backedge sat inside a branch ran once (every GPU bond list kept only
each atom's first bond).

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
