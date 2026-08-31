# Setup and correctness

[Back to the GROMACS documentation index](../README.md)

## Build and run

```bash
bash demos/gromacs/run.sh --quick   # villin only
bash demos/gromacs/run.sh           # villin + rnase
bash demos/gromacs/run.sh --all     # adds ADH (134k atoms)
```

`run.sh` calls `scripts/build_gromacs_cumetal.sh`, which fetches the GROMACS
2025.4 release tarball outside the repository and builds it twice from the same
source: once with `-DGMX_GPU=CUDA` against CuMetal's `nvcc` shim, and once with
`-DGMX_GPU=OFF` as the reference. Inputs come from the
[GROMACS benchmark set](https://gromacs-benchmarks-4ed623.gitlab.io/) (CC BY
4.0).

GROMACS is unmodified. The build script's CMake flags are toolchain
workarounds: Apple clang has no OpenMP, libc++ 23 no longer leaks
`std::ptrdiff_t` transitively, and bundled Colvars reads
`__cpp_lib_filesystem` without including `<filesystem>`. None changes a GPU
code path.

```text
$ gmx -version
GROMACS version:     2025.4
GPU support:         CUDA
NBNxM GPU setup:     super-cluster 2x2x2 / cluster 8 (cluster-pair splitting on)
GPU FFT library:     cuFFT
CUDA driver:         12.0
```

## What runs where

| Stage | Where | GROMACS kernels |
| --- | --- | --- |
| Short-range nonbonded (nbnxm) | **Apple GPU** | `nbnxn_kernel_ElecEwQSTab_VdwLJ*_{F,VF}[_prune]_cuda` |
| Pair-list pruning + bucket sci sort | **Apple GPU** | `nbnxn_kernel_prune_cuda`, `nbnxnKernelBucketSciSort` |
| Listed (bonded) forces | **Apple GPU** | `bonded_kernel_gpu<calcVir, calcEner>` |
| Constrained update (LINCS + SETTLE) | **Apple GPU** | `lincs_kernel`, `settle_kernel`, `updateMDLeapfrogSimple` |
| PME spread / solve / gather | **Apple GPU** | `pme_spline_and_spread_kernel`, `pme_solve_kernel`, `pme_gather_kernel` |
| PME 3D FFT | **Apple GPU** | VkFFT Metal backend through CuMetal's cuFFT adapter |
| Exclusive scan over the sci histogram | host UMA (CUB shim) | `cub::DeviceScan::ExclusiveSum` |

With `-nb gpu -pme gpu -bonded gpu -update gpu`, everything GROMACS can
offload is selected and the whole PME step is GPU-resident. Dense,
out-of-place, single-precision 3-D R2C/C2R transforms use vendored VkFFT 1.3.4
through its public Metal backend. CuMetal's project-owned Stockham/Bluestein
kernels remain the explicit fallback for layouts outside that bounded VkFFT
path. The CUB scan of the pair-list histogram remains on the host.

## Correctness gate

Molecular dynamics is chaotic, so a run that merely completes with plausible
numbers is insufficient. The gate performs a step-by-step energy comparison
against the same source built for the CPU.

`run.sh` rewrites each benchmark's `pme.mdp` into a deterministic short
variant: fixed step count, `nstcalcenergy = 1`, `nstlog = 1`, and `tcoupl = no`.
The thermostat is disabled because `v-rescale` draws random numbers that would
make the builds diverge for an unrelated reason. Both builds integrate from
the same `.tpr`, and `gate.py` compares every printed energy term at every
step:

```text
Bond  Angle  Proper Dih.  Per. Imp. Dih.  LJ-14  Coulomb-14
LJ (SR)  Coulomb (SR)  Coul. recip.  Potential
Kinetic En.  Total Energy  Temperature
```

The tolerance is `1e-2 + 2e-4 * |ref|`. Pressure is excluded because it is a
virial estimate that swings by hundreds of bar between neighbouring steps in
a 5,000-atom box; agreeing on it adds nothing the energy terms have not already
shown.

Twenty steps is deliberate. It is long enough for a wrong pair list or stale
force buffer to move the potential by percent—the compatibility defects in
this demo appeared at step 1—and short enough to avoid substantial
single-precision Lyapunov amplification. Set `CUMETAL_GROMACS_STEPS` to extend
the run, but expect honest trajectory divergence after a few hundred steps.

Two provenance checks must also pass:

- The GROMACS log must say the selected tasks were offloaded.
- `CUMETAL_TRACE_GPU=1` must show `device=apple_gpu` launches and no
  `source=approximate_stub`.

## Recorded correctness results

From one M4 Pro Debug/shim-on `bash demos/gromacs/run.sh` with GROMACS 2025.4
and `-nb gpu -pme gpu -bonded gpu -update gpu`:

| System | Atoms | Max relative energy difference, 20 steps |
| --- | ---: | ---: |
| villin | 5,006 | 2.66e-05 |
| rnase_cubic | 24,040 | 6.80e-05 |

ADH (134,177 atoms) is wired into `--all` and uses the same gate, but has no
recorded number here. Run it and use the table printed by `run.sh` rather than
assuming an undocumented result.

The two-step provenance run traces 88 Apple-GPU launches, versus 28 before the
FFT moved to Metal. Step-0 forces were separately compared atom by atom from
the `.trr`, with the nonbonded kernel on the GPU: the maximum relative
difference was **5.1e-05 over 5,006 atoms, with none above 1e-3**.

Performance evidence and the required cold/warm separation are documented in
[Performance and comparisons](performance.md).
