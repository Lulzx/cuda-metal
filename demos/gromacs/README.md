# GROMACS on CuMetal

**A production molecular dynamics engine's CUDA GPU path, running on an Apple
GPU and checked against its own CPU build.**

```bash
bash demos/gromacs/run.sh --quick   # villin only
bash demos/gromacs/run.sh           # villin + rnase
bash demos/gromacs/run.sh --all     # adds ADH (134k atoms)
```

Nothing from GROMACS lives in this tree. `run.sh` calls
`scripts/build_gromacs_cumetal.sh`, which fetches the GROMACS 2025.4 release
tarball outside the repo and builds it twice from that one source: once with
`-DGMX_GPU=CUDA` against CuMetal's `nvcc` shim, once with `-DGMX_GPU=OFF` as the
reference. Inputs come from the
[GROMACS benchmark set](https://gromacs-benchmarks-4ed623.gitlab.io/) (CC BY 4.0).

GROMACS is unmodified. The CMake flags in `build_gromacs_cumetal.sh` are all
toolchain workarounds — Apple clang has no OpenMP, libc++ 23 no longer leaks
`std::ptrdiff_t` transitively, the bundled Colvars reads `__cpp_lib_filesystem`
without including `<filesystem>`. None of them touch a GPU code path.

```
$ gmx -version
GROMACS version:     2025.4
GPU support:         CUDA
NBNxM GPU setup:     super-cluster 2x2x2 / cluster 8 (cluster-pair splitting on)
GPU FFT library:     cuFFT
CUDA driver:         12.0
```

---

## What runs where

| Stage | Where | GROMACS kernels |
| --- | --- | --- |
| Short-range nonbonded (nbnxm) | **Apple GPU** | `nbnxn_kernel_ElecEwQSTab_VdwLJ*_{F,VF}[_prune]_cuda` |
| Pair-list pruning + bucket sci sort | **Apple GPU** | `nbnxn_kernel_prune_cuda`, `nbnxnKernelBucketSciSort` |
| Listed (bonded) forces | **Apple GPU** | `bonded_kernel_gpu<calcVir, calcEner>` |
| Constrained update (LINCS + SETTLE) | **Apple GPU** | `lincs_kernel`, `settle_kernel`, `updateMDLeapfrogSimple` |
| Exclusive scan over the sci histogram | host UMA (CUB shim) | `cub::DeviceScan::ExclusiveSum` |
| PME mesh (3D FFT) | **CPU** | see [Known limits](#known-limits) |

`-nb gpu -bonded gpu -update gpu -pme cpu`. That is a normal GROMACS task split,
not a contrivance: PME is routinely assigned to separate ranks or to the CPU.

---

## The gate

MD is chaotic, so "it ran and printed plausible numbers" proves nothing. The
check is a **step-by-step energy comparison against the same source built for
the CPU**.

`run.sh` rewrites each benchmark's `pme.mdp` into a deterministic short variant:
fixed step count, `nstcalcenergy = 1`, `nstlog = 1`, and `tcoupl = no`. The
thermostat matters — `v-rescale` draws random numbers, so leaving it on would
make the two builds diverge for a reason that has nothing to do with the GPU.
Both builds then integrate the same trajectory from the same `.tpr`, and
`gate.py` compares every energy term GROMACS prints, at every step:

```
Bond  Angle  Proper Dih.  Per. Imp. Dih.  LJ-14  Coulomb-14
LJ (SR)  Coulomb (SR)  Coul. recip.  Potential
Kinetic En.  Total Energy  Temperature
```

Tolerance is `1e-2 + 2e-4 * |ref|`. Pressure is excluded — it is a virial
estimate that swings by hundreds of bar between neighbouring steps in a 5000-atom
box, so agreeing on it adds nothing the energies have not already said.

Twenty steps is deliberate. It is long enough that a wrong pair list or a stale
force buffer has already moved the potential by percent (the bugs below all
showed up at step 1), and short enough that single-precision reordering has not
had time to amplify. Extend it with `CUMETAL_GROMACS_STEPS`; expect honest
Lyapunov divergence past a few hundred steps.

Two more things must hold or the run fails:

- GROMACS's own log must say all three tasks were offloaded. A number that
  matches is not evidence the GPU produced it.
- `CUMETAL_TRACE_GPU=1` must show launches with `device=apple_gpu` and no
  `source=approximate_stub`, so a host fallback cannot pass as GPU execution.

---

## Results (M4 Pro, Debug, shim ON)

From one `bash demos/gromacs/run.sh`, GROMACS 2025.4, `-nb gpu -pme cpu
-bonded gpu -update gpu`:

| System | Atoms | Max relative energy difference, 20 steps |
| --- | --- | --- |
| villin | 5,006 | 2.37e-05 |
| rnase_cubic | 24,040 | 6.40e-05 |

ADH (134,177 atoms) is wired into `--all` and uses the same gate, but has no
recorded number here — run it and read the table `run.sh` prints rather than
trusting a figure copied into a README.

Step-0 forces were also compared directly, atom by atom, out of the `.trr`, with
the nonbonded kernel on the GPU: **max relative difference 5.1e-05 over 5,006
atoms, none above 1e-3.** That is the level you get from summing the same
interactions in a different order in binary32.

---

## What this found

Everything below was a CuMetal defect that GROMACS exposed and the demo's gate
caught. All produced wrong answers or hard failures, none produced a warning.

**`cudaDeviceReset` erased the kernel registry.** CuMetal's implementation
called `registration::clear()`, dropping the tables `__cudaRegisterFatBinary`
builds when the image loads. Those tables are not device-context state — real
CUDA keeps them across a reset and re-loads the modules on the next launch.
GROMACS calls `cudaDeviceReset` at the end of device detection, so *every*
kernel launch for the rest of the run failed. Now only the Metal buffers behind
`__device__` globals are released.

**Host-backed CUB algorithms ignored stream order.** Every `cub::Device*` shim is
a host loop over unified memory, and not one of them synchronized first — they
even discarded the `cudaStream_t` parameter. GROMACS fills `sciHistogram` in the
prune kernel and immediately exclusive-scans it; the scan ran while the kernel
was still in flight and returned a prefix sum of stale memory. The resulting
sorted pair list dropped about a quarter of the interactions: step 0 was correct
(it uses the *un*sorted list), and from step 1 the potential was off by 4%. All
33 entry points now synchronize. Same failure mode as the CPU-backed library
calls fixed earlier, in a shim family that had never been audited for it.

**`cudaDestroyTextureObject(0)` returned an error.** The null texture object is a
no-op to destroy, like freeing a null pointer. GROMACS's PME teardown calls it
on a lookup table that was never populated, and the `InternalError` it threw
aborted the process.

**A zero-parameter kernel could not be launched.** The registered-launch path
rejected `args == nullptr` outright. CUDA allows it when the kernel takes no
parameters — which is exactly GROMACS's `static __global__ void dummy_kernel(){}`
device sanity check, the first kernel it ever launches.

**Missing libdevice entry points**, each of which made a whole kernel
unlowerable rather than degrading: `__nv_rsqrt` (double reciprocal square root,
used by every nbnxm kernel) and the `__nv_float2int_rn` conversion family (used
by the bonded kernels' PBC image index). `__nv_rsqrt` now shares all three FP64
paths with `__nv_sqrt` and composes as `1 / sqrt(x)` in that mode's own
arithmetic. The conversions are table-driven across all three lowering paths,
with two things the obvious implementation gets wrong: the rounding mode in the
name is applied as a real rounding call before the cast rather than dropped, and
the signed variants cast through a signed integer, because the IR result type is
unsigned and casting straight to it turned every negative result into 0. Both
mistakes were caught by `tests/cuda_projects/libdevice`, which now probes
`rsqrt` and all four `__float2int_*` modes over inputs that straddle zero.

The `nvcc` shim also gained `-ccbin`, `-diag-suppress`, `-Xcompiler` quote
stripping, a two-phase compile-and-link (clang's CUDA-mode link mis-parses
Apple's `-lto_library`), `CUDA::cudart_static`, and directory symlinks for
`cub/` and `nvtx3/`. Those are build-plumbing, not semantics.

---

## Known limits

- **PME mesh runs on the CPU.** `-pme gpu` fails at `cufftPlanMany`: CuMetal's
  cuFFT is rank-1 only and GROMACS's PME needs a padded 3D R2C/C2R pair. This is
  a real gap, not a configuration choice — see
  [`docs/known-gaps/libraries.md`](../../docs/known-gaps/libraries.md).
- **Single rank, single GPU.** `GMX_MPI=OFF`, no domain decomposition, no
  halo exchange, no PME/PP split across ranks.
- **Mixed precision only.** GROMACS's double-precision build is untested here.
- **Coverage is these three systems.** They are all AMBER/CHARMM-style
  biomolecular systems with PME electrostatics. Reaction-field (`rf.mdp`),
  virtual sites, free-energy perturbation, and pressure coupling are unexercised.
- **This is a correctness demo, not a performance one.** The build is Debug-side
  CuMetal with fftpack for the CPU mesh, and `nstcalcenergy = 1` forces an
  energy reduction every step. No ns/day number here is worth quoting.

---

## File map

| Path | What |
| --- | --- |
| `run.sh` | fetch, build, run, gate |
| `gate.py` | parses GROMACS energy blocks, compares two logs term by term |
| `../../scripts/build_gromacs_cumetal.sh` | the two builds and why each flag is there |
| `out/` | build logs, `.tpr` inputs, per-run directories, gate output |
