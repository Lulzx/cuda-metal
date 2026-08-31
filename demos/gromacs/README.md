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
| PME spread / solve / gather | **Apple GPU** | `pme_spline_and_spread_kernel`, `pme_solve_kernel`, `pme_gather_kernel` |
| PME 3D FFT | **Apple GPU** | VkFFT Metal backend through CuMetal's cuFFT adapter |
| Exclusive scan over the sci histogram | host UMA (CUB shim) | `cub::DeviceScan::ExclusiveSum` |

`-nb gpu -pme gpu -bonded gpu -update gpu`: everything GROMACS can offload, and
the whole PME step is now GPU-resident. Dense, out-of-place, single-precision
3-D R2C/C2R transforms use vendored VkFFT 1.3.4 through its public Metal
backend. CuMetal's project-owned Stockham/Bluestein kernels remain the explicit
fallback for layouts outside that bounded VkFFT path. The one thing still on
the host is the CUB scan of the pair-list histogram.

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

From one `bash demos/gromacs/run.sh`, GROMACS 2025.4, `-nb gpu -pme gpu
-bonded gpu -update gpu`:

| System | Atoms | Max relative energy difference, 20 steps |
| --- | --- | --- |
| villin | 5,006 | 2.66e-05 |
| rnase_cubic | 24,040 | 6.80e-05 |

ADH (134,177 atoms) is wired into `--all` and uses the same gate, but has no
recorded number here — run it and read the table `run.sh` prints rather than
trusting a figure copied into a README.

88 kernel launches are traced with `device=apple_gpu` over a two-step
provenance run, against 28 before the FFT moved to Metal.

Moving the FFT to Metal took the 20-step villin run from 0.63 s to 0.46 s and
rnase_cubic from 8.95 s to 0.57 s, on a warm kernel cache. Both are exactly the
transform arithmetic: villin's FFT is 8.29 ms per step against 0.42 ms, which
over 21 steps is the 0.17 s the run saves. The backend comparison is in
[`docs/verified-results.md`](../../docs/verified-results.md).

Measure this warm. The first run of a GROMACS binary JIT-compiles its kernels
from a 5 MB PTX module, which takes about 56 s and which GROMACS charges to
`Launch PP GPU ops` — enough to swamp anything else being compared.

Step-0 forces were also compared directly, atom by atom, out of the `.trr`, with
the nonbonded kernel on the GPU: **max relative difference 5.1e-05 over 5,006
atoms, none above 1e-3.** That is the level you get from summing the same
interactions in a different order in binary32.

### Reading `ns/day` and comparing backends

`ns/day` is the number of nanoseconds of molecular trajectory GROMACS can
simulate in 24 hours of wall-clock time. Higher is better. For a timestep of
`dt` femtoseconds and a measured latency in milliseconds per step:

```text
ns/day = 86.4 * dt / (ms/step)
```

Every result below uses a 2 fs timestep, so `ns/day = 172.8 / (ms/step)`.
Consequently, lower `ms/step` and higher `ns/day` describe the same performance
win; they are not independent metrics.

The numbers are comparable only when the backend runs the same TPR, step count,
precision, host/toolchain class, and task placement. In particular, CuMetal's
`28.828 ns/day` result is the matched AdaptiveCpp experiment where only
short-range nonbonded work runs on the GPU. It must not be compared with the
MR overview's `32.79 ns/day` native-Metal result, where PME and its FFT also run
on the GPU. With matched full-GPU placement on the reconstructed 98,319-atom
case, CuMetal reaches `46.853 ns/day` against same-host native Metal at
`41.183 ns/day`.

The performance acceptance target is stricter than the currently recorded
cases: after correctness and Apple-GPU provenance pass, CuMetal's warm median
`ms/step` must be lower than both native Metal and AdaptiveCpp for every enrolled
GROMACS case under identical task placement. Conditioning/JIT processes are
reported separately and never mixed into the warm median. Current evidence is:

| Case and matched placement | CuMetal | Comparator | Status |
| --- | ---: | ---: | --- |
| Official 96k water, GPU nonbonded only | 28.828 ns/day | AdaptiveCpp Metal, 3.142 ns/day | **CuMetal 9.18x win** |
| Official 96k water, GPU nonbonded + PME | 63.392 ns/day | Native Metal, 57.797 ns/day | **CuMetal 1.097x win** |
| Reconstructed 98,319-atom water, full GPU | 46.853 ns/day | Native Metal, 41.183 ns/day | **CuMetal 1.138x win** |
| 1,005,375-atom structural stress, full GPU | 5.615 ns/day | Native Metal, 5.274 ns/day | **Provisional CuMetal 1.065x win** |

This does **not** yet prove the all-cases target. AdaptiveCpp generic/Metal still
lacks the GPU-FFT connection needed for a matched full-GPU comparison, and the
complete public benchmark corpus has not yet been run as paired warm series
through all three backends. Those remain explicit work for the proposed
`gromacs-metal` route.

### Official 96k water: CuMetal versus AdaptiveCpp Metal (2026-08-31)

The official
[GROMACS water benchmark suite](https://gromacs-benchmarks-4ed623.gitlab.io/)
ships a 96,000-atom case, authored by Szilárd Páll and Berk Hess under CC BY
4.0. Its unmodified `pme.mdp` uses V-rescale at 300 K and produced a
100×100×52 PME mesh. The same TPR and GROMACS commit
`c7fc4ef64a23f2fe4795d6342af5bcb769d9ca9a` were run on the same M4 Pro for
500×2 fs with:

```text
-ntmpi 1 -ntomp 12 -nb gpu -pme cpu -pmefft cpu -bonded cpu -update cpu -notunepme
```

This task placement is intentional. Current GROMACS main recognizes
AdaptiveCpp's generic/SSCP flow as experimental, and its VkFFT integration
accepts AdaptiveCpp CUDA and HIP targets but not generic/Metal. Therefore both
backends put only short-range nonbonded work on the GPU; PME, FFT, bonded,
update, and constraints stay on the CPU. Comparing this AdaptiveCpp build with
CuMetal's full-GPU PME path would be a different-work comparison.

The AdaptiveCpp build used the patched Metal installation from
`warpx-metal`, reported as AdaptiveCpp
`25.10.0+git.3733a565.20260520.branch.HEAD.dirty`, generic/SSCP, and Clang
20.1.8. CuMetal used its Release GROMACS CUDA build and Clang 23.1.0. Each
backend first ran a conditioning process, excluded because it populated a new
JIT/kernel-cache variant. Three subsequent independent warm processes gave:

| Backend | warm ms/step | median ms/step | median ns/day | relative throughput |
| --- | --- | ---: | ---: | ---: |
| AdaptiveCpp Metal | 55.003 / 55.006 / 54.997 | 55.003 | 3.142 | 1.000× |
| **CuMetal CUDA** | **6.053 / 5.963 / 5.994** | **5.994** | **28.828** | **9.175×** |

CuMetal is **89.1% lower latency** and **9.18× the throughput** of the
experimental AdaptiveCpp Metal route for this matched GPU-nonbonded workload.
The result does not include cold startup: the discarded conditioning processes
were 56.026 ms/step for AdaptiveCpp and 116.078 ms/step for CuMetal, with both
timings containing new kernel compilation. Those numbers are useful startup
evidence, but are not steady-state throughput.

A separate deterministic 20-step version of the same official case disabled
the thermostat and recorded energies every step. `gate.py` compared 147 terms
across 21 steps and passed; the largest relative difference between the two GPU
paths was `7.99e-06` in total energy. Both logs identify the Apple M4 Pro and
confirm that short-range interactions ran on the GPU.

This is not yet a full-GPU AdaptiveCpp comparison. A future `gromacs-metal`
route must connect GROMACS's SYCL FFT abstraction to a Metal-capable FFT,
validate GPU PME numerically, and then rerun the same corpus with matched
nonbonded/PME/FFT placement.

### Native Metal MR !6137 comparison (2026-08-31)

The current [native Apple Metal backend merge request](https://gitlab.com/gromacs/gromacs/-/merge_requests/6137)
was built alongside CuMetal from the exact same GROMACS commit,
`c7fc4ef64a23f2fe4795d6342af5bcb769d9ca9a`. Both were Release builds using
Homebrew clang 23.1.0, four OpenMP threads, fftpack on the CPU, and the same
96,000-atom water TPR on an M4 Pro. The common command line was deliberately
limited to the work both backends implement:

```text
-ntmpi 1 -ntomp 4 -nb gpu -pme gpu -bonded cpu -update cpu -notunepme
```

The native backend used Metal for nonbonded and PME with VkFFT. The CUDA build
used CuMetal for those same tasks, with the dense PME cuFFT plan now dispatched
through the same VkFFT Metal backend. The performance TPR used
`nstcalcenergy = 5000`; the separate correctness TPR below intentionally uses
`nstcalcenergy = 1`. After one conditioning process for each backend, three
independent warm 2,000-step runs produced:

| Backend | ms/step | ns/day |
| --- | ---: | ---: |
| Native Metal MR, run 1 / 2 / 3 | 2.990 / 2.990 / 2.976 | 57.797 / 57.786 / 58.071 |
| CuMetal CUDA, run 1 / 2 / 3 | 2.728 / 2.726 / 2.721 | 63.345 / 63.392 / 63.509 |

The rematched median steady-state result is **2.990 ms/step for native Metal
versus 2.726 ms/step for CuMetal**. CuMetal is **8.8% lower latency** and
**1.097x the throughput** on this bounded case. CuMetal's conditioning process
was 6.641 ms/step because lazy kernel compilation occurred inside the timed
region; native Metal's conditioning process was 2.998 ms/step. Do not merge
cold and warm numbers.

A separate deterministic 20-step TPR (`tcoupl = no`, energies every step) used
the same task mapping. `gate.py` compared 147 energy terms across 21 recorded
steps and passed; the largest relative difference was `7.99e-06` in total
energy. This validates this benchmark path, not all GROMACS inputs or all
features in the still-unmerged native backend.

### MR-overview water boxes

MR !6137's overview also reports two pure-water cases. Its published rows are
reproduced below next to CuMetal measurements, but they are deliberately tagged
by provenance: the overview does not publish its Mac model, complete `.mdp`,
equilibration protocol, or repetition policy, so arithmetic against those rows
is historical context rather than a same-machine speedup claim.

**98,319-atom SPC/E water box (10 nm cube, 84³ PME grid, 500×2 fs)**

| Backend | Provenance/config | ns/day | ms/step | vs reported CPU |
| --- | --- | ---: | ---: | ---: |
| CPU (vDSP FFT) | MR !6137: 12 thread-MPI ranks, `-nb cpu -pme cpu` | 8.77 | 19.72 | 1.00× |
| OpenCL | MR !6137: 1 rank, `-nb gpu -pme gpu -pmefft gpu` | 28.83 | 6.00 | 3.29× |
| Native Metal | MR !6137 overview | 32.79 | 5.27 | 3.74× |
| **CuMetal CUDA** | M4 Pro reconstruction, five paired warm runs | **46.853** | **3.688** | **5.34×*** |

The starred ratio divides by the rounded, externally reported CPU row and is
not a same-host CPU measurement. The reconstructed input was generated with
`gmx solvate -cs spc216.gro -box 10 10 10`, which deterministically produced
32,773 waters / 98,319 atoms. It was energy-minimized, equilibrated for 2 ps at
300 K, and then run for 500×2 fs with V-rescale, an explicit 84³ mesh,
`nstcalcenergy = 5000`, and:

```text
-ntmpi 1 -ntomp 12 -nb gpu -pme gpu -pmefft gpu
```

The exact reconstruction MDPs and command sequence are checked in under
[`mr6137-water/`](mr6137-water/README.md).

The same GROMACS commit and TPR were then run alternately through both Release
backends on the same M4 Pro. One CuMetal conditioning process was excluded;
the following five already-warm paired processes are the defensible comparison:

| Backend | warm ms/step | median ms/step | median ns/day | relative throughput |
| --- | --- | ---: | ---: | ---: |
| Native Metal MR | 4.179 / 4.211 / 4.180 / 4.234 / 4.196 | 4.196 | 41.183 | 1.000× |
| **CuMetal CUDA** | **3.688 / 3.657 / 3.786 / 3.678 / 3.796** | **3.688** | **46.853** | **1.138×** |

CuMetal is therefore **12.1% lower latency** than the native Metal backend on
this exact, same-host case. The result is not obtained by moving work back to
the CPU: CuMetal's log reports PP, PME, update, and constraints on the Apple
GPU, whereas the current native backend reports update and constraints on the
CPU. Increasing CuMetal's bounded command-buffer batch cap from 64 to 256
removes unnecessary commits in that short-kernel PP/update sequence. The cap
remains configurable with `CUMETAL_BATCH_DISPATCHES`.

A 20-step native-versus-CuMetal check on the same equilibrated state recorded
energies every step and passed all 147 term comparisons; the largest relative
difference was `7.73e-06` in total energy. The longer 500-step trajectories are not compared
term-for-term because ordinary binary32 reordering undergoes Lyapunov
amplification; performance and short deterministic correctness remain separate
measurements.

**1,005,375-atom SPC/E water box (21.70 nm cube, 192³ PME grid,
500×2 fs)**

| Backend | Provenance/config | ns/day | ms/step | vs reported CPU |
| --- | --- | ---: | ---: | ---: |
| CPU (vDSP FFT) | MR !6137: 12 thread-MPI ranks, `-nb cpu -pme cpu` | 1.20 | 144.5 | 1.00× |
| OpenCL | MR !6137: 1 rank, `-nb gpu -pme gpu -pmefft gpu` | 2.61 | 66.2 | 2.18× |
| Native Metal | MR !6137 overview | 4.36 | 39.7 | 3.63× |
| **CuMetal CUDA** | M4 Pro structural stress run, three warm runs | **5.615** | **30.772** | **4.68×*** |

The 21.70 nm `gmx solvate` box deterministically gives 335,125 waters, exactly
the overview's 1,005,375 atoms. This large reconstruction has not yet gone
through the small case's equilibration protocol, so it is recorded as a
structural throughput stress run rather than a final physical benchmark. Its
same-host warm medians were 32.766 ms/step for native Metal and 30.772 ms/step
for CuMetal: a bounded **6.1% CuMetal latency win**. A separate 20-step energy
gate passed with a largest relative difference of `5.39e-06`. The starred
4.68× value again uses the MR's rounded external CPU row and is not presented
as a same-host CPU comparison.

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

**A by-value struct kernel parameter had all of its fields aliased onto one
word.** A struct passed to a kernel by value arrives as `.param .align 4 .b8
name[24]`, and PTX reads its fields with `ld.param.<type> [name+offset]`. The
PTX->MSL path declared the whole thing as a single `constant uint&` and parsed
the offset off the parameter name and discarded it, so all six members of
GROMACS's barostat `ScalingMatrix` read the same four bytes -- and a float field
read through a `uint` declaration is its *bit pattern*, so 1.001f became 1.07e9.
Coordinates left the box at the first pressure-coupling step and the potential
went NaN. Nothing warned: the kernel lowered, compiled and launched. The fields
are now bound as words and read at their own offsets, and because optimised
NVPTX keeps floats in `.b32` registers (`ld.param.b32` feeding an `fma.rn.f32`),
the field's type is taken from its uses through the move chain rather than from
the load's own suffix. A struct access the model cannot express is declined
rather than guessed at. This is what `pcoupl` being unexercised was hiding.

**cuFFT was rank-1 only**, so `cufftPlanMany` with `rank = 3` returned
`CUFFT_NOT_SUPPORTED` and `-pme gpu` could not be used at all. It now executes
ranks 1 to 3 for every transform type, together with cuFFT's advanced data
layout — without `inembed`/`onembed` a padded grid cannot even be described, and
GROMACS's real-space mesh is padded on its fastest axis. Multidimensional
transforms are separable, so each runs as a sequence of 1-D transforms one axis
at a time; the previous code multiplied the dimensions together and did a single
flattened transform, which computes a different function entirely. Lengths that
vDSP cannot factor — a PME grid with a factor of 7 on some axis — go through
Bluestein's chirp-z algorithm instead of being rejected, which also removes the
1024-element ceiling the old direct-sum fallback carried.

**Then the FFT itself moved to Metal.** The first implementation used
project-owned Stockham autosort and Bluestein kernels. The production dense 3-D
R2C/C2R path now uses vendored VkFFT 1.3.4, with eager plan creation outside the
timed trajectory and CuMetal stream/resource ordering around each transform.
The Stockham/Bluestein implementation remains a tested fallback rather than a
silent CPU excursion.

The `nvcc` shim also gained `-ccbin`, `-diag-suppress`, `-Xcompiler` quote
stripping, a two-phase compile-and-link (clang's CUDA-mode link mis-parses
Apple's `-lto_library`), `CUDA::cudart_static`, and directory symlinks for
`cub/` and `nvtx3/`. Those are build-plumbing, not semantics.

The MR comparison exposed three more compatibility edges. GROMACS now compiles
some `.cpp` files as CUDA-language translation units, so the generated `nvcc`
shim enables CUDA mode for every recognized source suffix rather than only
`.cu`. Its device identity code also requires the standard 16-byte
`cudaDeviceProp::uuid`, which now matches CuMetal's deterministic Driver API
identity without growing the reserved ABI envelope. Finally, newer nonbonded
kernels call CUB's free `ShuffleIndex<32>` on `float3` and `float4`; CuMetal now
shuffles each 32-bit word of a trivially-copyable aggregate from the same lane,
with numerical Apple-GPU coverage for both exact vector shapes.

---

## Known limits

- **Only single precision is on the GPU.** Metal has no FP64, so the double
  cuFFT entry points keep the CPU implementation. GROMACS's mixed-precision
  build only uses the single ones. Small grids also stay on the CPU, where the
  dispatch cost would exceed the transform; `CUMETAL_FFT_METAL=1` overrides that
  and `CUMETAL_DEBUG_FFT=1` reports which path each transform took.
- **Single rank, single GPU.** `GMX_MPI=OFF`, no domain decomposition, no
  halo exchange, no PME/PP split across ranks.
- **Mixed precision only.** GROMACS's double-precision build is untested here.
- **`run.sh` covers three systems; `sweep.sh` covers the whole set.**
  `bash demos/gromacs/sweep.sh` runs every case in every archive of the
  benchmark collection through the same gate -- 82 cases up to 1.07M atoms,
  including the reaction-field, virtual-site, CHARMM force-switch and
  pressure-coupled variants `run.sh` never touches, and the pure-water systems
  that have no bonded interactions at all. `fetch.sh` downloads the archives;
  the origin throttles per connection, so it fetches byte ranges concurrently
  (~8x). Free-energy perturbation remains unexercised.
- **One case is still outside its noise floor.** Pure water at 768k atoms
  differs from the CPU build by 2.1e-03 at step 0 against a 1.6e-03 floor. It is
  the nonbonded/PME path, not the update: taking the update off the GPU does not
  move it. The same shape shows up as STMV's `Coul. recip.`, which is 2.3x its
  floor and unchanged by either FFT backend but drops 90x with PME on the CPU.
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
