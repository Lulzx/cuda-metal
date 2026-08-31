# Performance and backend comparisons

[Back to the GROMACS documentation index](../README.md)

## Reading `ns/day`

`ns/day` is the number of nanoseconds of molecular trajectory GROMACS can
simulate in 24 hours of wall-clock time. Higher is better. For a timestep of
`dt` femtoseconds and measured latency in milliseconds per step:

```text
ns/day = 86.4 * dt / (ms/step)
```

Every comparison here uses a 2 fs timestep, so `ns/day = 172.8 / (ms/step)`.
Lower `ms/step` and higher `ns/day` therefore describe the same performance
win; they are not independent metrics.

Results are comparable only when the backend runs the same TPR, step count,
precision, host/toolchain class, and task placement. In particular, CuMetal's
`28.828 ns/day` result is the matched AdaptiveCpp experiment where only
short-range nonbonded work runs on the GPU. It must not be compared with the
MR overview's `32.79 ns/day` native-Metal result, where PME and its FFT also run
on the GPU. With matched full-GPU placement on the reconstructed 98,319-atom
case, CuMetal reaches `46.853 ns/day` against same-host native Metal at
`41.183 ns/day`.

The acceptance target is stricter than the currently recorded cases: after
correctness and Apple-GPU provenance pass, CuMetal's warm median `ms/step` must
be lower than both native Metal and AdaptiveCpp for every enrolled GROMACS case
under identical task placement. Conditioning/JIT processes are reported
separately and never mixed into the warm median.

| Case and matched placement | CuMetal | Comparator | Status |
| --- | ---: | ---: | --- |
| Official 96k water, GPU nonbonded only | 28.828 ns/day | AdaptiveCpp Metal, 3.142 ns/day | **CuMetal 9.18x win** |
| Official 96k water, GPU nonbonded + PME | 63.392 ns/day | Native Metal, 57.797 ns/day | **CuMetal 1.097x win** |
| Reconstructed 98,319-atom water, full GPU | 46.853 ns/day | Native Metal, 41.183 ns/day | **CuMetal 1.138x win** |
| 1,005,375-atom structural stress, full GPU | 5.615 ns/day | Native Metal, 5.274 ns/day | **Provisional CuMetal 1.065x win** |

This does **not** yet prove the all-cases target. AdaptiveCpp generic/Metal
still lacks the GPU-FFT connection needed for a matched full-GPU comparison,
and the complete public benchmark corpus has not yet been run as paired warm
series through all three backends. Those remain explicit work for the proposed
`gromacs-metal` route.

## Demo timing notes

Moving the FFT to Metal took the 20-step villin run from 0.63 s to 0.46 s and
rnase_cubic from 8.95 s to 0.57 s on a warm kernel cache. Villin's FFT changed
from 8.29 ms per step to 0.42 ms, accounting for the 0.17 s saved over 21
steps. The broader backend comparison is also recorded in
[`docs/verified-results.md`](../../../docs/verified-results.md).

Measure warm performance. The first run of a GROMACS binary JIT-compiles its
kernels from a 5 MB PTX module, taking about 56 s; GROMACS charges that time to
`Launch PP GPU ops`, which can swamp the workload being compared.

## Official 96k water: CuMetal versus AdaptiveCpp Metal

Recorded 2026-08-31. The official
[GROMACS water benchmark suite](https://gromacs-benchmarks-4ed623.gitlab.io/)
ships a 96,000-atom case, authored by Szilárd Páll and Berk Hess under CC BY
4.0. Its unmodified `pme.mdp` uses V-rescale at 300 K and produced a
100×100×52 PME mesh. The same TPR and GROMACS commit
`c7fc4ef64a23f2fe4795d6342af5bcb769d9ca9a` were run on the same M4 Pro for
500×2 fs with:

```text
-ntmpi 1 -ntomp 12 -nb gpu -pme cpu -pmefft cpu -bonded cpu -update cpu -notunepme
```

This placement is intentional. Current GROMACS main recognizes AdaptiveCpp's
generic/SSCP flow as experimental, and its VkFFT integration accepts
AdaptiveCpp CUDA and HIP targets but not generic/Metal. Both backends therefore
put only short-range nonbonded work on the GPU; PME, FFT, bonded, update, and
constraints stay on the CPU. Comparing this AdaptiveCpp build with CuMetal's
full-GPU PME path would compare different work.

The AdaptiveCpp build used the patched Metal installation from `warpx-metal`,
reported as AdaptiveCpp
`25.10.0+git.3733a565.20260520.branch.HEAD.dirty`, generic/SSCP, and Clang
20.1.8. CuMetal used its Release GROMACS CUDA build and Clang 23.1.0. Each
backend first ran a conditioning process, excluded because it populated a new
JIT/kernel-cache variant. Three subsequent independent warm processes gave:

| Backend | Warm ms/step | Median ms/step | Median ns/day | Relative throughput |
| --- | --- | ---: | ---: | ---: |
| AdaptiveCpp Metal | 55.003 / 55.006 / 54.997 | 55.003 | 3.142 | 1.000× |
| **CuMetal CUDA** | **6.053 / 5.963 / 5.994** | **5.994** | **28.828** | **9.175×** |

CuMetal is **89.1% lower latency** and **9.18× the throughput** of the
experimental AdaptiveCpp Metal route for this matched GPU-nonbonded workload.
The result excludes cold startup: the discarded conditioning processes were
56.026 ms/step for AdaptiveCpp and 116.078 ms/step for CuMetal, both containing
new kernel compilation. These are startup evidence, not steady-state
throughput.

A deterministic 20-step version of the same official case disabled the
thermostat and recorded energies every step. `gate.py` compared 147 terms
across 21 steps and passed; the largest relative difference between GPU paths
was `7.99e-06` in total energy. Both logs identify the Apple M4 Pro and confirm
that short-range interactions ran on the GPU.

This is not a full-GPU AdaptiveCpp comparison. A future `gromacs-metal` route
must connect GROMACS's SYCL FFT abstraction to a Metal-capable FFT, validate
GPU PME numerically, and rerun the corpus with matched nonbonded/PME/FFT
placement.

## Same-host native Metal MR !6137 comparison

Recorded 2026-08-31. The current
[native Apple Metal backend merge request](https://gitlab.com/gromacs/gromacs/-/merge_requests/6137)
was built alongside CuMetal from the same GROMACS commit,
`c7fc4ef64a23f2fe4795d6342af5bcb769d9ca9a`. Both were Release builds using
Homebrew clang 23.1.0, four OpenMP threads, fftpack on the CPU, and the same
96,000-atom water TPR on an M4 Pro. The command line was limited to work both
backends implement:

```text
-ntmpi 1 -ntomp 4 -nb gpu -pme gpu -bonded cpu -update cpu -notunepme
```

The native backend used Metal for nonbonded and PME with VkFFT. The CUDA build
used CuMetal for the same tasks, with its dense PME cuFFT plan dispatched
through VkFFT's Metal backend. The performance TPR used `nstcalcenergy = 5000`;
the separate correctness TPR used `nstcalcenergy = 1`. After one conditioning
process per backend, three independent warm 2,000-step runs produced:

| Backend | ms/step | ns/day |
| --- | ---: | ---: |
| Native Metal MR, run 1 / 2 / 3 | 2.990 / 2.990 / 2.976 | 57.797 / 57.786 / 58.071 |
| CuMetal CUDA, run 1 / 2 / 3 | 2.728 / 2.726 / 2.721 | 63.345 / 63.392 / 63.509 |

The rematched medians are **2.990 ms/step for native Metal** and **2.726
ms/step for CuMetal**. CuMetal is **8.8% lower latency** and **1.097x the
throughput** on this bounded case. CuMetal's conditioning process was 6.641
ms/step because lazy kernel compilation occurred inside the timed region;
native Metal's was 2.998 ms/step. Cold and warm numbers must not be merged.

A deterministic 20-step TPR used the same task mapping. `gate.py` compared 147
energy terms across 21 steps and passed; the largest relative difference was
`7.99e-06` in total energy. This validates this benchmark path, not all GROMACS
inputs or all features in the still-unmerged native backend.

## MR-overview water boxes

MR !6137's overview reports two pure-water cases. Its published rows appear
below next to CuMetal measurements, tagged by provenance. The overview does not
publish its Mac model, complete `.mdp`, equilibration protocol, or repetition
policy, so arithmetic against those rows is historical context rather than a
same-machine speedup claim.

### 98,319-atom SPC/E water box

10 nm cube, 84³ PME grid, 500×2 fs:

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
300 K, and run for 500×2 fs with V-rescale, an explicit 84³ mesh,
`nstcalcenergy = 5000`, and:

```text
-ntmpi 1 -ntomp 12 -nb gpu -pme gpu -pmefft gpu
```

The exact reconstruction MDPs and commands are in
[`mr6137-water/`](../mr6137-water/README.md).

The same GROMACS commit and TPR were run alternately through both Release
backends on the same M4 Pro. One CuMetal conditioning process was excluded;
five already-warm paired processes gave:

| Backend | Warm ms/step | Median ms/step | Median ns/day | Relative throughput |
| --- | --- | ---: | ---: | ---: |
| Native Metal MR | 4.179 / 4.211 / 4.180 / 4.234 / 4.196 | 4.196 | 41.183 | 1.000× |
| **CuMetal CUDA** | **3.688 / 3.657 / 3.786 / 3.678 / 3.796** | **3.688** | **46.853** | **1.138×** |

CuMetal is **12.1% lower latency** than native Metal on this exact same-host
case. The result is not obtained by moving work back to the CPU: CuMetal's log
reports PP, PME, update, and constraints on the Apple GPU, whereas the current
native backend reports update and constraints on the CPU. Raising CuMetal's
bounded command-buffer batch cap from 64 to 256 removes unnecessary commits in
that short-kernel PP/update sequence. The cap remains configurable with
`CUMETAL_BATCH_DISPATCHES`.

A 20-step native-versus-CuMetal check on the equilibrated state recorded
energies every step and passed all 147 term comparisons; the largest relative
difference was `7.73e-06` in total energy. Longer 500-step trajectories are not
compared term for term because ordinary binary32 reordering undergoes Lyapunov
amplification; performance and short deterministic correctness remain separate
measurements.

### 1,005,375-atom SPC/E water box

21.70 nm cube, 192³ PME grid, 500×2 fs:

| Backend | Provenance/config | ns/day | ms/step | vs reported CPU |
| --- | --- | ---: | ---: | ---: |
| CPU (vDSP FFT) | MR !6137: 12 thread-MPI ranks, `-nb cpu -pme cpu` | 1.20 | 144.5 | 1.00× |
| OpenCL | MR !6137: 1 rank, `-nb gpu -pme gpu -pmefft gpu` | 2.61 | 66.2 | 2.18× |
| Native Metal | MR !6137 overview | 4.36 | 39.7 | 3.63× |
| **CuMetal CUDA** | M4 Pro structural stress run, three warm runs | **5.615** | **30.772** | **4.68×*** |

The 21.70 nm `gmx solvate` box deterministically gives 335,125 waters, exactly
the overview's 1,005,375 atoms. This large reconstruction has not yet gone
through the small case's equilibration protocol, so it is a structural
throughput stress run rather than a final physical benchmark. Its same-host
warm medians were 32.766 ms/step for native Metal and 30.772 ms/step for
CuMetal: a bounded **6.1% CuMetal latency win**. A separate 20-step energy gate
passed with a largest relative difference of `5.39e-06`. The starred 4.68×
value again uses the MR's rounded external CPU row and is not a same-host CPU
comparison.
