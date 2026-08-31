# GROMACS on CuMetal

GROMACS 2025.4's CUDA GPU path runs unmodified on an Apple GPU through
CuMetal. This directory contains the reproducible correctness gate, benchmark
drivers, and an indexed record of the current results.

```bash
bash demos/gromacs/run.sh --quick   # villin only
bash demos/gromacs/run.sh           # villin + rnase
bash demos/gromacs/run.sh --all     # adds ADH (134k atoms)
```

Nothing from GROMACS is vendored here. The scripts fetch the release source and
the [GROMACS benchmark set](https://gromacs-benchmarks-4ed623.gitlab.io/), then
build matching CuMetal CUDA and CPU-reference binaries.

## Documentation index

| Document | Use it for |
| --- | --- |
| [Setup and correctness](docs/validation.md) | What runs on the Apple GPU, how to run the demo, and what the energy/provenance gate proves |
| [Performance and comparisons](docs/performance.md) | `ns/day`, warm-run methodology, AdaptiveCpp and native Metal tables, and MR !6137 water results |
| [Compatibility findings](docs/compatibility-findings.md) | Runtime, compiler, cuFFT, CUB, texture, and toolchain defects exposed by GROMACS |
| [Known limits](docs/known-limits.md) | Precision, rank/GPU, benchmark coverage, and the remaining numerical outlier |
| [MR !6137 water reconstruction](mr6137-water/README.md) | Exact MDPs and commands for the 98,319- and 1,005,375-atom cases |

## Current comparison snapshot

All rows below are warm medians. Only rows with matched inputs and task
placement support a backend speedup claim; see the
[methodology and full run series](docs/performance.md).

| Case and matched placement | CuMetal | Comparator | Result |
| --- | ---: | ---: | --- |
| Official 96k water, GPU nonbonded only | 28.828 ns/day | AdaptiveCpp Metal, 3.142 ns/day | **9.18x throughput** |
| Official 96k water, GPU nonbonded + PME | 63.392 ns/day | Native Metal, 57.797 ns/day | **1.097x throughput** |
| Reconstructed 98,319-atom water, full GPU | 46.853 ns/day | Native Metal, 41.183 ns/day | **1.138x throughput** |
| 1,005,375-atom structural stress, full GPU | 5.615 ns/day | Native Metal, 5.274 ns/day | **Provisional 1.065x throughput** |

The all-cases target is not yet proven. AdaptiveCpp's generic/Metal path still
lacks matched GPU FFT placement, and the complete public corpus has not been
run as paired warm series through all three backends.

## File index

| Path | Purpose |
| --- | --- |
| [`run.sh`](run.sh) | Fetch, build, run, and gate the primary systems |
| [`sweep.sh`](sweep.sh) | Run the complete downloaded benchmark corpus through the correctness gate |
| [`fetch.sh`](fetch.sh) | Download the public benchmark archives |
| [`gate.py`](gate.py) | Compare GROMACS energy blocks term by term |
| [`../../scripts/build_gromacs_cumetal.sh`](../../scripts/build_gromacs_cumetal.sh) | Build matching GPU and CPU-reference binaries |
| `out/` | Local generated inputs, build logs, run directories, and gate output; not benchmark evidence by itself |
