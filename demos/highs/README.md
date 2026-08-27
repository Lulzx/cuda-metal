# HiGHS / cuPDLP-C demo

**An LP solver's CUDA GPU path, running on Apple Silicon, checked against its own CPU build.**

```bash
# from a built CuMetal tree
bash demos/highs/run.sh
```

Nothing from cuPDLP-C or HiGHS lives in this repo. `run.sh` calls
`scripts/build_cupdlp_cumetal.sh`, which clones cuPDLP-C outside the tree and
downloads the LP instances into `out/` (gitignored). HiGHS itself comes from
`brew install highs`.

---

## What this is

[cuPDLP-C](https://github.com/COPT-Public/cuPDLP-C) is the PDLP first-order LP
solver that HiGHS vendors as its GPU path; `libhighs` exports ~104 `cupdlp_*`
symbols of its own. So building cuPDLP-C unmodified and comparing it against its
CPU build tests the core GPU solver that HiGHS vendors, without yet exercising
HiGHS's own `CUPDLP_GPU=ON` integration layer.

It is built with `cupdlp_float = double`, the upstream default, rather than the
reduced `SFLOAT` configuration. **Storage is IEEE-754 binary64; arithmetic is
not.** CuMetal decodes each value into an FP32 pair, runs the operation at about
a 48-bit significand within binary32's exponent envelope, and encodes the result
back into binary64. The binary64 container is what every part of the CUDA ABI
sees, so `cudaMemcpy`, `.local` spills, warp shuffles and reading the same eight
bytes as a `uint64_t` all behave normally; only the arithmetic is reduced.

## What it checks

For each problem, both builds solve and must agree on:

- Model status class: `Optimal` vs `Iteration limit`. ("Optimal current" and
  "Optimal average" differ only in which iterate PDLP accepted.)
- Primal objective, within 1e-3 relative. PDLP's own default gap tolerance is
  1e-4, so two runs stop at different points inside it.
- Primal infeasibility, dual infeasibility, and relative duality gap. The
  objective alone is not enough: a run can report a plausible objective while its
  residuals say the point is not feasible. Each must be within 10× what the CPU
  build achieved, and, once the solver claims convergence, under 1e-3 absolute.
  A run stopped at the iteration limit has a large gap by construction (`stair`
  ends near 1e-2 on both builds), so applying the absolute ceiling there would
  fail a run that matches its reference.
- Every reported value finite.

Iteration counts are not compared. The builds follow different trajectories,
`shell` converging in 5,600 iterations against the CPU's 12,000, which says
nothing about whether they agree on the answer.

One problem is then re-solved with `CUMETAL_SPARSE_METAL=1`, which forces the
Metal SpMV path, and must clear the same gates. Auto mode keeps these instances'
sparse products on the CPU (see below), so without this stage the demo would
never exercise the GPU sparse kernels at all.

Finally one run is traced and must show `device=apple_gpu` with no
`source=approximate_stub`. A correct number without that provenance fails the
demo, because it would mean a CPU fallback produced it.

## Measured (M4 Pro, Debug build, shim ON)

| problem | cpu | metal | rel diff |
| --- | --- | --- | --- |
| afiro | Optimal −4.64750909e+02 (200) | Optimal −4.64750634e+02 (200) | 5.9e-07 |
| adlittle | Optimal +2.25476037e+05 (2720) | Optimal +2.25464108e+05 (2320) | 5.3e-05 |
| blending | Optimal −3.20000000e+03 (120) | Optimal −3.20000000e+03 (120) | 0 |
| e226 | Optimal −1.16392301e+01 (16760) | Optimal −1.16405125e+01 (15720) | 1.1e-04 |
| israel | Optimal −8.96580982e+05 (2560) | Optimal −8.96585250e+05 (2560) | 4.8e-06 |
| shell | Optimal +1.20883600e+09 (12000) | Optimal +1.20885006e+09 (5600) | 1.2e-05 |
| standata | Optimal +1.25745066e+03 (920) | Optimal +1.25777205e+03 (1080) | 2.6e-04 |
| stair | Iteration limit −2.51238976e+02 | Iteration limit −2.51313524e+02 | 3.0e-04 |

482 kernel launches traced, all `device=apple_gpu`. With `CUMETAL_SPARSE_METAL=1`
that becomes 608, the extra 126 being `cumetal_spmv_gather_f64` for `Ax` and
`A'y`. Because these kernels use `double`, they report:

```text
provenance=generic_ptx_lowering_fp64_emulated semantic_quality=reduced_precision_fp64
```

rather than `semantic_quality=exact`. The lowering translates this kernel's own
PTX with no substitution, so the translation provenance is honest; the numerical
contract is what differs, and `exact` would overstate what ran. Keeping those two
separate is the point of reporting both fields.

`stair` hits the iteration limit on both builds; it is a hard instance. Its
Metal residuals come out better than the CPU run's (gap 9.05e-03 vs 1.41e-02).

## Where the sparse products actually run

CuMetal has a Metal SpMV kernel for the gather shapes, which is what both of
cuPDLP's products reduce to: `Ax` is CSR non-transpose, and `A'y` is CSC
transpose, which is the same loop over the other compressed axis. Neither needs
atomic scatter.

It is not used here by default. On an M4 Pro a Metal dispatch costs on the order
of 100 us, and completed CSR SpMV only overtakes the CPU loop over unified memory
somewhere near 1e5 nonzeros:

| nonzeros | Metal | CPU | speedup |
| ---: | ---: | ---: | ---: |
| 3.2e4 | 36.9 us | 36.6 us | 0.99x |
| 1.3e5 | 149.6 us | 149.1 us | 1.00x |
| 5.1e5 | 227.8 us | 848.4 us | 3.72x |
| 2.0e6 | 427.0 us | 3859.8 us | 9.04x |
| 8.2e6 | 1987.4 us | 14869.3 us | 7.48x |
| 3.2e7 | 11220.6 us | 65799.4 us | 5.86x |

Every instance in this corpus is far below that, so auto mode routes them to the
CPU, and forcing the GPU path makes the solver 20-27% slower end to end. The
threshold is a conservative M4 Pro default measured with a synchronizing
microbenchmark; `CUMETAL_SPARSE_METAL=1`/`0` overrides the choice and
`CUMETAL_SPARSE_METAL_THRESHOLD_NNZ` moves it.

Two LPs from the Mittelmann test set are large enough to answer what that does to
a real solve. Solver wall time, median of 3, 99 fixed iterations, M4 Pro:

| instance | shape | cpu | auto | scalar kernel forced | cooperative kernel forced |
| --- | --- | ---: | ---: | ---: | ---: |
| `ex10` | 69,609 x 17,680, 1.18M nnz | 1.197 s | **0.323 s** | 0.322 s | 0.324 s |
| `datt256` | 11,078 x 262,144, 1.77M nnz | 1.905 s | **0.879 s** | 5.312 s | 1.071 s |

All four configurations reach bit-identical primal and dual objectives and the
same primal infeasibility on each instance.

These are measured with patch 4 from `scripts/build_cupdlp_cumetal.sh` applied.
Without it `PDHG_Power_Method` reads `ax` -- a vector `vec_Alloc` sized to
`nRows` -- for `nCols` elements, which segfaults on `datt256` often enough to
lose runs, and on `ex10` silently norms a quarter of the vector. The step size
the power method returns is unaffected either way, so the timings above did not
change when it was patched; the run-to-run crashes did. It is an upstream bug,
not a Metal one, and it reproduces without CuMetal -- CUDA's slab allocator just
hides the read where a per-allocation `MTLBuffer` faults on it.

`ex10` is the straightforward case: most of its CPU solve is `Ax` and `A'y`, both
go to the GPU, and the solve is 3.8x faster.

`datt256` is the interesting one. Forced onto the thread-per-row kernel it is
2.8x *slower* than the CPU, because after reformulation its longest row holds
57,840 entries against a mean of 136. One thread per row means one thread grinds
through that row while the other 11,076 idle, and no amount of parallelism hides
a serial loop. A synthetic matrix with the same dimensions and mean row length
runs 6x faster on the GPU, so a uniform-row benchmark would have missed this
entirely.

Splitting that row across a simdgroup cuts the serial depth by 32 and fixes the
kernel outright: 5.295 s becomes 1.084 s, and in isolation that SpMV goes from
22.9 ms to 1.8 ms, which is faster than the CPU's 2.0 ms rather than 11x slower.
Auto mode is faster still, at 0.883 s, because the two products want different
kernels and it picks per call rather than per solve. Which one runs is chosen
from the row distribution: the cooperative kernel wins as soon as rows are long
enough to fill a simdgroup and loses on uniformly short ones, so a matrix of
4-entry rows still takes the scalar kernel. `CUMETAL_SPARSE_METAL_KERNEL=scalar`
or `=simd` pins one, and `CUMETAL_DEBUG_SPARSE=1` reports which ran and why.

## What this does not show

No solver speedup on this corpus. These eight instances are all far below the
size where the GPU sparse path pays for itself, and forcing it makes them slower;
the solver speedups above are on two much larger LPs that are not part of this
demo. FP64 cuBLAS L1 (dot, norm) is still a scalar CPU loop
over unified memory, and solver-level synchronization is unoptimized. Do not read
these solve times as a benchmark.

On `datt256` that CPU loop is now most of what is left: with both products on the
GPU, `UpdateIterates` is 4.49 s of the 5.51 s solve.

This is standalone cuPDLP-C, not HiGHS's own GPU build. HiGHS's `CUPDLP_GPU=ON`
path additionally requires `cusparseDnVecSetValues`, `cusparseSpMV_preprocess`,
and CUDA-graph capture of library nodes, none of which CuMetal implements yet.
See `docs/known-gaps.md`.

The arithmetic is not IEEE-754 binary64, as above: about a 48-bit significand
within binary32's exponent envelope, against binary64's 53 bits and much wider
range. Measured worst relative round-trip error is 3.29e-15 against a 2^-48
(3.55e-15) contract. See `docs/known-gaps.md` and
`tests/cuda_projects/fp64/fp64_precision.cu`.

## Patches applied to cuPDLP-C

Three, none of them Metal-related. All follow from building an old standalone
cuPDLP-C against a current HiGHS, and `scripts/build_cupdlp_cumetal.sh` applies
them at build time:

1. HiGHS ≥ 1.7 exports the same `Init_Scaling` / `cupdlp_*` symbols, so
   `libcupdlp` must be named by path to stay ahead of `libhighs` at link time.
   Otherwise the first `Init_Scaling` call binds to HiGHS's copy and segfaults.
2. HiGHS 1.15's `HConst.h` defines `enum ConstraintType { EQ, LEQ, GEQ, BOUND }`
   at namespace scope, colliding with `wrapper_highs.h`.
3. `cupdlp/CMakeLists.txt` hardcodes `/usr/local/cuda/include`.
