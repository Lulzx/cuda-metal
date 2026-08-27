# HiGHS / cuPDLP-C

An LP solver's CUDA GPU path, running on Apple Silicon, checked against its own
CPU build.

```bash
bash demos/highs/run.sh          # from a built CuMetal tree
```

Nothing from cuPDLP-C or HiGHS lives here. `run.sh` calls
`scripts/build_cupdlp_cumetal.sh`, which clones cuPDLP-C outside the tree and
downloads the LPs into `out/`. HiGHS comes from `brew install highs`.

## What it is

[cuPDLP-C](https://github.com/COPT-Public/cuPDLP-C) is the PDLP solver HiGHS
vendors as its GPU path — `libhighs` exports ~104 `cupdlp_*` symbols of its own.
Building it unmodified and diffing it against its CPU build tests that solver,
not HiGHS's `CUPDLP_GPU=ON` integration layer.

Built with `cupdlp_float = double`, the upstream default, not the reduced
`SFLOAT` config. **Storage is IEEE-754 binary64. Arithmetic is not.** Values
decode to an FP32 pair, compute at ~48 significand bits inside binary32's
exponent range, and encode back. The binary64 container is what the CUDA ABI
sees, so `cudaMemcpy`, `.local` spills, warp shuffles, and `uint64_t` aliasing
all behave normally. Only the arithmetic is reduced.

## What it checks

Both builds must agree on:

- **Status class.** `Optimal` vs `Iteration limit`. ("Optimal current" and
  "Optimal average" differ only in which iterate PDLP took.)
- **Primal objective**, 1e-3 relative. PDLP's own gap tolerance is 1e-4, so two
  runs stop at different points inside it.
- **Primal infeasibility, dual infeasibility, relative gap.** The objective alone
  is not enough — a run can print a plausible objective while its residuals say
  the point is infeasible. Within 10× the CPU build, and under 1e-3 absolute once
  the solver claims convergence. Not applied to an iteration-limit run, which has
  a large gap by construction (`stair` ends near 1e-2 on both).
- Every value finite.

Iteration counts are **not** compared. Different trajectories are not
disagreement: `shell` converges in 5,600 against the CPU's 12,000.

Then one problem re-solves with `CUMETAL_SPARSE_METAL=1` under the same gates —
auto mode keeps this corpus on the CPU, so without that stage the demo would
never touch the GPU sparse kernels. Finally one run is traced and must show
`device=apple_gpu` with no `source=approximate_stub`. A correct number without
provenance fails: it means a CPU fallback produced it.

## Results (M4 Pro, Debug, shim ON)

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

482 launches traced, all `device=apple_gpu`; 608 with `CUMETAL_SPARSE_METAL=1`,
the extra 126 being `cumetal_spmv_gather_f64` for `Ax` and `A'y`. Those report:

```text
provenance=generic_ptx_lowering_fp64_emulated semantic_quality=reduced_precision_fp64
```

not `semantic_quality=exact`. The lowering translates that kernel's own PTX with
no substitution, so the translation provenance is honest; the numerical contract
is what differs. Reporting both separately is the point.

`stair` hits the iteration limit on both builds. It is a hard instance, and its
Metal residuals come out better than the CPU run's (gap 9.05e-03 vs 1.41e-02).

## SpMV: where it pays

Both of cuPDLP's products reduce to the gather shape — `Ax` is CSR
non-transpose, `A'y` is CSC transpose, the same loop over the other compressed
axis. Neither needs atomic scatter, which matters because Metal has no FP64
atomic.

A Metal dispatch costs ~100 us, so completed CSR SpMV only overtakes the CPU loop
near 1e5 nonzeros:

| nonzeros | Metal | CPU | speedup |
| ---: | ---: | ---: | ---: |
| 3.2e4 | 36.9 us | 36.6 us | 0.99x |
| 1.3e5 | 149.6 us | 149.1 us | 1.00x |
| 5.1e5 | 227.8 us | 848.4 us | 3.72x |
| 2.0e6 | 427.0 us | 3859.8 us | 9.04x |
| 8.2e6 | 1987.4 us | 14869.3 us | 7.48x |
| 3.2e7 | 11220.6 us | 65799.4 us | 5.86x |

This corpus is far below that, so auto routes it to the CPU and forcing the GPU
costs 20-27%. `CUMETAL_SPARSE_METAL=1`/`0` overrides,
`CUMETAL_SPARSE_METAL_THRESHOLD_NNZ` moves the threshold.

Two Mittelmann LPs are big enough to answer what it does to a real solve. Solver
wall time, median of 3, 99 fixed iterations:

| instance | shape | cpu | auto | scalar forced | cooperative forced |
| --- | --- | ---: | ---: | ---: | ---: |
| `ex10` | 69,609 x 17,680, 1.18M nnz | 1.180 s | **0.276 s** | 0.270 s | 0.270 s |
| `datt256` | 11,078 x 262,144, 1.77M nnz | 1.884 s | **0.823 s** | 5.360 s | 1.065 s |

All four configs reach bit-identical primal and dual objectives and the same
primal infeasibility.

`ex10` is the easy case: most of its CPU solve is `Ax` and `A'y`, both move to
the GPU, 3.8x faster.

`datt256` is the interesting one. On the thread-per-row kernel it is **2.8x
slower than the CPU**, because after reformulation its longest row holds 57,840
entries against a mean of 136. One thread grinds through that row while the other
11,076 idle, and no amount of parallelism hides a serial loop. A synthetic matrix
with the same dimensions and mean row length runs 6x *faster* — a uniform-row
benchmark would have missed this entirely.

Splitting that row across a simdgroup cuts serial depth by 32 and fixes it:
5.295 s becomes 1.084 s, and that SpMV goes 22.9 ms → 1.8 ms, beating the CPU's
2.0 ms instead of losing 11x. Auto is faster still at 0.883 s, because the two
products want different kernels and it picks per call. The choice comes from the
row distribution: the cooperative kernel wins once rows fill a simdgroup and
loses on uniformly short ones, so 4-entry rows still take the scalar kernel.
`CUMETAL_SPARSE_METAL_KERNEL=scalar|simd` pins one; `CUMETAL_DEBUG_SPARSE=1`
says which ran and why.

## cuBLAS level-1

FP64 level-1 is no longer a scalar CPU loop. `cublasDaxpy`, `cublasDscal`,
`cublasDdot` and `cublasDnrm2` have Metal kernels using the same Dekker pair as
the sparse ones, and route by length: **elementwise at 4096, reductions at
131072.**

That 32x gap is not arbitrary. An axpy only pays for its enqueue, which
command-buffer batching amortizes to microseconds. A dot has to synchronize to
hand a scalar back to the host, and that wait is a flat ~106 us floor no kernel
speed can move. `cumetal_cublas_blas1_metal_bench` prints the table both
defaults came from.

`datt256`, `--presolve off`, HiGHS 1.15.1 `CUPDLP_GPU=ON`, median of 3:
**3.96 s → 3.16 s**, against 5.17 s for the same HiGHS built `CUPDLP_GPU=OFF`.
The first run after a rebuild is several seconds slower while the new MSL
compiles and caches.

What is left is not level-1. Profiling puts 71% of that solve in
`cupdlp_movement_interaction_cuda` — cuPDLP's own FP64 tree reduction, already on
Metal — and 0.5% in the sparse products. Level-1 was 26% before this change.

## What this does not show

**No solver speedup on this corpus.** All eight instances are far below the size
where the GPU sparse path pays, and forcing it makes them slower. The speedups
above are on two much larger LPs that are not part of this demo. Solver-level
synchronization is unoptimized. Do not read these solve times as a benchmark.

**This is standalone cuPDLP-C, not HiGHS's own GPU build.** That additionally
needs `cusparseDnVecSetValues`, `cusparseSpMV_preprocess`, and CUDA-graph capture
of library nodes, none of which CuMetal implements. See `docs/known-gaps.md`.

**The arithmetic is not binary64.** ~48 significand bits inside binary32's
exponent range, against binary64's 53 and a much wider range. Worst measured
relative round-trip error 3.29e-15 against a 2^-48 (3.55e-15) contract. See
`tests/cuda_projects/fp64/fp64_precision.cu`.

## Patches to cuPDLP-C

Four, none Metal-related, applied at build time by
`scripts/build_cupdlp_cumetal.sh`. The first three follow from building an old
standalone cuPDLP-C against a current HiGHS:

1. HiGHS ≥ 1.7 exports the same `Init_Scaling` / `cupdlp_*` symbols, so
   `libcupdlp` must be named by path to stay ahead of `libhighs` at link time.
   Otherwise the first `Init_Scaling` binds to HiGHS's copy and segfaults.
2. HiGHS 1.15's `HConst.h` defines `enum ConstraintType { EQ, LEQ, GEQ, BOUND }`
   at namespace scope, colliding with `wrapper_highs.h`.
3. `cupdlp/CMakeLists.txt` hardcodes `/usr/local/cuda/include`.

The fourth is an upstream correctness bug, kept separate on purpose:
`PDHG_Power_Method` reads `ax` — a vector sized to `nRows` — for `nCols`
elements. It segfaults on `datt256` often enough to lose runs and silently norms
a quarter of the vector on `ex10`. The step size it returns is unaffected, so the
timings above did not move when it was patched; the crashes did. It reproduces
without CuMetal. CUDA's slab allocator just hides the read where a
per-allocation `MTLBuffer` faults on it.
