# HiGHS / cuPDLP-C

An LP solver's CUDA GPU path running on Apple Silicon, checked against its own
CPU build.

```bash
bash demos/highs/run.sh          # from a built CuMetal tree
```

Nothing from cuPDLP-C or HiGHS lives here. `run.sh` calls
`scripts/build_cupdlp_cumetal.sh`, which clones cuPDLP-C outside the tree and
downloads the LPs into `out/`. HiGHS comes from `brew install highs`.

[cuPDLP-C](https://github.com/COPT-Public/cuPDLP-C) is the PDLP solver HiGHS
vendors as its GPU path; `libhighs` exports ~104 `cupdlp_*` symbols of its own.
The `run.sh` comparison builds standalone cuPDLP-C so it can isolate that
solver's CPU and GPU behavior. Separately, `scripts/build_highs_cumetal.sh`
builds unmodified HiGHS itself with `CUPDLP_GPU=ON`; the focused HiGHS run
described below exercises that integration layer on the Apple GPU.

Built with `cupdlp_float = double`, the upstream default, not the reduced
`SFLOAT` config. Storage is IEEE-754 binary64; arithmetic is not. Values decode
to an FP32 pair, compute at ~48 significand bits inside binary32's exponent
range, and encode back. The binary64 container is what the CUDA ABI sees, so
`cudaMemcpy`, `.local` spills, warp shuffles and `uint64_t` aliasing behave
normally. Only the arithmetic is reduced.

## What it checks

Both builds must agree on:

- Status class, `Optimal` vs `Iteration limit`. "Optimal current" and "Optimal
  average" differ only in which iterate PDLP took.
- Primal objective, 1e-3 relative. PDLP's own gap tolerance is 1e-4, so two runs
  stop at different points inside it.
- Primal infeasibility, dual infeasibility, relative gap: within 10x the CPU
  build, and under 1e-3 absolute once the solver claims convergence. A run can
  print a plausible objective while its residuals say the point is infeasible.
  An iteration-limit run has a large gap by construction (`stair` ends near
  1e-2 on both), so the absolute ceiling is skipped there.
- Every value finite.

Iteration counts are not compared: `shell` converges in 5,600 against the CPU's
12,000, which is a different trajectory, not a disagreement.

Auto mode keeps this corpus on the CPU, so one problem re-solves with
`CUMETAL_SPARSE_METAL=1` under the same gates, or the demo would never touch the
GPU sparse kernels. One run is traced and must show `device=apple_gpu` with no
`source=approximate_stub`; a correct number without provenance means a CPU
fallback produced it.

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

The lowering translates that kernel's own PTX with no substitution, so the
provenance is honest; the numerical contract is what differs, which is why the
fields are separate.

`stair` hits the iteration limit on both builds, with better Metal residuals
than the CPU run: gap 9.05e-03 against 1.41e-02.

## SpMV

Both of cuPDLP's products reduce to the gather shape: `Ax` is CSR
non-transpose, `A'y` is CSC transpose, the same loop over the other compressed
axis. Neither needs atomic scatter, and Metal has no FP64 atomic.

A dispatch costs ~100 us, so completed CSR SpMV only overtakes the CPU loop near
1e5 nonzeros:

| nonzeros | Metal | CPU | speedup |
| ---: | ---: | ---: | ---: |
| 3.2e4 | 36.9 us | 36.6 us | 0.99x |
| 1.3e5 | 149.6 us | 149.1 us | 1.00x |
| 5.1e5 | 227.8 us | 848.4 us | 3.72x |
| 2.0e6 | 427.0 us | 3859.8 us | 9.04x |
| 8.2e6 | 1987.4 us | 14869.3 us | 7.48x |
| 3.2e7 | 11220.6 us | 65799.4 us | 5.86x |

This corpus is far below that, so auto routes it to the CPU and forcing the GPU
costs 20-27%. `CUMETAL_SPARSE_METAL=1`/`0` overrides;
`CUMETAL_SPARSE_METAL_THRESHOLD_NNZ` moves the threshold.

Two Mittelmann LPs are big enough to show what it does to a real solve. Solver
wall time, median of 3, 99 fixed iterations:

| instance | shape | cpu | auto | scalar forced | cooperative forced |
| --- | --- | ---: | ---: | ---: | ---: |
| `ex10` | 69,609 x 17,680, 1.18M nnz | 1.180 s | **0.276 s** | 0.270 s | 0.270 s |
| `datt256` | 11,078 x 262,144, 1.77M nnz | 1.884 s | **0.823 s** | 5.360 s | 1.065 s |

All four configs reach bit-identical primal and dual objectives and the same
primal infeasibility. Most of `ex10`'s CPU solve is `Ax` and `A'y`, both move to
the GPU, 3.8x faster.

`datt256` runs 2.8x slower than the CPU on the thread-per-row kernel. After
reformulation its longest row holds 57,840 entries against a mean of 136, so one
thread grinds through it while the other 11,076 idle, and parallelism does not
hide a serial loop. A synthetic matrix with the same dimensions and mean row
length runs 6x faster, so a uniform-row benchmark would have missed this.

Splitting that row across a simdgroup cuts serial depth by 32: 5.295 s becomes
1.084 s, and the SpMV itself goes from 22.9 ms to 1.8 ms, beating the CPU's
2.0 ms instead of losing 11x. Auto is faster still at 0.883 s, because the two
products want different kernels and it picks per call from the row distribution.
The cooperative kernel wins once rows fill a simdgroup and loses on uniformly
short ones, so 4-entry rows still take the scalar kernel.
`CUMETAL_SPARSE_METAL_KERNEL=scalar|simd` pins one, `CUMETAL_DEBUG_SPARSE=1`
reports which ran and why.

## cuBLAS level-1

`cublasDaxpy`, `cublasDscal`, `cublasDdot` and `cublasDnrm2` have Metal kernels
using the same Dekker pair as the sparse ones, and route by length: elementwise
at 4096, reductions at 131072.

They differ by 32x because the cost models differ. An axpy pays only for its
enqueue, which command-buffer batching amortizes to microseconds. A dot has to
synchronize to hand a scalar back to the host, a flat ~106 us floor no kernel
speed can move. `cumetal_cublas_blas1_metal_bench` prints the measurements both
defaults came from.

`datt256`, `--presolve off`, HiGHS 1.15.1 `CUPDLP_GPU=ON`, median of 3: 3.96 s
before, 3.16 s after, against 5.17 s for the same HiGHS built `CUPDLP_GPU=OFF`.
The first run after a rebuild is several seconds slower while the new MSL
compiles and caches.

Profiling puts 71% of that solve in `cupdlp_movement_interaction_cuda`, cuPDLP's
own FP64 tree reduction, which already runs on Metal, and 0.5% in the sparse
products. Level-1 was 26% before this change.

## What this does not show

No solver speedup on this corpus. All eight instances are far below the size
where the GPU sparse path pays, and forcing it makes them slower. The speedups
above are on two much larger LPs that are not part of this demo. Solver-level
synchronization is unoptimized. Do not read these solve times as a benchmark.

The eight-instance demo above exercises standalone cuPDLP-C. CuMetal now also
implements the three interfaces that had blocked HiGHS's own GPU integration:
`cusparseDnVecSetValues`, real `cusparseSpMV_preprocess` analysis, and captured
cuSPARSE SpMV graph nodes. `functional_cusparse_graph_capture` checks that
capture is non-eager, replay uses the captured descriptors, and later device
data is observed on both the forced-Metal and CPU sparse routes. An unmodified
HiGHS 1.15.1 `CUPDLP_GPU=ON` build from `scripts/build_highs_cumetal.sh` also
solves `afiro` with `--presolve off` to Optimal in 360 iterations while CuMetal
traces successful kernel launches on the Apple M4 Pro. This is focused
integration evidence, not a claim that every HiGHS model or CUDA-library graph
operation is supported. See `docs/known-gaps.md`.

The arithmetic is not binary64, per the FP32 pair above: 48 significand bits
against binary64's 53, and binary32's much narrower exponent range. Worst
measured relative round-trip error is 3.29e-15 against a 2^-48 (3.55e-15)
contract, gated by `tests/cuda_projects/fp64/fp64_precision.cu`.

## Patches to cuPDLP-C

Four, none Metal-related, applied at build time by
`scripts/build_cupdlp_cumetal.sh`. The first three follow from building an old
standalone cuPDLP-C against a current HiGHS:

1. HiGHS >= 1.7 exports the same `Init_Scaling` / `cupdlp_*` symbols, so
   `libcupdlp` must be named by path to stay ahead of `libhighs` at link time.
   Otherwise the first `Init_Scaling` binds to HiGHS's copy and segfaults.
2. HiGHS 1.15's `HConst.h` defines `enum ConstraintType { EQ, LEQ, GEQ, BOUND }`
   at namespace scope, colliding with `wrapper_highs.h`.
3. `cupdlp/CMakeLists.txt` hardcodes `/usr/local/cuda/include`.

The fourth is an upstream correctness bug, kept separate for that reason.
`PDHG_Power_Method` reads `ax`, a vector sized to `nRows`, for `nCols` elements:
it segfaults on `datt256` often enough to lose runs and silently norms a quarter
of the vector on `ex10`. The step size it returns is unaffected, so patching it
moved the crash rate, not the timings above. It reproduces without CuMetal;
CUDA's slab allocator hides the read where a per-allocation `MTLBuffer` faults.
