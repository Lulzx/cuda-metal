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
solver that HiGHS vendors as its GPU path — `libhighs` exports ~104 `cupdlp_*`
symbols of its own. So "does HiGHS's GPU solver work through CuMetal" is
answerable by building cuPDLP-C unmodified and comparing it to its CPU build.

It is built with `cupdlp_float = double`, the upstream default — not the reduced
`SFLOAT` configuration. Metal has no `double`, so every FP64 operation runs on
CuMetal's Dekker FP32-pair emulation while storage stays IEEE-754 binary64.

## What it checks

For each problem, both builds solve and must agree on:

- **model status class** — `Optimal` vs `Iteration limit`. ("Optimal current" vs
  "Optimal average" is only which iterate PDLP accepted, not a disagreement.)
- **primal objective**, within 1e-3 relative. PDLP's own default gap tolerance is
  1e-4, so two runs legitimately stop at different points inside it.
- **primal infeasibility, dual infeasibility and relative duality gap.** The
  objective alone is not enough: a run can report a plausible objective while its
  residuals say the point is not feasible. Each must be within 10× what the CPU
  build achieved, and — only when the solver claims convergence — under 1e-3
  absolute. A run stopped at the iteration limit legitimately has a large gap
  (`stair` ends near 1e-2 on both builds), so the absolute ceiling would fail a
  run that agrees perfectly well with its reference.
- every reported value **finite**.

Iteration counts are deliberately *not* compared. The builds follow different
trajectories — `shell` converges in 5,600 iterations against the CPU's 12,000 —
and that is a path difference, not a disagreement about the answer.

Then one run is traced and must show `device=apple_gpu` with no
`source=approximate_stub`. A correct number without that provenance is **not** a
pass — it would mean a CPU fallback produced it.

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

482 kernel launches traced, all `device=apple_gpu`. Because these kernels use
`double`, they report:

```text
provenance=generic_ptx_lowering_fp64_emulated semantic_quality=reduced_precision_fp64
```

not `semantic_quality=exact`. The lowering is a real, structurally faithful
translation with no substitution — but Metal has no `double`, so the arithmetic
is roughly a 48-bit significand rather than CUDA's FP64 semantics, and calling
that `exact` would be its own kind of green-wash.

`stair` hits the iteration limit on **both** builds; it is a hard instance, not a
Metal failure. Its Metal residuals are in fact slightly *better* than the CPU
run's (gap 9.05e-03 vs 1.41e-02).

## What this does not show

- **No speedup.** cuBLAS L1 and the FP64 sparse SpMV are still scalar CPU loops
  over unified memory, so this is a correctness result only. Do not read the
  timings as a benchmark.
- **Not HiGHS's own GPU build.** This is standalone cuPDLP-C. HiGHS's
  `CUPDLP_GPU=ON` path additionally requires `cusparseDnVecSetValues`,
  `cusparseSpMV_preprocess`, and CUDA-graph capture of library nodes, none of
  which CuMetal implements yet. See `docs/known-gaps.md`.
- **Not full IEEE-754 FP64.** ~48-bit significand within binary32's exponent
  envelope; see `docs/known-gaps.md` and
  `tests/cuda_projects/fp64/fp64_precision.cu`.

## Patches applied to cuPDLP-C

Three, none Metal-related — all consequences of building an old standalone
cuPDLP-C against a current HiGHS. They are applied by
`scripts/build_cupdlp_cumetal.sh` at build time:

1. HiGHS ≥ 1.7 exports the same `Init_Scaling` / `cupdlp_*` symbols, so
   `libcupdlp` must be named by path to stay ahead of `libhighs` at link time.
   Otherwise the first `Init_Scaling` call binds to HiGHS's copy and segfaults.
2. HiGHS 1.15's `HConst.h` defines `enum ConstraintType { EQ, LEQ, GEQ, BOUND }`
   at namespace scope, colliding with `wrapper_highs.h`.
3. `cupdlp/CMakeLists.txt` hardcodes `/usr/local/cuda/include`.
