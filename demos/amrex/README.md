# AMReX on CuMetal

[AMReX](https://github.com/AMReX-Codes/amrex) 25.09's CUDA GPU backend builds
and runs unmodified on an Apple GPU through CuMetal. This directory holds the
reproducible correctness gate and the CMake glue for the tutorial it runs.

```bash
bash demos/amrex/run.sh --quick   # 20 steps
bash demos/amrex/run.sh           # 200 steps
bash demos/amrex/run.sh --long    # 1000 steps, the tutorial's own input
```

Nothing from AMReX is vendored here. The scripts fetch AMReX 25.09 and
[amrex-tutorials](https://github.com/AMReX-Codes/amrex-tutorials) at pinned
revisions and build AMReX twice from one source tree: once with
`AMReX_GPU_BACKEND=CUDA` against CuMetal's CUDA toolkit shim, once with
`AMReX_GPU_BACKEND=NONE` as the reference. **AMReX itself is not patched** —
there are no AMReX changes in this path, only CuMetal ones.

## What runs

`HeatEquation_EX0_C` integrates the 3-D heat equation on a 32³ triply periodic
domain decomposed into eight 16³ boxes. On the GPU build, every part of the time
step is a CUDA kernel reached through `amrex::ParallelFor`: the Gaussian initial
condition, the three-direction flux computation, the flux-divergence update, and
the ghost-cell exchange between boxes.

The kernels are `__device__` lambdas in an ordinary `.cpp` file, which AMReX
compiles as CUDA. They reach the CuMetal runtime through
`__cudaRegisterFatBinary`, so this demo needs a build configured with
`-DCUMETAL_ENABLE_BINARY_SHIM=ON`; it reports SKIP otherwise.

## The gate

`run.sh` compares GPU and CPU plotfiles with AMReX's own `fcompare`, twice:

| Comparison | What it isolates |
| --- | --- |
| step 0 | the initial-condition kernel alone |
| final step | every stencil update and ghost exchange after it |

Both are reported because they fail differently. A wrong stencil, a stale ghost
cell, or a dropped launch moves the final field by percent while leaving step 0
untouched. Step 0 instead measures CuMetal's FP64 transcendentals: the initial
condition is a Gaussian, and under FP64 emulation `exp` on a `double` evaluates
through binary32 ([FP64 policy](../../docs/fp64-policy.md)), which sets the floor
on agreement for this problem.

Two further checks stop a degenerate pass. The GPU field must span the
tutorial's actual dynamic range — an all-zero field is exactly how both defects
below first appeared — and CuMetal's trace must show every launch on
`device=apple_gpu` with `launch_success=true`.

### Recorded result

Apple M4 Pro, macOS 27.0, Release build, `CUMETAL_ENABLE_BINARY_SHIM=ON`:

| Comparison | L∞ absolute | relative | gate |
| --- | ---: | ---: | ---: |
| step 0 (init) | 4.29e-08 | 2.22e-08 | 1e-6 |
| step 20 (final), `--quick` | 2.14e-08 | 1.17e-08 | 1e-6 |
| step 200 (final), default | 7.77e-09 | 5.53e-09 | 1e-6 |

243 kernel launches for `--quick`, 2,043 for the 200-step run; all
`device=apple_gpu`, none with `launch_success=false`.

The tolerance has roughly two orders of magnitude of headroom over the measured
agreement and eight over a genuine failure: running the same binary without
`CUMETAL_USE_METAL_DEVICE_ADDRESSES=1` — the defect described below — gives a
relative error of exactly 1.0 and a field range of `(0, 0)`, and both gates fire.

## One compatibility mode is required

`run.sh` exports `CUMETAL_USE_METAL_DEVICE_ADDRESSES=1`, and AMReX does not work
without it.

AMReX keeps a device-resident array of `Array4` descriptors, one per box, each
holding the raw pointer to that box's data. `MultiFab::min`, every `ParReduce`,
and the fused multi-box `ParallelFor` kernels index that array on the GPU and
dereference the pointer they find. Following a pointer loaded from device
memory — rather than one passed as a launch argument — requires the allocation
it points at to be resident for the dispatch, which is what this mode does. It
is the same requirement PhysX has
([feasibility notes](../../docs/physx-feasibility.md)).

Without it the loads return zeros and nothing reports an error, because from
Metal's side nothing invalid happened. That is why `run.sh` sets it rather than
leaving it to the caller. The cost is real: the mode marks every live allocation
resident on every dispatch, which removes cross-stream concurrency. Detecting
the pattern at lowering time and warning about it is
[open work](../../docs/known-gaps/runtime.md).

## Configuration

Four AMReX options are moved off their defaults, all about CuMetal's coverage
rather than about the comparison:

| Option | Why |
| --- | --- |
| `AMReX_CUDA_FASTMATH=OFF` | AMReX defaults nvcc's `--use_fast_math` on. The demo's claim is a numerical comparison, so both builds must share the floating-point contract. |
| `AMReX_CUDA_MAXREGCOUNT=0` | `-maxrregcount` has no meaning here: ptxas is a shim and Metal does its own register allocation. |
| `AMReX_GPU_RDC=OFF` | Relocatable device code fuses per-TU device images at link time. CuMetal has no such image; each translation unit registers its own kernels at load. |
| `AMReX_MPI=OFF` | Single process, single GPU, so both builds integrate the same decomposition. |

`AMReX_PRECISION` is left at its default of `DOUBLE`. On Metal that means the
FP32-pair emulation, and the demo reports the resulting agreement rather than
switching to single precision to make the number look better.

## Defects this exposed

Seven CuMetal defects, every one of them silent — a wrong number or a failed
compile, never a diagnostic. All seven are pinned by regression tests:
`functional_cuda_projects_amrex_device_idioms` runs each device-side idiom
against a host reference computed in the same process,
`functional_extended_api_v4` covers the batch pointer query, and
`unit_ptx_lower_to_llvm` pins the two new lowerings at the IR level.

1. **No `std::`-qualified math in device code at all.** CuMetal included Clang's
   CUDA math header, which declares only the C entry points in the global
   namespace, so libc++ had never nominated a `__device__` overload into `std`.
   `std::sqrt(x)` inside a `__device__` function did not compile. AMReX writes
   every `ParallelFor` body in `std::` form. The fix adds the binary32 overloads
   (without them an unqualified `float` call promotes to software binary64),
   `<cmath>`'s integral and mixed-argument promotions (without which
   `std::pow(2, n)` is ambiguous rather than merely unpromoted), and the `std`
   re-export.

2. **Clang's strict aliasing deleted AMReX's warp reductions.** AMReX reduces a
   tuple across a warp by reinterpreting it as an array of 32-bit words and
   shuffling one word at a time. That is formally undefined behaviour; nvcc does
   not act on it and Clang does. At `-O2` Clang's TBAA concluded the punned
   stores could not alias the struct and deleted them, so **every AMReX GPU
   min/max reduction returned zero** — with the right answer available in memory
   the whole time. CuMetal already compiled generated MSL with
   `-fno-strict-aliasing` for the same class of problem on Apple's compiler; it
   now does the same for device code on the CUDA→PTX side, matching nvcc.
   Device-only: the host compile keeps its own aliasing rules, as under nvcc.

3. **`__umul64hi` was unsupported by the PTX→LLVM lowering**, so the first AMReX
   kernel launched refused to compile. AMReX's `FastDivmodU64` replaces the
   64-bit division in every 3-D `ParallelFor`'s index decomposition with a
   multiply-and-shift built on it — it is on the hot path of every AMReX GPU
   kernel. Implemented from 32-bit limbs rather than by widening to `i128`,
   which Metal has no type for, along with the signed `__mul64hi`.

4. **`copysign` on a `double` was refused.** Unlike a transcendental, it is pure
   sign-bit manipulation of the binary64 storage word and is exact in every FP64
   mode; it was simply missing.

5. **No `long` overloads in the warp shuffle family.** On LP64 `long` and
   `long long` are distinct types of the same width, and CUDA declares the
   shuffles for both. With only the `long long` overload, shuffling a `long` was
   ambiguous against the `int` one. AMReX warp-reduces a `long` counter.

6. **`cuPointerGetAttributes` (the batch form) and `__nanosleep` were missing.**
   `AMReX::isManaged` uses the first; AMReX's `FillBoundary` device-side lock
   loop uses the second. The batch query's contract differs from the singular
   one in a way callers depend on: an attribute that does not apply leaves its
   slot untouched and does not fail the call.

7. **`curand_uniform_double` could not be lowered.** It converted a 53-bit
   integer to `double`, which lowers to PTX `cvt.rn.f64.u64`, and the FP32-pair
   emulation has no primitive for that. Rebuilt from the bit pattern — set the
   exponent, fill the mantissa, subtract one — which needs no integer-to-float
   conversion, is faster, and does not lose the low mantissa bits the way
   widening a binary32 draw would. `curand_normal_double`,
   `curand_log_normal_double`, and the Philox variants were added at the same
   time; AMReX's `amrex::RandomNormal` calls them whenever `Real` is `double`.

Separately, AMReX needed the nvcc shim to accept `--expt-extended-lambda`,
`--expt-relaxed-constexpr`, `-maxrregcount`, `--use_fast_math`, and
`--generate-line-info`, and needed
`__nv_is_extended_device_lambda_closure_type` to exist — CuMetal defines
`__NVCC__` for source compatibility, and AMReX gates on it to detect nvcc's
extended device lambdas. Clang's `__device__` lambdas are ordinary closure
types, so the answer is `false` for every type, spelled as an expression that
depends on `T` because AMReX feeds it to `std::enable_if_t` in a partial
specialization.

## Scope and limits

- One tutorial, one problem, one refinement level. This is not a statement about
  AMReX's AMR machinery, particles, embedded boundaries, or linear solvers, none
  of which this demo runs.
- Single process and single GPU. `AMReX_MPI=OFF`; multi-rank AMReX is untested.
- `double` throughout, on the FP32-pair FP64 emulation. Arithmetic and `sqrt`
  carry a ~48-bit significand; `exp`, `log`, and the trigonometric family
  evaluate through binary32. For a heat equation that is visible only in the
  initial condition, and the gate reports it separately for that reason. A
  stiff problem would not be so forgiving.
- Performance is not measured here. `CUMETAL_USE_METAL_DEVICE_ADDRESSES=1`
  serializes dispatches, so any timing from this configuration would describe
  that mode rather than the compiler.
- The next rung — PelePhysics' chemistry integrator and a PeleLMeX flame — is
  not attempted.

## File index

| Path | Purpose |
| --- | --- |
| [`run.sh`](run.sh) | Fetch, build, run, and gate |
| [`heat/CMakeLists.txt`](heat/CMakeLists.txt) | Build glue for the tutorial source |
| [`../../scripts/build_amrex_cumetal.sh`](../../scripts/build_amrex_cumetal.sh) | Build the CUDA and CPU-reference trees |
| [`../../tests/cuda_projects/amrex_device_idioms`](../../tests/cuda_projects/amrex_device_idioms) | Regression harness for the device-side defects above |
| `out/` | Generated inputs, build logs, run directories, plotfiles, and `fcompare` output |
