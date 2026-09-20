# CuMetal demos

These demos exercise increasingly demanding CUDA workloads on Apple Silicon.
Each runnable gate checks numerical or domain-specific output and Apple-GPU
provenance; process exit alone is not considered evidence of correct execution.

## Apollo demo

Apollo is the front door: one command climbs from vector addition through
reduction, SGEMM, and a path tracer, refusing any stage that lacks
`device=apple_gpu` provenance:

```bash
bash demos/apollo/run.sh
```

`bash demos/apollo/run.sh --full` also runs the llm.c GPT-2 FP32 gate. See
[the Apollo guide](../demos/apollo/README.md) for scope limits and artifacts.

## CUDA matmul vs MPS

Reproduce the matmul article's kernel progression, check numerical output and
Apple-GPU provenance, and compare a tuned CUDA kernel with MPS and Accelerate:

```bash
bash demos/matmul/run.sh --quick   # 512-cubed smoke comparison
bash demos/matmul/run.sh           # 4096-cubed, 3 rounds x 20 iterations
```

The recorded M4 Pro result is 2.03 TFLOP/s for the tuned CUDA kernel (1.48x the
article's final kernel), versus 5.06 TFLOP/s for MPS. The explicit CuMetal cuBLAS
path already calls MPS. See [the matmul demo guide](../demos/matmul/README.md)
for runnable modes and [the study](matmul-performance.md) for raw evidence,
article corrections, and hardware-specific limits.

## 3D Gaussian Splatting

The Inria tile-based Gaussian rasterizer (`renderCUDA` plus binning kernels)
runs on Apple Silicon and writes an image with `device=apple_gpu` provenance:

```bash
bash demos/3dgs/run.sh
open demos/3dgs/out/gaussians.png
```

Preprocessing currently runs on the host; tile sorting and blending run on the
GPU. See [the 3DGS guide](../demos/3dgs/README.md) for exact scope and limits.

## 3D SPH dam break

This is a roughly 200,000-particle weakly-compressible SPH simulation in the
DualSPHysics style. The neighbour search, physics, and particle rasterizer are
all CUDA kernels:

```bash
bash demos/sph/run.sh --selftest    # GPU vs host brute-force SPH reference
bash demos/sph/run.sh               # 1920x1080 60 fps -> demos/sph/out/dambreak.mp4
```

It uses `__shared__`/`__syncthreads()` prefix sums, `atomicAdd` counting sort,
and an `atomicMin` depth pass. The gate checks dam-break front speed and density
drift, not merely whether the process ran. See
[the SPH guide](../demos/sph/README.md).

## Tiny diffusion model

A roughly 310,000-parameter DDPM is trained on MNIST in PyTorch and then sampled
entirely by hand-written CUDA kernels: 1,000 denoising steps, measured at about
13 seconds for 16 images on the verified machine.

```bash
python3 demos/diffusion/train.py    # about 4 min on MPS -> out/model.bin
./demos/diffusion/run.sh --check    # forward pass vs the PyTorch reference
./demos/diffusion/run.sh            # sample -> demos/diffusion/out/samples.png
```

`--check` requires `max |cumetal - pytorch| < 2e-3`; the recorded result was
`5.2e-06`. Building it exposed a silent wrong-answer bug in PTX-to-MSL float
typing. See [the diffusion guide](../demos/diffusion/README.md).

## HiGHS / cuPDLP-C linear programming

The HiGHS demo builds the pinned CuMetal integration and compares the frozen
`afiro` problem on CPU and Apple GPU. It requires matching solver status and
objective, bounded primal/dual residuals, and successful GPU provenance:

```bash
bash demos/highs/run.sh
```

`wide48` and `ieee64` pass the recorded residual gate. `fast48` reaches Optimal
but misses the dual-residual limit, which is reported as a precision failure
rather than hidden. The `lpfeas/` harness fetches and reports a frozen
Mittelmann feasibility corpus. See [the HiGHS guide](../demos/highs/README.md)
for build pins, commands, FP64 semantics, and the current mixed cuSPARSE
precision boundary.

## GROMACS molecular dynamics

The GROMACS demo builds the unmodified 2025.4 release twice from one source --
`GMX_GPU=CUDA` against CuMetal and `GMX_GPU=OFF` as the reference -- and runs
benchmark systems from the public GROMACS benchmark set through both:

```bash
bash demos/gromacs/run.sh --quick    # villin, 5k atoms
bash demos/gromacs/run.sh            # adds rnase, 24k atoms
```

Short-range nonbonded, listed forces, the whole PME step including its 3D FFT,
and the LINCS/SETTLE constrained update all run on the Apple GPU. The gate
compares every energy term GROMACS prints at every step of a deterministic
20-step trajectory, and additionally requires that GROMACS's log shows all four
tasks offloaded and that CuMetal traced `device=apple_gpu`. The recorded results
are a maximum relative energy difference of `2.66e-05` for villin (5,006 atoms)
and `6.80e-05` for rnase_cubic (24,040).

The same guide records matched water-box performance comparisons against native
Metal and AdaptiveCpp Metal, explains `ns/day`, and keeps nonbonded-only and
full-GPU task placements separate. CuMetal wins every currently recorded
matched pair; the paired all-cases corpus and a full-GPU AdaptiveCpp/Metal FFT
route remain open rather than being inferred from those bounded results.

Building it exposed five CuMetal defects, all silent or fatal rather than
warned: `cudaDeviceReset` erasing the fatbin kernel registry, every host-backed
`cub::Device*` shim ignoring stream order, `cudaDestroyTextureObject(0)`
returning an error instead of a no-op, zero-parameter kernels rejecting a null
argument vector, and missing libdevice entry points. Offloading PME then
required cuFFT to grow rank-2 and rank-3 transforms and the advanced data layout
that describes a padded grid, and then a Metal implementation of the transform
itself. See
[the GROMACS guide](../demos/gromacs/README.md).

## AMReX block-structured AMR

AMReX 25.09's CUDA GPU backend builds unmodified against CuMetal. The demo
builds AMReX twice from one source tree -- `AMReX_GPU_BACKEND=CUDA` against
CuMetal and `AMReX_GPU_BACKEND=NONE` as the reference -- and runs the
`HeatEquation_EX0_C` tutorial through both:

```bash
bash demos/amrex/run.sh --quick    # 20 steps
bash demos/amrex/run.sh            # 200 steps
```

The Gaussian initial condition, the three-direction flux computation, the
flux-divergence update, and the ghost-cell exchange are all `amrex::ParallelFor`
kernels on the Apple GPU. The gate compares the two builds' plotfiles with
AMReX's own `fcompare` at step 0 and at the final step -- separately, because
the initial condition isolates CuMetal's FP64 transcendentals while the final
field is what a wrong stencil or a dropped launch moves. The recorded results on
an M4 Pro are a maximum relative difference of `2.22e-08` at step 0 and
`5.53e-09` after 200 steps, against a `1e-6` gate; the same binary without the
required address mode lands at `1.0`, so the gate has eight orders of magnitude
of margin.

AMReX needs `CUMETAL_USE_METAL_DEVICE_ADDRESSES=1`, because its reductions and
fused multi-box kernels dereference `Array4` pointers held in device memory
rather than passed as launch arguments -- the same requirement PhysX has.

Building it exposed seven CuMetal defects, all silent: `std::`-qualified math
was not usable in device code at all; Clang's strict aliasing deleted the punned
word-wise shuffles behind every AMReX GPU min/max reduction, which returned zero
instead; `__umul64hi` and `copysign(double)` were unsupported lowering targets;
the warp shuffle family had no `long` overloads; `cuPointerGetAttributes` and
`__nanosleep` were missing; and `curand_uniform_double` could not be lowered.
See [the AMReX guide](../demos/amrex/README.md) for the scope limits, the
measured FP64 accuracy, and what the demo does not claim.

## Run one sample

```bash
./build/cumetalc samples/vectorAdd/vectorAdd.cu -o /tmp/vectorAdd
CUMETAL_TRACE_GPU=1 /tmp/vectorAdd
```

The program should print a numerical `PASS`. The trace must contain a
`CUMETAL_PROVENANCE` record with `device=apple_gpu` and `launch_success=true`.
A correct number without GPU provenance is not proof of GPU execution.

Generated MSL is compiled under CUDA's floating-point contract: IEEE
comparisons with NaN and infinity preserved, FMA contraction allowed. Apple's
compiler defaults to fast math, under which `x != x` is false for NaN; set
`CUMETAL_MSL_MATH_MODE=fast` (or pass `--use_fast_math` to `cumetalc`) to opt
into that. GPU provenance reports the selected `math_mode`. Precompiled
metallibs retain the policy used when they were built.
