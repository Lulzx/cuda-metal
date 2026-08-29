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

## Run one sample

```bash
./build/cumetalc samples/vectorAdd/vectorAdd.cu -o /tmp/vectorAdd
CUMETAL_TRACE_GPU=1 /tmp/vectorAdd
```

The program should print a numerical `PASS`. The trace must contain a
`CUMETAL_PROVENANCE` record with `device=apple_gpu` and `launch_success=true`.
A correct number without GPU provenance is not proof of GPU execution.

Runtime-compiled MSL preserves Metal's fast-math default. Set
`CUMETAL_MSL_MATH_MODE=safe` to request safe Metal math for JIT-compiled source;
GPU provenance reports the selected `math_mode`. Precompiled metallibs retain
the policy used when they were built.
