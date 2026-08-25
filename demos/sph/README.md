# 3D SPH dam break on CuMetal

A DualSPHysics-style dam break — ~200k particles, weakly-compressible SPH, solved
*and* rendered by CUDA kernels, running on an Apple GPU through CuMetal.

Nothing here is a Metal shader. It is stock CUDA C++ (`__global__` kernels,
`<<<grid, block>>>` launches, `__shared__`, `__syncthreads()`, `atomicAdd`,
`atomicMin`, `float4`) compiled with clang's CUDA front end and executed by
`libcumetal`.

## What runs on the GPU

Per timestep, eight kernel launches:

| kernel | job |
| --- | --- |
| `k_fill_uint` | clear the cell histogram |
| `k_hash_count` | cell hash per particle + `atomicAdd` histogram |
| `k_scan_block` / `k_scan_blocksums` / `k_scan_add` | exclusive prefix sum over the cell histogram (shared-memory Hillis–Steele) |
| `k_scatter` | counting sort — reorders every particle array into cell order |
| `k_forces` | 27-cell neighbour sweep: continuity + pressure + artificial viscosity |
| `k_integrate` | Euler–Cromer update, density clamp, containment |

Per frame, three more:

| kernel | job |
| --- | --- |
| `k_clear` | clear depth + colour buffers |
| `k_splat_depth` | sphere-impostor splat, `atomicMin` depth pass |
| `k_splat_shade` | second pass, shades only the pixels the particle won |

The host builds the scene, drives the loop, and stamps the 2D chrome
(wireframe, colourbar, timer) onto the finished RGB frame.

## Physics

Weakly-compressible SPH:

- Wendland C2 kernel, `h = 1.3 dp`, support `2h` (~74 neighbours)
- Tait equation of state, `gamma = 7`, `c0 = 10 sqrt(2 g H)`
- Monaghan artificial viscosity, `alpha = 0.10`
- delta-SPH density diffusion (Antuono/Marrone), `delta = 0.10`, fluid-fluid
  only -- without it the density field goes noisy and the run walks into an
  instability around t = 0.9 s
- Dynamic boundary conditions (DualSPHysics DBC): 3 layers of boundary
  particles that carry density through the continuity equation, clamped to
  `rho >= rho0` so they only ever push
- Hydrostatic initial density profile, so the column does not start by
  radiating a pressure shock
- Uniform grid, rebuilt with a full GPU counting sort every step

Default scene: 3.2 x 1.4 x 1.0 m tank, 1.2 x 1.4 x 0.55 m water column, a solid
obstacle at x = 1.8 m. `dp = 0.021 m` gives 97,812 fluid + 103,853 boundary
particles, and 60,000 timesteps for 1.5 s of physics (~6.4 ms/step on an
M4 Pro, 16-core GPU).

## Running it

```bash
./demos/sph/run.sh --selftest     # numerical gate, seconds
./demos/sph/run.sh                # full render -> demos/sph/out/dambreak.mp4
```

Useful flags: `--dp`, `--frames`, `--dt`, `--steps-per-frame`, `--alpha`,
`--delta`, `--vscale`, `--width/--height`, `--az/--el/--dist`, `--no-video`,
`--ppm <file>`.

`ffmpeg` must be on `PATH` (the frames are piped to it as raw RGB); `--no-video`
skips it.

## Gates

`--selftest` builds a small scene and runs 20 steps twice: once through the full
GPU pipeline, once through a host brute-force O(N^2) SPH reference in the same
file. Particles are matched by position and compared:

```
selftest: matched 877/877 particles
  max |dx|      = 0.000e+00 m   (dp = 0.050 m)
  max |dv|      = 9.313e-10 m/s
  max drho/rho0 = 0.000e+00
PASS: GPU SPH matches host reference
```

The full run gates on physics, not on "it didn't crash":

- front speed within [0.6, 3.0] x `sqrt(2 g H)`, the shallow-water dam-break
  speed
- peak density excursion over the whole run within +/-20% of `rho0`. This is
  the single worst particle out of ~6e9 particle-steps, and it peaks during the
  impact on the far wall -- a loose bound that catches a blow-up, not ordinary
  impact compression
- **final-frame** instantaneous density spread within +/-8% of `rho0`. This is
  the real stability statement: an unstable WCSPH run cannot satisfy it, because
  its extremes keep growing instead of relaxing

A diverging run aborts mid-flight with `FAIL` rather than writing a
plausible-looking video.

## Notes

- `CUMETAL_TRACE_GPU=1` prints per-launch provenance. Every kernel reports
  `device=apple_gpu device_name="Apple M4 Pro" semantic_quality=exact` — no
  approximate kernels, no host fallback.
- Timestep is CFL-bound by the artificial sound speed. `dt = 2.5e-5` is stable;
  `5e-5` drifts and then runs away, which the density gate catches.
- Everything is single precision. FP64 would hit CuMetal's Dekker emulation.
