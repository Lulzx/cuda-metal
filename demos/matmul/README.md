# CUDA matmul vs MPS

Run the article's matrix multiplication progression on your Apple GPU, then
compare a tuned CUDA kernel with direct MPS, CuMetal cuBLAS, and Accelerate.

```bash
bash demos/matmul/run.sh --quick   # build + correctness + 512-cubed smoke comparison
bash demos/matmul/run.sh           # full 4096-cubed study, 3 rounds x 20 iterations
bash demos/matmul/run.sh --check   # correctness and GPU provenance only
```

Requires the usual [CuMetal build prerequisites](../../docs/build.md), Python 3,
and Apple's Metal/MPS/Accelerate frameworks. The runner configures and builds
`build-matmul-release` as Release with the binary shim OFF. To reuse another
build, set `CUMETAL_BUILD_DIR`; that directory is reconfigured with those settings.
The default full measurement can take several minutes, plus the first build.
Keep other GPU workloads idle while measuring.

The demo first checks six small/rectangular/tail shapes, then requires a
successful Apple-GPU provenance record and numerical verification from the
selected CUDA kernel. Timing runs have tracing disabled. A failed compile,
numerical check, provenance check, or measurement makes the command fail.
There is no hardware-independent performance pass threshold.

## What to look for

The measured M4 Pro study progressed through:

| Implementation | GFLOP/s |
|---|---:|
| Article naive | 338 |
| Article shared-memory tiled | 778 |
| Article 2x2 register tiled | 1,488 |
| Article float4 register tiled | 1,374 |
| Tuned CUDA 64x64 output tile / K32 | 2,031 |
| CuMetal cuBLAS -> MPS | 5,053 |
| Direct MPS | 5,061 |

The tuned source kernel was **1.48x faster than the article's final kernel**,
reaching **40.1% of MPS**. CuMetal's explicit cuBLAS SGEMM already uses MPS.
These are recorded M4 Pro results, not promises for other devices or the quick
512-cubed run. The study did not independently verify the author's M1 or T4
measurements. Vectorization was slower here, and the T4 final time/throughput
pair in the article is arithmetically inconsistent.

See the [full report](../../docs/matmul-performance.md) for all measurements,
article corrections, precision, timing boundaries, and remaining performance gap.

## Source and artifacts

The demo shares its implementation with [tools/matmul_bench](../../tools/matmul_bench/README.md):

- [kernels.cu](../../tools/matmul_bench/kernels.cu): article algorithms,
  axis-swapped naive control, and portable CUDA tile experiments.
- [main.cpp](../../tools/matmul_bench/main.cpp): host runner and correctness checks.
- [native.metal](../../tools/matmul_bench/native.metal): direct-Metal algorithm control.
- [MPS reference](../../runtime/metal_backend/matmul_bench_reference.mm): standalone
  comparison helper, kept inside the Metal API boundary.

Results are under `demos/matmul/out/full/`, `out/quick/`, or `out/check/`:
`correctness.log`, `provenance.log`, build logs, and (for measurements)
`report.md`, `measure.log`, `evidence/results.json`, and individual round logs.
Set `CUMETAL_MATMUL_DEMO_OUT` to change the parent output directory.

The demo compiles CUDA to a metallib and launches it through CuMetal's explicit
kernel API. It does not test CUDA launch-stub generation. The article algorithms
are re-expressed from the supplied listings; its original host benchmark was
not available. The float4 variant is restricted to aligned row strides; other
variants support rectangular tails. The optimization is in the source kernel,
not an automatic compiler substitution or reduced-precision path.
