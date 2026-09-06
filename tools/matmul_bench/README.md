# Matmul article reproduction and tuning

Reproduces the algorithms in Kartik Sirohi's
[44 to 376 GFLOP/s article](https://medium.com/@kartik.orion.dev/from-44-gflop-s-to-376-gflop-s-learning-gpu-performance-through-matmul-a88bcf2b7092).
The supplied kernel listings are expressed with shorter names and shared helpers;
this is an algorithm reproduction, not the author's unavailable host benchmark,
compiler revision, or M1 environment. Fixed launch sizes follow the article.

[`demos/matmul/kernels.cu`](../../demos/matmul/kernels.cu) includes the four article algorithms, an axis-swapped naive control,
and portable CUDA block-tile experiments. `block64_32` is the best balanced
configuration in this M4 Pro sweep: 64x64 output, K tile 32, 256 threads,
16 accumulators per thread. This is a source kernel optimization, not an
automatic compiler rewrite or replacement by a library call.

For a single-command build, validation, GPU provenance gate, and comparison,
use the [matmul demo](../../demos/matmul/README.md):
`bash demos/matmul/run.sh` (or `--quick`).

## Run

Build the checkout's compiler and runtime in Release first:

```bash
cmake -B build-matmul-release -DCMAKE_BUILD_TYPE=Release -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build build-matmul-release --target cumetalc cumetal_runtime -j6
CUMETAL_BUILD_DIR="$PWD/build-matmul-release" \
CUMETAL_MATMUL_OUT="$PWD/build-matmul-release/matmul-bench" \
bash tools/matmul_bench/test.sh
python3 tools/matmul_bench/measure.py "$PWD/build-matmul-release/matmul-bench" \
  --output "$PWD/build-matmul-release/matmul-bench/evidence"
```

For a single shape/variant, `run.sh M K N iterations variant` builds and runs it.
The already-built runner accepts `metallib M K N iterations variant`.
Use `all` to run every experiment; it includes deliberately unsuccessful tuning
choices. `measure.py` runs the comparison set in three rotated rounds of 20
iterations, saving logs and JSON. Do not run other GPU workloads concurrently.

## Measurement and correctness

- FP32 input, multiplication and accumulation; no half/TF32 conversion.
- Identical seeded signed random inputs for every variant, alpha=1, beta=0.
- Independent Accelerate SGEMM reference; 64 additional double-precision dot
  products check that reference. Every GPU output element is checked for finite
  values and error, with a NaN sentinel to expose missing stores.
- Error threshold: `2e-5*K + 2e-4*abs(reference)`; maximum absolute error and
  relative L2 error are always printed. The recorded runs had zero differences.
- Two warmups, followed by synchronized **wall time**, including launch/encode
  and wait overhead. Allocation, copies, source compilation and pipeline warmup
  are outside timed samples. Mean/median/min/max are reported, not fastest-only.
- Throughput is `2*M*K*N / (mean_ms*1e6)` in decimal GFLOP/s.
- `mps` uses direct MPSMatrixMultiplication with persistent descriptors/operator;
  `cublas_mps` measures CuMetal's existing cuBLAS-to-MPS implementation. The MPS
  reference is compiled only into this benchmark and lives inside the repository's
  Metal API boundary. `accelerate` measures the CPU library including any
  internal acceleration it chooses; it is not a scalar CPU baseline.
- The article `vectorized` variant is rejected when K or N is not divisible by
  four. Its bounds checks alone do not ensure 16-byte row alignment. Other
  variants cover arbitrary positive rectangular shapes and partial tiles.
- CLI dimensions are limited to 8192 to bound allocations and integer indexing.

The driver launches the source-compiled metallib through CuMetal's explicit
kernel API. It does not exercise native CUDA launch-stub generation. No NVIDIA
hardware is used. Set `CUMETAL_TRACE_GPU=1` for a separate provenance run; do not
mix tracing into the published timing runs.

## Native Metal control

[`demos/matmul/native.metal`](../../demos/matmul/native.metal) expresses the same blocked algorithm directly in Metal:

```bash
xcrun metal -fno-fast-math -ffp-contract=fast -c demos/matmul/native.metal -o /tmp/matmul-native.air
xcrun metallib /tmp/matmul-native.air -o /tmp/matmul-native.metallib
build-matmul-release/matmul-bench/matmul_bench /tmp/matmul-native.metallib 4096 4096 4096 20 block64_32
```

This control uses the same runtime launch path, isolating source translation
from algorithm choice. It is not MPS and does not use SIMD-group matrix intrinsics.

See [measured results](../../docs/matmul-performance.md) for provenance and limits.
