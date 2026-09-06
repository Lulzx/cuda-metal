# Matmul performance: article reproduction and CUDA tuning

Measured 2026-09-06 on **Apple M4 Pro, 16 GPU cores**, macOS 26.7,
Xcode 26.6, CUDA Clang 23.1.0. Release runtime, binary shim OFF. Base commit
`08bce3090256a7fc7d2b21ba4c7cdaf7fa9083f3` with existing uncommitted
compiler/runtime work. These are local-worktree results, not a clean-release
benchmark. [Exact metadata](evidence/matmul-2026-09-06/metadata.json).

Run the [matmul demo](../demos/matmul/README.md) with
`bash demos/matmul/run.sh` to build, validate, and repeat this comparison.
Use `--quick` for a smaller smoke run.

This page records the original source-tuning experiment. The later
[compiler optimization](compiler-performance.md) has separate before/after evidence.

## Outcome

The article's basic optimization story reproduces, but its final vectorization
step is slower on this machine. A new portable CUDA tile variant improves on
that final kernel by **1.48x**, reaching **2.03 TFLOP/s**, or **40.1% of MPS**.
CuMetal's existing `cublasSgemm` already reaches **99.8% of direct MPS** because
it calls MPS. This is an explicit library path, not compiler-generated GEMM.

No compiler/runtime implementation was changed for this experiment. The
improvement lives in a reusable CUDA kernel in
[`demos/matmul/kernels.cu`](../demos/matmul/kernels.cu).
It is not automatically applied to arbitrary CUDA kernels.

## Matched 4096 x 4096 x 4096 FP32 comparison

Each row uses the median of **three round means**, each with **20 timed
iterations after two warmups**. Variant order rotates between rounds.
GFLOP/s is derived from that time using `2*M*K*N`. Times are synchronized wall
times including launch/encode/wait, excluding compilation, allocation and
transfers. They are not GPU timestamp-only kernel times.

| Implementation | Time (ms) | GFLOP/s | MPS fraction |
|---|---:|---:|---:|
| Article naive | 406.266 | 338.3 | 6.7% |
| Naive with coalesced thread axes | 298.756 | 460.0 | 9.1% |
| Article shared-memory tiled | 176.548 | 778.5 | 15.4% |
| Article 2x2 register tiled | 92.346 | 1,488.3 | 29.4% |
| Article float4 register tiled | 100.047 | 1,373.7 | 27.1% |
| **New CUDA 64x64 tile, K=32** | **67.671** | **2,031.0** | **40.1%** |
| Accelerate SGEMM | 47.927 | 2,867.7 | 56.7% |
| CuMetal cuBLAS -> MPS | 27.198 | 5,053.2 | 99.8% |
| Direct MPS | 27.157 | 5,060.9 | 100% |

[Round results](evidence/matmul-2026-09-06/results.json) and
[full timing/check logs](evidence/matmul-2026-09-06/timings.log) retain variation.
For example, register tiling's three means are 92.12, 92.35, and 97.27 ms.
These measurements do not establish clock/thermal-independent peak throughput.

## What is and is not true in the article

Source: Kartik Sirohi,
[From 44 GFLOP/s to 376 GFLOP/s](https://medium.com/@kartik.orion.dev/from-44-gflop-s-to-376-gflop-s-learning-gpu-performance-through-matmul-a88bcf2b7092),
using the text/kernel listings supplied by the user.

- **The M1 time/throughput arithmetic is consistent.** The reported final
  365.357 ms gives 376.177 GFLOP/s; the overall 8.43x ratio is correct. It is
  plausible, but this M4 Pro cannot independently verify those M1 measurements.
  The original host benchmark, compiler version, timing boundaries, correctness
  checks, GPU core count, and thermal conditions were not supplied.
- **The T4 final row is inconsistent.** At 4096 cubed, 150.438 ms means
  **913.592 GFLOP/s**, not 756.53. The latter implies **181.670 ms**. The listed
  final time yields a 7.60x speedup over the listed naive time, while the listed
  throughput yields approximately 6.29x. No NVIDIA execution was performed here.
- **Tiling and register reuse help here**, but speedups are architecture and
  implementation dependent. Our four article variants progress
  338 -> 778 -> 1488 -> 1374 GFLOP/s, not monotonically through vectorization.
- **The naive-to-tiled comparison changes more than tiling.** The naive code
  maps thread x to output rows; tiled maps x to columns. Swapping only those
  axes gives 1.36x here, so the whole naive-to-tiled gain cannot be attributed
  exclusively to shared-memory reuse.
- **The float4 fallback is only bounds-safe.** For arbitrary K/N, a row may
  begin at an address not aligned to 16 bytes even when four elements remain.
  The harness rejects that variant unless both strides are multiples of four.
  The article's 4096-wide case satisfies that requirement.
- **376 versus 1372 is approximately 27.4%**, so that M1 comparison is
  arithmetically correct. It compares a teaching kernel with a tuned library;
  it is not a measurement of CUDA-to-Metal translation overhead in isolation.
- Accelerate is a tuned CPU-side library, not a scalar triple loop. Its internal
  acceleration choices were not profiled; do not infer ordinary CPU-core FP32
  efficiency from this result alone.

## Why the new CUDA kernel is faster

The article uses a 16x16 output tile with 2x2 output per thread. The selected
variant uses a **64x64 output tile, 32-element K tile, 256 threads, and 4x4
accumulators per thread**. Cooperative loading feeds more output reuse, each
barrier covers twice as much K work, and strided per-thread columns preserve
adjacent-lane memory access. Tail loads are zero-filled and stores are guarded.

This reduces the MPS gap from **3.68x to 2.49x** and improves over the article's
faster scalar-register variant by **1.36x**. It still does not reach MPS.
More tiling is not universally better: the 64x64/K32 variant with 8x8
accumulators measured only about 233 GFLOP/s in the exploratory sweep. Register
pressure/spilling is a hypothesis; hardware counters were not collected.

A directly written Metal implementation of the same selected algorithm measured
**71.592 ms / 1,919.7 GFLOP/s** over 20 iterations, versus approximately 67.7 ms
for translated CUDA. This limited control provides no evidence of a large
translation penalty for this algorithm. It is one control run, not proof of
compiler parity across kernels. [Native control log](evidence/matmul-2026-09-06/native.log).
Apple exposes [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
as a separate optimized matrix operation; matching it with custom kernels would
require further strategy/instruction-level work and profiling.

## Validation and reproduction

All timed GPU variants compare every output element with Accelerate on identical
seeded signed random input; the CPU reference is also checked against 64 double
precision dot products. All recorded GPU comparisons had **zero output error**.
That observation is input-specific, not a universal bit-exactness guarantee.
No reduced-precision multiplication or accumulation was selected.

The standalone test covers six rectangular/tail/small shapes across every
applicable variant, including 1x1x1 and 129x67x131, plus invalid dimensions,
unknown kernel names, and rejection of unsafe vector alignment. Article vector
runs are skipped on unaligned strides; the other variants must pass.
[Correctness log](evidence/matmul-2026-09-06/correctness.log).
A separate trace records `device=apple_gpu`, `launch_success=true`, and the
precompiled-metallib path for the selected kernel.
[Provenance log](evidence/matmul-2026-09-06/provenance.log).

See the [benchmark README](../tools/matmul_bench/README.md) for complete commands,
error thresholds, source/host-path scope, and native Metal control. The focused
benchmark tests passed on Release/shim-off. The repository-wide CTest suite
was not rerun: this change adds a standalone experiment and documentation and
preserves the pre-existing compiler/runtime work.

## Follow-up: compiler optimization

The measurements above precede the bounded private-array compiler pass. A
[separate compiler study](compiler-performance.md) tests the unchanged
large-accumulator kernel before and after that pass. It is distinct from the
source-level tiling improvement measured here.
