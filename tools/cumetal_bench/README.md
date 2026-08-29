# cumetal_bench

`cumetal_bench` compares direct Metal-backend kernel launch timing against launch timing
through the CuMetal CUDA runtime shim.

The `--all-kernels` release set is:

- vector add, SAXPY, STREAM copy, STREAM triad, and FP32 reduction
- Single-kernel benchmark (`vector_add` by default)
- Validates output correctness in both paths
- Prints fastest synchronized milliseconds, sample spread, and runtime/native ratio
- Optional perf gate via `--max-ratio` (non-zero exit on regression)

Usage:

```bash
./build/cumetal_bench \
  --metallib tests/air_abi/reference/reference.metallib \
  --kernel vector_add \
  --elements 262144 \
  --warmup 5 \
  --iterations 50 \
  --max-ratio 2.0
```

The metallib must already exist; generate it with:

```bash
./scripts/generate_reference_metallib.sh
```
