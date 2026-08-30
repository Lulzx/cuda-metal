# Library shim gaps

[Known-gaps index](../known-gaps.md) · [Library status](../status/libraries.md)

No library shim has full NVIDIA parity. Every operation is bounded by tested
datatype, layout, pointer location, stream, capture, and error behavior.

## Cross-library gaps

- Pointer modes and scalar residency are not complete for every routine.
- Stream ordering and graph capture need per-operation coverage.
- Datatype/layout/stride/batch combinations outside focused tests may reject or
  remain unimplemented.
- CPU or Accelerate work over UMA must not be counted as Apple-GPU execution.
- FP64 routines can use reduced-precision Metal paths and must report semantic
  quality honestly.
- Workspace, algorithm-selection, tuning, determinism, and version-specific API
  behavior are narrower than NVIDIA implementations.

## Library-specific boundaries

- **cuBLAS/cublasLt:** incomplete routine/type/epilogue/algorithm surface; not all
  batched, complex, tensor, or capture combinations are covered.
- **cuRAND:** bounded generator families and ordering; no complete distribution,
  quasi-random, state-serialization, or device API parity.
- **cuFFT:** ranks 1 to 3 execute for every transform type, including cuFFT's
  advanced data layout (`inembed`/`onembed`/stride/dist), which is what a padded
  grid such as GROMACS's PME mesh needs. Lengths vDSP cannot factor go through
  Bluestein's algorithm rather than being rejected, so there is no unsupported
  length and no size ceiling. Still absent: callbacks, multi-GPU and the rest of
  the Xt surface, and any execution on the GPU -- the transform runs on the CPU
  over unified memory, so offloading an FFT here buys correctness, not Metal.
- **cuSPARSE/cuSOLVER:** selected operations only; descriptor, format, solver,
  analysis/reuse, and datatype matrices remain incomplete.
- **cuDNN:** selected descriptors/operations only; algorithm, fusion, training,
  graph, datatype, and layout coverage is far from full.
- **NCCL:** single-device compatibility cannot provide collective multi-GPU
  semantics.
- **NVML:** compatibility queries cannot expose NVIDIA device management.
- **Thrust/CUB:** several algorithms are sequential/CPU over UMA; device-wide
  performance and full template/API compatibility are not claimed. Those
  host-backed `cub::Device*` entry points do now synchronize their stream before
  reading the input, which is a correctness requirement rather than a
  performance choice: without it a scan or reduction of a buffer a kernel is
  still writing silently returns stale memory.
- **NVTX:** annotations are no-ops.

The closure target is a generated support table from actual positive and
negative cases, not a list of exported symbol names.
