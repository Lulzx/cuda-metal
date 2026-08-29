# Library shim status

[Status index](../status.md) · [Known library gaps](../known-gaps/libraries.md)

CuMetal exports tested compatibility subsets from `libcumetal`; install-time
library-name aliases allow supported programs to resolve them. These are not
full NVIDIA library implementations.

## Implemented subsets

- **cuBLAS / cublasLt:** selected BLAS1/2/3, GEMM variants, batched/strided
  operations, pointer modes, descriptors/layouts, heuristics, and bounded
  epilogues. Metal/MPS paths exist for selected FP16, FP32, and FP64 behavior.
- **cuRAND:** pseudorandom generation for integer, uniform, normal, and
  log-normal families with seed, offset, generator lifetime, and stream ordering.
- **cuFFT:** rank-1 planning/execution for selected C2C/R2C/C2R and double forms;
  Accelerate backs supported lengths and a bounded direct DFT covers small
  fallback lengths.
- **cuSPARSE:** selected dense/sparse descriptors and operations, including GPU
  paths used by the HiGHS/cuPDLP-C integration.
- **cuSOLVER:** a bounded dense/sparse solver API subset.
- **cuDNN:** selected tensor, activation, pooling, convolution, softmax, and
  related descriptor/operation behavior.
- **NCCL / NVML:** single-device compatibility/query subsets.
- **Thrust / CUB:** header subsets; some algorithms are CPU/sequential over UMA
  and are labeled accordingly.
- **NVTX:** no-op annotation compatibility.

## Precision and execution labeling

Selected library operations use MPS/Metal, others use CPU or Accelerate over
shared memory. Documentation and provenance must distinguish them. FP64 GPU
paths must identify `fast48`, `wide48`, `ieee64`, or reduced-precision library
behavior and may not claim native binary64 Metal execution.

Focused functional tests cover positive, negative, pointer-mode, stream, graph,
datatype, and provenance behavior for the supported cases. The remaining audit
matrix is tracked in the library gaps page.
