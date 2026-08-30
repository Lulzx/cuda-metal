# Library shim status

[Status index](../status.md) · [Known library gaps](../known-gaps/libraries.md)

CuMetal exports tested compatibility subsets from `libcumetal`; install-time
library-name aliases allow supported programs to resolve them. These are not
full NVIDIA library implementations.

## Implemented subsets

- **cuBLAS / cublasLt:** selected BLAS1/2/3, GEMM variants, batched/strided
  operations, pointer modes, descriptors/layouts, heuristics, and bounded
  epilogues. Metal/MPS paths exist for selected FP16, FP32, and FP64 behavior.
  The bounded CPU/Accelerate cuBLASLt FP32/FP64 column-major path synchronizes
  its supplied CUDA stream and validates transpose, epilogue, datatype,
  compute/scale type, dimensions, leading dimensions, matrix geometry, batch
  counts/strides, and integer bounds before writing output. Unsupported layout
  orders, mixed datatypes, overlapping batch strides, FP64 epilogues, and
  opaque algorithms fail explicitly. Tracked unified-memory operands and bias
  vectors must cover their complete matrix, stride, and batch footprints;
  ordinary host buffers remain supported by this explicit CPU fallback.
- **cuRAND:** the default and MTGP32 compatibility generators provide integer,
  uniform, normal, log-normal, Poisson, and exponential output with seed,
  offset, generator lifetime, and stream ordering. MTGP32 has exact host/device
  self-consistency for the enrolled NVIDIA sample but is not claimed to match
  NVIDIA's bitstream. Other named generator descriptors remain introspectable,
  while generation from algorithms whose sequences are not implemented
  (including explicit MT19937, XORWOW, Philox, and Sobol forms) returns
  `CURAND_STATUS_TYPE_ERROR`. Normal/log-normal calls enforce finite parameters
  and even lengths. Device generation validates the complete typed output span,
  including interior pointers and multiplication-overflow-sized requests,
  before enqueueing any write. Pseudo-only seed, pseudo/quasi ordering, and
  quasi-dimension setters reject generator-type mismatches with CUDA-compatible
  statuses.
- **cuFFT:** rank-1 to rank-3 planning and execution for C2C/R2C/C2R and the
  double forms, with cuFFT's advanced data layout (`inembed`/`onembed`, stride,
  batch distance). The single-precision transforms run on the Apple GPU
  (Stockham autosort, Bluestein for non-power-of-two lengths); `CUMETAL_FFT_METAL`
  selects auto/always/never and `CUMETAL_DEBUG_FFT` reports the routing. Small
  grids and the double-precision forms use the Accelerate/Bluestein CPU path.
  Planning rejects overflowing advanced layouts without retaining a partial
  handle, and execution validates the full typed input/output allocation spans
  (including batch distances, padding, strides, and interior pointers) before
  either backend reads or writes unified memory.
- **cuSPARSE:** selected dense/sparse descriptors and operations, including GPU
  paths used by the HiGHS/cuPDLP-C integration. The implemented SpMV, SpMM,
  legacy CSR SpMV, and SpSV scalar inputs honor host/device pointer mode;
  tracked device scalars are stream-ordered, and captured SpMV reads device
  scalar values at replay rather than freezing them at capture. Generic
  descriptors reject invalid shapes, pointers, leading dimensions, and enum
  values; unsupported mixed datatype combinations fail explicitly. Legacy
  real CSR SpMV implements non-transpose, transpose, and conjugate-transpose
  and validates CSR bounds before modifying the output.
- **cuSOLVER:** a bounded dense/sparse solver API subset. Dense CPU/Accelerate
  calls synchronize both explicit and default CUDA streams before reading UMA
  operands. Implemented workspace queries validate handles, shapes, leading
  dimensions, enum values, output pointers, and integer-overflow bounds; the
  matching execution calls reject invalid handles, operations, fills, jobs,
  leading dimensions, and undersized workspaces before invoking LAPACK. Sparse
  Cholesky/QR fallbacks validate CSR offsets, column bounds, dimensions,
  tolerance, and allocation sizes before changing outputs, honor the
  singularity tolerance, and order reads from explicit and default streams.
- **cuDNN:** selected FP32/NCHW tensor, activation, pooling, convolution,
  softmax, batch-normalization, dropout, reduction, RNN, and bounded attention
  behavior. CPU/Accelerate-backed execution orders UMA reads against the
  handle's explicit or default CUDA stream. Descriptor geometry, supported
  layouts/datatypes, convolution groups/output shapes/workspaces, and matching
  tensor shapes are checked before output mutation; unsupported combinations
  fail explicitly. Regression coverage includes nonzero-`beta` instance
  softmax, backward-convolution accumulation, invalid descriptors, undersized
  workspaces, and stream-ordered input. The convolution family accepts only the
  contiguous NCHW layout it actually computes, uses overflow-checked
  forward/backward workspace geometry, and validates complete tracked
  tensor/filter/workspace spans before CPU access. Implemented activation,
  tensor, softmax, pooling, dropout, normalization, OpTensor, and reduction
  paths likewise validate tracked tensor, scalar, parameter, and reserve spans.
  The bounded forward RNN path consistently interprets contiguous
  `N x C x 1 x 1` timestep descriptors with `C` as the input feature count,
  validates every timestep and state descriptor, uses overflow-checked
  parameter/workspace/reserve geometry, and rejects short tracked input,
  weight, output, state, or scratch allocations before CPU access. Only
  zero-dropout, standard-algorithm FP32 linear-input RNN/GRU/LSTM modes are
  accepted. The bounded attention forward path accepts projection-free,
  dropout-free FP32 canonical time/batch/beam/vector descriptors; it validates
  configured maxima, checked tensor spans, non-overlapping output, and rejects
  residuals, windows, incremental mode, variable lengths, and scratch buffers
  that it does not implement. Projection-weight size and offset queries use
  checked arithmetic and reject undersized tracked or declared buffers.
- **NCCL / NVML:** single-device compatibility/query subsets. NCCL's one-rank
  collectives are stream-ordered identity copies with validated communicators,
  roots, reduction operations, datatypes, byte-count overflow, group state,
  and exact one-device initialization; multi-rank/device requests fail without
  partially initializing communicator arrays. NVML validates its synthetic
  device handle, reference-counts initialization, reports undersized strings,
  and returns `NOT_SUPPORTED` for utilization, temperature, power, and clocks
  that public macOS APIs cannot supply instead of fabricating measurements.
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
