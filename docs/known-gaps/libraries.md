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
  batched, complex, tensor, or capture combinations are covered. The hardened
  cuBLASLt CPU fallback is a bounded FP32/FP64 column-major, exact-shape,
  non-overlapping strided-batch path; row-major/special layouts, mixed
  datatypes, FP16/TF32 Lt compute, broadcast batches, general algorithm objects,
  and FP64 epilogues are rejected rather than emulated.
- **cuRAND:** the default and MTGP32 compatibility generators are not claimed
  as NVIDIA bitstream parity; MTGP32 is proven only for host/device
  self-consistency in the enrolled NVIDIA sample. Named XORWOW, MRG32k3a,
  MT19937, Philox, Sobol, and scrambled Sobol descriptors can be created and
  queried, but generation is rejected explicitly until the named algorithm is
  implemented. Ordering modes and quasi dimensions are descriptor state, not
  proof of their sequence semantics. Implemented device generation rejects
  output counts that exceed the tracked allocation remainder before enqueue;
  complete distributions, state
  serialization, and device API parity remain open.
- **cuFFT:** ranks 1 to 3 execute for every transform type, including cuFFT's
  advanced data layout (`inembed`/`onembed`/stride/dist), which is what a padded
  grid such as GROMACS's PME mesh needs. The single-precision transforms
  (`C2C`/`R2C`/`C2R`) run on the Apple GPU as Stockham autosort passes, with
  Bluestein on the same kernels for lengths that are not a power of two; grids
  below a dispatch-cost threshold and every double-precision entry point stay on
  the CPU, since Metal has no FP64. Still absent: callbacks, multi-GPU, the rest
  of the Xt surface, and a GPU path for the double transforms. Implemented
  execution rejects untracked, host, interior-short, and otherwise undersized
  input/output spans before dispatch; caller-supplied work areas are accepted
  for API compatibility but unused because both backends manage scratch.
- **cuSPARSE/cuSOLVER:** selected operations only; descriptor, format, solver,
  analysis/reuse, and datatype matrices remain incomplete. cuSPARSE host/device
  scalar pointer mode is covered for the implemented SpMV, SpMM, legacy CSR
  SpMV, and SpSV paths, including replay-time device scalar reads for captured
  SpMV. Generic SpMV/SpMM validate operation, algorithm, layout, and homogeneous
  FP32/FP64 descriptor types; other mixed-type combinations return an explicit
  unsupported status. This does not establish coverage for absent routines or
  additional datatype combinations. The implemented cuSOLVER dense query and
  execution entry points validate their current argument/workspace surface;
  sparse Cholesky/QR additionally validate CSR structure and singularity
  tolerance. Sparse reordering is not implemented and nonzero `reorder` is
  rejected instead of being silently ignored. Broader dense/sparse routine,
  datatype, batched, analysis/reuse, and GPU execution coverage remains open.
- **cuDNN:** selected descriptors/operations only. The hardened CPU-backed
  surface is primarily contiguous FP32/NCHW; it synchronizes the handle stream
  before reading UMA operands and rejects unsupported types/layouts/shapes for
  the covered calls. Convolution is the tested implicit-GEMM cross-correlation
  path; general algorithm selection, convolution-mode filter reversal, general
  OpTensor broadcasting, fusion, training/backward breadth, graph integration,
  datatype, and layout coverage remain incomplete. RNN and attention are
  bounded compatibility paths rather than full cuDNN implementations.
- **NCCL:** single-device compatibility cannot provide collective multi-GPU
  semantics. The implemented one-rank collectives are identity copies, not a
  transport; only device zero and rank zero are accepted, point-to-point calls
  fail, and multi-device initialization is rejected atomically.
- **NVML:** compatibility queries cannot expose NVIDIA device management. The
  single synthetic device reports Apple unified system-memory information, not
  dedicated VRAM; utilization, temperature, power, and clock telemetry are
  explicitly unsupported through the public-API-only boundary.
- **Thrust/CUB:** several algorithms are sequential/CPU over UMA; device-wide
  performance and full template/API compatibility are not claimed. Those
  host-backed `cub::Device*` entry points do now synchronize their stream before
  reading the input, which is a correctness requirement rather than a
  performance choice: without it a scan or reduction of a buffer a kernel is
  still writing silently returns stale memory.
- **NVTX:** annotations are no-ops.

The closure target is a generated support table from actual positive and
negative cases, not a list of exported symbol names.
