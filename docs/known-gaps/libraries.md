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

- **cuBLAS/cublasLt:** complex level-1/2/3 (`axpy`/`scal`/`dot`/`amax`/`asum`/
  `nrm2`/`ger`/`syrk`/`trsm`), `geam`, `dgmm`, `sbmv`, `tpttr`/`trttp`, batched
  `gemm`/`trsm`, and batched `getrf`/`getrs`/`getri` run on the CPU through
  Accelerate or plain loops. A NULL `PivotArray` selects unpivoted LU.
  `cublasSgemmEx` accepts only F32, F16 and BF16 operands. The cuBLAS 10
  `cudaDataType` compute-type overloads of `GemmEx`/`GemmStridedBatchedEx`
  map through `cublasMigrateComputeType`. Otherwise the routine/type/epilogue/algorithm surface is incomplete; not all
  batched, complex, tensor, or capture combinations are covered. The hardened
  cuBLASLt CPU fallback is a bounded FP32/FP64 column-major, exact-shape,
  non-overlapping strided-batch path; row-major/special layouts, mixed
  datatypes, FP16/TF32 Lt compute, broadcast batches, general algorithm objects,
  and FP64 epilogues are rejected rather than emulated. Tracked allocations are
  range-checked across their full strided-batch footprints; untracked host
  buffers remain accepted specifically for the CPU fallback. `cublasSetWorkspace`
  follows the CUDA contract (256-byte alignment, a span contained in one
  tracked allocation, NULL selects the default pool, `cublasSetStream` resets
  it unconditionally), but the recorded span is handle state only -- the
  backends still manage their own scratch rather than sub-allocating from it.
- **BLAS beta:** as in BLAS, routines do not read C (or `geam`'s B) when beta is
  zero, so NaN in a recycled output buffer cannot leak into the result.
- **cuRAND:** the default and MTGP32 compatibility generators are not claimed
  as NVIDIA bitstream parity; MTGP32 is proven only for host/device
  self-consistency in the enrolled NVIDIA sample. Named XORWOW, MRG32k3a,
  MT19937, Philox, Sobol, and scrambled Sobol descriptors can be created and
  queried, but generation is rejected explicitly until the named algorithm is
  implemented. Ordering modes and quasi dimensions are descriptor state, not
  proof of their sequence semantics. Implemented device generation rejects
  output counts that exceed the tracked allocation remainder before enqueue;
  complete distributions, state
  serialization, and device API parity remain open. In `curand_kernel.h`,
  `curand_uniform` returns (0, 1] as CUDA does. XORWOW, Philox and MRG32k3a
  have uniform, normal (float and double), log-normal and Poisson
  distributions. MRG32k3a's first draw from its default state matches the
  L'Ecuyer reference, but `curand_init` derives each state by hashing seed and
  subsequence rather than skipping 2^76 steps, so its streams are independent
  but not NVIDIA's sequence.
- **CPU-backed library calls and streams:** cuFFT, cuBLAS, cuBLASLt, cuSPARSE,
  cuSOLVER, cuDNN and cuRAND compute some calls on the CPU. Those calls first
  wait for the handle's stream, and on the legacy default stream for the whole
  device, since CUDA orders a default-stream call after every blocking stream.
  The device-wide wait also covers non-blocking streams, which costs
  concurrency, not correctness.
- **cuFFT:** ranks 1 to 3 execute for every transform type, including cuFFT's
  advanced data layout (`inembed`/`onembed`/stride/dist), which is what a padded
  grid such as GROMACS's PME mesh needs. Eligible dense, out-of-place rank-3
  single-precision R2C/C2R plans use vendored VkFFT 1.3.4 on Metal. Other
  single-precision transforms (`C2C`/`R2C`/`C2R`) use project-owned Stockham
  autosort and Bluestein GPU kernels; grids below a dispatch-cost threshold and
  every double-precision entry point stay on the CPU, since Metal has no FP64.
  `CUMETAL_FFT_VKFFT=0` explicitly disables the VkFFT route. Still absent:
  callbacks (`cufftXtSetCallback` returns `CUFFT_NOT_SUPPORTED`), multi-GPU
  (`cufftXtSetGPUs` accepts only device 0 alone; `cufftXtMemcpy` and the
  descriptor executes return `CUFFT_NOT_SUPPORTED`), and a GPU path for the
  double transforms. `cufftXtExec` dispatches to the plan's own transform type. Implemented
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
  rejected instead of being silently ignored. The generic 64-bit
  `cusolverDnXsyevd`/`cusolverDnXsyevBatched` surface exists for the
  homogeneous FP32/FP64 subset (data/compute/output types must agree) over the
  same Accelerate LAPACK path; other type combinations are rejected. Real
  Jacobi `cusolverDnS/DsyevjBatched` runs a CPU cyclic-Jacobi iteration that
  honors `syevjInfo_t` tolerance, max-sweeps, and ascending-sort controls and
  reports per-matrix nonconvergence through `devInfo`.
  The CuPy-driven additions all run on the CPU over unified memory:
  - **cuSOLVER dense:** complex `getrf`/`getrs`/`geqrf`/`potrf`/`potrs`/`gesvd`/
    `heevd`, `orgqr`/`ungqr`, `ormqr`/`unmqr`, `sytrf`, `gebrd`, batched
    `potrf`/`potrs`, single and batched `syevj`/`heevj`, `gesvdj`/
    `gesvdjBatched`, and `gesvdaStridedBatched`.
    - `gesvdj` computes with LAPACK `gesvd`, not Jacobi sweeps.
      `cusolverDnXgesvdjGetSweeps` reports 0. `GetResidual` is the explicitly
      measured `||diag(S) - U^H A V||_F`.
  - **IRS `gesv`/`gels` family:** solves in the main precision only and reports
    `niter = 0`.
  - **cuSOLVER sparse:** complex `csrlsvchol`/`csrlsvqr` and
    `S`/`D`/`C`/`Zcsreigvsi` (shifted inverse iteration).
  - **Status and pivot conventions:** LAPACK `info > 0` returns
    `CUSOLVER_STATUS_SUCCESS` with `devInfo` set, and a NULL `devIpiv` selects
    unpivoted LU, both as in cuSOLVER.
  - **Remaining cuSOLVER limits:**
    - Real `csrlsvchol`/`csrlsvqr` still reject `reorder != 0`.
    - `cusolverDnXgeev` is absent.
  - **cuSPARSE legacy and generic APIs:** format conversion and sorting,
    `nnz`/`nnz_compress`/`csr2csr_compress`, `csrgeam2`, `csrilu02` and
    `csric02` with truthful `zeroPivot`, the `gtsv2`/`gtsvInterleavedBatch`/
    `gpsvInterleavedBatch` banded solvers, `SpVV`, `Gather`, `SpSM`, `SpGEMM`,
    `SparseToDense`/`DenseToSparse`, and `Csr2cscEx2`.
  - **cuSPARSE refusals (`CUSPARSE_STATUS_NOT_SUPPORTED`):**
    - BSR incomplete factorizations (`bsrilu02`/`bsric02`).
    - `SpGEMM` with transposed operands, non-CSR matrices, or `beta != 0`.
    - `SpSM` with COO or batched dense matrices.
    - Batched dense/sparse conversion.
  `cusolverGetProperty` reports CuMetal's own version, not an NVIDIA release.
  Broader dense/sparse routine,
  datatype, batched, analysis/reuse, and GPU execution coverage remains open.
- **cuDNN:** selected descriptors/operations only. The hardened CPU-backed
  surface is primarily contiguous FP32/NCHW; it synchronizes the handle stream
  before reading UMA operands and rejects unsupported types/layouts/shapes for
  the covered calls. Convolution is the tested implicit-GEMM cross-correlation
  path; its tracked operands and workspaces are range-checked, and custom Nd
  strides are rejected because the implementation is contiguous. Other
  implemented tensor-operation families also range-check tracked operands;
  ordinary host buffers remain supported by the explicit CPU fallback. General
  algorithm selection, convolution-mode filter reversal, general
  OpTensor broadcasting, fusion, training/backward breadth, graph integration,
  datatype, and layout coverage remain incomplete. Forward RNN/GRU/LSTM is a
  bounded, CPU-backed FP32/NCHW path: standard algorithm, linear input, and
  zero dropout only. Its timestep/state geometry, parameter sizes, scratch
  sizes, and tracked allocation spans are checked, but backward RNN, packed or
  variable sequences, nonzero dropout, persistent algorithms, and broader
  descriptor formats are absent. Attention forward is limited to
  projection-free, dropout-free FP32 canonical descriptors with fixed full
  sequences and disjoint output. Learned projections, biases, residuals,
  windows, incremental decoding, variable lengths, backward/training reserve,
  and broader datatype/layout behavior are explicitly unsupported. Its covered
  tensor spans, configured maxima, and projection-weight queries are checked;
  this remains a bounded compatibility path rather than full cuDNN.
- **NCCL:** single-device compatibility cannot provide collective multi-GPU
  semantics. The implemented one-rank collectives are identity copies, not a
  transport; only device zero and rank zero are accepted, point-to-point calls
  fail, and multi-device initialization is rejected atomically. The header
  reports NCCL 2.18.0 to match `ncclGetVersion`. `ncclCommInitRankConfig` and
  `ncclCommSplit` produce one-rank communicators, and their config fields have no
  effect. `ncclCommGetAsyncError` always reports success because collectives
  finish synchronously.
- **NVML:** compatibility queries cannot expose NVIDIA device management. The
  single synthetic device reports Apple unified system-memory information, not
  dedicated VRAM; utilization, temperature, power, and clock telemetry are
  explicitly unsupported through the public-API-only boundary.
- **Thrust/CUB:** several algorithms are sequential/CPU over UMA; device-wide
  performance and full template/API compatibility are not claimed. CUDA-source
  `thrust::sort(cuda::par.on(stream), ...)` uses GPU merge and copy kernels,
  including a device-callable comparator, and completes the selected stream
  before releasing its temporary allocation. The gate covers signed integer
  ordering, duplicates, custom iterators/comparators, producer ordering and
  error paths. It is a correctness implementation, not a radix-sort performance
  claim. CUDA-policy `sort_by_key` and policy sort instantiated by a non-CUDA
  C++ compiler still refuse explicitly. Policy-free sort remains host-backed.
  Those host-backed `cub::Device*` entry points do now synchronize their stream before
  reading the input, which is a correctness requirement rather than a
  performance choice: without it a scan or reduction of a buffer a kernel is
  still writing silently returns stale memory. The tested aggregate
  `ShuffleIndex` helper covers trivially-copyable objects up to the fixed
  32-lane warp model, but broader CUB warp/block free-function and policy
  overload parity remains unclassified. `cub::BlockReduce` is a real cooperative
  reduction in device code and a sequential fallback on the host.
  `cub::BlockScan` is likewise a real cooperative scan in device code
  (multiple warps, 1D/2D/3D row-major indexing, custom operators,
  caller-barrier storage reuse, aggregate outputs) with a sequential host
  fallback. The other
  block and warp primitives are still host-only fallbacks and cannot be called
  from a kernel. `DeviceRadixSort` and `DeviceSegmentedRadixSort` accept
  `cub::DoubleBuffer` but sort in place, so the selector they return is always
  the one they were given -- correct for callers that read `Current()`, which is
  how CUB is meant to be used, but not the ping-pong a caller inspecting
  `selector` might expect.
- **NVTX:** annotations are no-ops.

The closure target is a generated support table from actual positive and
negative cases, not a list of exported symbol names.
