# Status

Current status: **experimental CUDA compatibility stack with a documented PTX
subset and real Metal GPU execution for covered kernels.**

The July 2026 Apple-GPU execution work, including the exact llama.cpp result,
runtime policy, provenance contract, toolchain shims, and completion validation,
is recorded in [apple-gpu-execution.md](apple-gpu-execution.md).

The 2026-07-26 silent-wrong-answer audit — two compiler miscompiles, two
harnesses that reported failures as skips, and the suite-wide skip review — is
recorded in [correctness-audit-2026-07-26.md](correctness-audit-2026-07-26.md).

Do not interpret the number of CTest registrations as a pass count. The suite
contains environment-dependent skips and external-project conformance gates;
each run must report passed, skipped, and failed totals separately with the
machine/Xcode configuration. General CUDA compatibility and broad llama.cpp GPU
offload are not complete.

Earlier phase labels below describe implemented project milestones, not proof
that arbitrary CUDA programs are supported. Phase 5 benchmark infrastructure is
implemented for the covered kernels.
Intentional non-goals per §2.2 (CUDA Graphs, dynamic parallelism, texture objects,
multi-GPU, graphics interop) remain deferred to v2.

v1 toolchain-completeness work (2026-07-26):

- **`cumetalc foo.cu -o foo` builds a runnable executable**, closing the spec §11 Phase 2/3 exit
  criterion. Previously `cumetalc` could only emit a `.metallib`, so every program had to be
  hand-split into a `.cu` plus a host `.cpp` that named kernels through `cumetalKernel_t` and
  took a metallib path as `argv[1]`. Gated by `functional_cumetalc_link_executable`, which
  requires both a numerical PASS and `device=apple_gpu` provenance. Verified from an install
  prefix as well as the build tree.

- **The CUDA registration ABI is decoupled from the binary shim.**
  `CUMETAL_ENABLE_CUDA_REGISTRATION` (always ON) builds `__cudaRegister*`; the renamed-in-scope
  `CUMETAL_ENABLE_BINARY_SHIM` now controls only the `libcuda.dylib` alias. These were one
  switch, so `CMAKE_BUILD_TYPE=Release` stubbed out the registration ABI and the entire
  source-recompilation path went unexercised in the configuration users install — the tests that
  covered it did not fail, they silently vanished from the suite.

- **`scripts/ci_report.sh` reports passed/skipped/failed separately** and names every skipped
  test, per this document's own warning that a registration count is not a pass count. Run it in
  place of bare `ctest`:

  ```bash
  bash scripts/ci_report.sh build --exclude-regex '^bench_'
  ```

  Layered GitHub Actions definitions exist for both configurations —
  Release/shim-off, which is what users install, and Debug/shim-on — but are
  temporarily disabled as `.github/workflows/*.yml.disabled`. When enabled,
  the hosted lane runs only tests explicitly labelled `hosted`, requires a
  non-empty selection, and verifies the installed prefix and shim layout.

  Spec §10.7's Apple-GPU correctness layer is defined separately for a
  self-hosted runner labelled `ci-m1`. When the workflow is re-enabled, it runs
  on pushes to `main` after the `CUMETAL_GPU_CI_ENABLED=true` repository
  variable is set and can be commissioned with a manual dispatch. A focused
  GPU/provenance set uses
  `--require-no-skips` before the full suite, so a missing device or Metal
  toolchain cannot appear as a green hardware result. Pull requests do not run
  untrusted code on the self-hosted machine.

- **The performance gate no longer flakes under load.** `cumetal_bench` averaged its iterations,
  and a mean is dominated by its worst sample, so a single scheduler stall failed the 2× ceiling
  on a busy machine. These kernels run in ~0.2 ms and are dispatch-jitter dominated — the
  per-iteration spread runs from a few percent to ~50% on a lightly loaded machine and climbs
  sharply under contention — so no measure of central tendency is stable:
  the median still reached 2.73× for saxpy under 8-way CPU saturation, because contention shifts
  the whole distribution and shifts CuMetal's path further (it does more host work per dispatch).
  The gate now reports the **fastest** iteration, the standard microbenchmark estimator for
  latency under interference. Verified 13/13 passes across idle and 8-way-saturated runs.
  The same change corrected a misleading published result: the mean-based numbers reported
  `vector_add` at 0.74×, i.e. CuMetal 26% *faster* than hand-written Metal, which was outliers
  inflating the native baseline rather than a real speedup. The results table now also prints a
  `spread` column so a reader can see how jittery a given run was.

- **The llm.c parity intermittency is root-caused and fixed** (0/75 after, 2/25 before). Two
  defects: a JIT cache key that did not identify the compiler build that produced each entry, so
  lowering changes silently reused stale kernels and a cache could hold kernels from several
  compiler versions at once; and a missing device-memory barrier in the specialized
  `fused_classifier_kernel3` template, where thread 0 read a logit while other threads overwrote
  it with gradients. See [known-gaps.md](known-gaps.md).

- **Name-match audit complete** ([name-match-audit-2026-07-26.md](name-match-audit-2026-07-26.md)).
  Every site where CuMetal decided behavior from a name it does not own was reviewed. Four body
  templates removed from the LLVM path, the 67-pattern MSL specialization table demoted to a
  fallback behind real translation, and a name-keyed argument-count override removed from the
  runtime launch path (instrumentation showed it was never reached). Intrinsic matching, cache
  identity lookups, and metallib byte signatures were cleared as legitimate.

- **Backend default is input-dependent and measured** rather than a single global setting; see
  the table in [known-gaps.md](known-gaps.md).

- **`ptx_sweep_numeric` executes each PTX opcode and checks the value** (spec §10.2). The
  pre-existing sweeps lower a kernel per opcode and grep the IR for `define void @name`, which
  proves a function was emitted and nothing about what it computes. The new harness runs each
  opcode on the GPU and compares bit-for-bit against a hand-derived ISA oracle. On its first run
  it found `neg.s32` returning a float sign-bit flip, which traced back to four name-matched
  body templates in `lower_to_llvm.cpp` that discarded real PTX bodies — see
  [known-gaps.md](known-gaps.md). Those are removed, and unlowerable kernels are now refused
  rather than emitted as an empty `ret void`.

- **AIR ABI results are attributable to a toolchain** (spec §10.5). Every AIR ABI test prints
  macOS build, Xcode version and build, selected `TOOLCHAINS`, chip, and metal compiler version.
  `air_abi_xcode_matrix_regression` previously defaulted both Xcode slots to the same developer
  directory and reported cross-version coverage it never had; it now identifies toolchains by
  compiler version, deduplicates, and says plainly when only one is present. Genuine
  cross-version coverage still requires two Xcodes installed side by side and
  `CUMETAL_XCODE15_DEVELOPER_DIR` / `CUMETAL_XCODE16_DEVELOPER_DIR` pointed at them.

- **Four test harnesses stopped reporting on stale artifacts.**
  `run_samples_vector_add.sh`, `run_cumetalc_cu_runtime_vector_add.sh`,
  `run_runtime_vector_add.sh`, and `run_runtime_axpy.sh` each checked for a pre-existing
  binary/metallib *before* checking whether the toolchain was available, so once any build
  produced one they stopped rebuilding and would have stayed green through a total compiler
  regression. They now rebuild whenever the toolchain is present and fall back to a stale
  artifact only when it is genuinely absent, with a warning. Same class as the
  `cumetal_cuda_projects_compile_link` stale-object bug in
  [correctness-audit-2026-07-26.md](correctness-audit-2026-07-26.md).

Phase 5 items implemented:

- `metal_backend::launch_kernel_timed()` — synchronous kernel launch that captures
  `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` for precise GPU-execution-time measurement.
- `metal_backend::GpuTimingResult` — GPU start/end in CFTimeInterval (seconds), with a
  `duration_ms()` helper.
- `tools/cumetal_bench/bench_kernels.metal` — native Metal MSL baseline kernels:
  `vector_add`, `saxpy` (memory-bound SAXPY with scalar alpha as a 1-element buffer),
  `reduce_f32` (tree reduction using threadgroup shared memory, one partial sum per
  threadgroup).
- `tools/cumetal_bench/main.cpp` — rewritten multi-kernel Phase 5 benchmark:
  - Supports `--all-kernels` to sweep vector_add, saxpy, and reduce_f32.
  - Reports native GPU time (from `MTLCommandBuffer` timestamps) and wall-clock time for
    both paths; ratio uses wall-clock (apples-to-apples: both paths synchronize per iteration).
  - Prints a tabular comparison: kernel | elements | native_gpu_ms | native_wall_ms |
    cumetal_wall_ms | ratio | PASS/FAIL.
  - `--max-ratio <x>` enforces the spec §5.7 / §10.6 gate (Phase 5 target: ≤ 2.0×).
  - Clean-rebuild Apple M4 Pro measurement on 2026-07-27 (fastest of 20 iterations):
    vector_add 1.063×, saxpy 1.036×, reduce_f32 1.008×. The benchmark metallib was
    regenerated after rebuilding `libcumetal`; older cross-rebuild measurements are
    not comparable because the former registration-JIT cache key could reuse kernels
    from a previous compiler build. These also supersede an earlier mean-based set
    (0.74×/0.98×/1.00×) that was outlier-dominated.
- `scripts/generate_bench_metallib.sh` — compiles `bench_kernels.metal` to
  `bench_kernels.metallib` via `xcrun metal` + `xcrun metallib`; exits 77 if
  toolchain is unavailable (CTest skip).
- `scripts/run_bench_phase5.sh` — end-to-end Phase 5 gate script: generates metallib,
  then runs `cumetal_bench --all-kernels --max-ratio 2.0`.
- `bench_phase5_all_kernels` CTest — registered in CMakeLists.txt (APPLE only,
  SKIP_RETURN_CODE 77); enforces the 2× ceiling defined in spec §5.7.

Post-Phase 5 work completed:

- **Selectable runtime MSL math policy**: `CUMETAL_MSL_MATH_MODE=fast|safe` controls
  runtime Metal source compilation, defaulting to the historical fast mode. Safe mode uses
  `MTLMathModeSafe` on macOS 15+ with a macOS 14-compatible fallback. The normalized policy is
  part of registration JIT cache identity and appears in `CUMETAL_TRACE_GPU=1` provenance;
  invalid values warn once and use fast mode. An Apple-GPU functional test verifies numerical
  correctness, distinct cache artifacts, and provenance for both modes.

- **Bounded ELF32/ELF64 fatbinary parsing**: CUDA registration and `cuModuleLoadData` now share
  section-table-driven extraction for little-endian ELF32 and ELF64 `.nv_fatbin` and raw-PTX
  sections.
  Standard and extended section counts/string-table indexes are resolved from the ELF header or
  section header 0 as required. Header, table, name, payload, and nested fatbin ranges are checked
  against the 64 MiB image ceiling; malformed and unsupported ELF images are refused. Functional
  tests launch the same PTX through registration and `cuModuleLoadData` for both ELF classes,
  including extended-index images, with malformed-image negative coverage.

- **Exact GPT-NeoX rotary embedding for covered GGML ABIs**: the concrete forward,
  no-frequency-factor `rope_neox` float-to-float and float-to-half variants now rotate paired
  lower/upper dimension halves with the full YaRN interpolation and fused set-rows indexing
  contract. Other `rope_neox` template variants remain classified approximate and refused.
  Unit tests pin the positive and negative mangled-name boundary, and a registration-path
  Apple-GPU test compares both output types against a CPU numerical oracle.

- **Correct legacy default-stream semantics**: runtime and Driver API stream flags are persisted
  and queryable, while Metal command submissions implement bidirectional implicit ordering between
  the legacy default stream and blocking user streams through `MTLSharedEvent` epochs. Non-blocking
  and per-thread streams remain independent. Positive and negative tests use disjoint allocations
  so conservative buffer-hazard fencing cannot mask a missing stream-order dependency.

- **CTest-wide Metal toolchain discovery**: configure now discovers Apple's
  separately installed Metal toolchain with
  `xcodebuild -showComponent MetalToolchain -json`, verifies its identifier
  through `xcrun -f metal`, and injects `TOOLCHAINS` into every registered
  test without overwriting existing test environments. Explicit cache and
  process-environment selections still win.

- **Classified standalone CUDA coverage sweep**:
  `tests/cuda_projects/sweep_cuda_projects.py` validates that every standalone
  `.cu` fixture is represented in a manifest, runs it with strict result
  semantics, and emits per-project logs plus TSV/JSON summaries. The current
  nine-project baseline is nine passes; unsupported lowering, compiler/link
  failures, crashes, and timeouts are distinguished rather than collapsed into
  skips. The CTest harness (`run_standalone_cu.sh`) likewise reports a kernel
  that runs but computes wrong results as a failure, never as a skip — only
  genuinely unavailable lowering (`registered kernel missing metallib`) skips.
  `cumetal_cuda_projects_compile_link` propagates the compiler's exit status and
  removes any previous object and binary first, so a failed compile cannot link a
  stale object from an earlier build and report a pass.

- **`functional_runtime_ptx_lowering_regression` executes again**: it previously
  passed `--mode experimental` unconditionally, always produced an unloadable
  container, and always skipped its own execution check on every machine. It now
  packages validated metallibs with `--mode xcrun`, skips only when `xcrun
  metal`/`metallib` are genuinely absent, and verifies negate, reduce_sum, and
  clamp_relu numerically on the GPU.

- **Typed NVVM/MSL migration**: the importer now covers common single-precision
  libdevice declarations including reciprocal square root, exponential,
  logarithmic, trigonometric, hyperbolic, rounding, and power functions. The
  hidden source-pattern-specific vector-add AIR template has been removed;
  direct AIR remains research tooling, while the explicit legacy backend
  remains available until the specification's parity gate is met.

- **Demand-driven binary-shim PTX ABI resolution**: fatbinary registration
  records kernel identities without scanning every module. The first actual
  launch resolves only its requested `.entry` signature; modules and entries
  that never launch are never parsed or inserted into a full ABI map. The
  scanner handles comments, strings, pointer qualifiers, scalar widths,
  aggregate byte arrays, and NVCC's unqualified 64-bit pointer convention. On
  Apple M4 Pro, a comment-safe 5,000-entry worst-case lookup takes 4 ms versus
  17 ms to build
  the complete index. The earlier lazy full-index implementation had already
  improved the llama.cpp NGL=1 one-token workload from 8.20 s to 1.00 s after
  the original linear scanner reduced it from 290.24 s. Full parsing remains
  part of actual kernel lowering.

- **Streaming, memoized registration JIT cache keys**: immutable module PTX is
  shared across kernel resolutions, and the persistent-cache FNV prefix is
  streamed and memoized once per module instead of copying and hashing
  multi-megabyte PTX once per launched kernel. Cache keys remain byte-for-byte
  compatible. Twelve controlled interleaved A/B runs of the SmolLM2 NGL=1
  16-token coherence gate improved from a 0.60 s warm median at `a41b4e5` to
  0.575 s (about 4.2%), with correct `Paris` output in every measured run.

- **Native FP16 `cublasGemmEx` library lowering**: all-FP16 GEMM with FP16
  compute now binds the tracked CUDA allocations directly as `MPSMatrix`
  operands instead of allocating and filling FP32 copies on the CPU. On the
  SmolLM2 NGL=1 gate this moves the 49,152×576 output projection to Apple GPU:
  the five-run warm median is 0.57 s for one token and 0.61 s for the 16-token
  coherence gate; generation rose from 8.1 to 279.2 tokens/s median (five runs).

- **GPU-address batched GEMM and mixed FP16/FP32 lowering**: device allocations
  now retain both their shared-memory CPU identity and native `MTLBuffer`
  virtual address in the allocation table. `cublasSgemmBatched`,
  `cublasDgemmBatched`, and `cublasGemmBatchedEx` decode device-resident pointer
  arrays through that table, including pointer values written by Metal kernels.
  FP16 A/B with FP32 compute and C now binds directly to mixed-type `MPSMatrix`
  descriptors instead of allocating CPU-filled FP32 operand copies. Truncated
  device tables fail with `CUBLAS_STATUS_INVALID_VALUE`. Test:
  `functional_cublas_device_pointer_table` (runs with native Metal addresses).

- **Truthful llama.cpp FlashAttention capability**: the CuMetal llama.cpp build
  sets `GGML_CUDA_FA=OFF`. llama.cpp's native CUDA backend probe consequently
  reports fused FlashAttention unsupported and its scheduler selects the
  ordinary attention graph rather than launching kernels CuMetal deliberately
  rejects. Test: `unit_llama_build_contract`.

- **Reproducible llama.cpp high-offload sweep**: the exact
  `k_compute_batched_ptrs` runtime helper now writes Metal GPU virtual addresses
  by default, clearing the former NGL=2 cuBLAS invalid-value failure. The
  `sweep_llama_cpp_ngl.py` runner classifies and logs NGL=1..99 probes. The
  former NGL=4 incoherence was isolated with llama.cpp's per-node evaluation
  callback to GGML's strided `cpy_scalar<half, half>` materialization of a
  transposed value-cache view. Exact typed MSL scalar-copy variants now preserve
  all four source and destination byte strides. The 2026-07-23 SmolLM2 sweep
  passes every NGL from 1 through 99; values above its layer count repeat the
  saturated full-offload configuration. Tests: `unit_ptx_lower_to_metal` and
  `unit_llama_ngl_sweep_classifier`.

- **Cross-command-queue resource fencing**: each Metal stream owns a public
  `MTLSharedEvent`, and kernel plus MPS command buffers publish one coalesced
  access value while waiting for prior accesses to their tracked buffers. This
  keeps registered launches asynchronous without stale alias reads. Five
  interleaved warm NGL=3 runs measured a 1.000 s asynchronous median versus
  1.024 s with `CUMETAL_SYNC_REGISTERED_LAUNCH=1` (0.98×) on Apple M4 Pro.
  Tests: `functional_metal_backend_cross_queue_fence` and
  `conformance_llama_cpp`; benchmark:
  `scripts/benchmark_llama_registered_launches.py`.

- **MTLHeap auto-threshold**: MTLHeap sub-allocation now auto-enabled for allocations ≥ 4 MiB
  (configurable via `CUMETAL_MTLHEAP_THRESHOLD_BYTES`). Three modes:
  - `CUMETAL_MTLHEAP_ALLOC` unset → auto (heap for size ≥ threshold, default 4 MiB)
  - `CUMETAL_MTLHEAP_ALLOC=1` → always use heap
  - `CUMETAL_MTLHEAP_ALLOC=0` → never use heap
  Tests: `functional_runtime_heap_auto_threshold`, `functional_runtime_heap_disabled`.

- **Binary shim JIT cache**: Registration-path PTX→metallib compilations are now cached
  persistently at `$CUMETAL_CACHE_DIR/registration-jit/<hash>.metallib` (default:
  `$HOME/Library/Caches/io.cumetal/registration-jit/`); the supported direct-source
  fallback stores `<hash>.metal` instead. The FNV-1a-64 key includes the libcumetal
  Mach-O `LC_UUID`, lowering policy, PTX source, and kernel name. Persistent cache
  files survive `__cudaUnregisterFatBinary` and process restart — the same build's
  second registration skips compilation, while a distinct build UUID uses a separate entry.
  Test: `functional_runtime_registration_jit_cache`.
- **`CUMETAL_DEBUG_REGISTRATION=1`** — opt-in stderr trace for binary shim diagnostics:
  logs fatbinary format detection, JIT compile path (Metal vs LLVM IR lowering),
  cache hits/misses, arg count inference, and kernel/symbol registration events.

Post-Phase 5 work completed (continued, part 2):

- **cuBLAS extended APIs** (`runtime/rt/cublas.cpp`, `runtime/api/cublas_v2.h`):
  Added `cudaDataType_t`, `cublasDiagType_t`, `cublasSideMode_t`, `cublasGemmAlgo_t` enums.
  New functions:
  - `cublasGemmEx` — extended GEMM: routes CUDA_R_32F → cublasSgemm,
    CUDA_R_64F → cublasDgemm, all-FP16/FP16-compute and
    FP16-input/FP32-output directly to MPS, and other mixed types through the
    FP32 conversion path.
  - `cublasGemmStridedBatchedEx` — batched strided GemmEx; routes fp32/fp64 to typed variants.
  - `cublasHgemm` — half-precision GEMM through the native FP16 GemmEx path.
  - `cublasSgemmBatched` / `cublasDgemmBatched` / `cublasGemmBatchedEx` —
    array-of-pointers batched GEMM with device-resident GPU-address table
    translation.
  - `cublasStrsm` / `cublasDtrsm` — triangular solve (BLAS3); supports LEFT/RIGHT side,
    UPPER/LOWER fill, N/T/C transpose, UNIT/NON_UNIT diagonal, alpha scaling.
  - `cublasSetVector` / `cublasGetVector` / `cublasSetMatrix` / `cublasGetMatrix` —
    strided host↔device copy helpers (no-op overhead on Apple Silicon UMA).
  - Async vector/matrix transfer variants enqueue their strided copies on the supplied CUDA
    stream; they no longer ignore it or execute inline.
  Tests: `functional_cublas_extended_api`,
  `functional_cublas_device_pointer_table`.


- **Miscellaneous extended APIs** (`runtime/api/`, `runtime/rt/`, `runtime/driver/`):
  Fills remaining API gaps identified in post-Phase-5 survey.
  - **cuRAND**: `curandGeneratePoisson(generator, ptr, n, lambda)` — Poisson-distributed
    uint32 via `std::poisson_distribution`; `curandGetProperty(type, value)` returning
    major/minor/patch version (mirrors CUDA `libraryPropertyType` enum).
  - **cuBLAS**: `cublasGetStatusName(status)` — returns enum-name string (e.g.
    `"CUBLAS_STATUS_SUCCESS"`); `cublasGetStatusString(status)` — returns human-readable
    description.
  - **cuFFT**: `cufftSetWorkArea(plan, workArea)` — no-op stub (vDSP manages its own
    scratch on UMA); `cufftEstimate1d/2d/3d/Many` — returns a conservative upper-bound
    scratch-size estimate without building a full plan.
  - **3D pitched memory** (`cuda_runtime.h`/`cuda.h`): Added types `cudaExtent`,
    `cudaPitchedPtr`, `cudaPos`, `cudaMemcpy3DParms` (with C++ `make_*` helpers) and
    opaque `cudaArray_t`. New runtime APIs:
    - `cudaMalloc3D(pitchedDevPtr, extent)` — allocates pitch×height×depth bytes,
      pitch aligned to 512 bytes.
    - `cudaMemcpy3D(parms)` / `cudaMemcpy3DAsync(parms, stream)` — 3D pitched copy
      (plane-by-row stride walk; the async form is stream ordered).
  - **Driver API 3D copy** (`cuda.h`/`cuda_driver.cpp`): Added `CUmemorytype` enum,
    `CUarray` opaque typedef, `CUDA_MEMCPY3D` struct, and:
    - `cuMemcpy3D(pCopy)` / `cuMemcpy3DAsync(pCopy, hStream)` — 3D strided copy
      resolving host/device ptrs from `CUmemorytype`; the async form is enqueued in the stream
      timeline.
  Test: `functional_misc_extended_api` (6 sub-tests covering all new APIs).

- **Extended APIs batch 2** (`runtime/api/`, `runtime/rt/`, `runtime/driver/`):
  - **cuRAND**: `curandCreateGeneratorHost` — on Apple Silicon UMA host=device, aliases
    `curandCreateGenerator` (no separate host/device distinction needed).
  - **cuBLAS**: `cublasGetProperty(type, value)` — returns cuBLAS version (major/minor/patch)
    via `libraryPropertyType` enum (same guard as `curand.h` to prevent double-definition).
    Symmetric BLAS:
    - `cublasSsyr`/`cublasDsyr` — symmetric rank-1 update: `A += alpha * x * x^T`
      (column-major, only upper or lower triangle updated).
    - `cublasSsyrk`/`cublasDsyrk` — symmetric rank-k update:
      `C = alpha * op(A) * op(A)^T + beta * C`.
    - `cublasSsyr2k`/`cublasDsyr2k` — symmetric rank-2k update:
      `C = alpha * (op(A)*op(B)^T + op(B)*op(A)^T) + beta * C`.
  - **Driver API**:
    - `cuFuncSetAttribute` — validates function handles; unsupported mutable Metal pipeline
      attributes return an explicit error.
    - `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` — delegates to base function,
      flags ignored.
    - `cuCtxPushCurrent`/`cuCtxPopCurrent` — thin wrappers around `cuCtxSetCurrent`/`GetCurrent`.
    - `cuDevicePrimaryCtxRetain`/`cuDevicePrimaryCtxRelease` — create/destroy primary context
      (single GPU on Apple Silicon).
    - `cuStreamGetPriority`/`cuStreamGetFlags` — return 0 (single-priority stream model).
    - `cuModuleGetGlobal` — stub returning `CUDA_ERROR_NOT_FOUND` (Driver-loaded modules do
      not yet expose globals; runtime-registration globals are covered separately below).
  - **Runtime peer copy**: `cudaMemcpyPeer`/`cudaMemcpyPeerAsync` — UMA single GPU;
    forward to `cudaMemcpy`/`cudaMemcpyAsync` with `cudaMemcpyDefault`.
  - `cudaLaunchHostFunc(stream, fn, userData)` — enqueues a CPU callback between shared-event
    markers so later stream work cannot overtake it and synchronization waits for it.
  - Runtime/Driver occupancy queries validate real kernel handles and derive limits from the
    corresponding Metal compute pipeline's maximum threads, execution width, and static
    threadgroup memory.
  Test: `functional_extended_api_v2` (18 sub-tests covering all new APIs).

- **Extended APIs batch 3** (`runtime/api/`, `runtime/rt/`):
  - **cuRAND**: `curandGenerateExponential`/`curandGenerateExponentialDouble` — exponential
    distribution via inverse transform: X = -ln(U), U ~ Uniform(0,1).
  - **cuFFT**: `cufftGetProperty(type, value)` — returns cuFFT version major/minor/patch
    (consistent with `curandGetProperty`/`cublasGetProperty` pattern).
  - **cuBLAS BLAS2**: `cublasSsyr2`/`cublasDsyr2` — symmetric rank-2 update:
    A += α·(x·yᵀ + y·xᵀ); only upper or lower triangle updated.
  - **cuBLAS BLAS2**: `cublasStrmv`/`cublasDtrmv` — triangular matrix-vector multiply:
    x := op(A)·x; supports UPPER/LOWER, NO_TRANS/TRANS, UNIT/NON_UNIT diagonal;
    uses temp buffer for in-place correctness.
  - **cuBLAS BLAS3**: `cublasSsymm`/`cublasDsymm` — symmetric matrix-matrix multiply:
    C = α·A·B + β·C (SIDE_LEFT) or C = α·B·A + β·C (SIDE_RIGHT); symmetric element
    lookup reconstructs missing triangle from stored half.
  - **cuBLAS BLAS3**: `cublasStrmm`/`cublasDtrmm` — triangular matrix-matrix multiply:
    C = α·op(A)·B (SIDE_LEFT) or C = α·B·op(A) (SIDE_RIGHT); output written to C
    (cuBLAS v2 API); supports all trans/diag/uplo/side combinations.
  - **cuBLAS BLAS1**: `cublasSrot`/`cublasDrot` — apply Givens rotation:
    x[i] = c·x[i] + s·y[i]; y[i] = c·y[i] - s·x[i].
  - **cuBLAS BLAS1**: `cublasSrotg`/`cublasDrotg` — construct Givens rotation:
    given (a,b) compute (c,s,r,z) such that [c s;-s c]·[a;b] = [r;0].
  Test: `functional_extended_api_v3` (14 sub-tests covering all new APIs).

- **Extended APIs batch 4** (`runtime/api/`, `runtime/rt/`, `runtime/driver/`):
  - **Runtime 3D memset**: `cudaMemset2DAsync` — stream-ordered 2D memset;
    `cudaMemset3D`/`cudaMemset3DAsync` — fill 3D pitched volume plane-by-row using
    `pitchedDevPtr.pitch × pitchedDevPtr.ysize` as the plane stride.
  - **Driver 2D memset**: `cuMemsetD2D8`/`cuMemsetD2D16`/`cuMemsetD2D32` — strided per-row fill
    (8-bit uses `memset`; 16/32-bit use typed element loops); `*Async` variants enqueue the
    typed fill in the selected stream.
  - **Driver allocation query**: `cuMemGetAddressRange(pbase, psize, dptr)` — queries CuMetal's
    allocation table via `cumetalRuntimeGetAllocationInfo` to return base address and allocation
    size for any pointer within a `cudaMalloc`-ed block.
  - **Driver pointer attributes**: `cuPointerGetAttribute(data, attribute, ptr)` — supports
    `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` (returns `CU_MEMORYTYPE_UNIFIED`),
    `CU_POINTER_ATTRIBUTE_DEVICE_POINTER`, `CU_POINTER_ATTRIBUTE_HOST_POINTER` (both return
    the pointer itself; UMA identity), `CU_POINTER_ATTRIBUTE_IS_MANAGED`,
    `CU_POINTER_ATTRIBUTE_MAPPED`, and `CU_POINTER_ATTRIBUTE_CONTEXT`.
  - **cuBLAS BLAS1**: `cublasSrotm`/`cublasDrotm` — apply modified Givens rotation H to (x,y);
    flag encoding: -2 = identity no-op, -1 = general [h11 h12; h21 h22],
    0 = diagonal-1 [1 h12; h21 1], 1 = off-diagonal [h11 1; -1 h22].
  - **cuBLAS BLAS1**: `cublasSrotmg`/`cublasDrotmg` — construct modified Givens rotation using
    Lawson et al. algorithm; encodes H into param[0..4] with rescaling loop to prevent
    overflow/underflow; updates d1, d2, x1 in-place.
  Test: `functional_extended_api_v4` (34 sub-tests covering all new APIs).

- **Threadgroup memory tiling hints** (`compiler/passes/src/threadgroup_tiling.cpp`):
  New `analyse_threadgroup_tiling()` pass that scans a PTX kernel's instruction
  stream for shared-memory bank-conflict patterns.  The pass detects `mul.lo`/`shl`
  stride constants (window of 4) immediately preceding `ld.shared`/`st.shared`/
  `atom.shared`/`red.shared` accesses and emits `TilingHint` entries for every
  power-of-2 stride ≥ 16 that aligns to the 32-bank, 4-byte-per-bank Metal
  threadgroup memory boundary.  Each hint carries the detected stride, element
  size, recommended padding (1 element = `elem_bytes`), and a human-readable
  reason string.  Covered by `unit_threadgroup_tiling` (9 sub-cases).

Items remaining (deferred per spec §2.2):

- Kernel fusion via MLIR GPU dialect (optional, deferred to v2).

Implemented:

- Phase 0.5 tooling:
  - `air_inspect`: `.metallib` container inspection
    - parses Apple function-list tags (`NAME`/`TYPE`/`HASH`/`MDSZ`/`OFFT`/`VERS`) on current Xcode layout
  - `cumetal-air-emitter`: `.metallib` emission (xcrun-backed + experimental mode)
  - `cumetalc`: thin compiler-driver CLI over the AIR emitter
  - `air_validate`: structural checks + optional `xcrun metal -validate`
  - `cumetal_metal_load_test`: `MTLDevice.newLibraryWithData:` acceptance test
- Phase 1 scaffolding:
  - minimal PTX text parser (`.version` / `.target` / `.entry` / `.param` + instruction stream)
    with tolerant/strict unsupported-op modes in `compiler/ptx/`
  - `cumetal-ptx2llvm`: PTX text to LLVM IR (AIR metadata scaffold) via the phase1 pipeline,
    including concrete vector-add and matrix-multiply body emission for recognized signatures
  - PTX signature lowering now also covers unary `negate` and `reduce_sum` (atomic add) kernels
    used in regression tests for `neg.f32`, `shl.b64`, and `atom.global.add.f32` paths
  - intrinsic-lowering opcode coverage expanded for `div`, `rem`, `and`, `or`, `xor`, `not`,
    `selp`, and `rcp` instruction roots, with strict PTX sweep coverage
  - math intrinsic lowering extended: `fma`, `max/min/abs` (with float/int variants),
    `sqrt`, `rsqrt`, `ex2`→`exp2`, `lg2`→`log2`, `sin`, `cos`
  - warp primitive lowering: `shfl.sync.{idx,down,up,bfly}` → `air.simdgroup.shuffle*`,
    `vote.sync.{ballot,any,all}` → `air.simdgroup.{ballot,any,all}`,
    `bar.warp.sync` → `air.simdgroup.barrier` (__syncwarp emulation)
  - memory barrier lowering: `membar.gl/sys` → `air.mem.barrier.device`,
    `membar.cta` → `air.mem.barrier.threadgroup` (__threadfence/__threadfence_block)
  - async copy lowering: `cp.async.*` → `air.cp_async` (serialized ld+st);
    `cp.async.commit_group/wait_group/wait_all` → `air.threadgroup_barrier`
  - phase-IR warp reduction classification maps `redux.sync.{add,and,or,xor,min,max}` to
    `air.simdgroup.reduce_*`; the generic LLVM emitter refuses these until the AIR ABI is
    validated rather than emitting the former per-lane identity placeholder.
  - parser: targeted error diagnostics for Hopper cluster ops (`cluster.*`, `mbarrier.*`),
    TMA (`cp.async.bulk.tensor.*`), and FP8 (`cvt.rn.f8*`) with specific messages
  - `cumetalc` accepts `.ptx` input via internal PTX->LLVM lowering (`--entry`, `--ptx-strict`)
  - `cumetalc` keeps the initial qualifier-stripping `.cu` prototype and now
    provides an opt-in real CUDA device frontend (`--cuda-device`) using
    Homebrew LLVM CUDA→PTX, with include/define/forced-include forwarding and
    an optional GPU inlining threshold
  - generic PTX LLVM lowering scalarizes `ld/st.v2` and `ld/st.v4` memory
    operations and lowers the scalar libdevice calls exercised by the reduced
    PhysX GRB path (sqrt/rsqrt, popcount, bit reinterpretation, min/max,
    fast division, and sin/cos)
  - PhysX 5.6 reduced GRB runtime executes production metallibs on Apple GPU;
    `cumetalc` emits exact source-level `.cumetal-abi` argument records plus
    static shared-memory byte requirements,
    Driver contexts are thread-local, and native `MTLBuffer.gpuAddress`
    allocation mode supports nested device pointers in PhysX descriptors
  - the 87-entry selected rigid PGS manifest compiles convex/convex GJK/EPA
    stage 2 from canonical non-inline NVVM through typed CuMetal IR; stage 1
    remains on the explicit legacy PTX backend because typed generic-pointer
    legalization rejects conflicting address-space flow
  - `conformance_physx_grb` compares CPU and GPU transforms for 30 resting
    contact steps at `1e-3` relative tolerance and requires Apple-GPU
    provenance through sphere narrowphase, contact pre-prep/prep, static
    solve, writeback, and integration
  - `conformance_physx_grb_friction` exercises the selected one-anchor
    sphere/plane friction solve for 60 steps, matches CPU through sliding and
    no-slip rolling, and verifies a friction-disabled negative control;
    generic friction correlation remains a measured gap
  - `conformance_physx_grb_multibody` runs two independent dynamic spheres
    against the plane for 30 steps, isolates each contact pre-prep/prepare
    batch in one Metal SIMD group, and checks all transforms and velocities
    against CPU
  - `conformance_physx_grb_stacked` runs one selected sphere/sphere dynamic
    contact above the plane for 30 frictional and frictionless steps, requires
    dynamic solve, slab reset, motion-writeback, and integration provenance,
    adds selected box/box and six-vertex convex/convex frictionless stacks, and
    rejects a one-body stacked scene; the convex gate uses a documented 1%
    component-wise envelope, while larger stacks, general convex topology, and
    general batching remain open
  - expanded PTX sweep harness (`tests/ptx_sweep`) for strict-mode supported/unsupported opcode checks
  - initial `intrinsic_lower` pass for thread-index/barrier/basic-math mappings
  - initial `printf_lower` pass for PTX `printf`/`vprintf` call extraction and format-table metadata
  - initial `addrspace` pass for shared/global/local load-store + `cvta.to.*` rewrites
  - initial `metadata` pass for AIR-style kernel metadata fields
  - initial phase1 pipeline API chaining parser + passes for a selected PTX entry
  - PTX parser handles entry attributes between signature/body (e.g. `.maxntid`, `.minnctapersm`)
    and `.param` qualifiers (`.ptr`, `.align`) used by clang-emitted PTX
- Early Phase 0 runtime path:
  - allocation tracking (`ptr -> MTLBuffer`) with offset resolution
  - optional `MTLHeap`-backed sub-allocation path for `cudaMalloc` / `cuMemAlloc`
    (`CUMETAL_MTLHEAP_ALLOC=1`, chunk size override: `CUMETAL_MTLHEAP_CHUNK_BYTES`)
  - synchronous `cudaMemcpy` on UMA via `memcpy`
  - kernel launch through Metal compute pipelines (`setBuffer` + `setBytes`)
  - default-stream, per-thread default stream, and user-stream execution
    (`cudaStreamCreate/Destroy/Synchronize`, `cudaStreamPerThread`, `cudaStreamLegacy`)
  - runtime functional tests for vector add, matrix multiply, and saxpy
  - initial library shims for cuRAND and cuBLAS v2
  - cuBLAS `cublasSgemm`/`cublasSgemmStridedBatched` backed by MetalPerformanceShaders GEMM
  - driver module loading from both in-memory metallib bytes and filesystem paths
  - on-disk cache for `cuModuleLoadData` metallib byte payloads
  - driver stream/event/memory APIs enforce `cuInit` + current-context requirements
  - shared runtime artifact: `libcumetal.dylib` (plus `cuda.h` / `cuda_runtime.h` install headers)
  - startup conflict warning if another `libcuda.dylib` is already loaded
  - Metal command-buffer failures map to CUDA timeout/illegal-address/devices-unavailable errors
  - default module cache root: `$HOME/Library/Caches/io.cumetal/kernels` (override: `CUMETAL_CACHE_DIR`)
  - `samples/vectorAdd` source flow exercised end-to-end (compile `.cu` with `cumetalc`, link host app
    against `libcumetal`, execute and validate output)
  - opt-in registration path symbols for binary-shim style launches
    (`__cudaRegisterFatBinary`, `__cudaRegisterFatBinary2`, `__cudaRegisterFatBinary3`,
    `__cudaRegisterFatBinaryEnd`, `__cudaRegisterFunction`, `__cudaRegisterVar`,
    `__cudaRegisterManagedVar`,
    `__cudaPushCallConfiguration`)
  - legacy runtime launch path (`cudaConfigureCall` / `cudaSetupArgument` / `cudaLaunch`)
  - llm.c FP32 CUDA stress binary can be built and executed through CuMetal registration path
    using `scripts/build_llmc_test_gpt2fp32cu.sh` + `scripts/run_llmc_test_gpt2fp32cu.sh`
  - `conformance_llmc_gpt2fp32cu` enforces numerical parity markers plus
    successful Apple-GPU launch provenance; measured on Apple M4 Pro, it passes
    with `overall okay: 1` and CPU emulation disabled
  - llm.c harness build shim supports `CUMETAL_LLMC_GRAD_TOL` (default `1.2e-2`) to tune
    gradient-check tolerance applied to the generated test translation unit
  - llm.c runtime CPU emulation is disabled by default and can be explicitly enabled only for
    diagnostics (`CUMETAL_ENABLE_LLMC_CPU_EMULATION=1`); it remains traceable with
    `CUMETAL_TRACE_LLMC_EMULATION=1`. Use `CUMETAL_TRACE_GPU=1` to verify Metal dispatch.
  - the passing llm.c workload uses specialized Metal replacements in
    `compiler/ptx/src/lower_to_metal.cpp`; it does not establish general PTX
    compatibility
  - PTX sweep extended with 30+ new test cases: `shfl.sync.{idx,down,up,bfly}`,
    `vote.sync.{ballot,any,all}`, `bar.warp.sync`, `membar.{gl,cta,sys}`,
    `cp.async.{ca,commit_group,wait_all}`,
    and math intrinsics `sqrt`, `rsqrt`, `ex2`, `lg2`, `sin`, `cos`, `fma`, `abs`, `min`, `max`
  - Unsupported-op sweep extended with targeted diagnostic cases for Hopper cluster ops
    (`cluster.sync.aligned`, `mbarrier.init`, `mbarrier.arrive`), TMA
    (`cp.async.bulk.tensor.1d.*`), and FP8 (`cvt.rn.f8x2.*`)
  - `--fp64=native|emulate|warn` flag added to `cumetalc` (spec §8.1); `warn` mode emits
    per-instruction warnings for `.f64` opcodes; `emulate` implements Dekker FP32-pair
    decomposition for generic FP64 register arithmetic, independent of entry name; unsupported
    binary64 memory/conversion boundaries fail lowering. Runtime defaults to `kEmulate` because
    Apple Silicon GPU rejects `fmul double` in Metal pipelines at runtime (set
    `CUMETAL_FP64_MODE=native` to force native mode for compilation-path testing)
  - functional tests added:
    - `functional_runtime_warp_shuffle` (simd_shuffle broadcast, 64 threads, lane-0 broadcast)
    - `functional_runtime_fp16_ops` (half-precision add, 256 elements, exact integer check)
    - `functional_runtime_shared_reduce` (256-thread tree reduction, output[0]==256.0)
    - `functional_runtime_grid_2d` (4×4 grid of 2×2 blocks, linear index check)
    - `functional_runtime_grid_3d` (2×3×4 grid of 2×2×2 blocks, 3D linear index check)
    - `functional_runtime_fp64_ops` (PTX fma.rn.f64 via driver API; PASS via emulate mode)
    - `functional_runtime_atomic_shared` (threadgroup atomic, 128 blocks×256 threads=32768)
    - `functional_runtime_warp_vote` (simd_any/all/ballot; 64 threads, ballot=0x55555555)
    - `functional_runtime_struct_arg` (struct by-value argument via CUMETAL_ARG_BYTES)
    - `functional_runtime_barrier_order` (thread 0 writes sentinel; all threads verify post-barrier)
    - `functional_runtime_cp_async_emul` (cp.async emulated as ld+st+threadgroup_barrier)
    - `functional_runtime_warp_partial_mask` (reference identity behavior for non-member shuffle lanes)
    - `functional_runtime_warp_mask_votes_cu` (real CUDA frontend/PTX/AIR execution for partial-mask vote, activemask, shuffle, divergent masked barriers, and automatic static shared memory)
  - intrinsic lowering: `brev.b32/b64` → `llvm.bitreverse.i32/i64` added to pass and parser
  - intrinsic_lower unit tests: Test 6 (abs/shr), Test 7 (brev), Test 8 (f32/f64 math, b64 bitwise)
  - PTX sweep: expanded to 93+ cases covering all kSupportedRoots opcode roots including:
    - `clz.b64`, `popc.b64` (64-bit bit-count ops)
    - `add/sub/mul/div.f32` (basic float arithmetic)
    - `neg/abs/min/max.f64` (double-precision unary/binary)
    - `and/or/xor/not.b64` (64-bit bitwise ops)
    - `mul.lo.u64`, `rem.u32`, `rem.s64`
    - `abs.{s32,s64,f32,f64}`, `shr.{b32,u32,s32,b64,u64,s64}`
    - `vote.{ballot,any,all}` non-sync forms
    - `st.global.{u32,u64,f64}`, `ld.global.{u8,s8,u16,s16}`
    - `atom.global.{cas,and,or,xor,min,max,exch}.b32`
    - `redux.sync.{min,max}.f32`
    - partial-mask variants: `shfl.sync` and `vote.sync.ballot` with mask=0x0000FFFF

Supported runtime API subset:

- `cudaInit`, `cudaDriverGetVersion`, `cudaRuntimeGetVersion`
- `cudaGetDeviceCount`, `cudaGetDevice`, `cudaSetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`
- `cudaSetDeviceFlags`, `cudaGetDeviceFlags`
- `cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaFree`
- `cudaMallocAsync`, `cudaMallocFromPoolAsync`, `cudaFreeAsync` (free lifetime is deferred in
  stream order; allocation does not drain the stream)
- `cudaHostAlloc`, `cudaFreeHost`, `cudaHostGetDevicePointer`, `cudaHostGetFlags`
- `cudaMemGetInfo`
- `cudaMemcpy`, `cudaMemcpyAsync`
- `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`, `cudaMemcpyToSymbolAsync`, `cudaMemcpyFromSymbolAsync`
  - External PTX constants referenced directly by a kernel entry are laid out in
    one aligned, at-most-64-KB module buffer and bound read-only at Metal index 30.
    The unmodified `LargeKernelParameter` CUDA sample passes both its 4 KB and
    32,764-byte parameter cases; unmodified `convolutionSeparable` passes with
    zero relative L2 error.
  - Runtime-registered writable `__device__` globals use persistent shared
    Metal buffers. Symbol APIs and kernels resolve the same bytes; the
    unmodified `threadFenceReduction` sample passes its single-pass 64-block
    reduction with exact CPU/GPU agreement.
- `cudaGetSymbolAddress`, `cudaGetSymbolSize`
- `cudaMemset`, `cudaMemsetAsync`
- `cudaLaunchKernel`
- `cudaConfigureCall`, `cudaSetupArgument`, `cudaLaunch`
- `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`
- `cudaStreamSynchronize`, `cudaStreamQuery`, `cudaStreamAddCallback`
- `cudaStreamWaitEvent`
- `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventRecord`
- `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventElapsedTime`, `cudaEventDestroy`
- `cudaDeviceReset`
- `cudaDeviceSynchronize`
- `cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorName`, `cudaGetErrorString`
- `cudaProfilerStart`, `cudaProfilerStop`
- `cudaFuncGetAttributes`, `cudaFuncSetCacheConfig`, `cudaFuncSetSharedMemConfig`, `cudaFuncSetAttribute`
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize`
- `cudaPointerGetAttributes`, `cudaChooseDevice`
- `cudaStreamCreateWithPriority` (priority ignored; creates regular stream)
- `cudaDeviceSetLimit` (no-op), `cudaDeviceGetLimit` (returns sensible defaults)
- `cudaLaunchCooperativeKernel` (single threadgroup supported; multi-block launch returns
  `cudaErrorNotSupported`)
- `cudaDeviceSetCacheConfig`, `cudaDeviceGetCacheConfig` (no-op stubs; all memory is UMA)
- `cudaDeviceSetSharedMemConfig`, `cudaDeviceGetSharedMemConfig` (no-op stubs)
- `cudaMemPrefetchAsync`, `cudaMemAdvise`, `cudaMemRangeGetAttribute` (meaningful no-ops on Apple Silicon UMA)
- `cudaDeviceGetStreamPriorityRange` (returns 0,0 — Metal has no priority queues)
- `cudaMemcpy2D`, `cudaMemcpy2DAsync`, `cudaMemset2D` (row-by-row on UMA)
- `cudaMallocPitch` (aligned 2D allocation; pitch rounded to 512 bytes)
- `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaDeviceDisablePeerAccess` (no-op stubs; single GPU)
- `cuda_runtime_api.h` forwarding header (programs that include this directly)

Device intrinsics added to `cuda_runtime.h`:
- Type-punning: `__int_as_float`, `__float_as_int`, `__uint_as_float`, `__float_as_uint`, `__longlong_as_double`, `__double_as_longlong`
- Integer: `__mulhi`, `__umulhi`, `__mul24`, `__umul24`, `__sad`, `__usad`
- Fast math: `__sinf`, `__cosf`, `__tanf`, `__expf`, `__exp2f`, `__logf`, `__log2f`, `__log10f`, `__powf`, `__sqrtf`, `__rsqrtf`, `__fdividef`, `__frcp_rn`, `__fsqrt_rn`
- Lane masks: `__lanemask_eq`, `__lanemask_lt`, `__lanemask_le`, `__lanemask_gt`, `__lanemask_ge`
- Warp reductions: `__reduce_add_sync`, `__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync`, `__reduce_min_sync`, `__reduce_max_sync`
- Warp shuffle: `__shfl_sync`, `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync` (int + float overloads; partial member masks predicate caller results)
- Warp vote: `__any_sync`, `__all_sync`, `__ballot_sync` (member mask intersected with the real active-lane ballot)
- Double atomics: `atomicAdd(double*, double)` via 64-bit CAS loop

`cuda_fp16.h` expanded:
- Comparison: `__hge`, `__hle`
- Math: `__hfma`, `__hneg`, `__habs`, `__hmax`, `__hmin`
- Conversions: `__half2int_rn`, `__half2uint_rn`, `__half2short_rn`, `__half2ll_rn`, `__int2half_rn`, `__uint2half_rn`, `__short2half_rn`, `__ll2half_rn`

Driver API additions:
- `cuMemAllocPitch`, `cuCtxEnablePeerAccess`, `cuCtxDisablePeerAccess`
- `cuCtxGetStreamPriorityRange` (returns 0,0)
- `cuLaunchHostFunc` (launches a CPU callback asynchronously on a stream; implemented via `cudaStreamAddCallback`)

`cudaDeviceProp` fields now populated per spec §6.8:
- `unifiedAddressing = 1`, `managedMemory = 1`, `concurrentManagedAccess = 1` (UMA)
- `maxBufferArguments = 31` (Metal buffer argument limit)
- `clockRate`, `memoryClockRate` (1296000 kHz), `memoryBusWidth` (128-bit)
- `totalConstMem` (64 KB), `sharedMemPerMultiprocessor`, `maxThreadsPerMultiProcessor` (2048)
- `l2CacheSize` (4 MB), `canMapHostMemory = 1`, `integrated = 1`, `concurrentKernels = 1`
- `asyncEngineCount = 0`, `computeMode = cudaComputeModeDefault`
- `pciBusID`, `pciDeviceID`, `pciDomainID` (all 0 — no discrete PCI GPU)
- `tccDriver = 0`, `kernelExecTimeoutEnabled = 0`
- `pageableMemoryAccess = 0`, `pageableMemoryAccessUsesHostPageTables = 0`;
  Apple Silicon is UMA, but CuMetal currently binds only tracked Metal-backed
  allocations, not arbitrary `malloc` pointers
- `persistingL2CacheMaxSize = 0`, `accessPolicyMaxWindowSize = 0`; the fields
  consume reserved ABI space instead of growing `cudaDeviceProp`

`cudaComputeMode` enum added: `cudaComputeModeDefault`, `cudaComputeModeExclusive`, `cudaComputeModeProhibited`, `cudaComputeModeExclusiveProcess`

`cudaDeviceGetAttribute` and `cuDeviceGetAttribute` now support additional attributes:
- `cudaDevAttrComputeCapabilityMajor` / `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` → 8
- `cudaDevAttrComputeCapabilityMinor` / `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR` → 0
- `cudaDevAttrMaxRegistersPerBlock` / `CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK` → 65536
- `cudaDevAttrClockRate` / `CU_DEVICE_ATTRIBUTE_CLOCK_RATE` → 1296000 kHz
- `cudaDevAttrTextureAlignment` → 512 bytes
- `cudaDevAttrGpuOverlap` / `CU_DEVICE_ATTRIBUTE_GPU_OVERLAP` → 1
- `cudaDevAttrMemoryBusWidth` → 128, `cudaDevAttrL2CacheSize` → 4 MB
- `cudaDevAttrMaxThreadsPerMultiProcessor` → 2048, `cudaDevAttrMemoryClockRate` → 1296000
- `cudaDevAttrIntegrated` → 1, `cudaDevAttrCanMapHostMemory` → 1
- `cudaDevAttrComputeMode` → 0, `cudaDevAttrConcurrentKernels` → 1
- `cudaDevAttrPciBusId`, `cudaDevAttrPciDeviceId`, `cudaDevAttrPciDomainId` → 0
- `cudaDevAttrTccDriver` → 0, `cudaDevAttrKernelExecTimeout` → 0, `cudaDevAttrAsyncEngineCount` → 0
- `cudaDevAttrPageableMemoryAccess` → 0, `cudaDevAttrPageableMemoryAccessUsesHostPageTables` → 0
- `cudaDevAttrSharedMemPerBlockOptin` → sharedMemPerBlock

`cooperative_groups::thread_block_tile<N>` extended with:
- `shfl(val, src_rank)`, `shfl_down(val, delta)`, `shfl_xor(val, mask)`
- `any(pred)`, `all(pred)`, `ballot(pred)` (via `__nvvm_vote_*` builtins)
- `cooperative_groups::less<T>` binary operator alongside existing `plus<T>` and `greater<T>`

CUDA vector types added to `cuda_runtime.h`:
- All standard types: `char2/3/4`, `short2/3/4`, `int2/3/4`, `uint2/4`,
  `long2/4`, `longlong2/4`, `ulong2/4`, `ulonglong2/4`,
  `float2/3`, `double2/3/4` with `__align__` annotations and `make_*` constructors

Device atomics added (CUDA device code path, spec §6.7):
- `atomicSub`, `atomicExch` (int/uint/float), `atomicMin`/`atomicMax` (int/uint),
  `atomicCAS` (uint/int/ull), `atomicAnd`/`atomicOr`/`atomicXor` (int/uint)
- System-scope 32-bit signed/unsigned
  `atomic{Add,Exch,Min,Max,CAS,And,Or,Xor,Inc,Dec}_system` on tracked managed
  allocations. A focused test observes host and GPU atomic contributions to the
  same UMA bytes. Arbitrary pageable pointers are not included in this claim.

Persisting-L2 compatibility surface added:
- `cudaAccessPolicyWindow`, `cudaStreamAttrValue`, access-property and stream-attribute enums
- `cudaStreamSetAttribute`, `cudaStreamGetAttribute`, and
  `cudaCtxResetPersistingL2Cache` return `cudaErrorNotSupported` because public
  Metal exposes no equivalent policy; nonzero `cudaLimitPersistingL2CacheSize`
  requests are likewise rejected rather than silently accepted

Device intrinsics added (guarded by `#ifndef __CLANG_CUDA_DEVICE_FUNCTIONS_H__`):
- `__syncwarp`, `__threadfence`, `__threadfence_block`, `__threadfence_system`
- `__activemask`, `__popc`/`__popcll`, `__clz`/`__clzll`, `__brev`/`__brevll`
- `__ffs`/`__ffsll`, `__fmaf_rn`, `__fma_rn`

`install.sh` / `uninstall.sh` now detect fish shell (`$SHELL=*/fish`) and write
`set -gx` syntax to `~/.config/fish/config.fish`; `CUMETAL_SHELL_RC` overrides.

Supported driver API subset:

- `cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem`, `cuDeviceGetAttribute`
- `cuCtxCreate`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxGetDevice`, `cuCtxGetFlags`, `cuCtxSetFlags`, `cuCtxSynchronize`
- `cuStreamCreate`, `cuStreamDestroy`, `cuStreamSynchronize`, `cuStreamQuery`, `cuStreamAddCallback`, `cuStreamWaitEvent`
- `cuEventCreate`, `cuEventDestroy`, `cuEventRecord`, `cuEventQuery`, `cuEventSynchronize`, `cuEventElapsedTime`
- `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleUnload`, `cuModuleGetFunction`
- `cuModuleLoadData` accepts metallib bytes/paths and PTX text images (including basic CUDA fatbin wrapper PTX variants)
- `cuModuleLoadDataEx` accepts option arrays in compatibility mode (options are currently ignored)
- `cuLaunchKernel` (kernel params path and `extra` packed-argument path)
- `cuMemAlloc`, `cuMemAllocManaged`, `cuMemFree`
- `cuMemGetInfo`
- `cuMemAllocHost`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostGetFlags`, `cuMemFreeHost`
- `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoD`
- `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`
- `cuMemsetD8`, `cuMemsetD8Async`
- `cuGetErrorName`, `cuGetErrorString`
- `cuProfilerStart`, `cuProfilerStop`
- `cuOccupancyMaxActiveBlocksPerMultiprocessor`, `cuOccupancyMaxPotentialBlockSize`
- `cuFuncGetAttribute`, `cuFuncSetCacheConfig`
- `cuStreamCreateWithPriority` (priority ignored; creates regular stream)
- `cuLaunchCooperativeKernel` (forwards to `cuLaunchKernel`; threadgroup CG works)
- `cuMemsetD16`, `cuMemsetD32`, `cuMemsetD16Async`, `cuMemsetD32Async`
- `cuDeviceComputeCapability` (returns 8.0 — synthetic Ampere-equivalent)
- `cuDeviceCanAccessPeer` (returns 0; single GPU on Apple Silicon)

Public headers now installed: `cuda.h`, `cuda_runtime.h`, `cuda_fp16.h`, `cuda_bf16.h`,
`cublas_v2.h`, `cublas_api.h`, `cublasLt.h`, `cufft.h`, `curand.h`, `cusparse.h`,
`cusolver_common.h`, `cusolverDn.h`, `cudnn.h`, `nvml.h`, `nccl.h`,
`cooperative_groups.h`, `cooperative_groups/reduce.h`, `cuComplex.h`,
`nvToolsExt.h`, `nvtx3/nvToolsExt.h`.

Forwarding headers (route to existing implementations):
`device_launch_parameters.h`, `driver_types.h`, `library_types.h`,
`channel_descriptor.h`, `device_atomic_functions.h`, `math_functions.h`,
`cuda_profiler_api.h`, `cuda_occupancy.h`, `cuda_runtime_api.h`,
`sm_20_intrinsics.h`, `sm_30_intrinsics.h`, `sm_60_intrinsics.h`,
`sm_70_intrinsics.h`, `sm_80_intrinsics.h`.

Header-only library shims:
- **thrust** (`thrust/`): `device_vector`, `host_vector`, `device_ptr`, `sort`,
  `sort_by_key`, `stable_sort`, `reduce`, `transform_reduce`, `inclusive_scan`,
  `exclusive_scan`, `transform`, `fill`, `copy`, `for_each`, `unique`, `sequence`,
  `counting_iterator`, `zip_iterator`, `transform_iterator`, `constant_iterator`,
  `discard_iterator`, `permutation_iterator`, `execution_policy`, `functional`, `pair`.
  CPU-backed on UMA (device memory is host-accessible).
- **CUB** (`cub/`): `BlockReduce`, `BlockScan`, `BlockExchange`, `BlockLoad`,
  `BlockStore`, `WarpReduce`, `WarpScan`, `DeviceReduce` (Sum/Min/Max/ArgMin/ArgMax),
  `DeviceScan` (Inclusive/Exclusive), `DeviceRadixSort` (SortKeys/SortPairs),
  `DeviceSelect` (If/Flagged/Unique), `DeviceHistogram` (Even/Range),
  `DeviceRunLengthEncode` (Encode/NonTrivialRuns).
  Sequential fallback for host-side compilation; device ops run on UMA.
- **NVTX** (`nvtx3/nvToolsExt.h`): No-op stubs for profiling annotations.
  Range push/pop, mark, domain API, naming API all silently ignored.
`cuda_fp16.h` provides host-side `__half` (IEEE 754 float16 via bit manipulation) and
device-side `__half = _Float16`; `atomicAdd(__half*, __half)` via CAS loop (spec §8).

Supported library shim subset:

- cuRAND (`curand.h`)
  - `curandCreateGenerator`, `curandDestroyGenerator`
  - `curandGetVersion`
  - `curandSetStream`, `curandGetStream`
  - `curandSetPseudoRandomGeneratorSeed`, `curandSetGeneratorOffset`
  - `curandGenerate` (uint32 output), `curandGenerateLongLong` (uint64 output)
  - `curandGenerateUniform`, `curandGenerateUniformDouble`
  - `curandGenerateNormal`, `curandGenerateNormalDouble`
  - `curandGenerateLogNormal`, `curandGenerateLogNormalDouble`
  - generation is enqueued on the bound stream; state-changing setters and destruction wait
    for prior generator work so callbacks cannot race generator lifetime/state.
- cuFFT (`cufft.h`)
  - `cufftCreate`, `cufftDestroy`, `cufftSetStream`, `cufftGetSize`, `cufftGetVersion`
  - `cufftPlan1d`, `cufftPlan2d`, `cufftPlan3d`, `cufftPlanMany`
  - `cufftMakePlan1d`, `cufftMakePlan2d`, `cufftMakePlan3d`, `cufftMakePlanMany`
  - `cufftExecC2C`, `cufftExecR2C`, `cufftExecC2R` (single-precision)
  - `cufftExecZ2Z`, `cufftExecD2Z`, `cufftExecZ2D` (double-precision)
  - Backed by Apple Accelerate `vDSP_DFT_Execute` (arbitrary N, any batch size)
  - `libcufft.dylib` symlink alias to `libcumetal.dylib`
- cuBLAS v2 (`cublas_v2.h`)
  - `cublasCreate`, `cublasDestroy`, `cublasGetVersion`
  - `cublasSetStream`, `cublasGetStream`
  - `cublasSetMathMode`, `cublasGetMathMode`
  - `cublasSaxpy`, `cublasSscal`, `cublasScopy`, `cublasSgemm`
  - `cublasSgemmStridedBatched`, `cublasDgemmStridedBatched`
  - `cublasSswap`, `cublasDswap`
  - `cublasSdot`, `cublasDdot`
  - `cublasSasum`, `cublasDasum`
  - `cublasSnrm2`, `cublasDnrm2`
  - `cublasIsamax`, `cublasIdamax`
  - `cublasIsamin`, `cublasIdamin`
  - `cublasSgemv`, `cublasDgemv`
  - `cublasSger`, `cublasDger`
  - `cublasSsymv`, `cublasDsymv`
  - `cublasDaxpy`, `cublasDscal`, `cublasDcopy`, `cublasDgemm`

Library alias compatibility:

- Build/install also provides `libcublas.dylib` and `libcurand.dylib` aliases to
  `libcumetal.dylib`, so software linked against CUDA library names can resolve shim symbols.
- Optional binary-shim alias: when `CUMETAL_ENABLE_BINARY_SHIM=ON`, build/install also provides
  `libcuda.dylib -> libcumetal.dylib`.

Known limitations (see spec §2.2 and §8):

- Default kernel launch uses a CuMetal descriptor (`cumetalKernel_t`).
- Binary-shim registration: CuMetal `CMTL` envelopes, direct PTX images, and basic CUDA
  fatbin PTX images are supported; full NVCC fatbinary variants are not yet implemented.
- CUDA Graphs: tested dependency-ordered kernel/linear-memcpy/memset/host-node capture and
  replay, cloning, root introspection, kernel-node parameter updates, and
  topology-compatible executable updates are implemented. Graph memory nodes,
  cross-stream event-capture topology, and advanced node types remain incomplete.
- Dynamic parallelism: compile-time error per spec §2.2.
- Texture/surface object lifecycle and CUDA-array memcpy are implemented; general
  device-side texture/surface sampling remains deferred per spec §2.2 and §8.
- Multi-GPU peer access: single GPU only on Apple Silicon; peer APIs return appropriate errors.
- CUDA graphics interop (OpenGL/Vulkan): non-goal per spec §2.2.
- Multi-block cooperative launch is rejected with not-supported because Metal has no
  cross-threadgroup barrier; single-block cooperative sync uses a threadgroup barrier.
- Masked `__syncwarp` uses AIR SIMD-group scope with threadgroup-memory visibility;
  divergent half-warp ordering, partial-mask vote/ballot, activemask, shuffle caller
  membership, and automatic source-path static shared memory are GPU-tested (spec §5.3).
- FP64: generic FP32-pair register emulation is available at ~44-bit mantissa; unsupported
  binary64 memory/conversion forms fail compilation (spec §8.1).
- Device printf: buffer-based; format strings limited to 256 bytes (spec §5.3).
