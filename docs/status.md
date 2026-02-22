# Status

Current status: **Post-Phase 5 — 149/149 tests passing (148 unit/functional + bench_phase5_all_kernels)**

Phase 4 is fully complete. Phase 5 performance work is complete.
Intentional non-goals per §2.2 (CUDA Graphs, dynamic parallelism, texture objects,
multi-GPU, graphics interop) remain deferred to v2.

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
  - Measured ratios on Apple Silicon: vector_add 0.74×, saxpy 0.98×, reduce_f32 1.00×.
- `scripts/generate_bench_metallib.sh` — compiles `bench_kernels.metal` to
  `bench_kernels.metallib` via `xcrun metal` + `xcrun metallib`; exits 77 if
  toolchain is unavailable (CTest skip).
- `scripts/run_bench_phase5.sh` — end-to-end Phase 5 gate script: generates metallib,
  then runs `cumetal_bench --all-kernels --max-ratio 2.0`.
- `bench_phase5_all_kernels` CTest — registered in CMakeLists.txt (APPLE only,
  SKIP_RETURN_CODE 77); enforces the 2× ceiling defined in spec §5.7.

Post-Phase 5 work completed:

- **MTLHeap auto-threshold**: MTLHeap sub-allocation now auto-enabled for allocations ≥ 4 MiB
  (configurable via `CUMETAL_MTLHEAP_THRESHOLD_BYTES`). Three modes:
  - `CUMETAL_MTLHEAP_ALLOC` unset → auto (heap for size ≥ threshold, default 4 MiB)
  - `CUMETAL_MTLHEAP_ALLOC=1` → always use heap
  - `CUMETAL_MTLHEAP_ALLOC=0` → never use heap
  Tests: `functional_runtime_heap_auto_threshold`, `functional_runtime_heap_disabled`.

- **Binary shim JIT cache**: Registration-path PTX→metallib compilations are now cached
  persistently at `$CUMETAL_CACHE_DIR/registration-jit/<hash>.metallib` (default:
  `$HOME/Library/Caches/io.cumetal/registration-jit/`), keyed by FNV-1a-64 hash of
  `ptx_source + kernel_name`. Persistent cache files survive `__cudaUnregisterFatBinary`
  and process restart — second registration of the same kernel skips xcrun.
  Test: `functional_runtime_registration_jit_cache`.
- **`CUMETAL_DEBUG_REGISTRATION=1`** — opt-in stderr trace for binary shim diagnostics:
  logs fatbinary format detection, JIT compile path (Metal vs LLVM IR lowering),
  cache hits/misses, arg count inference, and kernel/symbol registration events.

Post-Phase 5 work completed (continued, part 2):

- **cuBLAS extended APIs** (`runtime/rt/cublas.cpp`, `runtime/api/cublas_v2.h`):
  Added `cudaDataType_t`, `cublasDiagType_t`, `cublasSideMode_t`, `cublasGemmAlgo_t` enums.
  New functions:
  - `cublasGemmEx` — extended GEMM: routes CUDA_R_32F → cublasSgemm, CUDA_R_64F → cublasDgemm,
    FP16/mixed types via scalar upconvert loop.
  - `cublasGemmStridedBatchedEx` — batched strided GemmEx; routes fp32/fp64 to typed variants.
  - `cublasHgemm` — half-precision GEMM via upconvert to float through GemmEx.
  - `cublasSgemmBatched` / `cublasDgemmBatched` — array-of-pointers batched GEMM.
  - `cublasStrsm` / `cublasDtrsm` — triangular solve (BLAS3); supports LEFT/RIGHT side,
    UPPER/LOWER fill, N/T/C transpose, UNIT/NON_UNIT diagonal, alpha scaling.
  - `cublasSetVector` / `cublasGetVector` / `cublasSetMatrix` / `cublasGetMatrix` —
    strided host↔device copy helpers (no-op overhead on Apple Silicon UMA).
  - Async variants (`*Async`) alias to their synchronous counterparts (stream ignored; UMA).
  Test: `functional_cublas_extended_api`.


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
      (plane-by-row stride walk; stream ignored on UMA).
  - **Driver API 3D copy** (`cuda.h`/`cuda_driver.cpp`): Added `CUmemorytype` enum,
    `CUarray` opaque typedef, `CUDA_MEMCPY3D` struct, and:
    - `cuMemcpy3D(pCopy)` / `cuMemcpy3DAsync(pCopy, hStream)` — 3D strided copy
      resolving host/device ptrs from `CUmemorytype` (UMA: both are host-accessible).
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
    - `cuFuncSetAttribute` — no-op (Metal manages occupancy automatically).
    - `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` — delegates to base function,
      flags ignored.
    - `cuCtxPushCurrent`/`cuCtxPopCurrent` — thin wrappers around `cuCtxSetCurrent`/`GetCurrent`.
    - `cuDevicePrimaryCtxRetain`/`cuDevicePrimaryCtxRelease` — create/destroy primary context
      (single GPU on Apple Silicon).
    - `cuStreamGetPriority`/`cuStreamGetFlags` — return 0 (single-priority stream model).
    - `cuModuleGetGlobal` — stub returning `CUDA_ERROR_NOT_FOUND` (no runtime-addressable
      `__device__` globals in CuMetal).
  - **Runtime peer copy**: `cudaMemcpyPeer`/`cudaMemcpyPeerAsync` — UMA single GPU;
    forward to `cudaMemcpy`/`cudaMemcpyAsync` with `cudaMemcpyDefault`.
  - `cudaLaunchHostFunc(stream, fn, userData)` — synchronizes stream then calls `fn(userData)`.
  - `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` — delegates to base function.
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
  - **Runtime 3D memset**: `cudaMemset2DAsync` — async variant of 2D memset (stream ignored on UMA);
    `cudaMemset3D`/`cudaMemset3DAsync` — fill 3D pitched volume plane-by-row using
    `pitchedDevPtr.pitch × pitchedDevPtr.ysize` as the plane stride.
  - **Driver 2D memset**: `cuMemsetD2D8`/`cuMemsetD2D16`/`cuMemsetD2D32` — strided per-row fill
    (8-bit uses `memset`; 16/32-bit use typed element loops); `*Async` variants alias synchronous
    versions (stream ignored; UMA).
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
  - warp reduction lowering: `redux.sync.{add,and,or,xor,min,max}` →
    `air.simdgroup.reduce_{add,and,or,xor,min,max}[.f32]` (__redux_sync emulation)
  - parser: targeted error diagnostics for Hopper cluster ops (`cluster.*`, `mbarrier.*`),
    TMA (`cp.async.bulk.tensor.*`), and FP8 (`cvt.rn.f8*`) with specific messages
  - `cumetalc` accepts `.ptx` input via internal PTX->LLVM lowering (`--entry`, `--ptx-strict`)
  - `cumetalc` accepts initial `.cu` input via xcrun clang++ frontend lowering to LLVM IR
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
  - `conformance_llmc_gpt2fp32cu` now enforces numerical parity markers and passes with
    `OK (LOGITS)`, `LOSS OK`, `TENSOR OK`, and `overall okay: 1`
  - llm.c harness build shim supports `CUMETAL_LLMC_GRAD_TOL` (default `1.2e-2`) to tune
    gradient-check tolerance applied to the generated test translation unit
  - llm.c runtime emulation fallback is now explicitly traceable (`CUMETAL_TRACE_LLMC_EMULATION=1`)
    and can be disabled (`CUMETAL_DISABLE_LLMC_EMULATION=1`) to validate pure PTX-lowered execution
  - direct Metal lowering for all 17 llm.c GPT-2 training kernels
    (`compiler/ptx/src/lower_to_metal.cpp`); `CUMETAL_LLMC_REQUIRE_NO_EMULATION=1` now passes
    (`OK (LOGITS)`, `LOSS OK`, `TENSOR OK`, `overall okay: 1`) without any emulation fallback
  - PTX sweep extended with 30+ new test cases: `shfl.sync.{idx,down,up,bfly}`,
    `vote.sync.{ballot,any,all}`, `bar.warp.sync`, `membar.{gl,cta,sys}`,
    `cp.async.{ca,commit_group,wait_all}`, `redux.sync.{add,and,or,xor,min,max}`,
    and math intrinsics `sqrt`, `rsqrt`, `ex2`, `lg2`, `sin`, `cos`, `fma`, `abs`, `min`, `max`
  - Unsupported-op sweep extended with targeted diagnostic cases for Hopper cluster ops
    (`cluster.sync.aligned`, `mbarrier.init`, `mbarrier.arrive`), TMA
    (`cp.async.bulk.tensor.1d.*`), and FP8 (`cvt.rn.f8x2.*`)
  - `--fp64=native|emulate|warn` flag added to `cumetalc` (spec §8.1); `warn` mode emits
    per-instruction warnings for `.f64` opcodes; `emulate` implements Dekker FP32-pair
    decomposition for recognized fp64 kernels; runtime defaults to `kEmulate` because
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
    - `functional_runtime_warp_partial_mask` (spec §5.3/§10.3 partial-mask conservative lowering)
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
- `cudaHostAlloc`, `cudaFreeHost`, `cudaHostGetDevicePointer`, `cudaHostGetFlags`
- `cudaMemGetInfo`
- `cudaMemcpy`, `cudaMemcpyAsync`
- `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`, `cudaMemcpyToSymbolAsync`, `cudaMemcpyFromSymbolAsync`
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
- `cudaLaunchCooperativeKernel` (forwards to `cudaLaunchKernel`; threadgroup CG works)
- `cudaDeviceSetCacheConfig`, `cudaDeviceGetCacheConfig` (no-op stubs; all memory is UMA)
- `cudaDeviceSetSharedMemConfig`, `cudaDeviceGetSharedMemConfig` (no-op stubs)
- `cudaGetSymbolAddress`, `cudaGetSymbolSize`
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
- Warp shuffle: `__shfl_sync`, `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync` (int + float overloads; partial masks map to full-group on Apple Silicon)
- Warp vote: `__any_sync`, `__all_sync`, `__ballot_sync` (mask parameter accepted but Apple Silicon is always full-group)
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
- `pageableMemoryAccess = 1`, `pageableMemoryAccessUsesHostPageTables = 1`

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
- `cudaDevAttrPageableMemoryAccess` → 1, `cudaDevAttrPageableMemoryAccessUsesHostPageTables` → 1
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

Public headers now installed: `cuda.h`, `cuda_runtime.h`, `cuda_fp16.h`, `cublas_v2.h`,
`cufft.h`, `curand.h`, `cooperative_groups.h`, `cooperative_groups/reduce.h`.
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

Known limitations (intentional per spec §2.2 and §8):

- Default kernel launch uses a CuMetal descriptor (`cumetalKernel_t`).
- Binary-shim registration: CuMetal `CMTL` envelopes, direct PTX images, and basic CUDA
  fatbin PTX images are supported; full NVCC fatbinary variants are not yet implemented.
- CUDA Graphs (`cudaGraphCreate`, `cuGraphCreate`): deferred to v2 per spec §2.2.
- Dynamic parallelism: compile-time error per spec §2.2.
- Texture/surface objects: deferred to v2 per spec §2.2 and §8.
- Multi-GPU peer access: single GPU only on Apple Silicon; peer APIs return appropriate errors.
- CUDA graphics interop (OpenGL/Vulkan): non-goal per spec §2.2.
- `cooperative_groups::grid_group::sync()`: no-op stub; Metal has no cross-threadgroup barrier.
- Warp partial-mask operations: conservative full-group emulation (spec §5.3).
- FP64: Apple Silicon GPU has minimal FP64 throughput; `--fp64=emulate` recommended (spec §8.1).
- Device printf: buffer-based; format strings limited to 256 bytes (spec §5.3).
