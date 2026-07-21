# CuMetal — CUDA Compiler & Runtime for Apple Silicon

**Status:** Design Specification v0.1 (implementation of primary v1 goals complete per README.md and docs/status.md; see Phase 4/5 and post-Phase 5 work)  
**Target Platform:** macOS 14+ on Apple Silicon (M1 / M2 / M3 / M4 families)  
**License (proposed):** Apache 2.0  
**Changelog from v0.2:** See §16 for a categorized list of changes.

---

## 1. Problem Statement

CUDA is the de-facto kernel programming model for GPU compute. Apple Silicon is the most
accessible high-bandwidth unified-memory compute platform available to individual researchers
and developers. No open-source project provides a complete path from CUDA source to execution
on Apple's Metal GPU stack.

The gap is two-layered:

1. **Compiler gap** — no toolchain that compiles CUDA device code (`.cu` kernels) into valid
   Apple AIR (the LLVM-bitcode format the Metal driver consumes).
2. **Runtime gap** — no `libcuda.dylib` that maps CUDA driver/runtime API calls to
   Metal objects.

CuMetal closes both. The **primary path** is source recompilation via `cumetalc`. A binary
compatibility shim (`libcuda.dylib`) is provided as an opt-in convenience for programs that
ship PTX in their fatbinaries, but is not the recommended deployment model (see §12.1).

### 1.1 Why Source-First

Every prior CUDA translation layer (ZLUDA, HIP-on-non-AMD, chipStar) that prioritized
binary drop-in compatibility has faced the same failure cascade: NVIDIA's EULA prohibits
translation layers targeting non-NVIDIA hardware (clause added ~2024); detection is trivial
(`cudaGetDeviceProperties` vendor string check, driver version probing); commercial software
hardcodes NVIDIA assumptions.

The enduring value is *source-level* compatibility: researchers and developers who control
their own `.cu` source files can recompile with `cumetalc` and run on Apple Silicon. This
path is legally unambiguous, technically simpler, and invulnerable to vendor kill-switches.

---

## 2. Goals and Non-Goals

### 2.1 Goals

- **Primary:** Compile unmodified CUDA C++ source files (SM 5.0 – SM 8.6 feature set) to
  Apple Silicon GPUs via `cumetalc`, requiring only a recompile — no source changes.
- **Secondary:** Provide an opt-in `libcuda.dylib` shim that satisfies CUDA Driver/Runtime
  API symbol tables, enabling PTX-shipping programs to run via JIT without recompilation.
- Compile PTX and CUDA C++ to Apple AIR through an LLVM-based pipeline with no dependency
  on Apple's closed Metal toolchain for the compilation path (only the driver ABI).
- Achieve functional correctness as the first priority; performance parity as a later milestone.
- Remain legally clean: no NVIDIA header redistribution, clean-room runtime ABI, no
  proprietary PTX opcode documentation required.

### 2.2 Non-Goals (v1)

- Support for CUDA graphics interop (OpenGL / Vulkan / DirectX surface sharing).
- Dynamic parallelism (kernels launching kernels). Deferred to v2 with CPU trampoline
  emulation.
- Multi-GPU across discrete GPUs (Apple Silicon has one GPU die; eGPU via Thunderbolt is
  explicitly out of scope).
- CUDA Graphs (MTLCommandBuffer pre-recording is a viable path but deferred).
- cuDNN / cuBLAS / cuFFT — high-level library shims are a separate project (see §11.1).
- Windows or Linux ARM (x86 translation via Rosetta 2 is explicitly out of scope).
- CUDA texture/surface objects (bindless textures). Metal textures exist but the semantic
  mapping is complex; deferred to v2 (see §8).
- Drop-in compatibility for closed-source commercial binaries (see §12.1).

### 2.3 Why Not MLIR?

LLVM was chosen because the Clang CUDA frontend emits LLVM IR directly, and the NVPTX
backend's intrinsic vocabulary is what we need to lower from. Starting with MLIR's GPU
dialect would add a translation hop with no correctness benefit in v1.

However, MLIR is an excellent fit for Phase 5 performance optimization — the GPU dialect
enables kernel fusion, tiling, and other transforms that are painful to express as LLVM
passes. The architecture explicitly preserves this option: the `air_emitter` accepts any
conforming LLVM IR module regardless of what produced it.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          User Application                               │
│               #include <cuda_runtime.h>  // CuMetal's headers          │
│                                                                         │
│  PRIMARY PATH:   cumetalc foo.cu -o foo  (static link to libcumetal)   │
│  SECONDARY PATH: link against libcuda.dylib shim (opt-in, JIT)        │
└────────────────────────────┬────────────────────────────────────────────┘
                             │  CUDA Runtime / Driver API calls
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     CuMetal Runtime (libcumetal.dylib)                   │
│                                                                         │
│  cudaMalloc ──► MTLBuffer (StorageModeShared)                           │
│  cudaMemcpy ──► memcpy (UMA — zero-copy for all directions)            │
│  cudaStream ──► MTLCommandQueue                                         │
│  cudaEvent  ──► MTLSharedEvent                                          │
│  cudaLaunch ──► MTLComputeCommandEncoder + dispatchThreadgroups         │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │ Fatbinary Registration Interceptor (opt-in binary shim only)  │      │
│  │ __cudaRegisterFatBinary → extract PTX → JIT compile           │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │ Async Error Queue (cudaGetLastError / PeekAtLastError)        │      │
│  └───────────────────────────────────────────────────────────────┘      │
└────────────────────────────┬────────────────────────────────────────────┘
                             │  .metallib load
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CuMetal Compiler Pipeline                          │
│                                                                         │
│  .cu source ──► Clang CUDA frontend ──► LLVM IR (NVPTX dialect)        │
│      OR                                                                 │
│  .ptx source ──► PTX parser (fork of ZLUDA ptx crate)                  │
│                             │                                           │
│                             ▼                                           │
│              Intrinsic Lowering Pass (CUDA → AIR)                       │
│                             │                                           │
│                             ▼                                           │
│           LLVM IR with AIR intrinsics + kernel metadata                 │
│                             │                                           │
│                             ▼                                           │
│              AIR Bitcode Serializer (.metallib container)               │
│                             │                                           │
│                             ▼                                           │
│              Validation (MetalLibraryArchive / metallib-inspect)        │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    Metal Driver (closed, Apple)
                             │
                             ▼
                    Apple Silicon GPU
```

The runtime and compiler are separate libraries. In the primary path (`cumetalc`), compilation
happens ahead of time. In the secondary path (`libcuda.dylib` shim), the compiler is invoked
lazily at module-load time (JIT). Compiled `.metallib` files are cached on disk keyed by
input hash + GPU family.

---

## 4. Repository Layout

```
cumetal/
├── compiler/
│   ├── frontend/          # Clang CUDA → LLVM IR bridge
│   ├── ptx/               # PTX text parser → LLVM IR (derived from ZLUDA)
│   ├── passes/            # LLVM transformation passes
│   │   ├── intrinsic_lower.cpp   # CUDA built-ins → AIR intrinsics
│   │   ├── addrspace.cpp         # AS(3) shared memory rewriting
│   │   ├── printf_lower.cpp      # Device printf → buffer-based printf
│   │   └── metadata.cpp          # AIR kernel metadata injection
│   ├── air_emitter/       # LLVM bitcode → .metallib container
│   ├── air_validate/      # .metallib validation via MetalLibraryArchive
│   └── cumetalc/          # CLI driver
├── runtime/
│   ├── api/               # Public headers (cuda.h, cuda_runtime.h shims)
│   ├── driver/            # CUDA Driver API implementation
│   ├── rt/                # CUDA Runtime API implementation
│   ├── registration/      # __cudaRegisterFatBinary interceptor (opt-in)
│   ├── error/             # Async error queue (per-thread last error)
│   ├── metal_backend/     # Objective-C++ Metal wrapper layer
│   └── cache/             # Compiled .metallib disk cache
├── tests/
│   ├── unit/              # Per-pass LLVM IR transformation tests
│   ├── functional/        # End-to-end kernel correctness tests
│   ├── conformance/       # CUDA API conformance suite
│   ├── air_abi/           # Cross-Xcode AIR stability regression tests
│   └── ptx_sweep/         # Per-instruction PTX bit-accuracy tests (ZLUDA-derived)
├── tools/
│   ├── air_inspect/       # Dump AIR metadata from .metallib
│   ├── ptx_diff/          # Compare PTX semantics vs AIR output
│   └── cumetal_bench/     # Benchmark runner (translated vs. native Metal)
├── install/
│   ├── install.sh         # One-command installer
│   └── uninstall.sh       # Clean removal
└── docs/
    ├── intrinsic-map.md   # Complete CUDA→AIR intrinsic mapping table
    ├── known-gaps.md      # Unsupported features and workarounds
    ├── air-abi.md         # Reverse-engineered AIR format documentation
    ├── fp64-policy.md     # FP64 emulation strategy and precision guarantees
    └── legal-notice.md    # NVIDIA EULA translation-layer clause + project position
```

---

## 5. Compiler Pipeline — Detailed Specification

### 5.1 Input Formats

| Format | Path | Notes |
|--------|------|-------|
| CUDA C++ (`.cu`) | Clang CUDA frontend | Requires `--cuda-gpu-arch` flag to emit device IR |
| PTX text (`.ptx`) | Custom PTX parser | Supports PTX ISA 6.0 – 8.0+ with graceful degradation |
| Fatbinary (`.cubin`) | PTX extractor | Extracts embedded PTX; SASS is not supported |

SASS (native NVIDIA machine code) is explicitly unsupported. Applications that ship only
SASS with no PTX fallback cannot be translated.

#### 5.1.1 PTX ISA Feature Coverage

The parser accepts all PTX ISA versions and degrades gracefully per-instruction rather
than rejecting entire files at a version boundary. This is critical because many CUDA
toolkit outputs embed a high ISA version header while using only basic instructions.

| Feature Class | PTX Instructions | Status |
|---------------|------------------|--------|
| Core compute (Pascal/Volta) | `ld`, `st`, `add`, `mul`, `fma`, `setp`, `bra`, `bar.sync`, `shfl.sync` | Fully supported |
| Ampere atomics | `atom.global.add.f64`, `redux.sync` | Supported (atomics via LLVM IR) |
| Async copy | `cp.async`, `cp.async.commit_group`, `cp.async.wait_group` | `cp.async` → plain load + threadgroup barrier; functional but not performance-equivalent |
| Cluster ops (Hopper) | `cluster.*`, `mbarrier.*` (distributed shared memory) | Unsupported — per-instruction compile-time error with diagnostic |
| TMA (Tensor Memory Accelerator) | `cp.async.bulk.tensor.*` | Unsupported — per-instruction compile-time error |
| FP8 / Transformer Engine | `cvt.rn.f8x2.*` | Unsupported — per-instruction compile-time error |

**Parser tolerance:** The PTX parser (synced quarterly from ZLUDA's latest `ptx` crate,
which as of 2025 handles out-of-order modifiers, malformed inline asm, and relaxed syntax)
accepts any syntactically valid PTX and only errors on individual instructions that cannot
be lowered to AIR. The `--ptx-strict` flag restores hard errors on any unrecognized instruction.

### 5.2 LLVM IR Normalization

After parsing, the IR is normalized to a canonical form before lowering:

- All `addrspace(3)` (shared memory) allocations verified and tagged.
- `llvm.nvvm.*` intrinsics inventoried for lowering coverage check. Unknown intrinsics
  emit a compile-time warning and a trap instruction at runtime.
- NVPTX calling conventions rewritten to `amdgpu_kernel`-style (no-return, pointer args
  in flat address space) as an intermediate form before AIR finalization.
- **Opaque pointer canonicalization**: LLVM 18 uses opaque pointers by default. All IR is
  normalized to opaque-pointer form before metadata injection.

### 5.3 Intrinsic Lowering Pass

The core translation. Each CUDA/NVVM intrinsic maps to an AIR intrinsic or sequence:

**Thread indexing**

| CUDA | AIR Intrinsic |
|------|--------------|
| `threadIdx.{x,y,z}` | `llvm.air.thread.position.in.threadgroup.{v3i32,i32}` |
| `blockIdx.{x,y,z}` | `llvm.air.threadgroup.position.in.grid.{v3i32,i32}` |
| `blockDim.{x,y,z}` | `llvm.air.threads.per.threadgroup.{v3i32,i32}` |
| `gridDim.{x,y,z}` | `llvm.air.threadgroups.per.grid.{v3i32,i32}` |
| `warpSize` | Compile-time constant `32` (see §7) |

**Synchronization**

| CUDA | AIR Equivalent |
|------|---------------|
| `__syncthreads()` | `llvm.air.workgroup.barrier(scope=threadgroup, flags=0)` |
| `__syncwarp(mask)` | `llvm.air.simdgroup.barrier` + mask emulation |
| `__threadfence()` | `llvm.air.mem.barrier(scope=device)` |
| `__threadfence_block()` | `llvm.air.mem.barrier(scope=threadgroup)` |

**Warp-level primitives**

Apple Silicon SIMD-group width is architecturally fixed at 32 across all M-series chips
(see §7). No width indirection is needed.

| CUDA | AIR Equivalent |
|------|---------------|
| `__shfl_sync(mask, val, src)` | `llvm.air.simdgroup.shuffle(val, src)` |
| `__shfl_down_sync(mask, val, delta)` | `llvm.air.simdgroup.shuffle_down(val, delta)` |
| `__shfl_up_sync(mask, val, delta)` | `llvm.air.simdgroup.shuffle_up(val, delta)` |
| `__shfl_xor_sync(mask, val, lane_mask)` | `llvm.air.simdgroup.shuffle_xor(val, lane_mask)` |
| `__ballot_sync(mask, predicate)` | `llvm.air.simdgroup.ballot(predicate)` |
| `__any_sync(mask, predicate)` | `llvm.air.simdgroup.any(predicate)` |
| `__all_sync(mask, predicate)` | `llvm.air.simdgroup.all(predicate)` |

**Mask semantics**: All `_sync(mask, ...)` CUDA functions accept a lane participation mask.
CuMetal intersects AIR's active-lane ballot with that member mask for ballot/any/all and uses
an AIR ballot of `true` for `activemask`. Shuffle callers outside the member mask receive
their own value; CUDA specifies their result as undefined. A shuffle from an inactive source
lane is likewise undefined on both APIs. Masked `__syncwarp` remains conservative because
AIR does not consume CUDA's explicit member mask, so additional currently active lanes may
participate and receive stronger ordering than CUDA requires. Divergent lower/upper half-warp
barriers are GPU-tested, including shared-memory visibility. Broader application paths still
need independent conformance; PhysX's warp-cooperative `preIntegration` and
`updateBodiesLaunch` paths pass twenty consecutive 30-step conformance runs.

**Atomics**

PTX atomic operations map to LLVM atomic IR instructions, which the AIR backend already
handles. No special intrinsics needed. Address space must be correct (AS 1 = device, AS 3
= threadgroup).

**Math intrinsics**

`llvm.nvvm.sqrt.rn.f32` → `llvm.sqrt.f32`. Fast-math flags are preserved. Transcendentals
(`sin`, `cos`, `exp`, `log`) map to LLVM builtins; precision may differ from NVIDIA's
hardware by up to 1 ULP — documented in `known-gaps.md`.

**Device printf**

CUDA `printf` from device code is supported via a buffer-based mechanism:

1. The compiler allocates a ring buffer (`MTLBuffer`, 1 MB default, configurable via
   `CUMETAL_PRINTF_BUFFER_SIZE`) as a hidden kernel argument.
2. Each `printf` call in device code is lowered to: atomically increment write pointer,
   write format-string ID + arguments to the buffer.
3. After kernel completion, the runtime drains the buffer on the CPU and emits to stderr.
4. Format strings are deduplicated and stored in a side table compiled into the `.metallib`.

Limitations: output may be reordered relative to CUDA (which also does not guarantee ordering).
Maximum format string length: 256 bytes. If the buffer overflows, excess prints are silently
dropped (matching CUDA's behavior).

### 5.4 Address Space Map

| CUDA AS | Semantic | AIR AS | Notes |
|---------|----------|--------|-------|
| 0 | Generic | 0 | Flat pointer; resolved at runtime |
| 1 | Global (device) | 1 | `MTLBuffer` memory |
| 3 | Shared (threadgroup) | 3 | `setThreadgroupMemoryLength` |
| 4 | Constant | 2 | See §5.4.1 |
| 5 | Local (thread-private) | 0 (stack) | Lowered to stack allocations |

#### 5.4.1 Constant Memory

CUDA `__constant__` variables occupy a 64 KB per-module constant cache on NVIDIA hardware.
On Metal, constant address space maps to AIR AS 2 (constant buffers).

Implementation:

- `__constant__` declarations are collected into a single constant buffer per module.
- `cudaMemcpyToSymbol(symbol, src, size)` writes to the constant buffer's CPU-side shadow,
  which is flushed to the `MTLBuffer` before the next kernel launch that uses the module.
- The constant buffer is bound at a reserved buffer index (`kConstantBufferIndex = 30`).
- Maximum constant buffer size: 64 KB (matching CUDA). Exceeding this emits a compile error.

### 5.5 AIR Kernel Metadata Injection

Every kernel function must carry specific LLVM metadata for the Metal driver to accept it.
This is the most fragile part of the pipeline and requires ongoing maintenance as Apple
silently updates the AIR ABI across Xcode releases.

Required metadata per kernel (LLVM 18 opaque pointer form):

```llvm
define void @my_kernel(ptr addrspace(1) %buf, ptr addrspace(1) %printf_buf) #0 {
  ...
}

attributes #0 = { "air.kernel" "air.version"="2.6" }

!air.kernel = !{!0}
!0 = !{ptr @my_kernel, !1, !2, !3}
!1 = !{!"air.arg_type_size", i32 8}
!2 = !{!"air.arg_type_size", i32 8}
!3 = !{!"air.arg_name", !"buf"}
!air.compile_options = !{!4}
!4 = !{!"air.max_total_threads_per_threadgroup", i32 1024}
!air.language_version = !{!"Metal", i32 3, i32 1, i32 0}
```

#### 5.5.1 AIR ABI Stability and Community Tooling

The `air_validate` component integrates existing open-source reverse-engineering tools
to validate generated `.metallib` files before they reach the Metal driver:

| Tool | Purpose | Integration |
|------|---------|-------------|
| [MetalLibraryArchive](https://github.com/YuAo/MetalLibraryArchive) (YuAo) | Parse `.metallib` container, extract function metadata | Build dependency; validates container structure |
| [MetalShaderTools](https://github.com/zhuowei/MetalShaderTools) (zhuowei) | Disassemble AIR bitcode, inspect metadata | Optional dev tool; used in `air_inspect` |
| [MetallibSupportPkg](https://github.com/dortania/MetallibSupportPkg) (Dortania) | Cross-version metallib compatibility | Reference for version field requirements |

The `air_emitter` output is piped through `air_validate` before writing to disk. Validation
failures are compile-time errors, not silent Metal driver rejections. This turns the "most
fragile part" into a build-time-caught failure with a clear diagnostic.

**ABI stability testing**: The test suite includes reference kernels compiled with each
supported Xcode version (15.0, 15.4, 16.0, 16.2). The `tests/air_abi/` directory verifies
CuMetal-generated `.metallib` files are accepted by each Metal driver version. Breaking
changes are detected via CI and documented in `docs/air-abi.md`.

### 5.6 Compilation Cache

Compiled `.metallib` blobs are cached in `$HOME/Library/Caches/io.cumetal/kernels/` keyed
by `SHA256(ptx_source + gpu_family + compiler_version)`. Cache entries are invalidated when
the CuMetal compiler version changes. The cache is safe to delete at any time.

### 5.7 Performance Reality Check

Translation imposes overhead. Here is an honest accounting:

| Factor | Impact | Notes |
|--------|--------|-------|
| UMA memcpy (H↔D, D↔D) | **Huge win** | `memcpy` vs. PCIe DMA. `cudaMemcpy` is essentially free. |
| Metal command buffer submission | 1.2–1.5× overhead vs. hand-Metal | Per-dispatch overhead from encoder create/commit. Amortized for large kernels. |
| Intrinsic translation overhead | Negligible | 1:1 AIR intrinsic mapping for most operations |
| Warp primitive handling (partial masks) | Small predicate/ballot overhead | Vote masks are intersected explicitly; barriers use active SIMD-group scope |
| FP64 emulation (if `--fp64=emulate`) | ~4× vs. native FP32 per op | Only for double-precision code on emulation path |

**Targets:**

- Phase 4 (correctness): No performance gate. Any correct result is acceptable.
- Phase 5 (performance): Within **2× of hand-written Metal** for memory-bound kernels.
  This is a hard gate — releases must pass `cumetal_bench` against native Metal baselines.

The `tools/cumetal_bench/` runner executes every functional test kernel and prints a
side-by-side comparison against the equivalent native Metal implementation, reporting
wall-clock time, GPU time (from `MTLCommandBuffer` timestamps), and the ratio.

---

## 6. Runtime Shim — Detailed Specification

### 6.1 Initialization

`cuInit(0)` or any first CUDA Runtime call triggers:

1. `MTLCreateSystemDefaultDevice()` — obtain the default Metal device.
2. If device family < `MTLGPUFamilyApple7` (M1 baseline), emit a warning but continue.
3. Create a default `MTLCommandQueue` for the null stream.
4. Query and cache `threadExecutionWidth` from a dummy pipeline state. Assert == 32; if not,
   emit a hard error (see §7).
5. Initialize the compilation cache subsystem.
6. Initialize the per-thread error queue.
7. Register a `dyld` teardown handler for cleanup.
8. If another `libcuda.dylib` (NVIDIA's or another shim) is detected in the library path,
   emit a prominent warning to stderr.

### 6.2 Memory Management

Apple Silicon has a unified memory architecture. CPU and GPU share physical DRAM. This
eliminates the need for DMA transfers in the classical sense.

| CUDA API | Implementation | Notes |
|----------|---------------|-------|
| `cudaMalloc(ptr, size)` | `[device newBufferWithLength:size options:MTLStorageModeShared]`; return `buffer.contents` | Pointer returned is CPU-accessible |
| `cudaMallocManaged(ptr, size, flags)` | Same as `cudaMalloc` — all memory is already managed | UMA makes this a no-op distinction |
| `cudaMallocHost(ptr, size)` | `[device newBufferWithLength:size options:MTLStorageModeShared]`; return `buffer.contents` | Must be `MTLBuffer` — pointer needs to be trackable for kernel argument binding |
| `cudaFree(ptr)` | Look up `MTLBuffer` by `contents` pointer in allocation table; release reference | ARC handles dealloc after release |
| `cudaMemcpy(dst, src, size, kind)` | `memcpy(dst, src, size)` for all directions | UMA: H↔D, D↔H, D↔D all share physical memory |
| `cudaMemcpyAsync(dst, src, size, kind, stream)` | Enqueue `memcpy` as a CPU-side operation in the stream's work queue (see §6.2.1) | Maintains stream ordering |
| `cudaMemset(ptr, value, size)` | `memset(ptr, value, size)` for synchronous; blit encoder for async | |
| `cudaMemcpyToSymbol(sym, src, size)` | Write to constant buffer shadow; mark dirty (see §5.4.1) | |

#### 6.2.1 Allocation Tracking

The runtime maintains a `ptr → MTLBuffer` lookup table (concurrent hash map) for all
allocations made via `cudaMalloc*`. This is necessary because:

1. **Kernel argument binding**: When `cudaLaunchKernel` receives a `void**` argument that is
   a device pointer, the runtime must find the backing `MTLBuffer` and its offset to call
   `setBuffer:offset:atIndex:`.
2. **Free**: `cudaFree` receives a raw pointer and must find the `MTLBuffer` to release.
3. **Memcpy direction inference**: `cudaMemcpy` with `cudaMemcpyDefault` must determine if a
   pointer is host or device.

Implementation: a lock-free hash map keyed by `uintptr_t` base address, storing
`{MTLBuffer*, offset, size}`. Sub-buffer addressing (pointer arithmetic into a `cudaMalloc`
region) is resolved by range lookup.

**Future optimization**: For programs that allocate hundreds or thousands of buffers, switch
to `MTLHeap`-backed sub-allocation. This reduces Metal object overhead and enables bulk
deallocation. Deferred to Phase 5.

### 6.3 Stream Model

| CUDA Concept | Metal Equivalent | Notes |
|-------------|-----------------|-------|
| `cudaStream_t` (null stream) | Default `MTLCommandQueue` | Legacy default stream has implicit sync (see §6.3.1) |
| `cudaStream_t` (user stream) | Dedicated `MTLCommandQueue` | Independent execution |
| `cudaStreamSynchronize(s)` | `[commandBuffer waitUntilCompleted]` on the last committed buffer | |
| `cudaDeviceSynchronize()` | Flush and wait on all active command queues | |
| `cudaStreamWaitEvent(s, e)` | Encode `waitForEvent:value:` on stream's next command buffer | |
| `cudaStreamAddCallback(s, cb, data, flags)` | CPU dispatch queue callback on command buffer completion (§6.3.2) | |

Metal command buffers within the same `MTLCommandQueue` execute in submission order. This
matches CUDA's guarantee that operations within a stream are ordered. Different command
queues are independent, matching CUDA's guarantee that different streams are independent.

#### 6.3.1 Null Stream Semantics (Legacy Default Stream)

CUDA's legacy default stream has implicit synchronization behavior: a kernel launch on the
null stream waits for all prior work on all streams in the same context, and all streams
wait for the null stream to complete before starting new work.

CuMetal implements this by:

1. Before encoding work on the null stream: signal a shared `MTLSharedEvent` after all
   pending work on all user streams completes.
2. The null stream's command buffer waits on this event before executing.
3. After null stream work completes: all user streams wait on the null stream's completion
   event before executing new work.

**Per-thread default stream** (`cudaStreamPerThread`): allocated lazily per thread; behaves
as a regular (non-null) stream. This is the recommended mode and avoids the null stream
synchronization overhead.

#### 6.3.2 Stream Callbacks

`cudaStreamAddCallback(stream, callback, userData, flags)` is implemented via Metal's
command buffer completion handler:

```objc
[commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buf) {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        callback(stream, cudaSuccess, userData);
    });
}];
```

The callback is dispatched to a GCD queue to avoid blocking the Metal completion thread.

### 6.4 Event Model

| CUDA API | Metal Equivalent |
|----------|-----------------|
| `cudaEventCreate(e)` | Allocate `MTLSharedEvent` + monotonic counter |
| `cudaEventCreateWithFlags(e, flags)` | `cudaEventDisableTiming`: skip timestamp recording |
| `cudaEventRecord(e, stream)` | Encode `signalEvent:value:` at current point in command buffer |
| `cudaEventSynchronize(e)` | Spin-wait on `MTLSharedEvent.signaledValue` ≥ recorded value |
| `cudaEventElapsedTime(ms, start, end)` | Difference of `MTLCommandBuffer.GPUStartTime` / `GPUEndTime` |
| `cudaEventQuery(e)` | Non-blocking check of `signaledValue` ≥ recorded value |

Each event maintains a monotonic counter. `cudaEventRecord` increments the counter and
encodes a signal for the new value. This allows the same event to be recorded multiple
times without race conditions.

### 6.5 Kernel Launch

`cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream)` translates to:

1. Look up compiled `.metallib` for `func` in the kernel registry (loaded at module init).
2. Create or cache `MTLComputePipelineState` for the function.
3. If the module's constant buffer is dirty, flush it to the `MTLBuffer`.
4. Open `MTLComputeCommandEncoder` on the stream's command buffer.
5. **Bind arguments** — for each kernel parameter:
   - **Pointer arguments**: Look up the `MTLBuffer` in the allocation table. Call
     `setBuffer:offset:atIndex:` with the buffer and offset from base.
   - **Scalar/struct arguments (≤ 4 KB)**: Call `setBytes:length:atIndex:` to pass by value.
   - **Printf buffer**: Bind the printf ring buffer at the last argument index.
   - **Constant buffer**: Bind at reserved index `kConstantBufferIndex`.
6. Set threadgroup memory: `setThreadgroupMemoryLength:atIndex:` for `__shared__` allocations.
   Registered PTX records the computed static size; source-first metallibs carry it in the
   versioned `.cumetal-abi` sidecar. A nonzero `sharedMem` launch parameter supplies the
   dynamic allocation instead.
7. `dispatchThreadgroups: MTLSizeMake(gridDim.x, gridDim.y, gridDim.z)`
   `threadsPerThreadgroup: MTLSizeMake(blockDim.x, blockDim.y, blockDim.z)`.
8. End encoding.
9. If null stream: commit and wait (synchronous). Otherwise: commit (async).
10. After completion (or async via handler): drain printf buffer if non-empty.

### 6.6 Module Loading and Fatbinary Registration

**Primary path (cumetalc — AOT)**: `cumetalc` compiles `.cu` to a host executable that
statically links the compiled `.metallib` data. At program start, a constructor registers
the embedded `.metallib` with the runtime's kernel table. No JIT needed.

**Driver API path**: `cuModuleLoadData(module, image)` receives a fatbinary pointer. The
fatbinary is parsed to extract embedded PTX. The PTX is compiled via the CuMetal compiler
pipeline and the resulting `.metallib` is registered in the module's kernel function table.

**Runtime API path (binary shim — opt-in only)**: CUDA Runtime API programs
do not call `cuModuleLoad` explicitly. Instead, `nvcc` emits static initializers that call
hidden registration functions:

1. `__cudaRegisterFatBinary(fatbin)` — called from `__cuda_module_ctor` at program start.
2. `__cudaRegisterFunction(fatbin_handle, host_func, device_name, ...)` — maps host-side
   function pointers to device kernel names.
3. `__cudaRegisterVar(fatbin_handle, host_var, device_name, ...)` — registers `__device__`
   and `__constant__` variables.

CuMetal's opt-in `libcuda.dylib` shim exports these symbols. The shim is built only when
`CUMETAL_ENABLE_BINARY_SHIM=ON` (default: `OFF` in release builds, `ON` in dev builds).

`cuModuleGetFunction(func, module, name)` looks up the function by name in the registered
`.metallib` and caches the `MTLFunction` reference.

### 6.7 Error Model

CUDA has a per-thread asynchronous error reporting mechanism. CuMetal replicates this:

- **Per-thread last error**: Each thread maintains a `cudaError_t` in thread-local storage.
  Any runtime call that fails sets this value.
- `cudaGetLastError()`: Returns and clears the per-thread error.
- `cudaPeekAtLastError()`: Returns without clearing.
- **Async GPU errors**: Metal command buffer completion status is checked via the completion
  handler. If `commandBuffer.status == MTLCommandBufferStatusError`, the error code is
  translated to the nearest CUDA equivalent and stored for the next `cudaGetLastError` call
  on the submitting thread.

Error translation table:

| Metal Error | CUDA Error |
|-------------|------------|
| `MTLCommandBufferErrorTimeout` | `cudaErrorLaunchTimeout` |
| `MTLCommandBufferErrorPageFault` | `cudaErrorIllegalAddress` |
| `MTLCommandBufferErrorBlacklisted` | `cudaErrorDevicesUnavailable` |
| `MTLCommandBufferErrorInternal` | `cudaErrorUnknown` |

### 6.8 Device Property Queries

`cudaGetDeviceProperties` and the Driver API equivalent return a `cudaDeviceProp` struct
populated from Metal device queries:

| CUDA Property | Source |
|---------------|--------|
| `name` | `device.name` (e.g., "Apple M2 Pro") |
| `totalGlobalMem` | `device.recommendedMaxWorkingSetSize` |
| `sharedMemPerBlock` | `device.maxThreadgroupMemoryLength` |
| `warpSize` | `32` (architecturally fixed, see §7) |
| `maxThreadsPerBlock` | `device.maxThreadsPerThreadgroup` |
| `maxBufferArguments` | `31` (Metal limit; relevant for kernels with many parameters) |
| `multiProcessorCount` | Per-chip lookup from public specs (see table below) |
| `computeCapability` | `{8, 0}` (synthetic — indicates Ampere-equivalent feature set) |
| `unifiedAddressing` | `true` (always, on Apple Silicon) |
| `managedMemory` | `true` |
| `concurrentManagedAccess` | `true` |

**GPU core counts** (from Apple published specs, updated as new chips release):

| Chip | GPU Cores | Reported `multiProcessorCount` |
|------|-----------|-------------------------------|
| M1 | 7–8 | 7–8 |
| M1 Pro | 14–16 | 14–16 |
| M1 Max | 24–32 | 24–32 |
| M1 Ultra | 48–64 | 48–64 |
| M2 | 8–10 | 8–10 |
| M2 Pro | 16–19 | 16–19 |
| M2 Max | 30–38 | 30–38 |
| M2 Ultra | 60–76 | 60–76 |
| M3 | 8–10 | 8–10 |
| M3 Pro | 11–18 | 11–18 |
| M3 Max | 30–40 | 30–40 |
| M3 Ultra | 60 | 60 |
| M4 | 10 | 10 |
| M4 Pro | 16–20 | 16–20 |
| M4 Max | 32–40 | 32–40 |
| M4 Ultra | 60 | 60 |

The runtime detects the specific chip variant via `device.name` string matching and returns
the correct core count. Unknown chips return a conservative estimate of 8.

Properties that have no meaningful Metal equivalent return `0` or a conservative estimate
and are documented in `known-gaps.md`.

### 6.9 Installation and Conflict Avoidance

#### One-command install

```bash
curl -sSL https://cumetal.dev/install.sh | bash
# Installs to /opt/cumetal/{bin,lib,include}
# Adds /opt/cumetal/bin to PATH in ~/.zshrc
# Sets DYLD_FALLBACK_LIBRARY_PATH (not DYLD_LIBRARY_PATH — see below)
```

#### Library path strategy

- **AOT path (cumetalc)**: Programs statically link to `libcumetal.a` or dynamically link
  via `@rpath/libcumetal.dylib`. No global library path changes needed.
- **Binary shim path**: `libcuda.dylib` is installed to `/opt/cumetal/lib/`. Users who
  opt in must set `DYLD_FALLBACK_LIBRARY_PATH=/opt/cumetal/lib:$DYLD_FALLBACK_LIBRARY_PATH`.
  We use `FALLBACK` (not `LIBRARY_PATH`) to avoid overriding system libraries.
- **Conflict detection**: At initialization, the runtime checks `_dyld_get_image_name` for
  any other `libcuda.dylib` in the loaded image list. If found, a prominent warning is
  emitted to stderr identifying both libraries by path.
- **Per-app override**: For apps that hard-link `@rpath/libcuda.dylib`, use
  `install_name_tool -change @rpath/libcuda.dylib /opt/cumetal/lib/libcuda.dylib <binary>`.

#### Uninstall

```bash
/opt/cumetal/uninstall.sh   # Removes all CuMetal files, restores shell config
```

---

## 7. Warp Size

Apple Silicon SIMD-group width is **architecturally fixed at 32** across all M-series chips
(M1, M2, M3, M4 and their Pro/Max/Ultra variants).

Sources:

- Apple WWDC 2022 "Scale compute workloads across Apple GPUs": "The SIMD-group size on
  Apple GPUs is 32."
- `MTLComputePipelineState.threadExecutionWidth` returns 32 on all tested M-series hardware
  for all compute pipeline configurations.
- DougallJ's reverse-engineered Apple GPU ISA documentation confirms 32-wide SIMD across
  the AGX architecture family.

**CuMetal's policy:**

- `warpSize` is replaced with the compile-time constant `32`.
- At pipeline creation time, the runtime asserts `threadExecutionWidth == 32`. If this ever
  fails on a future Apple chip, it is a hard error — the runtime refuses to dispatch the
  kernel and prints a diagnostic directing the user to file a CuMetal issue.
- Ballot results are `uint32_t` (matching CUDA's 32-bit `__ballot_sync` return type).
- No runtime width indirection, no emulation modes, no performance penalty.

**Diagnostic annotations** (compile-time, advisory only):

- **Warning**: Integer literal `32` used as a divisor/modulus/shift on thread indices.
  Heuristic — detects patterns like `lane = threadIdx.x % 32` that assume a specific warp
  size. These are correct on CuMetal but may indicate non-portable code.

---

## 8. Known Semantic Gaps

The following CUDA features have no direct Metal equivalent and require workarounds or are
unsupported in v1:

| Feature | Gap | v1 Policy |
|---------|-----|-----------|
| Dynamic parallelism | Metal kernels cannot launch kernels | Compile-time error |
| Cooperative groups (grid-wide) | No cross-threadgroup barrier in Metal | Partial: threadgroup-scoped CG works |
| `__ldg()` (texture cache load) | No texture cache hint in AIR | Lowered to plain load; no perf impact on UMA |
| CUDA graphs | Pre-recorded command buffer equivalent exists | Deferred to v2 |
| Peer-to-peer memory | No multi-GPU on Apple Silicon | Compile-time error |
| Occupancy API | Apple GPU architecture differs | Returns conservative estimates (see §6.8) |
| Texture/surface objects | Metal textures exist but binding model differs | Deferred to v2; compile-time error for now |
| Half-precision atomics | Not universally supported in AIR | Software emulation via CAS loop |
| `cudaProfilerStart/Stop` | Metal GPU capture API exists but differs | Stub (no-op) |
| FP64 arithmetic | See §8.1 | Conditional support with fallback |
| `__constant__` memory > 64 KB | Metal constant buffer limit | Compile-time error |
| CUDA printf (> 256 byte format) | Buffer-based emulation limit | Truncation |

### 8.1 FP64 Policy

Apple Silicon GPUs have minimal FP64 support. The GPU ALUs can execute some FP64
instructions but at drastically reduced throughput (~1/32 of FP32). This is consistent
across M1–M4.

CuMetal's FP64 compilation modes:

| Mode | Flag | Behavior | Precision |
|------|------|----------|-----------|
| Native (default) | `--fp64=native` | Emit AIR FP64 instructions. Works but extremely slow. | IEEE 754 double |
| Emulate | `--fp64=emulate` | Decompose to FP32 pairs via Dekker's algorithm | ~44 bits mantissa |
| Warn | `--fp64=warn` | Same as native, but emit a per-instruction warning | IEEE 754 double |

**Guidance for users:**

- Programs that use occasional `double` for accumulation or reduction: use `--fp64=native`.
  The throughput penalty is per-instruction; if doubles are <5% of operations, the overall
  impact is tolerable.
- Programs dominated by FP64 (scientific simulation with `double` throughout): these are
  out of scope for GPU execution on Apple Silicon. The recommended path is CPU execution
  via Apple's AMX (Accelerate Matrix eXtensions) coprocessor, which provides full-speed
  FP64 SIMD. A Metal↔AMX bridge is out of scope for CuMetal but trivial to build on top.
- `--fp64=emulate` exists for the narrow case where ~44-bit precision is acceptable and
  FP64 throughput matters more than full IEEE precision.

---

## 9. Build System and Dependencies

### 9.1 Required Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| LLVM / Clang | 18+ | Compiler infrastructure, CUDA frontend |
| Rust | 1.78+ | PTX parser (forked from ZLUDA `ptx` crate) |
| CMake | 3.28+ | Build system |
| Xcode Command Line Tools | 15+ | Metal framework headers, `xcrun` for device queries |
| macOS SDK | 14.0+ | `Metal.framework`, `MTLDevice` APIs |

### 9.2 Optional Dependencies

| Dependency | Purpose |
|-----------|---------|
| [MetalLibraryArchive](https://github.com/YuAo/MetalLibraryArchive) (Swift pkg) | `.metallib` validation in `air_validate` |

### 9.3 Runtime Dependencies (end user)

- macOS 14.0+ (Sonoma) on Apple Silicon
- No NVIDIA drivers, CUDA toolkit, or Xcode installation required at runtime
- The Metal framework is part of macOS; no additional installation needed.

### 9.4 Build Targets

```
cumetalc              # Standalone compiler CLI
libcumetal.dylib      # Runtime + JIT compiler (primary dynamic library)
libcumetal.a          # Static link option (recommended for AOT path)
libcuda.dylib         # Binary compatibility shim (opt-in: CUMETAL_ENABLE_BINARY_SHIM=ON)
cumetal_tests         # Test runner
cumetal_bench         # Benchmark runner
```

---

## 10. Testing Strategy

### 10.1 Unit Tests (per LLVM pass)

Each lowering pass has a test suite using LLVM `FileCheck`. Input: LLVM IR with CUDA
intrinsics. Expected output: LLVM IR with AIR intrinsics. Tests run on any platform
(no Metal device required).

Example:
```
; RUN: opt -load-pass-plugin cumetal_passes.dylib -passes=intrinsic-lower %s | FileCheck %s
; CHECK: call i32 @llvm.air.thread.position.in.threadgroup
```

### 10.2 PTX Instruction Sweep

Per-instruction bit-accuracy tests derived from ZLUDA's PTX test harness:

- For each supported PTX instruction, generate a minimal kernel that exercises it with
  known inputs and expected outputs.
- Run on both CuMetal (Apple Silicon) and reference CUDA (NVIDIA GPU, if available).
- Compare outputs bit-for-bit for integer/FP32; within documented ULP bounds for
  transcendentals.
- Imported from ZLUDA quarterly; new instructions added as parser coverage expands.

### 10.3 Functional Tests

End-to-end tests run on Apple Silicon. Each test is a minimal CUDA program with a known
correct output. Test categories:

- **Memory**: malloc, memcpy, memset, unified memory access, `cudaMallocHost` GPU access,
  `cudaMemcpyToSymbol` for constant memory
- **Arithmetic**: FP32, FP64 (where supported), FP16 basic operations and precision checks
- **Indexing**: 1D / 2D / 3D grid and block configurations
- **Shared memory**: tiled matrix multiply, reduction patterns
- **Synchronization**: `__syncthreads`, barrier ordering guarantees
- **Warp primitives**: shuffle, ballot, vote operations (with partial masks)
- **Atomics**: global and shared memory atomics under contention
- **Printf**: format string output correctness and buffer overflow behavior
- **Error reporting**: verify `cudaGetLastError` returns correct codes after failures
- **Registration**: verify `__cudaRegisterFatBinary` path (binary shim build only)
- **Scalar arguments**: kernels taking `int`, `float`, struct arguments by value

### 10.4 Conformance Suite

A subset of NVIDIA's open CUDA samples adapted for conformance testing. The goal is
bit-accurate output for FP32 (matching CUDA within documented precision bounds) on a
defined set of reference kernels.

### 10.5 AIR ABI Regression Tests

For each supported Xcode version (15.0, 15.4, 16.0, 16.2+):

1. Compile a set of reference kernels to `.metallib` with CuMetal.
2. Validate with `air_validate` (MetalLibraryArchive parser).
3. Load the `.metallib` on a device running the corresponding macOS/Xcode version.
4. Verify the Metal driver accepts the kernel and produces correct output.

This catches silent AIR ABI changes across Xcode releases.

### 10.6 Benchmark Suite

`cumetal_bench` runs every functional test kernel and reports:

| Metric | Source |
|--------|--------|
| Wall-clock time | `mach_absolute_time` |
| GPU time | `MTLCommandBuffer.GPUStartTime` / `GPUEndTime` delta |
| Native Metal baseline | Hand-written Metal kernel for the same algorithm |
| Ratio | CuMetal GPU time / Native Metal GPU time |

Performance is not gated in CI for Phases 0–4. Starting Phase 5, the 2× ceiling is enforced
for memory-bound kernels in the benchmark suite.

### 10.7 Regression Baseline

Every merged commit runs the functional test suite on an M-series device via GitHub Actions
self-hosted runner. Performance is not gated in CI (Phases 0–4) — only correctness.

Self-hosted runner:

| Runner | Chip | Purpose |
|--------|------|---------|
| `ci-m1` | M1 | Baseline correctness |

---

## 11. Phased Roadmap

### Phase 0.5 — Metallib Validation Harness (2 weeks)

Goal: Prove that CuMetal can generate a `.metallib` that Apple's `metal` command-line tool
and `MTLDevice.newLibraryWithData:` both accept, using only public LLVM and existing
open-source RE tools. No kernel execution yet — just container acceptance.

- Integrate MetalLibraryArchive as a build dependency.
- Hand-write LLVM IR for a trivial kernel (vector add) with correct AIR metadata.
- Serialize to `.metallib` via `air_emitter`.
- Validate: (a) `air_validate` parses without error; (b) `xcrun metal -validate` accepts
  the file; (c) `MTLDevice.newLibraryWithData:` returns a valid `MTLLibrary`.
- **Exit criterion**: A `.metallib` generated entirely by CuMetal passes all three validation
  checks on M1 hardware.

### Phase 0 — Proof of Concept (4–6 weeks)

Goal: Execute a hand-compiled AIR kernel via Metal API; verify output.

- Extend the Phase 0.5 kernel to actually launch via `MTLComputeCommandEncoder`.
- Write `air_inspect` tool to dump `.metallib` structure in detail.
- Document AIR ABI findings in `docs/air-abi.md`.
- Implement `cudaMalloc` / `cudaMemcpy` / `cudaLaunchKernel` manually against the
  hand-written kernel.
- Implement allocation tracking table (§6.2.1).
- **Exit criterion**: vector addition kernel produces correct output on M-series hardware.

### Phase 1 — PTX Compiler (8–12 weeks)

Goal: PTX text → `.metallib` for the vector add and matrix multiply kernels.

- Fork ZLUDA's PTX parser crate (latest 2025 version); adapt to emit LLVM IR for AIR target.
- Implement `intrinsic_lower` pass covering thread indexing, `__syncthreads`, basic math.
- Implement `addrspace` pass.
- Implement `metadata` pass (with opaque pointer form and AIR version attributes).
- Implement `air_emitter` with `air_validate` integration.
- Implement printf lowering for basic format strings.
- Import ZLUDA PTX instruction sweep tests.
- **Exit criterion**: unmodified PTX for vector add and naive matrix multiply produce
  correct output.

### Phase 2 — Runtime Shim (6–8 weeks)

Goal: `libcumetal.dylib` satisfies enough of the CUDA Driver + Runtime API for simple
programs to link and run without modification.

- Implement `cuInit`, `cuDeviceGet`, `cuCtxCreate`, `cuModuleLoad*`, `cuLaunchKernel`.
- Implement `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaDeviceSynchronize`.
- Implement stream and event APIs (including null stream synchronization §6.3.1).
- Implement scalar/struct kernel argument passing via `setBytes`.
- Implement per-thread error queue.
- Compilation cache.
- **Exit criterion**: `samples/vectorAdd` from CUDA samples compiles with `cumetalc` and
  runs linked against `libcumetal` without modification.

### Phase 3 — CUDA C++ Frontend (8–12 weeks)

Goal: `.cu` source files compile and run without pre-compiling to PTX by the user.

- Integrate Clang CUDA device code lowering targeting the AIR backend.
- Implement `cumetalc` CLI: `cumetalc foo.cu -o foo`.
- Support `__global__`, `__device__`, `__host__`, `__shared__`, `__constant__` qualifiers.
- Implement constant memory (`cudaMemcpyToSymbol`) support.
- Implement `cumetal_bench` tool with native Metal baselines.
- **Exit criterion**: `nvcc`-compilable CUDA source files compile with `cumetalc` and
  produce correct output.

### Phase 4 — Correctness Hardening

- Full conformance suite pass rate ≥ 90%.
- FP32 bit-accuracy on all supported operations.
- Stress test: `llm.c` `test_gpt2fp32cu` runs to completion with correct loss values.
- Atomic correctness under high contention.
- Async memcpy and stream ordering.
- AIR ABI regression tests across Xcode 15/16.

### Phase 4.5 — High-Level Library Shims (4–6 weeks)

The CUDA ecosystem's real stickiness is in libraries, not the runtime. With a working
runtime, the following become thin dispatch layers:

| CUDA Library | Metal Backend | Shim Complexity |
|-------------|---------------|-----------------|
| cuBLAS (GEMM, BLAS) | `MPSMatrixMultiplication`, `MPSNDArray` | Low — API mapping |
| cuFFT | `vDSP_fft` (Accelerate) or Metal compute | Medium — plan API differs |
| cuRAND | `<random>` CPU + Metal compute | Low |
| cuDNN (conv, attention) | `MPSGraph` | High — semantic gaps |

Phase 4.5 implements cuBLAS and cuRAND shims (low complexity, high impact). cuDNN is
a separate project.

### Phase 5 — Performance (ongoing)

- Threadgroup memory tiling optimization hints.
- Kernel fusion opportunities (MLIR GPU dialect rewrite — optional path).
- Benchmark against native Metal implementations of the same algorithms.
- Performance parity target: within 2× of hand-written Metal for memory-bound kernels.
  This is a hard release gate enforced by `cumetal_bench`.
- `MTLHeap`-backed sub-allocation for high-allocation-count workloads.
- Binary shim (`libcuda.dylib`) hardening for the opt-in path.

---

## 12. Legal Considerations

### 12.1 Usage Models and Legal Status

> **⚠️ NVIDIA Translation Layer Warning**
>
> As of 2024, NVIDIA's CUDA EULA contains a clause prohibiting the use of translation
> layers to run CUDA code on non-NVIDIA hardware. This clause applies to binary drop-in
> shims (like `libcuda.dylib` intercepting fatbinary loads). It does **not** apply to
> source-level recompilation with a different compiler.
>
> CuMetal's primary path (`cumetalc` source recompilation) is legally unambiguous.
> The binary shim is provided as an opt-in convenience and its use is at the user's
> own risk.

| Usage Model | Supported | Legal Risk | Recommended For |
|-------------|-----------|------------|-----------------|
| Recompile `.cu` with `cumetalc` | ✅ Primary path | **None** — you compiled your own code with a different compiler | All users |
| Link to `libcumetal.dylib` (open-source CUDA programs) | ✅ Supported | **Low** — open-source code, no NVIDIA binary involved | Research, personal projects |
| Drop-in `libcuda.dylib` for closed-source PTX-shipping binaries | ⚠️ Opt-in | **High** — may violate NVIDIA EULA; detection is trivial | Discouraged; use at own risk |
| Drop-in for closed-source SASS-only binaries | ❌ Unsupported | N/A | Cannot work (SASS is not translatable) |

### 12.2 Clean-Room Implementation

- **No NVIDIA headers shipped.** Runtime API headers are clean-room implementations
  matching the public CUDA API specification.
- **No SASS decompilation.** Only PTX (documented virtual ISA) is processed.
- **AIR format documentation** is derived from reverse engineering of publicly distributed
  Apple toolchain outputs (`.metallib` files). This is a standard interoperability reverse
  engineering practice. Key legal basis:
  - **US**: *Sega v. Accolade* (9th Cir. 1992) — reverse engineering for interoperability
    is fair use. *Sony v. Connectix* (9th Cir. 2000) — intermediate copying during reverse
    engineering is permissible when the final product does not contain copied code.
  - **EU**: Directive 2009/24/EC, Article 6 — decompilation for interoperability is permitted
    without authorization when necessary to achieve interoperability.
  - **Note**: CuMetal does not decompile or redistribute any Apple binary. It generates
    `.metallib` files conforming to the reverse-engineered ABI. No Apple code is copied.
- **ZLUDA PTX parser** (Apache 2.0) may be incorporated with attribution. Any modifications
  must be upstreamed or maintained in a fork under the same license.
- **Contributors must sign a CLA** confirming clean-room implementation of any CUDA API
  surface they implement. The CLA also confirms no prior exposure to NVIDIA proprietary
  source code for the implemented API surface.

---

## 13. Open Questions

| # | Question | Impact | Status |
|---|----------|--------|--------|
| 1 | Does AIR ABI change silently between Xcode minor releases? | High | Mitigated by §5.5.1 validation + §10.5 regression tests; needs ongoing monitoring |
| 2 | Can PTX `fma.rn.f64` be translated accurately with LLVM `llvm.fma.f64` on AIR? | Medium | Depends on chip FP64 support; see §8.1 |
| 3 | Is `MTLStorageModeShared` always coherent from CPU after `waitUntilCompleted`? | High | Metal spec says yes; needs edge-case validation |
| 4 | What is the maximum `setThreadgroupMemoryLength` on M-series? | Medium | `maxThreadgroupMemoryLength` query at init; typically 32 KB |
| 5 | Can we use `MTLCommandBuffer` label API for `nvtxRangePush`-style profiling? | Low | Nice to have |
| 6 | How does `device.name` vary across chip binning variants for the GPU core count table? | Low | Need community hardware reports |
| 7 | Should we target macOS 15+ (Sequoia) to simplify AIR ABI if Xcode 16+ is more stable? | Medium | Needs data from §10.5 results |

---

## 14. Contributing

The highest-value contributions, in priority order:

1. **AIR ABI documentation** — compile Metal kernels with different Xcode versions,
   disassemble with MetalShaderTools, document metadata field changes.
2. **PTX parser coverage** — add lowering for PTX instructions not yet handled. Sync with
   upstream ZLUDA parser improvements.
3. **Functional test cases** — any CUDA kernel with a known correct output.
4. **FP64 characterization** — document which FP64 operations are natively supported on each
   chip family and their precision characteristics.
5. **GPU core count validation** — the §6.8 table now covers all M1–M4 variants including
   Ultra tiers. Community hardware reports for unreleased M5+ chips are welcome.
6. **Performance measurement** — baseline benchmarks of translated vs. native Metal kernels.
7. **cuBLAS/cuRAND shim** — thin wrappers over MPS/Accelerate (Phase 4.5).

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| AIR | Apple Intermediate Representation — LLVM bitcode variant consumed by Metal driver |
| `.metallib` | Container format for compiled Metal shader libraries |
| PTX | Parallel Thread Execution — NVIDIA's virtual ISA for CUDA kernels |
| SASS | Shader Assembly — NVIDIA's native GPU machine code (not translatable) |
| SIMD-group | Apple's term for a warp / wavefront (32 threads on Apple Silicon) |
| Threadgroup | Apple's term for a CUDA block |
| UMA | Unified Memory Architecture — CPU and GPU share physical DRAM |
| AMX | Apple Matrix eXtensions — CPU coprocessor with full-speed FP64 SIMD |

---

## 16. Changelog from v0.2

### Strategic Repositioning

| Change | Reason |
|--------|--------|
| Reordered §1 to put compiler gap before runtime gap | Compiler (source recompilation) is now the primary value proposition |
| Added §1.1 "Why Source-First" | Documents the strategic lesson from ZLUDA's failure mode |
| Moved binary shim from default-on to opt-in (`CUMETAL_ENABLE_BINARY_SHIM`) | Reduces legal exposure; the primary path is now `cumetalc` |
| Added §12.1 "Usage Models and Legal Status" with NVIDIA EULA warning | Prevents the project from being taken down the way ZLUDA nearly was |
| Added `docs/legal-notice.md` to repo layout | Ships the legal context with the code |

### Warp Size Simplification

| Change | Reason |
|--------|--------|
| Deleted `simd_width.cpp` pass and three compilation modes from §7 | Apple Silicon SIMD width is architecturally fixed at 32 (WWDC 2022, DougallJ docs, all hardware testing). The three-mode system was solving a problem that doesn't exist, at significant complexity cost. |
| §7 is now 30 lines instead of 80 | Massive simplification; removes the "most semantically dangerous" section |
| Removed `ci-m3` runner from CI matrix | No SIMD width regression to test for |

### AIR ABI Hardening

| Change | Reason |
|--------|--------|
| Added §5.5.1 with MetalLibraryArchive / MetalShaderTools / MetallibSupportPkg integration | Existing community RE tools turn build-time `.metallib` validation from "hope the driver accepts it" to "known-good container format" |
| Added `air_validate` component to repo layout | Build-time validation catches metadata errors before runtime |

### PTX Parser Updates

| Change | Reason |
|--------|--------|
| Raised PTX ISA ceiling from hard-reject at 7.8 to graceful per-instruction degradation | Many real-world PTX files declare high ISA versions but use only basic instructions; rejecting the whole file loses users unnecessarily |
| Added quarterly ZLUDA parser sync policy | ZLUDA's 2024–2025 parser fixes (out-of-order modifiers, relaxed syntax) solve real tolerance problems |
| Added `tests/ptx_sweep/` per-instruction test suite | Systematic bit-accuracy coverage imported from ZLUDA |

### New Sections

| Section | Content |
|---------|---------|
| §1.1 | "Why Source-First" strategic rationale |
| §2.3 | "Why Not MLIR?" FAQ |
| §5.5.1 | AIR ABI community tooling integration |
| §5.7 | Performance Reality Check (honest overhead accounting + `cumetal_bench`) |
| §6.3.2 | Stream callbacks (promoted from "no equivalent" to implemented) |
| §6.9 | Installation and conflict avoidance |
| §10.2 | PTX instruction sweep tests |
| §10.6 | Benchmark suite |
| §11 Phase 0.5 | Metallib validation harness (new first phase) |
| §11 Phase 4.5 | cuBLAS / cuRAND / cuFFT shims via MPS |
| §12.1 | Usage models and legal status table |
| §15 | Glossary |

### Minor Fixes

| Item | Change |
|------|--------|
| §6.2.1 | Added `MTLHeap` note for future large-allocation optimization |
| §6.8 | Replaced "hardcoded per-chip family" with actual per-chip GPU core count table |
| §6.8 | Added `maxBufferArguments` (31) to device properties |
| §6.8 | Added M3 Ultra (60 GPU cores) and M4 Ultra (60 GPU cores) to chip table |
| §9.4 | Split build targets: `libcumetal.dylib` (primary) vs `libcuda.dylib` (opt-in shim) |
| §8.1 | Added AMX recommendation for FP64-dominated workloads |
| §5.3 / §6.5 step 10 | Device printf fully implemented end-to-end: compiler ring-buffer injection, runtime buffer allocation/drain; both `cumetalKernel_t` and `__cudaRegisterFunction` paths covered by functional tests |

---

*End of specification.*
