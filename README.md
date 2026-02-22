CuMetal
=======

CuMetal is an experimental CUDA compiler and runtime for Apple Silicon GPUs.
It translates CUDA source code (`.cu`) and PTX assembly to Metal Shading Language,
and provides a CUDA-compatible runtime API backed by Metal and Apple frameworks.

Requirements
------------

- macOS 14+ (Sonoma)
- Apple M-series GPU
- Xcode command-line tools (`xcrun metal`, `xcrun metallib`)

Quick start
-----------

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

Install to a prefix:

```bash
cmake --install build --prefix /usr/local
```

Or use the provided scripts:

```bash
bash install/install.sh    # installs to /usr/local, sets DYLD_LIBRARY_PATH
bash install/uninstall.sh  # removes installed files
```

Fish shell is detected automatically; `install.sh` writes `set -gx` syntax to
`~/.config/fish/config.fish`. Override with `CUMETAL_SHELL_RC`.

Execution model
---------------

- **Source recompilation** (primary): compile `.cu` or PTX with `cumetalc`, producing
  a Metal-backed `.metallib`. Link the resulting object against `libcumetal.dylib`.
- **Binary shim** (optional): set `CUMETAL_ENABLE_BINARY_SHIM=ON` at build time to
  also emit `libcuda.dylib`. Software that was pre-linked against NVIDIA `libcuda.dylib`
  will load CuMetal without recompilation.

Tools
-----

| Tool | Description |
|------|-------------|
| `cumetalc` | Compiler driver: `.cu` / `.ptx` / `.ll` → `.metallib` |
| `cumetal-air-emitter` | Low-level AIR/metallib container writer |
| `cumetal-ptx2llvm` | PTX text → LLVM IR (AIR-annotated) |
| `air_inspect` | Inspect `.metallib` container (kernels, bitcode offsets, metadata) |
| `air_validate` | Validate `.metallib` structure and optionally xcrun-validate |
| `cumetal_bench` | Phase 5 performance benchmark: CuMetal vs native Metal |

`cumetalc` flags of note:
- `--fp64=native|emulate|warn` — FP64 mode (default: `emulate`; Apple Silicon GPU
  rejects native FP64 in Metal pipelines at runtime)
- `--entry <name>` — select a single PTX entry point
- `--ptx-strict` — treat unsupported PTX opcodes as errors

Library shims
-------------

`libcumetal.dylib` exports:

- Full CUDA Runtime API (see below)
- CUDA Driver API (`cuInit`, `cuLaunchKernel`, modules, streams, events, …)
- cuRAND (host-side random number generation via MT19937/XORWOW)
- cuBLAS v2 (GEMM, GEMV, BLAS 1 — backed by MetalPerformanceShaders and Accelerate)
- cuFFT (1D/2D/3D, any-N batched, backed by Apple Accelerate vDSP)

Build/install also provides dylib aliases so software linked against CUDA library
names can find the shims: `libcublas.dylib`, `libcurand.dylib`, `libcufft.dylib`.
With `CUMETAL_ENABLE_BINARY_SHIM=ON`, `libcuda.dylib` is also provided.

Runtime API coverage
--------------------

Memory:
`cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaMallocPitch`,
`cudaFree`, `cudaFreeHost`, `cudaHostAlloc`, `cudaHostGetDevicePointer`,
`cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemcpy2D`, `cudaMemcpy2DAsync`,
`cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`,
`cudaMemset`, `cudaMemsetAsync`, `cudaMemset2D`,
`cudaMemGetInfo`, `cudaPointerGetAttributes`

Launch:
`cudaLaunchKernel`, `cudaLaunchCooperativeKernel`,
`cudaConfigureCall`, `cudaSetupArgument`, `cudaLaunch`

Streams / events:
`cudaStreamCreate/Destroy/Synchronize/Query/AddCallback/WaitEvent`,
`cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`,
`cudaEventCreate/Destroy/Record/Query/Synchronize/ElapsedTime`,
`cudaDeviceSynchronize`, `cudaDeviceReset`

Device:
`cudaGetDeviceCount`, `cudaGetDevice`, `cudaSetDevice`,
`cudaGetDeviceProperties`, `cudaDeviceGetAttribute`,
`cudaSetDeviceFlags`, `cudaGetDeviceFlags`,
`cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`,
`cudaDeviceGetStreamPriorityRange`,
`cudaDeviceSetLimit`, `cudaDeviceGetLimit`,
`cudaDeviceSetCacheConfig`, `cudaDeviceGetCacheConfig`,
`cudaDeviceSetSharedMemConfig`, `cudaDeviceGetSharedMemConfig`

Misc:
`cudaInit`, `cudaDriverGetVersion`, `cudaRuntimeGetVersion`,
`cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorName`, `cudaGetErrorString`,
`cudaFuncGetAttributes`, `cudaFuncSetCacheConfig`, `cudaFuncSetAttribute`,
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize`,
`cudaGetSymbolAddress`, `cudaGetSymbolSize`, `cudaChooseDevice`,
`cudaMemPrefetchAsync`, `cudaMemAdvise`, `cudaMemRangeGetAttribute`,
`cudaProfilerStart`, `cudaProfilerStop`

Driver API coverage
-------------------

`cuInit`, `cuDriverGetVersion`, `cuDeviceGet*`, `cuCtx*`, `cuStream*`, `cuEvent*`,
`cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleUnload`, `cuModuleGetFunction`,
`cuLaunchKernel`, `cuLaunchCooperativeKernel`, `cuLaunchHostFunc`,
`cuMem*`, `cuGetErrorName`, `cuGetErrorString`,
`cuOccupancy*`, `cuFuncGetAttribute`, `cuFuncSetCacheConfig`,
`cuDeviceComputeCapability`, `cuDeviceCanAccessPeer`,
`cuMemAllocPitch`, `cuCtxGetStreamPriorityRange`

`cuModuleLoadData` accepts `.metallib` bytes, `.metallib` paths, PTX text images,
and basic CUDA fatbin wrapper PTX variants.

Device properties
-----------------

`cudaDeviceProp` is populated as an Ampere-equivalent (compute 8.0) for maximum
compatibility. Key fields: `unifiedAddressing=1`, `managedMemory=1`,
`concurrentManagedAccess=1`, `integrated=1`, `canMapHostMemory=1`,
`maxBufferArguments=31` (Metal limit), `l2CacheSize=4MB`, `sharedMemPerBlock=32KB`.

Headers installed
-----------------

`cuda.h`, `cuda_runtime.h`, `cuda_runtime_api.h`, `cuda_fp16.h`,
`cublas_v2.h`, `cufft.h`, `curand.h`,
`cooperative_groups.h`, `cooperative_groups/reduce.h`

Environment variables
---------------------

| Variable | Default | Description |
|----------|---------|-------------|
| `CUMETAL_MTLHEAP_ALLOC` | (unset) | MTLHeap mode: unset=auto, `1`=always, `0`=disabled |
| `CUMETAL_MTLHEAP_THRESHOLD_BYTES` | `4194304` | Auto-heap threshold (default 4 MiB) |
| `CUMETAL_MTLHEAP_CHUNK_BYTES` | `67108864` | Heap slab size (default 64 MiB) |
| `CUMETAL_CACHE_DIR` | `$HOME/Library/Caches/io.cumetal` | Root for module cache and JIT cache |
| `CUMETAL_DEBUG_REGISTRATION` | `0` | Set to `1` for binary-shim registration trace on stderr |
| `CUMETAL_FP64_MODE` | `emulate` | FP64 mode: `native`, `emulate`, `warn` |
| `CUMETAL_TRACE_LLMC_EMULATION` | `0` | Trace llm.c emulation fallback paths |
| `CUMETAL_DISABLE_LLMC_EMULATION` | `0` | Disable llm.c emulation; require pure PTX lowering |
| `CUMETAL_LLMC_REQUIRE_NO_EMULATION` | `0` | Fail if any llm.c kernel uses emulation fallback |
| `CUMETAL_LLMC_GRAD_TOL` | `1.2e-2` | Gradient tolerance for llm.c conformance test |

MTLHeap auto-threshold
----------------------

`cudaMalloc` automatically uses `MTLHeap` sub-allocation for allocations at or above
the threshold (default 4 MiB). This improves throughput for large allocations by
reducing Metal command encoder overhead. Set `CUMETAL_MTLHEAP_ALLOC=1` to force heap
for all allocations; `CUMETAL_MTLHEAP_ALLOC=0` to disable entirely.

Binary shim JIT cache
---------------------

The binary-shim registration path (`__cudaRegisterFatBinary`) compiles PTX kernels
to `.metallib` at first use and caches the result at
`$CUMETAL_CACHE_DIR/registration-jit/<hash>.metallib`.
The cache key is the FNV-1a-64 hash of `ptx_source + kernel_name`.
Cached files survive process restart and `__cudaUnregisterFatBinary` — the second
process to use the same kernel skips xcrun entirely.

Enable `CUMETAL_DEBUG_REGISTRATION=1` to trace: fatbinary format detection, JIT
compile vs cache hit, arg-count inference, and symbol registration events.

Performance
-----------

Phase 5 benchmark (`cumetal_bench --all-kernels --max-ratio 2.0`) measures
CuMetal wall-clock time against native Metal MSL for three kernels.
Typical results on Apple Silicon:

| Kernel | Elements | Ratio (CuMetal/Metal) |
|--------|----------|-----------------------|
| vector_add | 1M | ~0.74× |
| saxpy | 1M | ~0.98× |
| reduce_f32 | 1M | ~1.00× |

All measured ratios are well within the 2× spec gate (§5.7).

Conformance
-----------

The llm.c GPT-2 FP32 training binary can be built and executed via CuMetal:

```bash
bash scripts/build_llmc_test_gpt2fp32cu.sh
bash scripts/run_llmc_test_gpt2fp32cu.sh
```

Expected output includes `OK (LOGITS)`, `LOSS OK`, `TENSOR OK`, `overall okay: 1`.
All 17 GPT-2 training kernels are lowered directly to Metal MSL; no emulation
fallback is required (`CUMETAL_LLMC_REQUIRE_NO_EMULATION=1` passes).

### llama.cpp Conformance Test

[llama.cpp](https://github.com/ggml-org/llama.cpp) (95k+ stars, used by Ollama,
LM Studio, and every major local-LLM stack) is the most demanding real-world
CUDA workload available. Its GGML CUDA backend runs hundreds of quantized-matrix
kernels entirely unmodified — zero source changes required.

**Build llama.cpp with CuMetal as the CUDA provider:**

```bash
bash scripts/build_llama_cpp_cumetal.sh   # clones + builds in ../llama.cpp/
```

**Run the conformance test** (auto-downloads TinyLlama-1.1B Q4_K_M ~638 MB):

```bash
bash tests/conformance/run_llama_cpp_cumetal.sh
```

Expected output:

```
llama-cli: ../llama.cpp/build-cumetal/bin/llama-cli
Model cached: ~/.cache/cumetal/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

═══════════════════════════════════════════════════════════════
 llama.cpp CUDA backend conformance test via CuMetal
 Model:  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
 Prompt: Explain quantum entanglement in two short sentences.
 NGL:    99  (GPU layers offloaded)
 NTok:   128
═══════════════════════════════════════════════════════════════

Quantum entanglement is a phenomenon where two particles become
correlated such that the state of one instantly influences the
other, regardless of distance. This non-local connection is a
cornerstone of quantum information theory and has no classical
analogue.

─── Inference complete (8s wall-clock) ────────────────────────
Performance: 42.3 tokens per second

PASS: llama.cpp CUDA backend works perfectly on CuMetal
      Real production LLM kernels ran on Apple Silicon via Metal translation.
```

This is significant because it demonstrates that production CUDA LLM kernels —
quantized matrix multiplication, attention, RoPE, and more — execute correctly
through the full CUDA → Metal translation pipeline with no source modifications.
Point any pre-compiled llama.cpp binary at a different model by setting
`CUMETAL_LLAMA_MODEL=/path/to/model.gguf`.

Test suite
----------

152 tests are registered in CTest (unit + functional). An additional benchmark
gate test (`bench_phase5_all_kernels`) runs on Apple Silicon if xcrun is available.

```bash
ctest --test-dir build --output-on-failure      # run all tests
ctest --test-dir build -R functional_ -V        # functional tests only
ctest --test-dir build -R unit_ -V              # unit tests only
```

Known limitations
-----------------

- **CUDA Graphs**: deferred to v2 (spec §2.2)
- **Dynamic parallelism**: compile-time error (spec §2.2)
- **Texture/surface objects**: deferred to v2 (spec §2.2, §8)
- **Multi-GPU**: single GPU on Apple Silicon; peer APIs return appropriate errors
- **Graphics interop** (OpenGL/Vulkan): non-goal (spec §2.2)
- **`grid_group::sync()`**: no-op stub; Metal has no cross-threadgroup barrier
- **Warp partial-mask**: conservative full-group emulation (spec §5.3)
- **FP64**: Apple Silicon GPU has minimal FP64 throughput; `--fp64=emulate` recommended
- **Full NVCC fatbin**: binary shim supports CuMetal CMTL envelopes and PTX images;
  full NVCC ELF-embedded fatbinary variants are not yet implemented
- **Device printf**: buffer-based; format strings limited to 256 bytes

Documentation
-------------

- Implementation status and API coverage: [docs/status.md](./docs/status.md)
- Build and validation workflows: [docs/build.md](./docs/build.md)
- Test and conformance workflows: [docs/testing.md](./docs/testing.md)
- Known feature gaps: [docs/known-gaps.md](./docs/known-gaps.md)
- AIR/metallib ABI notes: [docs/air-abi.md](./docs/air-abi.md)
- Design specification: [spec.md](./spec.md)

License
-------

[Apache 2.0](./LICENSE)
