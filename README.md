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
- cuSPARSE (CSR/COO SpMV, SpMM, legacy `cusparseScsrmv`/`cusparseDcsrmv` — CPU-backed on UMA)
- cuSOLVER Dense (LU, QR, Cholesky, SVD, eigenvalue — backed by Apple Accelerate LAPACK)
- CUDA Graphs (stream capture, instantiate, launch — sequential replay)
- cublasLt (lightweight BLAS matmul with epilogues: bias, relu, gelu)
- cuDNN (convolution fwd/bwd via im2col+GEMM, pooling, activations fwd/bwd, dropout, softmax, batch norm, Nd tensor, tensor ops)
- NVML (device info, memory queries, driver version — Apple Silicon adapted)
- NCCL (single-rank collectives: allreduce, broadcast, reduce, allgather — identity ops on single GPU)
- thrust (device_vector, sort, reduce, scan, transform, fill, sequence, counting_iterator — CPU-backed on UMA)
- Async memory pool API (cudaMallocAsync/cudaFreeAsync — UMA synchronous aliases)
- Texture/Surface objects (array allocation, memcpy, object lifecycle)

Build/install also provides dylib aliases so software linked against CUDA library
names can find the shims: `libcublas.dylib`, `libcublasLt.dylib`, `libcudnn.dylib`,
`libcurand.dylib`, `libcufft.dylib`, `libcusparse.dylib`, `libcusolver.dylib`,
`libnvidia-ml.dylib`, `libnccl.dylib`.
With `CUMETAL_ENABLE_BINARY_SHIM=ON`, `libcuda.dylib` is also provided.


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

For the complete implementation record, GPU proof criteria, diagnostic switches,
and verified results, see
[docs/apple-gpu-execution.md](docs/apple-gpu-execution.md).

### Upstream CUDA source on Apple GPU

The upstream NVIDIA `cuda-samples` vectorAdd source can be compiled without
source modifications and executed through CuMetal:

```bash
CUMETAL_CUDA_SAMPLES_DIR=/path/to/cuda-samples \
  bash tests/cuda_projects/run_cuda_samples_vectoradd_gpu.sh "$PWD"
```

The gate requires the sample's numerical `Test PASSED` result and a provenance
record with `source=generic_ptx`, `device=apple_gpu`, and
`launch_success=true`; it rejects CPU fallbacks and stubs. This proves the
documented simple-kernel subset, not general CUDA or llama.cpp compatibility.

The llm.c GPT-2 FP32 training binary can be built and executed via CuMetal:

```bash
bash scripts/build_llmc_test_gpt2fp32cu.sh
bash scripts/run_llmc_test_gpt2fp32cu.sh
```

The conformance gate requires `OK (LOGITS)`, `LOSS OK`, `TENSOR OK`,
`overall okay: 1`, plus at least one successful Apple-GPU provenance record.
On the tested Apple M4 Pro, the strict gate passes for the llm.c GPT-2 FP32
workload with CPU emulation disabled. Its kernels use CuMetal's
`specialized_msl` path; this is not evidence that arbitrary PTX is supported.
The legacy llm.c CPU implementation is available only when explicitly requested
with `CUMETAL_ENABLE_LLMC_CPU_EMULATION=1`.
Set `CUMETAL_TRACE_GPU=1` to print a machine-readable `CUMETAL_PROVENANCE`
record for each Metal compute command, including whether it came from generic
PTX or a specialized MSL replacement, the execution device, launch status,
dimensions, and GPU duration when available.

### llama.cpp Conformance Test

[llama.cpp](https://github.com/ggml-org/llama.cpp) (95k+ stars, used by Ollama,
LM Studio, and every major local-LLM stack) is the most demanding real-world
CUDA workload available. Its GGML CUDA backend is built **unmodified** against
libcumetal as the CUDA provider — zero source changes required.

**Build llama.cpp with CuMetal as the CUDA provider:**

```bash
bash scripts/build_llama_cpp_cumetal.sh   # clones + builds in ../llama.cpp/
```

**Run the conformance test** (auto-downloads SmolLM2-135M-Instruct Q4_K_M ~105 MB):

```bash
bash tests/conformance/run_llama_cpp_cumetal.sh
```

**Status — verified on 2026-07-20:**

| What | Result |
| --- | --- |
| Build llama.cpp's GGML CUDA backend unmodified against libcumetal | ✅ works |
| Load model, init CUDA device, register fatbins/kernels, run end-to-end | ✅ works |
| SmolLM2-135M greedy output with one GPU-offloaded layer | ✅ coherent |

Measured on SmolLM2-135M, greedy decode of "The capital of France is":

- Stock CPU llama.cpp → `Paris.` ✅
- llama.cpp via libcumetal (NGL=0) → `Paris.` ✅
- llama.cpp via libcumetal (NGL=1) → `The capital of France is Paris.` ✅
  (Apple M4 Pro, 8.4 tokens/s generation in the latest verified run)

Registered fatbinary launches are conservatively synchronized by default because
the experimental asynchronous path can violate ordering when GGML uses adjacent
suballocations of a shared Metal arena. Direct/source-first launches remain
asynchronous. `CUMETAL_ENABLE_ASYNC_REGISTERED_LAUNCH=1` opts into the incomplete
asynchronous path for development. This result is a focused NGL=1 smoke test;
arbitrary models and high offload counts still require broader GGML kernel coverage.

The conformance test enforces a **coherence gate**: a greedy decode must contain the
expected answer (`CUMETAL_LLAMA_EXPECT`, default `Paris`). With `NGL>0` it also
requires a successful Apple-GPU provenance record and rejects CPU fallbacks and
stubs, so it FAILS on garbage or non-GPU execution rather than passing on "some
tokens were generated." The default NGL=1 probe is expected to pass on the
verified Apple M4 Pro path; unsupported models or larger offload counts fail
honestly instead of being reported as compatible.
Point it at another model/answer with `CUMETAL_LLAMA_MODEL` / `CUMETAL_LLAMA_EXPECT`.
The harness forces llama.cpp's `--simple-io` mode so generated tokens are
captured instead of being written only to an interactive terminal. Its output
checker treats the combined token/provenance stream as bytes and reconstructs
token text split by provenance records; focused regressions cover missing or
forbidden provenance and incoherent output.

Test suite
----------

CTest registers unit, functional, conformance, and benchmark tests. Results must
be reported as separate pass/skip/fail counts because external-project and
toolchain-dependent tests legitimately skip; registration is not a passing
compatibility claim. The `bench_phase5_all_kernels` gate runs only when its
required Metal toolchain is available.

```bash
ctest --test-dir build --output-on-failure      # run all tests
ctest --test-dir build -R functional_ -V        # functional tests only
ctest --test-dir build -R unit_ -V              # unit tests only
```

Known limitations
-----------------

- **Dynamic parallelism**: compile-time error (spec §2.2)
- **Multi-GPU**: single GPU on Apple Silicon; peer APIs return appropriate errors
- **Graphics interop** (OpenGL/Vulkan): non-goal (spec §2.2)
- **`grid_group::sync()`**: no-op stub; Metal has no cross-threadgroup barrier
- **Warp partial-mask**: conservative full-group emulation (spec §5.3)
- **FP64**: Apple Silicon GPU has minimal FP64 throughput; `--fp64=emulate` uses
  Dekker double-single decomposition (~44-bit mantissa via FP32 pairs)
- **CUDA Graphs**: stream capture, instantiate, and replay supported; memcpy/memset/kernel
  operations intercepted during capture; node addition APIs available
- **Texture/surface objects**: lifecycle and array memcpy supported; GPU-side texture
  sampling requires Metal shader integration (not yet wired)
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
