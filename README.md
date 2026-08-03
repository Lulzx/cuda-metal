# CuMetal

CUDA programs assume an NVIDIA machine. CuMetal makes a useful subset of them
run on Apple Silicon.

It is a compiler, a runtime, and a set of compatibility libraries. CUDA source
or PTX goes in. Metal runs on the GPU. There is no NVIDIA hardware in the loop.

This is experimental software. The covered paths execute real kernels and check
real answers. Unsupported paths are expected to fail explicitly. That is a
better failure mode than silently computing nonsense.

## Start here

You need:

- macOS 14 or newer
- an Apple M-series GPU
- CMake and the Xcode command-line tools
- Apple's Metal toolchain (`xcrun metal` and `xcrun metallib`)

Build it:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(sysctl -n hw.ncpu)"
bash scripts/ci_report.sh build --exclude-regex '^bench_'
```

### Apollo demo (the front door)

One command that climbs from vector-add through reduction, SGEMM, and a path
tracer, refusing any stage that lacks `device=apple_gpu` provenance:

```bash
bash demos/apollo/run.sh
```

Optional: `bash demos/apollo/run.sh --full` adds the llm.c GPT-2 FP32 gate.
Details, scope limits, and artifacts: [demos/apollo/README.md](demos/apollo/README.md).

### Single sample

```bash
./build/cumetalc samples/vectorAdd/vectorAdd.cu -o /tmp/vectorAdd
CUMETAL_TRACE_GPU=1 /tmp/vectorAdd
```

The program should print a numerical `PASS`. The trace should contain a
`CUMETAL_PROVENANCE` record with `device=apple_gpu` and
`launch_success=true`. A correct number without GPU provenance is not proof of
GPU execution.

Runtime-compiled MSL preserves Metal's fast-math default. Set
`CUMETAL_MSL_MATH_MODE=safe` to request safe Metal math for JIT-compiled source;
GPU provenance reports the selected `math_mode`. Precompiled metallibs retain
the policy used when they were built.

Install it:

```bash
cmake --install build --prefix /opt/cumetal
```

The installer scripts use the same prefix by default:

```bash
bash install/install.sh
bash install/uninstall.sh
```

`install.sh` also adds the prefix to your shell environment.

## The model

CuMetal is source-first.

```text
CUDA C++ ---> Clang / NVVM ---> typed CuMetal IR --+
                                                    |
PTX --------> legacy or typed lowering -------------+--> MSL
                                                          |
                                                          v
                                                Apple Metal toolchain
                                                          |
                                                          v
                                                     .metallib
                                                          |
                                                          v
                                              libcumetal -> Metal GPU
```

There are three entry paths:

1. **Source recompilation.** `cumetalc` compiles a `.cu` file into a runnable
   executable or a `.metallib`. This is the primary path.
2. **PTX compatibility.** CuMetal parses PTX, builds CFG/SSA, and lowers it
   toward the same Metal backend. This path currently covers more project-scale
   CUDA than the typed source path.
3. **Binary compatibility.** An opt-in `libcuda.dylib` alias accepts programs
   already linked against the CUDA Driver API and handles supported fatbinary
   registration. It is useful, narrower than CUDA, and not the architecture.

The SIMD width is 32. This is fixed. CUDA warp semantics are lowered onto
Metal SIMD-group operations; CuMetal does not pretend the machine has a
different warp size because that would make every hard problem harder.

Metal calls stay behind the Objective-C++ boundary in
`runtime/metal_backend/`. CUDA-facing headers are clean-room. No private Apple
API is used.

## Compiler reality

`cumetalc` can emit every useful stage:

```bash
cumetalc kernel.cu --emit=cumetal-ir -o kernel.cumetal
cumetalc kernel.cu --emit=msl        -o kernel.metal
cumetalc kernel.cu --emit=metallib   -o kernel.metallib
cumetalc kernel.cu --emit=exe        -o kernel
```

Important switches:

| Switch | Meaning |
| --- | --- |
| `--backend=cumetal-ir\|legacy` | Select the typed shared-IR backend or the compatibility backend. There is no silent fallback. |
| `--cuda-device` | Ask a CUDA-capable Clang to produce PTX before CuMetal lowering. |
| `--entry NAME` | Compile one kernel and its reachable device-call closure. |
| `--ptx-strict` | Reject unsupported PTX instead of tolerating it. |
| `--fp64=native\|emulate\|warn` | Choose the FP64 policy. Default: `emulate`. |
| `--save-temps` | Keep link intermediates. |

The default backend follows the input because measured coverage says it should:

| Input corpus | `legacy` | `cumetal-ir` |
| --- | ---: | ---: |
| direct `.cu` | 0/19 | **10/19** |
| `.cu --cuda-device` / PTX | **17/19** | 6/19 |

Direct `.cu` therefore defaults to `cumetal-ir`; PTX and `--cuda-device`
default to `legacy`. This is engineering, not ideology. When the measurements
change, the default should change.

The complete compiler boundary is in
[docs/compiler-architecture.md](docs/compiler-architecture.md). Unsupported
instructions, calls, pointer conversions, CFGs, and ABI forms are tracked in
[docs/known-gaps.md](docs/known-gaps.md).

## Runtime and libraries

`libcumetal.dylib` implements the CUDA Runtime and Driver API over Metal. It
tracks allocations, resolves CUDA pointers to Metal buffers, preserves the
per-thread error model, and maps streams and events onto command queues and
shared-event ordering.

The same library also exports compatibility surfaces for:

- cuBLAS and cublasLt
- cuRAND
- cuFFT
- cuSPARSE
- cuSOLVER Dense
- cuDNN
- CUDA Graphs
- NVML
- NCCL single-rank operations
- a small CPU-backed Thrust surface over unified memory
- async allocation, texture, and surface object lifecycle APIs

These names do not imply full NVIDIA parity. Some operations use
MetalPerformanceShaders, some use Accelerate, some exploit unified memory, and
some are deliberately partial. Read [docs/status.md](docs/status.md) before
building on one.

Large `cudaMalloc` allocations use `MTLHeap` suballocation at 4 MiB and above.
Override this for diagnosis:

```bash
CUMETAL_MTLHEAP_ALLOC=1 ./program   # always
CUMETAL_MTLHEAP_ALLOC=0 ./program   # never
```

## Binary shim

By default, Release builds keep source registration enabled and the
`libcuda.dylib` alias disabled. Enable the alias explicitly:

```bash
cmake -B build-shim \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUMETAL_ENABLE_BINARY_SHIM=ON
cmake --build build-shim
```

`CUMETAL_ENABLE_CUDA_REGISTRATION=ON` is the host registration ABI emitted by
Clang for recompiled `.cu` programs. It is part of the source path.
`CUMETAL_ENABLE_BINARY_SHIM=ON` only adds the drop-in `libcuda.dylib` name.
Do not confuse them.

The shim recognizes CMTL envelopes, raw PTX, and basic
FatBinary/FatBinary2/FatBinary3 PTX wrappers. It does not execute SASS and does
not understand every NVCC fatbinary variant.

Registered PTX is compiled on first use and cached under:

```text
$CUMETAL_CACHE_DIR/registration-jit/
```

The key includes the input, kernel, lowering policy, schema versions,
toolchain-dependent inputs, and the `libcumetal` Mach-O UUID. Set
`CUMETAL_DEBUG_REGISTRATION=1` to see format detection, compilation, cache hits,
ABI inference, and registration.

The legal boundary is documented in
[docs/legal-notice.md](docs/legal-notice.md).

## What has actually been demonstrated

### Small kernels

Vector add, SAXPY, reduction, matrix operations, atomics, shared memory, warp
operations, streams, events, and selected CUDA library calls have numerical GPU
tests. The suite includes negative cases because accepting a program is not the
same as implementing it correctly.

### Performance

The Phase 5 gate compares CuMetal with hand-written Metal for three
memory-bound kernels. On an Apple M4 Pro, rebuilt and measured on 2026-07-27:

| Kernel | Elements | CuMetal / native Metal |
| --- | ---: | ---: |
| vector add | 262,144 | 1.063× |
| SAXPY | 262,144 | 1.036× |
| FP32 reduction | 262,144 | 1.008× |

The gate uses the fastest of 20 synchronized wall-clock iterations. These
kernels take roughly 0.2 ms, so averages mostly measure scheduler interference.
The target is at most 2× native Metal, not a suspicious claim that translated
code beats the baseline.

Reproduce it:

```bash
ctest --test-dir build -R bench_phase5_all_kernels --output-on-failure
```

### Real programs

- Upstream `cuda-samples` vector add builds without source changes and passes a
  numerical plus Apple-GPU provenance gate.
- llm.c GPT-2 FP32 passes logits, loss, tensor, and GPU-provenance checks on the
  tested path. It uses explicit workload specializations and is not proof of
  arbitrary PTX support.
- llama.cpp's unmodified GGML CUDA backend builds against CuMetal. SmolLM2-135M
  greedy decoding was coherent from one-layer offload through saturation on the
  verified Apple M4 Pro setup. FlashAttention is advertised as unsupported, so
  llama.cpp selects its ordinary attention path.
- A reduced PhysX 5.6 GRB path runs selected sphere, box, convex, and triangle
  mesh contacts on the GPU. It is a selected-shape conformance target, not
  general PhysX GPU support.

Exact commands, models, tolerances, provenance requirements, results, and
scope boundaries live in
[docs/apple-gpu-execution.md](docs/apple-gpu-execution.md),
[docs/testing.md](docs/testing.md), and
[docs/known-gaps.md](docs/known-gaps.md). Compatibility claims without those
conditions are meaningless.

## Known hard limits

- No dynamic parallelism.
- No multi-GPU or peer-to-peer execution.
- No OpenGL, Vulkan, or DirectX interop.
- No SASS execution.
- Multi-block cooperative launch/grid sync is rejected; single-block
  cooperative launch is supported.
- FP64 register emulation provides about a 44-bit mantissa, not IEEE binary64;
  unsupported binary64 memory/conversion boundaries fail compilation.
- Texture and surface object lifecycle exists; general device-side sampling
  does not.
- Device `printf` uses a bounded buffer and limits format strings to 256 bytes.
- CUDA, cuDNN, PhysX, llama.cpp, and PTX coverage is incomplete.

This list is intentionally short. The authoritative list is
[docs/known-gaps.md](docs/known-gaps.md).

## Test without lying to yourself

Run both the normal source-first build and the binary-shim build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
bash scripts/ci_report.sh build --exclude-regex '^bench_'

cmake -B build-shim \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCUMETAL_ENABLE_BINARY_SHIM=ON
cmake --build build-shim
bash scripts/ci_report.sh build-shim --exclude-regex '^bench_'
```

For focused work:

```bash
ctest --test-dir build -R unit_ --output-on-failure
ctest --test-dir build -R functional_ --output-on-failure
ctest --test-dir build -R conformance_ --output-on-failure
```

Report passes, skips, and failures separately. A registered test is not a
passing test. A skip is not evidence of compatibility. A correct answer without
GPU provenance may be a CPU fallback. The test policy exists because all three
mistakes have happened before.

## Tools

| Tool | Job |
| --- | --- |
| `cumetalc` | Compile `.cu`, PTX, or NVVM IR to inspectable stages, `.metallib`, or an executable |
| `air_inspect` | Inspect kernels, bitcode offsets, and metadata in a `.metallib` |
| `air_validate` | Validate `.metallib` structure and optionally check it with `xcrun` |
| `cumetal-air-emitter` | AIR research and regression container generation |
| `cumetal-ptx2llvm` | Legacy PTX-to-LLVM inspection |
| `ptx_diff` | Compare PTX-related outputs |
| `cumetal_bench` | Compare covered CuMetal kernels with native Metal |

## Documentation

- [Design specification](spec.md) — canonical architecture and semantics
- [Current status](docs/status.md) — what is implemented
- [Known gaps](docs/known-gaps.md) — what is partial, wrong, or absent
- [Build guide](docs/build.md) — toolchains and validation
- [Testing guide](docs/testing.md) — gates and conformance workflows
- [Compiler architecture](docs/compiler-architecture.md) — lowering paths and migration boundaries
- [Apple-GPU execution record](docs/apple-gpu-execution.md) — evidence and provenance
- [AIR ABI notes](docs/air-abi.md) — metallib research and limitations
- [Correctness audit](docs/correctness-audit-2026-07-26.md) — failures found by testing the tests

If the README and the spec disagree, the spec wins.

## License

[Apache 2.0](LICENSE)
