# CuMetal

CUDA programs assume an NVIDIA machine. CuMetal makes a useful subset of them
run on Apple Silicon.

It is a compiler, a runtime, and a set of compatibility libraries. CUDA source
or PTX goes in. Metal runs on the GPU. There is no NVIDIA hardware in the loop.

This is experimental software. The covered paths execute real kernels and check
real answers. Unsupported paths are expected to fail explicitly. That is a
better failure mode than silently computing nonsense.

## Install

CuMetal requires macOS 14 or newer on Apple Silicon. Install the source-first
compiler and runtime from the official CuMetal tap:

```bash
brew install lulzx/tap/cumetal
cumetalc vectorAdd.cu -o vectorAdd
./vectorAdd
```

Verify the complete local toolchain with:

```bash
cumetal doctor
```

Homebrew installs CMake and LLVM as dependencies. Apple's Metal compiler still
comes from Xcode; if `xcrun --find metal` fails, install Xcode and its Metal
Toolchain component.

The formula deliberately installs the source-first compiler/runtime without
the optional `libcuda.dylib` binary shim.

## Build from source

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

### Try it

```bash
bash demos/apollo/run.sh
```

Apollo progresses from vector addition through a path tracer and requires
Apple-GPU provenance at every stage. The 3D Gaussian Splatting, SPH, diffusion,
and single-sample workflows are in [the demos guide](docs/demos.md).

Install it without changing shell startup files:

```bash
bash install/install.sh build /opt/cumetal
/opt/cumetal/bin/cumetal doctor
```

To also add CuMetal to your shell's `PATH`, opt in explicitly:

```bash
bash install/install.sh build /opt/cumetal --shell-config
```

Remove the installation with `/opt/cumetal/uninstall.sh`. The uninstaller uses
the recorded CMake install manifest, so every installed header, tool, shim, and
library is covered.

## Run

Source-built programs need no launcher:

```bash
cumetalc samples/vectorAdd/vectorAdd.cu -o vectorAdd
./vectorAdd
```

`cumetal run` is a convenience for launching a process with this installation's
runtime library path scoped to that child:

```bash
cumetal run ./cuda-application
```

It does not make unsupported binaries portable. Prebuilt CUDA applications
still require a compatible PTX payload and an installation configured with
`CUMETAL_ENABLE_BINARY_SHIM=ON`; SASS-only applications remain unsupported.

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

`cumetalc` can emit inspectable compiler stages, a `.metallib`, or a runnable
executable:

```bash
cumetalc kernel.cu --emit=cumetal-ir -o kernel.cumetal
cumetalc kernel.cu --emit=msl        -o kernel.metal
cumetalc kernel.cu --emit=metallib   -o kernel.metallib
cumetalc kernel.cu --emit=exe        -o kernel
```

Direct `.cu` input currently defaults to the typed shared-IR backend; PTX and
`--cuda-device` use the broader compatibility backend. The measured selection
gate, command-line policy switches, and legality stages are in
[the compiler architecture guide](docs/compiler-architecture.md). Unsupported
instructions, calls, pointer conversions, CFGs, and ABI forms remain in
[known gaps](docs/known-gaps.md).

## Runtime and libraries

`libcumetal.dylib` implements the CUDA Runtime and Driver API over Metal. It
tracks allocations, resolves CUDA pointers to Metal buffers, preserves the
per-thread error model, and maps streams and events onto command queues and
shared-event ordering.

The same library exports tested subsets of CUDA graphs, cuBLAS/cublasLt,
cuRAND, cuFFT, cuSPARSE, cuSOLVER, cuDNN, NVML, NCCL, Thrust, async allocation,
and texture/surface lifecycle APIs. Those names do not imply full NVIDIA
parity. Read [current status](docs/status.md) for implemented surfaces and the
[build guide](docs/build.md) for runtime diagnostic controls.

## Binary shim

Release builds keep source registration enabled and the `libcuda.dylib` alias
disabled. Enable the alias explicitly:

```bash
cmake -B build-shim \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUMETAL_ENABLE_BINARY_SHIM=ON
cmake --build build-shim
```

Source registration and the binary alias are independent switches. Supported
container forms, JIT-cache identity, diagnostics, and validation commands are
in [the build guide](docs/build.md). SASS execution remains unsupported, and
the legal boundary is in [the legal notice](docs/legal-notice.md).

## What has actually been demonstrated

Vector add, SAXPY, reduction, matrix operations, atomics, shared memory, warp
operations, streams, events, and selected CUDA library calls have numerical GPU
tests. The suite includes negative cases because accepting a program is not the
same as implementing it correctly.

The recorded native-Metal comparison, real-program gates, their scope, and
third-party projects using CuMetal are in
[verified results](docs/verified-results.md). Exact commands, models,
tolerances, and provenance requirements remain in
[the Apple-GPU execution record](docs/apple-gpu-execution.md).

## Known hard limits

Durable platform/legal boundaries:

- CuMetal targets macOS on Apple Silicon; Windows, Linux ARM, discrete GPUs,
  and Thunderbolt eGPU execution are outside the supported platform.
- No SASS execution or decompilation; binary compatibility requires embedded PTX.
- No multi-GPU or peer-to-peer execution on the single-GPU Apple Silicon target.
- No OpenGL, Vulkan, or DirectX interop.
- Metal has no single-dispatch cross-threadgroup barrier. Multi-block cooperative
  launch/grid sync is rejected; single-block cooperative launch is supported.
- Current public Metal compilation rejects native AIR `double`. FP64 register
  emulation provides about a 44-bit mantissa, and unsupported binary64
  memory/conversion boundaries fail compilation.
- Public Metal exposes no CUDA persisting-L2/access-policy-window control;
  CuMetal reports those capabilities as zero and rejects nontrivial requests.

Engineering gaps, not fundamental impossibilities:

- Dynamic parallelism needs a CPU trampoline and compatible scheduling/error semantics.
- Texture and surface object lifecycle exists; general device-side sampling needs
  a Metal texture binding ABI.
- CUDA graphs cover tested dependency-ordered kernel/linear-memcpy/memset/host-node
  capture and replay, cloning, and compatible executable updates, but memory nodes
  and other advanced behavior are incomplete.
- Device `printf` uses a bounded buffer and limits format strings to 256 bytes.
- CUDA, library-shim, PhysX, llama.cpp, and PTX coverage is incomplete.

This summary is intentionally short. The authoritative classification of
platform boundaries and implementable engineering gaps is in
[docs/known-gaps.md](docs/known-gaps.md).

## Test without lying to yourself

For routine source-first validation:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
bash scripts/ci_report.sh build --exclude-regex '^bench_'
```

Report passes, skips, and failures separately. A registered test is not a
passing test. A skip is not evidence of compatibility. A correct answer without
GPU provenance may be a CPU fallback. The test policy exists because all three
mistakes have happened before.

Binary-shim validation, focused test selections, CUDA sample setup, CI state,
and the runner contract are documented in [the testing guide](docs/testing.md).

## Tools

| Tool | Job |
| --- | --- |
| `cumetalc` | Compile `.cu`, PTX, or NVVM IR to inspectable stages, `.metallib`, or an executable |
| `cumetal` | Check an installation with `doctor` or launch a child process with `run` |
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
- [Demos](docs/demos.md) — runnable showcases and their evidence gates
- [Verified results](docs/verified-results.md) — benchmarks, real programs, and downstream usage
- [Testing guide](docs/testing.md) — gates and conformance workflows
- [Compiler architecture](docs/compiler-architecture.md) — lowering paths and migration boundaries
- [Apple-GPU execution record](docs/apple-gpu-execution.md) — evidence and provenance
- [AIR ABI notes](docs/air-abi.md) — metallib research and limitations
- [Correctness audit](docs/correctness-audit-2026-07-26.md) — failures found by testing the tests

If the README and the spec disagree, the spec wins.

## License

[Apache 2.0](LICENSE)
