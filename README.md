# CuMetal

CuMetal is a CUDA compiler and runtime for Apple Silicon. It compiles supported
CUDA C++ and PTX into Metal kernels, so you can run existing CUDA code on your
Mac’s GPU without rewriting it in Metal.

The project is experimental and supports a tested subset of CUDA and its
libraries. See [verified results](docs/verified-results.md) for what runs today
and [known gaps](docs/known-gaps.md) for the remaining limits.

## Install

Requires Apple Silicon and macOS 14 or newer. See the
[installation guide](docs/build.md) for compiler and Apple toolchain requirements.

```bash
brew install lulzx/tap/cumetal
cumetal doctor
```

Compile and run your CUDA source:

```bash
cumetalc kernel.cu -o kernel
./kernel
```

## Build from source

From a checkout with the [build prerequisites](docs/build.md) installed:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(sysctl -n hw.ncpu)"

build/cumetalc samples/vectorAdd/vectorAdd.cu -o vectorAdd
./vectorAdd
```

See [demos](docs/demos.md) for larger workloads and [testing](docs/testing.md)
for validation commands.

## How it works

```text
CUDA C++ / PTX → CuMetal compiler → Metal Shading Language → Apple tools → metallib
```

Source recompilation is the primary path. Direct CUDA C++ compilation uses
typed CuMetal IR and embeds the compiled Metal library in the executable,
with no first-launch PTX JIT. CuMetal uses no private Apple APIs.
See [compiler architecture](docs/compiler-architecture.md) for backend details.

## Limits

- CUDA and library APIs are tested subsets, not drop-in replacements.
- SASS execution is unsupported. The optional binary shim is
  disabled in Release builds unless explicitly enabled.
- SIMD/warp width is fixed at 32. Multi-GPU, peer access, and graphics-API
  interop are unsupported.
- FP64 uses emulation with mode-dependent precision; it is not native Metal FP64.
- Cooperative grids, dynamic launch, graphs, and textures have bounded or
  incomplete support. See [known gaps](docs/known-gaps.md) for exact limits.

## Documentation

- [Status](docs/status.md) and [verified results](docs/verified-results.md)
- [Roadmap](docs/spec-closure-roadmap.md) and [specification](spec.md)
- [All documentation](docs/README.md)

## License

[Apache 2.0](LICENSE) · [Legal notice](docs/legal-notice.md)
