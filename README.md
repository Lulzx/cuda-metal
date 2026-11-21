# CuMetal

CuMetal is a CUDA compiler and runtime for Apple Silicon. It compiles supported
CUDA C++ and PTX into Metal kernels, so you can run existing CUDA code on your
Mac’s GPU without rewriting it in Metal.

The project is experimental and supports a tested subset of CUDA and its
libraries. See [verified results](docs/verified-results.md) for what runs today
and [known gaps](docs/known-gaps.md) for the remaining limits.

<!-- cw:block readme.lammps -->
The [LAMMPS/Kokkos GPU demo](demos/lammps/README.md) compiles unmodified LAMMPS
and passes the stock 32,000-atom, 100-step full/half-neighbour numerical gate.
See [demos](docs/demos.md) for more workloads.

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

FP64 arithmetic comes from the `third_party/VF64-metal` submodule; initialize it
with `git submodule update --init --recursive`.

From a checkout with the [build prerequisites](docs/build.md) installed:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(sysctl -n hw.ncpu)"

build/cumetalc samples/vectorAdd/vectorAdd.cu -o vectorAdd
./vectorAdd
```

See [testing](docs/testing.md) for validation commands.

## How it works

```text
CUDA C++ / PTX → CuMetal compiler → Metal Shading Language → Apple tools → metallib
```

- **Direct CUDA C++, compiled ahead of time.** `cumetalc` compiles CUDA C++
  through typed CuMetal IR and embeds the Metal library in the executable, so
  there is no first-launch PTX JIT. CuMetal uses no private Apple APIs.
- **PTX import with SSA and pointer-provenance recovery.** PTX registers become
  typed SSA values, and addresses stay real pointers in their Metal address
  space. Bounded proofs recover pointer arithmetic; a Metal pointer is never
  turned into an integer, and what cannot be proved is rejected, not guessed.
- **FP64 emulation with [VF64-metal](third_party/VF64-metal).** Apple GPUs have
  no native FP64. `double` runs in a selectable emulation mode: `fast48` by
  default, `ieee64` for correctly rounded results
  ([FP64 policy](docs/fp64-policy.md)).
- **64-bit atomics and scoped memory (Metal 3.2+).** Typed lock-backed 64-bit
  atomics use device-coherent payload pointers and device fences, covering
  binary64 `atomicAdd` and cross-block scan descriptors.

The proof rules behind the PTX path are in
[compiler architecture](docs/compiler-architecture.md#ptx-memory-proof-rules);
[known gaps](docs/known-gaps.md) lists memory-provenance and legacy-backend
limits.

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
- [Matmul performance study](docs/matmul-performance.md) and [compiler timing and optimization](docs/compiler-performance.md)
- [All documentation](docs/README.md)

## License

[Apache 2.0](LICENSE) · [Legal notice](docs/legal-notice.md)
