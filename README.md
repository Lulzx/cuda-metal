# CuMetal

CuMetal recompiles a tested subset of CUDA C++ and PTX for Apple Metal and
provides a clean-room CUDA compatibility runtime for Apple Silicon. It is
experimental: supported paths run real Metal kernels, while unsupported paths
should fail explicitly.

## Start here

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(sysctl -n hw.ncpu)"
bash scripts/ci_report.sh build --exclude-regex '^bench_'

build/cumetalc samples/vectorAdd/vectorAdd.cu -o vectorAdd
./vectorAdd
```

Requirements are macOS 14 or newer on Apple Silicon, CMake, a CUDA-capable LLVM
Clang, and Apple's public `metal` and `metallib` tools. See the
[build and installation guide](docs/build.md) before packaging or installing.
The published Homebrew formula currently targets the older 0.1.3 release, so it
is not evidence for the behavior documented in this checkout.

For a staged GPU demonstration:

```bash
bash demos/apollo/run.sh
```

Apollo requires numerical checks and `device=apple_gpu` provenance at every
stage. Other bounded demonstrations are indexed in [the demos guide](docs/demos.md).

## Architecture

CuMetal is source-first:

```text
CUDA C++ -> Clang/NVVM -> typed CuMetal IR -> MSL -> Apple tools -> metallib
PTX -------------------> legacy or typed lowering --------------------^
```

- Direct `.cu` compilation is the primary path and defaults to typed CuMetal IR.
- PTX and `.cu --cuda-device` default to the broader legacy compatibility
  backend while typed PTX migration remains incomplete.
- The optional `libcuda.dylib` alias accepts only supported binary-registration
  and embedded-PTX forms. It is disabled in Release builds unless explicitly
  enabled. SASS execution is unsupported.
- SIMD/warp width is fixed at 32. Metal calls remain inside
  `runtime/metal_backend/`; CUDA-facing headers are clean-room; no private Apple
  API is used.

The canonical requirements are indexed by [the specification](spec.md). Current
compiler paths and their selection policy are in
[compiler architecture](docs/compiler-architecture.md).

## Current verified boundary

- The enrolled headless NVIDIA `cuda-samples` snapshot is **83/83 pass**, with
  zero waivers and zero nonpassing manifest entries.
- The reproducible 23-file production-metallib matrix is **9/23** for direct
  `.cu` through typed CuMetal IR, **6/23** for typed PTX, and **23/23** for the
  legacy PTX backend. These are compile results, not numerical runtime proof.
- Vector add, SAXPY, reduction, selected matrix/library operations, atomics,
  shared memory, warp operations, streams, and events have focused numerical
  Apple-GPU tests.
- `VF64-metal` is pinned and its `fast48`, `wide48`, and `ieee64` integration
  validation passes on the recorded Apple M4 Pro system. Each mode has different
  precision semantics; this is not native Metal FP64.

Exact commands, tolerances, device provenance, and third-party boundaries are
in [verified results](docs/verified-results.md). The executable source/sample
matrix is `tests/cuda_projects/backend_matrix_manifest.txt`.

## Important limits

- Apple Silicon/macOS only; no SASS, multi-GPU, peer access, or graphics-API
  interop.
- CUDA and library APIs are tested subsets, not NVIDIA-compatible drop-ins.
- Cross-threadgroup synchronization is limited to a resident cooperative grid
  capped at one block per reported GPU core.
- FP64 uses `fast48`, `wide48`, or software `ieee64`; observable IEEE exception
  status is not fully integrated.
- Dynamic launch uses a bounded device queue drained by the host.
- Texture/surface lifecycle and a source descriptor subset exist; direct PTX
  texture/surface instructions do not.
- Graph allocator reuse, cross-stream capture, and other advanced semantics are
  incomplete.

See [known gaps](docs/known-gaps.md) for the maintained classification and
[the closure roadmap](docs/spec-closure-roadmap.md) for prioritized work.

## Commands

```bash
# Inspect compiler stages
build/cumetalc kernel.cu --emit=cumetal-ir -o kernel.cmir
build/cumetalc kernel.cu --emit=msl -o kernel.metal
build/cumetalc kernel.cu --emit=metallib -o kernel.metallib

# Enable the optional binary alias in an explicit build
cmake -B build-shim -DCMAKE_BUILD_TYPE=Release \
  -DCUMETAL_ENABLE_BINARY_SHIM=ON
cmake --build build-shim

# Validate both source-first and no-shim configurations
ctest --test-dir build --output-on-failure
cmake -B build-noshim -DCMAKE_BUILD_TYPE=Debug \
  -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build build-noshim
ctest --test-dir build-noshim --output-on-failure
```

`cumetal doctor` checks an installed toolchain. `cumetal run` only scopes this
installation's runtime library path to a child process; it cannot make an
unsupported binary portable.

## Documentation index

The full documentation map is [docs/README.md](docs/README.md). Primary entries:

- [Specification](spec.md) — canonical requirements and chapter index
- [Status](docs/status.md) — implemented surfaces
- [Known gaps](docs/known-gaps.md) — partial, absent, and bounded behavior
- [Verified results](docs/verified-results.md) — measured evidence only
- [Testing](docs/testing.md) — gates and evidence policy
- [Build](docs/build.md) — toolchains, installation, and diagnostics
- [Legal notice](docs/legal-notice.md) — source and binary usage boundaries

If documentation conflicts, `spec.md` wins, followed by `AGENTS.md`. Status and
README material may lag and must not override the specification.

## License

[Apache 2.0](LICENSE)
