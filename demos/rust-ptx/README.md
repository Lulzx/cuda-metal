# Rust-generated PTX proof of concept

Boundary: Rust-CUDA exports PTX on Linux; CuMetal consumes the unchanged PTX
on macOS. Rust-CUDA does not depend on CuMetal. This harness links directly
against `libcumetal`, using its clean-room Driver API header, without `cust`,
NVIDIA SDK headers, or the optional `libcuda.dylib` alias on the Mac.

## Produce the artifact

Use the `examples/ptx_export` example on the Rust-CUDA `poc/portable-ptx-export`
branch. Its README documents the explicit pointer/count ABI. Run on Linux:

```sh
nix develop .#v19 --command cargo run -p ptx_export --features llvm19 -- artifacts/ptx
```

The fork's `Export portable PTX` GitHub workflow performs the same build without
an NVIDIA GPU and uploads `rust-ptx`, including the source commit, compiler version,
PTX checksum, and LLVM output. Copy/download those files to the Mac.

## Build and run on the Mac

Build CuMetal following `docs/build.md`, including its VF64-metal submodule.
For this PTX-only experiment, LLVM import is optional:
`-DCMAKE_DISABLE_FIND_PACKAGE_LLVM=TRUE` disables only the CUDA/NVVM frontend.
Keep `-DCUMETAL_ENABLE_BINARY_SHIM=OFF`.

For an Apple Silicon PTX-only build, the optional Nix shell supplies CMake,
Ninja, Python, LZ4, and Zstd while using Apple's installed host compiler/SDK:

```sh
git submodule update --init --recursive
nix develop
cmake -S . -B build-rust-ptx -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_LLVM=TRUE -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build build-rust-ptx --target cumetalc cumetal_runtime -j4
```

Use `--build-dir build-rust-ptx` below with this configuration. The flake does
not download Xcode, the Metal Toolchain, or NVIDIA's toolkit. It is intentionally
a PTX-only development shell, not the full supported CUDA C++ build environment.

```sh
python3 demos/rust-ptx/run.py /path/to/rust_kernels.ptx --build-dir build --mode vecadd
python3 demos/rust-ptx/run.py /path/to/rust_kernels.ptx --build-dir build --mode sha256
```

The script emits MSL through the strict typed PTX backend and requires the
compiler-generated `.cumetal-abi` sidecar. CuMetal compiles that MSL through
Metal's public runtime API; this route does not invoke `xcrun metal` or
`metallib`. Command Line Tools supply the host SDK/compiler; Xcode's separately
downloaded Metal tools are needed for offline metallib production, not this path.

Every run uses counts 1, 31, 32, 33, and 257 with oversized dispatches and output
tail guards. Vector addition uses exactly representable float inputs. SHA-256
hashes distinct 32-byte messages and compares every digest with CommonCrypto,
plus a fixed known answer for 32 zero bytes. Workload specializations are disabled.
Success requires both numerical results and generic-PTX Apple-GPU provenance for
each dispatch. A compiler error, GPU failure, or missing provenance fails the run.

`out/` holds checksums, generated MSL/ABI, compilation diagnostics, and execution
logs. Successful compilation alone is not evidence that the GPU results match.

This is an experimental integration fixture, not a claim of full Rust-CUDA or
vanity-miner compatibility. The next step after these kernels pass is a bounded
Shallenge batch and then individual crypto self-test kernels.

## Verified Rust artifact (2026-09-13)

Both kernels passed on Apple M5 for counts 1, 31, 32, 33, and 257, including
CPU comparisons, the SHA-256 fixed known answer, tail guards, and
`generic_ptx_lowering` / `device=apple_gpu` provenance with specializations off.
The [execution transcript](verified-apple-m5.txt) records these checks.
This uses runtime MSL compilation with Command Line Tools, without offline
`metal`/`metallib` tools. No performance comparison is claimed.

- Producer: Rust-CUDA commit `93d104d36708b9bfa7ad7fe170c36eb09768eed5`.
- [Successful export run](https://github.com/brandonros/Rust-CUDA/actions/runs/34773189004).
- Toolchain: the producer's locked `.#v19` shell, `--features llvm19`.
- PTX SHA-256: `237ae88084e727c0e2f9b1d51a1f9003cf5d5dfec4da3622c2fc9c8961967a69`.
- Byte-for-byte artifact: `tests/functional/reference/rust_ptx_export_93d104d.ptx`.
  It was generated from the Rust exporter kernels (including RustCrypto `sha2`);
  no instructions or target headers were rewritten for CuMetal.

The integration required typed lowering for `shf.{l,r}.wrap.b32` and generic
`prmt.b32`, vector stores with immediate lanes, and truncation of wider register
values for byte/halfword stores. Strict opcode validation now checks only the
selected entry and reachable device helpers. Other entries remain uncompiled.
`shf` clamp variants, specialized `prmt` modes, and predicated forms of these
operations remain unsupported by this typed path.

Reproduce with a tests-enabled build:

```sh
ctest --test-dir build-rust-ptx -R '^(unit_ptx_ir_msl|functional_ptx_bit_permutation|functional_rust_ptx_export)$' --output-on-failure
```

The independent bit-permutation GPU regression covers all 65,536 selectors
(including byte sign replication), ignored upper selector bits, distinct source
words, shift boundaries 0/1/7/16/31/32/33/63/64/UINT_MAX, mixed literal/register
stores, byte/halfword truncation, and output guards. Unsupported instruction
variants, malformed tuples, and reachable unsupported helpers have negative
host regression coverage.
