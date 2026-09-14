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

The experimental [Rust PTX harness](demos/rust-ptx/README.md) consumes
separately exported Rust-CUDA kernels and checks vector addition and SHA-256.
Both kernels from the unchanged, pinned Rust-CUDA PTX artifact pass numerical
checks on Apple M5 using runtime MSL compilation. See the harness's recorded
provenance and regression commands; this is not full Rust-CUDA compatibility.
The fresh full miner module passes its launch probe and its original
[Ed25519 known-answer self-test](docs/experiments/ptx-halfword-tuples.md) on
Apple M5, with intact result-buffer guards. Focused GPU regressions cover table
pointers, multiline calls, narrow loads, bit insertion and halfword packing.
Other miner kernels and broader input coverage still require validation.

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
- [Matmul performance study](docs/matmul-performance.md) and [compiler optimization](docs/compiler-performance.md)
- [All documentation](docs/README.md)

## License

[Apache 2.0](LICENSE) · [Legal notice](docs/legal-notice.md)

The new LLVM 7 and LLVM 19 miner artifacts both pass the launch probe and the
single Ed25519 known-answer fixture on Apple M5, with result and guard checks.
LLVM 19 required typed pointer relocation, a reviewed guarded-select rewrite,
and explicitly typed integer min/max operands. Full mining kernels remain
unvalidated. See [dual LLVM smoke evidence](docs/experiments/dual-llvm-miner-smoke.md).

For the next small validation steps and remaining work, see the
[vanity-miner validation checklist](docs/experiments/vanity-miner-validation-todo.md).

The LLVM 7 xoroshiro self-test also passes after bounded scalar tail-call
normalization and byte-array return packing. Runtime-seeded helper regressions
cover 261 seeds. LLVM 19's bounded local-buffer tail recursion also passes its
original xoroshiro self-test and 388 runtime seed cases. Both base58 variants
now get past trap lowering. LLVM 7 base58 numerically passes after correcting
unsigned integer widening; LLVM 19 now also numerically passes after preserving
pointer types and byte-offset subtraction through its reverse loop. See the
[miner fix backlog](docs/experiments/miner-fix-backlog.md) for scope and evidence.

The complete self-test rerun on `6d2549b` passes 90/118 numerical entries from
LLVM 7 and 82/118 from LLVM 19, plus both launch probes. All 64 remaining entries
fail compilation; no launched entry fails its numerical or guard checks.
[Full ledger and remaining blocker groups](docs/experiments/miner-self-test-sweep-6d2549b.md).
Passing fixed fixtures may be constant-folded; full mining kernels remain unvalidated.

Bounded trap propagation now supports device-call graphs: trapping/looping helpers
expand into the cancellation CFG, while proven finite helpers remain calls.
Both original compressed-mainnet WIF self-tests now pass on M5. After fixing
parameter-store truncation and signed-byte conversion, the original LLVM 19
compressed secp256k1 self-test now passes on M5 with guards intact. Six smaller
k256 checks also pass. LLVM 7 still has an earlier bounded runtime-attempt timeout.
See [fix 12](docs/experiments/miner-fix-backlog.md#fix-12-cvt-interprets-the-instructions-source-width)
and [intermediate-value evidence](docs/experiments/secp256k1-isolation.md).
At `93ebd6b`, the original LLVM 19 uncompressed secp256k1, Ethereum-address and
Bitcoin-Bech32-address self-tests also pass on M5, with guards intact.
[Targeted results](docs/experiments/secp256k1-dependent-checks.md).
The six remaining Ethereum/Bitcoin intermediate and matching checks also pass
individually at `b4894ed`, bringing that LLVM 19 group to **8/8 passing entries**.
[Evidence and coverage limits](docs/experiments/address-intermediates.md).
This does not establish a complete pass of the 118-test suite or the mining kernels.
