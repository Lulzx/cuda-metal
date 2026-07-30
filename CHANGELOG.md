# Changelog

All notable changes to CuMetal are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-07-30

### Changed

- **`cumetal doctor` now has a Flutter-style, color-aware summary.** Required components use
  green checks, failures use red crosses, the optional binary shim is clearly informational,
  redirected output stays free of ANSI escapes, and `NO_COLOR` is respected.
- **The doctor example is now real.** `vectorAdd.cu` is installed under CuMetal's shared examples
  directory, and doctor prints a copy-paste compile command using its resolved absolute path.
  The installed-prefix gate compiles that exact installed file.

## [0.1.0] - 2026-07-30

CuMetal compiles CUDA source to Metal and runs it on Apple Silicon GPUs, with a CUDA-compatible
runtime backed by Metal and Apple's acceleration frameworks. Read [What works](#what-works)
below before depending on it: CuMetal supports a documented subset of CUDA, not arbitrary CUDA
programs.

### Added

- **Homebrew tap packaging and an installed `cumetal` front door.** The
  `Lulzx/homebrew-tap` formula builds the source-first Release configuration with Homebrew LLVM
  and verifies it by compiling and running a CUDA kernel. `cumetal doctor` checks the complete
  local toolchain; `cumetal run` scopes runtime lookup to one child process without requiring
  global `DYLD_*` exports.
- **An installed-prefix end-to-end gate.** A fresh manifest-backed install must locate all of its
  headers, libraries, and Clang shims, pass `cumetal doctor`, compile the unmodified `vectorAdd`
  source, and execute it on the Apple GPU without caller environment setup.
- **`cumetalc foo.cu -o foo` builds a runnable executable.** An ordinary CUDA file — host code,
  `__global__` kernels, `<<<>>>` launches — compiles and runs with no host/device split and no
  `.metallib` path at runtime. Clang compiles the whole translation unit; device code travels as
  PTX inside a fatbinary and is lowered to a `.metallib` on first launch. Works from an install
  prefix as well as the build tree (`CUMETAL_ROOT` overrides discovery).
  `--link` / `--no-link` force the behavior; `--emit exe` is equivalent to `--link`.
- **`--save-temps`** keeps the intermediate object file from a link.
- **`cumetalc --version`**, plus `cumetalGetVersion()` / `cumetalGetVersionString()` in the
  runtime and `CUMETAL_VERSION*` macros in `cumetal_native.h`, so a version mismatch between
  headers and a loaded dylib is detectable.
- **`scripts/ci_report.sh`** reports passed/skipped/failed separately and names every skipped
  test, so a run cannot read as full coverage when part of the suite never executed.
- **`CUMETAL_ENABLE_CUDA_REGISTRATION`** build option (default `ON` in every build type),
  controlling the host CUDA registration ABI independently of the binary shim.
- **`samples/nativeLaunch`** documents the native `cumetalKernel_t` launch API, which
  `samples/vectorAdd` no longer needs to demonstrate.
- **`CONTRIBUTING.md`**, **`docs/cla.md`**, and **`SECURITY.md`** — the clean-room contribution
  certification that `docs/legal-notice.md` referred to now exists, signed via `git commit -s`.
- **`ptx_sweep_numeric`** (spec §10.2): executes each PTX opcode on the GPU and compares
  bit-for-bit against a hand-derived ISA oracle, classifying SUPPORTED / WRONG / UNSUPPORTED.
  Covers integer and float arithmetic, shifts, bit ops, and every `cvt` rounding mode.
- **AIR ABI toolchain provenance** (spec §10.5): every AIR ABI test prints the macOS build, Xcode
  version, selected `TOOLCHAINS`, chip, and metal compiler version, so a result is attributable
  to the toolchain that produced it.

### Changed

- **The installer no longer edits shell startup files by default.** `--shell-config` is an
  explicit opt-in, and even that only adds the installation's `bin` directory to `PATH`. Installs
  now retain CMake's exact manifest so uninstall covers every tool, header, library, and shim.
- **Name-selected llm.c/GGML workload bodies are opt-in.** Generic PTX lowering still runs first;
  if it declines, the specialized table is consulted only with
  `CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=1`. The strict llm.c, llama.cpp, and GGML conformance
  launchers opt in explicitly; arbitrary CUDA projects cannot acquire a body from a colliding
  entry name.
- **The manifest-complete CUDA-project sweep uses a fresh JIT cache per fixture** and now enrolls
  the libdevice and ray-tracer strict projects. This prevents a prior compiler build from
  concealing a lowering regression and raises the clean local sweep from 9 to 11 projects.
- **`CUMETAL_ENABLE_BINARY_SHIM` now controls only the `libcuda.dylib` alias.** It previously
  also gated the host registration ABI (`__cudaRegister*`), which Clang emits when compiling
  *your own* `.cu` source. Because the flag defaults off in Release, the source-recompilation
  path was stubbed out and untested in the configuration users install.
- **The `cumetalc` backend default follows the input** rather than being one global setting.
  A direct `.cu` defaults to `--backend=cumetal-ir`; `--cuda-device` and PTX inputs default to
  `--backend=legacy`. Measured over the 19-file in-tree `.cu` corpus, direct `.cu` is legacy
  0/19 vs cumetal-ir 10/19, and `--cuda-device` is legacy 17/19 vs cumetal-ir 6/19 — the
  backends are complementary, not ranked. `--backend` overrides.
- `samples/vectorAdd` is a single self-contained `.cu` built with `cumetalc -o`.

### Fixed

- **`cumetalc` now quotes the complete temporary `PATH` assignment.** A normal desktop `PATH`
  containing a directory with spaces previously prevented Clang from starting, breaking the
  supposedly simple compile command before source compilation began.
- **The JIT cache key now identifies the compiler build that produced each entry.** It was
  (hand-maintained schema string + policy + PTX + kernel name), which describes nothing about how
  a given build lowers PTX — so changing an MSL template, an instruction handler, or a
  legalization rule produced the same key and the runtime silently reused a metallib compiled by
  the previous build. A cache populated across several builds held kernels from different
  compiler versions at once, and it crossed build trees. The key now includes the libcumetal
  Mach-O `LC_UUID`, which the linker regenerates whenever the binary changes.
- **Fixed a race in the specialized `fused_classifier_kernel3` MSL template.** Thread 0 read
  `row_logits[target]` to compute the loss while every thread overwrote `row_logits[]` with
  gradients immediately below, without a barrier. Together with the cache defect above this made
  the llm.c parity gate fail 2-4 runs in 15; measured 0/75 after both fixes.
- **Removed four name-matched body templates that silently miscompiled real kernels.**
  `lower_to_llvm.cpp` replaced a kernel's actual PTX body with a canned implementation whenever
  the entry name contained `vector_add`, `matrix_mul`, `neg`, or `reduce_sum` and the parameters
  had roughly the right shape — before generic lowering was attempted, and bypassing
  `--ptx-strict`. A kernel named `neg_but_actually_triples` computing `x*3` was emitted as
  `fneg`; `neg.s32` returned a float sign-bit flip. The unit and AIR ABI fixtures covering these
  paths had stub bodies but asserted computed ones, so they verified the templates rather than
  the compiler. Found by the new numerical PTX sweep on its first run.
- **The name-matched MSL specialization table no longer pre-empts real translation.**
  `lower_to_metal.cpp` consulted its hardcoded llm.c/GGML name table *before* attempting generic
  PTX→MSL translation, so a kernel whose name merely contained e.g. `gelu_forward_kernel` had its
  real body replaced by a canned one. Same defect as the LLVM-path templates; generic translation
  now wins wherever it succeeds, and the table only covers what it cannot lower. The table is
  additionally disabled unless the caller explicitly opts into workload compatibility.
- **`cumetal_bench` gates on the fastest iteration instead of the mean.** These kernels run in
  ~0.2 ms and are dispatch-jitter dominated (per-iteration spread reaches ~50% when lightly
  loaded), so the
  mean flaked the 2× ceiling under load and the median still reached 2.73× under CPU saturation.
  The fastest iteration estimates the uncontended cost and holds under 8-way saturation. This
  also retired a fictitious published figure — CuMetal was never 26% faster than hand-written
  Metal at `vector_add`; that was outliers inflating the native baseline.
- **Unlowerable kernels are refused rather than emitted empty.** Tolerant mode previously fell
  through to a bare `ret void`, producing a kernel that launched and wrote nothing while the
  caller read back stale buffer contents with no diagnostic.
- **PTX `cvt` rounding modes were ignored on both lowering paths**, silently producing wrong
  answers. In the MSL path the result was typed from the register *name*, so `cvt.rni.f32.f32`
  (clang's `rintf`) emitted an integer truncation — `rintf(2.7)` gave 2 and `rintf(-1.5)` gave 0.
  In the LLVM path the same instruction became an identity copy.
- **libdevice math was a hand-written `if` chain**, so any unlisted function aborted the whole
  kernel. Now table-driven; the surface went from 18/41 to 42/42, verified by executing each
  function against host libm rather than string-matching emitted IR.
- **PTX `.local` stack depots were hardcoded to 256 bytes**, ignoring the declared size. Frames
  larger than that were truncated and out-of-range slots read 0 instead of faulting, so
  `sgemm_2d` silently computed all zeros.
- **PTX→MSL pointer bases were resolved flow-insensitively**, so a register reused for a second
  pointer base retargeted every earlier use to the last base.
- **Four test harnesses reported on stale artifacts.** `run_samples_vector_add`,
  `run_cumetalc_cu_runtime_vector_add`, `run_runtime_vector_add`, and `run_runtime_axpy` each
  checked for a pre-existing binary or metallib *before* checking toolchain availability, so
  once any build produced one they stopped rebuilding and would have stayed green through a
  total compiler regression.
- **`run_standalone_cu.sh` no longer downgrades a wrong answer to a skip.** A kernel that runs
  and computes incorrect results is a failure; only genuinely unavailable lowering skips.
- **llama.cpp coherence.** Registered launches synchronize before returning by default.

### Security

- Approximate and passthru kernel lowerings are **refused by default** rather than silently
  returning wrong results. Opt in with `CUMETAL_ENABLE_APPROX_KERNELS=1`.
- One-time `CUMETAL WARNING` diagnostics for grid-wide cooperative launch (a no-op on Metal) and
  FP64 Dekker emulation (~44-bit mantissa).

### What works

Verified on Apple M4 Pro, macOS 26.6, Xcode 26.6:

- 213 tests pass in Debug with the binary shim on; 210 in Release with it off. No skips, no
  failures. A clean-rebuild Apple M4 Pro re-measurement on 2026-07-27 reports vector_add
  1.063×, saxpy 1.036×, and reduce_f32 1.008× against hand-written Metal, well inside
  the 2× ceiling.
- **llm.c** GPT-2 FP32 training reaches numerical parity with the PyTorch reference.
- **llama.cpp** greedy-decodes coherently on SmolLM2-135M-Instruct-Q4_K_M at ~279 tok/s with
  full offload. This is one model on a covered kernel subset, not general GGML support.
- **PhysX 5.6** reduced GRB runs selected sphere/plane, box/plane, box/box, convex/convex, and
  sphere/triangle-mesh contacts against CPU reference. Selected shape paths, not general PhysX.
- Library shims: cuBLAS, cuBLASLt, cuDNN, cuRAND, cuFFT, cuSPARSE, cuSOLVER, NVML, NCCL, thrust,
  CUB, NVTX.

### Known limitations

Dynamic parallelism, graphics interop, multi-GPU, and device-side texture sampling are
unsupported by design. Grid-wide cooperative sync is a no-op. FP64 runs emulated at ~44-bit
precision. Broad GGML kernel coverage, general PhysX shapes, and arbitrary CUDA C++ are not
claimed. See [docs/known-gaps.md](docs/known-gaps.md) for the full list — it is long, specific,
and worth reading before adopting.
