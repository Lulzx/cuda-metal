# Changelog

All notable changes to CuMetal are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **A float `atomicAdd` on `__shared__` memory silently did nothing on the PTX path.** CuMetal's
  CUDA overlay spells `atomicAdd(float*, float)` as inline `atom.global.add.f32` so the lowering
  selects Metal's native float atomic. The PTX-to-LLVM path already resolved the pointer's real
  state space from `cvta.shared`, but then emitted `atomicrmw fadd float addrspace(3)*`, and Metal
  has no threadgroup float atomic in any language version. `xcrun metallib` accepted the
  instruction and produced a kernel whose add never landed, so a `__shared__` accumulator kept its
  initial value with no diagnostic anywhere -- the registration/JIT path returned 0 for a
  256-thread block reduction. Threadgroup float adds now expand to the same compare-and-swap loop
  MSL spells by hand, over `air.atomic.local.cmpxchg.weak.i32`; other threadgroup float operations
  are refused explicitly rather than miscompiled. Device float atomics still use the native
  instruction. This shipped with the float-atomic lowering while runtime tests could not execute,
  so it landed unverified; `functional_cuda_projects_float_atomics` now covers it.
- **Float atomics on the typed CuMetal IR backend computed garbage.** PTX keeps float temporaries in
  `.b32` registers, so an atomic's payload arrives typed as an integer holding the value's bit
  pattern. The Metal atomic lowering converted it numerically instead of reinterpreting it:
  `atomicAdd(p, 1.0f)` emitted `float(1065353216)`, and the CAS loop clang expands a system-scope
  float add into passed a `float` into a `uint` parameter, so an accumulator ended at 2.8e-45 --
  the float whose bits are 2. Payload and result now bitcast between the storage word and the
  float. This is the same class of defect as the 0.2.1 `.b32` register typing bug, at the atomic
  sites that audit did not reach. `conformance_cuda_projects_typed_ptx_corpus` now passes 28/28.
- **Constant-size aggregate copies between two host-populated device-buffer descriptors failed to
  lower on the source-first path.** `struct A { float4* data; int n; }` passed by value and copied
  element-wise emits `llvm.memcpy` between pointers derived from two byval parameters; the offset
  pointers created while expanding that memcpy were not marked generic, so the address-space
  legalizer rejected them with `host-populated pointer field reaches a conflicting concrete address
  space`. Generic status now propagates to those synthetic pointers. This was the blocker for nearly
  every generated NVIDIA Warp kernel; see `docs/warp-feasibility.md`.
- **`atomicAdd(float*, float)` did not compile on the source-first path.** The CUDA overlay spells
  it as inline `atom.global.add.f32` so the PTX path selects Metal's native float atomic, and the
  `cumetal-ir` backend rejected its own asm; `__fAtomicAdd` (`atomicrmw fadd`) then failed in Metal
  atomic lowering, which accepted only integer payloads. Both spellings, and PTX `atom.add.f32`, now
  lower to `atomic_fetch_add_explicit` on `device atomic_float`; threadgroup float add/sub use a
  bit-pattern compare-and-swap helper because Metal has no threadgroup float atomics. Float min/max
  and CAS remain explicit diagnostics. Every Warp adjoint kernel depends on this.
- **`cuMemcpy3DAsync` silently copied nothing.** Its host-func callback re-entered `cuMemcpy3D` on
  the stream worker thread, which holds no current CUDA context, so the copy failed the context
  check and the error was discarded. Copy operands are now resolved when the copy is enqueued.

### Changed

- The PhysX GRB conformance harnesses and their build scripts honour `CUMETAL_BUILD_DIR`, and ctest
  passes the tree that configured it. They hardcoded `build/`, so running the suite from any other
  build directory failed with `build is not a directory` rather than testing anything. They now also
  skip, rather than error, when CuMetal has not been built -- matching how every other prerequisite
  in those scripts is handled.
- The cuda-samples sweep-status check verifies the README headline only when the README states one.
  The README dropped its per-corpus figures when it was simplified, which left the check demanding a
  number the document no longer carried; the authoritative counts in
  `docs/known-gaps/verification.md` and `docs/verified-results.md` are still enforced
  unconditionally.

### Added

- **`scripts/build_warp_cumetal.sh`**, which clones NVIDIA Warp at `v1.12.0`, applies the two
  upstream changes in `scripts/warp-patches/`, generates the CuMetal CUDA toolkit shim, and
  compiles each of `libwarp`'s 11 `.cu` files through it, reporting per file and failing if one of
  the six that compile today regresses. The changes -- `crt.h` guarded on `WP_CUMETAL`, an
  `__APPLE__` driver `dlopen` branch, `--cuda-path` honoured on Darwin, a CuMetal branch in
  `build_dll.py`, and the `<new>` that `volume_builder.cu`'s placement `new` needs -- belong in
  NVIDIA's repository, so they are carried here as patches against a pinned clone rather than as a
  fork. Previously the only way to reproduce Warp results was a hand-patched local checkout.
- **`cuGetProcAddress`**, resolving against the library's own exported `cu*` symbols, plus
  `cuMemcpy2D`, `cuMemcpy2DAsync`, `cuMemcpyBatchAsync`, `cuEventRecordWithFlags` and
  `cuStreamGetCtx`. These are the driver entry points NVIDIA Warp resolves dynamically; with them
  every Warp entry point outside the OpenGL, IPC, graph-capture and CUDA-array groups is
  reachable. `functional_driver_proc_address` covers the lookup and each new entry point.
- NVIDIA Warp Phase 0 feasibility audit (`docs/warp-feasibility.md`).
- **`cudaTypedefs.h`**, generated from `cuda.h` by `scripts/generate_cuda_typedefs.py`, with the
  versioned `PFN_cu*` function-pointer typedefs hosts that load the driver dynamically declare.
  With it come the graph, stream-capture, IPC and graphics-interop types those signatures need
  (declared for compatibility; the entry points are not implemented), `CUDA_RESOURCE_VIEW_DESC`,
  the `CUfunction_attribute` spelling, and `cudaCpuDeviceId`.
- **`tex1D`**, and linear texture filtering for `float2`/`float4` fetches; the software filter
  previously only instantiated for scalar texel types.
- The `nvcc` shim accepts `-gencode=…` and the `-t`/`--threads` parallel-compilation flags.
- **The driver and runtime API surface NVIDIA Warp's `warp.cu` needs**: the
  `CU_DEVICE_ATTRIBUTE_PCI_*` identity triple, `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` and
  `MEMORY_POOLS_SUPPORTED` attributes; `CU_EVENT_RECORD_*` / `CU_EVENT_WAIT_*`,
  `CU_STREAM_ADD/SET_CAPTURE_DEPENDENCIES`, `CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS` and
  `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE`; `cudaStreamAdd/SetCaptureDependencies`;
  `cudaMemPoolGetAccess` / `cudaMemPoolSetAccess`; `cudaGraphUpload`; `cudaGraphDebugDotPrint`,
  which writes a real Graphviz file; and graph user objects (`cudaUserObjectCreate`,
  `cudaUserObjectRetain`, `cudaUserObjectRelease`, `cudaGraphRetainUserObject`,
  `cudaGraphReleaseUserObject`), whose reference counting ties a host resource's destructor to a
  graph's lifetime. Apple Silicon has no PCI enumeration, so the identity triple reports zeros
  rather than failing; the pool-access calls describe the single device, which always has
  read-write access to its own pool. With these, `warp.cu` compiles against CuMetal -- 6 of
  `libwarp`'s 11 `.cu` files now do, the remaining 5 blocked on CUB device headers.
- **An NVRTC and nvPTXCompiler surface** (`nvrtc.h`, `nvPTXCompiler.h`), exported from
  `libcumetal.dylib` and aliased as `libnvrtc.dylib`. `nvrtcCompileProgram` writes the program and
  any in-memory headers to a temporary directory and runs `cumetalc … --emit metallib`;
  `nvrtcGetCUBIN` returns those bytes, which `cuModuleLoadDataEx` already accepts by their `MTLB`
  magic, so a caller written against NVRTC never learns it is driving a Metal toolchain. NVRTC
  options that describe PTX/SASS code generation are recognised and dropped with a note in the
  program log; include paths, macros and the target architecture map onto `cumetalc` flags.
  `nvrtcGetPTX` fails and `compute_XX` architectures are rejected at compile time, because
  `cumetalc` lowers CUDA source to AIR and never to PTX; `nvPTXCompiler` passes PTX through, since
  the module loader compiles it. This is the surface NVIDIA Warp's `warp.cu` compiles its runtime
  kernels through.

## [0.2.1] - 2026-08-26

### Fixed

- **Float temporaries held in `.b32` registers were typed as unsigned integers, producing silently
  wrong results.** Optimized clang PTX keeps floating-point values in `.b32 %rN` registers, and the
  PTX->MSL emitter typed `neg`, `fma`, `mad`, `abs`, `min`, `max`, `not`, `rcp`, `selp`, `mov`, and
  the unary math intrinsics from the *register spelling* rather than the instruction suffix. A
  kernel computing `c * (out[i] - a * p[i]) + b * q[i]` therefore emitted `uint vr11 = -a;`, which
  truncated every intermediate and clamped negatives to zero -- a wrong answer with no diagnostic.
  The instruction suffix is now authoritative at all of those sites, as it already was for the
  binary operators. This is the same flaw as the 0.1.x `cvt` rounding-mode bug, in the handlers
  that audit had flagged but not reached.

### Added

- **A tiny diffusion-model demo** (`demos/diffusion`). A 312,769-parameter DDPM is trained on MNIST
  in PyTorch, then sampled entirely by hand-written CUDA kernels through CuMetal: 1000 denoising
  steps, 16 images in ~13 s on an M4 Pro. `run.sh --check` gates a forward pass against PyTorch's
  own output at `max |cumetal - pytorch| < 2e-3` (measured 5.2e-06). The demo is what surfaced the
  `.b32` typing bug above.

## [0.2.0] - 2026-08-26

### Added

- **Three runnable demos.** The Apollo demo is the front door; a 3D Gaussian Splatting demo runs
  industry CUDA on Apple GPUs; a 3D SPH dam break demo covers heavy simulation plus rendering.
- **`%lanemask_eq/le/lt/ge/gt` lower to real values,** derived from the simdgroup lane index.
- **Host `malloc`/`free`/`exit` are exposed via `cuda_runtime.h`.**
- **Layered CI groundwork and LLVM compatibility.**

### Fixed

- **`__activemask()` returned zero.** Clang lowers its inline asm to `mov.u32 %r, %activemask`
  rather than the standalone `activemask.b32` opcode, so it arrived as a special-register read and
  fell through to the generic path, which minted an uninitialised register slot. Any PTX special
  register CuMetal does not lower now refuses to lower instead of reading zero.
- **One `.callprototype` declaration made every entry in its module unlowerable,** including
  entries that never make an indirect call. The label is not followed directly by `:`, so the
  parser read it as an opcode.
- **Kernels lowered through the NVVM path shipped without a `.cumetal-abi` sidecar.**
  `cuLaunchKernel` then guessed the argument count by scanning `kernelParams` for a NULL
  terminator that CUDA does not guarantee, reading past the end of the caller's array. The sidecar
  is now derived from the imported IR's kernel ABI, and a launch that still has to fall back to the
  scan warns.
- **A double kernel launch on the source path under FP64 emulation.**
- **The PhysX GRB conformance build no longer configures against PhysX 5.6.1.** Four CMake
  variables were missing, including one that left the snippets directory never linking `PhysXGpu`.

### Changed

- **`cudaResourceType` is declared at namespace scope** with its four constants, matching the CUDA
  Toolkit, so stock sources compile without qualifying every use.
- **The legacy `__pipeline_*` helpers are callable from device code.** They are device primitives
  in CUDA; declaring them host-only made any device use a compile error. The copy stays
  synchronous, which trivially satisfies a later wait.
- **`assert` in `.cu` files no longer expands to nothing.** A `__device__` overload replaces the
  blanket macro no-op, so host-side assertions keep working and only device-side ones are dropped,
  which Metal cannot report anyway.
- **`cudaCreateTextureObject` warns once** when a linear or pitch2D resource is built while
  `CUMETAL_USE_METAL_DEVICE_ADDRESSES` is off. Device code dereferencing the resource pointer reads
  zeros with no error in that mode.

## [0.1.3] - 2026-07-30

### Removed

- **The installed `vectorAdd.cu` copy and its `cumetal doctor` suggestion have been removed.**
  Doctor now reports installation health only and ends at `No issues found!`.

## [0.1.2] - 2026-07-30

### Fixed

- **`cumetalc` no longer prints spurious `+ptxNN` target-feature warnings.** PTX version features
  are now scoped to Clang's CUDA device compilation instead of leaking into the Apple arm64 host
  compilation.

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
