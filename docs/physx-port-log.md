# PhysX port log

This log records progress and decisions for the PhysX GPU-on-CuMetal port.

## 2026-07-20 — Phase 0 feasibility audit

- Confirmed the working CuMetal repository is `/Users/lulzx/work/cumetal`;
  references to `./cuda-metal` in the port brief map to this checkout.
- Read `README.md`, `spec.md`, `docs/status.md`, and
  `docs/known-gaps.md` before making port decisions.
- Preserved three pre-existing deleted generated test binaries in the CuMetal
  working tree; they are unrelated to this work. The regression run rebuilt
  `cuda_samples_vectoradd/build/vectorAdd`, so that path now appears modified
  while the other two remain deleted.
- Cloned NVIDIA Omniverse PhysX into `../PhysX`, disabled Git LFS smudging for
  large sample assets, and pinned tag `107.3-physx-5.6.1` at commit
  `5ca9f472105a90d70d957c243cb0ef36fe251a9f`.
- Surveyed 61 CUDA translation units, 509 global-kernel declarations/textual
  occurrences, and the 501-name `PxgKernelNames.h` registry.
- Found no dynamic parallelism, cooperative groups, grid-wide sync,
  cooperative launch, active CUB/Thrust, active FP16, or callable device
  function pointers.
- Found the main correctness risk: extensive warp operations and active
  partial-mask behavior in rigid pipeline and solver code. CuMetal's current
  full-group mask emulation is not sufficient evidence of correctness.
- Found blockers outside the first target: SDF collision performs three
  active `tex3D` samples, and arbitrary device FP64 remains unsuitable for
  BVH/FEM paths.
- Mapped PhysX's module flow: nvcc static registration is captured by PhysX,
  fatbinary pointers are passed to `cuModuleLoadDataEx`, 501 functions are
  eagerly resolved with `cuModuleGetFunction`, and launches use
  `cuLaunchKernel`.
- Chose source-first AOT integration: compile a target-specific kernel subset
  with `cumetalc`, generate a metallib/function manifest, and make kernel
  lookup lazy or target-specific. The binary-shim JIT is not the preferred
  path.
- Selected a minimized non-rendering `SnippetHelloGRB`: one dynamic sphere,
  one static plane, GPU broadphase, PCM, and GPU dynamics. This exercises an
  honest rigid contact/solver path while excluding box stacks, joints,
  articulations, meshes, SDF, particles, cloth, and deformables.
- Static inspection found no core rigid solver dependence on an in-kernel
  cross-threadgroup barrier, so there is no architectural stop before Phase
  1.
- The first regression attempt exposed stale CTest registrations and stale
  binaries in the existing build tree. Reconfigured and rebuilt with
  `cmake -B build -DCMAKE_BUILD_TYPE=Debug` and `cmake --build build`, then
  reran the required full command:
  `ctest --test-dir build --output-on-failure`. All 186 current tests passed;
  platform/dependency-gated cases were reported as skips.
- Full report: [physx-feasibility.md](physx-feasibility.md).

Next action is intentionally paused pending author review of the Phase 0
report.

## 2026-07-20 — Phase 1 CPU SDK on macOS arm64

- Reused PhysX's Unix/Linux CMake source lists for the CPU-only build while
  retaining the compiler-selected `PX_OSX`, `PX_APPLE_FAMILY`, and `PX_A64`
  code paths. No CUDA language or GPU target is enabled.
- Added a revision-pinned, idempotent patch workflow under
  `scripts/physx-patches/`. The patch applies cleanly to
  `107.3-physx-5.6.1` at
  `5ca9f472105a90d70d957c243cb0ef36fe251a9f`.
- Kept the port narrow:
  - recognize CMake's `AppleClang` compiler ID and libc++ rather than applying
    libstdc++ ABI flags;
  - remove Linux-only `rt`, `dl`, `-m64`, `$ORIGIN`, X11, GL, and GLUT
    assumptions for a static, non-rendering macOS build;
  - use the scalar vector store path on every ARM-family target;
  - avoid x86 MXCSR intrinsics in macOS arm64 FPU guards;
  - resolve an AppleClang `size_t` overload ambiguity in the temporary
    allocator;
  - add a distinct `MA64` / `macaarch64` binary serialization platform tag.
- Added `build_physx_cpu_macos.sh`, which writes all generated build artifacts
  under CuMetal's `build/physx-cpu-macos-arm64` directory rather than
  polluting the PhysX checkout.
- Built the static Release CPU SDK dependencies and stock non-rendering
  `SnippetHelloWorld`. The result is a single-architecture Mach-O arm64
  executable.
- Ran the snippet's stock 100 fixed simulation steps. It completed with
  `SnippetHelloWorld done.` and the wrapper reported:
  `PASS: PhysX CPU SnippetHelloWorld completed 100 steps on macOS arm64`.
- The upstream build emits non-fatal AppleClang warnings and a final duplicate
  static-library linker warning; neither affects the successful build or
  simulation.
- Ran `ctest --test-dir build --output-on-failure` before commit. All 186
  current CuMetal tests passed; platform/dependency-gated cases were reported
  as skips.

## 2026-07-20 — Phase 2 CUDA source compilation

- Audited `cumetalc` before integrating PhysX and found that its original
  `.cu` path strips CUDA qualifiers and compiles host C++ LLVM IR. Preserved
  that prototype for compatibility and added an explicit `--cuda-device`
  path using CUDA-capable Homebrew Clang, device-only PTX emission, and the
  existing PTX-to-Metal pipeline.
- Added project-build arguments for include directories, preprocessor
  definitions, forced includes, CUDA architecture, and compiler override.
  The frontend defines the CUDA compiler identity macros used by the existing
  CuMetal source builds and disables PTX jump tables because `brx.idx` target
  tables are not yet lowered.
- Added clean-room `vector_types.h` and `vector_functions.h` compatibility
  headers. PhysX includes both directly; CuMetal already supplied their vector
  ABI types and `make_*` constructors through `cuda_runtime.h`.
- Fixed two PTX gaps exposed by the unmodified PhysX source:
  - scalarize both load and store forms of `.v2` and `.v4` memory operations;
  - lower the CUDA libdevice `__nv_sqrtf` call to the Metal sqrt intrinsic.
  Each fix has focused unit or AIR ABI regression coverage.
- Compiled the unmodified
  `gpunarrowphase/src/CUDA/cudaSphere.cu` translation unit through
  `cumetalc --cuda-device --ptx-strict`. Clang emitted a single
  `sphereNphase_Kernel` PTX entry (1,636 PTX lines in the initial audit run),
  and CuMetal produced a validated one-kernel metallib container.
- Added the revision-pinned `0002-cumetal-gpu-subset.patch`. Its opt-in
  `PhysXCumetalGpuKernels` CMake target does not enable nvcc/CMake CUDA; it
  invokes `cumetalc` for the explicit minimized-HelloGRB manifest and emits
  `kernels.json`.
- Added `build_physx_cumetal_kernels_macos.sh`. Production output defaults to
  `xcrun`; `CUMETAL_PHYSX_EMIT_MODE=experimental` is available only for
  compile/validation on machines without the optional Xcode Metal Toolchain.
- Verified the initial patched PhysX CMake target in experimental mode on this
  host. `air_validate` and `air_inspect` confirmed the bootstrap kernel named
  `sphereNphase_Kernel`. This host's `xcrun` is installed, but its optional
  Metal Toolchain component is not, so a GPU-executable production metallib
  could not be packaged locally. This is an environment dependency, not a
  CUDA/PTX compilation failure.

### Phase 2 completion sweep

- Mapped the reduced scene's broadphase, rigid simulation-controller,
  sphere/plane narrowphase, contact bookkeeping, and PGS solver dispatches to
  14 CUDA translation units.
- Added the clean-room CUDA declarations required by those sources:
  driver-compatible opaque stream/event spellings, `CUtexObject`,
  `__builtin_align__`, cache-hinted load/store shims, 64-bit/double shuffle
  overloads, and `sm_35_intrinsics.h`.
- Fixed PTX lowering exposed by the sweep:
  qualified shared-memory address-space operations; generic pointer loads and
  stores; destination-less calls; scalar bit reinterpretation; popcount;
  `fminf`; fast division; and fast sin/cos with local pointer outputs.
  Focused unit or AIR frontend regressions cover each compatibility class.
- Added `--cuda-inline-threshold`. Clang's GPU inliner removes PhysX's
  `updateCacheAndBound` helper `.func` without modifying NVIDIA source;
  standalone PTX `.func` lowering remains explicitly documented as a gap.
- Strict-lowered all 127 entries emitted by the selected translation units:
  122 lower directly. Four convex/joint entries remain outside the chosen
  scene, and the transform entry lowers when compiled with the new inlining
  option.
- Replaced the one-kernel bootstrap manifest with 83 entry points for reduced
  `SnippetHelloGRB` (PGS, sphere/plane). The manifest excludes articulations,
  joints, aggregates, freezing, threshold reporting, convex/mesh/SDF paths,
  deformables, particles, and Direct GPU API-only kernels.
- Compiled and individually validated/inspected all 83 entries using
  `CUMETAL_PHYSX_EMIT_MODE=experimental`; the build wrapper reports
  `PASS: 83 PhysX CuMetal sphere-plane PGS kernels compiled (experimental)`.
  Production `xcrun` packaging still requires downloading this host's optional
  Xcode Metal Toolchain component.
- Phase 2 only establishes compiler coverage and artifacts. Wiring these
  metallibs into `KernelWrangler`/Driver API module lookup belongs to Phase 3.
- Ran the required pre-commit gate:
  `ctest --test-dir build --output-on-failure`. All 187 registered tests
  passed; platform/toolchain-gated tests reported as skips.

## 2026-07-20 — Phase 3 runtime bring-up

- Built the PhysX GPU host implementation as ordinary arm64 C++ and linked a
  static `SnippetHelloGRB` against `libcumetal.dylib`. The SDK-wide
  `PX_CUMETAL` marker is required so `PX_SUPPORT_GPU_PHYSX` remains enabled on
  Apple arm64; without it PhysX accepts the scene flags but silently builds and
  runs the CPU implementation.
- Replaced nvcc-only static-link anchors with CuMetal no-op anchors and taught
  `KernelWrangler` to load source-recompiled kernels directly from
  `CUMETAL_PHYSX_KERNEL_DIR/<kernel>.metallib`. Missing artifacts are left
  unresolved so the reduced manifest does not need all 501 registered kernels.
- Reduced `SnippetHelloGRB` to one dynamic sphere and one static plane, removed
  PVD/network setup, and made GPU context/scene initialization failures fatal
  to the snippet instead of silently accepting CPU fallback.
- Added the clean-room host-side CUDA qualifier/constant/texture ABI surface
  needed to compile PhysX without nvcc. Texture object/array creation is an
  explicit `CUDA_ERROR_NOT_SUPPORTED` path because Metal texture sampling is
  still not wired.
- Runtime tracing found that CUDA Driver API `kernelParams` arrays were being
  scanned for a null sentinel. CUDA specifies an exact array with one slot per
  parameter and no sentinel; PhysX therefore exposed an out-of-bounds read.
  Experimental CuMetal containers now carry `kernel.arg_count` compiler
  metadata, `CUfunction` consumes it, and a protected-page regression proves
  that a non-terminated argument array is not scanned.
- Installed Apple's optional Metal Toolchain and compiled all 83 selected
  kernels as production metallibs. AIR reflection does not preserve CUDA's
  source argument grouping, so `cumetalc` now writes an exact
  `.cumetal-abi` sidecar and the Driver API consumes it when loading a module.
- Made the current CUDA context and context stack thread-local. PhysX performs
  GPU work from worker threads; the former process-global current context
  violated Driver API semantics and intermittently rejected valid launches.
- Found the central address-model mismatch after launches became stable.
  PhysX descriptor structs contain nested device pointers. A CPU mapping from
  `[MTLBuffer contents]` is not a GPU virtual address and cannot be
  dereferenced by Metal. The opt-in
  `CUMETAL_USE_METAL_DEVICE_ADDRESSES=1` mode returns public
  `MTLBuffer.gpuAddress` values for device allocations while CUDA memcpy and
  memset APIs translate them to the shared CPU mapping. A focused functional
  kernel now tests a nested pointer plus scalar argument end to end.
- The audited partial-warp-mask limitation affected `preIntegration`: its
  warp-swizzled body loads depend on a body-count-specific active mask.
  Patch 0005 uses an equivalent body-per-thread implementation under
  `PX_CUMETAL`; the upstream CUDA path is unchanged.
- Verified real Apple GPU execution with provenance tracing. The reduced
  scene runs 100 steps without a crash, launches solver and integration
  kernels from production metallibs, moves the sphere from `y=10` to
  `y=-3.76124477`, and reports `vy=-16.3499889`.
- Initial runtime bring-up established stable free-fall before contact. The
  subsequent Phase 4 hardening below replaces that provisional limitation
  with a selected resting-contact path.

## 2026-07-20 — Phase 4 conformance gate

- Added dual-mode CLI controls to the same reduced `SnippetHelloGRB` binary:
  `--cpu` or `--gpu`, `--steps N`, and `--dump FILE`. The dump contains the
  sphere position and quaternion for the initial state and every simulated
  step.
- Added `tests/conformance/run_physx_grb.sh` and a strict TSV comparator.
  The hardened default gate runs 30 resting-contact steps and uses `1e-3`
  relative plus `1e-5` absolute tolerance. It requires Apple-GPU provenance
  for narrowphase, constraint pre-prep/prep, static solve, writeback, and
  integration, preventing a CPU fallback or pre-contact-only path from passing.
- On Apple M4 Pro the hardened resting-contact CPU/GPU dumps remain within
  about `1.2e-7`; both finish at height `1.0` with zero velocity.
- Contact hardening fixed interior mapped-pointer aliases, CUDA-UVA offset
  arithmetic, lost/found compaction, static batch construction, contact
  pre-prep/prep, a reduced normal solver, and contact-aware integration. The
  30-step resting scene now differs from CPU by at most about `1.2e-7` in the
  measured transform. General frictional and multi-body scenes remain outside
  the selected target and may diverge chaotically over long runs.

## 2026-07-21 — Resting-contact hardening

- Made `cudaHostGetDevicePointer` accept interior pointers and preserve their
  byte offsets; PhysX uses interior addresses from pinned allocator slabs.
  Native Metal device-address allocations now register aliases without
  double-counting allocation bytes, with runtime/driver and nested-pointer
  regressions.
- Added patch 0007 for the explicitly reduced target. CUDA partial-warp scan
  stages are serialized, pointer-heavy descriptors are read from global
  memory, and CUDA-UVA pointer differences use device-buffer bases.
- Selected normal-only preparation and solve paths avoid partial-warp friction
  barriers and the generic solver metallib that Apple's pipeline compiler
  rejects. NVIDIA CUDA branches are unchanged.
