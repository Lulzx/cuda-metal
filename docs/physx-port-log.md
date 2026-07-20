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
