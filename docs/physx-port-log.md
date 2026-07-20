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

Phase 2 has not started.
