# PhysX patch workflow

These patches target NVIDIA Omniverse PhysX tag `107.3-physx-5.6.1`,
commit `5ca9f472105a90d70d957c243cb0ef36fe251a9f`.

Apply the patch set to a sibling checkout:

```bash
scripts/physx-patches/apply_physx_patches.sh ../PhysX
```

The application script is idempotent and rejects any other PhysX revision.
The patch reuses PhysX's Unix/Linux CMake source lists while allowing the
compiler to select the existing `PX_OSX`, `PX_APPLE_FAMILY`, and `PX_A64`
source paths. It does not add or enable GPU projects.

The second patch adds an opt-in `PhysXCumetalGpuKernels` target. It invokes
`cumetalc --cuda-device` for the explicit SnippetHelloGRB sphere-plane kernel
manifest without enabling CMake's CUDA language or modifying NVIDIA kernels.

The third patch enables PhysX's existing GPU-facing public declarations for
the opt-in `PX_CUMETAL` Apple ARM64 build and supplies the CuMetal-only CUDA
frontend definitions. Other Apple and upstream CUDA configurations are
unchanged.

The fourth patch expands the bootstrap to the 83 entry points statically
needed by reduced `SnippetHelloGRB` with PGS and sphere/plane geometry. It
deliberately excludes articulations, joints, aggregates, freezing, threshold
reporting, convex/mesh/SDF collision, deformables, particles, and Direct GPU
API-only entry points.

The fifth patch builds and links the GPU host runtime against
`libcumetal.dylib`, loads the source-recompiled per-kernel metallibs through
`CudaKernelWrangler`, minimizes `SnippetHelloGRB`, and uses a body-per-thread
pre-integration path for CuMetal's documented partial-warp-mask limitation.

The sixth patch adds CPU/GPU mode selection, step count, and per-step
transform dumps to the reduced snippet for the conformance gate.

The seventh patch brings up the selected sphere/plane contact path. It adds
CuMetal-safe scalar compaction and static-batch preparation, replaces CUDA-UVA
pointer subtraction with device-buffer offsets, and uses reduced normal-only
contact preparation/solve paths. Friction, joints, articulations, and general
multi-body scenes remain outside this target.

Build and verify the static CPU SDK and non-rendering HelloWorld snippet:

```bash
scripts/physx-patches/build_physx_cpu_macos.sh
```

By default, artifacts are written outside the PhysX checkout under
`build/physx-cpu-macos-arm64`. Set `PHYSX_REPO` or
`CUMETAL_PHYSX_BUILD_DIR` to override either location.

Build the Phase 2 kernel subset with a real Apple metallib:

```bash
scripts/physx-patches/build_physx_cumetal_kernels_macos.sh
```

This defaults to `xcrun` emission. Set
`CUMETAL_PHYSX_EMIT_MODE=experimental` to validate the compiler pipeline on a
machine that has Xcode but has not downloaded the optional Metal Toolchain
component; experimental containers are inspectable test artifacts and are not
GPU-executable.

The build script requires macOS on arm64, CMake, Ninja, and `xcrun`. It
compiles all 83 manifest entries, validates and inspects every output, and
prints a machine-readable `PASS` line.

Build and run the reduced GPU rigid-body snippet end to end:

```bash
scripts/physx-patches/build_physx_cumetal_grb_macos.sh
```

This enables native Metal GPU virtual addresses for CUDA device allocations,
which is required for the nested device pointers in PhysX descriptor structs.
The script verifies successful Apple GPU kernel dispatch and non-zero gravity
integration before printing `PASS`.

Run CPU/GPU transform conformance:

```bash
tests/conformance/run_physx_grb.sh
```

The sphere starts in resting contact with the plane, so the default 30-step
window exercises narrowphase, constraint preparation, the static contact
solver, writeback, and integration. It uses `1e-3` relative plus `1e-5`
absolute tolerance.
