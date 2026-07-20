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
