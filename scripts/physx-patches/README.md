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
`CudaKernelWrangler`, minimizes `SnippetHelloGRB`, and introduced the original
body-per-thread pre-integration compatibility path.

The sixth patch adds CPU/GPU mode selection, step count, and per-step
transform dumps to the reduced snippet for the conformance gate.

The seventh patch brings up the selected sphere/plane contact path. It adds
CuMetal-safe scalar compaction and static-batch preparation, replaces CUDA-UVA
pointer subtraction with device-buffer offsets, and uses reduced normal-only
contact preparation/solve paths. Friction, joints, articulations, and general
multi-body scenes remain outside this target.

The eighth patch removes the serialized `updateBodiesLaunch` and body-per-thread
pre-integration fallbacks. CuMetal's masked vote, shuffle, SIMD-group barrier,
and entry-specific static shared-memory paths now execute PhysX's upstream
warp-cooperative implementations through repeated 30-step conformance runs.

The ninth patch enables the selected sphere/plane kinetic-friction path. It
builds one friction anchor without expanding the unsupported generic patch
cache, restores the real contact solver's friction loop, and adds friction and
friction-disabled snippet modes with linear/angular velocity dumps. The
60-step gate matches CPU through the initial sliding phase and verifies a
material GPU friction response against the disabled control. Persistent static
friction and long-horizon rolling conformance remain out of scope.

The tenth patch closes the selected sphere/plane rolling-friction gap. It
clears the one-body accumulated solver deltas at each simulation step, stages
the bounded previous friction patch without the unsupported generic device
pointer traversal, and verifies CPU/GPU rolling agreement at step 60. Generic
friction correlation and multi-body batching remain out of scope.

The eleventh patch adds selected multibody rigid/static coverage. It schedules
each contact pre-prep and prepare batch in a dedicated 32-lane Metal SIMD
group, indexes the reduced static solver and delta reset across island bodies,
and adds `--bodies 1..16` to the snippet. The conformance claim covers two
separated dynamic spheres against one plane; dynamic/dynamic constraints and
packed general batching remain out of scope.

The twelfth patch adds selected dynamic/dynamic contact batching. It replaces
shared device-pointer staging in the zero and motion-writeback kernels with
direct Metal-safe indexing, runs the prepared rigid-contact block solver, and
serially aggregates and propagates each body's slab contributions. The snippet
adds a two-sphere `--stacked` layout. CPU and GPU agree for 30 frictional and
frictionless steps; larger stacks, joints, articulations, and packed general
batching remain out of scope.

The thirteenth patch adds the convex/plane narrowphase entry and a selectable
unit box to the reduced snippet. CuMetal's compiler now lays out only the
selected entry's aligned static shared objects, so the convex kernel's contact
scratch no longer starts beyond its allocated Metal threadgroup buffer. The
30-step frictionless box/plane gate preserves four distinct corner contacts
and matches CPU transforms. General convex meshes and other convex pair types
remain outside this claim.

The fourteenth patch adds PhysX's box/box narrowphase entry. CuMetal's CUDA
frontend forces all viable device calls to inline when the project requests an
inline threshold, eliminating the remaining `getIncidentPolygon4` PTX call.
The selected two-unit-box stack stays supported and matches CPU body states
over 30 frictionless steps. General oriented-box stress cases and larger box
stacks remain unverified.

The fifteenth patch adds the upstream two-stage convex/convex GJK/EPA entries
to the reproducible kernel manifest. Both canonical non-inline NVVM entries
compile to validated metallibs through the typed CuMetal IR backend. This is a
compiler and kernel-build claim only: no committed general convex-mesh scene
gate exists yet, so general convex/convex runtime support remains unclaimed.

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
compiles all 87 manifest entries, validates and inspects every output, and
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

Run the selected sliding-to-rolling friction gate:

```bash
tests/conformance/run_physx_grb_friction.sh
```

Run the selected two-body rigid/static batching gate:

```bash
tests/conformance/run_physx_grb_multibody.sh
```

Run the selected stacked dynamic/dynamic contact gate (frictional and
frictionless spheres plus frictionless boxes):

```bash
tests/conformance/run_physx_grb_stacked.sh
```

Run the selected four-point box/plane contact gate:

```bash
tests/conformance/run_physx_grb_box.sh
```
