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
to the reproducible kernel manifest. Stage 2 compiles from canonical non-inline
NVVM through the typed CuMetal IR backend. Stage 1 remains on the explicit
legacy PTX backend because typed generic-pointer legalization rejects conflicting
address-space flow.

The sixteenth patch adds a cooked six-vertex convex prism to the reduced
snippet and selects typed CuMetal IR only for convex/convex stage 2. The
30-step two-prism frictionless stack exercises both GJK/EPA stages, contact
finalization, dynamic and static preparation/solve, writeback, and integration.
CPU/GPU states stay within a documented 1% component-wise envelope; this is a
selected topology and pair, not a general convex-mesh compatibility claim.

The seventeenth patch adds six sphere/triangle-mesh midphase, narrowphase,
sorting, correlation, and finish entries. Its compact CuMetal path verifies one
frictionless unit sphere moving over the interior of one face in a two-triangle
static ground mesh for 30 steps, with byte-identical CPU/GPU states. The path
carries its one contact separation in the correlation index because the generic
temporary-contact record is not yet coherent across these dispatches. Seam
transitions, multiple bodies, friction, boxes/convexes against meshes, capsules,
heightfields, and SDFs remain outside this claim and are rejected by the snippet.

The eighteenth patch extends that selected flat mesh through a coplanar internal
triangle seam. PhysX stores `NONCONVEX_FLAG` in adjacency indices; the compact
path now masks that flag, recognizes when the projected sphere center lies on
the adjacent coplanar face, and carries plane separation during the cached-face
handoff. The 30-step trajectory starts at `x=-0.5`, crosses the diagonal, and
remains byte-identical to CPU. Non-coplanar seams and general mesh traversal are
still unverified.

The nineteenth patch removes the frictionless-only guard for the selected
one-sphere mesh path. The existing one-anchor friction preparation and static
solver reach no-slip rolling after crossing the seam: CPU/GPU state stays within
the established `3e-3` 60-step envelope, while a friction-disabled GPU control
retains `vx=5` and zero spin. Other shapes, multiple bodies, and generic mesh
friction correlation remain unsupported.

The twentieth patch makes the host runtime patch series self-contained by
adding the `CuMetalKernelInitStubs.cpp` file already referenced by patch 0005.
These empty link anchors replace the init symbols normally emitted by nvcc;
CuMetal loads the source-recompiled kernels by name at runtime.

The twenty-first patch includes PhysX's particle extension host helpers when
the CuMetal GPU runtime is enabled. This closes the native macOS link boundary
for the PBD cloth and inflatable snippets without enabling CUDA as a CMake
language; their device kernels remain explicit CuMetal manifest entries.

The twenty-second patch starts the recreation scenes by turning the PBD cloth
into a compact vertical flag with a pinned edge and time-varying wind. The
modern PhysX PBD and deformable paths retain their required TGS solver.

The twenty-third patch adds the explicit PBD cloth kernel closure observed by
the minimized flag scene, including particle integration, hashing/reordering,
spring projection, self-collision, aerodynamic forces, and finalization.

The twenty-fourth patch keeps particle grid quantization in single precision.
The operands are already `float`; spelling the operation `floorf` avoids an
otherwise needless libdevice double-precision call, which Metal does not
support, without changing the CUDA result type or grid indices.

The twenty-fifth patch gives the headless flag recreation a numerical
conformance result. After 100 frames it copies particle positions back through
the public PhysX CUDA context, rejects non-finite values, verifies that the
pinned edge stayed fixed, and requires measurable motion on the free cloth.

The twenty-sixth patch adds the four TGS scheduling kernels reached by the
minimal PBD flag. Modern PhysX drives PBD particle stepping and finalization
through the TGS solver, so these are required even in a scene with no rigid
obstacles.

The twenty-seventh patch supplies the flag recreation's external acceleration
directly from scene gravity and wind during particle pre-integration. This is a
scoped CuMetal workaround for the still-unverified GPU particle-material
`gravityScale` lookup. It also skips the monolithic rigid-island TGS kernel when
the island contains no rigid bodies; creating its Metal pipeline exceeds the
macOS compiler service's limits and it has no work in this particle-only scene.

The twenty-eighth patch adds the two pressure-volume kernels used by the
inflatable recreation on top of the common PBD particle closure.

The twenty-ninth patch completes three headless recreations. The flag uses
PhysX PBD cloth, springs, and aerodynamics. The inflatable adds a small
source-first CUDA pressure kernel, compiled by `cumetalc`, that acts on the
public PhysX particle buffers. Frog mode uses the same PBD path with a
procedurally generated closed frog-shaped surface. It is a visual/behavioral
recreation, not compatibility with the historical PhysX 2.x sample binary or
modern PhysX FEM deformable-volume kernels.

The thirtieth patch adds opt-in JSON frame capture to the flag snippet so its
real GPU particle positions can be rendered without changing the default
headless conformance run.

The thirty-first patch adds the cloth drape scene, selected with
`CUMETAL_PHYSX_DRAPE=1`. A free 31x33 sheet falls onto a sphere and settles.
Particle/rigid-body narrowphase is not part of the CuMetal PBD kernel closure —
a sheet dropped onto a `PxRigidStatic` sphere passes straight through it — so
the obstacle and ground are supplied by `cumetalClothCollide`, a source-first
CUDA kernel compiled by `cumetalc`, in the same way the inflatable supplies its
pressure. PhysX still owns integration, self-collision, and aerodynamics. The
patch also exports the surface topology and grid dimensions in the capture so
the recreation can be rendered as a shaded surface rather than a point cloud,
and it holds both cloth scenes under 1024 particles (see the known limits
below). The 240-step gate requires real contact, no penetration of the sphere,
wrap past its equator, and a settled mean velocity.

## Known limits in the CuMetal PBD path

These are reproducible with the snippets in this series and are not fixed here:

- **PBD spring constraints have no effect.** Sweeping the cloth stretch
  stiffness from `10` to `1000000`
  (`CUMETAL_PHYSX_CLOTH_STRETCH`/`CUMETAL_PHYSX_CLOTH_SHEAR`) produces
  bit-identical particle state. `ps_solveSpringsLaunch` dispatches and reports
  success, but its result never reaches the particles, so the cloth behaves as
  free particles with self-collision and aerodynamics only. Over a long window
  the flag stretches to many times its rest length instead of holding together.
- **Particle systems of 1024 or more particles silently stop integrating.**
  At 1023 particles the flag moves normally; at 1024 every particle stays at
  its initial position with zero velocity. The kernel sequence, launch counts,
  and grid/block dimensions are identical on both sides of the boundary, so the
  divergence is inside a kernel and data dependent. Both cloth scenes are sized
  under this ceiling.
- **`CUMETAL_USE_METAL_DEVICE_ADDRESSES=1` costs cross-stream concurrency.**
  PhysX reaches most of its solver state through raw device addresses embedded
  in descriptor structs, which are never bound with `setBuffer`, so that mode
  marks every live allocation read-write on every dispatch to keep it resident.
  Metal then treats unrelated work on different streams as dependent and
  serializes it. The runtime prints a one-time warning when the mode is active.
- **Velocity written to a particle buffer between steps is discarded unless
  the buffer is re-flagged.** `PxParticleBuffer::raiseFlags(eUPDATE_VELOCITY)`
  is required after a kernel writes `getVelocities()`; without it the next
  `simulate()` re-uploads the stale contents. The drape's collision kernel
  raises it. `cumetalInflatablePressure` does not, so the inflatable and frog
  scenes run with that kernel dispatching successfully but not affecting the
  simulation — the conformance gate only asserts that it launched. Raising the
  flag there makes those two scenes unstable, because the inflatable surface
  depends on the spring network above.

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

The build script requires macOS on arm64, CMake, Ninja, `xcrun`, and Xcode's
optional Metal Toolchain component (which it discovers automatically). It
compiles all 93 manifest entries, validates and inspects every output, and
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
frictionless spheres plus frictionless boxes and convex prisms):

```bash
tests/conformance/run_physx_grb_stacked.sh
```

Run the selected four-point box/plane contact gate:

```bash
tests/conformance/run_physx_grb_box.sh
```

Run the selected sphere/static-triangle-mesh gate and unsupported-shape
negative control:

```bash
tests/conformance/run_physx_grb_trimesh.sh
```

Build and run the flag, inflatable, and frog PBD recreations:

```bash
tests/conformance/run_physx_pbd_recreations.sh
```

The gate checks numerical scene invariants and verifies successful Apple GPU
dispatch of the aerodynamic, collision, and source-recompiled pressure kernels.
Each scene can also be captured to JSON for offline rendering; the renders
themselves are not kept in this repository.
