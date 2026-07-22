# PhysX GPU bring-up handoff

Last updated: 2026-07-22

This document is the restart point for CuMetal's PhysX work. The work is paused,
not complete. The canonical design remains `spec.md`; this file records the
empirical state of the selected PhysX 5.6.1 integration.

## Repository state

- CuMetal repository: `/Users/lulzx/work/cumetal`
- Branch: `main`
- Last runtime-support commit: `fbe8d6c` (`physx: verify friction across triangle seams`)
- PhysX checkout: `/Users/lulzx/work/PhysX`
- PhysX base commit: `5ca9f472105a90d70d957c243cb0ef36fe251a9f`
  (`107.3-physx-5.6.1`)
- Reproducible integration: `scripts/physx-patches/0001-*.patch` through
  `0020-cumetal-kernel-init-stubs.patch`

The external PhysX checkout is intentionally dirty because the patch series is
applied directly to it. Do not commit that checkout as the source of truth.
Recreate it from the base revision and the committed patch series when in doubt:

```bash
git -C ../PhysX reset --hard 5ca9f472105a90d70d957c243cb0ef36fe251a9f
scripts/physx-patches/apply_physx_patches.sh ../PhysX
```

The reset command is destructive. Use it only on a disposable PhysX checkout or
after confirming there are no independent local changes.

CuMetal itself has three pre-existing generated-artifact changes that were
deliberately excluded from every PhysX commit:

```text
 M tests/cuda_projects/cuda_samples_vectoradd/build/vectorAdd
 D tests/cuda_projects/flash_attn/build/flash
 D tests/cuda_projects/softmax/build/softmax
```

Preserve them unless their owner explicitly asks otherwise.

## Verified support

The 93-entry selected PhysX GPU manifest builds on macOS arm64 and executes real
Apple GPU metallibs. Verified reduced `SnippetHelloGRB` paths include:

- sphere/plane resting contact;
- selected one-anchor sliding and rolling friction;
- two separated sphere/plane static batches;
- selected two-sphere dynamic stack;
- unit box/plane and selected unit box/box contacts;
- one cooked six-vertex convex-prism pair through GJK/EPA;
- one sphere against a static two-triangle flat mesh;
- coplanar internal triangle-seam handoff; and
- 60-step sphere/mesh friction, including a friction-disabled control.

The triangle-mesh gate runs all six selected stages on Apple GPU:

```text
midphaseGeneratePairs
sphereTrimeshNarrowphase
sortTriangleIndices
convexTrimeshPostProcess
convexTrimeshCorrelate
convexTrimeshFinishContacts
```

The frictionless mesh trajectory starts at `x=-0.5`, crosses the internal seam,
and is byte-identical to CPU for 30 steps. The frictional trajectory stays within
the established `3e-3` envelope for 60 steps and reaches no-slip rolling; the
disabled control retains `vx=5` and zero spin.

These are selected paths, not general PhysX compatibility. Still unsupported or
unverified:

- multiple simultaneous sphere/mesh contact managers;
- non-coplanar mesh seams and general mesh traversal;
- box, convex, capsule, heightfield, and SDF mesh contacts;
- generic mesh temporary-contact and friction correlation;
- general packed batching and larger dynamic stacks;
- joints, articulations, CCD, deformables, cloth, particles, and fluids;
- long-running chaotic-scene determinism; and
- the complete 500+ kernel set and unmodified general PhysX applications.

## Important implementation details

The selected sphere/mesh kernel is intentionally compact because the full
upstream kernel exceeds or crashes the current Apple Metal pipeline compiler.
It computes sphere/triangle contact directly and carries the single separation
float through `SphereTriContacts::index`. This is a scoped workaround: the
generic temporary-contact record is not yet trustworthy across the correlation
dispatch.

PhysX stores `NONCONVEX_FLAG` in triangle adjacency indices. Patch 0018 masks
that flag before indexing the adjacent triangle and uses plane separation when
the projected sphere center lies on a coplanar adjacent face. Omitting the mask
turns the flagged value into an invalid triangle index and reintroduces a
contact gap during seam handoff.

Compiler commit `6344a47` added strict PTX lowering for tuple-source
`mov.b32`/`mov.b64`. It packs evenly sized 8-, 16-, or 32-bit elements using
zero-extension, shifts, and ORs, and rejects malformed tuple widths. Preserve
its positive and negative unit coverage when changing the legacy PTX path.

## Next confirmed blocker

Resume with multiple independent frictionless spheres against the triangle
mesh. The current snippet rejects this mode and must continue to do so until the
failure below is fixed.

The diagnostic scene used two separated spheres for 30 steps:

```bash
SnippetHelloGRB --gpu --sphere --trimesh --frictionless \
  --bodies 2 --steps 30 --dump gpu.tsv
```

Observed result: body 0 receives mesh contact, while body 1 falls immediately
(`py=0.997274995` at step 1 instead of CPU `py=1`). Two useful negative findings
have already ruled out common guesses:

1. Forcing every generated mesh separation to the physically correct `1.0`
   does not restore body 1. Its contact is missing before separation math is
   consumed.
2. Launching midphase, sort, postprocess, correlate, and finish with one SIMD
   group per block does not change the failure. Simple same-threadgroup shared
   scratch aliasing is therefore not the explanation.

Both experiments were reverted. The external PhysX checkout matches patches
0001-0020 aside from generated `PxConfig.h` and insignificant trailing blank
lines in two CUDA files; it contains no remaining diagnostic constants.

The next investigation should trace per-contact-manager identity through:

```text
ConvexMeshPair.cmIndex/startIndex/count
  -> sphereTrimeshNarrowphase pairOffset
  -> convexTrimeshCorrelate outBuffer[pair.cmIndex]
  -> convexTrimeshFinishContacts globalWarpIndex/cmOutputs
```

Prioritize the allocation and pointer-to-buffer identities for
`sphereTriNIGPU`, `sphereTriContactsGPU`, `sphereTriMaxDepthGPU`, and
`gpuIntermSphereMeshPair`. The symptom is consistent with only one contact
manager's manifold surviving, not with an incorrect contact distance.

## Rebuild and validation

Xcode's optional Metal Toolchain component is required. The PhysX build script
auto-discovers an installed component through:

```bash
xcodebuild -showComponent MetalToolchain -json
```

On the machine used for this handoff, the identifier was
`com.apple.dt.toolchain.Metal.32023.883`. General CTest invocations may still
need it exported explicitly even though the PhysX build script discovers it:

```bash
export TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883
```

Fast restart gate:

```bash
CUMETAL_PHYSX_BUILD_JOBS=1 tests/conformance/run_physx_grb_trimesh.sh
```

The external PhysX build tree is shared, so keep PhysX builds serial. The gate
rebuilds the selected integration, checks the unsupported box/mesh negative
path, proves the seam-crossing trajectory, and verifies 30-step frictionless and
60-step frictional CPU/GPU agreement.

Full CuMetal validation:

```bash
cmake --build build --parallel
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  ctest --test-dir build --output-on-failure

cmake -B build-nosshim -DCMAKE_BUILD_TYPE=Debug \
  -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build build-nosshim --parallel
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883 \
  ctest --test-dir build-nosshim --output-on-failure
```

At the pause point, the normal configuration passed 204/204 registered tests
with two expected skips. The binary-shim-disabled configuration passed 192/192
with eleven expected dependency/configuration skips. Patches 0018 and 0019 were
then revalidated through the focused triangle-mesh CTest gate.

## Commit sequence for orientation

```text
651e1a2 compiler: lower structured NVVM control flow correctly
56fc5a7 physx: track structured convex kernel build
812cdc6 compiler: preserve PTX symbol address spaces
07d82f7 physx: verify selected convex mesh contacts
6344a47 compiler: lower PTX mov source tuples
51ca3db physx: add selected sphere triangle mesh path
bb39aa9 physx: preserve sphere contacts across mesh seams
fbe8d6c physx: verify friction across triangle seams
```

When work resumes, update this handoff after each newly verified support tier and
keep `README.md`, `docs/known-gaps.md`, and the patch workflow aligned.
