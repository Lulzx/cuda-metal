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

Build and verify the static CPU SDK and non-rendering HelloWorld snippet:

```bash
scripts/physx-patches/build_physx_cpu_macos.sh
```

By default, artifacts are written outside the PhysX checkout under
`build/physx-cpu-macos-arm64`. Set `PHYSX_REPO` or
`CUMETAL_PHYSX_BUILD_DIR` to override either location.

The build script requires macOS on arm64, CMake, Ninja, and `xcrun`. It builds
the Release configuration, verifies that `SnippetHelloWorld` is a native
arm64 executable, runs its stock 100 simulation steps, and prints a
machine-readable `PASS` line.
