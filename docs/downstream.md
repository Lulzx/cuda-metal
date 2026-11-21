# Downstream users

Projects that build on CuMetal, what they exercise, and the CuMetal version they
pin or evaluated. A pinned version is the one the project reports using, not a
version this repository has re-verified. Before moving a pin forward, check the
issues linked below and [known gaps](known-gaps.md).

| Project | What it exercises | Pinned / evaluated | Tracking |
|---|---|---|---|
| [starknet-innovation/stwo-zig](https://github.com/starknet-innovation/stwo-zig) | `cumetalc` source path with `--ptx-strict` as a macOS correctness check for its CUDA backend; funnel shifts, `__match_any_sync`, 64-bit atomics, CUB | v0.1.3, v0.6.0, with local patches and a compat header | #221 |
| [jinzhezenggroup/generativeqc](https://github.com/jinzhezenggroup/generativeqc) | Quantum-chemistry SCF/DFT kernels on the typed PTX backend (`CUMETAL_PTX_BACKEND=cumetal-ir`, `CUMETAL_FP64_MODE=fast48`); very large translation units | `e87f368` (v0.6.0), 14 endpoint tests quarantined | #218, #220 |
| [lalomorales22/CUDA-M](https://github.com/lalomorales22/CUDA-M) | Cosmos-Predict2 tiled attention (softmax kernels plus the cuBLAS shim); Xcode 26/27 Metal toolchain setup | v0.6.0 | #217, #228 |
| [DeadlyLogically/Cuda-On-Metal-Lab](https://github.com/DeadlyLogically/Cuda-On-Metal-Lab) | llm.c GPT-2 training and NVIDIA Warp, via raw PTX through the driver API; overlays several CuMetal source files | `9126942` plus overlay | #222, #228 |
| [dasomel/kubemetal](https://github.com/dasomel/kubemetal) | Feasibility study: NVML, library-shim provenance, cuFFT determinism, toolchain detection | v0.7.1 | #214, #215, #216, #217 |
| [dimensionalOS/dimos](https://github.com/dimensionalOS/dimos), dimSLAM / cuVSLAM macOS packaging | Pointer-heavy code that needs `CUMETAL_USE_METAL_DEVICE_ADDRESSES=1`; wheels and tarballs that bundle `libcumetal.dylib` with a prewarmed cache | not stated | #224, [bundled cache](#bundled-kernel-cache) |
| [tsotchke/tensorcore](https://github.com/tsotchke/tensorcore) | Evaluation of failure paths and performance (copy kernels, half conversions, SGEMM) | Homebrew v0.1.3 | #223, #228 |
| [brandonros/vanity-miner-rs](https://github.com/brandonros/vanity-miner-rs) | Rust-CUDA PTX (LLVM 7 and 21 NVVM) for P-256 and RSA; very large generated MSL | September 2026 builds | #219, #220, #76 |
| [Glavin001/rubble](https://github.com/Glavin001/rubble) | PhysX GPU harness | not stated | #224, #225 |

Reduced kernels from these projects land as regression tests together with the
fix they cover (see each tracking issue). Re-measure a project at a new CuMetal
version with its own test suite; this table does not claim a project passes at
`main`.

## Detecting CuMetal

Do not infer CuMetal from `CUMETAL_ROOT`, the platform, or a missing
`nvidia-smi`. Use one of these instead.

### In source: `__CUMETAL__`

Every translation unit built against CuMetal's headers sees:

| Macro | Value |
|---|---|
| `__CUMETAL__` | `1` |
| `__CUMETAL_VERSION__` | `major * 10000 + minor * 100 + patch` (0.8.0 is `800`) |

They come from `cuda.h`, which `cuda_runtime.h` includes and `cumetalc`
force-includes on both the host and the device side, so they are visible in
`.cu` files without an explicit include, and in host code that includes either
header. NVIDIA's toolkit never defines them.

```cpp
#if defined(__CUMETAL__) && __CUMETAL_VERSION__ >= 800
// CuMetal 0.8.0 or newer
#endif
```

`cumetal_native.h` carries the same version as `CUMETAL_VERSION_MAJOR`,
`_MINOR`, `_PATCH` and `CUMETAL_VERSION`. `unit_version_matches` fails the
test suite if any of these drift from the CMake project version.

### From a build script: `cumetal version --json`

```console
$ cumetal version --json
{
  "name": "cumetal",
  "version": "0.8.0",
  "version_number": 800,
  "prefix": "/opt/homebrew/Cellar/cumetal/0.8.0",
  ...
  "bundled_cache_dir": null,
  "macros": {"__CUMETAL__": 1, "__CUMETAL_VERSION__": 800},
  "features": {
    "source_compiler": true,
    "runtime": true,
    "jit": true,
    "aot_metal_toolchain": true,
    "air_version": "2.9",
    "binary_shim": false,
    "compile_timing": true,
    "msl_size_limit": true
  }
}
```

Every feature flag is probed on the machine, not hard-coded:
`aot_metal_toolchain` runs the same toolchain probe as `cumetal doctor`
(compile and package a `-std=metal3.2` kernel, check the AIR version and the
deployment target). `cumetal doctor` exits 0 when ready, 3 when only runtime
JIT works, and 1 when CuMetal is not usable, so a CI lane can tell a missing
Metal toolchain from a broken install. Fields may be added in later versions;
existing fields keep their meaning.

## Bundled kernel cache

CuMetal caches compiled kernels under one root, resolved in this order:

1. `$CUMETAL_CACHE_DIR`, when set and non-empty.
2. `share/cumetal-cache` next to the loaded `libcumetal.dylib`
   (`<dir of libcumetal.dylib>/../share/cumetal-cache`), when that directory
   exists and is writable.
3. `~/Library/Caches/io.cumetal`, or `io.cumetal` in the temporary directory
   when `HOME` is unset.

Step 2 lets a wheel or tarball ship `lib/libcumetal.dylib` together with a
prewarmed `share/cumetal-cache` (dimSLAM and the cuVSLAM macOS packaging do
this). Registration-JIT entries are keyed to the dylib's `LC_UUID`, so entries
written by the bundled dylib hit in every installed copy of the same bundle and
are ignored by any other build. To prewarm, run the workload once with the
bundled dylib and no `CUMETAL_CACHE_DIR`, then package the directory.

A read-only bundled cache (for example inside a signed app bundle) is skipped
with a one-time warning, because CuMetal must be able to publish new entries;
set `CUMETAL_CACHE_DIR` to a writable copy instead. `functional_bundled_cache`
tests all three cases with a relocated dylib.

The subdirectories are `registration-jit/` (PTX JIT), `native-aot/` (embedded
`cumetalc` metallibs), `library-kernels/` (library shim kernels) and `kernels/`
(module images). For historical reasons `$CUMETAL_CACHE_DIR` itself, not
`$CUMETAL_CACHE_DIR/kernels`, holds module images when the variable is set.
