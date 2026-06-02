# AIR / metallib ABI Notes (Phase 0.5)

Status: draft with concrete reference bytes from:

- `tests/air_abi/reference/reference.metallib` (single kernel)
- `tests/air_abi/reference/multi_kernel.metal` compiled via `xcrun metal` + `xcrun metallib`

## Scope

This document records container-level findings for `.metallib` validation and CuMetal's
Phase 0.5 harness. The focus is structural acceptance, not kernel execution.

## Tooling in this repo

- `air_inspect`: structural dump with function-level output when available:
  - CuMetal experimental container v2 (`MTLB` + explicit function table)
  - Apple metallib function list tags (`NAME`, `MDSZ`, `OFFT`, `VERS`, `TYPE`, `HASH`) when parseable
- `cumetal-air-emitter`: emits a `.metallib` using one of:
  - `xcrun` mode: packages `.air` payloads with Apple tools (or `.metal` via `xcrun metal`).
  - `experimental` mode: CuMetal-owned provisional container format for local iteration.
- `air_validate`: validates container shape, function list, required metadata fields, and bitcode
  signatures. Optional checks:
  - `xcrun metal -validate`
  - `llvm-dis` bitcode verification (if installed)
  - MetalLibraryArchive bridge command (`tools/metal_library_archive_bridge`)
- `cumetal_metal_load_test`: checks `MTLDevice.newLibraryWithData:` acceptance.

## Reference layout snapshot

Reference artifact (`tests/air_abi/reference/reference.metallib`) from xcrun (or emitter fallback):

`air_inspect` summary (current):

- Size: `3760` bytes (`0xeb0`)
- Magic: `MTLB`
- Function list parser: `metallib-function-list`
- Function count: `1` (`vector_add`)
- Kernel bitcode: offset `0xf0` size `0xdc0`
- Observed: air.version=2.8, language.version=4.0 (references built against older xcrun; runtime accepts)

Header fields used by `parse_real_metallib` (compiler/common/src/metallib.cpp):

| File offset | Type | Meaning | Value (reference) |
|---|---|---|---|
| `0x18` | `u64` | function-list offset | `0x58` |
| `0x20` | `u64` | function-list size | `0x80` |
| `0x48` | `u64` | bitcode-section offset | `0xf0` |
| `0x50` | `u64` | bitcode-section size | `0xdc0` |

The parser accepts two size interpretations for function-list end ( +0 or +4 for entry_count variance).

## Function record tags

Tags parsed (NAME, TYPE, HASH, MDSZ, OFFT, VERS, ENDT etc.):

| Tag | Meaning | Example |
|---|---|---|
| `NAME` | function symbol | `vector_add` |
| `TYPE` | function kind | `2` (kernel) |
| `HASH` | digest | prefix e64ad8cd... |
| `MDSZ` | bitcode size | `0xdc0` |
| `OFFT` | offsets (pub/priv/bitcode) | `0/0/0` (single); non-zero for later kernels in multi |
| `VERS` | AIR/language (`u16` pairs) | AIR `2.8`, language `4.0` |
| `ENDT` | terminator | present |

## Multi-kernel / bench layout (fresh from build/bench_phase5/bench_kernels.metallib)

3 kernels (vector_add, saxpy, reduce_f32):

- Function count: 3
- Varying OFFT for kernels >0 (e.g. saxpy: public=8,private=8,bitcode=3520; reduce: 16/16/7040)
- All report air.version=2.8 , language 4.0 , type=2 kernel
- Bitcode sizes vary (0xdc0, 0xdc0, 0xf70)
- Confirms parser handles multiple entries + relative bitcode offsets correctly.

Practical notes:

1. Entry order matches declaration.
2. OFFT public/private often 0 for first kernel, positive for subsequent (sub-buffers?).
3. String table has kernel names + long mangled "air64_v28-..." build ids.
4. Current emitter + xcrun paths produce compatible layouts for runtime load/launch.

## Parser implementation notes (compiler/common/src/metallib.cpp + air_inspect)

- Supports MTLB Apple function-list + CuMetal experimental v2 containers.
- Bitcode sig checks: raw BC0C or wrapped.
- Used by air_emitter validation, air_inspect, and runtime module loading for registration/JIT path.
- Limitations tracked in known-gaps.md (experimental mode not production ABI).

Update this doc by re-running `./build/air_inspect <new-metallib>` after Xcode or emitter changes. Cross-check with `air_validate` and `MTLDevice.newLibraryWithData:`.
2. `OFFT.bitcode` for later entries can point to byte ranges after prior kernels.
3. The core tag set (`NAME`, `TYPE`, `HASH`, `MDSZ`, `OFFT`, `VERS`, `ENDT`) is stable across
   one- and two-kernel outputs on current Xcode.

## Validation pipeline

Validation should be treated as a pipeline, not a single check:

1. Local parser (`air_validate`)
2. Apple CLI validation (`xcrun metallib --app-store-validate` or `xcrun metal -validate`)
3. Runtime load validation (`MTLDevice.newLibraryWithData:`)

## Repro workflow

```bash
# 1) Build tools
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# 2) Generate reference assets (requires full Xcode shader tools)
./scripts/generate_reference_metallib.sh

# 3) Inspect + validate
./build/air_inspect tests/air_abi/reference/reference.metallib
./build/air_validate tests/air_abi/reference/reference.metallib --xcrun

# 4) Runtime acceptance
ctest --test-dir build -R air_abi_metal_load --output-on-failure
```

## Environment caveat

On systems where `xcrun metal` / `xcrun metallib` are unavailable (for example, Command Line Tools
without full Xcode shader utilities), use `cumetal-air-emitter --mode experimental` for local
container development and parser tests. The script `generate_reference_metallib.sh` falls back to
`tests/air_abi/reference/vector_add_air.ll` and generates
`tests/air_abi/reference/reference.experimental.metallib`.

The experimental container is useful for validating emitter/validator plumbing, but it is not an Apple
driver-compatible metallib.

## Next updates expected

- Compare this byte layout against additional Xcode builds and document field drift.
- Expand tag parsing beyond the currently consumed core set when new kernels require it.
- Keep MetalLibraryArchive validation wired in CI for parser cross-checks.
