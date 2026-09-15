# Build and Validation

Build
-----

Source builds require CMake, LLVM 18 or newer, and the LZ4 and Zstd development
libraries. On Homebrew these are `llvm`, `lz4`, and `zstd`.

On Apple Silicon, the Nix development shell pins LLVM/Clang 21, macOS SDK 15, CMake, Ninja,
Bash, Python, LZ4 and Zstd. It sets LLVM discovery and both CUDA Clang variables
used by the compiler and tests. Use a fresh build directory to avoid cached
Homebrew or Apple compiler paths:

```bash
nix develop
cmake -S . -B build-nix -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_SYSROOT="$SDKROOT" \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
  -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build build-nix -j2
BUILD_DIR="$PWD/build-nix" bash scripts/generate_reference_metallib.sh
ctest --test-dir build-nix --output-on-failure -j2
```

Apple's command-line Metal tools are installed separately. The shell uses
Apple's `xcrun` and the installation selected by `xcode-select`, while retaining
Nix's SDK for host compilation. To select a different Xcode for Metal, export
`DEVELOPER_DIR` after entering the shell. Modern Xcode also provides
`xcodebuild -downloadComponent metalToolchain`. Verify that
`/usr/bin/xcrun --find metal` and `/usr/bin/xcrun --find metallib` succeed before
running the AOT tests. The shell reports missing tools but remains usable for
host builds and runtime Metal compilation. Without these tools, the fixture
script produces only an experimental fixture, not `reference.metallib`.
The Apple tool wrappers clear `SDKROOT` for their subprocesses so Apple's host
linker can find its SDK libraries. Let CMake discover those wrappers from `PATH`.
Direct Nix compiler calls receive the pinned sysroot through `NIX_CFLAGS_COMPILE`.
The shell also selects macOS `mktemp`, matching the test scripts' BSD `-t` syntax.

For Debug/shim-on validation, configure another fresh directory with
`-DCMAKE_BUILD_TYPE=Debug -DCUMETAL_ENABLE_BINARY_SHIM=ON`. The default shell
provides Clang 21; the optional Clang 21/22/23 matrix requires the other two
compilers separately.
The module-scanning option avoids `clang-scan-deps` bypassing Nix's compiler
wrapper and losing its C++ header paths; this project does not use C++ modules.
The CUDA compiler wrapper also supplies the pinned SDK's C headers after libc++
headers, since the NVPTX device pass does not inherit Darwin's implicit search.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cmake --install build --prefix /tmp/cumetal-install
/tmp/cumetal-install/bin/cumetal doctor
# optional: also install the libcuda.dylib drop-in alias (see docs/legal-notice.md)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCUMETAL_ENABLE_BINARY_SHIM=ON
```

For a manifest-backed install with a matching uninstaller:

```bash
bash install/install.sh build /opt/cumetal
# Optional and explicit; the default does not edit shell startup files.
bash install/install.sh build /opt/cumetal --shell-config
```

The installer does not export global `DYLD_*` variables. Linked output from
`cumetalc` has an rpath to its installed runtime, and `cumetal run` scopes
runtime lookup changes to the process it launches.

Homebrew packaging
------------------

The formula is maintained in
[`Lulzx/homebrew-tap`](https://github.com/Lulzx/homebrew-tap). It builds a
Release configuration with Homebrew LLVM, keeps the binary shim off, and runs a
compile-and-execute GPU smoke test:

```bash
brew install lulzx/tap/cumetal
```

Two independent switches
------------------------

| Option | Default | Controls |
|--------|---------|----------|
| `CUMETAL_ENABLE_CUDA_REGISTRATION` | `ON` everywhere | The compatibility host CUDA registration ABI (`__cudaRegister*`) used by Clang-built objects and PTX/fatbinary paths. `cumetalc` native-AOT executables do not depend on it. |
| `CUMETAL_ENABLE_BINARY_SHIM` | `ON` except Release | The `libcuda.dylib` alias only — the drop-in for binaries pre-linked against NVIDIA's libcuda. |

These used to be a single flag, so a Release build silently replaced the registration ABI with a
stub and the source-recompilation path stopped being tested in the shipping configuration.
Enabling the binary shim without the registration ABI is a configure-time error.

Binary-shim formats and JIT cache
---------------------------------

The opt-in shim recognizes CMTL envelopes, raw PTX, basic
FatBinary/FatBinary2/FatBinary3 PTX wrappers, bounded version-`0x0101`
LZ4/Zstd-compressed PTX entries, and the ELF forms listed in
[known gaps](known-gaps.md). Decompression is capped at 64 MiB. It does not
execute SASS or accept every NVCC fatbinary variant.

Registered PTX is compiled on first use and cached under
`$CUMETAL_CACHE_DIR/registration-jit/` (by default under the user's Library
cache). Cache identity includes the input, kernel, lowering policy, compiler
schema, toolchain-dependent inputs, and the `libcumetal` Mach-O UUID. Set
`CUMETAL_DEBUG_REGISTRATION=1` to inspect format detection, compilation, cache
hits, ABI inference, and registration.

Runtime allocation diagnostics
------------------------------

Large `cudaMalloc` allocations use `MTLHeap` suballocation at 4 MiB and above.
Override that policy only for diagnosis:

```bash
CUMETAL_MTLHEAP_ALLOC=1 ./program   # always use heap allocation
CUMETAL_MTLHEAP_ALLOC=0 ./program   # never use heap allocation
```

Compile a CUDA program
----------------------

```bash
./build/cumetalc samples/vectorAdd/vectorAdd.cu -o /tmp/vectorAdd
CUMETAL_TRACE_GPU=1 /tmp/vectorAdd
```

An installed `cumetalc` finds its headers, `libcumetal.dylib`, and the `ptxas`/`fatbinary` shims
relative to its own path. Set `CUMETAL_ROOT` to point it at a prefix explicitly.

Use `cumetal doctor` to check Apple Silicon, macOS, LLVM, Metal tools, headers,
and runtime discovery in one pass.

Generate and validate a reference metallib (requires full Xcode)
-----------------------------------------------------------------

```bash
./scripts/generate_reference_metallib.sh
./build/air_inspect tests/air_abi/reference/reference.metallib
./build/air_validate tests/air_abi/reference/reference.metallib --xcrun
./build/cumetalc --mode xcrun --input tests/air_abi/reference/vector_add.metal --output /tmp/vector_add.cumetalc.metallib --overwrite
./build/cumetalc --mode xcrun tests/air_abi/reference/vector_add.metal -o /tmp/vector_add.cumetalc.positional.metallib --overwrite
./build/cumetalc --mode xcrun tests/air_abi/reference/vector_add.metal --overwrite
./build/cumetalc --mode experimental --input tests/air_abi/reference/vector_add.cu --output /tmp/vector_add.cumetalc.from_cu.experimental.metallib --overwrite
./build/cumetal-ptx2llvm --input tests/air_abi/reference/vector_add.ptx --output /tmp/vector_add.from_ptx.ll --entry vector_add --overwrite
./build/cumetal-ptx2llvm tests/air_abi/reference/vector_add.ptx --entry vector_add --overwrite
ctest --test-dir build -R air_abi_metal_load --output-on-failure
ctest --test-dir build -R air_abi_emit_validate_experimental --output-on-failure
ctest --test-dir build -R air_abi_validate_negative --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_positional_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_default_output_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_multikernel_emit_validate_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_ptx_to_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_matrix_ptx_to_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_ptx_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_matrix_ptx_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_cu_experimental_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_cu_default_output_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_cu_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_ptx_default_output_validate --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_ptx_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_cumetalc_matrix_ptx_emit_load_xcrun --output-on-failure
ctest --test-dir build -R air_abi_ptx2llvm_positional_default_output --output-on-failure
ctest --test-dir build -R air_abi_xcode_matrix_regression --output-on-failure
```

Optional Xcode 15/16 ABI matrix setup:

```bash
export CUMETAL_XCODE15_DEVELOPER_DIR="/Applications/Xcode_15.app/Contents/Developer"
export CUMETAL_XCODE16_DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
```

Optional manual llm.c stress-harness build
-------------------------------------------

```bash
export CUMETAL_LLMC_DIR="/path/to/llm.c"
# optional: tune gradient threshold used by patched test harness source
export CUMETAL_LLMC_GRAD_TOL="1.2e-2"
./scripts/build_llmc_test_gpt2fp32cu.sh "$CUMETAL_LLMC_DIR"
```
