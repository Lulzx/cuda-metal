# Build and Validation

Build
-----

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
cmake --install build --prefix /tmp/cumetal-install
# optional: also install the libcuda.dylib drop-in alias (see docs/legal-notice.md)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCUMETAL_ENABLE_BINARY_SHIM=ON
```

Two independent switches
------------------------

| Option | Default | Controls |
|--------|---------|----------|
| `CUMETAL_ENABLE_CUDA_REGISTRATION` | `ON` everywhere | The host CUDA registration ABI (`__cudaRegister*`) that Clang emits when compiling *your own* `.cu`. The source path needs it; it is not a binary shim. |
| `CUMETAL_ENABLE_BINARY_SHIM` | `ON` except Release | The `libcuda.dylib` alias only — the drop-in for binaries pre-linked against NVIDIA's libcuda. |

These used to be a single flag, so a Release build silently replaced the registration ABI with a
stub and the source-recompilation path stopped being tested in the shipping configuration.
Enabling the binary shim without the registration ABI is a configure-time error.

Compile a CUDA program
----------------------

```bash
./build/cumetalc samples/vectorAdd/vectorAdd.cu -o /tmp/vectorAdd
CUMETAL_TRACE_GPU=1 /tmp/vectorAdd
```

An installed `cumetalc` finds its headers, `libcumetal.dylib`, and the `ptxas`/`fatbinary` shims
relative to its own path. Set `CUMETAL_ROOT` to point it at a prefix explicitly.

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

llm.c conformance harness build
-------------------------------

```bash
export CUMETAL_LLMC_DIR="/path/to/llm.c"
# optional: tune gradient threshold used by patched test harness source
export CUMETAL_LLMC_GRAD_TOL="1.2e-2"
./scripts/build_llmc_test_gpt2fp32cu.sh "$CUMETAL_LLMC_DIR"
```
