# Testing and Benchmarking

The GPU-only proof strategy and July 2026 validation record are documented in
[apple-gpu-execution.md](apple-gpu-execution.md). In particular, a numerical
result is not a GPU pass unless the test also observes completed Apple-GPU
provenance and rejects CPU fallback/stub sources.

Runtime execution tests
-----------------------

These tests compile Metal kernels with `xcrun` and run them through the CuMetal runtime:

```bash
ctest --test-dir build -R functional_runtime_vector_add --output-on-failure
ctest --test-dir build -R functional_runtime_vector_add_heap_alloc --output-on-failure
ctest --test-dir build -R functional_runtime_vector_add_cu --output-on-failure
ctest --test-dir build -R functional_sample_vector_add --output-on-failure
ctest --test-dir build -R functional_runtime_matrix_mul --output-on-failure
ctest --test-dir build -R functional_runtime_stream_vector_add --output-on-failure
ctest --test-dir build -R functional_runtime_null_stream_sync --output-on-failure
ctest --test-dir build -R functional_runtime_stream_per_thread --output-on-failure
ctest --test-dir build -R functional_runtime_async_memops --output-on-failure
ctest --test-dir build -R functional_runtime_event --output-on-failure
ctest --test-dir build -R functional_runtime_stream_wait_event --output-on-failure
ctest --test-dir build -R functional_runtime_stream_query --output-on-failure
ctest --test-dir build -R functional_runtime_memcpy_kind --output-on-failure
ctest --test-dir build -R functional_runtime_symbol_memcpy --output-on-failure
ctest --test-dir build -R functional_runtime_mem_get_info --output-on-failure
ctest --test-dir build -R functional_runtime_host_alloc --output-on-failure
ctest --test-dir build -R functional_runtime_host_pointer_api --output-on-failure
ctest --test-dir build -R functional_runtime_stream_flags --output-on-failure
ctest --test-dir build -R functional_runtime_stream_callback --output-on-failure
ctest --test-dir build -R functional_runtime_device_api --output-on-failure
ctest --test-dir build -R functional_runtime_device_properties --output-on-failure
ctest --test-dir build -R functional_runtime_device_attribute --output-on-failure
ctest --test-dir build -R functional_runtime_device_reset --output-on-failure
ctest --test-dir build -R functional_runtime_device_flags --output-on-failure
ctest --test-dir build -R functional_runtime_error_api --output-on-failure
ctest --test-dir build -R functional_runtime_profiler_api --output-on-failure
ctest --test-dir build -R functional_curand_uniform --output-on-failure
ctest --test-dir build -R functional_cublas_api --output-on-failure
ctest --test-dir build -R functional_driver_vector_add --output-on-failure
ctest --test-dir build -R functional_driver_matrix_mul --output-on-failure
ctest --test-dir build -R functional_driver_null_stream_sync --output-on-failure
ctest --test-dir build -R functional_driver_device_api --output-on-failure
ctest --test-dir build -R functional_driver_error_api --output-on-failure
ctest --test-dir build -R functional_driver_profiler_api --output-on-failure
ctest --test-dir build -R functional_driver_device_query --output-on-failure
ctest --test-dir build -R functional_driver_device_attribute --output-on-failure
ctest --test-dir build -R functional_driver_stream_flags --output-on-failure
ctest --test-dir build -R functional_driver_stream_per_thread --output-on-failure
ctest --test-dir build -R functional_driver_stream_callback --output-on-failure
ctest --test-dir build -R functional_driver_context_switch --output-on-failure
ctest --test-dir build -R functional_driver_context_requirements --output-on-failure
ctest --test-dir build -R functional_driver_async_memcpy --output-on-failure
ctest --test-dir build -R functional_driver_memset --output-on-failure
ctest --test-dir build -R functional_driver_mem_get_info --output-on-failure
ctest --test-dir build -R functional_driver_mem_alloc_managed --output-on-failure
ctest --test-dir build -R functional_driver_host_alloc --output-on-failure
ctest --test-dir build -R functional_driver_host_pointer_api --output-on-failure
ctest --test-dir build -R functional_driver_module_load_data --output-on-failure
ctest --test-dir build -R functional_driver_module_load_data_ptx --output-on-failure
ctest --test-dir build -R functional_driver_launch_extra --output-on-failure
ctest --test-dir build -R functional_driver_launch_extra_scalar --output-on-failure
ctest --test-dir build -R functional_driver_stream_wait_event --output-on-failure
ctest --test-dir build -R functional_runtime_axpy_offset --output-on-failure
ctest --test-dir build -R functional_runtime_atomic --output-on-failure
ctest --test-dir build -R functional_runtime_atomic_shared --output-on-failure
ctest --test-dir build -R functional_runtime_warp_shuffle --output-on-failure
ctest --test-dir build -R functional_runtime_warp_vote --output-on-failure
ctest --test-dir build -R functional_runtime_warp_size_lane --output-on-failure
ctest --test-dir build -R functional_runtime_warp_partial_mask --output-on-failure
ctest --test-dir build -R functional_runtime_fp16_ops --output-on-failure
ctest --test-dir build -R functional_runtime_fp64_ops --output-on-failure
ctest --test-dir build -R functional_runtime_shared_reduce --output-on-failure
ctest --test-dir build -R functional_runtime_grid_2d --output-on-failure
ctest --test-dir build -R functional_runtime_grid_3d --output-on-failure
ctest --test-dir build -R functional_runtime_cooperative_launch --output-on-failure
ctest --test-dir build -R functional_runtime_struct_arg --output-on-failure
ctest --test-dir build -R functional_runtime_barrier_order --output-on-failure
ctest --test-dir build -R functional_runtime_cp_async_emul --output-on-failure
ctest --test-dir build -R functional_runtime_device_properties --output-on-failure
ctest --test-dir build -R functional_runtime_occupancy --output-on-failure
ctest --test-dir build -R functional_runtime_device_limits --output-on-failure
ctest --test-dir build -R functional_runtime_printf --output-on-failure
ctest --test-dir build -R functional_cufft_c2c --output-on-failure
ctest --test-dir build -R functional_runtime_ptx_lowering_regression --output-on-failure
ctest --test-dir build -R functional_runtime_matrix_mul_tiled --output-on-failure
ctest --test-dir build -R functional_runtime_dynamic_shared --output-on-failure
ctest --test-dir build -R functional_runtime_registration_printf --output-on-failure
ctest --test-dir build -R functional_driver_extended_api --output-on-failure
# binary-shim-only tests (`CUMETAL_ENABLE_BINARY_SHIM=ON`):
ctest --test-dir build -R functional_runtime_registration_path --output-on-failure
ctest --test-dir build -R functional_runtime_call_config_registration --output-on-failure
ctest --test-dir build -R functional_runtime_registration_fatbin_ptx --output-on-failure
ctest --test-dir build -R functional_runtime_legacy_launch_registration --output-on-failure
ctest --test-dir build -R functional_runtime_registration_fatbinary2_symbols --output-on-failure
ctest --test-dir build -R functional_runtime_registration_var_symbol --output-on-failure
ctest --test-dir build -R unit_allocation_table --output-on-failure
ctest --test-dir build -R unit_module_cache --output-on-failure
ctest --test-dir build -R unit_library_conflict --output-on-failure
ctest --test-dir build -R unit_metallib_parser --output-on-failure
ctest --test-dir build -R unit_ptx_parser --output-on-failure
ctest --test-dir build -R unit_intrinsic_lower --output-on-failure
ctest --test-dir build -R unit_printf_lower --output-on-failure
ctest --test-dir build -R unit_addrspace_pass --output-on-failure
ctest --test-dir build -R unit_metadata_pass --output-on-failure
ctest --test-dir build -R unit_phase1_pipeline --output-on-failure
ctest --test-dir build -R unit_ptx_lower_to_llvm --output-on-failure
ctest --test-dir build -R unit_cuda_fp16_host --output-on-failure
ctest --test-dir build -R unit_cuda_vector_types --output-on-failure
ctest --test-dir build -R unit_ptx_lower_to_metal --output-on-failure
ctest --test-dir build -R unit_cumetal_bench_help --output-on-failure
ctest --test-dir build -R unit_cumetal_bench_invalid_arg --output-on-failure
ctest --test-dir build -R unit_cumetal_bench_ratio_gate --output-on-failure
ctest --test-dir build -R unit_runtime_library_aliases --output-on-failure
# binary-shim-only unit tests (`CUMETAL_ENABLE_BINARY_SHIM=ON`):
ctest --test-dir build -R unit_binary_shim_symbol_exports --output-on-failure
ctest --test-dir build -R unit_binary_shim_library_alias --output-on-failure
ctest --test-dir build -R unit_binary_shim_link_alias --output-on-failure
ctest --test-dir build -R unit_library_link_aliases --output-on-failure
ctest --test-dir build -R ptx_sweep_supported_ops --output-on-failure
ctest --test-dir build -R ptx_sweep_unsupported_ops --output-on-failure
ctest --test-dir build -R unit_install_uninstall_scripts --output-on-failure
```

Conformance suite
-----------------

Phase 4 conformance gate over functional tests:

```bash
ctest --test-dir build -R conformance_phase4_functional --output-on-failure
ctest --test-dir build -R conformance_llmc_gpt2fp32cu --output-on-failure
ctest --test-dir build -R functional_cuda_projects_ --output-on-failure
```

Notes:
- `conformance_phase4_functional` now prints per-test progress (`[i/N]`) and applies a per-test timeout.
- Override per-test timeout with `CUMETAL_CONFORMANCE_SINGLE_TEST_TIMEOUT` (seconds, default `120`).
- `air_abi_xcode_matrix_regression` uses `CUMETAL_XCODE15_DEVELOPER_DIR`/`CUMETAL_XCODE16_DEVELOPER_DIR` when set.
  If unset, it falls back to `xcode-select -p` for both slots (single-Xcode mode).
- `functional_cuda_projects_*` compiles and runs standalone CUDA sample programs under
  `tests/cuda_projects/` (SGEMM naive/shmem/2d, reduction, transpose). It requires
  Clang, `xcrun`, and the matching `libcumetal` build. Unsupported lowering is
  reported as an exit-77 skip.
- `conformance_llmc_gpt2fp32cu` is registered only when llm.c is configured (set `CUMETAL_LLMC_DIR`
  before CMake configure, or place checkout at `../llm.c` relative to this repo root).
- `conformance_llmc_gpt2fp32cu` skips when `gpt2_124M.bin` is missing from the llm.c checkout
  (place under checkout root or `dev/data/` per upstream layout).
- When registered, `conformance_llmc_gpt2fp32cu` auto-wires CuMetal's LLVM+fatbin shim flow:
  `scripts/build_llmc_test_gpt2fp32cu.sh` + `scripts/run_llmc_test_gpt2fp32cu.sh`.
- `conformance_llmc_gpt2fp32cu` fails on any `TENSOR NOT OK` marker and requires
  `overall okay: 1` in output.
- By default, the gate disables llm.c CPU emulation, requires numerical parity,
  and requires successful Apple-GPU provenance. CPU fallback and stub
  provenance are failures.
- To force pure PTX-lowered execution (no llm.c emulation fallback), run:

```bash
CUMETAL_LLMC_REQUIRE_NO_EMULATION=1 \
ctest --test-dir build -R conformance_llmc_gpt2fp32cu --output-on-failure
```

Direct invocation with custom threshold/regex:

```bash
./tests/conformance/run_conformance_suite.sh build 90 '^functional_'
```

Optional llm.c stress harness setup:

```bash
export CUMETAL_LLMC_DIR="/path/to/llm.c"
bash scripts/fetch_llmc_assets.sh   # gpt2_124M.bin + debug state (once)
# optional overrides:
export CUMETAL_LLMC_BUILD_CMD="scripts/build_llmc_test_gpt2fp32cu.sh"
export CUMETAL_LLMC_TEST_CMD="scripts/run_llmc_test_gpt2fp32cu.sh"
# optional: gradient checker tolerance applied by build shim patching
export CUMETAL_LLMC_GRAD_TOL="1.2e-2"
# optional: hard-disable llm.c runtime emulation fallback
export CUMETAL_DISABLE_LLMC_EMULATION="1"
export CUMETAL_ENABLE_LLMC_CPU_EMULATION="0"
```

Kernel argument notes:
- Scalar kernel params should be passed as `CUMETAL_ARG_BYTES` (with `size_bytes`) or via a
  device buffer.
- Passing a host scalar pointer as `CUMETAL_ARG_BUFFER` is invalid and will fail launch with
  `cudaErrorInvalidDevicePointer`.

Benchmark runner
----------------

```bash
./scripts/generate_reference_metallib.sh
./build/cumetal_bench \
  --metallib tests/air_abi/reference/reference.metallib \
  --kernel vector_add \
  --elements 262144 \
  --warmup 5 \
  --iterations 50 \
  --max-ratio 2.0
```

If `xcrun metal`/`xcrun metallib` are unavailable
--------------------------------------------------

```bash
./build/cumetal-air-emitter \
  --input tests/air_abi/reference/vector_add_air.ll \
  --output /tmp/vector_add.experimental.metallib \
  --mode experimental \
  --overwrite

./build/air_validate /tmp/vector_add.experimental.metallib \
  --require-function-list --require-metadata
```
