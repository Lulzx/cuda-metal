# Testing and Benchmarking

The GPU-only proof strategy and July 2026 validation record are documented in
[apple-gpu-execution.md](apple-gpu-execution.md). In particular, a numerical
result is not a GPU pass unless the test also observes completed Apple-GPU
provenance and rejects CPU fallback/stub sources.

A skip is not a pass either. Harnesses must not downgrade a wrong answer to a
skip, and a skip must be a precondition checked *before* the work runs, never a
verdict reached after it. Read the pass count together with the skip count, and
inspect any long-lived skip with `ctest -V` before trusting it —
[correctness-audit-2026-07-26.md](correctness-audit-2026-07-26.md) records two
cases where a skip concealed a real failure and one where a test had never
executed on any machine.

Verification layers
-------------------

The repository intentionally contains no GitHub Actions workflows. Release and
GPU gates are explicit local/commissioned-machine commands; their results must
be recorded rather than inferred from repository automation.

Both supported build policies remain required:

| Configuration | Purpose |
| --- | --- |
| Release + binary shim off | Shipping, source-first configuration |
| Debug + binary shim on | Development and opt-in alias coverage |

The Phase 5 `bench_phase5_all_kernels` performance gate is registered only in
Release builds. It measures synchronized host launch overhead as well as GPU
execution, so an unoptimized Debug runtime is not a comparable performance
configuration. The test carries the `benchmark`, `gpu`, and `performance`
labels; Debug inventories contain correctness tests only.

The host selection covers parser, CFG/SSA, typed IR/MSL, PTX lowering, AIR
container, negative-path, headers, CLI, cache, ABI registration, and packaging
checks. It does not prove GPU execution:

```bash
bash scripts/ci_report.sh build \
  --require-tests \
  --label-regex '^hosted$'
```

On a commissioned Apple-GPU machine, first run a narrow no-skip proof spanning
Metal loading, runtime launch, streams, atomics, warp masks, native-AOT source
linking, and numerical PTX execution, then run the full correctness suite.
Optional external-project tests may still report explicit skips.

The report policy flags are intentionally distinct:

- `--require-tests` rejects an empty CTest label or regex selection.
- `--require-no-skips` rejects any skip in a prerequisite-complete proof gate.
- The full suite should not use `--require-no-skips`, because optional llm.c,
  llama.cpp, PhysX, and multi-Xcode checks are environment-dependent.

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
ctest --test-dir build -R '^functional_typed_(direct|ptx)_(device_atomics|system_atomics|fence)$' --output-on-failure
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
ctest --test-dir build -R functional_device_launch_queue --output-on-failure
ctest --test-dir build -R functional_runtime_registration_printf --output-on-failure
ctest --test-dir build -R functional_cuda_graph_api --output-on-failure
ctest --test-dir build -R functional_async_mempool_api --output-on-failure
ctest --test-dir build -R functional_driver_extended_api --output-on-failure
# CUDA registration-path tests (always built; CUMETAL_ENABLE_CUDA_REGISTRATION=ON):
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
# libcuda.dylib alias tests (`CUMETAL_ENABLE_BINARY_SHIM=ON` only):
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

Phase 4 conformance gate over the reviewed 185-test functional manifest:

```bash
ctest --test-dir build -R conformance_phase4_functional --output-on-failure
ctest --test-dir build -R functional_cuda_projects_ --output-on-failure

# Manifest-complete strict sweep with classified TSV/JSON output:
python3 tests/cuda_projects/sweep_cuda_projects.py
```

`tests/conformance/phase4_functional_manifest.txt` is the fixed denominator.
Every enrolled test is expected to pass; failures, timeouts, missing
registrations, disabled tests, and prerequisite skips remain non-passing
denominator entries. The 90% threshold therefore cannot be met by excluding
skips from the calculation.

The separate NVIDIA sample gate requires a full `cuda-samples` checkout. It is
intentionally run serially because several upstream workloads are large:

```bash
CUMETAL_CUDA_SAMPLES_DIR=/path/to/cuda-samples \
ctest --test-dir build -R '^conformance_cuda_samples_sweep$' --output-on-failure
```

Its expected outcomes live in
`tests/cuda_projects/cuda_samples_sweep_manifest.txt`. The gate skips unless
most enrolled samples are present, rejects regressions from `pass` or `waive`,
and also rejects an unsupported classification that starts succeeding. Keep
compile/link-only results as `run-unverified`; they are not runtime evidence.
The 2026-08-29 snapshot is 83 pass, 0 waive, and 0 without a passing runtime
result. See [known gaps](known-gaps.md) for the implementation details.

Notes:
- `conformance_phase4_functional` now prints per-test progress (`[i/N]`) and applies a per-test timeout.
- Override per-test timeout with `CUMETAL_CONFORMANCE_SINGLE_TEST_TIMEOUT` (seconds, default `120`).
- `air_abi_xcode_matrix_regression` uses `CUMETAL_XCODE15_DEVELOPER_DIR`/`CUMETAL_XCODE16_DEVELOPER_DIR` when set.
  If unset, it falls back to `xcode-select -p` for both slots (single-Xcode mode).
- `functional_cuda_projects_*` compiles and runs standalone CUDA sample programs under
  `tests/cuda_projects/` (SGEMM naive/shmem/2d, reduction, transpose). It requires
  Clang, `xcrun`, and the matching `libcumetal` build. Unsupported lowering is
  reported as an exit-77 skip.
- The standalone sweep uses strict classification instead of CTest skip
  semantics and distinguishes prerequisite skips, compile/link failures,
  unsupported kernels, numerical failures, crashes, timeouts, and other runtime
  errors. Its manifest check fails if a new standalone `.cu` fixture is not
  enrolled.
- llm.c is intentionally not registered in CTest. Its scripts remain an optional
  manual stress harness; successful runs must still require numerical parity,
  Apple-GPU provenance, and no CPU-emulation fallback.

Direct invocation with a custom threshold/manifest:

```bash
./tests/conformance/run_conformance_suite.sh build 90 \
  tests/conformance/phase4_functional_manifest.txt
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
bash tests/conformance/run_llmc_gpt2fp32cu.sh
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
