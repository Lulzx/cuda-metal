# Apple GPU execution implementation record

This document records the July 2026 work that moved CuMetal's CUDA source and
binary-shim paths away from implicit CPU emulation and established positive,
numerically checked Apple GPU execution.

## Result

Covered CUDA kernels now execute as Metal compute commands on Apple Silicon.
CPU kernel emulation and host helper fallbacks are disabled by default and can
only be enabled explicitly for diagnostics.

The following end-to-end results were verified on an Apple M4 Pro:

- A standalone `.cu` vector-add program compiled through CuMetal and produced
  the expected values on the Apple GPU.
- NVIDIA's unmodified `cuda-samples` `vectorAdd` source compiled through the
  in-tree CUDA toolchain shims, printed `Test PASSED`, and emitted completed
  `generic_ptx` Apple-GPU provenance.
- llm.c's GPT-2 FP32 conformance workload passed logits, loss, tensor, and
  overall numerical checks with CPU emulation disabled.
- llama.cpp's unmodified GGML CUDA backend, linked against `libcumetal`, loaded
  SmolLM2-135M-Instruct-Q4_K_M with one GPU-offloaded layer and greedily
  completed `The capital of France is` as `The capital of France is Paris.`
  The verified run generated at 5.8 tokens/s.

This is proof for the covered paths, not a claim of general CUDA compatibility.
Higher llama.cpp offload counts and arbitrary models still encounter unsupported
GGML kernels.

## Runtime policy

### CPU execution is opt-in

The legacy llm.c CPU implementation is disabled unless
`CUMETAL_ENABLE_LLMC_CPU_EMULATION=1` is set. GGML host kernel helpers are
disabled unless `CUMETAL_ENABLE_HOST_KERNEL_FALLBACKS=1` is set. Both diagnostic
modes emit a warning when enabled.

The strict conformance scripts reject provenance from `cpu_fallback` and `stub`
sources, so a CPU result cannot be mistaken for a GPU pass.

### Registered launches are correctness-serialized

llama.cpp allocates a large CUDA arena and passes adjacent regions of the same
underlying Metal buffer through several CUDA streams. The initial asynchronous
registration path produced timing-dependent stale reads: isolated RMS,
conversion, dequantization, and GEMM probes were exact, while the live model
alternated between exact and badly corrupted RMS results.

Synchronizing every registered launch removed the corruption and produced the
same greedy answer as the stock CPU build. CuMetal therefore synchronizes
fatbinary-registered launches before returning by default. Direct/source-first
kernels retain asynchronous stream behavior.

`CUMETAL_ENABLE_ASYNC_REGISTERED_LAUNCH=1` opts into the incomplete asynchronous
registration path for development. It is not a correctness mode. The long-term
fix is CUDA-equivalent cross-command-queue resource fencing for aliased Metal
buffers.

`CUMETAL_SYNC_EACH_LAUNCH=1` remains a broader diagnostic switch that also
synchronizes direct launches.

## GPU provenance

Set `CUMETAL_TRACE_GPU=1` to print one `CUMETAL_PROVENANCE` record per completed
Metal dispatch. Records identify:

- the Metal device;
- the kernel name;
- lowering source (`generic_ptx`, `specialized_msl`, or `metallib`);
- cache status;
- grid and block dimensions;
- launch success;
- completed GPU duration when available.

GPU conformance requires `device=apple_gpu` and `launch_success=true`. The gates
reject CPU fallback and stub provenance.

## Compiler and registration changes

The PTX-to-Metal path gained:

- `cvta.to.global` support;
- `mul.wide.s32` and `mul.wide.u32` support;
- floating-point opcode register typing fixes;
- a fast negative filter for unsupported large GGML kernel families;
- explicit classification of approximate/passthrough templates so they are
  refused by default instead of silently producing wrong values;
- exact specialized MSL for the covered GGML output path:
  - `rms_norm_f32`, including strided 3D input and mul/add broadcasting;
  - `k_bin_bcast` float add and multiply;
  - float-to-half and half-to-float `convert_unary`;
  - Q8_0-to-f16 block dequantization.

The RMS lowering uses one Metal threadgroup per CUDA row, a fixed 32-lane SIMD
width, `simd_sum` subtotals, and a small threadgroup reduction. The previous
global-thread mapping wrote beyond the destination arena.

Registration cache keys include a schema/version tag, so behavior-changing
lowering updates invalidate stale generated Metal sources. The runtime refuses
approximate registered kernels unless `CUMETAL_ENABLE_APPROX_KERNELS=1` is set.

## cuBLAS and Metal backend changes

Mixed-precision `cublasGemmEx` now honors the selected CUDA stream and waits for
the Metal/MPS result before host-side conversion or reuse. Tests cover the exact
SmolLM2 output-head dimensions:

- Q8_0 dequantization of 28,311,552 weights;
- f32-to-f16 conversion of a 576-element activation;
- a `49152 x 1 x 576` mixed GEMM;
- f16-to-f32 conversion of 49,152 logits.

`CUMETAL_CUBLAS_CPU_REFERENCE=1` is an opt-in diagnostic oracle used to separate
GEMM errors from surrounding kernel/scheduling errors. It is not enabled by
default and is not used by GPU conformance.

`CUMETAL_VALIDATE_GGML_RMS=1` synchronizes live GGML RMS launches and compares
their exact bound buffers, strides, dimensions, and broadcast metadata against a
CPU oracle. It was used to expose the alternating exact/corrupt results in the
asynchronous registration path. It is diagnostic-only and disabled by default.

The Metal backend records asynchronous command completion and reports positive
GPU duration provenance. macOS build/test helpers strip stale provenance
metadata and apply ad-hoc signatures to generated binaries where needed.

## CUDA source toolchain compatibility

CuMetal supplies narrow `ptxas` and `fatbinary` command-line shims for Clang and
upstream CUDA projects. They accept the invocation shapes used by the verified
samples and preserve embedded PTX for runtime registration. They do not emulate
NVIDIA SASS generation.

The strict CUDA project harnesses:

- require a real Apple GPU provenance record;
- require a numerical pass marker or exact output comparison;
- reject CPU fallbacks, approximate stubs, and silent skips;
- keep unsupported larger projects as explicit exit-77 skips.

## Validation commands

Configure and build the normal binary-shim path:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j8
```

Run the strict source and GGML operator gates:

```bash
ctest --test-dir build -R \
  'functional_cuda_source_gpu_vector_add|conformance_cuda_samples_vectoradd_gpu|functional_cuda_projects_ggml_output_ops' \
  --output-on-failure
```

Run the verified llama.cpp smoke test:

```bash
CUMETAL_LLAMA_NGL=1 \
bash tests/conformance/run_llama_cpp_cumetal.sh
```

Validate that the source-first build does not depend on the binary shim:

```bash
cmake -B build-nosshim -DCMAKE_BUILD_TYPE=Debug \
  -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build build-nosshim -j8
ctest --test-dir build-nosshim --output-on-failure
```

The July 20 completion audit selected 182 non-benchmark/non-external-model tests
in the default build and 171 in the binary-shim-off build. Both runs completed
with zero failures. Environment-dependent AIR/Xcode, standalone CUDA, and
binary-shim-only cases reported explicit skips instead of false passes.

## Remaining limitations

- The verified llama.cpp result uses NGL=1. Broad high-NGL support needs more
  dequantization, matrix, attention, RoPE, and copy kernels.
- Registered launches are synchronous until explicit Metal cross-queue fencing
  is implemented, which reduces throughput.
- Fatbinary registration still takes several minutes for llama.cpp because the
  process scans and prepares a very large kernel set.
- Approximate template bodies still exist for development but are refused by
  default.
- The generic PTX lowering surface remains a documented subset; successful
  specialized MSL kernels do not imply arbitrary PTX support.

See also [known-gaps.md](known-gaps.md), [testing.md](testing.md), and
[status.md](status.md).
