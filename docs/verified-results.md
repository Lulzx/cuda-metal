# Verified results and downstream projects

This document records the performance and application claims summarized in the
README. A successful process exit is insufficient: first-party results require
numerical or domain-specific checks and Apple-GPU provenance.

## Covered kernels

Vector addition, SAXPY, reduction, matrix operations, atomics, shared memory,
warp operations, streams, events, and selected CUDA library calls have numerical
GPU tests. The suite includes negative cases because accepting a program is not
the same as implementing it correctly.

## Performance gate

The Phase 5 gate compares CuMetal with hand-written Metal for three memory-bound
kernels. On an Apple M4 Pro, rebuilt and measured on 2026-07-27:

| Kernel | Elements | CuMetal / native Metal |
| --- | ---: | ---: |
| vector add | 262,144 | 1.063x |
| SAXPY | 262,144 | 1.036x |
| FP32 reduction | 262,144 | 1.008x |

The gate uses the fastest of 20 synchronized wall-clock iterations. These
kernels take roughly 0.2 ms, so averages mostly measure scheduler interference.
The target is at most 2x native Metal, not a claim that translated code beats
the baseline.

Reproduce it with:

```bash
ctest --test-dir build -R bench_phase5_all_kernels --output-on-failure
```

## Real programs

- Upstream `cuda-samples` vector addition builds without source changes and
  passes a numerical plus Apple-GPU provenance gate.
- Upstream `cuda-samples` `simplePrintf` builds without source changes and emits
  all 32 expected block/thread/value records through the Apple-GPU ring-buffer
  path. The focused in-tree Clang-ABI test independently checks the same record set.
- llm.c GPT-2 FP32 passes logits, loss, tensor, and GPU-provenance checks on the
  tested path. It uses explicit workload specializations and is not proof of
  arbitrary PTX support.
- llama.cpp's unmodified GGML CUDA backend builds against CuMetal. SmolLM2-135M
  greedy decoding was coherent from one-layer offload through saturation on the
  verified Apple M4 Pro setup. FlashAttention is advertised as unsupported, so
  llama.cpp selects its ordinary attention path.
- A reduced PhysX 5.6 GRB path runs selected sphere, box, convex, and triangle
  mesh contacts on the GPU. It is a selected-shape conformance target, not
  general PhysX GPU support.

Exact commands, models, tolerances, provenance requirements, and scope
boundaries live in [the Apple-GPU execution record](apple-gpu-execution.md),
[the testing guide](testing.md), and [known gaps](known-gaps.md).

## Projects using CuMetal

The following result is third-party work, verified by its author rather than by
this repository.

### cu_vslam_rs

[cu_vslam_rs](https://github.com/jeff-hykin/cu_vslam_rs) by
[@jeff-hykin](https://github.com/jeff-hykin) compiles NVIDIA's
[cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) visual-odometry stack for
Apple Silicon against CuMetal and packages it as an SDK with a Nix flake
(`nix build ...#sdk-metal`). NVIDIA ships no macOS build. Its CUDA kernels are
not rewritten in Metal; they use CuMetal's PTX path.

The project's `metal_smoke` test asserts actual camera motion rather than a
success status because CuMetal defects have historically returned success while
producing an identity pose. Stereo runs on either backend. RGB-D requires the
GPU because cuVSLAM v17 lifts depth into landmarks only in a CUDA kernel.

The working configuration required
`CUMETAL_USE_METAL_DEVICE_ADDRESSES=1`: its feature detector builds texture
objects over linear memory and dereferences the resource pointer in device code,
which reads as zero under CuMetal's default addressing without an error.
`cudaCreateTextureObject` now warns about that case instead of failing silently.

If you have shipped something on CuMetal, open a pull request adding it here.
