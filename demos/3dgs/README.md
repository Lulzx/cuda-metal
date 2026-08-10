# 3D Gaussian Splatting on CuMetal

**Inria’s tile-based CUDA rasterizer, running on Apple Silicon.**

```bash
# needs: built CuMetal + Homebrew glm (brew install glm)
bash demos/3dgs/run.sh
```

Open `out/gaussians.ppm` (or `out/gaussians.png`) in Preview. You should see three
colored splats (red / green / blue) on a dark background.

## What actually runs on the GPU

| Stage | Where | Source |
| --- | --- | --- |
| Preprocess (project, EWA conic, radii) | **host** | `host_preprocess.h` (same math as Inria forward preprocess) |
| Prefix sum of tile counts | host UMA (CUB shim) | CuMetal `cub::DeviceScan` |
| `duplicateWithKeys` | **Apple GPU** | Inria `rasterizer_impl.cu` |
| Radix sort of tile keys | host UMA (CUB shim) | CuMetal `cub::DeviceRadixSort` |
| `identifyTileRanges` | **Apple GPU** | Inria `rasterizer_impl.cu` |
| `renderCUDA` (tile alpha blend) | **Apple GPU** | Inria `forward.cu` |

Every GPU launch is gated by `device=apple_gpu` + `launch_success=true` in the log.

## Why host preprocess?

The full Inria `preprocessCUDA` pulls `glm` types and ~30 kernel parameters.
On this CuMetal stack that currently fails Metal pipeline creation
(`XPC_ERROR_CONNECTION_INTERRUPTED`). The hard, paper-famous path — per-tile
sorted alpha blending in `renderCUDA` — is what ships on the GPU today.

## Scope (read before claiming)

- Forward only. No training / backward / Adam.
- Synthetic 3-Gaussian scene, not a COLMAP-trained point cloud.
- Precomputed RGB (no SH evaluation on device).
- CUB scan/sort between kernels use CuMetal’s host-side UMA shims.
- Vendored kernels are non-commercial research code from
  [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization);
  see `vendor/LICENSE.md`.

## Why this target

3D Gaussian Splatting is one of the most-cited graphics/vision CUDA codebases of
the last few years. AR/VR, robotics, and film people already know it. Showing the
original tile rasterizer kernels launch on a Mac is a stronger story than another
vector-add.

## Example pass (M4 Pro)

```text
rendered instances: 60
radii: 30 30 27
non-background pixels: 3878 / 16384
max channel: ~0.95
rerun mean abs diff: 0
device=apple_gpu  launch_success=true   (duplicateWithKeys, identifyTileRanges, renderCUDA)
PASS: 3D Gaussian Splatting forward rendered on CuMetal
```
