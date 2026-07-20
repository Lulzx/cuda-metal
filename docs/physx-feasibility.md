# PhysX 5.6 GPU on CuMetal: Phase 0 feasibility audit

Date: 2026-07-20

## Executive conclusion

GPU rigid-body PhysX is feasible to bring up incrementally on CuMetal. The
PhysX 5.6.1 GPU source does not use dynamic parallelism, cooperative groups,
grid-wide synchronization, cooperative launch, active CUB or Thrust code, or
FP16. No core rigid-body solver kernel was found to require an in-kernel
cross-threadgroup barrier.

The first target should be a minimized, non-rendering `SnippetHelloGRB` scene
containing one dynamic sphere and one static plane. This keeps GPU broadphase,
contact generation, and the rigid solver active while avoiding articulation,
mesh, heightfield, SDF, particle, cloth, and deformable kernels. The stock
snippet is not small: it creates 40 stacks of 210 boxes (8,400 boxes) plus a
sphere.

The port is not a straight build-system substitution. The main risks are:

1. PhysX relies heavily on exact CUDA warp-mask behavior. There are 638
   `__shfl*`, 241 `__ballot*`, and 234 `__syncwarp` textual occurrences in
   `.cu`/`.cuh` files. Active partial masks occur in rigid-body update,
   broadphase/narrowphase helpers, radix/reduction code, and the PGS/TGS
   solver. CuMetal's conservative full-SIMD-group emulation must be improved
   or validated kernel by kernel.
2. PhysX's `KernelWrangler` eagerly resolves all 501 registered kernel names,
   although a minimal rigid scene executes only a subset. A subset build
   therefore needs a generated kernel manifest and lazy/optional lookup.
3. PhysX normally obtains modules from nvcc-generated fatbinary registration
   records. The preferred CuMetal integration is source-first AOT:
   `cumetalc` produces selected metallibs and a generated registration manifest
   supplies their paths and function names to the existing Driver API flow.
4. Three active `tex3D` calls make SDF collision unavailable until CuMetal
   wires device texture sampling. They are outside the proposed rigid-body
   target.

There is no architectural blocker requiring a redesign decision before Phase
1. Phase 2 should stop if runtime tracing reveals an unobserved cross-block
dependency or if exact partial-mask behavior cannot be implemented locally.

## Audit baseline and scope

- PhysX checkout: sibling directory `../PhysX`
- Upstream: `https://github.com/NVIDIA-Omniverse/PhysX.git`
- Tag: `107.3-physx-5.6.1`
- Commit: `5ca9f472105a90d70d957c243cb0ef36fe251a9f`
- LFS assets: checkout performed with LFS smudging disabled; source files are
  complete and large sample assets remain LFS pointers.
- Kernel scope: `physx/source/gpu*/src/CUDA`, plus GPU headers, `physxgpu`,
  `cudamanager`, GPU CMake files, and GPU-capable snippets.
- Inventory: 61 `.cu` translation units, 509 `__global__` declarations/textual
  occurrences, and 501 names in `PxgKernelNames.h`.

Counts below are textual grep counts and can include commented legacy code
where explicitly noted. The commands in the final section reproduce the
inventory.

## CUDA feature inventory and CuMetal classification

Classification meanings:

- **Fine**: supported directly or absent from the selected target.
- **Needs workaround**: feasible compiler/runtime work or a narrow upstream
  integration patch is required.
- **Blocker**: cannot execute the affected PhysX feature correctly with the
  current CuMetal architecture. A blocker outside the minimal GRB target does
  not block that target.

| Feature requested for audit | PhysX 5.6.1 evidence | CuMetal cross-reference | Classification |
| --- | --- | --- | --- |
| `__shfl*` | 638 occurrences in 46 `.cu`/`.cuh` files. Component counts: articulation 10, broadphase 13, common 111, narrowphase 440, simulation controller 25, solver 39. Most pass `FULL_MASK`; active partial/sub-warp masks occur in common reductions/radix/vector helpers, `cudaBox.cu`, `updateBodiesAndShapes.cu`, and PGS/TGS solver code. | Full-mask shuffle lowering exists. `docs/known-gaps.md` says partial masks are conservatively emulated as a full SIMD group and may differ from inactive-lane CUDA semantics. | **Fine** for full masks; **needs workaround** for partial masks. This is a minimal-GRB correctness risk because solver/update helpers use them. |
| `__ballot*` | 241 occurrences in 46 files. Component counts: articulation 12, broadphase 35, common 16, narrowphase 137, simulation controller 17, solver 24. Most use `FULL_MASK`; active variable-mask ballots occur in reduction/radix helpers, `updateBodiesAndShapes.cu`, and both multiblock solvers. Three legacy `__ballot(` references exist: one active in `cudaGJKEPA.cu`, one commented in `epa.cuh`, and two debug-print arguments in `epa.cuh`. | Full-mask ballot is lowered. Partial masks have the same conservative emulation gap. | **Fine** for full masks; **needs workaround** for active partial masks and the legacy intrinsic spelling if the mesh/GJK path is enabled. |
| `__syncwarp` | 234 occurrences in 44 `.cu`/`.cuh` files (240 when adjacent GPU `.h` files are included). Most are no-argument/full-warp barriers. Active masked calls occur in `internalConstraints2.cu`, `RadixSort.cuh`, `reduction.cuh`, `cudaBox.cu`, `epa.cuh`, and `updateBodiesAndShapes.cu`. | CuMetal lowers warp barriers to an AIR SIMD-group barrier. Its mask cannot exclude hardware lanes. | **Fine** for full-warp barriers; **needs workaround** for masked barriers. The one-sphere target avoids `cudaBox.cu` and articulation but still uses update/reduction infrastructure. |
| `__activemask` | No hits. | CuMetal provides the intrinsic, but it is not required here. | **Fine**. |
| Cooperative groups | No hits for `cooperative_groups`, `grid_group`, or `this_grid()`. `particleSystemHFMidPhaseCG.cu` uses “CG” in a contact-generation name, not the cooperative-groups API. | CuMetal's `grid_group::sync()` is a no-op, but PhysX does not call it. | **Fine**. |
| `grid_group::sync` | No hits. | Current no-op behavior is not exercised. | **Fine**. |
| Dynamic parallelism | No `<<<...>>>` launch syntax or device-launch API hits in the GPU sources. Host code launches through `cuLaunchKernel`. | Dynamic parallelism would be a compile-time error. | **Fine** because absent. |
| Textures/surfaces | Three active `tex3D<float>` calls in `gpunarrowphase/src/CUDA/sdfCollision.cuh` at lines 118, 134, and 147. Two `tex1Dfetch` lines in `bvh.cuh` are commented. `CudaKernelWrangler.cpp` defines an empty `__cudaRegisterTexture` compatibility function. Host GPU code also uses Driver API array/texture-object calls for SDF assets. No active surface instruction was found. | `docs/known-gaps.md` says object lifecycle/array copies exist at Runtime API level, but device `tex.*`/surface sampling is not wired; the equivalent Driver API array/texture symbols are also not currently implemented. | **Blocker** for SDF collision/SDF snippets. **Fine** for the sphere-plane GRB target, which does not select this code. |
| `cudaLaunchCooperativeKernel` | No Runtime or Driver API cooperative-launch hit in PhysX GPU code. | CuMetal's cooperative launch forwards to ordinary launch and cannot provide grid sync. | **Fine** because absent. |
| `cub::` | One hit, a commented `cub::BlockReduce` typedef in `particlesystem.cu:5222`. | CuMetal has a limited CUB header shim, but it is not needed for this source. | **Fine**. |
| `thrust::` | No hits or Thrust includes in GPU source. | CuMetal's host/UMA Thrust shim is not needed. | **Fine**. |
| `__half` / FP16 | No active `__half`, `half`, `half2`, or `cuda_fp16` usage in device source. “half” hits are comments such as half-angle or half of a buffer. | CuMetal FP16 coverage is not on the critical path. | **Fine**. |
| `double` | Active device double precision occurs in `gpusimulationcontroller/src/CUDA/bvh.cuh:356-363` (six robust geometric products) and `gpunarrowphase/src/CUDA/femClothPrimitives.cu:3578` (one plane distance). Other hits are comments or host-only code such as `PxgFEMCloth.cpp:632`. | General FP64 lowering is a known gap; current emulation is name-matched and arbitrary native FP64 is rejected by Metal pipeline creation on current Apple GPUs. | **Needs workaround** for BVH construction and FEM cloth. **Fine** for the primitive sphere-plane target because those units/paths are excluded. |
| Inline PTX (`asm volatile`) | Six active statements: `gpucommon/src/CUDA/atomic.cuh:149` uses `red.global.add.f32`; `gpunarrowphase/src/CUDA/convexMeshMidphase.cu:796,851,894,915,968` uses named `bar.sync 1..5`. | The PTX parser can lower `bar.sync N` to a full threadgroup barrier, but the generic `.cu` frontend does not yet establish that arbitrary inline PTX survives and lowers correctly. The reduction has an ordinary `atomicAdd` fallback. | **Needs workaround**: lower the reduction to `atomicAdd`; lower named full-block barriers to CuMetal's supported barrier form or teach the frontend to lower inline PTX. Both are outside the one-sphere narrowphase path, though `atomic.cuh` is broadly included. |
| Device function pointers | No callable device-function pointer tables or indirect device calls were found. `cudaGJKEPA.cu:1918-1989` does contain a `__constant__ __device__` table of C++ pointers-to-data-members (`FinishContactsWarpScratch::*`), not function pointers. | Indirect device calls are not required. C++ data-member pointer lowering should receive a frontend regression test before enabling the GJK/EPA file. | **Fine** for the audited feature; **needs validation/workaround** for the data-member pointer table on the convex/GJK path. |
| CUDA version/architecture guards | Device guards are limited: `atomic.cuh:148/151` selects inline reduction or `atomicAdd`; `particlesystem.cu:66` selects `__ldg`; `PxgArticulation.h:226` and `PxgIntrinsics.h:43` select architecture-dependent helpers; `PxgIntrinsics.h:57` checks CUDA compiler major version. Solver files include `sm_35_intrinsics.h`. GPU CMake requests SASS 70+ and PTX 120; `CudaContextManager.cpp:114-115` requires compute capability 7.0. | CuMetal reports synthetic compute capability 8.0, so the host capability gate passes. Build flags and NVIDIA-only `sm_35_intrinsics.h` headers need clean-room equivalents or include removal. `__CUDA_ARCH__` must be defined consistently enough to choose supported branches. | **Needs workaround** in build/frontend headers; no architectural blocker. Prefer the portable branches (`atomicAdd`, plain load) where possible. |

### Warp-mask impact on the minimal target

The one-sphere narrowphase kernel itself is unusually clean:
`gpunarrowphase/src/CUDA/cudaSphere.cu` has one `__global__` entry and no
shuffle, ballot, syncwarp, inline PTX, texture, half, double, or atomic hit.
It handles sphere-plane contact in its type switch.

That does not eliminate the mask risk. GPU broadphase, scan/radix helpers,
body/shape updates, contact compaction, constraint preparation, and PGS/TGS
solver kernels still use warp operations. Before trusting numerical output,
Phase 2 must add focused CuMetal tests for:

- sub-warp widths (especially 4, 6, 8, and 16 lanes);
- masks carried across loop iterations in `updateBodiesAndShapes.cu`;
- variable-mask `__shfl_xor_sync` plus `__ballot_sync` in
  `solverMultiBlock.cu` and `solverMultiBlockTGS.cu`;
- masked `__syncwarp` protecting shared-memory reads/writes.

## Kernel loading and launch path

PhysX does not load named `.ptx` files at runtime. Its normal GPU build treats
the `.cu` files as CMake CUDA sources. nvcc emits static initializers, and
PhysX supplies its own registration entry points in
`cudamanager/src/CudaKernelWrangler.cpp`:

1. `__cudaRegisterFatBinary` calls
   `PxGpuCudaRegisterFatBinary`, which unwraps CUDA's `0x466243B1` wrapper and
   stores the embedded fatbinary pointer in the 128-slot module table in
   `physxgpu/src/PxgPhysXGpu.cpp`.
2. `__cudaRegisterFunction` calls `PxGpuCudaRegisterFunction`, which records
   `(module index, device function name)` in a 1,024-slot function table.
3. During `PxCreateCudaContextManager`,
   `cudamanager/src/CudaContextManager.cpp:653-676` calls
   `cuModuleLoadDataEx` for every registered fatbinary.
4. `gpucommon/src/PxgKernelWrangler.cpp` builds the 501-name manifest from
   `PxgKernelNames.h`.
5. `KernelWrangler` eagerly searches the registration table and calls
   `cuModuleGetFunction` for every one of those names.
6. Host GPU components retrieve `CUfunction` handles by enum and ultimately
   call `cuLaunchKernel`; 525 host launch/get-function sites were found.

This is a Driver API architecture, not a CUDA Runtime API launch architecture.
PhysX's local no-op `cudaLaunch`/`cudaLaunchKernel` definitions only satisfy
nvcc object link dependencies; simulation launches do not use them.

### Recommended CuMetal fit

Use CuMetal's **primary source-recompile/AOT path**, not the binary-shim
registration JIT:

1. Compile only selected `.cu` entry points with `cumetalc` into metallibs.
2. Generate a small C++ manifest containing metallib paths (or embedded raw
   metallib bytes), kernel names, and module indices.
3. Feed those images through the existing `cuModuleLoadDataEx` path. CuMetal
   already accepts a metallib path string or metallib bytes.
4. Make `KernelWrangler` resolve functions lazily, or generate a target-specific
   name list, so the selected target does not require all 501 functions.
5. Link the GPU PhysX libraries directly against `libcumetal`; an opt-in
   `libcuda.dylib` alias is unnecessary when the link is under our control.

The binary-shim JIT is a fallback experiment only. PhysX owns the
`__cudaRegister*` entry points, so CuMetal's registration interceptor is not
normally invoked. Its Driver API can parse raw PTX and basic fatbinary wrappers,
but `docs/known-gaps.md` explicitly says full NVCC fatbinary variants are
incomplete. That path also defeats the Phase 2 requirement to compile a
deliberate kernel subset through `cumetalc`.

### Runtime API surface warning

The common Driver API path—contexts, modules, functions, streams, events,
memory, copies, and `cuLaunchKernel`—largely exists in CuMetal. PhysX also
references `cuArray3DCreate`, `cuArrayDestroy`, `cuTexObjectCreate`, and
`cuTexObjectDestroy`; these symbols are not currently implemented in CuMetal's
Driver API. They must be stubbed with honest unsupported errors or implemented
for linking in Phase 3. The minimal sphere-plane scene must not execute them.

## Smallest runnable GPU target

### Selected target: minimized `SnippetHelloGRB`

Use the non-rendering `snippethellogrb`, but patch its scene construction for
the port/conformance mode:

- retain the CUDA context manager;
- retain `eENABLE_GPU_DYNAMICS`, `eENABLE_PCM`, and `eGPU` broadphase;
- retain a static plane;
- remove all 40 box stacks;
- create one dynamic sphere above the plane with a deterministic initial
  pose/velocity;
- simulate enough fixed 1/60 s steps to cover free fall, first contact, and
  settling;
- dump the sphere transform each step for the later CPU/GPU comparison.

Why this is smaller than alternatives:

- Stock `SnippetHelloGRB` uses 8,401 dynamic bodies and both box and sphere
  contact code.
- `SnippetRBDirectGPUAPI` is based on the same 40-stack workload and adds
  direct-copy API activity.
- `SnippetDirectGPUAPIArticulation` adds the entire articulation kernel family.
- Particle, cloth, deformable, isosurface, and SDF snippets activate large
  specialized subsystems; SDF also hits the texture blocker.
- `SnippetJointDrive` can be forced to GPU and has only one dynamic actor in
  its default scene, but it activates joint constraint preparation/solve
  kernels. Its non-rendering default also leaves `gUseGPU == false`, so it is
  not an out-of-box GPU test.

The minimized HelloGRB sphere-plane path is therefore the smallest honest
end-to-end GPU rigid-body target. It cannot be reduced to a gravity-only body
with no collision: that would fail to exercise GPU narrowphase and meaningful
solver contact behavior.

### Initial subsystem/kernel-family boundary

The exact dynamic launch list cannot be proven from static grep alone because
PhysX selects kernels based on scene state. Runtime tracing is required once a
CUDA/CuMetal build exists. The expected first boundary is:

- `gpucommon`: memory copy, scan/reduction, and radix helpers used by pipeline
  bookkeeping;
- `gpubroadphase`: SAP update/report kernels, excluding aggregate-only work;
- `gpunarrowphase`: pair management/contact compaction plus
  `sphereNphase_Kernel`; exclude box, convex, mesh, heightfield, SDF, particle,
  cloth, and deformable contact generators;
- `gpusimulationcontroller`: rigid body/shape, transform/bounds, and changed
  handle updates; exclude articulation and non-rigid controllers;
- `gpusolver`: rigid PGS first, with no joint or articulation paths; defer TGS
  unless the scene configuration selects it.

Phase 2 should instrument `KernelWrangler::getCuFunction`/`cuLaunchKernel` and
grow the manifest from observed missing kernels rather than guessing all 501.

## Architectural-blocker check

The requested blocker patterns are absent from the core rigid path:

- no dynamic parallelism;
- no cooperative groups or cooperative launch;
- no `grid_group::sync`;
- no source-level grid-wide barrier helper;
- multiblock work is structured as separate host-dispatched kernels, atomics,
  grid-stride loops, and ordinary block/warp barriers.

Accordingly, no restructuring proposal is required before Phase 1. If later
runtime tracing disproves this static result, stop and evaluate:

1. split the affected kernel at the global barrier into two dispatches
   (roughly 2–5 engineering days per simple split);
2. use a single-threadgroup correctness path for the conformance scene
   (roughly 1–3 days, poor scalability);
3. move the scheduling loop to the CPU/command-buffer layer
   (roughly 1–3 weeks depending on state that must be externalized).

## Reproduction commands

Run from the CuMetal root:

```bash
git -C ../PhysX describe --tags --exact-match
git -C ../PhysX rev-parse HEAD

rg --files ../PhysX/physx/source/gpu*/src/CUDA -g '*.cu' | wc -l
rg -o '__global__' ../PhysX/physx/source/gpu* -g '*.{cu,cuh}' | wc -l
rg -c '^KERNEL_DEF\\(' ../PhysX/physx/source/gpucommon/include/PxgKernelNames.h

rg -n '__shfl|__ballot|__syncwarp|__activemask' \
  ../PhysX/physx/source/gpu* -g '*.{cu,cuh,h,cpp}'
rg -n 'cooperative_groups|grid_group|this_grid\\(|cudaLaunchCooperativeKernel|cuLaunchCooperativeKernel' \
  ../PhysX/physx/source/gpu* ../PhysX/physx/source/cudamanager \
  ../PhysX/physx/source/physxgpu -g '*.{cu,cuh,h,cpp}'
rg -n '<<<|cudaLaunchDevice|cudaGetParameterBuffer' \
  ../PhysX/physx/source/gpu* -g '*.{cu,cuh,h,cpp}'
rg -n 'texture\\s*<|surface\\s*<|tex1D|tex2D|tex3D|surf1D|surf2D|surf3D|cudaTextureObject_t|cudaSurfaceObject_t' \
  ../PhysX/physx/source/gpu* -g '*.{cu,cuh,h,cpp}'
rg -n 'cub::|thrust::|__half|\\bhalf2?\\b|cuda_fp16|\\bdouble\\b' \
  ../PhysX/physx/source/gpu* -g '*.{cu,cuh,h,cpp}'
rg -n 'asm[[:space:]]+(volatile[[:space:]]*)?\\(' \
  ../PhysX/physx/source/gpu* -g '*.{cu,cuh}'
rg -n '__CUDA_ARCH__|__CUDACC_VER_MAJOR__|CUDA_VERSION|CUDART_VERSION|sm_[0-9]|MIN_SM|ARCH_CODE_LIST' \
  ../PhysX/physx/source/gpu* ../PhysX/physx/source/cudamanager \
  ../PhysX/physx/source/compiler/cmakegpu -g '*.{cu,cuh,h,cpp,cmake,txt}'
```
