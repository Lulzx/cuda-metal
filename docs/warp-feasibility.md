# NVIDIA Warp on CuMetal: Phase 0 feasibility audit

Date: 2026-09-01

## Executive conclusion

Warp is a substantially better CuMetal target than PhysX was, and the reason is
structural rather than incidental: Warp already compiles CUDA C++ to PTX with a
bundled Clang/LLVM, links `libdevice.10.bc` itself, and never invokes NVCC or
NVRTC on that path. The compiler seam CuMetal needs already exists upstream and
is already exercised on machines with no CUDA toolkit installed.

The device-code surface is also far smaller than PhysX's. Across the 29 headers
that a generated Warp kernel actually includes (35,710 lines), there are 9
`__shfl*`, 6 `__ballot*`, 1 `__activemask`, 4 `__syncwarp`, and 50
`__syncthreads*` occurrences, and *every one of them* lives in `tile_*.h`. The
PhysX equivalent was 638 / 241 / 234. Warp's non-tile core — `vec.h`, `mat.h`,
`quat.h`, `spatial.h`, `intersect.h`, `array.h`, `mesh.h`, `bvh.h`,
`hashgrid.h`, `rand.h`, `noise.h`, `svd.h` — is per-thread scalar code with
atomics and no warp-level programming at all. That is precisely the shape
CuMetal handles today.

At the audit baseline, one defect blocked essentially every real Warp kernel,
and it was a CuMetal defect, not a Warp one. It reduces to two lines of CUDA:

```cuda
struct A { float4* data; int n; };
extern "C" __global__ void k(A x, A out) { int i = threadIdx.x; out.data[i] = x.data[i]; }
```

The baseline `cumetalc` rejected this with `cannot legalize CUDA generic pointers:
host-populated pointer field reaches a conflicting concrete address space`.
Clang lowers the aggregate assignment to `llvm.memcpy.p0.p0`, and the address
space legalizer cannot resolve a memcpy whose source and destination are two
distinct host-populated buffers. Since Warp's central type is
`array_t<T>` — a by-value struct holding `T* data` and `T* grad` — and since
most Warp arrays hold `vec3`, `mat33`, `quat`, `transform`, or a `wp.struct`,
almost every generated kernel emits exactly this memcpy.

**Follow-up status (2026-09-01):** the minimal reproducer now compiles through
the typed source-first path to a validated Apple metallib. The fix propagates
generic-pointer status to the synthetic constant-offset pointers created while
expanding `llvm.memcpy`; unit IR and direct `.cu` frontend regressions cover the
case. This is compiler/metallib evidence, not yet an unmodified Warp runtime
result.

There is no architectural blocker requiring a redesign decision before Phase 1.

## Audit baseline and scope

- Warp checkout: `/Users/lulzx/work/warp-cpu-threadpool`
- Upstream: `https://github.com/NVIDIA/warp.git`
- Version: 1.12.0, tag `v1.12.0`, commit `e6c3ba2d54bb048115760b5cd7a4bb2573329ae7`
- License: Apache-2.0 (the whole library, unlike Isaac Sim)
- CuMetal: `main` at `050642e`, `build/` (Debug, shim ON)
- The checkout is a local branch (`cpu-threadpool`) off `v1.12.0`; nothing in
  this audit depends on those local changes.

Counts below are textual grep counts over the named file sets and can include
inactive preprocessor branches. The commands in the final section reproduce
them.

## Warp has two distinct bodies of GPU code

This distinction governs everything that follows.

**1. The JIT path (user kernels).** `warp/_src/codegen.py` translates a Python
`@wp.kernel` into a `.cu` file that includes exactly one header, `builtin.h`,
which pulls in 28 more. That file is compiled at runtime and loaded with
`cuModuleLoadDataEx`. This is where user code lives, and it is the part worth
porting.

**2. The static native library.** 11 `.cu` files compiled into `libwarp` at
Warp *build* time: 50 `__global__` kernels, and CUB or Thrust in 6 of them
(`sort.cu` 15 uses, `sparse.cu` 12, `reduce.cu` 4, `scan.cu` 4,
`runlength_encode.cu` 2, `bvh.cu` 1). These back `wp.utils.array_sum`,
radix sort, BVH refit, sparse matrix ops, and volume building. CuMetal already
shims the CUB and Thrust device-level APIs these use.

## Device-code feature inventory (JIT header set)

29 headers, 35,710 lines. Localization matters more than the totals:

| Feature | Count | Where it lives | CuMetal status |
|---|---|---|---|
| `__shfl*` | 9 | `tile_reduce.h`, `tile_scan.h`, `tile_radix_sort.h` | Tile-only; deferrable |
| `__ballot*` | 6 | same | Tile-only; deferrable |
| `__activemask` | 1 | `tile_radix_sort.h` | Implemented (`activemask` fix) |
| `__syncwarp` | 4 | `tile_radix_sort.h` | Tile-only; deferrable |
| `__syncthreads*` | 50 | `tile.h`, `tile_radix_sort.h`, `tile_bvh.h`, `tile_mesh.h` | Supported |
| `__shared__` | 43 | `tile_*.h` (41), `mesh.h` (1), `bvh.h` (1) | Supported |
| Atomics | 40 | `builtin.h` (28), `tile_reduce.h` (2), `tile_bvh.h` (5), `tile_mesh.h` (5) | See defect 1 |
| Inline PTX | 6 | `builtin.h` (2: `atom.add.noftz.f16`, `atom.add.f64`), `tile.h` (4: `cvta`, `cp.async`) | Not lowered |
| `tex1D/2D/3D` | 9 | `texture.h` only | See defect 2 |
| FP16 | 4 | `builtin.h` | Partial |
| Cooperative groups | 0 | — | Not needed |
| Dynamic parallelism | 0 | — | Not needed |
| CUB / Thrust | 0 | — | Not needed in JIT path |
| `__launch_bounds__` | 0 | — | Not needed |
| MMA / WMMA | 0 | — | Not needed |

The two `__shared__` uses outside the tile headers are both the same construct:
a per-thread BVH traversal stack under `#if BVH_SHARED_STACK`, whose address is
stored into a pointer *field* (`query.stack.ptr = &stack[threadIdx.x]`). A
pointer field that can hold either a shared or a private address is the same
class of problem as defect 1 and should be expected to trip the same
legalizer. `BVH_SHARED_STACK` is compile-time optional.

`volume.h` pulls in NanoVDB (210 references). It is header-only C++ with no
warp-level intrinsics, but it is bulky and should be treated as its own tier.

## The compiler seam

`warp/_src/build.py:build_cuda` already branches:

```python
if warp.config.llvm_cuda:
    runtime.llvm.wp_compile_cuda(src, cu_path, inc_path, output_path, False)
else:
    runtime.core.wp_cuda_compile_program(...)   # NVRTC + nvJitLink
```

`wp_compile_cuda` (`warp/native/clang/clang.cpp:312`) builds a bare Clang cc1
invocation with `-triple nvptx64-nvidia-cuda -target-cpu sm_70 +ptx75`, links
`warp/native/libdevice/libdevice.10.bc`, and emits PTX assembly. No CUDA
toolkit, no NVRTC, no NVIDIA driver. `libwarp-clang.dylib` on this machine
already exports it (`nm` confirms `_wp_compile_cuda`).

Three integration options, in increasing order of nativeness:

1. **PTX handoff.** Set `llvm_cuda = True`, take the PTX, feed it to CuMetal's
   PTX lowering. Least invasive; inherits the legacy backend's coverage.
2. **Direct `.cu` handoff (recommended).** Replace the `build_cuda` body with a
   `cumetalc` invocation on the generated `.cu`, producing a metallib. This is
   CuMetal's primary, typed, source-first path — the one with the better
   numerics story — and it is what the experiments below use.
3. **Warp-native Metal backend.** Teach `codegen.py` to emit MSL. Large, and
   pointless while option 2 works.

Note that Warp's own `llvm_cuda` path does not currently compile out of the box
even on Linux-shaped inputs: the cc1 invocation has no sysroot and no
`__clang_cuda_runtime_wrapper.h`, so `__syncwarp`, `__ballot_sync`, `__shfl_sync`
and `atomicExch` are all undeclared (109 errors on a trivial kernel). It appears
to be an unexercised code path upstream. Option 2 sidesteps this entirely,
because `cumetalc` drives Clang in real CUDA mode with a real header set.

## Empirical results

A generated Warp kernel was produced with `ModuleBuilder.codegen("cuda")` and
compiled with `cumetalc` against a shadow copy of `warp/native`.

**Compiles to a real Apple metallib, end to end:**

```cuda
extern "C" __global__ void k(wp::launch_bounds_t dim,
                             wp::array_t<wp::float32> x,
                             wp::array_t<wp::float32> out)
{
    size_t _idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) return;
    wp::float32* p = wp::address(x, (int)_idx);
    wp::array_store(out, (int)_idx, wp::load(p) * 2.0f);
}
```

That covers `array_t` as a by-value kernel parameter, `wp::address` byte-offset
indexing, `wp::load`, `wp::array_store`, and the launch-bounds struct — the
spine of every Warp kernel. It works today.

**Did not compile at the audit baseline:** the same kernel with
`dtype=wp.vec3`, which exposed defect 1. **Rerun 2026-09-05:** after the
defect-1 and defect-2 fixes, both the `float` and `vec3` generated modules
compile end to end to validated metallibs, forward and backward kernels
included, with `texture.h` stubbed out of `builtin.h` (defect 3).

Two upstream frictions were also needed to get that far, both one-liners:

- `warp/native/crt.h:249` includes `cuda_crt.h` whenever `__CUDACC__ &&
  __clang__`, on the assumption that Clang here means *barebones* Clang with no
  CUDA headers. Under `cumetalc` that collides with
  `__clang_cuda_device_functions.h` (`__dAtomicAdd` and friends redefined). It
  needs a guard.
- `warp/native/tile.h:41` defines its own `float4` under
  `#else // If CUDA is not available (e.g., macOS build)`. Compiling with
  `-D WP_ENABLE_CUDA=1` is enough to take the correct branch.

## Defects this audit found in CuMetal

**1. Aggregate copy between two device buffers failed to lower.** Resolved in
the current working tree for the reduced constant-size memcpy form.

```cuda
struct A { float4* data; int n; };
extern "C" __global__ void k(A x, A out) { int i = threadIdx.x; out.data[i] = x.data[i]; }
```

The audit baseline produced `cannot legalize CUDA generic pointers: host-populated pointer field reaches
a conflicting concrete address space`. The emitted IR is a single
`llvm.memcpy.p0.p0.i64` between two pointers derived from distinct byval struct
parameters. A scalar element type (`float*`) succeeds; so does an aggregate
routed through a by-value helper that forces scalar loads. Only the memcpy form
failed. This was the same flow-insensitivity shape recorded for the `cvt` and
pointer-base bugs: memcpy-created offset pointers were not marked generic, so
they were prematurely treated as private instead of resolving from each use's
device base. The focused IR test and a direct `.cu` frontend fixture now pass.

**2. `atomicAdd(float*, float)` cannot be compiled on the default path.**
Blocking for any reduction or force accumulation. **Resolved (2026-09-05):**
the source-first path now lowers the overlay's inline `atom.global.add.f32`,
`atomicrmw fadd`/`fsub`, and PTX `atom.add.f32` to Metal's native device
`atomic_float`; threadgroup float add uses a bit-pattern CAS helper because no
Metal language version has threadgroup float atomics. With that, the generated
`kf`/`kv` modules below compile to metallibs *including their adjoint kernels*.
`tests/cuda_projects/float_atomics` is the numeric check; texture.h (defect 3)
must still be stubbed out of the include set.

```cuda
extern "C" __global__ void k(float* out, const float* in) { atomicAdd(out, in[threadIdx.x]); }
```

→ `unsupported LLVM inline assembly 'atom.global.add.f32 $0, [$1], $2;'`.
CuMetal's own `runtime/api/cuda_runtime.h:1862` spells float `atomicAdd` as
inline PTX specifically so the *PTX* path selects Metal's native float atomic;
the source-first `cumetal-ir` backend then cannot lower its own inline asm.
Substituting `__fAtomicAdd` moves the failure one stage later — `Metal atomic
lowering requires one 32-bit or lock-backed 64-bit integer result` — so the
`cumetal-ir` backend has no float-atomic lowering at all. Metal 3 supports
`atomic_fetch_add_explicit` on `atomic<float>` in device address space; this
looks like a straightforward gap rather than a hardware limit.

**3. Texture fetches are scalar-only.** Non-blocking; `wp.Texture` is used by
zero shipped Warp examples. `tex1D/2D/3D<float2>` and `<float4>` fail to
instantiate because CuMetal's bilinear helper (`cuda_runtime.h:1401`) does
scalar-times-vector arithmetic with no vector operator overloads in scope.
**Resolved (2026-09-05):** `tex1D` added (it did not exist at all) and the
linear filter now mixes through overloads for `float2`/`float4`. The generated
Warp modules compile with `texture.h` included; no stub is needed.

**4. `--backend legacy` ignores `-I`.** Minor. The same command line that works
under `--backend cumetal-ir` reports `'builtin.h' file not found`.

## Runtime and driver surface

Warp resolves its entire CUDA driver surface dynamically: it `dlopen`s
`libcuda.so`, pulls `cuGetProcAddress`, and resolves 70 driver entry points
through it (`warp/native/cuda_util.cpp:157`). It never links against the driver.
That is an unusually clean seam for `libcumetal.dylib`'s optional `libcuda`
alias — but it has one hard gate:

- **`cuGetProcAddress` is not exported by `libcumetal.dylib`.** Nothing resolves
  without it. It is a name→pointer table over symbols that mostly already exist.
  **Resolved (2026-09-05):** exported, resolving against the library's own
  `cu*` symbol table so it cannot drift from the real surface.
  `cuMemcpy2D`, `cuMemcpy2DAsync`, `cuMemcpyBatchAsync`,
  `cuEventRecordWithFlags` and `cuStreamGetCtx` were added at the same time;
  `functional_driver_proc_address` covers all six. Of Warp's 70 entry points,
  the 16 still missing are exactly the GL, IPC, graph-capture and
  `cuArrayCreate` groups below.

Of the 70 entry points Warp asks for, 49 are already exported. The 21 missing
ones cluster by feature, and most are optional:

| Group | Count | Needed for | Verdict |
|---|---|---|---|
| `cuGraphics*GL*` | 5 | OpenGL interop in `wp.render` | Skip |
| `cuIpc*` | 5 | Multi-process sharing | Skip |
| `cuGraph*`, `cuStream*Capture*` | 5 | CUDA graph capture (`wp.ScopedCapture`) | Defer; Warp degrades without it |
| `cuMemcpy2D`, `cuMemcpy2DAsync` | 2 | Strided array copies | Needed |
| `cuMemcpyBatchAsync` | 1 | Batched copies | Needed |
| `cuArrayCreate` | 1 | Texture arrays | With defect 3 |
| `cuEventRecordWithFlags` | 1 | Event timing | Easy |
| `cuStreamGetCtx` | 1 | Stream/context queries | Easy |

Warp also queries `cuDeviceGetAttribute` extensively and derives a compute
capability used to select a PTX target; the shim must present a coherent
synthetic device.

## Proposed phasing

**Phase 1 — scalar kernels, source-first.** Defects 1 and 2 are fixed,
`cuGetProcAddress` plus the 5 needed driver entry points are exported, and the
NVRTC / nvPTXCompiler shim is in (2026-09-05). Remaining: patch `build_cuda` to
call `cumetalc`, and add a `darwin-arm64 + CuMetal` branch to `build_dll.py` so
`WP_ENABLE_CUDA=1` is expressible. Gate: a handful of `warp/tests` modules
running green on `device="cuda:0"`.

**Phase 2 — the type zoo.** `vec`/`mat`/`quat`/`transform`/`spatial_vector`,
`wp.struct`, indexed arrays, RNG, noise, `svd`. This is where the 591 registered
builtins live. Mostly a conformance grind, not new architecture, once defect 1
is gone.

**Phase 3 — geometry queries.** `mesh_query_*`, `bvh_query_*`, `hashgrid`,
marching cubes. Adds the `BVH_SHARED_STACK` pointer-field question. 7 of 78
shipped examples need this.

**Phase 4 — the static library.** Port the 11 `.cu` files, leaning on CuMetal's
existing CUB/Thrust shims. Unlocks `wp.utils`, sorting, sparse, volumes.

**Phase 5 — tiles.** `tile_*.h`: every warp intrinsic in the codebase, plus
`cp.async` and `cvta` inline PTX, plus the MathDx/cuBLASDx LTO path
(`build_lto_dot`, `build_lto_solver`, `build_lto_fft`) which has no Metal
analogue and would have to be re-expressed against Metal Performance Shaders or
CuMetal's own cuBLAS. 17 of 78 examples. Largest and least certain tier.

NanoVDB (`wp.Volume`) is orthogonal and can be attempted at any point after
Phase 2.

## Building `libwarp` itself with `WP_ENABLE_CUDA=1` on macOS (2026-09-05)

Getting Warp's Python side to *use* CuMetal needs more than user-kernel
compilation: `wp_is_cuda_enabled()` comes from the native library, so the 11
`.cu` files in `libwarp` must compile and link. A sweep through CuMetal's
`nvcc` shim with Warp's real release flags, after the header additions below,
stands at **6 of 11 compiling** (`mesh`, `hashgrid`, `scan`, `volume`, `warp`,
`volume_builder`) as of 2026-09-05. The five that remain are the CUB device-header
group, which is Phase 4 work.

Landed on the CuMetal side for this: `cudaTypedefs.h` (generated by
`scripts/generate_cuda_typedefs.py`, covering every `PFN_cu*` name Warp
declares), the graph/capture/IPC/graphics types those typedefs need,
`CUDA_RESOURCE_VIEW_DESC`, `CUfunction_attribute`, `cudaCpuDeviceId`, and shim
acceptance of `-gencode=…`/`-t0`. On the Warp side (branch `cpu-threadpool`):
`crt.h` guarded on `WP_CUMETAL`, an `__APPLE__` `dlopen` branch in
`cuda_util.cpp`, `--cuda-path` honoured on Darwin in `build_lib.py`, and a
CuMetal branch in `build_dll.py` (detected from the toolkit's `version.json`)
that defines `WP_CUMETAL=1` and links `-lcuda` instead of the NVRTC/PTX static
libraries.

What still blocks the other seven, in the order to attack them:

| File | Blocker | Owner |
|---|---|---|
| ~~`warp.cu`~~ | **Compiles (2026-09-05).** Needed the NVRTC/nvPTXCompiler shim plus the driver and runtime API names below: the `CU_DEVICE_ATTRIBUTE_PCI_*` triple, `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`, `MEMORY_POOLS_SUPPORTED`, `CU_EVENT_RECORD_*` / `CU_EVENT_WAIT_*`, `CU_STREAM_ADD/SET_CAPTURE_DEPENDENCIES`, `CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS`, `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE`, `cudaStreamAdd/SetCaptureDependencies`, `cudaMemPoolGet/SetAccess`, `cudaGraphUpload`, `cudaGraphDebugDotPrint`, and graph user objects. | CuMetal |
| `reduce.cu`, `runlength_encode.cu`, `sparse.cu` | `cub/device/device_reduce.cuh`, `device_run_length_encode.cuh` missing from the CUB shim | CuMetal (Phase 4) |
| `sort.cu` | `cub::DoubleBuffer` missing | CuMetal (Phase 4) |
| `bvh.cu` | `cub::BlockReduce::Reduce` overload set incomplete; Clang rejects `__shared__` variables with initializers that nvcc accepts | CuMetal (Phase 4) / Warp |
| ~~`volume_builder.cu`~~ | **Compiles (2026-09-05)** after adding the `#include <new>` its placement `new` needs; nvcc includes it implicitly, Clang does not. | Warp (landed) |

`warp.cu` compiles as of 2026-09-05, so Phase 1's gate no longer depends on the
generated-kernel path alone; nothing else in Warp's Python layer needs to
change.

## The NVRTC shim (2026-09-05)

`runtime/api/nvrtc.h`, `runtime/api/nvPTXCompiler.h` and `runtime/rt/nvrtc.cpp`
give CuMetal a runtime-compilation surface, exported from `libcumetal.dylib`
(and aliased as `libnvrtc.dylib`) so a caller linking `-lnvrtc` or `-lcuda`
resolves it. `nvrtcCompileProgram` writes the program and any in-memory headers
into a temporary directory and runs `cumetalc … --emit metallib`; `nvrtcGetCUBIN`
returns those metallib bytes, which `cuModuleLoadDataEx` already recognises by
their `MTLB` magic. `cumetalc` is located through `CUMETAL_NVRTC_COMPILER`,
`CUMETAL_ROOT`, the directory of the loaded dylib, then `PATH`.

NVRTC's option surface is nvcc's, so most of it describes PTX/SASS code
generation. `--include-path`, `--define-macro`, `--undefine-macro`,
`--pre-include` and `--gpu-architecture` map onto `cumetalc` flags; the rest
(`--extra-device-vectorization`, `--diag-suppress=`, `--fmad=`, `-pch`,
`--Ofast-compile=`, …) is recognised and dropped, with a line in the program log
naming what was skipped. `runtime/rt/nvrtc_options.cpp` holds that mapping and
`tests/unit/nvrtc_options_test.cpp` covers it without spawning a compiler;
`tests/functional/nvrtc_compile_test.cpp` runs the full create/compile/
`cuModuleLoadDataEx` sequence.

Two deliberate limits. `nvrtcGetPTX` fails: `cumetalc` lowers CUDA source to
AIR, never to PTX. A request for a virtual architecture (`compute_XX`) is
therefore rejected by `nvrtcCompileProgram` itself, with the reason in the
program log, rather than returning bytes the caller would mis-handle.
`nvrtcGetSupportedArchs` reports the single architecture CuMetal presents
(sm_80), which is what steers Warp onto the CUBIN path unprompted. And
`nvPTXCompiler` is a pass-through: CuMetal's module loader parses PTX text
itself, so "compiling" hands the same PTX back.

## Stop conditions

Stop and reassess if a fresh generated-Warp rerun exposes broader aggregate
address-space cases beyond the reduced defect-1 fix, or if Phase 2 reveals that
Warp's adjoint (backward) kernels — roughly doubling every kernel body and
routing every store through a float atomic — cannot be made to lower without
defect 2 being solved natively rather than through the CAS lock bank.

## Reproducing the inventory

```bash
WARP=~/work/warp-cpu-threadpool/warp/native
SET="crt.h builtin.h vec.h mat.h quat.h spatial.h intersect.h intersect_adj.h \
     array.h tuple.h mesh.h bvh.h svd.h hashgrid.h volume.h volume_impl.h \
     texture.h range.h rand.h noise.h matnn.h tile.h tile_reduce.h tile_scan.h \
     tile_radix_sort.h tile_bvh.h tile_mesh.h solid_angle.h initializer_array.h"

cd "$WARP"
grep -cE "__shfl|__ballot|__activemask|__syncwarp" $SET | grep -v ":0"
grep -c  "__syncthreads" $SET | grep -v ":0"
grep -cE "atomic(Add|CAS|Min|Max|Exch|Or|And|Xor|Sub)" $SET | grep -v ":0"
grep -nE "asm volatile" $SET

# driver entry points Warp resolves, vs. what libcumetal exports
grep -ohE 'get_driver_entry_point\("[A-Za-z0-9_]+"' cuda_util.cpp \
  | sed 's/.*"\(.*\)"/\1/' | sort -u > /tmp/warp_drv.txt
nm -gU ~/work/cumetal/build/libcumetal.dylib | awk '{print $3}' | sed 's/^_//' \
  | grep -E '^cu[A-Z]' | sort -u > /tmp/cumetal_drv.txt
comm -23 /tmp/warp_drv.txt /tmp/cumetal_drv.txt
```

Generating a kernel and compiling it:

```bash
cd ~/work/warp-cpu-threadpool
PYTHONPATH=. python3 -c '
import warp as wp
from warp._src.context import ModuleBuilder
@wp.kernel
def k(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid(); out[i] = x[i] * 2.0
m = k.module
open("/tmp/k.cu", "w").write(ModuleBuilder(m, m.options).codegen("cuda"))'

# needs a shadow include dir: warp/native with crt.h's cuda_crt.h include guarded
~/work/cumetal/build/cumetalc /tmp/k.cu -o /tmp/k.metallib \
    -I <shadow-dir> -D WP_CUMETAL=1 -D WP_ENABLE_CUDA=1 --emit metallib
```
