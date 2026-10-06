# Runtime and CUDA semantic gaps

[Known-gaps index](../known-gaps.md) · [Runtime status](../status/runtime.md)

## Synchronization and launch

- Cooperative-grid synchronization is limited to a conservatively resident
  grid capped at one block per reported GPU core. Oversubscription is rejected.
- Dynamic launch uses a fixed 1 MiB device queue with at most 1,023 child
  records per parent dispatch and a host scheduling drain. Nested parent-child-
  grandchild execution, invalid child configurations, and record overflow have
  focused Apple-GPU tests. Queue growth and hardware-recursive scheduling parity
  remain absent.
- Contiguous half-warp and non-contiguous `0xa5a55a5a` masks have focused
  vote, shuffle, barrier-ordering, binary/labeled partition, and divergent
  coalesced-group tests. Arbitrary mask/topology interactions remain narrower
  than CUDA's complete surface.
- Kernel arguments bind sequentially at Metal buffer indices 0-30, the device
  maximum. Hidden arguments for grid Y offset, grid barrier, device clock,
  atomic lock bank, constant store and trap status occupy fixed reserved
  indices (25-30), so a kernel that uses one of those features loses the
  corresponding user-argument slots -- the lowering rejects the combination
  at compile time. Kernels that use none of them may bind all 31 slots; the
  runtime detects a launch argument occupying a reserved index and skips the
  hidden binding rather than overwriting the caller's buffer.
- A by-value aggregate kernel argument is bound as one buffer that every thread
  reads. When the source frontend (direct `.cu`, NVRTC) finds a kernel that may
  write the aggregate, directly or through a helper it calls, it gives each
  thread a private copy at entry, which costs a copy of the aggregate per
  thread. Kernels that only read it, such as large functors, keep reading the
  shared bytes.
- Stream priorities are reported as zero and are not Metal priority queues.
- CUDA device clocks use a device-wide atomic counter with a fixed monotonic
  quantum. They preserve wait-loop progress and unsigned wraparound behavior,
  but values are not GPU cycles and cannot be used for cycle-accurate timing.

## Zero-size allocation

`cudaMalloc(&pointer, 0)` succeeds, writes a null pointer, and consumes no
tracked allocation or reported free memory. A null output argument is invalid.
`cudaMallocManaged`, `cudaMallocAsync`, and driver `cuMemAlloc` still reject
zero sizes; this change covers the ordinary allocation used by Kokkos scratch
storage.

## Function preparation

`cuFuncLoad` prepares a function through the same path launches and attribute
queries use: it compiles or loads the library, resolves the entry, creates the
pipeline and runs the existing reflection checks, without dispatching. It is
idempotent, and concurrent preparation of one kernel coalesces on the backend
mutex rather than compiling twice.

`cuFuncIsLoaded` is query-only. It never starts a compilation, and it does not
take the backend mutex -- that mutex is held across Apple's compiler, so a
query that waited on it would block behind the compilation it exists to
observe. It reports preparation performed through any path.

Two boundaries worth stating. Readiness is a property of a live function in the
current context: an unloaded module destroys its functions, so a stale handle
returns `CUDA_ERROR_INVALID_HANDLE` rather than reporting loaded because the
process-wide pipeline cache still holds the same path and name. But a *new*
handle to the same metallib does report loaded without being prepared again,
because the artifact is immutable and the pipeline genuinely is ready for it.

There is no public "preparing" or "failed" state: CUDA defines only unloaded
and loaded, and CuMetal does not invent more. A preparation failure is reported
by `cuFuncLoad`'s return code, with the failing stage -- library compilation,
entry lookup or pipeline creation -- carried in a one-time `CUMETAL WARNING`
rather than collapsed into a bare `CUDA_ERROR_INVALID_VALUE`.

## Graphs and allocators

Tested graph capture/replay includes an event-linked two-stream dependency with
ordered numerical replay, stale-event lifetime rejection, and conflicting-
capture rejection. Explicit graph nodes also preserve pointer-backed pitched 3D
copy geometry and pitched 1/2/4-byte memset values; malformed pitch, extent,
element-size, overflow, dependency, and creation-flag inputs are rejected.
Array-backed graph-copy replay has focused host-to-array, offset array-to-array,
and array-to-host numerical coverage, including channel-width-aware pitched
geometry and out-of-bounds rejection. Graph and executable parameter setters
cover the supported kernel, 1D/3D memcpy, memset, and host node families with
node-identity, node-type, and malformed-parameter rejection. Clone/update and
memory-node lifetimes still do not cover child graphs, event/semaphore or
conditional nodes, every memory-node update interaction, or arbitrary
multi-stream topologies. Virtual/physical allocation reuse, allocator
caching/release-threshold behavior, and those advanced update cases remain
incomplete.

Device-side graph capability probes compile but honestly report the missing
feature: `cudaGetCurrentGraphExec()` returns null in device code and the
device `cudaGraphLaunch` overload returns `cudaErrorNotSupported`.
`cudaStreamGraphTailLaunch`/`cudaStreamGraphFireAndForget` are the CUDA
sentinel stream values for source compatibility; tail-launch and
fire-and-forget graph semantics are not implemented.

## Memory and pointers

- Arbitrary pageable `malloc` pointers are not kernel-bindable merely because
  Apple Silicon uses UMA; tracked Metal-backed allocations are required.
- Asynchronous copies follow CUDA's pageable-memory contract: a copy whose host
  end is pageable is synchronous with respect to the host (the source is staged
  at the call; a pageable destination is written before the call returns, in
  stream order). Only pinned memory from `cudaHostAlloc`/`cudaMallocHost` is
  copied asynchronously. A pageable-destination copy therefore drains the
  stream, which is the CUDA cost model rather than a CuMetal limitation, but
  the drain is a full stream synchronization rather than a wait on that one
  operation. Memsets targeting pageable host memory, which CUDA rejects and
  CuMetal accepts on unified memory, are completed before returning for the
  same reason.
- A CuMetal device pointer is the CPU mapping of a shared buffer, but Metal
  dereferences GPU virtual addresses. So that structures holding device pointers
  survive `cudaMemcpy`, host-to-device copies rewrite every pointer-aligned
  8-byte word that falls inside a live allocation to its GPU address, and
  device-to-host copies reverse it. The rewrite is heuristic: integer data that
  happens to equal such an address is rewritten too, so a kernel reading it as
  an integer sees the GPU address. Each copy filters words against one
  snapshot of the allocation intervals, so plain data costs a comparison
  rather than a locked lookup, but the scan still visits every word of every
  host/device copy.
- Managed-memory API compatibility does not imply CUDA concurrent managed access
  or CPU/GPU atomics. Prefetch/advice/range-query/attach calls validate tracked
  spans and arguments and preserve prefetch stream ordering, but do not control
  physical placement or reproduce NVIDIA residency state on unified memory.
- Persisting-L2/access-policy APIs preserve a conservative validated hint state,
  but public Metal offers no cache-residency control.
- Function/device cache preferences, shared-memory bank preferences, and
  carveout attributes are validated advisory calls only; they cannot change a
  Metal pipeline's cache or bank organization.
- Memory-pool attributes exceed the allocator's current reuse behavior.
- A kernel that dereferences a device pointer it *loaded from device memory*,
  rather than received as a launch argument, needs
  `CUMETAL_USE_METAL_DEVICE_ADDRESSES=1`. Without it the load silently reads
  zeros for any allocation larger than a few KiB: no error is raised, because
  from Metal's side nothing invalid happened -- the allocation simply was not
  resident for that dispatch. This affects every framework that keeps a
  device-side table of pointers, which is the normal way to express a
  multi-buffer kernel: PhysX's descriptor structs
  ([feasibility notes](../physx-feasibility.md)) and AMReX's per-box `Array4`
  array ([AMReX demo](../../demos/amrex/README.md)) both hit it. The mode marks
  every live allocation resident on every dispatch, which costs cross-stream
  concurrency, so it is opt-in. Detecting the pattern at lowering time and
  warning is open work.

## Textures, surfaces, and printf

Texture/surface object lifecycle, arrays, copies, and selected source descriptor
helpers exist. Source `tex1Dfetch` now has numerical scalar/vector element reads
and clamp/zero-border coverage for tracked linear resources. Direct PTX
indirect-object `txq`/`suq` width, height, and depth queries have strict-lowering
and numerical Apple-GPU coverage. Direct PTX sampling, surface load/store/reduction,
other query attributes, static opaque-reference operands, native Metal texture ABI,
LOD/gradient/gather families, and remaining addressing/filtering modes do not.
Device `printf` has a bounded
buffer and 256-byte format limit. Focused Clang-ABI tests cover 32/64-bit
signed/unsigned integers, hex flags, `size_t`, pointers, promoted binary64
floating values, characters, fixed precision, and escaped percent signs
on both PTX backends. Dynamic `*` width/precision and bounded `%s` reads from
tracked allocations are also tested on both paths. Registration-backed writable
module strings are materialized on legacy PTX, typed PTX, and native AOT;
arbitrary untracked addresses are rejected safely as `[string]`. Embedded
read-only module-constant strings remain a gap. `cudaLimitPrintfFifoSize`
configures the bounded ring; format-only and multi-argument calls return their
CUDA parsed-argument counts even when capacity rejects a record, and a complete
retained prefix is drained. A statically null format returns CUDA's specified
`-1` without reserving or writing a record on legacy PTX, typed PTX, and native
AOT. Dynamically selected formats remain unsupported. CUDA's circular overwrite
policy and the `-2` internal-error return remain outside the proved subset.

## FP64 and atomics

`fast48` has roughly 48-bit significand precision but binary32 exponent range;
`wide48` extends range handling; `ieee64` is software. Observable IEEE exception
status is not fully integrated. See [FP64 policy](../fp64-policy.md).

Atomic support is form-specific. Wider, floating, system-scope, ordering, and
address-space combinations outside focused tests remain gaps. Successful header
compilation is not atomic contention proof.

## Device properties

Several reported CUDA properties are conservative or synthetic compatibility
values (for example compute capability 8.0, zero PCI identifiers, priority range
0/0). `cudaDeviceProp.maxBlocksPerMultiProcessor` reports 1 and
`regsPerMultiprocessor` mirrors `regsPerBlock` -- consistent with the
one-resident-block occupancy guarantee, not an NVIDIA architectural limit --
and both are also queryable through `cudaDeviceGetAttribute`/`cuDeviceGetAttribute`.
They must not be used as proof that the corresponding NVIDIA hardware
feature exists.

`cudaDeviceProp` uses the CUDA 12 field order and size (1032 bytes). Earlier
CuMetal releases used a private 680-byte layout, so anything compiled against
the old header reads the wrong fields and must be rebuilt. The texture and
surface limits describe CuMetal's buffer-backed arrays. The mipmapped, gather
and cubemap limits are 0 because those forms are not implemented.

Interprocess sharing (`cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle` and the
event equivalents) always returns `cudaErrorNotSupported`.

`cuLinkCreate`/`cuLinkAddData`/`cuLinkComplete` cover only a link of exactly one
CUBIN, PTX or fatbinary image, which completes to that same image. CuMetal has
no device-code linker, so a second input, a relocatable object or a library
such as `libcudadevrt.a` returns `CUDA_ERROR_NOT_SUPPORTED`.

`cuModuleLoadData` accepts PTX text that opens with `//` or `/* */` comments,
as NVRTC, nvcc and Clang output all do. `cuModuleGetGlobal` finds any global
recorded in the module's ABI sidecar, whether or not a kernel that uses it has
been looked up yet.

`cudaFuncGetAttributes.numRegs` is a positive virtual occupancy cost derived
from that register budget and the Metal pipeline's actual thread limit. It
allows clients such as Kokkos to calculate nonempty launch sizes; it is not
measured physical register usage or an NVIDIA occupancy prediction.

`cudaFuncAttributes.ptxVersion` and `binaryVersion` (and the driver's
`CU_FUNC_ATTRIBUTE_PTX_VERSION`/`BINARY_VERSION`) report the device's
synthetic compute capability (80), not the PTX ISA the kernel came from. CUB
and Thrust choose their tuning policy from this value; a zero made CCCL select
no policy and its algorithms silently returned nothing. Unknown function
attributes return `CUDA_ERROR_INVALID_VALUE`.

## Error codes and driver entry points

`cudaError_t` uses CUDA 12's numbering (`cudaErrorNotReady` = 600,
`cudaErrorInvalidDevice` = 101, `cudaErrorApiFailureBase` = 10000). Earlier
CuMetal releases numbered the enum privately, so a binary compiled against the
old header misreads every error code and must be rebuilt.

`cudaGetDriverEntryPoint`/`cudaGetDriverEntryPointByVersion` resolve through
`cuGetProcAddress`; a missing symbol returns `cudaSuccess` with
`cudaDriverEntryPointSymbolNotFound`. The CUDA 12 library/kernel API
(`cuLibrary*`, `cuKernel*`) and kernel graph nodes are declared in `cuda.h` so
code that references them compiles, but they are not exported and
`cuGetProcAddress` reports them as not found. `cuLaunchKernelEx` launches only
when no attribute other than `CU_LAUNCH_ATTRIBUTE_IGNORE` is supplied; any
real attribute (cluster dimension, programmatic stream serialization, priority)
returns `CUDA_ERROR_NOT_SUPPORTED`. `cuStreamGetId` and the
`CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` query are implemented.

CPU-backed library calls wait for the device before reading their inputs. That
wait leaves a failed earlier launch pending for the caller's own synchronize,
so a refused kernel does not turn into an unrelated `CUBLAS_STATUS_EXECUTION_FAILED`.

NVRTC's CUBIN size includes a trailing NUL, matching real NVRTC; CuPy drops
the last byte of every image it receives and the module image still loads.

### Device assertions in synchronized kernels

The typed PTX path recognizes a narrow, proven terminal assertion of
`blockDim.axis & (blockDim.axis - 1)`. It restricts accepted launches to a
power-of-two dimension and checks the constraint at kernel entry before any
barrier or divergent work. This includes straight-line assertion forwarding
helpers. The launch reports `cudaErrorLaunchFailure` on an invalid shape.
Five-argument void `__assertfail` calls in kernels without barriers,
collectives, or device printf use the per-launch trap status and report
`cudaErrorLaunchFailure`; divergent and spinning peers are cancelled.
General assertions in synchronized kernels still refuse. Assertion messages
and exact `cudaErrorAssert` status are not implemented.
The constraint also applies when the assertion's original path would have been
untaken, so this is a deliberately narrower supported launch domain.
