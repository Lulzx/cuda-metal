# Compiler and toolchain status

[Status index](../status.md) · [Known compiler gaps](../known-gaps/compiler.md)

## Production paths

- `cumetalc file.cu -o program` builds a native-AOT source executable with an
  embedded metallib and versioned CuMetal registration descriptor; it has no
  `__cudaRegister*` or first-launch PTX-JIT dependency.
- Direct `.cu` defaults to typed CuMetal IR.
- PTX and `.cu --cuda-device` default to the broader legacy PTX backend.
- Both paths emit MSL and use Apple's public `metal`/`metallib` tools for
  production libraries.
- Direct AIR generation remains tooling/research only.

With CUDA Clang 21-23, the reviewed manifest-controlled 24-file
production-metallib matrix records:

| Frontend | Legacy | Typed CuMetal IR |
| --- | ---: | ---: |
| direct `.cu` | 0/24 | **24/24** |
| PTX / `--cuda-device` | **24/24** | **24/24** |

The legacy direct path is a qualifier-stripping prototype, not a fallback.
Matrix results prove compilation only. The versioned gate records each compiler
identity and requires the same manifest with CUDA Clang 21, 22, and 23.

The separate exact `coverage_manifest.json` numerical corpus passes all 22
projects through both typed PTX and direct native AOT on Apple M4 Pro. Both
gates disable workload specializations and require every enrolled project to
pass. The native-AOT gate launches embedded metallibs without registration JIT
or first-launch PTX compilation. Device `printf` uses native ABI version 3 so
its format table is embedded with each kernel descriptor and drained by the
ordinary runtime ring-record path.

## Typed representation

CuMetal IR provides typed SSA/CFG, block arguments, explicit address spaces,
pointer provenance, kernel ABI metadata, memory scope/order, verification, and
Metal legalization. The NVVM and PTX frontends share the typed MSL backend.

Recent typed coverage includes common FP32 libdevice calls, CFG/SSA loop joins,
source-level pointer flow, and host-populated device-pointer fields loaded from
device-backed descriptors. The descriptor rule applies only when no reachable
device-side store initializes the field; stored polymorphic pointers retain
their observed constraints. Runtime-sized `extern __shared__` storage is emitted
as Metal threadgroup binding 0; the reduction and softmax corpus cases have
numerical Apple-GPU proof. The direct typed libdevice surface passes its 42/42
per-function numerical harness, including explicit typed expansions for Metal
functions without direct equivalents.
The PTX typed path now preserves Clang call-sequence parameter slots and FP32
bit containers for that same libdevice surface. Direct and PTX-produced typed
artifacts have separate Apple-GPU numerical gates for the complete 32-bit CUDA
atomic family, device/threadgroup fences, and host-concurrent system atomics;
the broader libdevice numerical proof remains attributed to the direct path.
Both typed frontends also decode constant-format Clang `vprintf` into the same
bounded atomic ring-record ABI. Focused Apple-GPU tests validate every record
from a 32-lane multidimensional launch and prove a capacity-boundary record is
rejected without payload writes; unresolved formats and unsupported tuple
widths fail explicitly.
Both typed frontends lower FP64 values as raw binary64 storage and call private
software-ALU helpers in the kernel translation unit. The `fp64_precision`
corpus passes independently produced direct-NVVM and typed-PTX metallibs on
Apple GPU, covering the documented `fast48` precision/range contract, special
values, chained arithmetic, shared memory, 32-lane shuffles, store/reload,
`uint64_t` aliasing, comparisons, libdevice calls, and rounding. This does not
change the binary32 exponent-range limitation or make Metal FP64 native.
Externally initialized direct-NVVM `__constant__` and writable `__device__`
symbols use explicit hidden Metal buffers instead of embedded zero initializers.
The constant buffer has checked aligned offsets and a 64 KiB limit; the writable
buffer remains persistent across launches. A focused Apple-GPU test checks both
host-populated constant reads and two-launch writable-symbol persistence.
It also recognizes compiler-marked implicit definitions, bounded private local
depots, static/dynamic shared symbols, and canonical one-block or
unconditional-header natural loops without routing barrier code through the CFG
dispatcher.
Sequential natural loops now retain PHI values assigned by the preceding loop,
including the nested loop sequences in the reviewed 2D block-tiled SGEMM and
FlashAttention. Their typed direct metallibs have focused numerical Apple-GPU
comparisons.
Referenced PTX module constants use the reserved constant-symbol buffer at
binding 30 with checked byte offsets, and the proven float `frexp` pattern
normalizes Clang's double-width call-slot ABI without admitting general FP64.
CUDA's double-signature `frexp` call is narrowed only for the proven
float-to-double-call-to-float pattern, preserving the integer exponent output.
PTX memory operands retain literal byte displacements before typed load, store,
and atomic lowering. Relaxed CUDA system atomics use an explicit coherent-UMA
policy over tracked shared allocations; CAS retries spurious weak failures, and
signed min/max preserve their signed comparison domain.
Flat heterogeneous LLVM aggregates lower to typed MSL structs. The dynamic
cooperative-groups checks and raytracer CPU-reference comparison pass through
that direct typed path. Native AOT also distinguishes pointer-valued launch
arguments from by-value aggregates lowered through pointer-shaped host ABI
parameters, propagates hidden clock/grid-barrier state through device helpers,
and emits a device-fenced resident cooperative-grid barrier.

## PTX compatibility

The legacy PTX path covers the enrolled CUDA sample and project workloads more
broadly. It includes control flow, memory, synchronization, warp operations,
atomics, constants/globals, bounded dynamic launch, graph-related workloads,
and software FP64 paths exercised by focused tests. Instruction support is
semantic and per-form; an accepted PTX version header is not blanket support.

## Tools

- `cumetalc`: frontend and emission driver
- `cumetal-ptx2llvm`: legacy PTX/LLVM inspection
- `air_inspect` and `air_validate`: metallib structure and ABI checks
- `cumetal-air-emitter`: AIR/container research
- `ptx_diff`: PTX-related comparison
- `cumetal_bench`: native-Metal performance comparison

Detailed compiler selection and legality stages are in
[compiler architecture](../compiler-architecture.md).
