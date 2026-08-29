# Compiler and toolchain status

[Status index](../status.md) · [Known compiler gaps](../known-gaps/compiler.md)

## Production paths

- `cumetalc file.cu -o program` builds a source-recompiled executable.
- Direct `.cu` defaults to typed CuMetal IR.
- PTX and `.cu --cuda-device` default to the broader legacy PTX backend.
- Both paths emit MSL and use Apple's public `metal`/`metallib` tools for
  production libraries.
- Direct AIR generation remains tooling/research only.

With CUDA Clang 21-23, the reviewed manifest-controlled 23-file
production-metallib matrix records:

| Frontend | Legacy | Typed CuMetal IR |
| --- | ---: | ---: |
| direct `.cu` | 0/23 | **18/23** |
| PTX / `--cuda-device` | **23/23** | **20/23** |

The legacy direct path is a qualifier-stripping prototype, not a fallback.
Matrix results prove compilation only. The versioned gate records each compiler
identity and requires the same manifest with CUDA Clang 21, 22, and 23.

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
Externally initialized direct-NVVM `__constant__` and writable `__device__`
symbols use explicit hidden Metal buffers instead of embedded zero initializers.
The constant buffer has checked aligned offsets and a 64 KiB limit; the writable
buffer remains persistent across launches. A focused Apple-GPU test checks both
host-populated constant reads and two-launch writable-symbol persistence.
It also recognizes compiler-marked implicit definitions, bounded private local
depots, static/dynamic shared symbols, and canonical one-block or
unconditional-header natural loops without routing barrier code through the CFG
dispatcher.
Referenced PTX module constants use the reserved constant-symbol buffer at
binding 30 with checked byte offsets, and the proven float `frexp` pattern
normalizes Clang's double-width call-slot ABI without admitting general FP64.
CUDA's double-signature `frexp` call is narrowed only for the proven
float-to-double-call-to-float pattern, preserving the integer exponent output
without admitting general FP64 arithmetic.
PTX memory operands retain literal byte displacements before typed load, store,
and atomic lowering. Relaxed CUDA system atomics use an explicit coherent-UMA
policy over tracked shared allocations; CAS retries spurious weak failures, and
signed min/max preserve their signed comparison domain.
Flat heterogeneous LLVM aggregates lower to typed MSL structs. The dynamic
cooperative-groups checks and raytracer CPU-reference comparison pass through
that direct typed path.

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
