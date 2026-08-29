# Compiler and toolchain status

[Status index](../status.md) · [Known compiler gaps](../known-gaps/compiler.md)

## Production paths

- `cumetalc file.cu -o program` builds a source-recompiled executable.
- Direct `.cu` defaults to typed CuMetal IR.
- PTX and `.cu --cuda-device` default to the broader legacy PTX backend.
- Both paths emit MSL and use Apple's public `metal`/`metallib` tools for
  production libraries.
- Direct AIR generation remains tooling/research only.

The manifest-controlled 23-file production-metallib matrix currently records:

| Frontend | Legacy | Typed CuMetal IR |
| --- | ---: | ---: |
| direct `.cu` | 0/23 | **9/23** |
| PTX / `--cuda-device` | **23/23** | **7/23** |

The legacy direct path is a qualifier-stripping prototype, not a fallback.
Matrix results prove compilation only.

## Typed representation

CuMetal IR provides typed SSA/CFG, block arguments, explicit address spaces,
pointer provenance, kernel ABI metadata, memory scope/order, verification, and
Metal legalization. The NVVM and PTX frontends share the typed MSL backend.

Recent typed coverage includes common FP32 libdevice calls, CFG/SSA loop joins,
source-level pointer flow, and host-populated device-pointer fields loaded from
device-backed descriptors. The descriptor rule applies only when no reachable
device-side store initializes the field; stored polymorphic pointers retain
their observed constraints.

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
