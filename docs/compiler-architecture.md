# CuMetal compiler architecture

CuMetal's compiler center is a verified, typed SSA GPU IR shared by the CUDA
source and PTX compatibility frontends. Typed MSL is the stable production
boundary; Apple's supported tools own final Metal code generation.

```text
CUDA C++ → Clang device LLVM/NVVM ─┐
                                    ├→ CuMetal GPU IR
PTX → parser → CFG → SSA ──────────┘
  → Metal legalization → structurization → typed MSL
  → xcrun metal → xcrun metallib
```

## Current migration slice

The `cumetal-ir` backend currently provides:

- core `Module`, `Function`, `BasicBlock`, operation, value and type objects;
- block arguments, dominance checking, source locations and textual dumps;
- explicit device, constant, threadgroup and private address spaces;
- pointer provenance and typed memory scopes/orderings;
- separate GPU-semantic and Metal-legalized operation sets;
- direct, module-local, acyclic device-call verification;
- PTX CFG construction and register-to-SSA import for the initial arithmetic,
  indexing, branch, memory, synchronization and warp subset;
- LLVM/NVVM import using LLVM's IR reader when LLVM 18 or newer is available;
- typed MSL types, expressions, statements, functions, parameters, attributes,
  deterministic printing and controlled provenance comments;
- an explicit Metal legalization pass and a conservative structurizability
  check;
- inspection output through `--emit=llvm`, `cumetal-ir`, `metal-ir`, and `msl`;
- a versioned source-native runtime registration ABI in
  `runtime/api/cumetal_native.h`.

The new backend never falls back to legacy lowering. Unsupported opcodes,
intrinsics, memory semantics, CFG shapes, or MSL constructs are compile-time
errors.

## Backend selection

During the compatibility release:

```sh
cumetalc kernel.cu --backend=cumetal-ir --emit=msl -o kernel.metal
cumetalc kernel.ptx --backend=cumetal-ir --emit=cumetal-ir -o kernel.cmir
cumetalc kernel.ptx --backend=legacy -o kernel.metallib
```

The registration JIT selects the new PTX backend only when
`CUMETAL_PTX_BACKEND=cumetal-ir` is set. Cache keys include the selected
frontend/backend policy and compiler schema. There is no automatic retry with
the legacy backend.

## Legality stages

After frontend import, only core and `gpu.*` operations are legal. Metal
legalization converts supported GPU operations to explicit `metal.*`
operations. The verifier rejects any `gpu.*` operation surviving that stage,
and the typed MSL backend rejects anything it cannot represent faithfully.

The initial structurizer accepts straight-line control flow and a conservative
conditional-return shape. Loops, general if/else regions, and irreducible CFGs
are rejected until their named region transformations are implemented.

## Provenance and semantic quality

Execution provenance is independent of semantic quality.

Supported provenance vocabulary:

```text
generic_nvvm_lowering
generic_ptx_lowering
library_substitution
workload_specialization
precompiled_metallib
cpu_fallback
unsupported
```

Semantic quality vocabulary:

```text
exact
tolerance_bounded
semantic_emulation
performance_degraded
cpu_fallback
unsupported
```

Generated MSL contains controlled metadata comments, and completed runtime
dispatch traces report both fields. Workload substitution and CPU fallback do
not count as generic compiler coverage.

## Source-native registration

Source AOT modules use the versioned `CuMetalModuleDescriptor` ABI. Logical CUDA
arguments and concrete Metal bindings are separate tables. A module embeds
metallib bytes and maps host stubs directly to kernel descriptors through
`cumetalRegisterModule` and `cumetalUnregisterModule`.

The runtime validates descriptor versions, argument/binding ranges, SIMD width,
and host-stub uniqueness before registration. `__cudaRegister*` and fatbinary
parsing remain compatibility-only.

## Default-backend gate

The new backend becomes the default only when its source and PTX conformance
sets meet or exceed the legacy generic pass count, report zero silent fallback,
and pass correctness-critical indexing, control-flow, ABI, address-space,
shared-memory, synchronization, atomic, warp and math tests. AIR emission
remains available for ABI research and regression inspection, not production.
