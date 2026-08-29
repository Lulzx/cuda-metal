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
- a versioned source-native runtime registration ABI surface in
  `runtime/api/cumetal_native.h`; it is not yet wired into the default source
  executable path.

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

The legacy direct PTX-to-MSL compatibility path derives Metal element indices
from PTX byte addresses. It preserves both a derived pointer's byte stride and
literal byte displacement; for example, `gid * 8 + {0,4}` becomes adjacent
scalar indices rather than two aliases of `gid`. A focused regression covers
the `float2` access shape used by NVIDIA's `simpleCUFFT` sample.

### Command-line outputs and policy

`cumetalc` exposes the useful intermediate and final stages directly:

```bash
cumetalc kernel.cu --emit=cumetal-ir -o kernel.cumetal
cumetalc kernel.cu --emit=msl        -o kernel.metal
cumetalc kernel.cu --emit=metallib   -o kernel.metallib
cumetalc kernel.cu --emit=exe        -o kernel
```

| Switch | Meaning |
| --- | --- |
| `--backend=cumetal-ir\|legacy` | Select the typed shared-IR backend or compatibility backend. There is no silent fallback. |
| `--cuda-device` | Ask a CUDA-capable Clang to produce PTX before CuMetal lowering. |
| `--entry NAME` | Compile one kernel and its reachable device-call closure. |
| `--ptx-strict` | Reject unsupported PTX rather than tolerating it. |
| `--fp64=fast48\|wide48\|ieee64\|native\|emulate\|warn` | Select the virtual FP64 policy; direct typed `.cu` compilation and runtime/JIT default to `fast48` (`emulate` alias), while offline PTX compatibility retains `native`. |
| `--save-temps` | Retain link intermediates. |

The default follows measured production-metallib compilation coverage rather
than treating either backend as universally complete. The reviewed table below
uses CUDA Clang 21-23:

| Input corpus | `legacy` | `cumetal-ir` |
| --- | ---: | ---: |
| direct `.cu` | 0/24 | **24/24** |
| `.cu --cuda-device` / PTX | **24/24** | **24/24** |

Direct `.cu` therefore defaults to `cumetal-ir`; PTX and `--cuda-device`
default to `legacy`. Reproduce the reviewed per-file baseline with:

```bash
ctest --test-dir build -R '^conformance_compiler_backend_matrix$' --output-on-failure
```

The matrix proves that a backend produced and validated a production metallib;
it does not prove numerical correctness or GPU execution. Default-backend
promotion additionally requires the runtime evidence in the gate below.
`conformance_compiler_backend_matrix_versions` repeats the complete matrix with
CUDA Clang 21, 22, and 23 and records each compiler identity.

## Legality stages

After frontend import, only core and `gpu.*` operations are legal. Metal
legalization converts supported GPU operations to explicit `metal.*`
operations. The verifier rejects any `gpu.*` operation surviving that stage,
and the typed MSL backend rejects anything it cannot represent faithfully.

The structurizer accepts tested straight-line, conditional, and natural-loop
shapes and has a dispatcher fallback for selected barrier-free CFGs. Residual
loops, barrier-containing regions, and irreducible/general CFGs that cannot be
represented safely are rejected.

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

The versioned `CuMetalModuleDescriptor` ABI is implemented and used by linked
source executables. Logical CUDA arguments and concrete Metal bindings are separate
tables. It embeds metallib bytes and maps host stubs directly to kernel
descriptors through `cumetalRegisterModule` and `cumetalUnregisterModule`.

The runtime validates descriptor versions, argument/binding ranges, SIMD width,
and host-stub uniqueness before registration. `cumetalc file.cu -o program`
compiles host-only Clang launch stubs, embeds the typed direct-path metallib in a
generated native descriptor, and links both against `libcumetal`. Its executable
has no `__cudaRegister*` dependency and creates no registration-JIT cache on a
cold launch. CUDA registration and fatbinary parsing remain compatibility paths.

Native ABI version 2 describes constant and writable CUDA module globals,
including per-kernel symbol references, constant-buffer offsets, host shadows,
and persistent writable Metal buffers. Non-zero read-only tables remain
embedded directly in MSL.

## Default-backend gate

The new backend becomes the default only when its source and PTX conformance
sets meet or exceed the legacy generic pass count, report zero silent fallback,
and pass correctness-critical indexing, control-flow, ABI, address-space,
shared-memory, synchronization, atomic, warp and math tests. AIR emission
remains available for ABI research and regression inspection, not production.
