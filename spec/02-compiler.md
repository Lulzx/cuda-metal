# Compiler architecture

[Specification index](../spec.md)

## Canonical pipeline

```text
CUDA C++ -> Clang CUDA device LLVM/NVVM --+
                                           +-> verified CuMetal GPU IR
PTX -> parser -> CFG -> register SSA ------+
  -> Metal legalization -> CFG structurization -> typed MSL AST
  -> Apple metal/metallib -> .metallib
```

MSL is the stable production backend contract. Apple's public `metal` and
`metallib` tools produce deployable libraries. AIR inspection, validation, and
container generation remain separate research/regression tools.

## CuMetal IR

The shared IR must provide:

- typed SSA values and CFG blocks with block arguments;
- explicit device, constant, threadgroup, and private address spaces;
- pointer provenance through parameters, offsets, aggregates, memory, and CFG;
- explicit kernel ABI bindings and argument layout;
- typed memory scope and ordering for barriers, fences, and atomics;
- source locations and stable diagnostics;
- distinct generic GPU and Metal-legal operation families.

The verifier runs after import and every transformation. Undefined values,
invalid dominance, illegal casts, barriers or atomics, recursive/indirect device
calls, and unsupported ABI forms are hard failures.

## Frontends

- Direct `.cu` uses a stock CUDA-capable Clang device frontend and imports
  LLVM/NVVM into CuMetal IR.
- PTX uses the clean-room parser and constructs CFG/register SSA before
  normalization.
- PTX version headers do not grant support. Instructions are accepted only when
  their complete semantics are lowered or explicitly rejected.
- Binary containers are compatibility inputs only when a bounded parser can
  extract valid PTX. SASS-only input is rejected.

## Legalization and emission

Metal legalization is a hard boundary: generic GPU operations may not remain
after it, and Metal operations may not appear before it. CFG structurization
must preserve dominance, block arguments, masks, barriers, and exit behavior.
The typed MSL AST must preserve types and address spaces without textual
qualifier stripping.

## Backend migration

`--backend=legacy` may remain while typed coverage is incomplete. Defaults are
chosen per frontend and may change only when an executable corpus shows the
typed path matches or exceeds legacy production compilation and numerical
Apple-GPU correctness with no silent fallback. Compile counts and runtime proof
must be reported separately.

## Compilation cache and native AOT

Cache identity must cover input bytes, compiler version, normalized options,
GPU family, FP64 mode, and relevant toolchain identity. Corrupt or mismatched
entries must be rejected safely.

The target AOT architecture uses a versioned CuMetal-native registration and
launch ABI. NVIDIA `__cudaRegister*` and fatbinary handling remain compatibility
paths and must never be required by the final source-first architecture.
