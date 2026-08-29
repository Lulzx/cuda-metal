# CUDA semantic contracts

[Specification index](../spec.md)

## Warp and masks

Warp width is always 32. Shuffle, vote, ballot, match, and reduction operations
must honor CUDA participation masks and inactive-lane behavior for the supported
forms. A full-width Metal primitive is not sufficient evidence for irregular
masks.

## Memory and synchronization

- CUDA block scope maps to a Metal threadgroup; warp scope maps to a SIMD-group.
- Address spaces must remain explicit through loads, stores, aggregates, calls,
  shared staging, and CFG merges.
- Barrier and fence lowering must preserve memory scope and ordering.
- Metal has no general cross-threadgroup barrier. A cooperative-grid barrier is
  valid only for a resident grid capped by a conservative device limit; larger
  or non-cooperative use is rejected.

## Atomics

An atomic is supported only when operation, width, type, address space, scope,
ordering, return value, and contention behavior are all implemented. System
scope and wide/floating atomics require focused negative and contention tests.

## FP64

CUDA-visible storage remains ordinary binary64. Execution modes are explicit:

| Mode | Contract |
| --- | --- |
| `fast48` | pair-based, roughly 48-bit significand, binary32 exponent range |
| `wide48` | pair-based, roughly 48-bit significand with binary64 range handling |
| `ieee64` | software binary64 core with correctly rounded integrated operations |
| `native` | research only unless the selected public Metal target accepts FP64 |

Mode selection must never silently fall back. Pair representation, `mov.b64` /
`uint64_t`, memory, shared memory, shuffle, and reload observability must retain
the binary64 storage ABI. Reduced-precision modes must identify themselves as
reduced precision. `ieee64` is not complete until required arithmetic,
conversions, memory paths, and observable exception status are integrated and
tested. `docs/fp64-policy.md` holds the current operation-level boundary.

## Graphs, dynamic launch, textures, and printf

- Graph support is node- and topology-specific. Capture, clone, update, memory
  lifetime, and cross-stream behavior require separate evidence.
- Dynamic launch may use a bounded device record queue and host drain. Queue
  overflow, nesting, ordering, and child errors must be observable.
- Source descriptor texture/surface helpers may be supported independently of
  PTX `tex.*`, `suld.*`, and `sust.*`; documentation must distinguish them.
- Device `printf` is bounded. Format, argument, truncation, overflow, ordering,
  and drain behavior must be tested and documented.

## Library shims

cuBLAS/cublasLt, cuRAND, cuFFT, cuSPARSE, cuSOLVER, cuDNN, NCCL, NVML, and
Thrust surfaces are compatibility subsets. Each API needs tested datatype,
layout, pointer-mode, stream, capture, error, and provenance behavior. CPU/UMA
implementation must not be described as GPU execution.
