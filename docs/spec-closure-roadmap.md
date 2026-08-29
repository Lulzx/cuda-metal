# Specification closure roadmap

This document reconciles the canonical requirements in `spec.md` with current
repository evidence. It is a forward-looking closure plan, not a compatibility
claim. Implemented surfaces and detailed limitations remain in `status.md` and
`known-gaps.md`; measured results remain in `verified-results.md`.

## Current evidence boundary

- The manifest-driven NVIDIA sample gate contains 83 enrolled headless samples,
  all currently classified `pass`. This proves only that enrolled snapshot and
  does not establish general CUDA compatibility or the spec's undefined "full
  conformance suite" denominator.
- The typed shared-IR migration gate is not closed. The reproducible 23-file
  production-metallib matrix is 9/23 for direct `.cu` through `cumetal-ir` and
  6/23 for PTX/`--cuda-device`, versus 23/23 for the legacy PTX backend. The
  matrix is compile evidence only; promotion still requires numerical GPU tests.
- The three-kernel Phase 5 benchmark gate has measured Apple-GPU results below
  2x native Metal. It does not yet cover the broader release set required by
  [the verification specification](../spec/05-verification.md).
- llm.c, llama.cpp, PhysX, HiGHS, and multi-Xcode checks depend on external
  checkouts, assets, hardware, or toolchains. Their focused results are not a
  continuously enforced general-compatibility gate.
- `VF64-metal` is pinned at upstream `7290217`. Its CuMetal integration gate
  passes `fast48`, `wide48`, and `ieee64` on Apple M4 Pro. Recent upstream
  solver/resource-evidence commits do not change the linked support shader, so
  they strengthen provenance without expanding CuMetal's FP64 operation claims.

## Prioritized closure work

| Priority | Spec area | Remaining work | Closure evidence |
| --- | --- | --- | --- |
| P0 | Documentation integrity | Keep `spec.md`, README, status, known gaps, intrinsic map, and verified results consistent with executable manifests and source behavior. Remove historical statements once superseded instead of leaving contradictory snapshots. | A documentation consistency test derives sample totals and checks high-risk capability statements. |
| P1 | Canonical compiler path | Close typed-IR gaps in dynamic shared memory, atomics/reductions, FP64, generic pointers, device calls, and residual CFGs until it meets the legacy generic-correctness gate with no fallback. Wire the versioned native registration/launch-stub AOT path that the spec describes; the current complete executable flow still uses CUDA registration and first-launch PTX lowering. | A committed corpus matrix and numerical Apple-GPU tests show typed IR at least matches the legacy pass count; a source executable uses the native ABI with no first-launch PTX JIT; only then change remaining backend defaults. |
| P1 | Phase 4 correctness | Define the conformance denominator behind the 90% requirement; enable the hosted and self-hosted workflows; retain negative-path, numerical, and GPU-provenance gates. | Release/shim-off and debug/shim-on CI run; the M-series lane reports pass/skip/fail separately; the defined Phase 4 denominator is at least 90%. |
| P1 | AIR ABI | Exercise genuinely distinct Xcode 15.0, 15.4, 16.0, and 16.2+ toolchains. Current local matrix logic can detect duplicate toolchains but cannot supply the missing installations. | Attributable validation and runtime-load records for every required toolchain, with no duplicate compiler versions counted as cross-version coverage. |
| P2 | Runtime semantic subsets | Expand graph node/cross-stream/allocator semantics; direct PTX texture/surface instructions; nested/overflow/error behavior for the host-drained dynamic-launch queue; irregular cooperative-group masks; wider device `printf`; and observable `ieee64` exception status. | Focused positive and negative tests plus unmodified workloads where available; every unsupported branch fails explicitly. |
| P2 | Phase 4.5 libraries | Audit pointer modes, stream ordering, graph capture, datatype/layout variants, and unsupported returns across cuBLAS, cuRAND, cuFFT, cuSPARSE, cuSOLVER, and cuDNN. | Per-library support tables generated from tested API cases; no CPU/UMA fallback is reported as GPU execution. |
| P2 | Binary shim | Add compressed PTX payloads and remaining bounded fatbinary variants without weakening range validation or the source-first default. Big-endian and SASS-only inputs remain explicit non-goals unless the spec changes. | Registration and Driver API parity tests for every accepted container plus malformed/truncated negative cases in shim-on and shim-off builds. |
| P3 | Phase 5 performance | Extend native-Metal comparisons beyond vector add, SAXPY, and reduction and decide which memory-bound kernels form the release gate. Implement allocator reuse/heap work only against measured bottlenecks. | Reproducible benchmark artifacts with device/toolchain provenance; selected memory-bound release set remains at or below 2x native Metal. |

## Documentation debt found in this audit

- Several pages retained the old 33 pass / 3 waive / 47 nonpassing CUDA-sample
  snapshot after the manifest moved to 83/0/0.
- Historical sections still described dynamic parallelism, resident-grid
  cooperative launch, texture helpers, graph-memory samples, and persisting-L2
  hints as absent after bounded implementations landed.
- `spec.md`'s old architecture diagram contradicted its normative amendment
  by showing direct AIR emission as the production pipeline.
- The spec's FP64 table predated `fast48`, `wide48`, and `ieee64`.
- AIR cross-version claims need to distinguish matrix logic from actual access
  to four distinct Xcode toolchains.

## Working rule

A feature is closed only when source presence, tests, numerical correctness,
Apple-GPU provenance, and required environment coverage are each stated
separately. A passing enrolled sample does not close broader API parity, and a
manual external workload does not prove recurring CI behavior.
