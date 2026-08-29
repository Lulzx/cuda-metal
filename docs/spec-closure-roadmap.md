# Specification closure roadmap

This document reconciles every normative chapter indexed by `spec.md` with
current repository evidence. It is a forward-looking closure plan, not a
compatibility claim. Implemented surfaces and detailed limitations remain in
`status.md` and `known-gaps.md`; measured results remain in
`verified-results.md`.

## Current evidence boundary

- The manifest-driven NVIDIA sample gate contains 83 enrolled headless samples,
  all currently classified `pass` as of 2026-08-29. The authoritative inputs are
  `tests/cuda_projects/cuda_samples_sweep_manifest.txt` and
  `conformance_cuda_samples_sweep`. This proves only that enrolled snapshot and
  does not establish general CUDA compatibility or the spec's undefined "full
  conformance suite" denominator.
- The typed shared-IR migration gate is not closed. With CUDA Clang 21-23, the
  reviewed 23-file production-metallib matrix is 15/23 for direct `.cu` through
  `cumetal-ir` and 7/23 for PTX/`--cuda-device`, versus 23/23 for the legacy PTX
  backend. The authoritative reviewed manifest and gates are
  `tests/cuda_projects/backend_matrix_manifest.txt` and
  `conformance_compiler_backend_matrix{,_versions}`. The matrix is compile
  evidence only; promotion still requires numerical GPU tests.
- The three-kernel Phase 5 benchmark gate has measured Apple-GPU results below
  2x native Metal on the recorded Apple M4 Pro system. The reproducible gate is
  `bench_phase5_all_kernels`. It does not yet cover the broader release set
  required by [the verification specification](../spec/05-verification.md).
- llm.c, llama.cpp, PhysX, HiGHS, and multi-Xcode checks depend on external
  checkouts, assets, hardware, or toolchains. Their focused results are not a
  continuously enforced general-compatibility gate.
- `VF64-metal` is pinned at upstream `7290217`. Its CuMetal integration gate
  passes `fast48`, `wide48`, and `ieee64` on Apple M4 Pro. Recent upstream
  solver/resource-evidence commits do not change the linked support shader, so
  they strengthen provenance without expanding CuMetal's FP64 operation claims.

## Normative coverage map

Every normative requirement has at least one disposition: a durable boundary
that must remain enforced, an implemented subset whose regression gates must
remain green, or open work in the priority table below. A chapter can span more
than one disposition because implemented subsets and unclosed parity can
coexist.

| Normative chapter | Current disposition | Accountable evidence or roadmap row |
| --- | --- | --- |
| [Scope and principles](../spec/01-scope.md) | Durable platform, source-first, public-API, clean-room, fixed-warp, and explicit-failure boundaries remain release guardrails. Bounded compatibility areas remain subsets. | README architecture/limits, `known-gaps/platform.md`, P0 documentation integrity, and the relevant P2 subset rows. |
| [Compiler](../spec/02-compiler.md) | Typed IR/MSL is canonical but has not reached legacy coverage. The native source AOT ABI surface exists, but the executable path still uses CUDA registration and first-launch PTX lowering. Cache identity has focused tests and remains a regression invariant. | `conformance_compiler_backend_matrix_versions`, `unit_cumetal_ir`, `unit_native_registration`, `unit_module_cache`, and P1 canonical compiler path. |
| [Runtime](../spec/03-runtime.md) | Core allocation, pointer resolution, launch, stream/event, provenance, and per-thread error subsets are implemented and tested; advanced interactions remain bounded. | Runtime functional/unit tests, `status/runtime.md`, and P2 runtime semantic subsets/binary shim. |
| [CUDA semantics](../spec/04-semantics.md) | Fixed-width warp, memory, synchronization, atomic, FP64, graph, dynamic-launch, texture/surface, and `printf` behavior is form-specific rather than blanket compatible. | `known-gaps/runtime.md`, `fp64-policy.md`, P2 runtime semantic subsets, and P2 library rows. |
| [Verification](../spec/05-verification.md) | Evidence classes are separated, but the Phase 4 denominator, recurring CI, full Xcode matrix, and broader Phase 5 release set are open. | P1 Phase 4 correctness, P1 AIR ABI, and P3 Phase 5 performance. |
| [Roadmap](../spec/06-roadmap.md) | Priorities and closure criteria are instantiated by the table below. | This document and `unit_documentation_consistency`. |
| [Legal and clean room](../spec/07-legal.md) | No new compatibility work is authorized by the roadmap. Source-first packaging, public Apple APIs, clean-room headers, no SASS translation, attribution, and bounded opt-in binary language remain mandatory gates. The detailed legal/tooling notice still needs P0 reconciliation with the MSL production path and the spec's non-advice boundary. | `known-gaps/platform.md`, Release shim-off configuration, and P0 documentation integrity. |

The durable guardrails are not lower-priority work. A change that violates one
is a regression even if it improves a compatibility count.

## Prioritized closure work

| Priority | Spec area | Remaining work | Closure evidence |
| --- | --- | --- | --- |
| P0 | Documentation integrity | Keep `spec.md`, README, status, known gaps, intrinsic map, legal/tooling notes, and verified results consistent with executable manifests and source behavior. Remove historical statements once superseded instead of leaving contradictory snapshots. | `unit_documentation_consistency` derives sample/backend totals and checks high-risk boundaries. Release review also reconciles prose that cannot be derived mechanically. |
| P1 | Canonical compiler path | Close typed-IR gaps in dynamic shared memory, atomics/reductions, FP64, generic pointers, device calls, and residual CFGs until it meets the legacy generic-correctness gate with no fallback. Wire the versioned native registration/launch-stub AOT path that the spec describes; the current complete executable flow still uses CUDA registration and first-launch PTX lowering. Preserve cache identity/corruption rejection throughout the migration. | `conformance_compiler_backend_matrix_versions` plus numerical Apple-GPU tests show typed IR at least matches the legacy pass count across CUDA Clang 21-23; a source executable uses the native ABI with no first-launch PTX JIT; `unit_module_cache` remains green. Only then change remaining backend defaults. |
| P1 | Phase 4 correctness | Define the conformance denominator behind the 90% requirement; enable the hosted and self-hosted workflows; retain negative-path, numerical, and GPU-provenance gates. | Enabled Release/shim-off and Debug/shim-on workflows complete; the commissioned M-series lane reports pass/skip/fail separately; the named Phase 4 manifest reaches at least 90% without counting skips or waivers as passes. |
| P1 | AIR ABI | Exercise genuinely distinct Xcode 15.0, 15.4, 16.0, and 16.2+ toolchains. Current local matrix logic can detect duplicate toolchains but cannot supply the missing installations. | `air_abi_xcode_matrix_regression` records attributable validation and runtime-load results for every required toolchain, with no duplicate compiler identity counted as cross-version coverage. |
| P2 | Runtime semantic subsets | Expand graph node/cross-stream/allocator semantics; direct PTX texture/surface instructions; nested/overflow/error behavior for the host-drained dynamic-launch queue; irregular cooperative-group masks; wider device `printf`; and observable `ieee64` exception status. Preserve allocation/pointer, stream/event, launch ABI, provenance, and per-thread error invariants while doing so. | Focused positive and negative tests plus numerical Apple-GPU execution and unmodified workloads where available; every unsupported branch fails explicitly. |
| P2 | Phase 4.5 libraries | Audit pointer modes, stream ordering, graph capture, datatype/layout variants, and unsupported returns across cuBLAS, cuRAND, cuFFT, cuSPARSE, cuSOLVER, and cuDNN. | Per-library support tables generated from tested API cases; no CPU/UMA fallback is reported as GPU execution. |
| P2 | Binary shim | Add compressed PTX payloads and remaining bounded fatbinary variants without weakening range validation or the source-first default. Big-endian and SASS-only inputs remain explicit non-goals unless the spec changes. | Registration and Driver API parity tests for every accepted container plus malformed/truncated negative cases in Debug/shim-on and Release/shim-off builds. Release packaging must still omit the `libcuda.dylib` alias by default. |
| P3 | Phase 5 performance | Extend native-Metal comparisons beyond vector add, SAXPY, and reduction and decide which memory-bound kernels form the release gate. Implement allocator reuse/heap work only against measured bottlenecks. | Reproducible benchmark artifacts with device/toolchain provenance; selected memory-bound release set remains at or below 2x native Metal. |

## Reconciliation record

The 2026-08-29 reconciliation resolved these stale current-state claims:

- the old 33 pass / 3 waive / 47 nonpassing CUDA-sample snapshot;
- descriptions of dynamic launch, resident-grid cooperative launch, texture
  helpers, graph-memory samples, and persisting-L2 hints as wholly absent;
- the old production architecture diagram showing direct AIR emission;
- the FP64 table that predated `fast48`, `wide48`, and `ieee64`.

The following remain explicit P0/P1 work rather than silently accepted debt:

- keep legal and AIR/tooling notes aligned with MSL as the production contract
  and direct AIR/container generation as research/regression tooling;
- distinguish AIR matrix logic from actual access to four distinct Xcode
  toolchains;
- keep historical release/audit snapshots dated while current indexes derive
  their numerical headlines from executable manifests.

## Updating and closing rows

When evidence changes, update the authoritative manifest or measured record
first, run `unit_documentation_consistency`, then reconcile this roadmap,
`status.md`, `known-gaps.md`, `verified-results.md`, and README. Do not edit a
headline merely to make the consistency test pass.

A row can be removed from open work only after its complete closure-evidence
cell is satisfied. Partial progress belongs in status/gap pages and must not be
rewritten as closure. Durable boundaries stay in the normative coverage map
even when every open engineering row is complete.

## Working rule

A feature is closed only when source presence, tests, numerical correctness,
Apple-GPU provenance, and required environment coverage are each stated
separately. A passing enrolled sample does not close broader API parity, and a
manual external workload does not prove recurring CI behavior.
