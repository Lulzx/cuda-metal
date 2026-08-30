# Specification closure roadmap

This document reconciles every normative chapter indexed by `spec.md` with
current repository evidence. It is a forward-looking closure plan, not a
compatibility claim. Implemented surfaces and detailed limitations remain in
`status.md` and `known-gaps.md`; measured results remain in
`verified-results.md`.

## Current evidence boundary

- The manifest-driven NVIDIA sample gate contains 83 enrolled headless samples,
  all currently classified `pass` as of 2026-08-30. The authoritative inputs are
  `tests/cuda_projects/cuda_samples_sweep_manifest.txt` and
  `conformance_cuda_samples_sweep`. This proves only that enrolled snapshot and
  does not establish general CUDA compatibility or replace the separate fixed
  185-test Phase 4 functional denominator.
- The reviewed Phase 4 functional denominator passed 185/185 with zero skips on
  Apple M4 Pro on 2026-08-30. This closes the local percentage gate, but it does
  not substitute for a recurring verification mechanism or a commissioned GPU
  lane. GitHub Actions workflows are intentionally absent.
- The typed shared-IR migration gate is not closed. With CUDA Clang 21-23, the
  reviewed 26-file production-metallib matrix is 26/26 for direct `.cu` and PTX
  through `cumetal-ir`, while the legacy PTX backend is 25/26 because it rejects
  the aggregate device-call ABI. The authoritative reviewed manifest and gates are
  `tests/cuda_projects/backend_matrix_manifest.txt` and
  `conformance_compiler_backend_matrix{,_versions}`. The matrix is compile
  evidence only; promotion still requires numerical GPU tests.
- The exact 24-project in-tree corpus now passes 24/24 through typed PTX and
  24/24 through direct native AOT on Apple M4 Pro with workload
  specializations disabled. Native ABI v3 carries kernel, symbol, and device
  `printf` metadata and performs no first-launch PTX JIT. This closes the
  reviewed numerical corpus, not the residual language combinations in P1.
- The manifest still contains 83 enrolled headless samples and the 2026-08-30
  rerun passes 83/83. `conjugateGradientMultiBlockCG` now uses a
  guaranteed-progress occupancy-derived cooperative grid and must satisfy its
  independent host equation error, while the PhysX friction, stacked, and
  triangle-mesh comparisons pass with aggregate PTX ABI sizes preserved in
  their metallib sidecars.
- The named five-kernel Phase 5 release set—vector add, SAXPY, STREAM copy,
  STREAM triad, and FP32 reduction—has measured Apple-GPU results below 2x
  native Metal on the recorded Apple M4 Pro system. The reproducible gate is
  `bench_phase5_all_kernels`; this closes the selected-set Phase 5 criterion,
  not a whole-suite performance bound.
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
| [Compiler](../spec/02-compiler.md) | Typed IR/MSL exceeds the reviewed legacy PTX compile count by the aggregate device-call case. The source executable path embeds its metallib and registers host launch stubs through the versioned native ABI without first-launch PTX JIT; broader compiler combinations and default promotion remain gated. Cache identity remains a regression invariant. | `conformance_compiler_backend_matrix_versions`, `functional_cumetalc_link_executable`, `unit_native_registration`, `unit_module_cache`, and P1 canonical compiler path. |
| [Runtime](../spec/03-runtime.md) | Core allocation, pointer resolution, launch, stream/event, provenance, and per-thread error subsets are implemented and tested; advanced interactions remain bounded. | Runtime functional/unit tests, `status/runtime.md`, and P2 runtime semantic subsets/binary shim. |
| [CUDA semantics](../spec/04-semantics.md) | Fixed-width warp, memory, synchronization, atomic, FP64, graph, dynamic-launch, texture/surface, and `printf` behavior is form-specific rather than blanket compatible. | `known-gaps/runtime.md`, `fp64-policy.md`, P2 runtime semantic subsets, and P2 library rows. |
| [Verification](../spec/05-verification.md) | Evidence classes are separated, the Phase 4 denominator is fixed, and the named five-kernel Phase 5 release set passes; observed recurring verification and the full Xcode matrix remain open. | P1 Phase 4 correctness, P1 AIR ABI, and the Phase 5 closure record below. |
| [Roadmap](../spec/06-roadmap.md) | Priorities and closure criteria are instantiated by the table below. | This document and `unit_documentation_consistency`. |
| [Legal and clean room](../spec/07-legal.md) | Source-first packaging, public Apple APIs, clean-room headers, no SASS translation, attribution, bounded opt-in binary language, and the non-advice boundary are mandatory gates. The detailed notice identifies typed MSL through public Apple tools as the production contract and direct AIR/container generation as research/regression tooling. | `legal-notice.md`, `known-gaps/platform.md`, Release shim-off configuration, and `unit_documentation_consistency`. |

The durable guardrails are not lower-priority work. A change that violates one
is a regression even if it improves a compatibility count.

## Prioritized closure work

| Priority | Spec area | Remaining work | Closure evidence |
| --- | --- | --- | --- |
| P0 | Documentation integrity | Keep `spec.md`, README, status, known gaps, intrinsic map, legal/tooling notes, and verified results consistent with executable manifests and source behavior. Remove historical statements once superseded instead of leaving contradictory snapshots. | `unit_documentation_consistency` derives sample/backend totals and checks high-risk boundaries. Release review also reconciles prose that cannot be derived mechanically. |
| P1 | Canonical compiler path | Expand residual generic-pointer/device-call/CFG combinations beyond the 26-file compile corpus. Direct scalar-return helpers with pointer arguments, loops, merges, early exits, transitive `printf` state, and a flat 12-byte by-value/single aggregate-return ABI are proven; nested or irregular aggregates, multi-result/indirect call forms, compiler-promoted module-global aggregate literals, and broader barrier CFG remain open. Preserve native-AOT symbol metadata and cache identity/corruption rejection throughout the migration. | `conformance_compiler_backend_matrix_versions` plus numerical Apple-GPU tests show typed IR exceeds the reviewed legacy PTX pass count across CUDA Clang 21-23; `functional_cuda_projects_{device_calls,aggregate_device_calls}_typed_ptx` and the exact typed-PTX/native-AOT corpora prove the direct-call subsets without workload specialization; `functional_cumetalc_link_executable` proves an embedded native module with no `__cudaRegister*` dependency, no first-launch PTX JIT, and no registration-JIT cache; `functional_cumetalc_native_aot_symbols` proves host copies and persistent GPU writes for constant/device globals; `unit_module_cache` remains green. Only then change remaining backend defaults. |
| P1 | Phase 4 correctness | Keep the fixed 185-test denominator reviewed; establish recurring verification outside GitHub Actions and commission the Apple-GPU lane; retain negative-path, numerical, and GPU-provenance gates. | Recorded Release/shim-off and Debug/shim-on runs complete; the commissioned M-series lane reports pass/skip/fail separately; the named Phase 4 manifest reaches at least 90% without counting skips or waivers as passes. |
| P1 | AIR ABI | Exercise genuinely distinct Xcode 15.0, 15.4, 16.0, and 16.2+ toolchains. Current local matrix logic can detect duplicate toolchains but cannot supply the missing installations. | `air_abi_xcode_matrix_regression` records attributable validation and runtime-load results for every required toolchain, with no duplicate compiler identity counted as cross-version coverage. |
| P2 | Runtime semantic subsets | Expand remaining graph node/multi-stream-topology/allocator semantics; direct PTX texture/surface instructions; module-constant/untracked device-string `printf` and complete overflow-return parity; and observable `ieee64` exception status. Preserve the now-tested event-linked two-stream graph replay, contiguous/irregular group masks, nested/invalid/overflow dynamic-launch queue, and wide scalar, dynamic-width/precision, plus bounded tracked-string `printf` behavior along with allocation/pointer, stream/event, launch ABI, provenance, and per-thread error invariants. | Focused positive and negative tests plus numerical Apple-GPU execution and unmodified workloads where available; every unsupported branch fails explicitly. |
| P2 | Phase 4.5 libraries | Audit pointer modes, stream ordering, graph capture, datatype/layout variants, and unsupported returns across cuBLAS, cuRAND, cuFFT, cuSPARSE, cuSOLVER, and cuDNN. | Per-library support tables generated from tested API cases; no CPU/UMA fallback is reported as GPU execution. |
| P2 | Binary shim | Add remaining bounded fatbinary variants without weakening range validation or the source-first default. Preserve version-`0x0101` LZ4/Zstd PTX entry support, its 64 MiB decompression ceiling, and direct/ELF registration/Driver parity. Other entry versions/codecs remain explicit gaps. Big-endian and SASS-only inputs remain explicit non-goals unless the spec changes. | `functional_{driver_module_load_data_ptx,runtime_registration_fatbin_ptx}` execute every accepted compressed form on Apple GPU and reject malformed/truncated/oversized forms; keep them green in Debug/shim-on and Release/shim-off builds. Release packaging must still omit the `libcuda.dylib` alias by default. |

## Closed during this reconciliation

- **P3 Phase 5 performance:** the release set is explicitly vector add, SAXPY,
  STREAM copy, STREAM triad, and FP32 reduction. The reproducible
  `bench_phase5_all_kernels` gate passed all five below 2x native Metal on Apple
  M4 Pro with Metal 4, Xcode 26.6, and Apple Metal compiler 32023.883 on
  2026-08-29. Allocator/heap work remains measurement-driven rather than an
  unqualified compatibility requirement.

## Reconciliation record

The 2026-08-29 reconciliation resolved these stale current-state claims:

- the old 33 pass / 3 waive / 47 nonpassing CUDA-sample snapshot;
- descriptions of dynamic launch, resident-grid cooperative launch, texture
  helpers, graph-memory samples, and persisting-L2 hints as wholly absent;
- the old production architecture diagram showing direct AIR emission;
- the FP64 table that predated `fast48`, `wide48`, and `ieee64`.

The legal/tooling notice now matches MSL as the production contract, treats direct
AIR/container generation as research/regression tooling, and records engineering policy
without making legal determinations. The following remain explicit P0/P1 work rather than
silently accepted debt:

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
