# Issue #76 follow-up acceptance

This file separates compiler regression coverage from complete workload acceptance.
The supported proof and conversion gates have focused GPU validation. Native
legacy GPU acceptance remains incomplete. A focused GPU pass or emitted Metal
does not close a downstream workload ticket.

## Implemented gates

| Gate | Executable coverage | Evidence so far |
| --- | --- | --- |
| Aligned private-address OR | `ptx_address_alignment_test.cpp`, `run_ptx_address_alignment.py` | Unit and five 65-input GPU cases pass; all five original LLVM7 joins advance. |
| Guarded scalar and pointer store ranges | `ptx_scalar_ranges_test.cpp`, `ptx_pointer_ranges_test.cpp`, `run_ptx_memory_ranges.py` | Actual incoming-edge bounds, dynamic loop seeds, invariant pointer ends and copied/joined lengths; bypassed, overwritten, stale and complemented predicates remain negative controls. |
| Initialized memory prefix ranges | `ptx_local_memory_ranges_test.cpp`, `run_ptx_memory_ranges.py` | Finite pointer loops, scalar bytes beside pointer lanes, unknown-path exploration, effects and explicit resource limits. Ten 65-input GPU cases and sixteen rejection controls pass, including full array results for reversal and pointer/scalar loops. |
| Concrete pointer-cell refinement | `ptx_pointer_cell_refinement_test.cpp`, `run_ptx_pointer_cell_refinement.py` | Constant/device/private cells pass three 65-input GPU cases; mixed spaces, missing initialization and partial stores reject. |
| Wider conversion storage | `run_ptx_cvt_storage.py` | All 24 numerical cases pass, 65 inputs each, with twelve rejection controls. Includes signed/unsigned destination extension, copied/joined values, f32 bits in b64, float round trips, f16 extraction and exact float store width. |
| Exceptional f16/f32-to-integer conversion | `run_ptx_float_integer.py` | All 24 GPU fixtures pass: 144 source/rounding/destination/form contracts, 65 raw-bit inputs each, six exact output words per lane. Covers NaNs, infinities, threshold neighbors, ties, signed zero, subnormals and f32 FTZ. |
| Type-proof exhaustion | `ptx_type_proof_bounds_test.cpp` | Tight explicit solver limit rejects; sufficient limit passes the same defined program. Copy-ring tests reject reachable pointer-cell overwrites beyond early discovery rounds. |
| Legacy declaration widths | `ptx_legacy_register_width_test.cpp`, `run_ptx_legacy_register_width.py` | Declaration widths replace name-dependent storage hints; independent LLVM assembly passes the six original ReLU forms. |

All numerical fixtures check independent CPU values, exact ordered ABI, unchanged
input bytes, output guards and actual generic Apple-GPU launch provenance. The
conversion and pointer-cell runners retain input/output artifacts when requested.
Source/API-level proof tests are host tests, not GPU execution evidence.

## Bounded analysis contracts

Alignment evidence comes from an actual local allocation, never from numeric
register width or a guessed pointer value. Copies and joins must agree; unknown
or exhausted proofs do not permit OR-to-add rewriting.

Memory intervals discharge **non-overlap only**. For discovered pointer-cell
candidates, an initializing pointer store is checked separately against every
incoming path. Unknown calls/writes, partial overlaps and conflicting concrete
address spaces remain rejection cases within that proof. The separate escaped
cell discovery omission is recorded in [known gaps](known-gaps.md).
The prefix proof tracks known initialized scalar bytes and symbolic local
addresses. It never reads absent bytes as zero or replaces program execution.
Unknown branches are explored; incomplete exploration discards all observations.
Liveness pruning removes only register facts unused on every remaining path and
retains all memory bytes. Conditional definitions do not kill old facts.

The actual SSA type solver has a tightening-only test option,
`PtxImportOptions::type_solver_step_limit`. Zero preserves the normal limit;
positive values can lower, not raise, that limit. Separate private analysis APIs
expose alignment and prefix budgets for direct negative tests. CFG clone/live-fact
budgets retain their independent existing tests.

## Original ReLU acceptance remains incomplete

`run_ptx_reuse_acceptance.py` runs all six fixtures on both backends, retaining
failures and enforcing the 12-cell denominator. Each numerical cell has 65
exact binary32 input/reference values, 63 inactive tail lanes, exact ABI and
buffer guards. The measured candidate result is **7 numerical passes / 5 legacy
translation failures**: six typed forms and the original straight-line legacy
form pass. General CFG is unsupported in legacy direct MSL.

The separate native legacy PTX-to-LLVM path now emits valid LLVM for all six
forms, including renamed 64-bit registers. LLVM assembly is not native GPU
acceptance. This machine has Command Line Tools but no Xcode or offline
`metal`/`metallib` toolchain; the native AIR/metallib numerical route remains
unverified. Typed output is never substituted for that backend's result.

```sh
python3 tests/functional/run_ptx_reuse_acceptance.py /path/to/build \
  --stage numerical --output /path/to/evidence/relu
```

## Conversion semantics and limits

The tests establish instruction-format chopping, destination extension and
bit reinterpretation for the listed scalar forms. Generic f16/f32-to-integer
lowering now clamps to the destination range, handles NaN according to the PTX
destination format, and preserves all four integer-rounding modes. The independent
reference decodes raw IEEE bits using integer arithmetic. Invalid numeric casts
remain inside the in-range branch; Metal `select` is not used for this purpose.

- Saturating modifiers and directed integer-to-f32 rounding beyond RN are rejected
  explicitly, including wider destination containers. No MSL/ABI is published.
- The full FP64 conversion matrix under each emulation mode remains outside this
  measured gate. Its existing separate software conversion path is unchanged.

These are operation-semantics contracts, separate from correct SSA types. Neither
a host-language cast nor an unrelated finite-input pass serves as the reference.

## Full inputs and publication

Full replays retain the original producer `afe80210` PTX from Actions run
35055622652, with each compiler and input hash recorded. Clearing an original
error is recorded separately from later undefined-value, ABI, memory-proof or
Apple-compiler failures. The publication owner records final measured revisions,
Release/Debug results, exclusions and downstream stages in the linked issue/PR.
No fresh all-workload pass or upstream integration is claimed by this document.

The two Shallenge memory proofs now advance to separate guarded-payload (LLVM21)
and heterogeneous-vector pointer (LLVM7) gaps. Solana's aligned-address, reversal,
pointer-iterator and dynamic scalar-start cases advance to a helper-call effect
barrier. Caller-memory write footprints and escaped cells outside the normalization
proof path need separate acceptance; these tests do not establish arbitrary
interprocedural alias analysis.

## Recorded local validation

Measured on Apple M5 / macOS 26.6.2, 2026-09-16, using this follow-up source.
Both configurations built successfully. This is a development build, not a new
immutable downstream pin.

| Configuration | Selected unit checks | PTX functional checks |
| --- | --- | --- |
| Release, binary shim OFF | 54 passed, 1 environment skip | 55 passed, 1 offline-toolchain skip |
| Debug, binary shim ON | 57 passed, 1 environment skip | 55 passed, 1 offline-toolchain skip |

The Release unit run initially failed one expected-diagnostic assertion after
call-site detail was added. The assertion was corrected and its test passed on
rerun. Both configurations excluded the Apple-reference metallib parser check
and the CLI doctor check because their offline-toolchain prerequisites are absent.
Eleven additional typed-CUDA metallib tests per configuration were attempted and
all skipped for that same missing toolchain; none count as passes.

The unmodified published `4a207e2` compiler/runtime fails the new
`f32-rni-direct` numerical regression on the Apple GPU. Its first output is
`0xa5a50000` instead of zero, demonstrating the combined conversion/storage test
is not already green on the baseline. The aligned loop and guarded-memory
positive reducers also reject on that baseline; renamed legacy register storage
fails independent LLVM assembly.

All seven unchanged full-module replays still reject at later translation stages;
none is a full self-test pass. The Release compiler SHA-256 is
`72f3b8f4fe88e9eaf870e1b68173f611a2ee111e52ccd7edd80a4ceffd7898b0`.
The Debug compiler SHA-256 is
`be585116cd94ac4512c72af96f28d668bebe11f29abedd4dbae8c7231b0c4e7d`.
