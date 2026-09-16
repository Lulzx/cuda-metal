# Unsigned packed-value narrowing: issue #134 validation

The pre-SSA normalizer now handles a single `mov.b64` pack whose sole consumer
is exactly `cvt.u16.u64` or `cvt.u32.u64`. When the selected low32 source is
unchanged, it removes the pack and converts that source directly. The conversion
retains its destination format and wider storage extension. No undefined high
bits are materialized or filled with zero.

[Issue #134](https://github.com/Lulzx/cuda-metal/issues/134) remains distinct from
full [Bitcoin self-test acceptance #30](https://github.com/brandonros/vanity-miner-rs/issues/30).
This patch is based on #130's draft PR #138 at
`0f98856b9bcab06f0c41d239a86684fdf0d6371d`, following #76 PR #135 and its earlier stack.

## Implementation bounds

- Exact declarations establish widths, including compact register ranges;
  names do not supply type or use information. Declared builtin-looking names
  stay in use counts, including `%clock_payload` and an observed extraction lane.
- Whole-function single-definition/single-use counts include call returns and
  predicate reads. Source-write and call generations preserve the value captured
  at the pack. The consumer can overwrite its source only after reading it.
- Both instructions must be unpredicated and in the same block, with no
  intervening call or selected-source write. Other conversions and ambiguous
  declarations retain ordinary SSA checks.
- Bounded occurrence, instruction, text, declaration and candidate accounting
  replaces repeated interval rescans. All replacements are staged before a final
  commit pass; an exhausted proof leaves the original function unchanged.
- Existing low/high tuple extraction and named-unread destination rules remain
  covered. This is not general partial-value analysis.

## Validation

Apple M5 / macOS 26.6.2; LLVM/Clang 21.1.8:

| Configuration | Unit suite | PTX functional suite |
| --- | --- | --- |
| Release, binary shim OFF | 56 pass, one benchmark-precondition skip | 57 pass, one offline-Metal-tool skip |
| Debug, binary shim ON | 59 pass, same skip | 57 pass, same skip |

The two unit gates requiring unavailable Apple reference/CLI tool prerequisites
are excluded, as in the parent validation. These are scoped unit/PTX suites, not
an all-project test claim. Both configurations build successfully.

The new GPU runner checks **12 fixtures × 65 lanes**, each with five output words:
unsigned 16/32-bit formats, natural/wider64 storage, and straight/intervening-work/
loop layouts. An independently defined reference pack has a second full-width
use, so it cannot take this optimization. Independent CPU arithmetic, full wider
storage, observable side effects, buffer guards, unchanged inputs, exact ABI and
Apple GPU provenance are checked. These are 780 lane positions per configuration,
not 780 unique input values.

All **44 rejection controls** remain rejected. They cover genuinely observable
undefined bits and deliberately unsupported shapes such as intervening calls,
source overwrites and multiple uses. These two categories must not be conflated.
The unchanged parent compiler rejects all 12 positive fixtures and all 44
rejection controls. New host tests also cover malformed/ambiguous declarations,
call-return and vector/tuple writes, compact huge ranges, seven proof limits,
late all-or-nothing exhaustion and overlapping candidate intervals.

## Unchanged Bitcoin self-test input

Producer `afe80210ea28748cc58c3ba75f877bfa4a6b1ecd`,
[Actions35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652),
LLVM21 `self_test_bitcoin.ptx`, entry `kernel_self_test_bitcoin`.
SHA-256 `48a78769d28dc0e699b2e3209b332e8cd6efa5d3846cf39d1b9b0776adbe035b`.
Input hashes match before and after both 300-second translation attempts.

- Parent #130 compiler: rejects `%r1688` undefined at `$L__BB4_3` in **59.318 s**.
- This patch: clears that narrowing demand, then rejects IR pointer-type
  inconsistencies in **97.315 s**, starting at PTX line **32188**:
  `operand type ptr<device, i8> does not match value %197406 type i64`.
  No Metal source is emitted, and no full Bitcoin GPU execution is claimed.

The first next pointer comes from a heterogeneous `ld.local.v2.b64` pointer/length
reload; #118 is being checked as its owner. Later diagnostics are not all
attributed from their wording alone. A separate pointer fix and normal consumer
GPU acceptance remain necessary before downstream #30 can close.

These elapsed values are bounded acceptance observations, not isolated compiler
performance benchmarks; the candidate attempt overlapped the independent Debug
host build. The focused numerical fixtures and full-input translation are
separate gates. A matched immutable consumer replay is recorded separately after
publication.

- Parent compiler SHA-256: `7b7c6dac6330edde295baf19a865040922185a2f7412a613af66c6ad0df47d98`.
- Candidate compiler SHA-256: `2632e682ea0bb08c0a93e3c474790bdf6d6073ec24305c7d762c1e2773afc33f`.
