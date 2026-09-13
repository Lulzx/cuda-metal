# Vanity miner on CuMetal: incremental validation checklist

Goal: run all vanity-miner self-tests and all four mining kernels correctly on
CuMetal, from unchanged Rust-CUDA PTX. Advance one small test or failure at a
time; this checklist does not authorize or require running the entire suite
in one session. Correctness comes before throughput.

## Current baseline

Pinned inputs: run `34778991430`, x86-64 LLVM 7 (`sm_89`) and LLVM 19 (`sm_100`)
artifacts. Each contains 123 entries: 118 numerical self-tests, one plumbing
probe, and four mining kernels. ARM64-produced artifacts have the same entry
names but have not been executed in this comparison.

| Check | LLVM 7 | LLVM 19 | Scope |
| --- | --- | --- | --- |
| Full-module launch probe | Pass | Pass | Result and guard checks on M5 |
| Ed25519 primitive, slot 2 | Pass | Pass | One fixed known-answer fixture on M5 |
| Xoroshiro primitive, slot 0 | Pass | Pass | Scalar / local-buffer tail-call fixes; additional runtime-input regressions |
| Remaining numerical self-tests | Partial | Partial | See fix backlog; no aggregate pass claim |
| Four mining kernels | Pending | Pending | No end-to-end mining claim |

[Evidence, hashes and historical blockers](dual-llvm-miner-smoke.md).
[Subsequent fixes and targeted retests](miner-fix-backlog.md).
CuMetal commits through `967d8c7` include typed relocation, reviewed self-select
normalization and min/max typing. The earlier zero-pass compile inventory is
historical; do not use it as the current compatibility count.

## Next session: a deliberately small slice

- [ ] Inspect the emitted PTX for `kernel_self_test_arith_blackbox_identity_u64`
  and the corresponding u32 entry; confirm exact entry names against the
  artifact before invoking them. These correspond to result slots 57 and 58.
- [ ] Compile and run the u64 identity check from LLVM 7, then LLVM 19.
- [ ] Repeat for u32. Require the selected slot to equal 1 and all other slots
  and guards to remain untouched.
- [ ] Record whether each entry actually exercises the intended identity path
  or was constant-folded. A folded pass is only a launch/result-store check.
- [ ] If either fails, isolate that failure and add a regression before
  advancing. If both pass, take SHA-512 slot 1 as the next primitive, explicitly
  addressing the folded LLVM 19 fixture before claiming arithmetic coverage.

This small slice checks the test machinery used by many arithmetic diagnostics;
it is not a substitute for host-provided runtime inputs.

## Self-test rollout, one entry at a time

The source already has a substantial diagnostic suite. Preserve that work and
map results back to `logic/src/self_test.rs` labels and kernel result slots.
Do not equate the number of self-tests with independent arithmetic coverage.

- [ ] Add an opt-in runner/ledger for a selected entry and slot, recording both
  producer variants independently. No default "run everything" requirement.
- [ ] Validate the remaining top-level primitives (slots 0–9): xoroshiro,
  SHA-512, base58, secp256k1 compressed/uncompressed, Keccak-256, RIPEMD-160,
  SHA-256 fixed and variable length. Slot 2 already has one known-answer pass.
- [ ] Validate relevant raw arithmetic/identity checks (slots 31–40, 46–58)
  before or while diagnosing a failing primitive.
- [ ] Work through encoding and subsystem bisects (slots 41–45, 59–69), then
  curve/scalar/indexing/memory diagnostics (slots 70–117). Use the source labels
  as the authority; comments describing old bugs are not current CuMetal results.
- [ ] Validate composed checks (slots 10–30): Solana, Ethereum, Bitcoin,
  WIF variants, Shallenge and hash comparisons, after their primitives pass.
- [ ] Eventually account for all 118 numerical slots in both LLVM artifacts,
  including explicit folded-only, unsupported, timeout and numerical-fail states.
- [ ] Run a complete suite only once individual results justify it; retain
  per-entry isolation so a failure remains attributable.

## Strengthen the evidence where the current tests fall short

- [ ] Audit each emitted entry and reachable helpers for constant folding.
  LLVM 19's current SHA-512 primitive entry simply stores 1. `black_box` in Rust
  source alone does not establish that the intended GPU arithmetic survived.
- [ ] Add host-input primitive kernels that return actual digests/public keys,
  comparing full bytes with known-answer vectors and a CPU reference.
- [ ] Add multiple Ed25519 inputs, including scalar/clamping boundaries; the
  current success proves only one fixed derivation.
- [ ] Cover hash lengths around block/padding boundaries for variable-length
  APIs; use varied byte patterns and leading zeros for encoding tests.
- [ ] Add host-input arithmetic boundary cases for carries/borrows, high-bit
  multiplication, shifts, signed comparisons and narrow/wide conversions.
- [ ] Add multi-thread tests around launch boundaries (1, 31, 32, 33, 257),
  with per-thread outputs and guards, after single-thread cases are correct.
- [ ] Automate the small CuMetal regression set on a suitable Apple GPU runner;
  registered CTests are not evidence of a GitHub GPU CI gate.

## Mining kernels: one bounded correctness harness at a time

Suggested order follows increasing integration scope, not a promise about which
will compile first. For each: inspect ABI and termination first, bound work,
use deterministic seeds/inputs, then compare CPU and GPU results. Check candidate
counts, result capacity, found/no-match behavior and guards. Where atomics make
output order nondeterministic, compare result sets rather than buffer order.

- [ ] Shallenge: fixed starting state, bounded nonce batch, exact hashes and
  best-result comparison (including equal-score behavior).
- [ ] Solana: bounded candidates; compare private/public key and encoded-address
  results, then matching and result-buffer behavior.
- [ ] Ethereum: bounded candidates; compare public keys, address derivation,
  matching and result-buffer behavior.
- [ ] Bitcoin: bounded candidates; cover the address/network modes actually
  exposed by the kernel ABI, then matching and result-buffer behavior.
- [ ] Exercise each through the intended host integration after its standalone
  harness passes. PTX execution alone does not establish CLI compatibility.
- [ ] Broaden launch sizes, repeated launches and supported configurations.
- [ ] Benchmark only after correctness gates pass; report performance separately.

## Procedure for each bug we uncover

1. Pin PTX hash, producer, CuMetal commit/build hash, entry, input, expected
   result and actual result. Keep compile, Metal compile, launch and numerical
   outcomes separate; a timeout is not an unsupported-opcode finding.
2. Reduce to the smallest reproducer that preserves the failure. Keep the
   original full-module artifact as the integration retest.
3. Add a regression that fails before the fix, including boundary/negative
   cases relevant to the failure. Avoid expected values computed by the same
   GPU lowering being tested.
4. Fix the shared compiler/runtime behavior. Do not alter expected answers or
   add workload-specific output substitutions. Request independent review for
   control-flow, aliasing, pointer provenance or other proof-sensitive changes.
5. Run focused regressions, retry the original entry, update the ledger and
   commit the fix with its tests and evidence. Move to the next entry afterward.

## Known follow-up outside the passing smoke checks

- [ ] Investigate register-wide pointer inference when a PTX register is reused
  first as an integer offset and later as a pointer. The first min/max fixture
  exposed this; separate registers isolated min/max but did not fix reuse.
- [ ] Broaden guarded-select CFG coverage as real inputs require it; current
  support deliberately excludes unproven cases and retains SSA verification.

Ledger fields per attempt: artifact/hash, entry/slot, intended operation,
folded-versus-runtime assessment, PTX-to-MSL result, Metal compilation result,
launch result, numerical result, guard result, log path, blocker/regression link.
A checkbox closes only for its stated scope; new producer artifacts need their
own results rather than inheriting a pass by kernel name.
