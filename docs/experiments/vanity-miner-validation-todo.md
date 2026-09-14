# Vanity miner on CuMetal: incremental validation checklist

LLVM 21 update: [Ed25519](llvm21-loop-definedness.md) and
[compressed secp256k1](llvm21-guarded-load.md) now numerically pass on M5 after
bounded CFG proofs. The earlier [smoke-test ledger](llvm21-miner-smoke.md)
records probe/SHA-512 passes and the now-resolved import failures.
[Uncompressed secp256k1 and all eight Ethereum/Bitcoin checks now pass](llvm21-address-checks.md)
individually on M5, with other slots and guards intact.
[All four LLVM 21 WIF variants, slots 21–24, also pass](llvm21-wif-checks.md).
[All three LLVM 21 Shallenge checks, slots 25–27, also pass](llvm21-shallenge-checks.md).
Next bounded batch: hash comparisons, slots 28–30.
No complete LLVM 21 suite or mining validation is claimed.

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
| Base58 primitive, slot 3 | Pass | Pass | Unsigned widening and pointer subtraction fixes; see [fix backlog](miner-fix-backlog.md) |
| Compressed-mainnet WIF, slot 21 | Pass | Pass | Trap/call fix; unchanged PTX on M5 |
| Compressed secp256k1, slot 4 | Runtime attempt timeout | Pass | LLVM 19 signed-byte conversion fix; unchanged PTX, selected slot 1 and guards intact on M5 |
| Uncompressed secp256k1, slot 5 | Not retested | Pass | Full 65-byte known-answer check at `93ebd6b` |
| Ethereum address, slot 15 | Not retested | Pass | Seeded address known-answer check at `93ebd6b` |
| Bitcoin Bech32 address, slot 19 | Not retested | Pass | Full encoded address and length check at `93ebd6b` |
| Ethereum checks, slots 13–15 | Not retested | 3 / 3 pass | Address plus separately tested private/public keys; fixed fixtures |
| Bitcoin checks, slots 16–20 | Not retested | 5 / 5 pass | Private/public keys, hash, encoding and positive match; fixed fixtures |
| Full numerical inventory | 90 / 118 pass | 82 / 118 pass | All entries attempted on `6d2549b`; remaining entries fail compilation |
| Four mining kernels | Pending | Pending | No end-to-end mining claim |

[Evidence, hashes and historical blockers](dual-llvm-miner-smoke.md).
[Subsequent fixes and targeted retests](miner-fix-backlog.md).
CuMetal commits through `967d8c7` include typed relocation, reviewed self-select
normalization and min/max typing. The earlier zero-pass compile inventory is
historical; do not use it as the current compatibility count.

## Latest full sweep and next fix

[Full 238-attempt ledger](miner-self-test-sweep-6d2549b.md): all 118 numerical
entries and the plumbing probe were attempted for each producer on `6d2549b`.
Both probes pass. The 64 failures are all compiler rejections; all 174 launched
entries pass their result and guard checks. Folded fixtures remain limited evidence.

- [x] Compile/run both u64 and u32 identity checks in both producers.
- [x] Attempt every self-test entry in both original PTX modules.
- [x] Implement bounded trap reporting across device calls/helpers (fix 10).
- [x] Run the six existing LLVM 19 k256 bisects: all pass, including derivation
  for scalars 1 and 2; the initial nontrivial-scalar failure was then fixed. See
  [secp256k1 isolation](secp256k1-isolation.md).
- [x] Locate and fix the nontrivial-scalar failure: `cvt.s16.s8` lost sign
  extension during table selection. Original compressed LLVM 19 primitive passes
  after fix 12, with a runtime-input regression and CPU-checked intermediate snapshots.
- [x] Validate uncompressed secp256k1 (5), Ethereum address (15), and Bitcoin
  encoded address (19) individually on LLVM 19: all pass with guards intact.
  [Results and scope](secp256k1-dependent-checks.md).
- [x] Run the remaining Ethereum (13–14) and Bitcoin (16–18, 20) entries
  individually at `b4894ed`: all six pass with guards intact. Together with
  the previous address checks, all eight LLVM 19 entries pass.
  [Per-entry evidence](address-intermediates.md).
- [ ] Retest the 47 previously trap-blocked entries; both compressed-mainnet WIF
  entries pass. Both compressed secp256k1 entries clear MSL lowering but expose
  a subsequently fixed numerical failure (LLVM 19 passes after fix 12) and
  a 180-second runtime-attempt timeout (LLVM 7, not retested in fix 12).
- [ ] Reduce LLVM 19 loop-definedness failures: 9 entries across base58 and Dalek.
- [ ] Resolve LLVM 7 pointer/type gaps: 5 IR verification failures, 1 `mul.hi`
  operand mismatch, 1 subtraction form and 1 address-space conflict.
- [ ] Retest each affected original entry after its focused fix.

Do not change the PTX or treat a cleared compiler blocker as a numerical pass.
Keep the historical partial sweep separate from this complete rerun.

## Self-test rollout, one entry at a time

The source already has a substantial diagnostic suite. Preserve that work and
map results back to `logic/src/self_test.rs` labels and kernel result slots.
Do not equate the number of self-tests with independent arithmetic coverage.

- [x] Add an opt-in runner/ledger for a selected entry and slot, recording both
  producer variants independently. No default "run everything" requirement.
- [ ] Validate the remaining top-level primitives (slots 0–9): xoroshiro,
  SHA-512, base58, secp256k1 compressed/uncompressed, Keccak-256, RIPEMD-160,
  SHA-256 fixed and variable length. Slot 2 already has one known-answer pass.
- [x] Run raw arithmetic/identity checks (slots 31–40, 46–58) in both producers;
  all pass their fixed fixtures. Folding audit remains separate.
- [ ] Work through encoding and subsystem bisects (slots 41–45, 59–69), then
  curve/scalar/indexing/memory diagnostics (slots 70–117). Use the source labels
  as the authority; comments describing old bugs are not current CuMetal results.
- [ ] Validate composed checks (slots 10–30): Solana, Ethereum, Bitcoin,
  WIF variants, Shallenge and hash comparisons, after their primitives pass.
- [x] Account for all 118 numerical slots in both LLVM artifacts in the full sweep,
  recording compile/runtime outcomes; folding audit remains separate.
- [x] Run the complete self-test inventory after shared fixes; retain
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
