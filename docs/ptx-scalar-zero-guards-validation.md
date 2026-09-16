# Scalar zero-marker guards: issue #130 validation

The bounded CFG proof now carries exact scalar zero moves/copies into equality
and inequality guards before register SSA. A proven edge also folds an unselected
`selp.b32/b64` payload arm, including terminal blocks. Rewrites are staged from
that same proof and retain observable operations. Dynamic and private-address
markers are never assumed nonzero. Unknown facts, invalid declarations, writes,
conflicting joins and exhausted budgets preserve strict definedness checks.

Implementation issue: [#130](https://github.com/Lulzx/cuda-metal/issues/130).
This change depends on the #76 proof follow-up at
`13efc293c6cd06da5616a87b2dd8db81b1b0e0df` (draft PR #135).

## Validation

Tested on Apple M5 / macOS 26.6.2, with LLVM/Clang 21.1.8:

- Release, binary shim OFF: 55 unit passes and one benchmark-precondition skip.
- Debug, binary shim ON: 58 unit passes and the same skip.
- Each configuration: 56 PTX functional passes, one offline-Metal-tool skip.
- Two unrelated unit gates require unavailable Apple reference/toolchain inputs
  and were excluded. These are scoped unit/PTX suites, not a full project sweep.
- The new functional test numerically verifies 16 kernels, each across 65 lanes,
  with independent CPU expectations, exact output/guard/input checks and Apple
  GPU provenance. All 56 observable-undefined controls are rejected before GPU
  execution. Release and Debug both pass.
- Before this change, the 12 original positive zero-marker fixtures all fail
  translation on the unchanged parent compiler. The added scalar-select
  reproducers also fail before the select correction; direct-branch controls
  pass. Negative controls remain rejected.
- One existing CFG fixture used the branch predicate to select a defined
  constant on its missing-definition path. The new optimization correctly made
  that value unobservable. The fixture now uses an independent selector, retains
  its join/type assertions, and has a genuinely observable negative witness.
  Both configurations pass its targeted rerun after that correction; the other
  final suite results above needed no implementation changes.

Unit coverage includes declared widths and compact ranges, renamed registers,
copy snapshots, calls and returns, predicated overwrites, loop/join meets,
terminal and repeated-comparison selects, and conservative resource limits.

## Unchanged full-input translation replay

Compiler SHA-256:
`7b7c6dac6330edde295baf19a865040922185a2f7412a613af66c6ad0df47d98`.
This is the final Release development build, not the consumer Nix package.
Every input hash was checked before and after its serial 300-second attempt;
none timed out. The original 7484a5d producer includes its previously recorded
instrumentation. The remaining inputs came from miner `afe80210`, Actions run
[35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).

| Input | Result | Seconds |
| --- | --- | ---: |
| 7484a5d-llvm7-p256_public_key | MSL emitted | 13.110 |
| afe80210-llvm21-self_test_shallenge | MSL emitted | 16.642 |
| afe80210-llvm21-self_test_solana | MSL emitted | 79.644 |
| afe80210-llvm7-self_test_bitcoin | cumetalc failed: line 34534: unsupported local pointer memory proof: unresolved potentially overlapping store at line 33839 ([%rd3056+1]); prefix retained facts budget exhausted (used=1048711, limit=1048576) | 102.455 |
| afe80210-llvm7-self_test_ethereum | MSL emitted | 27.749 |
| afe80210-llvm7-p256_public_key | MSL emitted | 11.334 |
| afe80210-llvm7-self_test_p256_public_key | cumetalc failed: PTX register '%r29315' is undefined on an incoming edge to block '$L__BB19_1' | 30.941 |
| afe80210-llvm7-self_test_p256_signature | cumetalc failed: PTX register '%rs2098' is undefined on an incoming edge to block '$L__BB35_4' | 147.826 |
| afe80210-llvm7-self_test_rsa_pss | cumetalc failed: PTX memory-address demand proof budget exhausted (join edge construction: 1000000 used, 1 requested, limit 1000000) | 72.212 |

These are translation results, not Apple preparation or full-kernel GPU passes.
The original P-256 input hash is
`dbd5ada9e96613eb7f2da98e97b95310a5a6368744c5f8d1d84b9c184184e0da`.
The new memory-proof budgets and later undefined values are separately reported
first blockers, not silently accepted or initialized payloads. Consumer-pinned
GPU results must be recorded separately before closing workload issues.
