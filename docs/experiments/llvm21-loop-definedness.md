# LLVM 21 bounded self-select definedness

The LLVM 21 Ed25519 entry failed register SSA import before GPU execution. It
preserves an initially undefined value in a `selp` false arm, returns to a loop
header on that arm, and bypasses the select at the upper bound. The bypass goes
through a second comparison before reaching the consumer. Secp256k1 also failed
definedness, but its cause is a different guarded-load shape (see below).

## Proof and implementation

Two bounded transformations run before register SSA:

1. Specialize a compare-only successor for an incoming unsigned threshold fact.
   For example, an edge proving `index > 63` implies that `index < 64` is false.
   The specialized block retains its comparison instruction and gets a fresh
   instruction identity for SSA, but has only its proven successor. The original
   block remains available to other incoming edges.
2. Extend the existing unobserved self-select normalization to direct inverted
   branches and unchanged `mov.pred` / `not.pred` aliases. Follow the actual false
   edge. Replace the select only when no path reads its preserved value before an
   unconditional overwrite or the same select.

Threshold facts require the same register, unsigned width, and normalized bound.
Only scalar unsigned `lt`, `le`, `gt`, and `ge` with decimal constant bounds are
recognized. Overflowing normalization is rejected. Writes to the compared register
or controlling predicate discard the fact, including predicated writes. The
successor must contain exactly the comparison and branch; no side effects are
skipped. At most two blocks are added per original block, without recursive
specialization. General register-definedness checks remain enabled.

Keeping the impossible edge while merely rewriting `selp` is insufficient:
SSA would still demand the selected register on that edge. The CFG specialization
makes the control-flow proof explicit to later verification.

## Regression coverage

- Typed PTX/IR/MSL unit suite passes.
- Positive coverage includes a saturating bounded loop, a `not.pred` branch,
  a directly inverted branch, and the existing unbounded guarded-select shapes.
- Negative cases retain rejection for an observable false arm, a changed bound,
  signed/unsigned comparison mismatch, register writes, predicate clobbers,
  incorrect inversion, and a bypass directly to the consumer.
- Existing and bounded Apple GPU regressions each pass 65,541 inputs, with
  independent CPU expected values and 16 guard words: 131,082 tested inputs total.

The PTX fixture and the original miner PTX are not rewritten for execution.
These changes normalize the compiler's internal CFG before SSA construction.

## Reproduce

```sh
cmake --build build-rust-ptx-apple --target cumetalc cumetal_ptx_ir_msl_test -j 4
build-rust-ptx-apple/tests/unit/cumetal_ptx_ir_msl_test
ctest --test-dir build-rust-ptx-apple \
  -R '^functional_ptx_(guarded|bounded)_self_select$' --output-on-failure
```

Full-module retry logs are under `/tmp/miner-llvm21-loop-fixed`.

## Original LLVM 21 artifact results

[Evidence and hashes](llvm21-loop-definedness.json). Input PTX SHA-256 remains
`2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b`.

- Ed25519 compiles and numerically passes on Apple M5: slot 2 equals 1, the other
  117 result slots and 16 guards remain untouched. Import took about 109 seconds;
  host launch attempt (including Metal compilation) about 8.6 seconds;
  GPU trace duration about 2.8 ms. No PTX/MSL instrumentation or workload specialization.
- Compressed secp256k1 still fails import: `%rd34990` undefined on an incoming
  edge to `$L__BB127_1`. It did not reach Metal compilation or GPU execution.

The secp256k1 failure is in the `k256::arithmetic::mul::lincomb` helper. A load of
`%rd34990` is bypassed when `%rd34984 == %rd7`; a later consumer is bypassed when
that same equality OR a remaining-length test is true. Supporting this requires
proving implications involving repeated equality predicates and `or.pred`, while
preserving conditional-load safety. It is outside the implemented unsigned
threshold/compare-only specialization. Do not replace the missing value with zero.

This advances one numerical self-test; the remaining LLVM 21 suite and full
mining workloads are not validated by these results.
