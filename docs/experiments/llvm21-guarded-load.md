# LLVM 21 conditional-load definedness

The `k256::arithmetic::mul::lincomb` helper conditionally loads `%rd34990`.
It skips that load when two registers compare equal. A subsequent repeated
comparison, combined with a length check by `or.pred`, skips the consumer on
that same path. The next block branches on that combined predicate again,
guarding another conditionally defined register. Plain CFG joins lose these
relationships and require definitions on impossible incoming paths.

## Compiler change

Before register SSA, specialize incoming edges that establish a scalar register
`setp.eq`/`setp.ne` fact or the value of an `or.pred` result. Evaluate repeated
same-width integer/bit equality comparisons and `mov.pred`, `not.pred`, and
`or.pred` using only proven boolean values. A true OR input suffices even when
the other input is unknown. Writes, including predicated writes, invalidate
facts about the written register. Calls discard tracked facts.

When a successor's branch is proven, clone that block for the incoming edge,
retain every non-branch instruction with a fresh SSA instruction identity, and
replace only its final conditional branch with the proven edge. Carry facts
through at most eight successors, each with at most 32 instructions; stop at
unknown branches or revisited blocks. Original blocks remain available to
other predecessors. This is a bounded proof, not general symbolic execution.
Unsupported comparisons and genuinely undefined reads still fail validation.
No load is hoisted, and no undefined register is filled with zero.

## Regression coverage

The full typed PTX/IR/MSL unit suite passes. Positive cases cover the two-stage
guard, swapped comparison/OR operands, inverted equality branches, and predicate
aliases. Negative cases reject changed comparison operands (including conditional
writes), predicate clobbers, premature reads, opposite comparisons, and replacing
OR with AND.

`functional_ptx_guarded_load` passes 65,557 inputs on Apple M5, checking CPU
expected results and 16 untouched guard words. Inputs exercise both equality
outcomes, both length outcomes, and 32-bit overflow. The bypass leaves a null
load address and an undefined load-result register; the consumer must stay
unreachable. The existing guarded and bounded self-select GPU tests also pass
65,541 inputs each (196,639 inputs across all three tests).

```sh
cmake --build build-rust-ptx-apple --target cumetalc cumetal_ptx_ir_msl_test -j 4
build-rust-ptx-apple/tests/unit/cumetal_ptx_ir_msl_test
ctest --test-dir build-rust-ptx-apple \
  -R '^functional_ptx_(guarded_load|guarded_self_select|bounded_self_select)$' \
  --output-on-failure
```

Original miner PTX remains unchanged, SHA-256:
`2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b`.

## Original kernel result

Compressed secp256k1 now compiles and numerically passes on Apple M5. Slot 4
contains 1; the other 117 result slots and 16 guard words are unchanged.
The generic PTX path reports 18,653,250 ns of GPU execution, with workload
specializations disabled. Generated MSL is 7,467,333 bytes; the host launch
attempt also includes substantially longer Metal compilation/setup.
[Machine-readable evidence, logs, and hashes](llvm21-guarded-load.json).

```sh
build-rust-ptx-apple/cumetalc \
  ~/Downloads/vanity-miner-aarch64-llvm21/output-llvm21.ptx \
  --backend=cumetal-ir --ptx-strict --overwrite \
  --entry kernel_self_test_primitive_secp256k1_compressed \
  --emit=msl -o /tmp/secp.metal
python3 demos/rust-ptx/run_self_test.py /tmp/secp.metal \
  --build-dir build-rust-ptx-apple \
  --kernel kernel_self_test_primitive_secp256k1_compressed --slot 4
```

This validates one fixed compressed-key self-test. Next is the uncompressed
secp256k1 sibling, followed by dependent Ethereum/Bitcoin checks. The full
LLVM 21 suite and mining kernels remain unvalidated.

Follow-up: [the uncompressed sibling and all eight Ethereum/Bitcoin entries
also pass](llvm21-address-checks.md), without further compiler changes.
