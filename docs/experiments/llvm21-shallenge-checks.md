# LLVM 21 Shallenge validation

All three selected Shallenge entries compile and numerically pass on Apple M5
at CuMetal `debc412`, without additional compiler changes.
[Per-entry logs, timings, and hashes](llvm21-shallenge-checks.json).

| Slot | Check | Result | Import (s) | Runtime attempt (s) | GPU (µs) |
| --- | --- | --- | --- | --- | --- |
| 25 | Hash | Pass | 61.5 | 0.511 | 1.625 |
| 26 | Nonce length | Pass | 53.9 | 0.179 | 1.583 |
| 27 | Is better | Pass | 57.0 | 0.402 | 1.874 |

Every selected result slot equals 1, with the other 117 slots and 16 guard words
untouched. Runs use one GPU thread, the generic PTX path, and disabled workload
specializations. The original PTX and compiler-generated Metal are unchanged
and uninstrumented. All launch traces report compilation-cache misses.

## Scope

- Hash: compares all 32 expected bytes for username `brandonros`, seed 12345,
  logical thread 0, and a maximum target hash.
- Nonce length: checks that username length 10 yields nonce length 21. This is
  an arithmetic check, not a check of nonce generation or contents.
- Is better: checks the positive result against that maximum target. It does
  not exercise rejection or equality boundaries.

These are deterministic fixtures, so optimizers may fold parts of the work.
Microsecond execution times alone cannot establish where or how much folding
occurred. No claim of arbitrary runtime-input hashing, throughput, or concurrent
mining correctness follows from these passes.

Input: full 25,161,784-byte LLVM 21 PTX, CUDA 13.3 / NVVM 23, PTX 9.3, sm_100.
SHA-256: `2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b`.
Two compile workers were used, followed by sequential GPU launches. Import and
runtime attempt times are host wall-clock measurements; the latter include Metal
compilation and setup. GPU times come from the runtime trace.

## Reproduce

Use the entry names and slots in the JSON ledger:

```sh
build-rust-ptx-apple/cumetalc "$PTX" --backend=cumetal-ir --ptx-strict \
  --overwrite --entry "$ENTRY" --emit=msl -o "$OUT.metal"
python3 demos/rust-ptx/run_self_test.py "$OUT.metal" \
  --build-dir build-rust-ptx-apple --kernel "$ENTRY" --slot "$SLOT"
```

Imports were bounded at 600 seconds, runtime attempts at 300 seconds.
Local modules, ABI sidecars, and logs: `/tmp/llvm21-shallenge-validation`.
Next bounded batch: hash comparison checks (less/greater/equal), slots 28–30.
The full LLVM 21 suite and all four mining kernels remain unvalidated;
historical LLVM 7/19 totals are unchanged.

Follow-up: [all three LLVM 21 hash-comparison checks now pass](llvm21-hash-comparison-checks.md).
