# LLVM 21 WIF validation

All four WIF self-test entries compile and numerically pass on Apple M5 at
CuMetal `e4ba46b`, without additional compiler changes.
[Per-entry logs, timings, and binary hashes](llvm21-wif-checks.json).

| Slot | Variant | Result | Import (s) | Runtime attempt (s) | GPU (ms) |
| --- | --- | --- | --- | --- | --- |
| 21 | Compressed mainnet | Pass | 61.9 | 22.8 | 0.332 |
| 22 | Uncompressed mainnet | Pass | 61.2 | 26.4 | 0.333 |
| 23 | Compressed testnet | Pass | 81.7 | 16.6 | 0.336 |
| 24 | Uncompressed testnet | Pass | 82.7 | 14.4 | 0.327 |

Each selected result slot equals 1; the other 117 slots and all 16 guard words
remain untouched. These fixtures compare the complete WIF string and its length
(52 compressed, 51 uncompressed), using one fixed private key across both network
and compression flags. They do not establish arbitrary-input or mining coverage.

The full LLVM 21 PTX is unchanged: CUDA 13.3 / NVVM 23, PTX 9.3, sm_100,
25,161,784 bytes, SHA-256
`2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b`.
Neither PTX nor generated Metal was edited or instrumented. Runs use the generic
PTX path with workload specializations disabled, one thread, and sequential GPU
launches. Import used two workers. Runtime attempt times include Metal compilation
and setup; GPU times come from the launch trace. All report compilation-cache misses.

## Reproduce

Use the entry and slot from the JSON ledger:

```sh
build-rust-ptx-apple/cumetalc "$PTX" --backend=cumetal-ir --ptx-strict \
  --overwrite --entry "$ENTRY" --emit=msl -o "$OUT.metal"
python3 demos/rust-ptx/run_self_test.py "$OUT.metal" \
  --build-dir build-rust-ptx-apple --kernel "$ENTRY" --slot "$SLOT"
```

Attempts were bounded at 600 seconds for import and 300 seconds for runtime.
Local modules, ABI sidecars and logs: `/tmp/llvm21-wif-validation`.

Next bounded batch: Shallenge checks, slots 25–27. The full LLVM 21 numerical
suite and four mining kernels remain unvalidated; LLVM 7/19 totals are unchanged.

Follow-up: [all three LLVM 21 Shallenge checks now pass](llvm21-shallenge-checks.md).
