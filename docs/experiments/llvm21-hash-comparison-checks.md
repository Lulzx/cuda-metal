# LLVM 21 hash-comparison validation

All three hash-comparison entries compile and numerically pass on Apple M5
at CuMetal `1a3fc92`, without additional compiler changes.
[Per-entry logs, timings, and hashes](llvm21-hash-comparison-checks.json).

| Slot | Comparison | Result | Import (s) | Runtime attempt (s) | GPU (µs) |
| --- | --- | --- | --- | --- | --- |
| 28 | Less than | Pass | 54.5 | 0.253 | 1.583 |
| 29 | Greater than | Pass | 57.7 | 0.208 | 1.625 |
| 30 | Equal | Pass | 51.2 | 0.192 | 1.833 |

Each selected slot equals 1; the other 117 slots and all 16 guard words are
untouched. The fixtures compare zero against all-0xff (expected -1), the reverse
(expected 1), and zero against zero (expected 0). Inputs use Rust `black_box`,
but these remain fixed fixtures, not a randomized test or exhaustive coverage
of mismatches at each byte position.

The original full LLVM 21 PTX and generated Metal source are unchanged and
uninstrumented. Input: 25,161,784 bytes, CUDA 13.3 / NVVM 23, PTX 9.3, sm_100;
SHA-256 `2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b`.
Runs use the generic PTX path, disabled workload specializations, one GPU thread,
and sequential launches. All traces report compilation-cache misses. Import
used two workers; runtime attempt times include Metal compilation/setup, while
GPU durations come from launch traces.

## Reproduce

Use entry names and slots from the JSON ledger:

```sh
build-rust-ptx-apple/cumetalc "$PTX" --backend=cumetal-ir --ptx-strict \
  --overwrite --entry "$ENTRY" --emit=msl -o "$OUT.metal"
python3 demos/rust-ptx/run_self_test.py "$OUT.metal" \
  --build-dir build-rust-ptx-apple --kernel "$ENTRY" --slot "$SLOT"
```

Imports were bounded at 600 seconds and runtime attempts at 300 seconds.
Local output: `/tmp/llvm21-hash-comparison-validation`.
Next bounded batch: arithmetic checks, slots 31–40. The full LLVM 21 suite and
four mining kernels remain unvalidated; historical LLVM 7/19 totals are unchanged.
