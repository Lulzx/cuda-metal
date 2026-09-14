# LLVM 21 arithmetic validation

All ten arithmetic entries in slots 31–40 compile and numerically pass on
Apple M5 at CuMetal `40758eb`. No additional compiler changes were needed.
[Per-entry logs, timings, and hashes](llvm21-arithmetic-checks.json).

| Slot | Check | Result | Import (s) | Runtime attempt (s) | GPU (µs) |
| --- | --- | --- | --- | --- | --- |
| 31 | u32 division, variable divisor | Pass | 70.9 | 0.759 | 1.874 |
| 32 | u32 division, constant divisor | Pass | 72.4 | 0.360 | 1.500 |
| 33 | u64 division, variable divisor | Pass | 58.6 | 0.292 | 2.458 |
| 34 | u64 division, constant divisor | Pass | 57.4 | 0.185 | 1.666 |
| 35 | u32 remainder | Pass | 51.5 | 0.184 | 2.041 |
| 36 | u64 remainder | Pass | 50.7 | 0.411 | 2.416 |
| 37 | u32 low product | Pass | 61.1 | 0.188 | 1.625 |
| 38 | u64 low product | Pass | 60.4 | 0.181 | 1.624 |
| 39 | u64 high product | Pass | 87.3 | 0.299 | 1.624 |
| 40 | u128 wrapping product | Pass | 87.7 | 0.188 | 1.666 |

Each selected slot equals 1; the other 117 result slots and all 16 guard words
remain untouched. Runs use the generic PTX path, disabled workload specializations,
one GPU thread, and sequential launches. All traces report compilation-cache misses.

## Coverage and limits

The fixtures use Rust constant-evaluated expected values and `black_box` operands.
Division/remainder checks use divisor 58; multiplication uses fixed 32-, 64-,
and 128-bit operands. The high-product fixture compares the upper 64 bits of a
64-by-64 product; the 128-bit fixture compares the wrapping product.

These passes validate the emitted programs for these inputs. They are not an
exhaustive arithmetic proof, a randomized test, or confirmation that every source
operation maps to one particular PTX opcode. Old comments in the Rust fixtures
about broken arithmetic describe historical investigations, not this batch's
results. Full-suite and mining validation remain incomplete.

The original full LLVM 21 PTX and generated Metal are unchanged and uninstrumented.
Input: CUDA 13.3 / NVVM 23, PTX 9.3, sm_100, 25,161,784 bytes;
SHA-256 `2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b`.
Two compile workers were used. Host runtime-attempt durations include Metal
compilation/setup; GPU durations come from launch traces.

## Reproduce

Use entry names and slots from the JSON ledger:

```sh
build-rust-ptx-apple/cumetalc "$PTX" --backend=cumetal-ir --ptx-strict \
  --overwrite --entry "$ENTRY" --emit=msl -o "$OUT.metal"
python3 demos/rust-ptx/run_self_test.py "$OUT.metal" \
  --build-dir build-rust-ptx-apple --kernel "$ENTRY" --slot "$SLOT"
```

Imports were bounded at 600 seconds; runtime attempts at 300 seconds.
Local modules, ABI sidecars, and logs: `/tmp/llvm21-arithmetic-validation`.
Next bounded batch: composed-primitive checks, slots 41–45.
Historical LLVM 7/19 totals are unchanged.
