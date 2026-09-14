# LLVM 21 secp256k1 and address checks

All nine selected entries compile and numerically pass on Apple M5 using CuMetal
`f5e170f`. No additional compiler change was needed after the compound-predicate
load-guard fix. [Per-entry logs, timings, and hashes](llvm21-address-checks.json).

| Slot | Check | Result | Import (s) | Runtime attempt (s) | GPU (ms) |
| --- | --- | --- | --- | --- | --- |
| 5 | 65-byte uncompressed public key | Pass | 69.8 | 98.9 | 19.47 |
| 13 | Ethereum private key | Pass | 118.4 | 168.3 | 56.17 |
| 14 | Ethereum public key | Pass | 118.8 | 155.2 | 85.26 |
| 15 | Ethereum address | Pass | 118.3 | 103.3 | 38.78 |
| 16 | Bitcoin private key | Pass | 147.6 | 108.6 | 27.20 |
| 17 | Bitcoin public key | Pass | 147.2 | 113.1 | 29.25 |
| 18 | Bitcoin public-key hash | Pass | 148.1 | 112.0 | 27.73 |
| 19 | Bitcoin Bech32 address and length | Pass | 145.0 | 115.6 | 26.41 |
| 20 | Bitcoin positive matching flag | Pass | 144.4 | 120.0 | 27.98 |

Each selected slot equals 1, while the other 117 result slots and all 16 guard
words remain untouched. All runs use the generic PTX path, one GPU thread,
and disabled workload specializations. Original PTX and compiler-generated Metal
source were not edited or instrumented. Their hashes are recorded in the ledger.

The input is the full 25,161,784-byte LLVM 21 artifact, CUDA 13.3 / NVVM 23,
PTX 9.3, sm_100, SHA-256
`2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b`.

## Coverage and limits

The uncompressed sibling compares all 65 public-key bytes for its known private
key. Ethereum checks private/public keys and the final 20-byte address separately.
Bitcoin checks private/public keys, the 20-byte hash, the complete Bech32 address
and length, and a positive matching flag for prefix `bc1q` with an empty suffix.

These are deterministic existing fixtures; optimization can fold parts of them.
They do not establish randomized-input correctness, negative matching behavior,
nonempty suffix behavior, or concurrent mining correctness. LLVM 7/19 totals
remain separate historical results. The full LLVM 21 suite and all four mining
kernels remain unvalidated.

The uncompressed sibling ran first. The eight dependent entries then compiled
with three workers, and GPU runtime attempts ran sequentially. Import and runtime
attempt times are wall-clock measurements, affected by overlapping compilation.
Runtime attempts include Apple's Metal compilation and launch setup; they are
not GPU timings. Every successful trace reports `compile_cache_hit=false`.

## Reproduce

For the entry names and slots in the JSON ledger:

```sh
build-rust-ptx-apple/cumetalc "$PTX" --backend=cumetal-ir --ptx-strict \
  --overwrite --entry "$ENTRY" --emit=msl -o "$OUT.metal"
python3 demos/rust-ptx/run_self_test.py "$OUT.metal" \
  --build-dir build-rust-ptx-apple --kernel "$ENTRY" --slot "$SLOT"
```

Compilation was bounded at 600 seconds per entry, runtime attempts at 300 seconds.
Local output modules, ABI sidecars and logs: `/tmp/llvm21-address-validation`.
Next bounded batch: the four LLVM 21 WIF variants (slots 21–24).
