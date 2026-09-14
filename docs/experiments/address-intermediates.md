# Ethereum and Bitcoin intermediate checks

All six remaining LLVM 19 Ethereum/Bitcoin checks pass individually on Apple
M5 at revision `b4894ed`. No compiler change was needed. The compiler implementation
is unchanged from the `93ebd6b` signed-byte conversion fix.
[Per-entry results, logs and hashes](address-intermediates-b4894ed.json).

| Slot | Entry suffix (`kernel_self_test_…`) | Checked result | Outcome |
| --- | --- | --- | --- |
| 13 | `ethereum_priv` | All 32 private-key bytes | Pass |
| 14 | `ethereum_pub` | All 64 public-key bytes | Pass |
| 16 | `bitcoin_priv` | All 32 private-key bytes | Pass |
| 17 | `bitcoin_pub` | All 33 compressed public-key bytes | Pass |
| 18 | `bitcoin_pkh` | All 20 public-key hash bytes | Pass |
| 20 | `bitcoin_matches` | Positive matching flag | Pass |

Each selected result slot contains 1. The other 117 slots and all 16 guard
words remain untouched. Runs use the original full LLVM 19 PTX from artifact
run `34778991430`, one GPU thread, and disabled workload specializations. Neither
PTX nor generated MSL is manually edited or instrumented.

Combined with the [previous address checks](secp256k1-dependent-checks.md), all
**8 Ethereum/Bitcoin numerical entries** have now passed with this LLVM 19
compiler implementation: Ethereum slots 13–15 and Bitcoin slots 16–20.
This is a result for that group, not an updated total for the 118-test suite.
LLVM 7 counterparts and the four mining kernels remain outside this validation.

## Coverage limits

The checks use the existing deterministic fixtures: Ethereum seed
`15455378110306975741`, logical thread 0; Bitcoin seed `13278869120712471092`,
logical thread 1. Bitcoin's matching check expects true for prefix `bc1q` and
an empty suffix. It does not cover negative matches or nonempty suffixes.
Fixed fixtures can be partly constant-folded; these passes are not randomized
input or concurrent mining validation.

## Reproduce

For each entry and slot in the table:

```sh
build-rust-ptx-apple/cumetalc "$PTX" --backend=cumetal-ir --ptx-strict \
  --overwrite --entry "$ENTRY" --emit=msl -o "$MSL"
python3 demos/rust-ptx/run_self_test.py "$MSL" \
  --build-dir build-rust-ptx-apple --kernel "$ENTRY" --slot "$SLOT"
```

Local modules, ABI sidecars and logs are under `/tmp/address-intermediates-b4894ed`.
Compilation used three workers with a 600-second limit per entry; GPU runtime
attempts were sequential with a 300-second limit each. The ledger keeps host-side
attempt durations separate from the GPU trace's execution durations.
