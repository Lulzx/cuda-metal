# Uncompressed secp256k1 and dependent address checks

All three selected LLVM 19 entries numerically pass on Apple M5 at compiler
commit `93ebd6b`. No compiler changes were needed after the signed-byte conversion
fix. Inputs are the unchanged full PTX artifact from run `34778991430`.
[Results, logs and hashes](secp256k1-dependent-checks-93ebd6b.json).

| Slot | Entry suffix (`kernel_self_test_…`) | Result | GPU duration |
| --- | --- | --- | --- |
| 5 | `primitive_secp256k1_uncompressed` | Pass | 18.5 ms |
| 15 | `ethereum_address` | Pass | 31.7 ms |
| 19 | `bitcoin_encoded` | Pass | 23.7 ms |

Each entry launches with one GPU thread and writes 1 to its selected result slot.
The other 117 slots and all 16 guard words remain untouched. Workload
specializations are disabled. Generated MSL is not manually edited or instrumented.
The host-side runtime attempts take approximately 106, 105 and 125 seconds,
respectively; those include Metal compilation and are not GPU execution times.

## What was checked

- Uncompressed secp256k1 compares all 65 public-key bytes for the original
  private-key fixture used by the compressed primitive.
- Ethereum compares the final 20-byte address with
  `555563590c724a58f7bb48b6c847aa631a48651c`, using seed
  `15455378110306975741` and logical thread index 0. The source path includes
  seeded key generation, uncompressed secp256k1 and Keccak-256.
- Bitcoin compares both encoded length and the complete Bech32 address
  `bc1qqqqm20tdymccepdl72krc4c7ulsv3pllzju9s4`, using seed
  `13278869120712471092` and logical thread index 1. The source path includes
  seeded key generation, compressed secp256k1, SHA-256, RIPEMD-160 and Bech32.
  The logical index is a constant in the fixture, not the GPU launch index.

These are fixed known-answer tests, not randomized input coverage or a throughput
benchmark. Passing a final address does not mean every intermediate-field or
matching-flag entry was separately run. Optimizations may fold parts of fixed
fixtures. The four mining kernels remain unvalidated.

## Reproduce

For each entry and slot above:

```sh
build-rust-ptx-apple/cumetalc "$PTX" --backend=cumetal-ir --ptx-strict \
  --overwrite --entry "$ENTRY" --emit=msl -o "$MSL"
python3 demos/rust-ptx/run_self_test.py "$MSL" \
  --build-dir build-rust-ptx-apple --kernel "$ENTRY" --slot "$SLOT"
```

Local modules, ABI sidecars, compile logs and GPU logs are under
`/tmp/secp-dependent-93ebd6b`. Compilation was bounded at 600 seconds per entry;
runtime attempts were sequential and bounded at 300 seconds each.

Next small batch: separately validate Ethereum private/public-key checks
(13–14), Bitcoin private/public-key/hash checks (16–18), and Bitcoin's matching
flag (20). LLVM 7 has not been retested here; its historical failures and the
complete `6d2549b` sweep remain separate evidence.
