# Full miner self-test sweep on CuMetal 6d2549b

All 238 attempts completed on Apple M5, using the unchanged x86-64 LLVM 7 and
LLVM 19 artifacts from run `34778991430`. Compiler commit: `6d2549b`.
[Per-entry results, hashes, diagnostic excerpts and local log paths](miner-self-test-sweep-6d2549b.json).

| Outcome | LLVM 7 | LLVM 19 |
| --- | ---: | ---: |
| Numerical self-tests passed | 90 / 118 | 82 / 118 |
| Launch probe | Pass | Pass |
| PTX-to-MSL compile failures | 28 | 36 |
| Metal compilation failures | 0 | 0 |
| Numerical / guard / runtime failures | 0 | 0 |
| Timeouts | 0 | 0 |

The 174 successful launches include two plumbing probes. The remaining 172
passes are fixed-fixture self-tests, not 172 independent arithmetic proofs.
Constant folding was not audited in this sweep; LLVM 19 SHA-512 is already
known to be folded. No full mining kernel was compiled or run in this sweep.

Each entry compiled separately from the full original module with
`--backend=cumetal-ir --ptx-strict --entry NAME --emit=msl`. Eight compile workers
used a 600-second per-entry limit. Successful outputs ran sequentially through
`demos/rust-ptx/run_self_test.py`, in fresh processes, with a 180-second limit,
one GPU thread, the source-mapped selected slot required to equal 1, and all
other 117 slots plus 16 guard words required to remain untouched. Workload
specializations were disabled. Both input hashes and the compiler/runtime
binary hashes were rechecked after completion.

Local outputs: `/tmp/cumetal-self-test-sweep-6d2549b`; orchestration script:
`/tmp/run-miner-self-test-sweep-6d2549b.py`. A selected entry can be reproduced
with the compile flags above followed by:

```sh
python3 demos/rust-ptx/run_self_test.py /path/to/entry.metal \
  --build-dir build-rust-ptx-apple --kernel ENTRY --slot SLOT
```

## Remaining compile failures

| First blocker | LLVM 7 | LLVM 19 |
| --- | ---: | ---: |
| Kernel trap with calls | 19 | 27 |
| Helper trap | 1 | 0 |
| IR pointer type mismatch | 5 | 0 |
| SSA definedness | 0 | 9 |
| mul.hi operand types | 1 | 0 |
| Pointer subtraction form | 1 | 0 |
| Pointer address-space conflict | 1 | 0 |

Recommended next fix: extend trap reporting across device calls and helpers
(47 blocked entries). This is the largest first-blocker group; removing it may
expose further errors and does not guarantee 47 numerical passes. Preserve
trap semantics and add focused call-chain regressions before retesting these
original entries. Then isolate LLVM 19 loop-definedness cases and the remaining
pointer/type failures. The previous call-cycle rejection is no longer a first
blocker in this sweep.

## Per-entry ledger

Names below omit the common `kernel_self_test_` prefix. `Probe` writes slot 0
but is not one of the 118 numerical tests.

| Slot | Entry | LLVM 7 | LLVM 19 |
| ---: | --- | --- | --- |
| Probe | `stub` | Pass | Pass |
| 0 | `primitive_xoroshiro` | Pass | Pass |
| 1 | `primitive_sha512` | Pass | Pass |
| 2 | `primitive_ed25519` | Pass | Pass |
| 3 | `primitive_base58` | Pass | Pass |
| 4 | `primitive_secp256k1_compressed` | Kernel trap with calls | Kernel trap with calls |
| 5 | `primitive_secp256k1_uncompressed` | Kernel trap with calls | Kernel trap with calls |
| 6 | `primitive_keccak256` | Pass | Pass |
| 7 | `primitive_ripemd160` | Pass | Pass |
| 8 | `primitive_sha256_32` | Pass | Pass |
| 9 | `primitive_sha256_variable` | Pass | Pass |
| 10 | `solana_priv` | Kernel trap with calls | Kernel trap with calls |
| 11 | `solana_pub` | Kernel trap with calls | Kernel trap with calls |
| 12 | `solana_encoded` | Kernel trap with calls | Kernel trap with calls |
| 13 | `ethereum_priv` | Kernel trap with calls | Kernel trap with calls |
| 14 | `ethereum_pub` | Helper trap | Kernel trap with calls |
| 15 | `ethereum_address` | Kernel trap with calls | Kernel trap with calls |
| 16 | `bitcoin_priv` | IR pointer type mismatch | Kernel trap with calls |
| 17 | `bitcoin_pub` | IR pointer type mismatch | Kernel trap with calls |
| 18 | `bitcoin_pkh` | IR pointer type mismatch | Kernel trap with calls |
| 19 | `bitcoin_encoded` | IR pointer type mismatch | Kernel trap with calls |
| 20 | `bitcoin_matches` | IR pointer type mismatch | Kernel trap with calls |
| 21 | `wif_compressed_mainnet` | Kernel trap with calls | Kernel trap with calls |
| 22 | `wif_uncompressed_mainnet` | Kernel trap with calls | Kernel trap with calls |
| 23 | `wif_compressed_testnet` | Kernel trap with calls | Kernel trap with calls |
| 24 | `wif_uncompressed_testnet` | Kernel trap with calls | Kernel trap with calls |
| 25 | `shallenge_hash` | Pass | Pass |
| 26 | `shallenge_nonce_len` | Pass | Pass |
| 27 | `shallenge_is_better` | Pass | Pass |
| 28 | `compare_hashes_lt` | Pass | Pass |
| 29 | `compare_hashes_gt` | Pass | Pass |
| 30 | `compare_hashes_eq` | Pass | Pass |
| 31 | `arith_u32_div_var` | Pass | Pass |
| 32 | `arith_u32_div_const` | Pass | Pass |
| 33 | `arith_u64_div_var` | Pass | Pass |
| 34 | `arith_u64_div_const` | Pass | Pass |
| 35 | `arith_u32_rem_var` | Pass | Pass |
| 36 | `arith_u64_rem_var` | Pass | Pass |
| 37 | `arith_u32_mul_lo` | Pass | Pass |
| 38 | `arith_u64_mul_lo` | Pass | Pass |
| 39 | `arith_u64_mul_hi` | Pass | Pass |
| 40 | `arith_u128_mul` | Pass | Pass |
| 41 | `base58_var_len` | Kernel trap with calls | SSA definedness |
| 42 | `base58_var_len_leading_zero` | Kernel trap with calls | SSA definedness |
| 43 | `base58_all_zeros` | mul.hi operand types | Kernel trap with calls |
| 44 | `xoroshiro_base64_nonce` | Pass | Pass |
| 45 | `bech32_p2wpkh` | Pass | Kernel trap with calls |
| 46 | `arith_overflowing_add` | Pass | Pass |
| 47 | `arith_overflowing_sub` | Pass | Pass |
| 48 | `arith_carry_chain_3limb` | Pass | Pass |
| 49 | `arith_widening_mul_pair` | Pass | Pass |
| 50 | `arith_mad_lo_u64` | Pass | Pass |
| 51 | `arith_mad_hi_u64` | Pass | Pass |
| 52 | `arith_mul_wide_u32` | Pass | Pass |
| 53 | `arith_mask_blend_true` | Pass | Pass |
| 54 | `arith_mask_blend_false` | Pass | Pass |
| 55 | `arith_var_shr_u64` | Pass | Pass |
| 56 | `arith_var_shl_u64` | Pass | Pass |
| 57 | `arith_blackbox_identity_u64` | Pass | Pass |
| 58 | `arith_blackbox_identity_u32` | Pass | Pass |
| 59 | `base58_div_by_58` | Pass | Pass |
| 60 | `iter_static_table_lookup` | Pass | Pass |
| 61 | `iter_mut_slice_partial` | Pointer subtraction form | Pass |
| 62 | `iter_mut_alphabet_lookup` | Pass | Pass |
| 63 | `iter_static_slice_lookup` | Pass | Pass |
| 64 | `arith_divrem_by_58_pow_5` | Pass | Pass |
| 65 | `arith_i128_chain_add` | Pass | Pass |
| 66 | `base58_limb_divrem` | Pass | Pass |
| 67 | `dynamic_index_write` | Pass | Pass |
| 68 | `arith_widening_mul_chain_3term` | Pass | Pass |
| 69 | `base58_inner_mutate_phase` | Pass | Pass |
| 70 | `dalek_clamp_integer` | Pass | Pass |
| 71 | `dalek_scalar_round_trip_one` | Pass | Pass |
| 72 | `dalek_mul_base_scalar_one` | Pass | Pass |
| 73 | `k256_secret_from_bytes_one` | Pass | Pass |
| 74 | `k256_derive_scalar_one` | Kernel trap with calls | Kernel trap with calls |
| 75 | `k256_derive_scalar_two` | Kernel trap with calls | Kernel trap with calls |
| 76 | `static_u64_array_lookup` | Pass | Pass |
| 77 | `static_struct_wrapped_u64_lookup` | Pass | Pass |
| 78 | `k256_encode_generator` | Kernel trap with calls | Kernel trap with calls |
| 79 | `k256_double_generator` | Kernel trap with calls | Kernel trap with calls |
| 80 | `k256_scalar_one_round_trip` | Kernel trap with calls | Kernel trap with calls |
| 81 | `arith_u128_imm_shr_52` | Pass | Pass |
| 82 | `static_depth4_newtype_nesting` | Pass | Pass |
| 83 | `reverse_range_write` | Pass | Pass |
| 84 | `dalek_scalar52_from_bytes` | Pass | SSA definedness |
| 85 | `dalek_scalar52_montgomery_reduce_r` | Pass | Pass |
| 86 | `dalek_scalar52_mul_internal_then_reduce_one_r` | Pass | Pass |
| 87 | `dalek_scalar52_as_bytes_one` | Pass | Pass |
| 88 | `dalek_scalar52_sub_no_underflow` | Pass | Pass |
| 89 | `dalek_scalar52_sub_with_underflow` | Pass | Pass |
| 90 | `dalek_scalar52_montgomery_reduce_with_sub` | Pass | Pass |
| 91 | `index_trait_dispatch` | Pass | Pass |
| 92 | `dalek_scalar_one_to_bytes_direct` | Pass | Pass |
| 93 | `k256_affine_generator_encode` | Kernel trap with calls | Kernel trap with calls |
| 94 | `subtle_choice_u8_into_bool` | Pass | Pass |
| 95 | `subtle_conditional_select_u64` | Pass | Pass |
| 96 | `k256_encoded_point_from_affine_coords` | Pass | Pass |
| 97 | `index_trait_const_indices` | Pass | Pass |
| 98 | `generic_array_basic_index` | Pass | Pass |
| 99 | `generic_array_copy_from_slice` | Pass | Pass |
| 100 | `from_affine_coords_replica` | Pass | Pass |
| 101 | `generic_array_as_slice_last` | Pass | Pass |
| 102 | `dalek_scalar_round_trip_zero` | Pass | SSA definedness |
| 103 | `dalek_scalar_from_bytes_wide_zero` | Pointer address-space conflict | SSA definedness |
| 104 | `field_bytes_into_conversion` | Pass | Pass |
| 105 | `base58_min_nonzero` | Pass | Kernel trap with calls |
| 106 | `named_field_struct_return` | Pass | Pass |
| 107 | `base58_handrolled_no_seq` | Pass | Kernel trap with calls |
| 108 | `slice_reverse_partial` | Pass | Pass |
| 109 | `dalek_scalar_eq_zero` | Pass | SSA definedness |
| 110 | `generic_array_copy_from_ga_source` | Pass | Pass |
| 111 | `dalek_zero_eq_zero` | Pass | Pass |
| 112 | `dalek_from_canonical_zero` | Pass | SSA definedness |
| 113 | `dalek_scalar52_from_bytes_zero` | Pass | SSA definedness |
| 114 | `dalek_scalar52_mul_internal_zero` | Pass | Pass |
| 115 | `dalek_scalar52_montgomery_reduce_zero` | Pass | Pass |
| 116 | `dalek_scalar52_as_bytes_zero` | Pass | Pass |
| 117 | `dalek_reduce_pipeline_zero` | Pass | SSA definedness |
