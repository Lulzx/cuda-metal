# Narrow loads and helper return types

The full miner's `subtle::black_box` helper loads a byte into `%r1`, then stores
that register to its declared `.b32` return parameter. The typed importer inferred
the unsigned load's result as an 8-bit integer, even though `%r1` is a 32-bit
register. The return validator correctly rejected the resulting IR width mismatch.

[PTX load rules](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
require integer loads to extend to the destination register width: zero extension
for unsigned/bit types, sign extension for signed types. The importer previously
handled widening only for signed loads.

## Change

Ordinary integer memory loads now infer the width independently for each
destination register, including vector lanes. `memory_bit_width` and signedness
remain attached to the load, so the existing Metal emitter reads only the
requested bytes and performs the correct extension. This does not enlarge the
memory access or relax the return ABI validator. Parameter loads keep their
separate ABI/pointer handling. This is not a general repair of partial parameter
accesses, arbitrary register typing, or volatile memory conformance.

## Tests and evidence

A small helper with a local byte store/load and a `.b32` return reproduced the
original error before the change. It passes after the inference fix. Negative
coverage retains rejection of a 32-bit result in an incompatible 64-bit return
slot. Additional IR checks verify separate memory/result widths for mixed
16-/32-bit vector destinations and a 64-bit scalar destination.

`functional_ptx_narrow_loads` runs six helper paths (`u8`, `b8`, `s8`, `u16`,
`b16`, `s16`) across all 65,536 two-byte inputs. Host inputs have poisoned upper
bits. Each helper loads its parameter, stores to local memory, loads into a
32-bit register, and returns through a 32-bit slot. All 393,216 returned values
match the CPU extension oracle; 16 output guard words remain intact.

All twelve scoped CTests pass in the Release, PTX-only, binary-shim-disabled
build, including the Rust vector/SHA, multiline-call, table-relocation and
bit-permutation GPU regressions. The full CUDA C++/offline-metallib suite was not
run.

```sh
ctest --test-dir build-rust-ptx-apple -R '^functional_ptx_narrow_loads$' --output-on-failure
```

Compiler SHA-256: `f02861d9f1e52c84455a514fee18a6213aa168eb4efc9abd6d303d1d63f29fd4`.

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="narrow_loads" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=31000 grid=(1025,1,1) block=(64,1,1) unsupported_reason=""
NUMERICAL_PASS narrow_loads: 65536 inputs x 6 helper return paths; unsigned/bit zero extension, signed extension, guards
```

## Full miner follow-up

Producer: vanity-miner-rs `36ca4ed2be0cb16e6028f8ebd74deffd07500adc`, Actions run
`34774932262`. PTX SHA-256:
`41db05e2e1b6281923b9d9f8f157ced57f69a9a9e9d794d0513db6e618bbb6d7`.
The unchanged artifact was compiled with `--backend=cumetal-ir --ptx-strict
--entry kernel_self_test_primitive_ed25519 --emit=msl`.

The return mismatch is gone. Compilation now reaches line 328781 and fails with:

```text
PTX opcode 'bfi.b32' has no CuMetal IR normalization
```

This establishes progress past helper return typing, not a successful Ed25519
kernel compilation or execution. The full 123-entry inventory was not rerun.

Follow-up: [bit-field insertion](ptx-bit-insert.md) now has typed lowering and
numerical tests for both 32- and 64-bit forms.
