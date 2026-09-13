# Halfword tuples resolve the Ed25519 numerical failure

The original Ed25519 self-test now passes on Apple M5. Its result slot is 1;
the other 117 result slots and 16 guard words are unchanged. This verifies the
artifact's single Ed25519 known-answer fixture, not arbitrary input coverage or
all miner kernels.

## Root cause

The miner packs signed-digit bytes into local memory using eight instructions
of the form `mov.b32 %r81, {%rs248, %rs255}`. CuMetal only had explicit tuple
lowering for `mov.b64`. The 32-bit form fell through to a scalar move and selected
the first register. Type inference also made the destination a 16-bit value,
so later stores wrote only half the intended word. The upper halfword was lost
and some local bytes were left unwritten before table selection.

A standalone GPU regression reproduced the corruption before the fix:

```text
word 0: got a5a50000, expected ffff0000
```

The low/high source halfwords were `0x0000` and `0xffff`. `0xa5a5` came from the
pre-filled output buffer, demonstrating the partial write. This explained a
concrete divergence before curve arithmetic, without changing the expected
Ed25519 result or substituting a special-case implementation.

## Fix and boundaries

`mov.b32` now has explicit two-halfword packing and unpacking. Packing widens
each 16-bit bit container, shifts the high half by 16, and combines it with the
low half. Unpacking extracts both halves from a 32-bit value; either destination
may be `_`, provided at least one real destination remains. SSA destination
collection and inference retain the corresponding 32-/16-bit widths.

These are the [PTX pack/unpack semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov).
Malformed tuples, wrong register widths, packing sinks/immediates, duplicate or
all-sink destinations, and predicated forms fail explicitly. Four-byte `mov.b32`
tuples remain unsupported. Existing two-word `mov.b64` behavior is preserved.
Register width checks use the importer's existing PTX register-container naming
conventions (including the parser's renamed bare registers).

## Validation

All fourteen scoped CTests pass in the Release, PTX-only, binary-shim-disabled
build. The GPU regression tests 65,541 inputs: all halfword values paired with
their complements plus five boundary/mixed cases. It checks packing, independent
unpacking from host input, and both sink positions, comparing 262,164 output
words and 16 guards. This is not exhaustive over all possible halfword pairs.
Unit coverage includes malformed forms and width mismatches; previous Rust
vector/SHA, bit insertion/permutation, narrow loads and nested calls still pass.

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="tuple_move" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=15875 grid=(1025,1,1) block=(64,1,1) unsupported_reason=""
NUMERICAL_PASS tuple_move: 65541 inputs; packing, independent unpacking, both sink lanes, guards
```

## Original full-miner proof

Producer: vanity-miner-rs `36ca4ed2be0cb16e6028f8ebd74deffd07500adc`, Actions run
`34774932262`. The original 25,087,361-byte PTX artifact was consumed unchanged.
Its Ed25519 check derives a public key and compares it to the fixture:

```text
089a23ffc422f53d114587012bb2c028492fabdabe1266bc9ad6698ac43016bb
```

PTX SHA-256: `41db05e2e1b6281923b9d9f8f157ced57f69a9a9e9d794d0513db6e618bbb6d7`.
Compiler SHA-256: `b413b0d5778b154209d00d6752256941583580971af946accba8faaf2c494013`.
Generated MSL SHA-256: `711d5d49c972180c2f16b42942660767f7b14c96702bb27664082d508d482f55`.

```sh
build-rust-ptx-apple/cumetalc /tmp/vanity-miner-34774932262/output.ptx \
  --backend=cumetal-ir --ptx-strict --overwrite \
  --entry kernel_self_test_primitive_ed25519 --emit=msl \
  -o /tmp/miner-tuple-ed25519.metal
python3 demos/rust-ptx/run_self_test.py /tmp/miner-tuple-ed25519.metal \
  --build-dir build-rust-ptx-apple \
  --kernel kernel_self_test_primitive_ed25519 --slot 2
```

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="kernel_self_test_primitive_ed25519" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=9607166 grid=(1,1,1) block=(1,1,1) unsupported_reason=""
NUMERICAL_PASS kernel=kernel_self_test_primitive_ed25519 slot=2 actual=1 expected=1; other 117 slots and 16 guard words intact
```

The runtime used public Metal source compilation with workload specializations
disabled. The unchanged original kernel executed its arithmetic and comparison;
it was not replaced by the earlier table diagnostic or a constant result store.
The full CUDA C++/offline-metallib suite and a fresh 123-entry inventory were not
run for this change.
