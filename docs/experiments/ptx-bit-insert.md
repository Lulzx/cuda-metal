# PTX bit-field insertion

The typed importer supports unpredicated `bfi.b32` and `bfi.b64`, with register
or immediate sources, position and length. Position and length use their low
eight bits, as specified by [NVIDIA PTX](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi).
The insertion takes low bits from the first source and replaces the corresponding
bits in the second source. Zero length or a position beyond the destination
width leaves the second source unchanged; an overlapping field clips at the
most significant bit.

The lowering builds a low-bit mask, shifts it to the insertion position, and
merges the retained/inserted bits. Every shift count is masked to be strictly
below the operand width. Full-length masks and out-of-range positions use typed
selects; discarded expressions do not contain undefined full-width shifts.
There is no runtime helper or kernel-name special case.

Other instruction widths/modifiers, malformed operand counts, mismatched source
widths and predicated forms fail explicitly. This does not extend `bfe` support
or general predication semantics.

## Validation

Thirteen scoped CTests pass in the Release, PTX-only, binary-shim-disabled build.
Unit tests cover both widths, immediates/registers, unsupported forms and operand
width mismatches. Existing Rust SHA/vector, initialized-table, narrow-load and
multiline-call numerical regressions remain passing.

The Apple M5 numerical test covers all 65,536 position/length byte pairs for each
of three source patterns (all-zero/all-one in both directions and mixed bits).
High bits of both controls are set to verify the low-byte rule. Each case checks
32-bit insertion, 64-bit insertion, and the miner's immediate position=3,
length=13 form against an independent bit-by-bit CPU oracle. That is 196,608
input cases and 589,824 insertion results, with 16 intact output guard words.
This is exhaustive over the control-byte pairs for these patterns, not over all
possible source values.

```sh
ctest --test-dir build-rust-ptx-apple -R '^functional_ptx_bit_insert$' --output-on-failure
```

Compiler SHA-256: `de2d76d48bfb0679373f645df3a458de644132b54fb2646a64ec6272d6d9c90c`.

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="bit_insert" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=61375 grid=(3073,1,1) block=(64,1,1) unsupported_reason=""
NUMERICAL_PASS bit_insert: all 65536 position/length pairs x 3 patterns; b32/b64 and immediate insertion; guards
```

The full CUDA C++/offline-metallib suite and a fresh 123-entry inventory were not
run for this change.

## Historical Ed25519 result at 7c8e53c: numerical failure

The unchanged full miner PTX from commit
`36ca4ed2be0cb16e6028f8ebd74deffd07500adc`, Actions run `34774932262`, now compiles
`kernel_self_test_primitive_ed25519` to MSL. The MSL also compiles successfully
through the public Metal runtime and launches on Apple M5. Its self-test writes
**0 rather than 1 to result slot 2**; the other 117 result slots and 16 guard
words are intact. The failure reproduced with the committed runner. This is
an unresolved numerical mismatch, not a passing Ed25519 implementation.

The source check derives an Ed25519 public key from its fixed hashed private
input and compares the derived bytes with the expected public key. The generated
kernel contains the actual arithmetic; this is not a constant result-store test.
No full-miner throughput or correctness claim follows from launch success.

Original PTX SHA-256:
`41db05e2e1b6281923b9d9f8f157ced57f69a9a9e9d794d0513db6e618bbb6d7`.
Generated MSL SHA-256: `9d624643fb7fcbfca240b77ad89e87c4d6954d3bc2e7de40b823b73464715276`.

```sh
build-rust-ptx-apple/cumetalc /tmp/vanity-miner-34774932262/output.ptx \
  --backend=cumetal-ir --ptx-strict --overwrite \
  --entry kernel_self_test_primitive_ed25519 --emit=msl \
  -o /tmp/miner-bfi-ed25519.metal
python3 demos/rust-ptx/run_self_test.py /tmp/miner-bfi-ed25519.metal \
  --build-dir build-rust-ptx-apple \
  --kernel kernel_self_test_primitive_ed25519 --slot 2
```

The runner exits 1 for this numerical failure, checks every untouched slot and
guard, and frees the module, buffer and context. It is an artifact-dependent
manual diagnostic, not a passing CTest or a waiver.

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="kernel_self_test_primitive_ed25519" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=9999750 grid=(1,1,1) block=(1,1,1) unsupported_reason=""
NUMERICAL_FAIL kernel=kernel_self_test_primitive_ed25519 slot=2 actual=0 expected=1; other 117 slots and 16 guard words intact
```

The next investigation needs to locate the first arithmetic divergence against
a CPU reference; the previous missing-opcode compilation failure is resolved.

Resolved by the [halfword-tuple fix](ptx-halfword-tuples.md): the original
Ed25519 self-test now returns 1 on Apple M5 with all guards intact. The failed
run above is retained as historical evidence.
