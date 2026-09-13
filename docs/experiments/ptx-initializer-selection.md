# Selected-entry initialized data

## Problem and existing behavior

The full miner baseline failed at one Ed25519 global in every invocation,
including the launch probe. `import_ptx` decoded all initialized data before
selecting the entry. Later code already filtered decoded data using references
from that entry and its reachable device helpers. The failure happened before
that filter could run.

At commit `8ce368e`, the decoder supported numeric byte arrays and integer
scalars, but no symbolic addresses. The follow-up below adds a bounded form of
address relocation. Separately,
initialized mutable globals need persistent storage; a write in another entry
must prevent a shared global from becoming an embedded constant.

## Change

1. Parse entries and build the existing direct device-call graph.
2. Collect conservative symbol roots from the selected entry and all reachable
   helpers. Match complete identifier tokens, including inside address operands,
   rather than substrings. Exclude registers, numeric literals, and strings.
3. Identify an initialized declaration by its storage/type/name header before
   decoding its values. If its identity cannot be determined, retain a hard
   error instead of assuming it is unused.
4. Decode only rooted initialized declarations, using the existing value and
   size validation. Preserve the later printf-scaffold filtering and whole-module
   write check that determines constant versus persistent mutable storage.

This is conservative function-level selection, not instruction-level dead-code
elimination or a complete PTX name resolver. A symbol mentioned in a reachable
function remains a root even if that instruction is later optimized away.

## Complete pointer initializer resolution

The decoder now indexes declarations and follows initializer dependencies from
selected roots. It recognizes numeric `.b8`/`.u8` arrays and complete eight-byte
address initializers of this form (all eight masks, in increasing byte order):

```ptx
.global .align 8 .u8 pointer[8] = {
  0xFF(table), 0xFF00(table), 0xFF0000(table), 0xFF000000(table),
  0xFF00000000(table), 0xFF0000000000(table),
  0xFF000000000000(table), 0xFF00000000000000(table)
};
```

The example is wrapped for readability; the existing declaration scanner still
requires a declaration on one line. These masks extract address bytes, as
specified by [NVIDIA PTX initializers](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#initializers).

This is symbolic constant propagation, not a runtime relocation loader. For a
private, immutable pointer object whose only uses are direct complete 64-bit
loads, each load becomes a symbol-address move before register/SSA typing.
The referenced numeric table is retained and its address has Metal `constant`
provenance. No invented numeric address or zero pointer initializer is emitted.
The same logic applies to reachable helpers, without recognizing kernel names.

Before folding, the importer checks uses across all parsed entries and helpers.
It rejects pointer-object writes, address escapes, partial/offset loads, and
volatile loads. Conservative register dataflow follows the table address through
address arithmetic and rejects indirect writes or passing/storing the address.
Other global initializers referencing the pointer/table also prevent folding.
This analysis can reject safe programs; it does not assume an unknown use is safe.

The initial supported form requires explicit 64-bit addressing, eight ordered
masks naming the same private numeric target, and power-of-two alignment of at
least eight for the pointer. Missing targets, cycles, mixed bytes, and incomplete
relocations fail explicitly. Pointer chains, address addends, `generic()` forms,
mutable/exported pointer objects, packed pointer fields, and arbitrary runtime
relocations remain unsupported. None is silently replaced with zero data.

## Verification

`unit_ptx_global_reachability` covers unused data referenced only by another
entry, direct and multi-hop device-helper references, address/offset forms,
exact symbol prefixes, numeric operands, alias cycles, invalid required
initializers, missing declaration names, exact retained bytes, and globals
written by another entry. The existing typed PTX tests preserve numeric global
and printf behavior.

`functional_ptx_unused_initializer` runs vector addition on Apple GPU from an
unchanged handwritten PTX fixture containing an unused symbolic initializer.
The existing Rust-generated vector/SHA-256 artifact and bit-permutation GPU
regressions also run alongside it.

The full 25,087,361-byte miner artifact from producer commit
`36ca4ed2be0cb16e6028f8ebd74deffd07500adc` was separately passed unchanged to
`cumetalc --backend=cumetal-ir --ptx-strict --entry kernel_self_test_stub
--emit=msl`. Its generated MSL ran on Apple M5 through the Driver API:
slot 0 became 1; the other 117 result slots and 16 guard words retained their
poison value. Provenance reported `generic_ptx_lowering`, `device=apple_gpu`,
and `launch_success=true`, with workload specializations disabled.

This is evidence for the full-module launch probe only. The original
[123-entry baseline](vanity-miner-ptx-inventory.md) remains a historical report;
a post-fix full inventory and numerical validation of other miner kernels are
separate work. Some fixed-input self-test kernels, including this artifact's
`kernel_self_test_primitive_sha256_32`, contain only a constant result store.
Their numerical pass would not prove runtime crypto execution; the separate
runtime-input Rust SHA-256 harness supplies that evidence.

## Relocation follow-up validation

`functional_ptx_relocated_table` checks 257 runtime-selected table reads on the
Apple GPU, through the byte-encoded pointer, against a CPU byte-decoding oracle,
with output guards. Unit coverage includes forward declarations, dependencies,
exact retained bytes, malformed masks, cycles, missing targets, partial reads,
escapes, other initializer references, and direct/indirect writes in other entries.

The same runner accepts a full-miner PTX path as an optional second argument:

```sh
python3 tests/functional/run_ptx_relocated_table.py build-rust-ptx-apple \
  /tmp/vanity-miner-34774932262/output.ptx
```

That mode preserves the entire supplied PTX as a prefix and appends a diagnostic
entry. It reads the original Ed25519 alias and all 7,680 words of the original
30,720-byte table in runtime-selected order, plus a repeat, checking 7,681
outputs and 16 guard words. The CPU oracle zero-fills the two implicit trailing
initializer bytes. This is a table/relocation test, not Ed25519 arithmetic.

The initial unchanged `kernel_self_test_primitive_ed25519` attempt failed at
line 303275 because the parser split a multiline call. The subsequent
[multiline-call fix](ptx-multiline-calls.md) gets past that point and reaches a
return-width error in `subtle::black_box`. The original Ed25519 kernel has not
passed compilation or execution.

Final validation passed all ten scoped CTests and the full-miner table diagnostic
on Apple M5; see [compiler checksum and GPU transcript](full-miner-table-relocation.txt).
The full CUDA C++/offline-metallib suite and a new 123-entry inventory were not run.
