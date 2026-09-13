# Selected-entry initialized data

## Problem and existing behavior

The full miner baseline failed at one Ed25519 global in every invocation,
including the launch probe. `import_ptx` decoded all initialized data before
selecting the entry. Later code already filtered decoded data using references
from that entry and its reachable device helpers. The failure happened before
that filter could run.

The existing decoder supports numeric byte arrays and numeric integer scalars.
It does not support symbolic addresses or relocation expressions. Separately,
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

## Why this does not evaluate a global relocation graph

No accepted initializer can reference another global today: numeric bytes and
scalars contain no symbol dependencies. A reached symbolic alias is therefore
rejected by the existing decoder at the alias itself. An unused alias chain or
cycle can be omitted; a reached one cannot silently become zero data or cause
its target to be omitted from a successful compilation.

If symbolic initializers are implemented later, that change must introduce
proper global-to-global dependencies and address relocation semantics together.
This patch does not claim that support. Referenced `.u8` arrays and symbolic
address-byte expressions in the miner's Ed25519 alias remain unsupported.
The decoder's existing line-oriented declaration grammar is otherwise retained.

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
