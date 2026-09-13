# Dual LLVM miner smoke checks

Artifacts: Actions run 34778991430, x86-64 LLVM 7 and LLVM 19 downloads.
CuMetal source: 28cb908; Release build `build-rust-ptx-apple`, PTX-only importer.
Both unchanged full modules contain 123 entries. ARM64 artifacts were not executed.

| Producer | Launch probe | Ed25519 known-answer test |
| --- | --- | --- |
| LLVM 7, sm_89 | Apple M5 numerical pass | Apple M5 numerical pass |
| LLVM 19, sm_100 | Apple M5 numerical pass | PTX import failure; no GPU launch |

Every passing run checked its selected result slot equals 1, the other 117
result slots remain poisoned, and all 16 trailing guard words remain intact.
This is one Ed25519 fixture, not arbitrary-input or full mining validation.

## LLVM 19 blocker

LLVM 19 emits a `.global .align 8 .u64 TABLE[1] = {TARGET};` symbolic
pointer initializer. The existing importer recognizes LLVM 7's eight-byte
masked-address initializer representation; this typed symbolic representation
is rejected. The probe passes because its reachable code does not use this table.
The next implementation should normalize the typed initializer into the existing
symbolic relocation representation and retain the private/read-only target,
whole-pointer-load, addressing and escape checks. Add positive GPU table reads
and negative initializer/use tests before retrying this unchanged artifact.
Further LLVM 19 blockers may appear after this first import failure is resolved.

## Reproduction

For each producer V=7 or 19 and entry K=kernel_self_test_stub or
kernel_self_test_primitive_ed25519:

```sh
build-rust-ptx-apple/cumetalc /tmp/vanity-miner-dual-34778991430/vanity-miner-x86_64-llvm${V}/output-llvm${V}.ptx --backend=cumetal-ir --ptx-strict --overwrite --entry "$K" --emit=msl -o /tmp/check.metal
# Only after successful compilation; slot 0 for stub, 2 for Ed25519:
python3 demos/rust-ptx/run_self_test.py /tmp/check.metal --build-dir build-rust-ptx-apple --kernel "$K" --slot "$SLOT"
```

## Input identity and captured evidence

Compiler SHA-256: `b413b0d5778b154209d00d6752256941583580971af946accba8faaf2c494013`

LLVM 7 PTX SHA-256: `da18e436c1b5f54f2b9e1c44b4a6a0d62997ac8322a5e786419e2da215a114dc`

LLVM 19 PTX SHA-256: `f5c3d523a33e9ecc73f896ed26a46cda58039b16b1859af170e1d3af289be709`

### llvm19-kernel_self_test_primitive_ed25519.compile.log

```text
cumetalc failed: unsupported initialized PTX declaration: .global .align 8 .u64 _RNvNtNtNtNtCsbZJm7v6G7YB_16curve25519_dalek7backend6serial3u649constants23ED25519_BASEPOINT_TABLE[1] = {_RNvNtNtNtNtCsbZJm7v6G7YB_16curve25519_dalek7backend6serial3u649constants40ED25519_BASEPOINT_TABLE_INNER_DOC_HIDDEN};
```

### llvm19-kernel_self_test_stub.compile.log

```text
wrote "/tmp/cumetal-dual-smoke-34778991430/llvm19-kernel_self_test_stub.metal"
```

### llvm19-kernel_self_test_stub.gpu.log

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="kernel_self_test_stub" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=5500 grid=(1,1,1) block=(1,1,1) unsupported_reason=""
NUMERICAL_PASS kernel=kernel_self_test_stub slot=0 actual=1 expected=1; other 117 slots and 16 guard words intact
```

### llvm7-kernel_self_test_primitive_ed25519.compile.log

```text
wrote "/tmp/cumetal-dual-smoke-34778991430/llvm7-kernel_self_test_primitive_ed25519.metal"
```

### llvm7-kernel_self_test_primitive_ed25519.gpu.log

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="kernel_self_test_primitive_ed25519" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=9680749 grid=(1,1,1) block=(1,1,1) unsupported_reason=""
NUMERICAL_PASS kernel=kernel_self_test_primitive_ed25519 slot=2 actual=1 expected=1; other 117 slots and 16 guard words intact
```

### llvm7-kernel_self_test_stub.compile.log

```text
wrote "/tmp/cumetal-dual-smoke-34778991430/llvm7-kernel_self_test_stub.metal"
```

### llvm7-kernel_self_test_stub.gpu.log

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="kernel_self_test_stub" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=5458 grid=(1,1,1) block=(1,1,1) unsupported_reason=""
NUMERICAL_PASS kernel=kernel_self_test_stub slot=0 actual=1 expected=1; other 117 slots and 16 guard words intact
```


## Typed initializer implementation follow-up

The importer now accepts a single `.u64` or `.b64` array element containing a
bare target symbol and normalizes it into the existing pointer relocation.
The shared resolver checks size/alignment, private read-only storage, 64-bit
addressing and non-escaping whole-pointer loads. Multi-element arrays, address
addends and arbitrary runtime relocation remain unsupported.

Fifteen focused compiler/GPU CTests pass, including the new
`functional_ptx_typed_relocated_table` numerical test and the existing byte-form
and `mov.b32` tests. The new GPU test uses host-provided indices to read a
16-byte table through `ld.global.nc.b64`; it compares every result and guards.
Negative unit tests caught an alignment check confined to the byte decoder;
the final implementation also enforces that invariant in the shared resolver.

The unchanged LLVM 19 Ed25519 entry now gets past the initializer and fails at:

```text
PTX register '%rd22026' is undefined on an incoming edge to block '$L__BB95_1'
```

The PTX contains a loop-carried self-reference:

```ptx
selp.b64 %rd22026, %rd22025, %rd22026, %p2299;
@%p2299 bra $L__BB95_12;
bra.uni $L__BB95_1;
```

This requires a separate control-flow/definedness investigation, including
whether the undefined arm can reach an observable use. No zero initialization
or verifier relaxation was added. The full LLVM 19 kernel remains unvalidated.

Final compiler SHA-256: `88d06aa741003baf1a4d4bb85fecda246d0c37a59bdff97f7a7dbb9980dd3ee9`

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="relocated_table" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=6250 grid=(5,1,1) block=(64,1,1) unsupported_reason=""
NUMERICAL_PASS relocated_table: 257 runtime-selected reads from 16 table bytes through relocated pointer; guards intact
```
