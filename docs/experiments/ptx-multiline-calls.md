# Multiline PTX calls

The parser now assembles `call`/`call.uni` statements through their semicolon
before splitting operands. A physical newline inside a call is whitespace, so
LLVM's return tuple, callee and argument tuple stay in one instruction. Calls in
entries and device helpers use the same path. See the [PTX call syntax](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#control-flow-instructions-call).

Predicates, scoped bare-register renaming and the call's original source line
are retained. Text after the terminating semicolon is processed separately,
including closing scopes. Block-comment removal retains newlines and separates
adjacent tokens. Unterminated calls, unbalanced parentheses and malformed
outer operands produce unsupported instructions rather than successful partial
calls. This change does not generalize multiline parsing to every PTX opcode,
or add new indirect-call or predicated-call execution semantics.

## Validation

Eleven scoped CTests passed in the Release, PTX-only, binary-shim-disabled build.
Parser tests cover predicates, return tuples, no-argument calls, comments,
physical line numbers, trailing statements, scope boundaries and malformed
calls. A typed-IR test proves identical emitted MSL for the same one-line and
multiline call; multiline recursive calls still fail the call-graph check.
The final additional scope and recursion cases also passed the two unit suites.

`functional_ptx_multiline_call` executes vector addition through two nested
helpers on Apple M5. Both call sites use multiline argument/return syntax, and
the input arrays come from the host at runtime. Counts 1, 31, 32, 33 and 257
match CPU results with intact tail guards. Workload specializations are disabled.
The fixture is handwritten; it is not evidence of a Rust-generated kernel pass.

```sh
ctest --test-dir build-rust-ptx-apple -R '^functional_ptx_multiline_call$' --output-on-failure
```

Compiler SHA-256: `8dd5c37cc154e546c59dd1f907fbd76a6bf1c829ed4f4b0cf0141478effd2316`.
Fixture SHA-256: `975366fb69c2ffba313d538f46979f87a38cbec6491b2d01d94d685e6499819b`.

```text
CUMETAL_PROVENANCE event=kernel_launch kernel="rust_vecadd" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=false launch_success=true duration_ns=7083 grid=(1,1,1) block=(64,1,1) unsupported_reason=""
CUMETAL_PROVENANCE event=kernel_launch kernel="rust_vecadd" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=true launch_success=true duration_ns=6625 grid=(1,1,1) block=(64,1,1) unsupported_reason=""
CUMETAL_PROVENANCE event=kernel_launch kernel="rust_vecadd" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=true launch_success=true duration_ns=6374 grid=(1,1,1) block=(64,1,1) unsupported_reason=""
CUMETAL_PROVENANCE event=kernel_launch kernel="rust_vecadd" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=true launch_success=true duration_ns=6750 grid=(1,1,1) block=(64,1,1) unsupported_reason=""
CUMETAL_PROVENANCE event=kernel_launch kernel="rust_vecadd" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu device_name="Apple M5" math_mode=safe compile_cache_hit=true launch_success=true duration_ns=6000 grid=(5,1,1) block=(64,1,1) unsupported_reason=""
DEVICE Apple M5
NUMERICAL_PASS kernel=rust_vecadd count=1
NUMERICAL_PASS kernel=rust_vecadd count=31
NUMERICAL_PASS kernel=rust_vecadd count=32
NUMERICAL_PASS kernel=rust_vecadd count=33
NUMERICAL_PASS kernel=rust_vecadd count=257
```

## Full miner result

The unchanged 25,087,361-byte artifact from vanity-miner-rs commit
`36ca4ed2be0cb16e6028f8ebd74deffd07500adc`, Actions run `34774932262`, was compiled
with `--backend=cumetal-ir --ptx-strict --entry kernel_self_test_primitive_ed25519
--emit=msl`. Its SHA-256 is
`41db05e2e1b6281923b9d9f8f157ced57f69a9a9e9d794d0513db6e618bbb6d7`.

The prior multiline-call parser failure is gone. Compilation now reaches the
`subtle::black_box` helper and fails at line 633959:

```text
PTX device return value does not fit its declared return type
```

That helper loads an unsigned byte into `%r1` then stores it to a `.b32` return
slot. The remaining failure concerns narrow-value/register/return typing and
needs its own numerical and negative-path tests. Ed25519 arithmetic has not run.
A fresh 123-entry inventory and the full CUDA C++/offline-metallib suite were not
run for this change.

Follow-up: the [narrow-load inference fix](ptx-narrow-load-returns.md) resolves
that return mismatch, with exhaustive byte/halfword numerical checks. The
unchanged Ed25519 entry now reaches unsupported `bfi.b32` normalization.
