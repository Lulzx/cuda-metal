# Known gaps

This is the maintained gap index. A missing item is not automatically supported;
current status must be backed by tests and evidence.

## Gap groups

- [Platform and legal boundaries](known-gaps/platform.md)
- [Compiler and toolchain gaps](known-gaps/compiler.md)
- [Runtime and CUDA semantic gaps](known-gaps/runtime.md)
- [Library shim gaps](known-gaps/libraries.md)
- [Verification, CI, and downstream gaps](known-gaps/verification.md)

## Highest-priority open work

1. Expand typed CuMetal IR beyond the now-matched reviewed compile corpus and
   broaden numerical coverage.
2. Establish a recurring verification mechanism outside GitHub Actions and
   commission the trusted Apple-GPU lane; the fixed 185-test Phase 4 denominator
   is now defined.
3. Validate genuinely distinct supported Xcode toolchains.
4. Finish runtime/library semantic matrices and bounded binary-container forms.

The named five-kernel Phase 5 release set is closed for its selected-set
criterion; broader performance claims remain explicitly out of scope.
The [matmul study](matmul-performance.md) measures a remaining custom FP32
CUDA-kernel gap to MPS; its source-tile optimization is not an automatic
compiler optimization or evidence of M1/NVIDIA performance. A separate
[bounded private-array compiler pass](compiler-performance.md) targets large
per-thread arrays; it does not automatically retile kernels or provide MPS parity.

The executable priority/evidence table is in
[the specification closure roadmap](spec-closure-roadmap.md).

PTX recursive device calls remain unsupported except for the strictly checked
scalar and read-all / replace-all local-buffer tail-self-call forms documented
in the [miner fix backlog](experiments/miner-fix-backlog.md). General local-frame
recursion and barrier-aware trap cancellation remain open; the bounded LLVM 19 xoroshiro
fallback now passes numerical validation.

Vector parameter-slot transfers support the checked unpredicated direct-slot
`v2.b64` form. Other vector parameter forms fail explicitly; register-indirect
vector loads are not expanded because destination/address aliasing needs
separate handling.

[Bounded kernel trap reporting](experiments/trap-reporting.md) supports unchanged
PTX in kernels and supported acyclic device-call graphs without user barriers/collectives.
Trapping/looping helpers expand into the cancellation CFG; proven finite helpers
can remain calls. Expansion-size limits and unknown builtin restrictions remain. It reports a
stream-level software launch failure; full context-abort semantics and arbitrary
precompiled trap ABI are not implemented.

PTX pointer-minus-64-bit-integer byte offsets preserve pointer address spaces,
including local reverse loops. Pointer differences, integer-minus-pointer, and
narrow pointer subtraction remain explicitly rejected. Both pinned base58
primitive self-tests now pass; full mining kernels remain unvalidated. See the
[miner fix backlog](experiments/miner-fix-backlog.md).

The [complete miner self-test sweep](experiments/miner-self-test-sweep-6d2549b.md)
recorded 64 compile failures on `6d2549b`: 47 trap/call restrictions, 9 LLVM 19 SSA
definedness failures, and 8 LLVM 7 pointer/type failures. All 174 launched entries
pass, including two plumbing probes; this does not establish arithmetic
coverage for constant-folded fixtures.

Since that sweep, bounded trap/call support clears the WIF compressed-mainnet
checks in both producers. Compressed secp256k1 now reaches MSL but remains
unvalidated: after parameter-store truncation is fixed, LLVM 19 compiles and
launches but returns 0 instead of 1 with intact guards; LLVM 7 timed out during
its earlier 180-second runtime attempt. These targeted results do not replace the full
historical ledger; see [fix 10](experiments/miner-fix-backlog.md#fix-10-trap-propagation-through-supported-device-call-graphs).

Integer `st.param` stores truncate wider registers to the instruction width
before call/return ABI handling. Mismatched argument byte widths are rejected;
equal-width bit reinterpretations retain their semantics. See
[fix 11](experiments/miner-fix-backlog.md#fix-11-parameter-stores-preserve-their-instruction-width).
