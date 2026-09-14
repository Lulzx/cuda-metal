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

PTX parameter inference preserves address-register provenance across stores;
stored scalar values do not redefine the destination address register.

PTX tuple-move inference retains the full packed width across CFG edges:
`mov.b64` packs produce 64 bits and unpack to 32-bit halves; `mov.b32`
retains its 32-bit pack and 16-bit half behavior.

Pre-SSA tuple normalization can remove one unobserved 32-bit half of a
`mov.b64` pack/extract pair. The packed register must have one definition and
one source occurrence in the function; both instructions must be unpredicated
and in the same block, with no intervening call or write to the observed source.
The discarded extraction destination must be `_` or a named register with no
source occurrences anywhere in the function. Other partial-definedness cases
remain subject to ordinary SSA validation; undefined bits are never initialized.
