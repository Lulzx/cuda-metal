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

Unannotated 64-bit PTX parameters now require address-use evidence to be
classified as pointers; unused or ambiguous parameters remain scalars in the
parser. The typed importer may recover omitted pointer annotations from its
supported address paths and scalar parameter slots forwarded to already imported
pointer-taking helpers. Typed MSL and metallib sidecars use imported argument
types and scalar sizes; PTX static shared-memory reservations retain the existing
PTX calculation because import does not yet populate that IR field. Hidden
bindings are not user launch arguments. The legacy backend retains parser
classification and has no typed recovery: ambiguous pointers require `.ptr`
annotations instead of relying on the former width-only buffer default.

Commuted PTX address addition uses established pointer provenance to choose
the address operand during backward inference. The proof is bounded to twelve
passes through single-definition, unpredicated 64-bit register paths rooted
in known pointer parameters, address conversions, or local/shared/global symbols.
Unknown or reused paths retain the existing operand-one recovery fallback; this
is not a general register-provenance solver. Integer-minus-pointer forms remain
rejected. Parser inference does not attribute combined addresses to a scalar
parameter when another source has known symbol-address provenance; ordinary
untracked thread-index arithmetic retains its existing classification.

A translation-unit-private PTX `.global` already proven immutable and promoted
to Metal constant storage retains that physical storage through supported global
address conversions. Single-definition aliases are tracked during import.
Generic helper conversions retain a PTX-global constraint until call-site storage
is resolved; constant-space origins must all come from explicitly promoted
globals. Ordinary PTX `.const`, unknown constant origins, and private/shared
conversions do not qualify. Mutable globals retain their device storage.
Mutation through supported same-function register aliases prevents promotion,
including predicated writes and register reuse. Stores through a helper pointer
that resolves to constant storage are rejected; interprocedural mutation-based
reclassification of private globals is not yet implemented.
