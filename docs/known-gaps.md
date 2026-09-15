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

SSA allocation retains the inferred type of each PTX instruction definition,
so a register reused as a pointer does not retroactively retype earlier scalar
offsets. Block arguments and instructions synthesized during CFG normalization
retain the existing register-wide type seed. General scalar/pointer register
reuse across control-flow joins remains unsupported.
Literal-zero integer definitions passed to pointer block arguments are
materialized as typed nulls at the receiving edge. Nonzero integer inputs
to pointer block arguments are rejected; this does not provide general
integer-to-pointer conversion or repair unrelated scalar-offset joins.
An immediate that the typed importer has already assigned a concrete pointer
type is emitted with its exact 64-bit address and Metal address-space qualifier.
This covers nonzero dangling pointers selected inside one instruction; it does
not infer pointers from integer width, accept nonzero scalar/pointer block
arguments, or resolve conflicting address spaces.
Kernel-parameter values are bound before source-ordered dispatcher cases are
emitted. This permits a parameter load in a textually later block to serve a use
that it dominates in the CFG. Existing SSA validation still rejects paths that
bypass the load; this does not hoist ordinary instructions or repair undefined
registers.
Single-definition, unpredicated 64-bit values loaded inside a device helper
retain device pointer type when a later device-memory access proves that exact
definition is an address. This covers pointer fields in compiler-generated
private records; ordinary scalar fields remain integers. Reused, predicated,
narrow, and otherwise ambiguous loaded definitions remain conservative.
The Bitcoin self-test advances past its private-record pointer mismatches.
Trap-capable device helpers now share the kernel's hidden status binding instead
of being duplicated into one large kernel CFG. Helpers with cyclic CFGs poll
that status at loop boundaries, and callers stop before consuming a cancelled
return value. The exact Bitcoin module now translates to a 19.8-MB Metal source,
but a fresh Apple Metal compilation exceeded 10 minutes and 3.0 GB RSS. No full
Bitcoin self-test numerical pass is claimed.
Bounded guard specialization also follows literal predicate flags through
incoming branches and fallthroughs, including inversion. Predicate liveness
removes facts after their final possible use, so dead constants do not exhaust
the 128-live-fact limit; exceeding that live limit still disables the proof.
Direct `call`/`call.uni` preserves caller predicates declared at function-body
scope (including ranges),
except explicit return registers. Unknown/indirect call forms and undeclared
predicates remain conservative; nested declarations do not prove function-wide
locality. Overwrites invalidate these facts; exhausted
proof budgets still leave SSA to reject
unproven reads. This is not general path-sensitive register analysis.
That proof can cross bounded straight-line prefixes before a known branch,
preserving their memory operations. A separate bounded demand analysis removes
unobserved full-width scalar `mov.b16/b32/b64` chains, including retry cycles;
it does not remove other operations or initialize an undefined value. Copies
feeding stores, addresses, branches, or other retained operations remain live.
The P-256 public-key self-test now emits MSL through the guarded helper ABI.
The signature self-test still requires further guarded-SSA work; neither full
module has numerical Apple-GPU validation.
Repeated unchanged branch predicates use the known incoming edge value even
when their producing comparison is outside that bounded operand analysis.
Self-select guards can also use matching or complementary integer equality
comparisons from the preceding 64 instructions. Types and operands must match;
operand/predicate writes and calls discard these facts. Dynamic timer/counter
reads are excluded. This does not establish arbitrary predicate equivalence.
After incoming-register validation, trivial block arguments with one distinct,
type-identical input are folded, including loop self-references. Differing
definitions remain block arguments. This is not general loop optimization or
elimination of mutually dependent groups of block arguments.
Relocated immutable table pointers may pass through direct device helper
parameters when a bounded read-only proof covers every use, including forwarding
chains. Writes, pointer escapes, unknown/recursive calls, and ambiguous argument
staging still reject. This does not provide mutable global relocation support.

Repeated unsigned 32/64-bit register comparisons can now establish bounded
load/use path proofs before register SSA, including exact complements and
reversed operands. A bounded table also relates identical single `add.u32/s32/u64/s64`
and `cvt.u64.u32` expressions in different registers. It keeps at most 64 expressions;
operand/result writes, predication, calls, and eviction conservatively lose proof.
It can carry identities through at most eight unambiguous predecessor blocks and
256 instructions. A single-entry loop retains preheader identities only when no
instruction in its strongly connected region writes the result/input registers or
calls a helper; irreducible entries lose proof. It preserves the computations and
memory operations, and does not fold in-place updates, recursive
expressions, signed comparisons, or floating arithmetic. Full
RSA-PSS and P-256 signature compilation still encounter other guarded-SSA cases.
Pointer values stored in local-memory tables retain their address space when
loaded for indirect access, including scalar/vector stores and loads with
bounded dynamic indexing. Literal offsets distinguish pointer subtables inside
a larger compiler stack depot, and exact 64-bit cells may be proven even when
other fields in that subobject are integers. Homogeneous tables also support an
unknown dynamic index. Mixed pointer address spaces, ambiguous cells, escaped
subobjects, and unbounded mixed tables remain rejected.

IR verification computes immediate dominators and queries their tree intervals
instead of storing a hash set of dominators per block. Storage is linear in CFG
blocks and edges; the existing dominance relation is preserved, including the
convention for closed unreachable components. This reduces verifier overhead,
not the size of imported IR or the separate costs of PTX parsing/register SSA.
