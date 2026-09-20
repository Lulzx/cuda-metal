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
source occurrences anywhere in the function.

The #134 extension also recognizes a sole exact `cvt.u16.u64` or `cvt.u32.u64`
consumer. It removes the pack and converts the unchanged low32 source directly,
preserving unsigned narrowing and any wider destination container's extension.
Register declarations establish all widths; spelling does not imply a type.
Predication, calls, multiple packed uses/definitions, changed selected sources,
ambiguous declarations and other conversions retain ordinary SSA validation.
Whole-function counts and source-write generations are bounded; replacements
are staged and budget exhaustion leaves the function unchanged. Rejected
out-of-scope shapes are not necessarily semantically invalid PTX. This is not
general partial-value analysis, and undefined bits are never initialized.

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
is not a general register-provenance solver. Parser inference does not attribute combined addresses to a scalar
parameter when another source has known symbol-address provenance; ordinary
untracked thread-index arithmetic retains its existing classification.

A separate SSA pass on complete PTX functions recovers scalar counts when
64-bit `mov`, `add`, `sub`, `neg`, and `selp` paths provably cancel the same address
base. Branch and loop joins must agree on that base and its coefficient; shadow
integer offsets preserve wrapping arithmetic and register reuse. This includes
`base + (1 - cursor - length) + 30` when `cursor = base + written`, which becomes
`31 - length - written`. The pass retains actual pointer accesses and rejects
observable negative-base intermediates, unrelated bases, narrowing, and use of
the recovered scalar as a memory address without a proven pointer base.
After a successful rewrite, memory and call consumers must still receive a
proven pointer where their contract requires one; spilling a scalar count does
not grant it pointer provenance. Address conversions establish opaque roots:
equality of Metal address spaces
does not prove equality of PTX address bits across a conversion. Inline PTX with
external SSA bindings is outside this pass. Proof budgets bound analyzed values,
offset values, scalar-use traversal, liveness work, and generated names;
exhaustion leaves the original
unsupported expression rejected. General integer-minus-pointer expressions and
numeric pointer representations remain unsupported.

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

Promoted module globals are described in the `.cumetal-abi` sidecar as `global`
records, separately from the public argument list so a caller's argument count
is unchanged. A driver-API `cuModuleLoad` has no registration to own that
storage, so the module owns it: one allocation per symbol per module, seeded
with the compiler-recorded initializer (zero when the source had none),
reused by every launch so GPU writes persist, addressable through
`cuModuleGetGlobal`, and released on `cuModuleUnload`. Symbols the compiler did
not record remain `CUDA_ERROR_NOT_FOUND`. The typed backend emits one entry per
metallib, so cross-kernel sharing of a symbol within one module is implemented
but not exercised by the corpus.

A mutable module global referenced inside a reachable device helper is threaded
through that helper's signature as a hidden trailing pointer parameter and
passed at every direct call from the kernel, in module declaration order. The
raw PTX symbol spelling is never emitted into Metal. Registration metadata
discovers `.const` and `.global` symbols over the same reachable direct-call
closure, so the kernel's hidden binding is backed by the registered host shadow
instead of reading zeroed storage. Helpers that do not reference a global keep
their original signature; unreachable functions contribute nothing. Recursive
and indirect call graphs remain refusals.

PTX result types are resolved over the normalized SSA graph before materialization.
Conversions decode their destination independently of the source, wide arithmetic
uses its widened result, and predicate and tuple results retain per-lane contracts.
Copies, pointer arithmetic, branches and loops consume the actual reaching values;
a later assignment to the same register cannot seed an earlier definition or join.
Normalization records instruction origins for applicable memory proofs, while
clones keep distinct result identities. Imported result types are checked against
emission, and the IR verifier checks incoming edge types and dominance.

Direct PTX `clz.b32/b64` and `popc.b32/b64` retain their logical u32
destination through copies, joins and predicated updates. The 64-bit forms
combine two unsigned 32-bit Metal builtin results. Source bits, including
floating-register containers and integer literals, are preserved; unsupported
widths, modifiers and malformed operands reject explicitly. The generated u32
bit-count builtins are also accepted in trap-capable helper graphs.

PTX memory-address intermediates use byte pointers in the address's actual
storage space. Loads and stores retain their independent value types, and the
Metal emitter constructs the final typed dereference from those resolved types.
This avoids copying a provisional generic pointer field type into an earlier
address cast or offset, which could leave an unqualified nested pointer in MSL
after the field itself had resolved to device storage. The Metal IR verifier
recursively rejects unresolved nested pointer address spaces; concrete nested
pointer types and the existing tagged representation for top-level mixed
pointers remain supported. Focused unit coverage includes generic byte and
64-bit reads, generic stores, nonzero offsets, scalar fields, and proven private
and constant pointer fields. Full downstream compilation and GPU validation
remain separate gates. The memory-proof limits below still apply.
Pointer addition normalizes its sole pointer operand first, so both PTX operand
orders retain the same allocation identity in subsequent memory proofs.

Proven scalar zero definitions and copies may become typed nulls on pointer edges.
A packed tuple is not proved zero from its first lane. Concrete pointer inputs to
a generic pointer join receive explicit conversion values. Nonzero integer/pointer
joins, conflicting address spaces, undefined edges and unresolved cycles remain
errors. Memory pointer recovery separates generic pointer demand from concrete
address-space evidence. Local-cell candidates must pass a CFG reaching-store
check using final SSA operands; a store on another incoming path cannot supply
the load's type. Finite masked indices and small literal-initialized increment
loops with proven exit guards are supported. Bypassed bounds, unmodeled
conversions and incomplete discovery summaries supply no finite proof. Unresolved dynamic
addresses, partial writes, conflicting spaces and missing initialization remain
conservative proof limits. A call may preserve the proof when its imported body
is read-only or writes solely through addresses derived from its own private
allocations. Unknown, nested and external call effects remain barriers.
Backward pointer-demand discovery crosses only unique, unpredicated 64-bit
scalar aliases; reused aliases cannot reclassify an earlier scalar load.
Additional mixed-vector provenance is separate work. Call-return slots
also have per-definition SSA values: a reused slot name can hold different return
types on successive calls. Undefined or incompatible return values at an actual
join remain errors.
Argument-slot pointer recovery requires all associated signatures and stores to
agree; ambiguous reuse supplies no name-wide pointer evidence. Aggregate address
copies follow the reaching SSA value, preserving private storage and mutations.

Private-address `or.b64` is normalized to addition only when allocation-derived
known low bits prove the mask cannot carry. Copies, joins, selects and anchored
loops retain that proof; missing alignment, unrelated roots and exhausted
budgets supply no pointer evidence. Reaching-store validation can refine a
generic pointer load to one concrete space only after every incoming path and
overlapping store agrees. A later reuse of the cell-address register cannot
retag the earlier load.

Local-store disjointness also consumes unsigned guard bounds and bounded scalar
induction. Unit increments selected by a predicate are supported when every
backedge proves the selected value is one; unrelated or overwritten predicates,
OR-true branches, unknown seeds and unproved increments remain conservative. A separate bounded prefix analysis tracks initialized scalar bytes
beside pointer cells and finite pointer loops. It explores unknown paths and
keeps local addresses symbolic. Unknown effects invalidate facts; incomplete
exploration discards all observed bounds. These ranges establish disjointness,
not pointer contents or missing initialization. Arbitrary dynamic aliasing and
unbounded loops remain unsupported.

Store-range proofs retain bounds captured when a pointer is formed and merge
separately bounded origins through anchored copy/join cycles. Arithmetic
recurrences are not copies. Later scalar guards refine a captured offset only
when its SSA dependencies cannot change between creation and use; bounded
constant left shifts retain nonnegative intervals only without representational
overflow. Guard matching reuses affine origins before walking copy/join identities;
distinct concrete computations need no identity traversal. Completed identity
proofs are reused within the same immutable SSA analysis, with a bounded fact
count; exhausted attempts are not retained as semantic refusals. Unknown or
exhausted proofs remain conservative. The analysis keeps the existing work budget. These facts prove disjoint
writes only, never pointer-cell contents. The LLVM7 Ethereum self-test clears its recorded byte-store
range refusal with this correction, then still rejects an unproved reaching
pointer store; this is not full-module or GPU acceptance.

Address-demand discovery reuses the type solver's validated join index and
expands incoming edges only when a memory-address use demands them. It preserves pre-write definitions, all
demanded predecessors and independently pointer-typed helper-load candidates;
exhaustion discards partial demand sets. This reduces irrelevant join work without
raising the existing limit or supplying pointer provenance. Demand collection
shares the type solver's existing SSA walk, avoiding a second whole-function
result lookup, destination decode and environment rewrite. Only memory
instructions enter the collector; it uses one insertion lookup per demand and
borrows exact-copy and complete join edges from the same validated source index.
Memory observations and retained proof edges consume the unchanged work budget;
collectors are discarded when SSA is rebuilt. Other arithmetic entries remain
excluded. Concrete pointer values end demand traversal, but every address-conversion
source is independently seeded and applicable pointer-load candidates still undergo
reaching-store validation;
integer and generic-pointer paths retain their required edges. No second copy/join
index or per-edge register-name lookup is required. Provisional local-cell type hints
are invalidated by generic stores, so a former pointer cell
can subsequently hold scalar data. SSA reaching-store validation remains required
for later address uses, including scalar loads after hint invalidation. General
escaped-cell proof activation remains a separate limitation (#137). Fixed local `.v2`
and `.v4` 64-bit loads preserve pointer provenance independently per lane when
every reaching write proves that cell's contents; scalar metadata stays integer.
Partial, missing, conflicting and unsupported helper writes remain refusals.
Narrow vector lanes cannot supply full pointers. A bounded scalar analysis can
prove complete zero-length local cells before pointer demand, including exact
copies, agreeing incoming paths and unsigned `min(0,x)`. It removes only proved
impossible branch edges and rebuilds SSA; loads, stores and observable integer
sentinel bits remain unchanged. It recomputes guard facts after CFG pruning until
stable, with all rounds sharing a bounded analysis budget; a dead predecessor
cannot permanently hide a later zero-length guard. Local zero and pointer-field
checks share bounded store-range disjointness, including captured scalar offsets;
variable-address writes must remain inside their named allocation and outside the
loaded cell. Range indices are created only when exact addresses do not suffice.
Unknown, partial or overlapping writes, missing initializers, unsupported calls
and exhausted analysis supply no facts. The retained LLVM7 Solana self-test
module passes all 79 CPU/GPU checks with the matched `684bb03` package.
LLVM21 still rejects the separate helper-record proof; both-version acceptance
remains incomplete.
Ethereum and Bitcoin clear their empty private-key patterns but reach later
range/pointer-join failures. Zero markers survive scalar bitwise AND masks and
OR joins before SSA; a nonzero operand prevents an OR zero fact. The retained
Bitcoin LLVM7 input clears its masked-marker undefined value but still rejects
a later pointer/scalar branch argument. Small GPU fixtures are not full workload
acceptance. Explicit parameter slots with `+0` preserve the same
staged argument identity as an undisplaced slot. Once a zero length is proved,
unsigned comparisons can remove an empty iteration even when the index itself
is unknown; signed comparisons cannot borrow those identities. Register-width
lookups and completed exact-address facts are shared/reused within immutable
analysis, without relaxing budgets or reusing cell contents across writes.
Correlated pointer/length alternatives and dynamically indexed heterogeneous tables remain
outside this proof; an unused integer sentinel is not pointer provenance.

Scalar disjointness queries intersect dominating comparison bounds, trim excluded
interval endpoints, and relate plain 64-bit copy/add/sub siblings when their
shifted interval is representable. They do not bound the unobserved original
base from a modulo-wrapping sibling. Necessary facts from true AND, false OR,
predicate copies and negation are supported; opposite outcomes remain unknown.
Pointer iterators may use a positive literal length in the same allocation or
observe a selected zero marker as nonzero to prove the end was not reached.
Single-block fixed-stride pointer loops can also use a synchronized scalar
counter with a literal endpoint. Initial values, direction, every backedge,
no-wrap arithmetic and allocation bounds must be proved; equality endpoints
cannot be skipped. Unit-step scalar induction supports literal disequality
termination when every backedge observes the same update. Low-bit OR of a local
address is exact only within the allocation's declared alignment bits.
All-edge identity, overflow, entry and backedge checks remain required. Cached
completed scalar proofs survive later query-budget exhaustion; incomplete new
proofs remain unknown. That query limit does not bound the pre-existing scalar
constructor's SSA indexing work.

Passing a private pointer-cell address through a helper argument can suppress
candidate discovery before reaching-store validation. Without address
normalization, an overlapping byte-write control still emits an integer reload
and an invalid unqualified MSL pointer cast. The normalized path rejects that
control. This preexisting discovery gap is not a verified GPU result; escaped
cells need unconditional demand validation, separately from proving disjoint
caller-memory writes across helpers.

The issue #76 regressions cover reordered/renamed diamonds, zero-seeded and bounded
loops, guarded clones, conversions, wide products, tuples and rejected joins.

The #130 zero-marker proof carries exact unpredicated scalar zero moves/copies
into signed/unsigned/bitwise equality and inequality tests before register SSA.
Facts require unambiguous function-local 16/32/64-bit integer declarations with
matching instruction widths. Predicated writes, incompatible widths, unknown
calls and conflicting joins discard facts. A dynamic or private-address marker
does not imply nonzero: only the established zero path is specialized.
Known scalar-select predicates on an incoming edge can replace `selp.b32/b64`
with the selected move on that edge's clone, including a terminal block. Both
arms must have valid scalar forms/widths. Staged replacements use the same proof
that selected the edge; other paths, stores, loads and calls retain their order.
Declaration lookup, live facts, inspected paths and total clone growth are
bounded. Unsupported or exhausted proofs keep ordinary undefined-value
validation. These focused contracts are not arbitrary scalar constant
propagation, complete predication support or full downstream GPU acceptance.

The legacy PTX backend still cannot emit MSL for the joined ReLU fixtures; those
legacy numerical cells are reported as untested rather than successful. Passing
these focused tests does not establish full downstream compilation or GPU success.
Legacy LLVM lowering now uses declared register storage widths, including compact
register ranges, independently of register spelling or later opcode hints. Its
LLVM output passes the six ReLU assembly checks; this is not native legacy GPU
acceptance. Direct legacy MSL still lacks those general CFG forms, and the native
AIR/metallib route additionally requires the offline Apple toolchain.
Generic f16/f32-to-integer conversion retains destination signedness and integer
rounding mode, clamps to destination bounds, and handles NaN and subnormal bits
before any numeric cast. GPU fixtures cover all four integer rounding modes,
signed/unsigned 16/32/64-bit formats, direct/joined definitions and f32 FTZ.
Wider conversion destinations first produce the
instruction-format result, then sign-extend signed integer formats or
zero-extend other formats into declared integer storage. Wider integer source
containers are chopped before floating reinterpretation, including memory
stores. The numerical storage fixtures cover direct, copied and joined integer
results, f32 bit storage/round trips and f16 extraction. Saturating modifiers and
directed integer-to-f32 rounding beyond RN reject explicitly. The separate full
FP64 conversion matrix remains outside this measured gate.
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
A join whose only evidence is a null seed keeps that type provisionally and
stays out of conflict detection until a real incoming type settles it. Without
that, a loop-carried pointer whose other edge is a null constant deadlocked: the
latch committed to the seed's integer type before the pointer edge was known,
after which the header could never accept the pointer. The null proof still does
not propagate until every input proves zero, and a genuine non-zero integer
joined with a pointer remains a refusal once the provisional type resolves.

A device helper's own `cvta.to.global` or `cvta.to.local` proves its source is
a pointer, so a base arriving as a plain `.b64` parameter alongside a separate
integer index is recovered rather than guessed from 64-bit width. Bounds
branches and traps around the conversion are preserved, and conversions to a
conflicting address space remain refusals.

Single-definition, unpredicated 64-bit values loaded inside a device helper
retain device pointer type when a later device-memory access proves that exact
definition is an address. This covers pointer fields in compiler-generated
private records; ordinary scalar fields remain integers. Reused, predicated,
narrow, and otherwise ambiguous loaded definitions remain conservative.
Implicit kernel-address fields require a proven scalar offset for address
arithmetic. A failed address proof does not establish a scalar: cancellation,
two-address expressions and truncated address-derived offsets are rejected.
This preserves the existing scalar-byte argument ABI without admitting an
arbitrary integer as a pointer through a private helper record.
Private helper records with empty sentinel fields can use a bounded call-specific
zero context. An ordinary 64-bit load must have an exact in-bounds caller
allocation and a complete zero initializer on every incoming path, without an
intervening overlapping store or call. Identical proven contexts share a helper
clone; executable-edge scalar propagation removes only dead payload demands.
The sentinel's integer bits and live memory effects remain unchanged. Nonzero,
unknown, volatile, predicated, partially overwritten and observably used pointer
fields retain the original pointer proof. This is not general interprocedural
constant propagation: at most two rounds, 128 call contexts, 16 variants and
500,000 cloned operations are attempted, with bounded proof/transform work.
Small mixed empty/nonempty records pass numerical GPU checks. The unchanged
LLVM21 Solana self-test now emits MSL; full Apple preparation and numerical
acceptance remain pending for this correction.
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
Predicate threading has no separate eight-block depth cutoff. It follows a
cycle-free path until the existing 256-instruction inspection limit or shared
clone budgets stop the proof. Exhausting any bound still leaves SSA to reject
an unproven read; this does not initialize optional payloads.
Predicates assigned both literal presence states can also split a bounded live
region into true and false versions when they feed an absorbing `and.pred`
chain. The false version folds that chain and removes only dead, unpredicated
register operations from an exact integer/predicate whitelist. Loads, stores,
calls, traps, synchronization, unknown operations, escaped payload values, and
overwritten masks remain subject to strict SSA validation. The unchanged RSA
modulus self-test now emits MSL; its full Apple-GPU numerical run remains open.
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
