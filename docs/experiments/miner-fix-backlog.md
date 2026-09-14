# CuMetal miner fix backlog

Full-sweep baseline: [238 attempts on `6d2549b`](miner-self-test-sweep-6d2549b.md).
Subsequent bounded trap/call support and targeted retests are recorded in fix 10 below.
LLVM 7 passes 90/118 numerical tests; LLVM 19 passes 82/118. Both probes pass.
All 64 remaining failures occur during compilation. Trap propagation accounts
for 47 first blockers, LLVM 19 SSA definedness for 9, and LLVM 7 pointer/type
issues for 8. No device-call cycle is a first blocker in this rerun.

The historical sweep below stopped at user request after 121/238 attempts on compiler `967d8c7`.
The remaining attempts were not completed, not counted as failures.
[Raw results and input hashes](miner-partial-sweep.json). All executed checks used
a fresh process, one GPU thread, a selected result slot and memory guards.
Folded fixtures remain limited evidence of arithmetic.

| Outcome/group | Attempts |
| --- | ---: |
| Device-call cycle rejection | 30 |
| numerical_pass | 6 |
| Trap lowering | 13 |
| mul.hi operand widths | 14 |
| IR verifier | 4 |
| Pointer-to-integer MSL cast | 52 |
| SSA definedness | 2 |

## Ordered fix backlog

For each item: reproduce with a small fixture, add a regression and relevant
negative cases, fix, run focused tests, retry affected original entries on both
LLVM artifacts, commit and push. Keep each fix separate. A cleared first blocker
may expose another; update this list rather than declaring the kernel validated.

- [x] **Unused narrow pointer-to-integer conversions.** Start with black_box identity
  and small arithmetic checks. Determine whether results are used; never invent
  numeric pointer values to make observable casts compile.
- [ ] **Device-call cycle rejection.** The targeted LLVM 7 scalar and LLVM 19
  local-buffer RNG cycles are fixed and numerically tested (fixes 4 and 6).
  Retry the remaining cycle failures; this is not a claim that all 30 pass.
- [x] **Local-helper pointer IR verification failures.** Preserve the full verifier diagnostics and
  reduce secp256k1 failures; the first-line error alone is insufficient diagnosis.
- [x] **64-bit mul.hi support.** Reproduce the operand combinations in base58/WIF; cover
  signedness, narrow/wide boundaries and independent numerical expected results.
- [ ] **Trap handling.** Kernel reporting (fix 7) now supports bounded device-call
  expansion and proven finite helpers (fix 10). User barriers/collectives,
  expansion beyond its limits and full context-failure semantics remain open.
  Never silently turn traps into no-ops.
- [ ] **Additional SSA definedness.** The base58 primitive passes, but the full
  rerun finds 9 LLVM 19 loop-definedness failures across variable-length base58
  and Dalek. Reduce and prove each rewrite; see the complete ledger.
- [ ] **Retry affected entries after fixes**, recording newly exposed errors and
  genuine numerical failures as separate backlog items.
- [x] **Complete all entries and rerun previous failures** on `6d2549b`.
  The complete ledger preserves the original partial report as historical.

Full numerical correctness and the four mining kernels remain later milestones;
see the [validation checklist](vanity-miner-validation-todo.md).

## Fix 1: dead narrow pointer conversions

The MSL emitter omits a pointer-to-integer conversion narrower than 64 bits only
when its sole result has no SSA uses anywhere in the function, including edge
arguments. Observable conversions still fail explicitly. Memory reads/stores
remain intact. Independent review found no correctness issue; a diamond/phi
negative regression was added in response to review.

18 focused tests passed; the strengthened unit test also passed afterward.
Original LLVM 7 and LLVM 19 black_box-u64 identity (slot 57) and SHA-256-32
(slot 8) now numerically pass on M5 with other 117 slots and 16 guards intact.
Logs: `/tmp/cumetal-dead-cast-retest`. This establishes four retested passes,
not that all 52 original cast failures are resolved.

The call-cycle investigation also confirmed a real self-call in LLVM 19's
`rand_xoshiro::from_seed` zero-seed fallback, which constructs a fixed nonzero
seed then calls itself. It is not merely a misidentified call-graph edge.
Preserving its semantics needs a separate recursion-elimination design or a
producer-side change; do not disable the cycle rejection.

## Fix 2: exact 64-bit mul.hi

The MSL backend now handles signed and unsigned 64-bit high-half products using
four 32x32 partial products with bounded carry sums. Signed results apply the
standard unsigned-high correction modulo 2^64. Independent review verified the
sum bounds and signed correction. GPU tests compare 4,217 operand pairs against
Python arbitrary-precision multiplication, including all boundary cross-products
and 4,096 deterministic generated pairs, plus signed/unsigned immediate forms.
All 19 focused tests pass.

Original LLVM 7 base58 and both LLVM WIF-compressed-mainnet entries now get past
mul.hi and stop at unsupported trap lowering. LLVM 19 base58 still stops at its
pre-existing trap. Logs: `/tmp/cumetal-mulhi-retest`. No numerical pass is claimed
for those entries. The trap backlog remains open.

## Fix 3: generic helper pointers converted to local addresses

Helper `cvta.to.local.u64` establishes pointer-ness for its input parameter even
when the signature omits `.ptr` and every subsequent access is local. The
parameter remains generic until call-site specialization. GPU-stage verification
permits an explicitly tracked generic cast source with a concrete target;
Metal-stage verification still requires concrete address spaces on both sides.
Integer and incompatible device arguments remain rejected.

Independent review found no soundness issue. Added direct verifier tests cover
tracked/untracked sources, concrete/generic targets, and GPU-versus-Metal stages.
The local-memory helper identity passes on M5; 22 focused regressions passed,
followed by the additional verifier-boundary test. Both original LLVM variants
of secp256k1-compressed now pass IR verification and stop at trap lowering.
Logs: `/tmp/cumetal-local-helper-final`. No secp256k1 numerical pass is claimed.

## Remaining design work

Trap lowering is now the shared first blocker for the retested base58, WIF and
secp256k1 entries. A faithful implementation needs a defined runtime failure
channel and propagation through helper calls, including GPU tests where a trap
actually executes. Removing traps or returning normal success would hide bugs.
The targeted RNG cycles are now handled (fixes 4 and 6). Other cycle failures
need retesting; the remaining base58 SSA case still needs separate analysis.

## Fix 4: scalar tail recursion and byte-array return packing

The LLVM 7 `seed_from_u64` helper has a scalar tail self-call. A conservative
pre-import rewrite turns this into a loop: load the initial argument once,
update it at the tail call, and branch back to the body. There is no iteration
limit, invented seed, or removed trap. Eligibility requires one scalar 64-bit
argument, a 16-byte return, exact forwarding of both return words, no other
calls, and no memory or pointer operations beyond parameter loads/stores.
Non-tail, predicated, malformed and memory-dependent cycles still fail.

This exposed a separate ABI mismatch: a byte-array return is represented by
four u32 fields, while the helper writes and the caller reads two b64 words.
Complete contiguous integer words now split/recombine with explicit 64-bit
typing and checked slot offsets. Holes, overlaps, malformed offsets and
out-of-bounds reads remain errors. Independent review caught an immediate
shift-typing bug and permissive offset parsing; both were fixed and retested.

Regression evidence on Apple M5:

- Tail-count helper: 66 runtime inputs, zero through 1,024 tail iterations.
- Unchanged extracted LLVM 7 seed helper: 261 runtime seeds checked against
  an independent SplitMix64 oracle (including zero and integer boundaries).
- Nonrecursive immediate aggregate returns: zero, one, all bits set, high bit,
  and mixed bits; checked through both b64 and independent b32 field reads.
- All GPU cases check output guards. The 23 focused compiler/GPU tests pass.
- Original full-module LLVM 7 `kernel_self_test_primitive_xoroshiro`: slot 0
  returns 1, with other 117 slots and 16 guards intact. Input is the pinned
  run above; compile output `/tmp/tail-rng-llvm7.metal`.

LLVM 19 uses a pointer into a local frame in its `from_seed` self-call; it is
intentionally ineligible for this scalar rewrite. A separate proof of complete
input consumption, frame reuse and pointer non-escape is needed. The cycle
backlog remains partially open, and no mining-kernel pass is claimed.

The constant-return test also exposed a separate unused unannotated b64 helper
parameter being inferred as a pointer, causing a Metal integer-to-pointer cast
error. The ABI-only regression uses a no-argument constant helper; the unused
parameter inference case remains a follow-up rather than expanding this fix.

## Fix 5: vector parameter transfers

Exact unpredicated `ld.param.v2.b64` / `st.param.v2.b64` transfers through direct
parameter slots now expand into two scalar transfers before SSA construction.
Both lanes therefore use the existing aggregate ABI and definedness checks.
Malformed tuples, duplicate/narrow destinations, nonliteral or misaligned byte
offsets, predication, and register-indirect slots remain rejected. The last
restriction prevents a first lane from overwriting the address of the second.
Independent review identified that alias hazard; a negative test covers it.

Nonrecursive GPU regressions cover immediate zero/one, all bits set, high-bit
and mixed-bit return words. Existing scalar-return and independent u32-read
regressions remain enabled. This repairs an ABI blocker encountered while
working on LLVM 19 local-frame recursion; it does not itself eliminate cycles.

## Fix 6: read-all / replace-all local-buffer tail recursion

The original LLVM 19 `from_seed` helper now compiles and numerically passes.
The transform accepts a narrow tail-call diamond: one pointer argument, a
16-byte private frame, all 16 distinct input-byte reads in a straight-line
prefix, scalar byte assembly, one full-frame replacement, a self-call with that
frame's address, and complete unmodified return forwarding. Pointer arithmetic
is restricted to the checked zero-offset aliases; addresses cannot be used as
scalar data or escape. There are no intervening calls, memory accesses or
side effects. Any shape outside this proof remains a recursion error.

The frame is safe to reuse because every input byte is consumed before the
replacement write; the next iteration rereads the complete replacement. No
iteration cap, fixed seed substitution, trap removal or frame zeroing is added.
Independent review found no issue in this bounded frame-reuse proof.

Evidence on Apple M5:

- The unchanged extracted helper passes **388 runtime inputs**: the all-zero
  seed (which actually takes the fallback), each individual bit in both words,
  boundary inputs and 256 deterministic random pairs. Nonzero seeds preserve
  both words exactly; zero returns the known SplitMix64 seed expansion.
- Nine negative cases reject incomplete/out-of-bounds input consumption,
  partial/offset frame replacement, pointer use as scalar data, predicated
  writes and observable post-call work.
- **24 focused compiler/GPU tests pass**, including the existing scalar-tail,
  independent return packing, address-space and arithmetic regressions.
- The unchanged full-module LLVM 19 `kernel_self_test_primitive_xoroshiro`
  returns **slot 0 = 1**, with the other 117 slots and 16 guards intact.
  The pinned input hash is in the checked-in helper fixture; compile/GPU logs
  are `/tmp/local-tail-rng-llvm19.log` and `.gpu.log`.

Fresh full-module base58 retests remain blocked in both producers:
`kernel_self_test_primitive_base58` reaches `trap has no faithful MSL source
representation` at PTX line 305647 (LLVM 7) / 406366 (LLVM 19). Logs are
`/tmp/local-tail-base58-llvm7.log` and `/tmp/local-tail-base58-llvm19.log`.
Neither base58 variant reached GPU execution. Trap propagation remains the
next shared blocker; no result from these RNG fixes establishes mining success.

## Fix 7: bounded kernel trap reporting

Unchanged PTX now lowers kernel traps to a per-launch failure buffer with SIMD
publication and cooperative cancellation at dispatcher boundaries. Runtime
completion checks retain and inspect the buffer, report launch failure and latch
it on the affected stream. Calls, user barriers/collectives and helper traps
remain rejected; this is not full CUDA context abort. See the
[design, regression coverage and limitations](trap-reporting.md).

26 focused tests pass. Five additional stream tests pass and ten old fixture
checks skip without `xcrun metal`. Review caught and tests now protect remapped
binding collisions, completion races, spinning peers, and store-before-trap
instruction ordering.

New base58 backlog: LLVM 7 now executes and reports a taken trap (719); trace
its failing translated path next. LLVM 19 now exposes Metal pointer subtraction
and missing pointer-address-space errors before GPU execution. Neither variant
has a numerical pass. Full mining kernels remain unvalidated.

## Fix 8: unsigned integer widening preserves the source width

LLVM 7 base58's taken trap was the alphabet bounds check (`$L__BB94_46`,
PTX line 305662). Temporary generated-MSL diagnostics found an out-of-range
numeric digit (165) before alphabet lookup. The input PTX was never modified.

The divide-by-58 sequence uses
`mul.wide.u32 %rd, %r, -1925330167`. That literal represents the u32 bit pattern
`0x8d3dcb09`. The emitter widened it as `ulong(-1925330167)`, yielding
`0xffffffff8d3dcb09`, rather than `ulong(uint(-1925330167))`. This corrupted the
quotient/remainder and eventually triggered the legitimate bounds check.

Unsigned integer widening now explicitly establishes the source-width bit
pattern before converting to the wider result. Signed widening retains its
existing signed-source cast; floating, pointer and narrowing conversions are
unchanged. A small GPU regression failed before the fix for input 1 with exactly
those differing products, independently of base58.

Validation:

- Six GPU cases, each with 263 boundary/runtime inputs: unsigned and signed
  16-bit and 32-bit wide products, negative-spelled constants, -1 and minimum
  signed constants. Both outputs are checked against Python integer arithmetic,
  including output guards (1,578 input/coefficient pairs, 3,156 products).
- 27 focused compiler/GPU tests pass.
- The original full-module LLVM 7 `kernel_self_test_primitive_base58` now
  **numerically passes on Apple M5**, slot 3 = 1, with the other 117 slots and
  16 guards intact. Trap reporting remains enabled. Input remains the pinned
  run `34778991430`; no diagnostic MSL instrumentation is used for this pass.
- Logs: `/tmp/widen-before.log`, `/tmp/widen-after.log`,
  `/tmp/widen-base58-llvm7.log` and `/tmp/widen-base58-llvm7.gpu.log`.

LLVM 19's previously observed pointer-subtraction/address-space compilation
errors remain the next separate base58 task. No LLVM 19 numerical pass or full
mining-kernel validation is claimed.

## Fix 9: pointer subtraction preserves address spaces

LLVM 19 base58's reverse loop subtracts an integer byte offset from a local
pointer. Type inference previously recognized pointer addition but lost the
pointer on subtraction. The generated MSL consequently assigned a pointer to
an integer and later cast that integer to a pointer without an address space.

The importer now preserves the left pointer's type for 64-bit pointer-minus-
integer operations and records subtraction explicitly on the pointer-offset IR.
MSL emits byte subtraction in the original address space. Pointer differences,
integer-minus-pointer, and narrow pointer subtraction fail explicitly.

Validation:

- A focused GPU regression checks local byte reversal through a loop and device
  pointer subtraction for 261 boundary/random inputs, with output guards.
- Three compiler negative cases cover the rejected subtraction forms.
- All 28 focused compiler/GPU regression tests pass.
- The unchanged full-module LLVM 19 `kernel_self_test_primitive_base58` from
  pinned run `34778991430` now compiles and **numerically passes on Apple M5**:
  slot 3 = 1, other 117 slots and 16 guard words intact. Trap reporting remains
  enabled; no diagnostic instrumentation is used.
- Logs: `/tmp/pointer-sub-base58-llvm19.log`,
  `/tmp/pointer-sub-base58-llvm19.gpu.log`, `/tmp/pointer-sub-tests.log`.

This closes the observed base58 primitive failures for both producers. It does
not establish complete base58 input coverage or validate the full mining kernels.

## Fix 10: trap propagation through supported device-call graphs

Trap-capable kernels now expand trapping/looping device calls into their CFG
before pointer legalization. Normal helper returns branch to a typed caller
continuation; traps never do. Fresh block/value IDs preserve independent call
sites, loop calls, local pointers and scalar/aggregate returns. Both expanded
GPU IR and Metal IR are verified. This keeps trap publication and backedge
polling in the existing kernel dispatcher, including spinning sibling helpers.

Helpers remain calls only after proving their CFG and transitive call graph
finite and free of traps, barriers/collectives, printf, unknown calls and atomics.
Integer min/max/abs builtins are finite expressions. Retaining these helpers
avoids excessive duplication of straight-line secp256k1 arithmetic. Expansion
still rejects excessive code growth; it does not disable cancellation to fit.
See [the compiler/runtime contract](trap-reporting.md) for bounds and semantics.

Validation:

- All 31 focused compiler/GPU regressions pass.
- Nested helper tests cover untaken/all/divergent traps, a spinning sibling
  helper across two SIMD groups, multiple scalar returns, retained finite helper
  chains, concurrent streams, repeated 719 errors and output guards. A one-thread
  case verifies the store before a nested trap survives and stores after that
  call never execute. Both tracing modes pass.
- 261 runtime inputs cover aggregate returns, local-pointer side effects, two
  distinct call sites and repeated calls inside a loop, with guards.
- Negative tests retain transitive barrier rejection and enforce expansion limits.
- Original, unchanged full-module compressed-mainnet WIF (slot 21) numerically
  passes for **both LLVM 7 and LLVM 19 on Apple M5**: selected slot = 1, other
  117 slots and 16 guards intact. Inputs remain run `34778991430`.
- Both compressed secp256k1 primitive entries (slot 4) now compile to MSL.
  LLVM 19 reaches Metal compilation, which rejects two `as_type<uint>(ulong)`
  conversions. LLVM 7's first compile/load/launch attempt times out after 180
  seconds with no numerical result. Neither secp256k1 entry is a numerical pass.

Logs: `/tmp/trap-expand-tests-final.log`, `/tmp/trap-expand-final-unit.log`,
`/tmp/trap-expand-{wif,secp}-llvm{7,19}.{log,gpu.log}`. The final unit rerun also
checks the updated diagnostic wording. A follow-up host-only staged probe for
LLVM 7 is `/tmp/trap-expand-secp-llvm7.stage.log`: setup APIs complete, then
`cuLaunchKernel` does not return within 60 seconds. This does not distinguish
Metal library/pipeline compilation from other work inside that API. PTX/MSL
bytes are unchanged.

This is a targeted retest, not a rerun of all 47 previously trap-blocked entries.
The historical 238-attempt ledger remains unchanged. Next: reduce the secp256k1
bitcast failure, investigate the LLVM 7 timeout, and retest the other affected
entries in small batches. Barrier-aware cancellation, unknown builtins, general
recursion, expansion beyond its limits and full CUDA context-abort semantics
remain unsupported.

## Fix 11: parameter stores preserve their instruction width

LLVM 19 compressed secp256k1 stores a 64-bit register into a 32-bit call slot
at PTX lines 615739 and 616651, then calls `subtle::black_box`. The importer
retained the register's i64 type rather than the 32 bits written by `st.param.b32`.
Argument ABI conversion subsequently emitted the illegal `as_type<uint>(ulong)`.

Integer parameter stores now truncate wider registers to the store width before
recording scalar/aggregate argument or return-slot values. Matching-width
reinterpretation remains a bitcast; incompatible argument byte widths now fail
explicitly at import instead of reaching Metal as an invalid bitcast. Return
width diagnostics also show the actual and declared types.

Validation:

- All 32 focused compiler/GPU tests pass.
- 267 boundary/random runtime inputs per width test 64-to-32, 64-to-16 and
  64-to-8-bit argument and return stores: 1,602 output comparisons plus guards.
- A negative case rejects a 32-bit stored argument passed to a 64-bit parameter.
- The unchanged original LLVM 19 compressed secp256k1 self-test now **compiles
  and launches on Apple M5**. It returns **0 rather than 1** at slot 4, with the
  other 117 slots and all 16 guard words intact. GPU trace reports successful
  launch (about 18 ms). This is a numerical failure, not a pass or a taken trap.
- Logs: `/tmp/param-trunc-tests.log`, `/tmp/param-trunc-secp-llvm19.log`,
  `/tmp/param-trunc-secp-llvm19.gpu.log`. Original PTX remains pinned to run
  `34778991430`; the generated MSL is not manually edited.

Next: isolate secp256k1's numerical mismatch with its existing arithmetic/curve
bisects. LLVM 7's earlier timeout has not been retested in this milestone. The
complete sweep remains historical; no new aggregate compatibility count is claimed.

## Secp256k1 isolation after fix 11

All six existing LLVM 19 k256 bisects now pass on M5: affine generator encoding,
projective generator encoding, doubling, scalar-one round-trip, and full public
key derivation for scalars 1 and 2. The original nontrivial-key test still returns
0 with intact guards. Its expected key matches an independent CPU calculation.

This narrows the mismatch to scalar-dependent behavior without identifying a
faulty opcode. Next compare scalar decomposition, signed digits/table selection,
and accumulation intermediates. See [the isolation report](secp256k1-isolation.md)
for evidence, source-path analysis and reproduction. No compiler change in this
milestone; LLVM 7 and the historical full-sweep counts are unchanged.

## Fix 12: cvt interprets the instruction's source width

The original LLVM 19 compressed secp256k1 primitive (slot 4) now **numerically
passes on Apple M5**: 1 in the selected slot, other 117 slots and all 16 guards
intact. PTX remains unchanged, pinned to run `34778991430`. The full original
kernel was regenerated by the compiler; its MSL was not manually edited.

Diagnostic MSL snapshots first established that decomposition, component signs,
positive magnitudes, all 66 radix-16 digits and all 16 table points agree with
independent CPU arithmetic. The first identified incorrect operation is in
signed-digit selection: `ld.local.b8` zero-extends a byte into a `.b16` register,
then `cvt.s16.s8` must sign-extend that byte. Import previously retained the
16-bit source type, so Metal lowered the conversion to an identity. Negative
digits became positive numbers outside the table's 1–8 index range.

Integer `cvt` operands held in wider registers now truncate to the instruction's
source width before signed/unsigned conversion. Pointer handling and existing
bitcast validation are unchanged. No scalar, table or cryptographic code changes.

Validation:

- Before the fix, the new GPU regression fails at byte `0x81`: absolute value
  129 instead of 127.
- After the fix, all 256 byte encodings pass conversion/absolute-value checks.
  Four additional cases each cover 263 boundary/random inputs: signed/unsigned
  8-bit sources in 16/32-bit registers, 16-bit in 32-bit, and 32-bit in 64-bit.
  Total: 1,308 inputs, 2,616 output comparisons, plus guards.
- All 33 focused compiler/GPU regression tests pass.
- Original slot 4 passes, GPU duration about 18.5 ms. Logs:
  `/tmp/secp-cvt-fixed.{compile,gpu}.log`, `/tmp/secp-cvt-tests.log`.
- [Diagnostic evidence and CPU reference checker](secp256k1-isolation.md#follow-up-correct-decomposition-and-tables-incorrect-signed-conversion).

LLVM 7's earlier timeout remains uninvestigated. Uncompressed secp256k1 and
composed/mining kernels have not been retested in this milestone. The historical
full-sweep ledger is unchanged. Next: validate the uncompressed LLVM 19 sibling
and then the dependent Ethereum/Bitcoin self-tests individually.

## Targeted validation after fix 12

At `93ebd6b`, three more original LLVM 19 entries compile and numerically pass
on M5: uncompressed secp256k1 (slot 5), Ethereum address (15), and Bitcoin's
complete Bech32 address (19). Each selected slot is 1; the other 117 slots and
16 guards are intact. PTX and generated MSL are unchanged by instrumentation.
No additional compiler fix was needed.

[Per-entry evidence and reproduction](secp256k1-dependent-checks.md). These
fixed fixtures do not replace the historical full sweep or validate the mining
kernels. Next: separately check Ethereum's private/public-key entries (13–14),
Bitcoin's private/public-key/hash entries (16–18), and its matching flag (20).
LLVM 7 remains outside this targeted batch.

## Ethereum/Bitcoin intermediate validation

At `b4894ed`, the six remaining LLVM 19 Ethereum/Bitcoin entries all compile
and numerically pass on M5: Ethereum private/public keys (13–14), Bitcoin
private/public keys and public-key hash (16–18), and Bitcoin matching (20).
All selected slots contain 1; the other 117 slots and 16 guards are intact.
Original PTX and compiler-generated MSL are not manually edited. No further
compiler fix was required.

Together with slots 15 and 19 from the preceding batch, **all 8 Ethereum/Bitcoin
self-test entries now pass with LLVM 19**. The matching fixture only checks a
positive `bc1q` prefix with an empty suffix. Full-suite totals, LLVM 7 coverage
and mining-kernel validation are unchanged.
[Results, scope and reproduction](address-intermediates.md).

Next small batch: the remaining WIF variants (22–24), then the remaining
previously trap-blocked LLVM 19 entries. Negative matching and concurrent mining
validation remain separate checklist items.

## LLVM 21 bounded-loop definedness

Bounded comparison-edge specialization and inverted self-select handling now
unblock Ed25519: numerical pass on M5 with guards intact. Unit negative cases
and 131,082 GPU regression inputs pass. [Proof, results, and remaining guarded-load
blocker in secp256k1](llvm21-loop-definedness.md). Original PTX is unchanged.

## LLVM 21 compound-predicate guarded load

The `lincomb` conditional-load definedness blocker is fixed by bounded incoming
edge specialization for repeated equality and predicate OR. Compressed
secp256k1 now numerically passes on M5 (slot 4 = 1, other slots/guards intact).
Unit positive/negative checks and 196,639 GPU regression inputs pass.
[Proof and evidence](llvm21-guarded-load.md). PTX remains unchanged.
Next: validate the LLVM 21 uncompressed sibling and dependent address checks.

## LLVM 21 uncompressed secp256k1 and address validation

At `f5e170f`, the uncompressed sibling (5), Ethereum checks (13–15), and Bitcoin
checks (16–20) all compile and numerically pass on M5. Each selected slot equals 1;
other slots and all 16 guards remain intact. No additional compiler change was
needed. [Nine-entry evidence and timings](llvm21-address-checks.md).
Next bounded batch: LLVM 21 WIF variants (21–24); full-suite and mining validation
remain incomplete.

## LLVM 21 WIF validation

All four network/compression variants (21–24) compile and numerically pass on M5
at `e4ba46b`, without further compiler changes. Every selected slot equals 1;
other slots and 16 guards remain intact. Original PTX is unchanged.
[Evidence, timings, and scope](llvm21-wif-checks.md).
Next bounded batch: Shallenge checks (25–27). Full-suite and mining validation
remain incomplete.

## LLVM 21 Shallenge validation

Hash, nonce-length arithmetic, and positive is-better checks (25–27) compile and
numerically pass on M5 at `debc412`, without compiler changes. Every selected slot
is 1; other slots and 16 guards are intact. PTX remains unchanged.
[Evidence, timings, and fixed-fixture limits](llvm21-shallenge-checks.md).
Next bounded batch: hash comparisons (28–30). Full-suite and mining validation
remain incomplete.

## LLVM 21 hash-comparison validation

Less-than, greater-than, and equal checks (28–30) compile and numerically pass
on M5 at `1a3fc92`, without compiler changes. Selected slots equal 1, with other
slots and 16 guards intact. PTX remains unchanged.
[Evidence and scope](llvm21-hash-comparison-checks.md).
Next bounded batch: arithmetic checks (31–40). Full-suite and mining validation
remain incomplete.
