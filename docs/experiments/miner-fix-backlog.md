# CuMetal miner fix backlog

Latest: [complete 238-attempt sweep on `6d2549b`](miner-self-test-sweep-6d2549b.md).
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
- [ ] **Trap handling.** Bounded call-free kernel reporting is implemented (fix 7).
  Helper propagation, user barriers/collectives and full context-failure semantics
  remain open. Never silently turn traps into no-ops.
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
