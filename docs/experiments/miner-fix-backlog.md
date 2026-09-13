# CuMetal fixes from the partial miner sweep

Stopped at user request after 121/238 attempts on compiler `967d8c7`.
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
- [ ] **Device-call cycle rejection.** Inspect rand_xoshiro seed/from_seed call
  graph and compare with PTX definitions. Distinguish real recursion from a
  call-graph/importer bug before changing recursion handling.
- [x] **Local-helper pointer IR verification failures.** Preserve the full verifier diagnostics and
  reduce secp256k1 failures; the first-line error alone is insufficient diagnosis.
- [x] **64-bit mul.hi support.** Reproduce the operand combinations in base58/WIF; cover
  signedness, narrow/wide boundaries and independent numerical expected results.
- [ ] **Trap handling.** Separate provably unreachable panic paths from reachable
  device traps. Preserve failure semantics; do not silently turn traps into no-ops.
- [ ] **Additional SSA definedness.** Reduce the LLVM 19 base58 loop failure and
  prove any rewrite separately from the existing Ed25519 guarded-select case.
- [ ] **Retry affected entries after fixes**, recording newly exposed errors and
  genuine numerical failures as separate backlog items.
- [ ] **Resume uncompleted entries**, then rerun previously failing entries once
  shared fixes justify it. Preserve the original partial report as historical.

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
LLVM 19's local-memory rand_xoshiro cycle and the remaining base58 SSA case
remain open; their semantics require separate analyses before code changes.

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
