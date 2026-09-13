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

- [ ] **Unused pointer-to-integer conversions.** Start with black_box identity
  and small arithmetic checks. Determine whether results are used; never invent
  numeric pointer values to make observable casts compile.
- [ ] **Device-call cycle rejection.** Inspect rand_xoshiro seed/from_seed call
  graph and compare with PTX definitions. Distinguish real recursion from a
  call-graph/importer bug before changing recursion handling.
- [ ] **IR verification failures.** Preserve the full verifier diagnostics and
  reduce secp256k1 failures; the first-line error alone is insufficient diagnosis.
- [ ] **mul.hi widths.** Reproduce the operand combinations in base58/WIF; cover
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
