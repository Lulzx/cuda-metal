# ptx_sweep

Per-opcode PTX sweeps.

## `ptx_sweep_numeric` — numerical sweep (spec §10.2)

`ptx_sweep_numeric_test.cpp` **executes** each opcode on the Apple GPU and compares the result
bit-for-bit against a hand-derived PTX ISA oracle. Expected values are literals written from the
ISA specification, not computed by any code path CuMetal shares, which is what makes this a test
rather than a tautology. Every float case uses values exactly representable in binary32, so there
is no tolerance to tune.

Cases are classified `SUPPORTED` / `WRONG` / `UNSUPPORTED`. `WRONG` always fails. `UNSUPPORTED`
fails too unless `--allow-missing` is passed, so coverage cannot quietly regress. The output
buffer is poisoned before launch, so a kernel that never writes is caught rather than reading as
a zero that happens to match.

Covers integer arithmetic, division and remainder sign behavior, bitwise ops and shifts,
`popc`/`clz`/`brev`, float arithmetic, `sqrt`/`rcp`, and every `cvt` rounding mode including
round-half-to-even ties and negative operands.

Why it exists: the lowering-only sweeps below assert that lowering *succeeded* and grep the IR
for `define void @name`. That proves a function was emitted, not what it computes — every
silent-wrong-answer bug in `docs/correctness-audit-2026-07-26.md` would have passed them. On its
first run this harness found `neg.s32` returning a float sign-bit flip (`-(7)` → `0x80000007`),
which traced back to four name-matched body templates in `lower_to_llvm.cpp` that discarded real
PTX bodies; see `docs/known-gaps.md`.

## Lowering-only sweeps (Phase 1 scaffolding)

- `run_supported_ops.sh`: emits minimal PTX kernels for currently mapped opcodes and
  requires strict PTX->LLVM lowering success.
  - arithmetic: `add`, `sub`, `mul`, `mad`
  - special-register moves: `%tid.*`, `%ctaid.*`, `%ntid.*`, `%nctaid.*`
  - memory/addrspace: `ld/st` in shared/global/local + `cvta.to.*`
  - control/other: `bar.sync`, `setp`, `bra`
  - indirect object queries: `txq`/`suq` width, height, and depth; the numerical
    sweep executes all six against distinct GPU-resident descriptor values
- `run_unsupported_ops.sh`: verifies strict mode rejects unsupported instruction roots
  and forms (`foo`, `trap`, `tex`, `suld`, non-dimension `txq`/`suq`).
