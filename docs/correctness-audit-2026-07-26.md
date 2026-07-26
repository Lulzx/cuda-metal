# Correctness audit — 2026-07-26

A silent-wrong-answer audit of the CuMetal test suite and the two compiler bugs it
uncovered. Three defects were found and fixed; the suite went from 209 passed /
2 skipped to 209 passed / 0 skipped, and the classified CUDA project sweep from
eight passes and one numerical failure to nine passes.

Every defect here produced **wrong numbers rather than a crash or an error**, and
two of the three were concealed by test harnesses that reported a failure as a
skip. That combination — a silent miscompile behind a harness that green-washes
it — is the theme of this document.

Commits: `0209ede`, `4debf8e`, `c0ddf75`.

## How it started

A routine status check reported 209/209 passing with 2 skips. One of the skips,
`functional_cuda_projects_sgemm_2d`, was not a skip. Run under `ctest -V`:

```text
[0]: got 0.0000 ref 2.6865
FAIL: sgemm_2d: 16382 errors
SKIP: execution failed (likely due to incomplete PTX->Metal coverage for this project).
```

The kernel produced all zeros — 16382 mismatches out of 16384 — and the harness
downgraded that numerical failure to a skip. The pass count was therefore not
evidence of coverage, and the only way to see it was `-V`.

## Finding 1 — PTX `.local` stack depots were allocated at a fixed guessed size

**Severity: silent wrong answers. Any kernel with a stack frame over 256 bytes.**

`compiler/ptx/src/lower_to_llvm.cpp` hardcoded the frame size:

```cpp
struct LocalSymbolInfo {
    int size_bytes = 256;   // never set from the PTX declaration
};
```

`sgemm_2d` declares `.local .align 4 .b8 __local_depot0[288]`, holding
`threadResults[64]` at `%SPL+0` and `regM[8]` at `%SPL+256`. The lowering emitted
`alloca [256 x i8]`, so **every `regM` access fell outside the allocation**.
Out-of-range private memory reads zero rather than faulting, so the inner product
became `threadResults += 0 * regN`. The kernel launched with a correct grid,
correct threadgroup memory, ran to completion, and computed exact zeros.

Diagnosis path: `CUMETAL_TRACE=1 CUMETAL_DEBUG_REGISTRATION=1` confirmed a correct
launch (`grid=(2,2,1) block=(64,1,1)`, `static_shared=4096`), which ruled out
dispatch. Clearing the JIT cache and dumping with `CUMETAL_DEBUG_DUMP_PTX_DIR` /
`CUMETAL_DEBUG_DUMP_IR_DIR` showed the declared `[288]` against the emitted
`[256 x i8]`.

**Fix** (`0209ede`): `parse_ptx_local_depots()` reads the declared size and
alignment from the entry body, mirroring the existing `parse_ptx_shared_symbols()`;
the entry-body extraction both need was factored into `extract_entry_body()`. When
the depot size cannot be determined the lowering now **refuses** rather than
guessing, consistent with the project's existing policy of clean aborts over
silent garbage.

Covered by `unit_ptx_lower_to_llvm` (declared-size alloca, and refuse-when-unknown).

## Finding 2 — the cuda_projects harness masked compile failures

**Severity: a passing test could be verifying code that no longer exists.**

`tests/cuda_projects/_common.sh` discarded the compiler's exit status and never
invalidated the previous object file:

```bash
( "${CLANG_BIN}" -x cuda ... -c ... -o "${out_dir}/${src_cu%.cu}.o" 2>&1 || true ) \
    | grep -v -E '...' || true
```

Demonstrated with a deliberate syntax error against a previously-built object:

```text
compile_link returned: 0
run2 -> VERSION_ONE      # the binary from the PREVIOUS build
obj mtime: unchanged
```

The error text *was* printed, so it appeared in verbose logs — but the status was
thrown away, the stale `.o` was linked, and the test passed. A clean build
directory instead produced a confusing missing-`.o` link error, so this specifically
bit incremental local runs. Blast radius: all cuda_projects coverage (six CTest
entries, the sweep, and the strict variants).

**Fix** (`4debf8e`): propagate the compiler's exit status, remove the object and
binary before compiling, and filter known non-fatal warnings out of the printed
log only — never out of the exit status.

Separately, `run_standalone_cu.sh` no longer downgrades a wrong answer to a skip.
A kernel that runs but computes wrong results is a failure; only genuinely
unavailable lowering (`registered kernel missing metallib`) skips. The distinction
already existed in the script — it simply was not applied to the numerical-failure
branch.

## Finding 3 — PTX→MSL pointer bases were resolved per entry, not per use

**Severity: silent wrong answers. Triggered by ordinary nvcc register reuse.**

Found by making Finding 4's dead test actually run.

`compiler/ptx/src/lower_to_metal.cpp` kept **one** register→classification map for
the whole entry. Four consumers ran after classification and looked registers up in
that *final* map: address validation, element-type inference, global load/store
emission, and atomic emission. A register reassigned from one pointer base to
another therefore resolved **every** use to the last base assigned anywhere in the
kernel.

```ptx
add.u64     %rd3, %rd0, %rd2      ; param_in  + offset
ld.global.f32 %f0, [%rd3]
neg.f32     %f1, %f0
add.u64     %rd3, %rd1, %rd2      ; param_out + offset — same register
st.global.f32 [%rd3], %f1
```

lowered to:

```metal
float vf0 = param_out[gid];   // wrong — param_in dropped entirely
float vf1 = -vf0;
param_out[gid] = vf1;
```

Using a distinct destination register lowered correctly, which is what made this
easy to miss. Reusing a register across differing pointer bases is ordinary in
nvcc output, and the lowerer emitted wrong code rather than declining to match —
the dangerous failure mode for a pattern-matching lowerer that is otherwise
expected to bail out to the LLVM path.

**Fix** (`c0ddf75`): snapshot the address register's classification at each global
load/store/atomic while the forward classification pass is still at that
instruction (`addr_at_instr`), and resolve all four consumers against that
snapshot.

Covered by `unit_ptx_lower_to_metal`, and end to end by
`functional_runtime_ptx_lowering_regression`, whose `negate` fixture now
deliberately reuses one register across both bases so the fix is exercised through
metallib packaging and real GPU execution.

## Finding 4 — a regression test that never ran on any machine

**Severity: absent coverage presented as present.**

`tests/functional/run_runtime_ptx_lowering_regression.sh` passed
`--mode experimental` unconditionally, which always produced an unloadable
container, which always tripped its own skip:

```text
SKIP: experimental container produced (ptx lowering succeeded); full exec verification requires xcrun metallib
```

The inline comment claimed "In full-toolchain envs this runs". That was false: the
mode was not conditional on the toolchain, so the skip fired on every machine. In
the pre-fix file the skip sat at line 137 of 364, stranding roughly 225 lines —
the entire host-side execution verification — that had never executed anywhere.

Making it run exposed two fixtures that **could never have passed**, which is
presumably why it was left in that state:

- `negate` was a no-op stub (`neg.f32 %f1, %f0; ret;` — no loads, no stores, no
  thread indexing) while the host asserted `output[i] == -input[i]`.
- `reduce_sum` bound a device buffer to a `.param .u32` scalar, feeding the
  pointer's low bits in as the element count.

**Fix** (`4debf8e`): package validated metallibs with `--mode xcrun`; skip only when
`xcrun metal`/`metallib` are genuinely absent; fix both fixtures. All three kernels
(negate, reduce_sum, clamp_relu) now execute and are checked numerically.

## Suite-wide skip audit

All 74 `SKIP_RETURN_CODE` registrations and every skip site behind them were
reviewed. The pattern searched for was a skip taken *after* the thing under test
had run, as opposed to a precondition checked before any work.

Clean — precondition-only skipping:

| Area | Skip conditions |
| --- | --- |
| ~60 C++ functional / air_abi tests | input metallib or PTX missing; no Metal device |
| 16 air_abi scripts | missing `xcrun`, `metal`, `metallib`, or Xcode |
| 6 PhysX conformance scripts | non-Apple-Silicon; no Metal toolchain; no PhysX checkout |
| llama.cpp gate | strict — exit code, output length, fatal signals, CUDA errors, GPU provenance, coherence text |
| llm.c gate | capture-then-check with strict content assertions |
| `ptx_sweep` | `set -euo pipefail`, unguarded commands |
| `run_cuda_source_gpu.sh`, `run_strict_standalone_cu.sh` | hard-fail on wrong results; verify GPU provenance |

No `PASS_REGULAR_EXPRESSION` misuse; the single `WILL_FAIL` is intentional.

**Design note, not a defect:** `conformance_functional_suite` is registered with a
90% pass-rate threshold, so it goes green with up to 10% of functional tests
failing. It hides nothing today — every functional test is also registered
individually, and the script prints separate passed/failed/skipped counts — but the
aggregate gate is not a zero-failure gate.

## Verification

Each fix was checked against a deliberately broken build to confirm the new test is
not vacuous:

| Fix | Reverted behaviour | Test result when reverted |
| --- | --- | --- |
| `.local` depot sizing | restore `size_bytes = 256` | `FAIL: local depot alloca uses the declared frame size` |
| pointer-base snapshot | restore final-map lookup in `ld.global` | `FAIL: reused-base load reads the input parameter` |
| compile-failure masking | (reproducer script) | `compile_link returned: 0`, stale binary runs |

Final state, cold JIT cache:

```text
100% tests passed out of 209
The following tests did not run:   (none)
```

```text
Classifications: pass=9
```

`sgemm_2d` now reports genuine parity against its CPU reference:

```text
PASS: sgemm2DBlocktiling<BM=64,BN=64,BK=8,TM=8,TN=8> (M=128 N=128 K=64)
```

## Recurring shapes worth watching

The two compiler bugs were instances of two general patterns. Both are worth
grepping for elsewhere in the lowering code.

1. **A later pass re-reading a *final* map built by an earlier forward pass.**
   Correct while the forward pass builds it, wrong for every consumer afterwards.
   This was Finding 3. Any `std::unordered_map<std::string, …>` keyed by PTX
   register or symbol and consumed by a second iteration over the instruction list
   has this shape.

2. **A fixed-size constant standing in for a declared size.** This was Finding 1
   (`size_bytes = 256`). Under-allocation in private/threadgroup memory does not
   fault on Apple GPUs; it reads zero, so the symptom is plausible-looking numeric
   output rather than a crash.

Both produced silent wrong answers. Neither would have been caught by a test that
only asserts "the kernel launched" or "some bytes were produced" — the project's
existing provenance and coherence gates exist for exactly this reason, and are the
right model to extend.

## Limits of this audit

Stated plainly so this document is not read as a broader guarantee:

- The skip audit covered **skip-as-failure only**. It did not assess whether the
  non-skipping tests assert enough, or whether their tolerances are meaningful.
- Finding 3 was fixed at the four consumer sites that resolve an address register.
  The rest of the lowering was **not** audited for the same flow-insensitivity
  pattern; pattern 1 above is a lead, not a cleared area.
- Coverage gains are limited to what the existing fixtures exercise. `sgemm_2d`
  passing means one 2D-blocktiled SGEMM shape is correct, not that register-tiled
  kernels in general are.
- General CUDA compatibility and broad llama.cpp GPU offload remain incomplete, as
  [status.md](status.md) and [known-gaps.md](known-gaps.md) describe.
