# Name-Match Audit — 2026-07-26

Complete audit of every place CuMetal makes a decision based on a **name it does not own** —
a kernel or entry-point name chosen by the user or by an upstream project.

## Why

`ptx_sweep_numeric`, on its first run, reported `neg.s32` of 7 returning `0x80000007` — a float
sign-bit flip. The cause was not the `neg` handler, which types correctly, but a table in
`lower_to_llvm.cpp` that replaced a kernel's whole body with a canned implementation when its
entry name contained `negate`. A kernel named `neg_but_actually_triples` whose PTX computed `x*3`
was emitted as `fneg`.

Two more sites of the same shape turned up immediately afterwards. This audit exists to find the
rest of them rather than wait for the next one to surface as a wrong answer.

## The distinction that matters

| Kind | Example | Verdict |
|------|---------|---------|
| Name is the **specification** | `llvm.nvvm.barrier0`, `__nv_sqrtf`, `%laneid` | Fine. Reserved namespaces with defined meaning; matching them *is* reading the IR. |
| Name is an **identity** | `emitted_kernel_metallibs.find(kernel_name)`, cache keys, locating an `.entry` in PTX text | Fine. Exact lookup of a thing that was registered under that name. |
| Name **selects behavior** | "this kernel is called `gelu_forward_kernel`, so emit GELU" | Hazardous. The name is a label the user chose; it does not constrain what the code does. |

Only the third kind is in scope. Its failure mode is a silently wrong answer: the kernel loads,
launches, and returns plausible data computed from a body the caller never wrote.

## Findings

### 1. `lower_to_llvm.cpp` — four body templates — **REMOVED**

`vector_add`, `matrix_mul`, `negate`, `reduce_sum`. Substring match on the entry name plus a
loose parameter-shape check replaced the real body. Consulted **before** generic lowering was
attempted, which also bypassed `--ptx-strict`. The match additionally mutated the ABI (retyping
parameters, appending a thread-position builtin), which then made generic lowering fail — so the
fallback that produced the wrong body was triggered by the name match itself.

Removed outright. The generic path lowers every affected case; the suite stayed green.

The unit and AIR ABI fixtures that "covered" these were artifacts of them: each had a stub body
(`mov.u32 %r0, %tid.x; ret;`) while asserting a fully computed one. They verified the templates,
not the compiler, and could never have caught the miscompile. Now real kernels with real
assertions, plus negative tests.

### 2. `lower_to_metal.cpp` — MSL specialization table — **DEMOTED TO FALLBACK**

67 substring matches covering llm.c and GGML kernels. The code said what it did:

> `// First: try the hardcoded name-based lookup for known llm.c kernels.`
> `// Second: if no hardcoded match, attempt generic PTX → Metal translation.`

Generic translation now runs first; the table is consulted only when it declines. llama.cpp
coherence still passes, and the llm.c parity gate is no worse than before the change — measured
2/15 failures on current `main` against 4/15 on `fbaece1`, an intermittency that predates this
work (see below). So the table was only ever load-bearing where generic cannot reach.

The patterns are a mix of specific identifiers (`encoder_forward_kernel3`, `adamw_kernel2`,
mangled names like `_ZL10cpy_scalarI`) and ordinary words a user kernel could plausibly contain:
`silu`, `cpy_`, `q5_0`, `flash_attn`, `rms_norm_f32`, `rope_norm`. Seven of the riskiest are
swept by `unit_ptx_lower_to_metal`, each paired with PTX that doubles its input.

**Residual risk:** where generic translation genuinely cannot lower a kernel, a name collision
still selects the specialized body. Narrower than before but real, and reported as
`specialized_msl` / `workload_specialization` in `CUMETAL_TRACE_GPU=1` provenance.

The "fast negative filter" inside this table (`mul_mat_q`, `flash_attn`, `silu`, …) returns *no
specialization* rather than blocking translation, so it never suppressed generic lowering.

### 3. `cuda_runtime.cpp` — launch argument count — **REMOVED**

A table of llm.c kernel names forcing an argument count on the **real GPU launch path**,
ungated by any flag. A kernel whose name merely contained `layernorm_forward_kernel3` would have
had 8 arguments forced regardless of its actual signature, binding the wrong number of buffers.

Instrumenting the branch showed it is **never reached anywhere in the test suite**, including the
llm.c and llama.cpp conformance gates, because ABI resolution now populates `arg_info`. It
protected nothing and carried only collision risk. Removed; the argument count is inferred from
the caller's null-terminated argv, which is at least driven by the call rather than the name. That
inference now emits a one-time warning, since reaching it means the ABI is unknown.

### 4. `lower_to_llvm.cpp` — FP64 emulation template — **KEPT, DEMOTED TO FALLBACK**

`looks_like_fp64_mul_add_signature` matches `*fp64*{mul,fma,add}*`. Unlike the four above this is
**documented** behavior — `docs/known-gaps.md` states FP64 emulation activates only for
name-matched kernels — so it is a disclosed limitation rather than a hidden shortcut. Now guarded
by `!use_generic_body`, so real lowered PTX always wins.

### 5. `cuda_runtime.cpp` — llm.c CPU emulation — **NO CHANGE NEEDED**

Name-matched, but correctly built: off by default behind `CUMETAL_ENABLE_LLMC_CPU_EMULATION`,
emits a `warn_once`, and reports `semantic_quality=cpu_fallback device=cpu` in provenance. Its
own source comment records that it *used* to intercept silently and was fixed. This is the model
the other sites should have followed.

## Cleared

- `llvm.nvvm.*`, `__nv_*`, `%tid`/`%laneid` matching in `nvvm_importer.cpp`, `ptx_importer.cpp`,
  `lower_to_llvm.cpp` — reserved namespaces; the name is the specification.
- `looks_like_metallib`, `looks_like_bitcode_signature` in `metallib.cpp` — byte-signature checks
  on file contents, not names.
- `registration.cpp` map lookups and PTX `.entry` text search — identity, not behavior.
- `metal_backend.mm` provenance parsing — reads markers CuMetal itself emitted, defaults to
  `"unknown"` rather than guessing. (`library_substitution` is recognized but nothing emits it —
  a dead branch, harmless.)
- `compiler/passes/`, `compiler/ir/`, `compiler/metal/`, `compiler/air_emitter/` — no
  entry-name-driven behavior found.

## Unrelated finding: the llm.c parity gate is intermittent

Verifying that the `lower_to_metal.cpp` reordering had not disturbed llm.c surfaced a separate,
pre-existing problem. `conformance_llmc_gpt2fp32cu` fails roughly 2-4 runs in 15 with a genuine
numerical divergence (`LOSS MISMATCH AT STEP 1: 3.752515 4.059707`, sometimes a `-inf` loss) at a
step that varies between runs.

Measured 4/15 on `fbaece1` and 2/15 on current `main`, so it predates this work. Ruled out:
CTest timeout, machine load, and the asynchronous registered-launch hazard path. Recorded in
[known-gaps.md](known-gaps.md); not root-caused.

## Rule going forward

An entry name may be used to **look something up** or to **read a reserved intrinsic**. It may not
be used to **decide what code means**. Where a name-keyed fallback is genuinely unavoidable, it
must: run only after real translation has declined, be reported in provenance, and be documented
as a limitation. See [CONTRIBUTING.md](../CONTRIBUTING.md).
