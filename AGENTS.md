# AGENTS.md — CuMetal Development Guide

This file guides day-to-day implementation work in this repository.

Start with `/Users/lulzx/work/cumetal/spec.md` before architectural changes.
If there is a conflict:

1. `spec.md` wins.
2. `AGENTS.md` is next.
3. `README.md` is status-oriented and may lag.

## Project Identity

CuMetal is a CUDA compiler/runtime targeting Apple Metal on Apple Silicon.

- Primary path: source recompilation with `cumetalc`.
- Secondary path: opt-in binary-shim compatibility (`libcuda.dylib` alias and
  `__cudaRegister*` path) when `CUMETAL_ENABLE_BINARY_SHIM=ON`.

## Current Focus

The repository is beyond the bootstrap phases. Focus implementation on:

1. Phase 4 correctness hardening (conformance, ABI compatibility, negative-path behavior).
2. Phase 4.5 shim coverage quality (cuBLAS/cuRAND behavior and tests).
3. Binary-shim hardening (additional fatbinary variants and registration parity) without
   regressing the source-first path.
4. AIR ABI tooling hardening where practical (`air_inspect`, `air_validate`), with explicit
   limitations tracked in `docs/known-gaps.md`.

## Repository Structure

```
cumetal/
├── compiler/
│   ├── frontend/          # .cu frontend path
│   ├── ptx/               # PTX parser/lowering
│   ├── passes/            # LLVM transformation passes
│   ├── air_emitter/       # LLVM IR -> metallib emission
│   ├── air_validate/      # metallib validation
│   └── cumetalc/          # CLI
├── runtime/
│   ├── api/               # clean-room CUDA headers
│   ├── driver/            # Driver API shim
│   ├── rt/                # Runtime API shim
│   ├── registration/      # __cudaRegister* path (opt-in binary shim)
│   ├── error/             # per-thread error queue
│   ├── cache/             # metallib cache
│   └── metal_backend/     # Objective-C++ Metal boundary
├── tools/
│   ├── air_inspect/
│   ├── metal_library_archive_bridge/
│   ├── ptx_diff/
│   └── cumetal_bench/
├── tests/
│   ├── unit/
│   ├── functional/
│   ├── conformance/
│   ├── air_abi/
│   └── ptx_sweep/
├── docs/
│   ├── air-abi.md
│   └── known-gaps.md
├── spec.md                # canonical design spec
└── AGENTS.md              # this file
```

## Non-Negotiable Constraints

- SIMD/warp width is fixed at 32; no runtime width mode switching.
- Do not use private Apple APIs.
- Keep headers clean-room; do not import/ship NVIDIA headers.
- Keep Metal API calls inside `runtime/metal_backend/` as the Obj-C++ boundary.
- Preserve per-thread CUDA error model behavior.
- Preserve allocation tracking and pointer-to-buffer resolution.
- Never prioritize binary-shim behavior over source-first correctness.

## Build and Test

```bash
# configure + build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# full test suite
ctest --test-dir build --output-on-failure

# binary-shim OFF validation
cmake -B build-nosshim -DCMAKE_BUILD_TYPE=Debug -DCUMETAL_ENABLE_BINARY_SHIM=OFF
cmake --build build-nosshim
ctest --test-dir build-nosshim --output-on-failure
```

## Workflow Expectations

- Any behavior change must include tests (unit and/or functional).
- Add negative-path coverage whenever adding compatibility behavior.
- Commit messages carry no tool attribution or co-author trailers.
- Keep `README.md` and `docs/known-gaps.md` aligned with implementation reality.
- If a feature is intentionally partial, document it explicitly instead of implying full support.

## High-Value Next Steps

Prioritize these when choosing follow-on implementation work:

1. Expand fatbinary/PTX compatibility coverage in driver/registration paths and add focused tests.
2. Close conformance gaps and reduce skip-only coverage where dependencies permit.
3. Improve AIR ABI documentation from inspected reference outputs.
4. Keep Phase 4.5 shims behaviorally consistent with CUDA stream/error semantics.

## Anti-Patterns to Avoid

- Reintroducing early-phase assumptions as if they are current.
- Treating placeholder behavior as complete compatibility.
- Landing behavior changes without updating tests and docs.
- Regressing existing `ctest` behavior without explicit justification.

## Quick References

- Spec: `/Users/lulzx/work/cumetal/spec.md`
- Known gaps: `/Users/lulzx/work/cumetal/docs/known-gaps.md`
- Status summary: `/Users/lulzx/work/cumetal/README.md`
