# Contributing to CuMetal

Thanks for your interest. Before anything else, please read the **clean-room requirements**
below — they are not boilerplate. CuMetal's legal position (see
[docs/legal-notice.md](docs/legal-notice.md)) rests on every contribution being independently
written, and a single tainted patch would undermine that for the whole project.

---

## 1. Clean-room requirements

By contributing, you certify all of the following for **every** patch you submit:

1. **No NVIDIA proprietary source material** was referenced, copied, adapted, or consulted
   while writing the contribution. Public documentation — the CUDA Programming Guide, the PTX
   ISA specification, published header *interfaces* — is fine. Leaked headers, decompiled
   binaries, and internal NVIDIA source are not.
2. **No prior exposure to NVIDIA proprietary source code** for the specific API surface you are
   implementing. If you have worked on NVIDIA's implementation of an API, do not implement that
   API here. Contribute elsewhere in the codebase instead; this is not a judgment about you, it
   is about keeping provenance clean.
3. **No SASS.** CuMetal processes PTX, NVIDIA's documented virtual ISA. Do not contribute SASS
   decompilation, disassembly, or anything derived from it.
4. **Third-party code is attributed and license-compatible.** Apache 2.0 or a compatible
   permissive license, with the source recorded in the file header and in
   [docs/legal-notice.md](docs/legal-notice.md). The ZLUDA-derived PTX parser is the existing
   precedent.

Apple's AIR ABI is handled the same way: reverse engineering from *publicly distributed
toolchain output* (`.metallib` files that `xcrun metal` produced on your own machine) is the
sanctioned method. Do not contribute anything derived from Apple source code or from
decompiling an Apple binary.

If you are unsure whether something is contaminated, ask in an issue **before** writing code.
An unanswered question is cheaper than a patch we have to reject and quarantine.

## 2. Sign-off (DCO + clean-room CLA)

Every commit must carry a `Signed-off-by` line:

```bash
git commit -s -m "your message"
```

For CuMetal, that sign-off certifies both the standard
[Developer Certificate of Origin 1.1](https://developercertificate.org/) **and** clauses 1–4 of
the clean-room requirements above. The exact text you are certifying lives in
[docs/cla.md](docs/cla.md). Adding `Signed-off-by` with your real name and a reachable email is
how you sign it — there is no separate form to fill in.

Pull requests whose commits are not signed off will be asked to amend before review.

## 3. Development workflow

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(sysctl -n hw.ncpu)"
bash scripts/ci_report.sh build
```

See [docs/build.md](docs/build.md) for build options and [docs/testing.md](docs/testing.md) for
running individual suites. [AGENTS.md](AGENTS.md) describes repository layout and where each
kind of change belongs.

Before opening a pull request:

- Both configurations build and pass: Release with `CUMETAL_ENABLE_BINARY_SHIM=OFF` (what users
  install) and Debug with it `ON`. CI runs both.
- New behavior has a test that **fails without your change**. Verify this by reverting your
  change and watching it fail, not by assuming.
- `git status` is clean — no build artifacts, no generated binaries.

## 4. Testing standards

CuMetal is a translation layer, so the failure mode that matters most is a *silently wrong
answer*, not a crash. The project has shipped several of those (see
[docs/correctness-audit-2026-07-26.md](docs/correctness-audit-2026-07-26.md)), and every one got
through because a test asserted the wrong thing. Accordingly:

- **A test must check computed values, not that compilation succeeded.** Grepping emitted IR for
  a symbol name proves the symbol was emitted, not that it computes the right function. If a
  test can pass while the kernel returns garbage, it is not a test.
- **Never downgrade a wrong answer to a skip.** `exit 77` is for genuinely absent preconditions —
  no Metal toolchain, no external checkout. A kernel that runs and computes the wrong result is a
  **failure**. Check new harnesses with `ctest -V`, not just the pass count; a suspiciously fast
  "pass" is usually a skip or a stale artifact.
- **Rebuild inputs; never reuse a pre-existing artifact as a fast path.** Four harnesses once
  short-circuited on a previously built binary and would have stayed green through a total
  compiler regression. Fall back to a stale artifact only when the toolchain is genuinely
  unavailable, and print a warning saying the result does not verify the current build.
- **Prove GPU execution when you claim it.** Correct output alone does not distinguish a GPU
  dispatch from a host fallback. Assert on `CUMETAL_TRACE_GPU=1` provenance
  (`device=apple_gpu`, `launch_success=true`).
- **One kernel per function under test** where practical, so a single unsupported call cannot
  mask everything else. `functional_cuda_projects_libdevice_math` is the pattern to copy.

## 5. Scope and claims

Be precise in documentation about what is verified. "Runs llama.cpp" and "runs PhysX" are not
claims this codebase supports — one model on a covered kernel subset, and selected shape paths,
respectively. If you extend coverage, say exactly what you measured and on what hardware. If
something is approximate or a placeholder, mark it so the runtime can refuse it by default
rather than returning a wrong answer.

Intentional non-goals are listed in [docs/known-gaps.md](docs/known-gaps.md) and spec.md §2.2.
Please open an issue before building anything in those areas.

## 6. Reporting bugs

Useful reports include: the CuMetal commit, macOS and Xcode versions, chip (M1/M2/M3/M4), the
CMake configuration, the `.cu` or PTX that reproduces it, and `CUMETAL_TRACE_GPU=1` output.
Wrong-numerical-answer reports are the highest priority — please say what you expected and how
you obtained the expected value.

Security issues: see [SECURITY.md](SECURITY.md) — do not open a public issue.

## 7. License

Contributions are licensed under [Apache 2.0](LICENSE), matching the project.
