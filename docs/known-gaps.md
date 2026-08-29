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
2. Extend native-AOT descriptors to writable/zero-initialized CUDA globals.
3. Define the Phase 4 conformance denominator and enable real hosted/self-hosted
   CI without turning skips into passes.
4. Validate genuinely distinct supported Xcode toolchains.
5. Finish runtime/library semantic matrices and bounded binary-container forms.
6. Expand the performance release set beyond three kernels.

The executable priority/evidence table is in
[the specification closure roadmap](spec-closure-roadmap.md).
