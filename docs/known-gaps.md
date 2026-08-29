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

1. Bring typed CuMetal IR to legacy PTX production and numerical coverage.
2. Replace source AOT's remaining CUDA registration/first-launch PTX dependence
   with the versioned native ABI required by the spec.
3. Define the Phase 4 conformance denominator and enable real hosted/self-hosted
   CI without turning skips into passes.
4. Validate genuinely distinct supported Xcode toolchains.
5. Finish runtime/library semantic matrices and bounded binary-container forms.
6. Expand the performance release set beyond three kernels.

The executable priority/evidence table is in
[the specification closure roadmap](spec-closure-roadmap.md).
