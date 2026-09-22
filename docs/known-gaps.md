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
2. Establish a recurring verification mechanism outside GitHub Actions and
   commission the trusted Apple-GPU lane; the fixed 185-test Phase 4 denominator
   is now defined.
3. Validate genuinely distinct supported Xcode toolchains.
4. Finish runtime/library semantic matrices and bounded binary-container forms.

The named five-kernel Phase 5 release set is closed for its selected-set
criterion; broader performance claims remain explicitly out of scope.
The [matmul study](matmul-performance.md) measures a remaining custom FP32
CUDA-kernel gap to MPS; its source-tile optimization is not an automatic
compiler optimization or evidence of M1/NVIDIA performance. A separate
[bounded private-array compiler pass](compiler-performance.md) targets large
per-thread arrays; it does not automatically retile kernels or provide MPS parity.

The executable priority/evidence table is in
[the specification closure roadmap](spec-closure-roadmap.md).


## What will not work

Each entry is a refusal or a known semantic difference. Most refusals are
compile-time diagnostics rather than wrong answers. The linked group page
holds the exact boundary.

**Platform** ([details](known-gaps/platform.md))

- SASS-only binaries, non-Apple GPUs, multi-GPU/peer-to-peer, graphics interop.
- Native `double`: FP64 is software (`fast48` default: ~48-bit significand,
  binary32 exponent range; `ieee64` for full range). See [FP64 policy](fp64-policy.md).

**Compiler** ([details](known-gaps/compiler.md), [why a kernel is refused](ptx-proof-contracts.md))

- Recursive or indirect device-call graphs.
- Tensor-core `mma`/`wmma`/`ldmatrix`, `mbarrier`, TMA/cluster operations, FP8,
  and direct-PTX texture sampling/surface access.
- Inline `asm` containing control flow (`bra`, `call`, `ret`) or `${N:modifier}`
  operands; an empty `asm volatile("" ::: "memory")` is not a Metal barrier.
- Threadgroup float atomics other than add (add is a CAS loop, not hardware).
- Integer/pointer mixing that the proof passes cannot bound: unrelated-base
  pointer subtraction, numeric pointer representations, mutable relocations.
- The `legacy` PTX backend (still the registration-JIT default) lacks general
  CFG forms that the typed `cumetal-ir` backend accepts.
- NVRTC needs `cumetalc` and Xcode's Metal toolchain on disk; `nvrtcGetPTX`,
  LTO-IR, and template name expressions are unavailable.

**Runtime** ([details](known-gaps/runtime.md))

- Cooperative grid sync only up to one resident block per reported GPU core.
- 31 argument buffer slots; kernels using hidden features lose slots 25-30.
- Stream priorities are always zero; device clocks are not cycle counts.
- Child graphs and event/semaphore/conditional graph nodes.
- Device `printf` with a dynamically selected format or a read-only
  module-constant `%s` argument.
- Atomics are form-specific; untested width/scope/ordering combinations may refuse.
- `cudaMemcpy` rewrites 8-byte words that equal a live allocation address
  (to carry device pointers inside structs); such integer data is not preserved
  bit-for-bit on the device.

**Libraries** ([details](known-gaps/libraries.md))

- Datatype/layout/batch combinations outside focused tests; some routines run
  on the CPU through Accelerate and are not Apple-GPU execution.

**Verification and performance** ([details](known-gaps/verification.md))

- External workloads pass only at their pinned revisions and commands.
- The performance gate covers five named kernels, not arbitrary kernels.
