# FP64 Policy

CuMetal's handling of double-precision floating-point (`double`, `f64`) on Apple Silicon.

---

## Hardware Reality

Apple Silicon GPUs have highly limited FP64 support. While `fpext`/`fptrunc` type-conversion
instructions (float↔double) are accepted at runtime, double-precision arithmetic operations
(`fmul double`, `fadd double`, `@llvm.fma.f64`) cause Metal compute pipeline-state creation to
fail at runtime on M1–M4 hardware. `xcrun metal` compiles LLVM IR containing `fmul double`
without error, but the GPU driver rejects such pipelines when they are instantiated. This
means `--fp64=native` is effectively broken for arithmetic on current hardware.

As a result, **`--fp64=emulate` is the runtime default** (see `CUMETAL_FP64_MODE` env var
below). Emulation via Dekker's FP32-pair algorithm avoids double-precision ALU ops entirely.

This is not a CuMetal limitation; it reflects the hardware. NVIDIA A100 GPUs provide
1:2 FP64:FP32 throughput. Apple Silicon GPUs do not have a high-throughput FP64 path.

---

## Compilation Modes

CuMetal provides three FP64 compilation modes, selected via the `--fp64` flag:

| Mode | Flag | Runtime default | Behavior | Precision |
|------|------|-----------------|----------|-----------|
| **Native** | `--fp64=native` | No | Emit AIR FP64 instructions directly; fails at launch on Apple Silicon | IEEE 754 double (64-bit) |
| **Emulate** | `--fp64=emulate` | **Yes** (default) | Decompose to FP32 pairs via Dekker's algorithm | ~44 bits mantissa |
| **Warn** | `--fp64=warn` | No | Same as native + per-instruction compile warning | IEEE 754 double (64-bit) |

Set `CUMETAL_FP64_MODE=native` (or `warn`) to override the runtime default.

---

## Usage Guidance

**`--fp64=native` (default)**

Use this when:
- Your CUDA kernel uses `double` for occasional accumulation or reduction
- FP64 operations are less than ~5% of total GPU instructions
- IEEE 754 double precision is required

The throughput penalty is per FP64 instruction. For kernels where doubles are rare,
the overall impact on wall-clock time is tolerable.

**`--fp64=emulate`**

Use this when:
- ~48-bit significand precision, with binary32's exponent range, is acceptable
- FP64 throughput matters more than full IEEE precision
- Example use cases: some ML training loops, iterative solvers with loose tolerance

Emulation uses Dekker's algorithm to simulate double precision via pairs of `float` values.
This has roughly 4× overhead per FP64 operation vs. native FP32.

**`--fp64=warn`**

Use this when:
- You want to audit FP64 usage in a codebase
- You're unsure how much FP64 is in a kernel
- Debugging unexpected slowness that may be FP64-related

---

## Programs Dominated by FP64

Scientific simulation, CFD, climate modeling, and similar workloads that use `double`
throughout are **out of scope for GPU execution on Apple Silicon**. The hardware simply
does not provide competitive FP64 throughput.

**Recommended alternative**: Apple's AMX (Accelerate Matrix eXtensions) coprocessor
provides full-speed FP64 SIMD on the CPU. Workloads requiring competitive FP64 throughput
on Apple hardware should target AMX via the Accelerate framework or BLAS routines. A
Metal↔AMX bridge is out of scope for CuMetal but trivial to build on top of this runtime.

---

## Precision Notes

| Operation | Native Mode | Emulate Mode |
|-----------|------------|--------------|
| Addition, subtraction | IEEE 754 double | ~44 bits (Dekker) |
| Multiplication | IEEE 754 double | ~44 bits (Dekker) |
| `sqrt.rn.f64` | IEEE 754 double | ~44 bits |
| `fma.rn.f64` | IEEE 754 double | ~44 bits |
| `ex2.approx.f64` | Same as Metal `exp2` (double) | ~44 bits |
| Transcendentals (`sin`, `cos`, `lg2`) | May differ from NVIDIA by ≤1 ULP | ~44 bits |

---

## Known FP64 Gaps

- `atom.global.add.f64` (Ampere atomic): supported via LLVM atomic IR; requires AIR
  backend to emit the correct instruction. Verified on M1/M2.
- `fma.rn.f64`: translated via `llvm.fma.f64`; precision depends on Apple GPU FP64
  hardware (see §13 open question #2 in spec.md).
- No FP64 shuffle primitives: `shfl.sync.idx.f64` would require two 32-bit shuffles.
  Emitted as two `shfl.b32` operations on the high and low halves.
