# Compiler and toolchain gaps

[Known-gaps index](../known-gaps.md) · [Compiler status](../status/compiler.md)

## Typed CuMetal IR migration

With CUDA Clang 21-23, the reviewed production-metallib matrix is:

| Frontend | Legacy | Typed CuMetal IR |
| --- | ---: | ---: |
| direct `.cu` | 0/37 | **37/37** |
| PTX / `--cuda-device` | **33/37** | **34/37** |

The manifest is `tests/cuda_projects/backend_matrix_manifest.txt`; the CTest
gate is `conformance_compiler_backend_matrix`. Counts are compilation evidence,
not runtime correctness. `conformance_compiler_backend_matrix_versions` records
and checks the CUDA Clang 21, 22, and 23 identities separately.

Remaining typed-path blockers include combinations of:

- CFG structurization for residual irreducible or non-reconvergent
  barrier-containing regions beyond the proven uniform multi-exit helper;
- compound shared-memory layouts beyond the proven static arrays and single
  runtime-sized `extern __shared__` binding;
- generic pointer provenance: the legalizer is total for well-formed CUDA
  (unreachable helpers are pruned, a pointer with no in-module producer
  defaults to device memory, helpers are cloned per address space, and a
  reference-returning helper's result takes each call site's space), with two
  named residuals: a helper whose pointer return merges non-argument sources
  of different address spaces, and a helper called with mixed-space arguments
  in *different* spaces at one call site (`f(local, device_elem)`), which the
  single-space clone cannot serve and is refused with `polymorphic helper call
  has multiple concrete address spaces`;
- atomic scope/order/address-space combinations beyond the numerically proven
  32-bit direct/PTX family, lock-backed 64-bit typed-PTX family, and the
  32-bit float add/sub/exchange family (native `atomic_float` in device
  storage, a bit-pattern CAS loop in threadgroup storage; float min/max and
  CAS remain diagnostics);
- PTX call forms beyond the proven FP32 libdevice, constant-format `vprintf`,
  direct scalar-return/pointer-argument helpers, and flat 12-byte by-value/single
  aggregate-return ABI. The generic registration-JIT path additionally proves
  the bounded `cuda-samples/newdelete` virtual dispatch with one aligned
  16-byte by-value argument; nested/irregular aggregates, multi-result
  signatures, general indirect calls, and general double-signature calls remain
  open. A `__device__` function passed by value as a callback -- the classic
  reduction-operator spelling, and the one NVIDIA Warp's `bvh.cu` uses for
  `cub::BlockReduce::Reduce` -- decays to a function pointer and lands in that
  gap: the native-AOT path rejects it with `indirect device calls are
  unsupported` unless the optimizer devirtualizes it first, and `cumetalc` has
  no optimization level to ask for (`--cuda-inline-threshold` maps to
  `-fgpu-inline-threshold`, which Clang ignores without one). The same source
  compiles through the `nvcc` shim, which is why Warp builds. Passing a functor
  instead of a function avoids it entirely;
- aggregate insertion/extraction beyond the bounded NVVM reconstruction limit
  (depth 8, width 16, and 64 scalar leaves), plus irregularly padded nested
  device-call ABIs beyond the proven depth-two 12-byte fixture;
- initialized writable PTX `.global` forms beyond the proven visible numeric
  byte-array and translation-unit-private integer-scalar paths. Both frontends
  embed every referenced initialised read-only global as a `constant` byte
  array regardless of its LLVM address space or Clang/LLVM name (`__const_$`,
  `__const.<fn>.<var>`, `constinit`, `.str`), writable translation-unit-private
  globals use the hidden-buffer ABI, and an undefined `extern __device__`
  global is a compile-time error naming the symbol; unsupported initializer
  types still fail explicitly;
- pointer-returning device helpers on the typed PTX path: the PTX importer
  types a `.param .b64` return as an integer, so `int& at(const Arr&, int)`
  compiles on the direct `.cu` path (see `tests/cuda_projects/descriptor_copy`
  and `ref_return`, enrolled for the native corpus only) but not yet through
  PTX;
- FP64 modes and operations beyond the numerically proven `fast48`
  arithmetic/storage/comparison/rounding corpus, including observable IEEE
  exception status;
- native-AOT symbol combinations beyond the constant/writable multi-kernel
  paths covered by typed NVVM and linked-executable tests.

The old direct legacy `.cu` path is textual qualifier stripping and fails this
corpus. It is not a correctness fallback.

The exact 27-project in-tree numerical corpus passes both typed PTX and direct
native AOT on Apple M4 Pro with workload specializations disabled. This closes
the reviewed corpus, not the residual combinations listed above.

### FP64 libdevice math

Double-precision libdevice calls without a software-ALU primitive --
`exp`/`exp2`/`exp10`/`expm1`, `log`/`log2`/`log10`/`log1p`, the
`sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`atan2`, hyperbolic, and inverse
hyperbolic families, `pow`/`powi`, `fmod`, `fdim`, `hypot`, `cbrt`, `rcbrt`,
`erf`/`erfc`, `ldexp`/`scalbn`, `nextafter`, `sinpi`/`cospi`/`sincospi`,
`logb`/`ilogb`, `llrint`/`llround`, `remquo`, `norm3d`/`norm4d`,
`rhypot`/`rnorm3d`/`rnorm4d`, `erfcx`, `normcdf`/`normcdfinv`,
`erfinv`/`erfcinv`, and `tgamma`/`lgamma` -- evaluate through binary32:
each binary64 storage word decodes to `float`, the binary32 builtin or
its expansion runs, and the result re-encodes into binary64 storage.
This is not CUDA's binary64 transcendental semantics: precision is binary32's
(roughly 24 significant bits), and range is binary32's -- `exp`, for example,
overflows to `inf` near 88 rather than ~709. On the typed source/PTX-to-MSL
paths modules carry the `FP64 libdevice calls evaluate through binary32 under
emulation` semantic caveat in the generated source and report
`semantic_emulation` quality; the registration-JIT PTX-to-LLVM path applies
the same decode/evaluate/re-encode expansion in AIR text. The double
pointer-out builtins `sincos`, `modf`, `frexp`, and `sincospi` follow the
same fallback on all paths: out-params keep their ABI (`int` exponent for
`frexp`, binary64 storage elsewhere), and `modf`'s integral part is
re-encoded from the binary32 `trunc` result before the store. `remquo`'s
double form writes the low 7 quotient bits sign-adjusted (CUDA's contract is
the low 3), matching the platform `remquo` reference. A full
software binary64 transcendental library is intentionally deferred.

Helpers absent from MSL -- `logb`, `erfcx`, `normcdf`/`normcdfinv`,
`erfinv`/`erfcinv`, `tgamma`/`lgamma`, `llrint`/`llround`, the `norm`/`rnorm`
families, and the shared round-to-nearest-even primitive behind `remquo` --
live in `compiler/metal/support/cumetal_libdevice_support.metal`, an
`extern "C"` module textually included into typed MSL and linked into AIR
for the JIT path. The expansions are binary32-quality approximations
(Acklam inverse-normal, Lanczos gamma, Abramowitz-Stegun `erfc`), not
libdevice's own implementations.

Two libdevice families are exact rather than approximate. The
`__nv_{f,d}{add,sub,mul,div,rcp,sqrt}_{rd,ru,rz}` and `__nv_{fmaf,fma}_{rd,ru,rz}`
interval intrinsics route through the correctly-rounded vf64 software ALU in
every FP64 mode: binary64 operands run `vf64_*_round` directly; binary32
operands widen to binary64 exactly, run the vf64 op, and convert back with
the requested rounding -- exact for `add`/`sub`/`mul`, within one binary32
ulp for `div`/`rcp`/`sqrt`/`fma` where the correctly-rounded binary64
intermediate can double-round. Separately, the `_rn` spellings
(`__nv_dadd_rn`, `__nv_fadd_rn`, `__nv_fmaf_rn`, `__nv_drcp_rn`, and friends)
map onto the same IR opcodes as `add.rn.f64` and `fma.rn.f32`, so they run
in the active FP64 mode's own arithmetic, and the
`__nv_{int,ll,uint,ull}2double_*`/`__nv_double2{float,int,uint,ll,ull}_*`
conversion intrinsics plus `__nv_hiloint2double`/`__nv_double2hi/loint` are
exact through the software-FP64 conversion helpers.

## Source AOT architecture

The linked source flow uses native ABI version 3 with an embedded metallib and
no first-launch PTX lowering. Its descriptor carries per-kernel constant and
writable-global bindings plus the device-`printf` format table; focused tests
cover host symbol copies, constant offsets, persistent GPU writes across
launches, and exact 32-lane formatted output. ABI versions other than 3 are
rejected explicitly.

## PTX and fatbinary coverage

PTX support is per instruction form. Direct PTX indirect-object
`txq`/`suq` width, height, and depth queries are numerically tested; remaining
texture/surface forms, TMA/cluster operations, FP8, unrestricted device calls,
and other unsupported forms fail. The binary parser covers bounded raw PTX, CuMetal envelopes, common
fatbin PTX wrappers, version-`0x0101` LZ4/Zstd-compressed PTX entries, and
checked little-endian ELF32/ELF64 sections. Plausible framed entries with an
unknown kind cannot fall through to the legacy raw-PTX scanner or the
registration environment fallback. Other entry versions, codecs, and remaining
container variants are open; SASS-only and big-endian inputs are outside the
current target.

## Inline PTX

Inline `asm` in CUDA source is lowered by the same PTX instruction importer
the PTX frontend uses: operands are bound to synthetic registers from the
constraint string (`r`, `h`, `c`, `l`, `f`, `d`, `b`, immediates, tied `+r`
operands, multiple outputs) and the template's instructions are lowered in
place, so any instruction the typed PTX path supports works inside `asm`.
Refused with a diagnostic: control flow inside the template (`bra`, `call`,
`ret`), `${N:modifier}` operand modifiers, and instructions the PTX path does
not lower (tensor-core `mma`/`wmma`/`ldmatrix`, `mbarrier`, TMA, texture
instructions). A template with no instruction (`asm volatile("" ::: "memory")`)
lowers to nothing; it does not act as a compiler barrier for Metal. Predicated
instructions inside a block follow the PTX path's predication rules.

## Threadgroup float atomics

Metal has no threadgroup float atomic in any language version. CuMetal expands a
32-bit threadgroup float add into a compare-and-swap loop over the word's bit
pattern -- `cm_atomic_fadd_threadgroup` on the MSL path,
`air.atomic.local.cmpxchg.weak.i32` on the LLVM path -- which is correct but
serializes contending threads rather than using hardware. Other threadgroup
float operations (min, max, exchange, compare-and-swap) are refused with a
diagnostic. Device float add, subtract and exchange use Metal's native
`atomic_float`.

## Runtime compilation (NVRTC)

`nvrtcCompileProgram` compiles by spawning `cumetalc`, so runtime compilation
needs the compiler binary on disk and Xcode's Metal toolchain; a caller that
ships only `libcumetal.dylib` gets `NVRTC_ERROR_BUILTIN_OPERATION_FAILURE` with
the reason in the program log. `--device-as-default-execution-space` makes
unannotated functions `__host__ __device__` (Clang has no device-only
default), so a function that also has an explicit `__host__` declaration in the
SDK headers keeps that declaration; programs still see the macOS SDK headers
rather than NVRTC's freestanding set. `--ftz`, `--prec-div` and `--prec-sqrt`
have no Metal knob and are accepted as no-ops. Output is a metallib: `nvrtcGetPTX` and
`nvrtcGetLTOIR` fail, and a `compute_XX` architecture request is rejected at
compile time rather than served with bytes the caller would mis-handle. There is
no `-dlto` path. `nvrtcGetLoweredName` answers only for `extern "C"` entry
points, whose lowered name is the expression itself; template and namespace
expressions return `NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID` because the shim does
not recover the device compiler's mangling. `nvPTXCompiler` is a pass-through,
not a PTX compiler: it returns its input, which the module loader then compiles.

## AIR and Apple tools

Production output depends on Apple's public Metal compiler. `air_inspect`,
`air_validate`, and direct AIR container generation do not constitute a stable
private AIR compiler. Cross-Xcode evidence is incomplete without genuinely
distinct installations and runtime-load results.
