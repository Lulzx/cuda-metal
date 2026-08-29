# FP64 policy

CuMetal can lower CUDA `double` operations to software arithmetic that runs on
Apple GPUs. The CUDA-visible representation remains the ordinary eight-byte
IEEE-754 binary64 bit pattern; only arithmetic is virtualized.

## Why software FP64 is required

The public Apple GPU toolchain accepts some LLVM/AIR `double` instructions, but
pipelines containing native binary64 arithmetic fail during Metal pipeline
creation on the Apple Silicon generations tested by this project. `native` is
therefore a compiler-diagnostic mode, not a usable GPU arithmetic path on those
devices.

## Modes

`cumetalc --fp64=MODE` selects an offline mode. Runtime-registered CUDA code uses
the same names through `CUMETAL_FP64_MODE=MODE`.

| Mode | Contract | Range | Current role |
| --- | --- | --- | --- |
| `fast48` | FP32-pair arithmetic, approximately 48 significand bits | binary32 exponent envelope | Runtime default; `emulate` is a compatibility alias |
| `wide48` | Scaled FP32-pair arithmetic, approximately 48 significand bits | binary64 range, including subnormals and specials | Reduced-precision wide-range mode |
| `ieee64` | Correctly rounded binary64 core arithmetic and conversions | Full binary64 | Exact software mode |
| `native` | Native AIR binary64 operations | Full binary64 if supported | Expected to fail pipeline creation on current Apple GPUs |
| `warn` | Same code generation as `native`, with FP64 diagnostics | Same as `native` | Audit mode |

Examples:

```bash
cumetalc kernel.ptx --fp64=ieee64 --emit=metallib -o kernel.metallib
CUMETAL_FP64_MODE=wide48 ./cuda-program
```

## Storage and linking ABI

Registers, local spills, shared/global memory, kernel arguments, shuffles, and
integer aliases carry raw binary64 bits. This keeps `mov.b64`, `uint64_t`
type-punning, `cudaMemcpy`, and CPU library boundaries interoperable without a
private packed-pair format.

For `wide48` and `ieee64`, CuMetal statically links the pinned
`third_party/f64-metal` support module into each generated metallib. The
persistent JIT cache key includes the support source SHA-256, so a support
runtime update cannot reuse an older metallib. Linked helper functions are not
treated as kernel entry points by validation.

## Compiler coverage

The generic PTX-to-AIR path currently routes these source-level operations
through the virtual FP64 support ABI:

- add, subtract, multiply, divide, square root, and true fused FMA;
- negation, absolute value, raw moves, loads/stores, shared memory, and shuffles;
- ordered comparisons, equality/inequality, and NaN/number predicates;
- IEEE remainder and round-to-integer operations;
- binary16/binary32 and signed/unsigned 32/64-bit integer conversions.

Round-to-nearest-even and PTX's directed arithmetic modes are preserved in
`ieee64`. `wide48` intentionally accepts round-to-nearest-even arithmetic only.
The support runtime itself also exports classification, exception-status, and
flag operations, but
not all of those operations are wired into CUDA/PTX lowering yet. In
particular, source-level IEEE exception flags are not currently observable
through the CUDA runtime. These are compiler-integration gaps, not claims that
the underlying virtual runtime is absent.

## Provenance and cache behavior

GPU launch records distinguish the numerical contract:

| Mode | Provenance | Semantic quality |
| --- | --- | --- |
| `fast48` / `emulate` | `generic_ptx_lowering_fp64_emulated` | `reduced_precision_fp64` |
| `wide48` | `generic_ptx_lowering_fp64_wide48` | `reduced_precision_fp64` |
| `ieee64` | `generic_ptx_lowering_fp64_ieee64` | `exact` |

The mode and support-runtime hash are part of the persistent registration-JIT
cache identity. A warm-cache launch therefore retains the same provenance and
numerical implementation as a cold launch.

## Verified boundary

On the recorded Apple M4 Pro path, a CUDA source probe compiled and executed in
`ieee64` mode with zero violations across binary64 bit preservation, special
values, float-to-double widening, chained arithmetic, shared-memory reduction,
shuffle reduction, store/reload arithmetic, and `uint64_t` aliasing. This is a
focused integration proof; it is not yet complete CUDA source coverage for all
IEEE-754 operations.

The standalone `f64-metal` repository owns the full virtual runtime's
conformance corpus, mode contracts, benchmarks, and ISA documentation. CuMetal
pins that repository as a submodule so compiler claims can identify the exact
runtime revision they execute.
