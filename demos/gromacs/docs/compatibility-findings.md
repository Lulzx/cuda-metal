# Compatibility findings

[Back to the GROMACS documentation index](../README.md)

The correctness gate exposed every defect below. Each produced a wrong answer
or hard failure without a warning.

## Runtime and stream semantics

### `cudaDeviceReset` erased the kernel registry

CuMetal called `registration::clear()`, dropping the tables built by
`__cudaRegisterFatBinary` when the image loaded. Those tables are not device
context state: CUDA keeps them across a reset and reloads modules on the next
launch. GROMACS resets the device after detection, so all later launches
failed. CuMetal now releases only the Metal buffers behind `__device__`
globals.

### Host-backed CUB algorithms ignored stream order

Every `cub::Device*` shim was a host loop over unified memory and ignored its
`cudaStream_t`. GROMACS fills `sciHistogram` in the prune kernel and immediately
exclusive-scans it. The scan could read stale memory while the kernel was in
flight, causing the sorted pair list to drop about a quarter of the
interactions. Step 0 was correct because it uses the unsorted list; from step 1
the potential was off by 4%. All 33 entry points now synchronize.

### `cudaDestroyTextureObject(0)` returned an error

Destroying a null texture object is a no-op, like freeing a null pointer.
GROMACS's PME teardown does this for an unpopulated lookup table; CuMetal's
`InternalError` aborted the process.

### A zero-parameter kernel could not launch

The registered-launch path rejected `args == nullptr` unconditionally. CUDA
allows null arguments for a kernel with no parameters, which is exactly
GROMACS's first device sanity check:

```cuda
static __global__ void dummy_kernel() {}
```

## Compiler and libdevice lowering

### Missing libdevice entry points

`__nv_rsqrt` (double reciprocal square root) is used by every nbnxm kernel, and
the `__nv_float2int_rn` conversion family is used by bonded kernels for the PBC
image index. A missing entry point made the whole kernel unlowerable.

`__nv_rsqrt` now shares all three FP64 paths with `__nv_sqrt` and composes as
`1 / sqrt(x)` in the selected arithmetic. The conversions are table-driven
across all three lowering paths. The rounding mode is applied before the cast,
and signed variants cast through a signed integer; casting directly to the
unsigned IR result type had turned negative results into zero.
`tests/cuda_projects/libdevice` probes `rsqrt` and all four `__float2int_*`
modes on inputs straddling zero.

### By-value struct fields aliased onto one word

A by-value struct arrives as `.param .align 4 .b8 name[24]`, with fields read
by `ld.param.<type> [name+offset]`. The PTX-to-MSL path declared the object as a
single `constant uint&`, discarded the parsed offset, and made all six members
of GROMACS's barostat `ScalingMatrix` read the same four bytes. A float field
also read through a `uint` declaration as its bit pattern, so `1.001f` became
roughly `1.07e9`. Coordinates left the box at the first pressure-coupling step
and the potential became NaN.

Fields are now bound as words and read at their own offsets. Because optimized
NVPTX keeps floats in `.b32` registers—`ld.param.b32` feeding an
`fma.rn.f32`—the field type is inferred from uses through the move chain rather
than from the load suffix. Struct access outside the model is declined instead
of guessed.

## cuFFT and GPU FFT

The original cuFFT shim was rank-1 only, so `cufftPlanMany(rank = 3)` returned
`CUFFT_NOT_SUPPORTED` and prevented `-pme gpu`. It now supports ranks 1 through
3 for all transform types and cuFFT advanced data layouts. GROMACS's real-space
mesh is padded on its fastest axis, so `inembed` and `onembed` are required.

Multidimensional transforms run as separable sequences of one-dimensional
transforms. Treating the product of the dimensions as a single flattened
transform computes a different function. Axis lengths that vDSP cannot
factor—including a PME grid with factor 7—use Bluestein's chirp-z algorithm,
which also removes the old direct-sum fallback's 1,024-element ceiling.

The production dense 3-D R2C/C2R path now uses vendored VkFFT 1.3.4, with eager
plan creation outside the timed trajectory and CuMetal stream/resource ordering
around each transform. The project-owned Stockham/Bluestein implementation
remains a tested fallback, not a silent CPU excursion.

## Build and CUDA compatibility edges

The `nvcc` shim gained `-ccbin`, `-diag-suppress`, `-Xcompiler` quote stripping,
a two-phase compile and link—clang's CUDA-mode link mis-parses Apple's
`-lto_library`—`CUDA::cudart_static`, and directory symlinks for `cub/` and
`nvtx3/`. These are build plumbing rather than device semantics.

The native-Metal comparison exposed three additional edges:

- GROMACS compiles some `.cpp` files as CUDA-language translation units, so the
  generated `nvcc` shim enables CUDA mode for every recognized source suffix,
  not only `.cu`.
- `cudaDeviceProp::uuid` now exposes the standard 16-byte identity matching the
  deterministic Driver API value without growing the reserved ABI envelope.
- CUB's free `ShuffleIndex<32>` now shuffles every 32-bit word of trivially
  copyable `float3` and `float4` aggregates from the same lane, with numerical
  Apple-GPU tests for both vector shapes.
