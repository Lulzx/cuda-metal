# Platform and legal boundaries

[Known-gaps index](../known-gaps.md)

These are durable boundaries unless the canonical specification changes.

- macOS 14+ on Apple Silicon only.
- No Windows, Linux ARM, non-Apple discrete GPU, or Thunderbolt eGPU target.
- One Apple GPU device; no multi-GPU or peer-to-peer execution.
- No OpenGL, Vulkan, or DirectX interop. The `cudaGraphicsGL*` and legacy
  `cudaGL*BufferObject` entry points exist so programs with a headless path
  compile and link; every one returns `cudaErrorNotSupported`.
- No SASS execution, decompilation, or SASS-only binary support.
- No private Apple APIs.
- CUDA-facing headers remain clean-room; NVIDIA headers are not shipped.

The optional `libcuda.dylib` alias is a bounded PTX-bearing compatibility path,
not the project architecture. Closed-source drop-in use has legal and technical
risk distinct from source recompilation. See [the legal notice](../legal-notice.md).

Apple's public Metal compiler rejects native `double` arithmetic on the tested
targets. CuMetal software modes are explicit alternatives, not evidence of
native Metal FP64.

FP32 subnormals depend on the GPU, not on CuMetal's flags. CUDA's default
(`-ftz=false`) keeps single-precision subnormal operands and results. On an
Apple M4 Pro (macOS 27.2) FP32 arithmetic flushes them to signed zero in
every Metal math mode: `-fno-fast-math`, `-ffast-math`,
`-fmetal-math-mode=safe`, and with the `air.compile.denorms_disable` module
flag removed from the AIR. For example, `0x00c00000 * 0.5f` and
`0x00000003 * 0x1p24f` both give `+0`. The Apple Paravirtual GPU on hosted CI
flushes the same way. Cuport-Matrix reports that an Apple M6 keeps FP32
subnormals with the same CuMetal build, so newer GPUs follow CUDA. Loads,
stores, `__float_as_uint`/`__uint_as_float` and the software half/bfloat16
conversions keep subnormal bit patterns on every GPU. Only arithmetic
flushes. Emulating IEEE subnormals on every FP32 operation would slow down
all float code, so CuMetal does not do it. In the other direction, NVRTC's `-ftz`
is accepted as a no-op: only `--use_fast_math`'s module-wide fast-math asks
Metal to flush. `--fp64=fast48` doubles have binary32 range and lose
double subnormals on every GPU (see [FP64 policy](../fp64-policy.md)).
