# Platform and legal boundaries

[Known-gaps index](../known-gaps.md)

These are durable boundaries unless the canonical specification changes.

- macOS 14+ on Apple Silicon only.
- No Windows, Linux ARM, non-Apple discrete GPU, or Thunderbolt eGPU target.
- One Apple GPU device; no multi-GPU or peer-to-peer execution.
- No OpenGL, Vulkan, or DirectX interop.
- No SASS execution, decompilation, or SASS-only binary support.
- No private Apple APIs.
- CUDA-facing headers remain clean-room; NVIDIA headers are not shipped.

The optional `libcuda.dylib` alias is a bounded PTX-bearing compatibility path,
not the project architecture. Closed-source drop-in use has legal and technical
risk distinct from source recompilation. See [the legal notice](../legal-notice.md).

Apple's public Metal compiler rejects native `double` arithmetic on the tested
targets. CuMetal software modes are explicit alternatives, not evidence of
native Metal FP64.
