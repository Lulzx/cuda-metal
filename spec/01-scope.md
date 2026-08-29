# Purpose, principles, and scope

[Specification index](../spec.md)

CuMetal is a CUDA compiler and compatibility runtime targeting Apple's public
Metal stack on Apple Silicon. Its primary value is recompiling source that the
user controls. Binary compatibility is secondary and opt-in.

## Goals

- Compile supported CUDA C++ and PTX into Metal libraries and runnable programs
  without source changes inside the supported subset.
- Run accepted kernels on the Apple GPU and verify numerical results.
- Preserve CUDA-visible ABI and ordering contracts where implemented.
- Prefer correctness and explicit failure over broad acceptance.
- Keep compiler stages inspectable and runtime provenance observable.
- Use only public Apple APIs and clean-room CUDA-facing headers.

## Architecture invariants

- Source recompilation through `cumetalc` is the primary path.
- The canonical compiler representation is typed CuMetal GPU IR, followed by
  Metal legalization, a typed MSL AST, and Apple's supported compiler tools.
- Direct AIR generation is research and validation tooling, not the production
  compiler contract.
- SIMD/warp width is fixed at 32. There is no runtime width mode.
- Metal framework calls stay inside `runtime/metal_backend/`.
- Per-thread CUDA error behavior, allocation tracking, and pointer-to-buffer
  resolution are runtime invariants.
- No fallback, substitution, reduced precision, or workload specialization may
  be selected silently.

## Durable non-goals

- Windows, Linux ARM, non-Apple discrete GPUs, and Thunderbolt eGPU execution.
- SASS execution, decompilation, or translation.
- Multi-GPU and peer-to-peer execution on the single-GPU Apple Silicon target.
- OpenGL, Vulkan, or DirectX interop.
- Full parity with every CUDA toolkit, driver, or NVIDIA library version.
- Private Apple APIs.

## Bounded compatibility areas

The following are permitted only as explicitly tested subsets: CUDA graphs,
dynamic launch, texture/surface behavior, cooperative grids, high-level CUDA
libraries, PTX-bearing fatbinaries, and software FP64. Their exact present
limits belong in `docs/known-gaps.md`; adding a subset does not erase the larger
boundary.

## Unsupported behavior

Unsupported compiler semantics must produce a compile-time diagnostic.
Unsupported runtime or library requests must return a documented error. A CPU
fallback is allowed only behind an explicit user choice and must report CPU
provenance. An approximate GPU path must report its semantic quality.
