# Runtime architecture

[Specification index](../spec.md)

`libcumetal` maps tested CUDA Runtime and Driver API subsets to Metal while
preserving observable CUDA contracts inside the supported surface.

## Boundary and initialization

- Objective-C++ Metal calls remain in `runtime/metal_backend/`.
- Initialization is thread-safe and selects one supported Apple GPU device.
- Device properties report measured or conservatively derived capabilities;
  unsupported capabilities must not be advertised merely because an API exists.

## Memory

- Every device allocation is tracked with base, size, backing Metal buffer, and
  offset resolution.
- Interior pointers resolve to the correct backing object and byte offset.
- Invalid, stale, overlapping, or out-of-range pointers fail deterministically.
- Unified memory may use shared physical storage, but CUDA copy direction,
  ordering, and lifetime semantics still apply.
- Async allocation/free participates in stream ordering. Pool behavior not
  implemented by the allocator must remain documented as partial.

## Streams and events

- Streams map to ordered command submission and preserve dependencies across
  supported operations.
- Default-stream semantics are explicit and tested; they are not inferred from
  Metal queue behavior.
- Events use GPU-visible ordering where possible and report elapsed time only
  within the validated resolution/ordering contract.
- Callbacks and host functions run only after their preceding stream work and
  must not execute under runtime-internal locks that can deadlock re-entry.

## Launch and ABI

- Launch validates grid/block dimensions, resources, arguments, and function
  identity before dispatch.
- Scalar, aggregate, and pointer arguments preserve documented layout and
  alignment. Pointer arguments bind through allocation tracking.
- Dynamic shared memory and static threadgroup memory must be accounted together.
- A successful API return is not GPU proof; provenance records dispatch source,
  semantic quality, device, and launch success.

## Errors

The runtime preserves per-thread last-error behavior for supported APIs.
Synchronous validation errors and asynchronous GPU completion errors must not be
silently conflated. Peek does not clear; get clears according to CUDA behavior.

## Registration and binary shim

Source registration is distinct from the optional `libcuda.dylib` alias. The
alias is disabled by default in Release builds and accepts only bounded,
validated PTX-bearing forms. Range validation, malformed-container tests, and
cache identity are mandatory. SASS-only and unknown containers fail explicitly.
