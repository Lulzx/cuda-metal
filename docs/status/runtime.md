# Runtime and CUDA API status

[Status index](../status.md) · [Known runtime gaps](../known-gaps/runtime.md)

## Core runtime

`libcumetal` implements tested Runtime and Driver API subsets over Metal. It
tracks allocations and interior pointers, resolves them to Metal buffers,
preserves per-thread last-error behavior, and uses command queues/shared events
for stream and event ordering.

Covered families include:

- initialization, device selection/properties, contexts, modules, and functions;
- device/managed/host allocation, synchronous/asynchronous copies and memset;
- stream-ordered allocation/free and a conservative memory-pool subset;
- streams, priorities-as-zero, callbacks/host functions, waits, and events;
- kernel launch through Runtime, Driver, and source registration APIs;
- symbols/constants/globals, occupancy queries, attributes, and error strings;
- 2D pitched allocation/copy and conservative UMA advice/prefetch behavior.

Exact symbols are defined by the clean-room headers and focused API tests. API
presence does not imply every CUDA flag, datatype, or interaction is covered.

## Ordering and provenance

Legacy default-stream ordering is implemented bidirectionally against blocking
streams; nonblocking and per-thread streams remain independent. Cross-queue
resource fencing uses shared-event epochs for tracked buffers. Registered
launches can remain asynchronous while preserving tested alias ordering.

`CUMETAL_TRACE_GPU=1` reports dispatch source, semantic quality, device, and
launch success. CPU or approximate paths must identify themselves.

## Compatibility surfaces

- Source registration remains enabled independently of the binary alias.
- The opt-in binary path accepts bounded raw PTX, CuMetal envelopes, CUDA fatbin
  PTX forms including version-`0x0101` LZ4/Zstd entries, and checked
  little-endian ELF32/ELF64 PTX sections. Decompression is capped at 64 MiB.
- CUDA graph coverage includes tested kernel, linear memcpy/memset, host,
  clone/update, graph-memory node behavior, and event-linked two-stream ordered
  replay with capture-conflict rejection.
- Dynamic launch uses a bounded device record queue and recursive host drain;
  focused tests cover parent-child-grandchild ordering, invalid child
  configurations, and the 1,023-record overflow boundary.
- Texture/surface objects, arrays, memcpy, and source descriptor helpers cover a
  tested subset.
- Cooperative launch supports resident grids up to four blocks. CUDA-visible
  `multiProcessorCount` is 1 because public Metal exposes physical GPU-core
  count but no per-kernel simultaneous-residency query; this makes CUDA's
  occupancy-derived grid formula choose a guaranteed-progress partition while
  explicitly sized launches retain the tested four-block ceiling.
- Warp/cooperative-group coverage includes contiguous and irregular masked
  votes, shuffles, barrier ordering, and binary/labeled partitions.
- Device `printf` uses a bounded runtime buffer and 256-byte format limit;
  tested scalar formatting includes 32/64-bit integers, pointer, promoted
  binary64 floating, character, flags, fixed and dynamic width/precision, and
  `%%` forms. Bounded `%s` formatting materializes tracked allocations and
  registration-backed writable module strings; untracked addresses produce the
  safe `[string]` placeholder instead of being dereferenced. The FIFO size
  round-trips through `cudaDeviceSetLimit`/`cudaDeviceGetLimit`; complete
  retained records drain on overflow, and device calls return their parsed
  argument count independently of record acceptance.

## Installed headers

The install includes clean-room CUDA Runtime/Driver, FP16/BF16, cooperative
groups, CUDA library, NVML/NCCL, NVTX, Thrust, and CUB-facing headers required by
the tested subset. Forwarding headers cover common CUDA include names.
