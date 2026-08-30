# Runtime and CUDA semantic gaps

[Known-gaps index](../known-gaps.md) · [Runtime status](../status/runtime.md)

## Synchronization and launch

- Cooperative-grid synchronization is limited to a conservatively resident
  grid capped at one block per reported GPU core. Oversubscription is rejected.
- Dynamic launch uses a fixed 1 MiB device queue with at most 1,023 child
  records per parent dispatch and a host scheduling drain. Nested parent-child-
  grandchild execution, invalid child configurations, and record overflow have
  focused Apple-GPU tests. Queue growth and hardware-recursive scheduling parity
  remain absent.
- Contiguous half-warp and non-contiguous `0xa5a55a5a` masks have focused
  vote, shuffle, barrier-ordering, binary/labeled partition, and divergent
  coalesced-group tests. Arbitrary mask/topology interactions remain narrower
  than CUDA's complete surface.
- Stream priorities are reported as zero and are not Metal priority queues.
- CUDA device clocks use a device-wide atomic counter with a fixed monotonic
  quantum. They preserve wait-loop progress and unsigned wraparound behavior,
  but values are not GPU cycles and cannot be used for cycle-accurate timing.

## Graphs and allocators

Tested graph capture/replay includes an event-linked two-stream dependency with
ordered numerical replay, stale-event lifetime rejection, and conflicting-
capture rejection. Explicit graph nodes also preserve pointer-backed pitched 3D
copy geometry and pitched 1/2/4-byte memset values; malformed pitch, extent,
element-size, overflow, dependency, and creation-flag inputs are rejected.
Array-backed graph-copy replay is supported by the runtime path but lacks the
same focused numerical graph test. Clone/update and memory-node lifetimes still
do not cover all node types or multi-stream topologies. Virtual/physical
allocation reuse, allocator caching/release-threshold behavior, and advanced
update cases remain incomplete.

## Memory and pointers

- Arbitrary pageable `malloc` pointers are not kernel-bindable merely because
  Apple Silicon uses UMA; tracked Metal-backed allocations are required.
- Managed-memory API compatibility does not imply CUDA concurrent managed access
  or CPU/GPU atomics.
- Persisting-L2/access-policy APIs preserve a conservative validated hint state,
  but public Metal offers no cache-residency control.
- Memory-pool attributes exceed the allocator's current reuse behavior.

## Textures, surfaces, and printf

Texture/surface object lifecycle, arrays, copies, and selected source descriptor
helpers exist. Direct PTX texture/surface instructions, native Metal texture ABI,
and remaining addressing/filtering modes do not. Device `printf` has a bounded
buffer and 256-byte format limit. Focused Clang-ABI tests cover 32/64-bit
signed/unsigned integers, hex flags, `size_t`, pointers, promoted binary64
floating values, characters, fixed precision, and escaped percent signs
on both PTX backends. Dynamic `*` width/precision and bounded `%s` reads from
tracked allocations are also tested on both paths. Registration-backed writable
module strings are materialized on legacy PTX, typed PTX, and native AOT;
arbitrary untracked addresses are rejected safely as `[string]`. Embedded
read-only module-constant strings remain a gap. `cudaLimitPrintfFifoSize`
configures the bounded ring; format-only and multi-argument calls return their
CUDA parsed-argument counts even when capacity rejects a record, and a complete
retained prefix is drained. A statically null format returns CUDA's specified
`-1` without reserving or writing a record on legacy PTX, typed PTX, and native
AOT. Dynamically selected formats remain unsupported. CUDA's circular overwrite
policy and the `-2` internal-error return remain outside the proved subset.

## FP64 and atomics

`fast48` has roughly 48-bit significand precision but binary32 exponent range;
`wide48` extends range handling; `ieee64` is software. Observable IEEE exception
status is not fully integrated. See [FP64 policy](../fp64-policy.md).

Atomic support is form-specific. Wider, floating, system-scope, ordering, and
address-space combinations outside focused tests remain gaps. Successful header
compilation is not atomic contention proof.

## Device properties

Several reported CUDA properties are conservative or synthetic compatibility
values (for example compute capability 8.0, zero PCI identifiers, priority range
0/0). They must not be used as proof that the corresponding NVIDIA hardware
feature exists.
