# Bounded PTX trap reporting

The typed MSL backend can now compile unchanged PTX containing kernel traps.
This is software failure reporting for call-free kernels without user barriers,
collectives or printf. Traps in helpers and kernels outside this boundary still
fail compilation. It is not full CUDA context-abort behavior.

## Compiler contract

A reporting kernel receives `device atomic_uint* cm_trap_status` at buffer 25.
The compiler checks argument name/binding conflicts. The runtime also checks
each effective binding index, including explicit remapped arguments.

The kernel uses the CFG dispatcher. A block ending in trap executes its ordinary
operations first, then transitions to a unique synthetic trap state. At the next
iteration a SIMD vote publishes pending traps through atomic OR, followed by an
atomic status load. A nonzero status exits active lanes. Publication happens
before divergent switch arms: merely polling a flag after individual lanes
enter their trap arms hung a regression with spinning peers. The SIMD vote
resolves that tested case, including cancellation across two SIMD groups.

Every dispatcher iteration polls, including backedges. This adds overhead to
trap-capable kernels. Cancellation is cooperative at CFG boundaries, not an
immediate hardware abort. Outputs after a reported failure must not be trusted;
other threads may already have produced partial writes. A regression separately
checks that a single thread's store immediately before its trap is preserved.

## Runtime contract

Each normal launch gets a fresh zeroed four-byte status buffer. Such launches
are not batched. The pending command-buffer record retains the buffer until
completion, and tracing completion handlers retain it independently. CPU checks
occur only after GPU completion. Stream synchronization/query and event waits
observe `cudaErrorLaunchFailure` / driver `CUDA_ERROR_LAUNCH_FAILED` (719).
The error is latched on that stream, including repeated checks and checks from
other host threads; consuming a completion and publishing its error are protected
by the same mutex. Unrelated nonblocking streams keep their own results/status.

This is a stream-level failure contract. Context-wide poisoning, rollback,
general helper unwinding, barrier-aware cancellation and recovery semantics
remain future work. The alternate timed backend launch path rejects reporting
pipelines explicitly. Reporting is validated with CuMetal-produced MSL and its
ABI: pipeline reflection recognizes both buffer 25 and the name
`cm_trap_status`. Arbitrary stripped/renamed precompiled library ABI is not
covered. Reflection-name independence needs durable metadata or a broader
reserved-binding policy before extending that claim.

## Validation on Apple M5

- GPU regression in tracing and non-tracing modes: untaken traps, all lanes
  trapping, divergent lanes, spinning peers, and store-before-trap ordering.
- Concurrent independent nonblocking streams; event and stream completion,
  repeated error reports, mapped output values and 16 guard words.
- Backend negative tests for a short argument list explicitly remapped to
  binding 25 and the unsupported timed launch path.
- Compiler negative tests for helper traps, user barriers and hidden-name
  conflicts. Existing scalar/local-tail and arithmetic regressions remain on.
- 26 focused compiler/GPU tests pass. Five additional stream/callback/priority
  tests pass; ten older queue/event GPU tests skip because `xcrun metal` is
  unavailable. No full-suite pass is claimed.

The self-test runner preserves the original synchronization exception when
cleanup APIs report the same sticky failure.

## Base58 result

Pinned full-module inputs are run `34778991430`, x86 LLVM 7 and LLVM 19; PTX bytes
are unchanged. Both compile through CuMetal to MSL after this change.

- LLVM 7 `kernel_self_test_primitive_base58` reaches M5 execution and reports a
  taken trap at synchronization (719). This is not a numerical pass. Investigate
  which translated bounds/panic path is reached and why before changing it.
- LLVM 19 reaches Apple's MSL compiler, which rejects pointer subtraction
  assigned to ulong and pointer casts lacking an explicit address space.
  It does not reach GPU execution.

Logs: `/tmp/trap-base58-llvm7.log`, `/tmp/trap-base58-llvm7.gpu.log`, and the
corresponding `llvm19` files. These failures supersede the old first-blocker
message about missing trap lowering; they do not validate any mining kernel.

Subsequent investigation fixed the LLVM 7 numerical cause: an unsigned
32-to-64-bit immediate was widened without first preserving its 32-bit pattern.
The original base58 self-test now passes with trap reporting enabled; see
[fix 8](miner-fix-backlog.md#fix-8-unsigned-integer-widening-preserves-the-source-width).
The observations above describe the initial trap-reporting milestone.
