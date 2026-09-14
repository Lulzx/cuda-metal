# Bounded PTX trap reporting

The typed MSL backend can now compile unchanged PTX containing kernel traps.
This is software failure reporting for kernels and supported acyclic device-call
graphs without user barriers, collectives or printf. It is not full CUDA
context-abort behavior. Trapping/looping helpers are expanded into the kernel CFG;
finite trap-free helpers may remain ordinary calls under the proof below.

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

## Device-call cancellation legalization

Before address-space resolution, the compiler finds kernels whose transitive
call graph contains a trap. Direct helper calls that may trap or loop are
expanded into that kernel's CFG. Each call splits its caller block, clones the
callee with fresh block/value IDs, and connects normal returns to a continuation
through typed block arguments. Scalar and aggregate results retain their types.
Local allocations and pointer metadata are cloned; generic pointers resolve
through the new edges. A trap remains a trap, so it cannot reach the continuation
or execute caller stores after the call. Unreachable continuations are removed.
The expanded GPU IR and the resulting Metal IR both pass verification.

This also expands a nontrapping helper that loops: polling only in callers can
leave a sibling lane spinning inside a divergent call before another lane gets
to publish its trap. The regression exercises that exact shape, with two SIMD
groups, nested calls and repeated error checks.

To avoid duplicating large straight-line arithmetic helpers, the compiler retains
helpers only if their CFG is acyclic and every transitive operation is free of
traps, barriers, collectives, printf, unknown calls and atomics. Integer `min`,
`max` and signed `abs` are explicitly recognized finite expressions. Retained
helpers can finish between CFG-boundary cancellation polls. The proof is
recomputed for Metal lowering; a metadata assertion alone does not allow a call.
Other builtins remain rejected in trap-capable paths.

Expansion has explicit per-kernel ceilings: 1,024 expanded calls, 4,096 blocks
and a conservative 262,144-operation estimate. Excessive expansion fails rather
than disabling cancellation. Unnormalized recursion and indirect/undefined calls
remain rejected by IR verification. Standalone trapping helper libraries with
no kernel cancellation binding remain unsupported. User barriers, collectives
and printf anywhere in the expanded trapping path remain rejected.

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
expansion beyond the supported bounds, barrier-aware cancellation and recovery semantics
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
- Nested-helper GPU tests in tracing and non-tracing modes: scalar/multiple
  returns, retained finite helper chains, untaken/all/divergent traps, spinning
  helpers, and store-before-trap / no-store-after-trap ordering.
- 261 runtime inputs check aggregate returns, local pointer side effects,
  repeated call sites and calls inside a loop, with guards.
- Compiler negatives cover transitive user barriers, excessive CFG expansion
  and hidden-name conflicts. Existing scalar/local-tail regressions remain on.
- The call-expansion milestone passes all 31 focused compiler/GPU tests.
- Historical initial trap-reporting milestone:
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
