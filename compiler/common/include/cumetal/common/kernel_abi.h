#pragma once

#include <cstddef>

namespace cumetal::abi {

// Trap-capable MSL kernels declare this device atomic_uint parameter. The
// runtime recognizes both its binding and reflected name, allocates a zeroed
// 32-bit word per launch, and keeps it alive until command-buffer completion.
// Nonzero means a trap occurred. Every trap sets kTrapStatusLaunchFailure
// (cudaErrorLaunchFailure); a failed device assertion also sets
// kTrapStatusAssert, which the runtime reports as cudaErrorAssert instead.
// Both are sticky on the stream that ran the launch.
inline constexpr std::size_t kTrapStatusBindingIndex = 25;
inline constexpr char kTrapStatusName[] = "cm_trap_status";
inline constexpr unsigned kTrapStatusLaunchFailure = 1u;
inline constexpr unsigned kTrapStatusAssert = 2u;

// Shared-memory spill variants (compiler/metal/src/shared_spill.h): a kernel
// `K` that may need more shared memory than Metal's threadgroup limit also
// ships as `K__cm_spill_s<static bytes>`, whose __shared__ storage is a
// per-threadgroup slice of a device buffer bound at kSharedSpillBindingIndex.
// Word 0 of that buffer is the slice stride; slices start after the header.
inline constexpr char kSharedSpillKernelInfix[] = "__cm_spill_s";
inline constexpr std::size_t kSharedSpillBindingIndex = 24;
inline constexpr std::size_t kSharedSpillHeaderBytes = 16;

// Cooperative grid barrier state, bound at ptx::kGridBarrierBindingIndex:
// word 0 counts arrived threadgroups, word 1 is the release generation, and
// word 2 is the watchdog flag. Metal does not guarantee that every threadgroup
// of a dispatch is co-resident, so a grid sync can wait on a group that never
// runs. A waiting leader gives up after kGridSyncSpinBudget polls of the
// generation, sets the flag and leaves the barrier; other waiters see the flag
// at their next check (every kGridSyncFlagCheckInterval polls) and leave too.
// The runtime reports a set flag as cudaErrorLaunchTimeout. The budget was
// measured on an M4 Pro at roughly 6 seconds of spinning (see
// docs/known-gaps/runtime.md); a legitimate wait that long is not expected.
inline constexpr std::size_t kGridBarrierWords = 3;
inline constexpr std::size_t kGridBarrierTimeoutWord = 2;
inline constexpr unsigned kGridSyncFlagCheckInterval = 1024u;  // power of two
inline constexpr unsigned kGridSyncSpinBudget = 3u << 24;

}  // namespace cumetal::abi
