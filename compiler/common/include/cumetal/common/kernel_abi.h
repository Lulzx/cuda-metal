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

}  // namespace cumetal::abi
