#pragma once

#include <cstddef>

namespace cumetal::abi {

// Trap-capable MSL kernels declare this device atomic_uint parameter. The
// runtime recognizes both its binding and reflected name, allocates a zeroed
// 32-bit word per launch, and keeps it alive until command-buffer completion.
// Nonzero means a trap occurred; this is a launch failure, not a context abort.
inline constexpr std::size_t kTrapStatusBindingIndex = 25;
inline constexpr char kTrapStatusName[] = "cm_trap_status";

}  // namespace cumetal::abi
