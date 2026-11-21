#pragma once

#include "cumetal/common/kernel_abi.h"
#include "cumetal/ir/ir.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cumetal::metal {

// Shared-memory spill variants.
//
// Metal caps threadgroup memory at the device's maxThreadgroupMemoryLength
// (32 KB on every current Apple GPU); CUDA allows 48 KB without opt-in and
// more with cudaFuncAttributeMaxDynamicSharedMemorySize. For every kernel
// that can need more than Metal's limit -- one that uses dynamic shared
// memory, or whose static shared memory alone exceeds the limit -- the typed
// backend also emits a variant whose `__shared__` storage lives in a device
// buffer, one slice per threadgroup:
//
//   kernel void <kernel>__cm_spill_s<static_bytes>(..., device uchar* cm_shared_spill [[buffer(24)]])
//
// Buffer layout (written by the runtime): word 0 is the slice stride in
// bytes; slices start at byte kSharedSpillHeaderBytes, slice i belongs to
// the threadgroup whose linear position in this dispatch is i. Within a slice
// the static __shared__ arrays come first at the offsets the compiler chose,
// then dynamic shared memory at <static_bytes> (a multiple of 16). The
// runtime picks the variant only when static plus dynamic shared memory
// exceeds the Metal limit, so kernels within the limit keep real threadgroup
// memory. Barriers and threadgroup fences in the variant also order device
// memory, and its shared atomics are device atomics.
//
// A kernel whose code cannot be moved faithfully (address-space queries or
// integer conversions of shared pointers, generic pointers that may be
// shared, simdgroup matrix loads, or a launch argument at binding 24) gets no
// variant; the runtime then refuses an oversized launch.
inline constexpr std::string_view kSharedSpillKernelInfix = abi::kSharedSpillKernelInfix;
inline constexpr std::string_view kSharedSpillFunctionSuffix = "__cm_spill";
inline constexpr std::uint32_t kSharedSpillBindingIndex = abi::kSharedSpillBindingIndex;
inline constexpr std::uint32_t kSharedSpillHeaderBytes = abi::kSharedSpillHeaderBytes;
inline constexpr char kSharedSpillParameterName[] = "cm_shared_spill";
// Module attribute marking a module whose variants were already added.
inline constexpr char kSharedSpillDoneAttribute[] = "cumetal.shared_spill_done";
// Module attribute prefix holding a variant kernel's layout:
// "<global>=<offset>;..." keyed by "cumetal.shared_spill_layout:<kernel>".
inline constexpr char kSharedSpillLayoutAttributePrefix[] = "cumetal.shared_spill_layout:";

// Whether `name` is a function emitted for a spill variant.
bool is_shared_spill_function(std::string_view name);

// Adds spill variants for eligible kernels and marks the module done.
// `threadgroup_limit` is the Metal limit assumed at compile time.
ir::Module add_shared_spill_variants(const ir::Module& module,
                                     std::uint64_t threadgroup_limit = 32768);

// Parses a variant kernel's layout attribute; empty when absent.
std::unordered_map<std::string, std::uint64_t> shared_spill_layout(const ir::Module& module,
                                                                   std::string_view kernel);

} // namespace cumetal::metal
