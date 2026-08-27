#pragma once

#include "metal_backend.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace cumetal::rt {

// Stage one of the runtime-compiled library kernel sources to disk and return
// its path, or nullptr if it could not be written.
//
// The Metal backend compiles MSL from a file, so the sources that are baked into
// this binary have to be materialized once per process. The file is rewritten
// unconditionally rather than reused: a stale copy left by an older build must
// not win over the source this binary was compiled with.
//
// `body` is prepended with kF64DekkerMsl, which carries the `cumetal-math-mode:
// safe` marker the FP64 emulation depends on, so the marker always lands on the
// first line of the composed file where the backend looks for it.
//
// The returned pointer is owned by the cache and lives for the process.
const std::string* stage_library_kernel_source(std::string_view name,
                                               std::string_view body);

// Bind a device pointer as a Metal buffer argument, or fail so the caller can
// take its CPU path.
//
// A device pointer resolves to a buffer plus a byte offset, and Metal requires
// that offset to satisfy the bound type's alignment. An allocation that is too
// small or misaligned is a reason to decline, not to launch: the kernel would
// read past the end or bind a shifted view of the data.
bool resolve_kernel_buffer_arg(const void* ptr,
                               std::size_t required_bytes,
                               std::size_t alignment,
                               cumetal::metal_backend::KernelArg* out);

}  // namespace cumetal::rt
