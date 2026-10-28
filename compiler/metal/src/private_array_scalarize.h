#pragma once

#include "cumetal/ir/ir.h"

namespace cumetal::metal::detail {

// Rewrite small private (thread-local) arrays whose every access uses a
// constant element offset or a dynamic index of the form base + i * elem
// into scalar SSA values threaded through block arguments, with loads as
// select chains and stores as per-element conditional updates. Indexed
// thread-memory arrays keep Apple's shader compiler from holding elements
// in registers; the select chain exposes them as ordinary scalars.
//
// The rewrite is deliberately conservative: any use of the array address
// outside the proven access shapes (escapes, calls, block-argument
// propagation, byte-form dynamic offsets, mixed element types, guarded or
// volatile accesses) leaves the function untouched. Out-of-range dynamic
// indices are undefined behavior in the source program; loads then yield
// an arbitrary element and stores update nothing.
bool scalarize_private_arrays(ir::Function& function);

} // namespace cumetal::metal::detail
