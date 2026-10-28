#pragma once

#include "cumetal/ir/ir.h"

namespace cumetal::metal::detail {

// Threadgroup (shared) memory optimizations for the typed IR:
//
// 1. float4 coalescing: four adjacent 4-byte threadgroup loads (or stores)
//    at 16-byte-aligned addresses base+0/4/8/12 become one vector
//    load/store of a 4-lane aggregate with per-lane extracts (or a
//    4-lane construct). The static allocation's alignment is raised to 16
//    so the vector access is provably aligned.
//
// 2. Bank-conflict padding: a static threadgroup allocation whose
//    accesses all use 4-byte elements and whose address expressions show
//    a power-of-two row stride (e.g. tile[row][col] with 128-byte rows)
//    is remapped through the injective offset function
//    o -> o + (o / row_bytes) * 4, adding one padding element per row.
//    Warp lanes striding down a column then land in different banks.
//    The remap is a pure function of byte offset, so every access stays
//    consistent; anything unprovable (escapes, atomics, undescribed
//    threadgroup accesses, mixed element sizes) leaves the allocation
//    untouched.
void optimize_threadgroup_memory(ir::Module& module);

} // namespace cumetal::metal::detail
