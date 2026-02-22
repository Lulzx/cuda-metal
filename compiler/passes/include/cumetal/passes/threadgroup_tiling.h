#pragma once

#include "cumetal/ptx/parser.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cumetal::passes {

// Element type extracted from a shared-memory PTX opcode suffix.
enum class SharedMemElemType {
    kUnknown = 0,
    kB8,   // u8 / s8 / b8  — 1 byte
    kB16,  // u16 / s16 / b16 / f16 — 2 bytes
    kB32,  // u32 / s32 / b32 / f32 — 4 bytes
    kB64,  // u64 / s64 / b64 / f64 — 8 bytes
    kB128, // b128 / v4.f32 — 16 bytes
};

// Represents a single detected shared-memory access site.
struct SharedMemAccess {
    // Opcode of the PTX instruction (e.g. "ld.shared.f32", "st.shared.u32").
    std::string opcode;
    SharedMemElemType elem_type = SharedMemElemType::kUnknown;
    std::size_t elem_bytes = 0;
    bool is_write = false;  // false = read (ld), true = write (st/atom)
    int line = 0;
};

// A detected stride multiplier from a preceding mul/shl instruction:
// e.g. mul.lo.u32 %stride, %idx, 32  →  stride_value = 32
struct StrideHint {
    std::uint32_t stride_value = 0;
    int line = 0;  // line of the mul/shl that produced this hint
};

// A tiling hint for a single shared memory array dimension.
// When `needs_padding` is true, adding `padding_bytes` to the row stride
// of a threadgroup array of this element type eliminates the detected
// bank-conflict pattern.
struct TilingHint {
    std::uint32_t detected_stride = 0;   // power-of-2 stride that triggers conflict
    std::size_t elem_bytes = 0;          // element size in bytes
    std::uint32_t padding_bytes = 0;     // recommended additional bytes per row
    bool needs_padding = false;
    std::string reason;                  // human-readable description
};

struct TilingAnalysis {
    bool ok = false;

    // Whether any shared memory accesses were found.
    bool uses_shared_memory = false;

    // All detected shared-memory access sites.
    std::vector<SharedMemAccess> accesses;

    // Stride values detected immediately preceding shared-mem accesses.
    std::vector<StrideHint> stride_hints;

    // Emitted tiling hints (at most one per element type per conflicting stride).
    std::vector<TilingHint> hints;

    // Informational warnings (non-fatal).
    std::vector<std::string> warnings;
};

// Analyse the instruction stream of a single PTX kernel entry for shared
// memory bank-conflict patterns.  Emits TilingHint entries for every
// power-of-2 stride ≥ 16 that aligns to a multiple of the 32-bank
// threadgroup-memory bank width (4 bytes per bank on Apple Silicon).
//
// The analysis is conservative: it looks for mul.lo / shl instructions
// that produce a constant stride immediately before ld.shared / st.shared
// / atom.shared instructions.  False positives (hints for non-conflicting
// patterns) are possible; false negatives are unlikely for standard 2D
// tiled kernels.
TilingAnalysis analyse_threadgroup_tiling(const cumetal::ptx::EntryFunction& entry);

}  // namespace cumetal::passes
