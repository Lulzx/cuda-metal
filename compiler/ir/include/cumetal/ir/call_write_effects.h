#pragma once

#include "cumetal/ir/ir.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ir::detail {

struct CallWriteEffect {
    std::size_t argument = 0;
    std::int64_t offset = 0;
    std::uint64_t bytes = 0;
    bool operator==(const CallWriteEffect&) const = default;
};

struct CallEffectLimits {
    std::size_t work = 262144;
    std::size_t writes = 1024;
    std::size_t depth = 64;
};

struct CallEffectSummary {
    bool complete = false;
    bool budget_exhausted = false;
    std::string reason;
    std::size_t work = 0;
    std::vector<CallWriteEffect> writes;
};

// Analyze verified imported IR without modifying it. Effects include every
// possible write on every path, relative to the callee's formal pointer
// arguments. Writes confined to a callee-owned allocation are omitted. Thus a
// complete empty summary means the call preserves caller-owned memory.
//
// Direct nested calls are instantiated using their actual SSA operands. Only
// exact scalar pointer copies/casts, constant byte offsets and agreeing joins
// establish formal-relative addresses. Unknown/recursive calls, unsupported
// effects, ambiguous addresses and exhausted budgets return an incomplete
// summary with no partial writes. Callers must bind each formal effect to the
// actual argument at the particular PTX call site before checking disjointness.
// Returned summaries can be memoized for an immutable imported-module snapshot.
CallEffectSummary summarize_call_effects(const Module& module, std::string_view callee,
                                         CallEffectLimits limits = {});

// Only recognized builtin calls with entirely scalar operands/results qualify.
bool is_read_only_scalar_builtin(const Operation& operation);

} // namespace cumetal::ir::detail
