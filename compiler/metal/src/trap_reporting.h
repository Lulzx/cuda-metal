#pragma once

#include "cumetal/ir/ir.h"

#include <string>
#include <string_view>
#include <unordered_set>

namespace cumetal::metal {

// A trapping kernel and every device helper it can call share one hidden
// status pointer. Helpers are emitted as guarded variants so their loops can
// poll cancellation without duplicating their bodies into every call site.
struct TrapCallGraph {
    // Functions that directly or transitively reach a trap.
    std::unordered_set<std::string> trapping;
    // Functions that accept the hidden status pointer.
    std::unordered_set<std::string> guarded;
    // Functions that also need an unguarded variant for nontrapping kernels.
    std::unordered_set<std::string> ordinary;
    // Cyclic guarded functions use dispatcher lowering to guarantee polling.
    std::unordered_set<std::string> dispatch;
};

// Analyze and validate direct call graphs rooted at kernels that can trap.
// Barriers, collectives, printf, indirect calls, and undefined device calls
// remain unsupported because they cannot participate in this cancellation ABI.
bool analyze_trap_call_graph(const ir::Module& module, TrapCallGraph* graph,
                             std::string* error);

std::string guarded_trap_helper_name(std::string_view name);

} // namespace cumetal::metal
