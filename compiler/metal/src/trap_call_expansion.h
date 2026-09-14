#pragma once

#include "cumetal/ir/ir.h"
#include <string>
#include <unordered_set>

namespace cumetal::metal {
// These scalar integer intrinsics lower to finite expressions, not device calls.
bool is_bounded_trap_builtin(const ir::Operation& operation);

// Only acyclic CFGs whose transitive operations cannot trap, reach a collective, or
// wait on atomics qualify. Such helpers may finish between cancellation polls.
std::unordered_set<std::string> find_bounded_trap_helpers(const ir::Module& module);

// Expand direct calls in trap-capable kernels into one cancellation CFG.
// Requires verified GPU IR with an acyclic call graph. Failure leaves the module unchanged.
bool expand_trap_call_graphs(ir::Module* module, std::string* error);
}
