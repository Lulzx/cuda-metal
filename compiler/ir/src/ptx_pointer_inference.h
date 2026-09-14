#pragma once
#include "cumetal/ir/ir.h"
#include "cumetal/ptx/parser.h"
#include <unordered_map>
#include <unordered_set>

namespace cumetal::ir::detail {

struct PointerInference {
    std::unordered_set<std::string> promoted_global_registers;
};

// Bounded pre-SSA pointer recovery; unsupported/reused register paths remain
// subject to the normal typed importer and SSA checks.
PointerInference infer_entry_pointer_types(const ptx::EntryFunction& entry, const Module& module,
    bool is_kernel, const std::unordered_set<std::string>& pointer_symbols,
    const std::unordered_set<std::string>& promoted_global_symbols,
    std::unordered_map<std::string, Type>& parameter_types);

}  // namespace cumetal::ir::detail
