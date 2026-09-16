#pragma once

#include "ptx_cfg.h"

#include <map>
#include <functional>

namespace cumetal::ir::detail {

// Runs on connected, provisionally typed register SSA, before its types are
// validated or materialized. Rewrites only proven same-root address arithmetic
// into modular integer offsets. A true return requires rebuilding register SSA
// and discarding every provisional type; false leaves the graph unchanged.
bool cancel_same_base_addresses(
    std::vector<RawBlock>& blocks,
    const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
    const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
    const std::vector<std::map<std::string, ValueId>>& arguments,
    const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
    const std::unordered_map<ValueId, Type>& types,
    const std::unordered_map<std::string, Type>& parameters,
    const std::unordered_set<std::string>& symbols,
    const std::function<bool(const std::string&)>& declared_register,
    std::deque<Instruction>& storage,
    InstructionOrigins* origins);

}  // namespace cumetal::ir::detail
