#pragma once

#include "ptx_cfg.h"

#include <functional>
#include <map>
#include <optional>
#include <string>

namespace cumetal::ir::detail {

struct AddressDemandLimits {
    std::size_t work = 1'000'000;
};

struct AddressDemandLoad {
    const Instruction* instruction = nullptr;
    // The address operand's reaching SSA value before the load writes its
    // destination. A direct allocation symbol has no register value here.
    std::optional<ValueId> address;
};

struct AddressDemandResult {
    bool complete = false;
    bool budget_exhausted = false;
    std::string reason;
    std::size_t work = 0;
    std::size_t indexed_joins = 0;
    std::size_t expanded_joins = 0;
    std::size_t join_edges = 0;
    std::size_t copy_edges = 0;
    std::unordered_set<ValueId> values;
    std::vector<AddressDemandLoad> loads;
};

// Recover demand from real memory-address operands through exact scalar
// 64-bit copies/cvta and SSA joins. The graph is already reachable and has
// validated reaching definitions. Joins are indexed once; only demanded joins
// expand their incoming edges, including every predecessor and backedge.
//
// Demand is not proof of pointer contents or address space. The caller must
// validate any local load selected by this result against its reaching stores,
// and must also validate independently pointer-typed loads in `loads`.
// `ssa_destinations` must match the callback used to build `results`, including
// importer-specific call-return slots. Inputs are never changed. On exhaustion
// or inconsistent SSA, no partial values or load candidates are returned.
AddressDemandResult
compute_address_demands(const std::vector<RawBlock>& blocks,
                        const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
                        const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
                        const std::vector<std::map<std::string, ValueId>>& arguments,
                        const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
                        const std::function<std::vector<std::string>(const Instruction&)>& ssa_destinations,
                        AddressDemandLimits limits = {});

} // namespace cumetal::ir::detail
