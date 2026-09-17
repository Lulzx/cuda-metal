#pragma once

#include "ptx_cfg.h"
#include "ptx_module.h"

#include <map>

namespace cumetal::ir::detail {

struct LocalZeroLimits {
    std::size_t work = 1'000'000;
    std::size_t states = 16384;
    std::size_t depth = 128;
};

struct LocalZeroResult {
    ScalarZeroLoads loads;
    bool budget_exhausted = false;
    std::size_t work = 0;
};

// Prove scalar zero bits before pointer demand is collected. Each accepted
// lane is a full 64-bit cell in a named local allocation, initialized to zero
// on every reaching path. Exact SSA copies/joins and constant byte offsets
// establish cell identity; unknown/partial writes and unresolved calls stop
// the proof. No pointer types or replacement program values are produced.
// Exhaustion discards all facts. The input SSA and instructions are unchanged.
LocalZeroResult prove_local_zero_loads(
    const std::vector<RawBlock>& blocks,
    const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
    const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
    const std::vector<std::map<std::string, ValueId>>& arguments,
    const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
    const std::unordered_map<std::string, LocalDepot>& depots,
    const cumetal::ptx::EntryFunction& function, const Module& module,
    LocalZeroLimits limits = {});

} // namespace cumetal::ir::detail
