#pragma once

#include "ptx_cfg.h"
#include "ptx_module.h"

#include <cstddef>
#include <map>

namespace cumetal::ir::detail {

struct AddressAlignmentLimits {
    std::size_t max_values = 1'000'000;
    std::size_t max_edges = 4'000'000;
    std::size_t max_work = 64'000'000;
};

struct AddressAlignmentResult {
    std::size_t rewritten = 0;
    bool budget_exhausted = false;
};

// Prove local-allocation ancestry and low bits on connected provisional SSA.
// Rewrite P | C to P + C only when all set bits of the immediate C are zero
// in P. No numeric pointer values or new pointer roots are synthesized.
// Changes are committed together after convergence. On exhaustion, blocks,
// storage and origins are unchanged. After a rewrite the caller must rebuild
// SSA, type and memory evidence from the normalized instructions.
AddressAlignmentResult legalize_aligned_local_address_ors(
    std::vector<RawBlock>& blocks, const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
    const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
    const std::vector<std::map<std::string, ValueId>>& arguments,
    const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
    const std::unordered_map<ValueId, Type>& provisional_types,
    const std::unordered_map<std::string, LocalDepot>& local_depots, std::deque<Instruction>& storage,
    InstructionOrigins* origins, AddressAlignmentLimits limits = {});

} // namespace cumetal::ir::detail
