#pragma once

#include "ptx_cfg.h"

#include <functional>
#include <string>

namespace cumetal::ir::detail {

struct LocalStoreRange {
    std::string depot;
    // Inclusive start-address bounds, including the memory operand displacement.
    std::int64_t lower = 0;
    std::int64_t upper = 0;
};

struct LocalMemoryRangeLimits {
    std::size_t states = 16384;
    std::size_t operations = 1'000'000;
    std::size_t known_bytes_per_state = 65536;
    std::size_t register_facts_per_state = 8192;
    std::size_t retained_facts = 1'048'576;
};

struct LocalMemoryRangeProof {
    bool complete = false;
    std::string reason;
    // Missing entries are unknown, never evidence that the store is absent.
    std::unordered_map<const Instruction*, LocalStoreRange> stores;
    std::size_t states = 0;
    // Shared work count: CFG/liveness analysis, fact pruning and prefix steps.
    std::size_t operations = 0;
};

// Explores all feasible prefixes before a nonrepeating load. This is a bounded
// static proof, not an interpreter used to execute the program. Unknown inputs
// fork control flow; unsupported results lose facts. Incomplete exploration
// discards every observed address. Store ranges only discharge non-overlap;
// they do not prove pointer contents or supply missing program values.
// Scalar add/sub and comparisons use explicit 32/64-bit operand extraction;
// overflow is unknown. Narrow result facts never imply unknown upper bits are
// zero. Local addresses stay symbolic and within their named allocation.
// Enqueued states omit only register facts dead on every remaining prefix;
// conditional definitions never kill old facts and memory bytes are retained.
class LocalMemoryRanges {
  public:
    LocalMemoryRanges(const std::vector<RawBlock>& blocks,
                      const std::unordered_map<std::string, std::uint64_t>& depot_sizes,
                      std::function<bool(const Instruction&)> preserves_caller_memory = {},
                      LocalMemoryRangeLimits limits = {});
    LocalMemoryRangeProof prove_before(const Instruction* target_load) const;

  private:
    const std::vector<RawBlock>& blocks_;
    const std::unordered_map<std::string, std::uint64_t>& depot_sizes_;
    std::function<bool(const Instruction&)> preserves_caller_memory_;
    LocalMemoryRangeLimits limits_;
};

} // namespace cumetal::ir::detail
