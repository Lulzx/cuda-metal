#pragma once

#include "ptx_scalar_ranges.h"

#include <functional>

namespace cumetal::ir::detail {

struct ExactLocalAddress {
    std::string depot;
    std::int64_t offset = 0;
    std::uint64_t allocation_size = 0;
};

struct LocalPointerRange {
    std::string depot;
    std::int64_t lower = 0, upper = 0;
};

struct PointerRangeLimits {
    std::size_t work = 4'000'000;
};

// Bounds for a narrow, canonical same-allocation pointer iterator. These facts
// can discharge non-overlap only; they do not initialize memory or assign a
// pointer type. Each recurrence is checked on actual SSA and every backedge.
class PointerRanges {
  public:
    using AddressQuery = std::function<std::optional<ExactLocalAddress>(ValueId, const Instruction*)>;
    using ScalarQuery = std::function<std::optional<ScalarRange>(ValueId, const Instruction*)>;

    PointerRanges(const std::vector<RawBlock>& blocks,
                  const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
                  const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
                  const std::vector<std::map<std::string, ValueId>>& arguments,
                  const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
                  AddressQuery address, ScalarQuery scalar, PointerRangeLimits limits = {});
    ~PointerRanges();
    std::optional<LocalPointerRange> get(ValueId value, const Instruction* at);
    // Prove a store cannot touch an eight-byte cell. Captured scalar offsets
    // and anchored copy/join origins share the same proof for all consumers.
    bool disjoint(ValueId value, const Instruction* at, std::int64_t displacement,
                  std::uint64_t bytes, const std::string& depot, __int128 cell,
                  ScalarRanges& scalar_ranges);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cumetal::ir::detail
