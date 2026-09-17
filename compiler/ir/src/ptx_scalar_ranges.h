#pragma once

#include "ptx_cfg.h"
#include <map>
#include <memory>
#include <optional>

namespace cumetal::ir::detail {

struct ScalarRange {
    std::int64_t lower = 0;
    std::int64_t upper = 0;
};

struct ScalarRangeLimits {
    std::size_t work = 4'000'000;
};

// Bounds refer to a particular SSA value at a particular instruction. A bound
// learned from a branch is usable only below its successful edge. Unknown or
// exhausted proofs remain unknown; this analysis never rewrites the program.
class ScalarRanges {
  public:
    ScalarRanges(const std::vector<RawBlock>& blocks,
                 const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
                 const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
                 const std::vector<std::map<std::string, ValueId>>& arguments,
                 const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
                 const std::unordered_map<ValueId, Type>& types, ScalarRangeLimits limits = {});
    ~ScalarRanges();
    bool budget_exhausted() const;
    std::optional<ScalarRange> get(ValueId value, const Instruction* at);
    // A pointer captures its scalar offset at creation. Later guards may
    // refine that offset only while its SSA dependencies retain those values.
    std::optional<ScalarRange> captured(ValueId value, const Instruction* creation,
                                        const Instruction* use);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cumetal::ir::detail
