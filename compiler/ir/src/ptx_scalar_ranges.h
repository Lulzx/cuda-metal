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
                 const std::unordered_map<ValueId, Type>& types);
    ~ScalarRanges();
    std::optional<ScalarRange> get(ValueId value, const Instruction* at);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cumetal::ir::detail
