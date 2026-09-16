#pragma once
#include "ptx_cfg.h"

namespace cumetal::ir::detail {

// Pre-SSA: discard only a proven unobserved lane of a single-use, same-block
// pack/extract pair. Named lanes require zero source occurrences in the function.
// Storage owns replacement instructions until import completes.
void remove_discarded_pack_halves(std::vector<RawBlock>& blocks,
                                  std::deque<Instruction>& storage,
                                  InstructionOrigins* origins = nullptr);

}  // namespace cumetal::ir::detail
