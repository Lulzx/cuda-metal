#pragma once

#include "cumetal/ir/ir.h"

namespace cumetal::metal::detail {

struct RecordContext {
    std::size_t caller, block, operation, callee;
    std::vector<ir::ValueId> zero_loads;
};

// Specialize only proved memory-derived constants. Unknown or exhausted analyses
// leave the original call intact. No record bytes or pointer values are replaced.
bool specialize_record_contexts(ir::Module& module, const std::vector<RecordContext>& contexts);

} // namespace cumetal::metal::detail
