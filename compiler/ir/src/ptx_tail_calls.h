#pragma once

#include "ptx_module.h"

namespace cumetal::ir::detail {

// Match a supported self-call and prove its continuation before changing code.
// Ineligible recursion is left for the normal call-graph rejection.
void normalize_tail_calls(cumetal::ptx::EntryFunction& function,
                          const std::unordered_map<std::string, LocalDepot>& depots);

}  // namespace cumetal::ir::detail
