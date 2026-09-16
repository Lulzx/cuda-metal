#pragma once

#include "cumetal/ir/ir.h"

#include <string>
#include <cstddef>
#include <string_view>
#include <vector>

namespace cumetal::ir {

struct PtxImportOptions {
    bool strict = true;
    std::string entry_name;
    std::string source_name;
    std::string fp64_mode = "fast48";
    // Optional tighter per-function SSA type-proof work limit. Zero retains
    // the size-derived default. Exhaustion rejects the import; it never
    // materializes a partially solved graph.
    std::size_t type_solver_step_limit = 0;
};

struct PtxImportResult {
    bool ok = false;
    Module module;
    std::vector<std::string> warnings;
    std::vector<std::string> printf_formats;
    std::string error;
};

[[nodiscard]] PtxImportResult import_ptx(std::string_view ptx,
                                         const PtxImportOptions& options = {});

}  // namespace cumetal::ir
