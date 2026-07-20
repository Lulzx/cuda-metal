#pragma once

#include "cumetal/ir/ir.h"

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ir {

struct PtxImportOptions {
    bool strict = true;
    std::string entry_name;
    std::string source_name;
};

struct PtxImportResult {
    bool ok = false;
    Module module;
    std::vector<std::string> warnings;
    std::string error;
};

[[nodiscard]] PtxImportResult import_ptx(std::string_view ptx,
                                         const PtxImportOptions& options = {});

}  // namespace cumetal::ir
