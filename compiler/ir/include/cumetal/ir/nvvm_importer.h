#pragma once

#include "cumetal/ir/ir.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ir {

struct NvvmImportOptions {
    std::string source_name;
    // When set, import only this kernel and the transitive closure of direct
    // device functions it calls. This keeps source-AOT artifacts scoped to the
    // selected entry without requiring whole-kernel inlining.
    std::string entry_name;
};

struct NvvmImportResult {
    bool ok = false;
    Module module;
    std::vector<std::string> warnings;
    std::string error;
};

[[nodiscard]] bool llvm_frontend_available();
[[nodiscard]] NvvmImportResult import_nvvm_llvm_ir(
    std::string_view llvm_ir, const NvvmImportOptions& options = {});
[[nodiscard]] NvvmImportResult import_nvvm_bitcode_file(
    const std::filesystem::path& input, const NvvmImportOptions& options = {});

}  // namespace cumetal::ir
