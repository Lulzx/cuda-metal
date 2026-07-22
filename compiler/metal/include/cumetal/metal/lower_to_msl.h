#pragma once

#include "cumetal/ir/ir.h"
#include "cumetal/metal/msl_ast.h"

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::metal {

struct MetalLegalizeResult {
    bool ok = false;
    ir::Module module;
    std::vector<std::string> warnings;
    std::string error;
};

struct StructurizeResult {
    bool ok = false;
    std::string error;
};

struct LowerToMslResult {
    bool ok = false;
    MslModule ast;
    std::string source;
    std::vector<std::string> warnings;
    std::string error;
};

struct PtxToMslOptions {
    bool strict = true;
    std::string entry_name;
    std::string source_name;
};

struct PtxToMslResult {
    bool ok = false;
    ir::Module gpu_ir;
    ir::Module metal_ir;
    MslModule ast;
    std::string source;
    std::vector<std::string> warnings;
    std::string error;
};

using NvvmToMslResult = PtxToMslResult;

[[nodiscard]] MetalLegalizeResult legalize_for_metal(const ir::Module& module);
[[nodiscard]] StructurizeResult check_structurizable(const ir::Function& function);
[[nodiscard]] LowerToMslResult lower_to_msl(const ir::Module& metal_module);
[[nodiscard]] PtxToMslResult compile_ptx_to_msl(
    std::string_view ptx, const PtxToMslOptions& options = {});
[[nodiscard]] NvvmToMslResult compile_nvvm_to_msl(
    std::string_view llvm_ir, std::string_view source_name = {},
    std::string_view entry_name = {});

}  // namespace cumetal::metal
