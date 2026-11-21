#pragma once

#include "cumetal/ir/ir.h"
#include "cumetal/metal/msl_ast.h"

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::metal {

// Builtin callees for CTA-wide barrier reductions (__syncthreads_or/_and/
// _count, PTX bar.red.{or,and}.pred and bar.red.popc.u32). Each is a full
// threadgroup barrier backed by threadgroup scratch.
inline bool is_cta_reduction_callee(std::string_view callee) {
    return callee == "cm_cta_any" || callee == "cm_cta_all" || callee == "cm_cta_count";
}

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
    std::string fp64_mode = "fast48";
};

struct PtxToMslResult {
    bool ok = false;
    ir::Module gpu_ir;
    ir::Module metal_ir;
    MslModule ast;
    std::string source;
    std::vector<std::string> warnings;
    std::vector<std::string> printf_formats;
    bool uses_device_heap = false;
    bool uses_device_launch_queue = false;
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
    std::string_view entry_name = {}, std::string_view fp64_mode = "fast48");

}  // namespace cumetal::metal
