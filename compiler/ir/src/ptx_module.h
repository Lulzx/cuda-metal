#pragma once

#include "cumetal/ir/ir.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace cumetal::ir::detail {

// Module declarations are decoded independently of register SSA construction.
struct LocalDepot {
    std::string name;
    std::uint64_t byte_size = 0;
    std::uint32_t alignment = 1;
};

struct ModuleConstantSymbol {
    std::string name;
    std::uint64_t offset = 0;
    std::uint64_t byte_size = 0;
    std::uint32_t alignment = 1;
};

struct InitializedByteArray {
    std::string name;
    std::vector<std::uint8_t> bytes;
    std::uint32_t alignment = 1;
    bool constant_space = false;
    bool module_private = false;
};

struct InitializedByteArrayScan {
    std::vector<InitializedByteArray> arrays;
    std::string error;
};

std::vector<GlobalThreadgroup> scan_threadgroup_globals(std::string_view ptx);
std::vector<LocalDepot> scan_local_depots(std::string_view ptx);
std::unordered_set<std::string> scan_implicit_definitions(std::string_view ptx);
InitializedByteArrayScan scan_initialized_byte_arrays(std::string_view ptx);
std::vector<ModuleConstantSymbol> scan_module_constant_symbols(std::string_view ptx);
std::vector<ModuleConstantSymbol> scan_module_global_symbols(std::string_view ptx);

}  // namespace cumetal::ir::detail
