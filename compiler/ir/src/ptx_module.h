#pragma once

#include "cumetal/ir/ir.h"
#include "cumetal/ptx/parser.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cumetal::ir::detail {

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
    // A complete little-endian address relocation, not placeholder zero bytes.
    std::string pointer_target;
};

struct InitializedByteArrayScan {
    std::vector<InitializedByteArray> arrays;
    std::unordered_map<std::string, std::unordered_set<std::string>> dependencies;
    bool address_size_64 = false;
    std::string error;
};

std::vector<GlobalThreadgroup> scan_threadgroup_globals(std::string_view ptx);
std::vector<LocalDepot> scan_local_depots(std::string_view ptx);
std::unordered_set<std::string> scan_implicit_definitions(std::string_view ptx);
std::vector<ModuleConstantSymbol> scan_module_constant_symbols(std::string_view ptx);
std::vector<ModuleConstantSymbol> scan_module_global_symbols(std::string_view ptx);
InitializedByteArrayScan scan_initialized_byte_arrays(
    std::string_view ptx, const std::unordered_set<std::string>& referenced_symbols);
void collect_operand_symbols(std::string_view operand, std::unordered_set<std::string>* symbols);
bool symbol_is_written(const cumetal::ptx::ModuleInfo& module, std::string_view symbol);
bool resolve_immutable_table_pointers(cumetal::ptx::ModuleInfo& module,
                                      const InitializedByteArrayScan& initialized_arrays,
                                      std::string* error);

}  // namespace cumetal::ir::detail
