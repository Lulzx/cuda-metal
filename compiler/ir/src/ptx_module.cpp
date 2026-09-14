#include "ptx_module.h"
#include "ptx_instruction.h"
#include "ptx_text.h"

#include <algorithm>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace cumetal::ir::detail {

// Collect whole PTX identifier tokens, not substrings. Besides direct symbols,
// operands can contain addresses, offsets, tuples, or generic(symbol) forms.
// Register names, numeric literals, and quoted strings are not global roots.
void collect_operand_symbols(std::string_view operand,
                             std::unordered_set<std::string>* symbols) {
    const auto token_char = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) ||
               c == '_' || c == '.' || c == '$' || c == '%';
    };
    for (std::size_t cursor = 0; cursor < operand.size();) {
        if (operand[cursor] == '"') {
            ++cursor;
            while (cursor < operand.size()) {
                const char c = operand[cursor++];
                if (c == '\\' && cursor < operand.size()) ++cursor;
                else if (c == '"') break;
            }
            continue;
        }
        if (!token_char(operand[cursor])) {
            ++cursor;
            continue;
        }
        const std::size_t begin = cursor++;
        while (cursor < operand.size() && token_char(operand[cursor])) ++cursor;
        const char first = operand[begin];
        if (first != '%' && !std::isdigit(static_cast<unsigned char>(first))) {
            symbols->emplace(operand.substr(begin, cursor - begin));
        }
    }
}

std::vector<GlobalThreadgroup> scan_threadgroup_globals(std::string_view ptx) {
    const std::string source(ptx);
    const std::regex declaration(
        R"((?:\.extern\s+)?\.shared\s+\.align\s+([0-9]+)\s+\.(?:b|u|s|f)(8|16|32|64)\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*(?:\[\s*([0-9]*)\s*\])?\s*;)"
    );
    std::vector<GlobalThreadgroup> globals;
    for (std::sregex_iterator iterator(source.begin(), source.end(), declaration), end;
         iterator != end; ++iterator) {
        const std::uint64_t element_bytes =
            static_cast<std::uint64_t>(std::stoul((*iterator)[2].str())) / 8;
        const bool has_array_extent = (*iterator)[4].matched;
        const std::string extent = (*iterator)[4].str();
        const bool is_dynamic = has_array_extent && extent.empty();
        const std::uint64_t element_count =
            !has_array_extent ? 1 : (is_dynamic ? 0 : std::stoull(extent));
        globals.push_back({
            .name = (*iterator)[3].str(),
            .byte_size = element_bytes * element_count,
            .alignment = static_cast<std::uint32_t>(std::stoul((*iterator)[1].str())),
            .is_dynamic = is_dynamic,
        });
    }
    return globals;
}


std::vector<LocalDepot> scan_local_depots(std::string_view ptx) {
    const std::string source(ptx);
    const std::regex declaration(
        R"(\.local\s+\.align\s+([0-9]+)\s+\.b8\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*([0-9]+)\s*\]\s*;)"
    );
    std::vector<LocalDepot> depots;
    for (std::sregex_iterator iterator(source.begin(), source.end(), declaration), end;
         iterator != end; ++iterator) {
        depots.push_back({
            .name = (*iterator)[2].str(),
            .byte_size = std::stoull((*iterator)[3].str()),
            .alignment = static_cast<std::uint32_t>(std::stoul((*iterator)[1].str())),
        });
    }
    return depots;
}

std::unordered_set<std::string> scan_implicit_definitions(std::string_view ptx) {
    const std::string source(ptx);
    const std::regex marker(R"(//\s*implicit-def:\s*(%[A-Za-z0-9_.$]+))");
    std::unordered_set<std::string> definitions;
    for (std::sregex_iterator iterator(source.begin(), source.end(), marker), end;
         iterator != end; ++iterator) {
        definitions.insert((*iterator)[1].str());
    }
    return definitions;
}




InitializedByteArrayScan scan_initialized_byte_arrays(
    std::string_view ptx, const std::unordered_set<std::string>& referenced_symbols) {
    InitializedByteArrayScan result;
    std::istringstream lines{std::string(ptx)};
    std::string line;
    const std::regex declaration(
        R"(^\s*(?:(?:\.visible|\.extern|\.weak)\s+)?\.(const|global)\s+\.align\s+([0-9]+)\s+\.[bu]8\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*([0-9]+)\s*\]\s*=\s*\{([^}]*)\}\s*;\s*$)"
    );
    // One complete symbolic pointer object, as emitted by LLVM 19. Decode
    // into the same relocation representation as the masked byte form;
    // privacy, mutability and use validation remain in the common resolver.
    const std::regex pointer_declaration(
        R"(^\s*(?:(?:\.visible|\.extern|\.weak)\s+)?\.(const|global)\s+\.align\s+([0-9]+)\s+\.[bu]64\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*1\s*\]\s*=\s*\{\s*([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\}\s*;\s*$)"
    );
    const std::regex scalar_declaration(
        R"(^\s*(?:(?:\.visible|\.extern|\.weak)\s+)?\.(const|global)\s+\.align\s+([0-9]+)\s+\.[bus](8|16|32|64)\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*=\s*([^;]+)\s*;\s*$)"
    );
    // Recognize the declaration's identity independently of its supported
    // initializer encoding. Do not guess at an unidentifiable declaration.
    const std::regex declaration_name(
        R"(^\s*(?:(?:\.visible|\.extern|\.weak)\s+)?\.(?:const|global)\s+(?:\.align\s+[0-9]+\s+)?(?:\.v[24]\s+)?\.[busf][0-9]+\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*(?:\[[^\]\r\n]*\]\s*)*=)"
    );
    const std::regex address_size(R"(^\s*\.address_size\s+64\s*$)");
    std::unordered_map<std::string, std::string> declarations;
    std::vector<std::string> pending;
    while (std::getline(lines, line)) {
        const std::size_t comment = line.find("//");
        if (comment != std::string::npos) line.resize(comment);
        if (starts_with(trim(line), ".address_size")) {
            result.address_size_64 = std::regex_match(line, address_size);
        }
        if (line.find('=') == std::string::npos ||
            (line.find(".global") == std::string::npos &&
             line.find(".const") == std::string::npos)) {
            continue;
        }

        std::smatch match;
        if (!std::regex_search(line, match, declaration_name)) {
            result.error = "cannot identify initialized PTX declaration: " + trim(line);
            return result;
        }
        const std::string name = match[1].str();
        collect_operand_symbols(std::string_view(line).substr(line.find('=') + 1),
                                &result.dependencies[name]);
        if (referenced_symbols.contains(name)) pending.push_back(name);
        if (!declarations.emplace(name, line).second) {
            result.error = "duplicate initialized PTX symbol: " + match[1].str();
            return result;
        }
    }
    std::unordered_set<std::string> decoded;
    for (std::size_t next = 0; next < pending.size(); ++next) {
        const std::string name = pending[next];
        if (!decoded.insert(name).second || !declarations.contains(name)) continue;
        line = declarations.at(name);
        std::smatch match;
        if (std::regex_match(line, match, pointer_declaration)) {
            std::uint64_t alignment = 0;
            try { alignment = std::stoull(match[2].str()); }
            catch (...) {
                result.error = "invalid initialized PTX pointer alignment";
                return result;
            }
            const std::string target = match[4].str();
            if (alignment == 0 || alignment > UINT32_MAX ||
                !declarations.contains(target)) {
                result.error = "invalid or unresolved initialized PTX pointer declaration";
                return result;
            }
            pending.push_back(target);
            result.arrays.push_back({
                .name = match[3].str(),
                .bytes = std::vector<std::uint8_t>(8),
                .alignment = static_cast<std::uint32_t>(alignment),
                .constant_space = match[1].str() == "const",
                .module_private =
                    !starts_with(trim(line), ".visible") &&
                    !starts_with(trim(line), ".extern") &&
                    !starts_with(trim(line), ".weak"),
                .pointer_target = target,
            });
            continue;
        }
        if (line.find('{') == std::string::npos &&
            std::regex_match(line, match, scalar_declaration)) {
            std::uint64_t alignment = 0;
            std::uint64_t bits = 0;
            try {
                alignment = std::stoull(match[2].str());
                std::size_t consumed = 0;
                const long long value =
                    std::stoll(trim(match[5].str()), &consumed, 0);
                if (consumed != trim(match[5].str()).size()) {
                    throw std::invalid_argument("trailing scalar initializer text");
                }
                bits = static_cast<std::uint64_t>(value);
            } catch (...) {
                result.error = "invalid initialized PTX scalar declaration: " +
                               trim(line);
                return result;
            }
            const std::uint64_t byte_count = std::stoull(match[3].str()) / 8;
            if (alignment == 0 || alignment > UINT32_MAX || byte_count == 0) {
                result.error = "initialized PTX scalar has invalid size/alignment";
                return result;
            }
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(byte_count));
            for (std::size_t index = 0; index < bytes.size(); ++index) {
                bytes[index] = static_cast<std::uint8_t>(bits >> (index * 8));
            }
            result.arrays.push_back({
                .name = match[4].str(),
                .bytes = std::move(bytes),
                .alignment = static_cast<std::uint32_t>(alignment),
                .constant_space = match[1].str() == "const",
                .module_private =
                    !starts_with(trim(line), ".visible") &&
                    !starts_with(trim(line), ".extern") &&
                    !starts_with(trim(line), ".weak"),
            });
            continue;
        }
        if (!std::regex_match(line, match, declaration)) {
            result.error = "unsupported initialized PTX declaration: " +
                           trim(line);
            return result;
        }

        std::uint64_t declared_count = 0;
        std::uint64_t alignment = 0;
        try {
            alignment = std::stoull(match[2].str());
            declared_count = std::stoull(match[4].str());
        } catch (...) {
            result.error = "invalid initialized PTX byte-array size or alignment";
            return result;
        }
        constexpr std::uint64_t kMaxEmbeddedByteArray = 64u * 1024u * 1024u;
        if (alignment == 0 || alignment > UINT32_MAX || declared_count == 0 ||
            declared_count > kMaxEmbeddedByteArray) {
            result.error = "initialized PTX byte array has invalid or excessive size/alignment";
            return result;
        }

        std::vector<std::uint8_t> bytes;
        std::string pointer_target;
        const std::regex address_byte(
            R"(^0[xX]([0-9a-fA-F]+)\(([A-Za-z_.$][A-Za-z0-9_.$]*)\)$)");
        std::string initializer = trim(match[5].str());
        std::size_t begin = 0;
        while (begin < initializer.size()) {
            const std::size_t comma = initializer.find(',', begin);
            const std::size_t end =
                comma == std::string::npos ? initializer.size() : comma;
            const std::string item =
                trim(std::string_view(initializer).substr(begin, end - begin));
            if (item.empty()) {
                result.error = "initialized PTX byte array contains an empty element";
                return result;
            }
            std::smatch relocation;
            if (std::regex_match(item, relocation, address_byte)) {
                std::uint64_t mask = 0;
                try { mask = std::stoull(relocation[1].str(), nullptr, 16); }
                catch (...) { result.error = "invalid PTX address-byte mask"; return result; }
                if (declared_count != 8 || alignment < 8 || (alignment & (alignment - 1)) != 0 ||
                    bytes.size() >= 8 ||
                    mask != (std::uint64_t{255} << (bytes.size() * 8)) ||
                    (pointer_target.empty() && !bytes.empty()) ||
                    (!pointer_target.empty() && pointer_target != relocation[2].str())) {
                    result.error = "initialized PTX byte array has an incomplete or mixed address relocation";
                    return result;
                }
                pointer_target = relocation[2].str();
                bytes.push_back(0); // Size bookkeeping only; never emitted as data.
            } else try {
                if (!pointer_target.empty()) {
                    result.error = "initialized PTX byte array mixes address and numeric elements";
                    return result;
                }
                std::size_t consumed = 0;
                const long long value = std::stoll(item, &consumed, 0);
                if (consumed != item.size() || value < -128 || value > 255) {
                    result.error =
                        "initialized PTX byte array contains a non-byte element '" +
                        item + "'";
                    return result;
                }
                bytes.push_back(static_cast<std::uint8_t>(value & 0xff));
            } catch (...) {
                result.error =
                    "initialized PTX byte array contains an invalid element '" +
                    item + "'";
                return result;
            }
            if (bytes.size() > declared_count) {
                result.error =
                    "initialized PTX byte array has more elements than its declaration";
                return result;
            }
            if (comma == std::string::npos) break;
            begin = comma + 1;
        }
        if (!pointer_target.empty()) {
            if (bytes.size() != 8 || !declarations.contains(pointer_target)) {
                result.error = "initialized PTX byte array has an incomplete or unresolved address relocation";
                return result;
            }
            pending.push_back(pointer_target);
        }
        bytes.resize(static_cast<std::size_t>(declared_count), 0);
        result.arrays.push_back({
            .name = match[3].str(),
            .bytes = std::move(bytes),
            .alignment = static_cast<std::uint32_t>(alignment),
            .constant_space = match[1].str() == "const",
            .module_private =
                !starts_with(trim(line), ".visible") &&
                !starts_with(trim(line), ".extern") &&
                !starts_with(trim(line), ".weak"),
            .pointer_target = std::move(pointer_target),
        });
    }
    return result;
}

std::vector<ModuleConstantSymbol> scan_module_constant_symbols(std::string_view ptx) {
    const std::string source(ptx);
    const std::regex declaration(
        R"((?:\.visible\s+|\.extern\s+)?\.const\s+\.align\s+([0-9]+)\s+\.b8\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*([0-9]+)\s*\]\s*;)"
    );
    std::vector<ModuleConstantSymbol> symbols;
    std::uint64_t cursor = 0;
    for (std::sregex_iterator iterator(source.begin(), source.end(), declaration), end;
         iterator != end; ++iterator) {
        const std::uint32_t alignment =
            static_cast<std::uint32_t>(std::stoul((*iterator)[1].str()));
        cursor = (cursor + alignment - 1) / alignment * alignment;
        const std::uint64_t size = std::stoull((*iterator)[3].str());
        symbols.push_back({
            .name = (*iterator)[2].str(),
            .offset = cursor,
            .byte_size = size,
            .alignment = alignment,
        });
        cursor += size;
    }
    return symbols;
}

std::vector<ModuleConstantSymbol> scan_module_global_symbols(std::string_view ptx) {
    const std::string source(ptx);
    const std::regex declaration(
        R"((?:\.visible\s+|\.extern\s+)?\.global\s+\.align\s+([0-9]+)\s+\.b8\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*([0-9]+)\s*\]\s*;)"
    );
    std::vector<ModuleConstantSymbol> symbols;
    for (std::sregex_iterator iterator(source.begin(), source.end(), declaration), end;
         iterator != end; ++iterator) {
        symbols.push_back({
            .name = (*iterator)[2].str(),
            .offset = 0,
            .byte_size = std::stoull((*iterator)[3].str()),
            .alignment = static_cast<std::uint32_t>(
                std::stoul((*iterator)[1].str())),
        });
    }
    return symbols;
}

bool symbol_is_written(const cumetal::ptx::ModuleInfo& module, std::string_view symbol) {
    const auto writes_symbol = [&](const Instruction& instruction) {
        const std::string root = root_opcode(instruction.opcode);
        if (root != "st" && root != "atom" && root != "red") return false;
        return std::any_of(
            instruction.operands.begin(), instruction.operands.end(),
            [&](const std::string& operand) {
                return parameter_name_from_operand(operand) == symbol;
            });
    };
    const auto function_writes = [&](const cumetal::ptx::EntryFunction& function) {
        if (std::any_of(function.instructions.begin(), function.instructions.end(), writes_symbol)) return true;
        // Keep a conservative union across register reuse and predicates: any
        // possible store through a symbol-derived address prevents promotion.
        std::unordered_set<std::string> aliases;
        bool changed = true;
        for (int iteration = 0; iteration < 12 && changed; ++iteration) {
            changed = false;
            for (const auto& instruction : function.instructions) {
                const auto root = root_opcode(instruction.opcode);
                if (root != "mov" && root != "cvta" && root != "add" && root != "sub" &&
                    root != "selp" && root != "cvt") continue;
                bool derived = false;
                for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                    derived |= parameter_name_from_operand(instruction.operands[i]) == symbol;
                    for (const auto& source : registers_in(instruction.operands[i]))
                        derived |= aliases.contains(source);
                }
                if (derived)
                    for (const auto& destination : destination_registers(instruction))
                        changed |= aliases.insert(destination).second;
            }
        }
        if (changed) return true;  // An unfinished proof cannot justify promotion.
        for (const auto& instruction : function.instructions) {
            const auto root = root_opcode(instruction.opcode);
            if (root != "st" && root != "atom" && root != "red") continue;
            // Parameter stores forward values; they do not mutate their pointee.
            if (starts_with(instruction.opcode, "st.param")) continue;
            const std::size_t address = root == "atom" ? 1 : 0;
            if (instruction.operands.size() <= address) continue;
            for (const auto& source : registers_in(instruction.operands[address]))
                if (aliases.contains(source)) return true;
        }
        return false;
    };
    return std::any_of(module.entries.begin(), module.entries.end(),
                       function_writes) ||
           std::any_of(module.functions.begin(), module.functions.end(),
                       function_writes);
}

namespace {
struct PointerLoadRewrite {
    Instruction* instruction;
    std::string target;
};

bool plan_table_reads(cumetal::ptx::EntryFunction& function,
                      const InitializedByteArray& alias, const InitializedByteArray& target,
                      std::vector<PointerLoadRewrite>* rewrites, std::string* error) {
    // Conservative register dataflow across the whole function (including
    // loop backedges). A table address may only feed address arithmetic
    // and reads; storing/passing it would defeat the read-only proof.
    std::unordered_set<std::string> table_addresses;
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& instruction : function.instructions) {
            bool address_use = false;
            bool alias_load = false;
            for (const auto& operand : instruction.operands) {
                std::unordered_set<std::string> symbols;
                collect_operand_symbols(operand, &symbols);
                address_use |= symbols.contains(target.name);
                alias_load |= symbols.contains(alias.name);
            }
            for (const auto& source : source_registers(instruction)) {
                address_use |= table_addresses.contains(source);
            }
            const std::string root = root_opcode(instruction.opcode);
            if (address_use && root != "ld" && root != "mov" &&
                root != "add" && root != "sub" && root != "mad" &&
                root != "cvta" && root != "selp") {
                *error = "PTX relocated table address may escape or be written: " + target.name;
                return false;
            }
            if ((address_use && root != "ld") || (alias_load && root == "ld")) {
                for (const auto& destination : destination_registers(instruction)) {
                    changed |= table_addresses.insert(destination).second;
                }
            }
        }
    }
    for (auto& instruction : function.instructions) {
        bool uses_alias = false;
        for (const auto& operand : instruction.operands) {
            std::unordered_set<std::string> symbols;
            collect_operand_symbols(operand, &symbols);
            uses_alias |= symbols.contains(alias.name);
        }
        if (!uses_alias) continue;
        const bool whole_load = alias.constant_space
            ? (instruction.opcode == "ld.const.u64" || instruction.opcode == "ld.const.b64")
            : (instruction.opcode == "ld.global.u64" || instruction.opcode == "ld.global.b64" ||
               instruction.opcode == "ld.global.nc.u64" || instruction.opcode == "ld.global.nc.b64");
        if (!whole_load || instruction.operands.size() != 2 ||
            trim(instruction.operands[1]) != "[" + alias.name + "]") {
            *error = "PTX relocated pointer must only be read by direct whole-pointer loads: " + alias.name;
            return false;
        }
        rewrites->push_back({&instruction, alias.pointer_target});
    }
    return true;
}
}  // namespace

bool resolve_immutable_table_pointers(cumetal::ptx::ModuleInfo& module,
                                      const InitializedByteArrayScan& initialized_arrays,
                                      std::string* error) {
    // Fold only immutable, non-escaping pointer objects. This resolves the
    // relocation symbolically before SSA typing, preserving Metal address space.
    // Do not invent a numeric address or embed host/device pointer bytes in MSL.
    // Validate all uses before changing any instruction. A rejected relocation
    // leaves the parsed module untouched, including other kernels and helpers.
    std::vector<PointerLoadRewrite> rewrites;
    for (const auto& alias : initialized_arrays.arrays) {
        if (alias.pointer_target.empty()) continue;
        if (alias.bytes.size() != 8 || alias.alignment < 8 ||
            (alias.alignment & (alias.alignment - 1)) != 0) {
            *error = "PTX address relocation requires one aligned 64-bit pointer";
            return false;
        }
        if (!initialized_arrays.address_size_64) {
            *error = "PTX address relocation requires explicit 64-bit addressing";
            return false;
        }
        const auto target = std::find_if(initialized_arrays.arrays.begin(),
            initialized_arrays.arrays.end(), [&](const auto& value) {
                return value.name == alias.pointer_target;
            });
        if (!alias.module_private || target == initialized_arrays.arrays.end() ||
            !target->pointer_target.empty() || !target->module_private ||
            symbol_is_written(module, target->name)) {
            *error = "PTX address relocation requires a private read-only numeric target: " + alias.name;
            return false;
        }
        for (const auto& [name, dependencies] : initialized_arrays.dependencies) {
            if ((name != alias.name && dependencies.contains(target->name)) ||
                dependencies.contains(alias.name)) {
                *error = "PTX relocated table has another escaping initializer reference: " + name;
                return false;
            }
        }
        for (auto& function : module.entries) if (!plan_table_reads(function, alias, *target, &rewrites, error)) return false;
        for (auto& function : module.functions) if (!plan_table_reads(function, alias, *target, &rewrites, error)) return false;
    }
    for (const auto& rewrite : rewrites) {
        rewrite.instruction->opcode = "mov.u64";
        rewrite.instruction->operands[1] = rewrite.target;
    }
    return true;
}

}  // namespace cumetal::ir::detail
