#include "ptx_module.h"
#include "ptx_text.h"

#include <regex>
#include <sstream>
#include <stdexcept>

namespace cumetal::ir::detail {

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




InitializedByteArrayScan scan_initialized_byte_arrays(std::string_view ptx) {
    InitializedByteArrayScan result;
    std::istringstream lines{std::string(ptx)};
    std::string line;
    const std::regex declaration(
        R"(^\s*(?:(?:\.visible|\.extern|\.weak)\s+)?\.(const|global)\s+\.align\s+([0-9]+)\s+\.b8\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*([0-9]+)\s*\]\s*=\s*\{([^}]*)\}\s*;\s*$)"
    );
    const std::regex scalar_declaration(
        R"(^\s*(?:(?:\.visible|\.extern|\.weak)\s+)?\.(const|global)\s+\.align\s+([0-9]+)\s+\.[bus](8|16|32|64)\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*=\s*([^;]+)\s*;\s*$)"
    );
    while (std::getline(lines, line)) {
        const std::size_t comment = line.find("//");
        if (comment != std::string::npos) line.resize(comment);
        if (line.find('=') == std::string::npos ||
            (line.find(".global") == std::string::npos &&
             line.find(".const") == std::string::npos)) {
            continue;
        }

        std::smatch match;
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
            try {
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

}  // namespace cumetal::ir::detail
