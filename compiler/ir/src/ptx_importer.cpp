#include "cumetal/ir/ptx_importer.h"

#include "cumetal/passes/printf_lower.h"
#include "cumetal/ptx/parser.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace cumetal::ir {
namespace {

using Instruction = cumetal::ptx::EntryFunction::Instruction;

struct RawBlock {
    BlockId id = kInvalidBlock;
    std::string name;
    std::vector<const Instruction*> instructions;
    std::vector<std::size_t> successors;
    std::vector<std::size_t> predecessors;
    std::unordered_map<std::string, ValueId> last_definitions;
    std::unordered_set<std::string> uses_before_definition;
};

std::string trim(std::string_view input) {
    std::size_t begin = 0;
    while (begin < input.size() &&
           std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
        ++begin;
    }
    std::size_t end = input.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }
    return std::string(input.substr(begin, end - begin));
}

std::string root_opcode(std::string_view opcode) {
    const std::size_t dot = opcode.find('.');
    return std::string(opcode.substr(0, dot));
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

std::vector<std::string> registers_in(std::string_view input) {
    std::vector<std::string> registers;
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] != '%') {
            continue;
        }
        std::size_t end = i + 1;
        while (end < input.size()) {
            const unsigned char c = static_cast<unsigned char>(input[end]);
            if (std::isalnum(c) == 0 && c != '_' && c != '.' && c != '$') {
                break;
            }
            ++end;
        }
        if (end > i + 1) {
            registers.emplace_back(input.substr(i, end - i));
            i = end - 1;
        }
    }
    return registers;
}

std::string first_register(std::string_view input) {
    const std::vector<std::string> registers = registers_in(input);
    return registers.empty() ? std::string{} : registers.front();
}

std::vector<std::string> destination_registers(const Instruction& instruction) {
    const std::string root = root_opcode(instruction.opcode);
    if (instruction.opcode == "ptx.label" || instruction.operands.empty() ||
        root == "st" || root == "bra" || root == "bar" || root == "membar" ||
        root == "fence" || root == "ret" || root == "exit" || root == "trap" ||
        root == "call") {
        return {};
    }
    std::vector<std::string> destinations = registers_in(instruction.operands.front());
    const bool tuple_move = root == "mov" &&
                            instruction.opcode.find(".b64") != std::string::npos;
    if (root != "setp" && root != "shfl" && !tuple_move &&
        destinations.size() > 1) {
        destinations.resize(1);
    }
    return destinations;
}

std::vector<std::string> source_registers(const Instruction& instruction) {
    std::vector<std::string> sources;
    const std::string root = root_opcode(instruction.opcode);
    std::size_t first_source = destination_registers(instruction).empty() ? 0 : 1;
    if (root == "st") {
        first_source = 0;
    }
    for (std::size_t i = first_source; i < instruction.operands.size(); ++i) {
        const std::vector<std::string> found = registers_in(instruction.operands[i]);
        sources.insert(sources.end(), found.begin(), found.end());
    }
    if (!instruction.predicate.empty()) {
        const std::string predicate = first_register(instruction.predicate);
        if (!predicate.empty()) {
            sources.push_back(predicate);
        }
    }
    std::erase_if(sources, [](const std::string& name) {
        return starts_with(name, "%tid.") || starts_with(name, "%ctaid.") ||
               starts_with(name, "%ntid.") || starts_with(name, "%nctaid.") ||
               name == "%laneid" || name == "%warpid" || name == "%smid" ||
               name == "%activemask" || starts_with(name, "%clock");
    });
    return sources;
}

std::uint32_t ptx_type_bits(std::string_view type) {
    for (std::uint32_t bits : {8U, 16U, 32U, 64U}) {
        if (type.find(std::to_string(bits)) != std::string_view::npos) {
            return bits;
        }
    }
    return 32;
}

Type ptx_scalar_type(std::string_view spelling) {
    if (spelling.find(".pred") != std::string_view::npos) {
        return Type::predicate();
    }
    const std::uint32_t bits = ptx_type_bits(spelling);
    if (spelling.find(".f") != std::string_view::npos) {
        return Type::floating(bits);
    }
    return Type::integer(bits);
}

Type ptx_cvt_result_type(std::string_view opcode) {
    std::size_t cursor = opcode.find('.');
    while (cursor != std::string_view::npos && cursor + 1 < opcode.size()) {
        const std::size_t begin = cursor + 1;
        const std::size_t end = opcode.find('.', begin);
        const std::string_view token = opcode.substr(
            begin, (end == std::string_view::npos ? opcode.size() : end) - begin);
        if (token.size() >= 2 &&
            (token.front() == 'u' || token.front() == 's' ||
             token.front() == 'b' || token.front() == 'f') &&
            std::all_of(token.begin() + 1, token.end(), [](char c) {
                return c >= '0' && c <= '9';
            })) {
            const std::uint32_t bits = static_cast<std::uint32_t>(
                std::stoul(std::string(token.substr(1))));
            return token.front() == 'f' ? Type::floating(bits)
                                        : Type::integer(bits);
        }
        cursor = end;
    }
    return ptx_scalar_type(opcode);
}

Type ptx_cvt_source_type(std::string_view opcode) {
    Type source = ptx_scalar_type(opcode);
    std::size_t cursor = opcode.find('.');
    while (cursor != std::string_view::npos && cursor + 1 < opcode.size()) {
        const std::size_t begin = cursor + 1;
        const std::size_t end = opcode.find('.', begin);
        const std::string_view token = opcode.substr(
            begin, (end == std::string_view::npos ? opcode.size() : end) - begin);
        if (token.size() >= 2 &&
            (token.front() == 'u' || token.front() == 's' ||
             token.front() == 'b' || token.front() == 'f') &&
            std::all_of(token.begin() + 1, token.end(), [](char c) {
                return c >= '0' && c <= '9';
            })) {
            const std::uint32_t bits = static_cast<std::uint32_t>(
                std::stoul(std::string(token.substr(1))));
            source = token.front() == 'f' ? Type::floating(bits)
                                          : Type::integer(bits);
        }
        cursor = end;
    }
    return source;
}

Type parameter_type(const cumetal::ptx::Parameter& parameter) {
    if (parameter.is_pointer) {
        return Type::pointer(Type::integer(8), AddressSpace::kDevice);
    }
    const Type scalar = ptx_scalar_type(parameter.type);
    const std::uint32_t scalar_size =
        std::max<std::uint32_t>(1, scalar.bit_width / 8);
    if (parameter.byte_size <= scalar_size) return scalar;

    std::vector<Type> fields;
    if (parameter.byte_size % 4 == 0 && parameter.alignment >= 4) {
        fields.assign(parameter.byte_size / 4, Type::integer(32));
    } else {
        fields.assign(parameter.byte_size, Type::integer(8));
    }
    return Type::aggregate(
        std::move(fields),
        "CuMetalPackedParam" + std::to_string(parameter.byte_size));
}

std::uint32_t type_size(const Type& type) {
    if (type.is_pointer()) return 8;
    if (type.kind == TypeKind::kPredicate) return 1;
    if (type.kind == TypeKind::kAggregate) {
        std::uint32_t total = 0;
        for (const Type& element : type.elements) total += type_size(element);
        return total;
    }
    return std::max<std::uint32_t>(1, type.bit_width / 8);
}

std::string parameter_name_from_operand(std::string_view operand) {
    const std::size_t open = operand.find('[');
    const std::size_t close = operand.find(']');
    if (open == std::string_view::npos || close == std::string_view::npos || close <= open + 1) {
        return trim(operand);
    }
    std::string inside = trim(operand.substr(open + 1, close - open - 1));
    const std::size_t offset = inside.find_first_of(" +");
    if (offset != std::string::npos) {
        inside.resize(offset);
    }
    return inside;
}

std::vector<std::string> grouped_names(std::string_view operand) {
    std::string contents = trim(operand);
    if (contents.size() >= 2 && contents.front() == '(' && contents.back() == ')') {
        contents = trim(std::string_view(contents).substr(1, contents.size() - 2));
    }
    std::vector<std::string> names;
    std::size_t begin = 0;
    while (begin < contents.size()) {
        const std::size_t comma = contents.find(',', begin);
        const std::size_t end = comma == std::string::npos ? contents.size() : comma;
        const std::string name = trim(std::string_view(contents).substr(begin, end - begin));
        if (!name.empty()) names.push_back(name);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return names;
}

struct BuiltinSignature {
    std::string metal_name;
    Type return_type;
    std::vector<Type> argument_types;
    bool tolerance_bounded = false;
};

std::optional<BuiltinSignature> cuda_builtin_signature(std::string_view name) {
    static const std::unordered_map<std::string, std::string> kDoubleBuiltins = {
        {"__nv_fma", "fma"}, {"__nv_sqrt", "sqrt"},
        {"__nv_rsqrt", "rsqrt"},
        {"__nv_fmin", "fmin"}, {"__nv_fmax", "fmax"},
        {"__nv_remainder", "remainder"}, {"__nv_floor", "floor"},
        {"__nv_ceil", "ceil"}, {"__nv_trunc", "trunc"},
        {"__nv_round", "round"}, {"__nv_rint", "rint"},
    };
    const auto double_builtin = kDoubleBuiltins.find(std::string(name));
    if (double_builtin != kDoubleBuiltins.end()) {
        const std::size_t arity = name == "__nv_fma" ? 3 :
                                  (name == "__nv_fmin" || name == "__nv_fmax" ||
                                   name == "__nv_remainder" ? 2 : 1);
        return BuiltinSignature{
            .metal_name = double_builtin->second,
            .return_type = Type::floating(64),
            .argument_types = std::vector<Type>(arity, Type::floating(64)),
        };
    }
    static const std::unordered_map<std::string, std::string> kFloatBuiltins = {
        {"__nv_fminf", "fmin"}, {"__nv_fmaxf", "fmax"},
        {"__nv_sqrtf", "sqrt"}, {"__nv_rsqrtf", "rsqrt"},
        {"__nv_fabsf", "fabs"}, {"__nv_acosf", "acos"},
        {"__nv_expf", "exp"}, {"__nv_fast_expf", "exp"},
        {"__nv_exp2f", "exp2"}, {"__nv_exp10f", "exp10"},
        {"__nv_expm1f", "expm1"}, {"__nv_logf", "log"},
        {"__nv_log2f", "log2"}, {"__nv_log10f", "log10"},
        {"__nv_log1pf", "log1p"}, {"__nv_sinf", "sin"},
        {"__nv_cosf", "cos"}, {"__nv_tanf", "tan"},
        {"__nv_sinhf", "sinh"}, {"__nv_coshf", "cosh"},
        {"__nv_tanhf", "tanh"}, {"__nv_asinf", "asin"},
        {"__nv_atanf", "atan"}, {"__nv_atan2f", "atan2"},
        {"__nv_asinhf", "asinh"}, {"__nv_acoshf", "acosh"},
        {"__nv_atanhf", "atanh"}, {"__nv_cbrtf", "cbrt"},
        {"__nv_erff", "erf"}, {"__nv_erfcf", "erfc"},
        {"__nv_floorf", "floor"}, {"__nv_ceilf", "ceil"},
        {"__nv_truncf", "trunc"}, {"__nv_roundf", "round"},
        {"__nv_rintf", "rint"}, {"__nv_powf", "pow"},
        {"__nv_hypotf", "hypot"}, {"__nv_fmodf", "fmod"},
        {"__nv_copysignf", "copysign"}, {"__nv_fdimf", "fdim"},
        {"__nv_remainderf", "remainder"}, {"__nv_fmaf", "fma"},
    };
    const auto builtin = kFloatBuiltins.find(std::string(name));
    if (builtin != kFloatBuiltins.end()) {
        const std::size_t arity =
            name == "__nv_fmaf" ? 3 :
            (name == "__nv_atan2f" || name == "__nv_powf" ||
             name == "__nv_hypotf" || name == "__nv_fmodf" ||
             name == "__nv_copysignf" || name == "__nv_fdimf" ||
             name == "__nv_remainderf" || name == "__nv_fminf" ||
             name == "__nv_fmaxf" ? 2 : 1);
        static const std::unordered_set<std::string> kExpandedMath = {
            "__nv_expm1f", "__nv_log1pf", "__nv_cbrtf", "__nv_erff",
            "__nv_erfcf", "__nv_hypotf", "__nv_remainderf",
        };
        return BuiltinSignature{
            .metal_name = builtin->second,
            .return_type = Type::floating(32),
            .argument_types = std::vector<Type>(arity, Type::floating(32)),
            .tolerance_bounded = kExpandedMath.contains(std::string(name)),
        };
    }
    // Float <-> integer conversions. The rounding mode is part of the name and
    // applies to the float before the cast, so float->int goes through a helper
    // that lower_to_msl expands into round-then-cast rather than a bare cast.
    // Int->float is a plain numeric conversion in every mode.
    static const std::unordered_map<std::string, std::string> kFloatToIntBuiltins = {
        {"__nv_float2int_rn", "__cumetal_float2int_rne"},
        {"__nv_float2int_rz", "__cumetal_float2int_rtz"},
        {"__nv_float2int_ru", "__cumetal_float2int_rtp"},
        {"__nv_float2int_rd", "__cumetal_float2int_rtn"},
        {"__nv_float2uint_rn", "__cumetal_float2uint_rne"},
        {"__nv_float2uint_rz", "__cumetal_float2uint_rtz"},
        {"__nv_float2uint_ru", "__cumetal_float2uint_rtp"},
        {"__nv_float2uint_rd", "__cumetal_float2uint_rtn"},
    };
    const auto float_to_int = kFloatToIntBuiltins.find(std::string(name));
    if (float_to_int != kFloatToIntBuiltins.end()) {
        return BuiltinSignature{
            .metal_name = float_to_int->second,
            .return_type = Type::integer(32),
            .argument_types = {Type::floating(32)},
        };
    }
    static const std::unordered_map<std::string, std::string> kFloatToLongBuiltins = {
        {"__nv_float2ll_rn", "__cumetal_float2int_rne"},
        {"__nv_float2ll_rz", "__cumetal_float2int_rtz"},
        {"__nv_float2ll_ru", "__cumetal_float2int_rtp"},
        {"__nv_float2ll_rd", "__cumetal_float2int_rtn"},
        {"__nv_float2ull_rn", "__cumetal_float2uint_rne"},
        {"__nv_float2ull_rz", "__cumetal_float2uint_rtz"},
        {"__nv_float2ull_ru", "__cumetal_float2uint_rtp"},
        {"__nv_float2ull_rd", "__cumetal_float2uint_rtn"},
    };
    const auto float_to_long = kFloatToLongBuiltins.find(std::string(name));
    if (float_to_long != kFloatToLongBuiltins.end()) {
        return BuiltinSignature{
            .metal_name = float_to_long->second,
            .return_type = Type::integer(64),
            .argument_types = {Type::floating(32)},
        };
    }
    if (name == "__nv_float_as_uint" || name == "__nv_float_as_int") {
        return BuiltinSignature{
            .metal_name = "__cumetal_float_as_uint",
            .return_type = Type::integer(32),
            .argument_types = {Type::floating(32)},
        };
    }
    if (name == "__nv_uint_as_float" || name == "__nv_int_as_float") {
        return BuiltinSignature{
            .metal_name = "__cumetal_uint_as_float",
            .return_type = Type::floating(32),
            .argument_types = {Type::integer(32)},
        };
    }
    if (name == "__nv_popc" || name == "__nv_clz" || name == "__nv_abs" ||
        name == "__nv_ffs") {
        return BuiltinSignature{
            .metal_name = name == "__nv_popc" ? "popcount" :
                          name == "__nv_clz" ? "clz" :
                          name == "__nv_abs" ? "__cumetal_signed_abs" :
                                               "__cumetal_ffs",
            .return_type = Type::integer(32),
            .argument_types = {Type::integer(32)},
        };
    }
    if (name == "__nv_frexp") {
        return BuiltinSignature{
            .metal_name = "frexp",
            .return_type = Type::floating(32),
            .argument_types = {
                Type::floating(32),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
            },
        };
    }
    return std::nullopt;
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

struct LocalDepot {
    std::string name;
    std::uint64_t byte_size = 0;
    std::uint32_t alignment = 1;
};

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

std::int64_t memory_operand_offset(std::string_view operand) {
    const std::size_t open = operand.find('[');
    const std::size_t close = operand.find(']');
    if (open == std::string_view::npos || close == std::string_view::npos || close <= open) {
        return 0;
    }
    const std::string inside = trim(operand.substr(open + 1, close - open - 1));
    const std::size_t sign = inside.find_first_of("+-");
    if (sign == std::string::npos) return 0;
    try {
        const std::int64_t magnitude = std::stoll(trim(std::string_view(inside).substr(sign + 1)));
        return inside[sign] == '-' ? -magnitude : magnitude;
    } catch (...) {
        return 0;
    }
}

std::string branch_target(const Instruction& instruction) {
    return instruction.operands.empty() ? std::string{} : trim(instruction.operands.back());
}

bool is_conditional_branch(const Instruction& instruction) {
    return root_opcode(instruction.opcode) == "bra" && !instruction.predicate.empty();
}

bool is_terminating_instruction(const Instruction& instruction) {
    const std::string root = root_opcode(instruction.opcode);
    return root == "bra" || root == "ret" || root == "exit" || root == "trap";
}

std::optional<std::string> direct_call_target(const Instruction& instruction) {
    if (root_opcode(instruction.opcode) != "call") return std::nullopt;
    const bool has_return = instruction.operands.size() == 3;
    if ((!has_return && instruction.operands.size() != 2) ||
        (has_return && grouped_names(instruction.operands.front()).size() != 1)) {
        return std::nullopt;
    }
    return trim(instruction.operands[has_return ? 1 : 0]);
}

OpCode arithmetic_opcode(std::string_view root) {
    if (root == "add") return OpCode::kAdd;
    if (root == "sub") return OpCode::kSub;
    if (root == "mul" || root == "mad") return OpCode::kMul;
    if (root == "div") return OpCode::kDiv;
    if (root == "rem") return OpCode::kRemainder;
    if (root == "fma") return OpCode::kFma;
    if (root == "neg") return OpCode::kNegate;
    if (root == "and") return OpCode::kBitAnd;
    if (root == "or") return OpCode::kBitOr;
    if (root == "xor") return OpCode::kBitXor;
    if (root == "shl") return OpCode::kShiftLeft;
    if (root == "shr") return OpCode::kShiftRight;
    return OpCode::kInvalid;
}

std::string comparison_predicate(std::string_view opcode) {
    static constexpr std::string_view kPredicates[] = {
        "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs",
        "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan",
    };
    for (std::string_view predicate : kPredicates) {
        const std::string token = "." + std::string(predicate) + ".";
        if (opcode.find(token) != std::string_view::npos) {
            return std::string(predicate);
        }
    }
    return "eq";
}

bool has_signed_integer_type(std::string_view opcode) {
    return opcode.find(".s8") != std::string_view::npos ||
           opcode.find(".s16") != std::string_view::npos ||
           opcode.find(".s32") != std::string_view::npos ||
           opcode.find(".s64") != std::string_view::npos;
}

std::uint32_t ptx_register_container_bits(std::string_view name) {
    if (starts_with(name, "%rd") || starts_with(name, "%fd")) return 64;
    if (starts_with(name, "%rs") || starts_with(name, "%h")) return 16;
    if (starts_with(name, "%r") || starts_with(name, "%f")) return 32;
    return 0;
}

bool cvt_has_signed_source(std::string_view opcode) {
    return opcode.ends_with(".s8") || opcode.ends_with(".s16") ||
           opcode.ends_with(".s32") || opcode.ends_with(".s64");
}

std::pair<std::string, bool> normalized_predicate(std::string_view predicate) {
    const bool inverted = predicate.find('!') != std::string_view::npos;
    return {first_register(predicate), inverted};
}

MemoryScope memory_scope_from_opcode(std::string_view opcode) {
    if (opcode.find(".cta") != std::string_view::npos ||
        opcode.find(".shared") != std::string_view::npos) {
        return MemoryScope::kThreadgroup;
    }
    if (opcode.find(".warp") != std::string_view::npos) {
        return MemoryScope::kSimdgroup;
    }
    if (opcode.find(".sys") != std::string_view::npos) {
        return MemoryScope::kSystem;
    }
    return MemoryScope::kDevice;
}

MemoryOrdering memory_ordering_from_opcode(std::string_view opcode) {
    if (opcode.find(".acq_rel") != std::string_view::npos) return MemoryOrdering::kAcquireRelease;
    if (opcode.find(".acquire") != std::string_view::npos) return MemoryOrdering::kAcquire;
    if (opcode.find(".release") != std::string_view::npos) return MemoryOrdering::kRelease;
    if (opcode.find(".sc") != std::string_view::npos) return MemoryOrdering::kSequentiallyConsistent;
    return MemoryOrdering::kRelaxed;
}

std::string atomic_operation_from_opcode(std::string_view opcode) {
    static constexpr std::string_view kOperations[] = {
        "add", "sub", "and", "or", "xor", "cas", "min", "max", "exch",
    };
    for (std::string_view operation : kOperations) {
        const std::string token = "." + std::string(operation) + ".";
        if (opcode.find(token) != std::string_view::npos) {
            return std::string(operation);
        }
    }
    return {};
}

struct Importer {
    Builder builder;
    PtxImportResult result;
    const cumetal::ptx::EntryFunction* entry = nullptr;
    bool is_kernel = true;
    std::unordered_map<std::string, const cumetal::ptx::EntryFunction*>
        device_functions;
    std::unordered_set<std::string> printf_functions;
    std::unordered_map<std::string, Type> parameter_types;
    std::unordered_map<std::string, ValueId> parameter_values;
    // Older CUDA Clang releases materialize the address of a by-value
    // aggregate parameter with `mov.b64 %rd, param` and subsequently issue
    // `ld.param [%rd+offset]`.  CuMetal models the parameter as an SSA
    // aggregate, so retain which SSA aliases denote that address.
    std::unordered_map<ValueId, std::string> aggregate_parameter_addresses;
    // PTX virtual registers are SSA-like, but CuMetal's CFG construction can
    // replace a value with a block argument at a join.  Retain the symbolic
    // register alias as well so an aggregate parameter address survives that
    // representation change.
    std::unordered_map<std::string, std::string> aggregate_parameter_registers;
    std::unordered_map<std::string, Type> register_types;
    std::unordered_map<const Instruction*, std::vector<ValueId>> instruction_results;
    std::unordered_map<ValueId, Type> value_types;
    std::unordered_set<ValueId> integer_zero_values;
    std::vector<RawBlock> raw_blocks;
    std::unordered_map<std::string, std::size_t> label_blocks;
    std::vector<std::unordered_map<std::string, ValueId>> incoming;
    std::vector<std::unordered_map<std::string, ValueId>> outgoing;
    std::vector<std::map<std::string, ValueId>> block_arguments;
    std::unordered_map<std::string, Operand> call_parameter_slots;
    std::unordered_map<std::string, std::map<std::int64_t, Operand>>
        call_parameter_slot_fields;
    std::unordered_map<std::string, Operand> call_return_slots;
    std::unordered_set<std::string> threadgroup_symbols;
    std::unordered_map<std::string, LocalDepot> local_depots;
    std::unordered_map<std::string, Operand> local_depot_values;
    std::unordered_set<std::string> implicit_definitions;
    std::unordered_map<std::string, ValueId> implicit_values;
    std::unordered_map<std::string, ModuleConstantSymbol> module_constant_symbols;
    std::uint64_t module_constant_buffer_size = 0;
    std::optional<Operand> module_constant_buffer;
    std::vector<ModuleConstantSymbol> module_global_symbols;
    std::unordered_map<std::string, Operand> module_global_values;
    std::unordered_map<std::string, ModuleConstantSymbol>
        module_initialized_symbols;
    std::unordered_map<int, cumetal::passes::PrintfLoweredCall> printf_calls;
    std::unordered_set<int> printf_scaffold_lines;
    std::optional<Operand> printf_buffer;
    std::optional<Operand> printf_capacity;
    std::optional<Operand> function_return;
    std::map<std::int64_t, Operand> function_return_fields;

    bool fail(const Instruction* instruction, std::string message) {
        if (instruction != nullptr && instruction->line != 0) {
            message = "line " + std::to_string(instruction->line) + ": " + message;
        }
        result.error = std::move(message);
        return false;
    }

    std::optional<Operand> materialize_aggregate(
        BasicBlock* block, const Instruction* instruction, const Type& type,
        const std::map<std::int64_t, Operand>& fields,
        std::string_view description) {
        if (type.kind != TypeKind::kAggregate || type.elements.empty()) {
            fail(instruction,
                 std::string(description) + " does not have an aggregate type");
            return std::nullopt;
        }
        if (fields.size() != type.elements.size()) {
            fail(instruction, std::string(description) +
                                  " has missing, partial, or overlapping fields");
            return std::nullopt;
        }
        Operation construct;
        construct.opcode = OpCode::kAggregateConstruct;
        construct.attributes["aggregate_init"] = "true";
        construct.location = {
            .file = result.module.source_name,
            .line = static_cast<std::uint32_t>(
                std::max(0, instruction == nullptr ? 0 : instruction->line)),
        };
        std::int64_t byte_offset = 0;
        for (const Type& element_type : type.elements) {
            const auto field = fields.find(byte_offset);
            if (field == fields.end()) {
                fail(instruction, std::string(description) +
                                      " is missing field at byte offset " +
                                      std::to_string(byte_offset));
                return std::nullopt;
            }
            const std::uint32_t element_size = type_size(element_type);
            if (type_size(field->second.type) != element_size) {
                fail(instruction, std::string(description) +
                                      " has a partial or overlapping field at byte offset " +
                                      std::to_string(byte_offset));
                return std::nullopt;
            }
            Operand value = field->second;
            if (!(value.type == element_type)) {
                Operation conversion;
                conversion.opcode = OpCode::kConvert;
                conversion.location = construct.location;
                conversion.operands = {value};
                conversion.attributes["bitcast"] = "true";
                const ValueId converted = builder.next_value();
                conversion.results = {converted};
                conversion.result_types = {element_type};
                value_types[converted] = element_type;
                block->operations.push_back(std::move(conversion));
                value = Operand::value_ref(converted, element_type);
            }
            construct.operands.push_back(std::move(value));
            byte_offset += element_size;
        }
        if (byte_offset != static_cast<std::int64_t>(type_size(type))) {
            fail(instruction,
                 std::string(description) + " has an inconsistent aggregate layout");
            return std::nullopt;
        }
        const ValueId value = builder.next_value();
        construct.results = {value};
        construct.result_types = {type};
        value_types[value] = type;
        block->operations.push_back(std::move(construct));
        return Operand::value_ref(value, type);
    }

    bool select_entry(const cumetal::ptx::ParseResult& parsed, const PtxImportOptions& options) {
        if (parsed.module.entries.empty()) {
            result.error = "PTX module contains no kernel entries";
            return false;
        }
        if (options.entry_name.empty()) {
            entry = &parsed.module.entries.front();
            return true;
        }
        for (const auto& candidate : parsed.module.entries) {
            if (candidate.name == options.entry_name) {
                entry = &candidate;
                return true;
            }
        }
        result.error = "PTX entry not found: " + options.entry_name;
        return false;
    }

    void build_cfg() {
        const auto& instructions = entry->instructions;
        std::set<std::size_t> leaders = {0};
        for (std::size_t i = 0; i < instructions.size(); ++i) {
            if (instructions[i].opcode == "ptx.label") {
                leaders.insert(i);
            }
            if (is_terminating_instruction(instructions[i]) && i + 1 < instructions.size()) {
                leaders.insert(i + 1);
            }
        }
        std::vector<std::size_t> leader_list(leaders.begin(), leaders.end());
        for (std::size_t block_index = 0; block_index < leader_list.size(); ++block_index) {
            const std::size_t begin = leader_list[block_index];
            const std::size_t end =
                block_index + 1 < leader_list.size() ? leader_list[block_index + 1] : instructions.size();
            RawBlock block;
            block.id = builder.next_block();
            block.name = "bb" + std::to_string(block_index);
            for (std::size_t i = begin; i < end; ++i) {
                const Instruction& instruction = instructions[i];
                if (instruction.opcode == "ptx.label") {
                    if (!instruction.operands.empty()) {
                        block.name = instruction.operands.front();
                        label_blocks[block.name] = block_index;
                    }
                    continue;
                }
                block.instructions.push_back(&instruction);
            }
            raw_blocks.push_back(std::move(block));
        }

        for (std::size_t i = 0; i < raw_blocks.size(); ++i) {
            RawBlock& block = raw_blocks[i];
            const Instruction* last = block.instructions.empty() ? nullptr : block.instructions.back();
            if (last != nullptr && root_opcode(last->opcode) == "bra") {
                const auto target = label_blocks.find(branch_target(*last));
                if (target != label_blocks.end()) {
                    block.successors.push_back(target->second);
                }
                if (is_conditional_branch(*last) && i + 1 < raw_blocks.size()) {
                    block.successors.push_back(i + 1);
                }
            } else if (last == nullptr ||
                       (root_opcode(last->opcode) != "ret" &&
                        root_opcode(last->opcode) != "exit" &&
                        root_opcode(last->opcode) != "trap")) {
                if (i + 1 < raw_blocks.size()) {
                    block.successors.push_back(i + 1);
                }
            }
        }
        for (std::size_t i = 0; i < raw_blocks.size(); ++i) {
            for (std::size_t successor : raw_blocks[i].successors) {
                raw_blocks[successor].predecessors.push_back(i);
            }
        }
    }

    void infer_register_types() {
        for (const auto& parameter : entry->params) {
            parameter_types[parameter.name] = parameter_type(parameter);
        }

        // Older CUDA Clang PTX (notably 21) omits `.ptr` from device-function
        // parameters even when the CUDA source type is a pointer. Recover that
        // information from actual address use before forward type inference.
        // This is deliberately bounded to direct dataflow through the common
        // mov/ld.param, add, and selp forms; ambiguous integer-only values stay
        // integers instead of being guessed as pointers.
        std::unordered_set<std::string> required_device_pointers;
        for (const Instruction& instruction : entry->instructions) {
            const std::string root = root_opcode(instruction.opcode);
            if (root != "ld" && root != "st") continue;
            if (instruction.opcode.find(".param") != std::string::npos ||
                instruction.opcode.find(".shared") != std::string::npos ||
                instruction.opcode.find(".local") != std::string::npos ||
                instruction.opcode.find(".const") != std::string::npos) {
                continue;
            }
            const std::size_t memory_index = root == "st" ? 0 : 1;
            if (instruction.operands.size() <= memory_index) continue;
            const std::string base =
                first_register(instruction.operands[memory_index]);
            if (!base.empty()) required_device_pointers.insert(base);
        }
        bool pointer_changed = true;
        for (int iteration = 0; iteration < 12 && pointer_changed; ++iteration) {
            pointer_changed = false;
            for (const Instruction& instruction : entry->instructions) {
                const std::vector<std::string> destinations =
                    destination_registers(instruction);
                if (std::none_of(destinations.begin(), destinations.end(),
                                 [&](const std::string& destination) {
                                     return required_device_pointers.contains(destination);
                                 })) {
                    continue;
                }
                const std::string root = root_opcode(instruction.opcode);
                std::vector<std::size_t> pointer_sources;
                if (root == "mov" || starts_with(instruction.opcode, "ld.param")) {
                    pointer_sources = {1};
                } else if (root == "add") {
                    pointer_sources = {1};
                } else if (root == "selp") {
                    pointer_sources = {1, 2};
                }
                for (const std::size_t source_index : pointer_sources) {
                    if (instruction.operands.size() <= source_index) continue;
                    const std::string source =
                        first_register(instruction.operands[source_index]);
                    if (!source.empty() &&
                        required_device_pointers.insert(source).second) {
                        pointer_changed = true;
                    }
                    const std::string parameter = parameter_name_from_operand(
                        instruction.operands[source_index]);
                    const auto parameter_type_it = parameter_types.find(parameter);
                    if (parameter_type_it != parameter_types.end() &&
                        !parameter_type_it->second.is_pointer()) {
                        parameter_type_it->second = Type::pointer(
                            Type::integer(8), AddressSpace::kDevice);
                        pointer_changed = true;
                    }
                }
            }
        }

        // A CUDA kernel pointer parameter is a launch-time device pointer, but
        // an ordinary device function receives a CUDA generic pointer. Clang
        // 21-23 commonly drops `.ptr` entirely from the latter's PTX signature,
        // so the dataflow recovery above can prove pointer-ness but cannot pick
        // a concrete address space. Leave helpers generic and let the typed
        // interprocedural solver specialize them from direct call sites (for
        // example, a shared-memory argument passed through `cvta.shared`).
        if (!is_kernel) {
            for (const auto& parameter : entry->params) {
                Type& type = parameter_types[parameter.name];
                if (type.is_pointer()) type.address_space = AddressSpace::kNone;
            }
        }

        bool changed = true;
        for (int iteration = 0; iteration < 12 && changed; ++iteration) {
            changed = false;
            for (const Instruction& instruction : entry->instructions) {
                const std::vector<std::string> destinations = destination_registers(instruction);
                if (destinations.empty()) continue;
                Type inferred = ptx_scalar_type(instruction.opcode);
                const std::string root = root_opcode(instruction.opcode);
                if (root == "setp") {
                    inferred = Type::predicate();
                } else if (starts_with(instruction.opcode, "ld.") &&
                           has_signed_integer_type(instruction.opcode) &&
                           !destinations.empty()) {
                    const std::uint32_t container_bits =
                        ptx_register_container_bits(destinations.front());
                    if (container_bits > inferred.bit_width) {
                        inferred = Type::integer(container_bits);
                    }
                } else if (root == "mov" && instruction.operands.size() >= 2 &&
                           starts_with(trim(instruction.operands[1]), "0f")) {
                    inferred = Type::floating(32);
                } else if (root == "mov" && instruction.operands.size() >= 2 &&
                           parameter_types.contains(parameter_name_from_operand(
                               instruction.operands[1]))) {
                    inferred = parameter_types.at(parameter_name_from_operand(
                        instruction.operands[1]));
                } else if (root == "mov" && instruction.operands.size() >= 2 &&
                           register_types.contains(
                               first_register(instruction.operands[1]))) {
                    const std::string source =
                        first_register(instruction.operands[1]);
                    inferred = register_types.at(source);
                } else if (starts_with(instruction.opcode, "ld.param") &&
                           instruction.operands.size() >= 2) {
                    const auto parameter = parameter_types.find(
                        parameter_name_from_operand(instruction.operands[1]));
                    if (parameter != parameter_types.end() &&
                        parameter->second.kind != TypeKind::kAggregate) {
                        inferred = parameter->second;
                    } else {
                        const std::string base =
                            first_register(instruction.operands[1]);
                        const auto base_type = register_types.find(base);
                        if (base_type != register_types.end() &&
                            base_type->second.is_pointer()) {
                            inferred = base_type->second;
                        }
                    }
                } else if (root == "cvta") {
                    const AddressSpace space =
                        instruction.opcode.find(".shared") != std::string::npos
                            ? AddressSpace::kThreadgroup
                        : instruction.opcode.find(".local") != std::string::npos
                            ? AddressSpace::kPrivate
                            : AddressSpace::kDevice;
                    inferred = Type::pointer(Type::integer(8), space);
                } else if (root == "selp" && instruction.operands.size() >= 3) {
                    for (std::size_t source_index : {1U, 2U}) {
                        const std::string source =
                            first_register(instruction.operands[source_index]);
                        const auto type = register_types.find(source);
                        if (type != register_types.end() && type->second.is_pointer()) {
                            inferred = type->second;
                            break;
                        }
                    }
                } else if ((root == "add" || root == "mov" || root == "mad") &&
                           instruction.operands.size() >= 2) {
                    const std::string source_symbol =
                        parameter_name_from_operand(instruction.operands[1]);
                    if (threadgroup_symbols.contains(source_symbol)) {
                        inferred = Type::pointer(Type::integer(8),
                                                 AddressSpace::kThreadgroup);
                    } else if (module_initialized_symbols.contains(source_symbol)) {
                        inferred = Type::pointer(Type::integer(8),
                                                 AddressSpace::kConstant);
                    } else if (std::any_of(
                                   module_global_symbols.begin(),
                                   module_global_symbols.end(),
                                   [&](const auto& global) {
                                       return global.name == source_symbol;
                                   })) {
                        inferred = Type::pointer(Type::integer(8),
                                                 AddressSpace::kDevice);
                    } else if (local_depots.contains(source_symbol)) {
                        inferred = Type::pointer(Type::integer(8),
                                                 AddressSpace::kPrivate);
                    }
                    for (const std::string& source : source_registers(instruction)) {
                        const auto type = register_types.find(source);
                        if (type != register_types.end() && type->second.is_pointer()) {
                            inferred = type->second;
                            break;
                        }
                    }
                }
                for (const std::string& destination : destinations) {
                    const auto existing = register_types.find(destination);
                    if (existing == register_types.end() || !(existing->second == inferred)) {
                        register_types[destination] = inferred;
                        changed = true;
                    }
                }
            }
        }
    }

    void allocate_values() {
        for (const std::string& name : implicit_definitions) {
            const ValueId value = builder.next_value();
            implicit_values[name] = value;
            const auto type = register_types.find(name);
            value_types[value] =
                type == register_types.end() ? Type::integer(32) : type->second;
        }
        for (RawBlock& block : raw_blocks) {
            std::unordered_set<std::string> locally_defined;
            for (const Instruction* instruction : block.instructions) {
                for (const std::string& source : source_registers(*instruction)) {
                    if (!locally_defined.contains(source)) {
                        block.uses_before_definition.insert(source);
                    }
                }
                std::vector<ValueId> values;
                for (const std::string& destination : destination_registers(*instruction)) {
                    const ValueId value = builder.next_value();
                    values.push_back(value);
                    locally_defined.insert(destination);
                    block.last_definitions[destination] = value;
                    const auto type = register_types.find(destination);
                    value_types[value] =
                        type == register_types.end() ? Type::integer(32) : type->second;
                }
                instruction_results[instruction] = std::move(values);
            }
        }
    }

    bool construct_ssa() {
        incoming.resize(raw_blocks.size());
        outgoing.resize(raw_blocks.size());
        block_arguments.resize(raw_blocks.size());

        // Determine which registers must enter each block before assigning SSA
        // values.  In particular, loop backedges cannot be discovered reliably
        // by growing incoming maps from an initially empty fixed point: a
        // speculative block argument can be created before a dominating value
        // reaches every predecessor and then remain behind as an invalid phi.
        std::vector<std::set<std::string>> live_in(raw_blocks.size());
        std::vector<std::set<std::string>> live_out(raw_blocks.size());
        bool liveness_changed = true;
        while (liveness_changed) {
            liveness_changed = false;
            for (std::size_t reverse = raw_blocks.size(); reverse > 0; --reverse) {
                const std::size_t block_index = reverse - 1;
                const RawBlock& block = raw_blocks[block_index];
                std::set<std::string> next_out;
                for (std::size_t successor : block.successors) {
                    next_out.insert(live_in[successor].begin(), live_in[successor].end());
                }
                std::set<std::string> next_in(block.uses_before_definition.begin(),
                                              block.uses_before_definition.end());
                for (const std::string& name : next_out) {
                    if (!block.last_definitions.contains(name)) {
                        next_in.insert(name);
                    }
                }
                if (next_in != live_in[block_index] || next_out != live_out[block_index]) {
                    live_in[block_index] = std::move(next_in);
                    live_out[block_index] = std::move(next_out);
                    liveness_changed = true;
                }
            }
        }

        // A live value entering a join needs a block argument.  Preallocating
        // these arguments also breaks loop-header cycles: the backedge can
        // immediately refer to the header argument while the preheader carries
        // the dominating definition.  Redundant arguments where all incoming
        // values happen to match are valid SSA and can be folded later.
        for (std::size_t block_index = 0; block_index < raw_blocks.size(); ++block_index) {
            if (raw_blocks[block_index].predecessors.size() < 2) continue;
            for (const std::string& name : live_in[block_index]) {
                const ValueId argument = builder.next_value();
                block_arguments[block_index][name] = argument;
                const auto type = register_types.find(name);
                value_types[argument] =
                    type == register_types.end() ? Type::integer(32) : type->second;
            }
        }

        bool changed = true;
        const std::size_t iteration_limit = std::max<std::size_t>(1, raw_blocks.size() + 1);
        for (std::size_t iteration = 0; iteration < iteration_limit && changed; ++iteration) {
            changed = false;
            for (std::size_t block_index = 0; block_index < raw_blocks.size(); ++block_index) {
                const RawBlock& block = raw_blocks[block_index];
                std::unordered_map<std::string, ValueId> next_in;
                if (block_index == 0) {
                    next_in = implicit_values;
                } else if (block.predecessors.size() >= 2) {
                    for (const auto& [name, argument] : block_arguments[block_index]) {
                        next_in[name] = argument;
                    }
                } else if (block.predecessors.size() == 1) {
                    const auto& predecessor_out = outgoing[block.predecessors.front()];
                    for (const std::string& name : live_in[block_index]) {
                        const auto value = predecessor_out.find(name);
                        if (value != predecessor_out.end()) {
                            next_in[name] = value->second;
                        }
                    }
                }

                std::unordered_map<std::string, ValueId> next_out = next_in;
                for (const auto& [name, value] : block.last_definitions) {
                    next_out[name] = value;
                }
                if (next_in != incoming[block_index] || next_out != outgoing[block_index]) {
                    incoming[block_index] = std::move(next_in);
                    outgoing[block_index] = std::move(next_out);
                    changed = true;
                }
            }
        }

        for (std::size_t block_index = 0; block_index < raw_blocks.size(); ++block_index) {
            for (const auto& [name, value] : block_arguments[block_index]) {
                for (std::size_t predecessor : raw_blocks[block_index].predecessors) {
                    if (!outgoing[predecessor].contains(name)) {
                        return fail(nullptr, "PTX register '" + name +
                                                 "' is undefined on an incoming edge to block '" +
                                                 raw_blocks[block_index].name + "'");
                    }
                }
                (void)value;
            }
            for (const std::string& name : raw_blocks[block_index].uses_before_definition) {
                if (!incoming[block_index].contains(name)) {
                    return fail(nullptr, "PTX register '" + name + "' is used before definition in block '" +
                                             raw_blocks[block_index].name + "'");
                }
            }
        }
        return true;
    }

    Operand operand_for(std::string_view token,
                        const std::unordered_map<std::string, ValueId>& environment,
                        const Type& fallback_type) {
        const std::string register_name = first_register(token);
        if (!register_name.empty()) {
            const auto value = environment.find(register_name);
            if (value != environment.end()) {
                return Operand::value_ref(value->second, value_types[value->second]);
            }
        }
        const std::string symbol = parameter_name_from_operand(token);
        if (const auto global = module_global_values.find(symbol);
            global != module_global_values.end()) {
            return global->second;
        }
        if (module_initialized_symbols.contains(symbol)) {
            return Operand::symbol(
                symbol,
                Type::pointer(Type::integer(8), AddressSpace::kConstant));
        }
        if (threadgroup_symbols.contains(symbol)) {
            return Operand::symbol(
                symbol, Type::pointer(Type::integer(8), AddressSpace::kThreadgroup));
        }
        if (const auto depot = local_depot_values.find(symbol);
            depot != local_depot_values.end()) {
            return depot->second;
        }
        const std::string spelling = trim(token);
        if (spelling.size() == 10 && starts_with(spelling, "0f")) {
            return Operand::immediate(spelling, Type::floating(32));
        }
        return Operand::immediate(spelling, fallback_type);
    }

    bool append_guard(Operation* operation, const Instruction& instruction,
                      const std::unordered_map<std::string, ValueId>& environment) {
        if (instruction.predicate.empty() || root_opcode(instruction.opcode) == "bra") {
            return true;
        }
        const auto [name, inverted] = normalized_predicate(instruction.predicate);
        const auto predicate = environment.find(name);
        if (predicate == environment.end()) {
            return fail(&instruction, "predicate register '" + name + "' is undefined");
        }
        operation->attributes["guard_operand"] = std::to_string(operation->operands.size());
        operation->attributes["guard_inverted"] = inverted ? "true" : "false";
        operation->operands.push_back(
            Operand::value_ref(predicate->second, value_types[predicate->second]));
        return true;
    }

    bool translate_instruction(Function* function, BasicBlock* block,
                               const Instruction& instruction,
                               std::unordered_map<std::string, ValueId>* environment) {
        const std::string root = root_opcode(instruction.opcode);
        if (root == "bra" || root == "ret" || root == "exit" || root == "trap") {
            return true;
        }
        if (printf_scaffold_lines.contains(instruction.line)) return true;
        if (!instruction.supported) {
            return fail(&instruction, "unsupported PTX opcode '" + instruction.opcode + "'");
        }

        Operation operation;
        operation.location = {.file = result.module.source_name,
                              .line = static_cast<std::uint32_t>(std::max(0, instruction.line))};
        operation.attributes["ptx_opcode"] = instruction.opcode;
        if (instruction.opcode.find(".f64") != std::string::npos) {
            operation.attributes["fp64_mode"] =
                result.module.attributes.at("fp64_mode");
            result.module.semantic_quality = SemanticQuality::kSemanticEmulation;
            const std::string caveat =
                "FP64 uses raw binary64 storage with the selected software ALU mode";
            if (std::find(result.module.semantic_caveats.begin(),
                          result.module.semantic_caveats.end(), caveat) ==
                result.module.semantic_caveats.end()) {
                result.module.semantic_caveats.push_back(caveat);
            }
        }
        operation.results = instruction_results[&instruction];
        for (ValueId value : operation.results) {
            operation.result_types.push_back(value_types[value]);
        }
        if (instruction.opcode.find(".f64") != std::string::npos) {
            for (std::size_t i = 0; i < operation.results.size(); ++i) {
                operation.result_types[i] = Type::floating(64);
                value_types[operation.results[i]] = Type::floating(64);
            }
        }

        const std::vector<std::string> destinations = destination_registers(instruction);
        const auto source_operand = [&](std::size_t index, const Type& fallback = Type::integer(32)) {
            return index < instruction.operands.size()
                       ? operand_for(instruction.operands[index], *environment, fallback)
                       : Operand::immediate("0", fallback);
        };
        const auto bit_container_operand = [&](std::size_t index, const Type& expected) {
            Operand operand = source_operand(index, expected);
            const bool same_width = operand.type.bit_width == expected.bit_width;
            const bool float_integer_pair =
                (operand.type.kind == TypeKind::kFloat &&
                 expected.kind == TypeKind::kInteger) ||
                (operand.type.kind == TypeKind::kInteger &&
                 expected.kind == TypeKind::kFloat);
            if (!same_width || !float_integer_pair) return operand;
            Operation conversion;
            conversion.opcode = OpCode::kConvert;
            conversion.location = operation.location;
            conversion.operands.push_back(operand);
            const ValueId converted = builder.next_value();
            conversion.results.push_back(converted);
            conversion.result_types.push_back(expected);
            conversion.attributes["bitcast"] = "true";
            value_types[converted] = expected;
            block->operations.push_back(std::move(conversion));
            return Operand::value_ref(converted, expected);
        };
        const auto memory_address_operand = [&](std::size_t index,
                                                const Type& fallback_pointer) {
            Operand base = source_operand(index, fallback_pointer);
            if (index >= instruction.operands.size()) return base;
            if (base.type.is_pointer() && fallback_pointer.is_pointer() &&
                base.type.address_space != AddressSpace::kNone &&
                !(base.type == fallback_pointer)) {
                Type pointer_type = fallback_pointer;
                pointer_type.address_space = base.type.address_space;
                Operation cast;
                cast.opcode = OpCode::kAddressSpaceCast;
                cast.location = operation.location;
                cast.operands = {base};
                const ValueId pointer = builder.next_value();
                cast.results = {pointer};
                cast.result_types = {pointer_type};
                value_types[pointer] = pointer_type;
                block->operations.push_back(std::move(cast));
                base = Operand::value_ref(pointer, pointer_type);
            }
            const std::int64_t byte_offset =
                memory_operand_offset(instruction.operands[index]);
            if (byte_offset == 0) return base;
            if (!base.type.is_pointer()) {
                base.type = fallback_pointer;
            }
            Operation offset;
            offset.opcode = OpCode::kPointerOffset;
            offset.location = operation.location;
            offset.operands = {
                base,
                Operand::immediate(std::to_string(byte_offset), Type::integer(64)),
            };
            offset.attributes["offset_unit"] = "bytes";
            const ValueId pointer = builder.next_value();
            offset.results = {pointer};
            offset.result_types = {base.type};
            value_types[pointer] = base.type;
            block->operations.push_back(std::move(offset));
            return Operand::value_ref(pointer, base.type);
        };

        if (root == "mov" && destinations.size() == 1 &&
            instruction.operands.size() >= 2) {
            std::string parameter =
                parameter_name_from_operand(instruction.operands[1]);
            const std::string source_register =
                first_register(instruction.operands[1]);
            if (!source_register.empty()) {
                const auto source_value = environment->find(source_register);
                if (source_value != environment->end()) {
                    const auto alias = aggregate_parameter_addresses.find(
                        source_value->second);
                    if (alias != aggregate_parameter_addresses.end()) {
                        parameter = alias->second;
                    }
                }
            }
            const auto parameter_value = parameter_values.find(parameter);
            const auto parameter_type = parameter_types.find(parameter);
            if (parameter_value != parameter_values.end() &&
                parameter_type != parameter_types.end() &&
                parameter_type->second.kind == TypeKind::kAggregate) {
                if (!instruction.predicate.empty()) {
                    return fail(&instruction,
                                "predicated aggregate parameter address moves are unsupported");
                }
                const bool used_as_local_address = std::any_of(
                    entry->instructions.begin(), entry->instructions.end(),
                    [&](const Instruction& candidate) {
                        const std::string candidate_root =
                            root_opcode(candidate.opcode);
                        if ((candidate_root != "ld" && candidate_root != "st") ||
                            candidate.opcode.find(".local") == std::string::npos) {
                            return false;
                        }
                        const std::size_t memory_index =
                            candidate_root == "st" ? 0 : 1;
                        return candidate.operands.size() > memory_index &&
                               first_register(candidate.operands[memory_index]) ==
                                   destinations.front();
                    });
                // Most aggregate parameter-address idioms only feed ld.param;
                // those retain the aggregate SSA value so CFG block arguments
                // keep their established type. Clang uses ld/st.local when a
                // by-value parameter is mutated, which requires an addressable
                // private copy instead.
                if (!used_as_local_address) {
                    operation.opcode = OpCode::kParameter;
                    operation.operands = {Operand::value_ref(
                        parameter_value->second, parameter_type->second)};
                    operation.result_types = {parameter_type->second};
                    value_types[operation.results.front()] = parameter_type->second;
                    aggregate_parameter_addresses[operation.results.front()] =
                        parameter;
                    aggregate_parameter_registers[destinations.front()] = parameter;
                    block->operations.push_back(std::move(operation));
                    (*environment)[destinations.front()] =
                        instruction_results[&instruction].front();
                    return true;
                }

                const Type pointer_type = Type::pointer(
                    parameter_type->second, AddressSpace::kPrivate);
                operation.opcode = OpCode::kAlloca;
                operation.result_types = {pointer_type};
                value_types[operation.results.front()] = pointer_type;
                std::size_t alignment = 1;
                for (const auto& candidate : entry->params) {
                    if (candidate.name == parameter) {
                        alignment = candidate.alignment;
                        break;
                    }
                }
                operation.attributes["alignment"] = std::to_string(alignment);
                const ValueId address = operation.results.front();
                block->operations.push_back(std::move(operation));

                Operation initialize;
                initialize.opcode = OpCode::kStore;
                initialize.operands = {
                    Operand::value_ref(address, pointer_type),
                    Operand::value_ref(parameter_value->second,
                                       parameter_type->second),
                };
                initialize.attributes["alignment"] = std::to_string(alignment);
                initialize.location = {
                    .file = result.module.source_name,
                    .line = static_cast<std::uint32_t>(std::max(0, instruction.line)),
                };
                block->operations.push_back(std::move(initialize));
                function->pointer_provenance[address] = {
                    .base_kind = PointerBaseKind::kAllocation,
                    .base_name = parameter,
                    .known_byte_offset = 0,
                    .alignment = static_cast<std::uint32_t>(alignment),
                };
                aggregate_parameter_addresses[address] = parameter;
                aggregate_parameter_registers[destinations.front()] = parameter;
                (*environment)[destinations.front()] = address;
                return true;
            }
            if (parameter_value != parameter_values.end() &&
                parameter_type != parameter_types.end() &&
                parameter_type->second.is_pointer()) {
                if (!instruction.predicate.empty()) {
                    return fail(&instruction,
                                "predicated pointer parameter moves are unsupported");
                }
                // Clang represents the address of a pointer-valued PTX
                // parameter with `mov`, then may select between those addresses
                // before an indirect `ld.param`. CuMetal arguments already hold
                // the loaded pointer value, so retain the SSA relationship to
                // the argument rather than lowering the parameter name to a
                // disconnected symbolic pointer.
                operation.opcode = OpCode::kParameter;
                operation.operands = {Operand::value_ref(
                    parameter_value->second, parameter_type->second)};
                operation.result_types = {parameter_type->second};
                operation.attributes["parameter"] = parameter;
                value_types[operation.results.front()] = parameter_type->second;
                const auto provenance =
                    function->pointer_provenance.find(parameter_value->second);
                if (provenance != function->pointer_provenance.end()) {
                    function->pointer_provenance[operation.results.front()] =
                        provenance->second;
                }
                block->operations.push_back(std::move(operation));
                (*environment)[destinations.front()] =
                    instruction_results[&instruction].front();
                return true;
            }
        }

        if (root == "mov" && instruction.opcode.find(".b64") != std::string::npos &&
            destinations.size() == 2 && instruction.operands.size() >= 2) {
            if (!instruction.predicate.empty()) {
                return fail(&instruction, "predicated b64 tuple unpack is unsupported");
            }
            const Operand packed = source_operand(1, Type::integer(64));
            operation.opcode = OpCode::kConvert;
            operation.results = {instruction_results[&instruction][0]};
            operation.result_types = {Type::integer(32)};
            operation.operands = {packed};
            value_types[operation.results.front()] = Type::integer(32);
            block->operations.push_back(std::move(operation));

            Operation shift;
            shift.opcode = OpCode::kShiftRight;
            shift.location = {.file = result.module.source_name,
                              .line = static_cast<std::uint32_t>(
                                  std::max(0, instruction.line))};
            const ValueId shifted = builder.next_value();
            shift.results = {shifted};
            shift.result_types = {Type::integer(64)};
            shift.operands = {packed, Operand::immediate("32", Type::integer(64))};
            value_types[shifted] = Type::integer(64);
            block->operations.push_back(std::move(shift));

            Operation high;
            high.opcode = OpCode::kConvert;
            high.location = {.file = result.module.source_name,
                             .line = static_cast<std::uint32_t>(
                                 std::max(0, instruction.line))};
            high.results = {instruction_results[&instruction][1]};
            high.result_types = {Type::integer(32)};
            high.operands = {Operand::value_ref(shifted, Type::integer(64))};
            value_types[high.results.front()] = Type::integer(32);
            block->operations.push_back(std::move(high));
            (*environment)[destinations[0]] = instruction_results[&instruction][0];
            (*environment)[destinations[1]] = instruction_results[&instruction][1];
            return true;
        }
        if (root == "mov" && instruction.opcode.find(".b64") != std::string::npos &&
            destinations.size() == 1 && instruction.operands.size() >= 2 &&
            instruction.operands.front().find('{') != std::string::npos &&
            instruction.operands.front().find(',') != std::string::npos) {
            // Clang uses an anonymous inline-assembly temporary for one half of
            // a tuple, for example `{tmp, %r7}`. The parser intentionally only
            // assigns SSA values to `%` registers, so recover whether the sole
            // named destination is the low or high 32-bit half here.
            if (!instruction.predicate.empty()) {
                return fail(&instruction, "predicated partial b64 tuple unpack is unsupported");
            }
            const std::string& tuple = instruction.operands.front();
            const std::size_t comma = tuple.find(',');
            const bool named_high = tuple.find('%') > comma;
            const Operand packed = source_operand(1, Type::integer(64));
            Operand selected = packed;
            if (named_high) {
                Operation shift;
                shift.opcode = OpCode::kShiftRight;
                shift.location = operation.location;
                shift.operands = {packed,
                                  Operand::immediate("32", Type::integer(64))};
                const ValueId shifted = builder.next_value();
                shift.results = {shifted};
                shift.result_types = {Type::integer(64)};
                value_types[shifted] = Type::integer(64);
                block->operations.push_back(std::move(shift));
                selected = Operand::value_ref(shifted, Type::integer(64));
            }
            operation.opcode = OpCode::kConvert;
            operation.result_types = {Type::integer(32)};
            operation.operands = {selected};
            value_types[operation.results.front()] = Type::integer(32);
            block->operations.push_back(std::move(operation));
            (*environment)[destinations.front()] =
                instruction_results[&instruction].front();
            return true;
        }
        if (root == "mov" && instruction.opcode.find(".b64") != std::string::npos &&
            destinations.size() == 1 && instruction.operands.size() >= 2) {
            const std::vector<std::string> parts = registers_in(instruction.operands[1]);
            if (parts.size() == 2) {
                if (!instruction.predicate.empty()) {
                    return fail(&instruction, "predicated b64 tuple pack is unsupported");
                }
                const Operand low = operand_for(parts[0], *environment, Type::integer(32));
                const Operand high = operand_for(parts[1], *environment, Type::integer(32));
                const auto widen = [&](const Operand& input) {
                    Operation conversion;
                    conversion.opcode = OpCode::kConvert;
                    conversion.location = operation.location;
                    conversion.operands = {input};
                    const ValueId value = builder.next_value();
                    conversion.results = {value};
                    conversion.result_types = {Type::integer(64)};
                    value_types[value] = Type::integer(64);
                    block->operations.push_back(std::move(conversion));
                    return Operand::value_ref(value, Type::integer(64));
                };
                const Operand low64 = widen(low);
                const Operand high64 = widen(high);
                Operation shift;
                shift.opcode = OpCode::kShiftLeft;
                shift.location = operation.location;
                shift.operands = {high64,
                                  Operand::immediate("32", Type::integer(64))};
                const ValueId shifted = builder.next_value();
                shift.results = {shifted};
                shift.result_types = {Type::integer(64)};
                value_types[shifted] = Type::integer(64);
                block->operations.push_back(std::move(shift));
                operation.opcode = OpCode::kBitOr;
                operation.result_types = {Type::integer(64)};
                value_types[operation.results.front()] = Type::integer(64);
                operation.operands = {
                    low64, Operand::value_ref(shifted, Type::integer(64))};
                block->operations.push_back(std::move(operation));
                (*environment)[destinations[0]] = instruction_results[&instruction][0];
                return true;
            }
        }

        if (starts_with(instruction.opcode, "st.param")) {
            if (instruction.operands.size() < 2) {
                return fail(&instruction, "malformed st.param instruction");
            }
            const std::string name = parameter_name_from_operand(instruction.operands[0]);
            if (name.empty()) return fail(&instruction, "call parameter slot has no name");
            const Operand stored =
                source_operand(1, ptx_scalar_type(instruction.opcode));
            const std::int64_t byte_offset =
                memory_operand_offset(instruction.operands[0]);
            if (byte_offset < 0) {
                return fail(&instruction,
                            "PTX call parameter slot has a negative byte offset");
            }
            const bool is_return_slot = std::any_of(
                entry->return_params.begin(), entry->return_params.end(),
                [&](const cumetal::ptx::Parameter& parameter) {
                    return parameter.name == name;
                });
            if (is_return_slot) {
                if (entry->return_params.size() != 1) {
                    return fail(&instruction,
                                "typed PTX device calls support at most one return value");
                }
                const Type return_type =
                    parameter_type(entry->return_params.front());
                if (return_type.kind == TypeKind::kAggregate) {
                    function_return_fields[byte_offset] = stored;
                } else {
                    if (byte_offset != 0) {
                        return fail(&instruction,
                                    "scalar PTX device return has a nonzero byte offset");
                    }
                    function_return = stored;
                }
            } else {
                call_parameter_slot_fields[name][byte_offset] = stored;
                if (byte_offset == 0) call_parameter_slots[name] = stored;
            }
            return true;
        } else if (starts_with(instruction.opcode, "ld.param")) {
            if (instruction.operands.size() < 2 || operation.results.empty()) {
                return fail(&instruction, "malformed ld.param instruction");
            }
            std::string name = parameter_name_from_operand(instruction.operands[1]);
            const std::string base_register =
                first_register(instruction.operands[1]);
            if (!base_register.empty()) {
                const auto base_value = environment->find(base_register);
                if (base_value != environment->end()) {
                    const auto parameter_address =
                        aggregate_parameter_addresses.find(base_value->second);
                    if (parameter_address != aggregate_parameter_addresses.end()) {
                        name = parameter_address->second;
                    }
                }
                const auto symbolic_address =
                    aggregate_parameter_registers.find(base_register);
                if (symbolic_address != aggregate_parameter_registers.end()) {
                    name = symbolic_address->second;
                }
            }
            const auto argument = parameter_values.find(name);
            if (argument == parameter_values.end()) {
                const auto returned = call_return_slots.find(name);
                if (returned == call_return_slots.end()) {
                    const auto indirect = environment->find(base_register);
                    if (indirect == environment->end() ||
                        !value_types[indirect->second].is_pointer()) {
                        return fail(&instruction,
                                    "unknown kernel or call parameter '" + name + "'");
                    }
                    // Clang may select between addresses of pointer-valued PTX
                    // parameters and then load through the selected param-space
                    // address. Parameter operands already denote their loaded
                    // SSA values in CuMetal IR, so the selected value is the
                    // pointer itself and this ld.param is an exact typed copy.
                    operation.opcode = OpCode::kConvert;
                    operation.operands.push_back(Operand::value_ref(
                        indirect->second, value_types[indirect->second]));
                    operation.result_types.front() = value_types[indirect->second];
                    value_types[operation.results.front()] =
                        value_types[indirect->second];
                    const auto provenance =
                        function->pointer_provenance.find(indirect->second);
                    if (provenance != function->pointer_provenance.end()) {
                        function->pointer_provenance[operation.results.front()] =
                            provenance->second;
                    }
                } else {
                    if (returned->second.type.kind == TypeKind::kAggregate) {
                        const Type& aggregate_type = returned->second.type;
                        const std::int64_t byte_offset =
                            memory_operand_offset(instruction.operands[1]);
                        const std::uint32_t loaded_size =
                            type_size(operation.result_types.front());
                        if (byte_offset < 0 || loaded_size == 0 ||
                            aggregate_type.elements.empty() ||
                            type_size(aggregate_type.elements.front()) != loaded_size ||
                            byte_offset % loaded_size != 0 ||
                            static_cast<std::uint64_t>(byte_offset / loaded_size) >=
                                aggregate_type.elements.size()) {
                            return fail(
                                &instruction,
                                "aggregate PTX call return load is not an aligned field");
                        }
                        operation.opcode = OpCode::kAggregateExtract;
                        operation.operands.push_back(returned->second);
                        operation.operands.push_back(Operand::immediate(
                            std::to_string(byte_offset / loaded_size),
                            Type::integer(32)));
                    } else {
                        operation.opcode = OpCode::kConvert;
                        operation.operands.push_back(returned->second);
                        if (starts_with(instruction.opcode, "ld.param.b64") &&
                            returned->second.type == Type::floating(32)) {
                            operation.result_types.front() = Type::floating(32);
                            value_types[operation.results.front()] =
                                Type::floating(32);
                        } else if (!(returned->second.type ==
                                     operation.result_types.front())) {
                            operation.attributes["bitcast"] = "true";
                        }
                    }
                }
            } else {
                const Type& argument_type = parameter_types[name];
                if (argument_type.kind == TypeKind::kAggregate) {
                    const std::int64_t byte_offset =
                        memory_operand_offset(instruction.operands[1]);
                    const std::uint32_t loaded_size =
                        type_size(operation.result_types.front());
                    if (byte_offset < 0 || loaded_size == 0 ||
                        argument_type.elements.empty() ||
                        type_size(argument_type.elements.front()) != loaded_size ||
                        byte_offset % loaded_size != 0 ||
                        static_cast<std::uint64_t>(byte_offset / loaded_size) >=
                            argument_type.elements.size()) {
                        return fail(&instruction,
                                    "aggregate PTX parameter load is not an aligned field");
                    }
                    operation.opcode = OpCode::kAggregateExtract;
                    operation.operands.push_back(
                        Operand::value_ref(argument->second, argument_type));
                    operation.operands.push_back(Operand::immediate(
                        std::to_string(byte_offset / loaded_size),
                        Type::integer(32)));
                } else {
                    operation.opcode = OpCode::kParameter;
                    operation.operands.push_back(
                        Operand::value_ref(argument->second, argument_type));
                    operation.attributes["parameter"] = name;
                }
                if (operation.result_types.front().is_pointer()) {
                    function->pointer_provenance[operation.results.front()] =
                        function->pointer_provenance[argument->second];
                }
            }
        } else if (root == "mov" && instruction.operands.size() >= 2 &&
                   instruction.operands[1].find("%tid.") != std::string::npos) {
            operation.opcode = OpCode::kThreadId;
            operation.attributes["dimension"] =
                instruction.operands[1].find(".y") != std::string::npos
                    ? "y"
                    : (instruction.operands[1].find(".z") != std::string::npos ? "z" : "x");
        } else if (root == "mov" && instruction.operands.size() >= 2 &&
                   instruction.operands[1].find("%ctaid.") != std::string::npos) {
            operation.opcode = OpCode::kThreadgroupId;
            operation.attributes["dimension"] =
                instruction.operands[1].find(".y") != std::string::npos
                    ? "y"
                    : (instruction.operands[1].find(".z") != std::string::npos ? "z" : "x");
        } else if (root == "mov" && instruction.operands.size() >= 2 &&
                   instruction.operands[1].find("%ntid.") != std::string::npos) {
            operation.opcode = OpCode::kThreadgroupSize;
            operation.attributes["dimension"] =
                instruction.operands[1].find(".y") != std::string::npos
                    ? "y"
                    : (instruction.operands[1].find(".z") != std::string::npos ? "z" : "x");
        } else if (root == "mov" && instruction.operands.size() >= 2 &&
                   instruction.operands[1].find("%nctaid.") != std::string::npos) {
            operation.opcode = OpCode::kGridSize;
            operation.attributes["dimension"] =
                instruction.operands[1].find(".y") != std::string::npos
                    ? "y"
                    : (instruction.operands[1].find(".z") != std::string::npos ? "z" : "x");
        } else if (root == "mov" && instruction.operands.size() >= 2 &&
                   instruction.operands[1].find("%laneid") != std::string::npos) {
            operation.opcode = OpCode::kLaneId;
        } else if (root == "mov" && instruction.operands.size() >= 2 &&
                   instruction.operands[1].find("%activemask") != std::string::npos) {
            operation.opcode = OpCode::kBallot;
            operation.attributes["kind"] = "active_mask";
        } else if (root == "mov") {
            operation.opcode = OpCode::kConvert;
            operation.operands.push_back(
                bit_container_operand(1, operation.result_types.front()));
            if (trim(instruction.operands[1]) == "0") {
                integer_zero_values.insert(operation.results.begin(), operation.results.end());
            }
        } else if (root == "cvta") {
            operation.opcode = OpCode::kAddressSpaceCast;
            Operand source = source_operand(1, operation.result_types.front());
            if (source.kind == OperandKind::kValue &&
                integer_zero_values.contains(source.value)) {
                // Clang commonly spells a null generic address as `mov.b64 0`
                // followed by `cvta.to.global`. Preserve that provenance instead
                // of presenting an untyped integer to the address-space cast.
                source = Operand::immediate("null", operation.result_types.front());
            }
            operation.operands.push_back(std::move(source));
            if (!operation.results.empty() && operation.operands.front().kind == OperandKind::kValue) {
                const auto provenance =
                    function->pointer_provenance.find(operation.operands.front().value);
                if (provenance != function->pointer_provenance.end()) {
                    function->pointer_provenance[operation.results.front()] = provenance->second;
                }
            }
        } else if (root == "ld") {
            operation.opcode = OpCode::kLoad;
            if (instruction.operands.size() < 2) return fail(&instruction, "malformed load");
            const std::string referenced_symbol =
                parameter_name_from_operand(instruction.operands[1]);
            const auto module_constant = module_constant_symbols.find(referenced_symbol);
            if (module_constant != module_constant_symbols.end()) {
                if (!module_constant_buffer.has_value()) {
                    return fail(&instruction, "module constant buffer is unavailable");
                }
                const std::int64_t byte_offset =
                    static_cast<std::int64_t>(module_constant->second.offset) +
                    memory_operand_offset(instruction.operands[1]);
                if (byte_offset == 0) {
                    operation.operands.push_back(*module_constant_buffer);
                } else {
                    Operation offset;
                    offset.opcode = OpCode::kPointerOffset;
                    offset.location = operation.location;
                    offset.operands = {
                        *module_constant_buffer,
                        Operand::immediate(std::to_string(byte_offset), Type::integer(64)),
                    };
                    const ValueId pointer = builder.next_value();
                    const Type pointer_type = Type::pointer(
                        Type::integer(8), AddressSpace::kConstant);
                    offset.results = {pointer};
                    offset.result_types = {pointer_type};
                    value_types[pointer] = pointer_type;
                    block->operations.push_back(std::move(offset));
                    operation.operands.push_back(Operand::value_ref(pointer, pointer_type));
                }
            } else {
                const AddressSpace load_address_space =
                    instruction.opcode.find(".shared") != std::string::npos
                        ? AddressSpace::kThreadgroup
                    : instruction.opcode.find(".local") != std::string::npos
                        ? AddressSpace::kPrivate
                    : instruction.opcode.find(".const") != std::string::npos
                        ? AddressSpace::kConstant
                        : AddressSpace::kDevice;
                operation.operands.push_back(memory_address_operand(
                    1, Type::pointer(operation.result_types.front(),
                                     load_address_space)));
            }
            operation.attributes["address"] = instruction.operands[1];
            const Type memory_type = ptx_scalar_type(instruction.opcode);
            operation.attributes["memory_bit_width"] =
                std::to_string(memory_type.bit_width);
            if (has_signed_integer_type(instruction.opcode)) {
                operation.attributes["signed"] = "true";
            }
            operation.attributes["alignment"] =
                std::to_string(type_size(memory_type));
        } else if (root == "st") {
            operation.opcode = OpCode::kStore;
            if (instruction.operands.size() < 2) return fail(&instruction, "malformed store");
            const AddressSpace store_address_space =
                instruction.opcode.find(".shared") != std::string::npos
                    ? AddressSpace::kThreadgroup
                : instruction.opcode.find(".local") != std::string::npos
                    ? AddressSpace::kPrivate
                    : instruction.opcode.find(".const") != std::string::npos
                    ? AddressSpace::kConstant
                    : AddressSpace::kDevice;
            operation.operands.push_back(memory_address_operand(
                0, Type::pointer(ptx_scalar_type(instruction.opcode),
                                 store_address_space)));
            operation.operands.push_back(
                bit_container_operand(1, ptx_scalar_type(instruction.opcode)));
            operation.attributes["address"] = instruction.operands[0];
            operation.attributes["alignment"] =
                std::to_string(type_size(ptx_scalar_type(instruction.opcode)));
        } else if (root == "setp") {
            operation.opcode = OpCode::kCompare;
            operation.operands.push_back(
                bit_container_operand(1, ptx_scalar_type(instruction.opcode)));
            operation.operands.push_back(
                bit_container_operand(2, ptx_scalar_type(instruction.opcode)));
            operation.attributes["predicate"] = comparison_predicate(instruction.opcode);
            if (has_signed_integer_type(instruction.opcode)) {
                operation.attributes["signed"] = "true";
            }
        } else if (root == "selp") {
            operation.opcode = OpCode::kSelect;
            operation.operands.push_back(source_operand(3, Type::predicate()));
            // PTX keeps float temporaries in .b32 registers, so `selp.f32` reads
            // two integer-typed values and produces a float one. Taking them
            // through source_operand assigned the raw bit pattern to a float
            // result, which Metal then reads as a numeric conversion:
            // selp.f32 over the bits of 4.0f yielded 1082130432.0. Go through
            // the bit container so the mismatch becomes a bitcast, as every
            // other arithmetic form already does. Pointer selects are unchanged;
            // the helper only rewrites a same-width float/integer pair.
            operation.operands.push_back(
                bit_container_operand(1, operation.result_types.front()));
            operation.operands.push_back(
                bit_container_operand(2, operation.result_types.front()));
        } else if (root == "bar") {
            if (!instruction.predicate.empty()) {
                operation.attributes["predicate"] = instruction.predicate;
            }
            operation.opcode = OpCode::kBarrier;
            operation.memory_scope = MemoryScope::kThreadgroup;
        } else if (root == "membar" || root == "fence") {
            operation.opcode = OpCode::kFence;
            operation.memory_scope = memory_scope_from_opcode(instruction.opcode);
            operation.memory_ordering = memory_ordering_from_opcode(instruction.opcode);
            operation.attributes["cuda_membar"] = "true";
            if (operation.memory_scope == MemoryScope::kSystem) {
                operation.attributes["metal_uma_system_scope"] = "true";
            }
        } else if (root == "atom") {
            operation.opcode = OpCode::kAtomic;
            operation.memory_scope = memory_scope_from_opcode(instruction.opcode);
            operation.memory_ordering = memory_ordering_from_opcode(instruction.opcode);
            const std::string atomic_operation =
                atomic_operation_from_opcode(instruction.opcode);
            if (atomic_operation.empty()) {
                return fail(&instruction, "unsupported PTX atomic operation '" +
                                              instruction.opcode + "'");
            }
            operation.attributes["atomic_op"] = atomic_operation;
            if (has_signed_integer_type(instruction.opcode)) {
                operation.attributes["signed"] = "true";
            }
            // CUDA Clang 23 emits relaxed `.sys` atomics for the ordinary
            // source-level atomic family. On Apple Silicon, tracked CUDA
            // allocations use Metal shared storage, so this form has an
            // explicit coherent-UMA lowering policy rather than being silently
            // weakened by generic legalization.
            if (operation.memory_scope == MemoryScope::kSystem) {
                operation.attributes["metal_uma_system_scope"] = "true";
            }
            if (operation.memory_ordering == MemoryOrdering::kAcquire &&
                !block->operations.empty()) {
                const Operation& previous = block->operations.back();
                if (previous.opcode == OpCode::kFence &&
                    previous.attributes.contains("cuda_membar") &&
                    previous.attributes.at("cuda_membar") == "true" &&
                    previous.memory_ordering ==
                        MemoryOrdering::kSequentiallyConsistent) {
                    // CUDA Clang 21 spells legacy source atomics as a
                    // seq_cst system fence followed immediately by an acquire
                    // CAS. Metal atomics themselves are relaxed-only; retain
                    // the explicit fence and normalize only this proven pair.
                    operation.memory_ordering = MemoryOrdering::kRelaxed;
                    operation.attributes["cuda_fenced_acquire_atomic"] = "true";
                }
            }
            if (instruction.operands.size() < 3) {
                return fail(&instruction, "malformed PTX atomic instruction");
            }
            const AddressSpace atomic_address_space =
                instruction.opcode.find(".shared.") != std::string::npos
                    ? AddressSpace::kThreadgroup
                    : AddressSpace::kDevice;
            operation.operands.push_back(memory_address_operand(
                1, Type::pointer(ptx_scalar_type(instruction.opcode),
                                 atomic_address_space)));
            for (std::size_t i = 2; i < instruction.operands.size(); ++i) {
                operation.operands.push_back(
                    source_operand(i, ptx_scalar_type(instruction.opcode)));
            }
        } else if (root == "shfl") {
            operation.opcode = OpCode::kShuffle;
            if (instruction.opcode.find(".down.") != std::string::npos) {
                operation.attributes["kind"] = "down";
            } else if (instruction.opcode.find(".up.") != std::string::npos) {
                operation.attributes["kind"] = "up";
            } else if (instruction.opcode.find(".bfly.") != std::string::npos) {
                operation.attributes["kind"] = "xor";
            } else {
                operation.attributes["kind"] = "index";
            }
            if (instruction.operands.size() < 4) {
                return fail(&instruction, "malformed PTX shuffle instruction");
            }
            // PTX shfl is a bit-container operation. A float-valued register
            // named as the `.b32` source must be bitcast before Metal's typed
            // simd_shuffle; numeric float-to-uint conversion followed by an
            // as_type<float> turns ordinary values into denormals and makes
            // warp reductions appear to do nothing.
            operation.operands.push_back(
                bit_container_operand(1, ptx_scalar_type(instruction.opcode)));
            for (std::size_t i = 2; i < instruction.operands.size(); ++i) {
                operation.operands.push_back(
                    source_operand(i, Type::integer(32)));
            }
        } else if (root == "vote") {
            const bool ballot =
                instruction.opcode.find(".ballot.") != std::string::npos;
            const bool sync =
                instruction.opcode.find(".sync.") != std::string::npos;
            if (instruction.operands.size() < 2 ||
                (sync && instruction.operands.size() < 3)) {
                return fail(&instruction, "malformed PTX vote instruction");
            }
            if (instruction.opcode.find(".uni.") != std::string::npos) {
                return fail(&instruction,
                            "vote.uni is not supported by the typed Metal backend");
            }
            operation.opcode = ballot ? OpCode::kBallot : OpCode::kVote;
            operation.attributes["kind"] =
                ballot ? "ballot"
                       : (instruction.opcode.find(".all.") != std::string::npos
                              ? "all"
                              : "any");
            // The canonical GPU IR operand order is member-mask, predicate,
            // matching LLVM's nvvm.vote.*.sync intrinsics. PTX spells these as
            // destination, predicate, member-mask, so reorder rather than
            // importing the 32-bit mask as a predicate.
            operation.operands.push_back(
                sync ? source_operand(2, Type::integer(32))
                     : Operand::immediate("4294967295", Type::integer(32)));
            operation.operands.push_back(source_operand(1, Type::predicate()));
        } else if (root == "redux") {
            operation.opcode = OpCode::kReduction;
            for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                operation.operands.push_back(source_operand(i, ptx_scalar_type(instruction.opcode)));
            }
        } else if (root == "cvt") {
            operation.opcode = OpCode::kConvert;
            operation.result_types.front() = ptx_cvt_result_type(instruction.opcode);
            value_types[operation.results.front()] = operation.result_types.front();
            if (cvt_has_signed_source(instruction.opcode)) {
                operation.attributes["signed_input"] = "true";
            }
            if (instruction.opcode.find(".f32.f32") != std::string::npos &&
                (instruction.opcode.find(".rni.") != std::string::npos ||
                 instruction.opcode.find(".rmi.") != std::string::npos ||
                 instruction.opcode.find(".rpi.") != std::string::npos ||
                 instruction.opcode.find(".rzi.") != std::string::npos)) {
                operation.opcode = OpCode::kCall;
                operation.result_types.front() = Type::floating(32);
                value_types[operation.results.front()] = Type::floating(32);
                operation.operands.push_back(
                    bit_container_operand(1, Type::floating(32)));
                operation.attributes["builtin"] = "true";
                operation.attributes["callee"] =
                    instruction.opcode.find(".rni.") != std::string::npos
                        ? "rint"
                    : instruction.opcode.find(".rmi.") != std::string::npos
                        ? "floor"
                    : instruction.opcode.find(".rpi.") != std::string::npos
                        ? "ceil"
                        : "trunc";
            } else if (instruction.opcode.find(".rni.f64.f64") != std::string::npos) {
                operation.result_types.front() = Type::floating(64);
                value_types[operation.results.front()] = Type::floating(64);
                operation.operands.push_back(
                    bit_container_operand(1, Type::floating(64)));
                operation.attributes["fp64_conversion"] = "round_int";
                operation.attributes["rounding_mode"] = "0u";
            } else if (instruction.opcode.find(".f64.f32") != std::string::npos) {
                operation.result_types.front() = Type::floating(64);
                value_types[operation.results.front()] = Type::floating(64);
                operation.operands.push_back(
                    bit_container_operand(1, Type::floating(32)));
                operation.attributes["fp64_conversion"] = "f32_to_f64";
            } else if (instruction.opcode.find(".f32.f64") != std::string::npos) {
                operation.result_types.front() = Type::floating(32);
                value_types[operation.results.front()] = Type::floating(32);
                operation.operands.push_back(
                    bit_container_operand(1, Type::floating(64)));
                operation.attributes["fp64_conversion"] = "f64_to_f32";
            } else {
                operation.operands.push_back(
                    bit_container_operand(1, ptx_cvt_source_type(instruction.opcode)));
            }
        } else if (root == "rcp") {
            operation.opcode = OpCode::kDiv;
            operation.operands.push_back(
                Operand::immediate("1.0", operation.result_types.front()));
            operation.operands.push_back(
                bit_container_operand(1, operation.result_types.front()));
        } else if (root == "not") {
            const Type type = ptx_scalar_type(instruction.opcode);
            const bool predicate_not =
                type.kind == TypeKind::kPredicate ||
                (!destinations.empty() && starts_with(destinations.front(), "%p"));
            if (predicate_not) {
                operation.opcode = OpCode::kCompare;
                operation.operands.push_back(source_operand(1, Type::predicate()));
                operation.operands.push_back(
                    Operand::immediate("0", Type::predicate()));
                operation.attributes["predicate"] = "eq";
            } else {
                operation.opcode = OpCode::kBitXor;
                operation.operands.push_back(bit_container_operand(1, type));
                operation.operands.push_back(Operand::immediate(
                    type.bit_width == 64 ? "18446744073709551615" :
                    type.bit_width == 16 ? "65535" : "4294967295",
                    type));
            }
        } else if (root == "bfe") {
            if (instruction.operands.size() != 4 ||
                has_signed_integer_type(instruction.opcode)) {
                return fail(&instruction,
                            "typed PTX bfe currently requires an unsigned source");
            }
            const std::string position_spelling = trim(instruction.operands[2]);
            const std::string width_spelling = trim(instruction.operands[3]);
            const auto is_decimal = [](const std::string& spelling) {
                return !spelling.empty() &&
                       std::all_of(spelling.begin(), spelling.end(), [](char c) {
                           return c >= '0' && c <= '9';
                       });
            };
            if (!is_decimal(position_spelling) || !is_decimal(width_spelling)) {
                return fail(&instruction,
                            "typed PTX bfe requires immediate position and width");
            }
            const Type type = ptx_scalar_type(instruction.opcode);
            const std::uint32_t position = static_cast<std::uint32_t>(
                std::stoul(position_spelling));
            const std::uint32_t width = static_cast<std::uint32_t>(
                std::stoul(width_spelling));
            if (position >= type.bit_width || width > type.bit_width - position) {
                return fail(&instruction, "typed PTX bfe range exceeds its source width");
            }
            Operation shift;
            shift.opcode = OpCode::kShiftRight;
            shift.location = operation.location;
            shift.operands = {
                bit_container_operand(1, type),
                Operand::immediate(position_spelling, type),
            };
            const ValueId shifted = builder.next_value();
            shift.results = {shifted};
            shift.result_types = {type};
            value_types[shifted] = type;
            block->operations.push_back(std::move(shift));
            const std::uint64_t mask =
                width == 64 ? ~std::uint64_t{0}
                            : (width == 0 ? 0 : ((std::uint64_t{1} << width) - 1));
            operation.opcode = OpCode::kBitAnd;
            operation.operands = {
                Operand::value_ref(shifted, type),
                Operand::immediate(std::to_string(mask), type),
            };
        } else if (root == "abs" || root == "min" || root == "max") {
            const Type type = ptx_scalar_type(instruction.opcode);
            for (std::size_t i = 0; i < operation.results.size(); ++i) {
                operation.result_types[i] = type;
                value_types[operation.results[i]] = type;
            }
            operation.opcode = OpCode::kCall;
            operation.attributes["builtin"] = "true";
            operation.attributes["callee"] =
                root == "abs"
                    ? (type.kind == TypeKind::kFloat ? "fabs" : "__cumetal_signed_abs")
                    : (root == "min" ? "min" : "max");
            const std::size_t arity = root == "abs" ? 1 : 2;
            for (std::size_t i = 0; i < arity; ++i) {
                operation.operands.push_back(bit_container_operand(i + 1, type));
            }
        } else if (root == "call") {
            const bool has_return = instruction.operands.size() == 3;
            if ((!has_return && instruction.operands.size() != 2) ||
                (has_return && grouped_names(instruction.operands[0]).size() != 1)) {
                return fail(&instruction, "malformed or multi-result PTX call");
            }
            const std::size_t callee_index = has_return ? 1 : 0;
            const std::size_t arguments_index = has_return ? 2 : 1;
            const std::string callee = trim(instruction.operands[callee_index]);
            if (callee == "vprintf" || callee == "printf") {
                const auto decoded = printf_calls.find(instruction.line);
                if (decoded == printf_calls.end() || !printf_buffer.has_value() ||
                    !printf_capacity.has_value()) {
                    return fail(&instruction,
                                "typed printf is missing its decoded ring-buffer ABI");
                }
                operation.opcode = OpCode::kPrintf;
                operation.operands = {*printf_buffer, *printf_capacity};
                operation.attributes["format_id"] =
                    std::to_string(decoded->second.format_id);
                if (decoded->second.null_format) {
                    operation.attributes["null_format"] = "true";
                }
                std::ostringstream widths;
                for (std::size_t i = 0; i < decoded->second.arguments.size(); ++i) {
                    const int bits = decoded->second.argument_bits[i];
                    if (bits != 32 && bits != 64) {
                        return fail(&instruction,
                                    "typed printf argument is not 32 or 64 bits");
                    }
                    if (i != 0) widths << ',';
                    widths << bits;
                    operation.operands.push_back(operand_for(
                        decoded->second.arguments[i], *environment,
                        Type::integer(static_cast<std::uint32_t>(bits))));
                }
                operation.attributes["argument_bits"] = widths.str();
                if (has_return) {
                    const ValueId result_value = builder.next_value();
                    operation.results.push_back(result_value);
                    operation.result_types.push_back(Type::integer(32));
                    value_types[result_value] = Type::integer(32);
                    call_return_slots[grouped_names(instruction.operands[0]).front()] =
                        Operand::value_ref(result_value, Type::integer(32));
                }
                if (!append_guard(&operation, instruction, *environment)) return false;
                block->operations.push_back(std::move(operation));
                return true;
            }
            std::optional<BuiltinSignature> signature =
                cuda_builtin_signature(callee);
            const bool builtin_call = signature.has_value();
            if (!signature.has_value()) {
                const auto function = device_functions.find(callee);
                if (function == device_functions.end()) {
                    return fail(&instruction, "device call target '" + callee +
                                                  "' has no typed PTX definition");
                }
                if (function->second->return_params.size() > 1) {
                    return fail(&instruction, "device call target '" + callee +
                                                  "' has multiple return values");
                }
                std::vector<Type> argument_types;
                argument_types.reserve(function->second->params.size());
                const auto imported = std::find_if(
                    result.module.functions.begin(), result.module.functions.end(),
                    [&](const Function& candidate) {
                        return candidate.name == callee;
                    });
                for (std::size_t index = 0;
                     index < function->second->params.size(); ++index) {
                    if (imported != result.module.functions.end() &&
                        index < imported->arguments.size()) {
                        argument_types.push_back(imported->arguments[index].type);
                    } else {
                        argument_types.push_back(
                            parameter_type(function->second->params[index]));
                    }
                }
                signature = BuiltinSignature{
                    .metal_name = callee,
                    .return_type =
                        imported != result.module.functions.end()
                            ? imported->return_type
                            : function->second->return_params.empty()
                                  ? Type::void_type()
                                  : parameter_type(
                                        function->second->return_params.front()),
                    .argument_types = std::move(argument_types),
                };
            }
            const std::vector<std::string> argument_names =
                grouped_names(instruction.operands[arguments_index]);
            if (argument_names.size() != signature->argument_types.size()) {
                return fail(&instruction, "device call target '" + callee +
                                              "' received the wrong argument count");
            }
            operation.opcode = OpCode::kCall;
            operation.attributes["callee"] = signature->metal_name;
            if (builtin_call) operation.attributes["builtin"] = "true";
            if (signature->return_type == Type::floating(64)) {
                operation.attributes["fp64_mode"] =
                    result.module.attributes.at("fp64_mode");
            }
            for (std::size_t i = 0; i < argument_names.size(); ++i) {
                const std::string& argument_name = argument_names[i];
                Operand argument;
                if (signature->argument_types[i].kind == TypeKind::kAggregate) {
                    const auto fields =
                        call_parameter_slot_fields.find(argument_name);
                    if (fields == call_parameter_slot_fields.end()) {
                        return fail(&instruction,
                                    "aggregate call parameter slot '" +
                                        argument_name + "' was not initialized");
                    }
                    const std::optional<Operand> aggregate =
                        materialize_aggregate(
                            block, &instruction, signature->argument_types[i],
                            fields->second,
                            "aggregate call parameter slot '" + argument_name +
                                "'");
                    if (!aggregate.has_value()) return false;
                    argument = *aggregate;
                } else {
                    const auto slot = call_parameter_slots.find(argument_name);
                    if (slot == call_parameter_slots.end()) {
                        return fail(&instruction, "call parameter slot '" +
                                                      argument_name +
                                                      "' was not initialized");
                    }
                    argument = slot->second;
                }
                if (callee == "__nv_frexp" && i == 0 &&
                    argument.type == Type::floating(64) &&
                    signature->argument_types[i] == Type::floating(32) &&
                    argument.kind == OperandKind::kValue) {
                    // Clang widens float to the ABI's double slot before the
                    // call. This builtin is intentionally the proven float
                    // normalization: recover the exact f32 producer instead
                    // of bitcasting the now-correct 64-bit software value.
                    for (auto prior = block->operations.rbegin();
                         prior != block->operations.rend(); ++prior) {
                        if (prior->results.size() == 1 &&
                            prior->results.front() == argument.value &&
                            prior->attributes.contains("fp64_conversion") &&
                            prior->attributes.at("fp64_conversion") ==
                                "f32_to_f64" &&
                            prior->operands.size() == 1) {
                            argument = prior->operands.front();
                            break;
                        }
                    }
                }
                if (!(argument.type == signature->argument_types[i])) {
                    Operation conversion;
                    conversion.opcode = OpCode::kConvert;
                    conversion.location = operation.location;
                    conversion.operands.push_back(argument);
                    const ValueId converted = builder.next_value();
                    conversion.results.push_back(converted);
                    conversion.result_types.push_back(signature->argument_types[i]);
                    conversion.attributes["bitcast"] = "true";
                    value_types[converted] = signature->argument_types[i];
                    block->operations.push_back(std::move(conversion));
                    argument = Operand::value_ref(converted, signature->argument_types[i]);
                }
                operation.operands.push_back(std::move(argument));
                call_parameter_slots.erase(argument_name);
                call_parameter_slot_fields.erase(argument_name);
            }
            if (!builtin_call && printf_functions.contains(callee)) {
                if (!printf_buffer.has_value() || !printf_capacity.has_value()) {
                    return fail(&instruction,
                                "missing transitive printf binding for PTX device helper");
                }
                operation.operands.push_back(*printf_buffer);
                operation.operands.push_back(*printf_capacity);
            }
            if (has_return) {
                if (signature->return_type.kind == TypeKind::kVoid) {
                    return fail(&instruction, "void device call target '" + callee +
                                                  "' was given a return slot");
                }
                const ValueId result_value = builder.next_value();
                operation.results.push_back(result_value);
                operation.result_types.push_back(signature->return_type);
                value_types[result_value] = signature->return_type;
                call_return_slots[grouped_names(instruction.operands[0]).front()] =
                    Operand::value_ref(result_value, signature->return_type);
            }
            if (builtin_call && signature->tolerance_bounded) {
                result.module.semantic_quality = SemanticQuality::kToleranceBounded;
                const std::string caveat =
                    "Metal-missing float math functions use numerically tested typed expansions";
                if (std::find(result.module.semantic_caveats.begin(),
                              result.module.semantic_caveats.end(), caveat) ==
                    result.module.semantic_caveats.end()) {
                    result.module.semantic_caveats.push_back(caveat);
                }
            }
        } else {
            operation.opcode = arithmetic_opcode(root);
            if (operation.opcode == OpCode::kInvalid) {
                return fail(&instruction, "PTX opcode '" + instruction.opcode +
                                              "' has no CuMetal IR normalization");
            }
            const std::size_t first_source = destinations.empty() ? 0 : 1;
            const Type arithmetic_type = ptx_scalar_type(instruction.opcode);
            for (std::size_t i = first_source; i < instruction.operands.size(); ++i) {
                operation.operands.push_back(bit_container_operand(i, arithmetic_type));
            }
            if (!operation.result_types.empty() && operation.result_types.front().is_pointer()) {
                if (root == "mad") {
                    if (operation.operands.size() != 3 ||
                        !operation.operands[2].type.is_pointer()) {
                        return fail(&instruction,
                                    "pointer mad requires the base pointer as its addend");
                    }
                    Operation product;
                    product.opcode = OpCode::kMul;
                    product.location = operation.location;
                    product.operands = {operation.operands[0], operation.operands[1]};
                    const ValueId product_value = builder.next_value();
                    product.results = {product_value};
                    product.result_types = {arithmetic_type};
                    value_types[product_value] = arithmetic_type;
                    block->operations.push_back(std::move(product));
                    const Operand base = operation.operands[2];
                    operation.opcode = OpCode::kPointerOffset;
                    operation.operands = {
                        base, Operand::value_ref(product_value, arithmetic_type)};
                    operation.attributes["offset_unit"] = "bytes";
                } else {
                    operation.opcode = OpCode::kPointerOffset;
                }
                for (const Operand& operand : operation.operands) {
                    if (operand.kind != OperandKind::kValue) continue;
                    const auto provenance = function->pointer_provenance.find(operand.value);
                    if (provenance != function->pointer_provenance.end()) {
                        function->pointer_provenance[operation.results.front()] = provenance->second;
                        break;
                    }
                }
            } else if (root == "mad") {
                operation.attributes["combined"] = "mul_add";
            }
            if (root == "mul" &&
                instruction.opcode.find(".hi.") != std::string::npos) {
                operation.attributes["high_half"] = "true";
                if (has_signed_integer_type(instruction.opcode)) {
                    operation.attributes["signed"] = "true";
                }
            }
            if (has_signed_integer_type(instruction.opcode) &&
                (root == "div" || root == "rem" || root == "shr")) {
                operation.attributes["signed"] = "true";
            }
        }

        if (!append_guard(&operation, instruction, *environment)) return false;
        block->operations.push_back(std::move(operation));
        for (std::size_t i = 0; i < destinations.size(); ++i) {
            if (i < instruction_results[&instruction].size()) {
                (*environment)[destinations[i]] = instruction_results[&instruction][i];
            }
        }
        return true;
    }

    Successor make_successor(std::size_t source_index, std::size_t target_index) {
        Successor successor;
        successor.block = raw_blocks[target_index].id;
        for (const auto& [name, value] : block_arguments[target_index]) {
            (void)value;
            successor.arguments.push_back(outgoing[source_index].at(name));
        }
        return successor;
    }

    bool materialize_function() {
        Function function;
        function.name = entry->name;
        function.is_kernel = is_kernel;
        if (entry->return_params.size() > 1) {
            return fail(nullptr,
                        "typed PTX device functions support at most one return value");
        }
        function.return_type = entry->return_params.empty()
                                   ? Type::void_type()
                                   : parameter_type(entry->return_params.front());
        if (is_kernel) function.kernel_abi = KernelAbi{};

        for (std::size_t index = 0; index < entry->params.size(); ++index) {
            const auto& parameter = entry->params[index];
            const Type type = parameter_types[parameter.name];
            const ValueId value = builder.next_value();
            parameter_values[parameter.name] = value;
            value_types[value] = type;
            function.arguments.push_back({
                .value = value,
                .name = parameter.name,
                .type = type,
            });
            if (type.is_pointer()) {
                function.pointer_provenance[value] = {
                    .base_kind = PointerBaseKind::kKernelArgument,
                    .base_name = parameter.name,
                    .known_byte_offset = 0,
                    .alignment = 1,
                    .no_alias = false,
                };
            }
            if (is_kernel) {
                const std::uint32_t size = type_size(type);
                function.kernel_abi->arguments.push_back({
                    .name = parameter.name,
                    .kind = type.is_pointer() ? ArgumentKind::kPointer
                                              : ArgumentKind::kScalar,
                    .type = type,
                    .size = size,
                    .alignment = std::min<std::uint32_t>(size, 8),
                    .address_space = type.is_pointer() ? type.address_space
                                                       : AddressSpace::kConstant,
                    .binding_indices = {static_cast<std::uint32_t>(index)},
                });
                function.kernel_abi->bindings.push_back({
                    .kind = type.is_pointer() ? BindingKind::kBuffer
                                              : BindingKind::kBytes,
                    .binding_index = static_cast<std::uint32_t>(index),
                    .logical_argument_index = static_cast<std::uint32_t>(index),
                    .type = type,
                    .size = size,
                    .alignment = std::min<std::uint32_t>(size, 8),
                });
            }
        }

        // Registration appends referenced writable globals immediately after
        // the explicit CUDA arguments. Mirror that declaration order in the
        // typed ABI so every kernel sees the persistent registered buffer.
        if (is_kernel) {
            for (const auto& symbol : module_global_symbols) {
                const std::uint32_t binding_index =
                    static_cast<std::uint32_t>(function.arguments.size());
                if (binding_index >= 29u) {
                    return fail(
                        nullptr,
                        "CUDA device global conflicts with reserved Metal bindings");
                }
                const Type pointer_type =
                    Type::pointer(Type::integer(8), AddressSpace::kDevice);
                const ValueId value = builder.next_value();
                value_types[value] = pointer_type;
                const std::string argument_name =
                    "__cumetal_global_" + symbol.name;
                module_global_values.emplace(
                    symbol.name, Operand::value_ref(value, pointer_type));
                function.arguments.push_back({
                    .value = value,
                    .name = argument_name,
                    .type = pointer_type,
                });
                function.pointer_provenance[value] = {
                    .base_kind = PointerBaseKind::kAllocation,
                    .base_name = argument_name,
                    .known_byte_offset = 0,
                    .alignment = symbol.alignment,
                };
                const std::uint32_t logical_index =
                    static_cast<std::uint32_t>(
                        function.kernel_abi->arguments.size());
                const std::string hidden_role = "global_symbol:" + symbol.name;
                function.kernel_abi->arguments.push_back({
                    .name = argument_name,
                    .kind = ArgumentKind::kPointer,
                    .type = pointer_type,
                    .size = 8,
                    .alignment = 8,
                    .address_space = AddressSpace::kDevice,
                    .binding_indices = {binding_index},
                    .hidden_role = hidden_role,
                });
                function.kernel_abi->bindings.push_back({
                    .kind = BindingKind::kBuffer,
                    .binding_index = binding_index,
                    .logical_argument_index = logical_index,
                    .type = pointer_type,
                    .size = static_cast<std::uint32_t>(symbol.byte_size),
                    .alignment = symbol.alignment,
                    .hidden_role = hidden_role,
                });
            }
        }

        if (printf_functions.contains(entry->name)) {
            std::uint32_t binding_index =
                static_cast<std::uint32_t>(function.arguments.size());
            if (is_kernel && binding_index + 1 >= 29u) {
                return fail(nullptr,
                            "typed printf hidden arguments conflict with reserved Metal bindings");
            }
            const Type buffer_type =
                Type::pointer(Type::integer(32), AddressSpace::kDevice);
            const Type capacity_type = Type::integer(32);
            const ValueId buffer = builder.next_value();
            const ValueId capacity = builder.next_value();
            value_types[buffer] = buffer_type;
            value_types[capacity] = capacity_type;
            printf_buffer = Operand::value_ref(buffer, buffer_type);
            printf_capacity = Operand::value_ref(capacity, capacity_type);
            function.arguments.push_back({
                .value = buffer,
                .name = "__cumetal_printf_buffer",
                .type = buffer_type,
            });
            function.arguments.push_back({
                .value = capacity,
                .name = "__cumetal_printf_capacity",
                .type = capacity_type,
            });
            function.pointer_provenance[buffer] = {
                .base_kind = PointerBaseKind::kAllocation,
                .base_name = "__cumetal_printf_buffer",
                .known_byte_offset = 0,
                .alignment = 4,
            };
            if (is_kernel) {
                const std::uint32_t buffer_logical =
                    static_cast<std::uint32_t>(function.kernel_abi->arguments.size());
                function.kernel_abi->arguments.push_back({
                    .name = "__cumetal_printf_buffer",
                    .kind = ArgumentKind::kPointer,
                    .type = buffer_type,
                    .size = 8,
                    .alignment = 8,
                    .address_space = AddressSpace::kDevice,
                    .binding_indices = {binding_index},
                    .hidden_role = "printf_buffer",
                });
                function.kernel_abi->bindings.push_back({
                    .kind = BindingKind::kBuffer,
                    .binding_index = binding_index++,
                    .logical_argument_index = buffer_logical,
                    .type = buffer_type,
                    .size = 0,
                    .alignment = 4,
                    .hidden_role = "printf_buffer",
                });
                const std::uint32_t capacity_logical =
                    static_cast<std::uint32_t>(function.kernel_abi->arguments.size());
                function.kernel_abi->arguments.push_back({
                    .name = "__cumetal_printf_capacity",
                    .kind = ArgumentKind::kScalar,
                    .type = capacity_type,
                    .size = 4,
                    .alignment = 4,
                    .address_space = AddressSpace::kConstant,
                    .binding_indices = {binding_index},
                    .hidden_role = "printf_capacity",
                });
                function.kernel_abi->bindings.push_back({
                    .kind = BindingKind::kBytes,
                    .binding_index = binding_index,
                    .logical_argument_index = capacity_logical,
                    .type = capacity_type,
                    .size = 4,
                    .alignment = 4,
                    .hidden_role = "printf_capacity",
                });
            }
        }

        if (is_kernel && !module_constant_symbols.empty()) {
            if (function.arguments.size() > 30) {
                return fail(nullptr,
                            "kernel argument ABI conflicts with reserved constant buffer index 30");
            }
            const Type pointer_type =
                Type::pointer(Type::integer(8), AddressSpace::kConstant);
            const ValueId value = builder.next_value();
            value_types[value] = pointer_type;
            module_constant_buffer = Operand::value_ref(value, pointer_type);
            function.arguments.push_back({
                .value = value,
                .name = "__cumetal_constant_buffer",
                .type = pointer_type,
            });
            function.pointer_provenance[value] = {
                .base_kind = PointerBaseKind::kAllocation,
                .base_name = "__cumetal_constant_buffer",
                .known_byte_offset = 0,
                .alignment = 1,
            };
            const std::uint32_t logical_index =
                static_cast<std::uint32_t>(function.kernel_abi->arguments.size());
            function.kernel_abi->arguments.push_back({
                .name = "__cumetal_constant_buffer",
                .kind = ArgumentKind::kPointer,
                .type = pointer_type,
                .size = 8,
                .alignment = 8,
                .address_space = AddressSpace::kConstant,
                .binding_indices = {30},
            });
            function.kernel_abi->bindings.push_back({
                .kind = BindingKind::kBuffer,
                .binding_index = 30,
                .logical_argument_index = logical_index,
                .type = pointer_type,
                .size = static_cast<std::uint32_t>(module_constant_buffer_size),
                .alignment = 1,
                .hidden_role = "constant_symbols",
            });
        }

        for (std::size_t i = 0; i < raw_blocks.size(); ++i) {
            BasicBlock block;
            block.id = raw_blocks[i].id;
            block.name = raw_blocks[i].name;
            for (const auto& [name, value] : block_arguments[i]) {
                block.arguments.push_back({
                    .value = value,
                    .type = value_types[value],
                    .name = name,
                });
                if (value_types[value].is_pointer()) {
                    function.pointer_provenance[value] = {
                        .base_kind = PointerBaseKind::kUnknown,
                        .base_name = name,
                    };
                }
            }
            function.blocks.push_back(std::move(block));
        }


        for (const auto& [name, value] : implicit_values) {
            Operation initialization;
            initialization.opcode = OpCode::kConvert;
            initialization.results = {value};
            initialization.result_types = {value_types[value]};
            initialization.operands = {
                Operand::immediate("0", value_types[value]),
            };
            initialization.attributes["ptx_implicit_def"] = name;
            function.blocks.front().operations.push_back(std::move(initialization));
        }

        for (const auto& [name, depot] : local_depots) {
            const Type pointer_type =
                Type::pointer(Type::integer(8), AddressSpace::kPrivate);
            const ValueId value = builder.next_value();
            value_types[value] = pointer_type;
            local_depot_values[name] = Operand::value_ref(value, pointer_type);
            function.pointer_provenance[value] = {
                .base_kind = PointerBaseKind::kAllocation,
                .base_name = name,
                .known_byte_offset = 0,
                .alignment = depot.alignment,
            };
            Operation allocation;
            allocation.opcode = OpCode::kAlloca;
            allocation.results = {value};
            allocation.result_types = {pointer_type};
            allocation.attributes["byte_size"] = std::to_string(depot.byte_size);
            allocation.attributes["alignment"] = std::to_string(depot.alignment);
            function.blocks.front().operations.push_back(std::move(allocation));
        }

        for (std::size_t block_index = 0; block_index < raw_blocks.size(); ++block_index) {
            BasicBlock& block = function.blocks[block_index];
            std::unordered_map<std::string, ValueId> environment = incoming[block_index];
            function_return.reset();
            function_return_fields.clear();
            for (const Instruction* instruction : raw_blocks[block_index].instructions) {
                if (!translate_instruction(&function, &block, *instruction, &environment)) {
                    return false;
                }
            }

            Operation terminator;
            const Instruction* last =
                raw_blocks[block_index].instructions.empty()
                    ? nullptr
                    : raw_blocks[block_index].instructions.back();
            if (last != nullptr && root_opcode(last->opcode) == "bra") {
                if (raw_blocks[block_index].successors.empty()) {
                    return fail(last, "branch target '" + branch_target(*last) + "' does not exist");
                }
                if (is_conditional_branch(*last)) {
                    const auto [predicate_name, inverted] = normalized_predicate(last->predicate);
                    const auto predicate = environment.find(predicate_name);
                    if (predicate == environment.end()) {
                        return fail(last, "branch predicate '" + predicate_name + "' is undefined");
                    }
                    terminator.opcode = OpCode::kCondBranch;
                    terminator.operands.push_back(
                        Operand::value_ref(predicate->second, value_types[predicate->second]));
                    terminator.attributes["inverted"] = inverted ? "true" : "false";
                    for (std::size_t target : raw_blocks[block_index].successors) {
                        terminator.successors.push_back(make_successor(block_index, target));
                    }
                } else {
                    terminator.opcode = OpCode::kBranch;
                    terminator.successors.push_back(
                        make_successor(block_index, raw_blocks[block_index].successors.front()));
                }
                terminator.location = {
                    .file = result.module.source_name,
                    .line = static_cast<std::uint32_t>(std::max(0, last->line)),
                };
            } else if (last != nullptr && root_opcode(last->opcode) == "trap") {
                terminator.opcode = OpCode::kTrap;
                terminator.location = {
                    .file = result.module.source_name,
                    .line = static_cast<std::uint32_t>(std::max(0, last->line)),
                };
            } else if (last != nullptr &&
                       (root_opcode(last->opcode) == "ret" ||
                        root_opcode(last->opcode) == "exit")) {
                terminator.opcode = OpCode::kReturn;
                if (!is_kernel && function.return_type.kind != TypeKind::kVoid) {
                    if (function.return_type.kind == TypeKind::kAggregate) {
                        const std::optional<Operand> aggregate =
                            materialize_aggregate(
                                &block, last, function.return_type,
                                function_return_fields,
                                "aggregate PTX device return");
                        if (!aggregate.has_value()) return false;
                        terminator.operands.push_back(*aggregate);
                    } else {
                        if (!function_return.has_value()) {
                            return fail(
                                last,
                                "non-void PTX device function has no return value");
                        }
                        // The value reaching `st.param` carries the type of the
                        // instruction that produced it, which for a float in a
                        // .b32 register is not the declared return type. The
                        // aggregate path already reinterprets each field;
                        // without the same step here a scalar float return is
                        // emitted as a numeric conversion into the integer
                        // container. Widths must match -- anything else is a
                        // malformed return, not a reinterpretation.
                        Operand returned = *function_return;
                        if (!(returned.type == function.return_type)) {
                            if (type_size(returned.type) !=
                                type_size(function.return_type)) {
                                return fail(last,
                                            "PTX device return value does not fit "
                                            "its declared return type");
                            }
                            Operation conversion;
                            conversion.opcode = OpCode::kConvert;
                            conversion.location = {
                                .file = result.module.source_name,
                                .line = static_cast<std::uint32_t>(
                                    std::max(0, last->line)),
                            };
                            conversion.operands = {returned};
                            conversion.attributes["bitcast"] = "true";
                            const ValueId converted = builder.next_value();
                            conversion.results = {converted};
                            conversion.result_types = {function.return_type};
                            value_types[converted] = function.return_type;
                            block.operations.push_back(std::move(conversion));
                            returned =
                                Operand::value_ref(converted, function.return_type);
                        }
                        terminator.operands.push_back(returned);
                    }
                }
                terminator.location = {
                    .file = result.module.source_name,
                    .line = static_cast<std::uint32_t>(std::max(0, last->line)),
                };
            } else if (!raw_blocks[block_index].successors.empty()) {
                terminator.opcode = OpCode::kBranch;
                terminator.successors.push_back(
                    make_successor(block_index, raw_blocks[block_index].successors.front()));
            } else {
                terminator.opcode = OpCode::kReturn;
                if (!is_kernel && function.return_type.kind != TypeKind::kVoid) {
                    return fail(last,
                                "non-void PTX device function falls through without a return value");
                }
            }
            block.operations.push_back(std::move(terminator));
        }

        // Generic PTX helper pointers participate in Metal address-space
        // resolution exactly like addrspace(0) NVVM pointers. Mark every value
        // whose recovered type is still generic; call-site constraints will
        // resolve the complete def-use chain before MSL emission.
        for (const FunctionArgument& argument : function.arguments) {
            if (argument.type.is_pointer() &&
                argument.type.address_space == AddressSpace::kNone) {
                function.generic_pointer_values.insert(argument.value);
            }
        }
        for (const BasicBlock& block : function.blocks) {
            for (const BlockArgument& argument : block.arguments) {
                if (argument.type.is_pointer() &&
                    argument.type.address_space == AddressSpace::kNone) {
                    function.generic_pointer_values.insert(argument.value);
                }
            }
            for (const Operation& operation : block.operations) {
                for (std::size_t index = 0;
                     index < operation.results.size() &&
                     index < operation.result_types.size(); ++index) {
                    if (operation.result_types[index].is_pointer() &&
                        operation.result_types[index].address_space ==
                            AddressSpace::kNone) {
                        function.generic_pointer_values.insert(
                            operation.results[index]);
                    }
                }
            }
        }
        result.module.functions.push_back(std::move(function));
        return true;
    }
};

}  // namespace

PtxImportResult import_ptx(std::string_view ptx, const PtxImportOptions& options) {
    Importer importer;
    importer.result.module.source_name =
        options.source_name.empty() ? std::string("<ptx>") : options.source_name;
    importer.result.module.stage = IrStage::kGpuSemantic;
    importer.result.module.attributes["frontend"] = "ptx";
    importer.result.module.attributes["ir_schema"] = "1";
    if (ptx.find(".f64") != std::string_view::npos &&
        options.fp64_mode != "fast48" && options.fp64_mode != "wide48" &&
        options.fp64_mode != "ieee64") {
        importer.result.error =
            "typed PTX FP64 mode must be fast48, wide48, or ieee64";
        return importer.result;
    }
    importer.result.module.attributes["fp64_mode"] = options.fp64_mode;
    importer.result.module.global_threadgroups = scan_threadgroup_globals(ptx);
    for (const GlobalThreadgroup& global : importer.result.module.global_threadgroups) {
        importer.threadgroup_symbols.insert(global.name);
    }
    for (LocalDepot depot : scan_local_depots(ptx)) {
        importer.local_depots.emplace(depot.name, std::move(depot));
    }
    importer.implicit_definitions = scan_implicit_definitions(ptx);

    const InitializedByteArrayScan initialized_arrays =
        scan_initialized_byte_arrays(ptx);
    if (!initialized_arrays.error.empty()) {
        importer.result.error = initialized_arrays.error;
        return importer.result;
    }

    cumetal::ptx::ParseOptions parse_options;
    parse_options.strict = options.strict;
    const auto parsed = cumetal::ptx::parse_ptx(ptx, parse_options);
    if (!parsed.ok) {
        importer.result.error = parsed.error;
        return importer.result;
    }
    importer.result.warnings = parsed.warnings;
    if (!importer.select_entry(parsed, options)) return importer.result;
    const cumetal::ptx::EntryFunction* selected_entry = importer.entry;
    for (const cumetal::ptx::EntryFunction& function : parsed.module.functions) {
        importer.device_functions.emplace(function.name, &function);
    }

    std::vector<const cumetal::ptx::EntryFunction*> reachable_helpers;
    std::unordered_set<std::string> visiting;
    std::unordered_set<std::string> visited;
    const auto visit_call_graph = [&](const auto& self,
                                      const cumetal::ptx::EntryFunction& function)
        -> std::optional<bool> {
        if (visited.contains(function.name)) {
            return importer.printf_functions.contains(function.name);
        }
        if (!visiting.insert(function.name).second) {
            importer.result.error =
                "recursive PTX device-call cycle involving '" + function.name + "'";
            return std::nullopt;
        }
        bool uses_printf = false;
        for (const Instruction& instruction : function.instructions) {
            const std::optional<std::string> target = direct_call_target(instruction);
            if (!target.has_value()) continue;
            if (*target == "vprintf" || *target == "printf") {
                uses_printf = true;
                continue;
            }
            const auto callee = importer.device_functions.find(*target);
            if (callee == importer.device_functions.end()) continue;
            const std::optional<bool> child_uses_printf =
                self(self, *callee->second);
            if (!child_uses_printf.has_value()) return std::nullopt;
            uses_printf |= *child_uses_printf;
        }
        visiting.erase(function.name);
        visited.insert(function.name);
        if (uses_printf) importer.printf_functions.insert(function.name);
        if (!function.name.empty() && function.name != importer.entry->name &&
            importer.device_functions.contains(function.name)) {
            reachable_helpers.push_back(&function);
        }
        return uses_printf;
    };
    if (!visit_call_graph(visit_call_graph, *selected_entry).has_value()) {
        return importer.result;
    }
    std::unordered_set<int> decoded_printf_scaffold_lines;
    const auto collect_printf_scaffold = [&](
        const cumetal::ptx::EntryFunction& function) {
        const cumetal::passes::PrintfLowerResult lowered =
            cumetal::passes::lower_printf_calls(
                function, {.strict = options.strict, .ptx_source = ptx});
        if (!lowered.ok) return;
        for (const auto& call : lowered.calls) {
            decoded_printf_scaffold_lines.insert(call.abi_scaffold_lines.begin(),
                                                  call.abi_scaffold_lines.end());
        }
    };
    collect_printf_scaffold(*selected_entry);
    for (const auto* helper : reachable_helpers) collect_printf_scaffold(*helper);
    const auto symbol_is_referenced = [&](std::string_view symbol,
                                          bool include_printf_scaffold) {
        const auto instruction_references_symbol = [&](const Instruction& instruction) {
            if (!include_printf_scaffold &&
                decoded_printf_scaffold_lines.contains(instruction.line)) {
                return false;
            }
            return std::any_of(
                instruction.operands.begin(), instruction.operands.end(),
                [&](const std::string& operand) {
                    return parameter_name_from_operand(operand) == symbol;
                });
        };
        if (std::any_of(selected_entry->instructions.begin(),
                        selected_entry->instructions.end(),
                        instruction_references_symbol)) {
            return true;
        }
        return std::any_of(
            reachable_helpers.begin(), reachable_helpers.end(),
            [&](const cumetal::ptx::EntryFunction* helper) {
                return std::any_of(helper->instructions.begin(),
                                   helper->instructions.end(),
                                   instruction_references_symbol);
            });
    };
    const auto symbol_is_written = [&](std::string_view symbol) {
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
            return std::any_of(function.instructions.begin(),
                               function.instructions.end(), writes_symbol);
        };
        return std::any_of(parsed.module.entries.begin(), parsed.module.entries.end(),
                           function_writes) ||
               std::any_of(parsed.module.functions.begin(), parsed.module.functions.end(),
                           function_writes);
    };
    for (const InitializedByteArray& array : initialized_arrays.arrays) {
        if (!symbol_is_referenced(array.name, !array.module_private)) continue;
        const bool clang_promoted_literal =
            array.module_private && starts_with(array.name, "__const_$");
        const bool private_read_only =
            array.module_private && !symbol_is_written(array.name);
        if (!array.constant_space && !clang_promoted_literal && !private_read_only) {
            if (array.module_private) {
                // CUDA does not emit __cudaRegisterVar for translation-unit
                // private device storage. Keep the same hidden-buffer ABI as
                // visible globals, but describe its initializer so native AOT
                // and the registration JIT can create module-owned persistent
                // storage without inventing a public host symbol.
                importer.result.module.external_symbols.push_back({
                    .name = array.name,
                    .byte_size = array.bytes.size(),
                    .alignment = array.alignment,
                    .constant = false,
                    .module_private = true,
                    .initializer = array.bytes,
                });
            }
            // Ordinary initialized `.global` storage is mutable and must not
            // be embedded as a Metal constant. Give it the same hidden buffer
            // ABI as an uninitialized CUDA device symbol; registration copies
            // the host shadow's initializer into persistent Metal storage once
            // when the module is registered.
            importer.module_global_symbols.push_back({
                .name = array.name,
                .offset = 0,
                .byte_size = array.bytes.size(),
                .alignment = array.alignment,
            });
            continue;
        }
        importer.result.module.global_constants.push_back({
            .name = array.name,
            .bytes = array.bytes,
            .alignment = array.alignment,
        });
        importer.module_initialized_symbols.emplace(
            array.name,
            ModuleConstantSymbol{
                .name = array.name,
                .offset = 0,
                .byte_size = array.bytes.size(),
                .alignment = array.alignment,
            });
    }
    for (const ModuleConstantSymbol& symbol : scan_module_constant_symbols(ptx)) {
        importer.module_constant_buffer_size =
            std::max(importer.module_constant_buffer_size,
                     symbol.offset + symbol.byte_size);
        bool referenced = false;
        const auto symbol_referenced = [&](const Instruction& instruction) {
            referenced = std::any_of(
                instruction.operands.begin(), instruction.operands.end(),
                [&](const std::string& operand) {
                    return parameter_name_from_operand(operand) == symbol.name;
                });
            return referenced;
        };
        for (const Instruction& instruction : selected_entry->instructions) {
            if (symbol_referenced(instruction)) break;
        }
        for (const auto* helper : reachable_helpers) {
            if (referenced || std::any_of(helper->instructions.begin(),
                                          helper->instructions.end(),
                                          symbol_referenced)) {
                referenced = true;
                break;
            }
        }
        if (referenced) {
            importer.module_constant_symbols.emplace(symbol.name, symbol);
        }
    }
    if (!importer.module_constant_symbols.empty() &&
        importer.module_constant_buffer_size > 64u * 1024u) {
        importer.result.error = "external PTX constant buffer exceeds CUDA's 64 KB module limit";
        return importer.result;
    }
    for (const ModuleConstantSymbol& symbol : scan_module_global_symbols(ptx)) {
        const auto references_symbol = [&](const Instruction& instruction) {
            return std::any_of(
                instruction.operands.begin(), instruction.operands.end(),
                [&](const std::string& operand) {
                    return parameter_name_from_operand(operand) == symbol.name;
                });
        };
        bool referenced = std::any_of(
            selected_entry->instructions.begin(), selected_entry->instructions.end(),
            references_symbol);
        for (const auto* helper : reachable_helpers) {
            referenced |= std::any_of(helper->instructions.begin(),
                                      helper->instructions.end(), references_symbol);
        }
        if (referenced) importer.module_global_symbols.push_back(symbol);
    }

    const auto import_function = [&](const cumetal::ptx::EntryFunction* function,
                                     bool is_kernel) -> bool {
        Importer next;
        next.builder = importer.builder;
        next.result = std::move(importer.result);
        next.entry = function;
        next.is_kernel = is_kernel;
        next.device_functions = importer.device_functions;
        next.printf_functions = importer.printf_functions;
        next.threadgroup_symbols = importer.threadgroup_symbols;
        next.local_depots = importer.local_depots;
        next.implicit_definitions = importer.implicit_definitions;
        next.module_constant_symbols = importer.module_constant_symbols;
        next.module_constant_buffer_size = importer.module_constant_buffer_size;
        next.module_global_symbols = importer.module_global_symbols;
        next.module_initialized_symbols = importer.module_initialized_symbols;

        const cumetal::passes::PrintfLowerResult printf_lowered =
            cumetal::passes::lower_printf_calls(
                *function, {.strict = options.strict, .ptx_source = ptx});
        next.result.warnings.insert(next.result.warnings.end(),
                                    printf_lowered.warnings.begin(),
                                    printf_lowered.warnings.end());
        if (!printf_lowered.ok) {
            next.result.error = printf_lowered.error;
            importer = std::move(next);
            return false;
        }
        std::vector<std::uint32_t> format_ids;
        for (const auto& format : printf_lowered.formats) {
            if (!format.literal) {
                next.result.error =
                    "typed printf requires a decoded literal format string";
                importer = std::move(next);
                return false;
            }
            const auto existing = std::find(next.result.printf_formats.begin(),
                                            next.result.printf_formats.end(),
                                            format.token);
            if (existing == next.result.printf_formats.end()) {
                next.result.printf_formats.push_back(format.token);
                format_ids.push_back(static_cast<std::uint32_t>(
                    next.result.printf_formats.size() - 1));
            } else {
                format_ids.push_back(static_cast<std::uint32_t>(
                    std::distance(next.result.printf_formats.begin(), existing)));
            }
        }
        for (auto call : printf_lowered.calls) {
            if (call.format_id >= format_ids.size()) {
                next.result.error = "typed printf format id is out of range";
                importer = std::move(next);
                return false;
            }
            call.format_id = format_ids[call.format_id];
            next.printf_calls.emplace(call.source_line, call);
            next.printf_scaffold_lines.insert(call.abi_scaffold_lines.begin(),
                                               call.abi_scaffold_lines.end());
            next.printf_scaffold_lines.erase(call.source_line);
        }

        next.infer_register_types();
        next.build_cfg();
        next.allocate_values();
        if (!next.construct_ssa() || !next.materialize_function()) {
            importer = std::move(next);
            return false;
        }
        importer = std::move(next);
        return true;
    };

    for (const cumetal::ptx::EntryFunction* helper : reachable_helpers) {
        if (!import_function(helper, false)) return importer.result;
    }
    if (!import_function(selected_entry, true)) return importer.result;

    const VerifyResult verification = verify(importer.result.module);
    if (!verification.ok) {
        std::ostringstream error;
        error << "CuMetal IR verification failed";
        for (const Diagnostic& diagnostic : verification.diagnostics) {
            error << "\n";
            if (!diagnostic.location.str().empty()) {
                error << diagnostic.location.str() << ": ";
            }
            error << diagnostic.message;
        }
        importer.result.error = error.str();
        return importer.result;
    }
    importer.result.ok = true;
    return importer.result;
}

}  // namespace cumetal::ir
