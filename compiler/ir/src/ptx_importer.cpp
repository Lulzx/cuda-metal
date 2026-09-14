#include "cumetal/ir/ptx_importer.h"

#include "cumetal/passes/printf_lower.h"
#include "cumetal/ptx/parser.h"
#include "ptx_inline_asm.h"

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

// Element count of a `.v2`/`.v4` memory instruction, or 1.
std::size_t memory_vector_width(std::string_view opcode) {
    if (opcode.find(".v4.") != std::string_view::npos) return 4;
    if (opcode.find(".v2.") != std::string_view::npos) return 2;
    return 1;
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
                            (instruction.opcode == "mov.b32" ||
                             instruction.opcode.find(".b64") != std::string::npos);
    // `ld.*.v2/.v4 {a, b, ...}, [addr]` defines every register of the tuple.
    const bool vector_load = root == "ld" &&
                             (instruction.opcode.find(".v2.") != std::string::npos ||
                              instruction.opcode.find(".v4.") != std::string::npos);
    if (root != "setp" && root != "shfl" && !tuple_move && !vector_load &&
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
    // True when the double-precision builtin has no software-ALU helper and
    // lower_to_msl evaluates it through binary32.
    bool fp64_via_f32 = false;
    // Non-kCall opcodes let a device-math name ride the arithmetic or
    // conversion lowering instead of emitting a call: __nv_dadd_rn is a real
    // binary64 add and __nv_double2int_rd a real binary64->i32 conversion.
    OpCode opcode = OpCode::kCall;
    std::string fp64_conversion;
    std::string rounding_mode;
};

// Double-precision libdevice names whose callee has no FP64 software-ALU
// primitive; lower_to_msl decodes the binary64 words and runs the Metal
// binary32 builtin.
static const std::unordered_set<std::string> kFp64ViaF32Builtins = {
    "__nv_exp",   "__nv_exp2",    "__nv_exp10",  "__nv_expm1",
    "__nv_log",   "__nv_log2",    "__nv_log10",  "__nv_log1p",
    "__nv_sin",   "__nv_cos",     "__nv_tan",    "__nv_asin",
    "__nv_acos",  "__nv_atan",    "__nv_atan2",  "__nv_sinh",
    "__nv_cosh",  "__nv_tanh",    "__nv_asinh",  "__nv_acosh",
    "__nv_atanh", "__nv_cbrt",    "__nv_erf",    "__nv_erfc",
    "__nv_pow",   "__nv_powi",    "__nv_fmod",   "__nv_fdim",
    "__nv_hypot", "__nv_ldexp",   "__nv_scalbn", "__nv_nextafter",
    "__nv_rcbrt",
};

std::optional<BuiltinSignature> cuda_builtin_signature(std::string_view name) {
    // Classification returns an int and takes one float or double. The MSL
    // backend lowers the float forms to Metal's bit-pattern builtins and the
    // double forms on the binary64 storage word, so both are exact in every
    // math and FP64 mode.
    static const std::unordered_map<std::string, std::pair<std::string, unsigned>>
        kClassifyBuiltins = {
            {"__nv_isnanf", {"isnan", 32}},      {"__nv_isnand", {"isnan", 64}},
            {"__nv_isinff", {"isinf", 32}},      {"__nv_isinfd", {"isinf", 64}},
            {"__nv_finitef", {"isfinite", 32}},  {"__nv_isfinited", {"isfinite", 64}},
            {"__nv_signbitf", {"signbit", 32}},  {"__nv_signbitd", {"signbit", 64}},
        };
    const auto classify = kClassifyBuiltins.find(std::string(name));
    if (classify != kClassifyBuiltins.end()) {
        return BuiltinSignature{
            .metal_name = classify->second.first,
            .return_type = Type::integer(32),
            .argument_types = {Type::floating(classify->second.second)},
        };
    }
    static const std::unordered_map<std::string, std::string> kDoubleBuiltins = {
        {"__nv_fma", "fma"}, {"__nv_sqrt", "sqrt"},
        {"__nv_rsqrt", "rsqrt"},
        {"__nv_fmin", "fmin"}, {"__nv_fmax", "fmax"},
        {"__nv_remainder", "remainder"}, {"__nv_floor", "floor"},
        {"__nv_ceil", "ceil"}, {"__nv_trunc", "trunc"},
        {"__nv_round", "round"}, {"__nv_rint", "rint"},
        {"__nv_fabs", "fabs"}, {"__nv_copysign", "copysign"},
        // No software-ALU helper exists for these; lower_to_msl evaluates
        // them through binary32 (see kFp64ViaF32Builtins).
        {"__nv_exp", "exp"}, {"__nv_exp2", "exp2"}, {"__nv_exp10", "exp10"},
        {"__nv_expm1", "expm1"}, {"__nv_log", "log"}, {"__nv_log2", "log2"},
        {"__nv_log10", "log10"}, {"__nv_log1p", "log1p"}, {"__nv_sin", "sin"},
        {"__nv_cos", "cos"}, {"__nv_tan", "tan"}, {"__nv_asin", "asin"},
        {"__nv_acos", "acos"}, {"__nv_atan", "atan"}, {"__nv_atan2", "atan2"},
        {"__nv_sinh", "sinh"}, {"__nv_cosh", "cosh"}, {"__nv_tanh", "tanh"},
        {"__nv_asinh", "asinh"}, {"__nv_acosh", "acosh"}, {"__nv_atanh", "atanh"},
        {"__nv_cbrt", "cbrt"}, {"__nv_erf", "erf"}, {"__nv_erfc", "erfc"},
        {"__nv_pow", "pow"}, {"__nv_fmod", "fmod"}, {"__nv_fdim", "fdim"},
        {"__nv_hypot", "hypot"}, {"__nv_nextafter", "nextafter"},
        {"__nv_rcbrt", "__cumetal_rcbrt"},
    };
    const auto double_builtin = kDoubleBuiltins.find(std::string(name));
    if (double_builtin != kDoubleBuiltins.end()) {
        static const std::unordered_set<std::string> kBinaryDoubleBuiltins = {
            "__nv_fmin",   "__nv_fmax",    "__nv_remainder", "__nv_atan2",
            "__nv_pow",    "__nv_fmod",    "__nv_fdim",      "__nv_hypot",
            "__nv_copysign", "__nv_nextafter",
        };
        const std::size_t arity =
            name == "__nv_fma" ? 3
            : kBinaryDoubleBuiltins.contains(std::string(name)) ? 2 : 1;
        return BuiltinSignature{
            .metal_name = double_builtin->second,
            .return_type = Type::floating(64),
            .argument_types = std::vector<Type>(arity, Type::floating(64)),
            .fp64_via_f32 = kFp64ViaF32Builtins.contains(std::string(name)),
        };
    }
    static const std::unordered_map<std::string, std::string> kFloatBuiltins = {
        {"__nv_fminf", "fmin"}, {"__nv_fmaxf", "fmax"},
        {"__nv_sqrtf", "sqrt"}, {"__nv_rsqrtf", "rsqrt"},
        {"__nv_fabsf", "fabs"}, {"__nv_acosf", "acos"},
        {"__nv_expf", "exp"}, {"__nv_fast_expf", "exp"},
        {"__nv_exp2f", "exp2"}, {"__nv_exp10f", "exp10"},
        {"__nv_fast_exp10f", "exp10"},
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
        {"__nv_fast_sinf", "sin"}, {"__nv_fast_cosf", "cos"},
        {"__nv_fast_tanf", "tan"}, {"__nv_fast_logf", "log"},
        {"__nv_fast_log2f", "log2"}, {"__nv_fast_log10f", "log10"},
        {"__nv_fast_powf", "pow"}, {"__nv_nextafterf", "nextafter"},
        {"__nv_saturatef", "saturate"}, {"__nv_rcbrtf", "__cumetal_rcbrt"},
        {"__nv_fdividef", "__cumetal_fdivide"},
        {"__nv_fast_fdividef", "__cumetal_fdivide"},
    };
    const auto builtin = kFloatBuiltins.find(std::string(name));
    if (builtin != kFloatBuiltins.end()) {
        const std::size_t arity =
            name == "__nv_fmaf" ? 3 :
            (name == "__nv_atan2f" || name == "__nv_powf" ||
             name == "__nv_hypotf" || name == "__nv_fmodf" ||
             name == "__nv_copysignf" || name == "__nv_fdimf" ||
             name == "__nv_remainderf" || name == "__nv_fminf" ||
             name == "__nv_fmaxf" || name == "__nv_nextafterf" ||
             name == "__nv_fast_powf" || name == "__nv_fdividef" ||
             name == "__nv_fast_fdividef" ? 2 : 1);
        static const std::unordered_set<std::string> kExpandedMath = {
            "__nv_expm1f", "__nv_log1pf", "__nv_cbrtf", "__nv_erff",
            "__nv_erfcf", "__nv_hypotf", "__nv_remainderf", "__nv_rcbrtf",
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
    if (name == "__nv_frexpf") {
        return BuiltinSignature{
            .metal_name = "frexp",
            .return_type = Type::floating(32),
            .argument_types = {
                Type::floating(32),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
            },
        };
    }
    if (name == "__nv_frexp") {
        return BuiltinSignature{
            .metal_name = "frexp",
            .return_type = Type::floating(64),
            .argument_types = {
                Type::floating(64),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
            },
            .fp64_via_f32 = true,
        };
    }
    if (name == "__nv_modf") {
        return BuiltinSignature{
            .metal_name = "modf",
            .return_type = Type::floating(64),
            .argument_types = {
                Type::floating(64),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
            },
            .fp64_via_f32 = true,
        };
    }
    if (name == "__nv_sincos") {
        return BuiltinSignature{
            .metal_name = "sincos",
            .return_type = Type::void_type(),
            .argument_types = {
                Type::floating(64),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
            },
            .fp64_via_f32 = true,
        };
    }
    if (name == "__nv_modff") {
        return BuiltinSignature{
            .metal_name = "modf",
            .return_type = Type::floating(32),
            .argument_types = {
                Type::floating(32),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
            },
        };
    }
    if (name == "__nv_sincosf" || name == "__nv_fast_sincosf") {
        return BuiltinSignature{
            .metal_name = "sincos",
            .return_type = Type::void_type(),
            .argument_types = {
                Type::floating(32),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
            },
        };
    }
    if (name == "__nv_ldexpf" || name == "__nv_scalbnf") {
        return BuiltinSignature{
            .metal_name = "ldexp",
            .return_type = Type::floating(32),
            .argument_types = {Type::floating(32), Type::integer(32)},
        };
    }
    if (name == "__nv_powif") {
        return BuiltinSignature{
            .metal_name = "pow",
            .return_type = Type::floating(32),
            .argument_types = {Type::floating(32), Type::integer(32)},
        };
    }
    // Double libdevice calls whose second operand is an integer exponent or
    // that still lack a software-ALU primitive evaluate through binary32.
    if (name == "__nv_ldexp" || name == "__nv_scalbn") {
        return BuiltinSignature{
            .metal_name = "ldexp",
            .return_type = Type::floating(64),
            .argument_types = {Type::floating(64), Type::integer(32)},
            .fp64_via_f32 = true,
        };
    }
    if (name == "__nv_powi") {
        return BuiltinSignature{
            .metal_name = "pow",
            .return_type = Type::floating(64),
            .argument_types = {Type::floating(64), Type::integer(32)},
            .fp64_via_f32 = true,
        };
    }
    // IEEE arithmetic spelled as device functions. These become real IR ops
    // rather than calls, so the binary64 forms reach the same software-ALU
    // lowering as add.rn.f64 and friends in every FP64 mode.
    static const std::unordered_map<std::string, OpCode> kFp64ArithBuiltins = {
        {"__nv_dadd_rn", OpCode::kAdd}, {"__nv_dsub_rn", OpCode::kSub},
        {"__nv_dmul_rn", OpCode::kMul}, {"__nv_ddiv_rn", OpCode::kDiv},
    };
    if (const auto arith = kFp64ArithBuiltins.find(std::string(name));
        arith != kFp64ArithBuiltins.end()) {
        return BuiltinSignature{
            .return_type = Type::floating(64),
            .argument_types = {Type::floating(64), Type::floating(64)},
            .opcode = arith->second,
        };
    }
    static const std::unordered_map<std::string, OpCode> kFloatArithBuiltins = {
        {"__nv_fadd_rn", OpCode::kAdd}, {"__nv_fsub_rn", OpCode::kSub},
        {"__nv_fmul_rn", OpCode::kMul}, {"__nv_fdiv_rn", OpCode::kDiv},
        {"__nv_fmaf_rn", OpCode::kFma}, {"__nv_fmaf_ieee_rn", OpCode::kFma},
    };
    if (const auto arith = kFloatArithBuiltins.find(std::string(name));
        arith != kFloatArithBuiltins.end()) {
        const std::size_t arity = arith->second == OpCode::kFma ? 3 : 2;
        return BuiltinSignature{
            .return_type = Type::floating(32),
            .argument_types = std::vector<Type>(arity, Type::floating(32)),
            .opcode = arith->second,
        };
    }
    // Directed-rounding interval intrinsics stay calls (metal_name is the
    // stem plus suffix, e.g. "fadd_rd"); both lowerings route them through
    // the correctly-rounded vf64 software ALU. Binary32 forms widen each
    // operand to binary64, run the vf64 op, and convert back with the
    // requested rounding — exact for add/sub/mul since those operands'
    // products and sums always fit in the binary64 significand.
    if (name.starts_with("__nv_") && name.size() > 8) {
        const std::string_view suffix = name.substr(name.size() - 3);
        if (suffix == "_rd" || suffix == "_ru" || suffix == "_rz") {
            const std::string_view stem = name.substr(5, name.size() - 8);
            static const std::unordered_set<std::string_view> kDirectedStems = {
                "fadd", "fsub", "fmul", "fdiv", "frcp", "fsqrt", "fmaf",
                "dadd", "dsub", "dmul", "ddiv", "drcp", "dsqrt", "fma",
            };
            if (kDirectedStems.contains(stem)) {
                const bool is_double = stem.front() == 'd' || stem == "fma";
                const std::size_t arity =
                    stem == "fma" || stem == "fmaf" ? 3
                    : stem.ends_with("rcp") || stem.ends_with("sqrt") ? 1
                                                                      : 2;
                return BuiltinSignature{
                    .metal_name = std::string(stem) + std::string(suffix),
                    .return_type = Type::floating(is_double ? 64 : 32),
                    .argument_types = std::vector<Type>(
                        arity, Type::floating(is_double ? 64 : 32)),
                };
            }
        }
    }
    // __nv_frcp_rn is the correctly-rounded reciprocal; binary32 division is
    // already IEEE rne so it composes as 1.0f/x. __nv_frsqrt_rn has no exact
    // reciprocal-sqrt primitive, so it composes as 1.0f/sqrt(x) (~1 ulp).
    // __nv_drcp_rn divides in the active binary64 mode, which is exact.
    if (name == "__nv_frcp_rn" || name == "__nv_frsqrt_rn" ||
        name == "__nv_drcp_rn") {
        const bool is_double = name == "__nv_drcp_rn";
        return BuiltinSignature{
            .metal_name = name == "__nv_frsqrt_rn" ? "frsqrt_rn" : "rcp",
            .return_type = Type::floating(is_double ? 64 : 32),
            .argument_types = {Type::floating(is_double ? 64 : 32)},
            .tolerance_bounded = name == "__nv_frsqrt_rn",
        };
    }
    if (name == "__nv_fsqrt_rn") {
        return BuiltinSignature{
            .metal_name = "sqrt",
            .return_type = Type::floating(32),
            .argument_types = {Type::floating(32)},
        };
    }
    if (name == "__nv_dsqrt_rn") {
        return BuiltinSignature{
            .metal_name = "sqrt",
            .return_type = Type::floating(64),
            .argument_types = {Type::floating(64)},
        };
    }
    // __nv_<src>2double_<mode> and __nv_double2<dst>_<mode> are real
    // binary64 conversions, not calls: they become kConvert ops carrying the
    // software-FP64 conversion kind and the encoded rounding mode. The float
    // variants are already covered by kFloatToIntBuiltins/kIntToFloat above.
    if (name.starts_with("__nv_")) {
        const std::size_t mode_sep = name.rfind('_');
        const std::string_view suffix =
            mode_sep == std::string_view::npos
                ? std::string_view{}
                : std::string_view(name).substr(mode_sep + 1);
        static const std::unordered_map<std::string_view, const char*>
            kRoundSuffix = {
                {"rn", "0u"}, {"rz", "1u"}, {"rd", "2u"}, {"ru", "3u"},
            };
        const auto rounding = kRoundSuffix.find(suffix);
        if (rounding != kRoundSuffix.end() &&
            name.substr(0, mode_sep).find("double") != std::string_view::npos) {
            const std::string body(name.substr(5, mode_sep - 5));
            const std::size_t two = body.find('2');
            if (two != std::string::npos) {
                const std::string source = body.substr(0, two);
                const std::string destination = body.substr(two + 1);
                auto int_type = [](const std::string& kind) -> Type {
                    return kind == "ll" || kind == "ull" ? Type::integer(64)
                                                         : Type::integer(32);
                };
                if (destination == "double" &&
                    (source == "int" || source == "ll" || source == "uint" ||
                     source == "ull")) {
                    return BuiltinSignature{
                        .return_type = Type::floating(64),
                        .argument_types = {int_type(source)},
                        .opcode = OpCode::kConvert,
                        .fp64_conversion = source.front() == 'u'
                                               ? "unsigned_to_f64"
                                               : "signed_to_f64",
                        .rounding_mode = rounding->second,
                    };
                }
                if (source == "double" &&
                    (destination == "int" || destination == "ll" ||
                     destination == "uint" || destination == "ull" ||
                     destination == "float")) {
                    const bool to_float = destination == "float";
                    return BuiltinSignature{
                        .return_type = to_float ? Type::floating(32)
                                                : int_type(destination),
                        .argument_types = {Type::floating(64)},
                        .opcode = OpCode::kConvert,
                        .fp64_conversion =
                            to_float ? "f64_to_f32"
                            : destination.front() == 'u' ? "f64_to_unsigned"
                                                         : "f64_to_signed",
                        .rounding_mode = rounding->second,
                    };
                }
            }
        }
    }
    // Raw binary64 storage-word access. hiloint2double packs two u32 halves;
    // double2hiint/double2loint extract them. Bit-exact in every FP64 mode.
    if (name == "__nv_hiloint2double") {
        return BuiltinSignature{
            .metal_name = "hiloint2double",
            .return_type = Type::floating(64),
            .argument_types = {Type::integer(32), Type::integer(32)},
        };
    }
    if (name == "__nv_double2hiint" || name == "__nv_double2loint") {
        return BuiltinSignature{
            .metal_name = std::string(name == "__nv_double2hiint"
                                          ? "double2hiint"
                                          : "double2loint"),
            .return_type = Type::integer(32),
            .argument_types = {Type::floating(64)},
        };
    }
    // sinpi/cospi map to the Metal builtins; sincospi keeps the pointer-out
    // ABI like sincos. logb is the unbiased exponent as a float, ilogb as an
    // int. The double forms evaluate through binary32 like the rest of the
    // transcendental surface.
    if (name == "__nv_sinpif" || name == "__nv_sinpi" ||
        name == "__nv_cospif" || name == "__nv_cospi") {
        const bool is_double = name.back() != 'f';
        return BuiltinSignature{
            .metal_name = name.find("sinpi") != std::string_view::npos
                              ? "sinpi"
                              : "cospi",
            .return_type = Type::floating(is_double ? 64 : 32),
            .argument_types = {Type::floating(is_double ? 64 : 32)},
            .fp64_via_f32 = is_double,
        };
    }
    if (name == "__nv_sincospif" || name == "__nv_sincospi") {
        const bool is_double = name.back() != 'f';
        const Type scalar = Type::floating(is_double ? 64 : 32);
        return BuiltinSignature{
            .metal_name = "sincospi",
            .return_type = Type::void_type(),
            .argument_types = {
                scalar,
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
            },
            .fp64_via_f32 = is_double,
        };
    }
    if (name == "__nv_logbf" || name == "__nv_logb") {
        const bool is_double = name.back() != 'f';
        return BuiltinSignature{
            .metal_name = "logb",
            .return_type = Type::floating(is_double ? 64 : 32),
            .argument_types = {Type::floating(is_double ? 64 : 32)},
            .fp64_via_f32 = is_double,
        };
    }
    if (name == "__nv_ilogbf" || name == "__nv_ilogb") {
        const bool is_double = name.back() != 'f';
        return BuiltinSignature{
            .metal_name = "ilogb",
            .return_type = Type::integer(32),
            .argument_types = {Type::floating(is_double ? 64 : 32)},
            .fp64_via_f32 = is_double,
        };
    }
    if (name == "__nv_llrintf" || name == "__nv_llrint" ||
        name == "__nv_llroundf" || name == "__nv_llround") {
        const bool is_double = name.back() != 'f';
        return BuiltinSignature{
            .metal_name = name.find("llrint") != std::string_view::npos
                              ? "llrint"
                              : "llround",
            .return_type = Type::integer(64),
            .argument_types = {Type::floating(is_double ? 64 : 32)},
            .fp64_via_f32 = is_double,
        };
    }
    if (name == "__nv_remquof" || name == "__nv_remquo") {
        const bool is_double = name.back() != 'f';
        const Type scalar = Type::floating(is_double ? 64 : 32);
        return BuiltinSignature{
            .metal_name = "remquo",
            .return_type = scalar,
            .argument_types = {
                scalar, scalar,
                Type::pointer(Type::integer(8), AddressSpace::kPrivate),
            },
            .tolerance_bounded = !is_double,
            .fp64_via_f32 = is_double,
        };
    }
    // Multi-argument norms without a Metal builtin compose from sqrt/fabs.
    static const std::unordered_map<std::string, unsigned> kNormBuiltins = {
        {"__nv_norm3df", 3}, {"__nv_norm3d", 3},
        {"__nv_norm4df", 4}, {"__nv_norm4d", 4},
        {"__nv_rnorm3df", 3}, {"__nv_rnorm3d", 3},
        {"__nv_rnorm4df", 4}, {"__nv_rnorm4d", 4},
        {"__nv_rhypotf", 2}, {"__nv_rhypot", 2},
    };
    if (const auto norm = kNormBuiltins.find(std::string(name));
        norm != kNormBuiltins.end()) {
        const bool is_double = name.back() != 'f';
        const Type scalar = Type::floating(is_double ? 64 : 32);
        std::string metal = norm->first.substr(5);
        if (metal.back() == 'f') metal.pop_back();
        return BuiltinSignature{
            .metal_name = std::move(metal),
            .return_type = scalar,
            .argument_types = std::vector<Type>(norm->second, scalar),
            .tolerance_bounded = !is_double,
            .fp64_via_f32 = is_double,
        };
    }
    // Inverse-error-function and gamma family. None has a Metal builtin; each
    // lowers through a numerically-tested binary32 expansion, so the float
    // forms are tolerance-bounded and the double forms evaluate through
    // binary32 under emulation.
    static const std::unordered_set<std::string> kExpansionBuiltins = {
        "__nv_erfcxf",  "__nv_erfcx",
        "__nv_normcdff", "__nv_normcdf",
        "__nv_normcdfinvf", "__nv_normcdfinv",
        "__nv_erfinvf", "__nv_erfinv",
        "__nv_erfcinvf", "__nv_erfcinv",
        "__nv_tgammaf", "__nv_tgamma",
        "__nv_lgammaf", "__nv_lgamma",
    };
    if (kExpansionBuiltins.contains(std::string(name))) {
        // The double spelling is the bare base name; the float form appends
        // 'f'. A trailing 'f' only marks float when the stripped name is also
        // a known builtin -- __nv_normcdf is double, __nv_normcdff is float.
        const bool is_double =
            name.back() != 'f' ||
            !kExpansionBuiltins.contains(std::string(name.substr(0, name.size() - 1)));
        const Type scalar = Type::floating(is_double ? 64 : 32);
        std::string metal(name.substr(5));
        if (!is_double && metal.back() == 'f') metal.pop_back();
        return BuiltinSignature{
            .metal_name = std::move(metal),
            .return_type = scalar,
            .argument_types = {scalar},
            .tolerance_bounded = !is_double,
            .fp64_via_f32 = is_double,
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
    // A complete little-endian address relocation, not placeholder zero bytes.
    std::string pointer_target;
};

struct InitializedByteArrayScan {
    std::vector<InitializedByteArray> arrays;
    std::unordered_map<std::string, std::unordered_set<std::string>> dependencies;
    bool address_size_64 = false;
    std::string error;
};

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

// Parameter slots require a literal byte offset; never interpret malformed
// offsets as zero (the permissive address helper is used by other paths).
std::optional<std::int64_t> parameter_slot_offset(std::string operand, const std::string& name) {
    operand.erase(std::remove_if(operand.begin(), operand.end(),
        [](unsigned char c) { return std::isspace(c); }), operand.end());
    const std::string prefix = "[" + name;
    if (!starts_with(operand, prefix) || operand.back() != ']') return std::nullopt;
    const std::string suffix = operand.substr(prefix.size(), operand.size() - prefix.size() - 1);
    if (suffix.empty()) return 0;
    if (suffix.front() != '+' && suffix.front() != '-') return std::nullopt;
    try {
        std::size_t consumed = 0;
        const auto offset = std::stoll(suffix, &consumed, 0);
        if (consumed == suffix.size()) return offset;
    } catch (...) {}
    return std::nullopt;
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

std::uint32_t ptx_register_container_bits(std::string_view name);

// Bounded tail-self-call elimination for scalar integer helpers. Ineligible
// cycles retain the normal call-graph rejection. In particular, no local frame
// is reused: memory operations and pointer/address instructions are excluded.
bool normalize_scalar_tail_call(cumetal::ptx::EntryFunction* function) {
    if (function->params.size() != 1 || function->return_params.size() != 1 ||
        function->params.front().is_pointer ||
        (function->params.front().type != ".b64" && function->params.front().type != ".u64") ||
        function->params.front().byte_size != 8 || function->return_params.front().byte_size != 16) return false;
    const auto slot_at = [](std::string operand, const std::string& name, int offset) {
        operand.erase(std::remove_if(operand.begin(), operand.end(),
            [](unsigned char c) { return std::isspace(c); }), operand.end());
        return operand == "[" + name + "+" + std::to_string(offset) + "]" ||
               (offset == 0 && operand == "[" + name + "]");
    };
    auto& instructions = function->instructions;
    if (instructions.empty()) return false;
    const std::string parameter = function->params.front().name;
    const std::string return_parameter = function->return_params.front().name;
    std::size_t call_index = instructions.size();
    const std::unordered_set<std::string> pure_roots = {
        "mov", "add", "sub", "mul", "mad", "shr", "shl", "xor", "or", "and",
        "not", "neg", "setp", "selp", "bra", "ret", "call", "ptx"};
    for (std::size_t i = 0; i < instructions.size(); ++i) {
        const auto& instruction = instructions[i];
        const auto root = root_opcode(instruction.opcode);
        if (root == "call") {
            if (call_index != instructions.size() || !instruction.predicate.empty() ||
                (instruction.opcode != "call" && instruction.opcode != "call.uni") ||
                !instruction.supported || direct_call_target(instruction) != function->name) return false;
            call_index = i;
        } else if (root == "ld" || root == "st") {
            if (!starts_with(instruction.opcode, "ld.param.") &&
                !starts_with(instruction.opcode, "st.param.")) return false;
        } else if (!pure_roots.contains(root) ||
                   (root == "ptx" && instruction.opcode != "ptx.label")) return false;
        // Never let address-valued symbols (including local depots) enter the
        // scalar loop through arithmetic or selects. Match whole operands;
        // first_register alone would also accept tuples or address expressions.
        if (root != "ld" && root != "st" && root != "bra" && root != "call" &&
            root != "ret" && root != "ptx") {
            if (instruction.operands.empty() ||
                first_register(instruction.operands.front()) != trim(instruction.operands.front())) return false;
            for (const auto& operand : instruction.operands) {
                const std::string value = trim(operand);
                if (!value.empty() && first_register(value) == value) continue;
                try {
                    std::size_t consumed = 0;
                    std::stoll(value, &consumed, 0);
                    if (consumed != value.size()) return false;
                } catch (...) { return false; }
            }
        }
    }
    if (call_index == instructions.size() || call_index == 0) return false;
    const auto& call = instructions[call_index];
    if (call.operands.size() != 3) return false;
    const auto arguments = grouped_names(call.operands[2]);
    const auto returns = grouped_names(call.operands[0]);
    if (arguments.size() != 1 || returns.size() != 1) return false;
    const auto& argument_store = instructions[call_index - 1];
    if (argument_store.opcode != "st.param.b64" || !argument_store.predicate.empty() ||
        argument_store.operands.size() != 2 ||
        parameter_name_from_operand(argument_store.operands[0]) != arguments.front() ||
        !slot_at(argument_store.operands[0], arguments.front(), 0)) return false;
    const std::string argument = trim(argument_store.operands[1]);
    if (first_register(argument) != argument) {
        try {
            std::size_t consumed = 0;
            std::stoll(argument, &consumed, 0);
            if (consumed != argument.size()) return false;
        } catch (...) { return false; }
    } else if (ptx_register_container_bits(argument) != 64) return false;

    // The supported continuation is two scalar loads of the recursive result,
    // followed by optional join labels, two identical return stores and ret.
    // No computation, branch, memory effect or predication may intervene.
    std::map<std::int64_t, std::string> forwarded;
    std::set<std::size_t> remove;
    std::size_t cursor = call_index + 1;
    for (int lane = 0; lane < 2; ++lane, ++cursor) {
        if (cursor >= instructions.size()) return false;
        const auto& load = instructions[cursor];
        if (load.opcode != "ld.param.b64" || !load.predicate.empty() || load.operands.size() != 2 ||
            parameter_name_from_operand(load.operands[1]) != returns.front() ||
            ptx_register_container_bits(load.operands[0]) != 64) return false;
        const auto offset = slot_at(load.operands[1], returns.front(), 0) ? 0 :
                            slot_at(load.operands[1], returns.front(), 8) ? 8 : -1;
        if ((offset != 0 && offset != 8) || !forwarded.emplace(offset, load.operands[0]).second) return false;
        remove.insert(cursor);
    }
    if (forwarded.at(0) == forwarded.at(8)) return false;
    while (cursor < instructions.size() && instructions[cursor].opcode == "ptx.label") ++cursor;
    std::set<std::int64_t> stored;
    for (int lane = 0; lane < 2; ++lane, ++cursor) {
        if (cursor >= instructions.size()) return false;
        const auto& store = instructions[cursor];
        if (store.opcode != "st.param.b64" || !store.predicate.empty() || store.operands.size() != 2 ||
            parameter_name_from_operand(store.operands[0]) != return_parameter) return false;
        const auto offset = slot_at(store.operands[0], return_parameter, 0) ? 0 :
                            slot_at(store.operands[0], return_parameter, 8) ? 8 : -1;
        if (!forwarded.contains(offset) || store.operands[1] != forwarded.at(offset) || !stored.insert(offset).second) return false;
    }
    if (cursor + 1 != instructions.size() || instructions[cursor].opcode != "ret" ||
        !instructions[cursor].predicate.empty()) return false;

    const auto& first = instructions.front();
    if ((first.opcode != "ld.param.u64" && first.opcode != "ld.param.b64") ||
        !first.predicate.empty() || first.operands.size() != 2 ||
        parameter_name_from_operand(first.operands[1]) != parameter ||
        !slot_at(first.operands[1], parameter, 0) ||
        ptx_register_container_bits(first.operands[0]) != 64) return false;
    // No hidden parameter-slot aliases or extra reads/stores are allowed.
    for (std::size_t i = 1; i < instructions.size(); ++i) {
        const auto& instruction = instructions[i];
        if (starts_with(instruction.opcode, "ld.param") && !remove.contains(i)) return false;
        if (starts_with(instruction.opcode, "st.param") && i != call_index - 1 && i < cursor - 2) return false;
    }
    std::string current = "%rd_cm_tail_argument", header = "$cm_tail_header";
    const auto collides = [&](const std::string& name) {
        return std::any_of(instructions.begin(), instructions.end(), [&](const Instruction& instruction) {
            return std::any_of(instruction.operands.begin(), instruction.operands.end(),
                               [&](const std::string& operand) { return operand.find(name) != std::string::npos; });
        });
    };
    while (collides(current)) current += "_";
    while (collides(header)) header += "_";
    const auto make = [&](std::string opcode, std::vector<std::string> operands, int line) {
        Instruction result;
        result.opcode = std::move(opcode); result.operands = std::move(operands);
        result.line = line; result.supported = true;
        return result;
    };
    std::vector<Instruction> rewritten;
    rewritten.push_back(make(first.opcode, {current, first.operands[1]}, first.line));
    rewritten.push_back(make("ptx.label", {header}, first.line));
    rewritten.push_back(make("mov.u64", {first.operands[0], current}, first.line));
    for (std::size_t i = 1; i < instructions.size(); ++i) {
        if (i == call_index - 1 || remove.contains(i)) continue;
        if (i == call_index) {
            rewritten.push_back(make("mov.u64", {current, argument}, call.line));
            rewritten.push_back(make("bra.uni", {header}, call.line));
        } else rewritten.push_back(instructions[i]);
    }
    instructions = std::move(rewritten);
    function->register_declarations.push_back({current, "b64"});
    return true;
}

// Reuse a local frame only for a read-all / replace-all tail-call diamond.
// The input is consumed before any frame write, addresses cannot escape, and
// the continuation only forwards the complete result. No iteration cap or
// knowledge of a particular RNG's constants is needed.
bool normalize_local_buffer_tail_call(cumetal::ptx::EntryFunction* function,
                                     const std::unordered_map<std::string, LocalDepot>& depots) {
    auto& code = function->instructions;
    if (function->params.size() != 1 || function->params[0].byte_size != 8 ||
        function->return_params.size() != 1 || function->return_params[0].byte_size != 16 ||
        code.size() < 17) return false;
    const auto exact = [&](std::size_t i, std::string_view opcode, std::size_t operands) {
        return i < code.size() && code[i].opcode == opcode && code[i].supported &&
               code[i].predicate.empty() && code[i].operands.size() == operands;
    };
    if (!exact(0, "mov.b64", 2) || !exact(1, "cvta.local.u64", 2) ||
        !exact(2, "ld.param.b64", 2) || !exact(3, "cvta.to.local.u64", 2)) return false;
    const std::string local = code[0].operands[0], generic = code[1].operands[0];
    const std::string input = code[2].operands[0], read = code[3].operands[0];
    const auto depot = depots.find(code[0].operands[1]);
    if (depot == depots.end() || depot->second.byte_size != 16 || depot->second.alignment < 16 ||
        code[1].operands[1] != local || code[3].operands[1] != input ||
        parameter_slot_offset(code[2].operands[1], function->params[0].name) != 0) return false;
    std::set<std::string> pointers = {local, generic, input, read};
    if (pointers.size() != 4) return false;
    for (const auto& pointer : pointers)
        if (first_register(pointer) != pointer) return false;

    std::set<int> bytes;
    std::set<std::string> scalars;
    const auto scalar = [&](const std::string& operand) {
        if (scalars.contains(operand)) return true;
        try {
            std::size_t consumed = 0;
            std::stoll(operand, &consumed, 0);
            return consumed == operand.size();
        } catch (...) { return false; }
    };
    std::size_t branch = 4;
    for (; branch < code.size() && root_opcode(code[branch].opcode) != "bra"; ++branch) {
        const auto& instruction = code[branch];
        if (!instruction.predicate.empty() || !instruction.supported || instruction.operands.empty()) return false;
        const auto& destination = instruction.operands[0];
        if (first_register(destination) != destination || pointers.contains(destination)) return false;
        if (instruction.opcode == "ld.local.b8" && instruction.operands.size() == 2) {
            const auto offset = parameter_slot_offset(instruction.operands[1], read);
            if (!offset || *offset < 0 || *offset >= 16 || !bytes.insert(static_cast<int>(*offset)).second) return false;
        } else {
            // These integer operations cover byte assembly and its predicate;
            // no addresses, calls, stores, labels or other control flow occur.
            if ((instruction.opcode != "shl.b64" && instruction.opcode != "or.b64" &&
                 instruction.opcode != "setp.eq.b64") || instruction.operands.size() != 3 ||
                !scalar(instruction.operands[1]) || !scalar(instruction.operands[2])) return false;
        }
        scalars.insert(destination);
    }
    if (bytes.size() != 16 || branch + 12 != code.size()) return false;
    if (code[branch].opcode != "bra" || code[branch].predicate.empty() ||
        code[branch].operands.size() != 1 || !exact(branch+1, "bra.uni", 1) ||
        code[branch+2].opcode != "ptx.label" || code[branch+2].operands.size() != 1 ||
        code[branch].operands[0] != code[branch+2].operands[0]) return false;
    if (!exact(branch+3, "add.u64", 3) || !exact(branch+4, "add.u64", 3) ||
        code[branch+3].operands[1] != generic || code[branch+3].operands[2] != "0" ||
        code[branch+4].operands[1] != local || code[branch+4].operands[2] != "0") return false;
    const std::string argument_pointer = code[branch+3].operands[0];
    const std::string store_pointer = code[branch+4].operands[0];
    for (const auto& pointer : {argument_pointer, store_pointer}) {
        if (first_register(pointer) != pointer || scalars.contains(pointer) || !pointers.insert(pointer).second) return false;
    }
    if (!exact(branch+5, "st.local.v2.b64", 2) ||
        parameter_slot_offset(code[branch+5].operands[0], store_pointer) != 0) return false;
    const auto tuple = [](const std::string& operand) {
        const std::string value = trim(operand);
        if (value.size() < 2 || value.front() != '{' || value.back() != '}') return std::vector<std::string>{};
        return grouped_names(value.substr(1, value.size() - 2));
    };
    const auto stored = tuple(code[branch+5].operands[1]);
    if (stored.size() != 2 || !scalar(stored[0]) || !scalar(stored[1])) return false;
    if (!exact(branch+6, "st.param.b64", 2) || code[branch+6].operands[1] != argument_pointer ||
        !exact(branch+7, "call.uni", 3) || direct_call_target(code[branch+7]) != function->name) return false;
    const auto arguments = grouped_names(code[branch+7].operands[2]);
    const auto results = grouped_names(code[branch+7].operands[0]);
    if (arguments.size() != 1 || results.size() != 1 ||
        parameter_slot_offset(code[branch+6].operands[0], arguments[0]) != 0 ||
        !exact(branch+8, "ld.param.v2.b64", 2) ||
        parameter_slot_offset(code[branch+8].operands[1], results[0]) != 0) return false;
    if (code[branch+9].opcode != "ptx.label" || code[branch+9].operands.size() != 1 ||
        code[branch+1].operands[0] != code[branch+9].operands[0] ||
        code[branch+2].operands[0] == code[branch+9].operands[0] ||
        !exact(branch+10, "st.param.v2.b64", 2) || !exact(branch+11, "ret", 0) ||
        parameter_slot_offset(code[branch+10].operands[0], function->return_params[0].name) != 0) return false;
    const auto forwarded = tuple(code[branch+8].operands[0]);
    if (forwarded.size() != 2 || forwarded[0] == forwarded[1] ||
        tuple(code[branch+10].operands[1]) != forwarded) return false;
    for (const auto& value : forwarded)
        if (!scalars.contains(value) || ptx_register_container_bits(value) != 64) return false;

    std::string header = "$cm_local_tail_header";
    while (std::any_of(code.begin(), code.end(), [&](const Instruction& instruction) {
        return std::any_of(instruction.operands.begin(), instruction.operands.end(),
            [&](const std::string& operand) { return operand.find(header) != std::string::npos; });
    })) header += "_";
    Instruction label;
    label.opcode = "ptx.label"; label.operands = {header}; label.supported = true;
    label.line = code[3].line;
    Instruction update = code[2];
    update.opcode = "mov.b64"; update.operands = {input, argument_pointer};
    Instruction jump = code[branch+1]; jump.operands = {header};
    std::vector<Instruction> rewritten;
    for (std::size_t i = 0; i < code.size(); ++i) {
        if (i == 3) rewritten.push_back(label);
        if (i == branch+6 || i == branch+8) continue;
        if (i == branch+7) {
            rewritten.push_back(update);
            rewritten.push_back(jump);
        } else rewritten.push_back(code[i]);
    }
    code = std::move(rewritten);
    return true;
}

// Parameter slots are compiler-managed byte storage, not observable memory.
// Expand exact, aligned v2.b64 transfers before SSA so both lanes participate
// in normal definition tracking and the existing aggregate ABI checks.
void normalize_vector_parameter_transfers(cumetal::ptx::EntryFunction* function) {
    std::vector<Instruction> rewritten;
    for (const auto& instruction : function->instructions) {
        const bool load = instruction.opcode == "ld.param.v2.b64";
        if ((!load && instruction.opcode != "st.param.v2.b64") ||
            !instruction.predicate.empty() || instruction.operands.size() != 2) {
            rewritten.push_back(instruction);
            continue;
        }
        const auto& address = instruction.operands[load ? 1 : 0];
        const std::string name = parameter_name_from_operand(address);
        const auto offset = parameter_slot_offset(address, name);
        const std::string tuple = trim(instruction.operands[load ? 0 : 1]);
        if (name.empty() || !registers_in(name).empty() || !offset || *offset < 0 || *offset % 16 != 0 ||
            *offset > std::numeric_limits<std::int64_t>::max() - 8 ||
            tuple.size() < 2 || tuple.front() != '{' || tuple.back() != '}') {
            rewritten.push_back(instruction);
            continue;
        }
        const auto lanes = grouped_names(tuple.substr(1, tuple.size() - 2));
        bool valid = lanes.size() == 2 && (!load || lanes[0] != lanes[1]);
        for (const auto& lane : lanes) {
            if (first_register(lane) == lane && ptx_register_container_bits(lane) == 64) continue;
            if (load) { valid = false; break; }
            try {
                std::size_t consumed = 0;
                std::stoll(lane, &consumed, 0);
                valid &= consumed == lane.size();
            } catch (...) { valid = false; }
        }
        if (!valid) { rewritten.push_back(instruction); continue; }
        for (int i = 0; i < 2; ++i) {
            Instruction scalar = instruction;
            scalar.opcode = load ? "ld.param.b64" : "st.param.b64";
            const std::string slot = "[" + name + "+" + std::to_string(*offset + 8*i) + "]";
            scalar.operands = load ? std::vector<std::string>{lanes[i], slot} :
                                     std::vector<std::string>{slot, lanes[i]};
            rewritten.push_back(std::move(scalar));
        }
    }
    function->instructions = std::move(rewritten);
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

// The vf64 conversion helpers take the same mode encoding as
// vf64_rounding_mode in lower_to_llvm.cpp: 0=rne, 1=rtz, 2=rtn, 3=rtp.
// PTX spells them .rn/.rz/.rm/.rp for float destinations and
// .rni/.rzi/.rmi/.rpi for integer destinations; a bare cvt to an integer
// destination defaults to rtz and to float defaults to rne.
std::string cvt_rounding_mode(std::string_view opcode) {
    if (opcode.find(".rni.") != std::string::npos ||
        opcode.find(".rn.") != std::string::npos) return "0u";
    if (opcode.find(".rzi.") != std::string::npos ||
        opcode.find(".rz.") != std::string::npos) return "1u";
    if (opcode.find(".rmi.") != std::string::npos ||
        opcode.find(".rm.") != std::string::npos) return "2u";
    if (opcode.find(".rpi.") != std::string::npos ||
        opcode.find(".rp.") != std::string::npos) return "3u";
    const Type destination = ptx_cvt_result_type(opcode);
    return destination.kind == TypeKind::kInteger ? "1u" : "0u";
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
        auto normalized_fields = fields;
        // PTX parameter arrays are byte storage. A pair of b64 stores may
        // populate an ABI aggregate represented as four u32 fields. Split only
        // complete, contiguous integer words; holes/overlaps still fail below.
        const bool u32_layout = std::all_of(type.elements.begin(), type.elements.end(),
            [](const Type& element) { return element == Type::integer(32); });
        if (u32_layout && fields.size() * 2 == type.elements.size()) {
            bool complete_u64 = true;
            std::int64_t expected_offset = 0;
            for (const auto& [offset, value] : fields) {
                complete_u64 &= offset == expected_offset && value.type == Type::integer(64);
                expected_offset += 8;
            }
            if (complete_u64) {
                normalized_fields.clear();
                for (const auto& [offset, value] : fields) {
                    const auto emit = [&](OpCode opcode, Type result_type, std::vector<Operand> operands) {
                        Operation operation;
                        operation.opcode = opcode;
                        operation.operands = std::move(operands);
                        const ValueId id = builder.next_value();
                        operation.results = {id}; operation.result_types = {result_type};
                        value_types[id] = result_type;
                        block->operations.push_back(std::move(operation));
                        return Operand::value_ref(id, result_type);
                    };
                    // Materialize immediates as ulong before shifting: `1 >> 32`
                    // would otherwise use a 32-bit Metal literal.
                    const Operand wide = emit(OpCode::kConvert, Type::integer(64), {value});
                    normalized_fields[offset] = emit(OpCode::kConvert, Type::integer(32), {wide});
                    const Operand high = emit(OpCode::kShiftRight, Type::integer(64),
                        {wide, Operand::immediate("32", Type::integer(64))});
                    normalized_fields[offset + 4] = emit(OpCode::kConvert, Type::integer(32), {high});
                }
            }
        }
        if (normalized_fields.size() != type.elements.size()) {
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
            const auto field = normalized_fields.find(byte_offset);
            if (field == normalized_fields.end()) {
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

    static std::optional<Type> declared_register_type(const std::string& name) {
        if (name == "pred") return Type::predicate();
        if (name == "b8" || name == "u8" || name == "s8") return Type::integer(8);
        if (name == "b16" || name == "u16" || name == "s16") return Type::integer(16);
        if (name == "f16") return Type::floating(16);
        if (name == "b32" || name == "u32" || name == "s32") return Type::integer(32);
        if (name == "f32") return Type::floating(32);
        if (name == "b64" || name == "u64" || name == "s64") return Type::integer(64);
        if (name == "f64") return Type::floating(64);
        return std::nullopt;
    }

    void infer_register_types() {
        for (const auto& parameter : entry->params) {
            parameter_types[parameter.name] = parameter_type(parameter);
        }
        // Body-local `.reg` declarations name their type explicitly; use it
        // before inference so a predicate stays a predicate.
        for (const auto& declaration : entry->register_declarations) {
            if (register_types.contains(declaration.name)) continue;
            if (const auto type = declared_register_type(declaration.type)) {
                register_types[declaration.name] = *type;
            }
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
            // A helper's explicit generic-to-local address conversion proves
            // pointer-ness even when all subsequent accesses are ld.local.
            // Keep its argument generic; call-site specialization determines
            // the actual address space rather than guessing from integer width.
            if (!is_kernel && instruction.opcode == "cvta.to.local.u64" &&
                instruction.operands.size() == 2) {
                const std::string source = first_register(instruction.operands[1]);
                if (!source.empty()) required_device_pointers.insert(source);
            }
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
                } else if (instruction.opcode == "mov.b32" && instruction.operands.size() == 2 &&
                           (instruction.operands[0].find('{') != std::string::npos ||
                            instruction.operands[1].find('{') != std::string::npos)) {
                    inferred = Type::integer(instruction.operands[0].find('{') != std::string::npos ? 16 : 32);
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
                } else if (root == "sub" && instruction.operands.size() == 3 &&
                           ptx_scalar_type(instruction.opcode) == Type::integer(64)) {
                    const auto left = register_types.find(first_register(instruction.operands[1]));
                    const auto right = register_types.find(first_register(instruction.operands[2]));
                    if (left != register_types.end() && left->second.is_pointer() &&
                        (right == register_types.end() || !right->second.is_pointer()))
                        inferred = left->second;
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
                    Type destination_type = inferred;
                    if (root == "ld" && !starts_with(instruction.opcode, "ld.param") &&
                        inferred.kind == TypeKind::kInteger) {
                        // PTX integer loads extend to each destination register's
                        // width. Keep memory width/signedness on the load so MSL
                        // reads only the requested bytes and extends correctly.
                        // Parameter loads retain their separate ABI/pointer path.
                        destination_type = Type::integer(std::max(
                            ptx_scalar_type(instruction.opcode).bit_width,
                            ptx_register_container_bits(destination)));
                    }
                    const auto existing = register_types.find(destination);
                    if (existing == register_types.end() || !(existing->second == destination_type)) {
                        register_types[destination] = destination_type;
                        changed = true;
                    }
                }
            }
        }
    }

    // Stable storage for instructions normalized before register SSA.
    std::map<const Instruction*, Instruction> normalized_selects;

    void remove_unobserved_self_selects() {
        for (RawBlock& block : raw_blocks) {
            if (block.instructions.empty() || block.successors.size() != 2) continue;
            const Instruction* branch = block.instructions.back();
            if (root_opcode(branch->opcode) != "bra" || branch->predicate.empty()) continue;
            const auto [predicate, inverted] = normalized_predicate(branch->predicate);
            if (inverted) continue;
            for (std::size_t index = 0; index + 1 < block.instructions.size(); ++index) {
                const Instruction* select = block.instructions[index];
                if ((select->opcode != "selp.b32" && select->opcode != "selp.b64") ||
                    !select->predicate.empty() || select->operands.size() != 4 ||
                    trim(select->operands[3]) != predicate) continue;
                const std::string destination = trim(select->operands[0]);
                const std::string source = trim(select->operands[1]);
                const int width = select->opcode == "selp.b32" ? 32 : 64;
                // Do not turn malformed select operands into valid mov tuples.
                // Keep this proof restricted to matching scalar registers.
                if (first_register(source) != source ||
                    ptx_register_container_bits(source) != width ||
                    ptx_register_container_bits(destination) != width) continue;
                if (first_register(destination) != destination ||
                    trim(select->operands[2]) != destination ||
                    trim(select->operands[1]) == destination) continue;
                bool safe = true;
                for (std::size_t i = index + 1; i + 1 < block.instructions.size(); ++i) {
                    const auto sources = source_registers(*block.instructions[i]);
                    const auto destinations = destination_registers(*block.instructions[i]);
                    if (std::find(sources.begin(), sources.end(), destination) != sources.end() ||
                        std::find(destinations.begin(), destinations.end(), destination) != destinations.end() ||
                        std::find(destinations.begin(), destinations.end(), predicate) != destinations.end()) safe = false;
                }
                if (!safe) continue;
                // Follow the false edge until an unconditional overwrite or
                // this same select. Cycles are safe only if no path reads the
                // old value. A subsequent true edge supplies the new value.
                std::vector<std::size_t> pending{block.successors[1]};
                std::unordered_set<std::size_t> visited;
                while (!pending.empty() && safe) {
                    const auto current = pending.back();
                    pending.pop_back();
                    if (!visited.insert(current).second) continue;
                    bool killed = false;
                    for (const Instruction* instruction : raw_blocks[current].instructions) {
                        if (instruction == select) { killed = true; break; }
                        const auto sources = source_registers(*instruction);
                        if (std::find(sources.begin(), sources.end(), destination) != sources.end()) {
                            safe = false;
                            break;
                        }
                        const auto destinations = destination_registers(*instruction);
                        if (instruction->predicate.empty() &&
                            std::find(destinations.begin(), destinations.end(), destination) != destinations.end()) {
                            killed = true;
                            break;
                        }
                    }
                    if (!killed) {
                        pending.insert(pending.end(), raw_blocks[current].successors.begin(),
                                       raw_blocks[current].successors.end());
                    }
                }
                if (!safe) continue;
                Instruction replacement = *select;
                replacement.opcode = select->opcode == "selp.b32" ? "mov.b32" : "mov.b64";
                replacement.operands = {select->operands[0], select->operands[1]};
                auto [stored, inserted] = normalized_selects.emplace(select, std::move(replacement));
                block.instructions[index] = &stored->second;
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
        const auto bit_container_of = [&](Operand operand, const Type& expected) {
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

        if (instruction.opcode == "mov.b32" &&
            std::any_of(instruction.operands.begin(), instruction.operands.end(),
                        [](const std::string& operand) { return operand.find('{') != std::string::npos; })) {
            if (instruction.operands.size() != 2 || !instruction.predicate.empty()) {
                return fail(&instruction, "mov.b32 tuples require two operands and no predicate");
            }
            const bool unpack = instruction.operands[0].find('{') != std::string::npos;
            const std::string tuple = trim(instruction.operands[unpack ? 0 : 1]);
            const auto comma = tuple.find(',');
            if (tuple.size() < 5 || tuple.front() != '{' || tuple.back() != '}' ||
                comma == std::string::npos || tuple.find(',', comma + 1) != std::string::npos) {
                return fail(&instruction, "mov.b32 currently requires exactly two 16-bit tuple lanes");
            }
            const std::vector<std::string> lanes = {
                trim(tuple.substr(1, comma - 1)),
                trim(tuple.substr(comma + 1, tuple.size() - comma - 2)),
            };
            for (const auto& lane : lanes) {
                if (unpack && lane == "_") continue;
                if (lane.empty() || lane != first_register(lane) ||
                    ptx_register_container_bits(lane) != 16) {
                    return fail(&instruction, "mov.b32 tuple lanes must be 16-bit registers (or unpack sinks)");
                }
            }
            if (destinations.empty() || (unpack && lanes[0] == lanes[1])) {
                return fail(&instruction, "mov.b32 tuple needs distinct non-sink destinations");
            }
            const Type u16 = Type::integer(16), u32 = Type::integer(32);
            const auto emit = [&](OpCode opcode, Type type, std::vector<Operand> inputs) {
                Operation temporary;
                temporary.opcode = opcode;
                temporary.location = operation.location;
                const ValueId value = builder.next_value();
                temporary.results = {value};
                temporary.result_types = {type};
                temporary.operands = std::move(inputs);
                value_types[value] = type;
                block->operations.push_back(std::move(temporary));
                return Operand::value_ref(value, type);
            };
            if (!unpack) {
                if (destinations.size() != 1 || ptx_register_container_bits(destinations[0]) != 32) {
                    return fail(&instruction, "mov.b32 tuple packing requires a 32-bit scalar destination");
                }
                const Operand low = bit_container_of(operand_for(lanes[0], *environment, u16), u16);
                const Operand high = bit_container_of(operand_for(lanes[1], *environment, u16), u16);
                if (!(low.type == u16) || !(high.type == u16)) {
                    return fail(&instruction, "mov.b32 tuple source values must contain 16 bits");
                }
                const Operand low32 = emit(OpCode::kConvert, u32, {low});
                const Operand high32 = emit(OpCode::kConvert, u32, {high});
                const Operand shifted = emit(OpCode::kShiftLeft, u32,
                    {high32, Operand::immediate("16", u32)});
                operation.opcode = OpCode::kBitOr;
                operation.result_types = {u32};
                operation.operands = {low32, shifted};
                value_types[operation.results.front()] = u32;
                block->operations.push_back(std::move(operation));
                (*environment)[destinations[0]] = instruction_results[&instruction][0];
            } else {
                const Operand packed = bit_container_operand(1, u32);
                if (!(packed.type == u32)) {
                    return fail(&instruction, "mov.b32 tuple unpacking requires a 32-bit source value");
                }
                std::size_t result_index = 0;
                for (std::size_t lane = 0; lane < 2; ++lane) {
                    if (lanes[lane] == "_") continue;
                    const Operand selected = lane == 0 ? packed : emit(OpCode::kShiftRight, u32,
                        {packed, Operand::immediate("16", u32)});
                    Operation extract;
                    extract.opcode = OpCode::kConvert;
                    extract.location = operation.location;
                    const ValueId result_value = instruction_results[&instruction][result_index++];
                    extract.results = {result_value};
                    extract.result_types = {u16};
                    extract.operands = {selected};
                    value_types[result_value] = u16;
                    block->operations.push_back(std::move(extract));
                    (*environment)[lanes[lane]] = result_value;
                }
            }
            return true;
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

        if ((starts_with(instruction.opcode, "st.param") || starts_with(instruction.opcode, "ld.param")) &&
            memory_vector_width(instruction.opcode) != 1)
            return fail(&instruction, "unsupported or malformed vector PTX parameter transfer");
        if (starts_with(instruction.opcode, "st.param")) {
            if (instruction.operands.size() < 2) {
                return fail(&instruction, "malformed st.param instruction");
            }
            const std::string name = parameter_name_from_operand(instruction.operands[0]);
            if (name.empty()) return fail(&instruction, "call parameter slot has no name");
            const Type store_type = ptx_scalar_type(instruction.opcode);
            Operand stored = source_operand(1, store_type);
            if (stored.type.kind == TypeKind::kInteger && store_type.kind == TypeKind::kInteger &&
                stored.type.bit_width > store_type.bit_width) {
                // st.param writes the instruction width, not the register width.
                // Keep low bits before recording scalar or aggregate slot data.
                Operation truncate;
                truncate.opcode = OpCode::kConvert;
                truncate.location = operation.location;
                truncate.operands = {stored};
                const ValueId value = builder.next_value();
                truncate.results = {value};
                truncate.result_types = {store_type};
                value_types[value] = store_type;
                block->operations.push_back(std::move(truncate));
                stored = Operand::value_ref(value, store_type);
            }
            const auto checked_offset = parameter_slot_offset(instruction.operands[0], name);
            if (!checked_offset) return fail(&instruction, "invalid PTX parameter slot byte offset");
            const std::int64_t byte_offset = *checked_offset;
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
                        const auto checked_offset = parameter_slot_offset(instruction.operands[1], name);
                        if (!checked_offset) return fail(&instruction, "invalid PTX return slot byte offset");
                        const std::int64_t byte_offset = *checked_offset;
                        const std::uint32_t loaded_size =
                            type_size(operation.result_types.front());
                        if (loaded_size == 8 && operation.result_types.front() == Type::integer(64) &&
                            byte_offset >= 0 && byte_offset % 8 == 0 &&
                            (static_cast<std::uint64_t>(byte_offset) + 8) <= type_size(aggregate_type) &&
                            std::all_of(aggregate_type.elements.begin(), aggregate_type.elements.end(),
                                [](const Type& field) { return field == Type::integer(32); })) {
                            const auto emit = [&](OpCode opcode, Type type, std::vector<Operand> operands) {
                                Operation part;
                                part.opcode = opcode; part.location = operation.location;
                                part.operands = std::move(operands);
                                const ValueId id = builder.next_value();
                                part.results = {id}; part.result_types = {type};
                                value_types[id] = type;
                                block->operations.push_back(std::move(part));
                                return Operand::value_ref(id, type);
                            };
                            const auto word = [&](std::int64_t index) {
                                const Operand value = emit(OpCode::kAggregateExtract, Type::integer(32),
                                    {returned->second, Operand::immediate(std::to_string(index), Type::integer(32))});
                                return emit(OpCode::kConvert, Type::integer(64), {value});
                            };
                            const Operand low = word(byte_offset / 4);
                            const Operand high = emit(OpCode::kShiftLeft, Type::integer(64),
                                {word(byte_offset / 4 + 1), Operand::immediate("32", Type::integer(64))});
                            operation.opcode = OpCode::kBitOr;
                            operation.operands = {low, high};
                        } else {
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
                        }
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
            // Vector loads: `ld.global.v2.b32 {%r1, %r2}, [addr]` fills each
            // register from consecutive elements. Lanes after the first become
            // their own loads at the right byte displacement; the instruction's
            // own operation keeps lane 0, so guarding and result bookkeeping
            // stay unchanged.
            const std::size_t lanes = memory_vector_width(instruction.opcode);
            if (lanes > 1) {
                if (operation.results.size() != lanes) {
                    return fail(&instruction, "vector load destination tuple width mismatch");
                }
                if (module_constant_symbols.contains(
                        parameter_name_from_operand(instruction.operands[1]))) {
                    return fail(&instruction, "vector load from a module constant is unsupported");
                }
                const Type element_type = ptx_scalar_type(instruction.opcode);
                const AddressSpace lane_space =
                    instruction.opcode.find(".shared") != std::string::npos
                        ? AddressSpace::kThreadgroup
                    : instruction.opcode.find(".local") != std::string::npos
                        ? AddressSpace::kPrivate
                    : instruction.opcode.find(".const") != std::string::npos
                        ? AddressSpace::kConstant
                        : AddressSpace::kDevice;
                const Operand base = memory_address_operand(
                    1, Type::pointer(element_type, lane_space));
                for (std::size_t lane = 1; lane < lanes; ++lane) {
                    Operation offset;
                    offset.opcode = OpCode::kPointerOffset;
                    offset.location = operation.location;
                    offset.operands = {
                        base,
                        Operand::immediate(std::to_string(lane * type_size(element_type)),
                                           Type::integer(64)),
                    };
                    offset.attributes["offset_unit"] = "bytes";
                    const ValueId pointer = builder.next_value();
                    offset.results = {pointer};
                    offset.result_types = {base.type};
                    value_types[pointer] = base.type;
                    block->operations.push_back(std::move(offset));

                    Operation load;
                    load.opcode = OpCode::kLoad;
                    load.location = operation.location;
                    load.attributes["ptx_opcode"] = instruction.opcode;
                    load.operands.push_back(Operand::value_ref(pointer, base.type));
                    load.results = {operation.results[lane]};
                    load.result_types = {operation.result_types[lane]};
                    load.attributes["memory_bit_width"] = std::to_string(element_type.bit_width);
                    if (has_signed_integer_type(instruction.opcode)) {
                        load.attributes["signed"] = "true";
                    }
                    load.attributes["alignment"] = std::to_string(type_size(element_type));
                    if (!append_guard(&load, instruction, *environment)) return false;
                    block->operations.push_back(std::move(load));
                }
                operation.results.resize(1);
                operation.result_types.resize(1);
            }
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
            const Type element_type = ptx_scalar_type(instruction.opcode);
            const auto store_value = [&](Operand input) {
                input = bit_container_of(input, element_type);
                // PTX permits st.b8/st.b16 from a wider integer register.
                // Preserve the memory width instead of emitting a u32 store.
                if (input.type.kind == TypeKind::kInteger &&
                    element_type.kind == TypeKind::kInteger &&
                    input.type.bit_width > element_type.bit_width) {
                    Operation truncate;
                    truncate.opcode = OpCode::kConvert;
                    truncate.location = operation.location;
                    const ValueId value = builder.next_value();
                    truncate.results = {value};
                    truncate.result_types = {element_type};
                    truncate.operands = {input};
                    value_types[value] = element_type;
                    block->operations.push_back(std::move(truncate));
                    input = Operand::value_ref(value, element_type);
                }
                return input;
            };
            const Operand base = memory_address_operand(
                0, Type::pointer(element_type, store_address_space));
            // Vector stores: `st.global.v2.b32 [addr], {%r1, 0}` writes each
            // register or literal to consecutive elements. Clang emits these for adjacent
            // struct fields at -O2, and storing only the first lane silently
            // dropped the rest.
            const std::size_t lanes = memory_vector_width(instruction.opcode);
            std::vector<std::string> lane_operands;
            if (lanes > 1) {
                const std::string tuple = trim(instruction.operands[1]);
                if (tuple.size() < 2 || tuple.front() != '{' || tuple.back() != '}') {
                    return fail(&instruction, "vector store source requires a braced tuple");
                }
                const std::string contents = tuple.substr(1, tuple.size() - 2);
                std::size_t begin = 0;
                do {
                    const std::size_t end = contents.find(',', begin);
                    lane_operands.push_back(trim(contents.substr(begin, end - begin)));
                    if (lane_operands.back().empty()) {
                        return fail(&instruction, "vector store source tuple has an empty lane");
                    }
                    if (end == std::string::npos) break;
                    begin = end + 1;
                } while (true);
            }
            if (lanes > 1 && lane_operands.size() != lanes) {
                return fail(&instruction,
                            "vector store source tuple must provide one operand per lane");
            }
            for (std::size_t lane = 1; lane < lanes; ++lane) {
                Operation offset;
                offset.opcode = OpCode::kPointerOffset;
                offset.location = operation.location;
                offset.operands = {
                    base,
                    Operand::immediate(std::to_string(lane * type_size(element_type)),
                                       Type::integer(64)),
                };
                offset.attributes["offset_unit"] = "bytes";
                const ValueId pointer = builder.next_value();
                offset.results = {pointer};
                offset.result_types = {base.type};
                value_types[pointer] = base.type;
                block->operations.push_back(std::move(offset));

                Operation store;
                store.opcode = OpCode::kStore;
                store.location = operation.location;
                store.attributes["ptx_opcode"] = instruction.opcode;
                store.operands.push_back(Operand::value_ref(pointer, base.type));
                store.operands.push_back(store_value(
                    operand_for(lane_operands[lane], *environment, element_type)));
                store.attributes["alignment"] = std::to_string(type_size(element_type));
                if (!append_guard(&store, instruction, *environment)) return false;
                block->operations.push_back(std::move(store));
            }
            operation.operands.push_back(base);
            operation.operands.push_back(
                lanes > 1
                    ? store_value(operand_for(lane_operands[0], *environment, element_type))
                    : store_value(source_operand(1, element_type)));
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
        } else if (root == "shf" || root == "prmt") {
            const bool left = instruction.opcode == "shf.l.wrap.b32";
            const bool permute = instruction.opcode == "prmt.b32";
            if ((!left && !permute && instruction.opcode != "shf.r.wrap.b32") ||
                instruction.operands.size() != 4 || destinations.size() != 1) {
                return fail(&instruction, "typed bit permutation requires shf.{l,r}.wrap.b32 or prmt.b32 and four operands");
            }
            // PTX concatenates [b:a]. Use unsigned 64-bit intermediates so
            // shifts of zero (including wrapped 32) never shift a u32 by 32.
            const Type u32 = Type::integer(32);
            const Type u64 = Type::integer(64);
            const auto emit = [&](OpCode opcode, Type type, std::vector<Operand> inputs) {
                Operation temporary;
                temporary.opcode = opcode;
                temporary.location = operation.location;
                const ValueId value = builder.next_value();
                temporary.results = {value};
                temporary.result_types = {type};
                temporary.operands = std::move(inputs);
                value_types[value] = type;
                block->operations.push_back(std::move(temporary));
                return Operand::value_ref(value, type);
            };
            const Operand a = bit_container_operand(1, u32);
            const Operand b = bit_container_operand(2, u32);
            const Operand count = source_operand(3, u32);
            const Operand low = emit(OpCode::kConvert, u64, {a});
            const Operand high = emit(OpCode::kConvert, u64, {b});
            const Operand high_bits = emit(OpCode::kShiftLeft, u64,
                {high, Operand::immediate("32", u64)});
            const Operand packed = emit(OpCode::kBitOr, u64, {high_bits, low});
            Operand shifted;
            if (permute) {
                const Operand selector = emit(OpCode::kConvert, u64, {count});
                const auto imm = [&](unsigned n) { return Operand::immediate(std::to_string(n), u64); };
                shifted = imm(0);
                for (unsigned lane = 0; lane < 4; ++lane) {
                    const Operand nibble = emit(OpCode::kShiftRight, u64, {selector, imm(lane * 4)});
                    const Operand index = emit(OpCode::kBitAnd, u64, {nibble, imm(7)});
                    const Operand distance = emit(OpCode::kMul, u64, {index, imm(8)});
                    const Operand source = emit(OpCode::kShiftRight, u64, {packed, distance});
                    const Operand byte = emit(OpCode::kBitAnd, u64, {source, imm(255)});
                    const Operand sign = emit(OpCode::kShiftRight, u64, {byte, imm(7)});
                    const Operand replicated = emit(OpCode::kMul, u64, {sign, imm(255)});
                    const Operand flag_bits = emit(OpCode::kShiftRight, u64, {nibble, imm(3)});
                    const Operand flag = emit(OpCode::kBitAnd, u64, {flag_bits, imm(1)});
                    const Operand mask = emit(OpCode::kSub, u64, {imm(0), flag});
                    const Operand difference = emit(OpCode::kBitXor, u64, {byte, replicated});
                    const Operand selected_difference = emit(OpCode::kBitAnd, u64, {difference, mask});
                    const Operand selected = emit(OpCode::kBitXor, u64, {byte, selected_difference});
                    const Operand positioned = emit(OpCode::kShiftLeft, u64, {selected, imm(lane * 8)});
                    shifted = emit(OpCode::kBitOr, u64, {shifted, positioned});
                }
            } else {
                const Operand masked = emit(OpCode::kBitAnd, u32,
                    {count, Operand::immediate("31", u32)});
                const Operand shift = emit(OpCode::kConvert, u64, {masked});
                shifted = emit(left ? OpCode::kShiftLeft : OpCode::kShiftRight,
                               u64, {packed, shift});
                if (left) {
                    shifted = emit(OpCode::kShiftRight, u64,
                        {shifted, Operand::immediate("32", u64)});
                }
            }
            operation.opcode = OpCode::kConvert;
            operation.result_types = {u32};
            operation.operands = {shifted};
            value_types[operation.results.front()] = u32;
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
            } else if (ptx_cvt_source_type(instruction.opcode) ==
                           Type::floating(64) &&
                       operation.result_types.front().kind ==
                           TypeKind::kInteger) {
                // cvt.<round>.{s,u}{32,64}.f64: a binary64 -> integer
                // conversion. The generic path would emit `int(bits)` -- a
                // bit-pattern truncation that returns the low 32 bits of the
                // storage word instead of the numeric value.
                const bool signed_dest =
                    instruction.opcode.find(".s32.f64") != std::string::npos ||
                    instruction.opcode.find(".s64.f64") != std::string::npos;
                operation.attributes["fp64_conversion"] =
                    signed_dest ? "f64_to_signed" : "f64_to_unsigned";
                operation.attributes["rounding_mode"] =
                    cvt_rounding_mode(instruction.opcode);
                operation.operands.push_back(
                    bit_container_operand(1, Type::floating(64)));
            } else if (operation.result_types.front() == Type::floating(64) &&
                       ptx_cvt_source_type(instruction.opcode).kind ==
                           TypeKind::kInteger) {
                // cvt.<round>.f64.{s,u}{32,64}: integer -> binary64.
                operation.attributes["fp64_conversion"] =
                    cvt_has_signed_source(instruction.opcode)
                        ? "signed_to_f64"
                        : "unsigned_to_f64";
                operation.attributes["rounding_mode"] =
                    cvt_rounding_mode(instruction.opcode);
                operation.operands.push_back(
                    bit_container_operand(
                        1, ptx_cvt_source_type(instruction.opcode)));
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
        } else if (root == "bfi") {
            if ((instruction.opcode != "bfi.b32" && instruction.opcode != "bfi.b64") ||
                instruction.operands.size() != 5 || destinations.size() != 1 ||
                !instruction.predicate.empty()) {
                return fail(&instruction, "typed PTX bfi requires unpredicated bfi.b32/b64 with five operands");
            }
            const Type type = ptx_scalar_type(instruction.opcode);
            const Type u32 = Type::integer(32);
            const Operand a = bit_container_operand(1, type);
            const Operand b = bit_container_operand(2, type);
            const Operand position = source_operand(3, u32);
            const Operand length = source_operand(4, u32);
            if (!(a.type == type) || !(b.type == type) ||
                !(position.type == u32) || !(length.type == u32)) {
                return fail(&instruction, "typed PTX bfi operand widths do not match the instruction");
            }
            const auto emit = [&](OpCode opcode, Type result_type,
                                  std::vector<Operand> inputs, std::string predicate = "") {
                Operation temporary;
                temporary.opcode = opcode;
                temporary.location = operation.location;
                const ValueId value = builder.next_value();
                temporary.results = {value};
                temporary.result_types = {result_type};
                temporary.operands = std::move(inputs);
                if (!predicate.empty()) temporary.attributes["predicate"] = predicate;
                value_types[value] = result_type;
                block->operations.push_back(std::move(temporary));
                return Operand::value_ref(value, result_type);
            };
            const auto imm32 = [&](unsigned n) {
                return Operand::immediate(std::to_string(n), u32);
            };
            const Operand all = Operand::immediate(
                type.bit_width == 64 ? "18446744073709551615" : "4294967295", type);
            const Operand pos = emit(OpCode::kBitAnd, u32, {position, imm32(255)});
            const Operand len = emit(OpCode::kBitAnd, u32, {length, imm32(255)});
            // Even discarded select arms must avoid an undefined full-width
            // shift. Clamp shift counts by masking, then select the PTX result.
            const Operand safe_pos = emit(OpCode::kBitAnd, u32, {pos, imm32(type.bit_width - 1)});
            const Operand safe_len = emit(OpCode::kBitAnd, u32, {len, imm32(type.bit_width - 1)});
            const Operand shifted_ones = emit(OpCode::kShiftLeft, type, {all, safe_len});
            const Operand short_mask = emit(OpCode::kBitXor, type, {shifted_ones, all});
            const Operand full_length = emit(OpCode::kCompare, Type::predicate(),
                {len, imm32(type.bit_width)}, "ge");
            const Operand low_mask = emit(OpCode::kSelect, type, {full_length, all, short_mask});
            const Operand mask = emit(OpCode::kShiftLeft, type, {low_mask, safe_pos});
            const Operand inverse = emit(OpCode::kBitXor, type, {mask, all});
            const Operand retained = emit(OpCode::kBitAnd, type, {b, inverse});
            const Operand shifted_a = emit(OpCode::kShiftLeft, type, {a, safe_pos});
            const Operand inserted = emit(OpCode::kBitAnd, type, {shifted_a, mask});
            const Operand merged = emit(OpCode::kBitOr, type, {retained, inserted});
            const Operand in_range = emit(OpCode::kCompare, Type::predicate(),
                {pos, imm32(type.bit_width)}, "lt");
            operation.opcode = OpCode::kSelect;
            operation.result_types = {type};
            operation.operands = {in_range, merged, b};
            value_types[operation.results.front()] = type;
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
            if ((root == "min" || root == "max") && has_signed_integer_type(instruction.opcode)) {
                operation.attributes["signed"] = "true";
            }
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
            operation.opcode = signature->opcode;
            if (signature->opcode == OpCode::kCall) {
                operation.attributes["callee"] = signature->metal_name;
                if (builtin_call) operation.attributes["builtin"] = "true";
            }
            if (!signature->fp64_conversion.empty()) {
                operation.attributes["fp64_conversion"] =
                    signature->fp64_conversion;
            }
            if (!signature->rounding_mode.empty()) {
                operation.attributes["rounding_mode"] = signature->rounding_mode;
            }
            // A double crosses the call boundary as its 64-bit storage word;
            // mark the op whenever the signature carries binary64 anywhere --
            // __nv_sincos is void-returning but still takes a double input.
            const bool signature_uses_fp64 =
                signature->return_type == Type::floating(64) ||
                std::any_of(signature->argument_types.begin(),
                            signature->argument_types.end(),
                            [](const Type& argument_type) {
                                return argument_type == Type::floating(64);
                            });
            if (signature_uses_fp64) {
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
                if (!(argument.type == signature->argument_types[i])) {
                    if (type_size(argument.type) != type_size(signature->argument_types[i]))
                        return fail(&instruction, "PTX call parameter value does not fit its declared argument type");
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
            if (builtin_call && signature->fp64_via_f32) {
                if (result.module.semantic_quality == SemanticQuality::kExact) {
                    result.module.semantic_quality = SemanticQuality::kSemanticEmulation;
                }
                const std::string caveat =
                    "FP64 libdevice calls evaluate through binary32 under emulation";
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
            const Type source_type = ptx_scalar_type(instruction.opcode);
            // mul.wide / mad.wide produce a result twice as wide as their
            // operands: the operands are sign- or zero-extended first and the
            // destination register is 64-bit. Typing the product at the operand
            // width turned `mul.wide.s32 %rd, %r, 4` -- the byte offset of every
            // indexed access -- into a 32-bit multiply added to a pointer, which
            // both truncates real offsets and trips an Apple compiler
            // miscompile of 32-bit pointer displacements.
            const bool wide_product =
                (root == "mul" || root == "mad") &&
                source_type.kind == TypeKind::kInteger &&
                instruction.opcode.find(".wide.") != std::string::npos;
            const Type arithmetic_type =
                wide_product ? Type::integer(source_type.bit_width * 2) : source_type;
            const bool wide_signed = wide_product && has_signed_integer_type(instruction.opcode);
            for (std::size_t i = first_source; i < instruction.operands.size(); ++i) {
                Operand operand = bit_container_operand(i, source_type);
                if (wide_product && operand.type.kind == TypeKind::kInteger &&
                    operand.type.bit_width == source_type.bit_width) {
                    Operation extend;
                    extend.opcode = OpCode::kConvert;
                    extend.location = operation.location;
                    extend.operands.push_back(operand);
                    const ValueId extended = builder.next_value();
                    extend.results.push_back(extended);
                    extend.result_types.push_back(arithmetic_type);
                    if (wide_signed) extend.attributes["signed_input"] = "true";
                    value_types[extended] = arithmetic_type;
                    block->operations.push_back(std::move(extend));
                    operand = Operand::value_ref(extended, arithmetic_type);
                }
                operation.operands.push_back(std::move(operand));
            }
            if (wide_product) {
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    if (i < operation.result_types.size() &&
                        !operation.result_types[i].is_pointer()) {
                        operation.result_types[i] = arithmetic_type;
                        value_types[operation.results[i]] = arithmetic_type;
                    }
                }
            }
            if (root == "sub" && std::any_of(operation.operands.begin(), operation.operands.end(),
                                               [](const Operand& operand) { return operand.type.is_pointer(); })) {
                if (operation.operands.size() != 2 || !operation.operands[0].type.is_pointer() ||
                    operation.operands[1].type != Type::integer(64) || source_type != Type::integer(64))
                    return fail(&instruction, "pointer subtraction requires a pointer minus a 64-bit integer byte offset");
                if (operation.result_types.empty() || !operation.result_types[0].is_pointer())
                    return fail(&instruction, "pointer subtraction lost its result address space");
                operation.attributes["offset_direction"] = "subtract";
                operation.attributes["offset_unit"] = "bytes";
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
                                            "its declared return type (" + returned.type.str() + " to " +
                                                function.return_type.str() + ")");
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

namespace detail {

// Substitute `$N` placeholders with synthetic registers (or immediates), and
// put braces on their own lines so the module parser's scope handling sees
// them the way it does in a .ptx file.
static bool substitute_inline_asm_operands(const InlineAsmRequest& request,
                                           std::string* text, std::string* error) {
    text->clear();
    const std::string& in = request.text;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const char c = in[i];
        if (c == '{' || c == '}') {
            *text += '\n';
            *text += c;
            *text += '\n';
            continue;
        }
        if (c != '$') {
            *text += c;
            continue;
        }
        if (i + 1 < in.size() && in[i + 1] == '$') {
            *text += '$';
            ++i;
            continue;
        }
        if (i + 1 < in.size() && in[i + 1] == '{') {
            *error = "inline PTX operand modifiers are unsupported";
            return false;
        }
        std::size_t end = i + 1;
        while (end < in.size() && std::isdigit(static_cast<unsigned char>(in[end]))) ++end;
        if (end == i + 1) {
            *error = "malformed inline PTX operand reference";
            return false;
        }
        const std::size_t index = static_cast<std::size_t>(std::stoul(in.substr(i + 1, end - i - 1)));
        if (index >= request.bindings.size()) {
            *error = "inline PTX references operand $" + std::to_string(index) +
                     " but only " + std::to_string(request.bindings.size()) + " are bound";
            return false;
        }
        const InlineAsmBinding& binding = request.bindings[index];
        if (binding.is_immediate) {
            *text += binding.immediate_text;
        } else {
            *text += "%cm_asm_" + std::to_string(index);
        }
        i = end - 1;
    }
    return true;
}

InlineAsmResult lower_inline_ptx_asm(const InlineAsmRequest& request, Builder* builder,
                                     std::unordered_map<ValueId, Type>* value_types,
                                     Function* function, BasicBlock* block) {
    InlineAsmResult out;
    std::string text;
    if (!substitute_inline_asm_operands(request, &text, &out.error)) return out;

    std::vector<std::string> warnings;
    cumetal::ptx::EntryFunction entry = cumetal::ptx::parse_instruction_block(
        text, static_cast<int>(request.line), &warnings);

    std::size_t output_count = 0;
    for (const InlineAsmBinding& binding : request.bindings) {
        if (binding.is_output) ++output_count;
    }
    if (entry.instructions.empty()) {
        // `asm volatile("" ::: "memory")` and friends: a compiler barrier
        // with nothing for the GPU to do. Outputs would be undefined.
        if (output_count != 0) {
            out.error = "inline PTX defines no instruction but declares an output";
            return out;
        }
        out.ok = true;
        return out;
    }
    for (const auto& instruction : entry.instructions) {
        const std::string root = root_opcode(instruction.opcode);
        if (root == "bra" || root == "brx" || root == "call" || root == "ret" ||
            root == "exit") {
            out.error = "inline PTX control flow ('" + instruction.opcode +
                        "') is unsupported; keep branches in CUDA C++";
            return out;
        }
        if (!instruction.supported) {
            out.error = "unsupported PTX opcode '" + instruction.opcode + "' in inline asm";
            return out;
        }
    }

    Importer importer;
    importer.builder = *builder;
    importer.result.module.source_name = request.source_name;
    importer.result.module.attributes["fp64_mode"] =
        request.fp64_mode.empty() ? std::string("fast48") : request.fp64_mode;
    importer.entry = &entry;
    importer.is_kernel = false;

    std::unordered_map<std::string, ValueId> environment;
    for (std::size_t i = 0; i < request.bindings.size(); ++i) {
        const InlineAsmBinding& binding = request.bindings[i];
        const std::string name = "%cm_asm_" + std::to_string(i);
        importer.register_types[name] = binding.type;
        if (binding.is_immediate || !binding.input.has_value()) continue;
        const Operand& input = *binding.input;
        ValueId value = kInvalidValue;
        if (input.kind == OperandKind::kValue) {
            value = input.value;
            importer.value_types[value] = input.type;
        } else {
            // A symbol (module global) operand: give it an SSA name so the
            // PTX lowering can treat it like any register.
            value = importer.builder.next_value();
            Operation materialize;
            materialize.opcode = OpCode::kConvert;
            materialize.operands.push_back(input);
            materialize.results.push_back(value);
            materialize.result_types.push_back(input.type);
            materialize.attributes["bitcast"] = "true";
            materialize.location = {.file = request.source_name, .line = request.line};
            block->operations.push_back(std::move(materialize));
            importer.value_types[value] = input.type;
        }
        environment[name] = value;
        if (binding.tied_output.has_value()) {
            const std::string tied = "%cm_asm_" + std::to_string(*binding.tied_output);
            environment[tied] = value;
            importer.value_types[value] = input.type;
        }
    }

    importer.infer_register_types();
    RawBlock raw;
    raw.id = kInvalidBlock;
    raw.name = "cm_inline_asm";
    for (const auto& instruction : entry.instructions) raw.instructions.push_back(&instruction);
    importer.raw_blocks.push_back(std::move(raw));
    importer.allocate_values();

    for (const auto& instruction : entry.instructions) {
        if (!importer.translate_instruction(function, block, instruction, &environment)) {
            out.error = importer.result.error.empty()
                            ? "inline PTX lowering failed"
                            : importer.result.error;
            return out;
        }
    }

    for (std::size_t i = 0; i < request.bindings.size(); ++i) {
        if (!request.bindings[i].is_output) continue;
        const auto value = environment.find("%cm_asm_" + std::to_string(i));
        if (value == environment.end()) {
            out.error = "inline PTX never writes output operand $" + std::to_string(i);
            return out;
        }
        const auto type = importer.value_types.find(value->second);
        out.outputs.push_back(Operand::value_ref(
            value->second,
            type != importer.value_types.end() ? type->second : request.bindings[i].type));
    }

    *builder = importer.builder;
    for (const auto& [value, type] : importer.value_types) (*value_types)[value] = type;
    out.caveats = importer.result.module.semantic_caveats;
    out.ok = true;
    return out;
}

}  // namespace detail

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

    cumetal::ptx::ParseOptions parse_options;
    // Parse the module first; strict opcode checks apply to the selected entry
    // and its reachable helpers, not unrelated kernels in the same PTX file.
    parse_options.strict = false;
    auto parsed = cumetal::ptx::parse_ptx(ptx, parse_options);
    if (!parsed.ok) {
        importer.result.error = parsed.error;
        return importer.result;
    }
    importer.result.warnings = parsed.warnings;
    // Normalize copies before capturing pointers into the parsed function list.
    for (auto& function : parsed.module.functions) {
        if (!normalize_scalar_tail_call(&function))
            normalize_local_buffer_tail_call(&function, importer.local_depots);
        normalize_vector_parameter_transfers(&function);
    }
    for (auto& entry : parsed.module.entries) normalize_vector_parameter_transfers(&entry);
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
            if (options.strict && !instruction.supported) {
                importer.fail(&instruction, "unsupported opcode '" + instruction.opcode + "'");
                return std::nullopt;
            }
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

    std::unordered_set<std::string> referenced_symbols;
    std::unordered_set<std::string> non_printf_referenced_symbols;
    const auto collect_function_symbols = [&](const cumetal::ptx::EntryFunction& function) {
        for (const Instruction& instruction : function.instructions) {
            for (const std::string& operand : instruction.operands) {
                collect_operand_symbols(operand, &referenced_symbols);
                if (!decoded_printf_scaffold_lines.contains(instruction.line)) {
                    collect_operand_symbols(operand, &non_printf_referenced_symbols);
                }
            }
        }
    };
    collect_function_symbols(*selected_entry);
    for (const auto* helper : reachable_helpers) collect_function_symbols(*helper);

    const InitializedByteArrayScan initialized_arrays =
        scan_initialized_byte_arrays(ptx, referenced_symbols);
    if (!initialized_arrays.error.empty()) {
        importer.result.error = initialized_arrays.error;
        return importer.result;
    }
    for (const auto& array : initialized_arrays.arrays) {
        if (!array.pointer_target.empty()) {
            referenced_symbols.insert(array.pointer_target);
            non_printf_referenced_symbols.insert(array.pointer_target);
        }
    }
    const auto symbol_is_referenced = [&](std::string_view symbol,
                                          bool include_printf_scaffold) {
        const auto& symbols = include_printf_scaffold ? referenced_symbols
                                                      : non_printf_referenced_symbols;
        return symbols.contains(std::string(symbol));
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
    // Fold only immutable, non-escaping pointer objects. This resolves the
    // relocation symbolically before SSA typing, preserving Metal address space.
    // Do not invent a numeric address or embed host/device pointer bytes in MSL.
    for (const auto& alias : initialized_arrays.arrays) {
        if (alias.pointer_target.empty()) continue;
        if (alias.bytes.size() != 8 || alias.alignment < 8 ||
            (alias.alignment & (alias.alignment - 1)) != 0) {
            importer.result.error = "PTX address relocation requires one aligned 64-bit pointer";
            return importer.result;
        }
        if (!initialized_arrays.address_size_64) {
            importer.result.error = "PTX address relocation requires explicit 64-bit addressing";
            return importer.result;
        }
        const auto target = std::find_if(initialized_arrays.arrays.begin(),
            initialized_arrays.arrays.end(), [&](const auto& value) {
                return value.name == alias.pointer_target;
            });
        if (!alias.module_private || target == initialized_arrays.arrays.end() ||
            !target->pointer_target.empty() || !target->module_private ||
            symbol_is_written(target->name)) {
            importer.result.error = "PTX address relocation requires a private read-only numeric target: " + alias.name;
            return importer.result;
        }
        for (const auto& [name, dependencies] : initialized_arrays.dependencies) {
            if ((name != alias.name && dependencies.contains(target->name)) ||
                dependencies.contains(alias.name)) {
                importer.result.error = "PTX relocated table has another escaping initializer reference: " + name;
                return importer.result;
            }
        }
        const auto resolve = [&](cumetal::ptx::EntryFunction& function) {
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
                        address_use |= symbols.contains(target->name);
                        alias_load |= symbols.contains(alias.name);
                    }
                    for (const auto& source : source_registers(instruction)) {
                        address_use |= table_addresses.contains(source);
                    }
                    const std::string root = root_opcode(instruction.opcode);
                    if (address_use && root != "ld" && root != "mov" &&
                        root != "add" && root != "sub" && root != "mad" &&
                        root != "cvta" && root != "selp") {
                        importer.result.error = "PTX relocated table address may escape or be written: " + target->name;
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
                    importer.result.error = "PTX relocated pointer must only be read by direct whole-pointer loads: " + alias.name;
                    return false;
                }
                instruction.opcode = "mov.u64";
                instruction.operands[1] = alias.pointer_target;
            }
            return true;
        };
        for (auto& function : parsed.module.entries) if (!resolve(function)) return importer.result;
        for (auto& function : parsed.module.functions) if (!resolve(function)) return importer.result;
    }
    for (const InitializedByteArray& array : initialized_arrays.arrays) {
        if (!array.pointer_target.empty()) continue;
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
        next.remove_unobserved_self_selects();
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
