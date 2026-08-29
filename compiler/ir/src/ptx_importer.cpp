#include "cumetal/ir/ptx_importer.h"

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
    if (root != "setp" && root != "shfl" && destinations.size() > 1) {
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

Type parameter_type(const cumetal::ptx::Parameter& parameter) {
    if (parameter.is_pointer) {
        return Type::pointer(Type::integer(8), AddressSpace::kDevice);
    }
    return ptx_scalar_type(parameter.type);
}

std::uint32_t type_size(const Type& type) {
    if (type.is_pointer()) return 8;
    if (type.kind == TypeKind::kPredicate) return 1;
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
        R"((?:\.extern\s+)?\.shared\s+\.align\s+([0-9]+)\s+\.b8\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*([0-9]*)\s*\]\s*;)"
    );
    std::vector<GlobalThreadgroup> globals;
    for (std::sregex_iterator iterator(source.begin(), source.end(), declaration), end;
         iterator != end; ++iterator) {
        const std::string size = (*iterator)[3].str();
        globals.push_back({
            .name = (*iterator)[2].str(),
            .byte_size = size.empty() ? 0 : std::stoull(size),
            .alignment = static_cast<std::uint32_t>(std::stoul((*iterator)[1].str())),
            .is_dynamic = size.empty(),
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

struct Importer {
    Builder builder;
    PtxImportResult result;
    const cumetal::ptx::EntryFunction* entry = nullptr;
    std::unordered_map<std::string, Type> parameter_types;
    std::unordered_map<std::string, ValueId> parameter_values;
    std::unordered_map<std::string, Type> register_types;
    std::unordered_map<const Instruction*, std::vector<ValueId>> instruction_results;
    std::unordered_map<ValueId, Type> value_types;
    std::vector<RawBlock> raw_blocks;
    std::unordered_map<std::string, std::size_t> label_blocks;
    std::vector<std::unordered_map<std::string, ValueId>> incoming;
    std::vector<std::unordered_map<std::string, ValueId>> outgoing;
    std::vector<std::map<std::string, ValueId>> block_arguments;
    std::unordered_map<std::string, Operand> call_parameter_slots;
    std::unordered_map<std::string, Operand> call_return_slots;
    std::unordered_set<std::string> threadgroup_symbols;
    std::unordered_map<std::string, LocalDepot> local_depots;
    std::unordered_map<std::string, Operand> local_depot_values;
    std::unordered_set<std::string> implicit_definitions;
    std::unordered_map<std::string, ValueId> implicit_values;
    std::unordered_map<std::string, ModuleConstantSymbol> module_constant_symbols;
    std::uint64_t module_constant_buffer_size = 0;
    std::optional<Operand> module_constant_buffer;

    bool fail(const Instruction* instruction, std::string message) {
        if (instruction != nullptr && instruction->line != 0) {
            message = "line " + std::to_string(instruction->line) + ": " + message;
        }
        result.error = std::move(message);
        return false;
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
                } else if (root == "mov" && instruction.operands.size() >= 2 &&
                           starts_with(trim(instruction.operands[1]), "0f")) {
                    inferred = Type::floating(32);
                } else if (starts_with(instruction.opcode, "ld.param") &&
                           instruction.operands.size() >= 2) {
                    const auto parameter = parameter_types.find(
                        parameter_name_from_operand(instruction.operands[1]));
                    if (parameter != parameter_types.end()) inferred = parameter->second;
                } else if (root == "cvta") {
                    const AddressSpace space =
                        instruction.opcode.find(".shared") != std::string::npos
                            ? AddressSpace::kThreadgroup
                        : instruction.opcode.find(".local") != std::string::npos
                            ? AddressSpace::kPrivate
                            : AddressSpace::kDevice;
                    inferred = Type::pointer(Type::integer(8), space);
                } else if ((root == "add" || root == "mov") && instruction.operands.size() >= 2) {
                    const std::string source_symbol =
                        parameter_name_from_operand(instruction.operands[1]);
                    if (threadgroup_symbols.contains(source_symbol)) {
                        inferred = Type::pointer(Type::integer(8),
                                                 AddressSpace::kThreadgroup);
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
        if (!instruction.supported) {
            return fail(&instruction, "unsupported PTX opcode '" + instruction.opcode + "'");
        }

        Operation operation;
        operation.location = {.file = result.module.source_name,
                              .line = static_cast<std::uint32_t>(std::max(0, instruction.line))};
        operation.attributes["ptx_opcode"] = instruction.opcode;
        operation.results = instruction_results[&instruction];
        for (ValueId value : operation.results) {
            operation.result_types.push_back(value_types[value]);
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

        if (starts_with(instruction.opcode, "st.param")) {
            if (instruction.operands.size() < 2) {
                return fail(&instruction, "malformed st.param instruction");
            }
            const std::string name = parameter_name_from_operand(instruction.operands[0]);
            if (name.empty()) return fail(&instruction, "call parameter slot has no name");
            call_parameter_slots[name] =
                source_operand(1, ptx_scalar_type(instruction.opcode));
            return true;
        } else if (starts_with(instruction.opcode, "ld.param")) {
            if (instruction.operands.size() < 2 || operation.results.empty()) {
                return fail(&instruction, "malformed ld.param instruction");
            }
            const std::string name = parameter_name_from_operand(instruction.operands[1]);
            const auto argument = parameter_values.find(name);
            if (argument == parameter_values.end()) {
                const auto returned = call_return_slots.find(name);
                if (returned == call_return_slots.end()) {
                    return fail(&instruction, "unknown kernel or call parameter '" + name + "'");
                }
                operation.opcode = OpCode::kConvert;
                operation.operands.push_back(returned->second);
                if (starts_with(instruction.opcode, "ld.param.b64") &&
                    returned->second.type == Type::floating(32)) {
                    operation.result_types.front() = Type::floating(32);
                    value_types[operation.results.front()] = Type::floating(32);
                } else if (!(returned->second.type == operation.result_types.front())) {
                    operation.attributes["bitcast"] = "true";
                }
            } else {
                operation.opcode = OpCode::kParameter;
                operation.operands.push_back(
                    Operand::value_ref(argument->second, parameter_types[name]));
                operation.attributes["parameter"] = name;
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
            if (starts_with(trim(instruction.operands[1]), "0f")) {
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    operation.result_types[i] = Type::floating(32);
                    value_types[operation.results[i]] = Type::floating(32);
                }
            }
            operation.opcode = OpCode::kConvert;
            operation.operands.push_back(source_operand(1, operation.result_types.front()));
        } else if (root == "cvta") {
            operation.opcode = OpCode::kAddressSpaceCast;
            operation.operands.push_back(source_operand(1, operation.result_types.front()));
            if (!operation.results.empty() && operation.operands.front().kind == OperandKind::kValue) {
                const auto provenance =
                    function->pointer_provenance.find(operation.operands.front().value);
                if (provenance != function->pointer_provenance.end()) {
                    function->pointer_provenance[operation.results.front()] = provenance->second;
                }
            }
        } else if (starts_with(instruction.opcode, "ld.global") ||
                   starts_with(instruction.opcode, "ld.const") ||
                   starts_with(instruction.opcode, "ld.shared") ||
                   starts_with(instruction.opcode, "ld.local")) {
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
                operation.operands.push_back(source_operand(
                    1, Type::pointer(operation.result_types.front(),
                                     AddressSpace::kDevice)));
            }
            operation.attributes["address"] = instruction.operands[1];
            operation.attributes["alignment"] =
                std::to_string(type_size(operation.result_types.front()));
        } else if (starts_with(instruction.opcode, "st.global") ||
                   starts_with(instruction.opcode, "st.const") ||
                   starts_with(instruction.opcode, "st.shared") ||
                   starts_with(instruction.opcode, "st.local")) {
            operation.opcode = OpCode::kStore;
            if (instruction.operands.size() < 2) return fail(&instruction, "malformed store");
            operation.operands.push_back(
                source_operand(0, Type::pointer(ptx_scalar_type(instruction.opcode),
                                                AddressSpace::kDevice)));
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
            operation.operands.push_back(source_operand(1, operation.result_types.front()));
            operation.operands.push_back(source_operand(2, operation.result_types.front()));
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
        } else if (root == "atom") {
            operation.opcode = OpCode::kAtomic;
            operation.memory_scope = memory_scope_from_opcode(instruction.opcode);
            operation.memory_ordering = memory_ordering_from_opcode(instruction.opcode);
            for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                operation.operands.push_back(source_operand(i, ptx_scalar_type(instruction.opcode)));
            }
        } else if (root == "shfl") {
            operation.opcode = OpCode::kShuffle;
            for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                operation.operands.push_back(source_operand(i, ptx_scalar_type(instruction.opcode)));
            }
        } else if (root == "vote") {
            operation.opcode =
                instruction.opcode.find(".ballot.") != std::string::npos ? OpCode::kBallot : OpCode::kVote;
            for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                operation.operands.push_back(source_operand(i, Type::predicate()));
            }
        } else if (root == "redux") {
            operation.opcode = OpCode::kReduction;
            for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                operation.operands.push_back(source_operand(i, ptx_scalar_type(instruction.opcode)));
            }
        } else if (root == "cvt") {
            operation.opcode = OpCode::kConvert;
            if (instruction.opcode.find(".f64.f32") != std::string::npos ||
                instruction.opcode.find(".f32.f64") != std::string::npos) {
                operation.result_types.front() = Type::floating(32);
                value_types[operation.results.front()] = Type::floating(32);
                operation.operands.push_back(
                    bit_container_operand(1, Type::floating(32)));
                result.module.semantic_quality = SemanticQuality::kToleranceBounded;
                const std::string caveat =
                    "CUDA float-frexp double ABI is normalized at its float boundary";
                if (std::find(result.module.semantic_caveats.begin(),
                              result.module.semantic_caveats.end(), caveat) ==
                    result.module.semantic_caveats.end()) {
                    result.module.semantic_caveats.push_back(caveat);
                }
            } else {
                operation.operands.push_back(
                    source_operand(1, operation.result_types.front()));
            }
        } else if (root == "rcp") {
            operation.opcode = OpCode::kDiv;
            operation.operands.push_back(
                Operand::immediate("1.0", operation.result_types.front()));
            operation.operands.push_back(source_operand(1, operation.result_types.front()));
        } else if (root == "not") {
            const Type type = ptx_scalar_type(instruction.opcode);
            operation.opcode = OpCode::kBitXor;
            operation.operands.push_back(bit_container_operand(1, type));
            operation.operands.push_back(Operand::immediate(
                type.bit_width == 64 ? "18446744073709551615" :
                type.bit_width == 16 ? "65535" : "4294967295",
                type));
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
            const auto signature = cuda_builtin_signature(callee);
            if (!signature.has_value()) {
                return fail(&instruction, "device call target '" + callee +
                                              "' has no typed PTX signature");
            }
            const std::vector<std::string> argument_names =
                grouped_names(instruction.operands[arguments_index]);
            if (argument_names.size() != signature->argument_types.size()) {
                return fail(&instruction, "device call target '" + callee +
                                              "' received the wrong argument count");
            }
            operation.opcode = OpCode::kCall;
            operation.attributes["callee"] = signature->metal_name;
            operation.attributes["builtin"] = "true";
            for (std::size_t i = 0; i < argument_names.size(); ++i) {
                const auto slot = call_parameter_slots.find(argument_names[i]);
                if (slot == call_parameter_slots.end()) {
                    return fail(&instruction, "call parameter slot '" + argument_names[i] +
                                                  "' was not initialized");
                }
                Operand argument = slot->second;
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
            }
            if (has_return) {
                const ValueId result_value = builder.next_value();
                operation.results.push_back(result_value);
                operation.result_types.push_back(signature->return_type);
                value_types[result_value] = signature->return_type;
                call_return_slots[grouped_names(instruction.operands[0]).front()] =
                    Operand::value_ref(result_value, signature->return_type);
            }
            if (signature->tolerance_bounded) {
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
                operation.opcode = OpCode::kPointerOffset;
                for (const Operand& operand : operation.operands) {
                    if (operand.kind != OperandKind::kValue) continue;
                    const auto provenance = function->pointer_provenance.find(operand.value);
                    if (provenance != function->pointer_provenance.end()) {
                        function->pointer_provenance[operation.results.front()] = provenance->second;
                        break;
                    }
                }
            }
            if (root == "mad") operation.attributes["combined"] = "mul_add";
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
        function.is_kernel = true;
        function.return_type = Type::void_type();
        function.kernel_abi = KernelAbi{};

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
            const std::uint32_t size = type_size(type);
            function.kernel_abi->arguments.push_back({
                .name = parameter.name,
                .kind = type.is_pointer() ? ArgumentKind::kPointer : ArgumentKind::kScalar,
                .type = type,
                .size = size,
                .alignment = std::min<std::uint32_t>(size, 8),
                .address_space = type.is_pointer() ? type.address_space : AddressSpace::kConstant,
                .binding_indices = {static_cast<std::uint32_t>(index)},
            });
            function.kernel_abi->bindings.push_back({
                .kind = type.is_pointer() ? BindingKind::kBuffer : BindingKind::kBytes,
                .binding_index = static_cast<std::uint32_t>(index),
                .logical_argument_index = static_cast<std::uint32_t>(index),
                .type = type,
                .size = size,
                .alignment = std::min<std::uint32_t>(size, 8),
            });
        }

        if (!module_constant_symbols.empty()) {
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
            }
            block.operations.push_back(std::move(terminator));
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
    importer.result.module.global_threadgroups = scan_threadgroup_globals(ptx);
    for (const GlobalThreadgroup& global : importer.result.module.global_threadgroups) {
        importer.threadgroup_symbols.insert(global.name);
    }
    for (LocalDepot depot : scan_local_depots(ptx)) {
        importer.local_depots.emplace(depot.name, std::move(depot));
    }
    importer.implicit_definitions = scan_implicit_definitions(ptx);

    cumetal::ptx::ParseOptions parse_options;
    parse_options.strict = options.strict;
    const auto parsed = cumetal::ptx::parse_ptx(ptx, parse_options);
    if (!parsed.ok) {
        importer.result.error = parsed.error;
        return importer.result;
    }
    importer.result.warnings = parsed.warnings;
    if (!importer.select_entry(parsed, options)) return importer.result;
    for (const ModuleConstantSymbol& symbol : scan_module_constant_symbols(ptx)) {
        importer.module_constant_buffer_size =
            std::max(importer.module_constant_buffer_size,
                     symbol.offset + symbol.byte_size);
        bool referenced = false;
        for (const Instruction& instruction : importer.entry->instructions) {
            referenced = std::any_of(
                instruction.operands.begin(), instruction.operands.end(),
                [&](const std::string& operand) {
                    return parameter_name_from_operand(operand) == symbol.name;
                });
            if (referenced) break;
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
    importer.infer_register_types();
    importer.build_cfg();
    importer.allocate_values();
    if (!importer.construct_ssa()) return importer.result;
    if (!importer.materialize_function()) return importer.result;

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
