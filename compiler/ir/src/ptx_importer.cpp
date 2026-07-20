#include "cumetal/ir/ptx_importer.h"

#include "cumetal/ptx/parser.h"

#include <algorithm>
#include <cctype>
#include <map>
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
               starts_with(name, "%clock");
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
                } else if (starts_with(instruction.opcode, "ld.param") &&
                           instruction.operands.size() >= 2) {
                    const auto parameter = parameter_types.find(
                        parameter_name_from_operand(instruction.operands[1]));
                    if (parameter != parameter_types.end()) inferred = parameter->second;
                } else if (root == "cvta") {
                    inferred = Type::pointer(Type::integer(8), AddressSpace::kDevice);
                } else if ((root == "add" || root == "mov") && instruction.operands.size() >= 2) {
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

        bool changed = true;
        for (int iteration = 0; iteration < 64 && changed; ++iteration) {
            changed = false;
            for (std::size_t block_index = 0; block_index < raw_blocks.size(); ++block_index) {
                const RawBlock& block = raw_blocks[block_index];
                std::unordered_map<std::string, ValueId> next_in;
                if (!block.predecessors.empty()) {
                    std::set<std::string> registers;
                    for (std::size_t predecessor : block.predecessors) {
                        for (const auto& [name, value] : outgoing[predecessor]) {
                            (void)value;
                            registers.insert(name);
                        }
                    }
                    for (const std::string& name : registers) {
                        ValueId common = kInvalidValue;
                        bool all_present = true;
                        bool all_equal = true;
                        for (std::size_t predecessor : block.predecessors) {
                            const auto value = outgoing[predecessor].find(name);
                            if (value == outgoing[predecessor].end()) {
                                all_present = false;
                                continue;
                            }
                            if (common == kInvalidValue) {
                                common = value->second;
                            } else if (common != value->second) {
                                all_equal = false;
                            }
                        }
                        if (all_present && all_equal) {
                            next_in[name] = common;
                        } else if (block.uses_before_definition.contains(name) ||
                                   block_arguments[block_index].contains(name)) {
                            auto& argument = block_arguments[block_index][name];
                            if (argument == kInvalidValue) {
                                argument = builder.next_value();
                                const auto type = register_types.find(name);
                                value_types[argument] =
                                    type == register_types.end() ? Type::integer(32) : type->second;
                            }
                            next_in[name] = argument;
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
                if (block_index == 0 || !incoming[block_index].contains(name)) {
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
        return Operand::immediate(trim(token), fallback_type);
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

        if (starts_with(instruction.opcode, "ld.param")) {
            if (instruction.operands.size() < 2 || operation.results.empty()) {
                return fail(&instruction, "malformed ld.param instruction");
            }
            const std::string name = parameter_name_from_operand(instruction.operands[1]);
            const auto argument = parameter_values.find(name);
            if (argument == parameter_values.end()) {
                return fail(&instruction, "unknown kernel parameter '" + name + "'");
            }
            operation.opcode = OpCode::kParameter;
            operation.operands.push_back(
                Operand::value_ref(argument->second, parameter_types[name]));
            operation.attributes["parameter"] = name;
            if (operation.result_types.front().is_pointer()) {
                function->pointer_provenance[operation.results.front()] =
                    function->pointer_provenance[argument->second];
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
        } else if (root == "mov") {
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
            operation.operands.push_back(source_operand(1, Type::pointer(
                                                               operation.result_types.front(),
                                                               AddressSpace::kDevice)));
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
            operation.operands.push_back(source_operand(1, ptx_scalar_type(instruction.opcode)));
            operation.attributes["address"] = instruction.operands[0];
            operation.attributes["alignment"] =
                std::to_string(type_size(ptx_scalar_type(instruction.opcode)));
        } else if (root == "setp") {
            operation.opcode = OpCode::kCompare;
            operation.operands.push_back(source_operand(1, ptx_scalar_type(instruction.opcode)));
            operation.operands.push_back(source_operand(2, ptx_scalar_type(instruction.opcode)));
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
            operation.operands.push_back(source_operand(1, operation.result_types.front()));
        } else if (root == "call") {
            return fail(&instruction, "device calls are not yet supported by the PTX importer");
        } else {
            operation.opcode = arithmetic_opcode(root);
            if (operation.opcode == OpCode::kInvalid) {
                return fail(&instruction, "PTX opcode '" + instruction.opcode +
                                              "' has no CuMetal IR normalization");
            }
            const std::size_t first_source = destinations.empty() ? 0 : 1;
            for (std::size_t i = first_source; i < instruction.operands.size(); ++i) {
                operation.operands.push_back(
                    source_operand(i, operation.result_types.empty()
                                          ? ptx_scalar_type(instruction.opcode)
                                          : operation.result_types.front()));
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

    cumetal::ptx::ParseOptions parse_options;
    parse_options.strict = options.strict;
    const auto parsed = cumetal::ptx::parse_ptx(ptx, parse_options);
    if (!parsed.ok) {
        importer.result.error = parsed.error;
        return importer.result;
    }
    importer.result.warnings = parsed.warnings;
    if (!importer.select_entry(parsed, options)) return importer.result;
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
