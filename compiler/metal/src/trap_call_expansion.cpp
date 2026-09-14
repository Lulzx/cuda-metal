#include "trap_call_expansion.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace cumetal::metal {
namespace {
using namespace ir;

// Inlining is a cancellation legalization, not an unrestricted optimizer.
constexpr std::size_t kMaxCalls = 1024;
constexpr std::size_t kMaxBlocks = 4096;
constexpr std::size_t kMaxOperations = 262144;

void remove_unreachable_blocks(Function* function) {
    std::unordered_set<BlockId> reachable;
    std::vector<BlockId> pending{function->blocks.front().id};
    while (!pending.empty()) {
        const BlockId id = pending.back();
        pending.pop_back();
        if (!reachable.insert(id).second) continue;
        for (const auto& successor : function->find_block(id)->operations.back().successors)
            pending.push_back(successor.block);
    }
    std::erase_if(function->blocks, [&](const BasicBlock& block) {
        return !reachable.contains(block.id);
    });
}

bool expand_call(Function* caller, std::size_t block_index, std::size_t operation_index,
                 const Function& callee, ValueId* next_value, BlockId* next_block,
                 std::string* error) {
    const Operation call = caller->blocks[block_index].operations[operation_index];
    const bool returns_value = callee.return_type.kind != TypeKind::kVoid;
    if (callee.is_kernel || callee.blocks.empty() || !callee.blocks.front().arguments.empty() ||
        call.operands.size() != callee.arguments.size() ||
        call.results.size() != (returns_value ? 1u : 0u) ||
        call.result_types.size() != call.results.size() ||
        (returns_value && call.result_types.front() != callee.return_type)) {
        *error = "trap call expansion requires a defined helper with matching argument/return ABI";
        return false;
    }
    for (std::size_t i = 0; i < call.operands.size(); ++i) {
        if (call.operands[i].type != callee.arguments[i].type) {
            *error = "trap call expansion requires matching argument types";
            return false;
        }
    }
    const std::string prefix = "cm_inline_" + std::to_string(*next_block) + "_";
    std::unordered_map<ValueId, ValueId> values;
    std::unordered_map<BlockId, BlockId> blocks;
    for (const auto& argument : callee.arguments) values[argument.value] = (*next_value)++;
    for (const auto& block : callee.blocks) {
        blocks[block.id] = (*next_block)++;
        for (const auto& argument : block.arguments) values[argument.value] = (*next_value)++;
        for (const auto& operation : block.operations)
            for (const ValueId value : operation.results) values[value] = (*next_value)++;
    }
    for (const auto& [old_value, new_value] : values) {
        if (callee.pointer_provenance.contains(old_value))
            caller->pointer_provenance[new_value] = callee.pointer_provenance.at(old_value);
        if (callee.generic_pointer_values.contains(old_value))
            caller->generic_pointer_values.insert(new_value);
        if (callee.generic_null_pointer_values.contains(old_value))
            caller->generic_null_pointer_values.insert(new_value);
    }
    BasicBlock continuation;
    continuation.id = (*next_block)++;
    continuation.name = prefix + "return";
    for (std::size_t i = 0; i < call.results.size(); ++i)
        continuation.arguments.push_back({call.results[i], call.result_types[i], "result"});
    auto& original = caller->blocks[block_index];
    continuation.operations.assign(original.operations.begin() + operation_index + 1,
                                   original.operations.end());
    original.operations.resize(operation_index);

    // Branch arguments use value IDs, so materialize any immediate or symbol at
    // its original evaluation site. No arithmetic or address conversion is added.
    auto materialize = [&](Operand operand, std::vector<Operation>* operations) {
        if (operand.kind == OperandKind::kValue) return operand.value;
        Operation move;
        move.opcode = OpCode::kConvert;
        move.location = call.location;
        move.results = {(*next_value)++};
        move.result_types = {operand.type};
        move.operands = {operand};
        move.attributes["bitcast"] = "true";
        const ValueId value = move.results.front();
        operations->push_back(std::move(move));
        return value;
    };
    Operation enter;
    enter.opcode = OpCode::kBranch;
    enter.location = call.location;
    enter.successors.push_back({blocks.at(callee.blocks.front().id), {}});
    for (const auto& operand : call.operands)
        enter.successors.front().arguments.push_back(materialize(operand, &original.operations));
    original.operations.push_back(std::move(enter));

    std::vector<BasicBlock> clones = callee.blocks;
    for (auto& block : clones) {
        block.id = blocks.at(block.id);
        block.name = prefix + block.name;
        for (auto& argument : block.arguments) argument.value = values.at(argument.value);
        for (auto& operation : block.operations) {
            for (auto& value : operation.results) value = values.at(value);
            for (auto& operand : operation.operands)
                if (operand.kind == OperandKind::kValue) operand.value = values.at(operand.value);
            for (auto& successor : operation.successors) {
                successor.block = blocks.at(successor.block);
                for (auto& value : successor.arguments) value = values.at(value);
            }
        }
        if (block.operations.back().opcode == OpCode::kReturn) {
            Operation ret = std::move(block.operations.back());
            block.operations.pop_back();
            if (ret.operands.size() != call.results.size() ||
                (returns_value && ret.operands.front().type != callee.return_type)) {
                *error = "trap call expansion requires matching return operands";
                return false;
            }
            Operation leave;
            leave.opcode = OpCode::kBranch;
            leave.location = ret.location;
            leave.successors.push_back({continuation.id, {}});
            for (const auto& operand : ret.operands)
                leave.successors.front().arguments.push_back(materialize(operand, &block.operations));
            block.operations.push_back(std::move(leave));
        }
        // kTrap deliberately remains a trap: it never reaches the continuation.
    }
    for (const auto& argument : callee.arguments)
        clones.front().arguments.push_back({values.at(argument.value), argument.type, prefix + argument.name});
    caller->blocks.push_back(std::move(continuation));
    for (auto& block : clones) caller->blocks.push_back(std::move(block));
    return true;
}
}  // namespace

bool is_bounded_trap_builtin(const ir::Operation& operation) {
    if (operation.opcode != ir::OpCode::kCall ||
        !operation.attributes.contains("builtin") || operation.attributes.at("builtin") != "true" ||
        !operation.attributes.contains("callee") || operation.result_types.size() != 1) return false;
    const auto& name = operation.attributes.at("callee");
    const auto& type = operation.result_types.front();
    return (name == "min" || name == "max" || name == "__cumetal_signed_abs") &&
           type.kind == ir::TypeKind::kInteger &&
           (type.bit_width == 8 || type.bit_width == 16 || type.bit_width == 32 || type.bit_width == 64) &&
           operation.operands.size() == (name == "__cumetal_signed_abs" ? 1u : 2u) &&
           std::all_of(operation.operands.begin(), operation.operands.end(),
                       [&](const ir::Operand& operand) { return operand.type == type; });
}

std::unordered_set<std::string> find_bounded_trap_helpers(const ir::Module& module) {
    using namespace ir;
    std::unordered_set<std::string> bounded;
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& function : module.functions) {
            if (function.is_kernel || bounded.contains(function.name) || function.blocks.empty()) continue;
            bool safe = true;
            std::unordered_map<BlockId, std::size_t> incoming;
            for (const auto& block : function.blocks) incoming[block.id] = 0;
            for (const auto& block : function.blocks) {
                for (const auto& operation : block.operations) {
                    switch (operation.opcode) {
                        case OpCode::kTrap: case OpCode::kPrintf:
                        case OpCode::kBarrier: case OpCode::kMetalBarrier:
                        case OpCode::kAtomic: case OpCode::kMetalAtomic:
                        case OpCode::kShuffle: case OpCode::kMetalShuffle:
                        case OpCode::kBallot: case OpCode::kMetalBallot:
                        case OpCode::kVote: case OpCode::kMetalVote:
                        case OpCode::kReduction: case OpCode::kMetalReduction:
                            safe = false;
                            break;
                        case OpCode::kCall:
                            if (!is_bounded_trap_builtin(operation) &&
                                (!operation.attributes.contains("callee") ||
                                 !bounded.contains(operation.attributes.at("callee")))) safe = false;
                            break;
                        default: break;
                    }
                    for (const auto& successor : operation.successors) ++incoming[successor.block];
                }
            }
            if (!safe) continue;
            std::vector<BlockId> ready;
            for (const auto& [id, count] : incoming) if (count == 0) ready.push_back(id);
            std::size_t visited = 0;
            while (!ready.empty()) {
                const BlockId id = ready.back(); ready.pop_back(); ++visited;
                for (const auto& successor : function.find_block(id)->operations.back().successors)
                    if (--incoming[successor.block] == 0) ready.push_back(successor.block);
            }
            if (visited == function.blocks.size()) changed |= bounded.insert(function.name).second;
        }
    }
    return bounded;
}

bool expand_trap_call_graphs(ir::Module* module, std::string* error) {
    using namespace ir;
    const auto bounded_helpers = find_bounded_trap_helpers(*module);
    std::unordered_map<std::string, const Function*> functions;
    std::unordered_set<std::string> trapping;
    ValueId next_value = 1;
    BlockId next_block = 1;
    for (const auto& function : module->functions) {
        functions[function.name] = &function;
        for (const auto& argument : function.arguments) next_value = std::max(next_value, argument.value + 1);
        for (const auto& block : function.blocks) {
            next_block = std::max(next_block, block.id + 1);
            for (const auto& argument : block.arguments) next_value = std::max(next_value, argument.value + 1);
            for (const auto& operation : block.operations) {
                if (operation.opcode == OpCode::kTrap) trapping.insert(function.name);
                for (const auto value : operation.results) next_value = std::max(next_value, value + 1);
            }
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& function : module->functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations)
                    if (operation.opcode == OpCode::kCall && operation.attributes.contains("callee") &&
                        trapping.contains(operation.attributes.at("callee")))
                        changed |= trapping.insert(function.name).second;
    }
    // Build complete replacements before mutating the module. A rejected ABI
    // or expansion budget therefore leaves all original functions intact.
    std::vector<std::pair<std::size_t, Function>> replacements;
    for (std::size_t index = 0; index < module->functions.size(); ++index) {
        const auto& original = module->functions[index];
        if (!original.is_kernel || !trapping.contains(original.name)) continue;
        Function kernel = original;
        std::size_t expanded = 0;
        bool found = true;
        while (found) {
            found = false;
            for (std::size_t b = 0; b < kernel.blocks.size() && !found; ++b) {
                for (std::size_t o = 0; o < kernel.blocks[b].operations.size(); ++o) {
                    const auto& operation = kernel.blocks[b].operations[o];
                    if (operation.opcode != OpCode::kCall || is_bounded_trap_builtin(operation)) continue;
                    const auto name = operation.attributes.find("callee");
                    if (name != operation.attributes.end() && bounded_helpers.contains(name->second)) continue;
                    if (name == operation.attributes.end() || !functions.contains(name->second) ||
                        (operation.attributes.contains("builtin") && operation.attributes.at("builtin") == "true")) {
                        *error = "trap call expansion requires direct defined non-builtin helpers: " +
                            (name == operation.attributes.end() ? std::string("<indirect>") : name->second);
                        return false;
                    }
                    const auto& callee = *functions.at(name->second);
                    std::size_t operations = 0;
                    for (const auto& block : kernel.blocks) operations += block.operations.size();
                    for (const auto& block : callee.blocks) operations += block.operations.size();
                    if (++expanded > kMaxCalls || kernel.blocks.size() + callee.blocks.size() + 1 > kMaxBlocks ||
                        operations + operation.operands.size() + callee.blocks.size() + 1 > kMaxOperations) {
                        *error = "trap call expansion exceeds bounded CFG size (calls=" + std::to_string(expanded) +
                            ", blocks=" + std::to_string(kernel.blocks.size() + callee.blocks.size() + 1) +
                            ", operations=" + std::to_string(operations) + ")";
                        return false;
                    }
                    if (!expand_call(&kernel, b, o, callee, &next_value, &next_block, error)) return false;
                    remove_unreachable_blocks(&kernel);
                    found = true;
                    break;
                }
            }
        }
        replacements.emplace_back(index, std::move(kernel));
    }
    for (auto& [index, kernel] : replacements) module->functions[index] = std::move(kernel);
    return true;
}
}  // namespace cumetal::metal
