#include "trap_reporting.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace cumetal::metal {
namespace {

bool unsupported_in_guarded_graph(ir::OpCode opcode) {
    using ir::OpCode;
    switch (opcode) {
    case OpCode::kPrintf:
    case OpCode::kBarrier:
    case OpCode::kMetalBarrier:
    case OpCode::kShuffle:
    case OpCode::kMetalShuffle:
    case OpCode::kBallot:
    case OpCode::kMetalBallot:
    case OpCode::kVote:
    case OpCode::kMetalVote:
    case OpCode::kReduction:
    case OpCode::kMetalReduction:
        return true;
    default:
        return false;
    }
}

bool is_bounded_builtin(const ir::Operation& operation) {
    if (operation.opcode != ir::OpCode::kCall ||
        !operation.attributes.contains("builtin") ||
        operation.attributes.at("builtin") != "true" ||
        !operation.attributes.contains("callee") ||
        operation.result_types.size() != 1) {
        return false;
    }
    const std::string& name = operation.attributes.at("callee");
    const ir::Type& type = operation.result_types.front();
    const bool bit_count = name == "clz" || name == "popcount";
    const bool unary = bit_count || name == "__cumetal_signed_abs";
    return (unary || name == "min" || name == "max") &&
           type.kind == ir::TypeKind::kInteger &&
           (!bit_count || type.bit_width == 32 || type.bit_width == 64) &&
           (type.bit_width == 8 || type.bit_width == 16 ||
            type.bit_width == 32 || type.bit_width == 64) &&
           operation.operands.size() == (unary ? 1u : 2u) &&
           std::all_of(operation.operands.begin(), operation.operands.end(),
                       [&](const ir::Operand& operand) {
                           return operand.type == type;
                       });
}

bool cfg_is_cyclic(const ir::Function& function) {
    std::unordered_map<ir::BlockId, std::size_t> incoming;
    for (const ir::BasicBlock& block : function.blocks)
        incoming[block.id] = 0;
    for (const ir::BasicBlock& block : function.blocks) {
        for (const ir::Successor& successor :
             block.operations.back().successors) {
            ++incoming[successor.block];
        }
    }
    std::vector<ir::BlockId> ready;
    for (const auto& [id, count] : incoming) {
        if (count == 0)
            ready.push_back(id);
    }
    std::size_t visited = 0;
    while (!ready.empty()) {
        const ir::BlockId id = ready.back();
        ready.pop_back();
        ++visited;
        for (const ir::Successor& successor :
             function.find_block(id)->operations.back().successors) {
            if (--incoming[successor.block] == 0)
                ready.push_back(successor.block);
        }
    }
    return visited != function.blocks.size();
}

} // namespace

std::string guarded_trap_helper_name(std::string_view name) {
    return std::string(name) + "__cm_trap_guarded";
}

bool analyze_trap_call_graph(const ir::Module& module, TrapCallGraph* graph,
                             std::string* error) {
    using namespace ir;
    graph->trapping.clear();
    graph->guarded.clear();
    graph->ordinary.clear();
    graph->dispatch.clear();

    std::unordered_map<std::string, const Function*> functions;
    bool has_kernel = false;
    for (const Function& function : module.functions) {
        functions.emplace(function.name, &function);
        has_kernel |= function.is_kernel;
        for (const BasicBlock& block : function.blocks) {
            for (const Operation& operation : block.operations) {
                if (operation.opcode == OpCode::kTrap) {
                    graph->trapping.insert(function.name);
                }
            }
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (const Function& function : module.functions) {
            if (graph->trapping.contains(function.name))
                continue;
            for (const BasicBlock& block : function.blocks) {
                for (const Operation& operation : block.operations) {
                    if (operation.opcode == OpCode::kCall &&
                        operation.attributes.contains("callee") &&
                        graph->trapping.contains(
                            operation.attributes.at("callee"))) {
                        changed |= graph->trapping.insert(function.name).second;
                    }
                }
            }
        }
    }

    std::unordered_set<std::string> trap_reachable;
    std::vector<const Function*> pending;
    for (const Function& function : module.functions) {
        if (function.is_kernel && graph->trapping.contains(function.name)) {
            pending.push_back(&function);
        }
    }
    while (!pending.empty()) {
        const Function* function = pending.back();
        pending.pop_back();
        if (!trap_reachable.insert(function->name).second)
            continue;
        for (const BasicBlock& block : function->blocks) {
            for (const Operation& operation : block.operations) {
                if (unsupported_in_guarded_graph(operation.opcode)) {
                    *error = "trap reporting through device helpers does not "
                             "support " +
                             operation.location.str() +
                             " barriers, collectives, or printf";
                    return false;
                }
                if (operation.opcode != OpCode::kCall)
                    continue;
                const auto callee = operation.attributes.find("callee");
                if (callee == operation.attributes.end()) {
                    *error = "trap reporting requires direct device calls";
                    return false;
                }
                const auto target = functions.find(callee->second);
                if (target != functions.end() && !target->second->is_kernel &&
                    !target->second->blocks.empty()) {
                    pending.push_back(target->second);
                    continue;
                }
                if (!is_bounded_builtin(operation)) {
                    *error = "trap reporting requires defined device helper '" +
                             callee->second + "'";
                    return false;
                }
            }
        }
    }

    graph->guarded = graph->trapping;
    for (const Function& function : module.functions) {
        if (trap_reachable.contains(function.name) && cfg_is_cyclic(function)) {
            graph->guarded.insert(function.name);
            graph->dispatch.insert(function.name);
        }
    }
    changed = true;
    while (changed) {
        changed = false;
        for (const Function& function : module.functions) {
            if (!trap_reachable.contains(function.name) ||
                graph->guarded.contains(function.name)) {
                continue;
            }
            for (const BasicBlock& block : function.blocks) {
                for (const Operation& operation : block.operations) {
                    if (operation.opcode == OpCode::kCall &&
                        operation.attributes.contains("callee") &&
                        graph->guarded.contains(
                            operation.attributes.at("callee"))) {
                        changed |= graph->guarded.insert(function.name).second;
                    }
                }
            }
        }
    }

    for (const Function& function : module.functions) {
        if (function.is_kernel && !graph->trapping.contains(function.name)) {
            pending.push_back(&function);
        }
    }
    while (!pending.empty()) {
        const Function* function = pending.back();
        pending.pop_back();
        if (!graph->ordinary.insert(function->name).second)
            continue;
        for (const BasicBlock& block : function->blocks) {
            for (const Operation& operation : block.operations) {
                if (operation.opcode != OpCode::kCall ||
                    !operation.attributes.contains("callee")) {
                    continue;
                }
                const auto target =
                    functions.find(operation.attributes.at("callee"));
                if (target != functions.end())
                    pending.push_back(target->second);
            }
        }
    }
    for (const Function& function : module.functions) {
        if (trap_reachable.contains(function.name) &&
            !graph->guarded.contains(function.name)) {
            graph->ordinary.insert(function.name);
        }
    }
    if (!has_kernel) {
        for (const Function& function : module.functions) {
            graph->ordinary.insert(function.name);
        }
    }
    for (const Function& function : module.functions) {
        if (!function.is_kernel && graph->guarded.contains(function.name) &&
            functions.contains(guarded_trap_helper_name(function.name))) {
            *error = "trap helper name collision for '" + function.name + "'";
            return false;
        }
    }
    return true;
}

} // namespace cumetal::metal
