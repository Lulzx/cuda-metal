#include "record_contexts.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace cumetal::metal::detail {
namespace {
using namespace ir;
struct Fact {
    enum State { unknown, constant, varying } state = unknown;
    std::uint64_t bits = 0;
    bool operator==(const Fact&) const = default;
};
Fact meet(Fact a, Fact b) {
    if (a.state == Fact::unknown) return b;
    if (b.state == Fact::unknown) return a;
    return a == b ? a : Fact{Fact::varying};
}
unsigned width(const Type& type) {
    if (type.kind == TypeKind::kPredicate) return 1;
    return type.kind == TypeKind::kInteger && type.bit_width <= 64 ? type.bit_width : 0;
}
std::uint64_t mask(unsigned bits) { return bits == 64 ? UINT64_MAX : (std::uint64_t{1} << bits) - 1; }
bool ordinary_load(const Operation& op) {
    const auto ptx = op.attributes.find("ptx_opcode");
    return op.opcode == OpCode::kLoad && !op.attributes.contains("guard_operand") &&
        op.memory_ordering == MemoryOrdering::kNone && ptx != op.attributes.end() &&
        ptx->second.starts_with("ld.") && ptx->second.find("volatile") == std::string::npos &&
        ptx->second.find("acquire") == std::string::npos;
}
Fact immediate(const Operand& operand) {
    const auto bits = width(operand.type);
    if (operand.kind != OperandKind::kImmediate || !bits) return {Fact::varying};
    try {
        std::size_t used = 0;
        const auto n = std::stoull(operand.text, &used, 0);
        if (used == operand.text.size()) return {Fact::constant, n & mask(bits)};
    } catch (...) {}
    return {Fact::varying};
}

// Executable-edge constant propagation: a loop phi sees only proved executable
// inputs. Facts grow monotonically; no guessed zero initializes a cycle.
bool simplify(Function& function, const std::vector<ValueId>& zero_loads) {
    std::size_t work = 4'000'000;
    const auto charge = [&]() { if (!work) return false; --work; return true; };
    std::unordered_map<ValueId, Fact> facts;
    std::unordered_map<BlockId, std::size_t> blocks;
    for (std::size_t b = 0; b < function.blocks.size(); ++b) blocks[function.blocks[b].id] = b;
    for (const auto& a : function.arguments) facts[a.value] = {Fact::varying};
    const std::unordered_set<ValueId> zeros(zero_loads.begin(), zero_loads.end());
    const auto value = [&](const Operand& operand) {
        return operand.kind == OperandKind::kValue ? facts[operand.value] : immediate(operand);
    };
    std::vector<bool> executable(function.blocks.size(), false);
    if (executable.empty()) return false;
    executable[0] = true;
    std::set<std::pair<std::size_t, std::size_t>> edges;
    bool changed = true;
    while (changed) {
        changed = false;
        const auto merge = [&](ValueId id, Fact fact) {
            const auto next = meet(facts[id], fact);
            if (next != facts[id]) { facts[id] = next; changed = true; }
        };
        for (std::size_t b = 0; b < function.blocks.size(); ++b) {
            if (!charge()) return false;
            if (!executable[b]) continue;
            for (const auto& op : function.blocks[b].operations) {
                if (!charge()) return false;
                Fact result{Fact::varying};
                const unsigned bits = op.result_types.size() == 1 ? width(op.result_types[0]) : 0;
                if (op.results.size() == 1 && bits && !op.attributes.contains("guard_operand")) {
                    if (zeros.contains(op.results[0]) && ordinary_load(op)) result = {Fact::constant, 0};
                    else if (op.opcode == OpCode::kConstant && op.operands.size() == 1) result = value(op.operands[0]);
                    else if (op.opcode == OpCode::kParameter && op.operands.size() == 1 &&
                             op.operands[0].type == op.result_types[0]) result = value(op.operands[0]);
                    else if (op.opcode == OpCode::kConvert && op.operands.size() == 1 &&
                             op.operands[0].type == op.result_types[0] && op.attributes.contains("ptx_opcode") &&
                             op.attributes.at("ptx_opcode").starts_with("mov.")) result = value(op.operands[0]);
                    else if (op.opcode == OpCode::kSelect && op.operands.size() == 3) {
                        const auto c = value(op.operands[0]);
                        result = c.state == Fact::constant ? value(op.operands[c.bits ? 1 : 2])
                            : c.state == Fact::unknown ? Fact{} : meet(value(op.operands[1]), value(op.operands[2]));
                    } else if (op.operands.size() == 2 && width(op.operands[0].type) &&
                               (op.opcode == OpCode::kCompare || op.result_types[0] == op.operands[0].type) &&
                               (!op.attributes.contains("ptx_opcode") ||
                                (op.attributes.at("ptx_opcode").find(".sat") == std::string::npos &&
                                 op.attributes.at("ptx_opcode").find(".cc") == std::string::npos)) &&
                               op.operands[0].type == op.operands[1].type) {
                        const auto a = value(op.operands[0]), c = value(op.operands[1]);
                        const bool az = a.state == Fact::constant && a.bits == 0;
                        const bool cz = c.state == Fact::constant && c.bits == 0;
                        if (op.opcode == OpCode::kBitAnd && (az || cz)) result = {Fact::constant, 0};
                        else if (op.opcode == OpCode::kCompare && op.attributes.contains("predicate") &&
                                 !op.attributes.contains("signed") &&
                                 ((az && (op.attributes.at("predicate") == "gt" || op.attributes.at("predicate") == "le")) ||
                                  (cz && (op.attributes.at("predicate") == "lt" || op.attributes.at("predicate") == "ge")))) {
                            const auto& predicate = op.attributes.at("predicate");
                            result = {Fact::constant, predicate == "le" || predicate == "ge"};
                        } else if (a.state == Fact::unknown || c.state == Fact::unknown) result = {};
                        else if (a.state == Fact::constant && c.state == Fact::constant) {
                            if (op.opcode == OpCode::kAdd) result = {Fact::constant, (a.bits + c.bits) & mask(bits)};
                            else if (op.opcode == OpCode::kSub) result = {Fact::constant, (a.bits - c.bits) & mask(bits)};
                            else if (op.opcode == OpCode::kBitAnd) result = {Fact::constant, a.bits & c.bits};
                            else if (op.opcode == OpCode::kBitOr) result = {Fact::constant, a.bits | c.bits};
                            else if (op.opcode == OpCode::kBitXor) result = {Fact::constant, a.bits ^ c.bits};
                            else if (op.opcode == OpCode::kCompare && op.attributes.contains("predicate")) {
                                const auto& p = op.attributes.at("predicate");
                                if (p == "eq" || p == "ne") result = {Fact::constant, (a.bits == c.bits) == (p == "eq")};
                                else if (!op.attributes.contains("signed")) {
                                    if (p == "lt") result = {Fact::constant, a.bits < c.bits};
                                    if (p == "le") result = {Fact::constant, a.bits <= c.bits};
                                    if (p == "gt") result = {Fact::constant, a.bits > c.bits};
                                    if (p == "ge") result = {Fact::constant, a.bits >= c.bits};
                                }
                            }
                        }
                    }
                    merge(op.results[0], result);
                } else for (const auto id : op.results) merge(id, result);
                if (!op.is_terminator()) continue;
                Fact condition{Fact::varying};
                if (op.opcode == OpCode::kCondBranch && op.operands.size() == 1) condition = value(op.operands[0]);
                for (std::size_t e = 0; e < op.successors.size(); ++e) {
                    if (!charge()) return false;
                    if (op.opcode == OpCode::kCondBranch &&
                        (condition.state == Fact::unknown ||
                         (condition.state == Fact::constant && e != (condition.bits ? 0U : 1U)))) continue;
                    const auto& edge = op.successors[e];
                    const auto target = blocks.at(edge.block);
                    changed |= edges.insert({b, e}).second;
                    if (!executable[target]) { executable[target] = true; changed = true; }
                    const auto& arguments = function.blocks[target].arguments;
                    if (arguments.size() != edge.arguments.size()) return false;
                    for (std::size_t a = 0; a < arguments.size(); ++a) merge(arguments[a].value, facts[edge.arguments[a]]);
                }
            }
        }
    }
    // An unresolved executable branch is not permission to delete its paths.
    for (std::size_t b = 0; b < function.blocks.size(); ++b) if (executable[b]) {
        auto& block = function.blocks[b];
        auto& last = block.operations.back();
        if (last.opcode == OpCode::kCondBranch && last.operands.size() == 1) {
            const auto c = value(last.operands[0]);
            if (c.state == Fact::unknown) return false;
            if (c.state == Fact::constant) {
                const auto edge = last.successors[c.bits ? 0 : 1];
                last.opcode = OpCode::kBranch; last.operands.clear(); last.successors = {edge}; last.attributes.clear();
            }
        }
        for (auto& op : block.operations) if (op.results.size() == 1 && width(op.result_types[0]) &&
            facts[op.results[0]].state == Fact::constant &&
            (zeros.contains(op.results[0]) || op.opcode == OpCode::kCompare)) {
            const auto bits = facts[op.results[0]].bits;
            op.opcode = OpCode::kConstant;
            op.operands = {Operand::immediate(std::to_string(bits), op.result_types[0])};
            op.attributes.clear(); op.memory_scope = MemoryScope::kNone; op.memory_ordering = MemoryOrdering::kNone;
        }
    }
    std::size_t index = 0;
    std::erase_if(function.blocks, [&](const BasicBlock&) { return !executable[index++]; });
    // Mark from observable operations. Dead phi cycles must not keep each other
    // or an unused pointer load alive. Volatile/ordered loads remain observable.
    std::unordered_map<ValueId, std::vector<ValueId>> dependencies;
    std::unordered_set<ValueId> live;
    std::vector<ValueId> pending;
    const auto removable = [](const Operation& op) {
        if (ordinary_load(op)) return true;
        switch (op.opcode) {
            case OpCode::kConstant: case OpCode::kParameter: case OpCode::kAdd: case OpCode::kSub:
            case OpCode::kMul: case OpCode::kDiv: case OpCode::kRemainder: case OpCode::kFma:
            case OpCode::kNegate: case OpCode::kBitAnd: case OpCode::kBitOr: case OpCode::kBitXor:
            case OpCode::kShiftLeft: case OpCode::kShiftRight: case OpCode::kCompare: case OpCode::kSelect:
            case OpCode::kAggregateConstruct: case OpCode::kAggregateExtract: case OpCode::kConvert:
            case OpCode::kAddressSpaceCast: case OpCode::kAlloca: case OpCode::kPointerOffset: return true;
            default: return false;
        }
    };
    std::unordered_map<BlockId, const BasicBlock*> kept_blocks;
    for (const auto& block : function.blocks) kept_blocks[block.id] = &block;
    for (const auto& block : function.blocks) for (const auto& op : block.operations) {
        if (!charge()) return false;
        std::vector<ValueId> inputs;
        for (const auto& operand : op.operands) if (operand.kind == OperandKind::kValue) inputs.push_back(operand.value);
        if (op.attributes.contains("pointer_source_value")) {
            try { inputs.push_back(static_cast<ValueId>(std::stoul(op.attributes.at("pointer_source_value")))); }
            catch (...) { return false; }
        }
        for (const auto id : op.results) dependencies[id] = inputs;
        if (!removable(op)) pending.insert(pending.end(), inputs.begin(), inputs.end());
        for (const auto& edge : op.successors) {
            const auto found = kept_blocks.find(edge.block);
            if (found == kept_blocks.end()) return false;
            const auto* target = found->second;
            if (target->arguments.size() != edge.arguments.size()) return false;
            for (std::size_t a = 0; a < edge.arguments.size(); ++a)
                dependencies[target->arguments[a].value].push_back(edge.arguments[a]);
        }
    }
    while (!pending.empty()) {
        if (!charge()) return false;
        const auto id = pending.back(); pending.pop_back();
        if (!live.insert(id).second) continue;
        const auto& inputs = dependencies[id]; pending.insert(pending.end(), inputs.begin(), inputs.end());
    }
    std::unordered_map<BlockId, std::vector<bool>> keep;
    for (const auto& block : function.blocks) for (const auto& a : block.arguments) keep[block.id].push_back(live.contains(a.value));
    for (auto& block : function.blocks) {
        std::erase_if(block.arguments, [&](const BlockArgument& a) { return !live.contains(a.value); });
        std::erase_if(block.operations, [&](const Operation& op) {
            return removable(op) && std::none_of(op.results.begin(), op.results.end(), [&](ValueId id) { return live.contains(id); });
        });
        for (auto& op : block.operations) for (auto& edge : op.successors) {
            std::size_t n = 0;
            std::erase_if(edge.arguments, [&](ValueId) { return !keep[edge.block][n++]; });
        }
    }
    return true;
}

bool rename(Function& function, ValueId& next_value, BlockId& next_block) {
    std::unordered_map<ValueId, ValueId> values;
    std::unordered_map<BlockId, BlockId> blocks;
    const auto add = [&](ValueId value) { if (!values.contains(value)) values[value] = next_value++; };
    for (const auto& a : function.arguments) add(a.value);
    for (const auto& b : function.blocks) {
        blocks[b.id] = next_block++;
        for (const auto& a : b.arguments) add(a.value);
        for (const auto& op : b.operations) for (const auto id : op.results) add(id);
    }
    for (auto& a : function.arguments) a.value = values.at(a.value);
    for (auto& b : function.blocks) {
        b.id = blocks.at(b.id);
        for (auto& a : b.arguments) a.value = values.at(a.value);
        for (auto& op : b.operations) {
            for (auto& id : op.results) id = values.at(id);
            for (auto& operand : op.operands) if (operand.kind == OperandKind::kValue) operand.value = values.at(operand.value);
            for (auto& edge : op.successors) {
                edge.block = blocks.at(edge.block);
                for (auto& id : edge.arguments) id = values.at(id);
            }
            if (op.attributes.contains("pointer_source_value")) {
                const auto old = static_cast<ValueId>(std::stoul(op.attributes.at("pointer_source_value")));
                if (!values.contains(old)) return false;
                op.attributes["pointer_source_value"] = std::to_string(values.at(old));
            }
        }
    }
    const auto remap_map = [&](auto& map) {
        auto old = std::move(map); map.clear();
        for (auto& [id, metadata] : old) if (values.contains(id)) map.emplace(values.at(id), std::move(metadata));
    };
    remap_map(function.pointer_provenance); remap_map(function.mixed_pointer_address_spaces);
    const auto remap_set = [&](auto& set) {
        auto old = std::move(set); set.clear();
        for (const auto id : old) if (values.contains(id)) set.insert(values.at(id));
    };
    remap_set(function.generic_pointer_values); remap_set(function.generic_null_pointer_values);
    return true;
}
} // namespace

bool specialize_record_contexts(ir::Module& module, const std::vector<RecordContext>& contexts) {
    if (contexts.empty() || contexts.size() > 128) return false;
    using Key = std::pair<std::size_t, std::vector<ir::ValueId>>;
    std::map<Key, std::size_t> variants;
    std::vector<Function> clones;
    std::vector<std::pair<RecordContext, std::size_t>> redirects;
    std::size_t cloned_operations = 0;
    ValueId next_value = 1; BlockId next_block = 1;
    for (const auto& f : module.functions) {
        for (const auto& a : f.arguments) {
            if (a.value >= UINT32_MAX - 4'000'000) return false;
            next_value = std::max(next_value, a.value + 1);
        }
        for (const auto& b : f.blocks) {
            if (b.id >= UINT32_MAX - 1'000'000) return false;
            next_block = std::max(next_block, b.id + 1);
            for (const auto& a : b.arguments) {
                if (a.value >= UINT32_MAX - 4'000'000) return false;
                next_value = std::max(next_value, a.value + 1);
            }
            for (const auto& op : b.operations) for (const auto id : op.results) {
                if (id >= UINT32_MAX - 4'000'000) return false;
                next_value = std::max(next_value, id + 1);
            }
        }
    }
    if (next_value > UINT32_MAX - 4'000'000 || next_block > UINT32_MAX - 1'000'000) return false;
    for (const auto& context : contexts) {
        auto zeros = context.zero_loads; std::sort(zeros.begin(), zeros.end());
        if (zeros.empty() || context.callee >= module.functions.size()) continue;
        const Key key{context.callee, zeros};
        if (!variants.contains(key)) {
            if (clones.size() >= 16) return false;
            auto clone = module.functions[context.callee];
            if (clone.is_kernel) return false;
            for (const auto& b : clone.blocks) cloned_operations += b.operations.size();
            if (cloned_operations > 500'000) return false;
            if (!simplify(clone, zeros) || !rename(clone, next_value, next_block)) continue;
            clone.name += "__cm_record_zero_" + std::to_string(clones.size());
            if (std::any_of(module.functions.begin(), module.functions.end(), [&](const Function& f) { return f.name == clone.name; })) return false;
            variants[key] = clones.size(); clones.push_back(std::move(clone));
        }
        if (variants.contains(key)) redirects.push_back({context, variants.at(key)});
    }
    if (redirects.empty()) return false;
    for (const auto& [context, variant] : redirects)
        module.functions.at(context.caller).blocks.at(context.block).operations.at(context.operation).attributes["callee"] = clones.at(variant).name;
    for (auto& clone : clones) module.functions.push_back(std::move(clone));
    return true;
}
} // namespace cumetal::metal::detail
