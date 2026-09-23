#include "cumetal/ir/call_write_effects.h"

#include <algorithm>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace cumetal::ir::detail {
namespace {

struct Address {
    bool owned = false;
    std::size_t root = 0; // Allocation ValueId or formal argument index.
    std::optional<std::int64_t> offset = 0;
    bool operator==(const Address&) const = default;
};

struct FunctionGraph {
    const Function* function = nullptr;
    std::unordered_map<ValueId, std::size_t> arguments;
    std::unordered_map<ValueId, const Operation*> definitions;
    std::unordered_map<ValueId, std::vector<Operand>> incoming;
    std::unordered_set<ValueId> entry_joins, active_addresses;
    std::unordered_map<ValueId, std::optional<Address>> addresses;
    std::unordered_map<ValueId, std::optional<ValueId>> allocation_roots;
};

class Analysis {
public:
    Analysis(const Module& module, CallEffectLimits limits) : module_(module), limits_(limits) {}

    CallEffectSummary run(std::string_view callee) {
        for (const auto& function : module_.functions) {
            if (!charge(1, "function index")) return finish();
            if (!functions_.emplace(function.name, &function).second) {
                fail("duplicate function name");
                return finish();
            }
        }
        const auto target = functions_.find(std::string(callee));
        if (target == functions_.end()) {
            fail("unknown direct callee '" + std::string(callee) + "'");
            return finish();
        }
        if (!summarize(*target->second, 0)) return finish();
        result_.complete = true;
        result_.writes = summaries_.at(target->second);
        return finish();
    }

private:
    const Module& module_;
    CallEffectLimits limits_;
    CallEffectSummary result_;
    std::unordered_map<std::string, const Function*> functions_;
    std::unordered_map<const Function*, std::vector<CallWriteEffect>> summaries_;
    std::unordered_set<const Function*> active_functions_;

    CallEffectSummary finish() {
        if (!result_.complete) result_.writes.clear();
        return std::move(result_);
    }
    bool fail(const std::string& reason) {
        if (result_.reason.empty()) result_.reason = "PTX call write effect proof: " + reason;
        return false;
    }
    bool charge(std::size_t amount, const char* phase) {
        if (amount > limits_.work - result_.work) {
            result_.budget_exhausted = true;
            return fail(std::string("work budget exhausted (") + phase + ")");
        }
        result_.work += amount;
        return true;
    }
    bool depth(std::size_t value) {
        if (value <= limits_.depth) return true;
        result_.budget_exhausted = true;
        return fail("depth budget exhausted");
    }
    bool index(FunctionGraph& graph, const Function& function) {
        graph.function = &function;
        if (function.blocks.empty()) return fail("callee '" + function.name + "' has no body");
        std::unordered_map<BlockId, const BasicBlock*> blocks;
        for (std::size_t i = 0; i < function.arguments.size(); ++i) {
            if (!charge(1, "argument index")) return false;
            graph.arguments.emplace(function.arguments[i].value, i);
        }
        for (const auto& block : function.blocks) {
            if (!charge(1, "block index")) return false;
            if (!blocks.emplace(block.id, &block).second) return fail("duplicate block id");
            for (const auto& argument : block.arguments) {
                if (!charge(1, "join index")) return false;
                graph.incoming.emplace(argument.value, std::vector<Operand>{});
                if (&block == &function.blocks.front()) graph.entry_joins.insert(argument.value);
            }
            for (const auto& operation : block.operations) {
                if (!charge(1 + operation.results.size(), "definition index")) return false;
                if (operation.results.size() != operation.result_types.size()) return fail("malformed result types");
                for (const auto value : operation.results)
                    if (!graph.definitions.emplace(value, &operation).second) return fail("duplicate SSA definition");
            }
        }
        for (const auto& block : function.blocks) {
            if (block.operations.empty()) return fail("empty basic block");
            for (const auto& edge : block.operations.back().successors) {
                if (!charge(1 + edge.arguments.size(), "incoming edges")) return false;
                if (!blocks.contains(edge.block)) return fail("unknown successor block");
                const auto& target = *blocks.at(edge.block);
                if (edge.arguments.size() != target.arguments.size()) return fail("inconsistent successor arguments");
                for (std::size_t i = 0; i < edge.arguments.size(); ++i)
                    graph.incoming.at(target.arguments[i].value).push_back(
                        Operand::value_ref(edge.arguments[i], target.arguments[i].type));
            }
        }
        return true;
    }
    std::optional<std::int64_t> literal(FunctionGraph& graph, const Operand& operand, std::size_t level) {
        if (!charge(1, "constant offset") || !depth(level)) return std::nullopt;
        if (operand.type != Type::integer(64)) return std::nullopt;
        if (operand.kind == OperandKind::kImmediate) {
            try {
                std::size_t used = 0;
                const auto value = std::stoll(operand.text, &used, 0);
                if (used == operand.text.size()) return value;
            } catch (...) {}
        } else if (operand.kind == OperandKind::kValue && graph.definitions.contains(operand.value)) {
            const auto& operation = *graph.definitions.at(operand.value);
            if (operation.operands.size() != 1 || operation.result_types.size() != 1 ||
                operation.operands.front().type != Type::integer(64) ||
                operation.result_types.front() != Type::integer(64) ||
                operation.attributes.contains("guard_operand")) return std::nullopt;
            const auto opcode = operation.attributes.find("ptx_opcode");
            const bool plain_copy = operation.attributes.empty() ||
                (operation.attributes.size() == 1 && opcode != operation.attributes.end() &&
                 (opcode->second == "mov.b64" || opcode->second == "mov.u64" || opcode->second == "mov.s64"));
            if (operation.opcode == OpCode::kConstant || (operation.opcode == OpCode::kConvert && plain_copy))
                return literal(graph, operation.operands.front(), level + 1);
        }
        return std::nullopt;
    }
    std::optional<Address> address(FunctionGraph& graph, const Operand& operand, std::size_t level = 0) {
        if (!charge(1, "address derivation") || !depth(level)) return std::nullopt;
        if (operand.kind != OperandKind::kValue || !operand.type.is_pointer()) return std::nullopt;
        const auto value = operand.value;
        if (graph.addresses.contains(value)) return graph.addresses.at(value);
        if (!graph.active_addresses.insert(value).second) return std::nullopt;
        std::optional<Address> result;
        if (graph.arguments.contains(value)) {
            result = Address{false, graph.arguments.at(value), 0};
        } else if (graph.definitions.contains(value)) {
            const auto& operation = *graph.definitions.at(value);
            if (!operation.attributes.contains("guard_operand")) {
                if (operation.opcode == OpCode::kAlloca && operand.type.address_space == AddressSpace::kPrivate) {
                    result = Address{true, value, 0};
                } else if ((operation.opcode == OpCode::kConvert || operation.opcode == OpCode::kAddressSpaceCast ||
                            operation.opcode == OpCode::kParameter) && operation.operands.size() == 1 &&
                           operation.operands.front().type.is_pointer()) {
                    result = address(graph, operation.operands.front(), level + 1);
                } else if (operation.opcode == OpCode::kPointerOffset && operation.operands.size() == 2) {
                    auto base = address(graph, operation.operands.front(), level + 1);
                    const auto amount = literal(graph, operation.operands[1], level + 1);
                    const auto* pointee = operation.operands.front().type.pointee();
                    const auto unit = operation.attributes.find("offset_unit");
                    const bool bytes = (unit != operation.attributes.end() && unit->second == "bytes") ||
                        (unit == operation.attributes.end() && pointee &&
                         pointee->kind == TypeKind::kInteger && pointee->bit_width == 8);
                    if (base && bytes) {
                        if (base->offset && amount) {
                            const bool subtract = operation.attributes.contains("offset_direction") &&
                                operation.attributes.at("offset_direction") == "subtract";
                            const __int128 sum = static_cast<__int128>(*base->offset) +
                                (subtract ? -static_cast<__int128>(*amount) : *amount);
                            if (sum >= INT64_MIN && sum <= INT64_MAX) {
                                base->offset = std::int64_t(sum);
                                result = base;
                            }
                        } else if (base->owned) {
                            // Exact offsets are unnecessary for writes confined
                            // to this callee's allocation, but never for formals.
                            base->offset.reset();
                            result = base;
                        }
                    }
                } else if (operation.opcode == OpCode::kSelect && operation.operands.size() == 3) {
                    const auto left = address(graph, operation.operands[1], level + 1);
                    const auto right = address(graph, operation.operands[2], level + 1);
                    if (left && right && *left == *right) result = left;
                }
            }
        } else if (graph.incoming.contains(value) && !graph.entry_joins.contains(value)) {
            bool agrees = !graph.incoming.at(value).empty();
            for (const auto& source : graph.incoming.at(value)) {
                const auto incoming = address(graph, source, level + 1);
                if (!incoming || (result && *result != *incoming)) agrees = false;
                else result = incoming;
            }
            if (!agrees) result.reset();
        }
        graph.active_addresses.erase(value);
        graph.addresses[value] = result;
        return result;
    }
    std::optional<ValueId> allocation_root(FunctionGraph& graph, const Operand& operand) {
        if (!charge(1, "owned allocation query") || operand.kind != OperandKind::kValue ||
            !operand.type.is_pointer()) return std::nullopt;
        if (graph.allocation_roots.contains(operand.value)) return graph.allocation_roots.at(operand.value);
        auto& cached = graph.allocation_roots[operand.value];
        std::vector<Operand> pending{operand};
        std::unordered_set<ValueId> visited;
        std::unordered_map<ValueId, std::vector<ValueId>> users;
        std::optional<ValueId> root;
        while (!pending.empty()) {
            if (!charge(1, "owned allocation graph")) return std::nullopt;
            const auto current = pending.back();
            pending.pop_back();
            if (current.kind != OperandKind::kValue || !current.type.is_pointer()) return std::nullopt;
            const auto value = current.value;
            if (!visited.insert(value).second) continue;
            std::vector<Operand> inputs;
            if (graph.arguments.contains(value) || graph.entry_joins.contains(value)) return std::nullopt;
            if (graph.definitions.contains(value)) {
                const auto& operation = *graph.definitions.at(value);
                if (operation.attributes.contains("guard_operand")) return std::nullopt;
                if (operation.opcode == OpCode::kAlloca && operation.result_types.size() == 1 &&
                    operation.result_types.front().is_pointer() &&
                    operation.result_types.front().address_space == AddressSpace::kPrivate) {
                    if (root && *root != value) return std::nullopt;
                    root = value;
                    continue;
                }
                const bool copy = (operation.opcode == OpCode::kConvert ||
                    operation.opcode == OpCode::kAddressSpaceCast || operation.opcode == OpCode::kParameter) &&
                    operation.operands.size() == 1;
                const bool offset = operation.opcode == OpCode::kPointerOffset &&
                    operation.operands.size() == 2 && operation.operands[1].type == Type::integer(64);
                if (copy || offset) inputs.push_back(operation.operands.front());
                else if (operation.opcode == OpCode::kSelect && operation.operands.size() == 3)
                    inputs = {operation.operands[1], operation.operands[2]};
                else return std::nullopt;
            } else if (graph.incoming.contains(value)) {
                inputs = graph.incoming.at(value);
            } else return std::nullopt;
            if (inputs.empty()) return std::nullopt;
            for (const auto& input : inputs) {
                if (!charge(1, "owned allocation edges") || input.kind != OperandKind::kValue ||
                    !input.type.is_pointer()) return std::nullopt;
                users[input.value].push_back(value);
                pending.push_back(input);
            }
        }
        if (!root) return std::nullopt;
        // Every dependency must reach the allocation seed. Merely finding one
        // seed would also accept a select containing an independent empty cycle.
        std::vector<ValueId> reachable{*root};
        std::unordered_set<ValueId> seeded;
        while (!reachable.empty()) {
            if (!charge(1, "owned allocation reachability")) return std::nullopt;
            const auto value = reachable.back();
            reachable.pop_back();
            if (!seeded.insert(value).second) continue;
            if (users.contains(value)) for (const auto user : users.at(value)) {
                if (!charge(1, "owned allocation reachability edges")) return std::nullopt;
                reachable.push_back(user);
            }
        }
        if (seeded.size() == visited.size()) cached = root;
        return cached;
    }
    std::optional<Address> write_address(FunctionGraph& graph, const Operand& operand) {
        if (const auto exact = address(graph, operand)) return exact;
        if (!result_.reason.empty()) return std::nullopt;
        if (const auto root = allocation_root(graph, operand))
            return Address{true, *root, std::nullopt}; // Ownership proves no byte offset.
        return std::nullopt;
    }
    static std::optional<std::uint64_t> scalar_bytes(const Type& type) {
        if (type.is_pointer()) return 8;
        if (type.kind == TypeKind::kPredicate) return 1;
        if ((type.kind == TypeKind::kInteger || type.kind == TypeKind::kFloat) &&
            type.bit_width != 0 && type.bit_width <= 128 && type.bit_width % 8 == 0)
            return type.bit_width / 8;
        return std::nullopt; // Aggregate/vector padding is not a scalar contract.
    }
    static std::optional<std::uint64_t> write_bytes(const Operation& operation) {
        if (operation.operands.size() < 2) return std::nullopt;
        const auto opcode = operation.attributes.find("ptx_opcode");
        if (opcode != operation.attributes.end()) {
            const auto dot = opcode->second.rfind('.');
            if (dot == std::string::npos || dot + 2 >= opcode->second.size()) return std::nullopt;
            const auto suffix = std::string_view(opcode->second).substr(dot + 1);
            if (suffix.front() != 'b' && suffix.front() != 'u' && suffix.front() != 's' && suffix.front() != 'f')
                return std::nullopt;
            // Vector PTX stores have already become separate scalar IR stores.
            // The suffix and emitted scalar storage must agree. In particular,
            // a pointer expression emits eight bytes; a PTX b32 annotation
            // alone cannot prove a four-byte write in inconsistent IR.
            const auto bits = suffix.substr(1);
            std::optional<std::uint64_t> bytes;
            if (bits == "8") bytes = 1;
            else if (bits == "16") bytes = 2;
            else if (bits == "32") bytes = 4;
            else if (bits == "64") bytes = 8;
            return bytes == scalar_bytes(operation.operands[1].type) ? bytes : std::nullopt;
        }
        return scalar_bytes(operation.operands[1].type);
    }
    bool add_effect(std::set<std::tuple<std::size_t, std::int64_t, std::uint64_t>>& writes,
                    const Address& destination, std::int64_t displacement, std::uint64_t bytes) {
        if (!charge(1, "write footprint")) return false;
        if (destination.owned) return true;
        if (!destination.offset || !bytes) return fail("unknown formal write offset or width");
        const __int128 offset = static_cast<__int128>(*destination.offset) + displacement;
        if (offset < INT64_MIN || offset > INT64_MAX || offset + bytes > INT64_MAX)
            return fail("write footprint offset overflow");
        const auto effect = std::make_tuple(destination.root, std::int64_t(offset), bytes);
        if (!writes.contains(effect) && writes.size() >= limits_.writes) {
            result_.budget_exhausted = true;
            return fail("write footprint budget exhausted");
        }
        writes.insert(effect);
        return true;
    }
    bool summarize(const Function& function, std::size_t level) {
        if (!charge(1, "callee visit") || !depth(level)) return false;
        if (summaries_.contains(&function)) return true;
        if (!active_functions_.insert(&function).second) return fail("recursive call to '" + function.name + "'");
        FunctionGraph graph;
        if (!index(graph, function)) return false;
        std::set<std::tuple<std::size_t, std::int64_t, std::uint64_t>> writes;
        for (const auto& block : function.blocks) for (const auto& operation : block.operations) {
            if (!charge(1, "operation scan")) return false;
            if (operation.opcode == OpCode::kStore || operation.opcode == OpCode::kAtomic ||
                operation.opcode == OpCode::kMetalAtomic) {
                const auto bytes = write_bytes(operation);
                if (!bytes) return fail("unknown store width in '" + function.name + "'");
                const auto destination = write_address(graph, operation.operands.front());
                if (!destination) return fail("unknown or ambiguous write address in '" + function.name + "'");
                if (!add_effect(writes, *destination, 0, *bytes)) return false;
            } else if (operation.opcode == OpCode::kCall) {
                const auto name = operation.attributes.find("callee");
                if (name == operation.attributes.end() || name->second.empty() || operation.attributes.contains("indirect"))
                    return fail("indirect or unresolved call in '" + function.name + "'");
                const auto builtin = operation.attributes.find("builtin");
                if (builtin != operation.attributes.end() && builtin->second == "true") {
                    if (!is_read_only_scalar_builtin(operation)) return fail("unknown builtin effect for '" + name->second + "'");
                    continue;
                }
                const auto callee = functions_.find(name->second);
                if (callee == functions_.end()) return fail("unknown direct callee '" + name->second + "'");
                if (operation.operands.size() != callee->second->arguments.size()) return fail("callee argument count mismatch");
                if (!summarize(*callee->second, level + 1)) return false;
                for (const auto& effect : summaries_.at(callee->second)) {
                    if (!charge(1, "call argument substitution")) return false;
                    if (effect.argument >= operation.operands.size()) return fail("missing write argument");
                    const auto actual = write_address(graph, operation.operands[effect.argument]);
                    if (!actual) return fail("unknown or ambiguous actual write argument in '" + function.name + "'");
                    if (!add_effect(writes, *actual, effect.offset, effect.bytes)) return false;
                }
            } else if (operation.opcode == OpCode::kInvalid || operation.opcode == OpCode::kPrintf) {
                return fail("unsupported memory effect in '" + function.name + "'");
            }
        }
        if (!result_.reason.empty()) return false;
        auto& summary = summaries_[&function];
        for (const auto& [argument, offset, bytes] : writes) summary.push_back({argument, offset, bytes});
        active_functions_.erase(&function);
        return true;
    }
};

} // namespace

bool is_read_only_scalar_builtin(const Operation& operation) {
    const auto callee = operation.attributes.find("callee");
    const auto builtin = operation.attributes.find("builtin");
    if (operation.opcode != OpCode::kCall || callee == operation.attributes.end() ||
        builtin == operation.attributes.end() || builtin->second != "true" ||
        operation.attributes.contains("indirect")) return false;
    const auto& name = callee->second;
    static const std::unordered_set<std::string> known = {
        "sqrt", "rsqrt", "sin", "cos", "tan", "exp", "exp2", "log", "log2", "pow",
        "floor", "ceil", "trunc", "rint", "round", "fabs", "abs", "fma", "fmin", "fmax",
        "min", "max", "clz", "popcount", "reverse_bits", "isnan", "isinf", "isfinite", "signbit", "copysign",
    };
    if (!known.contains(name)) return false;
    const auto scalar = [](const Type& type) {
        return type.kind == TypeKind::kInteger || type.kind == TypeKind::kFloat || type.kind == TypeKind::kPredicate;
    };
    return std::all_of(operation.operands.begin(), operation.operands.end(),
                       [&](const Operand& operand) { return scalar(operand.type); }) &&
        std::all_of(operation.result_types.begin(), operation.result_types.end(), scalar);
}

CallEffectSummary summarize_call_effects(const Module& module, std::string_view callee, CallEffectLimits limits) {
    return Analysis(module, limits).run(callee);
}

} // namespace cumetal::ir::detail
