#include "ptx_address_cancellation.h"

#include "ptx_text.h"

#include <algorithm>
#include <deque>
#include <optional>
#include <set>
#include <unordered_set>

namespace cumetal::ir::detail {
namespace {

// An address is never represented by an integer. Only its symbolic coefficient
// and the ordinary integer residual are tracked. The small coefficient domain
// deliberately declines multiplication and other non-affine address operations.
struct Form {
    enum State { unknown, known, conflict } state = unknown;
    std::string root;
    int coefficient = 0;
    bool operator==(const Form&) const = default;
};

Form scalar() { return {Form::known, {}, 0}; }
Form pointer(std::string root) { return {Form::known, std::move(root), 1}; }
Form conflict() { return {Form::conflict, {}, 0}; }

struct Input {
    std::optional<ValueId> value;
    Form literal = scalar();
};

struct Node {
    enum Kind { leaf, copy, add, subtract, negate, select, join } kind = leaf;
    ValueId value = kInvalidValue;
    const Instruction* instruction = nullptr;
    std::string name;
    std::vector<Input> inputs;
    // Operand indexes are used only for instructions, never CFG joins.
    std::vector<std::size_t> operand_indexes;
    Form form;
};

bool integer_copy(const std::string& opcode) {
    return opcode == "mov.u64" || opcode == "mov.s64" || opcode == "mov.b64";
}

bool integer_add(const std::string& opcode) {
    return opcode == "add.u64" || opcode == "add.s64";
}

bool integer_subtract(const std::string& opcode) {
    return opcode == "sub.u64" || opcode == "sub.s64";
}

}  // namespace

bool cancel_same_base_addresses(
    std::vector<RawBlock>& blocks,
    const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
    const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
    const std::vector<std::map<std::string, ValueId>>& arguments,
    const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
    const std::unordered_map<ValueId, Type>& types,
    const std::unordered_map<std::string, Type>& parameters,
    const std::unordered_set<std::string>& symbols,
    const std::function<bool(const std::string&)>& declared_register,
    std::deque<Instruction>& storage,
    InstructionOrigins* origins) {
    // These are proof budgets, not licenses to accept an incomplete graph. On
    // exhaustion nothing changes and ordinary import reports the unsupported
    // address operation. Worklist propagation has at most two state changes per
    // value: unknown -> known -> conflict.
    constexpr std::size_t kMaxAnalyzedValues = 1'000'000;
    constexpr std::size_t kMaxOffsetValues = 16'384;
    constexpr std::size_t kMaxLivenessWork = 100'000'000;
    constexpr std::size_t kMaxShadowNameAttempts = 100'000;
    constexpr std::size_t kMaxScalarUseWork = 4'000'000;
    std::unordered_map<ValueId, Node> nodes;
    std::unordered_map<ValueId, std::vector<ValueId>> users;
    std::unordered_map<ValueId, std::vector<const Instruction*>> instruction_users;
    std::unordered_map<const Instruction*, std::unordered_map<std::string, ValueId>> sources;
    std::unordered_set<std::string> register_names;
    const auto pointer_type = [&](ValueId value) {
        const auto type = types.find(value);
        return type != types.end() && type->second.is_pointer();
    };

    for (std::size_t b = 0; b < blocks.size(); ++b) {
        auto environment = incoming[b];
        for (const auto* instruction : blocks[b].instructions) {
            auto& reaching = sources[instruction];
            for (const auto& name : source_registers(*instruction)) {
                register_names.insert(name);
                const auto value = environment.find(name);
                if (value == environment.end()) continue;
                reaching[name] = value->second;
                instruction_users[value->second].push_back(instruction);
            }
            const auto destinations = destination_registers(*instruction);
            const auto found = results.find(instruction);
            if (found == results.end()) continue;
            for (std::size_t i = 0; i < destinations.size() && i < found->second.size(); ++i) {
                const auto value = found->second[i];
                register_names.insert(destinations[i]);
                environment[destinations[i]] = value;
                Node node;
                node.value = value;
                node.name = destinations[i];
                node.instruction = instruction;
                // Opaque pointer definitions are distinct roots. Their original
                // instructions remain present, so this pass cannot legalize an
                // unsupported load/call/cast by erasing its implementation.
                node.form = pointer_type(value) ? pointer("value:" + std::to_string(value)) : scalar();
                // Guarded definitions normally become CFG edges before SSA.
                // An opaque residual guard cannot establish a shadow value on
                // its false path, so leave that uncommon form unsupported.
                if (!instruction->predicate.empty() && pointer_type(value)) node.form = conflict();
                if (destinations.size() == 1 && instruction->predicate.empty()) {
                    const auto& opcode = instruction->opcode;
                    const auto input = [&](std::size_t index) {
                        Input result;
                        const auto& token = instruction->operands[index];
                        const auto reg = first_register(token);
                        if (!reg.empty() && trim(token) != reg)
                            result.literal = conflict();
                        else if (const auto use = reaching.find(reg); use != reaching.end())
                            result.value = use->second;
                        else {
                            const auto symbol = parameter_name_from_operand(token);
                            if (symbols.contains(symbol)) result.literal = pointer("symbol:" + symbol);
                            else if (!first_register(token).empty()) result.literal = conflict();
                        }
                        node.inputs.push_back(std::move(result));
                        node.operand_indexes.push_back(index);
                    };
                    if (integer_copy(opcode) && instruction->operands.size() == 2 &&
                        instruction->operands[1].find('{') == std::string::npos) {
                        node.kind = Node::copy;
                        input(1);
                    } else if ((integer_add(opcode) || integer_subtract(opcode)) &&
                               instruction->operands.size() == 3) {
                        node.kind = integer_add(opcode) ? Node::add : Node::subtract;
                        input(1); input(2);
                    } else if (opcode == "neg.s64" && instruction->operands.size() == 2) {
                        node.kind = Node::negate;
                        input(1);
                    } else if ((opcode == "selp.u64" || opcode == "selp.s64" || opcode == "selp.b64") &&
                               instruction->operands.size() == 4) {
                        node.kind = Node::select;
                        input(1); input(2);
                    } else if (starts_with(opcode, "ld.param.") && pointer_type(value) &&
                               instruction->operands.size() == 2) {
                        const auto parameter = parameter_name_from_operand(instruction->operands[1]);
                        const auto contract = parameters.find(parameter);
                        if (contract != parameters.end() && contract->second.is_pointer() &&
                            instruction->operands[1] == "[" + parameter + "]")
                            node.form = pointer("parameter:" + parameter);
                    }
                    // cvta deliberately stays an opaque root: a generic and
                    // a space-specific PTX address can refer to the same byte
                    // while using different numerical representations. Equal
                    // abstract Metal address spaces do not prove equal bits.
                    if (node.kind != Node::leaf) node.form = {};
                }
                nodes.emplace(value, std::move(node));
            }
        }
        for (const auto& [name, value] : arguments[b]) {
            Node node;
            node.kind = Node::join;
            node.value = value;
            node.name = name;
            for (const auto predecessor : blocks[b].predecessors) {
                const auto source = outgoing[predecessor].find(name);
                if (source == outgoing[predecessor].end()) return false;
                node.inputs.push_back({source->second, {}});
            }
            nodes.emplace(value, std::move(node));
        }
    }
    if (nodes.size() > kMaxAnalyzedValues) return false;
    for (const auto& [value, node] : nodes)
        for (const auto& input : node.inputs)
            if (input.value) users[*input.value].push_back(value);

    const auto form = [&](const Input& input) -> Form {
        if (!input.value) return input.literal;
        const auto value = nodes.find(*input.value);
        // An external/implicit value has no allocation proof. Do not invent a
        // root or offset for an unseeded SSA component.
        return value == nodes.end() ? conflict() : value->second.form;
    };
    const auto evaluate = [&](const Node& node) -> Form {
        if (node.kind == Node::leaf) return node.form;
        if (node.inputs.empty()) return conflict();
        if (node.kind == Node::join || node.kind == Node::select) {
            Form result;
            for (const auto& input : node.inputs) {
                const auto candidate = form(input);
                if (candidate.state == Form::conflict) return conflict();
                if (candidate.state == Form::unknown) continue;
                if (result.state == Form::known && result != candidate) return conflict();
                result = candidate;
            }
            return result;
        }
        auto left = form(node.inputs.front());
        if (left.state != Form::known) return left;
        if (node.kind == Node::copy) return left;
        if (node.kind == Node::negate) {
            left.coefficient = -left.coefficient;
            return left;
        }
        if (node.inputs.size() != 2) return conflict();
        const auto right = form(node.inputs[1]);
        if (right.state != Form::known) return right;
        if (!left.root.empty() && !right.root.empty() && left.root != right.root) return conflict();
        const int coefficient = left.coefficient +
            (node.kind == Node::subtract ? -right.coefficient : right.coefficient);
        if (coefficient < -1 || coefficient > 1) return conflict();
        return {Form::known, coefficient == 0 ? "" : left.root.empty() ? right.root : left.root, coefficient};
    };
    std::deque<ValueId> pending;
    std::unordered_set<ValueId> queued;
    for (const auto& [value, node] : nodes) { pending.push_back(value); queued.insert(value); }
    std::size_t steps = 0;
    const auto propagate = [&]() {
        while (!pending.empty()) {
            if (++steps > nodes.size() * 8 + 1) return false;
            const auto value = pending.front(); pending.pop_front(); queued.erase(value);
            auto& node = nodes.at(value);
            if (node.form.state == Form::conflict) continue;
            auto next = evaluate(node);
            if (next.state == Form::unknown || next == node.form) continue;
            if (node.form.state == Form::known && next != node.form) next = conflict();
            node.form = std::move(next);
            for (const auto user : users[value])
                if (queued.insert(user).second) pending.push_back(user);
        }
        return true;
    };
    if (!propagate()) return false;
    // A seeded join may provisionally propagate across a backedge, but every
    // incoming edge must eventually prove the same form. Unknown SCCs and
    // unknown incoming edges invalidate the candidate and all its users.
    for (auto& [value, node] : nodes) {
        if (node.form.state != Form::known || node.kind == Node::leaf) continue;
        if (std::any_of(node.inputs.begin(), node.inputs.end(), [&](const Input& input) {
                return form(input).state != Form::known;
            })) {
            node.form = conflict();
            for (const auto user : users[value])
                if (queued.insert(user).second) pending.push_back(user);
        }
    }
    if (!propagate()) return false;

    std::unordered_set<ValueId> cancellations;
    for (const auto& [value, node] : nodes) {
        if (node.form.state != Form::known || node.form.coefficient != 0 ||
            (node.kind != Node::add && node.kind != Node::subtract)) continue;
        if (std::any_of(node.inputs.begin(), node.inputs.end(), [&](const Input& input) {
                return form(input).coefficient != 0;
            })) cancellations.insert(value);
    }
    if (cancellations.empty()) return false;

    // A cancelled address difference is an integer, not a new allocation.
    // The existing memory emitter can cast an integer address to an MSL
    // pointer, so allowing that use here would turn a formerly rejected
    // pointer difference into an unsupported raw numeric address. Follow the
    // residual through scalar operations and joins; a proven affine pointer
    // base must be reintroduced before it may address memory. A cvta is not
    // such proof: its input must already have an address representation.
    std::unordered_set<ValueId> scalar_residuals = cancellations;
    std::deque<ValueId> scalar_pending(cancellations.begin(), cancellations.end());
    std::size_t scalar_use_work = 0;
    const auto consume_scalar_work = [&](std::size_t amount) {
        if (amount > kMaxScalarUseWork - scalar_use_work) return false;
        scalar_use_work += amount;
        return true;
    };
    const auto follow_scalar = [&](ValueId value) {
        const auto found = nodes.find(value);
        if (found != nodes.end()) {
            const auto& node = found->second;
            // Opaque pointer typing alone does not prove that a scalar was
            // combined with a base. Only the checked affine graph can do so.
            if (node.kind != Node::leaf && node.form.state == Form::known &&
                node.form.coefficient == 1 &&
                std::all_of(node.inputs.begin(), node.inputs.end(), [&](const Input& input) {
                    return form(input).coefficient == 0 || !input.value ||
                           !scalar_residuals.contains(*input.value);
                })) return;
        }
        if (scalar_residuals.insert(value).second) scalar_pending.push_back(value);
    };
    while (!scalar_pending.empty()) {
        const auto value = scalar_pending.front(); scalar_pending.pop_front();
        if (!consume_scalar_work(instruction_users[value].size() + users[value].size() + 1))
            return false;
        for (const auto* use : instruction_users[value]) {
            const auto root = root_opcode(use->opcode);
            std::optional<std::size_t> address;
            if (root == "ld" || root == "atom" || root == "cvta") address = 1;
            else if (root == "st" || root == "red") address = 0;
            if (address && *address < use->operands.size()) {
                for (const auto& name : registers_in(use->operands[*address])) {
                    const auto reaching = sources.at(use).find(name);
                    if (reaching != sources.at(use).end() && reaching->second == value)
                        return false;
                }
            }
            // A read's contents are not its address's scalar arithmetic.
            // Direct use of an unbased residual as that address was rejected
            // above. Stored data and comparison operands remain permitted.
            if (root == "ld" || root == "atom" || root == "call") continue;
            const auto outputs = results.find(use);
            if (outputs == results.end()) continue;
            for (const auto output : outputs->second) follow_scalar(output);
        }
        for (const auto use : users[value])
            if (nodes.at(use).kind == Node::join) follow_scalar(use);
    }

    std::unordered_set<ValueId> offsets;
    std::vector<ValueId> needed(cancellations.begin(), cancellations.end());
    for (std::size_t i = 0; i < needed.size(); ++i) {
        const auto& node = nodes.at(needed[i]);
        for (const auto& input : node.inputs) {
            if (form(input).coefficient == 0 || !input.value) continue;
            if (!offsets.insert(*input.value).second) continue;
            if (offsets.size() > kMaxOffsetValues) return false;
            needed.push_back(*input.value);
        }
    }
    // Negative-root quantities cannot exist in Metal. Remove them only if all
    // original uses are inside the rewritten scalar cone. Stores, guards,
    // calls, conversions, nonlinear arithmetic, or a new pointer escape keep
    // the original import failure; they may not observe the residual alone.
    for (const auto value : offsets) {
        const auto& node = nodes.at(value);
        if (node.form.coefficient != -1) continue;
        for (const auto* use : instruction_users[value]) {
            const auto found = results.find(use);
            if (found == results.end() || found->second.size() != 1) return false;
            const auto destination = found->second.front();
            if (!cancellations.contains(destination) &&
                !(offsets.contains(destination) && nodes.at(destination).form.coefficient == -1)) return false;
        }
        for (const auto use : users[value]) {
            if (nodes.at(use).kind == Node::join &&
                (!offsets.contains(use) || nodes.at(use).form.coefficient != -1)) return false;
        }
    }

    std::unordered_map<std::string, std::string> shadows;
    std::size_t next_name = 0;
    const auto shadow = [&](const std::string& name) -> std::string {
        if (shadows.contains(name)) return shadows.at(name);
        std::string result;
        do {
            if (next_name >= kMaxShadowNameAttempts) return {};
            result = "%__cumetal_address_offset_" + std::to_string(next_name++);
        }
        while (register_names.contains(result) || declared_register(result));
        register_names.insert(result);
        shadows.emplace(name, result);
        return result;
    };
    // Shadow registers use the original register's control-flow lifetime, not
    // its spelling as provenance. The proof above uses distinct SSA values;
    // reconstruction below supplies a matching shadow phi on each live edge.
    for (const auto value : offsets)
        if (shadow(nodes.at(value).name).empty()) return false;
    const auto residual = [&](const Node& node, std::size_t input_index) -> std::string {
        const auto& input = node.inputs[input_index];
        const auto operand_index = node.operand_indexes[input_index];
        if (form(input).coefficient == 0) return node.instruction->operands[operand_index];
        if (!input.value) return "0"; // The symbolic root has zero residual.
        // At this instruction, this operand names exactly the reaching SSA
        // definition that the proof consumed, even if later PTX reuses it.
        return shadow(first_register(node.instruction->operands[operand_index]));
    };
    std::unordered_map<const Instruction*, Instruction> replacements;
    std::unordered_map<const Instruction*, Instruction> companions;
    std::unordered_set<const Instruction*> removable;
    for (const auto value : needed) {
        const auto& node = nodes.at(value);
        if (node.kind == Node::join || node.instruction == nullptr) continue;
        const bool cancellation = cancellations.contains(value);
        Instruction instruction = *node.instruction;
        instruction.operands[0] = cancellation ? node.name : shadow(node.name);
        if (node.kind == Node::leaf) {
            instruction.opcode = "mov.u64";
            instruction.operands = {shadow(node.name), "0"};
        } else {
            for (std::size_t i = 0; i < node.inputs.size(); ++i)
                instruction.operands[node.operand_indexes[i]] = residual(node, i);
            if (node.kind == Node::copy) instruction.opcode = "mov.u64";
            if (node.kind == Node::add) instruction.opcode = "add.u64";
            if (node.kind == Node::subtract) instruction.opcode = "sub.u64";
        }
        if (cancellation) replacements.emplace(node.instruction, std::move(instruction));
        else {
            companions.emplace(node.instruction, std::move(instruction));
            if (node.kind != Node::leaf || integer_copy(node.instruction->opcode))
                removable.insert(node.instruction);
        }
    }
    auto rewritten_blocks = blocks;
    for (auto& block : rewritten_blocks) {
        std::vector<const Instruction*> instructions;
        for (const auto* original : block.instructions) {
            // Capture scalar operands before an in-place PTX destination can
            // overwrite them with the original pointer result.
            if (const auto companion = companions.find(original); companion != companions.end()) {
                storage.push_back(companion->second);
                record_instruction_origin(origins, &storage.back(), original);
                instructions.push_back(&storage.back());
            }
            if (const auto replacement = replacements.find(original); replacement != replacements.end()) {
                storage.push_back(replacement->second);
                record_instruction_origin(origins, &storage.back(), original);
                instructions.push_back(&storage.back());
            } else instructions.push_back(original);
        }
        block.instructions = std::move(instructions);
    }
    // Only the proved pure address cone is eligible for deletion. In
    // particular, keep loads/calls and every pointer used for actual memory.
    // Compute the least live set while ignoring dead eligible instructions,
    // so pointer-only cycles disappear while shadow recurrences remain live.
    // Build the result separately: budget exhaustion cannot leave a partially
    // rewritten function active.
    std::size_t liveness_work = 0;
    const auto consume_work = [&](std::size_t amount) {
        if (amount > kMaxLivenessWork - liveness_work) return false;
        liveness_work += amount;
        return true;
    };
    {
        std::vector<std::set<std::string>> live_in(blocks.size()), live_out(blocks.size());
        bool changed = true;
        while (changed) {
            changed = false;
            for (std::size_t reverse = blocks.size(); reverse > 0; --reverse) {
                const auto b = reverse - 1;
                std::set<std::string> out;
                for (const auto successor : rewritten_blocks[b].successors) {
                    if (!consume_work(live_in[successor].size() + 1)) return false;
                    out.insert(live_in[successor].begin(), live_in[successor].end());
                }
                if (!consume_work(out.size() + rewritten_blocks[b].instructions.size())) return false;
                auto in = out;
                for (auto it = rewritten_blocks[b].instructions.rbegin(); it != rewritten_blocks[b].instructions.rend(); ++it) {
                    const auto destinations = destination_registers(**it);
                    if (removable.contains(*it) && std::none_of(destinations.begin(), destinations.end(),
                            [&](const std::string& name) { return in.contains(name); })) continue;
                    for (const auto& name : destinations) in.erase(name);
                    for (const auto& name : source_registers(**it)) in.insert(name);
                }
                if (in != live_in[b] || out != live_out[b]) {
                    live_in[b] = std::move(in); live_out[b] = std::move(out); changed = true;
                }
            }
        }
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            auto live = live_out[b];
            std::vector<const Instruction*> retained;
            for (auto it = rewritten_blocks[b].instructions.rbegin(); it != rewritten_blocks[b].instructions.rend(); ++it) {
                const auto destinations = destination_registers(**it);
                if (removable.contains(*it) && std::none_of(destinations.begin(), destinations.end(),
                        [&](const std::string& name) { return live.contains(name); })) {
                    continue;
                }
                retained.push_back(*it);
                for (const auto& name : destinations) live.erase(name);
                for (const auto& name : source_registers(**it)) live.insert(name);
            }
            std::reverse(retained.begin(), retained.end());
            rewritten_blocks[b].instructions = std::move(retained);
        }
    }
    blocks = std::move(rewritten_blocks);
    return true;
}

}  // namespace cumetal::ir::detail
