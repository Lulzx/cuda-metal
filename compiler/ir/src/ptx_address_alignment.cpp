#include "ptx_address_alignment.h"

#include "ptx_text.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_set>

namespace cumetal::ir::detail {
namespace {

std::uint64_t low_mask(unsigned bits) {
    return bits == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;
}

// Congruences describe low bits, not numeric addresses. A local fact must
// originate at a declared allocation. In particular, provisional pointer types
// and cvta instructions with arbitrary integer inputs do not seed provenance.
struct Fact {
    enum Kind { pending, scalar, local, opaque } kind = pending;
    unsigned bits = 0;
    std::uint64_t residue = 0;
    bool operator==(const Fact&) const = default;
};

Fact unknown() { return {Fact::opaque, 0, 0}; }

Fact meet(Fact a, Fact b) {
    if (a.kind == Fact::pending)
        return b;
    if (b.kind == Fact::pending)
        return a;
    if (a.kind != b.kind || a.kind == Fact::opaque)
        return unknown();
    const unsigned different = std::countr_zero(a.residue ^ b.residue);
    const unsigned bits = std::min({a.bits, b.bits, different});
    return {a.kind, bits, a.residue & low_mask(bits)};
}

// Parse only complete PTX integer literals. Modular negatives are useful for
// address subtraction/addition; OR masks below deliberately require a positive
// signed-representable offset. No floating encodings or expressions qualify.
std::optional<std::uint64_t> integer(std::string token) {
    token = trim(token);
    while (!token.empty() &&
           (token.back() == 'U' || token.back() == 'u' || token.back() == 'L' || token.back() == 'l'))
        token.pop_back();
    bool negative = false;
    if (!token.empty() && (token.front() == '-' || token.front() == '+')) {
        negative = token.front() == '-';
        token.erase(token.begin());
    }
    int radix = 10;
    if (token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) {
        radix = 16;
        token.erase(0, 2);
    }
    if (token.empty())
        return std::nullopt;
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value, radix);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size())
        return std::nullopt;
    return negative ? std::uint64_t{0} - value : value;
}

struct Input {
    std::optional<ValueId> value;
    Fact constant = unknown();
};

struct Node {
    enum Kind { leaf, copy, add, subtract, join, select, address_convert, bit_or } kind = leaf;
    Fact fact = unknown();
    std::vector<Input> inputs;
    const Instruction* instruction = nullptr;
    // For a candidate OR: which source operand is the address, and its mask.
    std::size_t address_operand = 0;
    std::uint64_t mask = 0;
};

} // namespace

AddressAlignmentResult legalize_aligned_local_address_ors(
    std::vector<RawBlock>& blocks, const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
    const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
    const std::vector<std::map<std::string, ValueId>>& arguments,
    const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
    const std::unordered_map<ValueId, Type>& provisional_types,
    const std::unordered_map<std::string, LocalDepot>& local_depots, std::deque<Instruction>& storage,
    InstructionOrigins* origins, AddressAlignmentLimits limits) {
    if (incoming.size() != blocks.size() || outgoing.size() != blocks.size() ||
        arguments.size() != blocks.size())
        return {};
    std::size_t work = 0, edges = 0;
    const auto consume = [&](std::size_t amount = 1) {
        if (amount > limits.max_work - work)
            return false;
        work += amount;
        return true;
    };
    const AddressAlignmentResult exhausted{0, true};
    std::unordered_map<ValueId, Node> nodes;
    std::unordered_map<ValueId, std::vector<ValueId>> users;
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        if (!consume(incoming[b].size() + 1))
            return exhausted;
        auto environment = incoming[b];
        for (const auto* instruction : blocks[b].instructions) {
            if (!consume())
                return exhausted;
            const auto destinations = destination_registers(*instruction);
            const auto found = results.find(instruction);
            if (found == results.end())
                continue;
            for (std::size_t lane = 0; lane < destinations.size() && lane < found->second.size(); ++lane) {
                if (!consume() || nodes.size() >= limits.max_values)
                    return exhausted;
                const ValueId value = found->second[lane];
                Node node;
                node.instruction = instruction;
                const auto type = provisional_types.find(value);
                if (type != provisional_types.end() && type->second.kind == TypeKind::kInteger)
                    node.fact = {Fact::scalar, 0, 0};
                const auto input = [&](std::size_t operand) -> Input {
                    const auto text = trim(instruction->operands[operand]);
                    const auto reg = first_register(text);
                    if (!reg.empty()) {
                        if (text != reg)
                            return {};
                        const auto source = environment.find(reg);
                        return source == environment.end() ? Input{} : Input{source->second, {}};
                    }
                    if (const auto literal = integer(text))
                        return {{}, {Fact::scalar, 64, *literal}};
                    const auto depot = local_depots.find(text);
                    if (depot != local_depots.end() && std::has_single_bit(depot->second.alignment))
                        return {{},
                                {Fact::local,
                                 static_cast<unsigned>(std::countr_zero(depot->second.alignment)), 0}};
                    return {};
                };
                const auto& opcode = instruction->opcode;
                if (destinations.size() == 1 && instruction->predicate.empty()) {
                    if ((opcode == "mov.u64" || opcode == "mov.s64" || opcode == "mov.b64") &&
                        instruction->operands.size() == 2 &&
                        instruction->operands[1].find('{') == std::string::npos) {
                        node.kind = Node::copy;
                        node.inputs = {input(1)};
                    } else if ((opcode == "add.u64" || opcode == "add.s64" || opcode == "sub.u64" ||
                                opcode == "sub.s64") &&
                               instruction->operands.size() == 3) {
                        node.kind = starts_with(opcode, "add.") ? Node::add : Node::subtract;
                        node.inputs = {input(1), input(2)};
                    } else if ((opcode == "selp.u64" || opcode == "selp.s64" || opcode == "selp.b64") &&
                               instruction->operands.size() == 4) {
                        node.kind = Node::select;
                        node.inputs = {input(1), input(2)};
                    } else if ((opcode == "cvta.local.u64" || opcode == "cvta.to.local.u64") &&
                               instruction->operands.size() == 2) {
                        node.kind = Node::address_convert;
                        node.inputs = {input(1)};
                    } else if (opcode == "or.b64" && instruction->operands.size() == 3) {
                        // An unproved pointer OR must never be treated as a
                        // scalar offset that seeds a new allocation proof.
                        node.fact = unknown();
                        for (std::size_t literal_operand : {2u, 1u}) {
                            const auto mask = integer(instruction->operands[literal_operand]);
                            if (!mask ||
                                *mask > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                                continue;
                            node.kind = Node::bit_or;
                            node.address_operand = 3 - literal_operand;
                            node.mask = *mask;
                            node.inputs = {input(node.address_operand)};
                            break;
                        }
                    }
                } else if (!instruction->predicate.empty()) {
                    node.fact = unknown();
                }
                if (node.kind != Node::leaf)
                    node.fact = {};
                if (node.inputs.size() > limits.max_edges - edges)
                    return exhausted;
                edges += node.inputs.size();
                nodes.emplace(value, std::move(node));
            }
            // Capture every source above before replacing any in-place result.
            for (std::size_t lane = 0; lane < destinations.size() && lane < found->second.size(); ++lane)
                environment[destinations[lane]] = found->second[lane];
        }
        for (const auto& [name, value] : arguments[b]) {
            if (!consume() || nodes.size() >= limits.max_values)
                return exhausted;
            Node node;
            node.kind = Node::join;
            node.fact = {};
            for (const auto predecessor : blocks[b].predecessors) {
                if (!consume())
                    return exhausted;
                if (edges >= limits.max_edges)
                    return exhausted;
                ++edges;
                if (predecessor >= outgoing.size())
                    return {};
                const auto source = outgoing[predecessor].find(name);
                if (source == outgoing[predecessor].end())
                    return {};
                node.inputs.push_back({source->second, {}});
            }
            nodes.emplace(value, std::move(node));
        }
    }
    for (const auto& [value, node] : nodes) {
        for (const auto& input : node.inputs) {
            if (!consume())
                return exhausted;
            if (input.value)
                users[*input.value].push_back(value);
        }
    }
    const auto fact = [&](const Input& input) {
        if (!input.value)
            return input.constant;
        const auto source = nodes.find(*input.value);
        return source == nodes.end() ? unknown() : source->second.fact;
    };
    const auto evaluate = [&](const Node& node) -> Fact {
        if (node.kind == Node::leaf)
            return node.fact;
        if (node.inputs.empty())
            return unknown();
        if (node.kind == Node::join || node.kind == Node::select) {
            Fact joined;
            for (const auto& input : node.inputs)
                joined = meet(joined, fact(input));
            return joined;
        }
        const auto left = fact(node.inputs[0]);
        if (left.kind == Fact::pending)
            return left;
        if (node.kind == Node::copy)
            return left;
        if (node.kind == Node::address_convert) {
            // Conversion of an actual local allocation preserves its storage
            // alignment and byte displacement, not equality of generic/local
            // numeric representations. Other conversions do not qualify.
            return left.kind == Fact::local ? left : unknown();
        }
        if (node.kind == Node::bit_or) {
            const auto zero_bits = low_mask(left.bits) & ~left.residue;
            if (left.kind != Fact::local || (node.mask & ~zero_bits) != 0)
                return unknown();
            return {Fact::local, left.bits, left.residue | node.mask};
        }
        const auto right = fact(node.inputs[1]);
        if (right.kind == Fact::pending)
            return right;
        const bool scalar_pair = left.kind == Fact::scalar && right.kind == Fact::scalar;
        const bool pointer_offset = left.kind == Fact::local && right.kind == Fact::scalar;
        const bool commuted =
            node.kind == Node::add && left.kind == Fact::scalar && right.kind == Fact::local;
        if (!scalar_pair && !pointer_offset && !commuted)
            return unknown();
        const unsigned bits = std::min(left.bits, right.bits);
        const auto residue =
            node.kind == Node::add ? left.residue + right.residue : left.residue - right.residue;
        return {scalar_pair ? Fact::scalar : Fact::local, bits, residue & low_mask(bits)};
    };
    std::deque<ValueId> pending;
    std::unordered_set<ValueId> queued;
    for (const auto& [value, node] : nodes) {
        pending.push_back(value);
        queued.insert(value);
    }
    const auto enqueue_users = [&](ValueId value) {
        if (!consume(users[value].size()))
            return false;
        for (const auto user : users[value])
            if (queued.insert(user).second)
                pending.push_back(user);
        return true;
    };
    const auto propagate = [&]() {
        while (!pending.empty()) {
            if (!consume())
                return false;
            const auto value = pending.front();
            pending.pop_front();
            queued.erase(value);
            auto& node = nodes.at(value);
            if (!consume(node.inputs.size()))
                return false;
            auto next = meet(node.fact, evaluate(node));
            if (next == node.fact)
                continue;
            node.fact = next;
            if (!enqueue_users(value))
                return false;
        }
        return true;
    };
    if (!propagate())
        return exhausted;
    // Unknown SCCs cannot lend facts to seeded joins. Make every unresolved
    // node opaque, then propagate that loss through all uses before committing.
    for (auto& [value, node] : nodes) {
        if (!consume())
            return exhausted;
        if (node.fact.kind != Fact::pending)
            continue;
        node.fact = unknown();
        if (!enqueue_users(value))
            return exhausted;
    }
    if (!propagate())
        return exhausted;

    std::unordered_map<const Instruction*, Instruction> replacements;
    for (const auto& [value, node] : nodes) {
        if (!consume())
            return exhausted;
        if (node.kind != Node::bit_or || node.fact.kind != Fact::local)
            continue;
        const auto input = fact(node.inputs.front());
        const auto zero_bits = low_mask(input.bits) & ~input.residue;
        if (input.kind != Fact::local || (node.mask & ~zero_bits) != 0)
            continue;
        Instruction replacement = *node.instruction;
        replacement.opcode = "add.u64";
        replacement.operands = {node.instruction->operands[0],
                                node.instruction->operands[node.address_operand],
                                node.instruction->operands[3 - node.address_operand]};
        replacements.emplace(node.instruction, std::move(replacement));
    }
    if (replacements.empty())
        return {};
    // No proof budget can fail once mutation begins. A one-for-one rewrite
    // preserves every effect and control-flow edge, including unsupported
    // producers which final type/materialization validation must still reject.
    for (auto& block : blocks) {
        for (auto*& instruction : block.instructions) {
            const auto found = replacements.find(instruction);
            if (found == replacements.end())
                continue;
            storage.push_back(found->second);
            record_instruction_origin(origins, &storage.back(), instruction);
            instruction = &storage.back();
        }
    }
    return {replacements.size(), false};
}

} // namespace cumetal::ir::detail
