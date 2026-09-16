#include "ptx_scalar_ranges.h"
#include "ptx_text.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <set>

namespace cumetal::ir::detail {
namespace {
std::optional<std::int64_t> literal(const std::string& text) {
    try {
        std::size_t size = 0;
        auto result = std::stoll(trim(text), &size, 0);
        if (size == trim(text).size())
            return result;
    } catch (...) {
    }
    return std::nullopt;
}
bool copy64(const Instruction& instruction) {
    return instruction.predicate.empty() && instruction.operands.size() == 2 &&
           (instruction.opcode == "mov.b64" || instruction.opcode == "mov.u64" ||
            instruction.opcode == "mov.s64") &&
           instruction.operands[1].find('{') == std::string::npos;
}
std::optional<ScalarRange> intersect(std::optional<ScalarRange> a, std::optional<ScalarRange> b) {
    if (!a)
        return b;
    if (!b)
        return a;
    ScalarRange result{std::max(a->lower, b->lower), std::min(a->upper, b->upper)};
    return result.lower <= result.upper ? std::optional(result) : std::nullopt;
}
std::optional<ScalarRange> arithmetic(ScalarRange a, ScalarRange b, bool subtract) {
    const __int128 lo =
        static_cast<__int128>(a.lower) + (subtract ? -static_cast<__int128>(b.upper) : b.lower);
    const __int128 hi =
        static_cast<__int128>(a.upper) + (subtract ? -static_cast<__int128>(b.lower) : b.upper);
    if (lo < INT64_MIN || hi > INT64_MAX)
        return std::nullopt;
    return ScalarRange{static_cast<std::int64_t>(lo), static_cast<std::int64_t>(hi)};
}
} // namespace

struct ScalarRanges::Impl {
    const std::vector<RawBlock>& blocks;
    const std::unordered_map<ValueId, Type>& types;
    const std::unordered_map<const Instruction*, std::vector<ValueId>>& results;
    std::unordered_map<ValueId, const Instruction*> definitions;
    std::unordered_map<const Instruction*, std::unordered_map<std::string, ValueId>> sources;
    std::unordered_map<const Instruction*, std::size_t> locations;
    std::unordered_map<ValueId, std::vector<ValueId>> joins;
    std::unordered_map<ValueId, std::size_t> join_blocks;
    struct Guard {
        std::size_t block, successor;
        const Instruction* comparison;
        bool truth;
    };
    std::vector<Guard> guards;
    std::vector<std::vector<std::size_t>> guards_by_block;
    std::map<std::pair<std::size_t, std::size_t>, std::vector<bool>> edge_reachable;
    std::map<std::pair<ValueId, std::size_t>, std::optional<ScalarRange>> cache;
    std::set<std::pair<ValueId, std::size_t>> active;
    std::size_t work = 0;
    static constexpr std::size_t kMaxWork = 4'000'000;

    Impl(const std::vector<RawBlock>& b,
         const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
         const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
         const std::vector<std::map<std::string, ValueId>>& arguments,
         const std::unordered_map<const Instruction*, std::vector<ValueId>>& r,
         const std::unordered_map<ValueId, Type>& t)
        : blocks(b), types(t), results(r), guards_by_block(b.size()) {
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            for (const auto& [name, id] : arguments[block]) {
                join_blocks[id] = block;
                for (auto predecessor : blocks[block].predecessors)
                    joins[id].push_back(outgoing[predecessor].at(name));
            }
            auto environment = incoming[block];
            for (const auto* instruction : blocks[block].instructions) {
                locations[instruction] = block;
                for (const auto& name : source_registers(*instruction))
                    if (environment.contains(name))
                        sources[instruction][name] = environment.at(name);
                const auto destinations = destination_registers(*instruction);
                const auto found = results.find(instruction);
                if (found == results.end())
                    continue;
                for (std::size_t i = 0; i < destinations.size() && i < found->second.size(); ++i) {
                    definitions[found->second[i]] = instruction;
                    environment[destinations[i]] = found->second[i];
                }
            }
        }
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            if (blocks[block].instructions.empty() || blocks[block].successors.size() != 2 ||
                blocks[block].successors[0] == blocks[block].successors[1])
                continue;
            const auto* branch = blocks[block].instructions.back();
            if (!is_conditional_branch(*branch))
                continue;
            const auto [name, inverted] = normalized_predicate(branch->predicate);
            auto predicate = source(branch, name);
            if (!predicate || !definitions.contains(*predicate))
                continue;
            const auto* comparison = definitions.at(*predicate);
            if (root_opcode(comparison->opcode) != "setp" || !comparison->predicate.empty() ||
                comparison->operands.size() != 3 || !results.contains(comparison) ||
                results.at(comparison).size() != 1)
                continue;
            for (std::size_t edge = 0; edge < 2; ++edge) {
                guards_by_block[block].push_back(guards.size());
                guards.push_back(
                    {block, blocks[block].successors[edge], comparison, (edge == 0) != inverted});
            }
        }
    }
    std::optional<ValueId> source(const Instruction* instruction, const std::string& operand) const {
        const auto found = sources.find(instruction);
        if (found == sources.end())
            return std::nullopt;
        const auto value = found->second.find(first_register(operand));
        return value == found->second.end() ? std::nullopt : std::optional(value->second);
    }
    // Identity is proved through all incoming edges, including anchored copy
    // cycles. A different concrete definition or disconnected cycle fails.
    bool forwards(ValueId value, ValueId target) {
        std::vector<ValueId> pending{value};
        std::unordered_set<ValueId> visited;
        std::unordered_map<ValueId, std::vector<ValueId>> reverse;
        while (!pending.empty()) {
            if (++work > kMaxWork)
                return false;
            auto current = pending.back();
            pending.pop_back();
            if (!visited.insert(current).second || current == target)
                continue;
            std::vector<ValueId> inputs;
            if (joins.contains(current))
                inputs = joins.at(current);
            else if (definitions.contains(current) && copy64(*definitions.at(current))) {
                auto input = source(definitions.at(current), definitions.at(current)->operands[1]);
                if (!input)
                    return false;
                inputs.push_back(*input);
            } else
                return false;
            if (inputs.empty())
                return false;
            for (auto input : inputs) {
                reverse[input].push_back(current);
                pending.push_back(input);
            }
        }
        if (!visited.contains(target))
            return false;
        pending = {target};
        std::unordered_set<ValueId> reached;
        while (!pending.empty()) {
            auto current = pending.back();
            pending.pop_back();
            if (!reached.insert(current).second)
                continue;
            for (auto user : reverse[current])
                pending.push_back(user);
        }
        return reached.size() == visited.size();
    }
    std::optional<ValueId> terminal(ValueId value) {
        std::unordered_set<ValueId> visited;
        std::vector<ValueId> pending{value};
        std::optional<ValueId> result;
        while (!pending.empty()) {
            if (++work > kMaxWork)
                return std::nullopt;
            auto current = pending.back();
            pending.pop_back();
            if (!visited.insert(current).second)
                continue;
            if (joins.contains(current) && !joins.at(current).empty()) {
                for (auto input : joins.at(current))
                    pending.push_back(input);
                continue;
            }
            if (definitions.contains(current) && copy64(*definitions.at(current))) {
                auto input = source(definitions.at(current), definitions.at(current)->operands[1]);
                if (input) {
                    pending.push_back(*input);
                    continue;
                }
            }
            if (result && *result != current)
                return std::nullopt;
            result = current;
        }
        return result && forwards(value, *result) ? result : std::nullopt;
    }
    bool dominates(const Guard& guard, std::size_t block) {
        auto key = std::make_pair(guard.block, guard.successor);
        if (!edge_reachable.contains(key)) {
            std::vector<bool> reachable(blocks.size(), false);
            std::vector<std::size_t> pending{0};
            while (!pending.empty()) {
                if (++work > kMaxWork)
                    return false;
                auto current = pending.back();
                pending.pop_back();
                if (reachable[current])
                    continue;
                reachable[current] = true;
                for (auto next : blocks[current].successors)
                    if (current != guard.block || next != guard.successor)
                        pending.push_back(next);
            }
            edge_reachable.emplace(key, std::move(reachable));
        }
        return !edge_reachable.at(key)[block];
    }
    std::optional<ScalarRange> operand(const Instruction* instruction, std::size_t index, std::size_t at) {
        if (index >= instruction->operands.size())
            return std::nullopt;
        if (auto n = literal(instruction->operands[index]))
            return ScalarRange{*n, *n};
        auto id = source(instruction, instruction->operands[index]);
        return id ? get(*id, at) : std::nullopt;
    }
    std::optional<ScalarRange> guard_bound(ValueId value, const Guard& guard, std::size_t at) {
        const auto* comparison = guard.comparison;
        for (std::size_t index : {1U, 2U}) {
            auto input = source(comparison, comparison->operands[index]);
            if (!input || (!forwards(*input, value) && !forwards(value, *input)))
                continue;
            auto limit = operand(comparison, 3 - index, at);
            if (!limit || limit->lower < 0)
                continue;
            auto opcode = comparison->opcode;
            std::string relation;
            for (auto op : {"lt", "le", "gt", "ge", "eq", "ne"})
                if (opcode == std::string("setp.") + op + ".u64" ||
                    ((std::string(op) == "eq" || std::string(op) == "ne") &&
                     (opcode == std::string("setp.") + op + ".s64" ||
                      opcode == std::string("setp.") + op + ".b64")))
                    relation = op;
            if (relation.empty())
                continue;
            if (index == 2) {
                if (relation == "lt")
                    relation = "gt";
                else if (relation == "le")
                    relation = "ge";
                else if (relation == "gt")
                    relation = "lt";
                else if (relation == "ge")
                    relation = "le";
            }
            if (!guard.truth) {
                if (relation == "lt")
                    relation = "ge";
                else if (relation == "le")
                    relation = "gt";
                else if (relation == "gt")
                    relation = "le";
                else if (relation == "ge")
                    relation = "lt";
                else if (relation == "eq")
                    relation = "ne";
                else if (relation == "ne")
                    relation = "eq";
            }
            if (relation == "lt" && limit->upper > 0)
                return ScalarRange{0, limit->upper - 1};
            if (relation == "le")
                return ScalarRange{0, limit->upper};
            if (relation == "eq")
                return limit;
            // A lower unsigned bound alone cannot exclude values > INT64_MAX.
        }
        return std::nullopt;
    }
    std::optional<ScalarRange> incoming_ranges(ValueId value) {
        const auto block = join_blocks.at(value);
        std::optional<ScalarRange> merged;
        for (std::size_t index = 0; index < joins.at(value).size(); ++index) {
            if (++work > kMaxWork)
                return std::nullopt;
            const auto input = joins.at(value)[index];
            const auto predecessor = blocks[block].predecessors[index];
            std::optional<ScalarRange> range;
            // This fact belongs to the edge, not the predecessor block: its
            // comparison can still be false earlier in that block.
            for (auto guard_index : guards_by_block[predecessor]) {
                if (++work > kMaxWork)
                    return std::nullopt;
                const auto& guard = guards[guard_index];
                if (guard.successor == block)
                    range = intersect(range, guard_bound(input, guard, predecessor));
            }
            if (!range)
                range = get(input, predecessor);
            if (!range)
                return std::nullopt;
            if (!merged)
                merged = range;
            else {
                merged->lower = std::min(merged->lower, range->lower);
                merged->upper = std::max(merged->upper, range->upper);
            }
        }
        return merged;
    }
    std::optional<ScalarRange> induction(ValueId header) {
        if (!joins.contains(header))
            return std::nullopt;
        auto block = join_blocks.at(header);
        std::optional<std::int64_t> start;
        std::optional<ValueId> update;
        std::vector<std::size_t> backedges;
        for (std::size_t i = 0; i < joins.at(header).size(); ++i) {
            auto terminal_value = terminal(joins.at(header)[i]);
            if (!terminal_value)
                return std::nullopt;
            auto input = *terminal_value;
            if (definitions.contains(input)) {
                const auto* definition = definitions.at(input);
                if (copy64(*definition))
                    if (auto n = literal(definition->operands[1])) {
                        if (start && *start != *n)
                            return std::nullopt;
                        start = *n;
                        continue;
                    }
                if (definition->predicate.empty() && definition->operands.size() == 3 &&
                    (definition->opcode == "add.u64" || definition->opcode == "add.s64")) {
                    bool right = literal(definition->operands[2]) == std::optional<std::int64_t>{1};
                    bool left = literal(definition->operands[1]) == std::optional<std::int64_t>{1};
                    auto prior = source(definition, definition->operands[right ? 1 : 2]);
                    if ((right || left) && prior && forwards(*prior, header) &&
                        (!update || *update == input)) {
                        update = input;
                        backedges.push_back(blocks[block].predecessors[i]);
                        continue;
                    }
                }
            }
            return std::nullopt;
        }
        if (!start || *start < 0 || !update)
            return std::nullopt;
        for (const auto& guard : guards) {
            auto bounded = guard_bound(*update, guard, guard.block);
            bool prior = false;
            if (!bounded) {
                bounded = guard_bound(header, guard, guard.block);
                prior = true;
            }
            if (!bounded || bounded->upper >= INT64_MAX - 1)
                continue;
            bool guarded = true;
            for (auto edge : backedges)
                if (!(guard.block == edge && guard.successor == block) && !dominates(guard, edge))
                    guarded = false;
            if (!guarded)
                continue;
            auto maximum = bounded->upper + (prior ? 1 : 0);
            if (maximum >= *start)
                return ScalarRange{*start, maximum};
        }
        return std::nullopt;
    }
    std::optional<ScalarRange> exclude_zero(ValueId value, std::size_t at, std::optional<ScalarRange> range) {
        // Unsigned !=0 alone still admits high-bit values. Tighten the lower
        // endpoint only after an independent proof bounds the value to a
        // nonnegative signed interval.
        if (!range || range->lower != 0 || range->upper <= 0)
            return range;
        for (const auto& guard : guards) {
            const auto* comparison = guard.comparison;
            const auto& opcode = comparison->opcode;
            const bool eq = opcode == "setp.eq.u64" || opcode == "setp.eq.s64" || opcode == "setp.eq.b64";
            const bool ne = opcode == "setp.ne.u64" || opcode == "setp.ne.s64" || opcode == "setp.ne.b64";
            if ((!eq || guard.truth) && (!ne || !guard.truth))
                continue;
            for (std::size_t index : {1U, 2U}) {
                if (literal(comparison->operands[3 - index]) != std::optional<std::int64_t>{0})
                    continue;
                const auto input = source(comparison, comparison->operands[index]);
                if (input && (forwards(*input, value) || forwards(value, *input)) && dominates(guard, at)) {
                    range->lower = 1;
                    return range;
                }
            }
        }
        return range;
    }
    std::optional<ScalarRange> get(ValueId value, std::size_t at) {
        if (++work > kMaxWork || !types.contains(value) || types.at(value) != Type::integer(64))
            return std::nullopt;
        const auto key = std::make_pair(value, at);
        if (cache.contains(key))
            return cache.at(key);
        if (active.size() > 256 || !active.insert(key).second)
            return std::nullopt;
        std::optional<ScalarRange> result;
        // A dominating guard is already a complete bound on this value. Use
        // it before recursively exploring the possibly expensive computation
        // that produced the value. Forwarded SSA copies are equal in either
        // direction only when every incoming edge supplies the same origin.
        for (const auto& guard : guards) {
            const auto left = source(guard.comparison, guard.comparison->operands[1]);
            const auto right = source(guard.comparison, guard.comparison->operands[2]);
            const auto same = [&](std::optional<ValueId> input) {
                return input && (forwards(*input, value) || forwards(value, *input));
            };
            if ((same(left) || same(right)) && dominates(guard, at)) {
                const auto bound = exclude_zero(value, at, guard_bound(value, guard, at));
                if (bound && work <= kMaxWork) {
                    active.erase(key);
                    cache[key] = bound;
                    return bound;
                }
            }
        }
        if (joins.contains(value)) {
            auto identity = terminal(value);
            if (identity && *identity != value)
                result = get(*identity, at);
            if (!result)
                result = induction(value);
            if (!result)
                result = incoming_ranges(value);
            if (!result) {
                bool complete = true;
                for (auto input : joins.at(value)) {
                    auto range = get(input, at);
                    if (!range) {
                        complete = false;
                        break;
                    }
                    if (!result)
                        result = range;
                    else {
                        result->lower = std::min(result->lower, range->lower);
                        result->upper = std::max(result->upper, range->upper);
                    }
                }
                if (!complete)
                    result.reset();
            }
        } else if (definitions.contains(value)) {
            const auto* instruction = definitions.at(value);
            if (copy64(*instruction))
                result = operand(instruction, 1, at);
            else if (instruction->predicate.empty() && instruction->opcode == "not.b64" &&
                     instruction->operands.size() == 2) {
                if (auto input = operand(instruction, 1, at)) {
                    const __int128 lower = -1 - static_cast<__int128>(input->upper);
                    const __int128 upper = -1 - static_cast<__int128>(input->lower);
                    if (lower >= INT64_MIN && upper <= INT64_MAX)
                        result =
                            ScalarRange{static_cast<std::int64_t>(lower), static_cast<std::int64_t>(upper)};
                }
            } else if (instruction->predicate.empty() && instruction->operands.size() == 3) {
                if (instruction->opcode == "shr.u64") {
                    const auto shift = literal(instruction->operands[2]);
                    const auto input = operand(instruction, 1, at);
                    if (shift && *shift >= 0 && *shift < 64 && input && input->lower >= 0)
                        result = ScalarRange{input->lower >> *shift, input->upper >> *shift};
                } else if (instruction->opcode == "and.b64") {
                    auto mask = literal(instruction->operands[2]);
                    if (!mask)
                        mask = literal(instruction->operands[1]);
                    if (mask && *mask >= 0)
                        result = ScalarRange{0, *mask};
                } else if (instruction->opcode == "add.s64" || instruction->opcode == "add.u64" ||
                           instruction->opcode == "sub.s64" || instruction->opcode == "sub.u64") {
                    auto a = operand(instruction, 1, at), b = operand(instruction, 2, at);
                    if (a && b)
                        result = arithmetic(*a, *b, root_opcode(instruction->opcode) == "sub");
                }
            }
        }
        for (const auto& guard : guards) {
            // Avoid computing reachability for guards unrelated to this value.
            auto left = source(guard.comparison, guard.comparison->operands[1]);
            auto right = source(guard.comparison, guard.comparison->operands[2]);
            if ((!left || (!forwards(*left, value) && !forwards(value, *left))) &&
                (!right || (!forwards(*right, value) && !forwards(value, *right))))
                continue;
            if (dominates(guard, at))
                result = intersect(result, guard_bound(value, guard, at));
        }
        result = exclude_zero(value, at, result);
        active.erase(key);
        if (work > kMaxWork)
            return std::nullopt;
        cache[key] = result;
        return result;
    }
};

ScalarRanges::ScalarRanges(const std::vector<RawBlock>& blocks,
                           const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
                           const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
                           const std::vector<std::map<std::string, ValueId>>& arguments,
                           const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
                           const std::unordered_map<ValueId, Type>& types)
    : impl_(std::make_unique<Impl>(blocks, incoming, outgoing, arguments, results, types)) {}
ScalarRanges::~ScalarRanges() = default;
std::optional<ScalarRange> ScalarRanges::get(ValueId value, const Instruction* at) {
    auto location = impl_->locations.find(at);
    if (location == impl_->locations.end())
        return std::nullopt;
    return impl_->get(value, location->second);
}
} // namespace cumetal::ir::detail
