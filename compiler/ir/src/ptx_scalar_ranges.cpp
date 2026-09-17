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
    std::unordered_map<const Instruction*, std::size_t> ordinals;
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
    const std::size_t kMaxWork;

    Impl(const std::vector<RawBlock>& b,
         const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
         const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
         const std::vector<std::map<std::string, ValueId>>& arguments,
         const std::unordered_map<const Instruction*, std::vector<ValueId>>& r,
         const std::unordered_map<ValueId, Type>& t, ScalarRangeLimits limits)
        : blocks(b), types(t), results(r), guards_by_block(b.size()), kMaxWork(limits.work) {
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            for (const auto& [name, id] : arguments[block]) {
                join_blocks[id] = block;
                for (auto predecessor : blocks[block].predecessors)
                    joins[id].push_back(outgoing[predecessor].at(name));
            }
            auto environment = incoming[block];
            std::size_t ordinal = 0;
            for (const auto* instruction : blocks[block].instructions) {
                locations[instruction] = block;
                ordinals[instruction] = ordinal++;
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
            const auto predicate = source(branch, name);
            if (!predicate)
                continue;
            for (std::size_t edge = 0; edge < 2; ++edge) {
                const bool truth = (edge == 0) != inverted;
                std::vector<std::pair<ValueId, bool>> pending{{*predicate, truth}};
                std::set<std::pair<ValueId, bool>> visited;
                std::vector<Guard> derived;
                constexpr std::size_t kMaxConditionValues = 512;
                bool complete = true;
                while (!pending.empty()) {
                    if (++work > kMaxWork || visited.size() >= kMaxConditionValues) {
                        complete = false;
                        break;
                    }
                    const auto condition = pending.back();
                    pending.pop_back();
                    if (!visited.insert(condition).second)
                        continue;
                    const auto found = definitions.find(condition.first);
                    if (found == definitions.end())
                        continue;
                    const auto* definition = found->second;
                    if (!definition->predicate.empty() || !results.contains(definition) ||
                        results.at(definition).size() != 1)
                        continue;
                    if (root_opcode(definition->opcode) == "setp" && definition->operands.size() == 3) {
                        derived.push_back(
                            {block, blocks[block].successors[edge], definition, condition.second});
                        continue;
                    }
                    const auto follow = [&](std::size_t operand, bool value) {
                        if (const auto input = source(definition, definition->operands[operand]))
                            pending.emplace_back(*input, value);
                    };
                    if (definition->operands.size() == 2 &&
                        (definition->opcode == "mov.pred" || definition->opcode == "not.pred")) {
                        follow(1, definition->opcode == "not.pred" ? !condition.second : condition.second);
                    } else if (definition->operands.size() == 3 &&
                               ((definition->opcode == "and.pred" && condition.second) ||
                                (definition->opcode == "or.pred" && !condition.second))) {
                        // AND true requires both operands true; OR false
                        // requires both false. The opposite outcomes imply
                        // neither individual fact and are left unknown.
                        follow(1, condition.second);
                        follow(2, condition.second);
                    }
                }
                if (!complete)
                    continue;
                for (const auto& guard : derived) {
                    guards_by_block[block].push_back(guards.size());
                    guards.push_back(guard);
                }
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
    std::map<std::pair<ValueId, ValueId>, bool> forward_cache;
    // Identity depends only on this immutable SSA graph, not the query's use
    // block. Reuse completed positive and negative proofs across store offsets.
    // The cap bounds retained facts; exhaustion must never be memoized as a
    // semantic refusal. This cache lives for one ScalarRanges instance only.
    bool forwards(ValueId value, ValueId target) {
        const auto key = std::make_pair(value, target);
        if (const auto found = forward_cache.find(key); found != forward_cache.end()) return found->second;
        const bool result = prove_forwarding(value, target);
        if (work <= kMaxWork && forward_cache.size() < 65536) forward_cache.emplace(key, result);
        return result;
    }
    // Identity is proved through all incoming edges, including anchored copy
    // cycles. A different concrete definition or disconnected cycle fails.
    bool prove_forwarding(ValueId value, ValueId target) {
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
    struct Affine {
        ValueId base;
        std::int64_t offset;
    };
    std::unordered_map<ValueId, std::optional<Affine>> affine_cache;
    std::unordered_set<ValueId> affine_active;

    std::optional<Affine> affine(ValueId value) {
        if (const auto found = affine_cache.find(value); found != affine_cache.end())
            return found->second;
        if (++work > kMaxWork || !types.contains(value) || types.at(value) != Type::integer(64) ||
            affine_active.size() >= 256 || !affine_active.insert(value).second)
            return std::nullopt;
        // Exact 64-bit add/sub identities are congruences modulo 2^64. A
        // range is transferred only when its shifted signed representative
        // stays representable; no claim is made about the original base range.
        std::optional<Affine> result = Affine{value, 0};
        if (joins.contains(value)) {
            const auto origin = terminal(value);
            if (origin && *origin != value)
                result = affine(*origin);
        } else if (const auto found = definitions.find(value); found != definitions.end()) {
            const auto* instruction = found->second;
            if (copy64(*instruction)) {
                const auto input = source(instruction, instruction->operands[1]);
                if (input)
                    result = affine(*input);
            } else if (instruction->predicate.empty() && instruction->operands.size() == 3 &&
                       (instruction->opcode == "add.u64" || instruction->opcode == "add.s64" ||
                        instruction->opcode == "sub.u64" || instruction->opcode == "sub.s64")) {
                const bool subtract = root_opcode(instruction->opcode) == "sub";
                for (std::size_t index : {1U, 2U}) {
                    if (subtract && index == 2)
                        break;
                    const auto constant = literal(instruction->operands[3 - index]);
                    const auto input = source(instruction, instruction->operands[index]);
                    if (!constant || !input)
                        continue;
                    const auto base = affine(*input);
                    if (!base) {
                        result.reset();
                        break;
                    }
                    const __int128 offset = static_cast<__int128>(base->offset) +
                                            (subtract ? -static_cast<__int128>(*constant) : *constant);
                    if (offset < INT64_MIN || offset > INT64_MAX)
                        result.reset();
                    else
                        result = Affine{base->base, static_cast<std::int64_t>(offset)};
                    break;
                }
            }
        }
        affine_active.erase(value);
        if (work > kMaxWork)
            return std::nullopt;
        affine_cache.emplace(value, result);
        return result;
    }
    std::optional<std::int64_t> relative_offset(ValueId value, ValueId compared) {
        if (value == compared) return 0;
        // Most comparisons concern different concrete computations. Resolve
        // cached affine origins before considering copy/join reachability;
        // walking both identity graphs for every unrelated guard consumes the
        // proof budget quadratically in large generated functions.
        const auto a = affine(value), b = affine(compared);
        const auto may_forward = [&](ValueId from, ValueId to) {
            if (from == to) return true;
            const auto definition = definitions.find(from);
            if (!joins.contains(from) &&
                (definition == definitions.end() || !copy64(*definition->second))) return false;
            return forwards(from, to);
        };
        if (!a || !b) {
            // Affine overflow can leave an otherwise exact copy unmodelled.
            // Do not lose that identity, and never turn failed range analysis
            // into evidence that different definitions are equal.
            return may_forward(value, compared) || may_forward(compared, value)
                ? std::optional<std::int64_t>{0} : std::nullopt;
        }
        if (a->base != b->base) {
            // A successful affine origin has already followed its exact
            // copies and any single-terminal join. Different origins can
            // still relay one another only when both are unresolved joins
            // (for example phi(other_join, other_join)). A concrete terminal
            // cannot be equal to an unresolved multi-terminal/cyclic join.
            if (!joins.contains(a->base) || !joins.contains(b->base)) return std::nullopt;
            if (!may_forward(a->base, b->base) && !may_forward(b->base, a->base)) return std::nullopt;
        }
        const __int128 delta = static_cast<__int128>(a->offset) - b->offset;
        return delta < INT64_MIN || delta > INT64_MAX ? std::nullopt
                                                      : std::optional(static_cast<std::int64_t>(delta));
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
            const auto delta = input ? relative_offset(value, *input) : std::nullopt;
            if (!delta)
                continue;
            const auto shifted = [&](ScalarRange bound) {
                return arithmetic(bound, ScalarRange{*delta, *delta}, false);
            };
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
                return shifted(ScalarRange{0, limit->upper - 1});
            if (relation == "le")
                return shifted(ScalarRange{0, limit->upper});
            if (relation == "eq")
                return shifted(*limit);
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
    std::optional<ScalarRange> exclude_endpoints(ValueId value, std::size_t at,
                                                 std::optional<ScalarRange> range) {
        if (!range)
            return range;
        // A disequality removes a point, not a whole side of the number line.
        // It tightens an interval only when that point is an endpoint. Gather
        // all proven exclusions before trimming so CFG/source order cannot
        // hide successive excluded endpoints. Interior holes remain in the
        // conservative enclosing interval.
        std::set<std::int64_t> excluded;
        for (const auto& guard : guards) {
            if (++work > kMaxWork)
                return std::nullopt;
            const auto* comparison = guard.comparison;
            const auto& op = comparison->opcode;
            const bool eq = op == "setp.eq.u64" || op == "setp.eq.s64" || op == "setp.eq.b64";
            const bool ne = op == "setp.ne.u64" || op == "setp.ne.s64" || op == "setp.ne.b64";
            if ((!eq || guard.truth) && (!ne || !guard.truth))
                continue;
            for (std::size_t index : {1U, 2U}) {
                const auto point = literal(comparison->operands[3 - index]);
                const auto input = source(comparison, comparison->operands[index]);
                if (!point || !input)
                    continue;
                const auto delta = relative_offset(value, *input);
                if (!delta)
                    continue;
                const __int128 shifted = static_cast<__int128>(*point) + *delta;
                if (shifted < range->lower || shifted > range->upper)
                    continue;
                if (dominates(guard, at))
                    excluded.insert(static_cast<std::int64_t>(shifted));
            }
        }
        while (excluded.contains(range->lower)) {
            if (range->lower == range->upper)
                return std::nullopt;
            ++range->lower;
        }
        while (excluded.contains(range->upper)) {
            if (range->lower == range->upper)
                return std::nullopt;
            --range->upper;
        }
        return range;
    }
    // Check the dynamic lifetime, not just the register name or static SSA
    // identity: a loop phi can execute again while a pointer retains its old
    // offset. Only changes on a path from creation to a use, without another
    // execution of creation, invalidate a later guard for that captured value.
    bool unchanged_since(ValueId value, const Instruction* creation, const Instruction* use) {
        const auto capture_block = locations.at(creation), capture_index = ordinals.at(creation);
        std::vector<std::set<std::size_t>> changes(blocks.size());
        std::vector<bool> changes_at_entry(blocks.size(), false);
        std::vector<ValueId> dependencies{value};
        std::set<ValueId> seen_values;
        while (!dependencies.empty()) {
            if (++work > kMaxWork) return false;
            const auto current = dependencies.back();
            dependencies.pop_back();
            if (!seen_values.insert(current).second) continue;
            if (joins.contains(current)) {
                changes_at_entry[join_blocks.at(current)] = true;
                for (auto input : joins.at(current)) dependencies.push_back(input);
            } else if (definitions.contains(current)) {
                const auto* definition = definitions.at(current);
                changes[locations.at(definition)].insert(ordinals.at(definition));
                // Loads snapshot their result. The addresses used by an older
                // load need not remain unchanged after that load completes.
                if (root_opcode(definition->opcode) != "ld" && sources.contains(definition))
                    for (const auto& [name, input] : sources.at(definition)) dependencies.push_back(input);
            } else {
                return false;
            }
        }
        using Point = std::pair<std::size_t, std::size_t>;
        // Instruction intervals from which the use is reachable without
        // crossing creation. Keep the initial partial block separate from a
        // full backedge visit; creation may lie between the two intervals.
        std::vector<std::vector<Point>> before_use(blocks.size());
        std::vector<bool> entry_before_use(blocks.size(), false);
        std::vector<Point> pending{{locations.at(use), ordinals.at(use)}};
        std::set<Point> visited;
        while (!pending.empty()) {
            if (++work > kMaxWork) return false;
            const auto [block, end] = pending.back();
            pending.pop_back();
            if (!visited.emplace(block, end).second) continue;
            const bool cut = block == capture_block && capture_index < end;
            before_use[block].emplace_back(cut ? capture_index + 1 : 0, end);
            if (!cut) {
                entry_before_use[block] = true;
                for (auto predecessor : blocks[block].predecessors)
                    pending.emplace_back(predecessor, blocks[predecessor].instructions.size());
            }
        }
        pending = {{capture_block, capture_index + 1}};
        visited.clear();
        while (!pending.empty()) {
            if (++work > kMaxWork) return false;
            const auto [block, begin] = pending.back();
            pending.pop_back();
            if (!visited.emplace(block, begin).second) continue;
            const bool cut = block == capture_block && begin <= capture_index;
            const auto end = cut ? capture_index : blocks[block].instructions.size();
            if (begin == 0 && entry_before_use[block] && changes_at_entry[block]) return false;
            for (auto changed = changes[block].lower_bound(begin);
                 changed != changes[block].end() && *changed < end; ++changed) {
                if (++work > kMaxWork) return false;
                for (const auto& [lo, hi] : before_use[block])
                    if (lo <= *changed && *changed < hi) return false;
            }
            if (!cut)
                for (auto successor : blocks[block].successors) pending.emplace_back(successor, 0);
        }
        return work <= kMaxWork;
    }
    std::optional<ScalarRange> captured(ValueId value, const Instruction* creation, const Instruction* use) {
        if (!locations.contains(creation) || !locations.contains(use)) return std::nullopt;
        const auto original = get(value, locations.at(creation));
        if (creation == use || work > kMaxWork) return original;
        const auto later = get(value, locations.at(use));
        if (!later || (original && later->lower <= original->lower && later->upper >= original->upper))
            return original;
        return unchanged_since(value, creation, use) ? intersect(original, later) : original;
    }
    std::optional<ScalarRange> get(ValueId value, std::size_t at) {
        if (!types.contains(value) || types.at(value) != Type::integer(64))
            return std::nullopt;
        const auto key = std::make_pair(value, at);
        // A completed fact is immutable for this SSA graph. Exhaustion limits
        // new proof work; it does not invalidate a previously proved answer.
        if (cache.contains(key))
            return cache.at(key);
        if (++work > kMaxWork)
            return std::nullopt;
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
                return input && relative_offset(value, *input).has_value();
            };
            if ((same(left) || same(right)) && dominates(guard, at))
                result = intersect(result, guard_bound(value, guard, at));
        }
        if (result && work <= kMaxWork) {
            result = exclude_endpoints(value, at, result);
            active.erase(key);
            if (work > kMaxWork)
                return std::nullopt;
            cache[key] = result;
            return result;
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
                if (instruction->opcode == "shl.b64") {
                    const auto shift = literal(instruction->operands[2]);
                    const auto input = operand(instruction, 1, at);
                    if (shift && *shift >= 0 && *shift < 64 && input && input->lower >= 0) {
                        const __int128 scale = static_cast<__int128>(1) << *shift;
                        const __int128 lower = static_cast<__int128>(input->lower) * scale;
                        const __int128 upper = static_cast<__int128>(input->upper) * scale;
                        if (upper <= INT64_MAX)
                            result = ScalarRange{static_cast<std::int64_t>(lower),
                                                 static_cast<std::int64_t>(upper)};
                    }
                } else if (instruction->opcode == "shr.u64") {
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
            if ((!left || !relative_offset(value, *left)) && (!right || !relative_offset(value, *right)))
                continue;
            if (dominates(guard, at))
                result = intersect(result, guard_bound(value, guard, at));
        }
        result = exclude_endpoints(value, at, result);
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
                           const std::unordered_map<ValueId, Type>& types, ScalarRangeLimits limits)
    : impl_(std::make_unique<Impl>(blocks, incoming, outgoing, arguments, results, types, limits)) {}
ScalarRanges::~ScalarRanges() = default;
bool ScalarRanges::budget_exhausted() const { return impl_->work > impl_->kMaxWork; }
std::optional<ScalarRange> ScalarRanges::get(ValueId value, const Instruction* at) {
    auto location = impl_->locations.find(at);
    if (location == impl_->locations.end())
        return std::nullopt;
    return impl_->get(value, location->second);
}

std::optional<ScalarRange> ScalarRanges::captured(ValueId value, const Instruction* creation,
                                                const Instruction* use) {
    return impl_->captured(value, creation, use);
}
} // namespace cumetal::ir::detail
