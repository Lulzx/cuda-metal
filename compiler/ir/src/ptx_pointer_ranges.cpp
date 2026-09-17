#include "ptx_pointer_ranges.h"
#include "ptx_text.h"

#include <algorithm>
#include <set>

namespace cumetal::ir::detail {
namespace {
bool copy(const Instruction& i) {
    return i.predicate.empty() && i.operands.size() == 2 &&
           (i.opcode == "mov.b64" || i.opcode == "mov.u64" || i.opcode == "mov.s64" ||
            i.opcode == "mov.pred") &&
           i.operands[1].find('{') == std::string::npos;
}
std::optional<std::int64_t> literal(const std::string& input) {
    try {
        const auto text = trim(input);
        std::size_t consumed = 0;
        const auto result = std::stoll(text, &consumed, 0);
        if (consumed == text.size())
            return result;
    } catch (...) {
    }
    return std::nullopt;
}
} // namespace

struct PointerRanges::Impl {
    struct Definition {
        const Instruction* instruction;
        std::size_t block;
    };
    struct Input {
        std::size_t predecessor;
        ValueId value;
    };
    struct Join {
        std::size_t block;
        std::vector<Input> inputs;
    };
    const std::vector<RawBlock>& blocks;
    AddressQuery address;
    std::unordered_map<std::string, std::uint64_t> allocation_sizes;
    ScalarQuery scalar;
    PointerRangeLimits limits;
    std::size_t work = 0;
    bool exhausted = false;
    std::unordered_map<ValueId, Definition> definitions;
    std::unordered_map<ValueId, Join> joins;
    std::unordered_map<const Instruction*, std::unordered_map<std::string, ValueId>> sources;
    std::unordered_map<const Instruction*, std::size_t> locations;
    std::unordered_map<const Instruction*, std::size_t> result_counts;
    std::vector<const Instruction*> comparisons;

    bool charge(std::size_t count = 1) {
        if (exhausted || count > limits.work - work) {
            exhausted = true;
            return false;
        }
        work += count;
        return true;
    }
    Impl(const std::vector<RawBlock>& b,
         const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
         const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
         const std::vector<std::map<std::string, ValueId>>& arguments,
         const std::unordered_map<const Instruction*, std::vector<ValueId>>& results, AddressQuery a,
         ScalarQuery s, PointerRangeLimits l)
        : blocks(b), scalar(std::move(s)), limits(l) {
        address = [this, query = std::move(a)](ValueId value, const Instruction* at) {
            const auto result = query(value, at);
            if (result) allocation_sizes[result->depot] = result->allocation_size;
            return result;
        };
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            if (!charge(1 + incoming[block].size()))
                return;
            for (const auto& [name, value] : arguments[block]) {
                if (!charge(1 + blocks[block].predecessors.size()))
                    return;
                auto& join = joins[value];
                join.block = block;
                for (auto predecessor : blocks[block].predecessors) {
                    const auto found = outgoing[predecessor].find(name);
                    if (found == outgoing[predecessor].end()) {
                        exhausted = true;
                        return;
                    }
                    join.inputs.push_back({predecessor, found->second});
                }
            }
            auto environment = incoming[block];
            for (const auto* instruction : blocks[block].instructions) {
                if (!charge())
                    return;
                locations[instruction] = block;
                for (const auto& name : source_registers(*instruction)) {
                    if (!charge())
                        return;
                    if (environment.contains(name))
                        sources[instruction][name] = environment.at(name);
                }
                const auto found = results.find(instruction);
                if (found == results.end())
                    continue;
                const auto destinations = destination_registers(*instruction);
                if (!charge(found->second.size()))
                    return;
                result_counts[instruction] = found->second.size();
                if (found->second.size() == 1 && instruction->predicate.empty() &&
                    instruction->operands.size() == 3 &&
                    (instruction->opcode == "setp.eq.s64" || instruction->opcode == "setp.eq.u64" ||
                     instruction->opcode == "setp.ne.s64" || instruction->opcode == "setp.ne.u64")) {
                    if (!charge())
                        return;
                    comparisons.push_back(instruction);
                }
                for (std::size_t i = 0; i < destinations.size() && i < found->second.size(); ++i) {
                    definitions[found->second[i]] = {instruction, block};
                    environment[destinations[i]] = found->second[i];
                }
            }
        }
    }
    std::optional<ValueId> source(const Instruction* i, const std::string& operand) const {
        const auto found = sources.find(i);
        if (found == sources.end())
            return std::nullopt;
        const auto value = found->second.find(first_register(operand));
        return value == found->second.end() ? std::nullopt : std::optional(value->second);
    }
    std::vector<ValueId> transparent_inputs(ValueId value) {
        if (joins.contains(value)) {
            std::vector<ValueId> inputs;
            for (const auto& input : joins.at(value).inputs)
                inputs.push_back(input.value);
            return inputs;
        }
        if (definitions.contains(value)) {
            const auto* i = definitions.at(value).instruction;
            if (copy(*i) && result_counts[i] == 1)
                if (auto input = source(i, i->operands[1]))
                    return {*input};
        }
        return {};
    }
    // All inputs must forward the same target, and cycles must be anchored at
    // that target. A disconnected relay cycle cannot invent an initial value.
    bool forwards(ValueId value, ValueId target) {
        std::vector<ValueId> pending{value};
        std::unordered_set<ValueId> visited;
        std::unordered_map<ValueId, std::vector<ValueId>> reverse;
        while (!pending.empty()) {
            if (!charge())
                return false;
            const auto current = pending.back();
            pending.pop_back();
            if (!visited.insert(current).second || current == target)
                continue;
            const auto inputs = transparent_inputs(current);
            if (inputs.empty() || !charge(inputs.size()))
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
            if (!charge())
                return false;
            auto current = pending.back();
            pending.pop_back();
            if (!reached.insert(current).second)
                continue;
            const auto found = reverse.find(current);
            if (found != reverse.end()) {
                if (!charge(found->second.size()))
                    return false;
                pending.insert(pending.end(), found->second.begin(), found->second.end());
            }
        }
        return reached.size() == visited.size();
    }
    std::optional<ValueId> terminal(ValueId value) {
        std::vector<ValueId> pending{value};
        std::unordered_set<ValueId> visited;
        std::optional<ValueId> result;
        while (!pending.empty()) {
            if (!charge())
                return std::nullopt;
            auto current = pending.back();
            pending.pop_back();
            if (!visited.insert(current).second)
                continue;
            const auto inputs = transparent_inputs(current);
            if (!inputs.empty()) {
                if (!charge(inputs.size()))
                    return std::nullopt;
                pending.insert(pending.end(), inputs.begin(), inputs.end());
            } else {
                if (result && *result != current)
                    return std::nullopt;
                result = current;
            }
        }
        return result && forwards(value, *result) ? result : std::nullopt;
    }
    // Reachability stops at a re-entry to cut. This gives guard facts a single
    // iteration lifetime, rather than reusing an earlier loop execution's fact.
    std::vector<bool> reachable(std::size_t start, std::optional<std::size_t> cut = {},
                                std::optional<std::pair<std::size_t, std::size_t>> omitted = {}) {
        std::vector<bool> seen(blocks.size(), false);
        std::vector<std::size_t> pending{start};
        while (!pending.empty()) {
            if (!charge())
                return {};
            const auto block = pending.back();
            pending.pop_back();
            if (seen[block])
                continue;
            seen[block] = true;
            for (auto successor : blocks[block].successors) {
                if (!charge())
                    return {};
                if ((cut && successor == *cut) || (omitted && *omitted == std::make_pair(block, successor)))
                    continue;
                pending.push_back(successor);
            }
        }
        return seen;
    }
    bool single(const Instruction* i) const {
        const auto found = result_counts.find(i);
        return i->predicate.empty() && found != result_counts.end() && found->second == 1;
    }
    bool equal_operand(const Instruction* i, std::size_t index, ValueId value) {
        auto input = source(i, i->operands[index]);
        return input && forwards(*input, value);
    }
    bool comparison(const Instruction* i, ValueId next, ValueId end, bool want_equal) {
        if (!single(i) || i->operands.size() != 3)
            return false;
        const auto expected = want_equal ? "setp.eq." : "setp.ne.";
        if (i->opcode != std::string(expected) + "s64" && i->opcode != std::string(expected) + "u64")
            return false;
        return (equal_operand(i, 1, next) && equal_operand(i, 2, end)) ||
               (equal_operand(i, 2, next) && equal_operand(i, 1, end));
    }
    // If a selected integer is observed nonzero, a literal-zero arm could not
    // have been selected. This implies the selector's truth value without
    // assigning any pointer meaning to the other arm. In particular, a loop
    // may encode its end test as select(end == next, 0, next) != 0.
    std::optional<std::pair<ValueId, bool>> nonzero_selection(const Instruction* comparison, bool truth) {
        if (!single(comparison) || comparison->operands.size() != 3 ||
            (comparison->opcode != "setp.ne.s64" && comparison->opcode != "setp.ne.u64" &&
             comparison->opcode != "setp.eq.s64" && comparison->opcode != "setp.eq.u64"))
            return std::nullopt;
        const bool equal = comparison->opcode.find(".eq.") != std::string::npos;
        if (truth == equal)
            return std::nullopt;
        for (std::size_t index : {1U, 2U}) {
            if (literal(comparison->operands[3 - index]) != 0)
                continue;
            const auto input = source(comparison, comparison->operands[index]);
            const auto selected = input ? terminal(*input) : std::nullopt;
            if (!selected || !definitions.contains(*selected))
                continue;
            const auto* select = definitions.at(*selected).instruction;
            if (!single(select) || select->operands.size() != 4 ||
                (select->opcode != "selp.b64" && select->opcode != "selp.u64" &&
                 select->opcode != "selp.s64"))
                continue;
            const bool first_zero = literal(select->operands[1]) == 0;
            const bool second_zero = literal(select->operands[2]) == 0;
            if (first_zero == second_zero)
                continue;
            const auto [name, inverted] = normalized_predicate(select->operands[3]);
            if (const auto selector = source(select, name))
                return std::pair{*selector, second_zero != inverted};
        }
        return std::nullopt;
    }
    bool implies_unequal(ValueId predicate, bool truth, ValueId next, ValueId end,
                         std::unordered_set<ValueId>& active) {
        if (!charge() || active.size() > 128 || !active.insert(predicate).second)
            return false;
        const auto resolved = terminal(predicate);
        bool result = false;
        if (resolved && definitions.contains(*resolved)) {
            const auto* i = definitions.at(*resolved).instruction;
            if (comparison(i, next, end, !truth))
                result = true;
            else if (const auto selected = nonzero_selection(i, truth))
                result = implies_unequal(selected->first, selected->second, next, end, active);
            else if (single(i) && i->operands.size() == 3 &&
                     ((i->opcode == "and.pred" && truth) || (i->opcode == "or.pred" && !truth))) {
                for (std::size_t index : {1U, 2U})
                    if (auto input = source(i, i->operands[index]))
                        result |= implies_unequal(*input, truth, next, end, active);
            } else if (single(i) && i->opcode == "not.pred" && i->operands.size() == 2) {
                if (auto input = source(i, i->operands[1]))
                    result = implies_unequal(*input, !truth, next, end, active);
            }
        }
        active.erase(predicate);
        return result;
    }
    bool guarded(std::size_t header, std::size_t predecessor, ValueId next, ValueId end,
                 const std::vector<bool>& iteration) {
        for (std::size_t block = 0; block < blocks.size(); ++block) {
            if (!charge())
                return false;
            if (!iteration[block] || blocks[block].instructions.empty() ||
                blocks[block].successors.size() != 2 ||
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
                std::unordered_set<ValueId> active;
                if (!implies_unequal(*predicate, (edge == 0) != inverted, next, end, active))
                    continue;
                const auto successor = blocks[block].successors[edge];
                if (block == predecessor && successor == header)
                    return true;
                const auto bypass = reachable(header, header, {{block, successor}});
                if (!bypass.empty() && !bypass[predecessor])
                    return true;
            }
        }
        return false;
    }
    std::optional<bool> predicate_truth(ValueId predicate, const Instruction* at) {
        const auto resolved = terminal(predicate);
        if (!resolved || !definitions.contains(*resolved))
            return std::nullopt;
        const auto* i = definitions.at(*resolved).instruction;
        if (!single(i) || i->operands.size() != 3 ||
            (i->opcode != "setp.eq.s64" && i->opcode != "setp.eq.u64" && i->opcode != "setp.ne.s64" &&
             i->opcode != "setp.ne.u64"))
            return std::nullopt;
        for (std::size_t index : {1U, 2U}) {
            const auto number = literal(i->operands[3 - index]);
            const auto value = source(i, i->operands[index]);
            if (!number || !value)
                continue;
            const auto range = scalar(*value, at);
            if (!range)
                continue;
            const bool eq = i->opcode.find(".eq.") != std::string::npos;
            if (*number < range->lower || *number > range->upper)
                return !eq;
            if (range->lower == *number && range->upper == *number)
                return eq;
        }
        return std::nullopt;
    }
    std::optional<ValueId> selected_seed(ValueId value, const Instruction* at) {
        auto resolved = terminal(value);
        if (!resolved || !definitions.contains(*resolved))
            return std::nullopt;
        const auto* i = definitions.at(*resolved).instruction;
        if (single(i) && i->opcode == "selp.b64" && i->operands.size() == 4) {
            const auto predicate = source(i, i->operands[3]);
            const auto truth = predicate ? predicate_truth(*predicate, at) : std::nullopt;
            if (!truth)
                return std::nullopt;
            const auto selected = source(i, i->operands[*truth ? 1 : 2]);
            return selected ? terminal(*selected) : std::nullopt;
        }
        return resolved;
    }
    bool unit_increment(ValueId value, ValueId next) {
        const auto resolved = terminal(value);
        if (!resolved || !definitions.contains(*resolved))
            return false;
        const auto* i = definitions.at(*resolved).instruction;
        if (!single(i) || (i->opcode != "add.s64" && i->opcode != "add.u64") || i->operands.size() != 3)
            return false;
        return (literal(i->operands[1]) == 1 && equal_operand(i, 2, next)) ||
               (literal(i->operands[2]) == 1 && equal_operand(i, 1, next));
    }
    bool update(ValueId value, ValueId next, ValueId end) {
        if (unit_increment(value, next))
            return true;
        const auto resolved = terminal(value);
        if (!resolved || !definitions.contains(*resolved))
            return false;
        const auto* i = definitions.at(*resolved).instruction;
        if (!single(i) || i->opcode != "selp.b64" || i->operands.size() != 4)
            return false;
        const auto predicate = source(i, i->operands[3]);
        const auto first = source(i, i->operands[1]), second = source(i, i->operands[2]);
        const auto condition = predicate ? terminal(*predicate) : std::nullopt;
        if (!condition || !definitions.contains(*condition) || !first || !second)
            return false;
        const auto* compare = definitions.at(*condition).instruction;
        return (comparison(compare, next, end, true) && forwards(*first, next) &&
                unit_increment(*second, next)) ||
               (comparison(compare, next, end, false) && unit_increment(*first, next) &&
                forwards(*second, next));
    }
    std::optional<LocalPointerRange> prove(ValueId previous, ValueId next, ValueId end,
                                           const Instruction* at) {
        const auto header = joins.at(previous).block;
        if (joins.at(next).block != header || header == 0)
            return std::nullopt;
        const auto iteration = reachable(header, header);
        const auto bypass = reachable(0, header);
        if (iteration.empty() || bypass.empty() || bypass[locations.at(at)] || !iteration[locations.at(at)])
            return std::nullopt;
        for (const auto predecessor : blocks[header].predecessors)
            if (iteration[predecessor] && bypass[predecessor])
                return std::nullopt;
        const auto end_terminal = terminal(end);
        if (!end_terminal || !definitions.contains(*end_terminal))
            return std::nullopt;
        const auto end_def = definitions.at(*end_terminal);
        // An end address recomputed in this iteration is not an invariant.
        if (iteration[end_def.block])
            return std::nullopt;
        const auto end_bypass = reachable(0, end_def.block);
        if (end_def.block != 0 && (end_bypass.empty() || end_bypass[header]))
            return std::nullopt;
        const auto* end_i = end_def.instruction;
        if (!single(end_i) || end_i->operands.size() != 3 ||
            (end_i->opcode != "add.s64" && end_i->opcode != "add.u64"))
            return std::nullopt;
        std::optional<ExactLocalAddress> base;
        for (const auto& input : joins.at(previous).inputs) {
            if (iteration[input.predecessor]) {
                if (!forwards(input.value, next))
                    return std::nullopt;
            } else {
                const auto seed = selected_seed(input.value, at);
                const auto candidate = seed ? address(*seed, at) : std::nullopt;
                if (!candidate || candidate->offset < 0 ||
                    static_cast<std::uint64_t>(candidate->offset) >= candidate->allocation_size)
                    return std::nullopt;
                if (base && (base->depot != candidate->depot || base->offset != candidate->offset ||
                             base->allocation_size != candidate->allocation_size))
                    return std::nullopt;
                base = candidate;
            }
        }
        if (!base)
            return std::nullopt;
        std::optional<ScalarRange> length;
        for (std::size_t index : {1U, 2U}) {
            const auto pointer = source(end_i, end_i->operands[index]);
            const auto count = source(end_i, end_i->operands[3 - index]);
            const auto constant_count = literal(end_i->operands[3 - index]);
            if (!pointer || (!count && !constant_count))
                continue;
            const auto candidate = address(*pointer, at);
            if (!candidate || candidate->depot != base->depot || candidate->offset != base->offset ||
                candidate->allocation_size != base->allocation_size)
                continue;
            // E captures this SSA value once. A completed outer-loop phi can
            // have several origins while remaining fixed throughout this loop.
            // Reject only values whose actual defining/join block may execute
            // again in the header-cut iteration; their current bound could
            // describe a newer value than the one captured by E.
            std::optional<ScalarRange> bound;
            if (constant_count) {
                bound = ScalarRange{*constant_count, *constant_count};
            } else {
                const auto count_block = definitions.contains(*count) ? definitions.at(*count).block
                                         : joins.contains(*count)     ? joins.at(*count).block
                                                                      : SIZE_MAX;
                if (count_block >= iteration.size() || iteration[count_block])
                    continue;
                bound = scalar(*count, at);
            }
            if (!bound || bound->lower < 1 || bound->upper < bound->lower ||
                static_cast<__int128>(base->offset) + bound->upper > base->allocation_size ||
                static_cast<__int128>(base->offset) + bound->upper > INT64_MAX)
                continue;
            length = bound;
        }
        if (!length)
            return std::nullopt;
        bool has_initial = false, has_update = false;
        for (const auto& input : joins.at(next).inputs) {
            if (iteration[input.predecessor]) {
                if (!update(input.value, next, *end_terminal) ||
                    !guarded(header, input.predecessor, next, *end_terminal, iteration))
                    return std::nullopt;
                has_update = true;
            } else {
                const auto seed = selected_seed(input.value, at);
                const auto candidate = seed ? address(*seed, at) : std::nullopt;
                if (!candidate || candidate->depot != base->depot ||
                    candidate->allocation_size != base->allocation_size ||
                    static_cast<__int128>(candidate->offset) != static_cast<__int128>(base->offset) + 1)
                    return std::nullopt;
                has_initial = true;
            }
        }
        if (!has_initial || !has_update || exhausted)
            return std::nullopt;
        // N starts B+1 and advances by exactly one only while N != E.
        // E is B+positive_length in this allocation, so neither N nor P can
        // skip E or wrap. P receives B initially and N on a guarded backedge.
        return LocalPointerRange{base->depot, base->offset, base->offset + length->upper};
    }
    std::vector<ValueId> ancestor_joins(ValueId value) {
        std::vector<ValueId> pending{value}, found;
        std::unordered_set<ValueId> visited;
        while (!pending.empty()) {
            if (!charge())
                return {};
            const auto current = pending.back();
            pending.pop_back();
            if (!visited.insert(current).second)
                continue;
            if (joins.contains(current))
                found.push_back(current);
            const auto inputs = transparent_inputs(current);
            if (!charge(inputs.size()))
                return {};
            pending.insert(pending.end(), inputs.begin(), inputs.end());
        }
        return found;
    }
    // A single-block loop may advance a pointer by a fixed stride while an
    // independent scalar reaches a literal bound. Both updates must occur exactly
    // once per backedge; equality endpoints must be reached without skipping.
    std::optional<LocalPointerRange> counted_loop(ValueId pointer, const Instruction* at) {
        if (!joins.contains(pointer) || !locations.contains(at)) return std::nullopt;
        const auto& join = joins.at(pointer);
        const auto header = join.block;
        if (header == 0 || blocks[header].instructions.empty() || blocks[header].successors.size() != 2)
            return std::nullopt;
        const auto& successors = blocks[header].successors;
        if ((successors[0] == header) == (successors[1] == header)) return std::nullopt;
        const auto* branch = blocks[header].instructions.back();
        if (!is_conditional_branch(*branch)) return std::nullopt;
        const auto bypass = reachable(0, header);
        if (bypass.empty() || bypass[locations.at(at)]) return std::nullopt;
        // A path from the exit back to an external predecessor would add an
        // unmodelled outer recurrence. Keep this proof's entry seeds invariant.
        const auto after = reachable(successors[0] == header ? successors[1] : successors[0], header);
        if (after.empty()) return std::nullopt;
        for (const auto predecessor : blocks[header].predecessors)
            if (predecessor != header && after[predecessor]) return std::nullopt;
        const auto [predicate_name, inverted] = normalized_predicate(branch->predicate);
        const auto predicate = source(branch, predicate_name);
        const auto predicate_origin = predicate ? terminal(*predicate) : std::nullopt;
        if (!predicate_origin || !definitions.contains(*predicate_origin)) return std::nullopt;
        const auto compare_definition = definitions.at(*predicate_origin);
        const auto* compare = compare_definition.instruction;
        const bool truth = (successors[0] == header) != inverted;
        if (compare_definition.block != header || !single(compare) || compare->operands.size() != 3)
            return std::nullopt;
        std::string relation;
        for (const auto* op : {"eq", "ne", "lt", "le", "gt", "ge"})
            if (compare->opcode == std::string("setp.") + op + ".u64" ||
                ((std::string(op) == "eq" || std::string(op) == "ne") &&
                 compare->opcode == std::string("setp.") + op + ".s64")) relation = op;
        if (relation.empty()) return std::nullopt;
        std::optional<ValueId> next_count;
        std::optional<std::int64_t> endpoint;
        for (std::size_t i : {1U, 2U}) {
            const auto limit = literal(compare->operands[3-i]);
            const auto input = source(compare, compare->operands[i]);
            if (!limit || !input) continue;
            next_count = terminal(*input);
            endpoint = limit;
            if (i == 2) {
                if (relation == "lt") relation = "gt";
                else if (relation == "le") relation = "ge";
                else if (relation == "gt") relation = "lt";
                else if (relation == "ge") relation = "le";
            }
            break;
        }
        if (!truth) {
            if (relation == "eq") relation = "ne";
            else if (relation == "ne") relation = "eq";
            else if (relation == "lt") relation = "ge";
            else if (relation == "le") relation = "gt";
            else if (relation == "gt") relation = "le";
            else if (relation == "ge") relation = "lt";
        }
        if (!endpoint || *endpoint < 0) return std::nullopt;
        if (!next_count || !definitions.contains(*next_count)) return std::nullopt;
        const auto count_definition = definitions.at(*next_count);
        const auto* count_update = count_definition.instruction;
        if (count_definition.block != header || !single(count_update) || count_update->operands.size() != 3)
            return std::nullopt;
        const bool subtract = count_update->opcode == "sub.s64" || count_update->opcode == "sub.u64";
        if (!subtract && count_update->opcode != "add.s64" && count_update->opcode != "add.u64")
            return std::nullopt;
        const auto decrement = literal(count_update->operands[2]);
        const auto count_input = source(count_update, count_update->operands[1]);
        if (!decrement || !count_input || *decrement == INT64_MIN) return std::nullopt;
        const std::int64_t step = subtract ? -*decrement : *decrement;
        if (step == 0) return std::nullopt;
        std::optional<ValueId> counter;
        for (const auto candidate : ancestor_joins(*count_input))
            if (joins.at(candidate).block == header && forwards(*count_input, candidate)) counter = candidate;
        if (!counter) return std::nullopt;
        std::optional<std::int64_t> trips;
        bool count_backedge = false;
        for (const auto& input : joins.at(*counter).inputs) {
            if (!charge()) return std::nullopt;
            if (input.predecessor == header) {
                if (!forwards(input.value, *next_count)) return std::nullopt;
                count_backedge = true;
            } else {
                const auto* site = blocks[input.predecessor].instructions.empty() ? nullptr
                    : blocks[input.predecessor].instructions.back();
                if (!site) return std::nullopt;
                const auto seed = scalar(input.value, site);
                if (!seed || seed->lower != seed->upper || seed->lower < 0) return std::nullopt;
                __int128 count = 0;
                const __int128 start = seed->lower, end = *endpoint, delta = step;
                if (relation == "ne") {
                    const auto distance = end - start;
                    if (distance == 0 || (distance > 0) != (delta > 0) || distance % delta != 0)
                        return std::nullopt;
                    count = distance / delta;
                } else if ((relation == "lt" || relation == "le") && delta > 0) {
                    const auto stop = end + (relation == "le" ? 1 : 0);
                    if (start >= stop) return std::nullopt;
                    count = (stop - start + delta - 1) / delta;
                } else if ((relation == "gt" || relation == "ge") && delta < 0) {
                    const auto stop = end - (relation == "ge" ? 1 : 0);
                    if (start <= stop) return std::nullopt;
                    count = (start - stop - delta - 1) / -delta;
                } else return std::nullopt;
                const auto last_count = start + count * delta;
                if (count <= 0 || count > INT64_MAX || last_count < 0 || last_count > INT64_MAX)
                    return std::nullopt;
                if (trips && *trips != count) return std::nullopt;
                trips = static_cast<std::int64_t>(count);
            }
        }
        if (!count_backedge || !trips) return std::nullopt;
        std::optional<ExactLocalAddress> base;
        std::optional<std::int64_t> stride;
        for (const auto& input : join.inputs) {
            if (!charge()) return std::nullopt;
            if (input.predecessor == header) {
                const auto origin = terminal(input.value);
                if (!origin || !definitions.contains(*origin)) return std::nullopt;
                const auto definition = definitions.at(*origin);
                const auto* update = definition.instruction;
                if (definition.block != header || !single(update) || update->operands.size() != 3 ||
                    (update->opcode != "add.s64" && update->opcode != "add.u64" &&
                     update->opcode != "sub.s64" && update->opcode != "sub.u64")) return std::nullopt;
                const auto amount = literal(update->operands[2]);
                const auto prior = source(update, update->operands[1]);
                if (!amount || !prior || !forwards(*prior, pointer) || *amount == INT64_MIN)
                    return std::nullopt;
                const auto offset = root_opcode(update->opcode) == "sub" ? -*amount : *amount;
                if (stride && *stride != offset) return std::nullopt;
                stride = offset;
            } else {
                const auto candidate = address(input.value, at);
                if (!candidate || candidate->offset < 0 ||
                    (base && (base->depot != candidate->depot || base->offset != candidate->offset ||
                              base->allocation_size != candidate->allocation_size))) return std::nullopt;
                base = candidate;
            }
        }
        if (!base || !stride || exhausted) return std::nullopt;
        const __int128 last = static_cast<__int128>(base->offset) + (*trips - 1) * static_cast<__int128>(*stride);
        const auto lower = std::min<__int128>(base->offset, last);
        const auto upper = std::max<__int128>(base->offset, last);
        // Include the final update in the no-wrap check even though the phi
        // sees only pre-update values. Its result can be used after the loop.
        const __int128 end = last + *stride;
        if (lower < 0 || upper >= base->allocation_size || upper > INT64_MAX ||
            end < 0 || end > base->allocation_size || end > INT64_MAX) return std::nullopt;
        return LocalPointerRange{base->depot, static_cast<std::int64_t>(lower), static_cast<std::int64_t>(upper)};
    }

    std::optional<LocalPointerRange> get(ValueId value, const Instruction* at) {
        if (exhausted || !locations.contains(at))
            return std::nullopt;
        // Start from the queried value and recurrence operands. Unrelated
        // registers and equality instructions cannot multiply candidate phis.
        for (const auto previous : ancestor_joins(value)) {
            if (!charge() || !forwards(value, previous))
                continue;
            if (const auto counted = counted_loop(previous, at)) return counted;
            const auto& join = joins.at(previous);
            const auto iteration = reachable(join.block, join.block);
            if (iteration.empty())
                return std::nullopt;
            std::set<ValueId> next_candidates;
            for (const auto& input : join.inputs) {
                if (!iteration[input.predecessor])
                    continue;
                for (auto candidate : ancestor_joins(input.value)) {
                    if (!charge())
                        return std::nullopt;
                    next_candidates.insert(candidate);
                }
            }
            for (const auto next : next_candidates) {
                if (!charge())
                    return std::nullopt;
                const auto& next_join = joins.at(next);
                if (next == previous || next_join.block != join.block)
                    continue;
                // End candidates must actually occur in an equality guard.
                for (const auto* i : comparisons) {
                    if (!charge())
                        return std::nullopt;
                    for (std::size_t index : {1U, 2U}) {
                        if (!equal_operand(i, index, next))
                            continue;
                        const auto end = source(i, i->operands[3 - index]);
                        if (end)
                            if (auto proof = prove(previous, next, *end, at))
                                return proof;
                    }
                }
            }
        }
        return std::nullopt;
    }
    std::map<std::pair<ValueId, const Instruction*>, std::optional<LocalPointerRange>> pointer_range_cache;

    // Shared by pointer-cell and scalar-zero proofs: ranges discharge only
    // non-overlap, never initialization or the type of a stored value.
    bool disjoint(ValueId value, const Instruction* event, std::int64_t displacement,
                  std::uint64_t bytes, const std::string& depot, __int128 cell,
                  ScalarRanges& scalar_ranges) {
        std::unordered_set<ValueId> active;
        std::unordered_map<ValueId, std::optional<LocalPointerRange>> cache;
        std::size_t work = 0;
        bool refine_at_use = false;
        const auto range = [&](auto&& self, ValueId value) -> std::optional<LocalPointerRange> {
            if (++work > 16384 || active.size() > 256) return std::nullopt;
            if (cache.contains(value)) return cache.at(value);
            if (!active.insert(value).second) return std::nullopt;
            const auto evaluate = [&]() -> std::optional<LocalPointerRange> {
                if (const auto exact = address(value, event))
                    return LocalPointerRange{exact->depot, exact->offset, exact->offset};
                if (joins.contains(value)) {
                    const auto forwarded = terminal(value);
                    if (forwarded && *forwarded != value) return self(self, *forwarded);
                    // A loop may carry several independently bounded
                    // pointer origins through copy-only joins. Collect
                    // all concrete origins without treating an arithmetic
                    // recurrence as another copy of its initial value.
                    std::vector<ValueId> pending{value}, origins;
                    std::unordered_set<ValueId> visited;
                    std::unordered_map<ValueId, std::vector<ValueId>> reverse;
                    while (!pending.empty()) {
                        if (++work > 16384) return std::nullopt;
                        const auto current = pending.back();
                        pending.pop_back();
                        if (!visited.insert(current).second) continue;
                        std::vector<ValueId> inputs;
                        if (joins.contains(current)) {
                            for (const auto& edge : joins.at(current).inputs) inputs.push_back(edge.value);
                            if (inputs.empty()) return std::nullopt;
                        } else if (definitions.contains(current)) {
                            const auto* copy = definitions.at(current).instruction;
                            if (copy->predicate.empty() && copy->operands.size() == 2 &&
                                (copy->opcode == "mov.b64" || copy->opcode == "mov.u64" ||
                                 copy->opcode == "mov.s64") &&
                                copy->operands[1].find('{') == std::string::npos) {
                                if (const auto input = source(copy, copy->operands[1]))
                                    inputs.push_back(*input);
                            }
                        }
                        if (inputs.empty()) origins.push_back(current);
                        for (const auto input : inputs) {
                            if (++work > 16384) return std::nullopt;
                            reverse[input].push_back(current);
                            pending.push_back(input);
                        }
                    }
                    // Every relay must be anchored. A separate cycle
                    // with no concrete incoming value remains unknown.
                    pending = origins;
                    std::unordered_set<ValueId> anchored;
                    while (!pending.empty()) {
                        if (++work > 16384) return std::nullopt;
                        const auto current = pending.back();
                        pending.pop_back();
                        if (!anchored.insert(current).second) continue;
                        if (reverse.contains(current))
                            for (const auto user : reverse.at(current)) pending.push_back(user);
                    }
                    if (anchored.size() != visited.size()) return std::nullopt;
                    std::optional<LocalPointerRange> merged;
                    for (auto input : origins) {
                        const auto next = self(self, input);
                        if (!next || (merged && next->depot != merged->depot)) return std::nullopt;
                        if (!merged) merged = next;
                        else {
                            merged->lower = std::min(merged->lower, next->lower);
                            merged->upper = std::max(merged->upper, next->upper);
                        }
                    }
                    return merged;
                }
                if (!definitions.contains(value)) return std::nullopt;
                const auto* definition = definitions.at(value).instruction;
                if (!definition->predicate.empty() ||
                    destination_registers(*definition).size() != 1) return std::nullopt;
                const auto input = [&](std::size_t index) -> std::optional<LocalPointerRange> {
                    const auto input_value = source(definition, definition->operands[index]);
                    return input_value ? self(self, *input_value) : std::nullopt;
                };
                if ((definition->opcode == "mov.b64" || definition->opcode == "mov.u64" ||
                     definition->opcode == "mov.s64" || definition->opcode == "cvta.local.u64" ||
                     definition->opcode == "cvta.to.local.u64") && definition->operands.size() == 2 &&
                    definition->operands[1].find('{') == std::string::npos) return input(1);
                const bool subtract = definition->opcode == "sub.s64" || definition->opcode == "sub.u64";
                if (definition->operands.size() != 3 ||
                    (!subtract && definition->opcode != "add.s64" && definition->opcode != "add.u64"))
                    return std::nullopt;
                for (std::size_t index : {1U, 2U}) {
                    if (subtract && index == 2) break;
                    const auto base = input(index);
                    if (!base) continue;
                    const auto number = literal(definition->operands[3-index]);
                    const auto input_value = source(definition, definition->operands[3-index]);
                    const auto offset = number ? std::optional(detail::ScalarRange{*number,*number})
                        : input_value ? (refine_at_use ? scalar_ranges.captured(*input_value,definition,event)
                                                : scalar_ranges.get(*input_value,definition)) : std::nullopt;
                    if (!offset) continue;
                    const __int128 lo = static_cast<__int128>(base->lower) +
                        (subtract ? -static_cast<__int128>(offset->upper) : offset->lower);
                    const __int128 hi = static_cast<__int128>(base->upper) +
                        (subtract ? -static_cast<__int128>(offset->lower) : offset->upper);
                    if (lo < INT64_MIN || hi > INT64_MAX) return std::nullopt;
                    return LocalPointerRange{base->depot,static_cast<std::int64_t>(lo),static_cast<std::int64_t>(hi)};
                }
                return std::nullopt;
            };
            auto result = evaluate();
            if (!result) {
                const auto key = std::make_pair(value, event);
                constexpr std::size_t kMaxPointerRangeQueries = 65536;
                if (!pointer_range_cache.contains(key) && pointer_range_cache.size() < kMaxPointerRangeQueries)
                    pointer_range_cache.emplace(key, get(value, event));
                if (const auto found = pointer_range_cache.find(key); found != pointer_range_cache.end())
                    if (const auto& iterator = found->second)
                        result = LocalPointerRange{iterator->depot, iterator->lower, iterator->upper};
            }
            active.erase(value);
            cache[value] = result;
            return result;
        };
        const auto separated = [&](const std::optional<LocalPointerRange>& written) {
            return written && work <= 16384 && allocation_sizes.contains(written->depot) &&
                static_cast<__int128>(written->lower) + displacement >= 0 &&
                static_cast<__int128>(written->upper) + displacement + bytes <= allocation_sizes.at(written->depot) &&
                (written->depot != depot ||
                static_cast<__int128>(written->lower) + displacement >= cell + 8 ||
                static_cast<__int128>(written->upper) + displacement + bytes <= cell);
        };
        if (!bytes) return false;
        if (separated(range(range, value))) return true;
        if (work > 16384) return false;
        refine_at_use = true;
        cache.clear();
        return separated(range(range, value));
    }

};

PointerRanges::PointerRanges(const std::vector<RawBlock>& blocks,
                             const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
                             const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
                             const std::vector<std::map<std::string, ValueId>>& arguments,
                             const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
                             AddressQuery address, ScalarQuery scalar, PointerRangeLimits limits)
    : impl_(std::make_unique<Impl>(blocks, incoming, outgoing, arguments, results, std::move(address),
                                   std::move(scalar), limits)) {}
PointerRanges::~PointerRanges() = default;
std::optional<LocalPointerRange> PointerRanges::get(ValueId value, const Instruction* at) {
    return impl_->get(value, at);
}
bool PointerRanges::disjoint(ValueId value, const Instruction* at, std::int64_t displacement,
                             std::uint64_t bytes, const std::string& depot, __int128 cell,
                             ScalarRanges& scalar_ranges) {
    return impl_->disjoint(value, at, displacement, bytes, depot, cell, scalar_ranges);
}
} // namespace cumetal::ir::detail
