#include "ptx_address_demands.h"

#include <utility>

namespace cumetal::ir::detail {

AddressDemandResult
compute_address_demands(const std::vector<RawBlock>& blocks,
                        const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
                        const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
                        const std::vector<std::map<std::string, ValueId>>& arguments,
                        const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
                        const std::function<std::vector<std::string>(const Instruction&)>& ssa_destinations,
                        AddressDemandLimits limits) {
    AddressDemandResult result;
    const auto charge = [&](std::size_t count, const char* phase) {
        if (count > limits.work - result.work) {
            result.budget_exhausted = true;
            result.reason = "PTX memory-address demand proof budget exhausted (" + std::string(phase) + ": " +
                            std::to_string(result.work) + " used, " + std::to_string(count) +
                            " requested, limit " + std::to_string(limits.work) + ")";
            return false;
        }
        result.work += count;
        return true;
    };
    const auto invalid = [&](const std::string& reason) {
        result.reason = "invalid PTX memory-address demand SSA: " + reason;
    };
    if (incoming.size() != blocks.size() || outgoing.size() != blocks.size() ||
        arguments.size() != blocks.size() || !ssa_destinations) {
        invalid("inconsistent block tables or missing destination callback");
        return result;
    }

    struct Join {
        std::size_t block;
        const std::string* name;
    };
    std::unordered_map<ValueId, Join> joins;
    std::unordered_map<ValueId, ValueId> copies;
    std::unordered_set<ValueId> demanded;
    std::vector<ValueId> pending;
    std::vector<AddressDemandLoad> loads;
    const auto demand = [&](ValueId value) {
        if (!charge(1, "demand membership lookup"))
            return false;
        if (demanded.contains(value))
            return true;
        if (!charge(2, "demand insertion and queue"))
            return false;
        demanded.insert(value);
        pending.push_back(value);
        return true;
    };

    for (std::size_t b = 0; b < blocks.size(); ++b) {
        if (!charge(1, "block index"))
            return result;
        for (const auto& [name, value] : arguments[b]) {
            if (!charge(1, "join index"))
                return result;
            if (!joins.emplace(value, Join{b, &name}).second) {
                invalid("duplicate join result");
                return result;
            }
            ++result.indexed_joins;
        }

        // Incoming maps can contain many live scalars unrelated to addresses.
        // Resolve reads through a sparse set of local writes instead of copying
        // those maps, preserving the pre-write value of self-updating operands.
        std::unordered_map<std::string, ValueId> written;
        const auto lookup = [&](const std::string& name, std::optional<ValueId>& value) {
            value.reset();
            if (name.empty())
                return true;
            if (!charge(1, "local environment lookup"))
                return false;
            if (const auto found = written.find(name); found != written.end()) {
                value = found->second;
                return true;
            }
            if (!charge(1, "incoming environment lookup"))
                return false;
            if (const auto found = incoming[b].find(name); found != incoming[b].end())
                value = found->second;
            return true;
        };
        for (const auto* instruction : blocks[b].instructions) {
            if (!charge(1, "instruction scan"))
                return result;
            if (instruction == nullptr) {
                invalid("null instruction");
                return result;
            }
            if (!charge(1, "instruction result lookup"))
                return result;
            const auto found_results = results.find(instruction);
            if (found_results == results.end()) {
                invalid("missing instruction results");
                return result;
            }
            const auto& values = found_results->second;
            const auto root = root_opcode(instruction->opcode);
            const bool parameter_load = instruction->opcode.starts_with("ld.param");
            std::optional<std::size_t> address_index;
            if ((root == "ld" || root == "atom") && !parameter_load)
                address_index = 1;
            else if (root == "st" || root == "red")
                address_index = 0;
            if (address_index && *address_index < instruction->operands.size()) {
                std::optional<ValueId> address;
                if (!lookup(first_register(instruction->operands[*address_index]), address))
                    return result;
                if (address && !demand(*address))
                    return result;
                if (root == "ld" && !values.empty()) {
                    if (!charge(1, "load candidate insertion"))
                        return result;
                    loads.push_back({instruction, address});
                }
            }
            if (values.size() == 1 && instruction->predicate.empty() && instruction->operands.size() == 2 &&
                (instruction->opcode == "mov.b64" || instruction->opcode == "mov.u64" ||
                 instruction->opcode == "mov.s64" || root == "cvta") &&
                instruction->operands[1].find('{') == std::string::npos) {
                std::optional<ValueId> source;
                if (!lookup(first_register(instruction->operands[1]), source))
                    return result;
                if (source) {
                    if (!charge(1, "copy index"))
                        return result;
                    if (!copies.emplace(values.front(), *source).second) {
                        invalid("duplicate copy result");
                        return result;
                    }
                    ++result.copy_edges;
                }
            }
            // Charge before invoking the same destination decoder used by SSA.
            // This also bounds callback invocations and retained overrides.
            if (!charge(1, "destination decoding") || !charge(values.size(), "environment update"))
                return result;
            const auto destinations = ssa_destinations(*instruction);
            if (destinations.size() != values.size()) {
                invalid("instruction destination/result count mismatch");
                return result;
            }
            for (std::size_t i = 0; i < destinations.size(); ++i)
                written[destinations[i]] = values[i];
        }
    }

    while (!pending.empty()) {
        if (!charge(1, "demand node visit"))
            return result;
        const auto value = pending.back();
        pending.pop_back();
        if (!charge(1, "copy lookup"))
            return result;
        if (const auto copy = copies.find(value); copy != copies.end()) {
            if (!charge(1, "copy edge visit") || !demand(copy->second))
                return result;
        }
        if (!charge(1, "join lookup"))
            return result;
        const auto join = joins.find(value);
        if (join == joins.end())
            continue;
        ++result.expanded_joins;
        const auto& predecessors = blocks[join->second.block].predecessors;
        if (predecessors.empty()) {
            invalid("demanded join has no incoming edge");
            return result;
        }
        for (const auto predecessor : predecessors) {
            if (!charge(1, "join edge visit"))
                return result;
            ++result.join_edges;
            if (predecessor >= outgoing.size()) {
                invalid("join predecessor is out of range");
                return result;
            }
            if (!charge(1, "predecessor value lookup"))
                return result;
            const auto source = outgoing[predecessor].find(*join->second.name);
            if (source == outgoing[predecessor].end()) {
                invalid("missing incoming definition for '" + *join->second.name + "'");
                return result;
            }
            if (!demand(source->second))
                return result;
        }
    }

    // Only complete demand and candidate sets may affect later memory proofs.
    result.values = std::move(demanded);
    result.loads = std::move(loads);
    result.complete = true;
    return result;
}

} // namespace cumetal::ir::detail
