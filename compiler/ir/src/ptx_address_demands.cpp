#include "ptx_address_demands.h"

#include <sstream>
#include <utility>

namespace cumetal::ir::detail {

void AddressDemandCollector::discard_partial() {
    result_.values.clear();
    result_.loads.clear();
    copies_.clear();
    pending_.clear();
}

bool AddressDemandCollector::charge(std::size_t count, const char* phase) {
    if (!result_.reason.empty()) return false;
    if (count > limits_.work - result_.work) {
        result_.budget_exhausted = true;
        result_.reason = "PTX memory-address demand proof budget exhausted (" + std::string(phase) + ": " +
                         std::to_string(result_.work) + " used, " + std::to_string(count) +
                         " requested, limit " + std::to_string(limits_.work) + ")";
        // Snapshot counters before discarding partial proof state. These
        // diagnostics do not change charging or permit an incomplete proof.
        constexpr const char* phases[] = {"observation", "closure"};
        std::ostringstream counters;
        counters << " [phase=" << phases[static_cast<std::size_t>(phase_)]
                 << " observation_work=" << phase_work_[0]
                 << " closure_work=" << phase_work_[1]
                 << " observations=" << observations_
                 << " memory_operands=" << memory_operands_
                 << " source_lookups=" << source_lookups_
                 << " loads=" << result_.loads.size()
                 << " demand_calls=" << demand_calls_
                 << " unique_demands=" << result_.values.size()
                 << " repeated_demands=" << repeated_demands_
                 << " pending=" << pending_.size()
                 << " nodes_started=" << nodes_started_
                 << " copies=" << result_.copy_edges
                 << " copy_lookups=" << copy_lookups_
                 << " copy_hits=" << copy_hits_
                 << " blocks=" << blocks_
                 << " reused_joins=" << reused_joins_
                 << " join_lookups=" << join_lookups_
                 << " expanded_joins=" << result_.expanded_joins
                 << " join_edges=" << result_.join_edges << ']';
        result_.reason += counters.str();
        discard_partial();
        return false;
    }
    result_.work += count;
    phase_work_[static_cast<std::size_t>(phase_)] += count;
    return true;
}

bool AddressDemandCollector::invalid(const std::string& reason) {
    result_.reason = "invalid PTX memory-address demand SSA: " + reason;
    discard_partial();
    return false;
}

bool AddressDemandCollector::demand(ValueId value) {
    ++demand_calls_;
    if (!charge(1, "demand membership lookup")) return false;
    if (result_.values.contains(value)) {
        ++repeated_demands_;
        return true;
    }
    if (!charge(2, "demand insertion and queue")) return false;
    result_.values.insert(value);
    pending_.push_back(value);
    return true;
}

bool AddressDemandCollector::observe(
    const Instruction& instruction, const std::vector<ValueId>& values,
    const std::unordered_map<std::string, ValueId>& environment) {
    // Classification shares the SSA solver's scan. Its result lookup,
    // destination decoding and environment writes have already been required
    // by that solver; this analysis performs none of them again.
    if (!charge(1, "instruction observation")) return false;
    ++observations_;
    const auto lookup = [&](const std::string& name, std::optional<ValueId>& value) {
        value.reset();
        if (name.empty()) return true;
        if (!charge(1, "source environment lookup")) return false;
        ++source_lookups_;
        if (const auto found = environment.find(name); found != environment.end())
            value = found->second;
        return true;
    };
    const auto root = root_opcode(instruction.opcode);
    const bool parameter_load = instruction.opcode.starts_with("ld.param");
    std::optional<std::size_t> address_index;
    if ((root == "ld" || root == "atom") && !parameter_load)
        address_index = 1;
    else if (root == "st" || root == "red")
        address_index = 0;
    if (address_index && *address_index < instruction.operands.size()) {
        ++memory_operands_;
        std::optional<ValueId> address;
        if (!lookup(first_register(instruction.operands[*address_index]), address)) return false;
        if (address && !demand(*address)) return false;
        if (root == "ld" && !values.empty()) {
            if (!charge(1, "load candidate insertion")) return false;
            result_.loads.push_back({&instruction, address});
        }
    }
    if (values.size() == 1 && instruction.predicate.empty() && instruction.operands.size() == 2 &&
        (instruction.opcode == "mov.b64" || instruction.opcode == "mov.u64" ||
         instruction.opcode == "mov.s64" || root == "cvta") &&
        instruction.operands[1].find('{') == std::string::npos) {
        std::optional<ValueId> source;
        if (!lookup(first_register(instruction.operands[1]), source)) return false;
        if (source) {
            if (!charge(1, "copy index")) return false;
            if (!copies_.emplace(values.front(), *source).second)
                return invalid("duplicate copy result");
            ++result_.copy_edges;
        }
    }
    return true;
}
AddressDemandResult AddressDemandCollector::finish(
    const JoinLookup& join_inputs, std::size_t reused_joins, std::size_t blocks) {
    if (!result_.reason.empty()) return std::move(result_);
    if (!join_inputs) {
        invalid("missing join lookup");
        return std::move(result_);
    }
    reused_joins_ = reused_joins;
    blocks_ = blocks;
    phase_ = Phase::kClosure;
    while (!pending_.empty()) {
        if (!charge(1, "demand node visit")) return std::move(result_);
        ++nodes_started_;
        const auto value = pending_.back();
        pending_.pop_back();
        if (!charge(1, "copy lookup")) return std::move(result_);
        ++copy_lookups_;
        if (const auto copy = copies_.find(value); copy != copies_.end()) {
            ++copy_hits_;
            if (!charge(1, "copy edge visit") || !demand(copy->second)) return std::move(result_);
        }
        if (!charge(1, "join lookup")) return std::move(result_);
        ++join_lookups_;
        const auto* inputs = join_inputs(value);
        if (!result_.reason.empty()) return std::move(result_);
        if (inputs == nullptr) continue;
        ++result_.expanded_joins;
        if (inputs->empty()) {
            invalid("demanded join has no incoming edge");
            return std::move(result_);
        }
        for (const auto source : *inputs) {
            if (!charge(1, "join edge visit")) return std::move(result_);
            ++result_.join_edges;
            if (!demand(source)) return std::move(result_);
        }
    }
    result_.complete = true;
    return std::move(result_);
}

} // namespace cumetal::ir::detail
