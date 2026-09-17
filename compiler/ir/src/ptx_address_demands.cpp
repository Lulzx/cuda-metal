#include "ptx_address_demands.h"

#include <sstream>
#include <utility>

namespace cumetal::ir::detail {

void AddressDemandCollector::discard_partial() {
    result_.values.clear();
    result_.loads.clear();
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
                 << " copy_edges=" << result_.copy_edges
                 << " blocks=" << blocks_
                 << " reused_joins=" << reused_joins_
                 << " definition_lookups=" << definition_lookups_
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
    if (!charge(1, "demand insertion lookup")) return false;
    if (!result_.values.insert(value).second) {
        ++repeated_demands_;
        return true;
    }
    if (!charge(1, "demand queue")) return false;
    pending_.push_back(value);
    return true;
}

bool AddressDemandCollector::observe_memory(
    const Instruction& instruction, const std::vector<ValueId>& values,
    const std::unordered_map<std::string, ValueId>& environment,
    std::size_t address_index, bool is_load) {
    // The SSA solver already classified this memory instruction. Its result
    // lookup, destination decoding and environment writes are also shared.
    if (!charge(1, "memory observation")) return false;
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
    if (address_index < instruction.operands.size()) {
        ++memory_operands_;
        std::optional<ValueId> address;
        if (!lookup(first_register(instruction.operands[address_index]), address)) return false;
        if (address && !demand(*address)) return false;
        if (is_load && !values.empty()) {
            if (!charge(1, "load candidate insertion")) return false;
            result_.loads.push_back({&instruction, address});
        }
    }
    return true;
}
AddressDemandResult AddressDemandCollector::finish(
    const SourceLookup& sources, std::size_t reused_joins, std::size_t blocks) {
    if (!result_.reason.empty()) return std::move(result_);
    if (!sources) {
        invalid("missing source lookup");
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
        if (!charge(1, "definition lookup")) return std::move(result_);
        ++definition_lookups_;
        const auto definition = sources(value);
        if (definition.kind == AddressDemandKind::kOther) continue;
        const auto* inputs = definition.inputs;
        if (inputs == nullptr) {
            invalid("demanded definition has no source vector");
            return std::move(result_);
        }
        if (definition.kind == AddressDemandKind::kCopy) {
            if (inputs->size() != 1) {
                invalid("demanded copy must have one source");
                return std::move(result_);
            }
            if (!charge(1, "copy edge visit")) return std::move(result_);
            ++result_.copy_edges;
            if (!demand(inputs->front())) return std::move(result_);
            continue;
        }
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
