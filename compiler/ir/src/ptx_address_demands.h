#pragma once

#include "ptx_cfg.h"

#include <array>
#include <functional>
#include <optional>
#include <string>

namespace cumetal::ir::detail {

struct AddressDemandLimits {
    std::size_t work = 1'000'000;
};

struct AddressDemandLoad {
    const Instruction* instruction = nullptr;
    // The address operand's reaching SSA value before the load writes its
    // destination. A direct allocation symbol has no register value here.
    std::optional<ValueId> address;
};

struct AddressDemandResult {
    bool complete = false;
    bool budget_exhausted = false;
    std::string reason;
    std::size_t work = 0;
    std::size_t expanded_joins = 0;
    std::size_t join_edges = 0;
    std::size_t copy_edges = 0;
    std::unordered_set<ValueId> values;
    std::vector<AddressDemandLoad> loads;
};

// Collect during the caller's existing walk of validated SSA, before updating
// the environment with each instruction's results. No second environment walk
// or destination decoding is needed. Every observed instruction and retained
// edge consumes work. The collector belongs to one graph revision and must be
// discarded whenever SSA is rebuilt.
//
// Demand is not proof of pointer contents or address space. The caller must
// validate any local load selected by this result against its reaching stores,
// and must also validate independently pointer-typed loads in `loads`.
// finish() reuses the caller's join index and expands every predecessor/backedge
// of each demanded join. On exhaustion or inconsistent SSA it publishes no partial
// values or load candidates. All inputs remain unchanged.
class AddressDemandCollector {
public:
    using JoinLookup = std::function<const std::vector<ValueId>*(ValueId)>;
    explicit AddressDemandCollector(AddressDemandLimits limits = {}) : limits_(limits) {}

    [[nodiscard]] bool observe(const Instruction& instruction, const std::vector<ValueId>& results,
                               const std::unordered_map<std::string, ValueId>& environment);
    [[nodiscard]] const std::string& reason() const { return result_.reason; }
    // Reuse an index the SSA solver has already built and validated. Return
    // null for non-joins; a join must expose every incoming edge, including
    // duplicates and backedges. Referenced vectors outlive this call.
    [[nodiscard]] AddressDemandResult finish(const JoinLookup& join_inputs,
                                             std::size_t reused_joins, std::size_t blocks);

private:
    enum class Phase : std::size_t { kObservation, kClosure };
    bool charge(std::size_t count, const char* phase);
    bool demand(ValueId value);
    bool invalid(const std::string& reason);
    void discard_partial();

    AddressDemandLimits limits_;
    AddressDemandResult result_;
    std::unordered_map<ValueId, ValueId> copies_;
    std::vector<ValueId> pending_;
    Phase phase_ = Phase::kObservation;
    std::array<std::size_t, 2> phase_work_{};
    std::size_t observations_ = 0;
    std::size_t memory_operands_ = 0;
    std::size_t source_lookups_ = 0;
    std::size_t demand_calls_ = 0;
    std::size_t repeated_demands_ = 0;
    std::size_t nodes_started_ = 0;
    std::size_t copy_lookups_ = 0;
    std::size_t copy_hits_ = 0;
    std::size_t join_lookups_ = 0;
    std::size_t blocks_ = 0;
    std::size_t reused_joins_ = 0;
};

} // namespace cumetal::ir::detail
