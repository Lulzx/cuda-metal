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
    std::size_t pointer_cutoffs = 0;
    std::unordered_set<ValueId> values;
    std::vector<AddressDemandLoad> loads;
};

enum class AddressDemandKind { kOther, kJoin, kCopy };
struct AddressDemandSources {
    AddressDemandKind kind = AddressDemandKind::kOther;
    const std::vector<ValueId>* inputs = nullptr;
};

// Collect during the caller's existing walk of validated SSA, before updating
// the environment with each instruction's results. No second environment walk
// or destination decoding is needed. Observe memory instructions and cvta source
// addresses selected by the caller's existing classification. Every observation and retained
// edge consumes work. The collector belongs to one graph revision and must be
// discarded whenever SSA is rebuilt.
//
// Demand is not proof of pointer contents or address space. The caller must
// validate any local load selected by this result against its reaching stores,
// and must also validate independently pointer-typed loads in `loads`.
// finish() reuses the caller's source index. Validated concrete pointers end
// demand propagation; every other demanded join expands all predecessor/backedges.
// cvta sources must be seeded independently because cvta can infer a pointer result
// before its scalar source has a memory-content proof. On exhaustion or inconsistent SSA
// it publishes no partial values or load candidates. All inputs remain unchanged.
class AddressDemandCollector {
public:
    using SourceLookup = std::function<AddressDemandSources(ValueId)>;
    using PointerPredicate = std::function<bool(ValueId)>;
    explicit AddressDemandCollector(AddressDemandLimits limits = {}) : limits_(limits) {}

    [[nodiscard]] bool observe_memory(const Instruction& instruction, const std::vector<ValueId>& results,
                                      const std::unordered_map<std::string, ValueId>& environment,
                                      std::size_t address_index, bool is_load);
    [[nodiscard]] const std::string& reason() const { return result_.reason; }
    // Reuse an index the SSA solver has already built and validated. Return
    // kOther for unrelated definitions; a join must expose every incoming edge,
    // including duplicates and backedges. A copy has exactly one source and is
    // an unpredicated, scalar mov.b64/u64/s64 or cvta with two operands. Referenced
    // vectors outlive this call. The predicate may return true only after final
    // SSA type validation, and only for concrete pointer types. It does not
    // discharge the caller's independent validation of pointer-typed loads.
    [[nodiscard]] AddressDemandResult finish(const SourceLookup& sources,
                                             const PointerPredicate& is_concrete_pointer,
                                             std::size_t reused_joins, std::size_t blocks);

private:
    enum class Phase : std::size_t { kObservation, kClosure };
    bool charge(std::size_t count, const char* phase);
    bool demand(ValueId value);
    bool invalid(const std::string& reason);
    void discard_partial();

    AddressDemandLimits limits_;
    AddressDemandResult result_;
    std::vector<ValueId> pending_;
    Phase phase_ = Phase::kObservation;
    std::array<std::size_t, 2> phase_work_{};
    std::size_t observations_ = 0;
    std::size_t memory_operands_ = 0;
    std::size_t source_lookups_ = 0;
    std::size_t demand_calls_ = 0;
    std::size_t repeated_demands_ = 0;
    std::size_t nodes_started_ = 0;
    std::size_t definition_lookups_ = 0;
    std::size_t type_lookups_ = 0;
    std::size_t blocks_ = 0;
    std::size_t reused_joins_ = 0;
};

} // namespace cumetal::ir::detail
