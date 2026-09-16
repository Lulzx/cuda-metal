#include "ptx_pointer_ranges.h"

#include <deque>
#include <iostream>

namespace {
namespace ir = cumetal::ir;
namespace detail = cumetal::ir::detail;

struct Graph {
    std::deque<detail::Instruction> storage;
    std::vector<detail::RawBlock> blocks;
    std::vector<std::unordered_map<std::string, ir::ValueId>> incoming, outgoing;
    std::vector<std::map<std::string, ir::ValueId>> arguments;
    std::unordered_map<const detail::Instruction*, std::vector<ir::ValueId>> results;
    explicit Graph(std::size_t count) : blocks(count), incoming(count), outgoing(count), arguments(count) {
        for (std::size_t i = 0; i < count; ++i) {
            blocks[i].id = static_cast<ir::BlockId>(i + 1);
            blocks[i].name = "B" + std::to_string(i);
        }
    }
    const detail::Instruction* add(std::size_t block, std::string opcode,
                                   std::vector<std::string> operands = {},
                                   std::vector<ir::ValueId> values = {}, std::string predicate = {}) {
        storage.push_back({std::move(predicate), std::move(opcode), std::move(operands), 1, true});
        auto* i = &storage.back();
        blocks[block].instructions.push_back(i);
        if (!values.empty())
            results[i] = std::move(values);
        return i;
    }
    void edge(std::size_t a, std::size_t b) {
        blocks[a].successors.push_back(b);
        blocks[b].predecessors.push_back(a);
    }
    void phi(std::size_t block, const std::string& name, ir::ValueId value) {
        incoming[block][name] = value;
        arguments[block][name] = value;
    }
};

enum class Case {
    plain,
    selected_seed,
    copied,
    bypass,
    beyond_end,
    step_two,
    overwritten_end,
    conflicting_base,
    missing_positive_length,
    past_allocation,
    opposite_guard,
    predicate_overwrite,
    dual_comparison,
    undefined_entry,
    exhausted,
    outer_phi,
    repeated_outer_phi
};

bool test(Case kind, std::int64_t upper = 64) {
    const bool joined_length = kind == Case::outer_phi || kind == Case::repeated_outer_phi;
    Graph graph(joined_length ? 9 : 6);
    graph.add(0, "mov.u64", {"%base", "depot"}, {1});
    graph.add(0, "ld.param.u64", {"%length", "[length]"}, {2});
    const std::size_t preheader = joined_length ? 8 : 0;
    if (joined_length) {
        // A completed preceding loop has two genuine count origins. The
        // pointer loop captures the resulting phi, not one arbitrary origin.
        graph.add(0, "bra", {"B6"});
        graph.edge(0, 6);
        graph.outgoing[0]["%length"] = 2;
        graph.phi(6, "%length", 20);
        graph.add(6, "bra", {"B7"}, {}, "%another");
        graph.edge(6, 7);
        graph.edge(6, 8);
        graph.incoming[7]["%length"] = 20;
        graph.add(7, "add.u64", {"%length", "%length", "1"}, {21});
        graph.add(7, "bra", {"B6"});
        graph.edge(7, 6);
        graph.outgoing[7]["%length"] = 21;
        graph.incoming[8] = {{"%base", 1}, {"%length", 20}};
    }
    graph.add(preheader, "setp.eq.s64", {"%empty", "%length", "0"}, {3});
    graph.add(preheader, "add.u64", {"%first", "%base", "1"}, {4});
    if (kind == Case::selected_seed)
        graph.add(preheader, "selp.b64", {"%seed", "%base", "%first", "%empty"}, {5});
    else
        graph.add(preheader, "mov.u64", {"%seed", "%first"}, {5});
    graph.add(preheader, "add.u64", {"%end", "%base", "%length"}, {6});
    graph.add(preheader, "bra", {"B1"});
    graph.edge(preheader, 1);
    graph.outgoing[preheader]["%previous"] = 1;
    graph.outgoing[preheader]["%next"] = kind == Case::undefined_entry ? 999 : 5;

    graph.phi(1, "%previous", 7);
    graph.phi(1, "%next", 8);
    graph.incoming[1]["%end"] = 6;
    const detail::Instruction* query;
    ir::ValueId queried = 7;
    if (kind == Case::copied) {
        graph.add(1, "mov.b64", {"%alias", "%previous"}, {9});
        queried = 9;
        query = graph.add(1, "st.u8", {"[%alias]", "0"});
    } else {
        query = graph.add(1, "st.u8", {"[%previous]", "0"});
    }
    graph.add(1, "bra", {"B3"});
    graph.edge(1, 3);
    graph.incoming[3] = {{"%next", 8}, {"%previous", 7}, {"%end", 6}};
    if (kind == Case::overwritten_end)
        graph.add(3, "add.u64", {"%end", "%end", "1"}, {17});
    graph.add(
        3, "setp.ne.s64", {kind == Case::dual_comparison ? "%more|%opposite" : "%more", "%end", "%next"},
        kind == Case::dual_comparison ? std::vector<ir::ValueId>{10, 18} : std::vector<ir::ValueId>{10});
    graph.add(3, "setp.eq.s64", {"%last", "%end", "%next"}, {11});
    graph.add(3, "add.s64", {"%incremented", "%next", kind == Case::step_two ? "2" : "1"}, {12});
    graph.add(3, "selp.b64", {"%updated", "%next", "%incremented", "%last"}, {13});
    graph.add(3, "setp.ne.s64", {"%nonnull", "%next", "0"}, {14});
    graph.add(3, "and.pred", {"%continue", "%nonnull", "%more"}, {15});
    if (kind == Case::predicate_overwrite)
        graph.add(3, "mov.pred", {"%continue", "%unknown"}, {19});
    graph.add(3, "mov.u64", {"%advanced", "%next"}, {16});
    graph.add(3, "bra", {"B2"}, {}, kind == Case::bypass ? "%choice" : "");
    graph.edge(3, 2);
    if (kind == Case::bypass)
        graph.edge(3, 4);
    graph.incoming[2] = {{"%continue", kind == Case::predicate_overwrite ? 19U : 15U}};
    graph.add(2, "bra", {"B1"}, {}, kind == Case::opposite_guard ? "!%continue" : "%continue");
    graph.edge(2, 1);
    graph.edge(2, 5);
    graph.outgoing[2] = {{"%previous", 16}, {"%next", 13}};
    graph.add(4, "bra", {"B1"});
    if (kind == Case::bypass) {
        graph.edge(4, 1);
        graph.outgoing[4] = {{"%previous", 16}, {"%next", 13}};
    }
    if (kind == Case::repeated_outer_phi) {
        // The same defining blocks can now execute again after leaving the
        // pointer loop. The narrow proof must reject this wider lifetime.
        graph.add(5, "bra", {"B6"});
        graph.edge(5, 6);
        graph.outgoing[5]["%length"] = 20;
    } else {
        graph.add(5, "ret");
    }

    // Independent supplied facts: only literal local anchors are addresses;
    // the iterator, updated values, and end are never supplied as ranges.
    auto address = [&](ir::ValueId value,
                       const detail::Instruction*) -> std::optional<detail::ExactLocalAddress> {
        if (value == 1)
            return detail::ExactLocalAddress{"depot", 64, 256};
        if (value == 4)
            return detail::ExactLocalAddress{kind == Case::conflicting_base ? "other" : "depot",
                                             kind == Case::beyond_end ? 130 : 65, 256};
        return std::nullopt;
    };
    auto scalar = [&](ir::ValueId value, const detail::Instruction*) -> std::optional<detail::ScalarRange> {
        if (value != (joined_length ? 20U : 2U))
            return std::nullopt;
        return detail::ScalarRange{kind == Case::missing_positive_length ? 0 : 1,
                                   kind == Case::past_allocation ? 193 : upper};
    };
    detail::PointerRanges ranges(graph.blocks, graph.incoming, graph.outgoing, graph.arguments, graph.results,
                                 address, scalar, {kind == Case::exhausted ? 8U : 100000U});
    const auto result = ranges.get(queried, query);
    const bool positive =
        kind == Case::plain || kind == Case::selected_seed || kind == Case::copied || kind == Case::outer_phi;
    const bool ok =
        positive ? result && result->depot == "depot" && result->lower == 64 && result->upper == 64 + upper
                 : !result;
    if (!ok)
        std::cerr << "FAIL: pointer iterator case " << static_cast<int>(kind) << " upper=" << upper << '\n';
    return ok;
}
} // namespace

int main() {
    bool ok = true;
    for (auto kind : {Case::plain, Case::selected_seed, Case::copied, Case::outer_phi})
        for (auto upper : {1, 17, 64, 192})
            ok &= test(kind, upper);
    for (auto kind : {Case::bypass, Case::beyond_end, Case::step_two, Case::overwritten_end,
                      Case::conflicting_base, Case::missing_positive_length, Case::past_allocation,
                      Case::opposite_guard, Case::predicate_overwrite, Case::dual_comparison,
                      Case::undefined_entry, Case::exhausted, Case::repeated_outer_phi})
        ok &= test(kind);
    return ok ? 0 : 1;
}
