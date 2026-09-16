#include "ptx_scalar_ranges.h"

#include <deque>
#include <iostream>
#include <limits>
#include <string>

namespace {
namespace ir = cumetal::ir;
namespace detail = cumetal::ir::detail;

// These graphs supply actual reaching SSA identities, independently of the
// importer. In particular, repeated PTX names do not mean repeated values.
struct Graph {
    std::deque<detail::Instruction> instructions;
    std::vector<detail::RawBlock> blocks;
    std::vector<std::unordered_map<std::string, ir::ValueId>> incoming, outgoing;
    std::vector<std::map<std::string, ir::ValueId>> arguments;
    std::unordered_map<const detail::Instruction*, std::vector<ir::ValueId>> results;
    std::unordered_map<ir::ValueId, ir::Type> types;

    explicit Graph(std::size_t count) : blocks(count), incoming(count), outgoing(count), arguments(count) {
        for (std::size_t i = 0; i < count; ++i) {
            blocks[i].id = static_cast<ir::BlockId>(i + 1);
            blocks[i].name = "B" + std::to_string(i);
        }
    }
    const detail::Instruction* add(std::size_t block, const std::string& opcode,
                                   std::vector<std::string> operands = {},
                                   std::vector<ir::ValueId> values = {}, const std::string& predicate = {}) {
        instructions.push_back({predicate, opcode, std::move(operands), 1, true});
        const auto* instruction = &instructions.back();
        blocks[block].instructions.push_back(instruction);
        if (!values.empty()) {
            results.emplace(instruction, values);
            for (const auto value : values)
                types[value] = opcode.starts_with("setp.") ? ir::Type::predicate() : ir::Type::integer(64);
        }
        return instruction;
    }
    void edge(std::size_t from, std::size_t to) {
        blocks[from].successors.push_back(to);
        blocks[to].predecessors.push_back(from);
    }
    void phi(std::size_t block, const std::string& name, ir::ValueId value) {
        arguments[block][name] = value;
        incoming[block][name] = value;
        types[value] = ir::Type::integer(64);
    }
    detail::ScalarRanges ranges() {
        return detail::ScalarRanges(blocks, incoming, outgoing, arguments, results, types);
    }
};

bool expect(bool condition, const std::string& message) {
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}
bool known(const std::optional<detail::ScalarRange>& range, std::int64_t lower, std::int64_t upper,
           const std::string& message) {
    return expect(range && range->lower == lower && range->upper == upper, message);
}

bool branch_bounds() {
    bool ok = true;
    for (const bool complement : {false, true}) {
        Graph graph(3);
        graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
        graph.add(0, "setp.lt.u64", {complement ? "%p|%opposite" : "%p", "%index", "16"},
                  complement ? std::vector<ir::ValueId>{2, 3} : std::vector<ir::ValueId>{2});
        graph.add(0, "bra", {"B1"}, {}, complement ? "%opposite" : "%p");
        graph.edge(0, 1);
        graph.edge(0, 2);
        const auto* taken = graph.add(1, "ret");
        const auto* other = graph.add(2, "ret");
        auto ranges = graph.ranges();
        if (complement) {
            // The second destination is !(%index < 16). Treating it as the
            // first result would incorrectly exclude an overlapping store.
            ok &= expect(!ranges.get(1, taken), "complemented predicate is not an upper bound");
        } else {
            ok &= known(ranges.get(1, taken), 0, 15, "taken unsigned < edge bounds its input");
            ok &= expect(!ranges.get(1, other), "unsigned lower bound does not exclude high-bit values");
        }
    }
    return ok;
}

bool shifted_reverse_indices() {
    bool ok = true;
    for (bool guarded : {false, true}) {
        Graph graph(3);
        graph.add(0, "ld.param.u64", {"%length", "[length]"}, {1});
        graph.add(0, "setp.le.u64", {"%fits", "%length", "64"}, {2});
        graph.add(0, "bra", {"B1"}, {}, guarded ? "%fits" : "");
        graph.edge(0, 1);
        if (guarded)
            graph.edge(0, 2);
        graph.incoming[1]["%length"] = 1;
        graph.add(1, "shr.u64", {"%half", "%length", "1"}, {3});
        graph.add(1, "not.b64", {"%reverse", "%half"}, {4});
        graph.add(1, "add.s64", {"%offset", "%length", "%reverse"}, {5});
        const auto* query = graph.add(1, "ret");
        graph.add(2, "ret");
        auto ranges = graph.ranges();
        if (guarded) {
            ok &= known(ranges.get(3, query), 0, 32,
                        "unsigned right shift preserves a proved nonnegative range");
            ok &= known(ranges.get(4, query), -33, -1, "bitwise not reverses signed interval endpoints");
            ok &= known(ranges.get(5, query), -33, 63, "bounded reverse address arithmetic remains finite");
        } else {
            ok &= expect(!ranges.get(3, query) && !ranges.get(4, query) && !ranges.get(5, query),
                         "bypassing the bound cannot invent a nonnegative shifted interval");
        }
    }
    return ok;
}

bool guarded_forwarded_copy() {
    Graph graph(5);
    graph.add(0, "ld.param.u64", {"%length", "[length]"}, {1});
    graph.add(0, "setp.le.u64", {"%fits", "%length", "64"}, {2});
    graph.add(0, "bra", {"B1"}, {}, "%fits");
    graph.edge(0, 1);
    graph.edge(0, 4);
    graph.incoming[1]["%length"] = 1;
    graph.add(1, "bra", {"B2"});
    graph.edge(1, 2);
    graph.outgoing[1]["%length"] = 1;
    graph.phi(2, "%length", 3);
    graph.add(2, "mov.u64", {"%copy", "%length"}, {4});
    const auto* query = graph.add(2, "bra", {"B3"});
    graph.edge(2, 3);
    graph.incoming[3]["%length"] = 3;
    graph.add(3, "bra", {"B2"}, {}, "%unknown");
    graph.edge(3, 2);
    graph.edge(3, 4);
    graph.outgoing[3]["%length"] = 3;
    graph.add(4, "ret");
    auto ranges = graph.ranges();
    return known(ranges.get(4, query), 0, 64,
                 "a guarded value retains its bound through anchored phi and copied result");
}

bool nonzero_refinement() {
    bool ok = true;
    for (unsigned variant = 0; variant < 3; ++variant) {
        Graph graph(4);
        graph.add(0, "ld.param.u64", {"%length", "[length]"}, {1});
        graph.add(0, "setp.le.u64", {"%fits", "%length", "64"}, {2});
        graph.add(0, "bra", {"B1"}, {}, variant == 1 ? "" : "%fits");
        graph.edge(0, 1);
        if (variant != 1)
            graph.edge(0, 3);
        graph.incoming[1]["%length"] = 1;
        graph.add(1, "setp.eq.u64", {"%empty", "%length", "0"}, {3});
        graph.add(1, "bra", {"B3"}, {}, "%empty");
        graph.edge(1, 3);
        graph.edge(1, 2);
        graph.incoming[2]["%length"] = 1;
        if (variant == 2)
            graph.add(2, "mov.u64", {"%length", "0"}, {4});
        const auto* query = graph.add(2, "ret");
        graph.add(3, "ret");
        auto ranges = graph.ranges();
        if (variant == 0)
            ok &= known(ranges.get(1, query), 1, 64, "nonzero guard tightens a separately bounded length");
        if (variant == 1)
            ok &= expect(!ranges.get(1, query), "nonzero alone cannot exclude high-bit unsigned values");
        if (variant == 2)
            ok &= known(ranges.get(4, query), 0, 0,
                        "a later zero definition does not inherit the old nonzero guard");
    }
    return ok;
}

bool guarded_redefinition() {
    Graph graph(3);
    graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
    graph.add(0, "setp.lt.u64", {"%p", "%index", "16"}, {2});
    graph.add(0, "bra", {"B1"}, {}, "%p");
    graph.edge(0, 1);
    graph.edge(0, 2);
    graph.incoming[1]["%index"] = 1;
    graph.add(1, "add.u64", {"%index", "%index", "64"}, {3});
    const auto* query = graph.add(1, "ret");
    graph.add(2, "ret");
    auto ranges = graph.ranges();
    return known(ranges.get(3, query), 64, 79,
                 "a reused register receives its new value's range after the guard");
}

bool stale_preheader_guard() {
    Graph graph(4);
    graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
    graph.add(0, "setp.lt.u64", {"%p", "%index", "16"}, {2});
    graph.add(0, "bra", {"B1"}, {}, "%p");
    graph.edge(0, 1);
    graph.edge(0, 3);
    graph.outgoing[0]["%index"] = 1;
    graph.phi(1, "%index", 3);
    const auto* query = graph.add(1, "bra", {"B2"});
    graph.edge(1, 2);
    graph.incoming[2]["%index"] = 3;
    graph.add(2, "add.u64", {"%index", "%index", "1"}, {4});
    graph.add(2, "bra", {"B1"});
    graph.edge(2, 1);
    graph.outgoing[2]["%index"] = 4;
    graph.add(3, "ret");
    auto ranges = graph.ranges();
    return expect(!ranges.get(3, query), "preheader fact cannot bound a subsequently changing loop phi");
}

bool induction_bounds(unsigned step, bool bypass, bool overflow) {
    // B1 computes next. B2 tests it. B3 is the backedge, additionally reachable
    // directly from B1 in the bypass case. The query precedes every increment.
    Graph graph(5);
    graph.add(0, "mov.u64", {"%index", overflow ? "9223372036854775806" : "0"}, {1});
    graph.add(0, "ld.param.u64", {"%choice", "[choice]"}, {2});
    graph.add(0, "setp.ne.u64", {"%choose", "%choice", "0"}, {3});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    graph.outgoing[0]["%index"] = 1;
    graph.phi(1, "%index", 4);
    graph.incoming[1]["%choose"] = 3;
    const auto* query = graph.add(1, "add.u64", {"%next", "%index", std::to_string(step)}, {5});
    graph.add(1, "bra", {"B2"}, {}, bypass ? "%choose" : "");
    graph.edge(1, 2);
    if (bypass)
        graph.edge(1, 3);
    graph.incoming[2]["%next"] = 5;
    graph.add(2, "setp.lt.u64", {"%again", "%next", overflow ? "9223372036854775807" : "16"}, {6});
    graph.add(2, "bra", {"B3"}, {}, "%again");
    graph.edge(2, 3);
    graph.edge(2, 4);
    graph.add(3, "bra", {"B1"});
    graph.edge(3, 1);
    graph.outgoing[3]["%index"] = 5;
    graph.add(4, "ret");
    auto ranges = graph.ranges();
    const auto range = ranges.get(4, query);
    if (!bypass)
        return known(range, 0, overflow ? INT64_MAX - 1 : 15,
                     "actual incoming values satisfy the guard regardless of recurrence step");
    return expect(!range, "an unguarded backedge prevents induction proof");
}

bool stale_loop_predicate() {
    // The loop branches on a carried predicate; computing a fresh comparison
    // without using it cannot justify the branch. This models a captured fact
    // whose compared value has changed on subsequent iterations.
    Graph graph(3);
    graph.add(0, "mov.u64", {"%index", "0"}, {1});
    graph.add(0, "setp.eq.u64", {"%again", "%index", "0"}, {2});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    graph.outgoing[0]["%index"] = 1;
    graph.outgoing[0]["%again"] = 2;
    graph.phi(1, "%index", 3);
    graph.phi(1, "%again", 4);
    graph.types[4] = ir::Type::predicate();
    const auto* query = graph.add(1, "add.u64", {"%index", "%index", "1"}, {5});
    graph.add(1, "setp.lt.u64", {"%fresh", "%index", "16"}, {6});
    graph.add(1, "bra", {"B1"}, {}, "%again");
    graph.edge(1, 1);
    graph.edge(1, 2);
    graph.outgoing[1]["%index"] = 5;
    graph.outgoing[1]["%again"] = 4;
    graph.add(2, "ret");
    auto ranges = graph.ranges();
    return expect(!ranges.get(3, query),
                  "a fresh but unused comparison does not refresh a carried predicate");
}

bool guarded_dynamic_seed() {
    bool ok = true;
    for (int variant = 0; variant < 3; ++variant) {
        Graph graph(4);
        graph.add(0, "ld.param.u64", {"%initial", "[initial]"}, {1});
        graph.add(0, variant == 1 ? "setp.ge.u64" : "setp.lt.u64", {"%entry", "%initial", "64"}, {5});
        graph.add(0, "bra", {"B1"}, {}, "%entry");
        graph.edge(0, 1);
        graph.edge(0, 3);
        graph.outgoing[0]["%index"] = 1;
        graph.phi(1, "%index", 2);
        const auto* store = graph.add(1, "st.local.u8", {"[%index]", "0"});
        graph.add(1, "add.u64", {"%index", "%index", "1"}, {3});
        graph.add(1, "bra", {"B2"});
        graph.edge(1, 2);
        graph.incoming[2]["%index"] = 3;
        graph.add(2, variant == 2 ? "setp.ge.u64" : "setp.lt.u64", {"%again", "%index", "64"}, {4});
        graph.add(2, "bra", {"B1"}, {}, "%again");
        graph.edge(2, 1);
        graph.edge(2, 3);
        graph.outgoing[2]["%index"] = 3;
        graph.add(3, "ret");
        auto ranges = graph.ranges();
        const auto result = ranges.get(2, store);
        ok &= variant == 0 ? known(result, 0, 63, "each incoming edge bounds dynamic loop seed")
                           : expect(!result, "unbounded initial or recurrence edge stays unknown");
    }
    return ok;
}

} // namespace

int main() {
    bool ok = branch_bounds();
    ok &= guarded_dynamic_seed();
    ok &= shifted_reverse_indices();
    ok &= guarded_forwarded_copy();
    ok &= nonzero_refinement();
    ok &= guarded_redefinition();
    ok &= stale_preheader_guard();
    ok &= induction_bounds(1, false, false);
    ok &= induction_bounds(1, true, false);
    ok &= induction_bounds(2, false, false);
    ok &= induction_bounds(1, false, true);
    ok &= induction_bounds(1, true, true);
    ok &= stale_loop_predicate();
    return ok ? 0 : 1;
}
