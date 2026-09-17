#include "ptx_scalar_ranges.h"
#include "ptx_pointer_ranges.h"

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
                types[value] = opcode.starts_with("setp.") || opcode.ends_with(".pred")
                                   ? ir::Type::predicate()
                                   : ir::Type::integer(64);
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
    detail::ScalarRanges ranges(detail::ScalarRangeLimits limits = {}) {
        return detail::ScalarRanges(blocks, incoming, outgoing, arguments, results, types, limits);
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

bool cached_bounds_survive_exhaustion() {
    // The two dynamic values have independent valid bounds at each query.
    // Proving one of them must not be undone when subsequent, unrelated
    // questions consume the finite analysis budget.
    constexpr std::size_t positions = 64, unknowns = 32;
    constexpr std::size_t exit = positions + 2;
    Graph graph(exit + 1);
    graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
    graph.add(0, "ld.param.u64", {"%other", "[other]"}, {2});
    for (std::size_t i = 0; i < unknowns; ++i)
        graph.add(0, "ld.param.u64", {"%unknown" + std::to_string(i), "[unknown]"},
                  {static_cast<ir::ValueId>(100 + i)});
    graph.add(0, "setp.le.u64", {"%fits", "%index", "255"}, {3});
    graph.add(0, "bra", {"B1"}, {}, "%fits");
    graph.edge(0, 1);
    graph.edge(0, exit);
    graph.incoming[1]["%other"] = 2;
    graph.add(1, "setp.le.u64", {"%other_fits", "%other", "15"}, {4});
    graph.add(1, "bra", {"B2"}, {}, "%other_fits");
    graph.edge(1, 2);
    graph.edge(1, exit);
    std::vector<const detail::Instruction*> queries;
    for (std::size_t i = 0; i < positions; ++i) {
        const auto block = i + 2;
        queries.push_back(graph.add(block, "nop"));
        graph.add(block, "bra", {"B" + std::to_string(block + 1)});
        graph.edge(block, block + 1);
    }
    graph.add(exit, "ret");

    // The initial proof is small; the distinct unknown queries deliberately
    // outnumber the work limit. Nothing about an unknown input's value is
    // inferred from reaching a budget boundary.
    auto ranges = graph.ranges(detail::ScalarRangeLimits{.work = 1024});
    bool ok = known(ranges.get(1, queries.front()), 0, 255, "initial dynamic bound is proved");
    for (const auto* query : queries)
        for (std::size_t i = 0; i < unknowns; ++i)
            ok &= expect(!ranges.get(static_cast<ir::ValueId>(100 + i), query),
                         "unrelated unknown inputs remain unknown");
    ok &= expect(!ranges.get(2, queries.front()), "an uncached question fails closed after exhaustion");
    ok &= expect(ranges.budget_exhausted(), "exhausted range analysis exposes its diagnostic status");
    ok &= known(ranges.get(1, queries.front()), 0, 255,
                "completed cached proof remains valid after unrelated budget exhaustion");
    return ok;
}

bool direct_guard_with_unrelated_guards(unsigned variant) {
    // Hundreds of comparisons concern different dynamic values. The final
    // comparison directly bounds index, including copies through a diamond
    // and an anchored phi cycle. Those unrelated guards must not obscure it.
    constexpr std::size_t unrelated = 512;
    constexpr std::size_t bound = unrelated, split = bound + 1, left = bound + 2, right = bound + 3,
                          join = bound + 4, body = bound + 5, exit = bound + 6;
    Graph graph(exit + 1);
    graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
    graph.add(0, "ld.param.u64", {"%choice", "[choice]"}, {2});
    graph.add(0, "ld.param.u64", {"%foreign", "[foreign]"}, {3});
    graph.add(0, "setp.ne.u64", {"%choose", "%choice", "0"}, {5});
    for (std::size_t block = 0; block < unrelated; ++block) {
        const auto name = "%noise" + std::to_string(block);
        const auto pred = "%test" + std::to_string(block);
        graph.add(block, "ld.param.u64", {name, "[noise]"}, {static_cast<ir::ValueId>(100 + 2 * block)});
        graph.add(block, "setp.ne.u64", {pred, name, "0"}, {static_cast<ir::ValueId>(101 + 2 * block)});
        graph.add(block, "bra", {"B" + std::to_string(block + 1)}, {}, pred);
        graph.edge(block, block + 1);
        // A bypass into the same copy path must defeat the final bound even
        // though all register names and copy instructions remain unchanged.
        graph.edge(block, variant == 1 && block == 0 ? left : exit);
    }
    graph.incoming[bound]["%index"] = 1;
    graph.add(bound, "setp.gt.u64", {"%too_large", "%index", "255"}, {4});
    graph.add(bound, "bra", {"B" + std::to_string(exit)}, {}, "%too_large");
    graph.edge(bound, exit);
    graph.edge(bound, split);
    graph.incoming[split]["%choose"] = 5;
    graph.add(split, "bra", {"B" + std::to_string(left)}, {}, "%choose");
    graph.edge(split, left);
    graph.edge(split, right);
    graph.incoming[left]["%index"] = 1;
    if (variant == 2)
        graph.add(left, "add.u64", {"%index", "%index", "279"}, {6});
    else
        graph.add(left, "mov.u64", {"%copy_left", "%index"}, {6});
    graph.add(left, "bra", {"B" + std::to_string(join)});
    graph.edge(left, join);
    graph.outgoing[left]["%merged"] = 6;
    graph.incoming[right]["%index"] = 1;
    graph.incoming[right]["%foreign"] = 3;
    graph.add(right, "mov.u64", {"%copy_right", variant == 3 ? "%foreign" : "%index"}, {7});
    graph.add(right, "bra", {"B" + std::to_string(join)});
    graph.edge(right, join);
    graph.outgoing[right]["%merged"] = 7;
    graph.phi(join, "%merged", 8);
    graph.incoming[join]["%choose"] = 5;
    graph.add(join, "mov.u64", {"%query", "%merged"}, {9});
    const auto* query = graph.add(join, "nop");
    graph.add(join, "bra", {"B" + std::to_string(body)}, {}, "%choose");
    graph.edge(join, body);
    graph.edge(join, exit);
    graph.incoming[body]["%merged"] = 8;
    if (variant == 4)
        graph.add(body, "add.u64", {"%merged", "%merged", "1"}, {10});
    graph.add(body, "bra", {"B" + std::to_string(join)});
    graph.edge(body, join);
    graph.outgoing[body]["%merged"] = variant == 4 ? 10 : 8;
    graph.add(exit, "ret");

    auto ranges = graph.ranges();
    const auto result = ranges.get(9, query);
    if (variant == 0)
        return known(result, 0, 255, "direct guard survives unrelated guards and copy/phi identities");
    if (variant == 2)
        return expect(!result || result->upper >= 534,
                      "a later addition cannot borrow its source's unmodified upper bound");
    return expect(!result, "bypassed, conflicting, or stale loop values cannot borrow the direct guard");
}

bool conditional_unit_increment(unsigned variant) {
    Graph graph(3);
    graph.add(0, variant == 4 ? "ld.param.u64" : "mov.u64",
              {"%initial", variant == 4 ? "[initial]" : "1"}, {1});
    graph.add(0, "ld.param.u64", {"%choice", "[choice]"}, {8});
    graph.add(0, "setp.ne.u64", {"%other", "%choice", "0"}, {9});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    graph.outgoing[0]["%index"] = 1;
    graph.phi(1, "%index", 2);
    graph.incoming[1]["%other"] = 9;
    graph.incoming[1]["%choice"] = 8;
    const auto* query = graph.add(1, "nop");
    graph.add(1, variant == 5 ? "setp.ge.u64" : "setp.lt.u64", {"%fits", "%index", "2"}, {3});
    graph.add(1, "selp.u64", {"%step", variant == 1 || variant == 5 ? "0" : "1",
                             variant == 1 || variant == 5 ? "1" : variant == 9 ? "99" : "0", "%fits"}, {4});
    graph.add(1, "add.u64", {"%next", "%index", variant == 3 ? "2" : "%step"}, {5});
    if (variant == 6) graph.add(1, "setp.ne.u64", {"%fits", "%choice", "0"}, {6});
    if (variant == 7 || variant == 8)
        graph.add(1, variant == 7 ? "and.pred" : "or.pred", {"%combined", "%fits", "%other"}, {7});
    const auto condition = variant == 2 ? "%other" : variant == 5 ? "!%fits" :
        variant == 7 || variant == 8 ? "%combined" : "%fits";
    graph.add(1, "bra", {"B1"}, {}, condition);
    graph.edge(1, 1);
    graph.edge(1, 2);
    graph.outgoing[1]["%index"] = 5;
    graph.add(2, "ret");
    auto ranges = graph.ranges();
    const auto result = ranges.get(2, query);
    if (variant == 3)
        return expect(!result || result->upper >= 3, "a two-step increment cannot borrow the unit-step bound");
    return variant == 0 || variant == 5 || variant == 7 || variant == 9
        ? known(result, 1, 2, "backedge proves the selected induction increment is one (" + std::to_string(variant) + ")")
        : expect(!result, "unknown step, stale predicate, bypass or dynamic seed cannot prove induction (" + std::to_string(variant) + ")");
}

bool guarded_join_relay(bool conflicting) {
    Graph graph(9);
    graph.add(0, "ld.param.u64", {"%left", "[left]"}, {1});
    graph.add(0, "ld.param.u64", {"%right", "[right]"}, {2});
    graph.add(0, "setp.ne.u64", {"%choice", "%left", "0"}, {3});
    graph.add(0, "bra", {"B1"}, {}, "%choice");
    graph.edge(0, 1); graph.edge(0, 2);
    for (std::size_t block : {1U, 2U}) {
        graph.incoming[block][block == 1 ? "%left" : "%right"] = block;
        graph.add(block, "mov.u64", {"%joined", block == 1 ? "%left" : "%right"},
                  {static_cast<ir::ValueId>(block + 3)});
        graph.add(block, "bra", {"B3"});
        graph.edge(block, 3);
        graph.outgoing[block]["%joined"] = block + 3;
    }
    graph.phi(3, "%joined", 6);
    graph.add(3, "setp.le.u64", {"%fits", "%joined", "31"}, {7});
    graph.add(3, "bra", {"B4"}, {}, "%fits");
    graph.edge(3, 4); graph.edge(3, 8);
    graph.incoming[4]["%choice"] = 3;
    graph.add(4, "bra", {"B5"}, {}, "%choice");
    graph.edge(4, 5); graph.edge(4, 6);
    for (std::size_t block : {5U, 6U}) {
        graph.incoming[block] = {{"%joined", 6}, {"%right", 2}};
        graph.add(block, "mov.u64", {"%relay", conflicting && block == 6 ? "%right" : "%joined"},
                  {static_cast<ir::ValueId>(block + 3)});
        graph.add(block, "bra", {"B7"});
        graph.edge(block, 7);
        graph.outgoing[block]["%relay"] = block + 3;
    }
    graph.phi(7, "%relay", 10);
    const auto* query = graph.add(7, "ret");
    graph.add(8, "ret");
    auto ranges = graph.ranges();
    const auto result = ranges.get(10, query);
    return conflicting ? expect(!result, "a conflicting relay cannot borrow a join's bound")
                       : known(result, 0, 31, "distinct multi-origin joins can relay the same bounded value");
}

bool many_independent_bounded_values() {
    // Many local store offsets have their own nearby bounds. Unrelated guards
    // must not consume the proof budget before later offsets are queried.
    constexpr std::size_t count = 128, exit = count * 2;
    Graph graph(exit + 1);
    std::vector<std::pair<ir::ValueId, const detail::Instruction*>> queries;
    for (std::size_t i = 0; i < count; ++i) {
        const auto block = i * 2;
        const auto value = static_cast<ir::ValueId>(i * 3 + 1);
        const auto name = "%index" + std::to_string(i);
        graph.add(block, "ld.param.u64", {name, "[index]"}, {value});
        graph.add(block, "setp.le.u64", {"%fits", name, "31"}, {value + 1});
        graph.add(block, "bra", {"B" + std::to_string(block + 1)}, {}, "%fits");
        graph.edge(block, block + 1);
        graph.edge(block, exit);
        queries.emplace_back(value, graph.add(block + 1, "nop"));
        graph.add(block + 1, "bra", {"B" + std::to_string(block + 2)});
        graph.edge(block + 1, block + 2);
    }
    const auto* after_merge = graph.add(exit, "ret");
    auto ranges = graph.ranges(detail::ScalarRangeLimits{.work = 175000});
    bool ok = true;
    for (const auto& [value, query] : queries) {
        if (!known(ranges.get(value, query), 0, 31, "independent guarded offsets fit bounded analysis")) {
            ok = false;
            break;
        }
    }
    // Every guard has an unbounded alternative into the exit. An affine origin
    // must never substitute for actual edge dominance.
    ok &= expect(!ranges.get(queries.back().first, after_merge),
                 "unbounded incoming edge remains unknown after affine matching");
    return ok;
}

bool excluded_endpoint_bounds() {
    bool ok = true;
    for (const std::string format : {"u64", "s64", "b64"}) {
        for (bool not_equal : {false, true}) {
            for (bool reverse : {false, true}) {
                // Variants 0/1 exclude upper/interior values; 2 bypasses the
                // exclusion, and 3 replaces its predicate before the branch.
                for (unsigned variant = 0; variant < 4; ++variant) {
                    Graph graph(4);
                    graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
                    graph.add(0, "ld.param.u64", {"%unknown", "[unknown]"}, {2});
                    graph.add(0, "setp.gt.u64", {"%too_large", "%index", "255"}, {3});
                    graph.add(0, "bra", {"B3"}, {}, "%too_large");
                    graph.edge(0, 3);
                    graph.edge(0, 1);
                    graph.incoming[1]["%index"] = 1;
                    graph.incoming[1]["%unknown"] = 2;
                    const std::string excluded = variant == 1 ? "254" : "255";
                    graph.add(1, "setp." + std::string(not_equal ? "ne." : "eq.") + format,
                              {"%test", reverse ? excluded : "%index", reverse ? "%index" : excluded}, {4});
                    if (variant == 3)
                        graph.add(1, "setp.ne.u64", {"%test", "%unknown", "0"}, {5});
                    if (variant == 2) {
                        graph.add(1, "bra", {"B2"});
                        graph.edge(1, 2);
                    } else {
                        graph.add(1, "bra", {not_equal ? "B2" : "B3"}, {}, "%test");
                        graph.edge(1, not_equal ? 2 : 3);
                        graph.edge(1, not_equal ? 3 : 2);
                    }
                    const auto* query = graph.add(2, "ret");
                    graph.add(3, "ret");
                    auto ranges = graph.ranges();
                    ok &= known(
                        ranges.get(1, query), 0, variant == 0 ? 254 : 255,
                        variant == 0
                            ? "excluding the bounded upper endpoint tightens its interval"
                            : "interior, bypassed, or overwritten tests do not remove the upper endpoint");
                }
            }
        }
    }

    // Endpoint removal is not a special zero rule. The arithmetic establishes
    // a separate 100..115 interval before a dominating comparison excludes 100.
    Graph graph(5);
    graph.add(0, "ld.param.u64", {"%input", "[input]"}, {1});
    graph.add(0, "setp.le.u64", {"%fits", "%input", "15"}, {2});
    graph.add(0, "bra", {"B1"}, {}, "%fits");
    graph.edge(0, 1);
    graph.edge(0, 4);
    graph.incoming[1]["%input"] = 1;
    graph.add(1, "add.u64", {"%index", "%input", "100"}, {3});
    graph.add(1, "setp.eq.u64", {"%endpoint", "%index", "100"}, {4});
    graph.add(1, "bra", {"B4"}, {}, "%endpoint");
    graph.edge(1, 4);
    graph.edge(1, 2);
    graph.incoming[2]["%index"] = 3;
    graph.add(2, "mov.u64", {"%copy", "%index"}, {5});
    graph.add(2, "bra", {"B3"});
    graph.edge(2, 3);
    const auto* query = graph.add(3, "ret");
    graph.add(4, "ret");
    auto ranges = graph.ranges();
    ok &= known(ranges.get(5, query), 101, 115,
                "excluding a nonzero lower endpoint tightens an independently established interval");
    return ok;
}

bool affine_sibling_bounds(unsigned variant) {
    // x+2 <=254 bounds the sibling x+3 to 1..255 even when x+2 itself
    // wrapped modulo 2^64. The proof relates actual SSA values, not names.
    enum : unsigned {
        plain,
        bypass,
        replaced_guard,
        different_base,
        replaced_base,
        narrow_guard,
        narrow_query,
        predicated_guard,
        predicated_query,
        overflowing_interval,
        negative_offsets,
        wrapped_base,
        subtraction,
        literal_left,
    };
    Graph graph(4);
    if (variant == wrapped_base)
        graph.add(0, "mov.u64", {"%base", "-2"}, {1});
    else
        graph.add(0, "ld.param.u64", {"%base", "[base]"}, {1});
    graph.add(0, "ld.param.u64", {"%foreign", "[foreign]"}, {2});
    graph.add(0, "setp.ne.u64", {"%choose", "%foreign", "0"}, {3});
    if (variant == bypass) {
        graph.add(0, "bra", {"B2"}, {}, "%choose");
        graph.edge(0, 2);
        graph.edge(0, 1);
    } else {
        graph.add(0, "bra", {"B1"});
        graph.edge(0, 1);
    }
    graph.incoming[1] = {{"%base", 1}, {"%foreign", 2}, {"%choose", 3}};
    graph.add(1, "mov.u64", {"%bounded", "%foreign"}, {4});
    const std::string guard_delta = variant == negative_offsets ? "-2" : "2";
    graph.add(1,
              variant == narrow_guard  ? "add.u32"
              : variant == subtraction ? "sub.u64"
                                       : "add.u64",
              {"%bounded", variant == literal_left ? guard_delta : "%base",
               variant == literal_left ? "%base" : guard_delta},
              {5}, variant == predicated_guard ? "%choose" : "");
    if (variant == replaced_guard)
        graph.add(1, "mov.u64", {"%bounded", "%foreign"}, {6});
    graph.add(1, "setp.le.u64", {"%fits", "%bounded", "254"}, {7});
    graph.add(1, "bra", {"B2"}, {}, "%fits");
    graph.edge(1, 2);
    graph.edge(1, 3);
    graph.incoming[2] = {{"%base", 1}, {"%foreign", 2}, {"%choose", 3}};
    if (variant == replaced_base)
        graph.add(2, "ld.param.u64", {"%base", "[replacement]"}, {8});
    graph.add(2, "mov.u64", {"%query", "%foreign"}, {9});
    const std::string delta = variant == overflowing_interval ? "9223372036854775807"
                              : variant == negative_offsets   ? "-1"
                              : variant == subtraction        ? "1"
                                                              : "3";
    const std::string query_base = variant == different_base ? "%foreign" : "%base";
    graph.add(2,
              variant == narrow_query  ? "add.u32"
              : variant == subtraction ? "sub.u64"
                                       : "add.u64",
              {"%query", variant == literal_left ? delta : query_base,
               variant == literal_left ? query_base : delta},
              {10}, variant == predicated_query ? "%choose" : "");
    graph.add(2, "mov.u64", {"%copy", "%query"}, {11});
    const auto* query = graph.add(2, "ret");
    graph.add(3, "ret");
    auto ranges = graph.ranges();
    const auto result = ranges.get(11, query);
    if (variant == plain || variant == negative_offsets || variant == subtraction || variant == literal_left)
        return known(result, 1, 255, "a guarded affine sibling bounds the queried value");
    if (variant == wrapped_base)
        return expect(result && result->lower == 1 && result->upper >= 1 && result->upper <= 255,
                      "modulo64 wrap in the shared base does not break a valid small sibling interval");
    return expect(
        !result,
        "unrelated, bypassed, overwritten, unsupported-width, predicated or overflowing affine case " +
            std::to_string(variant) + " remains unknown");
}

bool affine_joined_bases(bool conflicting) {
    Graph graph(6);
    graph.add(0, "ld.param.u64", {"%base", "[base]"}, {1});
    graph.add(0, "ld.param.u64", {"%foreign", "[foreign]"}, {2});
    graph.add(0, "setp.ne.u64", {"%choose", "%foreign", "0"}, {3});
    graph.add(0, "bra", {"B1"}, {}, "%choose");
    graph.edge(0, 1);
    graph.edge(0, 2);
    graph.incoming[1]["%base"] = 1;
    graph.add(1, "mov.u64", {"%joined", "%base"}, {4});
    graph.add(1, "bra", {"B3"});
    graph.edge(1, 3);
    graph.outgoing[1]["%joined"] = 4;
    graph.incoming[2] = {{"%base", 1}, {"%foreign", 2}};
    graph.add(2, "mov.u64", {"%joined", conflicting ? "%foreign" : "%base"}, {5});
    graph.add(2, "bra", {"B3"});
    graph.edge(2, 3);
    graph.outgoing[2]["%joined"] = 5;
    graph.phi(3, "%joined", 6);
    graph.add(3, "add.u64", {"%bounded", "%joined", "2"}, {7});
    graph.add(3, "mov.u64", {"%bound_copy", "%bounded"}, {8});
    graph.add(3, "setp.le.u64", {"%fits", "%bound_copy", "254"}, {9});
    graph.add(3, "bra", {"B4"}, {}, "%fits");
    graph.edge(3, 4);
    graph.edge(3, 5);
    // Deliberately query a sibling of the original value, not of the join.
    // Only agreement on every join edge permits transferring the bound.
    graph.incoming[4]["%base"] = 1;
    graph.add(4, "add.s64", {"%query", "%base", "3"}, {10});
    const auto* query = graph.add(4, "ret");
    graph.add(5, "ret");
    auto ranges = graph.ranges();
    const auto result = ranges.get(10, query);
    return conflicting
               ? expect(!result, "a different incoming base defeats affine identity through the join")
               : known(result, 1, 255, "all-edge copy identities relate affine siblings across a join");
}

bool boolean_guard_truth_table() {
    bool ok = true;
    for (bool conjunction : {false, true}) {
        for (bool truth : {false, true}) {
            for (bool inverted : {false, true}) {
                for (bool reverse : {false, true}) {
                    Graph graph(3);
                    graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
                    graph.add(0, "ld.param.u64", {"%other", "[other]"}, {2});
                    graph.add(0, conjunction ? "setp.le.u64" : "setp.gt.u64", {"%bound", "%index", "255"},
                              {3});
                    graph.add(0, "setp.ne.u64", {"%unrelated", "%other", "0"}, {4});
                    graph.add(
                        0, conjunction ? "and.pred" : "or.pred",
                        {"%combined", reverse ? "%unrelated" : "%bound", reverse ? "%bound" : "%unrelated"},
                        {5});
                    const bool taken = truth != inverted;
                    graph.add(0, "bra", {taken ? "B1" : "B2"}, {}, inverted ? "!%combined" : "%combined");
                    graph.edge(0, taken ? 1 : 2);
                    graph.edge(0, taken ? 2 : 1);
                    const auto* query = graph.add(1, "ret");
                    graph.add(2, "ret");
                    auto ranges = graph.ranges();
                    const auto bound = ranges.get(1, query);
                    if (conjunction == truth)
                        ok &= known(bound, 0, 255,
                                    "AND true / OR false imply each necessary comparison, including inverted "
                                    "branches");
                    else
                        ok &= expect(!bound,
                                     "AND false / OR true cannot choose which comparison controls the edge");
                }
            }
        }
    }
    return ok;
}

bool boolean_guard_redefinitions() {
    bool ok = true;
    for (unsigned variant = 0; variant < 4; ++variant) {
        Graph graph(3);
        graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
        graph.add(0, "ld.param.u64", {"%other", "[other]"}, {2});
        graph.add(0, "setp.gt.u64", {"%bound", "%index", "255"}, {3});
        graph.add(0, "setp.ne.u64", {"%unrelated", "%other", "0"}, {4});
        if (variant == 1)
            graph.add(0, "mov.pred", {"%bound", "%unrelated"}, {5});
        graph.add(0, "mov.pred", {"%combined", "%unrelated"}, {6});
        graph.add(0, variant == 3 ? "xor.pred" : "or.pred", {"%combined", "%bound", "%unrelated"}, {7},
                  variant == 2 ? "%unrelated" : "");
        if (variant == 0)
            graph.add(0, "mov.pred", {"%combined", "%unrelated"}, {8});
        graph.add(0, "bra", {"B2"}, {}, "%combined");
        graph.edge(0, 2);
        graph.edge(0, 1);
        const auto* query = graph.add(1, "ret");
        graph.add(2, "ret");
        auto ranges = graph.ranges();
        ok &= expect(!ranges.get(1, query), "unrelated/redefined/predicated/unsupported predicate case " +
                                                std::to_string(variant) +
                                                " cannot retain a stale comparison fact");
    }
    return ok;
}

bool nested_predicate_copies() {
    Graph graph(3);
    graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
    graph.add(0, "ld.param.u64", {"%other", "[other]"}, {2});
    graph.add(0, "setp.gt.u64", {"%over", "%index", "255"}, {3});
    graph.add(0, "setp.ne.u64", {"%unrelated", "%other", "0"}, {4});
    graph.add(0, "mov.pred", {"%copy", "%over"}, {5});
    graph.add(0, "not.pred", {"%fits", "%copy"}, {6});
    graph.add(0, "and.pred", {"%both", "%fits", "%unrelated"}, {7});
    graph.add(0, "mov.pred", {"%again", "%both"}, {8});
    graph.add(0, "not.pred", {"%skip", "%again"}, {9});
    graph.add(0, "bra", {"B2"}, {}, "%skip");
    graph.edge(0, 2);
    graph.edge(0, 1);
    const auto* query = graph.add(1, "ret");
    graph.add(2, "ret");
    auto ranges = graph.ranges();
    return known(ranges.get(1, query), 0, 255, "nested NOT/MOV preserves the required comparison polarity");
}

bool cyclic_predicate_cannot_reuse_entry_fact() {
    // When the entry comparison is false, the backedge toggles it and the
    // queried block is reached next. Its index therefore need not be <=255.
    Graph graph(4);
    graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
    graph.add(0, "setp.le.u64", {"%carried", "%index", "255"}, {2});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    graph.outgoing[0]["%carried"] = 2;
    graph.phi(1, "%carried", 3);
    graph.types[3] = ir::Type::predicate();
    graph.add(1, "bra", {"B2"}, {}, "%carried");
    graph.edge(1, 2);
    graph.edge(1, 3);
    const auto* query = graph.add(2, "ret");
    graph.incoming[3]["%carried"] = 3;
    graph.add(3, "not.pred", {"%carried", "%carried"}, {4});
    graph.add(3, "bra", {"B1"});
    graph.edge(3, 1);
    graph.outgoing[3]["%carried"] = 4;
    auto ranges = graph.ranges();
    return expect(!ranges.get(1, query), "a predicate join/backedge cannot borrow only the entry fact");
}

bool bounded_left_shifts() {
    bool ok = true;
    for (unsigned shift : {0U, 3U, 59U, 60U, 63U, 64U}) {
        Graph graph(3);
        graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
        graph.add(0, "setp.lt.u64", {"%fits", "%index", "16"}, {2});
        graph.add(0, "bra", {"B1"}, {}, "%fits");
        graph.edge(0, 1);
        graph.edge(0, 2);
        graph.incoming[1]["%index"] = 1;
        graph.add(1, "shl.b64", {"%offset", "%index", std::to_string(shift)}, {3});
        const auto* query = graph.add(1, "ret");
        graph.add(2, "ret");
        auto ranges = graph.ranges();
        if (shift <= 59)
            ok &= known(ranges.get(3, query), 0, static_cast<std::int64_t>(15ULL << shift),
                        "bounded left shift preserves its representable interval");
        else
            ok &= expect(!ranges.get(3, query), "wrapping or out-of-range left shifts stay unknown");
    }
    Graph unknown(1);
    unknown.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
    unknown.add(0, "shl.b64", {"%offset", "%index", "3"}, {2});
    const auto* query = unknown.add(0, "ret");
    auto ranges = unknown.ranges();
    ok &= expect(!ranges.get(2, query), "left shift alone does not establish a bound");
    return ok;
}

bool captured_offset_guards() {
    bool ok = true;
    for (const bool masked : {false, true}) {
        Graph graph(3);
        graph.add(0, "ld.param.u64", {"%input", "[input]"}, {1});
        graph.add(0, masked ? "and.b64" : "mov.u64",
                  masked ? std::vector<std::string>{"%index", "%input", "255"}
                         : std::vector<std::string>{"%index", "%input"}, {2});
        const auto* creation = graph.add(0, "add.u64", {"%pointer", "%base", "%index"}, {3});
        graph.add(0, "setp.le.u64", {"%fits", "%index", "32"}, {4});
        graph.add(0, "bra", {"B1"}, {}, "%fits");
        graph.edge(0, 1);
        graph.edge(0, 2);
        const auto* use = graph.add(1, "st.local.u8", {"[%pointer]", "47"});
        graph.add(2, "ret");
        auto ranges = graph.ranges();
        ok &= known(ranges.captured(2, creation, use), 0, 32,
                    "a later guard refines an unchanged captured scalar");
        auto exhausted = graph.ranges({.work = 0});
        ok &= expect(!exhausted.captured(2, creation, use),
                     "exhausted capture analysis does not invent a bound");
    }
    // The pointer keeps an older iteration's phi value. A guard on the next
    // iteration's value cannot constrain that captured offset.
    Graph graph(6);
    graph.add(0, "ld.param.u64", {"%index", "[index]"}, {1});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    graph.outgoing[0]["%index"] = 1;
    graph.phi(1, "%index", 2);
    graph.add(1, "ld.param.u64", {"%choice", "[choice]"}, {3});
    graph.add(1, "setp.eq.u64", {"%capture", "%choice", "0"}, {4});
    graph.add(1, "bra", {"B2"}, {}, "%capture");
    graph.edge(1, 2);
    graph.edge(1, 3);
    graph.incoming[2]["%index"] = 2;
    const auto* creation = graph.add(2, "add.u64", {"%pointer", "%base", "%index"}, {5});
    graph.add(2, "mov.u64", {"%index", "0"}, {6});
    graph.add(2, "bra", {"B1"});
    graph.edge(2, 1);
    graph.outgoing[2]["%index"] = 6;
    graph.incoming[3]["%index"] = 2;
    graph.add(3, "setp.le.u64", {"%fits", "%index", "32"}, {7});
    graph.add(3, "bra", {"B4"}, {}, "%fits");
    graph.edge(3, 4);
    graph.edge(3, 5);
    const auto* use = graph.add(4, "st.local.u8", {"[%pointer]", "47"});
    graph.add(5, "ret");
    auto ranges = graph.ranges();
    ok &= known(ranges.get(2, use), 0, 32, "current loop value has the guard's bound");
    ok &= expect(!ranges.captured(2, creation, use),
                 "the current loop bound cannot constrain a remembered older pointer");
    return ok;
}

bool equality_terminated_counter(unsigned variant) {
    Graph graph(3);
    graph.add(0, "mov.u64", {"%i", variant == 2 ? "4" : variant == 3 ? "5" : "1"}, {1});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    graph.outgoing[0]["%i"] = 1;
    graph.phi(1, "%i", 2);
    const auto* query = graph.add(1, "add.u64", {"%next", "%i", variant == 4 ? "2" : "1"}, {3});
    graph.add(1, variant == 1 ? "setp.eq.u64" : "setp.ne.u64",
              {"%again", variant == 5 ? "%i" : "%next", "4"}, {4});
    if (variant == 6) graph.add(1, "setp.ne.u64", {"%again", "%next", "0"}, {5});
    graph.add(1, "bra", {"B1"}, {}, variant == 1 ? "!%again" : "%again");
    graph.edge(1, 1);
    graph.edge(1, 2);
    graph.outgoing[1]["%i"] = 3;
    graph.add(2, "ret");
    auto ranges = graph.ranges();
    const auto range = ranges.get(2, query);
    if (variant == 0 || variant == 1 || variant == 5)
        return known(range, 1, variant == 5 ? 4 : 3,
                     "unit counter reaches its equality endpoint on every backedge");
    return expect(!range, "endpoint guard cannot bound a skipped/redefined/wrapping endpoint");
}

bool counted_pointer_loop(unsigned variant) {
    Graph graph(3);
    graph.add(0, "mov.u64", {"%pointer", "scratch"}, {1});
    const bool up = variant == 12 || variant == 13 || variant == 14 || variant == 19 || variant == 20;
    graph.add(0, variant == 7 ? "ld.param.u64" : "mov.u64",
              {"%remaining", variant == 20 ? "9223372036854775805" : up ? "0" :
                  variant == 3 ? "61" : variant == 4 ? "0" : variant == 7 ? "[unknown]" : "60"}, {2});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    if (variant == 5) graph.edge(0, 2);
    graph.outgoing[0] = {{"%pointer", 1}, {"%remaining", 2}};
    graph.phi(1, "%pointer", 3);
    graph.phi(1, "%remaining", 4);
    graph.add(1, variant == 6 ? "add.u32" : "add.u64",
              {"%pointer", "%pointer", variant == 1 ? "-4" : "4"}, {5});
    graph.add(1, up ? "add.u64" : "sub.u64",
              {"%remaining", "%remaining", variant == 19 ? "2" : up && variant != 20 ? "1" : "4"}, {6});
    const auto comparison = variant == 2 ? "setp.eq.u64" : variant == 12 || variant == 19 || variant == 20
        ? "setp.lt.u64" : variant == 14 ? "setp.le.u64" : variant == 15 || variant == 18 ? "setp.gt.u64"
        : variant == 16 || variant == 17 ? "setp.ge.u64" : "setp.ne.u64";
    const auto limit = variant == 20 ? "9223372036854775806" : variant == 19 ? "29" : variant == 12 || variant == 13
        ? "15" : variant == 14 ? "14" : variant == 16 ? "4" : variant == 8 || variant == 18 ? "1" : "0";
    graph.add(1, comparison, {"%again", "%remaining", limit}, {7});
    if (variant == 9) graph.add(1, "setp.ne.u64", {"%again", "%pointer", "0"}, {8});
    graph.add(1, "bra", {"B1"}, {}, variant == 2 ? "!%again" : "%again");
    graph.edge(1, 1);
    graph.edge(1, 2);
    graph.outgoing[1] = {{"%pointer", 5}, {"%remaining", variant == 10 ? 4u : 6u}};
    const auto* use = graph.add(2, "ret");
    auto scalars = graph.ranges();
    const auto base = variant == 1 ? 128 : variant == 11 ? 224 : 32;
    detail::PointerRanges pointers(graph.blocks, graph.incoming, graph.outgoing, graph.arguments,
        graph.results,
        [&](ir::ValueId value, const detail::Instruction*) -> std::optional<detail::ExactLocalAddress> {
            if (value == 1) return detail::ExactLocalAddress{"scratch", base, 256};
            return std::nullopt;
        }, [&](ir::ValueId value, const detail::Instruction* at) { return scalars.get(value, at); });
    const auto range = pointers.get(3, use);
    const auto label = "counted pointer variant " + std::to_string(variant);
    if (variant <= 2 || (variant >= 12 && variant <= 19 && variant != 17)) {
        const auto lo = variant == 1 ? 72 : 32;
        const auto hi = variant == 1 ? 128 : 88;
        return expect(range && range->depot == "scratch" && range->lower == lo && range->upper == hi,
                      label + " bounds the phi over all finite trips");
    }
    return expect(!range, label + " refuses an unproved trip count, update, guard or allocation extent");
}

} // namespace

int main() {
    bool ok = branch_bounds();
    for (unsigned variant = 0; variant < 7; ++variant) ok &= equality_terminated_counter(variant);
    for (unsigned variant = 0; variant < 21; ++variant) ok &= counted_pointer_loop(variant);
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
    ok &= cached_bounds_survive_exhaustion();
    for (unsigned variant = 0; variant < 5; ++variant)
        ok &= direct_guard_with_unrelated_guards(variant);
    for (unsigned variant = 0; variant < 10; ++variant)
        ok &= conditional_unit_increment(variant);
    ok &= guarded_join_relay(false);
    ok &= guarded_join_relay(true);
    ok &= many_independent_bounded_values();
    ok &= excluded_endpoint_bounds();
    for (unsigned variant = 0; variant < 14; ++variant)
        ok &= affine_sibling_bounds(variant);
    ok &= affine_joined_bases(false);
    ok &= affine_joined_bases(true);
    ok &= boolean_guard_truth_table();
    ok &= boolean_guard_redefinitions();
    ok &= nested_predicate_copies();
    ok &= cyclic_predicate_cannot_reuse_entry_fact();
    ok &= captured_offset_guards();
    ok &= bounded_left_shifts();
    return ok ? 0 : 1;
}
