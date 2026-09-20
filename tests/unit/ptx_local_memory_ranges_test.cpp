#include "ptx_local_memory_ranges.h"

#include <deque>
#include <iostream>
#include <tuple>

namespace {
namespace detail = cumetal::ir::detail;

struct Graph {
    std::deque<detail::Instruction> storage;
    std::vector<detail::RawBlock> blocks;
    std::unordered_map<std::string, std::uint64_t> depots{{"depot", 256}};
    explicit Graph(std::size_t count) : blocks(count) {
        for (std::size_t i = 0; i < count; ++i)
            blocks[i].name = "B" + std::to_string(i);
    }
    const detail::Instruction* add(std::size_t block, std::string opcode,
                                   std::vector<std::string> operands = {}, std::string predicate = {}) {
        storage.push_back({std::move(predicate), std::move(opcode), std::move(operands), 1, true});
        const auto* instruction = &storage.back();
        blocks[block].instructions.push_back(instruction);
        return instruction;
    }
    void edge(std::size_t from, std::size_t to) {
        blocks[from].successors.push_back(to);
        blocks[to].predecessors.push_back(from);
    }
    void record(std::uint64_t length = 10) {
        add(0, "mov.b64", {"%base", "depot"});
        add(0, "cvta.local.u64", {"%generic", "%base"});
        add(0, "add.u64", {"%record", "%base", "160"});
        // The pointer lane is not scalar bytes. The adjacent integer lane is
        // fully initialized and must survive an exact vector store.
        add(0, "st.local.v2.b64", {"[%record]", "{%generic, " + std::to_string(length) + "}"});
    }
    detail::LocalMemoryRangeProof run(const detail::Instruction* load,
                                      detail::LocalMemoryRangeLimits limits = {},
                                      std::function<bool(const detail::Instruction&)> preserves = {}) const {
        return detail::LocalMemoryRanges(blocks, depots, std::move(preserves), limits).prove_before(load);
    }
};

bool expect(bool condition, const std::string& message) {
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}
bool range(const detail::LocalMemoryRangeProof& proof, const detail::Instruction* store, std::int64_t lower,
           std::int64_t upper, const std::string& message) {
    const auto found = proof.stores.find(store);
    return expect(proof.complete && found != proof.stores.end() && found->second.depot == "depot" &&
                      found->second.lower == lower && found->second.upper == upper,
                  message + ": " + proof.reason);
}

bool initialized_pointer_loop(unsigned step, unsigned count) {
    Graph graph(3);
    graph.record(count);
    graph.add(0, "ld.local.b64", {"%count", "[%record+8]"});
    graph.add(0, "add.u64", {"%cursor", "%generic", "32"});
    graph.add(0, "add.u64", {"%end", "%cursor", "%count"});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    const auto* store = graph.add(1, "st.local.b8", {"[%cursor]", "%unknown"});
    graph.add(1, "add.s64", {"%cursor", "%cursor", std::to_string(step)});
    graph.add(1, "setp.ne.b64", {"%again", "%cursor", "%end"});
    graph.add(1, "bra", {"B1"}, "%again");
    graph.edge(1, 1);
    graph.edge(1, 2);
    const auto* target = graph.add(2, "ld.local.b64", {"%loaded", "[%record]"});
    return range(graph.run(target), store, 32, 32 + count - step,
                 "initialized scalar length bounds a pointer loop with step " + std::to_string(step));
}

bool two_phase_pointer_loop() {
    Graph graph(6);
    graph.record();
    graph.add(0, "ld.local.b64", {"%length", "[%record+8]"});
    graph.add(0, "sub.s64", {"%remaining", "31", "%length"});
    graph.add(0, "and.b64", {"%short", "%remaining", "3"});
    graph.add(0, "add.u64", {"%cursor", "%generic", "32"});
    graph.add(0, "add.u64", {"%end", "%cursor", "%remaining"});
    graph.add(0, "setp.eq.b64", {"%empty", "%short", "0"});
    graph.add(0, "bra", {"B2"}, "%empty");
    graph.edge(0, 2);
    graph.edge(0, 1);
    const auto* short_store = graph.add(1, "st.local.b8", {"[%cursor]", "255"});
    graph.add(1, "add.s64", {"%cursor", "%cursor", "1"});
    graph.add(1, "add.s64", {"%short", "%short", "-1"});
    graph.add(1, "setp.ne.b64", {"%again", "%short", "0"});
    graph.add(1, "bra", {"B1"}, "%again");
    graph.edge(1, 1);
    graph.edge(1, 2);
    graph.add(2, "setp.eq.b64", {"%done", "%cursor", "%end"});
    graph.add(2, "bra", {"B5"}, "%done");
    graph.edge(2, 5);
    graph.edge(2, 3);
    const auto* wide_store = graph.add(3, "st.local.v4.b8", {"[%cursor]", "{1, 2, 3, 4}"});
    graph.add(3, "add.s64", {"%cursor", "%cursor", "4"});
    graph.add(3, "bra", {"B4"});
    graph.edge(3, 4);
    graph.add(4, "setp.ne.b64", {"%again", "%cursor", "%end"});
    graph.add(4, "bra", {"B3"}, "%again");
    graph.edge(4, 3);
    graph.edge(4, 5);
    const auto* target = graph.add(5, "ld.local.b64", {"%loaded", "[%record]"});
    const auto proof = graph.run(target);
    return range(proof, short_store, 32, 32, "masked byte phase has one iteration") &
           range(proof, wide_store, 33, 49, "following four-byte phase covers five exact starts");
}

bool unknown_branch_union(bool unknown_address) {
    Graph graph(4);
    graph.record();
    graph.add(0, "bra", {"B1"}, "%choice");
    graph.edge(0, 1);
    graph.edge(0, 2);
    graph.add(1, "add.u64", {"%cursor", "%generic", "32"});
    graph.add(1, "bra", {"B3"});
    graph.edge(1, 3);
    if (unknown_address)
        graph.add(2, "ld.param.u64", {"%cursor", "[input]"});
    else
        graph.add(2, "add.u64", {"%cursor", "%generic", "200"});
    graph.add(2, "bra", {"B3"});
    graph.edge(2, 3);
    const auto* store = graph.add(3, "st.local.u8", {"[%cursor]", "0"});
    const auto* target = graph.add(3, "ld.local.u64", {"%loaded", "[%record]"});
    const auto proof = graph.run(target);
    if (unknown_address)
        return expect(proof.complete && !proof.stores.contains(store),
                      "one unknown visit permanently poisons a static store's address");
    return range(proof, store, 32, 200, "all branch outcomes contribute to the conservative interval");
}

bool overwritten_scalar_record(bool known_overwrite, bool conditional) {
    Graph graph(1);
    graph.record();
    graph.add(0, "st.local.b8", {"[%record+8]", known_overwrite ? "2" : "%unknown"},
              conditional ? "%choice" : "");
    graph.add(0, "ld.local.b64", {"%count", "[%record+8]"});
    graph.add(0, "add.u64", {"%cursor", "%generic", "%count"});
    const auto* store = graph.add(0, "st.local.u8", {"[%cursor]", "0"});
    const auto* target = graph.add(0, "ld.local.u64", {"%loaded", "[%record]"});
    const auto proof = graph.run(target);
    if (!known_overwrite)
        return expect(proof.complete && !proof.stores.contains(store),
                      "unknown partial scalar overwrite prevents an exact subsequent length");
    return range(proof, store, 2, conditional ? 10 : 2,
                 "byte writes replace initialized data and unknown predicates explore both outcomes");
}

bool calls_invalidate_memory(bool preserves) {
    Graph graph(1);
    graph.record();
    graph.add(0, "call.uni", {"helper", "()"});
    graph.add(0, "ld.local.b64", {"%count", "[%record+8]"});
    graph.add(0, "add.u64", {"%cursor", "%generic", "%count"});
    const auto* store = graph.add(0, "st.local.u8", {"[%cursor]", "0"});
    const auto* target = graph.add(0, "ld.local.u64", {"%loaded", "[%record]"});
    const auto proof = graph.run(target, {}, [preserves](const auto&) { return preserves; });
    if (preserves)
        return range(proof, store, 10, 10, "explicit call preservation retains scalar bytes");
    return expect(proof.complete && !proof.stores.contains(store),
                  "unclassified call invalidates caller bytes");
}

bool overwritten_register_and_boundaries() {
    bool ok = true;
    for (unsigned variant = 0; variant < 4; ++variant) {
        Graph graph(1);
        graph.record();
        graph.add(0, "add.u64", {"%cursor", "%generic", "64"});
        if (variant == 0)
            graph.add(0, "cvt.u32.u64", {"%cursor", "%cursor"});
        if (variant == 1)
            graph.add(0, "add.u64", {"%cursor", "%generic", "256"});
        if (variant == 2) {
            graph.add(0, "mov.s64", {"%index", "9223372036854775807"});
            graph.add(0, "add.s64", {"%index", "%index", "1"});
            graph.add(0, "add.u64", {"%cursor", "%generic", "%index"});
        }
        if (variant == 3) {
            graph.add(0, "mov.u64", {"%index", "-1"});
            graph.add(0, "add.s64", {"%cursor", "%cursor", "%index"});
        }
        const auto* store = graph.add(0, "st.local.u8", {"[%cursor]", "0"});
        const auto* target = graph.add(0, "ld.local.u64", {"%loaded", "[%record]"});
        const auto proof = graph.run(target);
        if (variant == 3)
            ok &= range(proof, store, 63, 63, "negative literal retains explicit 64-bit bit semantics");
        else
            ok &= expect(proof.complete && !proof.stores.contains(store),
                         "unsupported redefinition, one-past access or signed overflow yields unknown (" +
                             std::to_string(variant) + ")");
    }
    return ok;
}

bool independent_32bit_loop_counter() {
    Graph graph(3);
    graph.record();
    graph.add(0, "mov.u32", {"%count", "0"});
    graph.add(0, "add.u64", {"%cursor", "%generic", "32"});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    const auto* store = graph.add(1, "st.local.b8", {"[%cursor]", "0"});
    graph.add(1, "add.s32", {"%count", "%count", "1"});
    graph.add(1, "add.s64", {"%cursor", "%cursor", "1"});
    graph.add(1, "setp.lt.u32", {"%again", "%count", "32"});
    graph.add(1, "bra", {"B1"}, "%again");
    graph.edge(1, 1);
    graph.edge(1, 2);
    const auto* target = graph.add(2, "ld.local.b64", {"%loaded", "[%record]"});
    return range(graph.run(target), store, 32, 63,
                 "32-bit scalar induction bounds the independently updated 64-bit pointer");
}

bool width_and_overflow_boundaries() {
    bool ok = true;
    for (unsigned variant = 0; variant < 5; ++variant) {
        Graph graph(4);
        graph.record();
        if (variant == 0 || variant == 4)
            graph.add(0, "mov.u64", {"%input", "4294967296"});
        if (variant == 1 || variant == 3)
            graph.add(0, "mov.b32", {"%input", "4294967295"});
        if (variant == 2)
            graph.add(0, "mov.s32", {"%input", "2147483647"});
        if (variant != 4)
            graph.add(0, variant == 0 || variant == 3 ? "add.u32" : "add.s32", {"%result", "%input", "1"});
        if (variant == 0)
            graph.add(0, "setp.eq.u32", {"%choose", "%result", "1"});
        if (variant == 1)
            graph.add(0, "setp.eq.s32", {"%choose", "%result", "0"});
        if (variant == 2 || variant == 3)
            graph.add(0, "setp.lt.s32", {"%choose", "%result", "0"});
        if (variant == 4)
            graph.add(0, "setp.eq.u32", {"%choose", "%input", "0"});
        graph.add(0, "bra", {"B1"}, "%choose");
        graph.edge(0, 1);
        graph.edge(0, 2);
        graph.add(1, "add.u64", {"%cursor", "%generic", "32"});
        graph.add(1, "bra", {"B3"});
        graph.edge(1, 3);
        graph.add(2, "add.u64", {"%cursor", "%generic", "200"});
        graph.add(2, "bra", {"B3"});
        graph.edge(2, 3);
        const auto* store = graph.add(3, "st.local.u8", {"[%cursor]", "0"});
        const auto* target = graph.add(3, "ld.local.u64", {"%loaded", "[%record]"});
        ok &= range(graph.run(target), store, 32, variant == 2 || variant == 3 ? 200 : 32,
                    "32-bit extraction, signed interpretation and conservative overflow (" +
                        std::to_string(variant) + ")");
    }
    Graph narrow(1);
    narrow.record();
    narrow.add(0, "mov.u32", {"%index", "10"});
    narrow.add(0, "mov.b64", {"%index", "%index"});
    narrow.add(0, "add.u64", {"%cursor", "%generic", "%index"});
    const auto* store = narrow.add(0, "st.local.u8", {"[%cursor]", "0"});
    const auto* target = narrow.add(0, "ld.local.u64", {"%loaded", "[%record]"});
    const auto proof = narrow.run(target);
    ok &= expect(proof.complete && !proof.stores.contains(store),
                 "a 32-bit result cannot silently supply unknown upper bits to a 64-bit operand");
    return ok;
}

bool scalar_shifts() {
    bool ok = true;
    for (const auto& [opcode, source, count, expected] :
         std::vector<std::tuple<std::string, std::string, std::string, unsigned>>{
             {"shl.b64", "3", "3", 24},
             {"shl.b32", "4294967299", "4294967299", 24},
             {"shl.b64", "18446744073709551615", "64", 0},
             {"shl.b32", "4294967295", "32", 0},
             {"shr.u64", "192", "3", 24},
             {"shr.b32", "4294967488", "3", 24}}) {
        Graph graph(1);
        graph.record();
        graph.add(0, opcode, {"%index", source, count});
        // Narrow values intentionally cannot supply unknown upper address bits.
        const bool narrow = opcode.ends_with("32");
        graph.add(0, "add.u64", {"%cursor", "%generic", "%index"});
        const auto* store = graph.add(0, "st.local.u8", {"[%cursor]", "0"});
        const auto* target = graph.add(0, "ld.local.u64", {"%loaded", "[%record]"});
        const auto proof = graph.run(target);
        if (narrow)
            ok &=
                expect(proof.complete && !proof.stores.contains(store), "narrow shifted result stays narrow");
        else
            ok &= range(proof, store, expected, expected,
                        "logical shifts use chopped operands and saturated counts");
    }
    return ok;
}

bool dead_register_facts_are_pruned() {
    Graph graph(3);
    graph.record();
    graph.add(0, "add.u64", {"%cursor", "%generic", "32"});
    graph.add(0, "add.u64", {"%end", "%cursor", "10"});
    for (unsigned i = 0; i < 64; ++i)
        graph.add(0, "mov.u64", {"%dead" + std::to_string(i), std::to_string(i + 1)});
    // The skip outcome creates a mid-block state. Both it and the next block's
    // entry must discard dead facts while keeping cursor/end and all bytes.
    graph.add(0, "mov.u64", {"%unused", "1"}, "%unknown");
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    const auto* store = graph.add(1, "st.local.b8", {"[%cursor]", "%unknown_word"});
    graph.add(1, "add.s64", {"%cursor", "%cursor", "1"});
    graph.add(1, "setp.ne.b64", {"%again", "%cursor", "%end"});
    graph.add(1, "bra", {"B1"}, "%again");
    graph.edge(1, 1);
    graph.edge(1, 2);
    const auto* target = graph.add(2, "ld.local.b64", {"%loaded", "[%record]"});
    detail::LocalMemoryRangeLimits limits;
    limits.retained_facts = 200;
    bool ok = range(graph.run(target, limits), store, 32, 41,
                    "dead temporaries do not consume each retained prefix state");
    limits.operations = 150;
    const auto exhausted = graph.run(target, limits);
    ok &= expect(!exhausted.complete && exhausted.stores.empty() &&
                     exhausted.reason.find("operations budget exhausted") != std::string::npos &&
                     exhausted.reason.find("register liveness") != std::string::npos,
                 "unfinished liveness consumes the common work budget and publishes no facts: " +
                     exhausted.reason);
    return ok;
}

bool liveness_preserves_bypass_values() {
    Graph graph(4);
    graph.record();
    graph.add(0, "add.u64", {"%cursor", "%generic", "160"});
    graph.add(0, "bra", {"B1"}, "%choice");
    graph.edge(0, 1);
    graph.edge(0, 2);
    graph.add(1, "add.u64", {"%cursor", "%generic", "32"});
    graph.add(1, "bra", {"B3"});
    graph.edge(1, 3);
    graph.add(2, "bra", {"B3"});
    graph.edge(2, 3);
    const auto* store = graph.add(3, "st.local.b8", {"[%cursor]", "0"});
    const auto* target = graph.add(3, "ld.local.b64", {"%loaded", "[%record]"});
    // One path overwrites the pointer cell. A definition on the other path
    // cannot kill the old address when crossing the branch.
    return range(graph.run(target), store, 32, 160,
                 "liveness unions paths and retains the overlapping bypass address");
}

bool liveness_preserves_conditional_values(bool unsupported_definition) {
    Graph graph(2);
    graph.record();
    graph.add(0, "add.u64", {"%cursor", "%generic", "160"});
    graph.add(0, "bra", {"B1"});
    graph.edge(0, 1);
    if (unsupported_definition)
        graph.add(1, "cvt.u32.u64", {"%cursor", "%generic"}, "%choice");
    else
        graph.add(1, "add.u64", {"%cursor", "%generic", "32"}, "%choice");
    const auto* store = graph.add(1, "st.local.b8", {"[%cursor]", "0"});
    const auto* target = graph.add(1, "ld.local.b64", {"%loaded", "[%record]"});
    const auto proof = graph.run(target);
    if (unsupported_definition)
        return expect(proof.complete && !proof.stores.contains(store),
                      "unsupported conditional write poisons its visit while skip retains the old fact");
    return range(proof, store, 32, 160,
                 "conditional definition does not kill the incoming overlapping address");
}

bool incomplete_exploration() {
    bool ok = true;
    for (unsigned variant = 0; variant < 7; ++variant) {
        Graph graph(2);
        graph.record();
        const auto* store = graph.add(0, "st.local.u8", {"[%base+32]", "0"});
        graph.add(0, "bra", {"B1"});
        graph.edge(0, 1);
        const auto* target = graph.add(1, "ld.local.u64", {"%loaded", "[%record]"});
        if (variant == 4) {
            graph.add(1, "bra", {"B1"});
            graph.edge(1, 1);
        }
        if (variant == 5) {
            // The unknown condition permits arbitrarily many identical states
            // before leaving for the load. No constant trip count is invented.
            graph.blocks[0].instructions.pop_back();
            graph.add(0, "bra", {"B0"}, "%unknown");
            graph.blocks[0].successors = {0, 1};
        }
        detail::LocalMemoryRangeLimits limits;
        if (variant == 0)
            limits.states = 1;
        if (variant == 1)
            limits.operations = 1;
        if (variant == 2)
            limits.known_bytes_per_state = 0;
        if (variant == 3)
            limits.register_facts_per_state = 0;
        if (variant == 6)
            limits.retained_facts = 0;
        const auto proof = graph.run(target, limits);
        ok &= expect(!proof.complete && proof.stores.empty() && !proof.stores.contains(store),
                     "incomplete exploration discards every earlier store observation (" +
                         std::to_string(variant) + "): " + proof.reason);
        const char* resources[] = {"states",
                                   "operations",
                                   "known bytes per state",
                                   "register facts per state",
                                   "load belongs to a CFG cycle",
                                   "repeated prefix state",
                                   "retained facts"};
        ok &= expect(proof.reason.find(resources[variant]) != std::string::npos,
                     "failure names its actual exhausted resource or unsupported loop: " + proof.reason);
    }
    return ok;
}
} // namespace

int main() {
    bool ok = initialized_pointer_loop(1, 10);
    ok &= initialized_pointer_loop(4, 12);
    ok &= two_phase_pointer_loop();
    ok &= unknown_branch_union(false);
    ok &= unknown_branch_union(true);
    ok &= overwritten_scalar_record(false, false);
    ok &= overwritten_scalar_record(true, false);
    ok &= overwritten_scalar_record(true, true);
    ok &= calls_invalidate_memory(false);
    ok &= calls_invalidate_memory(true);
    ok &= overwritten_register_and_boundaries();
    ok &= independent_32bit_loop_counter();
    ok &= width_and_overflow_boundaries();
    ok &= dead_register_facts_are_pruned();
    ok &= liveness_preserves_bypass_values();
    ok &= liveness_preserves_conditional_values(false);
    ok &= liveness_preserves_conditional_values(true);
    ok &= incomplete_exploration();
    ok &= scalar_shifts();
    return ok ? 0 : 1;
}
