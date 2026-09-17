#include "cumetal/metal/lower_to_msl.h"
#include "ptx_address_demands.h"

#include <deque>
#include <iostream>
#include <string>

namespace {
using namespace cumetal::ir;
namespace detail = cumetal::ir::detail;

bool expect(bool condition, const std::string& message) {
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

struct Graph {
    std::deque<detail::Instruction> storage;
    std::vector<detail::RawBlock> blocks;
    std::vector<std::unordered_map<std::string, ValueId>> incoming, outgoing;
    std::vector<std::map<std::string, ValueId>> arguments;
    std::unordered_map<const detail::Instruction*, std::vector<ValueId>> results;
    std::unordered_map<const detail::Instruction*, std::vector<std::string>> destinations;

    explicit Graph(std::size_t count) : blocks(count), incoming(count), outgoing(count), arguments(count) {}

    const detail::Instruction* add(std::size_t block, std::string opcode, std::vector<std::string> operands,
                                   std::vector<ValueId> values = {}, std::string predicate = {}) {
        storage.push_back({std::move(predicate), std::move(opcode), std::move(operands), 1, true});
        const auto* instruction = &storage.back();
        blocks[block].instructions.push_back(instruction);
        results.emplace(instruction, std::move(values));
        destinations.emplace(instruction, detail::destination_registers(*instruction));
        return instruction;
    }

    void edge(std::size_t from, std::size_t to) {
        blocks[from].successors.push_back(to);
        blocks[to].predecessors.push_back(from);
    }

    detail::AddressDemandResult run(std::size_t budget = 1'000'000) const {
        return detail::compute_address_demands(
            blocks, incoming, outgoing, arguments, results,
            [&](const detail::Instruction& instruction) { return destinations.at(&instruction); }, {budget});
    }
};

bool copies_and_loop_joins() {
    Graph graph(3);
    graph.incoming[0] = {{"%cell", 1}};
    const auto* load = graph.add(0, "ld.local.u64", {"%loaded", "[%cell]"}, {2});
    graph.add(0, "mov.b64", {"%snapshot", "%loaded"}, {3});
    graph.add(0, "mov.b64", {"%loaded", "0"}, {4});
    graph.outgoing[0] = {{"%loop", 3}};
    graph.edge(0, 1);
    graph.edge(1, 1);
    graph.edge(1, 2);
    graph.arguments[1] = {{"%loop", 5}};
    graph.incoming[1] = {{"%loop", 5}};
    graph.add(1, "mov.u64", {"%copy", "%loop"}, {6});
    graph.add(1, "cvta.to.global.u64", {"%loop", "%copy"}, {7});
    graph.outgoing[1] = {{"%loop", 7}};
    graph.incoming[2] = {{"%address", 7}};
    graph.add(2, "st.global.u32", {"[%address+4]", "11"});
    const auto proof = graph.run();
    bool ok = expect(proof.complete, "copy/loop closure completes: " + proof.reason);
    ok &= expect(proof.values == std::unordered_set<ValueId>{1, 2, 3, 5, 6, 7},
                 "demand follows the captured source and all copy-cycle inputs, not a later overwrite");
    ok &= expect(proof.indexed_joins == 1 && proof.expanded_joins == 1 && proof.join_edges == 2,
                 "loop join expands each incoming edge once");
    ok &= expect(proof.loads.size() == 1 && proof.loads[0].instruction == load && proof.loads[0].address == 1,
                 "load candidate retains its original memory address");
    return ok;
}

bool sparse_environment_and_helper_candidates() {
    Graph graph(1);
    graph.incoming[0] = {{"%reused", 1}, {"%cell", 2}, {"%unrelated", 3}};
    const auto* self_load = graph.add(0, "ld.u64", {"%reused", "[%reused]"}, {4});
    graph.add(0, "st.u32", {"[%reused]", "0"});
    // This load has no direct address consumer in the caller. A helper's type
    // recovery can independently classify it as a pointer; it must be returned
    // for the same reaching-store validation as a directly demanded load.
    const auto* helper_load = graph.add(0, "ld.local.u64", {"%helper_argument", "[%cell]"}, {5});
    graph.add(0, "st.param.b64", {"[argument_slot]", "%helper_argument"});
    const auto* call = graph.add(0, "call.uni", {"(result_slot)", "reader", "(argument_slot)"}, {6});
    graph.destinations[call] = {"return-slot:result_slot"};
    graph.add(0, "ld.param.u64", {"%returned", "[result_slot]"}, {7});
    const auto* direct = graph.add(0, "ld.local.u64", {"%symbol_load", "[depot+16]"}, {8});
    const auto* vector = graph.add(0, "ld.local.v2.u64", {"{%first,%second}", "[%cell]"}, {9, 10});
    const auto proof = graph.run();
    bool ok = expect(proof.complete, "sparse environment and helper candidates: " + proof.reason);
    ok &= expect(proof.values == std::unordered_set<ValueId>{1, 2, 4},
                 "self-updating load reads old address; data/parameter operands are not demands");
    ok &= expect(proof.loads.size() == 4 && proof.loads[0].instruction == self_load &&
                     proof.loads[0].address == 1 && proof.loads[1].instruction == helper_load &&
                     proof.loads[1].address == 2 && proof.loads[2].instruction == direct &&
                     !proof.loads[2].address && proof.loads[3].instruction == vector,
                 "retain independent helper, symbol and vector candidates while excluding ld.param");
    return ok;
}

bool exact_copy_boundary() {
    Graph graph(1);
    graph.incoming[0] = {{"%source", 1}, {"%other", 2}, {"%guard", 3}};
    graph.add(0, "mov.b32", {"%narrow", "%source"}, {4});
    graph.add(0, "mov.b64", {"%predicated", "%source"}, {5}, "%guard");
    graph.add(0, "mov.b64", {"%packed", "{%source,%other}"}, {6});
    graph.add(0, "add.u64", {"%arithmetic", "%source", "4"}, {7});
    graph.add(0, "mov.b64", {"%literal", "1"}, {8});
    for (const auto* name : {"%narrow", "%predicated", "%packed", "%arithmetic", "%literal"})
        graph.add(0, "st.u8", {"[" + std::string(name) + "]", "0"});
    const auto proof = graph.run();
    return expect(proof.complete && proof.values == std::unordered_set<ValueId>{4, 5, 6, 7, 8} &&
                      proof.copy_edges == 0,
                  "unsupported transfers retain their own demands without inventing source provenance");
}

bool unused_scalar_joins_are_not_expanded() {
    constexpr std::size_t predecessors = 128;
    constexpr std::size_t scalar_joins = 128;
    Graph graph(predecessors + 2);
    const auto join = predecessors + 1;
    for (std::size_t b = 1; b <= predecessors; ++b) {
        // A chain of conditional predecessors can either reach the join or
        // continue. All join inputs have definitions, including scalar inputs.
        graph.edge(b - 1, b);
        graph.edge(b, join);
        graph.outgoing[b]["%address"] = 1;
        for (std::size_t i = 0; i < scalar_joins; ++i)
            graph.outgoing[b]["%scalar" + std::to_string(i)] = static_cast<ValueId>(100 + i);
    }
    graph.arguments[join]["%address"] = 2;
    graph.incoming[join]["%address"] = 2;
    for (std::size_t i = 0; i < scalar_joins; ++i) {
        const auto name = "%scalar" + std::to_string(i);
        graph.arguments[join][name] = static_cast<ValueId>(1000 + i);
        graph.incoming[join][name] = static_cast<ValueId>(1000 + i);
    }
    graph.add(join, "st.u32", {"[%address]", "0"});
    // The scalar joins are live and observable as data, but never addresses.
    for (std::size_t i = 0; i < scalar_joins; ++i)
        graph.add(join, "st.u32",
                  {"[%address+" + std::to_string(4 * (i + 1)) + "]", "%scalar" + std::to_string(i)});
    const auto proof = graph.run(1600);
    bool ok = expect(proof.complete, "bounded lazy join expansion: " + proof.reason);
    ok &= expect(proof.indexed_joins == scalar_joins + 1 && proof.expanded_joins == 1 &&
                     proof.join_edges == predecessors && proof.values == std::unordered_set<ValueId>{1, 2},
                 "only the demanded join expands, not the 16384 scalar incoming edges");
    // Indexing itself remains bounded even when no address reaches a join.
    const auto exhausted = graph.run(graph.blocks.size());
    ok &= expect(!exhausted.complete && exhausted.budget_exhausted && exhausted.values.empty() &&
                     exhausted.loads.empty() && exhausted.reason.find("join index") != std::string::npos,
                 "unused scalar joins still consume bounded indexing work");
    return ok;
}

bool every_predecessor_and_transactional_limits() {
    Graph graph(4);
    graph.edge(0, 3);
    graph.edge(1, 3);
    graph.edge(2, 3);
    graph.outgoing[0]["%joined"] = 1;
    graph.outgoing[1]["%joined"] = 2;
    graph.outgoing[2]["%joined"] = 3;
    graph.arguments[3]["%joined"] = 4;
    graph.incoming[3] = {{"%joined", 4}, {"%cell", 5}};
    graph.add(3, "ld.local.u64", {"%unused_load", "[%cell]"}, {6});
    graph.add(3, "mov.s64", {"%copy", "%joined"}, {7});
    graph.add(3, "st.u32", {"[%copy]", "0"});
    const auto complete = graph.run();
    bool ok = expect(complete.complete && complete.join_edges == 3 &&
                         complete.values == std::unordered_set<ValueId>{1, 2, 3, 4, 5, 7},
                     "every distinct incoming value remains demanded");
    const auto original_incoming = graph.incoming;
    const auto original_outgoing = graph.outgoing;
    const auto original_arguments = graph.arguments;
    const auto original_results = graph.results;
    for (std::size_t limit = 0; limit < complete.work; ++limit) {
        const auto partial = graph.run(limit);
        ok &= expect(!partial.complete && partial.budget_exhausted && partial.work <= limit &&
                         partial.values.empty() && partial.loads.empty(),
                     "budget " + std::to_string(limit) + " publishes no partial demand or load proof");
    }
    ok &= expect(graph.run(complete.work).complete, "exact work boundary succeeds");
    ok &= expect(graph.incoming == original_incoming && graph.outgoing == original_outgoing &&
                     graph.arguments == original_arguments && graph.results == original_results,
                 "all SSA inputs remain unchanged after success and exhaustion");
    graph.outgoing[2].erase("%joined");
    const auto invalid = graph.run();
    ok &= expect(!invalid.complete && !invalid.budget_exhausted && invalid.values.empty() &&
                     invalid.loads.empty() &&
                     invalid.reason.find("missing incoming definition") != std::string::npos,
                 "a missing last predecessor cannot be replaced with the other values");
    return ok;
}

std::string memory_fixture(bool helper, const std::string& stored) {
    std::string prefix = R"ptx(
.version 8.8
.target sm_89
.address_size 64
.func reader(.param .b64 pointer, .param .b64 output) {
  .reg .b64 %address, %out;
  .reg .b32 %value;
  ld.param.b64 %address, [pointer];
  ld.param.b64 %out, [output];
  ld.u32 %value, [%address];
  st.global.u32 [%out], %value;
  ret;
}
.visible .entry demand_probe(.param .u64 .ptr .global input,
                            .param .u64 .ptr .global output,
                            .param .u32 selector) {
  .local .align 16 .b8 depot[64];
  .reg .b64 %base, %odd, %input, %output, %loaded, %joined;
  .reg .b32 %selector, %value;
  .reg .pred %choose;
  .param .b64 arg_pointer;
  .param .b64 arg_output;
  ld.param.u64 %input, [input];
  ld.param.u64 %output, [output];
  ld.param.u32 %selector, [selector];
  mov.b64 %base, depot;
  st.local.u32 [%base], 23;
  or.b64 %odd, %base, 1;
  st.local.u8 [%odd], 7;
)ptx";
    std::string body = "st.local.u64 [%base+32], " + stored + ";\n";
    body += "ld.local.u64 %loaded, [%base+32];\n";
    if (helper) {
        // The existing helper recovery attaches pointer-load evidence only in
        // ordinary functions. A kernel-only call would test a separate missing
        // demand bridge, not preservation of this established proof path.
        const std::string declaration = ".visible .entry demand_probe(";
        prefix.replace(prefix.find(declaration), declaration.size(), ".func caller(");
        body += R"ptx(
  st.param.b64 [arg_pointer], %loaded;
  st.param.b64 [arg_output], %output;
  call.uni reader, (arg_pointer, arg_output);
)ptx";
    } else {
        body += R"ptx(
  setp.eq.u32 %choose, %selector, 0;
  @%choose bra ALTERNATE;
  mov.b64 %joined, %loaded;
  bra JOIN;
ALTERNATE:
  mov.b64 %joined, %input;
JOIN:
  ld.global.u32 %value, [%joined];
  st.global.u32 [%output], %value;
)ptx";
    }
    std::string source = prefix + body + "ret;\n}\n";
    if (helper) {
        source += R"ptx(
.visible .entry demand_probe(.param .u64 .ptr .global input,
                            .param .u64 .ptr .global output,
                            .param .u32 selector) {
  .reg .b64 %input, %output;
  .reg .b32 %selector;
  .param .b64 call_input;
  .param .b64 call_output;
  .param .b32 call_selector;
  ld.param.u64 %input, [input];
  ld.param.u64 %output, [output];
  ld.param.u32 %selector, [selector];
  st.param.b64 [call_input], %input;
  st.param.b64 [call_output], %output;
  st.param.b32 [call_selector], %selector;
  call.uni caller, (call_input, call_output, call_selector);
  ret;
}
)ptx";
    }
    return source;
}

bool importer_memory_proof_controls() {
    bool ok = true;
    for (bool helper : {false, true}) {
        const std::string label = helper ? "helper-recovered load" : "copy/join address load";
        // A caller-owned allocation supplies a concrete private address space
        // before interprocedural specialization of the helper's generic inputs.
        const auto valid =
            cumetal::metal::compile_ptx_to_msl(memory_fixture(helper, helper ? "%base" : "%input"));
        ok &= expect(valid.ok, label + " accepts proven stored pointer: " + valid.error);
        if (valid.ok)
            ok &= expect(verify(valid.gpu_ir).ok && verify(valid.metal_ir).ok,
                         label + " verifies both IR stages");
        const auto invalid = cumetal::metal::compile_ptx_to_msl(memory_fixture(helper, "1"));
        ok &= expect(!invalid.ok && (invalid.error.find("pointer") != std::string::npos ||
                                     invalid.error.find("incoming type") != std::string::npos),
                     label + " rejects integer storage even when later used as an address: " + invalid.error);
    }
    return ok;
}
} // namespace

int main() {
    bool ok = copies_and_loop_joins();
    ok &= sparse_environment_and_helper_candidates();
    ok &= exact_copy_boundary();
    ok &= unused_scalar_joins_are_not_expanded();
    ok &= every_predecessor_and_transactional_limits();
    ok &= importer_memory_proof_controls();
    if (!ok)
        return 1;
    std::cout << "PTX address-demand tests passed\n";
    return 0;
}
