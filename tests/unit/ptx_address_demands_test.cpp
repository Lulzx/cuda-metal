#include "cumetal/metal/lower_to_msl.h"
#include "ptx_address_demands.h"

#include <deque>
#include <iostream>
#include <map>
#include <string>

namespace {
using namespace cumetal::ir;
namespace detail = cumetal::ir::detail;

bool expect(bool condition, const std::string& message) {
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

// Model the existing producer's opcode classification. Non-memory operations
// never enter the collector, and parameter loads are not pointer-cell loads.
bool observe_memory(detail::AddressDemandCollector& collector, const detail::Instruction& instruction,
                    const std::vector<ValueId>& values,
                    const std::unordered_map<std::string, ValueId>& environment) {
    const auto root = detail::root_opcode(instruction.opcode);
    if ((root == "ld" && !instruction.opcode.starts_with("ld.param")) ||
        root == "atom" || root == "st" || root == "red" || root == "cvta")
        return collector.observe_memory(instruction, values, environment,
                                        root == "st" || root == "red" ? 0 : 1, root == "ld");
    return true;
}

struct Graph {
    std::deque<detail::Instruction> storage;
    std::vector<detail::RawBlock> blocks;
    std::vector<std::unordered_map<std::string, ValueId>> incoming, outgoing;
    std::vector<std::map<std::string, ValueId>> arguments;
    std::unordered_map<const detail::Instruction*, std::vector<ValueId>> results;
    std::unordered_map<const detail::Instruction*, std::vector<std::string>> destinations;
    std::unordered_set<ValueId> concrete_pointers;

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
        detail::AddressDemandCollector collector({budget});
        // This fixture supplies the tagged source index already available in
        // the production type solver. Graph validation belongs to that caller.
        struct Sources {
            std::vector<ValueId> inputs;
            detail::AddressDemandKind kind = detail::AddressDemandKind::kOther;
        };
        std::unordered_map<ValueId, Sources> sources;
        std::size_t joins = 0;
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            for (const auto& [name, value] : arguments[b]) {
                auto& entry = sources[value];
                entry.kind = detail::AddressDemandKind::kJoin;
                ++joins;
                for (const auto predecessor : blocks[b].predecessors)
                    entry.inputs.push_back(outgoing.at(predecessor).at(name));
            }
        }
        const auto lookup = [&](ValueId value) -> detail::AddressDemandSources {
            const auto found = sources.find(value);
            if (found == sources.end()) return {};
            return {found->second.kind, &found->second.inputs};
        };
        const auto is_pointer = [&](ValueId value) { return concrete_pointers.contains(value); };
        // The production caller already maintains this pre-write environment
        // while solving types. Feed precisely the same reaching SSA values.
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            auto environment = incoming[b];
            for (const auto* instruction : blocks[b].instructions) {
                const auto& values = results.at(instruction);
                if (!observe_memory(collector, *instruction, values, environment))
                    return collector.finish(lookup, is_pointer, joins, blocks.size());
                if (values.size() == 1 && instruction->predicate.empty() && instruction->operands.size() == 2 &&
                    (instruction->opcode == "mov.b64" || instruction->opcode == "mov.u64" ||
                     instruction->opcode == "mov.s64" || detail::root_opcode(instruction->opcode) == "cvta") &&
                    instruction->operands[1].find('{') == std::string::npos) {
                    const auto source = environment.find(detail::first_register(instruction->operands[1]));
                    if (source != environment.end())
                        sources[values.front()] = {{source->second}, detail::AddressDemandKind::kCopy};
                }
                const auto& writes = destinations.at(instruction);
                for (std::size_t i = 0; i < writes.size(); ++i) environment[writes[i]] = values[i];
            }
        }
        return collector.finish(lookup, is_pointer, joins, blocks.size());
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
    ok &= expect(proof.expanded_joins == 1 && proof.join_edges == 2 && proof.copy_edges == 3,
                 "loop join and exact copies expand each incoming edge once");
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
    const auto proof = graph.run(652);
    bool ok = expect(proof.complete, "bounded lazy join expansion: " + proof.reason);
    ok &= expect(proof.work <= 652 && proof.expanded_joins == 1 &&
                     proof.join_edges == predecessors && proof.values == std::unordered_set<ValueId>{1, 2},
                 "only the demanded join expands, not the 16384 scalar incoming edges");
    return ok;
}

bool shared_scan_scales_with_scalar_definitions() {
    // The retained RSA-PSS kernel has this many raw instructions. A separate
    // scan or collector call per scalar definition is unnecessary: the type
    // solver already classifies each opcode and indexes its sources.
    constexpr std::size_t scalars = 308214;
    detail::AddressDemandCollector collector;
    std::unordered_map<std::string, ValueId> environment = {{"%cell", 1}, {"%scratch", 2}};
    const detail::Instruction load{"", "ld.local.v2.u64", {"{%loaded,%length}", "[%cell]"}, 1, true};
    const detail::Instruction arithmetic{"", "add.u64", {"%scratch", "%scratch", "1"}, 2, true};
    const detail::Instruction store{"", "st.u32", {"[%loaded]", "%scratch"}, 3, true};
    if (!expect(observe_memory(collector, load, {3, 4}, environment), "collect vector load before scalar work"))
        return false;
    environment["%loaded"] = 3;
    environment["%length"] = 4;
    // Model the existing SSA producer supplying fresh definitions. The
    // collector must not retain another index of these unrelated operations.
    for (std::size_t i = 0; i < scalars; ++i) {
        const auto value = static_cast<ValueId>(100 + i);
        if (!observe_memory(collector, arithmetic, {value}, environment))
            return expect(false, "shared scalar scan remains bounded: " + collector.reason());
        environment["%scratch"] = value;
    }
    if (!expect(observe_memory(collector, store, {}, environment), "collect address after scalar work")) return false;
    const auto proof = collector.finish([](ValueId) { return detail::AddressDemandSources{}; },
                                        [](ValueId) { return false; }, 0, 1);
    bool ok = expect(proof.complete && proof.values == std::unordered_set<ValueId>{1, 3} &&
                         proof.loads.size() == 1 && proof.loads.front().instruction == &load &&
                         proof.loads.front().address == 1,
                     "large unrelated scalar stream preserves exact vector/address demands: " + proof.reason);
    ok &= expect(proof.work <= 16 &&
                     proof.copy_edges == 0 && proof.expanded_joins == 0,
                 "scalar definitions add no collector observation, index or closure work");
    return ok;
}

bool inconsistent_indices_discard_partial_results() {
    Graph graph(1);
    graph.incoming[0] = {{"%cell", 1}};
    graph.add(0, "ld.local.u64", {"%loaded", "[%cell]"}, {2});
    detail::AddressDemandCollector collector;
    bool ok = expect(observe_memory(collector, *graph.blocks[0].instructions[0], {2}, graph.incoming[0]),
                     "seed before missing source lookup");
    const auto invalid = collector.finish({}, [](ValueId) { return false; }, 0, 1);
    ok &= expect(!invalid.complete && invalid.values.empty() && invalid.loads.empty() &&
                     invalid.reason.find("missing source lookup") != std::string::npos,
                 "missing source lookup publishes no collected candidates");
    detail::AddressDemandCollector missing_types;
    ok &= expect(observe_memory(missing_types, *graph.blocks[0].instructions[0], {2}, graph.incoming[0]),
                 "seed before missing validated type lookup");
    const auto unknown_types = missing_types.finish(
        [](ValueId) { return detail::AddressDemandSources{}; }, {}, 0, 1);
    ok &= expect(!unknown_types.complete && unknown_types.values.empty() && unknown_types.loads.empty() &&
                     unknown_types.reason.find("missing validated pointer lookup") != std::string::npos,
                 "absent final type contract publishes no partial proof");
    for (const std::vector<ValueId> inputs : {std::vector<ValueId>{}, std::vector<ValueId>{3, 4}}) {
        detail::AddressDemandCollector bad_copy;
        ok &= expect(observe_memory(bad_copy, *graph.blocks[0].instructions[0], {2}, graph.incoming[0]),
                     "seed before malformed copy source vector");
        const auto invalid_copy = bad_copy.finish(
            [&](ValueId) { return detail::AddressDemandSources{detail::AddressDemandKind::kCopy, &inputs}; },
            [](ValueId) { return false; }, 0, 1);
        ok &= expect(!invalid_copy.complete && !invalid_copy.budget_exhausted && invalid_copy.values.empty() &&
                         invalid_copy.loads.empty() && invalid_copy.reason.find("one source") != std::string::npos,
                     "empty or multiple-source copy publishes no partial demand or load candidates");
    }
    Graph empty_join(1);
    empty_join.incoming[0] = {{"%address", 1}};
    empty_join.arguments[0] = {{"%address", 1}};
    empty_join.add(0, "ld.local.u64", {"%loaded", "[%address]"}, {2});
    const auto empty = empty_join.run();
    ok &= expect(!empty.complete && !empty.budget_exhausted && empty.values.empty() && empty.loads.empty() &&
                     empty.reason.find("no incoming edge") != std::string::npos,
                 "empty demanded join cannot prove an address and publishes no load candidates");
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
    return ok;
}

bool concrete_pointer_joins_end_demand() {
    constexpr std::size_t joins = 4096;
    Graph graph(2);
    graph.edge(0, 1);
    graph.edge(1, 1);
    graph.incoming[1]["%cell"] = 1;
    graph.concrete_pointers.insert(1);
    // Every cyclic forwarding join has a concrete pointer seed and a pointer
    // backedge. The caller has already validated their complete input types.
    for (std::size_t i = 0; i < joins; ++i) {
        const auto name = "%forward" + std::to_string(i);
        const auto value = static_cast<ValueId>(100 + i);
        graph.arguments[1][name] = value;
        graph.incoming[1][name] = value;
        graph.outgoing[0][name] = 1;
        graph.outgoing[1][name] = static_cast<ValueId>(100 + (i + 1) % joins);
        graph.concrete_pointers.insert(value);
    }
    const auto* load = graph.add(1, "ld.local.u64", {"%independent", "[%cell]"}, {5000});
    graph.concrete_pointers.insert(5000);
    graph.add(1, "st.u32", {"[%forward0]", "0"});
    const auto proof = graph.run(16);
    bool ok = expect(proof.complete && proof.pointer_cutoffs == 2 && proof.work <= 16 &&
                         proof.expanded_joins == 0 && proof.values == std::unordered_set<ValueId>{1, 100},
                     "validated pointer forwarding does not expand thousands of redundant joins: " + proof.reason);
    ok &= expect(proof.loads.size() == 1 && proof.loads.front().instruction == load,
                 "independently pointer-typed loads remain candidates without a demanded result");
    for (std::size_t limit = 0; limit < proof.work; ++limit) {
        const auto exhausted = graph.run(limit);
        ok &= expect(!exhausted.complete && exhausted.budget_exhausted &&
                         exhausted.values.empty() && exhausted.loads.empty(),
                     "pointer-cut budget boundary retains no partial proof");
    }
    return ok;
}

bool conversion_source_is_an_independent_demand() {
    Graph graph(1);
    graph.incoming[0] = {{"%cell", 1}};
    graph.concrete_pointers = {1, 4};
    const auto* load = graph.add(0, "ld.local.v2.u64", {"{%payload,%metadata}", "[%cell]"}, {2, 3});
    // Self-overwrite must retain the scalar input, even though the result is a
    // concrete pointer and the following memory consumer stops at that result.
    graph.add(0, "cvta.to.global.u64", {"%payload", "%payload"}, {4});
    graph.add(0, "st.u32", {"[%payload]", "0"});
    const auto proof = graph.run();
    return expect(proof.complete && proof.values == std::unordered_set<ValueId>{1, 2, 4} &&
                      proof.pointer_cutoffs == 2 && proof.loads.size() == 1 &&
                      proof.loads.front().instruction == load && proof.loads.front().address == 1,
                  "conversion seeds its scalar pre-write source independently of its pointer result: " + proof.reason);
}

std::string conversion_fixture(const std::string& space, const std::string& fault = {}, bool recover = true) {
    const std::string target = space == "private" ? "%base" : space == "device" ? "%input" :
        space == "constant" ? "constants" : "shared_data";
    const std::string cast_space = space == "private" ? "local" : space == "threadgroup" ? "shared" :
        space == "constant" ? "const" : "global";
    std::string source = R"ptx(
.version 8.8
.target sm_89
.address_size 64
.const .align 8 .b8 constants[8] = {73, 0, 0, 0, 0, 0, 0, 0};
.visible .entry conversion_probe(.param .u64 .ptr .global input,
                                 .param .u64 .ptr .global output, .param .u32 choice) {
 .local .align 16 .b8 depot[64];
 .shared .align 8 .b8 shared_data[16];
 .reg .b64 %base, %input, %output, %stored, %loaded, %metadata, %converted, %joined, %value;
 .reg .b32 %choice;
 .reg .pred %pick;
 ld.param.u64 %input, [input];
 ld.param.u64 %output, [output];
 ld.param.u32 %choice, [choice];
 setp.ne.u32 %pick, %choice, 0;
 mov.b64 %base, depot;
 st.local.u64 [%base], 73;
)ptx";
    source += "mov.b64 %stored, " + target + ";\n";
    if (fault == "missing") source += "@%pick bra AFTER_STORE;\n";
    source += "st.local.v2.b64 [%base+32], {%stored, 9};\nAFTER_STORE:\n";
    // Name-wide discovery sees a scalar overwrite, but SSA reaching-store
    // validation must recover the earlier pointer stored into the record.
    if (recover) source += "mov.b64 %stored, 1;\n";
    // Generic stores evade textual local-only discovery and must still be
    // rejected by independent validation of the pointer-typed load candidate.
    if (fault == "integer") source += "st.u64 [%base+32], 1;\n";
    if (fault == "partial") source += "st.u8 [%base+39], 0;\n";
    source += "ld.local.v2.b" + std::string(fault == "narrow" ? "32" : "64") +
        " {%loaded, %metadata}, [%base+32];\n";
    source += "cvta.to." + cast_space + ".u64 %converted, %loaded;\n";
    source += R"ptx(
 @%pick bra LEFT;
 mov.u64 %joined, %converted;
 bra JOIN;
LEFT:
 mov.b64 %joined, %converted;
JOIN:
 ld.u64 %value, [%joined];
 st.global.u64 [%output], %value;
 st.global.u64 [%output+8], %metadata;
 ret;
}
)ptx";
    return source;
}

bool conversion_recovery_and_pointer_join_refusals() {
    bool ok = true;
    for (const std::string space : {"private", "device", "constant", "threadgroup"}) {
        const auto recovered = cumetal::metal::compile_ptx_to_msl(conversion_fixture(space));
        ok &= expect(recovered.ok && verify(recovered.gpu_ir).ok && verify(recovered.metal_ir).ok,
                     space + " scalar lane recovers through conversion and concrete pointer join: " + recovered.error);
        for (const std::string fault : {"missing", "integer", "partial", "narrow"}) {
            const auto rejected = cumetal::metal::compile_ptx_to_msl(conversion_fixture(space, fault, false));
            ok &= expect(!rejected.ok && rejected.error.find("pointer memory proof") != std::string::npos,
                         space + " pointer join preserves " + fault + " refusal: " + rejected.error);
        }
    }
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

bool shared_index_excludes_arithmetic_sources() {
    const auto compiled = cumetal::metal::compile_ptx_to_msl(R"ptx(
.version 8.8
.target sm_89
.address_size 64
.visible .entry scalar_offset(.param .u64 .ptr .global input,
                              .param .u64 .ptr .global output) {
 .local .align 16 .b8 depot[16];
 .reg .b64 %cell, %offset, %metadata, %input, %output, %address;
 .reg .b32 %value;
 ld.param.u64 %input, [input];
 ld.param.u64 %output, [output];
 mov.b64 %cell, depot;
 st.local.v2.b64 [%cell], {4, 5};
 ld.local.v2.b64 {%offset, %metadata}, [%cell];
 add.u64 %address, %input, %offset;
 ld.global.u32 %value, [%address];
 st.global.u32 [%output], %value;
 st.global.u64 [%output+8], %metadata;
 ret;
}
)ptx");
    return expect(compiled.ok && verify(compiled.gpu_ir).ok && verify(compiled.metal_ir).ok,
                  "arithmetic source entries do not turn scalar vector lanes into pointer demands: " + compiled.error);
}
} // namespace

int main() {
    bool ok = copies_and_loop_joins();
    ok &= sparse_environment_and_helper_candidates();
    ok &= exact_copy_boundary();
    ok &= unused_scalar_joins_are_not_expanded();
    ok &= shared_scan_scales_with_scalar_definitions();
    ok &= inconsistent_indices_discard_partial_results();
    ok &= every_predecessor_and_transactional_limits();
    ok &= concrete_pointer_joins_end_demand();
    ok &= conversion_source_is_an_independent_demand();
    ok &= conversion_recovery_and_pointer_join_refusals();
    ok &= importer_memory_proof_controls();
    ok &= shared_index_excludes_arithmetic_sources();
    if (!ok)
        return 1;
    std::cout << "PTX address-demand tests passed\n";
    return 0;
}
