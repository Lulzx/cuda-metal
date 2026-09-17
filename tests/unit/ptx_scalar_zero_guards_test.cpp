#include "cumetal/ir/ptx_importer.h"
#include "cumetal/metal/lower_to_msl.h"
#include "ptx_cfg.h"

#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace ir = cumetal::ir;
namespace detail = cumetal::ir::detail;
std::size_t checks = 0;

bool expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

std::string replace(std::string text, const std::string& from, const std::string& to) {
    const auto at = text.find(from);
    if (at == std::string::npos) throw std::runtime_error("missing replacement: " + from);
    text.replace(at, from.size(), to);
    return text;
}

std::string fixture(unsigned width = 64, const std::string& type = "b",
                    const std::string& relation = "eq", bool copy = false) {
    const auto suffix = type + std::to_string(width);
    std::string source = R"ptx(.version 7.0
.target sm_80
.address_size 64
.visible .entry probe(.param .u64 .ptr .global output,
                     .param .u64 seed, .param .u32 choice) {
 .reg .b64 %output, %seed, %payload;
 .reg .b32 %choice, %counter;
 .reg .pred %skip, %absent, %again;
)ptx";
    source += " .reg .b" + std::to_string(width) + " %marker, %zero, %snapshot;\n";
    source += R"ptx( ld.param.u64 %output, [output];
 ld.param.u64 %seed, [seed];
 ld.param.u32 %choice, [choice];
)ptx";
    source += copy ? " mov." + suffix + " %zero, 0;\n mov." + suffix + " %marker, %zero;\n"
                   : " mov." + suffix + " %marker, 0;\n";
    source += R"ptx( setp.eq.u32 %skip, %choice, 0;
 @%skip bra JOIN;
 add.u64 %payload, %seed, 7;
)ptx";
    source += " cvt.u" + std::to_string(width) + ".u64 %marker, %seed;\nJOIN:\n";
    source += " setp." + relation + "." + suffix + " %absent, %marker, 0;\n";
    source += relation == "eq" ? " @%absent bra DONE;\n" : " @!%absent bra DONE;\n";
    source += " st.global.u64 [%output], %payload;\nDONE:\n ret;\n}\n";
    return source;
}

bool accepts(const std::string& source, const std::string& label) {
    const auto imported = ir::import_ptx(source);
    bool ok = expect(imported.ok, label + ": import: " + imported.error);
    if (!imported.ok) return ok;
    ok &= expect(ir::verify(imported.module).ok, label + ": verified SSA");
    bool payload_definition = false, payload_store = false;
    for (const auto& function : imported.module.functions)
        for (const auto& block : function.blocks)
            for (const auto& operation : block.operations) {
                payload_definition |= operation.opcode == ir::OpCode::kAdd;
                payload_store |= operation.opcode == ir::OpCode::kStore &&
                    operation.operands.size() >= 2 &&
                    operation.operands.back().kind == ir::OperandKind::kValue;
            }
    ok &= expect(payload_definition && payload_store, label + ": real payload calculation/store retained");
    const auto compiled = cumetal::metal::compile_ptx_to_msl(source);
    ok &= expect(compiled.ok, label + ": strict MSL lowering: " + compiled.error);
    return ok;
}

bool rejects(const std::string& source, const std::string& label) {
    const auto imported = ir::import_ptx(source);
    return expect(!imported.ok && imported.error.find("PTX register '") != std::string::npos &&
                      (imported.error.find(" is undefined ") != std::string::npos ||
                       imported.error.find(" is used before definition ") != std::string::npos),
                  label + ": strict undefined-value rejection: " + imported.error);
}

struct RawFixture {
    ir::Builder builder;
    std::deque<detail::Instruction> instructions;
    std::vector<detail::RawBlock> blocks;
    cumetal::ptx::EntryFunction function;

    explicit RawFixture(std::size_t count = 4) : blocks(count) {
        for (std::size_t i = 0; i < count; ++i) {
            blocks[i].id = builder.next_block();
            blocks[i].name = "block_" + std::to_string(i);
        }
        function.register_declarations = {{"%marker", "b64", true}, {"%zero", "b64", true}};
        function.register_ranges = {{"%p", "pred", 4, true}};
    }

    const detail::Instruction* add(std::string opcode, std::vector<std::string> operands,
                                   std::string predicate = {}) {
        detail::Instruction instruction;
        instruction.opcode = std::move(opcode);
        instruction.operands = std::move(operands);
        instruction.predicate = std::move(predicate);
        instruction.supported = true;
        instructions.push_back(std::move(instruction));
        return &instructions.back();
    }

    void basic() {
        blocks[0].instructions = {add("mov.b64", {"%marker", "0"}), add("bra", {blocks[1].name})};
        blocks[0].successors = {1};
        blocks[1].instructions = {add("setp.eq.b64", {"%p0", "%marker", "0"}),
                                  add("bra", {blocks[2].name}, "%p0")};
        blocks[1].successors = {2, 3};
        blocks[2].instructions = {add("ret", {})};
        blocks[3].instructions = {add("ret", {})};
    }

    bool specializes(bool with_function = true) {
        detail::simplify_guarded_paths(blocks, builder, instructions,
                                      with_function ? &function : nullptr);
        return blocks[0].successors[0] != 1;
    }
};

bool positive_and_negative_imports() {
    bool ok = true;
    for (const auto width : {16u, 32u, 64u})
        for (const std::string type : {"b", "u", "s"})
            for (const std::string relation : {"eq", "ne"})
                ok &= accepts(fixture(width, type, relation, true),
                    "copied zero " + type + std::to_string(width) + " " + relation);
    const auto base = fixture();
    ok &= accepts(base, "renamed direct dynamic marker");
    ok &= accepts(replace(base, "%absent, %marker, 0", "%absent, 0, %marker"), "commuted zero comparison");
    ok &= accepts(replace(base, "cvt.u64.u64 %marker, %seed;", "mov.b64 %marker, 1;"), "literal present marker");
    auto snapshot = fixture(64, "b", "eq", true);
    snapshot = replace(snapshot, "mov.b64 %marker, %zero;",
                       "mov.b64 %marker, %zero;\n mov.b64 %zero, 1;");
    ok &= accepts(snapshot, "copied snapshot survives source overwrite");
    auto joined_copy = replace(base, "JOIN:\n", "JOIN:\n mov.b64 %snapshot, %marker;\n");
    joined_copy = replace(joined_copy, "%absent, %marker, 0", "%absent, %snapshot, 0");
    ok &= accepts(joined_copy, "edge-local copy after join");
    auto two_skips = replace(base, "@%skip bra JOIN;", "@%skip bra JOIN;\n"
        " setp.eq.u32 %again, %choice, 1;\n @%again bra JOIN;");
    ok &= accepts(two_skips, "two skipped-definition edges");
    auto reordered = replace(base, " add.u64 %payload, %seed, 7;\n cvt.u64.u64 %marker, %seed;\n",
                             " bra DEFINE;\n");
    reordered = replace(reordered, "DONE:\n ret;", "DONE:\n ret;\nDEFINE:\n"
        " add.u64 %payload, %seed, 7;\n cvt.u64.u64 %marker, %seed;\n bra JOIN;");
    ok &= accepts(reordered, "reordered defining block");
    auto loop = replace(base, "@%skip bra JOIN;", "@%skip bra WAIT;");
    loop = replace(loop, "JOIN:\n", " bra JOIN;\nWAIT:\n mov.u32 %counter, 0;\nLOOP:\n"
        " add.u32 %counter, %counter, 1;\n setp.lt.u32 %again, %counter, 3;\n"
        " @%again bra LOOP;\nJOIN:\n");
    ok &= accepts(loop, "zero survives a bounded loop");
    auto called = replace(base, ".visible .entry", ".func helper() { .reg .b64 %marker;"
        " mov.b64 %marker, 1; ret; }\n.visible .entry");
    called = replace(called, "JOIN:\n", "JOIN:\n call.uni helper, ();\n");
    ok &= accepts(called, "direct callee same spelling is independent");
    auto private_marker = replace(base, " .reg .b64 %output", " .local .align 16 .b8 depot[16];\n .reg .b64 %base;\n .reg .b64 %output");
    private_marker = replace(private_marker, "cvt.u64.u64 %marker, %seed;",
        "mov.b64 %base, depot;\n cvta.local.u64 %marker, %base;");
    private_marker = replace(private_marker, "JOIN:\n", "JOIN:\n cvta.to.local.u64 %base, %marker;\n");
    ok &= accepts(private_marker, "private marker with intervening cvta");

    ok &= rejects(replace(base, " @%absent bra DONE;\n", ""), "unguarded payload");
    ok &= rejects(replace(base, " mov.b64 %marker, 0;\n", ""), "missing zero initialization");
    for (const std::string overwrite : {"mov.b64 %marker, 1;", "mov.b64 %marker, %seed;",
                                       "@%skip mov.b64 %marker, 1;"})
        ok &= rejects(replace(base, "JOIN:\n", "JOIN:\n " + overwrite + "\n"), "marker overwrite " + overwrite);
    ok &= rejects(replace(base, " @%absent bra DONE;", " @%skip mov.pred %absent, 0;\n @%absent bra DONE;"),
                  "predicated guard overwrite");
    ok &= rejects(replace(base, "JOIN:\n", "JOIN:\n st.global.u64 [%output+8], %payload;\n"),
                  "payload escapes before guard");
    ok &= rejects(replace(loop, " add.u32 %counter, %counter, 1;", " mov.b64 %marker, 1;\n add.u32 %counter, %counter, 1;"),
                  "changed marker on loop path/backedge");
    return ok;
}

bool raw_contracts() {
    bool ok = true;
    for (unsigned variant = 0; variant < 17; ++variant) {
        RawFixture test;
        test.basic();
        bool proven = variant == 0 || variant == 12 || variant == 13;
        if (variant == 1) test.function.register_declarations.clear();
        if (variant == 2) test.function.register_declarations[0].function_scope = false;
        if (variant == 3) test.function.register_declarations[0].type = "b32";
        if (variant == 4) test.function.register_declarations.push_back({"%marker", "b64", false});
        if (variant == 5) test.function.register_declarations[0].type = "b64garbage";
        if (variant == 6) test.blocks[0].instructions[0] = test.add("mov.b32", {"%marker", "0"});
        if (variant == 7) test.blocks[1].instructions[0] = test.add("setp.eq.b32", {"%p0", "%marker", "0"});
        if (variant == 8) test.blocks[1].instructions.insert(test.blocks[1].instructions.begin(),
            test.add("mov.b64", {"%marker", "{%zero, 1}"}));
        if (variant == 9) test.blocks[1].instructions.insert(test.blocks[1].instructions.begin(),
            test.add("call.uni", {"(%marker)", "helper", "()"}));
        if (variant == 10) test.blocks[1].instructions.insert(test.blocks[1].instructions.begin(),
            test.add("call.uni", {"%callee", "()"}));
        if (variant == 11) test.blocks[1].instructions.insert(test.blocks[1].instructions.begin(),
            test.add("call.uni", {"helper"}));
        if (variant == 12) test.blocks[1].instructions.insert(test.blocks[1].instructions.begin(),
            test.add("call.uni", {"helper", "()"}));
        if (variant == 13) test.blocks[1].instructions.insert(test.blocks[1].instructions.begin(),
            test.add("call.uni", {"(%zero)", "helper", "()"}));
        if (variant == 14) test.blocks[1].instructions.insert(test.blocks[1].instructions.begin(),
            test.add("call.uni", {"(%marker)", "helper", "()"}, "%p1"));
        if (variant == 16) test.blocks[1].instructions.insert(test.blocks[1].instructions.begin(),
            test.add("mov.b32", {"%marker", "1"}));
        ok &= expect(test.specializes(variant != 15) == proven,
                     "declaration/width/tuple/call contract " + std::to_string(variant));
    }
    for (unsigned variant = 0; variant < 4; ++variant) {
        RawFixture test;
        test.basic();
        test.function.register_declarations.clear();
        const std::string name = variant == 2 ? "%tag00" : "%tag0";
        test.function.register_ranges.push_back({"%tag", "b64",
            variant == 1 ? std::numeric_limits<std::size_t>::max() : 1, variant != 3});
        test.blocks[0].instructions[0] = test.add("mov.b64", {name, "0"});
        test.blocks[1].instructions[0] = test.add("setp.eq.b64", {"%p0", name, "0"});
        ok &= expect(test.specializes() == (variant < 2), "compact range binding " + std::to_string(variant));
    }
    RawFixture effects;
    effects.basic();
    effects.blocks[1].instructions.insert(effects.blocks[1].instructions.begin(),
        effects.add("st.global.u64", {"[%output]", "17"}));
    ok &= expect(effects.specializes(), "zero edge specializes across observable store");
    const auto clone = effects.blocks[0].successors[0];
    ok &= expect(effects.blocks[clone].instructions.front()->opcode == "st.global.u64",
                 "specialization retains observable store in clone");
    return ok;
}

bool conditional_select_payloads() {
    bool ok = true;
    for (const unsigned width : {32u, 64u}) {
        auto source = fixture();
        source = replace(source, " .reg .b32 %choice, %counter;",
            " .reg .b32 %choice, %counter;\n .reg .b" + std::to_string(width) +
            " %narrow, %selected;\n .reg .b64 %selected_result;");
        source = replace(source, " cvt.u64.u64 %marker, %seed;",
            " cvt.u" + std::to_string(width) + ".u64 %narrow, %payload;\n cvt.u64.u64 %marker, %seed;");
        const std::string select = " selp.b" + std::to_string(width) +
            " %selected, 8, %narrow, %absent;\n";
        source = replace(source, " @%absent bra DONE;", select +
            " cvt.u64.u" + std::to_string(width) + " %selected_result, %selected;\n"
            " st.global.u64 [%output+8], %selected_result;\n @%absent bra DONE;");
        source = replace(source, "JOIN:\n", "JOIN:\n st.global.u64 [%output+16], 77;\n");
        ok &= accepts(source, "known zero predicate selects default b" + std::to_string(width));
        auto terminal = replace(source, " @%absent bra DONE;\n st.global.u64 [%output], %payload;", "");
        ok &= accepts(terminal, "known select in terminal block b" + std::to_string(width));
        const auto compiled = cumetal::metal::compile_ptx_to_msl(terminal);
        bool observable_store = false;
        for (const auto& function : compiled.gpu_ir.functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations)
                    observable_store |= operation.opcode == ir::OpCode::kStore &&
                        !operation.operands.empty() &&
                        operation.operands.back().kind == ir::OperandKind::kImmediate &&
                        operation.operands.back().text == "77";
        ok &= expect(compiled.ok && observable_store,
                     "terminal select retains the observable preceding store");
        ok &= rejects(replace(source, select,
            " st.global.u64 [%output+24], %payload;\n" + select),
            "select proof cannot erase an observable undefined payload");
        ok &= rejects(replace(source, select,
            " selp.b" + std::to_string(width) + " %selected, %narrow, 8, %absent;\n"),
            "swapped select arms expose the absent payload");
        ok &= rejects(replace(source, select, " mov.pred %absent, 0;\n" + select),
            "select predicate overwritten before use");
        ok &= rejects(replace(source, " setp.eq.b64 %absent, %marker, 0;",
            " mov.b64 %marker, 1;\n setp.eq.b64 %absent, %marker, 0;"),
            "select marker overwritten before comparison");
        auto called = replace(source, ".visible .entry", ".func helper() { ret; }\n.visible .entry");
        called = replace(called, select, " call.uni helper, ();\n" + select);
        ok &= accepts(called, "caller-local predicate survives direct helper before select");
    }
    const std::string repeated = R"ptx(.version 7.1
.target sm_80
.address_size 64
.visible .entry probe(.param .u64 .ptr .global output,
                     .param .u64 left, .param .u64 right) {
 .reg .b64 %output, %left, %right, %payload, %selected;
 .reg .pred %first, %repeated;
 ld.param.u64 %output, [output];
 ld.param.u64 %left, [left];
 ld.param.u64 %right, [right];
 setp.eq.u64 %first, %left, %right;
 @%first bra JOIN;
 add.u64 %payload, %left, 11;
 bra JOIN;
JOIN:
 setp.eq.u64 %repeated, %left, %right;
 selp.b64 %selected, 8, %payload, %repeated;
 st.global.u64 [%output], %selected;
 ret;
}
)ptx";
    ok &= accepts(repeated, "terminal select consumes existing repeated-comparison edge proof");
    ok &= accepts(replace(repeated, " st.global.u64 [%output], %selected;",
        " bra STORE;\nSTORE:\n st.global.u64 [%output], %selected;"),
        "prefix select retains the repeated-comparison proof when cloned");
    ok &= rejects(replace(repeated, "selp.b64 %selected, 8, %payload, %repeated;",
        "selp.b64 %selected, %payload, 8, %repeated;"),
        "repeated-comparison select cannot discard a reachable undefined arm");
    ok &= rejects(replace(repeated, " selp.b64 %selected, 8, %payload, %repeated;",
        " setp.eq.u64 %repeated, %left, 0;\n selp.b64 %selected, 8, %payload, %repeated;"),
        "overwritten repeated predicate cannot use the earlier comparison proof");
    return ok;
}

bool bitwise_zero_markers() {
    bool ok = true;
    for (const unsigned width : {16U, 32U, 64U}) {
        const auto suffix = "b" + std::to_string(width);
        const auto compare = " setp.eq." + suffix + " %absent, %marker, 0;";
        for (const bool commute : {false, true}) {
            for (const auto* operation : {"and", "or"}) {
                const auto other = std::string(operation) == "and" ? "255" : "0";
                const auto operands = commute ? other + std::string(", %marker") : "%marker, " + std::string(other);
                const auto transfer = " " + std::string(operation) + "." + suffix + " %snapshot, " + operands + ";\n";
                const auto guard = " setp.eq." + suffix + " %absent, %snapshot, 0;";
                const auto label = std::string(operation) + " zero marker " + std::to_string(width) + (commute ? " commuted" : "");
                const auto source = replace(fixture(width), compare, transfer + guard);
                ok &= accepts(source, label);
                ok &= rejects(replace(source, guard, " mov." + suffix + " %snapshot, 1;\n" + guard),
                              label + " overwritten result");
                ok &= rejects(replace(source, " mov." + suffix + " %marker, 0;", " mov." + suffix + " %marker, 1;"),
                              label + " nonzero absent marker");
            }
        }
        const auto guard = " setp.eq." + suffix + " %absent, %snapshot, 0;";
        ok &= accepts(replace(fixture(width), compare,
            " and." + suffix + " %zero, %marker, 255;\n and." + suffix + " %snapshot, %marker, -256;\n"
            " or." + suffix + " %snapshot, %snapshot, %zero;\n" + guard),
            "masked marker reconstruction " + std::to_string(width));
        ok &= rejects(replace(fixture(width), compare,
            " or." + suffix + " %snapshot, %marker, 1;\n" + guard),
            "OR nonzero operand cannot establish zero " + std::to_string(width));
    }
    return ok;
}

bool budgets() {
    bool ok = true;
    for (const bool live : {false, true}) {
        RawFixture test;
        test.basic();
        test.function.register_ranges.push_back({"%tag", "b64", 129, true});
        for (unsigned i = 0; i < 129; ++i) {
            const auto name = "%tag" + std::to_string(i);
            test.blocks[0].instructions.insert(test.blocks[0].instructions.end() - 1,
                                               test.add("mov.b64", {name, "0"}));
            if (live) test.blocks[1].instructions.insert(test.blocks[1].instructions.begin(),
                test.add("st.global.u64", {"[%output]", name}));
        }
        ok &= expect(test.specializes() == !live, live ? "combined live-fact exhaustion declines proof"
                                                     : "dead scalar facts pruned before live-fact limit");
    }
    constexpr std::size_t incoming = 4097;
    RawFixture clones(incoming + 3);
    const auto zero = clones.add("mov.b64", {"%marker", "0"});
    const auto jump = clones.add("bra", {clones.blocks[incoming].name});
    for (std::size_t i = 0; i < incoming; ++i) {
        clones.blocks[i].instructions = {zero, jump};
        clones.blocks[i].successors = {incoming};
    }
    clones.blocks[incoming].instructions = {clones.add("setp.eq.b64", {"%p0", "%marker", "0"}),
        clones.add("bra", {clones.blocks[incoming + 1].name}, "%p0")};
    clones.blocks[incoming].successors = {incoming + 1, incoming + 2};
    clones.blocks[incoming + 1].instructions = {clones.add("ret", {})};
    clones.blocks[incoming + 2].instructions = {clones.add("ret", {})};
    detail::simplify_guarded_paths(clones.blocks, clones.builder, clones.instructions, &clones.function);
    ok &= expect(clones.blocks.size() == incoming + 3 + 4096, "scalar guards share the 4096 clone limit");
    ok &= expect(clones.blocks[incoming - 1].successors[0] == incoming,
                 "clone exhaustion retains the original unresolved edge");
    return ok;
}

bool loop_meets() {
    bool ok = true;
    for (unsigned variant = 0; variant < 4; ++variant) {
        RawFixture test(7);
        test.blocks[0].instructions = {test.add("bra", {test.blocks[1].name}, "%p1")};
        test.blocks[0].successors = {1, variant == 2 ? 3u : 2u};
        for (const auto block : {1u, 2u}) {
            test.blocks[block].instructions = {test.add("mov.b64",
                {"%marker", block == 2 && variant == 1 ? "1" : "0"}),
                test.add("bra", {test.blocks[3].name})};
            test.blocks[block].successors = {3};
        }
        // The unknown loop condition stops a local incoming-edge proof. Only
        // the completed predecessor meet can carry zero to the exit guard.
        if (variant == 3)
            test.blocks[3].instructions.push_back(test.add("mov.b64", {"%marker", "1"}));
        test.blocks[3].instructions.push_back(test.add("bra", {test.blocks[4].name}, "%p2"));
        test.blocks[3].successors = {4, 3};
        test.blocks[4].instructions = {test.add("setp.eq.b64", {"%p0", "%marker", "0"}),
            test.add("bra", {test.blocks[5].name}, "%p0")};
        test.blocks[4].successors = {5, 6};
        test.blocks[5].instructions = {test.add("ret", {})};
        test.blocks[6].instructions = {test.add("ret", {})};
        detail::simplify_guarded_paths(test.blocks, test.builder, test.instructions, &test.function);
        ok &= expect((test.blocks[3].successors[0] != 4) == (variant == 0),
                     "zero join, conflicting input, bypass, overwritten backedge " + std::to_string(variant));
    }
    return ok;
}
}  // namespace

int main() {
    bool ok = positive_and_negative_imports();
    ok &= raw_contracts();
    ok &= conditional_select_payloads();
    ok &= loop_meets();
    ok &= budgets();
    ok &= bitwise_zero_markers();
    if (ok) std::cout << "PASS scalar zero guards: " << checks << " checks\n";
    return ok ? 0 : 1;
}
