#include "cumetal/ir/ir.h"
#include "cumetal/metal/lower_to_msl.h"
#include "ptx_address_alignment.h"

#include <deque>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
using namespace cumetal;

bool expect(bool condition, const std::string& message) {
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

const std::string kPrefix = R"ptx(
.version 8.8
.target sm_89
.address_size 64
.visible .entry alignment_probe(.param .u64 .ptr .global output,
                               .param .u64 selector, .param .u64 offset) {
.local .align 16 .b8 scratch[256];
.local .align 16 .b8 other[256];
.local .align 1 .b8 unaligned[256];
.reg .b64 %rd<20>;
.reg .b32 %r<4>;
.reg .pred %p<4>;
ld.param.u64 %rd0, [output];
ld.param.u64 %rd1, [selector];
ld.param.u64 %rd2, [offset];
mov.u64 %rd3, scratch;
)ptx";

const std::string kOr = "or.b64 %rd6, %rd4, 1;\n";
const std::string kSuffix = R"ptx(
setp.eq.u64 %p3, %rd2, 0;
@%p3 bra ADDRESS_JOIN;
mov.u64 %rd10, scratch;
mov.u64 %rd6, %rd10;
ADDRESS_JOIN:
st.local.u8 [%rd6], 37;
ld.local.u8 %r0, [%rd6];
st.global.u32 [%rd0], %r0;
ret;
}
)ptx";

std::string replace_once(std::string text, const std::string& from, const std::string& to) {
    const auto position = text.find(from);
    if (position == std::string::npos)
        throw std::logic_error("missing test replacement");
    text.replace(position, from.size(), to);
    return text;
}

bool accepts(const std::string& body, const std::string& label) {
    const auto compiled = metal::compile_ptx_to_msl(kPrefix + body + kSuffix);
    if (!expect(compiled.ok, label + ": " + compiled.error))
        return false;
    bool ok = expect(ir::verify(compiled.gpu_ir).ok && ir::verify(compiled.metal_ir).ok,
                     label + " verifies both IR stages");
    unsigned private_stores = 0, private_loads = 0;
    for (const auto& function : compiled.gpu_ir.functions)
        for (const auto& block : function.blocks)
            for (const auto& operation : block.operations) {
                if (operation.operands.empty())
                    continue;
                if (operation.opcode == ir::OpCode::kStore &&
                    operation.operands.front().type.address_space == ir::AddressSpace::kPrivate)
                    ++private_stores;
                if (operation.opcode == ir::OpCode::kLoad &&
                    operation.operands.front().type.address_space == ir::AddressSpace::kPrivate)
                    ++private_loads;
            }
    ok &= expect(private_stores >= 1 && private_loads >= 1,
                 label + " preserves the actual private-memory effects");
    return ok;
}

bool rejects(const std::string& body, const std::string& label) {
    const auto compiled = metal::compile_ptx_to_msl(kPrefix + body + kSuffix);
    return expect(!compiled.ok, label + " must retain rejection without a proven address OR");
}

// Exercise the normalization API independently of importer hooks. In particular,
// late budget exhaustion must not commit the first of two available rewrites.
bool budget_and_origin_tests() {
    using namespace ir;
    using namespace ir::detail;
    bool ok = true;
    bool saw_exhaustion = false, saw_commit = false;
    for (std::size_t work = 0; work < 80; ++work) {
        std::deque<Instruction> original;
        Instruction base;
        base.opcode = "mov.u64";
        base.operands = {"%rd1", "scratch"};
        base.line = 10;
        original.push_back(base);
        Instruction first;
        first.opcode = "or.b64";
        first.operands = {"%rd2", "%rd1", "1"};
        first.line = 11;
        original.push_back(first);
        Instruction second = first;
        second.operands = {"%rd3", "%rd1", "4"};
        second.line = 12;
        original.push_back(second);
        std::vector<RawBlock> blocks(1);
        blocks[0].instructions = {&original[0], &original[1], &original[2]};
        const auto before = blocks[0].instructions;
        std::vector<std::unordered_map<std::string, ValueId>> incoming(1), outgoing(1);
        std::vector<std::map<std::string, ValueId>> arguments(1);
        std::unordered_map<const Instruction*, std::vector<ValueId>> results = {
            {&original[0], {1}}, {&original[1], {2}}, {&original[2], {3}}};
        std::unordered_map<ValueId, Type> types = {
            {1, Type::pointer(Type::integer(8), AddressSpace::kPrivate)},
            {2, Type::integer(64)},
            {3, Type::integer(64)}};
        std::unordered_map<std::string, LocalDepot> depots = {{"scratch", {"scratch", 256, 16}}};
        std::deque<Instruction> storage;
        InstructionOrigins origins;
        const auto result =
            legalize_aligned_local_address_ors(blocks, incoming, outgoing, arguments, results, types, depots,
                                               storage, &origins, {.max_work = work});
        if (result.budget_exhausted) {
            saw_exhaustion = true;
            ok &= expect(result.rewritten == 0 && blocks[0].instructions == before && storage.empty() &&
                             origins.empty(),
                         "budget exhaustion is atomic at work=" + std::to_string(work));
        } else {
            saw_commit = true;
            ok &= expect(result.rewritten == 2 && storage.size() == 2 && origins.size() == 2,
                         "both proven rewrites commit together");
            for (std::size_t i = 1; i < 3; ++i) {
                const auto* replacement = blocks[0].instructions[i];
                ok &= expect(replacement->opcode == "add.u64" && replacement->line == original[i].line &&
                                 replacement->operands == original[i].operands &&
                                 origins.at(replacement) == &original[i],
                             "rewrite preserves original operands and location identity");
            }
        }
    }
    ok &= expect(saw_exhaustion && saw_commit, "work-budget sweep covers both exhaustion and commit");
    return ok;
}

bool loop_proof_tests() {
    using namespace ir;
    using namespace ir::detail;
    bool ok = true;
    struct Case {
        const char* stride;
        bool from_or;
        std::size_t rewrites;
    };
    for (const Case test : {Case{"16", false, 1}, Case{"2", false, 1}, Case{"1", false, 0},
                            Case{"15", true, 1}, Case{"0", true, 0}}) {
        std::deque<Instruction> original;
        for (const auto& [opcode, operands] : std::vector<std::pair<std::string, std::vector<std::string>>>{
                 {"mov.u64", {"%rd1", "scratch"}},
                 {"or.b64", {"%rd2", "%rd1", "1"}},
                 {"add.u64", {"%rd1", test.from_or ? "%rd2" : "%rd1", test.stride}}}) {
            Instruction instruction;
            instruction.opcode = opcode;
            instruction.operands = operands;
            original.push_back(std::move(instruction));
        }
        std::vector<RawBlock> blocks(2);
        blocks[0].instructions = {&original[0]};
        blocks[0].successors = {1};
        blocks[1].instructions = {&original[1], &original[2]};
        blocks[1].predecessors = {0, 1};
        blocks[1].successors = {1};
        std::vector<std::unordered_map<std::string, ValueId>> incoming(2), outgoing(2);
        incoming[1]["%rd1"] = 2;
        outgoing[0]["%rd1"] = 1;
        outgoing[1]["%rd1"] = 4;
        std::vector<std::map<std::string, ValueId>> arguments(2);
        arguments[1]["%rd1"] = 2;
        std::unordered_map<const Instruction*, std::vector<ValueId>> results = {
            {&original[0], {1}}, {&original[1], {3}}, {&original[2], {4}}};
        std::unordered_map<ValueId, Type> types = {
            {1, Type::integer(64)}, {2, Type::integer(64)}, {3, Type::integer(64)}, {4, Type::integer(64)}};
        std::unordered_map<std::string, LocalDepot> depots = {{"scratch", {"scratch", 256, 16}}};
        std::deque<Instruction> storage;
        const auto result = legalize_aligned_local_address_ors(blocks, incoming, outgoing, arguments, results,
                                                               types, depots, storage, nullptr);
        ok &= expect(!result.budget_exhausted && result.rewritten == test.rewrites,
                     "SSA loop proof validates every backedge: stride=" + std::string(test.stride) +
                         (test.from_or ? " from OR" : " from incoming address"));
    }
    return ok;
}
} // namespace

int main() {
    bool ok = budget_and_origin_tests() && loop_proof_tests();
    const std::string aligned = "add.u64 %rd4, %rd3, 176;\n";
    ok &= accepts(aligned + kOr, "aligned allocation plus multiple-of-16 displacement");
    ok &= accepts(aligned + "or.b64 %rd6, 4, %rd4;\n", "commuted OR4 immediate");
    ok &= accepts("add.u64 %rd4, %rd3, 4;\n" + kOr, "nonzero low residue with disjoint mask");
    ok &= accepts("add.u64 %rd4, %rd3, 4;\nsub.u64 %rd4, %rd4, 4;\n" + kOr,
                  "subtracting an immediate restores low-bit alignment");
    ok &= accepts("add.s64 %rd4, %rd3, -16;\nadd.u64 %rd4, %rd4, 32;\n" + kOr,
                  "modular negative literal offsets");
    ok &= accepts("mov.u64 %rd8, 16;\nadd.u64 %rd4, %rd3, %rd8;\nmov.u64 %rd8, 1;\n" + kOr,
                  "literal offset captured before register overwrite");
    ok &= accepts("mov.b64 %rd8, %rd3;\nmov.u64 %rd4, %rd8;\nmov.u64 %rd3, 7;\n" + kOr,
                  "copied allocation remains proven after source-name overwrite");
    ok &= accepts("cvta.local.u64 %rd8, %rd3;\nadd.u64 %rd4, %rd8, 32;\n" + kOr,
                  "local conversion retains allocation alignment");
    ok &= accepts("cvta.local.u64 %rd8, %rd3;\ncvta.to.local.u64 %rd4, %rd8;\n" + kOr,
                  "conversion back to local retains allocation alignment");
    ok &= accepts("mov.u64 %rd4, %rd3;\nor.b64 %rd4, %rd4, 1;\nor.b64 %rd6, %rd4, 4;\n",
                  "in-place and dependent ORs use distinct SSA definitions");

    const std::string joined = R"ptx(
setp.eq.u64 %p0, %rd1, 0;
@%p0 bra ALTERNATE;
add.u64 %rd4, %rd3, 16;
bra JOIN;
ALTERNATE:
add.u64 %rd4, %rd3, 32;
JOIN:
)ptx";
    ok &= accepts(joined + kOr, "both aligned incoming edges at a join");
    ok &= accepts(
        replace_once(joined, "add.u64 %rd4, %rd3, 32;", "mov.u64 %rd8, other;\nadd.u64 %rd4, %rd8, 32;") +
            kOr,
        "distinct aligned private allocations retain local ancestry");
    ok &= accepts(R"ptx(
bra INIT;
USE:
or.b64 %rd6, %rd4, 1;
bra END;
INIT:
add.u64 %rd4, %rd3, 16;
bra USE;
END:
)ptx",
                  "out-of-order blocks use control-flow reaching definitions");
    const std::string selected = R"ptx(
add.u64 %rd8, %rd3, 16;
add.u64 %rd9, %rd3, 32;
setp.eq.u64 %p0, %rd1, 0;
selp.b64 %rd4, %rd8, %rd9, %p0;
)ptx";
    ok &= accepts(selected + kOr, "both aligned selected addresses");
    const std::string loop = R"ptx(
mov.u64 %rd4, %rd3;
mov.u64 %rd8, 0;
LOOP:
or.b64 %rd6, %rd4, 1;
add.u64 %rd4, %rd4, 16;
add.u64 %rd8, %rd8, 1;
setp.lt.u64 %p0, %rd8, 4;
@%p0 bra LOOP;
)ptx";
    ok &= accepts(loop, "anchored loop preserves 16-byte alignment");
    ok &= accepts(
        replace_once(loop, "add.u64 %rd4, %rd4, 16;", "or.b64 %rd4, %rd4, 1;\nadd.u64 %rd4, %rd4, 15;"),
        "OR-derived address feeds an alignment-preserving backedge");
    ok &= accepts(replace_once(loop, "add.u64 %rd4, %rd4, 16;", "add.u64 %rd4, %rd4, 2;"),
                  "loop loses stronger alignment but retains required evenness");
    ok &= accepts(R"ptx(
mov.u64 %rd4, %rd3;
or.b64 %rd5, %rd4, 1;
mov.u64 %rd8, 0;
BYTE_LOOP:
mov.u64 %rd4, %rd5;
add.u64 %rd5, %rd5, 1;
add.u64 %rd8, %rd8, 1;
setp.lt.u64 %p0, %rd8, 4;
@%p0 bra BYTE_LOOP;
mov.u64 %rd6, %rd4;
)ptx",
                  "pre-loop OR feeding a byte-stride pointer backedge");

    ok &= rejects("mov.u64 %rd4, unaligned;\n" + kOr, "alignment-one allocation");
    ok &= rejects("add.u64 %rd4, %rd3, 1;\n" + kOr, "OR overlaps a known-one low bit");
    ok &= rejects("add.u64 %rd4, %rd3, 4;\nor.b64 %rd6, %rd4, 4;\n", "OR4 overlaps residue4");
    ok &= rejects("add.u64 %rd4, %rd3, %rd2;\n" + kOr, "unbounded runtime displacement");
    ok &= rejects(aligned + "or.b64 %rd6, %rd4, %rd2;\n", "runtime OR mask has no immediate proof");
    ok &= rejects(replace_once(joined, "add.u64 %rd4, %rd3, 32;", "add.u64 %rd4, %rd3, 33;") + kOr,
                  "one misaligned predecessor invalidates the join proof");
    ok &= rejects(replace_once(joined, "add.u64 %rd4, %rd3, 32;", "mov.u64 %rd4, %rd2;") + kOr,
                  "an integer predecessor is not local allocation provenance");
    ok &= rejects(replace_once(selected, "add.u64 %rd9, %rd3, 32;", "add.u64 %rd9, %rd3, 33;") + kOr,
                  "one misaligned selected alternative");
    ok &= rejects(replace_once(loop, "add.u64 %rd4, %rd4, 16;", "add.u64 %rd4, %rd4, 1;"),
                  "byte-stride backedge invalidates alignment before commit");
    ok &= rejects("mov.u64 %rd4, %rd3;\nmov.u64 %rd4, %rd2;\n" + kOr,
                  "overwritten address name cannot retain its earlier allocation");
    ok &= rejects("mov.u64 %rd4, %rd3;\nsetp.eq.u64 %p0, %rd1, 0;\n@%p0 add.u64 %rd4, %rd3, 1;\n" + kOr,
                  "predicated overwrite cannot borrow the other edge's alignment");
    ok &= rejects("cvta.local.u64 %rd4, %rd2;\n" + kOr,
                  "cvta of an arbitrary integer cannot invent aligned allocation ancestry");
    ok &= rejects("cvt.u32.u64 %r1, %rd3;\ncvt.u64.u32 %rd4, %r1;\n" + kOr,
                  "narrowing and widening do not restore pointer ancestry");
    ok &= rejects("mov.u64 %rd4, %rd0;\n" + kOr, "global parameter has no declared local alignment proof");
    const auto scalar = metal::compile_ptx_to_msl(kPrefix + R"ptx(
or.b64 %rd6, %rd2, 1;
st.global.u64 [%rd0], %rd6;
ret;
}
)ptx");
    ok &= expect(scalar.ok, "ordinary scalar OR remains supported: " + scalar.error);
    if (!ok)
        return 1;
    std::cout << "PTX aligned address normalization tests passed\n";
    return 0;
}
