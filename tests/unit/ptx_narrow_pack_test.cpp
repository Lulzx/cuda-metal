#include "cumetal/ir/ptx_importer.h"
#include "cumetal/metal/lower_to_msl.h"
#include "ptx_tuple_normalization.h"

#include <algorithm>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace ir = cumetal::ir;
namespace detail = cumetal::ir::detail;
std::size_t checks = 0;

bool expect(bool condition, const std::string& label) {
    ++checks;
    if (!condition)
        std::cerr << "FAIL: " << label << '\n';
    return condition;
}

std::string replace(std::string text, const std::string& from, const std::string& to) {
    const auto at = text.find(from);
    if (at == std::string::npos)
        throw std::runtime_error("missing replacement: " + from);
    text.replace(at, from.size(), to);
    return text;
}

std::string rename(std::string text, const std::string& from, const std::string& to) {
    std::size_t at = 0;
    while ((at = text.find(from, at)) != std::string::npos) {
        text.replace(at, from.size(), to);
        at += to.size();
    }
    return text;
}

std::string fixture(unsigned format = 16, unsigned storage = 16, unsigned layout = 0) {
    std::string source = R"ptx(.version 7.0
.target sm_80
.address_size 64
.visible .entry probe(.param .u64 .ptr .global output,
                     .param .u32 value, .param .u32 choice) {
 .reg .b64 %output, %packed, %result;
 .reg .b32 %low, %high, %other, %counter, %choice;
 .reg .pred %condition;
)ptx";
    source += " .reg .b" + std::to_string(storage) + " %converted;\n";
    source += R"ptx( ld.param.u64 %output, [output];
 ld.param.u32 %low, [value];
 ld.param.u32 %choice, [choice];
 setp.ne.u32 %condition, %choice, 0;
)ptx";
    if (layout == 2)
        source += R"ptx( mov.u32 %counter, 0;
LOOP:
 add.u32 %counter, %counter, 1;
 setp.lt.u32 %condition, %counter, 2;
 @%condition bra LOOP;
)ptx";
    source += " mov.b64 %packed, {%low, %high};\n";
    if (layout == 1)
        source += " add.u32 %other, %low, 3;\n st.global.u32 [%output+8], %other;\n";
    source += " cvt.u" + std::to_string(format) + ".u64 %converted, %packed;\n";
    source += " cvt.u64.u" + std::to_string(storage) + " %result, %converted;\n";
    source += " st.global.u64 [%output], %result;\n ret;\n}\n";
    return source;
}

bool accepts(const std::string& source, const std::string& label, const std::string& rewritten = {}) {
    const auto imported = ir::import_ptx(source);
    bool ok = expect(imported.ok, label + ": " + imported.error);
    if (!imported.ok)
        return ok;
    ok &= expect(ir::verify(imported.module).ok, label + ": verified SSA");
    if (!rewritten.empty()) {
        bool found = false;
        for (const auto& function : imported.module.functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations) {
                    const auto opcode = operation.attributes.find("ptx_opcode");
                    found |= opcode != operation.attributes.end() && opcode->second == rewritten;
                }
        ok &= expect(found, label + ": low-source conversion retains instruction format");
    }
    const auto compiled = cumetal::metal::compile_ptx_to_msl(source);
    ok &= expect(compiled.ok, label + ": strict MSL lowering: " + compiled.error);
    return ok;
}

bool rejects(const std::string& source, const std::string& label) {
    const auto imported = ir::import_ptx(source);
    return expect(!imported.ok && imported.error.find("PTX register '") != std::string::npos &&
                      (imported.error.find(" is undefined ") != std::string::npos ||
                       imported.error.find(" is used before definition ") != std::string::npos),
                  label + ": observed undefined value rejected: " + imported.error);
}

bool imported_cases() {
    bool ok = true;
    for (const auto& [format, storage] :
         std::vector<std::pair<unsigned, unsigned>>{{16, 16}, {16, 32}, {16, 64}, {32, 32}, {32, 64}})
        for (unsigned layout = 0; layout < 3; ++layout)
            ok &= accepts(fixture(format, storage, layout),
                          "unsigned format " + std::to_string(format) + " storage " +
                              std::to_string(storage) + " layout " + std::to_string(layout),
                          "cvt.u" + std::to_string(format) + ".u32");

    const auto source = fixture();
    auto renamed = rename(rename(rename(fixture(16, 64), "%low", "%rd_low"), "%packed", "%rs_packed"),
                          "%converted", "%r_converted");
    ok &= accepts(renamed, "declarations override conventional register spelling", "cvt.u16.u32");
    auto ranged = rename(source, "%low", "%lane0");
    ranged = replace(ranged, ".reg .b32 %lane0, %high, %other, %counter, %choice;",
                     ".reg .b32 %lane<2>;\n .reg .b32 %other, %counter, %choice;");
    ranged = rename(ranged, "%high", "%lane1");
    ok &= accepts(ranged, "compact range declarations", "cvt.u16.u32");
    ok &= accepts(replace(source, " mov.b64 %packed,", " mov.u32 %high, 4294967295;\n mov.b64 %packed,"),
                  "defined high control", "cvt.u16.u32");
    ok &= accepts(replace(source, " cvt.u16.u64", " mov.u32 %high, 0;\n cvt.u16.u64"),
                  "unobserved high changes between pair", "cvt.u16.u32");
    auto inplace = fixture(32, 32);
    inplace = replace(inplace, "cvt.u32.u64 %converted, %packed;", "cvt.u32.u64 %low, %packed;");
    inplace = replace(inplace, "%result, %converted;", "%result, %low;");
    ok &= accepts(inplace, "consumer overwrites selected low after reading it", "cvt.u32.u32");
    ok &= accepts(replace(source, " cvt.u64.u16", " mov.u32 %low, 0;\n cvt.u64.u16"), "write after consumer",
                  "cvt.u16.u32");
    auto loop_pair =
        replace(source, " mov.b64 %packed,", " mov.u32 %counter, 0;\nPAIR_LOOP:\n mov.b64 %packed,");
    loop_pair = replace(loop_pair, " ret;",
                        " add.u32 %low, %low, 1;\n add.u32 %counter, %counter, 1;\n"
                        " setp.lt.u32 %condition, %counter, 2;\n @%condition bra PAIR_LOOP;\n ret;");
    ok &= accepts(loop_pair, "same-block pair in loop with a later low-source update", "cvt.u16.u32");

    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
             {"{%low, %high}", "{%high, %low}"},
             {" cvt.u16.u64", " st.global.u64 [%output+8], %packed;\n cvt.u16.u64"},
             {" cvt.u16.u64", " cvt.u32.u64 %other, %packed;\n cvt.u16.u64"},
             {" cvt.u16.u64", " mov.u32 %low, 0;\n cvt.u16.u64"},
             {" cvt.u16.u64", " @%condition mov.u32 %low, 0;\n cvt.u16.u64"},
             {" cvt.u16.u64", " mov.u16 %low, 0;\n cvt.u16.u64"},
             {" mov.b64 %packed,", " @%condition mov.b64 %packed,"},
             {" cvt.u16.u64", " @%condition cvt.u16.u64"},
             {" cvt.u16.u64", " bra CONSUME;\nCONSUME:\n cvt.u16.u64"},
             {" cvt.u16.u64 %converted, %packed;",
              "mov.b64 {_, %other}, %packed;\n cvt.u16.u32 %converted, %other;"}})
        ok &= rejects(replace(source, from, to), "refuse " + to);

    // Keep the pre-existing low/high extraction and unread named-lane contracts.
    for (bool high : {false, true}) {
        for (const auto& discarded : {std::string("_"), std::string("%other")}) {
            auto extraction = fixture(32, 32);
            if (high)
                extraction = replace(extraction, "{%low, %high}", "{%high, %low}");
            const auto lanes = high ? "{" + discarded + ", %converted}" : "{%converted, " + discarded + "}";
            extraction =
                replace(extraction, "cvt.u32.u64 %converted, %packed;", "mov.b64 " + lanes + ", %packed;");
            ok &= accepts(extraction, "existing extraction " + lanes);
            if (discarded != "_")
                ok &= rejects(replace(extraction, " ret;", " st.global.u32 [%output+8], %other;\n ret;"),
                              "observed named discarded lane");
        }
    }
    return ok;
}

struct RawFixture {
    std::deque<detail::Instruction> source, normalized;
    std::vector<detail::RawBlock> blocks{1};
    cumetal::ptx::EntryFunction function;
    detail::InstructionOrigins origins;

    RawFixture() {
        function.register_declarations = {
            {"%packed", "b64", true},    {"%low", "b32", true},   {"%high", "b32", true},
            {"%converted", "b16", true}, {"%other", "b32", true}, {"%dead", "b32", true},
            {"%output", "b64", true},    {"%wide", "b64", true},  {"%half", "b16", true}};
    }

    detail::Instruction* add(std::string opcode, std::vector<std::string> operands,
                             std::string predicate = {}, std::size_t block = 0) {
        detail::Instruction instruction;
        instruction.opcode = std::move(opcode);
        instruction.operands = std::move(operands);
        instruction.predicate = std::move(predicate);
        instruction.line = static_cast<int>(source.size() + 100);
        instruction.supported = true;
        source.push_back(std::move(instruction));
        blocks.at(block).instructions.push_back(&source.back());
        return &source.back();
    }

    void basic() {
        add("mov.u32", {"%low", "123456789"});
        add("mov.b64", {"%packed", "{%low, %high}"});
        add("cvt.u16.u64", {"%converted", "%packed"});
        add("st.global.u16", {"[%output]", "%converted"});
    }

    void normalize(detail::TupleNormalizationLimits limits = {}) {
        detail::remove_discarded_pack_halves(blocks, normalized, &function, &origins, limits);
    }

    bool unchanged(const std::string& label, detail::TupleNormalizationLimits limits = {}) {
        const auto before = blocks;
        normalize(limits);
        bool same = before.size() == blocks.size() && normalized.empty() && origins.empty();
        for (std::size_t b = 0; b < blocks.size(); ++b)
            same &= before[b].instructions == blocks[b].instructions;
        return expect(same, label + ": original instructions retained");
    }
};

bool raw_cases() {
    bool ok = true;
    {
        RawFixture raw;
        raw.basic();
        const auto* consumer = raw.blocks[0].instructions[2];
        raw.normalize();
        ok &= expect(raw.normalized.size() == 1 && raw.blocks[0].instructions.size() == 3,
                     "one pack removed and one consumer replaced");
        if (raw.normalized.size() == 1) {
            const auto& result = raw.normalized.front();
            ok &= expect(result.opcode == "cvt.u16.u32" &&
                             result.operands == std::vector<std::string>{"%converted", "%low"} &&
                             result.line == consumer->line && raw.origins.at(&result) == consumer,
                         "replacement format, operands and source origin");
        }
    }
    // These are proof refusal tests, including forms the ordinary importer may
    // support independently. They must not become new valid moves/conversions.
    for (unsigned variant = 0; variant < 26; ++variant) {
        RawFixture raw;
        raw.basic();
        auto& pack = raw.source[1];
        auto& consumer = raw.source[2];
        switch (variant) {
        case 0:
            consumer.opcode = "cvt.s16.u64";
            break;
        case 1:
            consumer.opcode = "cvt.u16.s64";
            break;
        case 2:
            consumer.opcode = "cvt.sat.u16.u64";
            break;
        case 3:
            consumer.opcode = "cvt.rn.f32.u64";
            break;
        case 4:
            consumer.opcode = "cvt.u64.u64";
            break;
        case 5:
            pack.predicate = "%condition";
            break;
        case 6:
            consumer.predicate = "%condition";
            break;
        case 7:
            pack.operands[1] = "{%low, %high, %other}";
            break;
        case 8:
            pack.operands[1] = "{%low, _}";
            break;
        case 9:
            pack.operands[1] = "{%low, %high+1}";
            break;
        case 10:
            consumer.operands[0] = "{%converted, %half}";
            break;
        case 11:
            consumer.operands.push_back("0");
            break;
        case 12:
            pack.operands[0] = "%packed+0";
            break;
        case 13:
            consumer.operands[1] = "%packed+0";
            break;
        case 14:
            raw.function.register_declarations[0].type = "b32";
            break;
        case 15:
            raw.function.register_declarations[1].type = "b64";
            break;
        case 16:
            raw.function.register_declarations[2].type = "b16";
            break;
        case 17:
            raw.function.register_declarations[1].type = "f32";
            break;
        case 18:
            raw.function.register_declarations[3].type = "f16";
            break;
        case 19:
            consumer.opcode = "cvt.u32.u64";
            break; // b16 destination is too small.
        case 20:
            raw.function.register_declarations.push_back({"%low", "b32", true});
            break;
        case 21:
            raw.function.register_declarations.push_back({"%low", "b64", true});
            break;
        case 22:
            raw.function.register_declarations[1].function_scope = false;
            break;
        case 23:
            raw.function.register_declarations.erase(raw.function.register_declarations.begin() + 1);
            break;
        case 24:
            std::swap(raw.blocks[0].instructions[1], raw.blocks[0].instructions[2]);
            break;
        case 25:
            raw.blocks.resize(2);
            raw.blocks[1].instructions = {raw.blocks[0].instructions[2], raw.blocks[0].instructions[3]};
            raw.blocks[0].instructions.resize(2);
            raw.blocks[0].successors = {1};
            break;
        }
        ok &= raw.unchanged("raw refusal " + std::to_string(variant));
    }
    for (const auto& [opcode, operands] : std::vector<std::pair<std::string, std::vector<std::string>>>{
             {"mov.u32", {"%low", "0"}},
             {"mov.u16", {"%low", "0"}},
             {"mov.b64", {"{%other, %low}", "%wide"}},
             {"ld.global.v2.u32", {"{%other, %low}", "[%output]"}},
             {"call.uni", {"helper", "()"}},
             {"call.uni", {"(%low)", "helper", "()"}},
             {"call.uni", {"(%packed)", "helper", "()"}}}) {
        for (const auto& predicate : {std::string{}, std::string{"%condition"}}) {
            RawFixture raw;
            raw.basic();
            const auto* added = raw.add(opcode, operands, predicate);
            raw.blocks[0].instructions.pop_back();
            raw.blocks[0].instructions.insert(raw.blocks[0].instructions.begin() + 2, added);
            ok &= raw.unchanged("intervening write/call " + opcode + " " + predicate);
        }
    }
    for (const auto& [opcode, operands] : std::vector<std::pair<std::string, std::vector<std::string>>>{
             {"st.global.u64", {"[%output]", "%packed"}},
             {"cvt.u32.u64", {"%other", "%packed"}},
             {"mov.b64", {"%packed", "0"}},
             {"call.uni", {"(%packed)", "helper", "()"}},
             {"call.uni", {"helper", "(%packed)"}}}) {
        RawFixture raw;
        raw.basic();
        raw.add(opcode, operands);
        ok &= raw.unchanged("whole-function extra use/definition " + opcode);
    }
    {
        RawFixture raw;
        raw.basic();
        raw.function.register_declarations[1].name = "%clock_payload";
        raw.source[0].operands[0] = "%other"; // Leave the selected source undefined.
        raw.source[1].operands[1] = "{%clock_payload, %high}";
        raw.normalize();
        ok &= expect(raw.normalized.size() == 1 && raw.normalized.front().operands[1] == "%clock_payload",
                     "builtin-looking declared source remains a demanded operand without throwing");
    }
    {
        RawFixture raw;
        raw.basic();
        raw.function.register_declarations.push_back({"%clock_discarded", "b32", true});
        raw.function.register_declarations[3].type = "b32";
        raw.source[2].opcode = "mov.b64";
        raw.source[2].operands = {"{%converted, %clock_discarded}", "%packed"};
        raw.add("st.global.u32", {"[%output+4]", "%clock_discarded"});
        ok &= raw.unchanged("builtin-looking observed extraction lane is not discarded");
    }
    {
        RawFixture raw;
        raw.basic();
        raw.function.register_declarations[3].type = "b32";
        raw.source[2].opcode = "mov.b64";
        raw.source[2].operands = {"{%converted, _}", "%packed"};
        raw.function.register_declarations[1].type = "f32";
        raw.function.register_declarations[2].type = "f32";
        raw.normalize();
        ok &= expect(raw.normalized.size() == 1 && raw.normalized.front().opcode == "mov.b32",
                     "existing extraction preserves floating bit-container widths");
    }
    return ok;
}

bool declaration_ranges() {
    bool ok = true;
    for (unsigned variant = 0; variant < 8; ++variant) {
        RawFixture raw;
        raw.basic();
        raw.function.register_declarations.erase(raw.function.register_declarations.begin() + 1);
        raw.function.register_ranges = {{"%lane", "b32", 2, true}};
        std::string name = "%lane1";
        if (variant == 1)
            name = "%lane01";
        if (variant == 2)
            name = "%lane2";
        if (variant == 3)
            name = "%lane184467440737095516160";
        if (variant == 4)
            raw.function.register_ranges.push_back({"%lane", "b32", 2, true});
        if (variant == 5)
            raw.function.register_declarations.push_back({"%lane1", "b32", true});
        if (variant == 6)
            raw.function.register_ranges[0].function_scope = false;
        if (variant == 7)
            raw.function.register_ranges[0].count = std::numeric_limits<std::size_t>::max();
        raw.source[0].operands[0] = name;
        raw.source[1].operands[1] = "{" + name + ", %high}";
        if (variant == 0 || variant == 7) {
            raw.normalize();
            ok &= expect(raw.normalized.size() == 1, "compact range accepted " + std::to_string(variant));
        } else
            ok &= raw.unchanged("ambiguous/noncanonical range " + std::to_string(variant));
    }
    return ok;
}

bool bounded_work() {
    bool ok = true;
    for (unsigned variant = 0; variant < 7; ++variant) {
        RawFixture raw;
        raw.basic();
        detail::TupleNormalizationLimits limits;
        switch (variant) {
        case 0:
            limits.instruction_visits = 0;
            break;
        case 1:
            limits.register_occurrences = 0;
            break;
        case 2:
            limits.text_bytes = 0;
            break;
        case 3:
            limits.tracked_registers = 0;
            break;
        case 4:
            limits.candidates = 0;
            break;
        case 5:
            limits.declaration_checks = 0;
            break;
        case 6:
            limits.declaration_entries = 0;
            break;
        }
        ok &= raw.unchanged("exhausted budget " + std::to_string(variant), limits);
    }
    {
        RawFixture raw;
        raw.basic();
        detail::TupleNormalizationLimits limits;
        limits.instruction_visits = 14; // Three passes over four instructions plus block.
        ok &= raw.unchanged("late commit-scan exhaustion is transactional", limits);
    }
    {
        RawFixture raw;
        raw.basic();
        detail::TupleNormalizationLimits limits;
        limits.instruction_visits = 15;
        raw.normalize(limits);
        ok &= expect(raw.normalized.size() == 1, "exact instruction visit budget succeeds");
    }
    for (bool exhaust : {false, true}) {
        RawFixture raw;
        constexpr std::size_t count = 128;
        raw.function.register_ranges = {{"%word", "b64", count, true}, {"%result", "b16", count, true}};
        raw.add("mov.u32", {"%low", "123"});
        for (std::size_t i = 0; i < count; ++i)
            raw.add("mov.b64", {"%word" + std::to_string(i), "{%low, %high}"});
        for (std::size_t i = 0; i < 256; ++i)
            raw.add("st.global.u32", {"[%output]", "%low"});
        for (std::size_t i = 0; i < count; ++i) {
            raw.add("cvt.u16.u64", {"%result" + std::to_string(i), "%word" + std::to_string(i)});
            raw.add("st.global.u16", {"[%output]", "%result" + std::to_string(i)});
        }
        detail::TupleNormalizationLimits limits;
        limits.candidates = exhaust ? count - 1 : count;
        if (exhaust)
            ok &= raw.unchanged("candidate exhaustion before many pending consumers", limits);
        else {
            raw.normalize(limits);
            ok &= expect(raw.normalized.size() == count, "many long overlapping stable intervals normalize");
            const auto stores =
                std::count_if(raw.blocks[0].instructions.begin(), raw.blocks[0].instructions.end(),
                              [](const auto* instruction) { return instruction->opcode.starts_with("st."); });
            ok &= expect(stores == 256 + static_cast<int>(count), "every independent store retained");
        }
    }
    return ok;
}
} // namespace

int main() {
    bool ok = imported_cases();
    ok &= raw_cases();
    ok &= declaration_ranges();
    ok &= bounded_work();
    if (ok)
        std::cout << "PTX narrow-pack contracts passed: " << checks << " checks\n";
    return ok ? 0 : 1;
}
