#include "cumetal/metal/lower_to_msl.h"
#include "ptx_local_zero_guards.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace ir = cumetal::ir;
namespace metal = cumetal::metal;

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

bool preflight_budget() {
    namespace detail = cumetal::ir::detail;
    detail::Instruction load;
    load.opcode = "ld.local.u64";
    load.operands = {"%length", "[record+8]"};
    detail::Instruction guard;
    guard.opcode = "setp.eq.u64";
    guard.operands = {"%empty", "%length", "0"};
    const auto run = [&](std::vector<const detail::Instruction*> instructions, std::size_t budget) {
        std::vector<detail::RawBlock> blocks(1);
        blocks.front().instructions = std::move(instructions);
        std::vector<std::unordered_map<std::string, ir::ValueId>> incoming(1), outgoing(1);
        std::vector<std::map<std::string, ir::ValueId>> arguments(1);
        std::unordered_map<const detail::Instruction*, std::vector<ir::ValueId>> results;
        std::unordered_map<std::string, detail::LocalDepot> depots{{"record", {"record", 16, 16}}};
        cumetal::ptx::EntryFunction function;
        ir::Module module;
        return detail::prove_local_zero_loads(blocks, incoming, outgoing, arguments, results,
                                             depots, function, module, {.work = budget});
    };
    bool ok = true;
    // These functions have a module depot but lack one necessary instruction
    // kind. A one-instruction allowance must suffice without entering SSA.
    for (const auto* instruction : {&load, &guard}) {
        const auto skipped = run({instruction}, 1);
        ok &= expect(skipped.loads.empty() && !skipped.budget_exhausted && skipped.work == 1,
                     "irrelevant function skips SSA indexing within its scan budget");
    }
    auto global = load;
    global.opcode = "ld.global.u64";
    const auto skipped = run({&guard, &global}, 2);
    ok &= expect(skipped.loads.empty() && !skipped.budget_exhausted && skipped.work == 2,
                 "global load cannot enroll local zero analysis");
    for (const auto budget : {1u, 2u}) {
        const auto exhausted = run({&guard, &load}, budget);
        ok &= expect(exhausted.loads.empty() && exhausted.budget_exhausted && exhausted.work == budget,
                     "preflight and SSA indexing share one cap without partial facts");
    }
    return ok;
}

std::string replace(std::string source, const std::string& from, const std::string& to) {
    const auto at = source.find(from);
    if (at == std::string::npos) throw std::runtime_error("missing fixture text: " + from);
    source.replace(at, from.size(), to);
    return source;
}

// These variants reproduce all six retained empty-slice-reducer/input.ptx
// files, including their original u32 output and optional raw u64 sentinel.
std::string fixture(bool derived, const std::string& kind) {
    std::string source = R"ptx(.version 7.1
.target sm_80
.address_size 64
.visible .entry probe(.param .u64 .ptr .global input,
                     .param .u64 .ptr .global output) {
 .local .align 16 .b8 record[16];
 .reg .b64 %record, %input, %output, %stored, %zero, %pointer, %length, %count;
 .reg .b32 %value;
 .reg .pred %empty;
 mov.u64 %record, record;
 ld.param.u64 %input, [input];
 ld.param.u64 %output, [output];
 mov.u32 %value, 0;
 st.local.v2.b64 [%record], {1, 0};
 ld.local.v2.b64 {%pointer, %length}, [%record];
 mov.b64 %count, %length;
 setp.eq.u64 %empty, %count, 0;
 @%empty bra DONE;
 ld.u8 %value, [%pointer];
DONE:
 st.global.u32 [%output], %value;
 ret;
}
)ptx";
    if (derived) {
        source = replace(source,
            " st.local.v2.b64 [%record], {1, 0};\n ld.local.v2.b64 {%pointer, %length}, [%record];\n mov.b64 %count, %length;",
            " mov.u64 %stored, 1;\n mov.u64 %zero, 0;\n st.local.v2.u64 [%record], {%stored, %zero};\n"
            " ld.local.v2.u64 {%pointer, %length}, [%record];\n min.u64 %count, %length, 8;");
    }
    if (kind == "concrete") {
        source = derived ? replace(source, "mov.u64 %stored, 1;", "mov.u64 %stored, %input;")
                         : replace(source, "[%record], {1, 0}", "[%record], {%input, 0}");
    } else if (kind == "observable") {
        source = replace(source, " setp.eq.u64 %empty, %count, 0;",
                         " st.global.u64 [%output+8], %pointer;\n setp.eq.u64 %empty, %count, 0;");
    }
    return source;
}

bool accepts(const std::string& source, const std::string& label, bool sentinel, bool observable) {
    const auto result = metal::compile_ptx_to_msl(source, {.entry_name = "probe"});
    if (!expect(result.ok, label + " compiles: " + result.error)) return false;
    bool ok = expect(ir::verify(result.gpu_ir).ok && ir::verify(result.metal_ir).ok,
                     label + " verifies before and after Metal lowering");
    unsigned result_stores = 0, sentinel_stores = 0, generic_dereferences = 0;
    for (const auto& function : result.gpu_ir.functions) {
        for (const auto& block : function.blocks) {
            for (const auto& operation : block.operations) {
                const auto attribute = operation.attributes.find("ptx_opcode");
                if (attribute == operation.attributes.end()) continue;
                const auto& opcode = attribute->second;
                if (operation.opcode == ir::OpCode::kLoad && opcode == "ld.u8") ++generic_dereferences;
                if (sentinel && operation.opcode == ir::OpCode::kLoad && opcode.starts_with("ld.local.")) {
                    // Folding the loads to constants is also valid. Any loads
                    // that remain must preserve both raw lanes as integers.
                    for (const auto& type : operation.result_types)
                        ok &= expect(type == ir::Type::integer(64), label + " retained raw lane stays i64");
                }
                if (operation.opcode != ir::OpCode::kStore || operation.operands.size() < 2) continue;
                if (opcode == "st.global.u32") {
                    ++result_stores;
                    ok &= expect(operation.operands[1].type == ir::Type::integer(32),
                                 label + " retains the original u32 result store");
                }
                if (sentinel && opcode == "st.global.u64") {
                    ++sentinel_stores;
                    ok &= expect(operation.operands[1].type == ir::Type::integer(64),
                                 label + " observable sentinel remains integer bits");
                }
            }
        }
    }
    ok &= expect(result_stores == 1, label + " preserves the result output");
    if (sentinel) {
        ok &= expect(generic_dereferences == 0, label + " removes the unreachable sentinel dereference");
        ok &= expect(sentinel_stores == unsigned(observable), label + " preserves exactly the requested sentinel output");
    }
    return ok;
}

std::string rejection(const std::string& kind) {
    auto source = fixture(false, "guarded");
    const std::string initialize = " st.local.v2.b64 [%record], {1, 0};";
    const std::string load = " ld.local.v2.b64 {%pointer, %length}, [%record];";
    if (kind == "nonzero-length") {
        source = replace(source, initialize, " st.local.v2.b64 [%record], {1, 1};");
    } else if (kind == "overwritten-length") {
        source = replace(source, load, " st.local.u64 [%record+8], 1;\n" + load);
    } else if (kind == "missing-initializer") {
        source = replace(source, initialize, "");
    } else if (kind == "predicated-initializer") {
        source = replace(source, " .reg .b32 %value;", " .reg .b64 %unknown;\n .reg .pred %maybe;\n .reg .b32 %value;");
        source = replace(source, initialize, " ld.global.u64 %unknown, [%input];\n"
            " setp.ne.u64 %maybe, %unknown, 0;\n @%maybe st.local.v2.b64 [%record], {1, 0};");
    } else if (kind == "backedge-only-zero") {
        source = replace(source, initialize, " st.local.u64 [%record], 1;");
        source = replace(source, load, "READ_SLICE:\n" + load);
        source = replace(source, " ld.u8 %value, [%pointer];",
            " ld.u8 %value, [%pointer];\n st.local.u64 [%record+8], 0;\n bra READ_SLICE;");
    } else if (kind == "partial-write") {
        source = replace(source, load, " st.local.u8 [%record+15], 1;\n" + load);
    } else if (kind == "overlapping-write") {
        source = replace(source, load, " st.local.u64 [%record+4], 4294967296;\n" + load);
    } else if (kind == "unknown-helper-clobber") {
        source = replace(source, ".visible .entry", R"ptx(.func clobber(.param .b64 address, .param .b64 offset) {
 .reg .b64 %raw, %pointer, %offset;
 ld.param.b64 %raw, [address];
 cvta.to.local.u64 %pointer, %raw;
 ld.param.b64 %offset, [offset];
 add.u64 %pointer, %pointer, %offset;
 st.local.u8 [%pointer], 1;
 ret;
}
.visible .entry)ptx");
        source = replace(source, " .reg .b32 %value;",
            " .reg .b64 %unknown;\n .param .b64 clobber_record;\n .param .b64 clobber_offset;\n .reg .b32 %value;");
        source = replace(source, load, " ld.global.u64 %unknown, [%input];\n"
            " st.param.b64 [clobber_record], %record;\n st.param.b64 [clobber_offset], %unknown;\n"
            " call.uni clobber, (clobber_record, clobber_offset);\n" + load);
    } else if (kind == "unguarded-sentinel") {
        source = replace(source, " @%empty bra DONE;\n", "");
    } else if (kind == "signed-min-negative") {
        source = replace(source, " mov.b64 %count, %length;", " min.s64 %count, %length, -1;");
    } else {
        throw std::runtime_error("unknown refusal: " + kind);
    }
    return source;
}

// st.param captures a value. Reassigning its source register before the call
// must neither invent disjointness nor discard an already disjoint object.
std::string staged_helper_fixture(bool overlap) {
    auto source = fixture(false, "observable");
    source = replace(source, ".visible .entry", R"ptx(.func write_byte(.param .b64 address) {
 .reg .b64 %raw, %pointer;
 ld.param.b64 %raw, [address];
 cvta.to.local.u64 %pointer, %raw;
 st.local.u8 [%pointer], 1;
 ret;
}
.visible .entry)ptx");
    source = replace(source, " .local .align 16 .b8 record[16];",
        " .local .align 16 .b8 record[16];\n .local .align 8 .b8 scratch[8];\n"
        " .param .b64 write_actual;\n .reg .b64 %actual, %observed;");
    const std::string record = " add.u64 %actual, %record, 8;\n";
    const std::string scratch = " mov.u64 %actual, scratch;\n";
    source = replace(source, " ld.local.v2.b64 {%pointer, %length}, [%record];",
        " st.local.u64 [scratch], 0;\n" + (overlap ? record : scratch) +
        " st.param.b64 [write_actual], %actual;\n" + (overlap ? scratch : record) +
        " call.uni write_byte, (write_actual);\n"
        // Keep the reassignment observable: the safe case reads record length
        // zero, while the refusal case reads the separately initialized scratch.
        " ld.local.u64 %observed, [%actual];\n cvt.u32.u64 %value, %observed;\n"
        " ld.local.v2.b64 {%pointer, %length}, [%record];");
    return source;
}

bool rejects_source(const std::string& source, const std::string& kind) {
    const auto result = metal::compile_ptx_to_msl(source, {.entry_name = "probe"});
    const bool proof_error = result.error.find("pointer memory proof") != std::string::npos ||
        result.error.find("pointer field proof") != std::string::npos ||
        result.error.find("operand type") != std::string::npos ||
        result.error.find("incompatible pointer") != std::string::npos;
    return expect(!result.ok && proof_error, kind + " refuses an unproved sentinel address: " + result.error);
}
} // namespace

int main() {
    bool ok = preflight_budget();
    for (const bool derived : {false, true}) {
        for (const std::string kind : {"guarded", "observable", "concrete"})
            ok &= accepts(fixture(derived, kind), std::string(derived ? "llvm7/" : "llvm21/") + kind,
                          kind != "concrete", kind == "observable");
    }
    for (const bool derived : {false, true}) {
        auto source = replace(fixture(derived, "observable"),
            "setp.eq.u64 %empty, %count, 0;", "setp.lt.u64 %empty, %count, 64;");
        ok &= accepts(source, derived ? "unsigned min8 lt64" : "direct length lt64", true, true);
    }
    for (const bool less_than : {false, true}) {
        auto source = fixture(true, "observable");
        source = replace(source, " .reg .b32 %value;", " .reg .b64 %unknown;\n .reg .b32 %value;");
        source = replace(source, " min.u64 %count, %length, 8;",
            " ld.global.u64 %unknown, [%input];\n min.u64 %count, %length, %unknown;");
        if (less_than) source = replace(source, "setp.eq.u64 %empty, %count, 0;",
                                        "setp.lt.u64 %empty, %count, 64;");
        ok &= accepts(source, less_than ? "unsigned min unknown lt64" : "unsigned min unknown eq0", true, true);
    }
    for (const std::string kind : {"nonzero-length", "overwritten-length", "missing-initializer",
                                   "predicated-initializer", "backedge-only-zero", "partial-write",
                                   "overlapping-write", "unknown-helper-clobber", "unguarded-sentinel",
                                   "signed-min-negative"}) ok &= rejects_source(rejection(kind), kind);
    ok &= accepts(staged_helper_fixture(false), "disjoint staged helper actual survives register overwrite", true, true);
    ok &= rejects_source(staged_helper_fixture(true), "overlapping staged helper actual survives register overwrite");
    return ok ? 0 : 1;
}
