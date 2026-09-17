#include "cumetal/metal/lower_to_msl.h"

#include <iostream>
#include <string>

namespace {
namespace ir = cumetal::ir;
namespace metal = cumetal::metal;

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

std::string fixture(const std::string& kind, const std::string& mode = {}) {
    const bool shifted = mode == "offset-copy";
    const bool scratch_loop = mode == "helper-loop-scratch";
    const bool scratch_write = mode == "helper-dynamic-scratch" || scratch_loop;
    const bool intervening_scratch = mode == "intervening-scratch-loop";
    const bool record_write = mode == "helper-unknown-alias";
    const std::string offset = shifted ? "24" : "0";
    std::string source = R"ptx(.version 7.1
.target sm_80
.address_size 64
.const .align 8 .b8 bytes[8] = {97,0,1,127,128,254,255,19};
)ptx";
    if (intervening_scratch) source += R"ptx(.func scratch_only(.param .u64 seed) {
 .local .align 16 .b8 scratch[32];
 .reg .b64 %base, %cursor, %seed;
 .reg .b32 %iteration;
 .reg .pred %more;
 ld.param.u64 %seed, [seed];
 and.b64 %seed, %seed, 7;
 mov.u64 %base, scratch;
 add.u64 %cursor, %base, %seed;
 mov.u32 %iteration, 0;
LOOP:
 st.local.u8 [%cursor], 0;
 add.u64 %cursor, %cursor, 1;
 add.u32 %iteration, %iteration, 1;
 setp.lt.u32 %more, %iteration, 2;
 @%more bra LOOP;
 ret;
}
)ptx";
    source += ".func (.param .b32 result) read_record(.param .b64 request) {\n";
    if (intervening_scratch) source += " .param .u64 scratch_argument;\n";
    if (scratch_write) source += " .local .align 16 .b8 scratch[32];\n";
    if (scratch_write || record_write) {
        source += " .reg .b64 %scratch_base, %write_address, %write_offset;\n";
        source += " .reg .b32 %scratch_step;\n .reg .pred %scratch_more;\n";
    }
    source += R"ptx(
 .reg .b64 %base, %record, %pointer, %length;
 .reg .b32 %value;
 .reg .pred %empty;
 ld.param.b64 %base, [request];
 cvta.to.local.u64 %record, %base;
 ld.local.b64 %length, [%record+16];
)ptx";
    if (intervening_scratch) {
        // This call lies between caller initialization and the helper's field
        // reload, exercising the shared summary at the Metal proof boundary.
        source += " st.param.u64 [scratch_argument], %length;\n";
        source += " call.uni scratch_only, (scratch_argument);\n";
    }
    if (scratch_write || record_write) {
        source += " and.b64 %write_offset, %length, 7;\n";
        if (scratch_write) {
            // Every cursor starts in the helper's own allocation. The variable
            // offset prevents an exact (root, byte offset) identity proof.
            source += " mov.u64 %scratch_base, scratch;\n";
            source += " add.u64 %write_address, %scratch_base, %write_offset;\n";
        } else {
            // The same bounded dynamic write can overlap the incoming pointer
            // field at any byte from +8 through +15 and must remain a refusal.
            source += " add.u64 %write_offset, %write_offset, 8;\n";
            source += " add.u64 %write_address, %record, %write_offset;\n";
        }
        if (scratch_loop) source += " mov.u32 %scratch_step, 0;\nSCRATCH_LOOP:\n";
        source += " st.local.u8 [%write_address], 0;\n";
        if (scratch_loop) {
            // The loop carries a pointer initialized outside the loop; both
            // writes stay within scratch[0..8] for every scalar length.
            source += " add.u64 %write_address, %write_address, 1;\n";
            source += " add.u32 %scratch_step, %scratch_step, 1;\n";
            source += " setp.lt.u32 %scratch_more, %scratch_step, 2;\n @%scratch_more bra SCRATCH_LOOP;\n";
        }
    }
    source += R"ptx(
 ld.local.b64 %pointer, [%record+8];
 mov.u32 %value, 0;
 setp.eq.u64 %empty, %length, 0;
 @%empty bra DONE;
 ld.b8 %value, [%pointer];
DONE:
 st.param.b32 [result], %value;
 ret;
}
.visible .entry probe(.param .u64 .ptr .global output, .param .u32 choice) {
 .local .align 16 .b8 depot[96];
 .local .align 16 .b8 second[96];
 .reg .b64 %base, %record, %copy, %payload, %constant, %output, %other, %dynamic;
 .reg .b64 %wide_bits;
 .reg .b32 %low_bits;
 .reg .b32 %value, %choice;
 .reg .pred %p;
 .param .b64 argument;
 .param .b32 answer;
 ld.param.u64 %output, [output];
 ld.param.u32 %choice, [choice];
 setp.ne.u32 %p, %choice, 0;
 mov.u64 %base, depot;
 add.u64 %payload, %base, 80;
 st.local.u8 [%payload], 97;
 mov.u64 %constant, bytes;
 mov.u64 %other, second;
 st.local.u64 [%other+8], 9;
)ptx";
    source += " add.u64 %record, %base, " + offset + ";\n";
    source += " mov.b64 %copy, %record;\n";
    const auto pointer = kind == "private" ? "%payload" : kind == "constant" ? "%constant" : "%output";
    if (mode == "branch-conflict") {
        source += " @%p bra DEVICE;\n st.local.b64 [%copy+8], %payload;\n bra STORED;\n";
        source += "DEVICE:\n st.local.b64 [%copy+8], %output;\nSTORED:\n";
    } else if (mode != "missing") {
        source += mode == "predicated" ? " @%p " : " ";
        source += std::string(mode == "narrow-pointer-store" ? "st.local.b32" : "st.local.b64") +
            " [%copy+8], " + std::string(mode == "integer" ? "1" : pointer) + ";\n";
    }
    source += " st.local.b64 [%copy+16], " + std::string(mode == "empty" ? "0" : "1") + ";\n";
    if (mode == "partial") source += " st.local.u8 [%copy+15], 0;\n";
    if (mode == "unknown-alias") {
        source += " cvt.u64.u32 %dynamic, %choice;\n add.u64 %dynamic, %record, %dynamic;\n";
        source += " st.local.u8 [%dynamic], 0;\n";
    }
    if (mode == "truncated-overlap") {
        // Truncation removes 2^32: the byte store really overlaps field +8.
        // Treating the truncation as an identity invents a disjoint write.
        source += " mov.u64 %wide_bits, 4294967304;\n cvt.u32.u64 %low_bits, %wide_bits;\n";
        source += " cvt.u64.u32 %dynamic, %low_bits;\n add.u64 %dynamic, %copy, %dynamic;\n";
        source += " st.local.u8 [%dynamic], 0;\n";
    }
    const std::string call = " st.param.b64 [argument], %copy;\n call.uni (answer), read_record, (argument);\n"
                             " ld.param.b32 %value, [answer];\n st.global.u32 [%output], %value;\n";
    source += call;
    if (mode == "second-record" || mode == "callsite-conflict") {
        source += " st.local.b64 [%other+8], " + std::string(mode == "callsite-conflict" ? "%output" : "%payload") + ";\n";
        source += " st.local.b64 [%other+16], 1;\n mov.b64 %copy, %other;\n";
        source += call;
    }
    source += " ret;\n}\n";
    return source;
}

std::string entry_backedge_fixture() {
    // HEADER is the helper's first block. Its complete store initializes later
    // iterations, but the caller never initializes field +8 for the first read.
    return R"ptx(.version 7.1
.target sm_80
.address_size 64
.func (.param .b32 result) read_record(
 .param .b64 request, .param .b64 replacement, .param .u32 repeat) {
 .reg .b64 %raw, %record, %pointer, %raw_payload, %payload;
 .reg .b32 %value, %repeat;
 .reg .pred %again;
HEADER:
 ld.param.b64 %raw, [request];
 cvta.to.local.u64 %record, %raw;
 ld.param.b64 %raw_payload, [replacement];
 cvta.to.local.u64 %payload, %raw_payload;
 ld.param.u32 %repeat, [repeat];
 ld.local.b64 %pointer, [%record+8];
 ld.u8 %value, [%pointer];
 st.local.b64 [%record+8], %payload;
 setp.ne.u32 %again, %repeat, 0;
 @%again bra HEADER;
 st.param.b32 [result], %value;
 ret;
}
.visible .entry probe(.param .u64 .ptr .global output, .param .u32 choice) {
 .local .align 16 .b8 depot[32];
 .reg .b64 %record, %payload, %output;
 .reg .b32 %choice, %value;
 .param .b64 argument;
 .param .b64 payload_argument;
 .param .b32 repeat_argument;
 .param .b32 answer;
 ld.param.u64 %output, [output];
 ld.param.u32 %choice, [choice];
 mov.u64 %record, depot;
 add.u64 %payload, %record, 24;
 st.local.u8 [%payload], 97;
 st.param.b64 [argument], %record;
 st.param.b64 [payload_argument], %payload;
 st.param.b32 [repeat_argument], %choice;
 call.uni (answer), read_record, (argument, payload_argument, repeat_argument);
 ld.param.b32 %value, [answer];
 st.global.u32 [%output], %value;
 ret;
}
)ptx";
}

bool positive(const std::string& kind, const std::string& mode = {}) {
    const auto result = metal::compile_ptx_to_msl(fixture(kind, mode), {.entry_name = "probe"});
    const auto label = kind + "/" + mode;
    if (!expect(result.ok, label + " compiles: " + result.error)) return false;
    bool ok = expect(ir::verify(result.gpu_ir).ok && ir::verify(result.metal_ir).ok, label + " verifies");
    const auto expected = kind == "private" ? ir::AddressSpace::kPrivate :
        kind == "constant" ? ir::AddressSpace::kConstant : ir::AddressSpace::kDevice;
    unsigned pointer_fields = 0, lengths = 0;
    for (const auto& function : result.metal_ir.functions) {
        if (function.name != "read_record") continue;
        for (const auto& block : function.blocks) for (const auto& operation : block.operations) {
            const auto opcode = operation.attributes.find("ptx_opcode");
            if (operation.opcode != ir::OpCode::kLoad || opcode == operation.attributes.end() ||
                opcode->second != "ld.local.b64") continue;
            if (operation.result_types.front().is_pointer()) {
                ++pointer_fields;
                ok &= expect(operation.result_types.front().address_space == expected,
                             label + " field preserves pointee address space");
            } else {
                ++lengths;
                ok &= expect(operation.result_types.front() == ir::Type::integer(64), label + " length remains i64");
            }
            ok &= expect(operation.operands.front().type.is_pointer() &&
                operation.operands.front().type.address_space == ir::AddressSpace::kPrivate,
                label + " field storage remains private");
        }
    }
    return expect(pointer_fields == 1 && lengths == 1, label + " checks both fields") && ok;
}

bool negative(const std::string& mode) {
    const auto source = mode == "entry-backedge" ? entry_backedge_fixture() : fixture("private", mode);
    const auto result = metal::compile_ptx_to_msl(source, {.entry_name = "probe"});
    const auto diagnostic = mode == "narrow-pointer-store" ? "narrow PTX store" : "private helper pointer field proof";
    return expect(!result.ok && result.error.find(diagnostic) != std::string::npos,
                  mode + " refuses unproved private field: " + result.error);
}
} // namespace

int main() {
    bool ok = true;
    for (const std::string kind : {"private", "device", "constant"}) {
        ok &= positive(kind);
        ok &= positive(kind, "offset-copy");
    }
    ok &= positive("private", "empty");
    ok &= positive("private", "second-record");
    ok &= positive("private", "helper-dynamic-scratch");
    ok &= positive("private", "helper-loop-scratch");
    ok &= positive("private", "intervening-scratch-loop");
    for (const std::string mode : {"missing", "integer", "partial", "narrow-pointer-store", "predicated", "unknown-alias",
                                   "branch-conflict", "callsite-conflict", "truncated-overlap",
                                   "entry-backedge", "helper-unknown-alias"}) ok &= negative(mode);
    return ok ? 0 : 1;
}
