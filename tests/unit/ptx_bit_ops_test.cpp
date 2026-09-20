#include "cumetal/metal/lower_to_msl.h"

#include <iostream>
#include <string>

namespace {
bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << "\n";
    return condition;
}

bool permutation_shape(const std::string& selector, bool compact,
                       bool sign_replicating = true,
                       const std::string& a = "%a", const std::string& b = "%b") {
    namespace ir = cumetal::ir;
    const auto source = R"ptx(.version 7.1
.target sm_80
.address_size 64
.visible .entry permutation(.param .u64 .ptr .global output,
                            .param .b32 first, .param .b32 second) {
 .reg .b64 %address, %wide;
 .reg .b32 %a, %b, %selector, %answer;
 ld.param.u64 %address, [output];
 ld.param.b32 %a, [first];
 ld.param.b32 %b, [second];
 mov.u32 %selector, %tid.x;
 cvt.u64.u32 %wide, %a;
 or.b64 %wide, %wide, 18446744069414584320;
 prmt.b32 %answer, )ptx" + a + ", " + b + ", " + selector + R"ptx(;
 st.global.b32 [%address], %answer;
 ret;
}
)ptx";
    const auto compiled = cumetal::metal::compile_ptx_to_msl(source, {.entry_name = "permutation"});
    if (!expect(compiled.ok, "prmt selector " + selector + " compiles: " + compiled.error)) return false;
    bool ok = expect(ir::verify(compiled.gpu_ir).ok && ir::verify(compiled.metal_ir).ok,
                     "prmt verifies before and after legalization");
    unsigned line = 0;
    for (const auto& function : compiled.gpu_ir.functions)
        for (const auto& block : function.blocks)
            for (const auto& operation : block.operations)
                if (operation.attributes.contains("ptx_opcode") &&
                    operation.attributes.at("ptx_opcode") == "prmt.b32") line = operation.location.line;
    unsigned operations = 0;
    for (const auto& function : compiled.gpu_ir.functions) {
        for (const auto& block : function.blocks) {
            for (const auto& operation : block.operations) {
                if (line == 0 || operation.location.line != line) continue;
                ++operations;
                if (compact) {
                    ok &= expect(operation.result_types == std::vector<ir::Type>{ir::Type::integer(32)},
                                 "immediate prmt uses only 32-bit intermediate results");
                    if (operation.opcode == ir::OpCode::kShiftLeft ||
                        operation.opcode == ir::OpCode::kShiftRight) {
                        const auto& shift = operation.operands.at(1);
                        ok &= expect(shift.kind == ir::OperandKind::kImmediate &&
                                         std::stoul(shift.text) < 32,
                                     "immediate prmt uses defined constant shift distances");
                    }
                }
            }
        }
    }
    ok &= expect(compact ? (operations > 0 && operations <= 21) : operations == 66,
                 "prmt " + selector + " operation count: " + std::to_string(operations));
    if (compact && !sign_replicating) {
        // No nibble asks for sign replication, so the instruction is a byte
        // shuffle: one call, lowered to a Metal swizzle rather than the
        // fifteen-value arithmetic expansion.
        // One call, plus a conversion for any operand that is not already a
        // 32-bit value (a wider PTX register container, for instance).
        const std::size_t conversions =
            (a == "%a" ? 0u : 1u) + (b == "%b" ? 0u : 1u);
        ok &= expect(operations <= 1 + conversions,
                     "sign-free immediate prmt " + selector + " is one operation plus " +
                         std::to_string(conversions) + " conversion(s), got " +
                         std::to_string(operations));
        ok &= expect(compiled.source.find("as_type<uchar4>") != std::string::npos,
                     "sign-free immediate prmt " + selector + " emits a byte swizzle");
    }
    if (compact && sign_replicating) {
        ok &= expect(operations > 1 &&
                         compiled.source.find("as_type<uchar4>") == std::string::npos,
                     "sign-replicating prmt " + selector +
                         " keeps the arithmetic expansion");
    }
    return ok;
}
}  // namespace

int main() {
    using namespace cumetal;
    bool ok = true;
    const std::string funnel_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry funnel(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r<4>;
    ld.param.u64 %rd1, [output];
    mov.u32 %r1, %tid.x;
    shf.l.wrap.b32 %r2, 305419896, 2596069104, %r1;
    shf.r.wrap.b32 %r3, 305419896, 2596069104, %r1;
    st.global.v4.b32 [%rd1], {%r2, -1, %r3, 0};
    st.global.b8 [%rd1+15], %r2;
    st.global.b16 [%rd1+12], %r3;
    ret;
}
)ptx";
    metal::PtxToMslOptions strict_options;
    strict_options.entry_name = "funnel";
    const auto funnel = metal::compile_ptx_to_msl(funnel_ptx, strict_options);
    ok &= expect(funnel.ok, "wrapped funnel shifts and mixed literal vector stores compile: " + funnel.error);
    ok &= expect(funnel.source.find("uchar(") != std::string::npos &&
                 funnel.source.find("ushort(") != std::string::npos,
                 "narrow stores truncate wider PTX source registers");
    for (const auto* invalid_opcode : {"shf.l.clamp.b32", "shf.l.wrap.b64", "shf.l.extra.wrap.b32"}) {
        auto invalid_ptx = funnel_ptx;
        invalid_ptx.replace(invalid_ptx.find("shf.l.wrap.b32"), 14, invalid_opcode);
        ok &= expect(!metal::compile_ptx_to_msl(invalid_ptx, strict_options).ok,
                     "unsupported funnel shift variants fail");
    }
    for (const auto* tuple : {"{%r2, 0}", "{%r2, , %r3, 0}", "{%r2, 0, %r3,}"}) {
        auto invalid_ptx = funnel_ptx;
        const std::string original = "{%r2, -1, %r3, 0}";
        invalid_ptx.replace(invalid_ptx.find(original), original.size(), tuple);
        ok &= expect(!metal::compile_ptx_to_msl(invalid_ptx, strict_options).ok,
                     "malformed vector store tuples fail");
    }
    for (const auto* opcode : {"prmt.b32", "prmt.b32.f4e", "prmt.b64"}) {
        auto permute_ptx = funnel_ptx;
        permute_ptx.replace(permute_ptx.find("shf.l.wrap.b32"), 14, opcode);
        const auto compiled = metal::compile_ptx_to_msl(permute_ptx, strict_options);
        ok &= expect(compiled.ok == (std::string(opcode) == "prmt.b32"),
                     "only generic prmt.b32 is supported");
    }

    // Retained RSA selectors, every sign-copy nibble, and PTX literal spellings.
    for (const auto* selector : {"0x0123U", "0x7771U", "0x7772U", "0x7773U",
                                 "0x7770U", "0x3340U", "0x5410U", "0x7600U",
                                 "0x8888", "0x9999", "0xaaaa", "0xbbbb",
                                 "0xcccc", "0xdddd", "0xeeee", "0xffff",
                                 "0xabcd5410U", "21520", "052020", "0b0101010000010000U",
                                 "0XFFFF5410u", "+21520", "-1", "-0xABEF",
                                 "0", "18446744073709551615U"}) {
        // A selector whose low four nibbles are all below eight is a pure byte
        // shuffle; anything with bit three set still needs sign replication.
        unsigned long long value = 0;
        bool literal = true;
        try {
            std::string text = selector;
            while (!text.empty() && (text.back() == 'U' || text.back() == 'u')) text.pop_back();
            std::size_t consumed = 0;
            const bool negative = !text.empty() && text.front() == '-';
            if (negative) text.erase(text.begin());
            if (text.rfind("0b", 0) == 0 || text.rfind("0B", 0) == 0) {
                value = std::stoull(text.substr(2), &consumed, 2);
                literal = consumed + 2 == text.size();
            } else {
                value = std::stoull(text, &consumed, 0);
                literal = consumed == text.size();
            }
            if (negative) value = static_cast<unsigned long long>(-static_cast<long long>(value));
        } catch (...) {
            literal = false;
        }
        bool sign_replicating = !literal;
        for (unsigned lane = 0; lane < 4 && literal; ++lane) {
            if (((value >> (lane * 4)) & 8ull) != 0) sign_replicating = true;
        }
        ok &= permutation_shape(selector, true, sign_replicating);
    }
    ok &= permutation_shape("0x7543", true, false, "%wide", "%b");
    ok &= permutation_shape("0xfedc", true, true, "-1", "0x80000000U");
    ok &= permutation_shape("%selector", false);
    // Valid constant expressions outside the literal fast path stay generic.
    ok &= permutation_shape("(0x5400 | 0x10)", false);


    const auto bfi_module = [](const std::string& instruction) {
        return ".version 7.1\n.target sm_80\n.address_size 64\n"
               ".visible .entry insert_bits(.param .u64 input) {\n"
               ".reg .b32 %r<3>;\n.reg .b64 %rd<3>;\n.reg .pred %p1;\n"
               "ld.param.u64 %rd1, [input];\n"
               "ld.global.u32 %r1, [%rd1];\nld.global.u64 %rd2, [%rd1];\n"
               "setp.eq.u32 %p1, %r1, 0;\n" + instruction + "\nret;\n}\n";
    };
    for (const auto* instruction : {"bfi.b32 %r2, %r1, 42, 3, 13;",
                                    "bfi.b64 %rd2, %rd2, 42, %r1, 64;"}) {
        const auto result = metal::compile_ptx_to_msl(bfi_module(instruction));
        ok &= expect(result.ok, "32/64-bit bit insertion supports register and immediate operands");
        if (!result.ok) std::cerr << result.error << "\n";
    }
    for (const auto* instruction : {"bfi.b16 %r2, %r1, 0, 3, 13;",
                                    "bfi.u32 %r2, %r1, 0, 3, 13;",
                                    "bfi.b32.extra %r2, %r1, 0, 3, 13;",
                                    "bfi.b32 %r2, %r1, 0, 3;",
                                    "bfi.b32 %r2, %rd2, 0, 3, 13;",
                                    "bfi.b32 %r2, %r1, 0, %rd2, 13;",
                                    "@%p1 bfi.b32 %r2, %r1, 0, 3, 13;"}) {
        const auto result = metal::compile_ptx_to_msl(bfi_module(instruction));
        ok &= expect(!result.ok && result.error.find("bfi") != std::string::npos,
                     "unsupported bit-insertion forms and mismatched operands fail explicitly");
    }


    const auto tuple_module = [](const std::string& instruction) {
        return ".version 7.1\n.target sm_80\n.address_size 64\n"
               ".visible .entry tuple_move(.param .u64 input) {\n"
               ".reg .b64 %rd1;\n.reg .b32 %r<3>;\n.reg .b16 %rs<5>;\n.reg .pred %p1;\n"
               "ld.param.u64 %rd1, [input];\nld.global.u32 %r1, [%rd1];\n"
               "cvt.u16.u32 %rs1, %r1;\nmov.u16 %rs2, 43981;\n"
               "setp.eq.u32 %p1, %r1, 0;\n" + instruction + "\nret;\n}\n";
    };
    for (const auto* instruction : {"mov.b32 %r2, {%rs1, %rs2};",
                                    "mov.b32 {%rs3, %rs4}, %r1;",
                                    "mov.b32 {_, %rs4}, %r1;",
                                    "mov.b32 {%rs3, _}, %r1;"}) {
        const auto result = metal::compile_ptx_to_msl(tuple_module(instruction));
        ok &= expect(result.ok, "mov.b32 halfword packing/unpacking and sink lanes compile");
        if (!result.ok) std::cerr << result.error << "\n";
    }
    for (const auto* instruction : {"mov.b32 %r2, {%rs1};",
                                    "mov.b32 %r2, {%rs1, %rs2, %rs1, %rs2};",
                                    "mov.b32 %r2, {%r1, %rs2};",
                                    "mov.b32 %r2, {%rs1, _};",
                                    "mov.b32 %r2, {%rs1, 7};",
                                    "mov.b32 {_, _}, %r1;",
                                    "mov.b32 {%rs3, %rs3}, %r1;",
                                    "mov.b32 {%rs3, %rs4}, %rd1;",
                                    "@%p1 mov.b32 %r2, {%rs1, %rs2};"}) {
        const auto result = metal::compile_ptx_to_msl(tuple_module(instruction));
        ok &= expect(!result.ok && result.error.find("mov.b32") != std::string::npos,
                     "unsupported/malformed mov.b32 tuples are rejected rather than scalarized");
    }

    return ok ? 0 : 1;
}
