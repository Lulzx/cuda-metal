#include "cumetal/metal/lower_to_msl.h"

#include <iostream>
#include <string>
#include <string_view>

namespace {
namespace ir = cumetal::ir;
namespace metal = cumetal::metal;

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

std::string fixture(std::string body, const std::string& opcode) {
    for (std::size_t at = 0; (at = body.find("OP.", at)) != std::string::npos;) {
        body.replace(at, 2, opcode);
        at += opcode.size() + 1;
    }
    return R"ptx(.version 7.1
.target sm_80
.address_size 64
.visible .entry bit_count_probe(.param .u64 .ptr .global output,
                         .param .b64 bits, .param .u32 choice) {
 .reg .b64 %address, %data, %reused;
 .reg .b32 %answer, %copy, %low, %choice;
 .reg .b16 %small;
 .reg .f32 %floating;
 .reg .pred %choose;
 ld.param.u64 %address, [output];
 ld.param.b64 %data, [bits];
 ld.param.u32 %choice, [choice];
 cvt.u32.u64 %low, %data;
 cvt.u16.u64 %small, %data;
 setp.ne.u32 %choose, %choice, 0;
)ptx" + std::string(body) + R"ptx(
 mov.b32 %copy, %answer;
 st.global.u32 [%address], %copy;
 ret;
}
)ptx";
}

bool positive(const std::string& root, const std::string& label, const std::string& body) {
    const auto compiled = metal::compile_ptx_to_msl(fixture(body, root), {.entry_name = "bit_count_probe"});
    if (!expect(compiled.ok, label + " compiles: " + compiled.error)) return false;
    bool ok = expect(ir::verify(compiled.gpu_ir).ok && ir::verify(compiled.metal_ir).ok,
                     label + " verifies before and after legalization");
    for (const auto* module : {&compiled.gpu_ir, &compiled.metal_ir}) {
        unsigned results = 0, calls = 0;
        for (const auto& function : module->functions) {
            for (const auto& block : function.blocks) {
                for (const auto& operation : block.operations) {
                    const auto opcode = operation.attributes.find("ptx_opcode");
                    if (opcode != operation.attributes.end() && opcode->second.starts_with(root + ".")) {
                        ++results;
                        ok &= expect(operation.result_types.size() == 1 &&
                                         operation.result_types.front() == ir::Type::integer(32),
                                     label + " PTX " + root + " has a logical u32 result");
                    }
                    const auto callee = operation.attributes.find("callee");
                    if (operation.opcode == ir::OpCode::kCall &&
                        callee != operation.attributes.end() && callee->second == (root == "clz" ? "clz" : "popcount")) {
                        ++calls;
                        ok &= expect(operation.operands.front().kind == ir::OperandKind::kValue &&
                                         operation.operands.front().type == ir::Type::integer(32) &&
                                         operation.result_types.size() == 1 &&
                                         operation.result_types.front() == ir::Type::integer(32),
                                     label + " every Metal bit count uses the u32 overload");
                    }
                }
            }
        }
        ok &= expect(results != 0 && calls != 0, label + " retains checked bit-count operations");
    }
    return ok;
}

}  // namespace

bool run(const std::string& root) {
    bool ok = true;
    const auto check = [&](const std::string& label, const std::string& body) {
        return positive(root, root + " " + label, body);
    };
    ok &= check("b32 register", "OP.b32 %answer, %low;");
    ok &= check("b64 register", "OP.b64 %answer, %data;");
    ok &= check("inside a trap-capable function", R"ptx(
 @%choose bra FAULT;
 OP.b64 %answer, %data;
 bra DONE;
FAULT:
 trap;
DONE:
)ptx");
    ok &= check("b32 low bits of wider storage", "OP.b32 %answer, %data;");
    ok &= check("zero b32", "OP.b32 %answer, 0;");
    ok &= check("zero b64", "OP.b64 %answer, 0;");
    ok &= check("all-one immediate b32", "OP.b32 %answer, 4294967295;");
    ok &= check("wide immediate b32 low bits", "OP.b32 %answer, 4294967296;");
    ok &= check("all-one immediate b64", "OP.b64 %answer, 18446744073709551615;");
    for (const std::string literal : {"-1", "+1", "0x8000000000000000U", "0b100000001", "0377U"})
        ok &= check("integer literal " + literal, "OP.b64 %answer, " + literal + ";");
    ok &= check("float bit container", "mov.f32 %floating, 0f80000000;\nOP.b32 %answer, %floating;");
    ok &= check("reused arbitrary wide register",
        "mov.b64 %reused, %data;\nOP.b64 %reused, %reused;\nmov.b32 %answer, %reused;");
    ok &= check("copied result across branch join", R"ptx(
 @%choose bra ALTERNATE;
 OP.b64 %answer, %data;
 bra JOIN;
ALTERNATE:
 mov.u32 %answer, 65;
JOIN:
)ptx");
    ok &= check("predicated b32", "mov.u32 %answer, 77;\n@%choose OP.b32 %answer, %low;");
    ok &= check("predicated b64", "mov.u32 %answer, 77;\n@!%choose OP.b64 %answer, %data;");
    ok &= check("predicated incoming value and consecutive guards", R"ptx(
 mov.u32 %answer, 77;
 bra GUARD;
GUARD:
 @%choose OP.b64 %answer, %data;
 @!%choose OP.b32 %answer, %low;
)ptx");
    ok &= check("predicated loop-carried destination", R"ptx(
 mov.u32 %answer, 77;
LOOP:
 @%choose OP.b64 %answer, %data;
 sub.u32 %choice, %choice, 1;
 setp.gt.u32 %choose, %choice, 0;
 @%choose bra LOOP;
)ptx");

    for (const std::string body : {
             "OP.b16 %answer, %small;", "OP.u32 %answer, %low;",
             "OP.u64 %answer, %data;", "OP.b64.extra %answer, %data;",
             "OP.b32 %answer;", "OP.b64 %answer, %data, %low;",
             "OP.b32 {%answer, %copy}, %low;", "OP.b32 %answer, {%low};",
             "OP.b32 %answer, [%low];", "OP.b32 %answer, %low+1;",
             "OP.b32 %answer, nonsense;", "OP.b32 %answer, 1.5;",
             "OP.b64 %answer, 18446744073709551616;", "OP.b64 %answer, 0x10000000000000000;",
             "OP.b64 %answer, 0b102;", "OP.b32 %answer, 089;", "OP.b32 %answer, 1UU;",
             "OP.b64 %answer, %small;", "OP.b64 %answer, %choose;",
             "mov.u32 %answer, 77;\n@%choose OP.b64 %answer, %data, %low;"}) {
        const auto compiled = metal::compile_ptx_to_msl(fixture(body, root), {.entry_name = "bit_count_probe"});
        ok &= expect(!compiled.ok && compiled.error.find(root) != std::string::npos,
                     root + " " + body + " refuses malformed or unsupported bit count: " + compiled.error);
    }
    return ok;
}

int main() {
    const bool clz = run("clz"), popc = run("popc");
    if (clz && popc) std::cout << "PASS: PTX clz/popc logical widths, copies, joins and predication\n";
    return clz && popc ? 0 : 1;
}
