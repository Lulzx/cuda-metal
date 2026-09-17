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

std::string fixture(std::string_view body) {
    return R"ptx(.version 7.1
.target sm_80
.address_size 64
.visible .entry clz_probe(.param .u64 .ptr .global output,
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

bool positive(const std::string& label, const std::string& body) {
    const auto compiled = metal::compile_ptx_to_msl(fixture(body), {.entry_name = "clz_probe"});
    if (!expect(compiled.ok, label + " compiles: " + compiled.error)) return false;
    bool ok = expect(ir::verify(compiled.gpu_ir).ok && ir::verify(compiled.metal_ir).ok,
                     label + " verifies before and after legalization");
    for (const auto* module : {&compiled.gpu_ir, &compiled.metal_ir}) {
        unsigned results = 0, calls = 0;
        for (const auto& function : module->functions) {
            for (const auto& block : function.blocks) {
                for (const auto& operation : block.operations) {
                    const auto opcode = operation.attributes.find("ptx_opcode");
                    if (opcode != operation.attributes.end() && opcode->second.starts_with("clz.")) {
                        ++results;
                        ok &= expect(operation.result_types.size() == 1 &&
                                         operation.result_types.front() == ir::Type::integer(32),
                                     label + " PTX clz has a logical u32 result");
                    }
                    const auto callee = operation.attributes.find("callee");
                    if (operation.opcode == ir::OpCode::kCall &&
                        callee != operation.attributes.end() && callee->second == "clz") {
                        ++calls;
                        ok &= expect(operation.operands.front().kind == ir::OperandKind::kValue &&
                                         operation.operands.front().type == ir::Type::integer(32) &&
                                         operation.result_types.size() == 1 &&
                                         operation.result_types.front() == ir::Type::integer(32),
                                     label + " every Metal clz uses the u32 overload");
                    }
                }
            }
        }
        ok &= expect(results != 0 && calls != 0, label + " retains checked clz operations");
    }
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= positive("b32 register", "clz.b32 %answer, %low;");
    ok &= positive("b64 register", "clz.b64 %answer, %data;");
    ok &= positive("b32 low bits of wider storage", "clz.b32 %answer, %data;");
    ok &= positive("zero b32", "clz.b32 %answer, 0;");
    ok &= positive("zero b64", "clz.b64 %answer, 0;");
    ok &= positive("all-one immediate b32", "clz.b32 %answer, 4294967295;");
    ok &= positive("wide immediate b32 low bits", "clz.b32 %answer, 4294967296;");
    ok &= positive("all-one immediate b64", "clz.b64 %answer, 18446744073709551615;");
    ok &= positive("float bit container", "mov.f32 %floating, 0f80000000;\nclz.b32 %answer, %floating;");
    ok &= positive("reused arbitrary wide register",
        "mov.b64 %reused, %data;\nclz.b64 %reused, %reused;\nmov.b32 %answer, %reused;");
    ok &= positive("copied result across branch join", R"ptx(
 @%choose bra ALTERNATE;
 clz.b64 %answer, %data;
 bra JOIN;
ALTERNATE:
 mov.u32 %answer, 65;
JOIN:
)ptx");
    ok &= positive("predicated b32", "mov.u32 %answer, 77;\n@%choose clz.b32 %answer, %low;");
    ok &= positive("predicated b64", "mov.u32 %answer, 77;\n@!%choose clz.b64 %answer, %data;");
    ok &= positive("predicated incoming value and consecutive guards", R"ptx(
 mov.u32 %answer, 77;
 bra GUARD;
GUARD:
 @%choose clz.b64 %answer, %data;
 @!%choose clz.b32 %answer, %low;
)ptx");
    ok &= positive("predicated loop-carried destination", R"ptx(
 mov.u32 %answer, 77;
LOOP:
 @%choose clz.b64 %answer, %data;
 sub.u32 %choice, %choice, 1;
 setp.gt.u32 %choose, %choice, 0;
 @%choose bra LOOP;
)ptx");

    for (const std::string body : {
             "clz.b16 %answer, %small;", "clz.u32 %answer, %low;",
             "clz.u64 %answer, %data;", "clz.b64.extra %answer, %data;",
             "clz.b32 %answer;", "clz.b64 %answer, %data, %low;",
             "clz.b32 {%answer, %copy}, %low;", "clz.b32 %answer, {%low};",
             "clz.b32 %answer, [%low];", "clz.b32 %answer, %low+1;",
             "clz.b32 %answer, nonsense;", "clz.b32 %answer, 1.5;",
             "clz.b64 %answer, %small;", "clz.b64 %answer, %choose;",
             "mov.u32 %answer, 77;\n@%choose clz.b64 %answer, %data, %low;"}) {
        const auto compiled = metal::compile_ptx_to_msl(fixture(body), {.entry_name = "clz_probe"});
        ok &= expect(!compiled.ok && compiled.error.find("clz") != std::string::npos,
                     body + " refuses malformed or unsupported clz: " + compiled.error);
    }
    if (ok) std::cout << "PASS: PTX clz logical widths, copies, joins and predication\n";
    return ok ? 0 : 1;
}
