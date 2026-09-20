#include "cumetal/metal/lower_to_msl.h"

#include <iostream>
#include <string>

namespace {
namespace ir = cumetal::ir;
namespace metal = cumetal::metal;

bool expect(bool condition, const std::string& message) {
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

// Retains the RSA reducer's important trigger: the slot's generic address is
// subsequently reused by a bounded scalar-load loop, after its pointer contents
// have been overwritten. That later private use cannot type the earlier payload.
std::string fixture(const std::string& kind, const std::string& fault = {}) {
    std::string source = R"ptx(.version 7.1
.target sm_80
.address_size 64
.global .align 4 .b8 literal_data[8] = {57,48,0,0,50,9,1,0};
.visible .entry probe(.param .u64 .ptr .global output,
                     .param .u64 .ptr .global input, .param .u32 choice) {
 .local .align 16 .b8 depot[176];
 .reg .b64 %base, %local, %cell, %slot, %stored, %loaded, %next, %index, %out;
 .reg .b32 %value, %scratch, %choice;
 .reg .pred %select, %again;
 mov.b64 %local, depot;
 cvta.local.u64 %base, %local;
 st.local.u64 [%base], 0;
 add.u64 %cell, %base, 64;
 cvta.to.local.u64 %slot, %cell;
)ptx";
    if (kind == "constant")
        source += " mov.u64 %stored, literal_data;\n cvta.global.u64 %stored, %stored;\n";
    else if (kind == "device")
        source += " ld.param.u64 %stored, [input];\n";
    else
        source += " add.u64 %stored, %base, 16;\n st.local.u32 [%stored], 12345;\n";
    if (fault == "mixed" || fault == "missing") {
        source += R"ptx( ld.param.u32 %choice, [choice];
 setp.eq.u32 %select, %choice, 0;
 @%select bra ALTERNATE;
 st.local.u64 [%slot], %stored;
 bra RELOAD;
ALTERNATE:
)ptx";
        if (fault == "mixed")
            source += " st.local.u64 [%slot], %base;\n";
        source += "RELOAD:\n";
    } else
        source += " st.local.u64 [%slot], %stored;\n";
    if (fault == "partial")
        source += " st.local.u32 [%slot+4], 7;\n";
    source += R"ptx( ld.local.u64 %loaded, [%slot];
 ld.u32 %value, [%loaded];
 st.local.u64 [%slot], 0;
 st.local.u64 [%slot+8], 0;
 st.local.u64 [%slot+16], 0;
 st.local.u64 [%slot+24], 0;
 st.local.u64 [%slot+32], 0;
 st.local.u64 [%slot+40], 0;
 st.local.u64 [%slot+48], 0;
 st.local.u64 [%slot+56], 0;
 or.b64 %next, %cell, 4;
 mov.u64 %index, 0;
LOOP:
 ld.u8 %scratch, [%cell];
 add.u64 %index, %index, 1;
 setp.lt.u64 %again, %index, 16;
 selp.b64 %cell, %next, %cell, %again;
 add.u64 %next, %next, 4;
 @%again bra LOOP;
 ld.param.u64 %out, [output];
 st.global.u32 [%out], %value;
 ret;
}
)ptx";
    return source;
}

bool positive(const std::string& kind, ir::AddressSpace space) {
    const auto compiled = metal::compile_ptx_to_msl(fixture(kind), {.entry_name = "probe"});
    bool ok = expect(compiled.ok, kind + " unique pointer initialization compiles: " + compiled.error);
    if (!compiled.ok)
        return false;
    ok &= expect(ir::verify(compiled.gpu_ir).ok, kind + " GPU IR verifies");
    ok &= expect(ir::verify(compiled.metal_ir).ok, kind + " Metal IR verifies");
    unsigned pointer_loads = 0;
    for (const auto& function : compiled.gpu_ir.functions)
        if (function.name == "probe")
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations)
                    if (operation.opcode == ir::OpCode::kLoad && operation.result_types.size() == 1 &&
                        operation.result_types.front().is_pointer()) {
                        ++pointer_loads;
                        ok &= expect(operation.result_types.front().address_space == space,
                                     kind + " field payload has the proven concrete space");
                        ok &= expect(
                            !operation.operands.empty() && operation.operands.front().type.is_pointer() &&
                                operation.operands.front().type.address_space == ir::AddressSpace::kPrivate,
                            kind + " field address remains private independently of its payload");
                    }
    return expect(pointer_loads == 1, kind + " checked exactly one pointer-valued field load") && ok;
}

bool negative(const std::string& fault, const std::string& diagnostic) {
    const auto compiled = metal::compile_ptx_to_msl(fixture("constant", fault), {.entry_name = "probe"});
    return expect(!compiled.ok && compiled.error.find(diagnostic) != std::string::npos,
                  fault + " rejects the actual incomplete/conflicting memory proof: " + compiled.error);
}
} // namespace

int main() {
    bool ok = positive("constant", ir::AddressSpace::kConstant);
    ok &= positive("device", ir::AddressSpace::kDevice);
    ok &= positive("private", ir::AddressSpace::kPrivate);
    ok &= negative("mixed", "conflicting local pointer memory proof");
    ok &= negative("missing", "no initializing store");
    ok &= negative("partial", "partial overlapping store");
    return ok ? 0 : 1;
}
