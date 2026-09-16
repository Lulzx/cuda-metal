#include "cumetal/ir/ir.h"
#include "cumetal/ir/ptx_importer.h"

#include <iostream>
#include <string>

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

// A runtime-controlled scalar copy ring is initialized entirely to zero. On
// each iteration its oldest word becomes the store offset, then a selected
// nonzero word advances one place. The nonzero offset eventually reaches the
// store; source-order discovery must not treat an unfinished {0} set as a proof.
std::string copy_ring(unsigned depth, unsigned injected_offset) {
    std::string source = R"ptx(.version 7.1
.target sm_80
.address_size 64
.visible .entry proof_ring(.param .u64 .ptr output, .param .u32 count) {
    .local .align 16 .b8 depot[512];
    .reg .b64 %base, %cell, %data, %write, %loaded, %output;
    .reg .u32 %count, %iteration;
    .reg .pred %inject, %again;
)ptx";
    source += "    .reg .b64 %word<" + std::to_string(depth + 1) + ">;\n";
    source += R"ptx(    ld.param.u64 %output, [output];
    ld.param.u32 %count, [count];
    mov.u64 %base, depot;
    add.u64 %cell, %base, 128;
    add.u64 %data, %base, 384;
    st.local.u64 [%cell], %data;
    mov.u32 %iteration, 0;
    setp.ne.u32 %inject, %count, 0;
)ptx";
    for (unsigned i = 0; i <= depth; ++i)
        source += "    mov.u64 %word" + std::to_string(i) + ", 0;\n";
    source += "LOOP:\n    add.u64 %write, %base, %word" + std::to_string(depth) + ";\n";
    source += "    st.local.u32 [%write], 7;\n";
    for (unsigned i = depth; i > 0; --i)
        source += "    mov.b64 %word" + std::to_string(i) + ", %word" + std::to_string(i - 1) + ";\n";
    source += "    selp.b64 %word0, " + std::to_string(injected_offset) + ", %word" + std::to_string(depth) +
              ", %inject;\n";
    source += R"ptx(    add.u32 %iteration, %iteration, 1;
    setp.lt.u32 %again, %iteration, %count;
    @%again bra LOOP;
    ld.local.u64 %loaded, [%cell];
    st.global.u64 [%output], %loaded;
    ret;
}
)ptx";
    return source;
}

bool has_pointer_cell_load(const cumetal::ir::Module& module) {
    for (const auto& function : module.functions)
        for (const auto& block : function.blocks)
            for (const auto& operation : block.operations) {
                // The fixture has exactly one non-parameter load. Its address
                // is always a pointer; only the loaded result is under test.
                if (operation.opcode != cumetal::ir::OpCode::kLoad)
                    continue;
                for (const auto& type : operation.result_types)
                    if (type.is_pointer())
                        return true;
            }
    return false;
}

bool rejects_false_pointer_proof(unsigned depth) {
    namespace ir = cumetal::ir;
    const auto imported = ir::import_ptx(copy_ring(depth, 128));
    const std::string label = std::to_string(depth) + "-stage copy ring";
    if (!imported.ok) {
        // Parse/undefined-register failures would not exercise the memory
        // proof. Every register is initialized and all memory accesses fit.
        return expect(imported.error.find("pointer memory proof") != std::string::npos,
                      label + " rejects the overlapping memory proof: " + imported.error);
    }
    return expect(ir::verify(imported.module).ok, label + " accepted IR verifies") &&
           expect(!has_pointer_cell_load(imported.module),
                  label + " cannot retain pointer evidence after a scalar overwrite");
}

} // namespace

int main() {
    namespace ir = cumetal::ir;
    const auto disjoint = ir::import_ptx(copy_ring(4, 64));
    bool ok = expect(disjoint.ok, "small disjoint ring retains a usable proof: " + disjoint.error);
    if (disjoint.ok) {
        ok &= expect(ir::verify(disjoint.module).ok, "small disjoint ring IR verifies");
        ok &= expect(has_pointer_cell_load(disjoint.module),
                     "small disjoint ring preserves the initialized pointer cell");
    }
    ok &= rejects_false_pointer_proof(4);
    // Exceeds the current 16 discovery rounds. A future stronger proof may
    // resolve the cycle, but must still observe the reachable scalar overwrite.
    ok &= rejects_false_pointer_proof(20);
    // Exercise the actual SSA solver's explicit failure path with the same
    // valid input. Raising the requested limit cannot loosen its default cap.
    ir::PtxImportOptions limited;
    limited.type_solver_step_limit = 1;
    const auto exhausted = ir::import_ptx(copy_ring(4, 64), limited);
    ok &= expect(!exhausted.ok &&
                     exhausted.error.find("type proof budget exhausted (limit=1)") != std::string::npos,
                 "a tighter type-proof budget rejects instead of materializing partial facts");
    limited.type_solver_step_limit = 4096;
    const auto sufficient = ir::import_ptx(copy_ring(4, 64), limited);
    ok &= expect(sufficient.ok && ir::verify(sufficient.module).ok,
                 "the same defined program verifies when its type-proof budget is sufficient");
    return ok ? 0 : 1;
}
