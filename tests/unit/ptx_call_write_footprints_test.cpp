#include "cumetal/metal/lower_to_msl.h"
#include "cumetal/ptx/parser.h"

#include <iostream>
#include <string>

namespace {
namespace ir = cumetal::ir;
namespace metal = cumetal::metal;

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

// The public issue-136 reducer saves a device pointer in depot[96..103],
// invokes write_byte on depot+16, then reloads the pointer. The generic helper
// store is intentional: its actual argument supplies the private address space.
std::string fixture(const std::string& mode, bool private_payload = false) {
    std::string source = ".version 7.1\n.target sm_80\n.address_size 64\n";
    if (mode == "unknown-call") {
        source += ".extern .func write_byte(.param .u64 address);\n";
    } else if (mode == "two-arguments" || mode == "narrow-pointer-store") {
        source += R"ptx(.func write_byte(.param .u64 address, .param .u64 second) {
 .reg .b64 %raw, %pointer, %raw_second, %global;
 ld.param.u64 %raw, [address];
 cvta.to.local.u64 %pointer, %raw;
 ld.param.u64 %raw_second, [second];
 cvta.to.global.u64 %global, %raw_second;
)ptx";
        source += mode == "two-arguments" ?
            " st.local.u8 [%pointer], 7;\n st.global.u8 [%global], 9;\n" :
            " st.local.b32 [%pointer], %global;\n";
        source += " ret;\n}\n";
    } else if (mode == "callee-unknown-offset") {
        source += R"ptx(.func write_byte(.param .u64 address, .param .u64 offset) {
 .reg .b64 %pointer, %offset;
 ld.param.u64 %pointer, [address];
 ld.param.u64 %offset, [offset];
 add.u64 %pointer, %pointer, %offset;
 st.u8 [%pointer], 7;
 ret;
}
)ptx";
    } else {
        source += R"ptx(.func write_byte(.param .u64 address) {
 .reg .b64 %pointer;
 .reg .b32 %byte;
)ptx";
        if (mode == "recursive") source += " .param .u64 next_argument;\n";
        source += " ld.param.u64 %pointer, [address];\n";
        if (mode == "read-only") source += " ld.u8 %byte, [%pointer];\n";
        else source += std::string(mode == "partial-overlap" ? " st.u32" : " st.u8") + " [%pointer], 7;\n";
        if (mode == "recursive") {
            // A non-tail recursive call must not acquire an empty summary.
            source += " st.param.u64 [next_argument], %pointer;\n"
                      " call.uni write_byte, (next_argument);\n st.u8 [%pointer], 8;\n";
        }
        source += " ret;\n}\n";
    }
    if (mode == "nested") source += R"ptx(.func wrapper(.param .u64 address) {
 .reg .b64 %pointer, %shifted;
 .param .u64 nested_argument;
 ld.param.u64 %pointer, [address];
 add.u64 %shifted, %pointer, 8;
 st.param.u64 [nested_argument], %shifted;
 call.uni write_byte, (nested_argument);
 ret;
}
)ptx";
    source += R"ptx(.visible .entry probe(.param .u64 .ptr .global input,
                     .param .u64 .ptr .global output, .param .u64 offset) {
 .local .align 16 .b8 depot[128];
 .local .align 16 .b8 other[16];
 .reg .b64 %base, %cell, %stored, %loaded, %write, %out, %offset, %aligned;
 .reg .b32 %value;
 .reg .pred %choice;
 .param .u64 argument;
 .param .u64 second_argument;
 mov.u64 %base, depot;
 ld.param.u64 %out, [output];
 ld.param.u64 %offset, [offset];
 setp.ne.u64 %choice, %offset, 0;
)ptx";
    if (mode == "overlap" || mode == "partial-overlap") {
        // Retain the normalized candidate-discovery control separately from
        // escaped-cell, which must also refuse without this extra scalar use.
        source += " or.b64 %aligned, %base, 1;\n st.local.u8 [%aligned], 0;\n";
    }
    source += " add.u64 %cell, %base, 96;\n";
    source += private_payload ?
        " add.u64 %stored, %base, 64;\n st.local.u32 [%stored], 12345;\n" :
        " ld.param.u64 %stored, [input];\n";
    source += " st.local.u64 [%cell], %stored;\n";
    if (mode == "separate-allocation") source += " mov.u64 %write, other;\n";
    else if (mode == "unknown-offset") source += " add.u64 %write, %base, %offset;\n";
    else {
        const std::string offset = mode == "neighbor-before" ? "95" :
            mode == "neighbor-after" ? "104" : mode == "nested" ? "8" :
            mode == "overlap" || mode == "escaped-cell" || mode == "staged-overlap" ? "96" :
            mode == "partial-overlap" ? "100" : mode == "narrow-pointer-store" ? "92" : "16";
        source += " add.u64 %write, %base, " + offset + ";\n";
    }
    if (mode == "read-only") source += " st.local.u8 [%write], 7;\n";
    if (mode == "ambiguous-staging") {
        source += " @%choice bra ALTERNATE;\n st.param.u64 [argument], %write;\n bra CALL;\n"
                  "ALTERNATE:\n st.param.u64 [argument], %cell;\nCALL:\n";
    } else if (mode != "missing-staging") {
        source += std::string(mode == "guarded-staging" ? " @%choice " : " ") +
            "st.param.u64 [argument], %write;\n";
    }
    // st.param captures an SSA value. Reassigning its source register before
    // the call must neither invent nor hide an overlap with the saved cell.
    if (mode == "staged-disjoint") source += " mov.b64 %write, %cell;\n";
    if (mode == "staged-overlap") source += " add.u64 %write, %base, 16;\n";
    if (mode == "two-arguments") source += " st.param.u64 [second_argument], %out;\n";
    if (mode == "narrow-pointer-store") source += " st.param.u64 [second_argument], %stored;\n";
    if (mode == "callee-unknown-offset") source += " st.param.u64 [second_argument], %offset;\n";
    source += " call.uni " + std::string(mode == "nested" ? "wrapper" : "write_byte") + ", (argument";
    if (mode == "two-arguments" || mode == "callee-unknown-offset" || mode == "narrow-pointer-store")
        source += ", second_argument";
    source += ");\n ld.local.u64 %loaded, [%cell];\n";
    source += mode == "public-reducer" ? " ld.global.u32 %value, [%loaded];\n" :
                                         " ld.u32 %value, [%loaded];\n";
    source += " st.global.u32 [%out], %value;\n ret;\n}\n";
    return source;
}

bool positive(const std::string& mode, bool private_payload = false) {
    const auto result = metal::compile_ptx_to_msl(fixture(mode, private_payload), {.entry_name = "probe"});
    const std::string label = mode + (private_payload ? "/private-payload" : "");
    if (!expect(result.ok, label + " compiles: " + result.error)) return false;
    bool ok = expect(ir::verify(result.gpu_ir).ok && ir::verify(result.metal_ir).ok, label + " verifies");
    const auto expected = private_payload ? ir::AddressSpace::kPrivate : ir::AddressSpace::kDevice;
    for (const auto* module : {&result.gpu_ir, &result.metal_ir}) {
        unsigned cells = 0;
        for (const auto& function : module->functions) {
            if (function.name != "probe") continue;
            for (const auto& block : function.blocks) for (const auto& operation : block.operations) {
                const auto opcode = operation.attributes.find("ptx_opcode");
                if (operation.opcode != ir::OpCode::kLoad || opcode == operation.attributes.end() ||
                    opcode->second != "ld.local.u64") continue;
                ++cells;
                ok &= expect(operation.result_types.size() == 1 && operation.result_types.front().is_pointer() &&
                    operation.result_types.front().address_space == expected,
                    label + " saved pointer reload has its concrete payload address space");
                ok &= expect(!operation.operands.empty() && operation.operands.front().type.is_pointer() &&
                    operation.operands.front().type.address_space == ir::AddressSpace::kPrivate,
                    label + " saved pointer cell storage remains private");
            }
        }
        ok &= expect(cells == 1, label + " checks exactly one cell reload in each verified IR");
    }
    return ok;
}

bool negative(const std::string& mode) {
    const auto source = fixture(mode);
    const auto parsed = cumetal::ptx::parse_ptx(source, {.strict = true});
    if (!expect(parsed.ok, mode + " is syntactically valid PTX: " + parsed.error)) return false;
    const auto result = metal::compile_ptx_to_msl(source, {.entry_name = "probe"});
    return expect(!result.ok && !result.error.empty() && result.source.empty(),
                  mode + " refuses translation before unsafe MSL is emitted: " + result.error);
}
} // namespace

int main() {
    bool ok = true;
    for (const std::string mode : {"public-reducer", "disjoint", "separate-allocation", "nested", "read-only",
                                   "neighbor-before", "neighbor-after", "two-arguments", "staged-disjoint"})
        ok &= positive(mode);
    ok &= positive("disjoint", true);
    for (const std::string mode : {"overlap", "partial-overlap", "escaped-cell", "unknown-offset",
                                   "callee-unknown-offset", "unknown-call", "recursive", "missing-staging",
                                   "guarded-staging", "ambiguous-staging", "staged-overlap", "narrow-pointer-store"})
        ok &= negative(mode);
    return ok ? 0 : 1;
}
