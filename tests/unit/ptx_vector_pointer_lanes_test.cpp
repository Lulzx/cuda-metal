#include "cumetal/metal/lower_to_msl.h"

#include <fstream>
#include <iterator>
#include <iostream>
#include <string>
#include <vector>

namespace {
namespace ir = cumetal::ir;
namespace metal = cumetal::metal;

bool expect(bool condition, const std::string& message) {
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

// Fixed vector destinations have independent types. Every lane is observed:
// pointer lanes are dereferenced and scalar lanes participate in arithmetic.
std::string fixture(std::size_t width, std::size_t pointer_lane, const std::string& fault = {},
                    bool reverse_blocks = false) {
    std::string source = R"ptx(.version 7.1
.target sm_80
.address_size 64
.visible .entry probe(.param .u64 .ptr .global output, .param .u32 choice) {
 .local .align 32 .b8 depot[128];
 .reg .b64 %base, %cell, %pointer, %output, %sum, %value, %alias, %index, %address;
 .reg .b64 %v<4>;
 .reg .b32 %choice;
 .reg .pred %p;
 mov.u64 %base, depot;
 add.u64 %cell, %base, 32;
 add.u64 %pointer, %base, 96;
 st.local.u64 [%pointer], 97;
 ld.param.u64 %output, [output];
 ld.param.u32 %choice, [choice];
 setp.ne.u32 %p, %choice, 0;
)ptx";
    const auto store = [&](const std::string& pointer) {
        std::string result = " st.local.v" + std::to_string(width) + ".b64 [%cell], {";
        for (std::size_t lane = 0; lane < width; ++lane) {
            if (lane)
                result += ", ";
            result += lane == pointer_lane ? pointer : std::to_string(10 + lane);
        }
        return result + "};\n";
    };
    const bool branch =
        fault == "missing" || fault == "conflicting" || fault == "predicated" || fault == "branch";
    if (branch) {
        source += " @%p bra RIGHT;\n bra LEFT;\n";
        const auto left = "LEFT:\n" + store("%pointer") + " bra JOIN;\n";
        const auto right =
            "RIGHT:\n" +
            (fault == "missing"      ? std::string{}
             : fault == "predicated" ? " @!%p" + store("%pointer")
                                     : store(fault == "conflicting" ? "%output" : "%pointer")) +
            " bra JOIN;\n";
        source += reverse_blocks ? right + left : left + right;
        source += "JOIN:\n";
    } else {
        source += store(fault == "integer" ? "1" : "%pointer");
    }
    if (fault == "partial")
        source += " st.local.u8 [%cell+" + std::to_string(pointer_lane * 8 + 7) + "], 0;\n";
    if (fault == "overwrite")
        source += " st.local.u64 [%cell+" + std::to_string(pointer_lane * 8) + "], 1;\n";
    source += " ld.local.v" + std::to_string(width) + ".b64 {";
    for (std::size_t lane = 0; lane < width; ++lane) {
        if (lane)
            source += ", ";
        source += "%v" + std::to_string(lane);
    }
    source += "}, [%cell];\n mov.u64 %sum, 0;\n";
    for (std::size_t lane = 0; lane < width; ++lane) {
        const auto value = "%v" + std::to_string(lane);
        if (lane == pointer_lane) {
            source += " mov.u64 %alias, " + value + ";\n ld.u64 %value, [%alias];\n";
            source += " add.u64 %sum, %sum, %value;\n";
        } else {
            source += " add.u64 %sum, %sum, " + value + ";\n";
        }
    }
    source += " st.global.u64 [%output], %sum;\n ret;\n}\n";
    return source;
}

bool positive(std::size_t width, std::size_t lane, bool branch, bool reversed) {
    const auto compiled = metal::compile_ptx_to_msl(fixture(width, lane, branch ? "branch" : "", reversed),
                                                    {.entry_name = "probe"});
    const auto label = "v" + std::to_string(width) + " pointer lane " + std::to_string(lane) +
                       " branch=" + std::to_string(branch) + " reversed=" + std::to_string(reversed);
    bool ok = expect(compiled.ok, label + " compiles: " + compiled.error);
    if (!compiled.ok)
        return false;
    ok &= expect(ir::verify(compiled.gpu_ir).ok && ir::verify(compiled.metal_ir).ok,
                 label + " both IR stages verify");
    unsigned pointer_count = 0, scalar_count = 0;
    const auto opcode = "ld.local.v" + std::to_string(width) + ".b64";
    for (const auto& function : compiled.gpu_ir.functions)
        for (const auto& block : function.blocks)
            for (const auto& operation : block.operations) {
                const auto original = operation.attributes.find("ptx_opcode");
                if (operation.opcode != ir::OpCode::kLoad || original == operation.attributes.end() ||
                    original->second != opcode || operation.result_types.size() != 1)
                    continue;
                const auto type = operation.result_types.front();
                if (type.is_pointer()) {
                    ++pointer_count;
                    ok &= expect(type.address_space == ir::AddressSpace::kPrivate,
                                 label + " pointer payload retains private space");
                } else {
                    ++scalar_count;
                    ok &= expect(type == ir::Type::integer(64), label + " adjacent scalar stays i64");
                }
                ok &= expect(!operation.operands.empty() && operation.operands.front().type.is_pointer() &&
                                 operation.operands.front().type.address_space == ir::AddressSpace::kPrivate,
                             label + " record address remains independently private");
            }
    return expect(pointer_count == 1 && scalar_count == width - 1,
                  label + " exactly one pointer result and all scalar results checked") &&
           ok;
}

bool negative(std::size_t width, std::size_t lane, const std::string& fault, bool reversed = false) {
    const auto compiled =
        metal::compile_ptx_to_msl(fixture(width, lane, fault, reversed), {.entry_name = "probe"});
    return expect(!compiled.ok && compiled.error.find("pointer memory proof") != std::string::npos,
                  "v" + std::to_string(width) + " lane " + std::to_string(lane) + " " + fault +
                      " refuses unproved cell contents: " + compiled.error);
}

bool replace_once(std::string& text, const std::string& before, const std::string& after) {
    const auto at = text.find(before);
    if (!expect(at != std::string::npos, "fixture replacement exists: " + before))
        return false;
    text.replace(at, before.size(), after);
    return true;
}

bool scalar_only(std::size_t width, unsigned bits = 64) {
    auto source = fixture(width, width - 1, "integer");
    if (!replace_once(source, "ld.u64 %value, [%alias];", "mov.u64 %value, %alias;"))
        return false;
    if (bits != 64 && !replace_once(source, "ld.local.v" + std::to_string(width) + ".b64",
                                   "ld.local.v" + std::to_string(width) + ".b" + std::to_string(bits)))
        return false;
    const auto compiled = metal::compile_ptx_to_msl(source, {.entry_name = "probe"});
    const auto label = "scalar-only v" + std::to_string(width);
    bool ok = expect(compiled.ok, label + " compiles without pointer promotion: " + compiled.error);
    if (!compiled.ok)
        return false;
    unsigned lanes = 0;
    for (const auto& function : compiled.gpu_ir.functions)
        for (const auto& block : function.blocks)
            for (const auto& operation : block.operations) {
                const auto opcode = operation.attributes.find("ptx_opcode");
                if (operation.opcode != ir::OpCode::kLoad || opcode == operation.attributes.end() ||
                    opcode->second != "ld.local.v" + std::to_string(width) + ".b" + std::to_string(bits))
                    continue;
                ++lanes;
                // PTX extends a narrow load into its declared b64 register;
                // the widened result must remain an integer, never a pointer.
                ok &= expect(operation.result_types.size() == 1 &&
                                 operation.result_types.front() == ir::Type::integer(64),
                             label + " loaded lane remains a widened integer");
            }
    return expect(lanes == width && ir::verify(compiled.gpu_ir).ok && ir::verify(compiled.metal_ir).ok,
                  label + " all lanes counted and both IR stages verify") &&
           ok;
}

bool malformed_tuple(std::size_t width, std::size_t destinations) {
    auto source = fixture(width, 0);
    if (!replace_once(source, "%v<4>", "%v<8>"))
        return false;
    const std::string prefix = " ld.local.v" + std::to_string(width) + ".b64 {";
    const auto begin = source.find(prefix);
    if (!expect(begin != std::string::npos, "vector load exists"))
        return false;
    const auto end = source.find('}', begin);
    if (!expect(end != std::string::npos, "vector destination tuple ends"))
        return false;
    std::string tuple;
    for (std::size_t lane = 0; lane < destinations; ++lane) {
        if (lane)
            tuple += ", ";
        tuple += "%v" + std::to_string(lane);
    }
    source.replace(begin + prefix.size(), end - begin - prefix.size(), tuple);
    const auto compiled = metal::compile_ptx_to_msl(source, {.entry_name = "probe"});
    return expect(!compiled.ok, "v" + std::to_string(width) + " rejects " + std::to_string(destinations) +
                                    " destinations: " + compiled.error);
}

bool narrow_payload(std::size_t width, unsigned bits) {
    auto source = fixture(width, width - 1);
    const auto prefix = "ld.local.v" + std::to_string(width) + ".b";
    if (!replace_once(source, prefix + "64", prefix + std::to_string(bits)))
        return false;
    const auto compiled = metal::compile_ptx_to_msl(source, {.entry_name = "probe"});
    return expect(!compiled.ok, "v" + std::to_string(width) + " b" + std::to_string(bits) +
                                    " cannot recover a full pointer from a narrow lane: " + compiled.error);
}
bool reused_scalar_cells(std::string source, bool scalar) {
    if (scalar) {
        const std::string vector = "ld.local.v2.u64 {%x, %y}, [%cell];";
        while (source.find(vector) != std::string::npos)
            if (!replace_once(source, vector, "ld.local.u64 %x, [%cell];\n ld.local.u64 %y, [%cell+8];"))
                return false;
    }
    const auto compiled = metal::compile_ptx_to_msl(source, {.entry_name = "probe"});
    if (!expect(compiled.ok, "reused local cell is scalar after generic writes: " + compiled.error)) return false;
    bool ok = expect(ir::verify(compiled.gpu_ir).ok && ir::verify(compiled.metal_ir).ok,
                     "reused local cell verifies at both IR boundaries");
    // Observing the old pointer before the overwrite is valid; dereferencing
    // the new scalar after it still requires pointer evidence and must refuse.
    if (!replace_once(source, "mul.lo.u64 %result, %x, 5;", "ld.u64 %result, [%x];")) return false;
    const auto refused = metal::compile_ptx_to_msl(source, {.entry_name = "probe"});
    ok &= expect(!refused.ok && refused.error.find("pointer memory proof") != std::string::npos,
                 "overwritten scalar cannot be dereferenced: " + refused.error);
    return ok;
}

bool invalidated_cell_join() {
    const std::string source = R"ptx(.version 7.1
.target sm_80
.address_size 64
.visible .entry cell_join(.param .u64 .ptr .global input,
                         .param .u64 .ptr .global output,
                         .param .u32 choice) {
 .local .align 8 .b8 scratch[16];
 .reg .b64 %input, %output, %cell, %loaded, %joined;
 .reg .b32 %choice, %value;
 .reg .pred %pick;
 ld.param.u64 %input, [input];
 ld.param.u64 %output, [output];
 ld.param.u32 %choice, [choice];
 setp.ne.u32 %pick, %choice, 0;
 mov.b64 %cell, scratch;
 st.local.u64 [%cell], %input;
 // Device storage is disjoint from the private pointer cell.
 st.u32 [%output+8], 0;
 ld.local.u64 %loaded, [%cell];
 @%pick bra FROM_INPUT;
 mov.b64 %joined, %loaded;
 bra JOIN;
FROM_INPUT:
 mov.b64 %joined, %input;
JOIN:
 ld.global.u32 %value, [%joined];
 st.global.u32 [%output], %value;
 ret;
}
)ptx";
    const auto compiled = metal::compile_ptx_to_msl(source, {.entry_name = "cell_join"});
    bool ok = expect(compiled.ok, "recover invalidated cell before pointer join: " + compiled.error);
    if (compiled.ok) ok &= expect(ir::verify(compiled.gpu_ir).ok && ir::verify(compiled.metal_ir).ok,
                                "recovered pointer join verifies");
    for (bool overlap : {false, true}) {
        std::string negative = source;
        if (overlap) replace_once(negative, "st.u32 [%output+8], 0;", "st.u32 [%cell+4], 0;");
        else replace_once(negative, "st.local.u64 [%cell], %input;", "st.local.u64 [%cell], 1;");
        const auto refused = metal::compile_ptx_to_msl(negative, {.entry_name = "cell_join"});
        ok &= expect(!refused.ok, "invalidated cell join retains proof refusal");
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1]);
    if (!input) return 2;
    const std::string reused((std::istreambuf_iterator<char>(input)), {});
    bool ok = invalidated_cell_join();
    ok &= reused_scalar_cells(reused, false);
    ok &= reused_scalar_cells(reused, true);
    for (std::size_t width : {2U, 4U})
        for (std::size_t lane = 0; lane < width; ++lane) {
            ok &= positive(width, lane, false, false);
            for (bool reversed : {false, true}) {
                ok &= positive(width, lane, true, reversed);
                for (const std::string fault : {"missing", "conflicting", "predicated"})
                    ok &= negative(width, lane, fault, reversed);
            }
            for (const std::string fault : {"partial", "overwrite", "integer"})
                ok &= negative(width, lane, fault);
        }
    for (std::size_t width : {2U, 4U}) {
        ok &= scalar_only(width);
        ok &= malformed_tuple(width, width - 1);
        ok &= malformed_tuple(width, width + 1);
        for (unsigned bits : {8U, 16U, 32U}) {
            ok &= narrow_payload(width, bits);
            ok &= scalar_only(width, bits);
        }
    }
    return ok ? 0 : 1;
}
