#include "cumetal/ir/ir.h"
#include "cumetal/ir/ptx_importer.h"
#include "cumetal/metal/lower_to_msl.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

bool expect_concrete_pointer_type(const cumetal::ir::Type& type,
                                  const std::string& context) {
    using namespace cumetal;
    bool ok = true;
    if (type.is_pointer()) {
        ok &= expect(type.address_space != ir::AddressSpace::kNone,
                     context + " has a concrete pointer address space: " + type.str());
        ok &= expect(type.pointee() != nullptr,
                     context + " has a pointer pointee");
    }
    for (const auto& element : type.elements)
        ok &= expect_concrete_pointer_type(element, context + " element");
    return ok;
}

bool expect_concrete_metal_pointer_types(const cumetal::ir::Module& module,
                                         const std::string& context) {
    bool ok = true;
    // A correct load result alone is insufficient: the old legalization left
    // generic pointers nested in earlier casts, offsets, and their operands.
    for (const auto& function : module.functions) {
        const std::string where = context + " in " + function.name;
        ok &= expect_concrete_pointer_type(function.return_type, where + " return");
        for (const auto& argument : function.arguments)
            ok &= expect_concrete_pointer_type(argument.type, where + " argument");
        if (function.kernel_abi) {
            for (const auto& argument : function.kernel_abi->arguments)
                ok &= expect_concrete_pointer_type(argument.type, where + " ABI argument");
            for (const auto& binding : function.kernel_abi->bindings)
                ok &= expect_concrete_pointer_type(binding.type, where + " ABI binding");
        }
        for (const auto& block : function.blocks) {
            for (const auto& argument : block.arguments)
                ok &= expect_concrete_pointer_type(argument.type, where + " block argument");
            for (const auto& operation : block.operations) {
                for (const auto& type : operation.result_types)
                    ok &= expect_concrete_pointer_type(type, where + " operation result");
                for (const auto& operand : operation.operands)
                    ok &= expect_concrete_pointer_type(operand.type, where + " operation operand");
            }
        }
    }
    return ok;
}

bool test_private_record_pointer_qualifiers(const std::string& source) {
    using namespace cumetal;
    bool ok = true;
    for (const std::string variant : {"explicit u64", "generic u64", "generic u8",
                                      "generic store", "generic nonzero offsets"}) {
        auto fixture = source;
        const auto replace = [&](const std::string& from, const std::string& to) {
            fixture.replace(fixture.find(from), from.size(), to);
        };
        if (variant == "generic u64") replace("ld.global.u64", "ld.u64");
        if (variant == "generic u8") {
            replace(".reg .b64 %rd<7>;", ".reg .b64 %rd<7>;\n    .reg .b32 %r0;");
            replace("ld.global.u64 %rd5, [%rd2];",
                    "ld.u8 %r0, [%rd2];\n    cvt.u64.u32 %rd5, %r0;");
        }
        if (variant == "generic store") replace("st.global.u64", "st.u64");
        if (variant == "generic nonzero offsets") {
            replace("record[24]", "record[32]");
            replace("ld.local.u64 %rd2, [%rd1];", "ld.local.u64 %rd2, [%rd1+24];");
            replace("st.local.u64 [%rd3], %rd0;", "st.local.u64 [%rd3+24], %rd0;");
            replace("ld.global.u64 %rd5, [%rd2];", "ld.u64 %rd5, [%rd2+8];");
            replace("st.global.u64 [%rd3], %rd5;", "st.u64 [%rd3+8], %rd5;");
        }
        const auto compiled = metal::compile_ptx_to_msl(fixture);
        const auto context = "private record " + variant;
        ok &= expect(compiled.ok, context + " emits Metal: " + compiled.error);
        if (!compiled.ok) continue;
        ok &= expect(ir::verify(compiled.metal_ir).ok, context + " Metal IR verifies");
        ok &= expect_concrete_metal_pointer_types(compiled.metal_ir, context);
        for (const auto& function : compiled.metal_ir.functions) {
            if (!function.is_kernel) continue;
            ok &= expect(function.kernel_abi && function.kernel_abi->arguments.size() == 3,
                         context + " retains three legacy ABI arguments");
            if (!function.kernel_abi || function.kernel_abi->arguments.size() != 3) continue;
            for (std::size_t i = 0; i < 3; ++i) {
                const auto& argument = function.kernel_abi->arguments[i];
                ok &= expect(function.arguments[i].type == ir::Type::integer(64) &&
                                 argument.kind == ir::ArgumentKind::kScalar && argument.size == 8,
                             context + " preserves raw address bits and scalar bytes ABI");
            }
        }
        unsigned device_fields = 0, scalar_fields = 0;
        for (const auto& function : compiled.metal_ir.functions) {
            if (function.name != "read_record") continue;
            for (const auto& block : function.blocks) {
                for (const auto& operation : block.operations) {
                    if (operation.opcode != ir::OpCode::kLoad ||
                        operation.result_types.size() != 1 || operation.operands.empty()) continue;
                    const auto& address = operation.operands[0].type;
                    const auto& result = operation.result_types[0];
                    if (!address.is_pointer() || address.address_space != ir::AddressSpace::kPrivate)
                        continue;
                    if (result.is_pointer()) {
                        ++device_fields;
                        ok &= expect(result.address_space == ir::AddressSpace::kDevice,
                                     context + " pointer field resolves to device storage");
                        ok &= expect(address.pointee() &&
                                         (*address.pointee() == ir::Type::integer(8) ||
                                          *address.pointee() == result),
                                     context + " pointer-field address is a byte view or resolved typed view");
                    } else if (result == ir::Type::integer(64)) {
                        ++scalar_fields;
                    }
                }
            }
        }
        ok &= expect(device_fields == 2 && scalar_fields == 1,
                     context + " preserves both pointer fields and the independent scalar field");
        if (variant == "explicit u64") {
            // Concrete semantic pointer-to-pointer views remain legal. Only
            // unresolved nested address spaces must fail the Metal IR gate.
            auto concrete_view = compiled.metal_ir;
            for (auto& function : concrete_view.functions) {
                if (function.name != "read_record" || function.arguments.empty()) continue;
                ir::ValueId next = 1;
                for (const auto& argument : function.arguments) next = std::max(next, argument.value + 1);
                for (const auto& block : function.blocks) {
                    for (const auto& argument : block.arguments) next = std::max(next, argument.value + 1);
                    for (const auto& operation : block.operations)
                        for (const auto value : operation.results) next = std::max(next, value + 1);
                }
                ir::Operation cast;
                cast.opcode = ir::OpCode::kConvert;
                cast.results = {next};
                cast.result_types = {ir::Type::pointer(
                    ir::Type::pointer(ir::Type::integer(8), ir::AddressSpace::kDevice),
                    ir::AddressSpace::kPrivate)};
                cast.operands = {ir::Operand::value_ref(function.arguments.front().value,
                                                       function.arguments.front().type)};
                cast.attributes["kind"] = "bitcast";
                auto& operations = function.blocks.front().operations;
                operations.insert(operations.end() - 1, cast);
                ok &= expect(ir::verify(concrete_view).ok,
                             "a concrete semantic pointer-to-pointer view verifies");
                operations[operations.size() - 2].result_types.front().elements.front().address_space =
                    ir::AddressSpace::kNone;
                ok &= expect(!ir::verify(concrete_view).ok,
                             "Metal IR rejects an unresolved nested pointer in a cast result");
            }
        }
    }
    for (const std::string invalid : {"literal", "truncated", "two-addresses"}) {
        auto fixture = source;
        const auto store = fixture.find("st.local.u64 [%rd3], %rd0;");
        if (invalid == "literal") {
            fixture.replace(store, std::string("st.local.u64 [%rd3], %rd0;").size(),
                            "st.local.u64 [%rd3], 1;");
        } else if (invalid == "truncated") {
            fixture.insert(store, ".reg .b32 %narrow;\n    cvt.u32.u64 %narrow, %rd0;\n"
                                  "    cvt.u64.u32 %rd0, %narrow;\n    ");
        } else {
            fixture.insert(store, "add.u64 %rd0, %rd0, %rd1;\n    ");
        }
        const auto rejected = metal::compile_ptx_to_msl(fixture);
        ok &= expect(!rejected.ok && rejected.error.find("private helper pointer field proof") != std::string::npos,
                     "private record rejects " + invalid + " scalar producer: " + rejected.error);
    }
    return ok;
}

constexpr const char* kVectorAddPtx = R"ptx(
.version 7.0
.target sm_80
.address_size 64

.visible .entry vector_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
{
    .reg .pred %p1;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE:
    ret;
}
)ptx";

bool test_instruction_result_contracts() {
    using namespace cumetal;
    bool ok = true;
    struct ConversionCase {
        const char* opcode;
        const char* source_spelling;
        const char* destination_spelling;
        ir::Type source_type;
        ir::Type result_type;
    };
    // Place the join before both definitions in source order. Materialization
    // cannot repair a conversion's result after the join has consumed its type.
    const std::vector<ConversionCase> conversions = {
        {"cvt.u64.u32", "u32", "u64", ir::Type::integer(32), ir::Type::integer(64)},
        {"cvt.u64.s32", "s32", "u64", ir::Type::integer(32), ir::Type::integer(64)},
        {"cvt.u32.u64", "u64", "u32", ir::Type::integer(64), ir::Type::integer(32)},
        {"cvt.s32.s16", "s16", "s32", ir::Type::integer(16), ir::Type::integer(32)},
        {"cvt.u32.u16", "u16", "u32", ir::Type::integer(16), ir::Type::integer(32)},
        {"cvt.rn.f32.u32", "u32", "f32", ir::Type::integer(32), ir::Type::floating(32)},
        {"cvt.rzi.u32.f32", "f32", "u32", ir::Type::floating(32), ir::Type::integer(32)},
        {"cvt.f64.f32", "f32", "f64", ir::Type::floating(32), ir::Type::floating(64)},
        {"cvt.rn.f32.f64", "f64", "f32", ir::Type::floating(64), ir::Type::floating(32)},
    };
    for (const auto& test : conversions) {
        const std::string source_type = test.source_spelling;
        const std::string destination_type = test.destination_spelling;
        const std::string opcode = test.opcode;
        const std::string source =
            ".version 7.1\n.target sm_80\n.address_size 64\n"
            ".visible .entry conversion_contract(.param .u64 .ptr output, .param ." +
            source_type + " lhs, .param ." + source_type + " rhs, .param .u32 choice) {\n"
            ".reg .b64 %rd1;\n.reg .u32 %r1;\n.reg .pred %p1;\n.reg ." +
            source_type + " %source<2>;\n.reg ." + destination_type + " %value;\n"
            "ld.param.u64 %rd1, [output];\nld.param.u32 %r1, [choice];\nld.param." +
            source_type + " %source0, [lhs];\nld.param." + source_type + " %source1, [rhs];\n"
            "setp.ne.u32 %p1, %r1, 0;\n@%p1 bra LEFT;\nbra RIGHT;\n"
            "JOIN:\nst.global." + destination_type + " [%rd1], %value;\nret;\n"
            "RIGHT:\n" + opcode + " %value, %source1;\nbra JOIN;\n"
            "LEFT:\n" + opcode + " %value, %source0;\nbra JOIN;\n}\n";
        const auto imported = ir::import_ptx(source);
        ok &= expect(imported.ok, opcode + " through reordered join: " + imported.error);
        if (!imported.ok) continue;
        ok &= expect(ir::verify(imported.module).ok, opcode + " produces verified IR");
        unsigned definitions = 0;
        bool joined = false;
        for (const auto& function : imported.module.functions) {
            for (const auto& block : function.blocks) {
                for (const auto& argument : block.arguments) {
                    if (block.name == "JOIN" && argument.name == "%value") {
                        joined = true;
                        ok &= expect(argument.type == test.result_type,
                                     opcode + " join uses destination, not source, type");
                    }
                }
                for (const auto& operation : block.operations) {
                    const auto origin = operation.attributes.find("ptx_opcode");
                    if (origin == operation.attributes.end() || origin->second != opcode) continue;
                    ++definitions;
                    ok &= expect(operation.result_types == std::vector<ir::Type>{test.result_type},
                                 opcode + " has the specified result type");
                    ok &= expect(operation.operands.size() == 1 &&
                                     operation.operands.front().type == test.source_type,
                                 opcode + " retains its independent source type");
                }
            }
        }
        ok &= expect(definitions == 2 && joined, opcode + " checks both definitions and their join");
    }

    for (const std::string opcode : {"mul.wide.u32", "mul.wide.s32", "mad.wide.u32"}) {
        const std::string operands = opcode.starts_with("mad") ? ", 4, 4294967296" : ", 4";
        const std::string source = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry wide_contract(.param .u64 .ptr output, .param .u32 index) {
.reg .b64 %rd<3>;
.reg .b32 %r<3>;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
ld.param.u32 %r1, [index];
setp.ne.u32 %p1, %r1, 0;
@%p1 bra LEFT;
bra RIGHT;
JOIN:
st.global.u64 [%rd1], %rd2;
ret;
LEFT:
)ptx" + opcode + " %rd2, %r1" + operands + ";\nbra JOIN;\n"
            "RIGHT:\nadd.u32 %r2, %r1, 1;\n" + opcode + " %rd2, %r2" + operands +
            ";\nbra JOIN;\n}\n";
        const auto imported = ir::import_ptx(source);
        ok &= expect(imported.ok, opcode + " through reordered join: " + imported.error);
        if (!imported.ok) continue;
        bool joined = false;
        unsigned definitions = 0;
        for (const auto& function : imported.module.functions)
            for (const auto& block : function.blocks) {
                for (const auto& argument : block.arguments)
                    if (block.name == "JOIN" && argument.name == "%rd2") {
                        joined = true;
                        ok &= expect(argument.type == ir::Type::integer(64),
                                     opcode + " wide join is 64 bits before emission");
                    }
                for (const auto& operation : block.operations) {
                    const auto origin = operation.attributes.find("ptx_opcode");
                    if (origin == operation.attributes.end() || origin->second != opcode) continue;
                    ++definitions;
                    ok &= expect(operation.result_types == std::vector<ir::Type>{ir::Type::integer(64)},
                                 opcode + " wide result is 64 bits");
                }
            }
        ok &= expect(joined && definitions == 2 && ir::verify(imported.module).ok,
                     opcode + " checks both wide definitions and verified joined IR");
    }

    const std::string double_predicate = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry double_predicate(.param .u64 .ptr output, .param .f64 lhs, .param .f64 rhs) {
.reg .b64 %rd1;
.reg .f64 %fd<3>;
.reg .b32 %r1;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
ld.param.f64 %fd1, [lhs];
ld.param.f64 %fd2, [rhs];
setp.lt.f64 %p1, %fd1, %fd2;
selp.u32 %r1, 1, 0, %p1;
st.global.u32 [%rd1], %r1;
ret;
}
)ptx";
    const auto compared = ir::import_ptx(double_predicate);
    ok &= expect(compared.ok, "f64 comparison produces a predicate: " + compared.error);
    bool predicate_result = false;
    if (compared.ok)
        for (const auto& function : compared.module.functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations)
                    if (operation.opcode == ir::OpCode::kCompare)
                        predicate_result |= operation.result_types ==
                            std::vector<ir::Type>{ir::Type::predicate()} &&
                            operation.operands.size() == 2 &&
                            operation.operands[0].type == ir::Type::floating(64) &&
                            operation.operands[1].type == ir::Type::floating(64);
    ok &= expect(predicate_result, "comparison result category does not follow operand f64 category");

    for (const std::string tuple : {"{%r3, %r4}", "{%r3, _}", "{_, %r3}"}) {
        const bool both_lanes = tuple == "{%r3, %r4}";
        const std::string second_store = both_lanes ? "st.global.u32 [%rd1+4], %r4;\n" : "";
        const std::string source = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry tuple_contract(.param .u64 .ptr output, .param .u64 lhs,
                              .param .u64 rhs, .param .u32 choice) {
.reg .b64 %rd<4>;
.reg .b32 %r<5>;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
ld.param.u64 %rd2, [lhs];
ld.param.u64 %rd3, [rhs];
ld.param.u32 %r1, [choice];
setp.ne.u32 %p1, %r1, 0;
@%p1 bra LEFT;
bra RIGHT;
JOIN:
st.global.u32 [%rd1], %r3;
)ptx" + second_store + "ret;\nLEFT:\nmov.b64 " + tuple +
            ", %rd2;\nbra JOIN;\nRIGHT:\nmov.b64 " + tuple + ", %rd3;\nbra JOIN;\n}\n";
        const auto imported = ir::import_ptx(source);
        ok &= expect(imported.ok, "tuple destinations through reordered join " + tuple + ": " + imported.error);
        if (!imported.ok) continue;
        unsigned joined_lanes = 0;
        for (const auto& function : imported.module.functions)
            for (const auto& block : function.blocks)
                if (block.name == "JOIN")
                    for (const auto& argument : block.arguments)
                        if (argument.name == "%r3" || argument.name == "%r4") {
                            ++joined_lanes;
                            ok &= expect(argument.type == ir::Type::integer(32),
                                         "tuple lane retains its own 32-bit result type " + tuple);
                        }
        ok &= expect(joined_lanes == (both_lanes ? 2u : 1u) && ir::verify(imported.module).ok,
                     "tuple metadata count and lane order survive normalization " + tuple);
    }
    return ok;
}

bool test_float_to_integer_contracts() {
    using namespace cumetal;
    bool ok = true;
    struct Rounding { const char* ptx; const char* mode; const char* metal; };
    const Rounding roundings[] = {
        {"rni", "0u", "rint"}, {"rzi", "1u", "trunc"},
        {"rmi", "2u", "floor"}, {"rpi", "3u", "ceil"},
    };
    for (const auto& rounding : roundings) {
        for (const bool signed_destination : {true, false}) {
            const std::string destination = signed_destination ? "s32" : "u32";
            const std::string opcode = "cvt." + std::string(rounding.ptx) + "." + destination + ".f32";
            const std::string source =
                ".version 7.1\n.target sm_80\n.address_size 64\n"
                ".visible .entry float_to_integer(.param .f32 input, .param .u64 .ptr output) {\n"
                ".reg .f32 %f1;\n.reg .b32 %r1;\n.reg .b64 %rd1;\n"
                "ld.param.f32 %f1, [input];\nld.param.u64 %rd1, [output];\n" +
                opcode + " %r1, %f1;\nst.global.b32 [%rd1], %r1;\nret;\n}\n";
            const auto compiled = metal::compile_ptx_to_msl(source);
            ok &= expect(compiled.ok, opcode + " compiles: " + compiled.error);
            if (!compiled.ok) continue;
            bool checked = false;
            for (const auto& function : compiled.gpu_ir.functions) {
                for (const auto& block : function.blocks) {
                    for (const auto& operation : block.operations) {
                        const auto spelling = operation.attributes.find("ptx_opcode");
                        if (spelling == operation.attributes.end() || spelling->second != opcode) continue;
                        const auto mode = operation.attributes.find("rounding_mode");
                        const auto sign = operation.attributes.find("signed_output");
                        checked = operation.opcode == ir::OpCode::kConvert &&
                            operation.result_types == std::vector<ir::Type>{ir::Type::integer(32)} &&
                            mode != operation.attributes.end() && mode->second == rounding.mode &&
                            (signed_destination ? sign != operation.attributes.end() && sign->second == "true"
                                                : sign == operation.attributes.end());
                    }
                }
            }
            ok &= expect(checked, opcode + " retains destination signedness and integer rounding in signless IR");
            const std::string expected = (signed_destination ? "as_type<uint>(int(" : "uint(") +
                std::string(rounding.metal) + "(";
            ok &= expect(compiled.source.find(expected) != std::string::npos,
                         opcode + " rounds, converts numerically, then retains the result bits");
        }
    }
    return ok;
}

bool test_call_return_slot_definitions() {
    using namespace cumetal;
    bool ok = true;
    const std::string helpers = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.func (.param .b32 returned) make_word() {
    st.param.b32 [returned], 11;
    ret;
}
.func (.param .b32 returned) make_other_word() {
    st.param.b32 [returned], 29;
    ret;
}
.func (.param .b64 returned) make_wide() {
    st.param.b64 [returned], 4294967297;
    ret;
}
.func (.param .f32 returned) make_float() {
    st.param.f32 [returned], 0f40800000;
    ret;
}
)ptx";
    const auto reused = ir::import_ptx(helpers + R"ptx(
.visible .entry reused_return_slot(.param .u64 .ptr output) {
    .reg .b64 %rd_output, %rd_wide, %rd_float;
    .reg .b32 %r_word;
    .param .b64 retval0;
    ld.param.u64 %rd_output, [output];
    call.uni (retval0), make_word, ();
    ld.param.b32 %r_word, [retval0];
    st.global.b32 [%rd_output], %r_word;
    call.uni (retval0), make_wide, ();
    ld.param.b64 %rd_wide, [retval0];
    st.global.b64 [%rd_output+8], %rd_wide;
    call.uni (retval0), make_float, ();
    ld.param.b64 %rd_float, [retval0];
    st.global.f32 [%rd_output+16], %rd_float;
    ret;
}
)ptx");
    ok &= expect(reused.ok, "one return slot accepts distinct successive call signatures: " + reused.error);
    if (reused.ok) {
        ok &= expect(ir::verify(reused.module).ok, "reused return slot has verified def-use types");
        unsigned matched_loads = 0;
        for (const auto& function : reused.module.functions) {
            if (!function.is_kernel) continue;
            for (const auto& block : function.blocks) {
                ir::ValueId last_call = ir::kInvalidValue;
                ir::Type last_type;
                for (const auto& operation : block.operations) {
                    if (operation.opcode == ir::OpCode::kCall && operation.results.size() == 1) {
                        last_call = operation.results.front();
                        last_type = operation.result_types.front();
                    }
                    const auto opcode = operation.attributes.find("ptx_opcode");
                    if (opcode == operation.attributes.end() || !opcode->second.starts_with("ld.param.b")) continue;
                    ok &= expect(operation.opcode == ir::OpCode::kConvert && operation.operands.size() == 1 &&
                        operation.operands.front().kind == ir::OperandKind::kValue &&
                        operation.operands.front().value == last_call &&
                        operation.result_types == std::vector<ir::Type>{last_type},
                        "return load uses its reaching call result, including f32 through b64 ABI");
                    ++matched_loads;
                }
            }
        }
        ok &= expect(matched_loads == 3, "all three reused return-slot loads are checked");
    }

    const std::string prefix = helpers + R"ptx(
.visible .entry joined_return_slot(.param .u64 .ptr output, .param .u32 choice) {
    .reg .b64 %rd_output;
    .reg .b32 %r_choice, %r_value;
    .reg .pred %p;
    .param .b64 retval0;
    ld.param.u64 %rd_output, [output];
    ld.param.u32 %r_choice, [choice];
    setp.ne.u32 %p, %r_choice, 0;
    @%p bra LEFT;
    bra RIGHT;
)ptx";
    const std::string join = R"ptx(
JOIN:
    ld.param.b32 %r_value, [retval0];
    st.global.b32 [%rd_output], %r_value;
    ret;
)ptx";
    const std::string calls = R"ptx(
LEFT:
    call.uni (retval0), make_word, ();
    bra JOIN;
RIGHT:
    call.uni (retval0), make_other_word, ();
    bra JOIN;
)ptx";
    for (const bool join_first : {true, false}) {
        const auto imported = ir::import_ptx(prefix + (join_first ? join + calls : calls + join) + "}\n");
        ok &= expect(imported.ok, "return-slot join is independent of block source order: " + imported.error);
        if (!imported.ok) continue;
        ok &= expect(ir::verify(imported.module).ok, "joined call return dominates its parameter load");
        bool reads_join = false;
        for (const auto& function : imported.module.functions) {
            if (!function.is_kernel) continue;
            for (const auto& block : function.blocks) {
                if (block.name != "JOIN") continue;
                for (const auto& argument : block.arguments) {
                    if (argument.name != "return-slot:retval0") continue;
                    for (const auto& operation : block.operations) {
                        reads_join |= operation.opcode == ir::OpCode::kConvert &&
                            operation.operands.size() == 1 &&
                            operation.operands.front().kind == ir::OperandKind::kValue &&
                            operation.operands.front().value == argument.value;
                    }
                }
            }
        }
        ok &= expect(reads_join, "ld.param consumes the SSA join of the two call results");
    }
    std::string conflicting = calls;
    conflicting.replace(conflicting.find("make_other_word"), std::string("make_other_word").size(), "make_wide");
    const auto rejected = ir::import_ptx(prefix + join + conflicting + "}\n");
    ok &= expect(!rejected.ok && rejected.error.find("conflicting PTX incoming type") != std::string::npos,
                 "incompatible reaching return-slot types remain rejected: " + rejected.error);
    std::string undefined = calls;
    const std::string right_call = "call.uni (retval0), make_other_word, ();";
    undefined.erase(undefined.find(right_call), right_call.size());
    const auto missing = ir::import_ptx(prefix + join + undefined + "}\n");
    ok &= expect(!missing.ok && missing.error.find("undefined on an incoming edge") != std::string::npos,
                 "return-slot load with an undefined incoming call remains rejected: " + missing.error);
    return ok;
}

bool test_local_pointer_memory_proof_paths() {
    using namespace cumetal;
    bool ok = true;
    const std::string head = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry pointer_cell_probe(
    .param .u64 .ptr input, .param .u64 .ptr output, .param .u32 choice
) {
    .reg .b64 %rd<8>;
    .reg .b32 %r<3>;
    .reg .pred %p0;
    .local .align 8 .b8 slot[8];
    .local .align 8 .b8 data[8];
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r0, [choice];
    mov.u64 %rd2, slot;
    mov.u64 %rd3, data;
    st.local.u32 [%rd3], 37;
    cvta.local.u64 %rd4, %rd3;
    setp.eq.u32 %p0, %r0, 0;
    @%p0 bra PRIVATE_STORE;
    bra DEVICE_STORE;
)ptx";
    const std::string join = R"ptx(
JOIN:
    ld.local.u64 %rd5, [%rd2];
    ld.u32 %r1, [%rd5];
    st.global.u32 [%rd1], %r1;
    ret;
)ptx";
    const std::vector<std::vector<unsigned>> layouts = {
        {0, 2, 1}, {1, 2, 0}, {0, 1, 2}, {1, 0, 2},
    };
    for (const std::string mode : {"mixed", "private", "device"}) {
        const std::vector<std::string> blocks = {
            "PRIVATE_STORE:\nst.local.u64 [%rd2], " + std::string(mode == "device" ? "%rd0" : "%rd4") + ";\nbra JOIN;\n",
            "DEVICE_STORE:\nst.local.u64 [%rd2], " + std::string(mode == "private" ? "%rd4" : "%rd0") + ";\nbra JOIN;\n",
            join,
        };
        for (const auto& layout : layouts) {
            std::string fixture = head;
            for (const auto block : layout) fixture += blocks[block];
            fixture += "}\n";
            const auto imported = ir::import_ptx(fixture);
            if (mode == "mixed") {
                ok &= expect(!imported.ok && imported.error.find("conflicting local pointer memory proof") != std::string::npos,
                             "mixed reaching pointer stores reject independently of block layout: " + imported.error);
            } else {
                ok &= expect(imported.ok, "uniform " + mode + " pointer stores prove every incoming path: " + imported.error);
                if (imported.ok) {
                    bool proven = false;
                    const auto expected = mode == "private" ? ir::AddressSpace::kPrivate : ir::AddressSpace::kDevice;
                    for (const auto& function : imported.module.functions)
                        for (const auto& block : function.blocks)
                            for (const auto& operation : block.operations)
                                if (operation.opcode == ir::OpCode::kLoad && operation.result_types.size() == 1 &&
                                    operation.result_types.front().is_pointer())
                                    proven |= operation.result_types.front().address_space == expected;
                    ok &= expect(proven && ir::verify(imported.module).ok, "uniform reaching stores preserve verified concrete pointer type");
                }
            }
        }
    }
    for (const std::string alternate : {"", "st.local.u32 [%rd2], 7;\n"}) {
        const auto imported = ir::import_ptx(head +
            "PRIVATE_STORE:\nst.local.u64 [%rd2], %rd4;\nbra JOIN;\n" + join +
            "DEVICE_STORE:\n" + alternate + "bra JOIN;\n}\n");
        ok &= expect(!imported.ok && imported.error.find("unsupported local pointer memory proof") != std::string::npos,
                     "missing or partial incoming initialization cannot prove a pointer cell: " + imported.error);
    }
    return ok;
}

bool test_pointer_memory_bounded_loops() {
    using namespace cumetal;
    bool ok = true;
    const auto fixture = [](int scratch_offset, bool reversed_layout, bool inverted_guard,
                            int step = 1, int limit = 16, bool bypass = false) {
        const std::string prefix = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry bounded_pointer_cell(.param .u64 .ptr output, .param .u32 choice) {
    .local .align 8 .b8 depot[512];
    .reg .b64 %base, %table, %data, %scratch, %index, %offset, %write, %loaded, %output;
    .reg .b32 %word, %choice;
    .reg .pred %done, %bypass;
    ld.param.u64 %output, [output];
    ld.param.u32 %choice, [choice];
    mov.u64 %base, depot;
    add.u64 %table, %base, 256;
    add.u64 %data, %base, 480;
    st.local.u32 [%data], 42;
    st.local.u64 [%table], %data;
    add.u64 %scratch, %base, )ptx" + std::to_string(scratch_offset) + R"ptx(;
    mov.u64 %index, 0;
    setp.ne.u32 %bypass, %choice, 0;
    bra BODY;
)ptx";
        const std::string body = R"ptx(
BODY:
    shl.b64 %offset, %index, 3;
    add.u64 %write, %scratch, %offset;
    st.local.u64 [%write], 1;
    add.u64 %index, %index, )ptx" + std::to_string(step) + ";\n" +
            (bypass ? "    @%bypass bra BODY;\n" : "") +
            "    setp." + (inverted_guard ? std::string("ne") : std::string("eq")) +
            ".u64 %done, %index, " + std::to_string(limit) + ";\n" +
            (inverted_guard ? "    @%done bra BODY;\n    bra EXIT;\n" :
                              "    @%done bra EXIT;\n    bra BODY;\n");
        const std::string exit = R"ptx(
EXIT:
    ld.local.u64 %loaded, [%table];
    ld.u32 %word, [%loaded];
    st.global.u32 [%output], %word;
    ret;
)ptx";
        return prefix + (reversed_layout ? exit + body : body + exit) + "}\n";
    };
    for (const int offset : {0, 320}) for (const bool reverse : {false, true})
        for (const bool inverted : {false, true}) {
            const auto source = fixture(offset, reverse, inverted);
            const auto imported = ir::import_ptx(source);
            ok &= expect(imported.ok, "bounded same-depot writes preserve a disjoint pointer cell: " + imported.error);
            if (imported.ok) ok &= expect(ir::verify(imported.module).ok, "bounded pointer-cell loop verifies");
            const auto compiled = metal::compile_ptx_to_msl(source);
            ok &= expect(compiled.ok, "bounded pointer-cell loop emits Metal: " + compiled.error);
        }
    for (const auto& source : {
            fixture(192, false, false), // [192,320) overlaps the pointer cell at 256.
            fixture(0, false, false, 2, 15), // The equality exit cannot bound this recurrence.
            fixture(0, false, false, 1, 16, true)}) { // A backedge bypasses the bound.
        const auto imported = ir::import_ptx(source);
        ok &= expect(!imported.ok, "overlapping or unproved loop writes cannot preserve pointer-cell evidence");
    }
    const std::string narrowed_source = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry narrowed_pointer_overwrite(.param .u64 .ptr output) {
    .local .align 8 .b8 depot[512];
    .reg .b64 %base, %cell, %data, %large, %offset, %write, %loaded, %output;
    ld.param.u64 %output, [output];
    mov.u64 %base, depot;
    add.u64 %cell, %base, 256;
    add.u64 %data, %base, 384;
    st.local.u64 [%cell], %data;
    mov.u64 %large, 65536;
    cvt.u64.u16 %offset, %large;
    add.u64 %write, %cell, %offset;
    st.local.u32 [%write], 7;
    ld.local.u64 %loaded, [%cell];
    st.global.u64 [%output], %loaded;
    ret;
}
)ptx";
    const auto narrowed = ir::import_ptx(narrowed_source);
    // The source interpretation is u16, so the actual offset is zero. A
    // name-wide summary treating cvt as a copy would miss this partial write.
    bool claims_pointer_load = false;
    if (narrowed.ok) for (const auto& function : narrowed.module.functions)
        for (const auto& block : function.blocks) for (const auto& operation : block.operations)
            if (operation.opcode == ir::OpCode::kLoad)
                for (const auto& type : operation.result_types) claims_pointer_load |= type.is_pointer();
    ok &= expect(!narrowed.ok || !claims_pointer_load,
                 "narrow conversion cannot supply a pointer type after an overlapping scalar write");
    auto predicated_source = narrowed_source;
    const std::string index_sequence = "mov.u64 %large, 65536;\n    cvt.u64.u16 %offset, %large;\n    add.u64 %write, %cell, %offset;";
    predicated_source.replace(predicated_source.find(index_sequence), index_sequence.size(),
        "mov.u64 %write, %cell;\n    mov.u32 %lane, %tid.x;\n    setp.ne.u32 %skip, %lane, 0;\n"
        "    @%skip add.u64 %write, %base, 0;");
    predicated_source.insert(predicated_source.find(".reg .b64"), ".reg .b32 %lane;\n    .reg .pred %skip;\n    ");
    const auto predicated = ir::import_ptx(predicated_source);
    claims_pointer_load = false;
    if (predicated.ok) for (const auto& function : predicated.module.functions)
        for (const auto& block : function.blocks) for (const auto& operation : block.operations)
            if (operation.opcode == ir::OpCode::kLoad)
                for (const auto& type : operation.result_types) claims_pointer_load |= type.is_pointer();
    ok &= expect(!predicated.ok || !claims_pointer_load,
                 "a skipped address update cannot hide an overlapping scalar write");
    return ok;
}

bool test_named_tuple_register_contracts() {
    using namespace cumetal;
    const std::string source = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry named_tuple(.param .u64 .ptr output) {
    .reg .b64 %address;
    .reg .b16 %half<4>;
    .reg .b32 %packed;
    ld.param.u64 %address, [output];
    mov.u16 %half0, 4660;
    mov.u16 %half1, 43981;
    mov.b32 %packed, {%half0, %half1};
    mov.b32 {%half2, %half3}, %packed;
    st.global.u16 [%address], %half2;
    st.global.u16 [%address+2], %half3;
    ret;
}
)ptx";
    const auto compiled = metal::compile_ptx_to_msl(source);
    bool ok = expect(compiled.ok, "tuple lane/container widths come from declarations, not register names: " + compiled.error);
    auto invalid = source;
    invalid.replace(invalid.find(".reg .b16 %half<4>"), std::string(".reg .b16 %half<4>").size(), ".reg .b32 %half<4>");
    for (std::size_t at = 0; (at = invalid.find("%half", at)) != std::string::npos; at += 3)
        invalid.replace(at, 5, "%rs");
    ok &= expect(!metal::compile_ptx_to_msl(invalid).ok,
                 "conventional 16-bit register names do not override 32-bit declarations in tuple operands");
    return ok;
}

bool test_call_argument_slot_evidence() {
    using namespace cumetal;
    bool ok = true;
    const std::string helpers = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.func (.param .b64 retval) scalar_identity(.param .u64 value) {
    .reg .b64 %value;
    ld.param.u64 %value, [value];
    st.param.b64 [retval], %value;
    ret;
}
.func (.param .b64 retval) pointer_read(.param .u64 .ptr input) {
    .reg .b64 %address, %value;
    ld.param.u64 %address, [input];
    ld.u64 %value, [%address];
    st.param.b64 [retval], %value;
    ret;
}
)ptx";
    const auto mixed_source = helpers + R"ptx(
.visible .entry mixed_argument_slot(
    .param .u64 scalar, .param .u64 .ptr input, .param .u64 .ptr output
) {
    .reg .b64 %scalar, %input, %output, %first, %second;
    .param .b64 argument;
    .param .b64 returned;
    ld.param.u64 %scalar, [scalar];
    ld.param.u64 %input, [input];
    ld.param.u64 %output, [output];
    st.param.b64 [argument], %scalar;
    call.uni (returned), scalar_identity, (argument);
    ld.param.b64 %first, [returned];
    st.param.b64 [argument], %input;
    call.uni (returned), pointer_read, (argument);
    ld.param.b64 %second, [returned];
    st.global.u64 [%output], %first;
    st.global.u64 [%output+8], %second;
    ret;
}
)ptx";
    const auto mixed = ir::import_ptx(mixed_source);
    ok &= expect(mixed.ok, "reused argument slot keeps distinct scalar/pointer call roles: " + mixed.error);
    bool scalar_preserved = false;
    if (mixed.ok) for (const auto& function : mixed.module.functions)
        if (function.name == "mixed_argument_slot") for (const auto& argument : function.arguments)
            if (argument.name == "scalar") scalar_preserved = argument.type == ir::Type::integer(64);
    ok &= expect(scalar_preserved, "a later pointer call does not turn the earlier scalar parameter into a pointer");
    const auto mixed_msl = metal::compile_ptx_to_msl(mixed_source);
    ok &= expect(mixed_msl.ok, "sequential scalar/pointer argument-slot reuse emits Metal: " + mixed_msl.error);

    auto overwritten_source = mixed_source;
    const std::string first_call = "call.uni (returned), scalar_identity, (argument);\n    ld.param.b64 %first, [returned];";
    overwritten_source.replace(overwritten_source.find(first_call), first_call.size(), "mov.u64 %first, %scalar;");
    const auto overwritten = ir::import_ptx(overwritten_source);
    ok &= expect(overwritten.ok, "overwritten scalar argument store does not borrow the sole pointer call's role: " + overwritten.error);
    scalar_preserved = false;
    if (overwritten.ok) for (const auto& function : overwritten.module.functions)
        if (function.name == "mixed_argument_slot") for (const auto& argument : function.arguments)
            if (argument.name == "scalar") scalar_preserved = argument.type == ir::Type::integer(64);
    ok &= expect(scalar_preserved, "dead earlier argument-slot store cannot reclassify its scalar source");
    const auto overwritten_msl = metal::compile_ptx_to_msl(overwritten_source);
    ok &= expect(overwritten_msl.ok, "overwritten scalar slot followed by a pointer call emits Metal: " + overwritten_msl.error);

    const auto stable_source = helpers + R"ptx(
.func (.param .b64 retval) forward_slot(.param .u64 input) {
    .reg .b64 %input, %first, %second;
    .param .b64 argument;
    .param .b64 returned;
    ld.param.u64 %input, [input];
    st.param.b64 [argument], %input;
    call.uni (returned), pointer_read, (argument);
    ld.param.b64 %first, [returned];
    st.param.b64 [argument], %input;
    call.uni (returned), pointer_read, (argument);
    ld.param.b64 %second, [returned];
    st.param.b64 [retval], %second;
    ret;
}
.visible .entry stable_argument_slot(.param .u64 .ptr input, .param .u64 .ptr output) {
    .reg .b64 %input, %output, %value;
    .param .b64 argument;
    .param .b64 returned;
    ld.param.u64 %input, [input];
    ld.param.u64 %output, [output];
    st.param.b64 [argument], %input;
    call.uni (returned), forward_slot, (argument);
    ld.param.b64 %value, [returned];
    st.global.u64 [%output], %value;
    ret;
}
)ptx";
    const auto stable = ir::import_ptx(stable_source);
    ok &= expect(stable.ok, "matching pointer call signatures retain forwarding evidence: " + stable.error);
    bool pointer_inferred = false;
    if (stable.ok) for (const auto& function : stable.module.functions)
        if (function.name == "forward_slot") for (const auto& argument : function.arguments)
            if (argument.name == "input") pointer_inferred = argument.type.is_pointer();
    ok &= expect(pointer_inferred, "consistent pointer-only call slots recover an unannotated forwarded pointer");
    const auto stable_msl = metal::compile_ptx_to_msl(stable_source);
    ok &= expect(stable_msl.ok, "consistent reused pointer slot emits Metal: " + stable_msl.error);
    return ok;
}

bool test_pointer_memory_call_effects() {
    using namespace cumetal;
    bool ok = true;
    const std::string helpers = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.func own_stack_write(.param .u64 .ptr input) {
    .local .align 1 .b8 own[1];
    .reg .b64 %address;
    .reg .b32 %value;
    mov.u64 %address, own;
    add.u64 %address, %address, 0;
    st.local.u8 [%address], 7;
    ld.volatile.local.u8 %value, [%address];
    ret;
}
.func read_only(.param .u64 .ptr input) {
    .reg .b64 %address;
    .reg .b32 %value;
    ld.param.u64 %address, [input];
    ld.u32 %value, [%address];
    ret;
}
.func caller_write(.param .u64 .ptr input) {
    .reg .b64 %address;
    ld.param.u64 %address, [input];
    st.u32 [%address], 7;
    ret;
}
.func nested_write(.param .u64 .ptr input) {
    .reg .b64 %address;
    .param .b64 forwarded;
    ld.param.u64 %address, [input];
    st.param.b64 [forwarded], %address;
    call.uni caller_write, (forwarded);
    ret;
}
.visible .entry pointer_cell_call(.param .u64 .ptr output, .param .u32 choice) {
    .local .align 8 .b8 slot[8];
    .local .align 4 .b8 data[4];
    .reg .b64 %slot, %data, %loaded, %output, %argument;
    .reg .b32 %word, %choice;
    .reg .pred %select;
    .param .b64 argument;
    ld.param.u64 %output, [output];
    ld.param.u32 %choice, [choice];
    mov.u64 %slot, slot;
    mov.u64 %data, data;
    st.local.u32 [%data], 42;
    st.local.u64 [%slot], %data;
)ptx";
    for (const std::string callee : {"own_stack_write", "read_only", "caller_write", "nested_write"}) {
        const bool preserves = callee == "own_stack_write" || callee == "read_only";
        // The may-write cases select an alias which can reach the pointer
        // cell. Existing name-wide escape discovery cannot identify one
        // depot for that alias; the call-effect guard must still reject it.
        const std::string argument = preserves ? "st.param.b64 [argument], %data;\n"
            : "setp.eq.u32 %select, %choice, 0;\nselp.u64 %argument, %slot, %data, %select;\nst.param.b64 [argument], %argument;\n";
        const std::string source = helpers + argument + "call.uni " + callee + R"ptx(, (argument);
    ld.local.u64 %loaded, [%slot];
    ld.u32 %word, [%loaded];
    st.global.u32 [%output], %word;
    ret;
}
)ptx";
        const auto imported = ir::import_ptx(source);
        if (preserves) {
            ok &= expect(imported.ok, "proven caller-memory-preserving helper retains pointer cell proof: " + callee + ": " + imported.error);
            if (imported.ok) ok &= expect(ir::verify(imported.module).ok, "preserved pointer-cell call IR verifies");
            const auto compiled = metal::compile_ptx_to_msl(source);
            ok &= expect(compiled.ok, "safe helper call across a proven pointer cell emits Metal: " + callee + ": " + compiled.error);
        } else {
            ok &= expect(!imported.ok && imported.error.find("intervening call") != std::string::npos &&
                             imported.error.find("may modify the cell") != std::string::npos,
                         "direct and nested writes through caller pointers invalidate the proof: " + callee + ": " + imported.error);
        }
    }
    return ok;
}

bool test_pointer_demand_alias_definitions() {
    using namespace cumetal;
    bool ok = true;
    for (const bool overwrite : {false, true}) {
        const std::string source = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.func (.param .b64 retval) pointer_alias(.param .u64 .ptr input) {
    .reg .b64 %base, %loaded, %alias, %widened;
    .reg .b32 %word;
    ld.param.u64 %base, [input];
    ld.global.u64 %loaded, [%base];
    mov.u64 %alias, %loaded;
)ptx" + std::string(overwrite ? "mov.u64 %alias, %base;\n" : "") + R"ptx(
    ld.global.u32 %word, [%alias];
    cvt.u64.u32 %widened, %word;
    st.param.b64 [retval], )ptx" + (overwrite ? "%loaded" : "%widened") + R"ptx(;
    ret;
}
.visible .entry alias_probe(.param .u64 .ptr input, .param .u64 .ptr output) {
    .reg .b64 %base, %out, %answer;
    .param .b64 argument;
    .param .b64 returned;
    ld.param.u64 %base, [input];
    ld.param.u64 %out, [output];
    st.param.b64 [argument], %base;
    call.uni (returned), pointer_alias, (argument);
    ld.param.b64 %answer, [returned];
    st.global.u64 [%out], %answer;
    ret;
}
)ptx";
        const auto imported = ir::import_ptx(source);
        ok &= expect(imported.ok, "pointer demand follows only a unique alias definition: " + imported.error);
        if (!imported.ok) continue;
        bool checked_load = false;
        for (const auto& function : imported.module.functions)
            if (function.name == "pointer_alias")
                for (const auto& block : function.blocks)
                    for (const auto& operation : block.operations)
                        if (operation.opcode == ir::OpCode::kLoad && operation.result_types.size() == 1 &&
                            (operation.result_types.front().is_pointer() || operation.result_types.front() == ir::Type::integer(64))) {
                            checked_load = true;
                            ok &= expect(overwrite ? operation.result_types.front() == ir::Type::integer(64)
                                                   : operation.result_types.front().is_pointer(),
                                         overwrite ? "overwritten alias does not contaminate the earlier scalar field load"
                                                   : "unique alias retains proven pointer field demand");
                        }
        ok &= expect(checked_load && ir::verify(imported.module).ok, "alias-demand fixture checks the field load in verified IR");
        const auto compiled = metal::compile_ptx_to_msl(source);
        ok &= expect(compiled.ok, "unique and overwritten pointer aliases emit Metal: " + compiled.error);
    }
    return ok;
}

bool test_pointer_load_address_constraints() {
    using namespace cumetal;
    bool ok = true;
    const std::string source = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.func (.param .b32 retval) local_pointer_field() {
    .local .align 8 .b8 depot[16];
    .reg .b64 %addr<4>;
    .reg .b32 %word;
    mov.u64 %addr0, depot;
    add.u64 %addr1, %addr0, 8;
    st.local.u64 [%addr0], %addr1;
    st.local.u32 [%addr1], 42;
    ld.local.u64 %addr2, [%addr0];
    ld.u32 %word, [%addr2];
    st.param.b32 [retval], %word;
    ret;
}
.visible .entry pointer_field(.param .u64 .ptr .global output) {
    .reg .b64 %out;
    .reg .b32 %value;
    .param .b32 returned;
    ld.param.u64 %out, [output];
    call.uni (returned), local_pointer_field, ();
    ld.param.b32 %value, [returned];
    st.global.u32 [%out], %value;
    ret;
}
)ptx";
    for (const bool conventional_names : {false, true}) {
        std::string fixture = source;
        if (conventional_names) {
            for (std::size_t at = 0; (at = fixture.find("%addr", at)) != std::string::npos; at += 3)
                fixture.replace(at, 5, "%rd");
        }
        const auto imported = ir::import_ptx(fixture);
        ok &= expect(imported.ok, "generic dereference preserves a private pointer field with declared register widths: " + imported.error);
        if (imported.ok) {
            bool private_pointer_load = false;
            for (const auto& function : imported.module.functions)
                if (function.name == "local_pointer_field")
                    for (const auto& block : function.blocks)
                        for (const auto& operation : block.operations)
                            if (operation.opcode == ir::OpCode::kLoad && operation.result_types.size() == 1 &&
                                operation.result_types.front().is_pointer())
                                private_pointer_load |= operation.result_types.front().address_space == ir::AddressSpace::kPrivate;
            ok &= expect(private_pointer_load, "local field load has a proven private address space, independent of register spelling");
            ok &= expect(ir::verify(imported.module).ok, "refined pointer field IR verifies");
        }
        const auto compiled = metal::compile_ptx_to_msl(fixture);
        ok &= expect(compiled.ok, "refined private pointer field emits Metal: " + compiled.error);
        if (compiled.ok)
            ok &= expect_concrete_metal_pointer_types(compiled.metal_ir,
                                                     "proven private pointer field");
    }
    auto constant_source = source;
    constant_source.insert(constant_source.find(".func"),
                           ".const .align 4 .b8 constant_data[4] = {42, 0, 0, 0};\n");
    const std::string local_address = "add.u64 %addr1, %addr0, 8;";
    constant_source.replace(constant_source.find(local_address), local_address.size(),
                            "mov.u64 %addr1, constant_data;");
    const std::string local_write = "st.local.u32 [%addr1], 42;";
    constant_source.erase(constant_source.find(local_write), local_write.size());
    const auto constant = metal::compile_ptx_to_msl(constant_source);
    ok &= expect(constant.ok, "generic dereference retains a proven constant pointer field: " + constant.error);
    if (constant.ok) {
        ok &= expect_concrete_metal_pointer_types(constant.metal_ir, "proven constant pointer field");
        bool constant_pointer_load = false;
        for (const auto& function : constant.metal_ir.functions)
            if (function.name == "local_pointer_field")
                for (const auto& block : function.blocks)
                    for (const auto& operation : block.operations)
                        if (operation.opcode == ir::OpCode::kLoad && operation.result_types.size() == 1)
                            constant_pointer_load |= operation.result_types.front().is_pointer() &&
                                operation.result_types.front().address_space == ir::AddressSpace::kConstant;
        ok &= expect(constant_pointer_load, "constant pointer field does not default to device storage");
    }
    // A generic use must not erase an independent explicit global constraint.
    // The same field cannot simultaneously prove private and device storage.
    auto conflicting = source;
    const std::string generic_load = "ld.u32 %word, [%addr2];";
    conflicting.replace(conflicting.find(generic_load), generic_load.size(),
        generic_load + "\nld.global.u32 %word, [%addr2];");
    const auto invalid = ir::import_ptx(conflicting);
    ok &= expect(!invalid.ok && invalid.error.find("conflicting proven pointer load types") != std::string::npos,
                 "explicit global address demand still rejects a proven private pointer: " + invalid.error);
    return ok;
}

bool test_mutated_aggregate_pointer_copy() {
    using namespace cumetal;
    bool ok = true;
    for (const std::string load : {"ld.local.u32", "ld.param.u32"}) {
        const std::string source = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry aggregate_copy(.param .align 4 .b8 packed[8], .param .u64 .ptr output) {
    .reg .b64 %address, %alias, %output;
    .reg .b32 %value;
    ld.param.u64 %output, [output];
    mov.b64 %address, packed;
    st.local.u32 [%address], 99;
    mov.b64 %alias, %address;
)ptx" + load + R"ptx( %value, [%alias];
    st.global.u32 [%output], %value;
    ret;
}
)ptx";
        const auto imported = ir::import_ptx(source);
        ok &= expect(imported.ok, "copy of a mutated aggregate retains its private allocation: " + load + ": " + imported.error);
        if (!imported.ok) continue;
        unsigned allocations = 0, initializations = 0;
        ir::ValueId allocation = ir::kInvalidValue, alias = ir::kInvalidValue;
        std::vector<ir::ValueId> aliases;
        bool reads_copy = false;
        for (const auto& function : imported.module.functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations) {
                    if (operation.opcode == ir::OpCode::kAlloca) {
                        ++allocations;
                        allocation = operation.results.front();
                    }
                    if (operation.opcode == ir::OpCode::kStore && operation.operands.size() == 2 &&
                        operation.operands[1].type.kind == ir::TypeKind::kAggregate) ++initializations;
                    if (operation.opcode == ir::OpCode::kConvert && operation.operands.size() == 1 &&
                        operation.operands[0].kind == ir::OperandKind::kValue && operation.operands[0].value == allocation &&
                        !operation.result_types.empty() && operation.result_types.front().is_pointer()) {
                        alias = operation.results.front();
                        aliases.push_back(alias);
                    }
                    if ((operation.opcode == ir::OpCode::kAddressSpaceCast || operation.opcode == ir::OpCode::kPointerOffset) &&
                        !operation.operands.empty() && operation.operands[0].kind == ir::OperandKind::kValue &&
                        std::find(aliases.begin(), aliases.end(), operation.operands[0].value) != aliases.end())
                        aliases.insert(aliases.end(), operation.results.begin(), operation.results.end());
                    if (operation.opcode == ir::OpCode::kLoad && !operation.operands.empty() &&
                        operation.operands[0].kind == ir::OperandKind::kValue)
                        reads_copy |= std::find(aliases.begin(), aliases.end(), operation.operands[0].value) != aliases.end();
                }
        ok &= expect(allocations == 1 && initializations == 1 && alias != ir::kInvalidValue && reads_copy,
                     "aggregate address copy reads the existing mutation without allocating or initializing again: " + load);
        ok &= expect(ir::verify(imported.module).ok, "mutated aggregate pointer copy IR verifies");
        const auto compiled = metal::compile_ptx_to_msl(source);
        ok &= expect(compiled.ok, "mutated aggregate pointer copy emits Metal: " + load + ": " + compiled.error);
    }
    return ok;
}

bool test_joined_aggregate_parameter_addresses() {
    using namespace cumetal;
    bool ok = true;
    const std::string prefix = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry joined_aggregate_parameters(
    .param .align 4 .b8 parameter_a[8],
    .param .align 4 .b8 parameter_b[8],
    .param .u64 .ptr output,
    .param .u32 choice
) {
    .reg .b64 %rd_output, %rd_selected, %rd_intermediate;
    .reg .b32 %r_choice, %r_result;
    .reg .pred %p_outer, %p_inner;
    ld.param.u64 %rd_output, [output];
    ld.param.u32 %r_choice, [choice];
    setp.eq.u32 %p_outer, %r_choice, 0;
    setp.eq.u32 %p_inner, %r_choice, 1;
    @%p_outer bra EARLY_A;
    bra DISPATCH_B;
)ptx";
    const std::string outer = R"ptx(
OUTER_JOIN:
    ld.param.u32 %r_result, [%rd_selected+4];
    st.global.u32 [%rd_output], %r_result;
    ret;
EARLY_A:
    mov.b64 %rd_selected, parameter_a;
    bra OUTER_JOIN;
)ptx";
    const std::string inner = R"ptx(
INNER_JOIN:
    mov.b64 %rd_selected, %rd_intermediate;
    bra OUTER_JOIN;
B_LEFT:
    mov.b64 %rd_intermediate, parameter_b;
    bra INNER_JOIN;
B_RIGHT:
    mov.b64 %rd_intermediate, parameter_b;
    bra INNER_JOIN;
DISPATCH_B:
    @%p_inner bra B_LEFT;
    bra B_RIGHT;
)ptx";
    // In the first ordering the outer join sees A before its B input can
    // resolve through the inner join. An early cached parameter-name alias
    // would survive resolution and incorrectly extract A on every path.
    for (const bool outer_first : {true, false}) for (const bool later_local_reuse : {false, true}) {
        std::string source = prefix + (outer_first ? outer + inner : inner + outer) + "}\n";
        if (later_local_reuse) {
            source.insert(source.find(".reg .b64"), ".local .align 4 .b8 scratch[4];\n");
            const std::string result_store = "st.global.u32 [%rd_output], %r_result;";
            source.insert(source.find(result_store) + result_store.size(),
                "\nmov.u64 %rd_selected, scratch;\nst.local.u32 [%rd_selected], 42;\n"
                "ld.local.u32 %r_result, [%rd_selected];\nst.global.u32 [%rd_output+4], %r_result;\n");
        }
        const auto imported = ir::import_ptx(source);
        const std::string context = std::string(outer_first ? "outer-first" : "inner-first") +
            (later_local_reuse ? " with later local-pointer reuse" : "");
        ok &= expect(imported.ok, context + " aggregate address joins import: " + imported.error);
        if (!imported.ok) continue;
        ok &= expect(ir::verify(imported.module).ok, context + " aggregate address joins verify");
        bool reads_joined_value = false;
        for (const auto& function : imported.module.functions) {
            for (const auto& block : function.blocks) {
                if (block.name != "OUTER_JOIN") continue;
                ir::ValueId selected = ir::kInvalidValue;
                for (const auto& argument : block.arguments) {
                    if (argument.name == "%rd_selected" && argument.type.kind == ir::TypeKind::kAggregate)
                        selected = argument.value;
                }
                for (const auto& operation : block.operations) {
                    if (operation.opcode != ir::OpCode::kAggregateExtract || operation.operands.size() != 2) continue;
                    reads_joined_value |= selected != ir::kInvalidValue &&
                        operation.operands[0].kind == ir::OperandKind::kValue &&
                        operation.operands[0].value == selected &&
                        operation.operands[1].kind == ir::OperandKind::kImmediate &&
                        operation.operands[1].text == "1";
                }
            }
        }
        ok &= expect(reads_joined_value, context + " indirect ld.param reads the joined aggregate, not a cached parameter");
    }
    const auto same_space = ir::import_ptx(R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry local_aggregate_address(
    .param .align 4 .b8 packed[8], .param .u32 choice, .param .u64 .ptr output
) {
    .reg .b64 %rd_selected, %rd_output;
    .reg .b32 %r_choice, %r_value;
    .reg .pred %p;
    ld.param.u64 %rd_output, [output];
    ld.param.u32 %r_choice, [choice];
    mov.b64 %rd_selected, packed;
    setp.ne.u32 %p, %r_choice, 0;
    @%p bra JOIN;
    cvta.local.u64 %rd_selected, %rd_selected;
JOIN:
    ld.local.u32 %r_value, [%rd_selected];
    st.global.u32 [%rd_output], %r_value;
    ret;
}
)ptx");
    ok &= expect(same_space.ok, "same-space cvta retains the aggregate allocation pointee at a join: " + same_space.error);
    if (same_space.ok) ok &= expect(ir::verify(same_space.module).ok,
        "same-space aggregate pointer join has exact incoming types");
    return ok;
}

}  // namespace

int main() {
    using namespace cumetal;
    bool ok = true;
    ok &= test_instruction_result_contracts();
    ok &= test_float_to_integer_contracts();
    ok &= test_call_return_slot_definitions();
    ok &= test_local_pointer_memory_proof_paths();
    ok &= test_pointer_memory_bounded_loops();
    ok &= test_named_tuple_register_contracts();
    ok &= test_call_argument_slot_evidence();
    ok &= test_pointer_memory_call_effects();
    ok &= test_pointer_demand_alias_definitions();
    ok &= test_pointer_load_address_constraints();
    ok &= test_mutated_aggregate_pointer_copy();
    ok &= test_joined_aggregate_parameter_addresses();

    const std::string reused_register = R"ptx(
.version 7.0
.target sm_80
.visible .entry reuse(.param .u64 .ptr input, .param .u32 index) {
.reg .b64 %rd<3>;
.reg .b32 %r<3>;
ld.param.u64 %rd1, [input];
ld.param.u32 %r1, [index];
cvt.u64.u32 %rd2, %r1;
shl.b64 %rd2, %rd2, 2;
add.u64 %rd2, %rd1, %rd2;
ld.global.u32 %r2, [%rd2];
ret;
}
)ptx";
    const auto reused = metal::compile_ptx_to_msl(reused_register);
    bool scalar_shift = false, pointer_offset = false;
    if (reused.ok) {
        for (const auto& function : reused.gpu_ir.functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations) {
                    scalar_shift |= operation.opcode == ir::OpCode::kShiftLeft &&
                        operation.result_types == std::vector<ir::Type>{ir::Type::integer(64)};
                    pointer_offset |= operation.opcode == ir::OpCode::kPointerOffset &&
                        operation.result_types.size() == 1 && operation.result_types.front().is_pointer();
                }
    }
    ok &= expect(reused.ok && scalar_shift && pointer_offset,
                 "reused register definitions retain scalar and pointer SSA types: " + reused.error);
    const std::string promoted_global = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.global .align 8 .b8 private$table[64] = {74, 75, 72, 73, 78, 79, 76, 77, 66, 67, 64, 65, 70, 71, 68, 69, 122, 123, 120, 121, 126, 127, 124, 125, 114, 115, 112, 113, 118, 119, 116, 117, 106, 107, 104, 105, 110, 111, 108, 109, 98, 99, 96, 97, 102, 103, 100, 101, 26, 27, 24, 25, 30, 31, 28, 29, 18, 19, 16, 17, 22, 23, 20, 21};
.visible .entry promoted_global(.param .u64 .ptr input, .param .u64 length, .param .u64 .ptr output) {
.reg .b64 %rd<9>;
.reg .b32 %r1;
mov.b64 %rd6, private$table;
cvta.global.u64 %rd7, %rd6;
cvta.to.global.u64 %rd1, %rd7;
ld.param.u64 %rd2, [length];
ld.param.u64 %rd3, [output];
sub.u64 %rd4, 31, %rd2;
add.u64 %rd5, %rd2, %rd1;
ld.global.u8 %r1, [%rd5];
st.global.u64 [%rd3], %rd4;
st.global.u8 [%rd3+8], %r1;
ret;
}
)ptx";
    const auto promoted_global_result = metal::compile_ptx_to_msl(promoted_global);
    ok &= expect(promoted_global_result.ok,
                 "cvta.global preserves promoted read-only storage: " + promoted_global_result.error);
    auto invalid_global_to_local = promoted_global;
    invalid_global_to_local.replace(invalid_global_to_local.find("cvta.to.global"), 14, "cvta.to.local");
    ok &= expect(!metal::compile_ptx_to_msl(invalid_global_to_local).ok,
                 "promoted global cannot be converted to private storage");
    auto actual_constant = promoted_global;
    actual_constant.replace(actual_constant.find(".global .align"), 7, ".const");
    ok &= expect(!metal::compile_ptx_to_msl(actual_constant).ok,
                 "PTX constant declarations do not get the promoted-global exception");

    const std::string promoted_helper = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.global .align 8 .b8 private$table[64] = {74, 75, 72, 73, 78, 79, 76, 77, 66, 67, 64, 65, 70, 71, 68, 69, 122, 123, 120, 121, 126, 127, 124, 125, 114, 115, 112, 113, 118, 119, 116, 117, 106, 107, 104, 105, 110, 111, 108, 109, 98, 99, 96, 97, 102, 103, 100, 101, 26, 27, 24, 25, 30, 31, 28, 29, 18, 19, 16, 17, 22, 23, 20, 21};
.func (.param .b32 retval) load_byte(.param .u64 address) {
.reg .b64 %rd<3>;
.reg .b32 %r1;
ld.param.u64 %rd1, [address];
cvta.to.global.u64 %rd2, %rd1;
ld.global.u8 %r1, [%rd2];
st.param.b32 [retval], %r1;
ret;
}
.visible .entry promoted_helper(.param .u64 .ptr input, .param .u64 length, .param .u64 .ptr output) {
.reg .b64 %rd<12>;
.reg .b32 %r<3>;
mov.b64 %rd6, private$table;
cvta.global.u64 %rd7, %rd6;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [length];
ld.param.u64 %rd3, [output];
add.u64 %rd5, %rd7, %rd2;
add.u64 %rd8, %rd1, %rd2;
{
.param .b64 arg;
.param .b32 result;
st.param.b64 [arg], %rd5;
call.uni (result), load_byte, (arg);
ld.param.b32 %r1, [result];
st.param.b64 [arg], %rd8;
call.uni (result), load_byte, (arg);
ld.param.b32 %r2, [result];
}
cvt.u64.u32 %rd9, %r1;
cvt.u64.u32 %rd10, %r2;
shl.b64 %rd10, %rd10, 32;
or.b64 %rd4, %rd9, %rd10;
st.global.u64 [%rd3], %rd4;
st.global.u8 [%rd3+8], %r1;
ret;
}
)ptx";
    const auto promoted_helper_result = metal::compile_ptx_to_msl(promoted_helper);
    ok &= expect(promoted_helper_result.ok,
                 "helper global conversion accepts device and promoted storage: " + promoted_helper_result.error);
    auto invalid_helper_constant = promoted_helper;
    invalid_helper_constant.replace(invalid_helper_constant.find(".global .align"), 7, ".const");
    const std::string global_conversion = "cvta.global.u64 %rd7, %rd6;";
    invalid_helper_constant.replace(invalid_helper_constant.find(global_conversion), global_conversion.size(),
                                    "mov.b64 %rd7, %rd6;");
    ok &= expect(!metal::compile_ptx_to_msl(invalid_helper_constant).ok,
                 "helper global conversion rejects actual PTX constant origins");
    auto mixed_constant_origins = promoted_helper;
    mixed_constant_origins.insert(mixed_constant_origins.find(".func"),
        ".const .align 8 .b8 other_constant[4] = {1, 2, 3, 4};\n");
    const std::string helper_input = "ld.param.u64 %rd1, [input];";
    mixed_constant_origins.replace(mixed_constant_origins.find(helper_input), helper_input.size(),
                                  "mov.b64 %rd1, other_constant;");
    ok &= expect(!metal::compile_ptx_to_msl(mixed_constant_origins).ok,
                 "a promoted call site does not excuse another ordinary constant origin");
    auto writes_from_helper = promoted_helper;
    writes_from_helper.insert(writes_from_helper.find("ld.global.u8 %r1, [%rd2];"),
                              "st.global.u8 [%rd2], 7;\n");
    const auto helper_write = metal::compile_ptx_to_msl(writes_from_helper);
    ok &= expect(!helper_write.ok && helper_write.error.find("read-only constant storage") != std::string::npos,
                 "a helper write cannot use the promoted-global conversion exemption");
    auto invalid_helper_local = promoted_helper;
    const std::string table_move = "mov.b64 %rd6, private$table;";
    invalid_helper_local.replace(invalid_helper_local.find(table_move), table_move.size(),
                                ".local .align 8 .b8 depot[64];\nmov.b64 %rd6, depot;");
    invalid_helper_local.replace(invalid_helper_local.find(global_conversion), global_conversion.size(),
                                "mov.b64 %rd7, %rd6;");
    ok &= expect(!metal::compile_ptx_to_msl(invalid_helper_local).ok,
                 "helper global conversion rejects private origins");
    auto invalid_helper_shared = invalid_helper_local;
    invalid_helper_shared.replace(invalid_helper_shared.find(".local .align"), 6, ".shared");
    ok &= expect(!metal::compile_ptx_to_msl(invalid_helper_shared).ok,
                 "helper global conversion rejects shared origins");
    auto mutable_global = promoted_global;
    mutable_global.insert(mutable_global.find("cvta.global.u64 %rd7"), "st.global.u8 [%rd6], 7;\n");
    const auto mutable_global_result = metal::compile_ptx_to_msl(mutable_global);
    ok &= expect(mutable_global_result.ok &&
                 !mutable_global_result.gpu_ir.attributes.contains("ptx_promoted_global:private$table"),
                 "written globals retain device storage without a promotion exemption: " + mutable_global_result.error);
    auto mutable_literal = mutable_global;
    for (std::size_t at = 0; (at = mutable_literal.find("private$table", at)) != std::string::npos; at += 16)
        mutable_literal.replace(at, 13, "__const_$literal");
    const auto mutable_literal_result = metal::compile_ptx_to_msl(mutable_literal);
    ok &= expect(mutable_literal_result.ok &&
                 !mutable_literal_result.gpu_ir.attributes.contains("ptx_promoted_global:__const_$literal"),
                 "literal-like names do not override actual write evidence");





    // A runtime scalar remains an integer when it precedes an annotated pointer
    // in commuted address addition. The subtraction exposes false promotion.
    const std::string commuted_pointer = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry commuted_pointer(
.param .u64 .ptr .global input,
.param .u64 length,
.param .u64 .ptr .global output
) {
.reg .b64 %rd<8>;
.reg .b32 %r1;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [length];
ld.param.u64 %rd3, [output];
sub.u64 %rd4, 31, %rd2;
add.u64 %rd5, %rd2, %rd1;
ld.global.u8 %r1, [%rd5];
st.global.u64 [%rd3], %rd4;
st.global.u8 [%rd3+8], %r1;
ret;
}
)ptx";
    const auto commuted_result = metal::compile_ptx_to_msl(commuted_pointer);
    ok &= expect(commuted_result.ok, "commuted pointer addition preserves scalar length: " +
                 commuted_result.error);
    std::string ordered_pointer = commuted_pointer;
    ordered_pointer.replace(ordered_pointer.find("%rd5, %rd2, %rd1"),
                            std::string("%rd5, %rd2, %rd1").size(), "%rd5, %rd1, %rd2");
    const auto ordered_result = metal::compile_ptx_to_msl(ordered_pointer);
    ok &= expect(ordered_result.ok, "ordered pointer addition preserves scalar length: " +
                 ordered_result.error);
    std::string local_commuted_pointer = commuted_pointer;
    const std::string input_load = "ld.param.u64 %rd1, [input];";
    local_commuted_pointer.replace(local_commuted_pointer.find(input_load), input_load.size(),
                                  ".local .align 8 .b8 depot[64];\nmov.u64 %rd1, depot;");
    const std::string global_load = "ld.global.u8 %r1, [%rd5];";
    local_commuted_pointer.replace(local_commuted_pointer.find(global_load), global_load.size(),
                                  "st.u8 [%rd5], 7;\nld.u8 %r1, [%rd5];");
    const auto local_commuted_result = metal::compile_ptx_to_msl(local_commuted_pointer);
    ok &= expect(local_commuted_result.ok,
                 "local symbol provenance survives commuted pointer addition: " +
                 local_commuted_result.error);
    auto alias_pointer = commuted_pointer;
    alias_pointer.replace(alias_pointer.find(".reg .b64 %rd<8>"), std::string(".reg .b64 %rd<8>").size(), ".reg .b64 %rd<9>");
    const std::string alias_load = "ld.param.u64 %rd1, [input];";
    alias_pointer.replace(alias_pointer.find(alias_load), alias_load.size(),
        alias_load + "\nmov.u64 %rd6, %rd1;\nadd.u64 %rd7, %rd6, 0;\nsub.u64 %rd8, %rd7, 0;");
    alias_pointer.replace(alias_pointer.find("%rd5, %rd2, %rd1"), std::string("%rd5, %rd2, %rd1").size(), "%rd5, %rd2, %rd8");
    const auto aliases = metal::compile_ptx_to_msl(alias_pointer);
    ok &= expect(aliases.ok, "commuted pointer survives single-definition aliases: " + aliases.error);
    for (const auto& replacement : {"add.u64 %rd5, %rd3, %rd1;",
                                   "mov.u64 %rd6, %rd1;\nmov.u64 %rd6, 7;\nadd.u64 %rd5, %rd2, %rd6;"}) {
        auto invalid = commuted_pointer;
        const std::string addition = "add.u64 %rd5, %rd2, %rd1;";
        invalid.replace(invalid.find(addition), addition.size(), replacement);
        ok &= expect(!metal::compile_ptx_to_msl(invalid).ok,
                     std::string("conflicting/reused address evidence does not waive typed validation: ") + replacement);
    }
    auto named_stack = local_commuted_pointer;
    for (std::size_t at = 0; (at = named_stack.find("%rd1", at)) != std::string::npos; at += 3)
        named_stack.replace(at, 4, "%SP");
    named_stack.insert(named_stack.find(".reg .b64"), ".reg .b64 %SP;\n");
    const auto stack_result = metal::compile_ptx_to_msl(named_stack);
    ok &= expect(stack_result.ok, "declared stack-pointer names retain provenance: " + stack_result.error);
    std::string invalid_pointer = commuted_pointer;
    invalid_pointer.replace(invalid_pointer.find("%rd4, 31, %rd2"),
                            std::string("%rd4, 31, %rd2").size(), "%rd4, 31, %rd1");
    const auto invalid_pointer_result = metal::compile_ptx_to_msl(invalid_pointer);
    ok &= expect(!invalid_pointer_result.ok &&
                 invalid_pointer_result.error.find("pointer subtraction") != std::string::npos,
                 "integer minus pointer remains rejected");



    metal::PtxToMslOptions options;
    options.entry_name = "vector_add";
    options.source_name = "vector_add.ptx";
    const metal::PtxToMslResult result =
        metal::compile_ptx_to_msl(kVectorAddPtx, options);

    if (!result.ok) {
        std::cerr << result.error << "\n";
        return 1;
    }
    ok &= expect(ir::print(result.gpu_ir).find("gpu.thread_id") != std::string::npos,
                 "PTX importer normalizes thread identity");
    ok &= expect(ir::print(result.gpu_ir).find("cond_branch") != std::string::npos,
                 "PTX importer constructs typed CFG");
    ok &= expect(ir::print(result.metal_ir).find("metal.thread_position") !=
                     std::string::npos,
                 "Metal legalization removes generic GPU builtin");
    ok &= expect(result.source.find("kernel void vector_add") != std::string::npos,
                 "typed backend emits a kernel");
    ok &= expect(result.source.find("[[buffer(0)]]") != std::string::npos,
                 "typed backend emits explicit bindings");
    ok &= expect(result.source.find("threadgroup_position_in_grid") != std::string::npos,
                 "typed backend emits Metal threadgroup builtin");
    ok &= expect(result.source.find("if (") != std::string::npos,
                 "simple forward branch is structurized");
    ok &= expect(result.source.find("reinterpret_cast<device cm_alias_float*>") !=
                     std::string::npos,
                 "typed backend emits checked pointer casts");

    const std::string undefined_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry bad() {
    add.u32 %r1, %r2, 1;
    ret;
}
)ptx";
    const metal::PtxToMslResult undefined =
        metal::compile_ptx_to_msl(undefined_ptx);
    ok &= expect(!undefined.ok &&
                     undefined.error.find("used before definition") != std::string::npos,
                 "undefined PTX registers fail before MSL emission");

    const std::string explicit_implicit_def_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry explicit_implicit_def() {
    .reg .b32 %r<3>;
    // implicit-def: %r1
    add.u32 %r2, %r1, 1;
    ret;
}
)ptx";
    const metal::PtxToMslResult explicit_implicit_def =
        metal::compile_ptx_to_msl(explicit_implicit_def_ptx);
    ok &= expect(explicit_implicit_def.ok,
                 "compiler-emitted PTX implicit-def markers receive a valid refinement");

    const std::string loop_join_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry loop_join(
    .param .u64 output,
    .param .u32 count
) {
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r1, [count];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra DONE;
    mov.u32 %r2, 0;
LOOP:
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p2, %r2, %r1;
    @%p2 bra LOOP;
DONE:
    st.global.u32 [%rd1], %r1;
    ret;
}
)ptx";
    metal::PtxToMslOptions loop_options;
    loop_options.entry_name = "loop_join";
    loop_options.source_name = "loop_join.ptx";
    const metal::PtxToMslResult loop_join =
        metal::compile_ptx_to_msl(loop_join_ptx, loop_options);
    ok &= expect(loop_join.ok,
                 "SSA construction carries dominating values through loop and exit joins");
    if (!loop_join.ok) {
        std::cerr << loop_join.error << "\n";
    }

    const std::string inverted_loop_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry inverted_loop(.param .u32 count) {
    .reg .pred %p1;
    .reg .b32 %r<3>;
    ld.param.u32 %r1, [count];
    mov.u32 %r2, 0;
LOOP:
    add.u32 %r2, %r2, 1;
    setp.eq.u32 %p1, %r2, %r1;
    @!%p1 bra LOOP;
    ret;
}
)ptx";
    const metal::PtxToMslResult inverted_loop =
        metal::compile_ptx_to_msl(inverted_loop_ptx);
    ok &= expect(inverted_loop.ok &&
                     inverted_loop.source.find("if (!!") != std::string::npos,
                 "inverted PTX loop predicates preserve the back-edge condition");

    const std::string b64_shuffle_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry b64_shuffle(.param .u64 input, .param .u64 output) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.b64 %rd3, [%rd1];
    cvt.u32.u64 %r1, %rd3;
    { .reg .b32 tmp; mov.b64 {tmp, %r2}, %rd3; }
    shfl.sync.down.b32 %r3, %r2, 1, 31, -1;
    cvt.u64.u32 %rd4, %r3;
    st.global.b64 [%rd2], %rd4;
    ret;
}
)ptx";
    const metal::PtxToMslResult b64_shuffle =
        metal::compile_ptx_to_msl(b64_shuffle_ptx);
    const std::string b64_shuffle_ir = ir::print(b64_shuffle.gpu_ir);
    ok &= expect(b64_shuffle.ok &&
                     b64_shuffle_ir.find("shr") != std::string::npos &&
                     b64_shuffle_ir.find("-> i64") != std::string::npos &&
                     b64_shuffle.source.find("cm_lane_id +") != std::string::npos &&
                     b64_shuffle.source.find("simd_shuffle(") != std::string::npos,
                 "partial mov.b64 tuples and cvt.u64 preserve 32-bit shuffle halves");

    const std::string float_shuffle_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry float_shuffle(.param .u64 output) {
    .reg .f32 %f1;
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    mov.f32 %f1, 0f3f800000;
    shfl.sync.down.b32 %r1, %f1, 1, 31, -1;
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult float_shuffle =
        metal::compile_ptx_to_msl(float_shuffle_ptx);
    ok &= expect(float_shuffle.ok &&
                     float_shuffle.source.find("as_type<uint>(") !=
                         std::string::npos &&
                     float_shuffle.source.find("simd_shuffle(") !=
                         std::string::npos,
                 "PTX b32 shuffle preserves float register bits before typed SIMD exchange");
    if (!float_shuffle.ok) std::cerr << float_shuffle.error << "\n";

    const std::string masked_vote_mul_hi_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry masked_vote_mul_hi(.param .u64 output) {
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    mov.u32 %r1, %laneid;
    mul.hi.u32 %r2, %r1, 1431655766;
    setp.eq.u32 %p1, %r2, 0;
    vote.sync.any.pred %p2, %p1, 255;
    vote.sync.all.pred %p3, %p1, 255;
    vote.sync.ballot.b32 %r3, %p1, 255;
    selp.u32 %r4, 1, 0, %p2;
    selp.u32 %r5, 1, 0, %p3;
    add.u32 %r6, %r4, %r5;
    add.u32 %r7, %r6, %r3;
    st.global.u32 [%rd1], %r7;
    ret;
}
)ptx";
    const metal::PtxToMslResult masked_vote_mul_hi =
        metal::compile_ptx_to_msl(masked_vote_mul_hi_ptx);
    const std::string masked_vote_mul_hi_ir =
        ir::print(masked_vote_mul_hi.gpu_ir);
    const bool masked_vote_mul_hi_valid = masked_vote_mul_hi.ok &&
                     masked_vote_mul_hi_ir.find("high_half=\"true\"") !=
                         std::string::npos &&
                     masked_vote_mul_hi_ir.find("kind=\"all\"") !=
                         std::string::npos &&
                     masked_vote_mul_hi.source.find("ulong(") !=
                         std::string::npos &&
                     masked_vote_mul_hi.source.find(">> 32u") !=
                         std::string::npos &&
                     masked_vote_mul_hi.source.find("simd_ballot(") !=
                         std::string::npos &&
                     masked_vote_mul_hi.source.find(
                         "simd_active_threads_mask()") != std::string::npos &&
                     masked_vote_mul_hi.source.find("simd_any(") ==
                         std::string::npos &&
                     masked_vote_mul_hi.source.find("simd_all(") ==
                         std::string::npos;
    ok &= expect(masked_vote_mul_hi_valid,
                 "PTX mul.hi and masked vote operands retain CUDA semantics");
    if (!masked_vote_mul_hi_valid) {
        std::cerr << masked_vote_mul_hi.error << "\n"
                  << masked_vote_mul_hi_ir << "\n"
                  << masked_vote_mul_hi.source << "\n";
    }

    const std::string barrier_self_loop_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.extern .shared .align 4 .b8 scratch[];
.visible .entry barrier_self_loop(.param .u32 count) {
    .reg .pred %p1;
    .reg .b32 %r<3>;
    .reg .b64 %rd1;
    ld.param.u32 %r1, [count];
    mov.b64 %rd1, scratch;
    mov.u32 %r2, 0;
LOOP:
    st.shared.b32 [%rd1], %r2;
    bar.sync 0;
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, %r1;
    @%p1 bra LOOP;
    ret;
}
)ptx";
    const metal::PtxToMslResult barrier_self_loop =
        metal::compile_ptx_to_msl(barrier_self_loop_ptx);
    ok &= expect(barrier_self_loop.ok &&
                     barrier_self_loop.source.find("while (true)") != std::string::npos &&
                     barrier_self_loop.source.find("threadgroup_barrier") != std::string::npos &&
                     barrier_self_loop.source.find("[[threadgroup(0)]]") != std::string::npos,
                 "single-block barrier loops preserve dynamic shared memory and structure");
    if (!barrier_self_loop.ok) std::cerr << barrier_self_loop.error << "\n";

    const std::string unconditional_loop_header_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry unconditional_loop_header(.param .u32 count) {
    .reg .pred %p1;
    .reg .b32 %r<3>;
    ld.param.u32 %r1, [count];
    mov.u32 %r2, 0;
HEADER:
    bra BODY;
BODY:
    bar.sync 0;
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, %r1;
    @%p1 bra HEADER;
    ret;
}
)ptx";
    const metal::PtxToMslResult unconditional_loop_header =
        metal::compile_ptx_to_msl(unconditional_loop_header_ptx);
    ok &= expect(unconditional_loop_header.ok &&
                     unconditional_loop_header.source.find("while (true)") !=
                         std::string::npos,
                 "unconditional natural-loop headers structurize through their latch");
    if (!unconditional_loop_header.ok) {
        std::cerr << unconditional_loop_header.error << "\n";
    }

    const std::string generic_shared_helper_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .func shared_barrier_helper(.param .b64 ptr) {
    .reg .b64 %rd1;
    ld.param.b64 %rd1, [ptr];
    st.b32 [%rd1], 7;
    bar.sync 0;
    ret;
}
.visible .entry shared_barrier_call(.param .u64 output) {
    .shared .align 4 .b8 tile[4];
    .reg .b32 %r1;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [output];
    mov.b64 %rd2, tile;
    cvta.shared.u64 %rd3, %rd2;
    .param .b64 param0;
    st.param.b64 [param0], %rd3;
    call.uni shared_barrier_helper, (param0);
    ld.shared.b32 %r1, [tile];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult generic_shared_helper =
        metal::compile_ptx_to_msl(generic_shared_helper_ptx);
    const bool generic_shared_helper_valid =
        generic_shared_helper.ok &&
        generic_shared_helper.source.find(
            "shared_barrier_helper(threadgroup uchar*") != std::string::npos &&
        generic_shared_helper.source.find("threadgroup_barrier") !=
            std::string::npos;
    ok &= expect(generic_shared_helper_valid,
                 "generic PTX helper pointers specialize to shared memory at call sites");
    if (!generic_shared_helper_valid) {
        std::cerr << generic_shared_helper.error << "\n"
                  << generic_shared_helper.source << "\n";
    }

    const std::string predicated_barrier_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry predicated_barrier(.param .u32 enabled) {
    .reg .pred %p1;
    .reg .b32 %r1;
    ld.param.u32 %r1, [enabled];
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bar.sync 0;
    ret;
}
)ptx";
    const metal::PtxToMslResult predicated_barrier =
        metal::compile_ptx_to_msl(predicated_barrier_ptx);
    ok &= expect(!predicated_barrier.ok &&
                     predicated_barrier.error.find("predicated barriers") !=
                         std::string::npos,
                 "predicated PTX barriers fail with an explicit diagnostic");

    const std::string local_depot_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry local_depot() {
    .local .align 8 .b8 depot[32];
    .reg .b64 %rd1;
    mov.b64 %rd1, depot;
    st.local.b32 [%rd1], 7;
    ret;
}
)ptx";
    const metal::PtxToMslResult local_depot =
        metal::compile_ptx_to_msl(local_depot_ptx);
    ok &= expect(local_depot.ok &&
                     local_depot.source.find("thread uchar") != std::string::npos &&
                     local_depot.source.find("reinterpret_cast<thread cm_alias_uint*>") !=
                         std::string::npos,
                 "PTX local depots retain bounded private byte-array storage");
    if (!local_depot.ok) std::cerr << local_depot.error << "\n";

    const std::string module_constant_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .const .align 16 .b8 table[32];
.visible .entry module_constant(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [output];
    ld.const.b32 %r1, [table+16];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult module_constant =
        metal::compile_ptx_to_msl(module_constant_ptx);
    ok &= expect(module_constant.ok &&
                     module_constant.source.find("[[buffer(30)]]") != std::string::npos &&
                     module_constant.source.find(" + 16") != std::string::npos,
                 "PTX module constants use reserved binding 30 and byte offsets");
    if (!module_constant.ok) std::cerr << module_constant.error << "\n";

    const std::string module_global_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .global .align 4 .b8 state[32];
.visible .entry module_global(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [state+28];
    add.u32 %r1, %r1, 3;
    st.global.b32 [state+28], %r1;
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult module_global =
        metal::compile_ptx_to_msl(module_global_ptx);
    const bool module_global_valid = module_global.ok &&
                     module_global.source.find(
                         "device uchar* cm___cumetal_global_state [[buffer(1)]]") !=
                         std::string::npos &&
                     module_global.source.find(
                         "reinterpret_cast<device cm_alias_uchar*>") !=
                         std::string::npos &&
                     module_global.source.find(" + 28)") != std::string::npos &&
                     module_global.source.find("[state+28]") == std::string::npos;
    ok &= expect(module_global_valid,
                 "PTX writable module globals use ordered hidden persistent buffers");
    if (!module_global_valid) {
        std::cerr << module_global.error << "\n" << module_global.source << "\n";
    }

    const std::string promoted_aggregate_literal_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.global .align 4 .b8 __const_$record[12] = {3, 0, 0, 0, 0, 0, 32, 64, 4};
.visible .entry promoted_aggregate_literal(.param .u64 output) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [__const_$record];
    ld.global.b32 %r2, [__const_$record+4];
    ld.global.b32 %r3, [__const_$record+8];
    add.u32 %r4, %r1, %r2;
    add.u32 %r4, %r4, %r3;
    st.global.b32 [%rd1], %r4;
    ret;
}
)ptx";
    const metal::PtxToMslResult promoted_aggregate_literal =
        metal::compile_ptx_to_msl(promoted_aggregate_literal_ptx);
    ok &= expect(
        promoted_aggregate_literal.ok &&
            promoted_aggregate_literal.source.find(
                "constant uchar cm___const__record[12]") != std::string::npos &&
            promoted_aggregate_literal.source.find(
                "0x00, 0x00, 0x20, 0x40, 0x04, 0x00, 0x00, 0x00") !=
                std::string::npos &&
            promoted_aggregate_literal.source.find("global_symbol:") ==
                std::string::npos,
        "Clang-promoted aggregate literals embed exact zero-filled bytes without a runtime global binding");
    if (!promoted_aggregate_literal.ok) {
        std::cerr << promoted_aggregate_literal.error << "\n";
    }

    const std::string initialized_writable_global_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .global .align 4 .b8 mutable_state[4] = {1};
.visible .entry initialized_writable_global(.param .u64 output) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [mutable_state];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult initialized_writable_global =
        metal::compile_ptx_to_msl(initialized_writable_global_ptx);
    const bool initialized_writable_global_valid =
        initialized_writable_global.ok &&
        initialized_writable_global.source.find(
            "device uchar* cm___cumetal_global_mutable_state [[buffer(1)]]") !=
            std::string::npos &&
        initialized_writable_global.source.find(
            "constant uchar cm_mutable_state") == std::string::npos;
    ok &= expect(
        initialized_writable_global_valid,
        "initialized writable PTX globals retain registration-backed mutable storage");
    if (!initialized_writable_global_valid) {
        std::cerr << initialized_writable_global.error << "\n"
                  << initialized_writable_global.source << "\n";
    }

    const std::string private_initialized_global_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.global .align 4 .b8 private_state[8] = {13, 0, 0, 0, 254, 255, 255, 255};
.visible .entry private_initialized_global(.param .u64 output) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [private_state];
    add.u32 %r1, %r1, 4;
    st.global.b32 [private_state], %r1;
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult private_initialized_global =
        metal::compile_ptx_to_msl(private_initialized_global_ptx);
    const bool private_initialized_global_valid =
        private_initialized_global.ok &&
        private_initialized_global.source.find(
            "device uchar* cm___cumetal_global_private_state [[buffer(1)]]") !=
            std::string::npos &&
        private_initialized_global.source.find(
            "cumetal-native-symbol: private-global private_state 8 4 0") !=
            std::string::npos &&
        private_initialized_global.source.find(
            "cumetal-native-symbol-initializer: private_state 0d000000feffffff") !=
            std::string::npos &&
        private_initialized_global.source.find(
            "constant uchar cm_private_state") == std::string::npos;
    ok &= expect(
        private_initialized_global_valid,
        "module-private initialized PTX globals use persistent module-owned storage");
    if (!private_initialized_global_valid) {
        std::cerr << private_initialized_global.error << "\n"
                  << private_initialized_global.source << "\n";
    }

    const std::string oversized_writable_initializer_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .global .align 4 .b8 bad_state[4] = {1, 2, 3, 4, 5};
.visible .entry oversized_writable_initializer(.param .u64 output) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [bad_state];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult oversized_writable_initializer =
        metal::compile_ptx_to_msl(oversized_writable_initializer_ptx);
    ok &= expect(
        !oversized_writable_initializer.ok &&
            oversized_writable_initializer.error.find(
                "more elements than its declaration") != std::string::npos,
        "oversized writable PTX global initializers fail explicitly");

    const std::string aggregate_param_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry aggregate_param(
    .param .align 4 .b8 packed[12],
    .param .u64 output
) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.b32 %r1, [packed+8];
    ld.param.u64 %rd1, [output];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult aggregate_param =
        metal::compile_ptx_to_msl(aggregate_param_ptx);
    ok &= expect(aggregate_param.ok &&
                     aggregate_param.source.find(
                         "struct CuMetalPackedParam12") != std::string::npos &&
                     aggregate_param.source.find("packed.field2") !=
                         std::string::npos,
                 "typed PTX reads aligned fields from by-value aggregate arguments");
    if (!aggregate_param.ok) std::cerr << aggregate_param.error << "\n";

    const std::string frexp_abi_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.extern .func (.param .b64 result) __nv_frexp(
    .param .b64 value,
    .param .b64 exponent
);
.visible .entry frexp_abi(.param .f32 input) {
    .local .align 4 .b8 exponent_slot[4];
    .reg .b32 %r1;
    .reg .b64 %rd<4>;
    ld.param.f32 %r1, [input];
    mov.b64 %rd1, exponent_slot;
    cvt.f64.f32 %rd2, %r1;
    .param .b64 param0;
    .param .b64 param1;
    .param .b64 retval0;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param1], %rd1;
    call.uni (retval0), __nv_frexp, (param0, param1);
    ld.param.b64 %rd3, [retval0];
    cvt.rn.f32.f64 %r1, %rd3;
    ret;
}
)ptx";
    const metal::PtxToMslResult frexp_abi =
        metal::compile_ptx_to_msl(frexp_abi_ptx);
    ok &= expect(frexp_abi.ok && frexp_abi.source.find("frexp(") != std::string::npos &&
                     frexp_abi.source.find("\n    double") == std::string::npos,
                 "proven float frexp pattern normalizes its PTX double ABI boundary");
    if (!frexp_abi.ok) {
        std::cerr << frexp_abi.error << "\n";
    } else if (frexp_abi.source.find("frexp(") == std::string::npos ||
               frexp_abi.source.find("\n    double") != std::string::npos) {
        std::cerr << frexp_abi.source << "\n";
    }

    const std::string double_exp_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry double_exp(.param .u64 output) {
    .reg .b64 %rd<4>;
    .reg .f32 %f1;
    ld.param.u64 %rd1, [output];
    mov.f32 %f1, 0f3f800000;
    cvt.f64.f32 %rd2, %f1;
    .param .b64 param0;
    .param .b64 retval0;
    st.param.b64 [param0], %rd2;
    call.uni (retval0), __nv_exp, (param0);
    ld.param.b64 %rd3, [retval0];
    st.global.b64 [%rd1], %rd3;
    ret;
}
)ptx";
    const metal::PtxToMslResult double_exp =
        metal::compile_ptx_to_msl(double_exp_ptx);
    ok &= expect(double_exp.ok &&
                     double_exp.source.find("vf64_f64_to_f32(") !=
                         std::string::npos &&
                     double_exp.source.find("exp(as_type<float>") !=
                         std::string::npos &&
                     double_exp.source.find("cm_fp64_fast_f32_to_f64(") !=
                         std::string::npos &&
                     double_exp.source.find(
                         "cumetal-semantic-caveat: FP64 libdevice calls "
                         "evaluate through binary32") != std::string::npos,
                 "double-precision __nv_exp evaluates through binary32 with an "
                 "explicit caveat: " + double_exp.error);
    if (!double_exp.ok) std::cerr << double_exp.error << "\n";

    const std::string sincos_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry sincos_probe(.param .f32 input, .param .u64 output) {
    .local .align 4 .b8 sin_slot[4];
    .local .align 4 .b8 cos_slot[4];
    .reg .b32 %r1;
    .reg .b64 %rd<5>;
    ld.param.f32 %r1, [input];
    ld.param.u64 %rd1, [output];
    mov.b64 %rd2, sin_slot;
    mov.b64 %rd3, cos_slot;
    .param .b32 param0;
    .param .b64 param1;
    .param .b64 param2;
    st.param.b32 [param0], %r1;
    st.param.b64 [param1], %rd2;
    st.param.b64 [param2], %rd3;
    call.uni __nv_sincosf, (param0, param1, param2);
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult sincos_probe =
        metal::compile_ptx_to_msl(sincos_ptx);
    ok &= expect(sincos_probe.ok &&
                     sincos_probe.source.find("= sin(") != std::string::npos &&
                     sincos_probe.source.find("= cos(") != std::string::npos,
                 "void __nv_sincosf writes both results through its output "
                 "pointers: " + sincos_probe.error);
    if (!sincos_probe.ok) std::cerr << sincos_probe.error << "\n";

    const std::string sincos_double_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry sincos_double(.param .f32 input, .param .u64 output) {
    .local .align 8 .b8 sin_slot[8];
    .local .align 8 .b8 cos_slot[8];
    .reg .b32 %r1;
    .reg .b64 %rd<6>;
    .reg .f32 %f1;
    ld.param.f32 %r1, [input];
    ld.param.u64 %rd1, [output];
    cvt.f64.f32 %rd2, %r1;
    mov.b64 %rd3, sin_slot;
    mov.b64 %rd4, cos_slot;
    .param .b64 param0;
    .param .b64 param1;
    .param .b64 param2;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param1], %rd3;
    st.param.b64 [param2], %rd4;
    call.uni __nv_sincos, (param0, param1, param2);
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult sincos_double_probe =
        metal::compile_ptx_to_msl(sincos_double_ptx);
    ok &= expect(sincos_double_probe.ok &&
                     sincos_double_probe.source.find("vf64_f64_to_f32(") !=
                         std::string::npos &&
                     sincos_double_probe.source.find("sin(as_type<float>") !=
                         std::string::npos &&
                     sincos_double_probe.source.find("cos(as_type<float>") !=
                         std::string::npos &&
                     sincos_double_probe.source.find(
                         "cm_fp64_fast_f32_to_f64(") != std::string::npos,
                 "void __nv_sincos evaluates through binary32 and writes "
                 "binary64 storage through both output pointers: " +
                     sincos_double_probe.error);
    if (!sincos_double_probe.ok) {
        std::cerr << sincos_double_probe.error << "\n";
    } else if (sincos_double_probe.source.find("vf64_f64_to_f32(") ==
                   std::string::npos ||
               sincos_double_probe.source.find("sin(as_type<float>") ==
                   std::string::npos ||
               sincos_double_probe.source.find("cos(as_type<float>") ==
                   std::string::npos ||
               sincos_double_probe.source.find("cm_fp64_fast_f32_to_f64(") ==
                   std::string::npos) {
        std::cerr << sincos_double_probe.source << "\n";
    }

    const std::string modf_double_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry modf_double(.param .f32 input, .param .u64 output) {
    .local .align 8 .b8 integral_slot[8];
    .reg .b32 %r1;
    .reg .b64 %rd<6>;
    ld.param.f32 %r1, [input];
    ld.param.u64 %rd1, [output];
    cvt.f64.f32 %rd2, %r1;
    mov.b64 %rd3, integral_slot;
    .param .b64 param0;
    .param .b64 param1;
    .param .b64 retval0;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param1], %rd3;
    call.uni (retval0), __nv_modf, (param0, param1);
    ld.param.b64 %rd4, [retval0];
    st.global.b64 [%rd1], %rd4;
    ret;
}
)ptx";
    const metal::PtxToMslResult modf_double_probe =
        metal::compile_ptx_to_msl(modf_double_ptx);
    ok &= expect(modf_double_probe.ok &&
                     modf_double_probe.source.find("vf64_f64_to_f32(") !=
                         std::string::npos &&
                     modf_double_probe.source.find("trunc(") !=
                         std::string::npos &&
                     modf_double_probe.source.find(
                         "cm_fp64_fast_f32_to_f64(") != std::string::npos,
                 "double __nv_modf stores a binary64 integral part and "
                 "returns the fraction through binary32: " +
                     modf_double_probe.error);
    if (!modf_double_probe.ok) {
        std::cerr << modf_double_probe.error << "\n";
    } else if (!ok) {
        std::cerr << modf_double_probe.source << "\n";
    }

    const std::string directed_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry directed_probe(.param .f32 input, .param .f32 input2,
                               .param .u64 output) {
    .reg .f32 %f<4>;
    .reg .b64 %rd<4>;
    ld.param.f32 %f1, [input];
    ld.param.f32 %f2, [input2];
    ld.param.u64 %rd1, [output];
    cvt.f64.f32 %rd2, %f1;
    .param .b64 param0;
    .param .b64 param1;
    .param .b64 retval0;
    .param .b32 param2;
    .param .b32 param3;
    .param .b32 retval1;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param1], %rd2;
    call.uni (retval0), __nv_dadd_rd, (param0, param1);
    ld.param.b64 %rd3, [retval0];
    st.global.b64 [%rd1], %rd3;
    st.param.b32 [param2], %f1;
    st.param.b32 [param3], %f2;
    call.uni (retval1), __nv_fmul_ru, (param2, param3);
    ret;
}
)ptx";
    const metal::PtxToMslResult directed_probe =
        metal::compile_ptx_to_msl(directed_ptx);
    ok &= expect(directed_probe.ok &&
                     directed_probe.source.find("vf64_add_round(") !=
                         std::string::npos &&
                     directed_probe.source.find("vf64_mul_round(") !=
                         std::string::npos &&
                     directed_probe.source.find("vf64_f64_to_f32(") !=
                         std::string::npos &&
                     directed_probe.source.find("cm_fp64_fast_f32_to_f64(") !=
                         std::string::npos,
                 "directed-rounding intrinsics lower through the "
                 "correctly-rounded vf64 ALU: " + directed_probe.error);
    if (!directed_probe.ok) {
        std::cerr << directed_probe.error << "\n";
    } else if (!ok) {
        std::cerr << directed_probe.source << "\n";
    }

    const std::string rint_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry rint_probe(.param .f32 input, .param .u64 output) {
    .reg .f32 %f<3>;
    .reg .b64 %rd1;
    ld.param.f32 %f1, [input];
    ld.param.u64 %rd1, [output];
    cvt.rni.f32.f32 %f2, %f1;
    st.global.f32 [%rd1], %f2;
    ret;
}
)ptx";
    const metal::PtxToMslResult rint_probe =
        metal::compile_ptx_to_msl(rint_ptx);
    ok &= expect(rint_probe.ok &&
                     rint_probe.source.find("floor(") != std::string::npos &&
                     rint_probe.source.find("fmod(") != std::string::npos &&
                     rint_probe.source.find("copysign(") != std::string::npos &&
                     rint_probe.source.find("rint(") == std::string::npos,
                 "PTX rintf spells deterministic round-to-nearest-even MSL");
    if (!rint_probe.ok) std::cerr << rint_probe.error << "\n";

    const std::string hex_float_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry hex_float(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .f32 %f1;
    ld.param.u64 %rd1, [output];
    mov.f32 %f1, 0f3f800000;
    st.global.f32 [%rd1], %f1;
    ret;
}
)ptx";
    const metal::PtxToMslResult hex_float =
        metal::compile_ptx_to_msl(hex_float_ptx);
    ok &= expect(hex_float.ok &&
                     hex_float.source.find(
                         "as_type<float>(0x3f800000u)") != std::string::npos,
                 "PTX hexadecimal float bit patterns become valid exact MSL literals");

    const std::string signed_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry signed_div(.param .s32 value) {
    .reg .s32 %r<3>;
    ld.param.s32 %r1, [value];
    div.s32 %r2, %r1, -2;
    ret;
}
)ptx";
    const metal::PtxToMslResult signed_result =
        metal::compile_ptx_to_msl(signed_ptx);
    ok &= expect(signed_result.ok &&
                     signed_result.source.find("int(") != std::string::npos &&
                     signed_result.source.find(" / int(-2)") != std::string::npos,
                 "signed PTX division preserves signed semantics in MSL");

    const std::string call_slots_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.extern .func (.param .b32 result) __nv_fmaxf(
    .param .b32 lhs,
    .param .b32 rhs
);
.visible .entry call_slots(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r<4>;
    ld.param.u64 %rd1, [output];
    mov.b32 %r1, 0f3f800000;
    mov.b32 %r2, 0f40000000;
    .param .b32 param0;
    .param .b32 param1;
    .param .b32 retval0;
    st.param.b32 [param0], %r1;
    st.param.b32 [param1], %r2;
    call.uni (retval0), __nv_fmaxf, (param0, param1);
    ld.param.b32 %r3, [retval0];
    st.global.b32 [%rd1], %r3;
    ret;
}
)ptx";
    const metal::PtxToMslResult call_slots =
        metal::compile_ptx_to_msl(call_slots_ptx);
    ok &= expect(call_slots.ok &&
                     call_slots.source.find("fmax(") != std::string::npos &&
                     call_slots.source.find("as_type<float>") != std::string::npos &&
                     call_slots.source.find("as_type<uint>") != std::string::npos,
                 "PTX call slots preserve float bits across a typed libdevice call");
    if (!call_slots.ok) std::cerr << call_slots.error << "\n";

    const std::string direct_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .func (.param .b32 add_one_ret) add_one(
    .param .b32 add_one_value
) {
    .reg .b32 %r<3>;
    ld.param.b32 %r1, [add_one_value];
    add.u32 %r2, %r1, 1;
    st.param.b32 [add_one_ret], %r2;
    ret;
}
.visible .entry direct_device_call(.param .u64 output) {
    .reg .b32 %r<3>;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    mov.u32 %r1, 41;
    .param .b32 param0;
    .param .b32 retval0;
    st.param.b32 [param0], %r1;
    call.uni (retval0), add_one, (param0);
    ld.param.b32 %r2, [retval0];
    st.global.b32 [%rd1], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult direct_device_call =
        metal::compile_ptx_to_msl(direct_device_call_ptx);
    ok &= expect(direct_device_call.ok &&
                     direct_device_call.source.find("uint add_one(uint") !=
                         std::string::npos &&
                     direct_device_call.source.find("add_one(") !=
                         direct_device_call.source.rfind("add_one("),
                 "typed PTX materializes direct scalar device helpers and return slots");
    if (!direct_device_call.ok) std::cerr << direct_device_call.error << "\n";


    std::string multiline_device_call_ptx = direct_device_call_ptx;
    const std::string one_line_call = "call.uni (retval0), add_one, (param0);";
    multiline_device_call_ptx.replace(multiline_device_call_ptx.find(one_line_call),
        one_line_call.size(), "call.uni (retval0),\n add_one,\n (\n param0\n );");
    const auto multiline_device_call = metal::compile_ptx_to_msl(multiline_device_call_ptx);
    ok &= expect(multiline_device_call.ok && multiline_device_call.source == direct_device_call.source,
                 "multiline direct device call produces identical MSL to the one-line form");
    if (!multiline_device_call.ok) std::cerr << multiline_device_call.error << "\n";

    std::string byte_return_ptx = direct_device_call_ptx;
    const std::string addition = "    add.u32 %r2, %r1, 1;";
    byte_return_ptx.replace(byte_return_ptx.find(addition), addition.size(),
        "    .local .align 1 .b8 scratch[1];\n"
        "    .reg .b64 %rd1;\n"
        "    mov.u64 %rd1, scratch;\n"
        "    st.local.u8 [%rd1], %r1;\n"
        "    ld.volatile.local.u8 %r2, [%rd1];");
    const auto byte_return = metal::compile_ptx_to_msl(byte_return_ptx);
    ok &= expect(byte_return.ok,
                 "unsigned byte load into a 32-bit register satisfies a 32-bit helper return");
    if (!byte_return.ok) std::cerr << byte_return.error << "\n";


    std::string mismatched_return_ptx = byte_return_ptx;
    const auto return_decl = mismatched_return_ptx.find(".param .b32 add_one_ret");
    mismatched_return_ptx.replace(return_decl, std::string(".param .b32 add_one_ret").size(),
                                  ".param .b64 add_one_ret");
    const auto mismatched_return = metal::compile_ptx_to_msl(mismatched_return_ptx);
    ok &= expect(!mismatched_return.ok &&
                 mismatched_return.error.find("does not fit its declared return type") != std::string::npos,
                 "load widening does not weaken the return ABI width check");


    const auto mixed_load = metal::compile_ptx_to_msl(R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry mixed_load(.param .u64 input) {
 .reg .b64 %rd<3>;
 .reg .b16 %rs1;
 .reg .b32 %r1;
 ld.param.u64 %rd1, [input];
 ld.global.v2.u8 {%rs1, %r1}, [%rd1];
 ld.global.u16 %rd2, [%rd1];
 ret;
}
)ptx");
    bool byte16 = false, byte32 = false, half64 = false;
    if (mixed_load.ok) {
        for (const auto& fn : mixed_load.gpu_ir.functions)
            for (const auto& block : fn.blocks)
                for (const auto& op : block.operations) {
                    if (op.opcode != ir::OpCode::kLoad || op.result_types.size() != 1 ||
                        !op.attributes.contains("memory_bit_width")) continue;
                    const auto bits = op.result_types[0].bit_width;
                    const auto memory = op.attributes.at("memory_bit_width");
                    byte16 |= memory == "8" && bits == 16;
                    byte32 |= memory == "8" && bits == 32;
                    half64 |= memory == "16" && bits == 64;
                }
    }
    ok &= expect(mixed_load.ok && byte16 && byte32 && half64,
                 "vector lanes and 64-bit destinations retain independent register and memory widths");


    const std::string discarded_half = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry discarded_half(.param .u64 input, .param .u64 output, .param .u32 count) {
.reg .b64 %rd<8>;
.reg .b32 %r<10>;
.reg .pred %p1;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
ld.param.u32 %r1, [count];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.u32 %r2, %r2, %r3, %r4;
setp.ge.u32 %p1, %r2, %r1;
@%p1 bra DONE;
mul.wide.u32 %rd3, %r2, 4;
add.u64 %rd4, %rd1, %rd3;
add.u64 %rd5, %rd2, %rd3;
ld.global.u32 %r5, [%rd4];
mov.b64 %rd6, {%r6, %r5};
add.u32 %r7, %r5, 3;
mov.b64 {_, %r8}, %rd6;
st.global.u32 [%rd5], %r8;
DONE:
ret;
}
)ptx";
    auto half_result = metal::compile_ptx_to_msl(discarded_half);
    ok &= expect(half_result.ok, "unused packed low half does not need a definition: " + half_result.error);
    auto low_half = discarded_half;
    low_half.replace(low_half.find("{%r6, %r5}"), 10, "{%r5, %r6}");
    low_half.replace(low_half.find("{_, %r8}"), 8, "{%r8, _}");
    ok &= expect(metal::compile_ptx_to_msl(low_half).ok, "unused packed high half also compiles");
    auto dead_named_half = discarded_half;
    dead_named_half.replace(dead_named_half.find("{_, %r8}"), 8, "{%r9, %r8}");
    ok &= expect(metal::compile_ptx_to_msl(dead_named_half).ok,
                 "named but unread packed half does not require a definition");
    auto dead_named_high = low_half;
    dead_named_high.replace(dead_named_high.find("{%r8, _}"), 8, "{%r8, %r9}");
    ok &= expect(metal::compile_ptx_to_msl(dead_named_high).ok,
                 "named unread high half also compiles");
    auto observed_named_half = dead_named_half;
    observed_named_half.insert(observed_named_half.find("DONE:\n"),
                               "st.global.u32 [%rd5+4], %r9;\n");
    ok &= expect(!metal::compile_ptx_to_msl(observed_named_half).ok,
                 "observed named undefined half remains rejected");
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"{_, %r8}", "{%r8, _}"},
        {"{_, %r8}", "{_, _}"},
        {"{%r6, %r5}", "{%r6, %rd5}"},
        {"{%r6, %r5}", "{%r6, %r5, %r7}"},
        {"add.u32 %r7, %r5, 3;", "mov.u32 %r5, 0;"},
        {"add.u32 %r7, %r5, 3;", "@%p1 mov.u32 %r5, 0;"},
        {"add.u32 %r7, %r5, 3;", "st.global.u64 [%rd5], %rd6;"},
        {"add.u32 %r7, %r5, 3;", "bra EXTRACT;\nEXTRACT:"},
        {"mov.b64 %rd6,", "@%p1 mov.b64 %rd6,"},
        {"mov.b64 {_, %r8}", "@%p1 mov.b64 {_, %r8}"}}) {
        auto invalid = discarded_half;
        invalid.replace(invalid.find(from), from.size(), to);
        ok &= expect(!metal::compile_ptx_to_msl(invalid).ok,
                     "observable/stale/malformed packed half stays rejected: " + to);
    }
    const std::string inferred_pointer_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .func (.param .b32 pointer_load_ret) pointer_load(
    .param .b64 pointer_load_address
) {
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.b64 %rd1, [pointer_load_address];
    ld.global.b32 %r1, [%rd1];
    st.param.b32 [pointer_load_ret], %r1;
    ret;
}
.visible .entry inferred_pointer_device_call(
    .param .u64 .ptr .align 1 output
) {
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.b64 %rd1, [output];
    .param .b64 argument0;
    st.param.b64 [argument0], %rd1;
    .param .b32 retval0;
    call.uni (retval0), pointer_load, (argument0);
    ld.param.b32 %r1, [retval0];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult inferred_pointer_device_call =
        metal::compile_ptx_to_msl(inferred_pointer_device_call_ptx);
    ok &= expect(inferred_pointer_device_call.ok &&
                     inferred_pointer_device_call.source.find(
                         "uint pointer_load(device uchar*") != std::string::npos,
                 "typed PTX infers pointer-valued device parameters when older Clang omits .ptr");
    if (!inferred_pointer_device_call.ok) {
        std::cerr << inferred_pointer_device_call.error << "\n";
    }

    const std::string private_record_pointer_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.func read_record(.param .b64 record_address) {
    .reg .b64 %rd<7>;
    ld.param.b64 %rd0, [record_address];
    cvta.to.local.u64 %rd1, %rd0;
    ld.local.u64 %rd2, [%rd1];
    ld.local.u64 %rd3, [%rd1+8];
    ld.local.u64 %rd4, [%rd1+16];
    ld.global.u64 %rd5, [%rd2];
    add.u64 %rd5, %rd5, %rd4;
    st.global.u64 [%rd3], %rd5;
    ret;
}
.visible .entry private_record_pointer(
    .param .u64 input, .param .u64 output, .param .u64 scalar) {
    .local .align 8 .b8 record[24];
    .reg .b64 %rd<6>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u64 %rd2, [scalar];
    mov.u64 %rd3, record;
    st.local.u64 [%rd3], %rd0;
    st.local.u64 [%rd3+8], %rd1;
    st.local.u64 [%rd3+16], %rd2;
    cvta.local.u64 %rd4, %rd3;
    .param .b64 argument;
    st.param.b64 [argument], %rd4;
    call.uni read_record, (argument);
    ret;
}
)ptx";
    ok &= test_private_record_pointer_qualifiers(private_record_pointer_ptx);

    const std::string aggregate_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .func (.param .align 4 .b8 make_ret[12]) make_record(
    .param .b32 make_first,
    .param .b32 make_second,
    .param .b32 make_third
) {
    .reg .b32 %r<4>;
    ld.param.b32 %r1, [make_first];
    ld.param.b32 %r2, [make_second];
    ld.param.b32 %r3, [make_third];
    st.param.b32 [make_ret], %r1;
    st.param.b32 [make_ret+4], %r2;
    st.param.b32 [make_ret+8], %r3;
    ret;
}
.visible .func (.param .b32 consume_ret) consume_record(
    .param .align 4 .b8 consume_value[12]
) {
    .reg .b32 %r<6>;
    ld.param.b32 %r1, [consume_value];
    ld.param.b32 %r2, [consume_value+4];
    ld.param.b32 %r3, [consume_value+8];
    add.u32 %r4, %r1, %r2;
    add.u32 %r5, %r4, %r3;
    st.param.b32 [consume_ret], %r5;
    ret;
}
.visible .entry aggregate_device_call(
    .param .u64 .ptr .align 1 output
) {
    .reg .b32 %r<8>;
    .reg .b64 %rd<2>;
    ld.param.b64 %rd1, [output];
    .param .b32 make_arg0;
    .param .b32 make_arg1;
    .param .b32 make_arg2;
    st.param.b32 [make_arg0], 3;
    st.param.b32 [make_arg1], 7;
    st.param.b32 [make_arg2], 11;
    .param .align 4 .b8 make_result[12];
    call.uni (make_result), make_record, (make_arg0, make_arg1, make_arg2);
    ld.param.b32 %r1, [make_result];
    ld.param.b32 %r2, [make_result+4];
    ld.param.b32 %r3, [make_result+8];
    .param .align 4 .b8 consume_arg[12];
    st.param.b32 [consume_arg], %r1;
    st.param.b32 [consume_arg+4], %r2;
    st.param.b32 [consume_arg+8], %r3;
    .param .b32 consume_result;
    call.uni (consume_result), consume_record, (consume_arg);
    ld.param.b32 %r4, [consume_result];
    st.global.b32 [%rd1], %r4;
    ret;
}
)ptx";
    const metal::PtxToMslResult aggregate_device_call =
        metal::compile_ptx_to_msl(aggregate_device_call_ptx);
    ok &= expect(aggregate_device_call.ok &&
                     aggregate_device_call.source.find(
                         "CuMetalPackedParam12 make_record(") !=
                         std::string::npos &&
                     aggregate_device_call.source.find(
                         "consume_record(CuMetalPackedParam12") !=
                         std::string::npos,
                 "typed PTX materializes aggregate device arguments and returns");
    if (!aggregate_device_call.ok) {
        std::cerr << aggregate_device_call.error << "\n";
    }

    const std::string incomplete_aggregate_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .func consume_incomplete(
    .param .align 4 .b8 consume_value[12]
) {
    ret;
}
.visible .entry incomplete_aggregate_device_call() {
    .reg .b32 %r<2>;
    mov.u32 %r1, 1;
    .param .align 4 .b8 consume_arg[12];
    st.param.b32 [consume_arg], %r1;
    st.param.b32 [consume_arg+8], %r1;
    call.uni consume_incomplete, (consume_arg);
    ret;
}
)ptx";
    const metal::PtxToMslResult incomplete_aggregate_device_call =
        metal::compile_ptx_to_msl(incomplete_aggregate_device_call_ptx);
    ok &= expect(!incomplete_aggregate_device_call.ok &&
                     incomplete_aggregate_device_call.error.find(
                         "missing, partial, or overlapping fields") !=
                         std::string::npos,
                 "incomplete aggregate PTX device-call slots fail explicitly");

    const std::string recursive_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .func recursive_helper() {
    call.uni
    recursive_helper,
    (
    );
    ret;
}
.visible .entry recursive_device_call() {
    call.uni
    recursive_helper,
    (
    );
    ret;
}
)ptx";
    const metal::PtxToMslResult recursive_device_call =
        metal::compile_ptx_to_msl(recursive_device_call_ptx);
    ok &= expect(!recursive_device_call.ok &&
                     recursive_device_call.error.find(
                         "recursive PTX device-call cycle") != std::string::npos,
                 "recursive typed PTX device-call graphs fail explicitly");

    const std::string missing_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry missing_device_call() {
    call.uni absent_helper, ();
    ret;
}
)ptx";
    const metal::PtxToMslResult missing_device_call =
        metal::compile_ptx_to_msl(missing_device_call_ptx);
    ok &= expect(!missing_device_call.ok &&
                     missing_device_call.error.find(
                         "has no typed PTX definition") != std::string::npos,
                 "undefined typed PTX device-call targets fail explicitly");

    const std::string missing_call_slot_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry missing_call_slot() {
    .param .b32 retval0;
    call.uni (retval0), __nv_expf, (param0);
    ret;
}
)ptx";
    const metal::PtxToMslResult missing_call_slot =
        metal::compile_ptx_to_msl(missing_call_slot_ptx);
    ok &= expect(!missing_call_slot.ok &&
                     missing_call_slot.error.find("was not initialized") != std::string::npos,
                 "uninitialized PTX call slots fail explicitly");

    const std::string reciprocal_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry reciprocal(.param .f32 value) {
    .reg .f32 %f<3>;
    ld.param.f32 %f1, [value];
    rcp.rn.f32 %f2, %f1;
    ret;
}
)ptx";
    const metal::PtxToMslResult reciprocal =
        metal::compile_ptx_to_msl(reciprocal_ptx);
    ok &= expect(reciprocal.ok && reciprocal.source.find(" / ") != std::string::npos,
                 "PTX reciprocal lowers to a typed floating division");

    const std::string atomic_f32_ptx = R"ptx(
.version 8.8
.target sm_80
.address_size 64
.visible .entry atomic_f32(.param .u64 output) {
    .reg .f32 %f<3>;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    mov.f32 %f1, 0f3F800000;
    atom.global.add.f32 %f2, [%rd1], %f1;
    st.global.f32 [%rd1+4], %f2;
    ret;
}
)ptx";
    const metal::PtxToMslResult atomic_f32 =
        metal::compile_ptx_to_msl(atomic_f32_ptx);
    ok &= expect(atomic_f32.ok &&
                     atomic_f32.source.find(
                         "atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>") !=
                         std::string::npos,
                 "PTX atom.add.f32 lowers to Metal's native device atomic_float add: " +
                     atomic_f32.error);

    const std::string atomic32_ptx = R"ptx(
.version 8.8
.target sm_80
.address_size 64
.visible .entry atomic32(.param .u64 output) {
    .reg .b32 %r<12>;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    atom.relaxed.sys.global.add.u32 %r1, [%rd1], 1;
    atom.relaxed.sys.global.and.b32 %r2, [%rd1+4], 255;
    atom.relaxed.sys.global.or.b32 %r3, [%rd1+8], 2;
    atom.relaxed.sys.global.xor.b32 %r4, [%rd1+12], 4;
    atom.relaxed.sys.global.exch.b32 %r5, [%rd1+16], 7;
    atom.relaxed.sys.global.max.s32 %r6, [%rd1+20], 8;
    atom.relaxed.sys.global.min.s32 %r7, [%rd1+24], -1;
    atom.relaxed.sys.global.cas.b32 %r8, [%rd1+28], 0, 9;
    membar.gl;
    ret;
}
)ptx";
    const metal::PtxToMslResult atomic32 =
        metal::compile_ptx_to_msl(atomic32_ptx);
    ok &= expect(atomic32.ok &&
                     atomic32.source.find("atomic_fetch_add_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_and_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_or_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_xor_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_exchange_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_max_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_min_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("cm_atomic_cas_device_u32") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_compare_exchange_weak_explicit") !=
                         std::string::npos,
                 "typed PTX lowers the complete 32-bit CUDA atomic family with explicit UMA policy");
    ok &= expect(atomic32.ok &&
                     atomic32.source.find(
                         "reinterpret_cast<device cm_alias_uchar*>") !=
                         std::string::npos &&
                     atomic32.source.find(" + 28)") != std::string::npos,
                 "PTX atomic memory operands retain literal byte displacements");
    if (!atomic32.ok) std::cerr << atomic32.error << "\n";

    const std::string unfenced_acquire_atomic_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry unfenced_acquire(.param .u64 output) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    atom.acquire.global.cas.b32 %r1, [%rd1], 0, 1;
    ret;
}
)ptx";
    const metal::PtxToMslResult unfenced_acquire_atomic =
        metal::compile_ptx_to_msl(unfenced_acquire_atomic_ptx);
    ok &= expect(!unfenced_acquire_atomic.ok &&
                     unfenced_acquire_atomic.error.find(
                         "requires relaxed CUDA ordering") != std::string::npos,
                 "an acquire atomic without Clang's preceding system fence is not silently weakened");

    const std::string wide_atomic_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.shared .align 8 .f64 block_sum;
.visible .entry wide_atomic(.param .u64 output) {
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [output];
    atom.relaxed.sys.global.add.u64 %rd2, [%rd1], 1;
    st.shared.b64 [block_sum], 0;
    atom.relaxed.sys.shared.cas.b64 %rd2, [block_sum], 0, 1;
    ret;
}
)ptx";
    const metal::PtxToMslResult wide_atomic =
        metal::compile_ptx_to_msl(wide_atomic_ptx);
    ok &= expect(wide_atomic.ok &&
                     wide_atomic.source.find(
                         "cm_atomic_lock_bank [[buffer(29)]]") !=
                         std::string::npos &&
                     wide_atomic.source.find("cm_wide_atomic_add_device_u64") !=
                         std::string::npos &&
                     wide_atomic.source.find(
                         "cm_wide_atomic_cas_threadgroup_u64") !=
                         std::string::npos &&
                     wide_atomic.source.find("atomic_exchange_explicit") !=
                         std::string::npos &&
                     wide_atomic.source.find("threadgroup uchar cm_shared_block_sum[8]") !=
                         std::string::npos,
                 "typed PTX lowers device and static-shared 64-bit atomics through the lock-bank ABI");
    if (!wide_atomic.ok) std::cerr << wide_atomic.error << "\n";

    const std::string clang_printf_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.extern .func (.param .b32 func_retval0) vprintf(
    .param .b64 vprintf_param_0, .param .b64 vprintf_param_1);
.global .align 1 .b8 format[18] = {80, 82, 73, 78, 84, 70, 91, 37, 100, 44, 37, 100, 93, 61, 37, 100, 10};
.visible .entry typed_printf(.param .u32 value) {
    .local .align 8 .b8 __local_depot0[16];
    .reg .b32 %r<4>;
    .reg .b64 %rd<5>;
    mov.b64 %rd1, __local_depot0;
    cvta.local.u64 %rd2, %rd1;
    ld.param.u32 %r1, [value];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    st.local.v2.b32 [%rd1], {%r2, %r3};
    st.local.b32 [%rd1+8], %r1;
    .param .b64 param0;
    .param .b64 param1;
    .param .b32 retval0;
    st.param.b64 [param1], %rd2;
    mov.b64 %rd3, format;
    cvta.global.u64 %rd4, %rd3;
    st.param.b64 [param0], %rd4;
    call.uni (retval0), vprintf, (param0, param1);
    ret;
}
)ptx";
    const metal::PtxToMslResult clang_printf =
        metal::compile_ptx_to_msl(clang_printf_ptx);
    ok &= expect(clang_printf.ok && clang_printf.printf_formats.size() == 1 &&
                     clang_printf.printf_formats.front() == "PRINTF[%d,%d]=%d\n" &&
                     clang_printf.source.find("atomic_fetch_add_explicit") !=
                         std::string::npos &&
                     clang_printf.source.find(" = 3;") != std::string::npos &&
                     clang_printf.source.find("vprintf(") == std::string::npos,
                 "typed PTX decodes Clang vprintf into the bounded ring and parsed-count return ABI");
    if (!clang_printf.ok) std::cerr << clang_printf.error << "\n";

    const std::string clang_null_cvta_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry clang_null_cvta(.param .u64 output) {
    .reg .pred %p<2>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [output];
    mov.b64 %rd2, 0;
    cvta.to.global.u64 %rd3, %rd2;
    setp.eq.b64 %p1, %rd1, %rd3;
    ret;
}
)ptx";
    const metal::PtxToMslResult clang_null_cvta =
        metal::compile_ptx_to_msl(clang_null_cvta_ptx);
    ok &= expect(clang_null_cvta.ok &&
                     clang_null_cvta.source.find("nullptr") != std::string::npos,
                 "typed PTX preserves Clang mov-zero plus cvta null pointers");
    if (!clang_null_cvta.ok) std::cerr << clang_null_cvta.error << "\n";

    const std::string clang_call_slot_rcp_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry clang_call_slot_rcp(.param .u64 output) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    mov.b32 %r1, 0f40000000;
    rcp.rn.f32 %r2, %r1;
    st.global.b32 [%rd1], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult clang_call_slot_rcp =
        metal::compile_ptx_to_msl(clang_call_slot_rcp_ptx);
    ok &= expect(clang_call_slot_rcp.ok &&
                     clang_call_slot_rcp.source.find("as_type<float>") !=
                         std::string::npos &&
                     clang_call_slot_rcp.source.find("1.0 /") != std::string::npos,
                 "typed PTX reciprocal reinterprets b32 call-slot containers as f32");
    if (!clang_call_slot_rcp.ok) std::cerr << clang_call_slot_rcp.error << "\n";

    const std::string pointer_select_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry pointer_select(
    .param .u64 lhs, .param .u64 rhs, .param .u64 output, .param .u32 pick_lhs) {
    .reg .pred %p<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<7>;
    ld.param.u64 %rd1, [lhs];
    ld.param.u64 %rd2, [rhs];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r1, [pick_lhs];
    setp.ne.u32 %p1, %r1, 0;
    selp.b64 %rd4, %rd1, %rd2, %p1;
    mov.u64 %rd6, 1;
    mad.lo.s64 %rd5, %rd6, 4, %rd4;
    ld.global.b32 %r2, [%rd5];
    st.global.b32 [%rd3], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult pointer_select =
        metal::compile_ptx_to_msl(pointer_select_ptx);
    ok &= expect(pointer_select.ok &&
                     pointer_select.source.find("device uchar*") !=
                         std::string::npos &&
                     pointer_select.source.find("reinterpret_cast<device cm_alias_uint*>") !=
                         std::string::npos,
                 "typed PTX propagates device pointers through selp and pointer arithmetic");
    if (!pointer_select.ok) std::cerr << pointer_select.error << "\n";

    const std::string pointer_literal_select_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry pointer_literal_select(
    .param .u64 input, .param .u64 output, .param .u32 choose_literal) {
    .reg .pred %p<3>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [choose_literal];
    setp.ne.u32 %p1, %r1, 0;
    selp.b64 %rd3, 1, %rd1, %p1;
    ld.global.u64 %rd5, [%rd3];
    selp.u64 %rd4, 2, 3, %p1;
    st.global.u64 [%rd2], %rd4;
    ret;
}
)ptx";
    const metal::PtxToMslResult pointer_literal_select =
        metal::compile_ptx_to_msl(pointer_literal_select_ptx);
    ok &= expect(pointer_literal_select.ok &&
                     pointer_literal_select.source.find(
                         "reinterpret_cast<device cm_alias_uchar*>(ulong(1))") !=
                         std::string::npos &&
                     pointer_literal_select.source.find("? 2 : 3") !=
                         std::string::npos,
                 "typed PTX pointer literals retain address bits while scalar literals stay integers");
    if (!pointer_literal_select.ok) std::cerr << pointer_literal_select.error << "\n";

    const std::string out_of_order_parameter_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry out_of_order_parameter(
    .param .u64 input, .param .u64 output, .param .u32 count) {
    .reg .pred %p<3>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r1, [count];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra TRAP;
    bra LOAD_PARAMETER;
READ:
    add.u64 %rd3, %rd2, 8;
    ld.global.u64 %rd4, [%rd3];
    st.global.u64 [%rd1], %rd4;
    ret;
TRAP:
    trap;
LOAD_PARAMETER:
    ld.param.u64 %rd2, [input];
    bra READ;
}
)ptx";
    const metal::PtxToMslResult out_of_order_parameter =
        metal::compile_ptx_to_msl(out_of_order_parameter_ptx);
    ok &= expect(out_of_order_parameter.ok &&
                     out_of_order_parameter.source.find("input + 8") !=
                         std::string::npos,
                 "dispatcher binds CFG-dominating parameters before source-ordered block emission");
    if (!out_of_order_parameter.ok) std::cerr << out_of_order_parameter.error << "\n";

    const std::string signed_narrow_load_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry signed_narrow_load(.param .u64 input, .param .u64 output) {
    .reg .b16 %rs<3>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.s8 %rs1, [%rd1];
    cvt.rn.f16.s16 %rs2, %rs1;
    st.global.b16 [%rd2], %rs2;
    ret;
}
)ptx";
    const metal::PtxToMslResult signed_narrow_load =
        metal::compile_ptx_to_msl(signed_narrow_load_ptx);
    ok &= expect(signed_narrow_load.ok &&
                     signed_narrow_load.source.find(
                         "reinterpret_cast<device cm_alias_char*>") != std::string::npos &&
                     signed_narrow_load.source.find("short(") != std::string::npos,
                 "typed PTX sign-extends narrow signed loads before numeric conversion");
    if (!signed_narrow_load.ok) std::cerr << signed_narrow_load.error << "\n";

    const std::string float_bit_container_cvt_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry float_bit_container_cvt(.param .u64 input, .param .u64 output) {
    .reg .b16 %rs<2>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.b32 %r1, [%rd1];
    cvt.rn.f16.f32 %rs1, %r1;
    st.global.b16 [%rd2], %rs1;
    ret;
}
)ptx";
    const metal::PtxToMslResult float_bit_container_cvt =
        metal::compile_ptx_to_msl(float_bit_container_cvt_ptx);
    ok &= expect(float_bit_container_cvt.ok &&
                     float_bit_container_cvt.source.find("as_type<float>") !=
                         std::string::npos &&
                     float_bit_container_cvt.source.find("half(") !=
                         std::string::npos,
                 "typed PTX reinterprets b32 containers before f32-to-f16 conversion");
    if (!float_bit_container_cvt.ok) std::cerr << float_bit_container_cvt.error << "\n";

    const std::string immediate_bfe_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry immediate_bfe(.param .u64 output, .param .u32 input) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r1, [input];
    bfe.u32 %r2, %r1, 4, 1;
    st.global.u32 [%rd1], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult immediate_bfe =
        metal::compile_ptx_to_msl(immediate_bfe_ptx);
    ok &= expect(immediate_bfe.ok &&
                     immediate_bfe.source.find(">> 4") != std::string::npos &&
                     immediate_bfe.source.find("& 1") != std::string::npos,
                 "typed PTX lowers bounded unsigned immediate bfe exactly");
    if (!immediate_bfe.ok) std::cerr << immediate_bfe.error << "\n";

    const std::string predicate_not_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry predicate_not(.param .u64 output, .param .u32 input) {
    .reg .pred %p<3>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r1, [input];
    setp.ne.u32 %p1, %r1, 0;
    not.pred %p2, %p1;
    @!%p2 bra done;
    st.global.u32 [%rd1], 1;
done:
    ret;
}
)ptx";
    const metal::PtxToMslResult predicate_not =
        metal::compile_ptx_to_msl(predicate_not_ptx);
    ok &= expect(predicate_not.ok &&
                     predicate_not.source.find("4294967295") == std::string::npos,
                 "typed PTX lowers not.pred as boolean negation");
    if (!predicate_not.ok) std::cerr << predicate_not.error << "\n";

    const std::string mov_bit_container_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry mov_bit_container(.param .u64 output) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    add.f32 %r1, 0f3F800000, 0f3F800000;
    mov.b32 %r2, %r1;
    st.global.b32 [%rd1], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult mov_bit_container =
        metal::compile_ptx_to_msl(mov_bit_container_ptx);
    ok &= expect(mov_bit_container.ok &&
                     mov_bit_container.source.find("as_type<uint>") !=
                         std::string::npos,
                 "typed PTX mov.b32 preserves float register bit containers");
    if (!mov_bit_container.ok) std::cerr << mov_bit_container.error << "\n";

    const std::string indirect_aggregate_param_ptx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry indirect_aggregate_param(
    .param .u64 output,
    .param .align 4 .b8 dims[12]
) {
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [output];
    ld.b32 %r1, [%rd1];
    st.b32 [%rd1], %r1;
    mov.b64 %rd2, dims;
    mov.pred %p1, 1;
    @%p1 bra indirect_join;
    mov.u32 %r1, 0;
indirect_join:
    ld.param.b32 %r1, [%rd2+8];
    st.global.u32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult indirect_aggregate_param =
        metal::compile_ptx_to_msl(indirect_aggregate_param_ptx);
    ok &= expect(indirect_aggregate_param.ok &&
                     indirect_aggregate_param.source.find(".field2") !=
                         std::string::npos,
                 "typed PTX resolves CUDA Clang 21 generic memory and indirect ld.param addresses");
    if (!indirect_aggregate_param.ok) {
        std::cerr << indirect_aggregate_param.error << "\n";
    }

    const std::string symbolic_printf_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry symbolic_printf() {
    call.uni vprintf, (format_symbol, 7);
    ret;
}
)ptx";
    const metal::PtxToMslResult symbolic_printf =
        metal::compile_ptx_to_msl(symbolic_printf_ptx);
    ok &= expect(!symbolic_printf.ok &&
                     symbolic_printf.error.find("literal format") !=
                         std::string::npos,
                 "typed PTX rejects unresolved printf formats instead of emitting a fallback");

    const std::string dead_pointer_cast = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry dead_pointer_cast(.param .u64 input, .param .u64 output, .param .u32 count) {
.local .align 8 .b8 depot[8];
.reg .b64 %rd<10>;
.reg .b32 %r<7>;
.reg .pred %p1;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
ld.param.u32 %r1, [count];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.u32 %r2, %r2, %r3, %r4;
setp.ge.u32 %p1, %r2, %r1;
@%p1 bra DONE;
mul.wide.u32 %rd3, %r2, 8;
add.u64 %rd4, %rd1, %rd3;
add.u64 %rd5, %rd2, %rd3;
ld.global.u64 %rd6, [%rd4];
mov.u64 %rd7, depot;
st.local.u64 [%rd7], %rd6;
cvt.u32.u64 %r5, %rd7;
ld.volatile.local.u64 %rd8, [%rd7];
st.global.u64 [%rd5], %rd8;
DONE:
ret;
}
)ptx";
    ok &= expect(metal::compile_ptx_to_msl(dead_pointer_cast).ok,
                 "unused pointer truncation does not require numeric Metal pointer semantics");
    for (const std::string use : {
        "st.global.u32 [%rd5], %r5;",
        "bra OBSERVE;\nOBSERVE:\nst.global.u32 [%rd5], %r5;",
        "setp.eq.u32 %p1, %r5, 0;\n@%p1 bra DONE;"}) {
        std::string observed = dead_pointer_cast;
        observed.insert(observed.find("ld.volatile.local.u64"), use + "\n");
        const auto rejected = metal::compile_ptx_to_msl(observed);
        ok &= expect(!rejected.ok && rejected.error.find("observable pointer-to-integer") != std::string::npos,
                     "observable truncation, cross-block and predicate uses stay rejected: " + rejected.error);
    }
    std::string merged_pointer_cast = dead_pointer_cast;
    merged_pointer_cast.insert(merged_pointer_cast.find("ld.volatile.local.u64"),
        "@%p1 bra ZERO;\nbra MERGE;\nZERO:\nmov.u32 %r5, 0;\nMERGE:\nst.global.u32 [%rd5], %r5;\n");
    const auto merged_cast = metal::compile_ptx_to_msl(merged_pointer_cast);
    ok &= expect(!merged_cast.ok && merged_cast.error.find("observable pointer-to-integer") != std::string::npos,
                 "pointer truncation passed as a CFG block argument remains observable");
    if (!ok) return 1;
    std::cout << "PTX -> CuMetal IR -> typed MSL tests passed\n";
    return 0;
}
