#include "cumetal/metal/lower_to_msl.h"
#include "cumetal/ir/ptx_importer.h"
#include "ptx_cfg.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

// Check the public SSA contract, independently of how the importer solves it.
// In particular, an operation must not retain an operand type copied before its
// incoming definition or loop argument acquired its final type.
bool expect_ssa_types(const cumetal::ir::Module& module, const std::string& context) {
    namespace ir = cumetal::ir;
    bool ok = expect(ir::verify(module).ok, context + ": imported IR verifies");
    for (const auto& function : module.functions) {
        std::unordered_map<ir::ValueId, ir::Type> definitions;
        const auto define = [&](ir::ValueId value, const ir::Type& type) {
            ok &= expect(value != ir::kInvalidValue && definitions.emplace(value, type).second,
                         context + ": every result has a distinct SSA identity");
        };
        for (const auto& argument : function.arguments) define(argument.value, argument.type);
        for (const auto& block : function.blocks) {
            for (const auto& argument : block.arguments) define(argument.value, argument.type);
            for (const auto& operation : block.operations) {
                ok &= expect(operation.results.size() == operation.result_types.size(),
                             context + ": complete per-destination types");
                for (std::size_t i = 0; i < operation.results.size() &&
                                        i < operation.result_types.size(); ++i)
                    define(operation.results[i], operation.result_types[i]);
            }
        }
        for (const auto& block : function.blocks) {
            for (const auto& operation : block.operations) {
                for (const auto& operand : operation.operands) {
                    if (operand.kind != ir::OperandKind::kValue) continue;
                    const auto definition = definitions.find(operand.value);
                    ok &= expect(definition != definitions.end() && definition->second == operand.type,
                                 context + ": use type agrees with its actual definition");
                }
                for (const auto& successor : operation.successors) {
                    const auto* target = function.find_block(successor.block);
                    ok &= expect(target && successor.arguments.size() == target->arguments.size(),
                                 context + ": edge supplies every block argument");
                    if (!target) continue;
                    for (std::size_t i = 0; i < successor.arguments.size() &&
                                            i < target->arguments.size(); ++i) {
                        const auto definition = definitions.find(successor.arguments[i]);
                        ok &= expect(definition != definitions.end() &&
                                         definition->second == target->arguments[i].type,
                                     context + ": incoming definition agrees with block argument");
                    }
                }
            }
        }
    }
    return ok;
}

bool expect_block_argument(const cumetal::ir::Module& module, const std::string& block_name,
                           const std::string& name, const cumetal::ir::Type& type) {
    for (const auto& function : module.functions)
        for (const auto& block : function.blocks)
            if (block.name == block_name)
                for (const auto& argument : block.arguments)
                    if (argument.name == name)
                        return expect(argument.type == type,
                                      block_name + ": " + name + " has type " + type.str());
    return expect(false, block_name + ": expected nontrivial argument " + name);
}

bool test_definition_type_cfg() {
    namespace ir = cumetal::ir;
    bool ok = true;
    const std::string prefix = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry converted_join(.param .u64 .ptr output, .param .u32 index,
                              .param .u32 choice) {
.reg .b64 %rd<4>;
.reg .b32 %r<4>;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
ld.param.u32 %r1, [index];
ld.param.u32 %r2, [choice];
setp.ne.u32 %p1, %r2, 0;
@%p1 bra LEFT;
bra RIGHT;
)ptx";
    const std::string left = "LEFT:\ncvt.u64.u32 %rd2, %r1;\nbra JOIN;\n";
    const std::string right =
        "RIGHT:\nadd.u32 %r3, %r1, 1;\ncvt.u64.u32 %rd2, %r3;\nbra JOIN;\n";
    const std::string join = R"ptx(JOIN:
mov.b64 %rd3, %rd2;
selp.b64 %rd3, %rd3, 8, %p1;
shl.b64 %rd3, %rd3, 2;
sub.u64 %rd2, %rd1, %rd3;
st.global.u32 [%rd2], %r1;
ret;
)ptx";
    // Neither textual block order nor a register's spelling may determine the
    // join's type. The same mutable register becomes a pointer only after JOIN.
    for (unsigned layout = 0; layout < 3; ++layout) {
        for (const bool rename : {false, true}) {
            std::string source = prefix +
                (layout == 0 ? left + right + join :
                 layout == 1 ? join + right + left : right + join + left) + "}\n";
            if (rename) {
                for (const auto& [from, to] :
                     std::vector<std::pair<std::string, std::string>>{
                         {"%rd", "%wide"}, {"%r", "%word"}, {"%p", "%flag"}}) {
                    std::size_t cursor = 0;
                    while ((cursor = source.find(from, cursor)) != std::string::npos) {
                        source.replace(cursor, from.size(), to);
                        cursor += to.size();
                    }
                }
            }
            const std::string context = "converted join layout " + std::to_string(layout) +
                                        (rename ? " renamed" : "");
            const auto imported = ir::import_ptx(source);
            ok &= expect(imported.ok, context + ": " + imported.error);
            if (!imported.ok) continue;
            ok &= expect_ssa_types(imported.module, context);
            ok &= expect_block_argument(imported.module, "JOIN", rename ? "%wide2" : "%rd2",
                                        ir::Type::integer(64));
        }
    }

    const std::string loop_prefix = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry converted_loop(.param .u64 .ptr output, .param .u32 start,
                              .param .u32 count) {
.reg .b64 %rd<4>;
.reg .b32 %r<4>;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
ld.param.u32 %r1, [start];
ld.param.u32 %r2, [count];
cvt.u64.u32 %rd2, %r1;
mov.u32 %r3, 0;
bra HEADER;
)ptx";
    const std::string loop_body = R"ptx(HEADER:
setp.ge.u32 %p1, %r3, %r2;
@%p1 bra EXIT;
bra BODY;
BODY:
mov.b64 %rd3, %rd2;
shl.b64 %rd3, %rd3, 2;
add.u64 %rd3, %rd1, %rd3;
st.global.u32 [%rd3], %r3;
add.u64 %rd2, %rd2, 1;
add.u32 %r3, %r3, 1;
bra HEADER;
)ptx";
    const std::string loop_exit = R"ptx(EXIT:
sub.u64 %rd2, %rd1, %rd2;
st.global.u32 [%rd2], %r3;
ret;
)ptx";
    for (const bool exit_first : {false, true}) {
        const auto imported = ir::import_ptx(loop_prefix +
            (exit_first ? loop_exit + loop_body : loop_body + loop_exit) + "}\n");
        const std::string context = exit_first ? "loop with exit before header" : "loop offset reuse";
        ok &= expect(imported.ok, context + ": " + imported.error);
        if (!imported.ok) continue;
        ok &= expect_ssa_types(imported.module, context);
        ok &= expect_block_argument(imported.module, "HEADER", "%rd2", ir::Type::integer(64));
        ok &= expect_block_argument(imported.module, "HEADER", "%r3", ir::Type::integer(32));
    }
    auto missing_seed = loop_prefix;
    const std::string seed = "cvt.u64.u32 %rd2, %r1;\n";
    missing_seed.erase(missing_seed.find(seed), seed.size());
    const auto unseeded = ir::import_ptx(missing_seed + loop_body + loop_exit + "}\n");
    ok &= expect(!unseeded.ok && unseeded.error.find("undefined") != std::string::npos,
                 "loop backedge cannot invent the missing entry value: " + unseeded.error);

    auto undefined_arm = right;
    const std::string definition = "cvt.u64.u32 %rd2, %r3;\n";
    undefined_arm.erase(undefined_arm.find(definition), definition.size());
    const auto undefined = ir::import_ptx(prefix + left + undefined_arm + join + "}\n");
    ok &= expect(!undefined.ok && undefined.error.find("undefined") != std::string::npos,
                 "typed join still rejects an undefined incoming definition: " + undefined.error);

    // A concrete, nonzero integer is not evidence of a pointer. This must not
    // be repaired by taking the register's pointer type from the other arm.
    const std::string pointer_left = "LEFT:\nmov.b64 %rd2, %rd1;\nbra JOIN;\n";
    const std::string null_right = "RIGHT:\nmov.u64 %rd2, 0;\nbra JOIN;\n";
    const std::string null_join = R"ptx(JOIN:
setp.eq.u64 %p1, %rd2, 0;
@%p1 bra DONE;
st.global.u32 [%rd2], %r1;
DONE:
ret;
)ptx";
    const auto nullable = ir::import_ptx(prefix + pointer_left + null_right + null_join + "}\n");
    ok &= expect(nullable.ok, "pointer/known-zero join: " + nullable.error);
    if (nullable.ok) ok &= expect_ssa_types(nullable.module, "pointer/known-zero join");
    const auto nonzero = ir::import_ptx(prefix + pointer_left +
        "RIGHT:\nmov.u64 %rd2, 7;\nbra JOIN;\n" + null_join + "}\n");
    ok &= expect(!nonzero.ok, "pointer/nonzero-integer join remains rejected");
    const auto packed_nonzero = ir::import_ptx(prefix + pointer_left +
        "RIGHT:\nmov.u32 %r2, 0;\nmov.u32 %r3, 1;\n"
        "mov.b64 %rd2, {%r2, %r3};\nbra JOIN;\n" + null_join + "}\n");
    ok &= expect(!packed_nonzero.ok,
                 "a zero low half cannot prove a packed nonzero 64-bit value is null");
    auto zero_loop_prefix = loop_prefix;
    zero_loop_prefix.replace(zero_loop_prefix.find(seed), seed.size(), "mov.u64 %rd2, 0;\n");
    const auto zero_loop = ir::import_ptx(zero_loop_prefix +
        "HEADER:\nsetp.ge.u32 %p1, %r3, %r2;\n@%p1 bra EXIT;\n"
        "mov.u64 %rd2, 0;\nadd.u32 %r3, %r3, 1;\nbra HEADER;\n"
        "EXIT:\nst.global.u64 [%rd1], %rd2;\nret;\n}\n");
    ok &= expect(zero_loop.ok, "all-zero loop joins resolve to an integer type: " + zero_loop.error);
    if (zero_loop.ok) {
        ok &= expect_ssa_types(zero_loop.module, "all-zero loop joins");
        ok &= expect_block_argument(zero_loop.module, "HEADER", "%rd2", ir::Type::integer(64));
    }
    auto local_prefix = prefix;
    local_prefix.insert(local_prefix.find(".reg .b64"), ".local .align 8 .b8 scratch[8];\n");
    const auto wrong_space = cumetal::metal::compile_ptx_to_msl(local_prefix + pointer_left +
        "RIGHT:\nmov.u64 %rd2, scratch;\nbra JOIN;\n"
        "JOIN:\ncvta.to.global.u64 %rd3, %rd2;\nst.global.u32 [%rd3], %r1;\nret;\n}\n");
    ok &= expect(!wrong_space.ok, "joining a private pointer cannot authorize a global-only cast");
    auto pointer_loop = loop_prefix;
    pointer_loop.replace(pointer_loop.find(seed), seed.size(), "mov.b64 %rd2, %rd1;\n");
    const auto invalid_backedge = ir::import_ptx(pointer_loop +
        "HEADER:\nsetp.ge.u32 %p1, %r3, %r2;\n@%p1 bra EXIT;\n"
        "mov.u64 %rd2, 7;\nadd.u32 %r3, %r3, 1;\nbra HEADER;\n"
        "EXIT:\nst.global.u32 [%rd2], %r3;\nret;\n}\n");
    ok &= expect(!invalid_backedge.ok, "pointer/integer loop backedges remain rejected");
    return ok;
}

bool test_context_dependent_clones() {
    namespace ir = cumetal::ir;
    bool ok = true;
    const std::string source = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry cloned_copy(.param .u64 .ptr input, .param .u64 .ptr output,
                           .param .u32 choice) {
.reg .b64 %rd<6>;
.reg .b32 %r1;
.reg .pred %p<2>;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd5, [output];
ld.param.u32 %r1, [choice];
setp.eq.u32 %p0, %r1, 0;
@%p0 bra POINTER;
mov.u64 %rd2, 7;
mov.pred %p1, 0;
bra JOIN;
POINTER:
mov.u64 %rd2, %rd1;
mov.pred %p1, 1;
bra JOIN;
JOIN:
mov.b64 %rd3, %rd2;
@%p1 bra USE_POINTER;
st.global.u64 [%rd5], %rd3;
ret;
USE_POINTER:
ld.global.u64 %rd4, [%rd3];
st.global.u64 [%rd5], %rd4;
ret;
}
)ptx";
    const auto imported = ir::import_ptx(source);
    ok &= expect(imported.ok, "cloned mov consumes each path's actual reaching value: " + imported.error);
    if (imported.ok) {
        ok &= expect_ssa_types(imported.module, "context-dependent cloned mov");
        bool scalar_copy = false, pointer_copy = false;
        std::unordered_set<ir::ValueId> inputs;
        std::unordered_set<std::uint32_t> source_lines;
        unsigned copies = 0;
        for (const auto& function : imported.module.functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations) {
                    const auto opcode = operation.attributes.find("ptx_opcode");
                    if (opcode == operation.attributes.end() || opcode->second != "mov.b64") continue;
                    ++copies;
                    source_lines.insert(operation.location.line);
                    if (operation.operands.size() != 1 || operation.result_types.size() != 1) continue;
                    inputs.insert(operation.operands[0].value);
                    scalar_copy |= operation.operands[0].type == ir::Type::integer(64) &&
                                   operation.result_types[0] == ir::Type::integer(64);
                    pointer_copy |= operation.operands[0].type.is_pointer() &&
                                    operation.result_types[0] == operation.operands[0].type;
                }
        ok &= expect(copies >= 2 && inputs.size() >= 2 && source_lines.size() == 1 &&
                         scalar_copy && pointer_copy,
                     "one original mov becomes distinct integer and pointer definitions after specialization");
    }
    auto conflicting = source;
    conflicting.insert(conflicting.find("JOIN:\n") + 6, "setp.eq.u32 %p1, %r1, 42;\n");
    ok &= expect(!ir::import_ptx(conflicting).ok,
                 "overwriting the guard cannot authorize incompatible incoming copy types");
    return ok;
}

bool test_coupled_phi_types() {
    namespace ir = cumetal::ir;
    bool ok = true;
    const std::string prefix = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry three_way_join(.param .u64 .ptr output, .param .u32 choice) {
.reg .b64 %rd<4>;
.reg .b32 %r1;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
ld.param.u32 %r1, [choice];
setp.eq.u32 %p1, %r1, 0;
@%p1 bra LEFT;
setp.eq.u32 %p1, %r1, 1;
@%p1 bra MIDDLE;
bra RIGHT;
)ptx";
    const std::string arms =
        "LEFT:\ncvt.u64.u32 %rd2, %r1;\nbra JOIN;\n"
        "MIDDLE:\nmul.wide.u32 %rd2, %r1, 4;\nbra JOIN;\n"
        "RIGHT:\nmov.u64 %rd2, 4294967296;\nbra JOIN;\n";
    const std::string join =
        "JOIN:\nmov.b64 %rd3, %rd2;\nsub.u64 %rd2, %rd1, %rd3;\n"
        "st.global.u32 [%rd2], %r1;\nret;\n";
    for (const bool join_first : {false, true}) {
        const auto imported = ir::import_ptx(prefix + (join_first ? join + arms : arms + join) + "}\n");
        ok &= expect(imported.ok, "three different incoming definitions converge: " + imported.error);
        if (!imported.ok) continue;
        ok &= expect_ssa_types(imported.module, "three-predecessor converted offset");
        ok &= expect_block_argument(imported.module, "JOIN", "%rd2", ir::Type::integer(64));
        unsigned incoming_edges = 0;
        for (const auto& function : imported.module.functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations)
                    for (const auto& successor : operation.successors) {
                        const auto* target = function.find_block(successor.block);
                        incoming_edges += target != nullptr && target->name == "JOIN";
                    }
        ok &= expect(incoming_edges == 3, "three-predecessor regression retains all three incoming edges");
    }

    const std::string coupled = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry coupled_loop(.param .u64 .ptr output, .param .u32 count) {
.reg .b64 %rd<5>;
.reg .b32 %r<3>;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
ld.param.u32 %r1, [count];
mov.u64 %rd2, 0;
cvt.u64.u32 %rd3, %r1;
mov.u32 %r2, 0;
bra HEADER;
HEADER:
setp.ge.u32 %p1, %r2, %r1;
@%p1 bra EXIT;
mov.b64 %rd4, %rd2;
mov.b64 %rd2, %rd3;
add.u64 %rd3, %rd4, 1;
add.u32 %r2, %r2, 1;
bra HEADER;
EXIT:
add.u64 %rd2, %rd1, %rd2;
st.global.u64 [%rd2], %rd3;
ret;
}
)ptx";
    const auto imported = ir::import_ptx(coupled);
    ok &= expect(imported.ok, "mutually dependent loop arguments converge: " + imported.error);
    if (imported.ok) {
        ok &= expect_ssa_types(imported.module, "mutually dependent loop arguments");
        ok &= expect_block_argument(imported.module, "HEADER", "%rd2", ir::Type::integer(64));
        ok &= expect_block_argument(imported.module, "HEADER", "%rd3", ir::Type::integer(64));
    }
    auto conflicting = coupled;
    const std::string update = "add.u64 %rd3, %rd4, 1;";
    conflicting.replace(conflicting.find(update), update.size(), "mov.u64 %rd3, %rd1;");
    ok &= expect(!ir::import_ptx(conflicting).ok,
                 "coupled loop cannot absorb an incompatible pointer backedge into its integer cycle");
    return ok;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto fixture = [&](const char* name) {
        std::ifstream input(std::filesystem::path(argv[1]) / name);
        if (!input) throw std::runtime_error(std::string("missing fixture: ") + name);
        return std::string(std::istreambuf_iterator<char>(input), {});
    };
    namespace metal = cumetal::metal;
    bool ok = true;
    ok &= test_definition_type_cfg();
    ok &= test_context_dependent_clones();
    ok &= test_coupled_phi_types();
    const auto invariant_loop = cumetal::ir::import_ptx(fixture("ptx_trivial_block_arguments.ptx"));
    ok &= expect(invariant_loop.ok, "invariant-loop import: " + invariant_loop.error);
    if (invariant_loop.ok) {
        bool found_loop = false;
        for (const auto& function : invariant_loop.module.functions) {
            for (const auto& block : function.blocks) {
                if (block.name != "LOOP") continue;
                found_loop = true;
                ok &= expect(block.arguments.size() == 1 && block.arguments.front().name == "%r2",
                             "only the changing induction value remains a loop argument");
            }
        }
        ok &= expect(found_loop, "invariant-loop fixture retains its loop");
    }
    const std::string diamond = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry diamond(.param .u64 output, .param .u32 input) {
.reg .b64 %rd1;
.reg .b32 %r<4>;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
ld.param.u32 %r1, [input];
setp.eq.u32 %p1, %r1, 0;
@%p1 bra OTHER;
mov.u32 %r2, 7;
bra JOIN;
OTHER:
mov.u32 %r2, 11;
JOIN:
add.u32 %r3, %r1, %r2;
st.global.u32 [%rd1], %r3;
ret;
}
)ptx";
    for (const bool different_pointer : {false, true}) {
        auto source = diamond;
        if (different_pointer)
            source.insert(source.find("OTHER:\n") + 7, "add.u64 %rd1, %rd1, 4;\n");
        const auto imported = cumetal::ir::import_ptx(source);
        ok &= expect(imported.ok, "diamond import: " + imported.error);
        bool found_join = false;
        for (const auto& function : imported.module.functions) {
            for (const auto& block : function.blocks) {
                if (block.name != "JOIN") continue;
                found_join = true;
                ok &= expect(block.arguments.size() == (different_pointer ? 2u : 1u),
                             "identical join inputs fold; differing scalars and pointers remain");
                for (const auto& argument : block.arguments)
                    if (argument.name == "%rd1")
                        ok &= expect(argument.type.is_pointer(), "pointer join retains pointer type");
            }
        }
        ok &= expect(found_join, "diamond fixture retains its join");
    }
    auto undefined_loop = fixture("ptx_trivial_block_arguments.ptx");
    undefined_loop.erase(undefined_loop.find("mov.u32 %r2, 0;"), 16);
    ok &= expect(!cumetal::ir::import_ptx(undefined_loop).ok,
                 "folding does not invent a missing loop entry value");
    const auto repeated_predicate = fixture("ptx_repeated_predicate.ptx");
    const auto repeated = metal::compile_ptx_to_msl(repeated_predicate);
    ok &= expect(repeated.ok, "repeated 16-bit predicate guards the load: " + repeated.error);
    for (const std::string insertion : {
        "not.pred %p0, %p0;\n",
        "st.global.u32 [%rd3], %r1;\n"}) {
        auto invalid = repeated_predicate;
        invalid.insert(invalid.find("JOIN:\n") + 6, insertion);
        ok &= expect(!metal::compile_ptx_to_msl(invalid).ok,
                     "repeated predicate proof cannot hide overwritten guards or undefined reads");
    }
    const std::string constant_guard = fixture("ptx_constant_predicate.ptx");
    const auto guarded_constant = metal::compile_ptx_to_msl(constant_guard);
    ok &= expect(guarded_constant.ok,
                 "constant predicate guards a conditionally defined value: " + guarded_constant.error);
    auto cloned_conversion = constant_guard;
    cloned_conversion.replace(cloned_conversion.find(".reg .b64 %rd<5>;"),
                              std::string(".reg .b64 %rd<5>;").size(), ".reg .b64 %rd<6>;");
    cloned_conversion.insert(cloned_conversion.find("JOIN:\n") + 6,
                             "cvt.u64.u32 %rd5, %r0;\n");
    cloned_conversion.insert(cloned_conversion.find("st.global.u32 [%rd3], 99;"),
                             "st.global.u64 [%rd3+8], %rd5;\n");
    cloned_conversion.insert(cloned_conversion.find("DONE:\n"),
        "st.global.u64 [%rd3+8], %rd5;\n"
        "add.u64 %rd5, %rd0, %rd5;\nld.global.u8 %r1, [%rd5];\n"
        "st.global.u8 [%rd3+16], %r1;\n");
    const auto cloned = cumetal::ir::import_ptx(cloned_conversion);
    ok &= expect(cloned.ok, "guard-specialized conversion retains its own result type: " + cloned.error);
    if (cloned.ok) {
        ok &= expect_ssa_types(cloned.module, "guard-specialized conversion");
        unsigned conversions = 0;
        for (const auto& function : cloned.module.functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations) {
                    const auto opcode = operation.attributes.find("ptx_opcode");
                    if (opcode == operation.attributes.end() || opcode->second != "cvt.u64.u32") continue;
                    ++conversions;
                    ok &= expect(operation.result_types ==
                                     std::vector<cumetal::ir::Type>{cumetal::ir::Type::integer(64)},
                                 "every normalized conversion remains an integer before pointer reuse");
                }
        ok &= expect(conversions >= 2, "guard specialization actually exercises multiple conversion definitions");
    }
    for (const std::string replacement : {
        "JOIN:\n@%p0 mov.pred %p1, 0;\n",
        "JOIN:\nsetp.ne.u32 %p1, %r0, 0;\n",
        "JOIN:\nst.global.u32 [%rd3], %r1;\n"}) {
        auto invalid = constant_guard;
        invalid.replace(invalid.find("JOIN:\n"), 6, replacement);
        ok &= expect(!metal::compile_ptx_to_msl(invalid).ok,
                     "constant facts cannot hide overwritten guards or undefined reads");
    }
    auto call_preserves_local_constant = constant_guard;
    call_preserves_local_constant.insert(call_preserves_local_constant.find(".visible .entry"),
                              ".func opaque_helper() { ret; }\n");
    call_preserves_local_constant.insert(call_preserves_local_constant.find("@%p0 bra JOIN;"),
                              "call.uni opaque_helper, ();\n");
    ok &= expect(metal::compile_ptx_to_msl(call_preserves_local_constant).ok,
                 "direct calls preserve explicitly local constant predicate facts");
    auto carried_constant = constant_guard;
    carried_constant.insert(carried_constant.find("@%p0 bra JOIN;"),
                            "bra CHECK;\nCHECK:\n");
    ok &= expect(metal::compile_ptx_to_msl(carried_constant).ok,
                 "constant flags propagate across an intervening block");
    const std::string deep_guard = fixture("ptx_deep_guarded_payload.ptx");
    const auto deep = metal::compile_ptx_to_msl(deep_guard);
    ok &= expect(deep.ok, "predicate facts cross more than eight guarded blocks: " + deep.error);
    auto invalid_deep_guard = deep_guard;
    invalid_deep_guard.insert(invalid_deep_guard.find("MERGE:\n") + 7,
                              "mov.pred %p0, 1;\n");
    ok &= expect(!metal::compile_ptx_to_msl(invalid_deep_guard).ok,
                 "deep specialization does not invent an undefined payload");
    const std::string masked_payload = fixture("ptx_masked_optional_payload.ptx");
    const auto masked = metal::compile_ptx_to_msl(masked_payload);
    ok &= expect(masked.ok,
                 "absent 128-byte payload is absorbed by its false mask: " + masked.error);
    auto commuted_mask = masked_payload;
    commuted_mask.replace(commuted_mask.find("and.pred %p4, %p0, %p3;"),
                          std::string("and.pred %p4, %p0, %p3;").size(),
                          "and.pred %p4, %p3, %p0;");
    ok &= expect(metal::compile_ptx_to_msl(commuted_mask).ok,
                 "AND absorption is independent of operand order");
    auto called_mask = masked_payload;
    called_mask.insert(called_mask.find(".visible .entry"),
                       ".func helper() { ret; }\n");
    called_mask.insert(called_mask.find("JOIN:\n") + 6,
                       "call.uni helper, ();\n");
    ok &= expect(metal::compile_ptx_to_msl(called_mask).ok,
                 "direct calls preserve caller-local payload masks");
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"ld.param.u32 %r3, [last];",
         "ld.param.u32 %r3, [last];\nsetp.eq.u32 %p0, %r3, 42;"},
        {"ld.param.u32 %r3, [last];",
         "ld.param.u32 %r3, [last];\ncvt.u32.u16 %r5, %rs0;\n"
         "st.global.u32 [%rd0+8], %r5;"}}) {
        auto invalid_mask = masked_payload;
        invalid_mask.replace(invalid_mask.find(from), from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(invalid_mask);
        ok &= expect(!rejected.ok && rejected.error.find("undefined") != std::string::npos,
                     "overwritten masks and escaped payload values remain rejected");
    }
    auto exhausted_constants = carried_constant;
    exhausted_constants.insert(exhausted_constants.find(".reg .pred"),
                               ".reg .pred %q<129>;\n");
    std::string flags;
    for (unsigned i = 0; i < 129; ++i)
        flags += "mov.pred %q" + std::to_string(i) + ", -1;\n";
    exhausted_constants.insert(exhausted_constants.find("bra CHECK;"), flags);
    ok &= expect(metal::compile_ptx_to_msl(exhausted_constants).ok,
                 "dead constant facts are pruned before the live-fact limit");
    auto live_exhausted_constants = exhausted_constants;
    std::string flag_uses;
    for (unsigned i = 1; i < 129; ++i)
        flag_uses += "or.pred %q0, %q0, %q" + std::to_string(i) + ";\n";
    live_exhausted_constants.insert(live_exhausted_constants.find("CHECK:\n") + 7,
                                    flag_uses);
    ok &= expect(!metal::compile_ptx_to_msl(live_exhausted_constants).ok,
                 "live constant fact budget exhaustion does not invent SSA definitions");
    const std::string guarded_select = fixture("ptx_guarded_self_select.ptx");
    const auto selected = metal::compile_ptx_to_msl(guarded_select);
    ok &= expect(selected.ok, "unobserved loop-carried select arm is eliminated: " + selected.error);
    std::string observable_select = guarded_select;
    const auto false_edge = observable_select.find("@%p3 bra USE;\nbra HEAD;");
    observable_select.replace(false_edge, std::string("@%p3 bra USE;\nbra HEAD;").size(),
        "@%p3 bra USE;\nst.global.u32 [%rd5], %r9;\nbra HEAD;");
    const auto observable = metal::compile_ptx_to_msl(observable_select);
    ok &= expect(!observable.ok && observable.error.find("undefined") != std::string::npos,
                 "observable undefined false arm remains rejected");
    for (const std::string replacement : {
        "@%p2 bra USE;", "@!%p3 bra USE;",
        "st.global.u32 [%rd5], %r9;\n@%p3 bra USE;",
        "setp.eq.u32 %p3, %r6, 0;\n@%p3 bra USE;"}) {
        std::string unsafe_select = guarded_select;
        unsafe_select.replace(unsafe_select.find("@%p3 bra USE;"),
                              std::string("@%p3 bra USE;").size(), replacement);
        const auto rejected = metal::compile_ptx_to_msl(unsafe_select);
        ok &= expect(!rejected.ok, "select rewrite requires matching unchanged predicate and no intervening use");
    }
    std::string tuple_select = guarded_select;
    tuple_select.insert(tuple_select.find(".reg .pred"), ".reg .b16 %rs1;\n.reg .b16 %rs2;\n");
    tuple_select.insert(tuple_select.find("selp.b32 %r9"), "mov.u16 %rs1, 1;\nmov.u16 %rs2, 2;\n");
    tuple_select.replace(tuple_select.find("%r6, %r9, %p3"), std::string("%r6, %r9, %p3").size(), "{%rs1,%rs2}, %r9, %p3");
    ok &= expect(!metal::compile_ptx_to_msl(tuple_select).ok, "malformed tuple cannot become a valid mov");
    for (const std::string false_path : {
        "bra USE;", "@%p3 mov.u32 %r9, 7;\nst.global.u32 [%rd5], %r9;\nbra HEAD;"}) {
        std::string crossing = guarded_select;
        crossing.replace(crossing.find("@%p3 bra USE;\nbra HEAD;"),
                         std::string("@%p3 bra USE;\nbra HEAD;").size(), "@%p3 bra USE;\n" + false_path);
        ok &= expect(!metal::compile_ptx_to_msl(crossing).ok,
                     "false-path joins and predicated kills cannot hide an undefined read");
    }
    std::string initialized = observable_select;
    initialized.insert(initialized.find("HEAD:"), "mov.u32 %r9, 99;\n");
    const auto initialized_result = metal::compile_ptx_to_msl(initialized);
    ok &= expect(initialized_result.ok && initialized_result.source.find(" ? ") != std::string::npos,
                 "observable but initialized false arm remains valid");
    std::string wide_select = guarded_select;
    wide_select.insert(wide_select.find(".reg .pred"), ".reg .b64 %rd90;\n.reg .b64 %rd91;\n");
    const std::string narrow = "selp.b32 %r9, %r6, %r9, %p3;";
    wide_select.replace(wide_select.find(narrow), narrow.size(),
        "cvt.u64.u32 %rd90, %r6;\nselp.b64 %rd91, %rd90, %rd91, %p3;");
    wide_select.insert(wide_select.find("add.u32 %r7, %r7, %r9;"), "cvt.u32.u64 %r9, %rd91;\n");
    ok &= expect(metal::compile_ptx_to_msl(wide_select).ok, "64-bit guarded self-select compiles");

    const std::string bounded_select = fixture("ptx_bounded_self_select.ptx");
    const auto bounded = metal::compile_ptx_to_msl(bounded_select);
    ok &= expect(bounded.ok, "bounded inverted-predicate self-select compiles: " + bounded.error);
    auto directly_inverted = bounded_select;
    directly_inverted.replace(directly_inverted.find("not.pred %p4, %p3;\n@%p4 bra HEAD;"),
        std::string("not.pred %p4, %p3;\n@%p4 bra HEAD;").size(), "@!%p3 bra HEAD;");
    ok &= expect(metal::compile_ptx_to_msl(directly_inverted).ok,
                 "directly inverted branch uses the same false-path proof");
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"setp.gt.u64 %p2, %rd7, 3;", "setp.gt.u64 %p2, %rd7, 3;\nmov.u64 %rd7, 0;"},
        {"@%p2 bra CHECK;", "not.pred %p2, %p2;\n@%p2 bra CHECK;"},
        {"not.pred %p4, %p3;", "not.pred %p4, %p3;\nmov.pred %p4, 1;"},
        {"setp.lt.u64 %p5, %rd7, 4;", "setp.le.u64 %p5, %rd7, 4;"},
        {"CHECK:\n", "CHECK:\nmov.u64 %rd7, 0;\n"},
        {"CHECK:\n", "CHECK:\n@%p1 mov.u64 %rd7, 0;\n"},
        {"not.pred %p4, %p3;", "mov.pred %p4, %p3;"},
        {"@%p2 bra CHECK;", "@%p2 bra USE;"},
        {"setp.lt.u64 %p5, %rd7, 4;", "setp.lt.s64 %p5, %rd7, 4;"}}) {
        auto invalid = bounded_select;
        invalid.replace(invalid.find(from), from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(invalid);
        ok &= expect(!rejected.ok, "observable/stale bounded self-select must remain rejected: " + to);
    }

    const std::string guarded_load = fixture("ptx_guarded_load.ptx");
    const auto guarded_load_result = metal::compile_ptx_to_msl(guarded_load);
    ok &= expect(guarded_load_result.ok, "repeated equality and combined predicate guard a load: " + guarded_load_result.error);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"setp.eq.b64 %p3, %rd6, %rd7;", "setp.eq.b64 %p3, %rd7, %rd6;"},
        {"setp.eq.b64 %p2, %rd6, %rd7;\n@%p2 bra CHECK;",
         "setp.ne.b64 %p2, %rd6, %rd7;\n@!%p2 bra CHECK;"},
        {"setp.eq.b64 %p3, %rd6, %rd7;",
         "setp.ne.b64 %p3, %rd6, %rd7;\nnot.pred %p3, %p3;"},
        {"or.pred %p5, %p3, %p4;", "or.pred %p5, %p4, %p3;"},
        {"SECOND:\n@%p5 bra STORE;", "SECOND:\nmov.pred %p6, %p5;\n@%p6 bra STORE;"}}) {
        auto equivalent = guarded_load;
        equivalent.replace(equivalent.find(from), from.size(), to);
        const auto compiled = metal::compile_ptx_to_msl(equivalent);
        ok &= expect(compiled.ok, "equivalent load guard compiles: " + to + ": " + compiled.error);
    }
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"or.pred %p5, %p3, %p4;", "and.pred %p5, %p3, %p4;"},
        {"@%p2 bra CHECK;", "mov.u64 %rd6, 1;\n@%p2 bra CHECK;"},
        {"@%p2 bra CHECK;", "not.pred %p2, %p2;\n@%p2 bra CHECK;"},
        {"@%p5 bra SECOND;", "not.pred %p5, %p5;\n@%p5 bra SECOND;"},
        {"CHECK:\n", "CHECK:\nmov.u64 %rd6, 1;\n"},
        {"CHECK:\n", "CHECK:\nmov.u64 %rd7, 1;\n"},
        {"CHECK:\n", "CHECK:\n@%p1 mov.u64 %rd6, 1;\n"},
        {"CHECK:\n", "CHECK:\nst.global.u32 [%rd5], %r8;\n"},
        {"SECOND:\n", "SECOND:\nnot.pred %p5, %p5;\n"},
        {"setp.eq.b64 %p3, %rd6, %rd7;", "setp.ne.b64 %p3, %rd6, %rd7;"}}) {
        auto invalid = guarded_load;
        invalid.replace(invalid.find(from), from.size(), to);
        ok &= expect(!metal::compile_ptx_to_msl(invalid).ok,
                     "observable or stale conditional-load proof must be rejected: " + to);
    }
    // Many independent incoming edges must not cause unbounded code growth.
    // The last edge remains unsimplified when the shared cloning budget fills.
    using namespace cumetal::ir;
    detail::Instruction comparison, branch;
    comparison.opcode = "setp.eq.b64";
    comparison.operands = {"%p1", "%rd1", "%rd2"};
    branch.opcode = "bra";
    branch.predicate = "%p1";
    branch.operands = {"exit"};
    constexpr std::size_t incoming_count = 4097;
    Builder builder;
    std::vector<detail::RawBlock> blocks(incoming_count + 2);
    for (std::size_t i = 0; i <= incoming_count; ++i) {
        blocks[i].id = builder.next_block();
        blocks[i].name = "incoming_" + std::to_string(i);
        blocks[i].instructions = {&comparison, &branch};
        blocks[i].successors = {incoming_count, incoming_count + 1};
    }
    blocks[incoming_count].successors = {incoming_count + 1, incoming_count + 1};
    blocks.back().id = builder.next_block();
    blocks.back().name = "exit";
    std::deque<detail::Instruction> storage;
    detail::simplify_guarded_paths(blocks, builder, storage);
    ok &= expect(blocks.size() == incoming_count + 2 + 4096 && storage.size() == 8192,
                 "guard specialization has a bounded shared clone budget");
    ok &= expect(blocks[incoming_count - 1].successors[0] == incoming_count,
                 "budget exhaustion retains the original edge");
    // A mask region larger than the shared instruction budget must be left
    // untouched for strict SSA validation; the pass cannot partially clone it.
    detail::Instruction split, absent, present, jump, absorbed;
    split.opcode = "bra";
    split.predicate = "%selector";
    split.operands = {"present"};
    absent.opcode = "mov.pred";
    absent.operands = {"%mask", "0"};
    present.opcode = "mov.pred";
    present.operands = {"%mask", "1"};
    jump.opcode = "bra";
    jump.operands = {"join"};
    absorbed.opcode = "and.pred";
    absorbed.operands = {"%combined", "%mask", "%other"};
    Builder mask_builder;
    std::vector<detail::RawBlock> oversized(4);
    for (std::size_t i = 0; i < oversized.size(); ++i) {
        oversized[i].id = mask_builder.next_block();
        oversized[i].name = "mask_budget_" + std::to_string(i);
    }
    oversized[0].instructions = {&split};
    oversized[0].successors = {2, 1};
    oversized[1].instructions = {&absent, &jump};
    oversized[1].successors = {3};
    oversized[1].predecessors = {0};
    oversized[2].instructions = {&present, &jump};
    oversized[2].successors = {3};
    oversized[2].predecessors = {0};
    oversized[3].instructions.assign(131073, &absorbed);
    oversized[3].predecessors = {1, 2};
    std::deque<detail::Instruction> mask_storage;
    detail::simplify_guarded_paths(oversized, mask_builder, mask_storage);
    ok &= expect(oversized.size() == 4 && mask_storage.empty(),
                 "masked payload growth budget retains the original CFG");
    // Check liveness for forms rejected by later lowering as well: this pass
    // must not erase a copy feeding an address or a predicated overwrite.
    for (const auto& root : {std::string("st.global.u32"), std::string("red.global.add.u32"),
                             std::string("ld.global.u32"), std::string("predicated")}) {
        std::deque<detail::Instruction> owned;
        auto add = [&](std::string opcode, std::vector<std::string> operands,
                       std::string predicate = {}) {
            detail::Instruction instruction;
            instruction.opcode = std::move(opcode);
            instruction.operands = std::move(operands);
            instruction.predicate = std::move(predicate);
            owned.push_back(std::move(instruction));
        };
        add("mov.b32", {"%r1", "%r2"});
        const auto* copy = &owned.back();
        if (root == "predicated") {
            add("mov.u32", {"%r1", "0"}, "%p0");
            add("st.global.u32", {"[%rd0]", "%r1"});
        } else if (root == "ld.global.u32") add(root, {"%r3", "[%r1]"});
        else add(root, {"[%r1]", "1"});
        std::vector<detail::RawBlock> one(1);
        one[0].id = builder.next_block();
        one[0].name = "observe";
        for (const auto& instruction : owned) one[0].instructions.push_back(&instruction);
        detail::simplify_guarded_paths(one, builder, owned);
        ok &= expect(!one[0].instructions.empty() && one[0].instructions.front() == copy,
                     "retain observed copy before " + root);
    }
    // Timer/counter reads need not retain their value between comparisons.
    // Also verify that the bounded prefix does not consume an older predicate.
    for (const auto& [operand, padding, rewritten] :
         std::vector<std::tuple<std::string, unsigned, bool>>{
             {"%clock64", 0, false}, {"%globaltimer", 0, false},
             {"%globaltimer_lo", 0, false}, {"%globaltimer_hi", 0, false},
             {"%pm0", 0, false}, {"%pm0_64", 0, false},
             {"%rd8", 62, true}, {"%rd8", 63, false}}) {
        std::deque<detail::Instruction> owned;
        auto add = [&](std::string opcode, std::vector<std::string> operands,
                       std::string predicate = {}) {
            detail::Instruction instruction;
            instruction.opcode = std::move(opcode);
            instruction.operands = std::move(operands);
            instruction.predicate = std::move(predicate);
            owned.push_back(std::move(instruction));
        };
        add("setp.eq.u64", {"%p1", operand, "1"});
        for (unsigned i = 0; i < padding; ++i) add("mov.u32", {"%r3", "0"});
        add("setp.ne.u64", {"%p2", operand, "1"});
        const auto select_index = owned.size();
        add("selp.b64", {"%rd7", "%rd6", "%rd7", "%p2"});
        add("bra", {"retry"}, "%p1");
        std::vector<detail::RawBlock> cfg(2);
        cfg[0].id = builder.next_block();
        cfg[0].name = "retry";
        cfg[0].successors = {0, 1};
        for (const auto& instruction : owned) cfg[0].instructions.push_back(&instruction);
        cfg[1].id = builder.next_block();
        cfg[1].name = "exit";
        add("st.global.u64", {"[%rd0]", "%rd7"});
        cfg[1].instructions.push_back(&owned.back());
        add("ret", {});
        cfg[1].instructions.push_back(&owned.back());
        detail::simplify_guarded_paths(cfg, builder, owned);
        ok &= expect(cfg[0].instructions[select_index]->opcode == (rewritten ? "mov.b64" : "selp.b64"),
                     "comparison alias eligibility: " + operand + "/" + std::to_string(padding));
    }
    // Exercise call clobbers before SSA, including forms the importer may reject.
    for (unsigned variant = 0; variant < 9; ++variant) {
        cumetal::ptx::EntryFunction function;
        function.register_ranges.push_back({"%p", "pred", variant == 4 ? 3u : 4u, variant != 8});
        std::deque<detail::Instruction> owned;
        auto add = [&](std::string opcode, std::vector<std::string> operands,
                       std::string predicate = {}) -> const detail::Instruction* {
            detail::Instruction instruction;
            instruction.opcode = std::move(opcode);
            instruction.operands = std::move(operands);
            instruction.predicate = std::move(predicate);
            owned.push_back(std::move(instruction));
            return &owned.back();
        };
        const std::string reg = variant == 5 ? "%p03" : "%p3";
        std::vector<detail::RawBlock> cfg(4);
        for (unsigned i = 0; i < 4; ++i) {
            cfg[i].id = builder.next_block();
            cfg[i].name = "call_scope_" + std::to_string(i);
        }
        cfg[0].instructions = {add("mov.pred", {reg, "1"}), add("bra", {cfg[1].name})};
        cfg[0].successors = {1};
        const auto operands = variant == 1 ? std::vector<std::string>{"(%p3)", "helper", "()"}
            : variant == 2 ? std::vector<std::string>{"%rd0", "()"}
            : variant == 3 ? std::vector<std::string>{"helper"}
            : variant == 7 ? std::vector<std::string>{"(%p0)", "helper", "()"}
            : std::vector<std::string>{"helper", "()"};
        cfg[1].instructions = {add("call.uni", operands), add("bra", {cfg[2].name}, reg)};
        cfg[1].successors = {2, 3};
        cfg[2].instructions = {add("ret", {})};
        cfg[3].instructions = {add("ret", {})};
        detail::simplify_guarded_paths(cfg, builder, owned, variant == 6 ? nullptr : &function);
        const bool proven = variant == 0 || variant == 7;
        ok &= expect((cfg[0].successors[0] != 1) == proven,
                     "caller predicate scope and explicit output clobbers: " + std::to_string(variant));
    }
    return ok ? 0 : 1;
}
