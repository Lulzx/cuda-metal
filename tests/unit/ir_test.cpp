#include "cumetal/ir/ir.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

cumetal::ir::Module make_valid_module() {
    using namespace cumetal::ir;
    Module module;
    module.source_name = "ir_test";

    Function function;
    function.name = "kernel";
    function.is_kernel = true;
    function.kernel_abi = KernelAbi{};

    BasicBlock entry;
    entry.id = 1;
    entry.name = "entry";
    Operation constant;
    constant.opcode = OpCode::kConstant;
    constant.results = {1};
    constant.result_types = {Type::integer(32)};
    constant.operands = {Operand::immediate("7", Type::integer(32))};
    entry.operations.push_back(constant);
    Operation ret;
    ret.opcode = OpCode::kReturn;
    entry.operations.push_back(ret);
    function.blocks.push_back(entry);
    module.functions.push_back(function);
    return module;
}

cumetal::ir::Module make_edge_module(const cumetal::ir::Type& incoming,
                                    const cumetal::ir::Type& argument) {
    using namespace cumetal::ir;
    Module module;
    Function function;
    function.name = "edge_helper";
    function.arguments.push_back({1, "input", incoming});
    if (incoming.is_pointer()) function.pointer_provenance[1] = PointerProvenance{};
    if (incoming.is_pointer() && incoming.address_space == AddressSpace::kNone)
        function.generic_pointer_values.insert(1);
    if (argument.is_pointer() && argument.address_space == AddressSpace::kNone)
        function.generic_pointer_values.insert(2);
    BasicBlock entry;
    entry.id = 1;
    entry.name = "entry";
    Operation branch;
    branch.opcode = OpCode::kBranch;
    branch.successors = {{2, {1}}};
    entry.operations.push_back(branch);
    BasicBlock target;
    target.id = 2;
    target.name = "target";
    target.arguments.push_back({2, argument, "joined"});
    Operation ret;
    ret.opcode = OpCode::kReturn;
    target.operations.push_back(ret);
    function.blocks = {entry, target};
    module.functions.push_back(function);
    return module;
}

bool has_diagnostic(const cumetal::ir::VerifyResult& result, const std::string& text) {
    for (const auto& diagnostic : result.diagnostics)
        if (diagnostic.message.find(text) != std::string::npos) return true;
    return false;
}

bool test_successor_values() {
    using namespace cumetal::ir;
    bool ok = true;
    const Type i32 = Type::integer(32);
    const Type i64 = Type::integer(64);
    const Type device = Type::pointer(Type::integer(8), AddressSpace::kDevice);
    const Type generic = Type::pointer(Type::integer(8), AddressSpace::kNone);
    for (const Type type : {i32, i64, Type::floating(32), Type::predicate(), device, generic})
        ok &= expect(verify(make_edge_module(type, type)).ok,
                     "equal successor types verify, including pending generic pointers: " + type.str());
    for (const auto& types : std::vector<std::pair<Type, Type>>{
             {i64, i32}, {Type::floating(32), i32}, {Type::predicate(), i32},
             {i64, device}, {device, i64},
             {device, Type::pointer(Type::integer(8), AddressSpace::kPrivate)},
             {device, Type::pointer(Type::integer(32), AddressSpace::kDevice)},
             {generic, device}}) {
        const auto result = verify(make_edge_module(types.first, types.second));
        ok &= expect(!result.ok && has_diagnostic(result, "to block argument type"),
                     "successor needs an explicit type conversion: " + types.first.str() +
                         " -> " + types.second.str());
    }
    auto pending_generic = make_edge_module(device, generic);
    ok &= expect(verify(pending_generic).ok,
                 "tracked generic destination can receive a concrete space before specialization");
    auto untracked_generic = pending_generic;
    untracked_generic.functions[0].generic_pointer_values.clear();
    ok &= expect(!verify(untracked_generic).ok,
                 "untracked generic destination does not authorize an address-space mismatch");
    auto provisional_private = make_edge_module(
        device, Type::pointer(Type::integer(8), AddressSpace::kPrivate));
    provisional_private.functions[0].generic_pointer_values.insert(2);
    ok &= expect(verify(provisional_private).ok,
                 "tracked NVVM generic PHI may retain its provisional private-space spelling");
    auto wrong_generic_pointee = make_edge_module(
        device, Type::pointer(Type::integer(32), AddressSpace::kNone));
    ok &= expect(!verify(wrong_generic_pointee).ok,
                 "tracked generic metadata cannot excuse a different pointee type");
    auto wrong_generic_scalar = make_edge_module(i64, generic);
    ok &= expect(!verify(wrong_generic_scalar).ok,
                 "tracked generic metadata cannot convert a scalar edge into a pointer");
    auto mixed = pending_generic;
    mixed.stage = IrStage::kMetalLegalized;
    ok &= expect(!verify(mixed).ok,
                 "pending generic metadata alone is insufficient after Metal legalization");
    const auto space_bit = [](AddressSpace space) {
        return static_cast<std::uint8_t>(1u << static_cast<unsigned>(space));
    };
    const std::uint8_t supported_spaces = space_bit(AddressSpace::kDevice) |
                                          space_bit(AddressSpace::kThreadgroup);
    mixed.functions[0].mixed_pointer_address_spaces[2] = supported_spaces;
    ok &= expect(verify(mixed).ok,
                 "legalized mixed pointer accepts a concrete alternative present in its tag mask");
    auto excluded_space = mixed;
    excluded_space.functions[0].mixed_pointer_address_spaces[2] = space_bit(AddressSpace::kThreadgroup);
    ok &= expect(!verify(excluded_space).ok,
                 "legalized mixed pointer rejects a concrete space absent from its mask");
    auto mixed_source = make_edge_module(generic, generic);
    mixed_source.stage = IrStage::kMetalLegalized;
    mixed_source.functions[0].mixed_pointer_address_spaces[1] = supported_spaces;
    mixed_source.functions[0].mixed_pointer_address_spaces[2] = supported_spaces;
    ok &= expect(verify(mixed_source).ok, "matching mixed-pointer alternatives verify");
    mixed_source.functions[0].mixed_pointer_address_spaces[2] = space_bit(AddressSpace::kDevice);
    ok &= expect(!verify(mixed_source).ok,
                 "equal generic pointer spellings cannot hide incompatible mixed-pointer alternatives");
    auto undefined = make_edge_module(i32, i32);
    undefined.functions[0].blocks[0].operations[0].successors[0].arguments[0] = 99;
    const auto missing = verify(undefined);
    ok &= expect(!missing.ok && has_diagnostic(missing, "uses undefined value %99"),
                 "successor values are checked for definedness");

    auto typed_null = make_edge_module(device, device);
    typed_null.functions[0].arguments.clear();
    typed_null.functions[0].pointer_provenance.clear();
    Operation null;
    null.opcode = OpCode::kConvert;
    null.results = {1};
    null.result_types = {device};
    null.operands = {Operand::immediate("null", device)};
    typed_null.functions[0].blocks[0].operations.insert(
        typed_null.functions[0].blocks[0].operations.begin(), null);
    ok &= expect(verify(typed_null).ok, "explicit typed null can enter a pointer block argument");
    auto raw_zero = typed_null;
    auto& zero = raw_zero.functions[0].blocks[0].operations[0];
    zero.opcode = OpCode::kConstant;
    zero.result_types = {i64};
    zero.operands = {Operand::immediate("0", i64)};
    const auto unconverted_zero = verify(raw_zero);
    ok &= expect(!unconverted_zero.ok && has_diagnostic(unconverted_zero, "to block argument type"),
                 "proven zero still requires an explicit typed null on a pointer edge");

    Module loop;
    Function function;
    function.name = "edge_loop";
    function.arguments = {{1, "initial", i32}};
    BasicBlock entry;
    entry.id = 1;
    entry.name = "entry";
    Operation enter;
    enter.opcode = OpCode::kBranch;
    enter.successors = {{2, {1}}};
    entry.operations.push_back(enter);
    BasicBlock header;
    header.id = 2;
    header.name = "header";
    header.arguments = {{2, i32, "carried"}};
    Operation condition;
    condition.opcode = OpCode::kCondBranch;
    condition.operands = {Operand::immediate("true", Type::predicate())};
    condition.successors = {{3, {}}, {4, {2}}};
    header.operations.push_back(condition);
    BasicBlock body;
    body.id = 3;
    body.name = "body";
    Operation next;
    next.opcode = OpCode::kConstant;
    next.results = {3};
    next.result_types = {i32};
    next.operands = {Operand::immediate("7", i32)};
    body.operations.push_back(next);
    Operation backedge;
    backedge.opcode = OpCode::kBranch;
    backedge.successors = {{2, {3}}};
    body.operations.push_back(backedge);
    BasicBlock exit;
    exit.id = 4;
    exit.name = "exit";
    exit.arguments = {{4, i32, "result"}};
    Operation ret;
    ret.opcode = OpCode::kReturn;
    exit.operations.push_back(ret);
    function.blocks = {entry, header, body, exit};
    loop.functions.push_back(function);
    ok &= expect(verify(loop).ok, "matching loop entry, backedge and exit values verify");
    auto bad_backedge = loop;
    auto& wide = bad_backedge.functions[0].blocks[2].operations[0];
    wide.result_types = {i64};
    wide.operands = {Operand::immediate("7", i64)};
    const auto mismatched = verify(bad_backedge);
    ok &= expect(!mismatched.ok && has_diagnostic(mismatched, "edge from block 'body' to block 'header'"),
                 "loop backedge types are checked against the header");
    auto escaping_body = loop;
    escaping_body.functions[0].blocks[1].operations[0].successors[1].arguments[0] = 3;
    const auto nondominating = verify(escaping_body);
    ok &= expect(!nondominating.ok && has_diagnostic(nondominating, "does not dominate the edge"),
                 "body definition cannot escape along an edge that bypasses its definition");
    auto self_definition = make_edge_module(i32, i32);
    auto& self_branch = self_definition.functions[0].blocks[0].operations[0];
    self_branch.results = {3};
    self_branch.result_types = {i32};
    self_branch.successors[0].arguments[0] = 3;
    const auto late = verify(self_definition);
    ok &= expect(!late.ok && has_diagnostic(late, "before its definition"),
                 "a branch cannot use its own result as an incoming edge value");
    return ok;
}

}  // namespace

int main() {
    using namespace cumetal::ir;
    bool ok = true;
    ok &= test_successor_values();

    Module valid = make_valid_module();
    ok &= expect(verify(valid).ok, "well-formed module verifies");
    ok &= expect(print(valid).find("%1 = constant") != std::string::npos,
                 "textual IR includes SSA definition");

    Module undefined = make_valid_module();
    Operation use;
    use.opcode = OpCode::kAdd;
    use.results = {2};
    use.result_types = {Type::integer(32)};
    use.operands = {
        Operand::value_ref(99, Type::integer(32)),
        Operand::value_ref(1, Type::integer(32)),
    };
    undefined.functions.front().blocks.front().operations.insert(
        undefined.functions.front().blocks.front().operations.end() - 1, use);
    ok &= expect(!verify(undefined).ok, "undefined SSA use is rejected");

    Module predicated_barrier = make_valid_module();
    Operation barrier;
    barrier.opcode = OpCode::kBarrier;
    barrier.memory_scope = MemoryScope::kThreadgroup;
    barrier.attributes["predicate"] = "%p";
    predicated_barrier.functions.front().blocks.front().operations.insert(
        predicated_barrier.functions.front().blocks.front().operations.end() - 1, barrier);
    ok &= expect(!verify(predicated_barrier).ok, "predicated barrier is rejected");

    Module metal_before_legalize = make_valid_module();
    Operation metal_op;
    metal_op.opcode = OpCode::kMetalLaneId;
    metal_op.results = {2};
    metal_op.result_types = {Type::integer(32)};
    metal_before_legalize.functions.front().blocks.front().operations.insert(
        metal_before_legalize.functions.front().blocks.front().operations.end() - 1, metal_op);
    ok &= expect(!verify(metal_before_legalize).ok,
                 "Metal operations are rejected in GPU-semantic IR");

    Module forward_call = make_valid_module();
    Operation call;
    call.opcode = OpCode::kCall;
    call.attributes["callee"] = "helper";
    forward_call.functions.front().blocks.front().operations.insert(
        forward_call.functions.front().blocks.front().operations.end() - 1, call);
    Function helper;
    helper.name = "helper";
    BasicBlock helper_entry;
    helper_entry.id = 2;
    helper_entry.name = "helper_entry";
    Operation helper_return;
    helper_return.opcode = OpCode::kReturn;
    helper_entry.operations.push_back(helper_return);
    helper.blocks.push_back(helper_entry);
    forward_call.functions.push_back(helper);
    ok &= expect(verify(forward_call).ok,
                 "direct calls may resolve to functions declared later in the module");

    Module recursive_call = forward_call;
    Operation recurse;
    recurse.opcode = OpCode::kCall;
    recurse.attributes["callee"] = "helper";
    recursive_call.functions.back().blocks.front().operations.insert(
        recursive_call.functions.back().blocks.front().operations.end() - 1, recurse);
    ok &= expect(!verify(recursive_call).ok, "recursive device calls are rejected");

    Module generic_cast;
    Function generic_helper;
    generic_helper.name = "generic_helper";
    const Type generic_pointer = Type::pointer(Type::integer(8), AddressSpace::kNone);
    generic_helper.arguments.push_back({1, "address", generic_pointer});
    generic_helper.generic_pointer_values.insert(1);
    generic_helper.pointer_provenance[1] = PointerProvenance{};
    BasicBlock cast_block;
    cast_block.id = 1;
    cast_block.name = "entry";
    Operation cast;
    cast.opcode = OpCode::kAddressSpaceCast;
    cast.results = {2};
    cast.result_types = {Type::pointer(Type::integer(8), AddressSpace::kPrivate)};
    cast.operands = {Operand::value_ref(1, generic_pointer)};
    cast_block.operations.push_back(cast);
    Operation cast_return;
    cast_return.opcode = OpCode::kReturn;
    cast_block.operations.push_back(cast_return);
    generic_helper.blocks.push_back(cast_block);
    generic_cast.functions.push_back(generic_helper);
    ok &= expect(verify(generic_cast).ok, "tracked generic source can await GPU call-site specialization");
    Module concrete_required = generic_cast;
    concrete_required.stage = IrStage::kMetalLegalized;
    ok &= expect(!verify(concrete_required).ok, "Metal IR rejects unresolved generic cast sources");
    Module untracked = generic_cast;
    untracked.functions.front().generic_pointer_values.clear();
    ok &= expect(!verify(untracked).ok, "untracked generic cast sources remain invalid");
    Module generic_target = generic_cast;
    generic_target.functions.front().blocks.front().operations.front().result_types.front() = generic_pointer;
    ok &= expect(!verify(generic_target).ok, "generic cast targets remain invalid");

    Module pointer_offset = generic_cast;
    auto& offset_function = pointer_offset.functions.front();
    const Type local_pointer = Type::pointer(Type::integer(8), AddressSpace::kPrivate);
    offset_function.arguments[0].type = local_pointer;
    offset_function.generic_pointer_values.clear();
    auto& offset = offset_function.blocks[0].operations[0];
    offset.opcode = OpCode::kPointerOffset;
    offset.operands = {Operand::value_ref(1, local_pointer),
                       Operand::immediate("8", Type::integer(64))};
    offset.attributes = {{"offset_direction", "subtract"}, {"offset_unit", "bytes"}};
    ok &= expect(verify(pointer_offset).ok, "pointer subtraction preserves byte units and address space");
    for (int invalid_case = 0; invalid_case < 5; ++invalid_case) {
        Module invalid = pointer_offset;
        auto& operation = invalid.functions[0].blocks[0].operations[0];
        switch (invalid_case) {
            case 0: operation.attributes["offset_direction"] = "unknown"; break;
            case 1: operation.attributes.erase("offset_unit"); break;
            case 2: operation.operands[1].type = Type::integer(32); break;
            case 3: operation.result_types[0] = generic_pointer; break;
            case 4: operation.attributes["combined"] = "mul_add"; break;
        }
        ok &= expect(!verify(invalid).ok, "invalid pointer subtraction contract is rejected");
    }

    if (!ok) return 1;
    std::cout << "CuMetal IR tests passed\n";
    return 0;
}
