#include "cumetal/ir/ir.h"

#include <iostream>
#include <string>

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

}  // namespace

int main() {
    using namespace cumetal::ir;
    bool ok = true;

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
