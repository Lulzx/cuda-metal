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

    if (!ok) return 1;
    std::cout << "CuMetal IR tests passed\n";
    return 0;
}
