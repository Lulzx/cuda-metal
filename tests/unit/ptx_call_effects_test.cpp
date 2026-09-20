#include "cumetal/ir/call_write_effects.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
namespace ir = cumetal::ir;
namespace detail = cumetal::ir::detail;

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

ir::Type pointer_type() {
    return ir::Type::pointer(ir::Type::integer(8), ir::AddressSpace::kPrivate);
}

struct Graph {
    ir::Module module;
    ir::Builder ids;

    ir::Function function(const std::string& name, const std::vector<ir::Type>& types,
                          std::size_t blocks = 1) {
        ir::Function result;
        result.name = name;
        for (std::size_t i = 0; i < types.size(); ++i) {
            const auto value = ids.next_value();
            result.arguments.push_back({value, "arg" + std::to_string(i), types[i]});
            if (types[i].is_pointer()) result.pointer_provenance[value] = {};
        }
        for (std::size_t i = 0; i < blocks; ++i)
            result.blocks.push_back({ids.next_block(), name + ".b" + std::to_string(i), {}, {}});
        return result;
    }

    ir::Operand argument(const ir::Function& function, std::size_t index) const {
        const auto& argument = function.arguments.at(index);
        return ir::Operand::value_ref(argument.value, argument.type);
    }

    ir::Operand block_argument(ir::BasicBlock& block, const ir::Type& type) {
        const auto value = ids.next_value();
        block.arguments.push_back({value, type, "joined"});
        return ir::Operand::value_ref(value, type);
    }

    ir::Operation& add(ir::BasicBlock& block, ir::OpCode opcode,
                       std::vector<ir::Operand> operands = {}) {
        ir::Operation operation;
        operation.opcode = opcode;
        operation.operands = std::move(operands);
        block.operations.push_back(std::move(operation));
        return block.operations.back();
    }

    ir::Operand value(ir::BasicBlock& block, ir::OpCode opcode, const ir::Type& type,
                      std::vector<ir::Operand> operands = {}) {
        const auto id = ids.next_value();
        auto& operation = add(block, opcode, std::move(operands));
        operation.results = {id};
        operation.result_types = {type};
        return ir::Operand::value_ref(id, type);
    }

    ir::Operand constant(ir::BasicBlock& block, std::int64_t number) {
        return value(block, ir::OpCode::kConstant, ir::Type::integer(64),
                     {ir::Operand::immediate(std::to_string(number), ir::Type::integer(64))});
    }

    ir::Operand offset(ir::BasicBlock& block, ir::Operand base, ir::Operand bytes,
                       bool subtract = false) {
        auto result = value(block, ir::OpCode::kPointerOffset, base.type, {base, bytes});
        block.operations.back().attributes["offset_unit"] = "bytes";
        if (subtract) block.operations.back().attributes["offset_direction"] = "subtract";
        return result;
    }

    ir::Operand offset(ir::BasicBlock& block, ir::Operand base, std::int64_t bytes) {
        return offset(block, base, ir::Operand::immediate(std::to_string(bytes), ir::Type::integer(64)));
    }

    void store(ir::BasicBlock& block, ir::Operand address, unsigned bits = 8) {
        add(block, ir::OpCode::kStore,
            {address, ir::Operand::immediate("0", ir::Type::integer(bits))});
    }

    ir::Operation& call(ir::BasicBlock& block, const std::string& callee,
                       std::vector<ir::Operand> operands = {}) {
        auto& operation = add(block, ir::OpCode::kCall, std::move(operands));
        operation.attributes["callee"] = callee;
        return operation;
    }

    void branch(ir::BasicBlock& block, ir::BlockId target,
                std::vector<ir::ValueId> arguments = {}) {
        add(block, ir::OpCode::kBranch).successors = {{target, std::move(arguments)}};
    }

    void finish(ir::Function function) {
        module.functions.push_back(std::move(function));
    }

    detail::CallEffectSummary run(const std::string& name, detail::CallEffectLimits limits = {}) const {
        return detail::summarize_call_effects(module, name, limits);
    }
};

std::vector<detail::CallWriteEffect> canonical(std::vector<detail::CallWriteEffect> effects) {
    std::sort(effects.begin(), effects.end(), [](const auto& a, const auto& b) {
        return std::tie(a.argument, a.offset, a.bytes) < std::tie(b.argument, b.offset, b.bytes);
    });
    effects.erase(std::unique(effects.begin(), effects.end()), effects.end());
    return effects;
}

bool complete(const Graph& graph, const std::string& function,
              std::vector<detail::CallWriteEffect> expected, const std::string& label) {
    bool ok = expect(ir::verify(graph.module).ok, label + " has valid SSA and control flow");
    const auto summary = graph.run(function);
    ok &= expect(summary.complete && !summary.budget_exhausted && summary.work > 0,
                 label + " completes: " + summary.reason);
    ok &= expect(canonical(summary.writes) == canonical(std::move(expected)), label + " exact write ranges");
    return ok;
}

bool refused(const Graph& graph, const std::string& function, const std::string& label,
             detail::CallEffectLimits limits = {}, bool budget = false) {
    const auto summary = graph.run(function, limits);
    return expect(!summary.complete && summary.writes.empty() && !summary.reason.empty() &&
                      summary.budget_exhausted == budget,
                  label + " refuses without partial writes: " + summary.reason);
}

void write_byte(Graph& graph, const std::string& name = "write_byte", std::int64_t offset = 0,
                unsigned bits = 8) {
    auto function = graph.function(name, {pointer_type()});
    auto& block = function.blocks.front();
    graph.store(block, graph.offset(block, graph.argument(function, 0), offset), bits);
    graph.add(block, ir::OpCode::kReturn);
    graph.finish(std::move(function));
}

bool precise_writes() {
    Graph byte;
    write_byte(byte);
    bool ok = complete(byte, "write_byte", {{0, 0, 1}}, "write_byte argument 0");

    Graph fields;
    auto blackbox = fields.function("blackbox", {pointer_type()});
    fields.value(blackbox.blocks[0], ir::OpCode::kLoad, ir::Type::integer(64),
                 {fields.argument(blackbox, 0)});
    fields.add(blackbox.blocks[0], ir::OpCode::kReturn);
    fields.finish(std::move(blackbox));
    auto function = fields.function("field_invert", {pointer_type()});
    auto& block = function.blocks[0];
    const auto output = fields.argument(function, 0);
    for (const auto offset : {0, 8, 16, 24, 32}) fields.store(block, fields.offset(block, output, offset), 64);
    fields.store(block, fields.offset(block, output, 40), 8);
    fields.call(block, "blackbox", {output});
    fields.add(block, ir::OpCode::kReturn);
    fields.finish(std::move(function));
    ok &= complete(fields, "field_invert", {{0, 0, 8}, {0, 8, 8}, {0, 16, 8},
                                             {0, 24, 8}, {0, 32, 8}, {0, 40, 1}},
                   "field-invert ranges plus nested read-only blackbox");
    return ok;
}

bool imported_store_footprints() {
    Graph narrow;
    auto function = narrow.function("narrow", {pointer_type(), pointer_type()});
    auto& block = function.blocks[0];
    // Inconsistent narrow pointer stores must not receive a smaller footprint
    // than the actual pointer expression emitted by the backend.
    narrow.add(block, ir::OpCode::kStore,
               {narrow.argument(function, 0), narrow.argument(function, 1)}).attributes["ptx_opcode"] = "st.local.b32";
    narrow.add(block, ir::OpCode::kReturn);
    narrow.finish(std::move(function));
    bool ok = refused(narrow, "narrow", "pointer store width cannot contradict emitted storage");

    Graph lanes;
    auto vector = lanes.function("vector_lanes", {pointer_type(), pointer_type()});
    auto& vector_block = vector.blocks[0];
    for (const auto offset : {0, 8}) {
        const auto address = lanes.offset(vector_block, lanes.argument(vector, 0), offset);
        // Vector lowering retains the original PTX opcode on each scalar lane.
        lanes.add(vector_block, ir::OpCode::kStore,
                  {address, lanes.argument(vector, 1)}).attributes["ptx_opcode"] = "st.local.v2.b64";
    }
    lanes.add(vector_block, ir::OpCode::kReturn);
    lanes.finish(std::move(vector));
    ok &= complete(lanes, "vector_lanes", {{0, 0, 8}, {0, 8, 8}},
                   "split vector stores each retain one eight-byte lane footprint");
    return ok;
}

bool substituted_offset_overflow() {
    Graph graph;
    write_byte(graph, "overflow_child", 8, 64);
    auto function = graph.function("overflow", {pointer_type()});
    auto& block = function.blocks[0];
    const auto base = graph.argument(function, 0);
    graph.store(block, base); // A later overflow must discard this known write.
    const auto actual = graph.offset(block, base, std::numeric_limits<std::int64_t>::max());
    graph.call(block, "overflow_child", {actual});
    graph.add(block, ir::OpCode::kReturn);
    graph.finish(std::move(function));
    return refused(graph, "overflow", "INT64_MAX root plus child write range overflows");
}

bool own_allocations_and_calls() {
    Graph scratch;
    write_byte(scratch);
    auto local = scratch.function("local", {});
    auto& block = local.blocks[0];
    const auto storage = scratch.value(block, ir::OpCode::kAlloca, pointer_type());
    block.operations.back().attributes = {{"byte_size", "64"}, {"alignment", "8"}};
    scratch.store(block, storage, 64);
    scratch.store(block, scratch.offset(block, storage, 24), 64);
    scratch.call(block, "write_byte", {scratch.offset(block, storage, 16)});
    scratch.add(block, ir::OpCode::kReturn);
    scratch.finish(std::move(local));
    bool ok = complete(scratch, "local", {}, "own allocation writes including a nested call");

    Graph nested;
    write_byte(nested, "child", 8, 64);
    auto caller = nested.function("caller", {pointer_type(), pointer_type()});
    auto& caller_block = caller.blocks[0];
    const auto actual = nested.offset(caller_block, nested.argument(caller, 1), 16);
    nested.call(caller_block, "child", {actual});
    nested.add(caller_block, ir::OpCode::kReturn);
    nested.finish(std::move(caller));
    ok &= complete(nested, "caller", {{1, 24, 8}}, "child offset substituted into caller argument 1");
    return ok;
}

Graph scratch_loop(const std::string& mode) {
    Graph graph;
    if (mode == "nested") write_byte(graph);
    auto function = graph.function("scratch_loop", {pointer_type(), ir::Type::predicate()}, 3);
    auto& entry = function.blocks[0];
    auto& loop = function.blocks[1];
    auto& exit = function.blocks[2];
    const auto storage = graph.value(entry, ir::OpCode::kAlloca, pointer_type());
    entry.operations.back().attributes = {{"byte_size", "64"}, {"alignment", "8"}};
    const auto other = graph.value(entry, ir::OpCode::kAlloca, pointer_type());
    entry.operations.back().attributes = {{"byte_size", "64"}, {"alignment", "8"}};
    const auto zero = graph.value(entry, ir::OpCode::kConstant, ir::Type::integer(32),
                                 {ir::Operand::immediate("0", ir::Type::integer(32))});
    const auto pointer = graph.block_argument(loop, pointer_type());
    const auto iteration = graph.block_argument(loop, ir::Type::integer(32));
    loop.arguments.back().name = "iteration";
    const auto seed = mode == "formal" ? graph.argument(function, 0) : storage;
    graph.branch(entry, loop.id, {seed.value, zero.value});
    if (mode == "nested") graph.call(loop, "write_byte", {pointer});
    else graph.store(loop, pointer);
    auto next = graph.offset(loop, pointer, 1);
    if (mode == "mixed-roots" || mode == "unknown-alias")
        next = graph.value(loop, ir::OpCode::kSelect, pointer_type(),
                           {graph.argument(function, 1), next,
                            mode == "mixed-roots" ? other : graph.argument(function, 0)});
    const auto count = graph.value(loop, ir::OpCode::kAdd, ir::Type::integer(32),
        {iteration, ir::Operand::immediate("1", ir::Type::integer(32))});
    const auto again = graph.value(loop, ir::OpCode::kCompare, ir::Type::predicate(),
        {count, ir::Operand::immediate("2", ir::Type::integer(32))});
    loop.operations.back().attributes["predicate"] = "lt";
    graph.add(loop, ir::OpCode::kCondBranch, {again}).successors =
        {{loop.id, {next.value, count.value}}, {exit.id, {}}};
    graph.add(exit, ir::OpCode::kReturn);
    graph.finish(std::move(function));
    return graph;
}

bool owned_pointer_loops() {
    const auto direct = scratch_loop("direct");
    bool ok = complete(direct, "scratch_loop", {}, "loop-carried scratch writes stay owned");
    ok &= complete(scratch_loop("nested"), "scratch_loop", {}, "loop-carried scratch actual for nested write");
    for (const std::string mode : {"formal", "mixed-roots", "unknown-alias"}) {
        const auto graph = scratch_loop(mode);
        ok &= expect(ir::verify(graph.module).ok, mode + " loop has valid SSA");
        ok &= refused(graph, "scratch_loop", mode + " loop cannot acquire ownership or exact offsets");
    }
    const auto work = direct.run("scratch_loop").work;
    ok &= expect(work > 0, "owned loop proof charges work");
    if (work) ok &= refused(direct, "scratch_loop", "owned loop work budget", {.work = work - 1}, true);

    // A disconnected cycle has no initial pointer. Even selecting an allocation
    // alongside that cycle does not initialize the cyclic alternative.
    for (const bool seed_alternative : {false, true}) {
        Graph graph;
        auto function = graph.function("unseeded", {ir::Type::predicate()}, 2);
        graph.add(function.blocks[0], ir::OpCode::kReturn);
        auto& loop = function.blocks[1];
        const auto pointer = graph.block_argument(loop, pointer_type());
        auto stored = pointer;
        if (seed_alternative) {
            const auto local = graph.value(loop, ir::OpCode::kAlloca, pointer_type());
            loop.operations.back().attributes = {{"byte_size", "16"}, {"alignment", "8"}};
            stored = graph.value(loop, ir::OpCode::kSelect, pointer_type(),
                                  {graph.argument(function, 0), local, pointer});
        }
        graph.store(loop, stored);
        graph.branch(loop, loop.id, {pointer.value});
        graph.finish(std::move(function));
        ok &= expect(ir::verify(graph.module).ok, "unseeded cycle has structurally valid SSA");
        ok &= refused(graph, "unseeded", seed_alternative ?
            "allocation alternative does not seed an independent cycle" : "unseeded pointer cycle");
    }
    return ok;
}

bool copied_constant_offsets() {
    Graph graph;
    auto function = graph.function("copies", {pointer_type()});
    auto& block = function.blocks[0];
    const auto amount = graph.constant(block, 32);
    const auto copy = graph.value(block, ir::OpCode::kConvert, ir::Type::integer(64), {amount});
    const auto second_copy = graph.value(block, ir::OpCode::kConvert, ir::Type::integer(64), {copy});
    const auto pointer = graph.value(block, ir::OpCode::kConvert, pointer_type(), {graph.argument(function, 0)});
    const auto cast = graph.value(block, ir::OpCode::kAddressSpaceCast, pointer_type(), {pointer});
    const auto added = graph.offset(block, cast, second_copy);
    const auto address = graph.offset(block, added, graph.constant(block, 8), true);
    graph.store(block, address, 64);
    graph.add(block, ir::OpCode::kReturn);
    graph.finish(std::move(function));
    return complete(graph, "copies", {{0, 24, 8}}, "64-bit constant copies and pointer casts preserve byte offsets");
}

Graph branches(const std::string& mode) {
    Graph graph;
    auto function = graph.function("branches", {pointer_type(), pointer_type(), ir::Type::predicate()}, 4);
    auto& entry = function.blocks[0];
    auto& left = function.blocks[1];
    auto& right = function.blocks[2];
    auto& join = function.blocks[3];
    const auto a = graph.argument(function, 0), b = graph.argument(function, 1);
    graph.add(entry, ir::OpCode::kCondBranch, {graph.argument(function, 2)}).successors =
        {{left.id, {}}, {right.id, {}}};
    graph.store(left, a);
    graph.store(right, graph.offset(right, b, 8), 64);
    const auto left_pointer = graph.offset(left, a, 16);
    const auto right_pointer = graph.offset(right, mode == "different-argument" ? b : a,
                                             mode == "different-offset" ? 24 : 16);
    const auto joined = graph.block_argument(join, pointer_type());
    graph.branch(left, join.id, {left_pointer.value});
    graph.branch(right, join.id, {right_pointer.value});
    graph.store(join, joined, 32);
    graph.add(join, ir::OpCode::kReturn);
    graph.finish(std::move(function));
    return graph;
}

bool control_flow() {
    bool ok = complete(branches("agree"), "branches", {{0, 0, 1}, {1, 8, 8}, {0, 16, 4}},
                       "all branch stores plus an agreeing pointer join");
    ok &= refused(branches("different-argument"), "branches", "join of different formal arguments");
    ok &= refused(branches("different-offset"), "branches", "join of different byte offsets");

    Graph graph;
    auto function = graph.function("entry_join", {pointer_type(), ir::Type::predicate()}, 2);
    auto& entry = function.blocks[0];
    auto& exit = function.blocks[1];
    const auto joined = graph.block_argument(entry, pointer_type());
    // The backedge supplies a formal pointer, but no initial invocation binds
    // the entry block argument. Its first write cannot acquire that identity.
    graph.store(entry, joined);
    graph.add(entry, ir::OpCode::kCondBranch, {graph.argument(function, 1)}).successors =
        {{entry.id, {graph.argument(function, 0).value}}, {exit.id, {}}};
    graph.add(exit, ir::OpCode::kReturn);
    graph.finish(std::move(function));
    ok &= refused(graph, "entry_join", "entry join initialized only by a backedge");
    return ok;
}

Graph unsupported_address(const std::string& mode) {
    Graph graph;
    auto function = graph.function("bad_address", {pointer_type(), ir::Type::integer(64)});
    auto& block = function.blocks[0];
    const auto base = graph.argument(function, 0);
    graph.store(block, graph.offset(block, base, 64)); // Must be discarded on failure.
    ir::Operand address;
    if (mode == "dynamic-offset") {
        address = graph.offset(block, base, graph.argument(function, 1));
    } else if (mode == "truncated-offset") {
        const auto wide = graph.constant(block, 4294967304);
        const auto narrow = graph.value(block, ir::OpCode::kConvert, ir::Type::integer(32), {wide});
        const auto widened = graph.value(block, ir::OpCode::kConvert, ir::Type::integer(64), {narrow});
        address = graph.offset(block, base, widened);
    } else {
        const auto bits = graph.value(block, ir::OpCode::kConvert,
                                     ir::Type::integer(mode == "truncated-address" ? 32 : 64), {base});
        address = graph.value(block, ir::OpCode::kConvert, pointer_type(), {bits});
    }
    graph.store(block, address);
    graph.add(block, ir::OpCode::kReturn);
    graph.finish(std::move(function));
    return graph;
}

bool refusal_boundaries() {
    bool ok = true;
    for (const std::string mode : {"dynamic-offset", "truncated-offset", "integer-roundtrip", "truncated-address"})
        ok &= refused(unsupported_address(mode), "bad_address", mode);

    for (const std::string mode : {"unknown-call", "indirect-call", "recursive-call", "unknown-width", "printf", "invalid"}) {
        Graph graph;
        write_byte(graph, "known");
        auto function = graph.function("unsupported", {pointer_type()});
        auto& block = function.blocks[0];
        const auto argument = graph.argument(function, 0);
        graph.store(block, argument); // A refusal must erase even earlier known effects.
        if (mode == "unknown-width") {
            graph.store(block, argument, 0);
        } else if (mode == "printf" || mode == "invalid") {
            graph.add(block, mode == "printf" ? ir::OpCode::kPrintf : ir::OpCode::kInvalid);
        } else {
            auto& call = graph.call(block, mode == "recursive-call" ? "unsupported" :
                                           mode == "unknown-call" ? "missing" : "known", {argument});
            if (mode == "indirect-call") call.attributes["indirect"] = "true";
        }
        graph.add(block, ir::OpCode::kReturn);
        graph.finish(std::move(function));
        ok &= refused(graph, "unsupported", mode);
    }
    return ok;
}

bool builtins() {
    bool ok = true;
    for (const bool known : {true, false}) {
        Graph graph;
        auto function = graph.function("builtin", {ir::Type::floating(32)});
        auto& block = function.blocks[0];
        graph.value(block, ir::OpCode::kCall, ir::Type::floating(32), {graph.argument(function, 0)});
        block.operations.back().attributes = {{"callee", known ? "sqrt" : "unknown_builtin"}, {"builtin", "true"}};
        graph.add(block, ir::OpCode::kReturn);
        graph.finish(std::move(function));
        if (known) ok &= complete(graph, "builtin", {}, "known scalar sqrt builtin");
        else ok &= refused(graph, "builtin", "unknown builtin with scalar arguments");
    }
    return ok;
}

bool budgets() {
    Graph graph;
    write_byte(graph, "leaf", 8, 64);
    auto middle = graph.function("middle", {pointer_type()});
    graph.call(middle.blocks[0], "leaf", {graph.argument(middle, 0)});
    graph.add(middle.blocks[0], ir::OpCode::kReturn);
    graph.finish(std::move(middle));
    auto function = graph.function("budgets", {pointer_type()});
    auto& block = function.blocks[0];
    const auto argument = graph.argument(function, 0);
    graph.store(block, argument);
    graph.store(block, graph.offset(block, argument, 16));
    graph.call(block, "middle", {argument});
    graph.add(block, ir::OpCode::kReturn);
    graph.finish(std::move(function));
    bool ok = complete(graph, "budgets", {{0, 0, 1}, {0, 16, 1}, {0, 8, 8}}, "unrestricted budget control");
    ok &= refused(graph, "budgets", "work budget", {.work = 2}, true);
    ok &= refused(graph, "budgets", "write budget", {.writes = 1}, true);
    ok &= refused(graph, "budgets", "nested depth budget", {.depth = 1}, true);
    return ok;
}
} // namespace

int main() {
    bool ok = true;
    ok &= precise_writes();
    ok &= imported_store_footprints();
    ok &= substituted_offset_overflow();
    ok &= own_allocations_and_calls();
    ok &= owned_pointer_loops();
    ok &= copied_constant_offsets();
    ok &= control_flow();
    ok &= refusal_boundaries();
    ok &= builtins();
    ok &= budgets();
    return ok ? 0 : 1;
}
