#include "cumetal/ir/ir.h"
#include "dominance.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

namespace {
using Graph = std::vector<std::vector<std::size_t>>;

// Deliberately simple reference equations, independent of immediate dominators.
bool compare_with_reference(const Graph& predecessors) {
    const auto count = predecessors.size();
    std::vector<std::unordered_set<std::size_t>> reference(count);
    reference[0].insert(0);
    for (std::size_t node = 1; node < count; ++node)
        for (std::size_t candidate = 0; candidate < count; ++candidate)
            reference[node].insert(candidate);
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t node = 1; node < count; ++node) {
            std::unordered_set<std::size_t> next;
            if (!predecessors[node].empty()) {
                next = reference[predecessors[node].front()];
                for (const auto pred : predecessors[node])
                    std::erase_if(next, [&](auto candidate) { return !reference[pred].contains(candidate); });
            }
            next.insert(node);
            if (next != reference[node]) {
                reference[node] = std::move(next);
                changed = true;
            }
        }
    }
    const cumetal::ir::detail::Dominance actual(predecessors);
    for (std::size_t definition = 0; definition < count; ++definition)
        for (std::size_t use = 0; use < count; ++use)
            if (actual.dominates(definition, use) != reference[use].contains(definition)) {
                std::cerr << "dominance mismatch: definition=" << definition << " use=" << use << '\n';
                return false;
            }
    return true;
}

cumetal::ir::Module reverse_chain(std::size_t count, std::size_t definition, std::size_t use) {
    using namespace cumetal::ir;
    Module module;
    Function function;
    function.name = "reverse_chain";
    function.is_kernel = true;
    function.kernel_abi = KernelAbi{};
    for (std::size_t i = 0; i < count; ++i) {
        BasicBlock block;
        block.id = static_cast<BlockId>(i + 1);
        block.name = "block" + std::to_string(i);
        if (i == definition) {
            Operation constant;
            constant.opcode = OpCode::kConstant;
            constant.results = {1};
            constant.result_types = {Type::integer(32)};
            constant.operands = {Operand::immediate("7", Type::integer(32))};
            block.operations.push_back(constant);
        }
        if (i == use) {
            Operation add;
            add.opcode = OpCode::kAdd;
            add.results = {2};
            add.result_types = {Type::integer(32)};
            add.operands = {Operand::value_ref(1, Type::integer(32)),
                            Operand::immediate("1", Type::integer(32))};
            block.operations.push_back(add);
        }
        Operation terminator;
        if (i != 1) {
            terminator.opcode = OpCode::kBranch;
            terminator.successors.push_back(Successor{.block = static_cast<BlockId>(i == 0 ? count : i)});
        } else {
            terminator.opcode = OpCode::kReturn;
        }
        block.operations.push_back(terminator);
        function.blocks.push_back(block);
    }
    module.functions.push_back(function);
    return module;
}
}  // namespace

int main() {
    // Exhaustive small graphs include entry back edges, unreachable roots,
    // closed components, self loops, diamonds and irreducible cycles.
    for (unsigned mask = 0; mask < 65536; ++mask) {
        Graph predecessors(4);
        for (unsigned from = 0; from < 4; ++from)
            for (unsigned to = 0; to < 4; ++to)
                if ((mask >> (from * 4 + to)) & 1) predecessors[to].push_back(from);
        if (!compare_with_reference(predecessors)) return 1;
    }
    std::mt19937 random(0xC0FFEE);
    for (unsigned trial = 0; trial < 1000; ++trial) {
        const unsigned count = 1 + random() % 32;
        Graph predecessors(count);
        for (unsigned from = 0; from < count; ++from)
            for (unsigned to = 0; to < count; ++to)
                if (random() % 12 == 0) predecessors[to].push_back(from);
        if (!compare_with_reference(predecessors)) return 1;
    }
    std::cout << "Dominance matches the reference on exhaustive and random CFGs\n" << std::flush;
    // Reversed layout makes the old full-set iteration propagate just one
    // block per pass. This also exercises the verifier's actual query wiring.
    if (!cumetal::ir::verify(reverse_chain(2000, 0, 1)).ok) return 1;
    if (cumetal::ir::verify(reverse_chain(20, 10, 19)).ok) {
        std::cerr << "non-dominating definition accepted\n";
        return 1;
    }
    if (cumetal::ir::verify(reverse_chain(20, 20, 1)).ok) {
        std::cerr << "undefined value accepted\n";
        return 1;
    }
    std::cout << "Verifier dominance regressions passed\n";
}
