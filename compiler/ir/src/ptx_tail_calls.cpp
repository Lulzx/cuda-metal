#include "ptx_tail_calls.h"
#include "ptx_instruction.h"
#include "ptx_parameters.h"
#include "ptx_text.h"

#include <algorithm>
#include <map>
#include <set>
#include <unordered_set>

namespace cumetal::ir::detail {
namespace {

struct ScalarTailPlan {
    std::size_t call_index;
    std::set<std::size_t> remove;
    std::string argument;
};
struct LocalTailPlan {
    std::size_t branch;
    std::string input, argument_pointer;
};

std::string unused_name(const cumetal::ptx::EntryFunction& function, std::string name) {
    const auto collides = [&] {
        if (std::any_of(function.register_declarations.begin(), function.register_declarations.end(),
                        [&](const auto& reg) { return reg.name == name; })) return true;
        return std::any_of(function.instructions.begin(), function.instructions.end(), [&](const auto& instruction) {
            return std::any_of(instruction.operands.begin(), instruction.operands.end(),
                               [&](const auto& operand) { return operand.find(name) != std::string::npos; });
        });
    };
    while (collides()) name += "_";
    return name;
}

// Bounded tail-self-call elimination for scalar integer helpers. Ineligible
// cycles retain the normal call-graph rejection. In particular, no local frame
// is reused: memory operations and pointer/address instructions are excluded.
std::optional<ScalarTailPlan> match_scalar_tail_call(const cumetal::ptx::EntryFunction* function) {
    if (function->params.size() != 1 || function->return_params.size() != 1 ||
        function->params.front().is_pointer ||
        (function->params.front().type != ".b64" && function->params.front().type != ".u64") ||
        function->params.front().byte_size != 8 || function->return_params.front().byte_size != 16) return std::nullopt;
    const auto slot_at = [](const std::string& operand, const std::string& name, int offset) {
        return parameter_slot_offset(operand, name) == offset;
    };
    auto& instructions = function->instructions;
    if (instructions.empty()) return std::nullopt;
    const std::string parameter = function->params.front().name;
    const std::string return_parameter = function->return_params.front().name;
    std::size_t call_index = instructions.size();
    const std::unordered_set<std::string> pure_roots = {
        "mov", "add", "sub", "mul", "mad", "shr", "shl", "xor", "or", "and",
        "not", "neg", "setp", "selp", "bra", "ret", "call", "ptx"};
    for (std::size_t i = 0; i < instructions.size(); ++i) {
        const auto& instruction = instructions[i];
        const auto root = root_opcode(instruction.opcode);
        if (root == "call") {
            if (call_index != instructions.size() || !instruction.predicate.empty() ||
                (instruction.opcode != "call" && instruction.opcode != "call.uni") ||
                !instruction.supported || direct_call_target(instruction) != function->name) return std::nullopt;
            call_index = i;
        } else if (root == "ld" || root == "st") {
            if (!starts_with(instruction.opcode, "ld.param.") &&
                !starts_with(instruction.opcode, "st.param.")) return std::nullopt;
        } else if (!pure_roots.contains(root) ||
                   (root == "ptx" && instruction.opcode != "ptx.label")) return std::nullopt;
        // Never let address-valued symbols (including local depots) enter the
        // scalar loop through arithmetic or selects. Match whole operands;
        // first_register alone would also accept tuples or address expressions.
        if (root != "ld" && root != "st" && root != "bra" && root != "call" &&
            root != "ret" && root != "ptx") {
            if (instruction.operands.empty() ||
                first_register(instruction.operands.front()) != trim(instruction.operands.front())) return std::nullopt;
            for (const auto& operand : instruction.operands) {
                const std::string value = trim(operand);
                if (!value.empty() && first_register(value) == value) continue;
                try {
                    std::size_t consumed = 0;
                    (void)std::stoll(value, &consumed, 0);
                    if (consumed != value.size()) return std::nullopt;
                } catch (...) { return std::nullopt; }
            }
        }
    }
    if (call_index == instructions.size() || call_index == 0) return std::nullopt;
    const auto& call = instructions[call_index];
    if (call.operands.size() != 3) return std::nullopt;
    const auto arguments = grouped_names(call.operands[2]);
    const auto returns = grouped_names(call.operands[0]);
    if (arguments.size() != 1 || returns.size() != 1) return std::nullopt;
    const auto& argument_store = instructions[call_index - 1];
    if (argument_store.opcode != "st.param.b64" || !argument_store.predicate.empty() ||
        argument_store.operands.size() != 2 ||
        parameter_name_from_operand(argument_store.operands[0]) != arguments.front() ||
        !slot_at(argument_store.operands[0], arguments.front(), 0)) return std::nullopt;
    const std::string argument = trim(argument_store.operands[1]);
    if (first_register(argument) != argument) {
        try {
            std::size_t consumed = 0;
            (void)std::stoll(argument, &consumed, 0);
            if (consumed != argument.size()) return std::nullopt;
        } catch (...) { return std::nullopt; }
    } else if (ptx_register_container_bits(argument) != 64) return std::nullopt;

    // The supported continuation is two scalar loads of the recursive result,
    // followed by optional join labels, two identical return stores and ret.
    // No computation, branch, memory effect or predication may intervene.
    std::map<std::int64_t, std::string> forwarded;
    std::set<std::size_t> remove;
    std::size_t cursor = call_index + 1;
    for (int lane = 0; lane < 2; ++lane, ++cursor) {
        if (cursor >= instructions.size()) return std::nullopt;
        const auto& load = instructions[cursor];
        if (load.opcode != "ld.param.b64" || !load.predicate.empty() || load.operands.size() != 2 ||
            parameter_name_from_operand(load.operands[1]) != returns.front() ||
            ptx_register_container_bits(load.operands[0]) != 64) return std::nullopt;
        const auto offset = slot_at(load.operands[1], returns.front(), 0) ? 0 :
                            slot_at(load.operands[1], returns.front(), 8) ? 8 : -1;
        if ((offset != 0 && offset != 8) || !forwarded.emplace(offset, load.operands[0]).second) return std::nullopt;
        remove.insert(cursor);
    }
    if (forwarded.at(0) == forwarded.at(8)) return std::nullopt;
    while (cursor < instructions.size() && instructions[cursor].opcode == "ptx.label") ++cursor;
    std::set<std::int64_t> stored;
    for (int lane = 0; lane < 2; ++lane, ++cursor) {
        if (cursor >= instructions.size()) return std::nullopt;
        const auto& store = instructions[cursor];
        if (store.opcode != "st.param.b64" || !store.predicate.empty() || store.operands.size() != 2 ||
            parameter_name_from_operand(store.operands[0]) != return_parameter) return std::nullopt;
        const auto offset = slot_at(store.operands[0], return_parameter, 0) ? 0 :
                            slot_at(store.operands[0], return_parameter, 8) ? 8 : -1;
        if (!forwarded.contains(offset) || store.operands[1] != forwarded.at(offset) || !stored.insert(offset).second) return std::nullopt;
    }
    if (cursor + 1 != instructions.size() || instructions[cursor].opcode != "ret" ||
        !instructions[cursor].predicate.empty()) return std::nullopt;

    const auto& first = instructions.front();
    if ((first.opcode != "ld.param.u64" && first.opcode != "ld.param.b64") ||
        !first.predicate.empty() || first.operands.size() != 2 ||
        parameter_name_from_operand(first.operands[1]) != parameter ||
        !slot_at(first.operands[1], parameter, 0) ||
        ptx_register_container_bits(first.operands[0]) != 64) return std::nullopt;
    // No hidden parameter-slot aliases or extra reads/stores are allowed.
    for (std::size_t i = 1; i < instructions.size(); ++i) {
        const auto& instruction = instructions[i];
        if (starts_with(instruction.opcode, "ld.param") && !remove.contains(i)) return std::nullopt;
        if (starts_with(instruction.opcode, "st.param") && i != call_index - 1 && i < cursor - 2) return std::nullopt;
    }
    return ScalarTailPlan{call_index, std::move(remove), argument};
}

void rewrite_scalar_tail_call(cumetal::ptx::EntryFunction* function, const ScalarTailPlan& plan) {
    auto& instructions = function->instructions;
    const auto& first = instructions.front();
    const auto& call = instructions[plan.call_index];
    const auto& [call_index, remove, argument] = plan;
    const auto current = unused_name(*function, "%rd_cm_tail_argument");
    const auto header = unused_name(*function, "$cm_tail_header");
    const auto make = [&](std::string opcode, std::vector<std::string> operands, int line) {
        Instruction result;
        result.opcode = std::move(opcode); result.operands = std::move(operands);
        result.line = line; result.supported = true;
        return result;
    };
    std::vector<Instruction> rewritten;
    rewritten.push_back(make(first.opcode, {current, first.operands[1]}, first.line));
    rewritten.push_back(make("ptx.label", {header}, first.line));
    rewritten.push_back(make("mov.u64", {first.operands[0], current}, first.line));
    for (std::size_t i = 1; i < instructions.size(); ++i) {
        if (i == call_index - 1 || remove.contains(i)) continue;
        if (i == call_index) {
            rewritten.push_back(make("mov.u64", {current, argument}, call.line));
            rewritten.push_back(make("bra.uni", {header}, call.line));
        } else rewritten.push_back(instructions[i]);
    }
    instructions = std::move(rewritten);
    function->register_declarations.push_back({current, "b64"});
}

// Reuse a local frame only for a read-all / replace-all tail-call diamond.
// The input is consumed before any frame write, addresses cannot escape, and
// the continuation only forwards the complete result. No iteration cap or
// knowledge of a particular RNG's constants is needed.
std::optional<LocalTailPlan> match_local_tail_call(const cumetal::ptx::EntryFunction* function,
                                     const std::unordered_map<std::string, LocalDepot>& depots) {
    auto& code = function->instructions;
    if (function->params.size() != 1 || function->params[0].byte_size != 8 ||
        function->return_params.size() != 1 || function->return_params[0].byte_size != 16 ||
        code.size() < 17) return std::nullopt;
    const auto exact = [&](std::size_t i, std::string_view opcode, std::size_t operands) {
        return i < code.size() && code[i].opcode == opcode && code[i].supported &&
               code[i].predicate.empty() && code[i].operands.size() == operands;
    };
    if (!exact(0, "mov.b64", 2) || !exact(1, "cvta.local.u64", 2) ||
        !exact(2, "ld.param.b64", 2) || !exact(3, "cvta.to.local.u64", 2)) return std::nullopt;
    const std::string local = code[0].operands[0], generic = code[1].operands[0];
    const std::string input = code[2].operands[0], read = code[3].operands[0];
    const auto depot = depots.find(code[0].operands[1]);
    if (depot == depots.end() || depot->second.byte_size != 16 || depot->second.alignment < 16 ||
        code[1].operands[1] != local || code[3].operands[1] != input ||
        parameter_slot_offset(code[2].operands[1], function->params[0].name) != 0) return std::nullopt;
    std::set<std::string> pointers = {local, generic, input, read};
    if (pointers.size() != 4) return std::nullopt;
    for (const auto& pointer : pointers)
        if (first_register(pointer) != pointer) return std::nullopt;

    std::set<int> bytes;
    std::set<std::string> scalars;
    const auto scalar = [&](const std::string& operand) {
        if (scalars.contains(operand)) return true;
        try {
            std::size_t consumed = 0;
            (void)std::stoll(operand, &consumed, 0);
            return consumed == operand.size();
        } catch (...) { return false; }
    };
    std::size_t branch = 4;
    for (; branch < code.size() && root_opcode(code[branch].opcode) != "bra"; ++branch) {
        const auto& instruction = code[branch];
        if (!instruction.predicate.empty() || !instruction.supported || instruction.operands.empty()) return std::nullopt;
        const auto& destination = instruction.operands[0];
        if (first_register(destination) != destination || pointers.contains(destination)) return std::nullopt;
        if (instruction.opcode == "ld.local.b8" && instruction.operands.size() == 2) {
            const auto offset = parameter_slot_offset(instruction.operands[1], read);
            if (!offset || *offset < 0 || *offset >= 16 || !bytes.insert(static_cast<int>(*offset)).second) return std::nullopt;
        } else {
            // These integer operations cover byte assembly and its predicate;
            // no addresses, calls, stores, labels or other control flow occur.
            if ((instruction.opcode != "shl.b64" && instruction.opcode != "or.b64" &&
                 instruction.opcode != "setp.eq.b64") || instruction.operands.size() != 3 ||
                !scalar(instruction.operands[1]) || !scalar(instruction.operands[2])) return std::nullopt;
        }
        scalars.insert(destination);
    }
    if (bytes.size() != 16 || branch + 12 != code.size()) return std::nullopt;
    if (code[branch].opcode != "bra" || code[branch].predicate.empty() ||
        code[branch].operands.size() != 1 || !exact(branch+1, "bra.uni", 1) ||
        code[branch+2].opcode != "ptx.label" || code[branch+2].operands.size() != 1 ||
        code[branch].operands[0] != code[branch+2].operands[0]) return std::nullopt;
    if (!exact(branch+3, "add.u64", 3) || !exact(branch+4, "add.u64", 3) ||
        code[branch+3].operands[1] != generic || code[branch+3].operands[2] != "0" ||
        code[branch+4].operands[1] != local || code[branch+4].operands[2] != "0") return std::nullopt;
    const std::string argument_pointer = code[branch+3].operands[0];
    const std::string store_pointer = code[branch+4].operands[0];
    for (const auto& pointer : {argument_pointer, store_pointer}) {
        if (first_register(pointer) != pointer || scalars.contains(pointer) || !pointers.insert(pointer).second) return std::nullopt;
    }
    if (!exact(branch+5, "st.local.v2.b64", 2) ||
        parameter_slot_offset(code[branch+5].operands[0], store_pointer) != 0) return std::nullopt;
    const auto tuple = [](const std::string& operand) {
        const std::string value = trim(operand);
        if (value.size() < 2 || value.front() != '{' || value.back() != '}') return std::vector<std::string>{};
        return grouped_names(value.substr(1, value.size() - 2));
    };
    const auto stored = tuple(code[branch+5].operands[1]);
    if (stored.size() != 2 || !scalar(stored[0]) || !scalar(stored[1])) return std::nullopt;
    if (!exact(branch+6, "st.param.b64", 2) || code[branch+6].operands[1] != argument_pointer ||
        !exact(branch+7, "call.uni", 3) || direct_call_target(code[branch+7]) != function->name) return std::nullopt;
    const auto arguments = grouped_names(code[branch+7].operands[2]);
    const auto results = grouped_names(code[branch+7].operands[0]);
    if (arguments.size() != 1 || results.size() != 1 ||
        parameter_slot_offset(code[branch+6].operands[0], arguments[0]) != 0 ||
        !exact(branch+8, "ld.param.v2.b64", 2) ||
        parameter_slot_offset(code[branch+8].operands[1], results[0]) != 0) return std::nullopt;
    if (code[branch+9].opcode != "ptx.label" || code[branch+9].operands.size() != 1 ||
        code[branch+1].operands[0] != code[branch+9].operands[0] ||
        code[branch+2].operands[0] == code[branch+9].operands[0] ||
        !exact(branch+10, "st.param.v2.b64", 2) || !exact(branch+11, "ret", 0) ||
        parameter_slot_offset(code[branch+10].operands[0], function->return_params[0].name) != 0) return std::nullopt;
    const auto forwarded = tuple(code[branch+8].operands[0]);
    if (forwarded.size() != 2 || forwarded[0] == forwarded[1] ||
        tuple(code[branch+10].operands[1]) != forwarded) return std::nullopt;
    for (const auto& value : forwarded)
        if (!scalars.contains(value) || ptx_register_container_bits(value) != 64) return std::nullopt;

    return LocalTailPlan{branch, input, argument_pointer};
}

void rewrite_local_tail_call(cumetal::ptx::EntryFunction* function, const LocalTailPlan& plan) {
    auto& code = function->instructions;
    const auto& [branch, input, argument_pointer] = plan;
    const auto header = unused_name(*function, "$cm_local_tail_header");
    Instruction label;
    label.opcode = "ptx.label"; label.operands = {header}; label.supported = true;
    label.line = code[3].line;
    Instruction update = code[2];
    update.opcode = "mov.b64"; update.operands = {input, argument_pointer};
    Instruction jump = code[branch+1]; jump.operands = {header};
    std::vector<Instruction> rewritten;
    for (std::size_t i = 0; i < code.size(); ++i) {
        if (i == 3) rewritten.push_back(label);
        if (i == branch+6 || i == branch+8) continue;
        if (i == branch+7) {
            rewritten.push_back(update);
            rewritten.push_back(jump);
        } else rewritten.push_back(code[i]);
    }
    code = std::move(rewritten);
}

}  // namespace

void normalize_tail_calls(cumetal::ptx::EntryFunction& function,
                          const std::unordered_map<std::string, LocalDepot>& depots) {
    if (const auto plan = match_scalar_tail_call(&function)) {
        rewrite_scalar_tail_call(&function, *plan);
    } else if (const auto plan = match_local_tail_call(&function, depots)) {
        rewrite_local_tail_call(&function, *plan);
    }
}

}  // namespace cumetal::ir::detail
