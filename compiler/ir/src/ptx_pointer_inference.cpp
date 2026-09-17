#include "ptx_pointer_inference.h"
#include "ptx_instruction.h"
#include "ptx_text.h"
#include <algorithm>
#include <cctype>
#include <optional>

namespace cumetal::ir::detail {

namespace {

bool has_integer_64_bit_type(std::string_view opcode) {
    for (const std::string_view type : {".b64", ".u64", ".s64"}) {
        std::size_t position = opcode.find(type);
        while (position != std::string::npos) {
            const std::size_t end = position + type.size();
            if (end == opcode.size() || opcode[end] == '.') return true;
            position = opcode.find(type, position + 1);
        }
    }
    return false;
}

bool is_64_bit_load(const Instruction& instruction) {
    return root_opcode(instruction.opcode) == "ld" &&
        !starts_with(instruction.opcode, "ld.param") &&
        instruction.operands.size() == 2 && has_integer_64_bit_type(instruction.opcode);
}

}  // namespace

PointerInference infer_entry_pointer_types(const ptx::EntryFunction& entry, const Module& module,
    bool is_kernel, const std::unordered_set<std::string>& pointer_symbols,
    const std::unordered_set<std::string>& promoted_global_symbols,
    std::unordered_map<std::string, Type>& parameter_types) {
    PointerInference evidence;
    // Preserve established pointer evidence before backward recovery. Only
    // single-definition, unpredicated 64-bit registers participate in this proof.
    const auto is_declared64 = [&](const std::string& name) {
        const auto is_integer64 = [](const std::string& type) {
            return type == "b64" || type == "u64" || type == "s64";
        };
        for (const auto& declaration : entry.register_declarations)
            if (declaration.name == name) return is_integer64(declaration.type);
        for (const auto& range : entry.register_ranges) {
            if (!starts_with(name, range.prefix)) continue;
            const std::string suffix = name.substr(range.prefix.size());
            if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(),
                    [](unsigned char c) { return std::isdigit(c); })) continue;
            try {
                if (std::stoull(suffix) < range.count) return is_integer64(range.type);
            } catch (...) {}
        }
        return false;
    };
    std::unordered_map<std::string, std::size_t> definitions;
    std::unordered_map<std::string, const Instruction*> defining_instructions;
    for (const auto& instruction : entry.instructions)
        for (const auto& destination : destination_registers(instruction)) {
            ++definitions[destination];
            defining_instructions[destination] = &instruction;
        }
    std::unordered_set<std::string> known_pointers;
    bool known_changed = true;
    for (int iteration = 0; iteration < 12 && known_changed; ++iteration) {
        known_changed = false;
        for (const auto& instruction : entry.instructions) {
            const auto destinations = destination_registers(instruction);
            if (destinations.size() != 1 || definitions[destinations.front()] != 1 ||
                !instruction.predicate.empty() || instruction.operands.size() < 2 ||
                !is_declared64(destinations.front())) continue;
            const auto root = root_opcode(instruction.opcode);
            const auto known = [&](std::size_t index) {
                return instruction.operands.size() > index &&
                    known_pointers.contains(first_register(instruction.operands[index]));
            };
            bool pointer = root == "cvta";
            if (starts_with(instruction.opcode, "ld.param")) {
                const auto parameter = parameter_types.find(parameter_name_from_operand(instruction.operands[1]));
                pointer = parameter != parameter_types.end() && parameter->second.is_pointer();
            } else if (root == "mov" && instruction.operands[1].find('{') == std::string::npos) {
                pointer = known(1) || pointer_symbols.contains(parameter_name_from_operand(instruction.operands[1]));
            } else if (root == "add") {
                pointer = known(1) != known(2);
            } else if (root == "sub") {
                pointer = known(1) && !known(2);
            }
            if (pointer && known_pointers.insert(destinations.front()).second) known_changed = true;
            // A promoted PTX global remains physical Metal constant storage.
            // Ordinary .const symbols and unrelated conversions are excluded.
            const auto promoted_source = [&](std::size_t index) {
                return instruction.operands.size() > index && evidence.promoted_global_registers.contains(
                    first_register(instruction.operands[index]));
            };
            bool promoted = false;
            if (root == "mov" && instruction.operands[1].find('{') == std::string::npos)
                promoted = promoted_source(1) || promoted_global_symbols.contains(
                    parameter_name_from_operand(instruction.operands[1]));
            else if (root == "add") promoted = promoted_source(1) != promoted_source(2);
            else if (root == "sub") promoted = promoted_source(1) && !promoted_source(2);
            else if (instruction.opcode == "cvta.global.u64" || instruction.opcode == "cvta.to.global.u64")
                promoted = promoted_source(1);
            if (promoted && evidence.promoted_global_registers.insert(destinations.front()).second)
                known_changed = true;
        }
    }

    // Older CUDA Clang PTX (notably 21) omits `.ptr` from device-function
    // parameters even when the CUDA source type is a pointer. Recover that
    // information from actual address use before forward type inference.
    // This is deliberately bounded to direct dataflow through the common
    // mov/ld.param, add, and selp forms; ambiguous integer-only values stay
    // integers instead of being guessed as pointers.
    std::unordered_map<std::string, AddressSpace> required_pointers;
    const auto require_pointer = [&](const std::string& name, AddressSpace space) {
        if (name.empty()) return false;
        const auto [found, inserted] = required_pointers.emplace(name, space);
        if (inserted) return true;
        if (found->second == AddressSpace::kNone && space != AddressSpace::kNone) {
            found->second = space;
            return true;
        }
        return false;
    };
    // Reachable helpers are imported before their callers. A parameter slot
    // supplies pointer evidence only when every use of that slot name agrees
    // on the pointer contract. PTX reuses lexical argument-slot names across
    // unrelated calls; a later pointer call must not retag an earlier scalar.
    std::unordered_map<std::string, std::optional<Type>> pointer_slots;
    for (const Instruction& instruction : entry.instructions) {
        if (root_opcode(instruction.opcode) != "call" || instruction.operands.empty()) continue;
        const auto callee = direct_call_target(instruction);
        const auto imported = std::find_if(module.functions.begin(), module.functions.end(),
            [&](const Function& function) { return callee && function.name == *callee; });
        const auto arguments = grouped_names(instruction.operands.back());
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            std::optional<Type> contract;
            if (imported != module.functions.end() && i < imported->arguments.size() &&
                imported->arguments[i].type.is_pointer()) contract = imported->arguments[i].type;
            const auto [slot, inserted] = pointer_slots.emplace(arguments[i], contract);
            if (!inserted && slot->second != contract) slot->second.reset();
        }
    }
    std::unordered_map<std::string, std::optional<std::string>> slot_sources;
    for (const Instruction& instruction : entry.instructions) {
        if (!starts_with(instruction.opcode, "st.param") || instruction.operands.size() < 2) continue;
        const auto slot = parameter_name_from_operand(instruction.operands[0]);
        if (!pointer_slots.contains(slot) || !pointer_slots.at(slot)) continue;
        const auto source = first_register(instruction.operands[1]);
        std::optional<std::string> evidence;
        if ((instruction.opcode == "st.param.b64" || instruction.opcode == "st.param.u64") &&
            instruction.predicate.empty() && trim(instruction.operands[0]) == "[" + slot + "]" &&
            !source.empty() && trim(instruction.operands[1]) == source && definitions[source] == 1 &&
            defining_instructions.at(source)->predicate.empty() && is_declared64(source)) evidence = source;
        const auto [known, inserted] = slot_sources.emplace(slot, evidence);
        if (!inserted && known->second != evidence) known->second.reset();
    }
    for (const Instruction& instruction : entry.instructions) {
        if ((instruction.opcode == "st.param.b64" || instruction.opcode == "st.param.u64") &&
            instruction.operands.size() == 2) {
            const auto slot = parameter_name_from_operand(instruction.operands[0]);
            if (pointer_slots.contains(slot) && pointer_slots.at(slot) && slot_sources.contains(slot) && slot_sources.at(slot) &&
                trim(instruction.operands[0]) == "[" + slot + "]") {
                const auto source = first_register(instruction.operands[1]);
                require_pointer(source, AddressSpace::kNone);
            }
        }
    }
    for (const Instruction& instruction : entry.instructions) {
        const std::string root = root_opcode(instruction.opcode);
        // A helper's explicit generic-to-local address conversion proves
        // pointer-ness even when all subsequent accesses are ld.local.
        // Keep its argument generic; call-site specialization determines
        // the actual address space rather than guessing from integer width.
        if (!is_kernel && instruction.opcode == "cvta.to.local.u64" &&
            instruction.operands.size() == 2) {
            const std::string source = first_register(instruction.operands[1]);
            require_pointer(source, AddressSpace::kNone);
        }
        if (root != "ld" && root != "st") continue;
        if (instruction.opcode.find(".param") != std::string::npos ||
            instruction.opcode.find(".shared") != std::string::npos ||
            instruction.opcode.find(".local") != std::string::npos ||
            instruction.opcode.find(".const") != std::string::npos) {
            continue;
        }
        const std::size_t memory_index = root == "st" ? 0 : 1;
        if (instruction.operands.size() <= memory_index) continue;
        const std::string base =
            first_register(instruction.operands[memory_index]);
        require_pointer(base, instruction.opcode.find(".global") != std::string::npos
            ? AddressSpace::kDevice : AddressSpace::kNone);
    }
    bool pointer_changed = true;
    for (int iteration = 0; iteration < 12 && pointer_changed; ++iteration) {
        pointer_changed = false;
        for (const Instruction& instruction : entry.instructions) {
            const std::vector<std::string> destinations =
                destination_registers(instruction);
            if (std::none_of(destinations.begin(), destinations.end(),
                             [&](const std::string& destination) {
                                 return required_pointers.contains(destination);
                             })) {
                continue;
            }
            // Each vector destination is an independent SSA value. Address
            // demand for a pointer lane says nothing about adjacent lengths
            // or pointers in a different address space. Keep the same unique,
            // unconditional-definition boundary as scalar load recovery.
            // The cell address does not inherit its payload's address space.
            if (!is_kernel && is_64_bit_load(instruction) && instruction.predicate.empty()) {
                for (std::size_t lane = 0; lane < destinations.size(); ++lane) {
                    const auto& destination = destinations[lane];
                    const auto required = required_pointers.find(destination);
                    if (required != required_pointers.end() && definitions[destination] == 1 &&
                        is_declared64(destination))
                        evidence.pointer_loads[&instruction][lane] = required->second;
                }
                continue;
            }
            // This pre-SSA recovery can attach a name-wide demand only when
            // the name denotes one unconditional definition. In particular,
            // a later pointer assignment must not retag an earlier scalar
            // source copied into the same mutable PTX register.
            if (destinations.size() != 1 || definitions[destinations.front()] != 1 ||
                !instruction.predicate.empty() || !is_declared64(destinations.front()) ||
                !has_integer_64_bit_type(instruction.opcode)) continue;
            AddressSpace required_space = AddressSpace::kNone;
            for (const auto& destination : destinations) {
                const auto required = required_pointers.find(destination);
                if (required != required_pointers.end() && required->second != AddressSpace::kNone)
                    required_space = required->second;
            }
            const std::string root = root_opcode(instruction.opcode);
            std::vector<std::size_t> pointer_sources;
            if ((root == "mov" && instruction.operands.size() == 2 &&
                 instruction.operands[1].find('{') == std::string::npos) ||
                starts_with(instruction.opcode, "ld.param")) {
                pointer_sources = {1};
            } else if (root == "add") {
                const bool right = instruction.operands.size() > 2 &&
                    known_pointers.contains(first_register(instruction.operands[2]));
                const bool left = instruction.operands.size() > 1 &&
                    known_pointers.contains(first_register(instruction.operands[1]));
                pointer_sources = {right && !left ? 2U : 1U};
            } else if (root == "selp") {
                pointer_sources = {1, 2};
            }
            for (const std::size_t source_index : pointer_sources) {
                if (instruction.operands.size() <= source_index) continue;
                const std::string source =
                    first_register(instruction.operands[source_index]);
                if (require_pointer(source, required_space)) {
                    pointer_changed = true;
                }
                const std::string parameter = parameter_name_from_operand(
                    instruction.operands[source_index]);
                const auto parameter_type_it = parameter_types.find(parameter);
                if (parameter_type_it != parameter_types.end() &&
                    !parameter_type_it->second.is_pointer()) {
                    parameter_type_it->second = Type::pointer(
                        Type::integer(8), AddressSpace::kDevice);
                    pointer_changed = true;
                }
            }
        }
    }

    return evidence;
}

}  // namespace cumetal::ir::detail
