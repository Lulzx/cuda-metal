#include "ptx_pointer_inference.h"
#include "ptx_instruction.h"
#include "ptx_text.h"
#include <algorithm>

namespace cumetal::ir::detail {

PointerInference infer_entry_pointer_types(const ptx::EntryFunction& entry, const Module& module,
    bool is_kernel, const std::unordered_set<std::string>& pointer_symbols,
    const std::unordered_set<std::string>& promoted_global_symbols,
    std::unordered_map<std::string, Type>& parameter_types) {
    PointerInference evidence;
    // Preserve established pointer evidence before backward recovery. Only
    // single-definition, unpredicated 64-bit registers participate in this proof.
    std::unordered_set<std::string> declared64;
    for (const auto& declaration : entry.register_declarations) {
        if (declaration.type == "b64" || declaration.type == "u64" || declaration.type == "s64")
            declared64.insert(declaration.name);
    }
    std::unordered_map<std::string, std::size_t> definitions;
    for (const auto& instruction : entry.instructions)
        for (const auto& destination : destination_registers(instruction)) ++definitions[destination];
    std::unordered_set<std::string> known_pointers;
    bool known_changed = true;
    for (int iteration = 0; iteration < 12 && known_changed; ++iteration) {
        known_changed = false;
        for (const auto& instruction : entry.instructions) {
            const auto destinations = destination_registers(instruction);
            if (destinations.size() != 1 || definitions[destinations.front()] != 1 ||
                !instruction.predicate.empty() || instruction.operands.size() < 2 ||
                (ptx_register_container_bits(destinations.front()) != 64 &&
             !declared64.contains(destinations.front()))) continue;
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
    std::unordered_set<std::string> required_device_pointers;
    // Reachable helpers are imported before their callers. Forwarding a
    // scalar parameter slot to a proven pointer argument is pointer evidence
    // even if this function never dereferences the address itself.
    std::unordered_set<std::string> pointer_slots;
    for (const Instruction& instruction : entry.instructions) {
        const auto callee = direct_call_target(instruction);
        if (!callee) continue;
        const auto imported = std::find_if(module.functions.begin(), module.functions.end(),
            [&](const Function& function) { return function.name == *callee; });
        if (imported == module.functions.end()) continue;
        const auto arguments = grouped_names(instruction.operands.back());
        for (std::size_t i = 0; i < arguments.size() && i < imported->arguments.size(); ++i) {
            if (imported->arguments[i].type.is_pointer()) pointer_slots.insert(arguments[i]);
        }
    }
    for (const Instruction& instruction : entry.instructions) {
        if ((instruction.opcode == "st.param.b64" || instruction.opcode == "st.param.u64") &&
            instruction.operands.size() == 2) {
            const auto slot = parameter_name_from_operand(instruction.operands[0]);
            if (pointer_slots.contains(slot) && trim(instruction.operands[0]) == "[" + slot + "]") {
                const auto source = first_register(instruction.operands[1]);
                if (!source.empty()) required_device_pointers.insert(source);
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
            if (!source.empty()) required_device_pointers.insert(source);
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
        if (!base.empty()) required_device_pointers.insert(base);
    }
    bool pointer_changed = true;
    for (int iteration = 0; iteration < 12 && pointer_changed; ++iteration) {
        pointer_changed = false;
        for (const Instruction& instruction : entry.instructions) {
            const std::vector<std::string> destinations =
                destination_registers(instruction);
            if (std::none_of(destinations.begin(), destinations.end(),
                             [&](const std::string& destination) {
                                 return required_device_pointers.contains(destination);
                             })) {
                continue;
            }
            const std::string root = root_opcode(instruction.opcode);
            std::vector<std::size_t> pointer_sources;
            if (root == "mov" || starts_with(instruction.opcode, "ld.param")) {
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
                if (!source.empty() &&
                    required_device_pointers.insert(source).second) {
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
