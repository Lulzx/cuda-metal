#include "shared_spill.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace cumetal::metal {
namespace {

std::vector<std::string> split_callees(const std::string& list) {
    std::vector<std::string> names;
    std::size_t begin = 0;
    while (begin <= list.size()) {
        const std::size_t comma = list.find(',', begin);
        const std::size_t end = comma == std::string::npos ? list.size() : comma;
        if (end > begin) names.push_back(list.substr(begin, end - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return names;
}

std::vector<std::string> callees_of(const ir::Operation& operation) {
    std::vector<std::string> names;
    if (const auto callee = operation.attributes.find("callee");
        callee != operation.attributes.end()) {
        names.push_back(callee->second);
    }
    if (const auto callees = operation.attributes.find("callees");
        callees != operation.attributes.end()) {
        for (std::string& name : split_callees(callees->second)) names.push_back(std::move(name));
    }
    return names;
}

bool is_threadgroup_pointer(const ir::Type& type) {
    return type.is_pointer() && type.address_space == ir::AddressSpace::kThreadgroup;
}

void move_to_device(ir::Type* type) {
    if (type->address_space == ir::AddressSpace::kThreadgroup) {
        type->address_space = ir::AddressSpace::kDevice;
    }
    for (ir::Type& element : type->elements) move_to_device(&element);
}

bool is_shared_symbol(const ir::Operand& operand) {
    return operand.kind == ir::OperandKind::kSymbol && is_threadgroup_pointer(operand.type);
}

// Why a function cannot run with its shared memory in a device buffer, or
// empty when it can.
std::string spill_obstacle(const ir::Function& function) {
    if (!function.mixed_pointer_address_spaces.empty() ||
        function.mixed_pointer_return_spaces != 0) {
        return "generic pointers with several address spaces";
    }
    for (const ir::BasicBlock& block : function.blocks) {
        for (const ir::Operation& operation : block.operations) {
            if (operation.opcode == ir::OpCode::kAddressSpaceTest) {
                return "address-space query";
            }
            if (operation.attributes.contains("wmma_8x8")) return "simdgroup matrix tile";
            if (operation.opcode == ir::OpCode::kConvert ||
                operation.opcode == ir::OpCode::kAddressSpaceCast) {
                const bool from_shared =
                    !operation.operands.empty() && is_threadgroup_pointer(operation.operands[0].type);
                const bool to_shared = !operation.result_types.empty() &&
                                       is_threadgroup_pointer(operation.result_types[0]);
                if (from_shared || to_shared) {
                    const ir::Type& other =
                        from_shared ? operation.result_types[0] : operation.operands[0].type;
                    const bool device_like =
                        other.is_pointer() && (other.address_space == ir::AddressSpace::kDevice ||
                                               other.address_space == ir::AddressSpace::kNone ||
                                               other.address_space == ir::AddressSpace::kThreadgroup);
                    if (!device_like) return "shared pointer conversion";
                }
            }
        }
    }
    return {};
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return alignment <= 1 ? value : (value + alignment - 1) / alignment * alignment;
}

} // namespace

bool is_shared_spill_function(std::string_view name) {
    return name.find(kSharedSpillFunctionSuffix) != std::string_view::npos;
}

std::unordered_map<std::string, std::uint64_t> shared_spill_layout(const ir::Module& module,
                                                                   std::string_view kernel) {
    std::unordered_map<std::string, std::uint64_t> layout;
    const auto found =
        module.attributes.find(std::string(kSharedSpillLayoutAttributePrefix) + std::string(kernel));
    if (found == module.attributes.end()) return layout;
    std::istringstream entries(found->second);
    std::string entry;
    while (std::getline(entries, entry, ';')) {
        const std::size_t equals = entry.rfind('=');
        if (equals == std::string::npos) continue;
        layout[entry.substr(0, equals)] = std::stoull(entry.substr(equals + 1));
    }
    return layout;
}

ir::Module add_shared_spill_variants(const ir::Module& module, std::uint64_t threadgroup_limit) {
    ir::Module result = module;
    result.attributes[kSharedSpillDoneAttribute] = "1";

    std::unordered_map<std::string, const ir::Function*> by_name;
    for (const ir::Function& function : module.functions) by_name.emplace(function.name, &function);
    std::unordered_map<std::string, const ir::GlobalThreadgroup*> shared_globals;
    for (const ir::GlobalThreadgroup& global : module.global_threadgroups) {
        shared_globals.emplace(global.name, &global);
    }

    std::set<std::string> cloned;  // original names whose spill clone exists
    std::vector<ir::Function> clones;
    for (const ir::Function& kernel : module.functions) {
        if (!kernel.is_kernel || is_shared_spill_function(kernel.name)) continue;

        // Call closure in module order (callees before callers), and the
        // shared globals it names.
        std::unordered_set<std::string> reachable;
        std::vector<std::string> pending{kernel.name};
        while (!pending.empty()) {
            const std::string name = std::move(pending.back());
            pending.pop_back();
            const auto found = by_name.find(name);
            if (found == by_name.end() || !reachable.insert(name).second) continue;
            for (const ir::BasicBlock& block : found->second->blocks) {
                for (const ir::Operation& operation : block.operations) {
                    for (std::string& callee : callees_of(operation)) pending.push_back(std::move(callee));
                }
            }
        }
        std::set<std::string> used_globals;
        bool obstacle = false;
        for (const std::string& name : reachable) {
            const ir::Function& function = *by_name.at(name);
            if (function.is_kernel && name != kernel.name) continue;  // device launch target
            if (!spill_obstacle(function).empty()) obstacle = true;
            for (const ir::BasicBlock& block : function.blocks) {
                for (const ir::Operation& operation : block.operations) {
                    for (const ir::Operand& operand : operation.operands) {
                        if (is_shared_symbol(operand)) used_globals.insert(operand.text);
                    }
                }
            }
        }
        if (used_globals.empty() || obstacle) continue;

        // Layout: static arrays in name order, each at least 16-byte aligned,
        // then dynamic shared memory at a 16-byte boundary.
        std::uint64_t offset = 0;
        bool dynamic = false;
        std::string layout;
        std::vector<std::string> dynamic_names;
        for (const std::string& name : used_globals) {
            const auto global = shared_globals.find(name);
            if (global == shared_globals.end()) {
                obstacle = true;
                break;
            }
            if (global->second->is_dynamic) {
                dynamic = true;
                dynamic_names.push_back(name);
                continue;
            }
            offset = align_up(offset, std::max<std::uint64_t>(16, global->second->alignment));
            layout += name + "=" + std::to_string(offset) + ";";
            offset += global->second->byte_size;
        }
        if (obstacle) continue;
        const std::uint64_t static_bytes = align_up(offset, 16);
        for (const std::string& name : dynamic_names) {
            layout += name + "=" + std::to_string(static_bytes) + ";";
        }
        // Only a kernel that can exceed the Metal limit needs a variant.
        if (!dynamic && static_bytes <= threadgroup_limit) continue;

        // The buffer binds at index 24; a launch argument there would collide.
        bool binding_conflict = kernel.arguments.size() > kSharedSpillBindingIndex;
        if (kernel.kernel_abi.has_value()) {
            for (const ir::ArgumentDescriptor& argument : kernel.kernel_abi->arguments) {
                for (const std::uint32_t index : argument.binding_indices) {
                    binding_conflict |= index == kSharedSpillBindingIndex;
                }
            }
        }
        if (binding_conflict) continue;

        const std::string kernel_clone =
            kernel.name + std::string(kSharedSpillKernelInfix) + std::to_string(static_bytes);
        result.attributes[std::string(kSharedSpillLayoutAttributePrefix) + kernel_clone] = layout;

        const auto clone_name = [&](const std::string& name) {
            return name == kernel.name ? kernel_clone
                                       : name + std::string(kSharedSpillFunctionSuffix);
        };
        for (const ir::Function& function : module.functions) {
            if (!reachable.contains(function.name)) continue;
            if (function.is_kernel && function.name != kernel.name) continue;
            if (function.name != kernel.name && cloned.contains(function.name)) continue;
            if (function.name != kernel.name) cloned.insert(function.name);

            ir::Function clone = function;
            clone.name = clone_name(function.name);
            move_to_device(&clone.return_type);
            for (ir::FunctionArgument& argument : clone.arguments) move_to_device(&argument.type);
            if (clone.kernel_abi.has_value()) {
                for (ir::ArgumentDescriptor& argument : clone.kernel_abi->arguments) {
                    move_to_device(&argument.type);
                    if (argument.address_space == ir::AddressSpace::kThreadgroup) {
                        argument.address_space = ir::AddressSpace::kDevice;
                    }
                }
                for (ir::BindingDescriptor& binding : clone.kernel_abi->bindings) {
                    move_to_device(&binding.type);
                }
                clone.kernel_abi->static_threadgroup_memory = 0;
            }
            for (ir::BasicBlock& block : clone.blocks) {
                for (ir::BlockArgument& argument : block.arguments) move_to_device(&argument.type);
                for (ir::Operation& operation : block.operations) {
                    for (ir::Type& type : operation.result_types) move_to_device(&type);
                    // Shared globals become device pointers too; the lowering
                    // recognizes them by name and binds them to slices of the
                    // spill buffer.
                    for (ir::Operand& operand : operation.operands) move_to_device(&operand.type);
                    if (operation.opcode == ir::OpCode::kCall ||
                        operation.opcode == ir::OpCode::kIndirectCall) {
                        if (auto callee = operation.attributes.find("callee");
                            callee != operation.attributes.end() &&
                            reachable.contains(callee->second) &&
                            !by_name.at(callee->second)->is_kernel) {
                            callee->second = clone_name(callee->second);
                        }
                        if (auto callees = operation.attributes.find("callees");
                            callees != operation.attributes.end()) {
                            std::string renamed;
                            for (const std::string& name : split_callees(callees->second)) {
                                if (!renamed.empty()) renamed += ",";
                                renamed += reachable.contains(name) && !by_name.at(name)->is_kernel
                                               ? clone_name(name)
                                               : name;
                            }
                            callees->second = renamed;
                        }
                    }
                    if (operation.opcode == ir::OpCode::kMetalBarrier ||
                        operation.opcode == ir::OpCode::kBarrier ||
                        operation.opcode == ir::OpCode::kMetalFence ||
                        operation.opcode == ir::OpCode::kFence) {
                        operation.attributes["shared_in_device_memory"] = "true";
                    }
                    // cm_cta_any/cm_cta_count already fence device memory.
                }
            }
            clones.push_back(std::move(clone));
        }
    }
    // Clones follow every original so callees still precede their callers.
    for (ir::Function& clone : clones) result.functions.push_back(std::move(clone));
    return result;
}

} // namespace cumetal::metal
