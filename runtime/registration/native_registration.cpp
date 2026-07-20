#include "native_registration.h"

#include "cumetal/common/metallib.h"
#include "cumetal_native.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cumetal::native_registration {
namespace {

struct NativeModule {
    std::string metallib_path;
    std::vector<const void*> host_functions;
};

struct State {
    std::mutex mutex;
    std::unordered_map<CuMetalModuleHandle, std::unique_ptr<NativeModule>> modules;
    std::unordered_map<const void*, cumetal::registration::RegisteredKernel> kernels;
};

State& state() {
    static State value;
    return value;
}

std::uint64_t fnv1a(const std::uint8_t* bytes, std::size_t size) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::filesystem::path cache_root() {
    if (const char* configured = std::getenv("CUMETAL_CACHE_DIR");
        configured != nullptr && configured[0] != '\0') {
        return std::filesystem::path(configured) / "native-aot";
    }
    if (const char* user_home = std::getenv("HOME");
        user_home != nullptr && user_home[0] != '\0') {
        return std::filesystem::path(user_home) / "Library" / "Caches" /
               "io.cumetal" / "native-aot";
    }
    return std::filesystem::temp_directory_path() / "io.cumetal" / "native-aot";
}

std::filesystem::path materialize_metallib(const CuMetalModuleDescriptor& descriptor,
                                           std::string* error) {
    const auto* bytes = static_cast<const std::uint8_t*>(descriptor.metallib_data);
    const std::uint64_t digest = fnv1a(bytes, descriptor.metallib_size);
    std::ostringstream name;
    name << "abi" << descriptor.abi_version << "-" << std::hex
         << std::setfill('0') << std::setw(16) << digest << ".metallib";

    const std::filesystem::path root = cache_root();
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        if (error != nullptr) *error = "failed to create native AOT cache: " + ec.message();
        return {};
    }
    const std::filesystem::path output = root / name.str();
    if (std::filesystem::exists(output, ec) && !ec) return output;
    const std::vector<std::uint8_t> data(bytes, bytes + descriptor.metallib_size);
    if (!cumetal::common::write_file_bytes(output, data, error)) return {};
    return output;
}

bool validate_descriptor(const CuMetalModuleDescriptor* descriptor,
                         std::string* error) {
    if (descriptor == nullptr ||
        descriptor->abi_version != CUMETAL_NATIVE_ABI_VERSION ||
        descriptor->metallib_data == nullptr || descriptor->metallib_size == 0 ||
        descriptor->kernel_count == 0 || descriptor->kernels == nullptr) {
        if (error != nullptr) *error = "invalid CuMetal native module descriptor";
        return false;
    }
    if (descriptor->binding_count > 0 && descriptor->bindings == nullptr) {
        if (error != nullptr) *error = "native module binding table is missing";
        return false;
    }
    std::unordered_set<const void*> host_stubs;
    for (std::uint32_t i = 0; i < descriptor->kernel_count; ++i) {
        const CuMetalKernelDescriptor& kernel = descriptor->kernels[i];
        if (kernel.cuda_name == nullptr || kernel.metal_name == nullptr ||
            kernel.host_stub == nullptr || kernel.required_simd_width != 32 ||
            (kernel.argument_count > 0 && kernel.arguments == nullptr) ||
            !host_stubs.insert(kernel.host_stub).second) {
            if (error != nullptr) *error = "invalid CuMetal native kernel descriptor";
            return false;
        }
        for (std::uint32_t argument_index = 0;
             argument_index < kernel.argument_count; ++argument_index) {
            const CuMetalArgumentDescriptor& argument =
                kernel.arguments[argument_index];
            if (argument.size == 0 || argument.alignment == 0 ||
                argument.first_binding > descriptor->binding_count ||
                argument.binding_count >
                    descriptor->binding_count - argument.first_binding) {
                if (error != nullptr) *error = "invalid CuMetal native argument descriptor";
                return false;
            }
            for (std::uint32_t binding_offset = 0;
                 binding_offset < argument.binding_count; ++binding_offset) {
                const CuMetalBindingDescriptor& binding =
                    descriptor->bindings[argument.first_binding + binding_offset];
                if (binding.logical_argument_index != argument_index ||
                    binding.size == 0 || binding.alignment == 0) {
                    if (error != nullptr) *error = "invalid CuMetal native binding descriptor";
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace

bool lookup_kernel(const void* host_function,
                   cumetal::registration::RegisteredKernel* out) {
    if (host_function == nullptr || out == nullptr) return false;
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto found = s.kernels.find(host_function);
    if (found == s.kernels.end()) return false;
    *out = found->second;
    return true;
}

void clear() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.kernels.clear();
    s.modules.clear();
}

}  // namespace cumetal::native_registration

extern "C" {

CuMetalModuleHandle cumetalRegisterModule(
    const CuMetalModuleDescriptor* descriptor) {
    std::string error;
    if (!cumetal::native_registration::validate_descriptor(descriptor, &error)) {
        return nullptr;
    }
    const std::filesystem::path metallib =
        cumetal::native_registration::materialize_metallib(*descriptor, &error);
    if (metallib.empty()) return nullptr;

    auto module = std::make_unique<cumetal::native_registration::NativeModule>();
    module->metallib_path = metallib.string();
    CuMetalModuleHandle handle = module.get();

    cumetal::native_registration::State& s =
        cumetal::native_registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (std::uint32_t i = 0; i < descriptor->kernel_count; ++i) {
        if (s.kernels.contains(descriptor->kernels[i].host_stub)) {
            return nullptr;
        }
    }
    for (std::uint32_t i = 0; i < descriptor->kernel_count; ++i) {
        const CuMetalKernelDescriptor& kernel = descriptor->kernels[i];
        cumetal::registration::RegisteredKernel record;
        record.metallib_path = module->metallib_path;
        record.kernel_name = kernel.metal_name;
        record.static_shared_bytes = kernel.static_threadgroup_memory;
        record.provenance =
            descriptor->provenance != nullptr
                ? descriptor->provenance
                : "generic_nvvm_lowering";
        record.semantic_quality =
            descriptor->semantic_quality != nullptr
                ? descriptor->semantic_quality
                : "exact";
        for (std::uint32_t argument_index = 0;
             argument_index < kernel.argument_count; ++argument_index) {
            const CuMetalArgumentDescriptor& argument =
                kernel.arguments[argument_index];
            record.arg_info.push_back({
                .kind = argument.kind == CUMETAL_NATIVE_ARGUMENT_POINTER
                            ? CUMETAL_ARG_BUFFER
                            : CUMETAL_ARG_BYTES,
                .size_bytes = argument.size,
            });
        }
        module->host_functions.push_back(kernel.host_stub);
        s.kernels[kernel.host_stub] = std::move(record);
    }
    s.modules[handle] = std::move(module);
    return handle;
}

void cumetalUnregisterModule(CuMetalModuleHandle module_handle) {
    if (module_handle == nullptr) return;
    cumetal::native_registration::State& s =
        cumetal::native_registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto module = s.modules.find(module_handle);
    if (module == s.modules.end()) return;
    for (const void* host_function : module->second->host_functions) {
        s.kernels.erase(host_function);
    }
    s.modules.erase(module);
}

}  // extern "C"
