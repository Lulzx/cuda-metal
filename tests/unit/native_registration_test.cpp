#include "cumetal_native.h"
#include "native_registration.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {

void host_stub() {}

int fail(const std::string& message) {
    std::cerr << "native registration test failed: " << message << "\n";
    return 1;
}

}  // namespace

int main() {
    const std::filesystem::path cache =
        std::filesystem::temp_directory_path() /
        ("cumetal-native-registration-" + std::to_string(getpid()));
    if (setenv("CUMETAL_CACHE_DIR", cache.c_str(), 1) != 0) {
        return fail("could not set isolated cache directory");
    }

    const unsigned char metallib[] = {0x4d, 0x54, 0x4c, 0x42};
    const CuMetalArgumentDescriptor arguments[] = {{
        .kind = CUMETAL_NATIVE_ARGUMENT_POINTER,
        .size = 8,
        .alignment = 8,
        .address_space = CUMETAL_NATIVE_ADDRESS_DEVICE,
        .first_binding = 0,
        .binding_count = 1,
    }};
    const CuMetalBindingDescriptor bindings[] = {{
        .kind = CUMETAL_NATIVE_BINDING_BUFFER,
        .metal_index = 0,
        .logical_argument_index = 0,
        .size = 8,
        .alignment = 8,
    }};
    const CuMetalKernelDescriptor kernels[] = {{
        .cuda_name = "vector_add",
        .metal_name = "vector_add",
        .host_stub = reinterpret_cast<const void*>(&host_stub),
        .argument_count = 1,
        .arguments = arguments,
        .static_threadgroup_memory = 64,
        .required_simd_width = 32,
    }};
    CuMetalModuleDescriptor descriptor = {
        .abi_version = CUMETAL_NATIVE_ABI_VERSION,
        .metallib_data = metallib,
        .metallib_size = sizeof(metallib),
        .kernel_count = 1,
        .kernels = kernels,
        .binding_count = 1,
        .bindings = bindings,
        .provenance = "generic_nvvm_lowering",
        .semantic_quality = "exact",
    };

    const CuMetalModuleHandle module = cumetalRegisterModule(&descriptor);
    if (module == nullptr) return fail("valid descriptor was rejected");

    cumetal::registration::RegisteredKernel registered;
    if (!cumetal::native_registration::lookup_kernel(
            reinterpret_cast<const void*>(&host_stub), &registered)) {
        return fail("registered host stub was not found");
    }
    if (registered.kernel_name != "vector_add" ||
        registered.static_shared_bytes != 64 ||
        registered.provenance != "generic_nvvm_lowering" ||
        registered.semantic_quality != "exact" ||
        registered.arg_info.size() != 1 ||
        registered.arg_info.front().kind != CUMETAL_ARG_BUFFER) {
        return fail("registered kernel metadata did not round-trip");
    }
    if (cumetalRegisterModule(&descriptor) != nullptr) {
        return fail("duplicate live host stub registration was accepted");
    }

    cumetalUnregisterModule(module);
    if (cumetal::native_registration::lookup_kernel(
            reinterpret_cast<const void*>(&host_stub), &registered)) {
        return fail("unregistered host stub remained visible");
    }

    descriptor.abi_version = CUMETAL_NATIVE_ABI_VERSION + 1;
    if (cumetalRegisterModule(&descriptor) != nullptr) {
        return fail("unsupported ABI version was accepted");
    }

    std::error_code error;
    std::filesystem::remove_all(cache, error);
    return 0;
}
