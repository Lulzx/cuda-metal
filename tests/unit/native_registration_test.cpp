#include "cumetal_native.h"
#include "native_registration.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {

void host_stub() {}
unsigned char constant_shadow[8] = {1, 2, 3, 4, 5, 6, 7, 8};
unsigned char global_shadow[8] = {9, 10, 11, 12, 13, 14, 15, 16};

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
    const std::uint32_t kernel_symbol_indices[] = {0, 1};
    const char* const printf_formats[] = {"value=%d\n"};
    const CuMetalKernelDescriptor kernels[] = {{
        .cuda_name = "vector_add",
        .metal_name = "vector_add",
        .host_stub = reinterpret_cast<const void*>(&host_stub),
        .argument_count = 1,
        .arguments = arguments,
        .static_threadgroup_memory = 64,
        .required_simd_width = 32,
        .symbol_count = 2,
        .symbol_indices = kernel_symbol_indices,
        .printf_format_count = 1,
        .printf_formats = printf_formats,
    }};
    const CuMetalSymbolDescriptor symbols[] = {
        {
            .name = "constant_shadow",
            .host_symbol = constant_shadow,
            .size = sizeof(constant_shadow),
            .alignment = 8,
            .constant_offset = 16,
            .kind = CUMETAL_NATIVE_SYMBOL_CONSTANT,
        },
        {
            .name = "global_shadow",
            .host_symbol = global_shadow,
            .size = sizeof(global_shadow),
            .alignment = 8,
            .constant_offset = 0,
            .kind = CUMETAL_NATIVE_SYMBOL_GLOBAL,
        },
    };
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
        .symbol_count = 2,
        .symbols = symbols,
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
        registered.printf_formats.size() != 1 ||
        registered.printf_formats.front() != "value=%d\n" ||
        registered.arg_info.size() != 1 ||
        registered.arg_info.front().kind != CUMETAL_ARG_BUFFER) {
        return fail("registered kernel metadata did not round-trip");
    }
    if (registered.constant_symbols.size() != 1 ||
        registered.constant_symbols.front().offset != 16 ||
        registered.global_symbols.size() != 1 ||
        registered.global_symbols.front().buffer == nullptr) {
        return fail("native symbol metadata did not reach the registered kernel");
    }
    // Advisory function controls must validate registration identity without
    // trying to load this deliberately minimal fake metallib.
    if (cudaFuncSetCacheConfig(reinterpret_cast<const void*>(&host_stub),
                               cudaFuncCachePreferEqual) != cudaSuccess ||
        cudaFuncSetSharedMemConfig(reinterpret_cast<const void*>(&host_stub),
                                   cudaSharedMemBankSizeEightByte) != cudaSuccess) {
        return fail("advisory function controls forced metallib loading");
    }
    const void* resolved_symbol = nullptr;
    std::size_t resolved_size = 0;
    if (!cumetal::native_registration::lookup_symbol(
            global_shadow, &resolved_symbol, &resolved_size) ||
        resolved_symbol == nullptr || resolved_size != sizeof(global_shadow)) {
        return fail("native writable symbol did not resolve to persistent storage");
    }
    if (cumetalRegisterModule(&descriptor) != nullptr) {
        return fail("duplicate live host stub registration was accepted");
    }
    CuMetalSymbolDescriptor invalid_symbols[] = {symbols[0], symbols[1]};
    invalid_symbols[0].kind = static_cast<CuMetalSymbolKind>(99);
    CuMetalModuleDescriptor invalid_descriptor = descriptor;
    invalid_descriptor.symbols = invalid_symbols;
    if (cumetalRegisterModule(&invalid_descriptor) != nullptr) {
        return fail("invalid native symbol kind was accepted");
    }
    CuMetalKernelDescriptor invalid_kernel = kernels[0];
    invalid_kernel.printf_formats = nullptr;
    invalid_descriptor = descriptor;
    invalid_descriptor.kernels = &invalid_kernel;
    if (cumetalRegisterModule(&invalid_descriptor) != nullptr) {
        return fail("missing native printf format table was accepted");
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
