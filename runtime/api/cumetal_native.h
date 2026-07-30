#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUMETAL_NATIVE_ABI_VERSION 1u

/* CuMetal release version. Distinct from CUMETAL_NATIVE_ABI_VERSION above, which versions the
 * native registration struct layout and moves only on an ABI break.
 *
 * These must stay in step with project(cumetal VERSION ...) in CMakeLists.txt. They are
 * duplicated rather than generated so this header stays usable standalone; unit_version_matches
 * fails the build's test suite if the two ever drift. */
#define CUMETAL_VERSION_MAJOR 0
#define CUMETAL_VERSION_MINOR 1
#define CUMETAL_VERSION_PATCH 0
#define CUMETAL_VERSION_STRING "0.1.0"

/* Encoded as major*10000 + minor*100 + patch, so versions compare with < and >. */
#define CUMETAL_VERSION \
    (CUMETAL_VERSION_MAJOR * 10000 + CUMETAL_VERSION_MINOR * 100 + CUMETAL_VERSION_PATCH)

/* Version of the loaded libcumetal, which may differ from the headers compiled against. */
int cumetalGetVersion(void);
const char* cumetalGetVersionString(void);

typedef enum CuMetalArgumentKind {
    CUMETAL_NATIVE_ARGUMENT_POINTER = 0,
    CUMETAL_NATIVE_ARGUMENT_SCALAR = 1,
    CUMETAL_NATIVE_ARGUMENT_AGGREGATE = 2,
    CUMETAL_NATIVE_ARGUMENT_DYNAMIC_THREADGROUP_MEMORY = 3,
} CuMetalArgumentKind;

typedef enum CuMetalBindingKind {
    CUMETAL_NATIVE_BINDING_BUFFER = 0,
    CUMETAL_NATIVE_BINDING_BYTES = 1,
    CUMETAL_NATIVE_BINDING_THREADGROUP_MEMORY = 2,
} CuMetalBindingKind;

typedef enum CuMetalAddressSpace {
    CUMETAL_NATIVE_ADDRESS_NONE = 0,
    CUMETAL_NATIVE_ADDRESS_DEVICE = 1,
    CUMETAL_NATIVE_ADDRESS_CONSTANT = 2,
    CUMETAL_NATIVE_ADDRESS_THREADGROUP = 3,
    CUMETAL_NATIVE_ADDRESS_PRIVATE = 4,
} CuMetalAddressSpace;

typedef struct CuMetalArgumentDescriptor {
    CuMetalArgumentKind kind;
    uint32_t size;
    uint32_t alignment;
    CuMetalAddressSpace address_space;
    uint32_t first_binding;
    uint32_t binding_count;
} CuMetalArgumentDescriptor;

typedef struct CuMetalBindingDescriptor {
    CuMetalBindingKind kind;
    uint32_t metal_index;
    uint32_t logical_argument_index;
    uint32_t size;
    uint32_t alignment;
} CuMetalBindingDescriptor;

typedef struct CuMetalKernelDescriptor {
    const char* cuda_name;
    const char* metal_name;
    const void* host_stub;
    uint32_t argument_count;
    const CuMetalArgumentDescriptor* arguments;
    uint32_t static_threadgroup_memory;
    uint32_t required_simd_width;
} CuMetalKernelDescriptor;

typedef struct CuMetalModuleDescriptor {
    uint32_t abi_version;
    const void* metallib_data;
    size_t metallib_size;
    uint32_t kernel_count;
    const CuMetalKernelDescriptor* kernels;
    uint32_t binding_count;
    const CuMetalBindingDescriptor* bindings;
    const char* provenance;
    const char* semantic_quality;
} CuMetalModuleDescriptor;

typedef void* CuMetalModuleHandle;

CuMetalModuleHandle cumetalRegisterModule(const CuMetalModuleDescriptor* descriptor);
void cumetalUnregisterModule(CuMetalModuleHandle module);

#ifdef __cplusplus
}  // extern "C"
#endif
