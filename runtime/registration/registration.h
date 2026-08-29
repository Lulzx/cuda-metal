#pragma once

#include "cuda_runtime.h"
#include "metal_backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cumetal::registration {

struct RegisteredConstantSymbol {
    std::string name;
    const void* address = nullptr;
    std::size_t offset = 0;
    std::size_t size = 0;
};

struct RegisteredGlobalSymbol {
    std::string name;
    std::shared_ptr<cumetal::metal_backend::Buffer> buffer;
    std::size_t size = 0;
};

struct RegisteredKernel {
    void* module_handle = nullptr;
    std::string metallib_path;
    std::string kernel_name;
    std::vector<cumetalKernelArgInfo_t> arg_info;
    // Device printf format table (spec §5.3): non-empty iff kernel uses printf.
    // printf_formats[i] is the format string for format id i.
    std::vector<std::string> printf_formats;
    bool uses_device_heap = false;
    bool uses_device_launch_queue = false;
    // Total bytes of static __shared__ memory (non-extern .shared declarations).
    // Used to call setThreadgroupMemoryLength when no dynamic shared memory is specified.
    std::size_t static_shared_bytes = 0;
    // Module-scope external `.const` storage referenced by this entry. These
    // become hidden read-only Metal buffer arguments after the CUDA arguments.
    std::vector<RegisteredConstantSymbol> constant_symbols;
    std::size_t constant_buffer_size = 0;
    // Persistent writable module-scope `.global` storage. Each buffer is shared
    // by symbol APIs and every kernel launch in the registration module.
    std::vector<RegisteredGlobalSymbol> global_symbols;
    std::string provenance;
    std::string semantic_quality;
};

struct LaunchConfiguration {
    dim3 grid_dim{};
    dim3 block_dim{};
    std::size_t shared_mem = 0;
    cudaStream_t stream = nullptr;
};

// Build the CUDA launch-argument ABI index from PTX entry signatures. This is
// intentionally a lightweight registration-time scan; full PTX parsing remains
// part of kernel lowering.
std::unordered_map<std::string, std::vector<cumetalKernelArgInfo_t>>
build_arg_info_index_from_ptx(const std::string& ptx_source);

// Resolve one launch ABI without allocating metadata for every other entry in
// a large fatbinary module. Returns false when the entry is absent or malformed.
bool find_arg_info_for_ptx_entry(const std::string& ptx_source,
                                 std::string_view entry_name,
                                 std::vector<cumetalKernelArgInfo_t>* out);

bool lookup_registered_kernel(const void* host_function, RegisteredKernel* out);
// Resolve a device-side kernel token emitted by the PTX lowering into a normal
// host launch alias backed by the same registration module.
bool lookup_device_kernel_alias(void* module_handle,
                                std::uint64_t token,
                                const void** out_host_function,
                                RegisteredKernel* out_kernel = nullptr);
bool lookup_registered_symbol(const void* host_symbol,
                              const void** out_device_symbol,
                              std::size_t* out_size);
void clear();

}  // namespace cumetal::registration

extern "C" {

void** __cudaRegisterFatBinary(const void* fat_cubin);
void** __cudaRegisterFatBinary2(const void* fat_cubin, ...);
void** __cudaRegisterFatBinary3(const void* fat_cubin, ...);
void __cudaRegisterLinkedBinary(void (*register_globals)(void**),
                                void* fatbin_wrapper,
                                void* module_id,
                                void (*callback)(void));
void __cudaRegisterFatBinaryEnd(void** fat_cubin_handle);
void __cudaUnregisterFatBinary(void** fat_cubin_handle);
void __cudaRegisterFunction(void** fat_cubin_handle,
                            const void* host_function,
                            char* device_function,
                            const char* device_name,
                            int thread_limit,
                            void* thread_id,
                            void* block_id,
                            void* block_dim,
                            void* grid_dim,
                            int* warp_size);
void __cudaRegisterVar(void** fat_cubin_handle,
                       char* host_var,
                       char* device_address,
                       const char* device_name,
                       int ext,
                       std::size_t size,
                       int constant,
                       int global);
void __cudaRegisterManagedVar(void** fat_cubin_handle,
                              void** host_var_ptr_address,
                              char* device_address,
                              const char* device_name,
                              int ext,
                              std::size_t size,
                              int constant,
                              int global);
cudaError_t __cudaPushCallConfiguration(dim3 grid_dim,
                                        dim3 block_dim,
                                        std::size_t shared_mem,
                                        cudaStream_t stream);
cudaError_t __cudaPopCallConfiguration(dim3* grid_dim,
                                       dim3* block_dim,
                                       std::size_t* shared_mem,
                                       void** stream);

}  // extern "C"
