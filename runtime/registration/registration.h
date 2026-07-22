#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace cumetal::registration {

struct RegisteredKernel {
    std::string metallib_path;
    std::string kernel_name;
    std::vector<cumetalKernelArgInfo_t> arg_info;
    // Device printf format table (spec §5.3): non-empty iff kernel uses printf.
    // printf_formats[i] is the format string for format id i.
    std::vector<std::string> printf_formats;
    // Total bytes of static __shared__ memory (non-extern .shared declarations).
    // Used to call setThreadgroupMemoryLength when no dynamic shared memory is specified.
    std::size_t static_shared_bytes = 0;
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

bool lookup_registered_kernel(const void* host_function, RegisteredKernel* out);
bool lookup_registered_symbol(const void* host_symbol,
                              const void** out_device_symbol,
                              std::size_t* out_size);
void clear();

}  // namespace cumetal::registration

extern "C" {

void** __cudaRegisterFatBinary(const void* fat_cubin);
void** __cudaRegisterFatBinary2(const void* fat_cubin, ...);
void** __cudaRegisterFatBinary3(const void* fat_cubin, ...);
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
