#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cumetal::metal_backend {

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual void* contents() const = 0;
    virtual std::uintptr_t device_address() const {
        return reinterpret_cast<std::uintptr_t>(contents());
    }
    virtual std::size_t length() const = 0;
};

class Stream {
public:
    virtual ~Stream() = default;
};

struct DeviceProperties {
    std::string name;
    std::size_t total_global_mem = 0;
    int shared_mem_per_block = 0;
    int max_threads_per_block = 0;
    int multi_processor_count = 8;
};

struct KernelArg {
    enum class Kind {
        kBuffer,
        kBytes,
    };

    Kind kind = Kind::kBytes;
    std::shared_ptr<Buffer> buffer;
    std::size_t offset = 0;
    std::vector<std::uint8_t> bytes;
};

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t shared_memory_bytes = 0;
};

cudaError_t initialize(std::string* error_message);
cudaError_t query_device_properties(DeviceProperties* out_properties, std::string* error_message);
cudaError_t allocate_buffer(std::size_t size,
                            std::shared_ptr<Buffer>* out_buffer,
                            std::string* error_message);
cudaError_t create_stream(std::shared_ptr<Stream>* out_stream, std::string* error_message);
cudaError_t destroy_stream(const std::shared_ptr<Stream>& stream, std::string* error_message);
cudaError_t stream_synchronize(const std::shared_ptr<Stream>& stream, std::string* error_message);
cudaError_t stream_tail_ticket(const std::shared_ptr<Stream>& stream,
                               std::uint64_t* out_ticket,
                               std::string* error_message);
cudaError_t stream_query_ticket(const std::shared_ptr<Stream>& stream,
                                std::uint64_t ticket,
                                bool* out_complete,
                                std::string* error_message);
cudaError_t stream_wait_ticket(const std::shared_ptr<Stream>& stream,
                               std::uint64_t ticket,
                               std::string* error_message);
cudaError_t launch_kernel(const std::string& metallib_path,
                          const std::string& kernel_name,
                          const LaunchConfig& config,
                          const std::vector<KernelArg>& args,
                          const std::shared_ptr<Stream>& stream,
                          std::string* error_message);

// GPU timing result from MTLCommandBuffer.GPUStartTime / GPUEndTime.
// Both fields are in seconds (CFTimeInterval). Duration = gpu_end_s - gpu_start_s.
struct GpuTimingResult {
    double gpu_start_s = 0.0;
    double gpu_end_s = 0.0;
    double duration_ms() const { return (gpu_end_s - gpu_start_s) * 1000.0; }
};

// Synchronous kernel launch that captures GPU execution time.
// Runs on the default queue; does not accept a stream.
cudaError_t launch_kernel_timed(const std::string& metallib_path,
                                const std::string& kernel_name,
                                const LaunchConfig& config,
                                const std::vector<KernelArg>& args,
                                GpuTimingResult* out_timing,
                                std::string* error_message);
cudaError_t gemm_f32(bool transa,
                     bool transb,
                     int m,
                     int n,
                     int k,
                     float alpha,
                     const std::shared_ptr<Buffer>& a_buffer,
                     std::size_t a_offset_bytes,
                     int lda,
                     const std::shared_ptr<Buffer>& b_buffer,
                     std::size_t b_offset_bytes,
                     int ldb,
                     float beta,
                     const std::shared_ptr<Buffer>& c_buffer,
                     std::size_t c_offset_bytes,
                     int ldc,
                     const std::shared_ptr<Stream>& stream,
                     std::string* error_message);
cudaError_t gemm_strided_batched_f32(bool transa,
                                     bool transb,
                                     int m,
                                     int n,
                                     int k,
                                     float alpha,
                                     const std::shared_ptr<Buffer>& a_buffer,
                                     std::size_t a_offset_bytes,
                                     int lda,
                                     std::size_t stridea_bytes,
                                     const std::shared_ptr<Buffer>& b_buffer,
                                     std::size_t b_offset_bytes,
                                     int ldb,
                                     std::size_t strideb_bytes,
                                     float beta,
                                     const std::shared_ptr<Buffer>& c_buffer,
                                     std::size_t c_offset_bytes,
                                     int ldc,
                                     std::size_t stridec_bytes,
                                     int batch_count,
                                     const std::shared_ptr<Stream>& stream,
                                     std::string* error_message);
cudaError_t synchronize(std::string* error_message);

}  // namespace cumetal::metal_backend
