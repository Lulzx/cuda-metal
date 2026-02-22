#include "metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cumetal::metal_backend {
namespace {

cudaError_t check_command_buffer_status(id<MTLCommandBuffer> command_buffer, std::string* error_message);
cudaError_t map_command_buffer_error(NSError* command_error);

constexpr std::size_t kDefaultHeapChunkBytes = 64ull * 1024ull * 1024ull;
// Default auto-enable threshold: 4 MiB.  Allocations at or above this size use
// MTLHeap sub-allocation automatically (unless CUMETAL_MTLHEAP_ALLOC=0 disables it).
constexpr std::size_t kDefaultHeapAutoThresholdBytes = 4ull * 1024ull * 1024ull;

// CUMETAL_MTLHEAP_ALLOC controls heap sub-allocation:
//   unset / "auto"  → heap used for allocations >= CUMETAL_MTLHEAP_THRESHOLD_BYTES (default 4 MiB)
//   "1" / "true"    → heap always used
//   "0" / "false"   → heap never used
enum class HeapMode { kAuto, kAlways, kDisabled };

bool env_truthy(const char* value) {
    if (value == nullptr) {
        return false;
    }
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

std::size_t parse_size_env(const char* value, std::size_t fallback) {
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0) {
        return fallback;
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const std::size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return value + (alignment - remainder);
}

class BufferImpl final : public Buffer {
public:
    explicit BufferImpl(id<MTLBuffer> buffer) : buffer_(buffer) {}

    void* contents() const override {
        return [buffer_ contents];
    }

    std::size_t length() const override {
        return static_cast<std::size_t>([buffer_ length]);
    }

    id<MTLBuffer> handle() const {
        return buffer_;
    }

private:
    id<MTLBuffer> buffer_;
};

class StreamImpl final : public Stream {
public:
    explicit StreamImpl(id<MTLCommandQueue> queue) : queue_(queue) {}

    id<MTLCommandQueue> queue() const {
        return queue_;
    }

    std::uint64_t add_pending(id<MTLCommandBuffer> command_buffer) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::uint64_t ticket = next_ticket_++;
        pending_buffers_.push_back(PendingBuffer{.ticket = ticket, .command_buffer = command_buffer});
        return ticket;
    }

    std::uint64_t tail_ticket() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return next_ticket_ - 1;
    }

    cudaError_t poll_completed(std::string* error_message) {
        for (;;) {
            id<MTLCommandBuffer> completed_buffer = nil;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (pending_buffers_.empty()) {
                    return cudaSuccess;
                }

                const PendingBuffer& front = pending_buffers_.front();
                const MTLCommandBufferStatus status = [front.command_buffer status];
                if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError) {
                    return cudaSuccess;
                }

                completed_ticket_ = front.ticket;
                completed_buffer = front.command_buffer;
                pending_buffers_.erase(pending_buffers_.begin());
            }

            const cudaError_t status = check_command_buffer_status(completed_buffer, error_message);
            if (status != cudaSuccess) {
                return status;
            }
        }
    }

    cudaError_t query_ticket(std::uint64_t ticket, bool* out_complete, std::string* error_message) {
        if (out_complete == nullptr) {
            if (error_message != nullptr) {
                *error_message = "query_ticket missing out_complete";
            }
            return cudaErrorInvalidValue;
        }

        if (ticket == 0) {
            *out_complete = true;
            return cudaSuccess;
        }

        const cudaError_t poll_status = poll_completed(error_message);
        if (poll_status != cudaSuccess) {
            return poll_status;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (ticket >= next_ticket_) {
            if (error_message != nullptr) {
                *error_message = "query_ticket received unknown ticket";
            }
            return cudaErrorInvalidValue;
        }

        *out_complete = (ticket <= completed_ticket_);
        return cudaSuccess;
    }

    cudaError_t wait_ticket(std::uint64_t ticket, std::string* error_message) {
        if (ticket == 0) {
            return cudaSuccess;
        }

        for (;;) {
            const cudaError_t poll_status = poll_completed(error_message);
            if (poll_status != cudaSuccess) {
                return poll_status;
            }

            id<MTLCommandBuffer> next_wait = nil;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (ticket >= next_ticket_) {
                    if (error_message != nullptr) {
                        *error_message = "wait_ticket received unknown ticket";
                    }
                    return cudaErrorInvalidValue;
                }
                if (ticket <= completed_ticket_) {
                    return cudaSuccess;
                }
                if (pending_buffers_.empty()) {
                    if (error_message != nullptr) {
                        *error_message = "wait_ticket has no pending command buffers";
                    }
                    return cudaErrorUnknown;
                }
                next_wait = pending_buffers_.front().command_buffer;
            }

            [next_wait waitUntilCompleted];
            const cudaError_t status = check_command_buffer_status(next_wait, error_message);
            if (status != cudaSuccess) {
                return status;
            }
        }
    }

private:
    struct PendingBuffer {
        std::uint64_t ticket = 0;
        id<MTLCommandBuffer> command_buffer = nil;
    };

    id<MTLCommandQueue> queue_;
    mutable std::mutex mutex_;
    std::uint64_t next_ticket_ = 1;
    std::uint64_t completed_ticket_ = 0;
    std::vector<PendingBuffer> pending_buffers_;
};

struct BackendState {
    struct HeapArena {
        id<MTLHeap> heap = nil;
        std::size_t size = 0;
    };

    std::mutex mutex;
    bool initialized = false;
    HeapMode heap_mode = HeapMode::kAuto;
    std::size_t heap_auto_threshold = kDefaultHeapAutoThresholdBytes;
    std::size_t heap_chunk_bytes = kDefaultHeapChunkBytes;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::unordered_map<std::string, id<MTLLibrary>> library_cache;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;
    std::vector<HeapArena> buffer_heaps;
    std::vector<std::weak_ptr<StreamImpl>> streams;
};

BackendState& state() {
    static BackendState kState;
    return kState;
}

bool ensure_initialized(std::string* error_message) {
    BackendState& backend = state();
    std::lock_guard<std::mutex> lock(backend.mutex);
    if (backend.initialized) {
        return true;
    }

    @autoreleasepool {
        backend.device = MTLCreateSystemDefaultDevice();
        if (backend.device == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create default Metal device";
            }
            return false;
        }

        backend.queue = [backend.device newCommandQueue];
        if (backend.queue == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create Metal command queue";
            }
            return false;
        }

        {
            const char* heap_env = std::getenv("CUMETAL_MTLHEAP_ALLOC");
            if (heap_env == nullptr || heap_env[0] == '\0') {
                backend.heap_mode = HeapMode::kAuto;
            } else if (env_truthy(heap_env)) {
                backend.heap_mode = HeapMode::kAlways;
            } else {
                backend.heap_mode = HeapMode::kDisabled;
            }
        }
        backend.heap_auto_threshold =
            parse_size_env(std::getenv("CUMETAL_MTLHEAP_THRESHOLD_BYTES"),
                           kDefaultHeapAutoThresholdBytes);
        backend.heap_chunk_bytes =
            parse_size_env(std::getenv("CUMETAL_MTLHEAP_CHUNK_BYTES"), kDefaultHeapChunkBytes);

        backend.initialized = true;
        return true;
    }
}

std::string to_lower_copy(const std::string& input) {
    std::string lowered = input;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lowered;
}

int infer_multi_processor_count(const std::string& device_name) {
    const std::string lowered = to_lower_copy(device_name);

    if (lowered.find("m1 ultra") != std::string::npos) {
        return 48;
    }
    if (lowered.find("m1 max") != std::string::npos) {
        return 24;
    }
    if (lowered.find("m1 pro") != std::string::npos) {
        return 14;
    }
    if (lowered.find("m1") != std::string::npos) {
        return 8;
    }

    if (lowered.find("m2 ultra") != std::string::npos) {
        return 60;
    }
    if (lowered.find("m2 max") != std::string::npos) {
        return 30;
    }
    if (lowered.find("m2 pro") != std::string::npos) {
        return 16;
    }
    if (lowered.find("m2") != std::string::npos) {
        return 8;
    }

    if (lowered.find("m3 ultra") != std::string::npos) {
        return 60;
    }
    if (lowered.find("m3 max") != std::string::npos) {
        return 30;
    }
    if (lowered.find("m3 pro") != std::string::npos) {
        return 11;
    }
    if (lowered.find("m3") != std::string::npos) {
        return 8;
    }

    if (lowered.find("m4 ultra") != std::string::npos) {
        return 60;
    }
    if (lowered.find("m4 max") != std::string::npos) {
        return 32;
    }
    if (lowered.find("m4 pro") != std::string::npos) {
        return 16;
    }
    if (lowered.find("m4") != std::string::npos) {
        return 10;
    }

    return 8;
}

id<MTLLibrary> load_library_locked(BackendState& backend,
                                   const std::string& metallib_path,
                                   std::string* error_message) {
    const auto found = backend.library_cache.find(metallib_path);
    if (found != backend.library_cache.end()) {
        return found->second;
    }

    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
        NSError* read_error = nil;
        NSData* data = [NSData dataWithContentsOfFile:path options:0 error:&read_error];
        if (data == nil || [data length] == 0) {
            if (error_message != nullptr) {
                *error_message = "failed to read metallib file: " + metallib_path;
                if (read_error != nil) {
                    *error_message += " (" + std::string([[read_error localizedDescription] UTF8String]) + ")";
                }
            }
            return nil;
        }

        dispatch_data_t dispatch_data = dispatch_data_create(
            [data bytes], [data length], dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);

        NSError* library_error = nil;
        id<MTLLibrary> library = [backend.device newLibraryWithData:dispatch_data error:&library_error];
        if (library == nil) {
            if (error_message != nullptr) {
                *error_message = "newLibraryWithData failed for metallib: " + metallib_path;
                if (library_error != nil) {
                    *error_message +=
                        " (" + std::string([[library_error localizedDescription] UTF8String]) + ")";
                }
            }
            return nil;
        }

        backend.library_cache.emplace(metallib_path, library);
        return library;
    }
}

id<MTLComputePipelineState> load_pipeline_locked(BackendState& backend,
                                                 const std::string& metallib_path,
                                                 const std::string& kernel_name,
                                                 std::string* error_message) {
    const std::string cache_key = metallib_path + "::" + kernel_name;
    const auto found = backend.pipeline_cache.find(cache_key);
    if (found != backend.pipeline_cache.end()) {
        return found->second;
    }

    id<MTLLibrary> library = load_library_locked(backend, metallib_path, error_message);
    if (library == nil) {
        return nil;
    }

    @autoreleasepool {
        NSString* function_name = [NSString stringWithUTF8String:kernel_name.c_str()];
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to find kernel function: " + kernel_name;
            }
            return nil;
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            [backend.device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (pipeline == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create compute pipeline for function: " + kernel_name;
                if (pipeline_error != nil) {
                    *error_message +=
                        " (" + std::string([[pipeline_error localizedDescription] UTF8String]) + ")";
                }
            }
            return nil;
        }

        if ([pipeline threadExecutionWidth] != 32) {
            if (error_message != nullptr) {
                *error_message = "unsupported Metal threadExecutionWidth (expected 32)";
            }
            return nil;
        }

        backend.pipeline_cache.emplace(cache_key, pipeline);
        return pipeline;
    }
}

cudaError_t check_command_buffer_status(id<MTLCommandBuffer> command_buffer, std::string* error_message) {
    if ([command_buffer status] != MTLCommandBufferStatusError) {
        return cudaSuccess;
    }

    NSError* command_error = [command_buffer error];
    const cudaError_t mapped_error = map_command_buffer_error(command_error);
    if (error_message != nullptr) {
        *error_message = "command buffer failed";
        if (command_error != nil) {
            *error_message += " (" + std::string([[command_error localizedDescription] UTF8String]) + ")";
        }
    }
    return mapped_error;
}

cudaError_t map_command_buffer_error(NSError* command_error) {
    if (command_error == nil) {
        return cudaErrorUnknown;
    }

    if (![[command_error domain] isEqualToString:MTLCommandBufferErrorDomain]) {
        return cudaErrorUnknown;
    }

    const MTLCommandBufferError code = static_cast<MTLCommandBufferError>([command_error code]);
    switch (code) {
        case MTLCommandBufferErrorTimeout:
            return cudaErrorLaunchTimeout;
        case MTLCommandBufferErrorPageFault:
            return cudaErrorIllegalAddress;
        case MTLCommandBufferErrorAccessRevoked:
            return cudaErrorDevicesUnavailable;
        case MTLCommandBufferErrorInternal:
            return cudaErrorUnknown;
        default:
            return cudaErrorUnknown;
    }
}

std::vector<std::shared_ptr<StreamImpl>> collect_live_streams_locked(BackendState& backend) {
    std::vector<std::shared_ptr<StreamImpl>> live;
    std::vector<std::weak_ptr<StreamImpl>> retained;
    retained.reserve(backend.streams.size());
    live.reserve(backend.streams.size());

    for (const std::weak_ptr<StreamImpl>& weak_stream : backend.streams) {
        if (auto stream = weak_stream.lock()) {
            live.push_back(stream);
            retained.push_back(stream);
        }
    }

    backend.streams.swap(retained);
    return live;
}

id<MTLBuffer> allocate_buffer_from_heap_locked(BackendState& backend,
                                               std::size_t size,
                                               std::string* error_message) {
    constexpr MTLResourceOptions kBufferOptions = MTLResourceStorageModeShared;
    for (BackendState::HeapArena& arena : backend.buffer_heaps) {
        id<MTLBuffer> buffer = [arena.heap newBufferWithLength:size options:kBufferOptions];
        if (buffer != nil) {
            return buffer;
        }
    }

    const MTLSizeAndAlign size_and_align =
        [backend.device heapBufferSizeAndAlignWithLength:size options:kBufferOptions];
    const std::size_t aligned_size = align_up(
        static_cast<std::size_t>(size_and_align.size), static_cast<std::size_t>(size_and_align.align));
    const std::size_t arena_size = std::max(backend.heap_chunk_bytes, aligned_size);
    if (arena_size == 0) {
        if (error_message != nullptr) {
            *error_message = "heapBufferSizeAndAlignWithLength returned invalid size";
        }
        return nil;
    }

    MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
    [descriptor setStorageMode:MTLStorageModeShared];
    [descriptor setSize:arena_size];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 110000
    [descriptor setType:MTLHeapTypeAutomatic];
#endif

    id<MTLHeap> heap = [backend.device newHeapWithDescriptor:descriptor];
    if (heap == nil) {
        if (error_message != nullptr) {
            *error_message = "newHeapWithDescriptor failed";
        }
        return nil;
    }

    backend.buffer_heaps.push_back(BackendState::HeapArena{.heap = heap, .size = arena_size});
    id<MTLBuffer> buffer = [heap newBufferWithLength:size options:kBufferOptions];
    if (buffer == nil && error_message != nullptr) {
        *error_message = "newBufferWithLength from heap failed";
    }
    return buffer;
}

bool checked_matrix_span_bytes(std::size_t offset_bytes,
                               std::size_t span_bytes,
                               std::size_t buffer_length,
                               const char* label,
                               std::string* error_message) {
    if (offset_bytes > buffer_length || span_bytes > (buffer_length - offset_bytes)) {
        if (error_message != nullptr) {
            *error_message = std::string("matrix span exceeds buffer bounds for ") + label;
        }
        return false;
    }
    return true;
}

cudaError_t resolve_queue_for_stream(const std::shared_ptr<Stream>& stream,
                                     std::shared_ptr<StreamImpl>* out_stream_impl,
                                     id<MTLCommandQueue>* out_queue,
                                     std::string* error_message) {
    if (out_queue == nullptr) {
        if (error_message != nullptr) {
            *error_message = "missing output queue";
        }
        return cudaErrorInvalidValue;
    }

    std::shared_ptr<StreamImpl> stream_impl;
    if (stream != nullptr) {
        stream_impl = std::dynamic_pointer_cast<StreamImpl>(stream);
        if (stream_impl == nullptr) {
            if (error_message != nullptr) {
                *error_message = "received unknown stream type";
            }
            return cudaErrorInvalidValue;
        }
    }

    BackendState& backend = state();
    {
        std::lock_guard<std::mutex> lock(backend.mutex);
        *out_queue = (stream_impl != nullptr) ? stream_impl->queue() : backend.queue;
    }

    if (out_stream_impl != nullptr) {
        *out_stream_impl = std::move(stream_impl);
    }
    return cudaSuccess;
}

}  // namespace

cudaError_t initialize(std::string* error_message) {
    return ensure_initialized(error_message) ? cudaSuccess : cudaErrorInitializationError;
}

cudaError_t query_device_properties(DeviceProperties* out_properties, std::string* error_message) {
    if (out_properties == nullptr) {
        if (error_message != nullptr) {
            *error_message = "query_device_properties missing output";
        }
        return cudaErrorInvalidValue;
    }

    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    DeviceProperties props;
    BackendState& backend = state();
    {
        std::lock_guard<std::mutex> lock(backend.mutex);

        NSString* ns_name = [backend.device name];
        if (ns_name != nil) {
            props.name = [ns_name UTF8String];
        }
        if (props.name.empty()) {
            props.name = "Apple GPU";
        }

        props.total_global_mem = static_cast<std::size_t>([backend.device recommendedMaxWorkingSetSize]);
        if (props.total_global_mem == 0) {
            props.total_global_mem =
                static_cast<std::size_t>([[NSProcessInfo processInfo] physicalMemory]);
        }

        props.shared_mem_per_block =
            static_cast<int>([backend.device maxThreadgroupMemoryLength]);

        const MTLSize max_threads_per_group = [backend.device maxThreadsPerThreadgroup];
        const std::uint64_t max_threads_product =
            static_cast<std::uint64_t>(max_threads_per_group.width) *
            static_cast<std::uint64_t>(max_threads_per_group.height) *
            static_cast<std::uint64_t>(max_threads_per_group.depth);
        const std::uint64_t max_int = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
        props.max_threads_per_block = static_cast<int>(
            std::min(max_threads_product, max_int));

        props.multi_processor_count = infer_multi_processor_count(props.name);
    }

    *out_properties = std::move(props);
    return cudaSuccess;
}

cudaError_t allocate_buffer(std::size_t size,
                            std::shared_ptr<Buffer>* out_buffer,
                            std::string* error_message) {
    if (out_buffer == nullptr || size == 0) {
        if (error_message != nullptr) {
            *error_message = "allocate_buffer invalid argument";
        }
        return cudaErrorInvalidValue;
    }

    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    BackendState& backend = state();
    std::lock_guard<std::mutex> lock(backend.mutex);

    id<MTLBuffer> buffer = nil;
    const bool use_heap = (backend.heap_mode == HeapMode::kAlways) ||
                          (backend.heap_mode == HeapMode::kAuto &&
                           size >= backend.heap_auto_threshold);
    if (use_heap) {
        buffer = allocate_buffer_from_heap_locked(backend, size, error_message);
    }
    if (buffer == nil) {
        buffer = [backend.device newBufferWithLength:size options:MTLResourceStorageModeShared];
    }
    if (buffer == nil) {
        if (error_message != nullptr) {
            *error_message = "newBufferWithLength failed";
        }
        return cudaErrorMemoryAllocation;
    }
    if (error_message != nullptr) {
        error_message->clear();
    }

    *out_buffer = std::make_shared<BufferImpl>(buffer);
    return cudaSuccess;
}

cudaError_t create_stream(std::shared_ptr<Stream>* out_stream, std::string* error_message) {
    if (out_stream == nullptr) {
        if (error_message != nullptr) {
            *error_message = "create_stream invalid argument";
        }
        return cudaErrorInvalidValue;
    }

    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    BackendState& backend = state();
    std::lock_guard<std::mutex> lock(backend.mutex);

    id<MTLCommandQueue> queue = [backend.device newCommandQueue];
    if (queue == nil) {
        if (error_message != nullptr) {
            *error_message = "failed to create stream command queue";
        }
        return cudaErrorUnknown;
    }

    std::shared_ptr<StreamImpl> stream = std::make_shared<StreamImpl>(queue);
    backend.streams.push_back(stream);
    *out_stream = stream;
    return cudaSuccess;
}

cudaError_t destroy_stream(const std::shared_ptr<Stream>& stream, std::string* error_message) {
    if (stream == nullptr) {
        if (error_message != nullptr) {
            *error_message = "destroy_stream invalid argument";
        }
        return cudaErrorInvalidValue;
    }

    return stream_synchronize(stream, error_message);
}

cudaError_t stream_synchronize(const std::shared_ptr<Stream>& stream, std::string* error_message) {
    auto stream_impl = std::dynamic_pointer_cast<StreamImpl>(stream);
    if (stream_impl == nullptr) {
        if (error_message != nullptr) {
            *error_message = "stream_synchronize received unknown stream type";
        }
        return cudaErrorInvalidValue;
    }

    return stream_impl->wait_ticket(stream_impl->tail_ticket(), error_message);
}

cudaError_t stream_tail_ticket(const std::shared_ptr<Stream>& stream,
                               std::uint64_t* out_ticket,
                               std::string* error_message) {
    if (out_ticket == nullptr) {
        if (error_message != nullptr) {
            *error_message = "stream_tail_ticket missing output";
        }
        return cudaErrorInvalidValue;
    }

    auto stream_impl = std::dynamic_pointer_cast<StreamImpl>(stream);
    if (stream_impl == nullptr) {
        if (error_message != nullptr) {
            *error_message = "stream_tail_ticket received unknown stream type";
        }
        return cudaErrorInvalidValue;
    }

    *out_ticket = stream_impl->tail_ticket();
    return cudaSuccess;
}

cudaError_t stream_query_ticket(const std::shared_ptr<Stream>& stream,
                                std::uint64_t ticket,
                                bool* out_complete,
                                std::string* error_message) {
    auto stream_impl = std::dynamic_pointer_cast<StreamImpl>(stream);
    if (stream_impl == nullptr) {
        if (error_message != nullptr) {
            *error_message = "stream_query_ticket received unknown stream type";
        }
        return cudaErrorInvalidValue;
    }

    return stream_impl->query_ticket(ticket, out_complete, error_message);
}

cudaError_t stream_wait_ticket(const std::shared_ptr<Stream>& stream,
                               std::uint64_t ticket,
                               std::string* error_message) {
    auto stream_impl = std::dynamic_pointer_cast<StreamImpl>(stream);
    if (stream_impl == nullptr) {
        if (error_message != nullptr) {
            *error_message = "stream_wait_ticket received unknown stream type";
        }
        return cudaErrorInvalidValue;
    }

    return stream_impl->wait_ticket(ticket, error_message);
}

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
                     std::string* error_message) {
    if (m < 0 || n < 0 || k < 0 || lda <= 0 || ldb <= 0 || ldc <= 0 || a_buffer == nullptr ||
        b_buffer == nullptr || c_buffer == nullptr) {
        if (error_message != nullptr) {
            *error_message = "gemm_f32 invalid argument";
        }
        return cudaErrorInvalidValue;
    }
    if (m == 0 || n == 0 || k == 0) {
        return cudaSuccess;
    }

    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    auto* a_impl = dynamic_cast<BufferImpl*>(a_buffer.get());
    auto* b_impl = dynamic_cast<BufferImpl*>(b_buffer.get());
    auto* c_impl = dynamic_cast<BufferImpl*>(c_buffer.get());
    if (a_impl == nullptr || b_impl == nullptr || c_impl == nullptr) {
        if (error_message != nullptr) {
            *error_message = "gemm_f32 unexpected buffer type";
        }
        return cudaErrorInvalidValue;
    }

    const std::size_t a_cols = static_cast<std::size_t>(transa ? m : k);
    const std::size_t b_cols = static_cast<std::size_t>(transb ? k : n);
    const std::size_t c_cols = static_cast<std::size_t>(n);
    const std::size_t a_span = static_cast<std::size_t>(lda) * a_cols * sizeof(float);
    const std::size_t b_span = static_cast<std::size_t>(ldb) * b_cols * sizeof(float);
    const std::size_t c_span = static_cast<std::size_t>(ldc) * c_cols * sizeof(float);

    if (!checked_matrix_span_bytes(a_offset_bytes, a_span, a_impl->length(), "A", error_message) ||
        !checked_matrix_span_bytes(b_offset_bytes, b_span, b_impl->length(), "B", error_message) ||
        !checked_matrix_span_bytes(c_offset_bytes, c_span, c_impl->length(), "C", error_message)) {
        return cudaErrorInvalidValue;
    }

    std::shared_ptr<StreamImpl> stream_impl;
    id<MTLCommandQueue> queue = nil;
    const cudaError_t queue_status =
        resolve_queue_for_stream(stream, &stream_impl, &queue, error_message);
    if (queue_status != cudaSuccess) {
        return queue_status;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = "gemm_f32 failed to create command buffer";
            }
            return cudaErrorUnknown;
        }

        const NSUInteger left_rows = static_cast<NSUInteger>(b_cols);
        const NSUInteger left_cols = static_cast<NSUInteger>(transb ? n : k);
        const NSUInteger right_rows = static_cast<NSUInteger>(a_cols);
        const NSUInteger right_cols = static_cast<NSUInteger>(transa ? k : m);
        const NSUInteger result_rows = static_cast<NSUInteger>(n);
        const NSUInteger result_cols = static_cast<NSUInteger>(m);
        const NSUInteger interior_cols = static_cast<NSUInteger>(k);

        MPSMatrixDescriptor* left_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:left_rows
                                                  columns:left_cols
                                                 rowBytes:static_cast<NSUInteger>(ldb) * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* right_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:right_rows
                                                  columns:right_cols
                                                 rowBytes:static_cast<NSUInteger>(lda) * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* result_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:result_rows
                                                  columns:result_cols
                                                 rowBytes:static_cast<NSUInteger>(ldc) * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrix* left =
            [[MPSMatrix alloc] initWithBuffer:b_impl->handle()
                                       offset:static_cast<NSUInteger>(b_offset_bytes)
                                   descriptor:left_desc];
        MPSMatrix* right =
            [[MPSMatrix alloc] initWithBuffer:a_impl->handle()
                                       offset:static_cast<NSUInteger>(a_offset_bytes)
                                   descriptor:right_desc];
        MPSMatrix* result =
            [[MPSMatrix alloc] initWithBuffer:c_impl->handle()
                                       offset:static_cast<NSUInteger>(c_offset_bytes)
                                   descriptor:result_desc];

        MPSMatrixMultiplication* op =
            [[MPSMatrixMultiplication alloc] initWithDevice:state().device
                                              transposeLeft:(transb ? YES : NO)
                                             transposeRight:(transa ? YES : NO)
                                                resultRows:result_rows
                                             resultColumns:result_cols
                                           interiorColumns:interior_cols
                                                      alpha:static_cast<double>(alpha)
                                                       beta:static_cast<double>(beta)];
        if (op == nil) {
            if (error_message != nullptr) {
                *error_message = "gemm_f32 failed to create MPSMatrixMultiplication";
            }
            return cudaErrorUnknown;
        }

        [op encodeToCommandBuffer:command_buffer leftMatrix:left rightMatrix:right resultMatrix:result];
        [command_buffer commit];

        if (stream_impl != nullptr) {
            stream_impl->add_pending(command_buffer);
            return cudaSuccess;
        }

        [command_buffer waitUntilCompleted];
        return check_command_buffer_status(command_buffer, error_message);
    }
}

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
                                     std::string* error_message) {
    if (batch_count < 0) {
        if (error_message != nullptr) {
            *error_message = "gemm_strided_batched_f32 invalid batch_count";
        }
        return cudaErrorInvalidValue;
    }
    if (batch_count == 0 || m == 0 || n == 0 || k == 0) {
        return cudaSuccess;
    }

    if (m < 0 || n < 0 || k < 0 || lda <= 0 || ldb <= 0 || ldc <= 0 || a_buffer == nullptr ||
        b_buffer == nullptr || c_buffer == nullptr) {
        if (error_message != nullptr) {
            *error_message = "gemm_strided_batched_f32 invalid argument";
        }
        return cudaErrorInvalidValue;
    }

    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    auto* a_impl = dynamic_cast<BufferImpl*>(a_buffer.get());
    auto* b_impl = dynamic_cast<BufferImpl*>(b_buffer.get());
    auto* c_impl = dynamic_cast<BufferImpl*>(c_buffer.get());
    if (a_impl == nullptr || b_impl == nullptr || c_impl == nullptr) {
        if (error_message != nullptr) {
            *error_message = "gemm_strided_batched_f32 unexpected buffer type";
        }
        return cudaErrorInvalidValue;
    }

    const std::size_t a_cols = static_cast<std::size_t>(transa ? m : k);
    const std::size_t b_cols = static_cast<std::size_t>(transb ? k : n);
    const std::size_t c_cols = static_cast<std::size_t>(n);
    const std::size_t a_span = static_cast<std::size_t>(lda) * a_cols * sizeof(float);
    const std::size_t b_span = static_cast<std::size_t>(ldb) * b_cols * sizeof(float);
    const std::size_t c_span = static_cast<std::size_t>(ldc) * c_cols * sizeof(float);
    const std::size_t batch_index = static_cast<std::size_t>(batch_count - 1);
    const std::size_t a_total_span = batch_index * stridea_bytes + a_span;
    const std::size_t b_total_span = batch_index * strideb_bytes + b_span;
    const std::size_t c_total_span = batch_index * stridec_bytes + c_span;

    if (!checked_matrix_span_bytes(a_offset_bytes, a_total_span, a_impl->length(), "A", error_message) ||
        !checked_matrix_span_bytes(b_offset_bytes, b_total_span, b_impl->length(), "B", error_message) ||
        !checked_matrix_span_bytes(c_offset_bytes, c_total_span, c_impl->length(), "C", error_message)) {
        return cudaErrorInvalidValue;
    }

    std::shared_ptr<StreamImpl> stream_impl;
    id<MTLCommandQueue> queue = nil;
    const cudaError_t queue_status =
        resolve_queue_for_stream(stream, &stream_impl, &queue, error_message);
    if (queue_status != cudaSuccess) {
        return queue_status;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = "gemm_strided_batched_f32 failed to create command buffer";
            }
            return cudaErrorUnknown;
        }

        const NSUInteger left_rows = static_cast<NSUInteger>(b_cols);
        const NSUInteger left_cols = static_cast<NSUInteger>(transb ? n : k);
        const NSUInteger right_rows = static_cast<NSUInteger>(a_cols);
        const NSUInteger right_cols = static_cast<NSUInteger>(transa ? k : m);
        const NSUInteger result_rows = static_cast<NSUInteger>(n);
        const NSUInteger result_cols = static_cast<NSUInteger>(m);
        const NSUInteger interior_cols = static_cast<NSUInteger>(k);

        MPSMatrixDescriptor* left_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:left_rows
                                                  columns:left_cols
                                                 rowBytes:static_cast<NSUInteger>(ldb) * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* right_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:right_rows
                                                  columns:right_cols
                                                 rowBytes:static_cast<NSUInteger>(lda) * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* result_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:result_rows
                                                  columns:result_cols
                                                 rowBytes:static_cast<NSUInteger>(ldc) * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrixMultiplication* op =
            [[MPSMatrixMultiplication alloc] initWithDevice:state().device
                                              transposeLeft:(transb ? YES : NO)
                                             transposeRight:(transa ? YES : NO)
                                                resultRows:result_rows
                                             resultColumns:result_cols
                                           interiorColumns:interior_cols
                                                      alpha:static_cast<double>(alpha)
                                                       beta:static_cast<double>(beta)];
        if (op == nil) {
            if (error_message != nullptr) {
                *error_message = "gemm_strided_batched_f32 failed to create MPSMatrixMultiplication";
            }
            return cudaErrorUnknown;
        }

        for (int batch = 0; batch < batch_count; ++batch) {
            const std::size_t bindex = static_cast<std::size_t>(batch);
            MPSMatrix* left =
                [[MPSMatrix alloc] initWithBuffer:b_impl->handle()
                                           offset:static_cast<NSUInteger>(b_offset_bytes + bindex * strideb_bytes)
                                       descriptor:left_desc];
            MPSMatrix* right =
                [[MPSMatrix alloc] initWithBuffer:a_impl->handle()
                                           offset:static_cast<NSUInteger>(a_offset_bytes + bindex * stridea_bytes)
                                       descriptor:right_desc];
            MPSMatrix* result =
                [[MPSMatrix alloc] initWithBuffer:c_impl->handle()
                                           offset:static_cast<NSUInteger>(c_offset_bytes + bindex * stridec_bytes)
                                       descriptor:result_desc];
            [op encodeToCommandBuffer:command_buffer leftMatrix:left rightMatrix:right resultMatrix:result];
        }

        [command_buffer commit];
        if (stream_impl != nullptr) {
            stream_impl->add_pending(command_buffer);
            return cudaSuccess;
        }

        [command_buffer waitUntilCompleted];
        return check_command_buffer_status(command_buffer, error_message);
    }
}

cudaError_t launch_kernel(const std::string& metallib_path,
                          const std::string& kernel_name,
                          const LaunchConfig& config,
                          const std::vector<KernelArg>& args,
                          const std::shared_ptr<Stream>& stream,
                          std::string* error_message) {
    if (metallib_path.empty() || kernel_name.empty()) {
        if (error_message != nullptr) {
            *error_message = "launch_kernel requires metallib path and kernel name";
        }
        return cudaErrorInvalidValue;
    }

    if (args.size() > 31) {
        if (error_message != nullptr) {
            *error_message = "kernel argument count exceeds Metal argument index limit (31)";
        }
        return cudaErrorInvalidValue;
    }

    if (config.grid.x == 0 || config.grid.y == 0 || config.grid.z == 0 || config.block.x == 0 ||
        config.block.y == 0 || config.block.z == 0) {
        if (error_message != nullptr) {
            *error_message = "launch dimensions must be non-zero";
        }
        return cudaErrorInvalidValue;
    }

    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    std::shared_ptr<StreamImpl> stream_impl;
    if (stream != nullptr) {
        stream_impl = std::dynamic_pointer_cast<StreamImpl>(stream);
        if (stream_impl == nullptr) {
            if (error_message != nullptr) {
                *error_message = "launch_kernel received unknown stream type";
            }
            return cudaErrorInvalidValue;
        }
    }

    BackendState& backend = state();
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandQueue> queue = nil;
    {
        std::lock_guard<std::mutex> lock(backend.mutex);
        pipeline = load_pipeline_locked(backend, metallib_path, kernel_name, error_message);
        if (pipeline == nil) {
            return cudaErrorInvalidValue;
        }

        if (stream_impl != nullptr) {
            queue = stream_impl->queue();
        } else {
            queue = backend.queue;
        }
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create command buffer";
            }
            return cudaErrorUnknown;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "failed to create compute encoder";
            }
            return cudaErrorUnknown;
        }

        [encoder setComputePipelineState:pipeline];

        for (std::size_t i = 0; i < args.size(); ++i) {
            const KernelArg& arg = args[i];
            if (arg.kind == KernelArg::Kind::kBuffer) {
                if (arg.buffer == nullptr) {
                    [encoder setBuffer:nil offset:0 atIndex:i];
                    continue;
                }

                auto* buffer_impl = dynamic_cast<BufferImpl*>(arg.buffer.get());
                if (buffer_impl == nullptr) {
                    if (error_message != nullptr) {
                        *error_message = "kernel arg " + std::to_string(i) + " has unexpected buffer type";
                    }
                    return cudaErrorInvalidValue;
                }

                [encoder setBuffer:buffer_impl->handle() offset:arg.offset atIndex:i];
            } else {
                if (arg.bytes.empty()) {
                    if (error_message != nullptr) {
                        *error_message = "kernel arg " + std::to_string(i) + " has empty byte payload";
                    }
                    return cudaErrorInvalidValue;
                }
                if (arg.bytes.size() > 4096) {
                    if (error_message != nullptr) {
                        *error_message = "kernel arg " + std::to_string(i) +
                                         " byte payload exceeds 4KB setBytes limit";
                    }
                    return cudaErrorInvalidValue;
                }

                [encoder setBytes:arg.bytes.data() length:arg.bytes.size() atIndex:i];
            }
        }

        if (config.shared_memory_bytes > 0) {
            [encoder setThreadgroupMemoryLength:config.shared_memory_bytes atIndex:0];
        }

        const MTLSize threadgroups = MTLSizeMake(config.grid.x, config.grid.y, config.grid.z);
        const MTLSize threads_per_threadgroup =
            MTLSizeMake(config.block.x, config.block.y, config.block.z);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];

        [encoder endEncoding];

        [command_buffer commit];

        if (stream_impl != nullptr) {
            stream_impl->add_pending(command_buffer);
            return cudaSuccess;
        }

        [command_buffer waitUntilCompleted];
        return check_command_buffer_status(command_buffer, error_message);
    }

    return cudaSuccess;
}

cudaError_t launch_kernel_timed(const std::string& metallib_path,
                                const std::string& kernel_name,
                                const LaunchConfig& config,
                                const std::vector<KernelArg>& args,
                                GpuTimingResult* out_timing,
                                std::string* error_message) {
    if (metallib_path.empty() || kernel_name.empty()) {
        if (error_message != nullptr) {
            *error_message = "launch_kernel_timed requires metallib path and kernel name";
        }
        return cudaErrorInvalidValue;
    }

    if (args.size() > 31) {
        if (error_message != nullptr) {
            *error_message = "kernel argument count exceeds Metal argument index limit (31)";
        }
        return cudaErrorInvalidValue;
    }

    if (config.grid.x == 0 || config.grid.y == 0 || config.grid.z == 0 || config.block.x == 0 ||
        config.block.y == 0 || config.block.z == 0) {
        if (error_message != nullptr) {
            *error_message = "launch dimensions must be non-zero";
        }
        return cudaErrorInvalidValue;
    }

    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    BackendState& backend = state();
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLCommandQueue> queue = nil;
    {
        std::lock_guard<std::mutex> lock(backend.mutex);
        pipeline = load_pipeline_locked(backend, metallib_path, kernel_name, error_message);
        if (pipeline == nil) {
            return cudaErrorInvalidValue;
        }
        queue = backend.queue;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            if (error_message != nullptr) {
                *error_message = "launch_kernel_timed failed to create command buffer";
            }
            return cudaErrorUnknown;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            if (error_message != nullptr) {
                *error_message = "launch_kernel_timed failed to create compute encoder";
            }
            return cudaErrorUnknown;
        }

        [encoder setComputePipelineState:pipeline];

        for (std::size_t i = 0; i < args.size(); ++i) {
            const KernelArg& arg = args[i];
            if (arg.kind == KernelArg::Kind::kBuffer) {
                if (arg.buffer == nullptr) {
                    [encoder setBuffer:nil offset:0 atIndex:i];
                    continue;
                }
                auto* buffer_impl = dynamic_cast<BufferImpl*>(arg.buffer.get());
                if (buffer_impl == nullptr) {
                    if (error_message != nullptr) {
                        *error_message = "launch_kernel_timed arg " + std::to_string(i) +
                                         " has unexpected buffer type";
                    }
                    return cudaErrorInvalidValue;
                }
                [encoder setBuffer:buffer_impl->handle() offset:arg.offset atIndex:i];
            } else {
                if (arg.bytes.empty()) {
                    if (error_message != nullptr) {
                        *error_message = "launch_kernel_timed arg " + std::to_string(i) +
                                         " has empty byte payload";
                    }
                    return cudaErrorInvalidValue;
                }
                if (arg.bytes.size() > 4096) {
                    if (error_message != nullptr) {
                        *error_message = "launch_kernel_timed arg " + std::to_string(i) +
                                         " byte payload exceeds 4KB setBytes limit";
                    }
                    return cudaErrorInvalidValue;
                }
                [encoder setBytes:arg.bytes.data() length:arg.bytes.size() atIndex:i];
            }
        }

        if (config.shared_memory_bytes > 0) {
            [encoder setThreadgroupMemoryLength:config.shared_memory_bytes atIndex:0];
        }

        const MTLSize threadgroups = MTLSizeMake(config.grid.x, config.grid.y, config.grid.z);
        const MTLSize threads_per_threadgroup =
            MTLSizeMake(config.block.x, config.block.y, config.block.z);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        const cudaError_t status = check_command_buffer_status(command_buffer, error_message);
        if (status == cudaSuccess && out_timing != nullptr) {
            out_timing->gpu_start_s = [command_buffer GPUStartTime];
            out_timing->gpu_end_s   = [command_buffer GPUEndTime];
        }
        return status;
    }
}

cudaError_t synchronize(std::string* error_message) {
    if (!ensure_initialized(error_message)) {
        return cudaErrorInitializationError;
    }

    BackendState& backend = state();
    std::vector<std::shared_ptr<StreamImpl>> streams;
    {
        std::lock_guard<std::mutex> lock(backend.mutex);
        streams = collect_live_streams_locked(backend);
    }

    for (const std::shared_ptr<StreamImpl>& stream : streams) {
        const cudaError_t status = stream_synchronize(stream, error_message);
        if (status != cudaSuccess) {
            return status;
        }
    }

    return cudaSuccess;
}

}  // namespace cumetal::metal_backend
