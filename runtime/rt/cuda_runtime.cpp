#include "cuda_runtime.h"

#include "allocation_table.h"
#include "library_conflict.h"
#include "metal_backend.h"
#include "registration.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

struct cudaStream_st {};
struct cudaEvent_st {
    bool disable_timing = false;
    bool recorded_once = false;
    bool complete = true;
    bool timing_valid = false;
    std::shared_ptr<cumetal::metal_backend::Stream> stream;
    std::uint64_t ticket = 0;
    std::chrono::steady_clock::time_point timestamp{};
    std::mutex mutex;
};

namespace {

constexpr int kCudaCompatVersion = 12000;

struct RuntimeState {
    std::once_flag init_once;
    cudaError_t init_status = cudaSuccess;
    std::string init_error;
    int current_device = 0;
    unsigned int device_flags = cudaDeviceScheduleAuto;
    cumetal::rt::AllocationTable allocations;
    std::mutex stream_mutex;
    std::unordered_map<cudaStream_t, std::shared_ptr<cumetal::metal_backend::Stream>> streams;
};

RuntimeState& runtime_state() {
    static RuntimeState state;
    return state;
}

thread_local cudaError_t tls_last_error = cudaSuccess;
thread_local std::shared_ptr<cumetal::metal_backend::Stream> tls_per_thread_stream;

struct PendingLaunchArgument {
    size_t offset = 0;
    size_t size = 0;
};

struct PendingLaunchState {
    bool configured = false;
    dim3 grid_dim{};
    dim3 block_dim{};
    size_t shared_mem = 0;
    cudaStream_t stream = nullptr;
    std::vector<std::uint8_t> storage;
    std::vector<PendingLaunchArgument> arguments;
};

thread_local PendingLaunchState tls_pending_launch;

void clear_pending_launch_state() {
    tls_pending_launch.configured = false;
    tls_pending_launch.grid_dim = dim3{};
    tls_pending_launch.block_dim = dim3{};
    tls_pending_launch.shared_mem = 0;
    tls_pending_launch.stream = nullptr;
    tls_pending_launch.storage.clear();
    tls_pending_launch.arguments.clear();
}

void set_last_error(cudaError_t error) {
    tls_last_error = error;
}

// ── Device printf drain (spec §5.3) ─────────────────────────────────────────
// Ring-buffer layout (all words are uint32):
//   buf[0]          = atomic write-word-count (total words written after index 0)
//   buf[1..]        = packed records: [fmt_id, n_args, arg0, ..., argN-1]
// Float args are stored as as_type<uint>(f); integers as uint casts.

void drain_one_printf_record(const std::string& fmt,
                             const std::uint32_t* args,
                             std::uint32_t n_args) {
    std::uint32_t arg_idx = 0;
    for (std::size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] != '%') {
            std::fputc(fmt[i], stderr);
            continue;
        }
        ++i;
        if (i >= fmt.size()) { break; }
        if (fmt[i] == '%') {
            std::fputc('%', stderr);
            continue;
        }
        // Reconstruct specifier string
        std::string spec = "%";
        // Flags
        while (i < fmt.size() &&
               (fmt[i] == '-' || fmt[i] == '+' || fmt[i] == ' ' ||
                fmt[i] == '#' || fmt[i] == '0')) {
            spec += fmt[i++];
        }
        // Width
        while (i < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[i]))) {
            spec += fmt[i++];
        }
        // Precision
        if (i < fmt.size() && fmt[i] == '.') {
            spec += fmt[i++];
            while (i < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[i]))) {
                spec += fmt[i++];
            }
        }
        // Skip length modifiers (all device args are stored as 32-bit)
        while (i < fmt.size() &&
               (fmt[i] == 'l' || fmt[i] == 'h' || fmt[i] == 'z' ||
                fmt[i] == 'j' || fmt[i] == 't')) {
            ++i;
        }
        if (i >= fmt.size()) { break; }
        const char conv = fmt[i];
        spec += conv;

        if (arg_idx >= n_args) {
            std::fputs(spec.c_str(), stderr);
            continue;
        }
        const std::uint32_t raw = args[arg_idx++];
        if (conv == 'f' || conv == 'e' || conv == 'g' ||
            conv == 'F' || conv == 'E' || conv == 'G') {
            float fval;
            std::memcpy(&fval, &raw, sizeof(fval));
            std::fprintf(stderr, spec.c_str(), fval);
        } else if (conv == 'd' || conv == 'i') {
            std::fprintf(stderr, spec.c_str(), static_cast<int>(raw));
        } else if (conv == 'u' || conv == 'o' || conv == 'x' || conv == 'X') {
            std::fprintf(stderr, spec.c_str(), raw);
        } else if (conv == 'c') {
            std::fprintf(stderr, spec.c_str(), static_cast<int>(raw));
        } else if (conv == 's') {
            std::fputs("[string]", stderr);
        } else {
            std::fputs(spec.c_str(), stderr);
        }
    }
}

void drain_printf_buffer(const void* buf_bytes,
                         std::uint32_t cap_words,
                         const std::vector<std::string>& formats) {
    if (buf_bytes == nullptr || cap_words == 0 || formats.empty()) {
        return;
    }
    const std::uint32_t* buf = static_cast<const std::uint32_t*>(buf_bytes);
    const std::uint32_t total_words = buf[0];
    if (total_words == 0 || total_words > cap_words - 1u) {
        return;
    }
    // Walk records starting at index 1
    std::uint32_t i = 1u;
    while (i + 1u <= total_words) {
        const std::uint32_t fmt_id = buf[i];
        const std::uint32_t n_args = buf[i + 1u];
        if (n_args > total_words || i + 2u + n_args > total_words + 1u) {
            break;
        }
        if (fmt_id < static_cast<std::uint32_t>(formats.size())) {
            drain_one_printf_record(formats[fmt_id], buf + i + 2u, n_args);
        }
        i += 2u + n_args;
    }
}

cudaError_t ensure_initialized() {
    RuntimeState& state = runtime_state();
    std::call_once(state.init_once, [&state]() {
        const std::string conflict_warning =
            cumetal::error::detect_loaded_libcuda_conflict(reinterpret_cast<const void*>(&cudaInit));
        if (!conflict_warning.empty()) {
            std::fprintf(stderr, "%s\n", conflict_warning.c_str());
        }

        std::string error;
        state.init_status = cumetal::metal_backend::initialize(&error);
        state.init_error = error;
    });
    return state.init_status;
}

cudaError_t fail(cudaError_t error) {
    set_last_error(error);
    return error;
}

bool resolve_stream_handle(cudaStream_t stream,
                           std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (stream == nullptr || out_stream == nullptr) {
        return false;
    }

    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.stream_mutex);
    const auto found = state.streams.find(stream);
    if (found == state.streams.end()) {
        return false;
    }

    *out_stream = found->second;
    return true;
}

bool is_legacy_stream_handle(cudaStream_t stream) {
    return stream == nullptr || stream == cudaStreamLegacy;
}

bool is_per_thread_stream_handle(cudaStream_t stream) {
    return stream == cudaStreamPerThread;
}

cudaError_t ensure_per_thread_stream(std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (out_stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (tls_per_thread_stream != nullptr) {
        *out_stream = tls_per_thread_stream;
        return cudaSuccess;
    }

    std::string error;
    std::shared_ptr<cumetal::metal_backend::Stream> created;
    const cudaError_t status = cumetal::metal_backend::create_stream(&created, &error);
    if (status != cudaSuccess || created == nullptr) {
        return status == cudaSuccess ? cudaErrorUnknown : status;
    }
    tls_per_thread_stream = created;
    *out_stream = std::move(created);
    return cudaSuccess;
}

cudaError_t resolve_runtime_stream(cudaStream_t stream,
                                   std::shared_ptr<cumetal::metal_backend::Stream>* out_stream,
                                   bool* is_legacy_stream) {
    if (out_stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    out_stream->reset();
    if (is_legacy_stream != nullptr) {
        *is_legacy_stream = false;
    }

    if (is_legacy_stream_handle(stream)) {
        if (is_legacy_stream != nullptr) {
            *is_legacy_stream = true;
        }
        return cudaSuccess;
    }

    if (is_per_thread_stream_handle(stream)) {
        return ensure_per_thread_stream(out_stream);
    }

    if (!resolve_stream_handle(stream, out_stream)) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

bool erase_stream_handle(cudaStream_t stream,
                         std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (stream == nullptr || out_stream == nullptr) {
        return false;
    }

    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.stream_mutex);
    const auto found = state.streams.find(stream);
    if (found == state.streams.end()) {
        return false;
    }

    *out_stream = std::move(found->second);
    state.streams.erase(found);
    return true;
}

cudaError_t validate_memcpy_kind(cudaMemcpyKind kind) {
    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyDeviceToDevice:
        case cudaMemcpyDefault:
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t validate_host_alloc_flags(unsigned int flags) {
    constexpr unsigned int kSupportedHostAllocFlags =
        cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined;
    if ((flags & ~kSupportedHostAllocFlags) != 0) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t validate_device_flags(unsigned int flags) {
    constexpr unsigned int kSupportedDeviceFlags =
        cudaDeviceScheduleSpin | cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync |
        cudaDeviceMapHost | cudaDeviceLmemResizeToMax;

    if ((flags & ~kSupportedDeviceFlags) != 0) {
        return cudaErrorInvalidValue;
    }

    const unsigned int schedule_bits =
        flags & (cudaDeviceScheduleSpin | cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync);
    if (schedule_bits == (cudaDeviceScheduleSpin | cudaDeviceScheduleYield) ||
        schedule_bits == (cudaDeviceScheduleSpin | cudaDeviceScheduleBlockingSync) ||
        schedule_bits == (cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync) ||
        schedule_bits == (cudaDeviceScheduleSpin | cudaDeviceScheduleYield |
                          cudaDeviceScheduleBlockingSync)) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

bool is_device_pointer(const void* ptr) {
    if (ptr == nullptr) {
        return false;
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    return state.allocations.resolve(ptr, &resolved);
}

cudaError_t resolve_memcpy_kind(void* dst, const void* src, cudaMemcpyKind kind, cudaMemcpyKind* resolved_kind) {
    if (resolved_kind == nullptr) {
        return cudaErrorInvalidValue;
    }

    const cudaError_t kind_status = validate_memcpy_kind(kind);
    if (kind_status != cudaSuccess) {
        return kind_status;
    }

    const bool dst_is_device = is_device_pointer(dst);
    const bool src_is_device = is_device_pointer(src);

    if (kind == cudaMemcpyDefault) {
        if (dst_is_device && src_is_device) {
            *resolved_kind = cudaMemcpyDeviceToDevice;
        } else if (dst_is_device && !src_is_device) {
            *resolved_kind = cudaMemcpyHostToDevice;
        } else if (!dst_is_device && src_is_device) {
            *resolved_kind = cudaMemcpyDeviceToHost;
        } else {
            *resolved_kind = cudaMemcpyHostToHost;
        }
        return cudaSuccess;
    }

    switch (kind) {
        case cudaMemcpyHostToHost:
            break;
        case cudaMemcpyHostToDevice:
            if (!dst_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            break;
        case cudaMemcpyDeviceToHost:
            if (!src_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            break;
        case cudaMemcpyDeviceToDevice:
            if (!dst_is_device || !src_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            break;
        case cudaMemcpyDefault:
            break;
    }

    *resolved_kind = kind;
    return cudaSuccess;
}

cudaError_t resolve_memcpy_to_symbol_kind(const void* src,
                                          cudaMemcpyKind kind,
                                          cudaMemcpyKind* resolved_kind) {
    if (resolved_kind == nullptr) {
        return cudaErrorInvalidValue;
    }

    const bool src_is_device = is_device_pointer(src);
    if (kind == cudaMemcpyDefault) {
        *resolved_kind = src_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        return cudaSuccess;
    }

    switch (kind) {
        case cudaMemcpyHostToDevice:
            *resolved_kind = cudaMemcpyHostToDevice;
            return cudaSuccess;
        case cudaMemcpyDeviceToDevice:
            if (!src_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            *resolved_kind = cudaMemcpyDeviceToDevice;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t resolve_memcpy_from_symbol_kind(void* dst,
                                            cudaMemcpyKind kind,
                                            cudaMemcpyKind* resolved_kind) {
    if (resolved_kind == nullptr) {
        return cudaErrorInvalidValue;
    }

    const bool dst_is_device = is_device_pointer(dst);
    if (kind == cudaMemcpyDefault) {
        *resolved_kind = dst_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
        return cudaSuccess;
    }

    switch (kind) {
        case cudaMemcpyDeviceToHost:
            *resolved_kind = cudaMemcpyDeviceToHost;
            return cudaSuccess;
        case cudaMemcpyDeviceToDevice:
            if (!dst_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            *resolved_kind = cudaMemcpyDeviceToDevice;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t checked_symbol_ptr(const void* symbol,
                               size_t count,
                               size_t offset,
                               const unsigned char** out_ptr) {
    if (symbol == nullptr || out_ptr == nullptr) {
        return cudaErrorInvalidValue;
    }

    const void* resolved_symbol = symbol;
    std::size_t resolved_size = 0;
    if (cumetal::registration::lookup_registered_symbol(symbol, &resolved_symbol, &resolved_size)) {
        if (resolved_size > 0 && (offset > resolved_size || count > (resolved_size - offset))) {
            return cudaErrorInvalidValue;
        }
    }

    if (offset > (std::numeric_limits<size_t>::max() - count)) {
        return cudaErrorInvalidValue;
    }

    *out_ptr = static_cast<const unsigned char*>(resolved_symbol) + offset;
    return cudaSuccess;
}

cudaError_t synchronize_stream_for_host_op(cudaStream_t stream,
                                           std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return resolve_status;
    }

    if (legacy_stream) {
        std::string error;
        return cumetal::metal_backend::synchronize(&error);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::stream_synchronize(backend_stream, &error);
    if (status != cudaSuccess) {
        return status;
    }

    if (out_stream != nullptr) {
        *out_stream = std::move(backend_stream);
    }
    return cudaSuccess;
}

cudaError_t update_event_completion(cudaEvent_t event, bool wait_for_completion) {
    if (event == nullptr) {
        return cudaErrorInvalidValue;
    }

    std::shared_ptr<cumetal::metal_backend::Stream> stream;
    std::uint64_t ticket = 0;
    {
        std::lock_guard<std::mutex> lock(event->mutex);
        if (event->complete) {
            return cudaSuccess;
        }
        stream = event->stream;
        ticket = event->ticket;
    }

    if (stream == nullptr || ticket == 0) {
        std::lock_guard<std::mutex> lock(event->mutex);
        event->complete = true;
        if (!event->disable_timing) {
            event->timestamp = std::chrono::steady_clock::now();
            event->timing_valid = true;
        }
        return cudaSuccess;
    }

    std::string error;
    bool is_complete = false;
    if (wait_for_completion) {
        const cudaError_t status = cumetal::metal_backend::stream_wait_ticket(stream, ticket, &error);
        if (status != cudaSuccess) {
            return status;
        }
        is_complete = true;
    } else {
        const cudaError_t status = cumetal::metal_backend::stream_query_ticket(stream, ticket,
                                                                                &is_complete, &error);
        if (status != cudaSuccess) {
            return status;
        }
        if (!is_complete) {
            return cudaErrorNotReady;
        }
    }

    std::lock_guard<std::mutex> lock(event->mutex);
    event->complete = true;
    if (!event->disable_timing) {
        event->timestamp = std::chrono::steady_clock::now();
        event->timing_valid = true;
    }
    return cudaSuccess;
}

template <typename T>
T read_scalar_launch_arg(void** args, std::uint32_t index) {
    T value{};
    std::memcpy(&value, args[index], sizeof(T));
    return value;
}

template <typename T>
T* read_pointer_launch_arg(void** args, std::uint32_t index) {
    T* value = nullptr;
    std::memcpy(&value, args[index], sizeof(value));
    return value;
}

bool kernel_name_contains(const std::string& kernel_name, std::string_view needle) {
    return kernel_name.find(needle) != std::string::npos;
}

bool kernel_name_matches_env_list(const std::string& kernel_name, const char* env_var) {
    if (env_var == nullptr || env_var[0] == '\0') {
        return false;
    }

    std::string token;
    for (const char* p = env_var;; ++p) {
        const char c = *p;
        if (c == ',' || c == '\0') {
            std::size_t begin = 0;
            while (begin < token.size() &&
                   std::isspace(static_cast<unsigned char>(token[begin])) != 0) {
                ++begin;
            }
            std::size_t end = token.size();
            while (end > begin &&
                   std::isspace(static_cast<unsigned char>(token[end - 1])) != 0) {
                --end;
            }
            if (end > begin) {
                const std::string_view needle(token.data() + begin, end - begin);
                if (kernel_name.find(needle) != std::string::npos) {
                    return true;
                }
            }
            token.clear();
            if (c == '\0') {
                break;
            }
            continue;
        }
        token.push_back(c);
    }

    return false;
}

bool env_truthy(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return normalized == "1" || normalized == "true" || normalized == "yes" ||
           normalized == "on";
}

bool llmc_emulation_enabled() {
    return !env_truthy(std::getenv("CUMETAL_DISABLE_LLMC_EMULATION"));
}

bool llmc_emulation_skips_kernel(const std::string& kernel_name) {
    return kernel_name_matches_env_list(kernel_name, std::getenv("CUMETAL_LLMC_EMULATION_SKIP"));
}

bool llmc_emulation_trace_enabled() {
    return env_truthy(std::getenv("CUMETAL_TRACE_LLMC_EMULATION"));
}

std::atomic<std::uint64_t>& llmc_emulation_count() {
    static std::atomic<std::uint64_t> count{0};
    return count;
}

void note_llmc_emulation_hit(const std::string& kernel_name, std::uint32_t arg_count) {
    const std::uint64_t hit = llmc_emulation_count().fetch_add(1, std::memory_order_relaxed) + 1;
    if (llmc_emulation_trace_enabled()) {
        std::fprintf(stderr,
                     "INFO: CUMETAL_LLMC_EMULATION kernel=%s arg_count=%u hit=%llu\n",
                     kernel_name.c_str(),
                     static_cast<unsigned int>(arg_count),
                     static_cast<unsigned long long>(hit));
    }
}

std::uint32_t llmc_expected_arg_count(const std::string& kernel_name) {
    if (kernel_name_contains(kernel_name, "encoder_forward_kernel3")) {
        return 7;
    }
    if (kernel_name_contains(kernel_name, "encoder_backward_kernel")) {
        return 7;
    }
    if (kernel_name_contains(kernel_name, "layernorm_forward_kernel3")) {
        return 8;
    }
    if (kernel_name_contains(kernel_name, "unpermute_kernel_backward")) {
        return 6;
    }
    if (kernel_name_contains(kernel_name, "unpermute_kernel")) {
        return 6;
    }
    if (kernel_name_contains(kernel_name, "permute_kernel_backward") &&
        !kernel_name_contains(kernel_name, "unpermute_kernel_backward")) {
        return 8;
    }
    if (kernel_name_contains(kernel_name, "permute_kernel") &&
        !kernel_name_contains(kernel_name, "unpermute_kernel")) {
        return 8;
    }
    if (kernel_name_contains(kernel_name, "softmax_forward_kernel5")) {
        return 5;
    }
    if (kernel_name_contains(kernel_name, "residual_forward_kernel")) {
        return 4;
    }
    if (kernel_name_contains(kernel_name, "gelu_forward_kernel")) {
        return 3;
    }
    if (kernel_name_contains(kernel_name, "gelu_backward_kernel")) {
        return 4;
    }
    if (kernel_name_contains(kernel_name, "matmul_backward_bias_kernel4")) {
        return 5;
    }
    if (kernel_name_contains(kernel_name, "layernorm_backward_kernel2")) {
        return 11;
    }
    if (kernel_name_contains(kernel_name, "softmax_autoregressive_backward_kernel")) {
        return 7;
    }
    if (kernel_name_contains(kernel_name, "adamw_kernel2")) {
        return 12;
    }
    if (kernel_name_contains(kernel_name, "fused_classifier_kernel3")) {
        return 9;
    }
    if (kernel_name_contains(kernel_name, "matmul_forward_kernel4")) {
        return 6;
    }
    return 0;
}

cudaError_t synchronize_for_emulated_kernel(
    bool legacy_stream,
    const std::shared_ptr<cumetal::metal_backend::Stream>& backend_stream) {
    if (legacy_stream || backend_stream == nullptr) {
        return cudaSuccess;
    }

    std::string error;
    return cumetal::metal_backend::stream_synchronize(backend_stream, &error);
}

cudaError_t emulate_matmul_forward_kernel4(
    dim3 grid_dim,
    dim3 block_dim,
    void** args,
    bool legacy_stream,
    const std::shared_ptr<cumetal::metal_backend::Stream>& backend_stream) {
    float* out = read_pointer_launch_arg<float>(args, 0);
    const float* inp = read_pointer_launch_arg<const float>(args, 1);
    const float* weight = read_pointer_launch_arg<const float>(args, 2);
    const float* bias = read_pointer_launch_arg<const float>(args, 3);
    const int c = read_scalar_launch_arg<int>(args, 4);
    const int oc = read_scalar_launch_arg<int>(args, 5);

    if (out == nullptr || inp == nullptr || weight == nullptr || c <= 0 || oc <= 0) {
        return cudaErrorInvalidValue;
    }

    const int tile_rows = static_cast<int>(block_dim.x) * 8;
    const int tile_cols = static_cast<int>(block_dim.y) * 8;
    if (tile_rows <= 0 || tile_cols <= 0) {
        return cudaErrorInvalidValue;
    }

    const int m = static_cast<int>(grid_dim.x) * tile_rows;
    if (m <= 0) {
        return cudaSuccess;
    }

    const cudaError_t sync_status = synchronize_for_emulated_kernel(legacy_stream, backend_stream);
    if (sync_status != cudaSuccess) {
        return sync_status;
    }

    if (bias != nullptr) {
        for (int row = 0; row < m; ++row) {
            std::memcpy(out + static_cast<std::size_t>(row) * static_cast<std::size_t>(oc),
                        bias,
                        static_cast<std::size_t>(oc) * sizeof(float));
        }
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation weight_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation inp_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation out_resolved;
    if (!state.allocations.resolve(weight, &weight_resolved) ||
        !state.allocations.resolve(inp, &inp_resolved) ||
        !state.allocations.resolve(out, &out_resolved)) {
        return cudaErrorInvalidDevicePointer;
    }

    std::string error;
    return cumetal::metal_backend::gemm_f32(
        /*transpose_left=*/true,
        /*transpose_right=*/false,
        oc,
        m,
        c,
        1.0f,
        weight_resolved.buffer,
        weight_resolved.offset,
        c,
        inp_resolved.buffer,
        inp_resolved.offset,
        c,
        bias != nullptr ? 1.0f : 0.0f,
        out_resolved.buffer,
        out_resolved.offset,
        oc,
        backend_stream,
        &error);
}

cudaError_t try_emulate_llmc_registered_kernel(
    const std::string& kernel_name,
    std::uint32_t arg_count,
    dim3 grid_dim,
    dim3 block_dim,
    void** args,
    bool legacy_stream,
    const std::shared_ptr<cumetal::metal_backend::Stream>& backend_stream,
    bool* handled) {
    if (handled == nullptr) {
        return cudaErrorInvalidValue;
    }
    *handled = false;

    if (args == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (kernel_name_contains(kernel_name, "matmul_forward_kernel4")) {
        *handled = true;
        if (arg_count < 6) {
            return cudaErrorInvalidValue;
        }
        return emulate_matmul_forward_kernel4(grid_dim, block_dim, args, legacy_stream, backend_stream);
    }

    const bool known_llmc_kernel =
        kernel_name_contains(kernel_name, "encoder_forward_kernel3") ||
        kernel_name_contains(kernel_name, "encoder_backward_kernel") ||
        kernel_name_contains(kernel_name, "layernorm_forward_kernel3") ||
        kernel_name_contains(kernel_name, "permute_kernel") ||
        kernel_name_contains(kernel_name, "unpermute_kernel") ||
        kernel_name_contains(kernel_name, "softmax_forward_kernel5") ||
        kernel_name_contains(kernel_name, "residual_forward_kernel") ||
        kernel_name_contains(kernel_name, "gelu_forward_kernel") ||
        kernel_name_contains(kernel_name, "gelu_backward_kernel") ||
        kernel_name_contains(kernel_name, "matmul_backward_bias_kernel4") ||
        kernel_name_contains(kernel_name, "layernorm_backward_kernel2") ||
        kernel_name_contains(kernel_name, "softmax_autoregressive_backward_kernel") ||
        kernel_name_contains(kernel_name, "adamw_kernel2") ||
        kernel_name_contains(kernel_name, "fused_classifier_kernel3");
    if (!known_llmc_kernel) {
        return cudaSuccess;
    }

    const cudaError_t sync_status = synchronize_for_emulated_kernel(legacy_stream, backend_stream);
    if (sync_status != cudaSuccess) {
        return sync_status;
    }

    if (kernel_name_contains(kernel_name, "encoder_forward_kernel3")) {
        if (arg_count < 7) {
            return cudaErrorInvalidValue;
        }
        float* out = read_pointer_launch_arg<float>(args, 0);
        const int* inp = read_pointer_launch_arg<const int>(args, 1);
        const float* wte = read_pointer_launch_arg<const float>(args, 2);
        const float* wpe = read_pointer_launch_arg<const float>(args, 3);
        const int b = read_scalar_launch_arg<int>(args, 4);
        const int t = read_scalar_launch_arg<int>(args, 5);
        const int c = read_scalar_launch_arg<int>(args, 6);
        if (out == nullptr || inp == nullptr || wte == nullptr || wpe == nullptr || b <= 0 || t <= 0 ||
            c <= 0) {
            return cudaErrorInvalidValue;
        }
        for (int bi = 0; bi < b; ++bi) {
            for (int ti = 0; ti < t; ++ti) {
                const int token = inp[bi * t + ti];
                const std::size_t out_base =
                    (static_cast<std::size_t>(bi) * static_cast<std::size_t>(t) +
                     static_cast<std::size_t>(ti)) *
                    static_cast<std::size_t>(c);
                const std::size_t wte_base = static_cast<std::size_t>(token) * static_cast<std::size_t>(c);
                const std::size_t wpe_base = static_cast<std::size_t>(ti) * static_cast<std::size_t>(c);
                for (int ci = 0; ci < c; ++ci) {
                    out[out_base + static_cast<std::size_t>(ci)] =
                        wte[wte_base + static_cast<std::size_t>(ci)] +
                        wpe[wpe_base + static_cast<std::size_t>(ci)];
                }
            }
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "encoder_backward_kernel")) {
        if (arg_count < 7) {
            return cudaErrorInvalidValue;
        }
        float* dwte = read_pointer_launch_arg<float>(args, 0);
        float* dwpe = read_pointer_launch_arg<float>(args, 1);
        const float* dout = read_pointer_launch_arg<const float>(args, 2);
        const int* inp = read_pointer_launch_arg<const int>(args, 3);
        const int b = read_scalar_launch_arg<int>(args, 4);
        const int t = read_scalar_launch_arg<int>(args, 5);
        const int c = read_scalar_launch_arg<int>(args, 6);
        if (dwte == nullptr || dwpe == nullptr || dout == nullptr || inp == nullptr || b <= 0 || t <= 0 ||
            c <= 0) {
            return cudaErrorInvalidValue;
        }
        for (int bi = 0; bi < b; ++bi) {
            for (int ti = 0; ti < t; ++ti) {
                const int token = inp[bi * t + ti];
                const std::size_t dout_base =
                    (static_cast<std::size_t>(bi) * static_cast<std::size_t>(t) +
                     static_cast<std::size_t>(ti)) *
                    static_cast<std::size_t>(c);
                const std::size_t dwte_base = static_cast<std::size_t>(token) * static_cast<std::size_t>(c);
                const std::size_t dwpe_base = static_cast<std::size_t>(ti) * static_cast<std::size_t>(c);
                for (int ci = 0; ci < c; ++ci) {
                    const float grad = dout[dout_base + static_cast<std::size_t>(ci)];
                    dwte[dwte_base + static_cast<std::size_t>(ci)] += grad;
                    dwpe[dwpe_base + static_cast<std::size_t>(ci)] += grad;
                }
            }
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "layernorm_forward_kernel3")) {
        if (arg_count < 8) {
            return cudaErrorInvalidValue;
        }
        float* out = read_pointer_launch_arg<float>(args, 0);
        float* mean = read_pointer_launch_arg<float>(args, 1);
        float* rstd = read_pointer_launch_arg<float>(args, 2);
        const float* inp = read_pointer_launch_arg<const float>(args, 3);
        const float* weight = read_pointer_launch_arg<const float>(args, 4);
        const float* bias = read_pointer_launch_arg<const float>(args, 5);
        const int n = read_scalar_launch_arg<int>(args, 6);
        const int c = read_scalar_launch_arg<int>(args, 7);
        if (out == nullptr || inp == nullptr || weight == nullptr || bias == nullptr || n <= 0 || c <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int row = 0; row < n; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
            const float* x = inp + base;
            float sum = 0.0f;
            for (int ci = 0; ci < c; ++ci) {
                sum += x[ci];
            }
            const float m = sum / static_cast<float>(c);
            if (mean != nullptr) {
                mean[row] = m;
            }
            float var_sum = 0.0f;
            for (int ci = 0; ci < c; ++ci) {
                const float diff = x[ci] - m;
                var_sum += diff * diff;
            }
            const float s = 1.0f / std::sqrt(var_sum / static_cast<float>(c) + 1.0e-5f);
            if (rstd != nullptr) {
                rstd[row] = s;
            }
            float* o = out + base;
            for (int ci = 0; ci < c; ++ci) {
                const float norm = (x[ci] - m) * s;
                o[ci] = norm * weight[ci] + bias[ci];
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "permute_kernel_backward") &&
        !kernel_name_contains(kernel_name, "unpermute_kernel_backward")) {
        if (arg_count < 8) {
            return cudaErrorInvalidValue;
        }
        float* dinp = read_pointer_launch_arg<float>(args, 0);
        const float* dq = read_pointer_launch_arg<const float>(args, 1);
        const float* dk = read_pointer_launch_arg<const float>(args, 2);
        const float* dv = read_pointer_launch_arg<const float>(args, 3);
        const int b = read_scalar_launch_arg<int>(args, 4);
        const int n = read_scalar_launch_arg<int>(args, 5);
        const int nh = read_scalar_launch_arg<int>(args, 6);
        const int d = read_scalar_launch_arg<int>(args, 7);
        if (dinp == nullptr || dq == nullptr || dk == nullptr || dv == nullptr || b <= 0 || n <= 0 ||
            nh <= 0 || d <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int bi = 0; bi < b; ++bi) {
            for (int nhi = 0; nhi < nh; ++nhi) {
                for (int ni = 0; ni < n; ++ni) {
                    for (int di = 0; di < d; ++di) {
                        const std::size_t idx =
                            (((static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) +
                               static_cast<std::size_t>(nhi)) *
                                  static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(ni)) *
                                 static_cast<std::size_t>(d)) +
                            static_cast<std::size_t>(di);
                        const std::size_t inp_idx =
                            (static_cast<std::size_t>(bi) * static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(3 * nh * d)) +
                            (static_cast<std::size_t>(ni) * static_cast<std::size_t>(3 * nh * d)) +
                            static_cast<std::size_t>(nhi * d + di);
                        dinp[inp_idx] = dq[idx];
                        dinp[inp_idx + static_cast<std::size_t>(nh * d)] = dk[idx];
                        dinp[inp_idx + static_cast<std::size_t>(2 * nh * d)] = dv[idx];
                    }
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "permute_kernel") &&
        !kernel_name_contains(kernel_name, "unpermute_kernel")) {
        if (arg_count < 8) {
            return cudaErrorInvalidValue;
        }
        float* q = read_pointer_launch_arg<float>(args, 0);
        float* k = read_pointer_launch_arg<float>(args, 1);
        float* v = read_pointer_launch_arg<float>(args, 2);
        const float* inp = read_pointer_launch_arg<const float>(args, 3);
        const int b = read_scalar_launch_arg<int>(args, 4);
        const int n = read_scalar_launch_arg<int>(args, 5);
        const int nh = read_scalar_launch_arg<int>(args, 6);
        const int d = read_scalar_launch_arg<int>(args, 7);
        if (q == nullptr || k == nullptr || v == nullptr || inp == nullptr || b <= 0 || n <= 0 ||
            nh <= 0 || d <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int bi = 0; bi < b; ++bi) {
            for (int nhi = 0; nhi < nh; ++nhi) {
                for (int ni = 0; ni < n; ++ni) {
                    for (int di = 0; di < d; ++di) {
                        const std::size_t idx =
                            (((static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) +
                               static_cast<std::size_t>(nhi)) *
                                  static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(ni)) *
                                 static_cast<std::size_t>(d)) +
                            static_cast<std::size_t>(di);
                        const std::size_t inp_idx =
                            (static_cast<std::size_t>(bi) * static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(3 * nh * d)) +
                            (static_cast<std::size_t>(ni) * static_cast<std::size_t>(3 * nh * d)) +
                            static_cast<std::size_t>(nhi * d + di);
                        q[idx] = inp[inp_idx];
                        k[idx] = inp[inp_idx + static_cast<std::size_t>(nh * d)];
                        v[idx] = inp[inp_idx + static_cast<std::size_t>(2 * nh * d)];
                    }
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "unpermute_kernel_backward")) {
        if (arg_count < 6) {
            return cudaErrorInvalidValue;
        }
        float* dinp = read_pointer_launch_arg<float>(args, 0);
        const float* dout = read_pointer_launch_arg<const float>(args, 1);
        const int b = read_scalar_launch_arg<int>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        const int nh = read_scalar_launch_arg<int>(args, 4);
        const int d = read_scalar_launch_arg<int>(args, 5);
        if (dinp == nullptr || dout == nullptr || b <= 0 || n <= 0 || nh <= 0 || d <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int bi = 0; bi < b; ++bi) {
            for (int nhi = 0; nhi < nh; ++nhi) {
                for (int ni = 0; ni < n; ++ni) {
                    for (int di = 0; di < d; ++di) {
                        const std::size_t idx =
                            (((static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) +
                               static_cast<std::size_t>(nhi)) *
                                  static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(ni)) *
                                 static_cast<std::size_t>(d)) +
                            static_cast<std::size_t>(di);
                        const std::size_t other_idx =
                            (static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) *
                                 static_cast<std::size_t>(n * d)) +
                            (static_cast<std::size_t>(ni) * static_cast<std::size_t>(nh * d)) +
                            static_cast<std::size_t>(nhi * d + di);
                        dinp[idx] = dout[other_idx];
                    }
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "unpermute_kernel")) {
        if (arg_count < 6) {
            return cudaErrorInvalidValue;
        }
        float* inp = read_pointer_launch_arg<float>(args, 0);
        float* out = read_pointer_launch_arg<float>(args, 1);
        const int b = read_scalar_launch_arg<int>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        const int nh = read_scalar_launch_arg<int>(args, 4);
        const int d = read_scalar_launch_arg<int>(args, 5);
        if (inp == nullptr || out == nullptr || b <= 0 || n <= 0 || nh <= 0 || d <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int bi = 0; bi < b; ++bi) {
            for (int nhi = 0; nhi < nh; ++nhi) {
                for (int ni = 0; ni < n; ++ni) {
                    for (int di = 0; di < d; ++di) {
                        const std::size_t idx =
                            (((static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) +
                               static_cast<std::size_t>(nhi)) *
                                  static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(ni)) *
                                 static_cast<std::size_t>(d)) +
                            static_cast<std::size_t>(di);
                        const std::size_t other_idx =
                            (static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) *
                                 static_cast<std::size_t>(n * d)) +
                            (static_cast<std::size_t>(ni) * static_cast<std::size_t>(nh * d)) +
                            static_cast<std::size_t>(nhi * d + di);
                        out[other_idx] = inp[idx];
                    }
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "softmax_forward_kernel5")) {
        if (arg_count < 5) {
            return cudaErrorInvalidValue;
        }
        float* out = read_pointer_launch_arg<float>(args, 0);
        const float inv_temperature = read_scalar_launch_arg<float>(args, 1);
        const float* inp = read_pointer_launch_arg<const float>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        const int t = read_scalar_launch_arg<int>(args, 4);
        if (out == nullptr || inp == nullptr || n <= 0 || t <= 0) {
            return cudaErrorInvalidValue;
        }

        const std::size_t rows = static_cast<std::size_t>(n) * static_cast<std::size_t>(t);
        for (std::size_t row = 0; row < rows; ++row) {
            const int own_pos = static_cast<int>(row % static_cast<std::size_t>(t));
            const float* x = inp + row * static_cast<std::size_t>(t);
            float* y = out + row * static_cast<std::size_t>(t);

            float max_val = -FLT_MAX;
            for (int i = 0; i <= own_pos; ++i) {
                max_val = std::max(max_val, x[i]);
            }

            float sum = 0.0f;
            for (int i = 0; i <= own_pos; ++i) {
                sum += std::exp(inv_temperature * (x[i] - max_val));
            }
            const float norm = sum > 0.0f ? (1.0f / sum) : 0.0f;

            for (int i = 0; i <= own_pos; ++i) {
                y[i] = std::exp(inv_temperature * (x[i] - max_val)) * norm;
            }
            for (int i = own_pos + 1; i < t; ++i) {
                y[i] = 0.0f;
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "residual_forward_kernel")) {
        if (arg_count < 4) {
            return cudaErrorInvalidValue;
        }
        float* out = read_pointer_launch_arg<float>(args, 0);
        const float* inp1 = read_pointer_launch_arg<const float>(args, 1);
        const float* inp2 = read_pointer_launch_arg<const float>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        if (out == nullptr || inp1 == nullptr || inp2 == nullptr || n < 0) {
            return cudaErrorInvalidValue;
        }
        for (int i = 0; i < n; ++i) {
            out[i] = inp1[i] + inp2[i];
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "gelu_forward_kernel")) {
        if (arg_count < 3) {
            return cudaErrorInvalidValue;
        }
        constexpr float kGeluScaling = 0.7978845608028654f;
        float* out = read_pointer_launch_arg<float>(args, 0);
        const float* inp = read_pointer_launch_arg<const float>(args, 1);
        const int n = read_scalar_launch_arg<int>(args, 2);
        if (out == nullptr || inp == nullptr || n < 0) {
            return cudaErrorInvalidValue;
        }
        for (int i = 0; i < n; ++i) {
            const float x = inp[i];
            const float cube = 0.044715f * x * x * x;
            out[i] = 0.5f * x * (1.0f + std::tanh(kGeluScaling * (x + cube)));
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "gelu_backward_kernel")) {
        if (arg_count < 4) {
            return cudaErrorInvalidValue;
        }
        constexpr float kGeluScaling = 0.7978845608028654f;
        float* dinp = read_pointer_launch_arg<float>(args, 0);
        const float* inp = read_pointer_launch_arg<const float>(args, 1);
        const float* dout = read_pointer_launch_arg<const float>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        if (dinp == nullptr || inp == nullptr || dout == nullptr || n < 0) {
            return cudaErrorInvalidValue;
        }
        for (int i = 0; i < n; ++i) {
            const float x = inp[i];
            const float cube = 0.044715f * x * x * x;
            const float tanh_arg = kGeluScaling * (x + cube);
            const float tanh_out = std::tanh(tanh_arg);
            const float cosh_out = std::cosh(tanh_arg);
            const float sech2 = 1.0f / (cosh_out * cosh_out);
            const float local_grad = 0.5f * (1.0f + tanh_out) +
                                     x * 0.5f * sech2 * kGeluScaling *
                                         (1.0f + 3.0f * 0.044715f * x * x);
            dinp[i] = local_grad * dout[i];
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "matmul_backward_bias_kernel4")) {
        if (arg_count < 5) {
            return cudaErrorInvalidValue;
        }
        float* dbias = read_pointer_launch_arg<float>(args, 0);
        const float* dout = read_pointer_launch_arg<const float>(args, 1);
        const int b = read_scalar_launch_arg<int>(args, 2);
        const int t = read_scalar_launch_arg<int>(args, 3);
        const int oc = read_scalar_launch_arg<int>(args, 4);
        if (dbias == nullptr || dout == nullptr || b <= 0 || t <= 0 || oc <= 0) {
            return cudaErrorInvalidValue;
        }
        const int rows = b * t;
        for (int col = 0; col < oc; ++col) {
            float sum = 0.0f;
            for (int row = 0; row < rows; ++row) {
                sum += dout[static_cast<std::size_t>(row) * static_cast<std::size_t>(oc) +
                            static_cast<std::size_t>(col)];
            }
            dbias[col] += sum;
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "layernorm_backward_kernel2")) {
        if (arg_count < 11) {
            return cudaErrorInvalidValue;
        }
        float* dinp = read_pointer_launch_arg<float>(args, 0);
        float* dweight = read_pointer_launch_arg<float>(args, 1);
        float* dbias = read_pointer_launch_arg<float>(args, 2);
        const float* dout = read_pointer_launch_arg<const float>(args, 3);
        const float* inp = read_pointer_launch_arg<const float>(args, 4);
        const float* weight = read_pointer_launch_arg<const float>(args, 5);
        const float* mean = read_pointer_launch_arg<const float>(args, 6);
        const float* rstd = read_pointer_launch_arg<const float>(args, 7);
        const int b = read_scalar_launch_arg<int>(args, 8);
        const int t = read_scalar_launch_arg<int>(args, 9);
        const int c = read_scalar_launch_arg<int>(args, 10);
        if (dinp == nullptr || dweight == nullptr || dbias == nullptr || dout == nullptr || inp == nullptr ||
            weight == nullptr || mean == nullptr || rstd == nullptr || b <= 0 || t <= 0 || c <= 0) {
            return cudaErrorInvalidValue;
        }

        const int n = b * t;
        const float inv_c = 1.0f / static_cast<float>(c);
        const int warps_per_block = std::max(1u, block_dim.x / 32u);
        const int block_count = static_cast<int>(grid_dim.x);

        std::vector<float> block_dbias(static_cast<std::size_t>(c));
        std::vector<float> block_dweight(static_cast<std::size_t>(c));

        for (int block = 0; block < block_count; ++block) {
            std::fill(block_dbias.begin(), block_dbias.end(), 0.0f);
            std::fill(block_dweight.begin(), block_dweight.end(), 0.0f);

            for (int warp_rank = 0; warp_rank < warps_per_block; ++warp_rank) {
                const int row = block * warps_per_block + warp_rank;
                if (row >= n) {
                    continue;
                }

                const std::size_t base =
                    static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
                const float* dout_row = dout + base;
                const float* inp_row = inp + base;
                float* dinp_row = dinp + base;
                const float mean_row = mean[row];
                const float rstd_row = rstd[row];

                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int ci = 0; ci < c; ++ci) {
                    const float norm = (inp_row[ci] - mean_row) * rstd_row;
                    const float dnorm = weight[ci] * dout_row[ci];
                    dnorm_mean += dnorm;
                    dnorm_norm_mean += dnorm * norm;
                }
                dnorm_mean *= inv_c;
                dnorm_norm_mean *= inv_c;

                for (int ci = 0; ci < c; ++ci) {
                    const float norm = (inp_row[ci] - mean_row) * rstd_row;
                    const float dnorm = weight[ci] * dout_row[ci];
                    block_dbias[static_cast<std::size_t>(ci)] += dout_row[ci];
                    block_dweight[static_cast<std::size_t>(ci)] += norm * dout_row[ci];
                    const float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_row;
                    dinp_row[ci] += dval;
                }
            }

            for (int ci = 0; ci < c; ++ci) {
                dbias[ci] += block_dbias[static_cast<std::size_t>(ci)];
                dweight[ci] += block_dweight[static_cast<std::size_t>(ci)];
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "softmax_autoregressive_backward_kernel")) {
        if (arg_count < 7) {
            return cudaErrorInvalidValue;
        }
        float* dpreatt = read_pointer_launch_arg<float>(args, 0);
        const float* datt = read_pointer_launch_arg<const float>(args, 1);
        const float* att = read_pointer_launch_arg<const float>(args, 2);
        const int t = read_scalar_launch_arg<int>(args, 4);
        const float scale = read_scalar_launch_arg<float>(args, 6);
        if (dpreatt == nullptr || datt == nullptr || att == nullptr || t <= 0) {
            return cudaErrorInvalidValue;
        }

        const int heads = static_cast<int>(grid_dim.y);
        if (heads <= 0) {
            return cudaErrorInvalidValue;
        }

        const std::size_t head_stride =
            static_cast<std::size_t>(t) * static_cast<std::size_t>(t);
        for (int head = 0; head < heads; ++head) {
            const std::size_t head_base = static_cast<std::size_t>(head) * head_stride;
            for (int row = 0; row < t; ++row) {
                const std::size_t row_base =
                    head_base + static_cast<std::size_t>(row) * static_cast<std::size_t>(t);
                float local_sum = 0.0f;
                for (int col = 0; col <= row; ++col) {
                    local_sum += att[row_base + static_cast<std::size_t>(col)] *
                                 datt[row_base + static_cast<std::size_t>(col)];
                }
                for (int col = 0; col <= row; ++col) {
                    const float a = att[row_base + static_cast<std::size_t>(col)];
                    const float da = datt[row_base + static_cast<std::size_t>(col)];
                    dpreatt[row_base + static_cast<std::size_t>(col)] = scale * a * (da - local_sum);
                }
                for (int col = row + 1; col < t; ++col) {
                    dpreatt[row_base + static_cast<std::size_t>(col)] = 0.0f;
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "adamw_kernel2")) {
        if (arg_count < 12) {
            return cudaErrorInvalidValue;
        }
        float* params = read_pointer_launch_arg<float>(args, 0);
        float* grads = read_pointer_launch_arg<float>(args, 1);
        float* m = read_pointer_launch_arg<float>(args, 2);
        float* v = read_pointer_launch_arg<float>(args, 3);
        const std::int64_t num_parameters = read_scalar_launch_arg<std::int64_t>(args, 4);
        const float learning_rate = read_scalar_launch_arg<float>(args, 5);
        const float beta1 = read_scalar_launch_arg<float>(args, 6);
        const float beta2 = read_scalar_launch_arg<float>(args, 7);
        const float beta1_correction = read_scalar_launch_arg<float>(args, 8);
        const float beta2_correction = read_scalar_launch_arg<float>(args, 9);
        const float eps = read_scalar_launch_arg<float>(args, 10);
        const float weight_decay = read_scalar_launch_arg<float>(args, 11);
        if (params == nullptr || grads == nullptr || m == nullptr || v == nullptr || num_parameters < 0) {
            return cudaErrorInvalidValue;
        }

        const std::size_t count = static_cast<std::size_t>(num_parameters);
        for (std::size_t i = 0; i < count; ++i) {
            const float grad = grads[i];
            float m_val = m[i];
            float v_val = v[i];
            m_val = beta1 * m_val + (1.0f - beta1) * grad;
            v_val = beta2 * v_val + (1.0f - beta2) * (grad * grad);
            m[i] = m_val;
            v[i] = v_val;
            const float m_hat = m_val / beta1_correction;
            const float v_hat = v_val / beta2_correction;
            params[i] -= learning_rate *
                         (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * params[i]);
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "fused_classifier_kernel3")) {
        if (arg_count < 9) {
            return cudaErrorInvalidValue;
        }
        float* logits = read_pointer_launch_arg<float>(args, 0);
        float* losses = read_pointer_launch_arg<float>(args, 1);
        float* probs = read_pointer_launch_arg<float>(args, 2);
        const float* dlosses = read_pointer_launch_arg<const float>(args, 3);
        const int* targets = read_pointer_launch_arg<const int>(args, 4);
        const int b = read_scalar_launch_arg<int>(args, 5);
        const int t = read_scalar_launch_arg<int>(args, 6);
        const int v = read_scalar_launch_arg<int>(args, 7);
        const int p = read_scalar_launch_arg<int>(args, 8);
        if (logits == nullptr || losses == nullptr || targets == nullptr || b <= 0 || t <= 0 || v <= 0 ||
            p <= 0 || v > p) {
            return cudaErrorInvalidValue;
        }

        const int n = b * t;
        const float default_dloss = 1.0f / static_cast<float>(n);
        for (int row = 0; row < n; ++row) {
            const int target = targets[row];
            if (target < 0 || target >= v) {
                return cudaErrorInvalidValue;
            }

            float* row_logits = logits + static_cast<std::size_t>(row) * static_cast<std::size_t>(p);
            float max_val = -FLT_MAX;
            for (int col = 0; col < v; ++col) {
                max_val = std::max(max_val, row_logits[col]);
            }

            float sum = 0.0f;
            for (int col = 0; col < v; ++col) {
                sum += std::exp(row_logits[col] - max_val);
            }
            const float inv_sum = sum > 0.0f ? (1.0f / sum) : 0.0f;
            const float target_prob = std::exp(row_logits[target] - max_val) * inv_sum;
            losses[row] = -std::log(std::max(target_prob, 1.0e-30f));

            const float dloss = dlosses != nullptr ? dlosses[row] : default_dloss;
            for (int col = 0; col < v; ++col) {
                const float prob = std::exp(row_logits[col] - max_val) * inv_sum;
                if (probs != nullptr) {
                    probs[static_cast<std::size_t>(row) * static_cast<std::size_t>(p) +
                          static_cast<std::size_t>(col)] = prob;
                }
                const float indicator = (col == target) ? 1.0f : 0.0f;
                row_logits[col] = (prob - indicator) * dloss;
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    return cudaSuccess;
}

}  // namespace

namespace cumetal::rt {

bool resolve_allocation_for_pointer(const void* ptr, AllocationTable::ResolvedAllocation* out) {
    if (ptr == nullptr || out == nullptr) {
        return false;
    }
    RuntimeState& state = runtime_state();
    return state.allocations.resolve(ptr, out);
}

}  // namespace cumetal::rt

extern "C" {

int cumetalRuntimeIsDevicePointer(const void* ptr) {
    if (ptr == nullptr) {
        return 0;
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    return state.allocations.resolve(ptr, &resolved) ? 1 : 0;
}

// Returns 1 if ptr is in a known allocation; sets *base_out and *size_out.
// base_out = start of the containing allocation, size_out = total allocation size.
int cumetalRuntimeGetAllocationInfo(const void* ptr, void** base_out, size_t* size_out) {
    if (ptr == nullptr) {
        return 0;
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(ptr, &resolved)) {
        return 0;
    }
    const auto raw = reinterpret_cast<std::uintptr_t>(ptr);
    // base = ptr - offset into allocation
    const std::uintptr_t base_addr = raw - resolved.offset;
    if (base_out) *base_out = reinterpret_cast<void*>(base_addr);
    if (size_out) *size_out = resolved.offset + resolved.remaining_size;
    return 1;
}

// Returns 1 if ptr is a managed (unified) allocation.
int cumetalRuntimeIsManaged(const void* ptr) {
    if (ptr == nullptr) {
        return 0;
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(ptr, &resolved)) {
        return 0;
    }
    // cuMemAllocManaged allocations have the same coherent UMA properties.
    // We can distinguish them by host_alloc_flags (managed = no host flags set).
    return (resolved.kind == cumetal::rt::AllocationKind::kDevice) ? 1 : 0;
}

cudaError_t cudaInit(unsigned int flags) {
    if (flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t status = ensure_initialized();
    return fail(status);
}

cudaError_t cudaDriverGetVersion(int* driver_version) {
    if (driver_version == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    *driver_version = kCudaCompatVersion;
    return fail(cudaSuccess);
}

cudaError_t cudaRuntimeGetVersion(int* runtime_version) {
    if (runtime_version == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    *runtime_version = kCudaCompatVersion;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDeviceCount(int* count) {
    if (count == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    *count = 1;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDevice(int* device) {
    if (device == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    *device = state.current_device;
    return fail(cudaSuccess);
}

cudaError_t cudaSetDevice(int device) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (device != 0) {
        return fail(cudaErrorInvalidValue);
    }

    RuntimeState& state = runtime_state();
    state.current_device = device;
    return fail(cudaSuccess);
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    const cudaError_t flags_status = validate_device_flags(flags);
    if (flags_status != cudaSuccess) {
        return fail(flags_status);
    }

    RuntimeState& state = runtime_state();
    state.device_flags = flags;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
    if (flags == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    *flags = state.device_flags;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (prop == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (device != 0) {
        return fail(cudaErrorInvalidValue);
    }

    cumetal::metal_backend::DeviceProperties backend_props;
    std::string error;
    const cudaError_t query_status =
        cumetal::metal_backend::query_device_properties(&backend_props, &error);
    if (query_status != cudaSuccess) {
        return fail(query_status);
    }

    std::memset(prop, 0, sizeof(*prop));
    std::strncpy(prop->name, backend_props.name.c_str(), sizeof(prop->name) - 1);
    prop->name[sizeof(prop->name) - 1] = '\0';
    prop->totalGlobalMem = backend_props.total_global_mem;
    prop->warpSize = 32;
    prop->multiProcessorCount = backend_props.multi_processor_count;
    prop->maxThreadsPerBlock =
        backend_props.max_threads_per_block > 0 ? backend_props.max_threads_per_block : 1024;
    prop->maxThreadsDim[0] = prop->maxThreadsPerBlock;
    prop->maxThreadsDim[1] = prop->maxThreadsPerBlock;
    prop->maxThreadsDim[2] = prop->maxThreadsPerBlock;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->sharedMemPerBlock =
        backend_props.shared_mem_per_block > 0 ? backend_props.shared_mem_per_block : (32 * 1024);
    prop->regsPerBlock = 65536;
    prop->major = 8;
    prop->minor = 0;
    prop->unifiedAddressing = 1;        // UMA: CPU and GPU share physical DRAM
    prop->managedMemory = 1;            // cudaMallocManaged == cudaMalloc on UMA
    prop->concurrentManagedAccess = 1;  // CPU+GPU can access managed memory simultaneously
    prop->maxBufferArguments = 31;      // Metal buffer argument limit per kernel
    // Additional fields (spec §6.8)
    prop->clockRate = 1296000;          // ~1.3 GHz in kHz (conservative estimate)
    prop->memoryClockRate = 1296000;    // Same as GPU clock on UMA (shared memory controller)
    prop->memoryBusWidth = 128;         // 128-bit memory bus (conservative; M-series varies)
    prop->totalConstMem = 64 * 1024;    // 64 KB constant memory per module (spec §5.4.1)
    prop->sharedMemPerMultiprocessor = prop->sharedMemPerBlock; // Same as per-block on Metal
    prop->maxThreadsPerMultiProcessor = 2048; // Conservative estimate for M-series
    prop->l2CacheSize = 4 * 1024 * 1024; // 4 MB L2 (varies by chip; conservative)
    prop->canMapHostMemory = 1;          // UMA: all host memory is device-accessible
    prop->integrated = 1;               // Apple Silicon is an integrated GPU
    prop->concurrentKernels = 1;        // Metal supports concurrent command buffer execution
    prop->asyncEngineCount = 0;         // UMA makes async memcpy a memcpy, no DMA engine
    prop->computeMode = 0;              // 0 = cudaComputeModeDefault
    prop->pciBusID = 0;                 // No PCI bus on Apple Silicon
    prop->pciDeviceID = 0;
    prop->pciDomainID = 0;
    prop->tccDriver = 0;                // Not a Tesla compute cluster driver
    prop->kernelExecTimeoutEnabled = 0; // Metal does not enforce kernel timeout by default
    prop->pageableMemoryAccess = 1;     // UMA: device can access host pageable memory
    prop->pageableMemoryAccessUsesHostPageTables = 1; // Shared page tables on Apple Silicon

    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device) {
    if (value == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) {
        return fail(status);
    }

    switch (attr) {
        case cudaDevAttrMaxThreadsPerBlock:
            *value = prop.maxThreadsPerBlock;
            break;
        case cudaDevAttrMaxBlockDimX:
            *value = prop.maxThreadsDim[0];
            break;
        case cudaDevAttrMaxBlockDimY:
            *value = prop.maxThreadsDim[1];
            break;
        case cudaDevAttrMaxBlockDimZ:
            *value = prop.maxThreadsDim[2];
            break;
        case cudaDevAttrMaxGridDimX:
            *value = prop.maxGridSize[0];
            break;
        case cudaDevAttrMaxGridDimY:
            *value = prop.maxGridSize[1];
            break;
        case cudaDevAttrMaxGridDimZ:
            *value = prop.maxGridSize[2];
            break;
        case cudaDevAttrMaxSharedMemoryPerBlock:
            *value = prop.sharedMemPerBlock;
            break;
        case cudaDevAttrWarpSize:
            *value = prop.warpSize;
            break;
        case cudaDevAttrMultiProcessorCount:
            *value = prop.multiProcessorCount;
            break;
        case cudaDevAttrMaxRegistersPerBlock:
            *value = 65536;  // Metal has no per-block register limit; return generous value
            break;
        case cudaDevAttrClockRate:
            *value = 1296000;  // kHz — conservative estimate for M-series GPU
            break;
        case cudaDevAttrTextureAlignment:
            *value = 512;
            break;
        case cudaDevAttrGpuOverlap:
            *value = 1;  // Metal supports async compute + copy overlap
            break;
        case cudaDevAttrComputeCapabilityMajor:
            *value = 8;  // Ampere-equivalent feature set (spec §6.8)
            break;
        case cudaDevAttrComputeCapabilityMinor:
            *value = 0;  // Ampere-equivalent feature set (spec §6.8)
            break;
        case cudaDevAttrUnifiedAddressing:
        case cudaDevAttrManagedMemory:
        case cudaDevAttrConcurrentManagedAccess:
        case cudaDevAttrCanMapHostMemory:
        case cudaDevAttrIntegrated:
        case cudaDevAttrConcurrentKernels:
        case cudaDevAttrPageableMemoryAccess:
        case cudaDevAttrPageableMemoryAccessUsesHostPageTables:
            *value = 1;
            break;
        case cudaDevAttrMemoryBusWidth:
            *value = 128;
            break;
        case cudaDevAttrL2CacheSize:
            *value = 4 * 1024 * 1024;
            break;
        case cudaDevAttrMaxThreadsPerMultiProcessor:
            *value = 2048;
            break;
        case cudaDevAttrMemoryClockRate:
            *value = 1296000;
            break;
        case cudaDevAttrComputeMode:
        case cudaDevAttrPciBusId:
        case cudaDevAttrPciDeviceId:
        case cudaDevAttrPciDomainId:
        case cudaDevAttrTccDriver:
        case cudaDevAttrKernelExecTimeout:
        case cudaDevAttrAsyncEngineCount:
            *value = 0;
            break;
        case cudaDevAttrSharedMemPerBlockOptin: {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device);
            *value = static_cast<int>(prop.sharedMemPerBlock);
            break;
        }
        default:
            return fail(cudaErrorInvalidValue);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cumetal::metal_backend::DeviceProperties backend_props;
    std::string error;
    const cudaError_t query_status =
        cumetal::metal_backend::query_device_properties(&backend_props, &error);
    if (query_status != cudaSuccess) {
        return fail(query_status);
    }

    RuntimeState& state = runtime_state();
    const std::size_t allocated_bytes = state.allocations.total_allocated_size();
    const std::size_t total_mem = backend_props.total_global_mem;

    *total_bytes = total_mem;
    *free_bytes = allocated_bytes >= total_mem ? 0 : (total_mem - allocated_bytes);
    return fail(cudaSuccess);
}

cudaError_t cudaMalloc(void** dev_ptr, size_t size) {
    if (dev_ptr == nullptr || size == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> buffer;
    std::string error;
    const cudaError_t alloc_status = cumetal::metal_backend::allocate_buffer(size, &buffer, &error);
    if (alloc_status != cudaSuccess || buffer == nullptr) {
        return fail(alloc_status == cudaSuccess ? cudaErrorMemoryAllocation : alloc_status);
    }

    void* base = buffer->contents();
    if (base == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    RuntimeState& state = runtime_state();
    if (!state.allocations.insert(base, size, cumetal::rt::AllocationKind::kDevice,
                                  /*host_alloc_flags=*/0,
                                  std::move(buffer), &error)) {
        return fail(cudaErrorMemoryAllocation);
    }

    *dev_ptr = base;
    return fail(cudaSuccess);
}

cudaError_t cudaMallocManaged(void** dev_ptr, size_t size, unsigned int flags) {
    if (flags != 0) {
        return fail(cudaErrorInvalidValue);
    }
    return cudaMalloc(dev_ptr, size);
}

// Pitched 2D allocation — align pitch to 512 bytes (matching cudaDevAttrTextureAlignment).
cudaError_t cudaMallocPitch(void** dev_ptr, size_t* pitch, size_t width, size_t height) {
    if (dev_ptr == nullptr || pitch == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    constexpr size_t kAlign = 512;
    *pitch = (width + kAlign - 1) & ~(kAlign - 1);
    return cudaMalloc(dev_ptr, *pitch * height);
}

cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) {
    if (pitchedDevPtr == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
        pitchedDevPtr->ptr   = nullptr;
        pitchedDevPtr->pitch = 0;
        pitchedDevPtr->xsize = extent.width;
        pitchedDevPtr->ysize = extent.height;
        return fail(cudaSuccess);
    }
    constexpr size_t kAlign = 512;
    size_t pitch = (extent.width + kAlign - 1) & ~(kAlign - 1);
    void* ptr = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, pitch * extent.height * extent.depth);
    if (err != cudaSuccess) return err;
    pitchedDevPtr->ptr   = ptr;
    pitchedDevPtr->pitch = pitch;
    pitchedDevPtr->xsize = extent.width;
    pitchedDevPtr->ysize = extent.height;
    return fail(cudaSuccess);
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    if (ptr == nullptr || size == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t flags_status = validate_host_alloc_flags(flags);
    if (flags_status != cudaSuccess) {
        return fail(flags_status);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> buffer;
    std::string error;
    const cudaError_t alloc_status = cumetal::metal_backend::allocate_buffer(size, &buffer, &error);
    if (alloc_status != cudaSuccess || buffer == nullptr) {
        return fail(alloc_status == cudaSuccess ? cudaErrorMemoryAllocation : alloc_status);
    }

    void* base = buffer->contents();
    if (base == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    RuntimeState& state = runtime_state();
    if (!state.allocations.insert(base, size, cumetal::rt::AllocationKind::kHost, flags,
                                  std::move(buffer), &error)) {
        return fail(cudaErrorMemoryAllocation);
    }

    *ptr = base;
    return fail(cudaSuccess);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

cudaError_t cudaHostGetDevicePointer(void** dev_ptr, void* host_ptr, unsigned int flags) {
    if (dev_ptr == nullptr || host_ptr == nullptr || flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(host_ptr, &resolved) || resolved.offset != 0 ||
        resolved.kind != cumetal::rt::AllocationKind::kHost) {
        return fail(cudaErrorInvalidValue);
    }

    *dev_ptr = host_ptr;
    return fail(cudaSuccess);
}

cudaError_t cudaHostGetFlags(unsigned int* flags, void* host_ptr) {
    if (flags == nullptr || host_ptr == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(host_ptr, &resolved) || resolved.offset != 0 ||
        resolved.kind != cumetal::rt::AllocationKind::kHost) {
        return fail(cudaErrorInvalidValue);
    }

    *flags = resolved.host_alloc_flags;
    return fail(cudaSuccess);
}

cudaError_t cudaFreeHost(void* ptr) {
    return cudaFree(ptr);
}

cudaError_t cudaFree(void* dev_ptr) {
    if (dev_ptr == nullptr) {
        return fail(cudaSuccess);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    RuntimeState& state = runtime_state();
    if (!state.allocations.erase(dev_ptr)) {
        return fail(cudaErrorInvalidDevicePointer);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if ((dst == nullptr || src == nullptr) && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_kind(dst, src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(dst, src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyAsync(void* dst,
                            const void* src,
                            size_t count,
                            cudaMemcpyKind kind,
                            cudaStream_t stream) {
    if ((dst == nullptr || src == nullptr) && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_kind(dst, src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(dst, src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyToSymbol(const void* symbol,
                               const void* src,
                               size_t count,
                               size_t offset,
                               cudaMemcpyKind kind) {
    if (symbol == nullptr || (src == nullptr && count > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_to_symbol_kind(src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(const_cast<unsigned char*>(symbol_ptr), src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyFromSymbol(void* dst,
                                 const void* symbol,
                                 size_t count,
                                 size_t offset,
                                 cudaMemcpyKind kind) {
    if ((dst == nullptr && count > 0) || symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_from_symbol_kind(dst, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(dst, symbol_ptr, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol,
                                    const void* src,
                                    size_t count,
                                    size_t offset,
                                    cudaMemcpyKind kind,
                                    cudaStream_t stream) {
    if (symbol == nullptr || (src == nullptr && count > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_to_symbol_kind(src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(const_cast<unsigned char*>(symbol_ptr), src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyFromSymbolAsync(void* dst,
                                      const void* symbol,
                                      size_t count,
                                      size_t offset,
                                      cudaMemcpyKind kind,
                                      cudaStream_t stream) {
    if ((dst == nullptr && count > 0) || symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_from_symbol_kind(dst, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(dst, symbol_ptr, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemset(void* dev_ptr, int value, size_t count) {
    if (dev_ptr == nullptr && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memset(dev_ptr, value, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemsetAsync(void* dev_ptr, int value, size_t count, cudaStream_t stream) {
    if (dev_ptr == nullptr && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memset(dev_ptr, value, count);
    }

    return fail(cudaSuccess);
}

// 2D pitched memcpy — on UMA, copy width bytes per row for height rows.
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch,
                          const void* src, size_t spitch,
                          size_t width, size_t height,
                          cudaMemcpyKind /*kind*/) {
    if ((dst == nullptr || src == nullptr) && (width > 0 && height > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    auto* d = static_cast<uint8_t*>(dst);
    const auto* s = static_cast<const uint8_t*>(src);
    for (size_t row = 0; row < height; ++row) {
        if (width > 0) {
            std::memcpy(d + row * dpitch, s + row * spitch, width);
        }
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
                               const void* src, size_t spitch,
                               size_t width, size_t height,
                               cudaMemcpyKind kind, cudaStream_t stream) {
    const cudaError_t sync_status = synchronize_stream_for_host_op(stream, nullptr);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemset2D(void* dev_ptr, size_t pitch,
                          int value, size_t width, size_t height) {
    if (dev_ptr == nullptr && (width > 0 && height > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    auto* d = static_cast<uint8_t*>(dev_ptr);
    for (size_t row = 0; row < height; ++row) {
        if (width > 0) {
            std::memset(d + row * pitch, value, width);
        }
    }

    return fail(cudaSuccess);
}

// cudaMemset2DAsync — UMA: stream ignored; same as synchronous variant.
cudaError_t cudaMemset2DAsync(void* dev_ptr, size_t pitch,
                               int value, size_t width, size_t height,
                               cudaStream_t /*stream*/) {
    return cudaMemset2D(dev_ptr, pitch, value, width, height);
}

// cudaMemset3D — fills a 3D pitched allocation plane-by-row.
cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) {
    if (pitchedDevPtr.ptr == nullptr && (extent.width > 0 && extent.height > 0 && extent.depth > 0)) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    auto* base = static_cast<uint8_t*>(pitchedDevPtr.ptr);
    const size_t pitch      = pitchedDevPtr.pitch;
    const size_t plane_size = pitch * pitchedDevPtr.ysize;
    for (size_t z = 0; z < extent.depth; ++z) {
        for (size_t y = 0; y < extent.height; ++y) {
            if (extent.width > 0) {
                std::memset(base + z * plane_size + y * pitch, value, extent.width);
            }
        }
    }
    return fail(cudaSuccess);
}

// cudaMemset3DAsync — UMA: stream ignored; same as synchronous variant.
cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent,
                               cudaStream_t /*stream*/) {
    return cudaMemset3D(pitchedDevPtr, value, extent);
}

cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) {
    if (p == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    // Array-backed 3D copies are not supported on this implementation.
    if (p->srcArray != nullptr || p->dstArray != nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    const char* src_base  = static_cast<const char*>(p->srcPtr.ptr);
    char*       dst_base  = static_cast<char*>(p->dstPtr.ptr);
    size_t      src_pitch  = p->srcPtr.pitch  ? p->srcPtr.pitch  : p->extent.width;
    size_t      dst_pitch  = p->dstPtr.pitch  ? p->dstPtr.pitch  : p->extent.width;
    size_t      src_height = p->srcPtr.ysize  ? p->srcPtr.ysize  : p->extent.height;
    size_t      dst_height = p->dstPtr.ysize  ? p->dstPtr.ysize  : p->extent.height;

    for (size_t z = 0; z < p->extent.depth; ++z) {
        const size_t src_z_off = (p->srcPos.z + z) * src_pitch * src_height;
        const size_t dst_z_off = (p->dstPos.z + z) * dst_pitch * dst_height;
        for (size_t y = 0; y < p->extent.height; ++y) {
            const char* src_row = src_base + src_z_off
                                + (p->srcPos.y + y) * src_pitch + p->srcPos.x;
            char*       dst_row = dst_base + dst_z_off
                                + (p->dstPos.y + y) * dst_pitch + p->dstPos.x;
            std::memcpy(dst_row, src_row, p->extent.width);
        }
    }
    return fail(cudaSuccess);
}

cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) {
    (void)stream;
    return cudaMemcpy3D(p);
}

cudaError_t cudaDeviceReset(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    RuntimeState& state = runtime_state();
    std::vector<std::pair<cudaStream_t, std::shared_ptr<cumetal::metal_backend::Stream>>> streams;
    {
        std::lock_guard<std::mutex> lock(state.stream_mutex);
        streams.reserve(state.streams.size());
        for (auto& [handle, backend_stream] : state.streams) {
            streams.emplace_back(handle, std::move(backend_stream));
        }
        state.streams.clear();
    }

    for (auto& [handle, backend_stream] : streams) {
        if (backend_stream != nullptr) {
            std::string destroy_error;
            (void)cumetal::metal_backend::destroy_stream(backend_stream, &destroy_error);
        }
        delete handle;
    }

    if (tls_per_thread_stream != nullptr) {
        std::string destroy_error;
        (void)cumetal::metal_backend::destroy_stream(tls_per_thread_stream, &destroy_error);
        tls_per_thread_stream.reset();
    }

    state.allocations.clear();
    cumetal::registration::clear();
    clear_pending_launch_state();
    state.current_device = 0;
    state.device_flags = cudaDeviceScheduleAuto;
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceSynchronize(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::synchronize(&error);
    return fail(status);
}

cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    return cudaStreamCreateWithFlags(stream, cudaStreamDefault);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    if (stream == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (flags != cudaStreamDefault && flags != cudaStreamNonBlocking) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    std::string error;
    const cudaError_t status = cumetal::metal_backend::create_stream(&backend_stream, &error);
    if (status != cudaSuccess || backend_stream == nullptr) {
        return fail(status == cudaSuccess ? cudaErrorUnknown : status);
    }

    auto* handle = new (std::nothrow) cudaStream_st{};
    if (handle == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    RuntimeState& state = runtime_state();
    {
        std::lock_guard<std::mutex> lock(state.stream_mutex);
        state.streams.emplace(handle, std::move(backend_stream));
    }

    *stream = handle;
    return fail(cudaSuccess);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    if (!erase_stream_handle(stream, &backend_stream)) {
        return fail(cudaErrorInvalidValue);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::destroy_stream(backend_stream, &error);
    delete stream;
    return fail(status);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (is_legacy_stream_handle(stream)) {
        return cudaDeviceSynchronize();
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess || backend_stream == nullptr) {
        return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::stream_synchronize(backend_stream, &error);
    return fail(status);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (is_legacy_stream_handle(stream)) {
        std::vector<std::shared_ptr<cumetal::metal_backend::Stream>> streams;
        {
            RuntimeState& state = runtime_state();
            std::lock_guard<std::mutex> lock(state.stream_mutex);
            streams.reserve(state.streams.size());
            for (const auto& it : state.streams) {
                streams.push_back(it.second);
            }
        }
        if (tls_per_thread_stream != nullptr) {
            streams.push_back(tls_per_thread_stream);
        }

        for (const auto& backend_stream : streams) {
            std::uint64_t tail_ticket = 0;
            bool complete = true;
            std::string error;
            const cudaError_t tail_status =
                cumetal::metal_backend::stream_tail_ticket(backend_stream, &tail_ticket, &error);
            if (tail_status != cudaSuccess) {
                return fail(tail_status);
            }
            const cudaError_t query_status =
                cumetal::metal_backend::stream_query_ticket(backend_stream, tail_ticket,
                                                            &complete, &error);
            if (query_status != cudaSuccess) {
                return fail(query_status);
            }
            if (!complete) {
                return fail(cudaErrorNotReady);
            }
        }

        return fail(cudaSuccess);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess || backend_stream == nullptr) {
        return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
    }

    std::uint64_t tail_ticket = 0;
    bool complete = true;
    std::string error;
    const cudaError_t tail_status =
        cumetal::metal_backend::stream_tail_ticket(backend_stream, &tail_ticket, &error);
    if (tail_status != cudaSuccess) {
        return fail(tail_status);
    }
    const cudaError_t query_status =
        cumetal::metal_backend::stream_query_ticket(backend_stream, tail_ticket, &complete, &error);
    if (query_status != cudaSuccess) {
        return fail(query_status);
    }
    return fail(complete ? cudaSuccess : cudaErrorNotReady);
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void* user_data,
                                  unsigned int flags) {
    if (callback == nullptr || flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return fail(resolve_status);
    }

    std::uint64_t tail_ticket = 0;
    if (!legacy_stream) {
        std::string error;
        const cudaError_t ticket_status =
            cumetal::metal_backend::stream_tail_ticket(backend_stream, &tail_ticket, &error);
        if (ticket_status != cudaSuccess) {
            return fail(ticket_status);
        }
    }

    std::thread([stream, callback, user_data, backend_stream, tail_ticket, legacy_stream]() mutable {
        std::string error;
        cudaError_t callback_status = cudaSuccess;
        if (legacy_stream) {
            callback_status = cumetal::metal_backend::synchronize(&error);
        } else {
            callback_status =
                cumetal::metal_backend::stream_wait_ticket(backend_stream, tail_ticket, &error);
        }
        callback(stream, callback_status, user_data);
    }).detach();

    return fail(cudaSuccess);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    if (event == nullptr || flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    if (!is_legacy_stream_handle(stream)) {
        std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
        const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, nullptr);
        if (resolve_status != cudaSuccess || backend_stream == nullptr) {
            return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
        }
    }

    const cudaError_t status = update_event_completion(event, /*wait_for_completion=*/true);
    return fail(status);
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (event == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const unsigned int unsupported_flags = flags & ~(cudaEventDefault | cudaEventBlockingSync |
                                                      cudaEventDisableTiming);
    if (unsupported_flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    auto* created = new (std::nothrow) cudaEvent_st{};
    if (created == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    created->disable_timing = (flags & cudaEventDisableTiming) != 0;
    created->complete = true;
    created->recorded_once = false;
    created->timing_valid = false;
    created->ticket = 0;
    *event = created;
    return fail(cudaSuccess);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (event == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    delete event;
    return fail(cudaSuccess);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (event == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return fail(resolve_status);
    }

    std::uint64_t tail_ticket = 0;
    bool complete = true;
    if (!legacy_stream) {
        std::string error;
        const cudaError_t tail_status =
            cumetal::metal_backend::stream_tail_ticket(backend_stream, &tail_ticket, &error);
        if (tail_status != cudaSuccess) {
            return fail(tail_status);
        }

        if (tail_ticket > 0) {
            const cudaError_t query_status =
                cumetal::metal_backend::stream_query_ticket(backend_stream, tail_ticket,
                                                            &complete, &error);
            if (query_status != cudaSuccess) {
                return fail(query_status);
            }
        }
    } else {
        std::string error;
        const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
        if (sync_status != cudaSuccess) {
            return fail(sync_status);
        }
    }

    {
        std::lock_guard<std::mutex> lock(event->mutex);
        event->stream = std::move(backend_stream);
        event->ticket = tail_ticket;
        event->recorded_once = true;
        event->complete = complete;
        event->timing_valid = false;
        if (event->complete && !event->disable_timing) {
            event->timestamp = std::chrono::steady_clock::now();
            event->timing_valid = true;
        }
    }

    return fail(cudaSuccess);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    const cudaError_t status = update_event_completion(event, /*wait_for_completion=*/true);
    return fail(status);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    const cudaError_t status = update_event_completion(event, /*wait_for_completion=*/false);
    return fail(status);
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (ms == nullptr || start == nullptr || end == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t start_status = update_event_completion(start, /*wait_for_completion=*/false);
    if (start_status != cudaSuccess) {
        return fail(start_status);
    }
    const cudaError_t end_status = update_event_completion(end, /*wait_for_completion=*/false);
    if (end_status != cudaSuccess) {
        return fail(end_status);
    }

    std::chrono::steady_clock::time_point start_timestamp;
    std::chrono::steady_clock::time_point end_timestamp;
    {
        std::lock_guard<std::mutex> lock(start->mutex);
        if (start->disable_timing || !start->recorded_once || !start->timing_valid) {
            return fail(cudaErrorInvalidValue);
        }
        start_timestamp = start->timestamp;
    }
    {
        std::lock_guard<std::mutex> lock(end->mutex);
        if (end->disable_timing || !end->recorded_once || !end->timing_valid) {
            return fail(cudaErrorInvalidValue);
        }
        end_timestamp = end->timestamp;
    }

    *ms = std::chrono::duration<float, std::milli>(end_timestamp - start_timestamp).count();
    return fail(cudaSuccess);
}

cudaError_t cudaLaunchKernel(const void* func,
                             dim3 grid_dim,
                             dim3 block_dim,
                             void** args,
                             size_t shared_mem,
                             cudaStream_t stream) {
    if (func == nullptr || grid_dim.x == 0 || grid_dim.y == 0 || grid_dim.z == 0 || block_dim.x == 0 ||
        block_dim.y == 0 || block_dim.z == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cumetal::registration::RegisteredKernel registered_kernel;
    const bool use_registered_kernel =
        cumetal::registration::lookup_registered_kernel(func, &registered_kernel);

    cumetalKernel_t kernel_copy{};
    const cumetalKernel_t* kernel = nullptr;
    std::uint32_t arg_count = 0;
    const cumetalKernelArgInfo_t* arg_info = nullptr;

    if (use_registered_kernel) {
        if (registered_kernel.metallib_path.empty() || registered_kernel.kernel_name.empty() ||
            args == nullptr) {
            return fail(cudaErrorInvalidValue);
        }

        if (!registered_kernel.arg_info.empty()) {
            const std::uint32_t ptx_arg_count =
                static_cast<std::uint32_t>(registered_kernel.arg_info.size());
            // Clip to null-terminator: some callers pass nullptr as sentinel after real args
            std::uint32_t clipped = 0;
            for (; clipped < ptx_arg_count; ++clipped) {
                if (args[clipped] == nullptr) {
                    break;
                }
            }
            arg_count = clipped;
            arg_info = registered_kernel.arg_info.data();
        } else {
            const std::uint32_t known_arg_count = llmc_expected_arg_count(registered_kernel.kernel_name);
            if (known_arg_count > 0) {
                arg_count = known_arg_count;
            } else {
                std::size_t inferred_count = 0;
                for (; inferred_count < 31; ++inferred_count) {
                    if (args[inferred_count] == nullptr) {
                        break;
                    }
                }
                if (inferred_count == 31) {
                    return fail(cudaErrorInvalidValue);
                }
                arg_count = static_cast<std::uint32_t>(inferred_count);
            }
        }
    } else {
        std::memcpy(&kernel_copy, func, sizeof(kernel_copy));
        kernel = &kernel_copy;
        if (kernel->metallib_path == nullptr || kernel->kernel_name == nullptr || kernel->arg_count > 31) {
            return fail(cudaErrorInvalidValue);
        }
        if (kernel->arg_count > 0 && args == nullptr) {
            return fail(cudaErrorInvalidValue);
        }
        arg_count = kernel->arg_count;
        arg_info = kernel->arg_info;
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return fail(resolve_status);
    }
    if (legacy_stream) {
        std::string error;
        const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
        if (sync_status != cudaSuccess) {
            return fail(sync_status);
        }
    }

    if (use_registered_kernel && llmc_emulation_enabled() &&
        !llmc_emulation_skips_kernel(registered_kernel.kernel_name)) {
        bool emulated = false;
        const cudaError_t emulation_status =
            try_emulate_llmc_registered_kernel(registered_kernel.kernel_name,
                                               arg_count,
                                               grid_dim,
                                               block_dim,
                                               args,
                                               legacy_stream,
                                               backend_stream,
                                               &emulated);
        if (emulation_status != cudaSuccess) {
            return fail(emulation_status);
        }
        if (emulated) {
            note_llmc_emulation_hit(registered_kernel.kernel_name, arg_count);
            return fail(cudaSuccess);
        }
    }

    // Printf ring buffer size: 1 MB default, overridable via CUMETAL_PRINTF_BUFFER_SIZE (bytes).
    const std::uint32_t kPrintfCapWords = [&]() -> std::uint32_t {
        constexpr std::uint32_t kDefault = 256u * 1024u;  // 1 MB
        const char* env = std::getenv("CUMETAL_PRINTF_BUFFER_SIZE");
        if (env == nullptr || env[0] == '\0') return kDefault;
        const long val = std::strtol(env, nullptr, 10);
        if (val <= 0) return kDefault;
        const std::uint32_t words = static_cast<std::uint32_t>(val) / sizeof(std::uint32_t);
        return words > 0 ? words : kDefault;
    }();
    const bool needs_printf = use_registered_kernel && !registered_kernel.printf_formats.empty();

    std::vector<cumetal::metal_backend::KernelArg> launch_args;
    launch_args.reserve(static_cast<std::size_t>(arg_count) + (needs_printf ? 2u : 0u));

    RuntimeState& state = runtime_state();

    for (std::uint32_t i = 0; i < arg_count; ++i) {
        if (args == nullptr || args[i] == nullptr) {
            return fail(cudaErrorInvalidValue);
        }

        cumetalKernelArgInfo_t info{
            .kind = CUMETAL_ARG_BUFFER,
            .size_bytes = static_cast<std::uint32_t>(sizeof(void*)),
        };
        if (arg_info != nullptr) {
            info = arg_info[i];
        } else if (use_registered_kernel) {
            std::uintptr_t value = 0;
            std::memcpy(&value, args[i], sizeof(value));
            cumetal::rt::AllocationTable::ResolvedAllocation resolved_ptr;
            if (!state.allocations.resolve(reinterpret_cast<void*>(value), &resolved_ptr)) {
                info.kind = CUMETAL_ARG_BYTES;
                info.size_bytes = value <= 0xFFFFFFFFull ? 4u : 8u;
            }
        }

        if (info.kind == CUMETAL_ARG_BUFFER) {
            void* device_ptr = *reinterpret_cast<void**>(args[i]);
            if (use_registered_kernel && device_ptr == nullptr) {
                cumetal::metal_backend::KernelArg arg;
                arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
                arg.buffer = nullptr;
                arg.offset = 0;
                launch_args.push_back(std::move(arg));
                continue;
            }

            cumetal::rt::AllocationTable::ResolvedAllocation resolved;
            if (!state.allocations.resolve(device_ptr, &resolved)) {
                return fail(cudaErrorInvalidDevicePointer);
            }

            cumetal::metal_backend::KernelArg arg;
            arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
            arg.buffer = std::move(resolved.buffer);
            arg.offset = resolved.offset;
            launch_args.push_back(std::move(arg));
        } else {
            if (info.size_bytes == 0 || info.size_bytes > 4096) {
                return fail(cudaErrorInvalidValue);
            }

            cumetal::metal_backend::KernelArg arg;
            arg.kind = cumetal::metal_backend::KernelArg::Kind::kBytes;
            arg.bytes.resize(info.size_bytes);
            std::memcpy(arg.bytes.data(), args[i], info.size_bytes);
            launch_args.push_back(std::move(arg));
        }
    }

    // Append hidden printf ring-buffer args if the kernel uses device printf (spec §5.3).
    std::shared_ptr<cumetal::metal_backend::Buffer> printf_buffer;
    if (needs_printf) {
        const std::size_t kBufBytes =
            static_cast<std::size_t>(kPrintfCapWords) * sizeof(std::uint32_t);
        std::string alloc_error;
        const cudaError_t alloc_status =
            cumetal::metal_backend::allocate_buffer(kBufBytes, &printf_buffer, &alloc_error);
        if (alloc_status != cudaSuccess) {
            return fail(alloc_status);
        }
        std::memset(printf_buffer->contents(), 0, kBufBytes);
        {
            cumetal::metal_backend::KernelArg buf_arg;
            buf_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
            buf_arg.buffer = printf_buffer;
            buf_arg.offset = 0;
            launch_args.push_back(std::move(buf_arg));
        }
        {
            cumetal::metal_backend::KernelArg cap_arg;
            cap_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBytes;
            cap_arg.bytes.resize(sizeof(std::uint32_t));
            const std::uint32_t cap = kPrintfCapWords;
            std::memcpy(cap_arg.bytes.data(), &cap, sizeof(cap));
            launch_args.push_back(std::move(cap_arg));
        }
    }

    cumetal::metal_backend::LaunchConfig config{
        .grid = grid_dim,
        .block = block_dim,
        .shared_memory_bytes = shared_mem,
    };

    const char* metallib_path =
        use_registered_kernel ? registered_kernel.metallib_path.c_str() : kernel->metallib_path;
    const char* kernel_name =
        use_registered_kernel ? registered_kernel.kernel_name.c_str() : kernel->kernel_name;

    std::string error;
    const cudaError_t status =
        cumetal::metal_backend::launch_kernel(metallib_path, kernel_name, config, launch_args,
                                              backend_stream, &error);

    // Drain device printf output after kernel completes.
    if (needs_printf && printf_buffer != nullptr && status == cudaSuccess) {
        if (backend_stream != nullptr) {
            // Async stream: synchronize to ensure kernel output is visible.
            std::string sync_error;
            cumetal::metal_backend::stream_synchronize(backend_stream, &sync_error);
        }
        drain_printf_buffer(printf_buffer->contents(), kPrintfCapWords,
                            registered_kernel.printf_formats);
    }

    return fail(status);
}

cudaError_t cudaConfigureCall(dim3 grid_dim,
                              dim3 block_dim,
                              size_t shared_mem,
                              cudaStream_t stream) {
    if (grid_dim.x == 0 || grid_dim.y == 0 || grid_dim.z == 0 || block_dim.x == 0 ||
        block_dim.y == 0 || block_dim.z == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> resolved_stream;
    bool legacy_stream = false;
    const cudaError_t stream_status =
        resolve_runtime_stream(stream, &resolved_stream, &legacy_stream);
    if (stream_status != cudaSuccess) {
        return fail(stream_status);
    }

    clear_pending_launch_state();
    tls_pending_launch.configured = true;
    tls_pending_launch.grid_dim = grid_dim;
    tls_pending_launch.block_dim = block_dim;
    tls_pending_launch.shared_mem = shared_mem;
    tls_pending_launch.stream = stream;
    return fail(cudaSuccess);
}

cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset) {
    if (!tls_pending_launch.configured || arg == nullptr || size == 0) {
        return fail(cudaErrorInvalidValue);
    }

    if (offset > (std::numeric_limits<size_t>::max() - size)) {
        return fail(cudaErrorInvalidValue);
    }

    const size_t end = offset + size;
    if (end > tls_pending_launch.storage.size()) {
        tls_pending_launch.storage.resize(end, 0);
    }
    std::memcpy(tls_pending_launch.storage.data() + offset, arg, size);

    bool found = false;
    for (PendingLaunchArgument& pending_arg : tls_pending_launch.arguments) {
        if (pending_arg.offset != offset) {
            continue;
        }
        pending_arg.size = size;
        found = true;
        break;
    }
    if (!found) {
        tls_pending_launch.arguments.push_back(PendingLaunchArgument{
            .offset = offset,
            .size = size,
        });
    }

    return fail(cudaSuccess);
}

cudaError_t cudaLaunch(const void* func) {
    if (func == nullptr || !tls_pending_launch.configured) {
        return fail(cudaErrorInvalidValue);
    }

    std::vector<PendingLaunchArgument> ordered_args = tls_pending_launch.arguments;
    std::sort(ordered_args.begin(),
              ordered_args.end(),
              [](const PendingLaunchArgument& lhs, const PendingLaunchArgument& rhs) {
                  return lhs.offset < rhs.offset;
              });

    std::vector<void*> launch_args;
    launch_args.reserve(ordered_args.size() + 1);

    size_t previous_end = 0;
    for (const PendingLaunchArgument& pending_arg : ordered_args) {
        if (pending_arg.size == 0 ||
            pending_arg.offset > (std::numeric_limits<size_t>::max() - pending_arg.size)) {
            clear_pending_launch_state();
            return fail(cudaErrorInvalidValue);
        }

        const size_t end = pending_arg.offset + pending_arg.size;
        if (end > tls_pending_launch.storage.size() || pending_arg.offset < previous_end) {
            clear_pending_launch_state();
            return fail(cudaErrorInvalidValue);
        }
        previous_end = end;

        launch_args.push_back(reinterpret_cast<void*>(tls_pending_launch.storage.data() +
                                                      pending_arg.offset));
    }
    launch_args.push_back(nullptr);

    const dim3 grid_dim = tls_pending_launch.grid_dim;
    const dim3 block_dim = tls_pending_launch.block_dim;
    const size_t shared_mem = tls_pending_launch.shared_mem;
    const cudaStream_t stream = tls_pending_launch.stream;

    const cudaError_t status =
        cudaLaunchKernel(func, grid_dim, block_dim, launch_args.data(), shared_mem, stream);
    clear_pending_launch_state();
    return status;
}

cudaError_t cudaGetLastError(void) {
    const cudaError_t value = tls_last_error;
    tls_last_error = cudaSuccess;
    return value;
}

cudaError_t cudaPeekAtLastError(void) {
    return tls_last_error;
}

const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";
        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";
        case cudaErrorNotReady:
            return "cudaErrorNotReady";
        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";
        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";
        case cudaErrorUnknown:
            return "cudaErrorUnknown";
    }
    return "cudaErrorUnknown";
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";
        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";
        case cudaErrorNotReady:
            return "cudaErrorNotReady";
        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";
        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";
        case cudaErrorUnknown:
            return "cudaErrorUnknown";
    }
    return "cudaErrorUnknown";
}

cudaError_t cudaProfilerStart(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaProfilerStop(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    return fail(cudaSuccess);
}

// Occupancy API — returns conservative estimates (spec §8).
// Metal exposes no equivalent to SM occupancy; we return sensible defaults
// that allow block-size auto-tuning code to proceed without crashing.
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                          const void* /*func*/,
                                                          int /*blockSize*/,
                                                          size_t /*dynamicSMemSize*/) {
    if (numBlocks == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    *numBlocks = 2;  // conservative estimate
    return fail(cudaSuccess);
}

cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize,
                                               int* blockSize,
                                               const void* /*func*/,
                                               size_t /*dynamicSMemSize*/,
                                               int blockSizeLimit) {
    if (minGridSize == nullptr || blockSize == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    // Default to 256 threads/block unless the caller constrains it.
    const int chosen_block = (blockSizeLimit > 0 && blockSizeLimit < 256) ? blockSizeLimit : 256;
    *blockSize = chosen_block;
    // minGridSize = multiProcessorCount * 2 blocks/SM, rounded to grid coverage.
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.multiProcessorCount > 0) {
        *minGridSize = prop.multiProcessorCount * 2;
    } else {
        *minGridSize = 16;  // safe fallback
    }
    return fail(cudaSuccess);
}

// Function attribute query — returns zeroed/default attributes (spec §8).
// Metal pipelines expose no per-function register or shared-memory counts.
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* /*func*/) {
    if (attr == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    *attr = {};
    attr->maxThreadsPerBlock = 1024;
    attr->ptxVersion = 80;   // report Ampere-equivalent PTX ISA
    attr->binaryVersion = 80;
    return fail(cudaSuccess);
}

// No-ops — Metal has no L1/shared-memory configuration knobs.
cudaError_t cudaFuncSetCacheConfig(const void* /*func*/, cudaFuncCache /*cacheConfig*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaFuncSetSharedMemConfig(const void* /*func*/, cudaSharedMemConfig /*config*/) {
    return fail(cudaSuccess);
}

// No-op: Metal has no per-function attribute knobs corresponding to CUDA's.
// cudaFuncAttributeMaxDynamicSharedMemorySize is validated at launch time instead.
cudaError_t cudaFuncSetAttribute(const void* /*func*/, cudaFuncAttribute attr, int /*value*/) {
    if (attr != cudaFuncAttributeMaxDynamicSharedMemorySize &&
        attr != cudaFuncAttributePreferredSharedMemoryCarveout) {
        return fail(cudaErrorInvalidValue);
    }
    return fail(cudaSuccess);
}

// Device-level L1/shared-memory config — no-ops on Metal.
cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache /*cacheConfig*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) {
    if (pCacheConfig == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *pCacheConfig = cudaFuncCachePreferNone;
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig /*config*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) {
    if (pConfig == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *pConfig = cudaSharedMemBankSizeFourByte;  // default on CUDA
    return fail(cudaSuccess);
}

// Symbol address query: CuMetal registers __device__ variables as host-accessible
// pointers (UMA). The symbol pointer is the device address directly.
cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
    if (devPtr == nullptr || symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    // On UMA, the symbol's host address IS the device address.
    *devPtr = const_cast<void*>(symbol);
    return fail(cudaSuccess);
}

// Symbol size query: CuMetal doesn't track symbol sizes separately from the
// symbol pointer. Return cudaErrorInvalidSymbol to indicate the symbol table
// doesn't store sizes; callers should use sizeof() at compile time.
cudaError_t cudaGetSymbolSize(size_t* /*size*/, const void* symbol) {
    if (symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    // Symbol table doesn't store sizes separately; callers should use sizeof().
    return fail(cudaErrorInvalidValue);
}

// Unified Memory advisory APIs — no-ops on Apple Silicon UMA.
// Prefetch hints, locality advice, and access pattern hints have no effect when
// CPU and GPU share the same physical memory (UMA).
cudaError_t cudaMemPrefetchAsync(const void* /*devPtr*/,
                                  size_t /*count*/,
                                  int /*dstDevice*/,
                                  cudaStream_t /*stream*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaMemAdvise(const void* /*devPtr*/,
                           size_t /*count*/,
                           cudaMemoryAdvise /*advice*/,
                           int /*device*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaMemRangeGetAttribute(void* data,
                                      size_t dataSize,
                                      cudaMemRangeAttribute attribute,
                                      const void* /*devPtr*/,
                                      size_t /*count*/) {
    if (data == nullptr || dataSize == 0) {
        return fail(cudaErrorInvalidValue);
    }
    // On UMA: read-mostly is effectively always on; preferred location is device 0.
    if (attribute == cudaMemRangeAttributeReadMostly && dataSize >= sizeof(int)) {
        *reinterpret_cast<int*>(data) = 1;
    } else if (attribute == cudaMemRangeAttributePreferredLocation && dataSize >= sizeof(int)) {
        *reinterpret_cast<int*>(data) = 0;  // device 0
    } else if (attribute == cudaMemRangeAttributeLastPrefetchLocation && dataSize >= sizeof(int)) {
        *reinterpret_cast<int*>(data) = 0;
    } else if (dataSize >= sizeof(int)) {
        *reinterpret_cast<int*>(data) = 0;
    }
    return fail(cudaSuccess);
}

// Pointer attribute query — classifies a pointer as host, device, or managed.
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
    if (attributes == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    *attributes = {};
    attributes->device = 0;
    // On UMA, every CuMetal allocation is simultaneously host- and device-accessible.
    // Report the pointer as managed so callers handle it correctly.
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    const bool is_device_ptr =
        (ptr != nullptr) && runtime_state().allocations.resolve(ptr, &resolved);
    if (is_device_ptr) {
        attributes->type = cudaMemoryTypeManaged;
        attributes->devicePointer = const_cast<void*>(ptr);
        attributes->hostPointer = const_cast<void*>(ptr);
    } else {
        attributes->type = cudaMemoryTypeHost;
        attributes->hostPointer = const_cast<void*>(ptr);
    }
    return fail(cudaSuccess);
}

// Device selection by property — always returns device 0 (single GPU on Apple Silicon).
cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* /*prop*/) {
    if (device == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *device = 0;
    return fail(cudaSuccess);
}

// Peer access — Apple Silicon has a single GPU; no peer-to-peer access (spec §2.2).
cudaError_t cudaDeviceCanAccessPeer(int* can_access_peer, int /*device*/, int /*peer_device*/) {
    if (can_access_peer == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *can_access_peer = 0;
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceEnablePeerAccess(int /*peer_device*/, unsigned int /*flags*/) {
    return fail(cudaErrorInvalidValue);
}

cudaError_t cudaDeviceDisablePeerAccess(int /*peer_device*/) {
    return fail(cudaErrorInvalidValue);
}

// Stream with priority — priority is ignored; Metal has no priority queue.
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags,
                                          int /*priority*/) {
    return cudaStreamCreateWithFlags(stream, flags);
}

// Priority range — Metal has no stream priority; both bounds are 0.
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;
    return fail(cudaSuccess);
}

// Device limits — Metal exposes no equivalent knobs; no-op set, sensible get.
cudaError_t cudaDeviceSetLimit(cudaLimit /*limit*/, size_t /*value*/) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) {
    if (pValue == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    switch (limit) {
        case cudaLimitStackSize:
            *pValue = 1024;
            break;
        case cudaLimitPrintfFifoSize:
            *pValue = 1024u * 1024u;  // 1 MB (matches CUMETAL_PRINTF_BUFFER_SIZE default)
            break;
        case cudaLimitMallocHeapSize:
            *pValue = 8u * 1024u * 1024u;  // 8 MB
            break;
        default:
            *pValue = 0;
            break;
    }
    return fail(cudaSuccess);
}

// Peer memcpy — single GPU; ignore device IDs and forward to standard memcpy.
cudaError_t cudaMemcpyPeer(void* dst, int /*dstDevice*/,
                            const void* src, int /*srcDevice*/,
                            size_t count) {
    return cudaMemcpy(dst, src, count, cudaMemcpyDefault);
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int /*dstDevice*/,
                                 const void* src, int /*srcDevice*/,
                                 size_t count, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream);
}

// cudaLaunchHostFunc — synchronizes the stream then invokes fn(userData) on the host.
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    if (fn == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        return fail(sync_err);
    }
    fn(userData);
    return fail(cudaSuccess);
}

// cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags — flags ignored on Metal.
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                                    const void* func,
                                                                    int blockSize,
                                                                    size_t dynamicSMemSize,
                                                                    unsigned int /*flags*/) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                         dynamicSMemSize);
}

// Cooperative kernel launch — grid-wide CG is not supported on Metal (no cross-
// threadgroup barrier), but threadgroup-scoped CG works.  Forward to cudaLaunchKernel
// so programs that only use thread_block CG continue to function (spec §8).
cudaError_t cudaLaunchCooperativeKernel(const void* func,
                                         dim3 gridDim,
                                         dim3 blockDim,
                                         void** args,
                                         size_t sharedMem,
                                         cudaStream_t stream) {
    return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

}  // extern "C"
