#include "cuda.h"

#include "cumetal/air_emitter/emitter.h"
#include "cumetal/common/metallib.h"
#include "cumetal/ptx/lower_to_llvm.h"
#include "cuda_runtime.h"
#include "module_cache.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <new>
#include <string>
#include <unordered_set>
#include <vector>

struct CUctx_st {
    CUdevice device = 0;
    unsigned int flags = 0;
};

struct CUfunc_st;

struct CUmod_st {
    std::string metallib_path;
    bool owns_metallib_path = false;
    std::vector<CUfunc_st*> functions;
};

struct CUfunc_st {
    CUmod_st* module = nullptr;
    std::string kernel_name;
};

namespace {

extern "C" int cumetalRuntimeIsDevicePointer(const void* ptr);
extern "C" int cumetalRuntimeGetAllocationInfo(const void* ptr, void** base_out, size_t* size_out);
extern "C" int cumetalRuntimeIsManaged(const void* ptr);

constexpr int kCudaCompatVersion = 12000;
constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr std::uint32_t kFatbinBlobMagic = 0xBA55ED50u;
constexpr std::uint16_t kFatbinHeaderMinSize = 16u;
constexpr std::size_t kMaxImageBytes = 64ull * 1024ull * 1024ull;

struct FatbinWrapper {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

struct FatbinBlobHeader {
    std::uint32_t magic = 0;
    std::uint16_t version = 0;
    std::uint16_t header_size = 0;
    std::uint64_t fat_size = 0;
};

struct DriverState {
    std::mutex mutex;
    bool initialized = false;
    std::unordered_set<CUctx_st*> contexts;
    std::unordered_set<CUmod_st*> modules;
    std::unordered_set<CUfunc_st*> functions;
    CUctx_st* current_context = nullptr;
};

struct DriverStreamCallbackPayload {
    CUstream stream = nullptr;
    CUstreamCallback callback = nullptr;
    void* user_data = nullptr;
};

DriverState& driver_state() {
    static DriverState state;
    return state;
}

CUresult map_cuda_error(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return CUDA_SUCCESS;
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevicePointer:
            return CUDA_ERROR_INVALID_VALUE;
        case cudaErrorMemoryAllocation:
            return CUDA_ERROR_OUT_OF_MEMORY;
        case cudaErrorInitializationError:
            return CUDA_ERROR_NOT_INITIALIZED;
        case cudaErrorLaunchTimeout:
            return CUDA_ERROR_LAUNCH_TIMEOUT;
        case cudaErrorNotReady:
            return CUDA_ERROR_NOT_READY;
        case cudaErrorIllegalAddress:
            return CUDA_ERROR_ILLEGAL_ADDRESS;
        case cudaErrorDevicesUnavailable:
            return CUDA_ERROR_DEVICES_UNAVAILABLE;
        case cudaErrorUnknown:
            return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_ERROR_UNKNOWN;
}

bool has_current_context_locked(const DriverState& state) {
    return state.current_context != nullptr;
}

CUresult require_initialized_context() {
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    return CUDA_SUCCESS;
}

bool is_valid_function_locked(const DriverState& state, CUfunction function) {
    return function != nullptr && state.functions.find(function) != state.functions.end();
}

bool is_valid_module_locked(const DriverState& state, CUmodule module) {
    return module != nullptr && state.modules.find(module) != state.modules.end();
}

CUresult create_module_from_path(const std::string& path, bool owns_path, CUmodule* out_module) {
    if (out_module == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!std::filesystem::exists(path)) {
        return CUDA_ERROR_INVALID_IMAGE;
    }

    auto* loaded = new (std::nothrow) CUmod_st{};
    if (loaded == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    loaded->metallib_path = path;
    loaded->owns_metallib_path = owns_path;

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.modules.insert(loaded);
    }

    *out_module = loaded;
    return CUDA_SUCCESS;
}

bool parse_metallib_size(const void* image, std::size_t* out_size) {
    if (image == nullptr || out_size == nullptr) {
        return false;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(image);
    if (std::memcmp(bytes, "MTLB", 4) != 0) {
        return false;
    }

    std::uint64_t raw_size = 0;
    std::memcpy(&raw_size, bytes + 0x10, sizeof(raw_size));
    if (raw_size < 0x20 || raw_size > (1ull << 31)) {
        return false;
    }

    *out_size = static_cast<std::size_t>(raw_size);
    return true;
}

bool parse_metallib_path_image(const void* image, std::string* out_path) {
    if (image == nullptr || out_path == nullptr) {
        return false;
    }

    const auto* chars = static_cast<const char*>(image);
    constexpr std::size_t kMaxPathBytes = 4096;
    std::size_t len = 0;
    for (; len < kMaxPathBytes; ++len) {
        if (chars[len] == '\0') {
            break;
        }
    }
    if (len == 0 || len == kMaxPathBytes) {
        return false;
    }

    std::error_code ec;
    std::filesystem::path candidate(std::string(chars, len));
    if (!std::filesystem::exists(candidate, ec) || ec) {
        return false;
    }
    if (!std::filesystem::is_regular_file(candidate, ec) || ec) {
        return false;
    }

    *out_path = candidate.string();
    return true;
}

bool extract_ptx_cstr(const char* chars, std::size_t max_bytes, std::string* out_ptx) {
    if (chars == nullptr || out_ptx == nullptr || max_bytes == 0) {
        return false;
    }

    const void* terminator = std::memchr(chars, '\0', max_bytes);
    if (terminator == nullptr) {
        return false;
    }

    const std::size_t size = static_cast<const char*>(terminator) - chars;
    if (size == 0) {
        return false;
    }

    const std::string candidate(chars, size);
    if (candidate.find(".version") == std::string::npos ||
        candidate.find(".entry") == std::string::npos) {
        return false;
    }

    *out_ptx = candidate;
    return true;
}

bool extract_ptx_from_blob(const std::uint8_t* bytes,
                           std::size_t size,
                           std::string* out_ptx) {
    if (bytes == nullptr || out_ptx == nullptr || size < 16) {
        return false;
    }

    static constexpr char kMarker[] = ".version";
    constexpr std::size_t kMarkerLen = sizeof(kMarker) - 1;

    for (std::size_t i = 0; i + kMarkerLen < size; ++i) {
        if (std::memcmp(bytes + i, kMarker, kMarkerLen) != 0) {
            continue;
        }

        std::string candidate;
        if (extract_ptx_cstr(reinterpret_cast<const char*>(bytes + i), size - i, &candidate)) {
            *out_ptx = std::move(candidate);
            return true;
        }

        const char* candidate_start = reinterpret_cast<const char*>(bytes + i);
        std::size_t candidate_size = size - i;
        const void* terminator = std::memchr(candidate_start, '\0', candidate_size);
        if (terminator != nullptr) {
            candidate_size = static_cast<const char*>(terminator) - candidate_start;
        } else {
            for (std::size_t j = candidate_size; j > 0; --j) {
                if (candidate_start[j - 1] == '}') {
                    candidate_size = j;
                    break;
                }
            }
        }

        if (candidate_size == 0) {
            continue;
        }

        const std::string sliced(candidate_start, candidate_size);
        if (sliced.find(".version") == std::string::npos || sliced.find(".entry") == std::string::npos) {
            continue;
        }
        *out_ptx = sliced;
        return true;
    }

    return false;
}

bool parse_direct_ptx_image(const void* image, std::string* out_ptx) {
    if (image == nullptr || out_ptx == nullptr) {
        return false;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(image);
    if (bytes[0] != static_cast<std::uint8_t>('.')) {
        return false;
    }

    return extract_ptx_cstr(static_cast<const char*>(image), 1ull << 20, out_ptx);
}

bool parse_fatbin_blob_ptx(const void* image, std::string* out_ptx) {
    if (image == nullptr || out_ptx == nullptr) {
        return false;
    }

    const auto* blob = static_cast<const std::uint8_t*>(image);
    FatbinBlobHeader header{};
    std::memcpy(&header, blob, sizeof(header));
    if (header.magic != kFatbinBlobMagic || header.header_size < kFatbinHeaderMinSize) {
        return false;
    }

    const std::size_t header_size = static_cast<std::size_t>(header.header_size);
    const std::size_t fat_size = static_cast<std::size_t>(header.fat_size);
    if (fat_size == 0 || header_size > kMaxImageBytes || fat_size > kMaxImageBytes ||
        header_size > (kMaxImageBytes - fat_size)) {
        return false;
    }

    return extract_ptx_from_blob(blob + header_size, fat_size, out_ptx);
}

bool parse_fatbin_wrapper_ptx(const void* image, std::string* out_ptx) {
    if (image == nullptr || out_ptx == nullptr) {
        return false;
    }

    // Some fatbin wrappers prepend private fields before the canonical wrapper.
    const auto* raw = static_cast<const std::uint8_t*>(image);
    constexpr std::size_t kOffsets[] = {0u, 16u};
    for (const std::size_t offset : kOffsets) {
        std::uint32_t magic = 0;
        std::uint32_t version = 0;
        std::memcpy(&magic, raw + offset, sizeof(magic));
        std::memcpy(&version, raw + offset + sizeof(magic), sizeof(version));
        if (magic != kFatbinWrapperMagic || version == 0 || version > 3) {
            continue;
        }

        const void* data = nullptr;
        std::memcpy(&data, raw + offset + sizeof(magic) + sizeof(version), sizeof(data));
        if (data == nullptr) {
            continue;
        }

        if (parse_direct_ptx_image(data, out_ptx)) {
            return true;
        }
        if (parse_fatbin_blob_ptx(data, out_ptx)) {
            return true;
        }
    }
    return false;
}

bool parse_ptx_image(const void* image, std::string* out_ptx) {
    if (image == nullptr || out_ptx == nullptr) {
        return false;
    }

    if (parse_direct_ptx_image(image, out_ptx)) {
        return true;
    }
    if (parse_fatbin_blob_ptx(image, out_ptx)) {
        return true;
    }
    return parse_fatbin_wrapper_ptx(image, out_ptx);
}

bool emit_ptx_to_temp_metallib(const std::string& ptx, std::string* out_path) {
    if (ptx.empty() || out_path == nullptr) {
        return false;
    }

    // Default to kEmulate: Apple Silicon GPU rejects double-precision ALU ops
    // (fmul double, @llvm.fma.f64) at pipeline-creation time even though the
    // xcrun metal compiler accepts them.  kEmulate uses Dekker FP32-pair
    // arithmetic (~44-bit mantissa) which runs on all Apple Silicon hardware.
    // Set CUMETAL_FP64_MODE=native to force kNative (IEEE 754 double, fails
    // at runtime on current hardware but useful for testing the compilation path).
    cumetal::ptx::LowerToLlvmOptions lower_opts;
    lower_opts.fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
    const char* fp64_env = std::getenv("CUMETAL_FP64_MODE");
    if (fp64_env != nullptr) {
        if (std::string(fp64_env) == "native") {
            lower_opts.fp64_mode = cumetal::ptx::Fp64Mode::kNative;
        } else if (std::string(fp64_env) == "warn") {
            lower_opts.fp64_mode = cumetal::ptx::Fp64Mode::kWarn;
        }
        // "emulate" is already the default; any other value is ignored
    }
    const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(ptx, lower_opts);
    if (!lowered.ok || lowered.llvm_ir.empty()) {
        return false;
    }

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path ll_path =
        std::filesystem::temp_directory_path() / ("cumetal-driver-" + std::to_string(stamp) + ".ll");
    const std::filesystem::path metallib_path =
        std::filesystem::temp_directory_path() / ("cumetal-driver-" + std::to_string(stamp) + ".metallib");

    std::string io_error;
    const std::vector<std::uint8_t> ll_bytes(lowered.llvm_ir.begin(), lowered.llvm_ir.end());
    if (!cumetal::common::write_file_bytes(ll_path, ll_bytes, &io_error)) {
        return false;
    }

    cumetal::air_emitter::EmitOptions emit_options;
    emit_options.input = ll_path;
    emit_options.output = metallib_path;
    emit_options.mode = cumetal::air_emitter::EmitMode::kXcrun;
    emit_options.overwrite = true;
    emit_options.validate_output = false;
    emit_options.fallback_to_experimental = true;
    emit_options.kernel_name = lowered.entry_name.empty() ? "vector_add" : lowered.entry_name;

    const auto emitted = cumetal::air_emitter::emit_metallib(emit_options);
    std::error_code ec;
    std::filesystem::remove(ll_path, ec);
    if (!emitted.ok || emitted.output.empty()) {
        std::filesystem::remove(metallib_path, ec);
        return false;
    }

    *out_path = emitted.output.string();
    return true;
}

bool stage_module_image_to_tempfile(const void* image, std::size_t size, std::string* out_path) {
    if (image == nullptr || size == 0 || out_path == nullptr) {
        return false;
    }

    const std::string filename =
        "cumetal_module_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".metallib";
    const std::filesystem::path path = std::filesystem::temp_directory_path() / filename;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out.write(static_cast<const char*>(image), static_cast<std::streamsize>(size));
    out.close();
    if (!out.good()) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return false;
    }

    *out_path = path.string();
    return true;
}

bool valid_context_flags(unsigned int flags) {
    constexpr unsigned int kSupportedContextFlags =
        CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD | CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST |
        CU_CTX_LMEM_RESIZE_TO_MAX;

    if ((flags & ~kSupportedContextFlags) != 0) {
        return false;
    }

    const unsigned int schedule_bits =
        flags & (CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD | CU_CTX_SCHED_BLOCKING_SYNC);
    if (schedule_bits == (CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD) ||
        schedule_bits == (CU_CTX_SCHED_SPIN | CU_CTX_SCHED_BLOCKING_SYNC) ||
        schedule_bits == (CU_CTX_SCHED_YIELD | CU_CTX_SCHED_BLOCKING_SYNC) ||
        schedule_bits == (CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD | CU_CTX_SCHED_BLOCKING_SYNC)) {
        return false;
    }

    return true;
}

void driver_stream_callback_bridge(cudaStream_t stream, cudaError_t status, void* user_data) {
    (void)stream;
    auto* payload = static_cast<DriverStreamCallbackPayload*>(user_data);
    if (payload == nullptr || payload->callback == nullptr) {
        return;
    }

    payload->callback(payload->stream, map_cuda_error(status), payload->user_data);
    delete payload;
}

bool is_runtime_device_pointer_word(CUdeviceptr value) {
    const std::uintptr_t raw = static_cast<std::uintptr_t>(value);
    if (raw == 0) {
        return false;
    }
    return cumetalRuntimeIsDevicePointer(reinterpret_cast<const void*>(raw)) != 0;
}

}  // namespace

extern "C" {

CUresult cuInit(unsigned int flags) {
    if (flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const CUresult status = map_cuda_error(cudaInit(0));
    if (status != CUDA_SUCCESS) {
        return status;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.initialized = true;
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int* driverVersion) {
    if (driverVersion == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *driverVersion = kCudaCompatVersion;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int* count) {
    if (count == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    *count = 1;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (ordinal != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    *device = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    if (name == nullptr || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, dev);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    std::strncpy(name, prop.name, static_cast<std::size_t>(len - 1));
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    if (bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, dev);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *bytes = prop.totalGlobalMem;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    if (pi == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, dev);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = prop.maxThreadsPerBlock;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
            *pi = prop.maxThreadsDim[0];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
            *pi = prop.maxThreadsDim[1];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
            *pi = prop.maxThreadsDim[2];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
            *pi = prop.maxGridSize[0];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
            *pi = prop.maxGridSize[1];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
            *pi = prop.maxGridSize[2];
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
            *pi = prop.sharedMemPerBlock;
            break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
            *pi = prop.warpSize;
            break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            *pi = prop.multiProcessorCount;
            break;
        case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
            *pi = 65536;
            break;
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
            *pi = 1296000;  // kHz — conservative estimate for M-series GPU
            break;
        case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
            *pi = 1;  // Metal supports async compute + copy overlap
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
            *pi = 8;  // Ampere-equivalent feature set (spec §6.8)
            break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
            *pi = 0;  // Ampere-equivalent feature set (spec §6.8)
            break;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
        case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
            *pi = 1;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }

    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!valid_context_flags(flags)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    auto* context = new (std::nothrow) CUctx_st{};
    if (context == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    context->device = dev;
    context->flags = flags;

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.contexts.insert(context);
        if (state.current_context == nullptr) {
            state.current_context = context;
        }
    }

    *pctx = context;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
    if (ctx == nullptr) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.contexts.find(ctx) == state.contexts.end()) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        state.contexts.erase(ctx);
        if (state.current_context == ctx) {
            state.current_context = state.contexts.empty() ? nullptr : *state.contexts.begin();
        }
    }

    delete ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (ctx != nullptr && state.contexts.find(ctx) == state.contexts.end()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    state.current_context = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    if (pctx == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    *pctx = state.current_context;
    return CUDA_SUCCESS;
}

// cuCtxPushCurrent — push context onto a per-thread stack (simplified: just sets current).
CUresult cuCtxPushCurrent(CUcontext ctx) {
    return cuCtxSetCurrent(ctx);
}

// cuCtxPopCurrent — pop context from per-thread stack (simplified: returns current, resets to null).
CUresult cuCtxPopCurrent(CUcontext* pctx) {
    if (pctx != nullptr) {
        CUresult r = cuCtxGetCurrent(pctx);
        if (r != CUDA_SUCCESS) return r;
    }
    return cuCtxSetCurrent(nullptr);
}

// Primary context — on Apple Silicon, the primary context *is* the only context.
// Retain just creates a fresh context for device 0.
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    if (pctx == nullptr) return CUDA_ERROR_INVALID_VALUE;
    return cuCtxCreate(pctx, 0, dev);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice /*dev*/) {
    // Destroy the current context if any; ignore errors (idempotent).
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (ctx != nullptr) {
        cuCtxDestroy(ctx);
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice* device) {
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    *device = state.current_context->device;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetFlags(unsigned int* flags) {
    if (flags == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    *flags = state.current_context->flags;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetFlags(unsigned int flags) {
    if (!valid_context_flags(flags)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!has_current_context_locked(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    state.current_context->flags = flags;
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }

    return map_cuda_error(cudaDeviceSynchronize());
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags) {
    if (flags != CU_STREAM_DEFAULT && flags != CU_STREAM_NON_BLOCKING) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t*>(phStream), flags));
}

CUresult cuStreamDestroy(CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuStreamSynchronize(CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuStreamQuery(CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamQuery(reinterpret_cast<cudaStream_t>(hStream)));
}

// Stream getters — Metal has no stream priority; both return 0.
CUresult cuStreamGetPriority(CUstream /*hStream*/, int* priority) {
    if (priority == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
    if (flags == nullptr) return CUDA_ERROR_INVALID_VALUE;
    // hStream==null is legacy stream (default), non-null was created with CU_STREAM_DEFAULT (0)
    // or CU_STREAM_NON_BLOCKING (1).  We don't store flags per stream; return 0.
    (void)hStream;
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamAddCallback(CUstream hStream,
                             CUstreamCallback callback,
                             void* userData,
                             unsigned int flags) {
    if (callback == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    auto* payload = new (std::nothrow) DriverStreamCallbackPayload{};
    if (payload == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    payload->stream = hStream;
    payload->callback = callback;
    payload->user_data = userData;

    const cudaError_t status = cudaStreamAddCallback(reinterpret_cast<cudaStream_t>(hStream),
                                                     driver_stream_callback_bridge,
                                                     payload,
                                                     flags);
    if (status != cudaSuccess) {
        delete payload;
        return map_cuda_error(status);
    }

    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(hStream),
                                              reinterpret_cast<cudaEvent_t>(hEvent),
                                              flags));
}

CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) {
    if (fn == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    // Wrap CUhostFn as a stream callback.
    struct Payload {
        CUhostFn fn;
        void* userData;
    };
    auto* payload = new Payload{fn, userData};
    auto wrapper = [](cudaStream_t, cudaError_t, void* arg) {
        auto* p = static_cast<Payload*>(arg);
        p->fn(p->userData);
        delete p;
    };
    cudaError_t status = cudaStreamAddCallback(reinterpret_cast<cudaStream_t>(hStream),
                                               wrapper, payload, 0);
    if (status != cudaSuccess) {
        delete payload;
        return map_cuda_error(status);
    }
    return CUDA_SUCCESS;
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int flags) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t*>(phEvent), flags));
}

CUresult cuEventDestroy(CUevent hEvent) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(hEvent)));
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventRecord(reinterpret_cast<cudaEvent_t>(hEvent),
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuEventSynchronize(CUevent hEvent) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(hEvent)));
}

CUresult cuEventQuery(CUevent hEvent) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventQuery(reinterpret_cast<cudaEvent_t>(hEvent)));
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaEventElapsedTime(pMilliseconds,
                                               reinterpret_cast<cudaEvent_t>(hStart),
                                               reinterpret_cast<cudaEvent_t>(hEnd)));
}

// cuModuleGetGlobal — global device variable lookup.
// CuMetal doesn't support runtime-addressable __device__ globals; return NOT_FOUND.
CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                            CUmodule /*hmod*/, const char* name) {
    if (name == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (dptr)  *dptr  = 0;
    if (bytes) *bytes = 0;
    return CUDA_ERROR_NOT_FOUND;
}

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    if (module == nullptr || fname == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }

    return create_module_from_path(fname, /*owns_path=*/false, module);
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    if (module == nullptr || image == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
    }

    std::string ptx_text;
    if (parse_ptx_image(image, &ptx_text)) {
        std::string compiled_metallib_path;
        if (emit_ptx_to_temp_metallib(ptx_text, &compiled_metallib_path)) {
            const CUresult load_status =
                create_module_from_path(compiled_metallib_path, /*owns_path=*/true, module);
            if (load_status != CUDA_SUCCESS) {
                std::error_code ec;
                std::filesystem::remove(compiled_metallib_path, ec);
            }
            return load_status;
        }
    }

    std::size_t size = 0;
    if (parse_metallib_size(image, &size)) {
        std::filesystem::path cache_path;
        if (cumetal::cache::stage_metallib_bytes(image, size, &cache_path, nullptr)) {
            return create_module_from_path(cache_path.string(), /*owns_path=*/false, module);
        }

        std::string staged_path;
        if (!stage_module_image_to_tempfile(image, size, &staged_path)) {
            return CUDA_ERROR_UNKNOWN;
        }

        const CUresult load_status = create_module_from_path(staged_path, /*owns_path=*/true, module);
        if (load_status != CUDA_SUCCESS) {
            std::error_code ec;
            std::filesystem::remove(staged_path, ec);
        }
        return load_status;
    }

    std::string image_path;
    if (parse_metallib_path_image(image, &image_path)) {
        return create_module_from_path(image_path, /*owns_path=*/false, module);
    }

    return CUDA_ERROR_INVALID_IMAGE;
}

CUresult cuModuleLoadDataEx(CUmodule* module,
                            const void* image,
                            unsigned int numOptions,
                            void* options,
                            void* optionValues) {
    if ((numOptions > 0) && (options == nullptr || optionValues == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule module) {
    if (module == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    DriverState& state = driver_state();
    std::string owned_path;
    bool remove_owned_path = false;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!is_valid_module_locked(state, module)) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        for (CUfunc_st* function : module->functions) {
            state.functions.erase(function);
            delete function;
        }
        module->functions.clear();
        if (module->owns_metallib_path) {
            owned_path = module->metallib_path;
            remove_owned_path = true;
        }
        state.modules.erase(module);
    }

    delete module;
    if (remove_owned_path) {
        std::error_code ec;
        std::filesystem::remove(owned_path, ec);
    }
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (hfunc == nullptr || hmod == nullptr || name == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (!is_valid_module_locked(state, hmod)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }

    auto* function = new (std::nothrow) CUfunc_st{};
    if (function == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    function->module = hmod;
    function->kernel_name = name;

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        hmod->functions.push_back(function);
        state.functions.insert(function);
    }

    *hfunc = function;
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void** kernelParams,
                        void** extra) {
    if (f == nullptr || gridDimX == 0 || gridDimY == 0 || gridDimZ == 0 || blockDimX == 0 ||
        blockDimY == 0 || blockDimZ == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (kernelParams != nullptr && extra != nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    DriverState& state = driver_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return CUDA_ERROR_NOT_INITIALIZED;
        }
        if (!has_current_context_locked(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (!is_valid_function_locked(state, f)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }

    std::vector<void*> launch_params;
    std::vector<CUdeviceptr> packed_arg_values;
    std::vector<cumetalKernelArgInfo_t> arg_info;

    if (extra != nullptr) {
        void* packed_buffer = nullptr;
        std::size_t packed_size = 0;
        bool have_buffer = false;
        bool have_size = false;

        for (std::size_t i = 0;;) {
            void* key = extra[i++];
            if (key == CU_LAUNCH_PARAM_END) {
                break;
            }
            if (key == CU_LAUNCH_PARAM_BUFFER_POINTER) {
                if (extra[i] == nullptr) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                packed_buffer = extra[i++];
                have_buffer = true;
                continue;
            }
            if (key == CU_LAUNCH_PARAM_BUFFER_SIZE) {
                if (extra[i] == nullptr) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                packed_size = *reinterpret_cast<const std::size_t*>(extra[i]);
                have_size = true;
                ++i;
                continue;
            }
            return CUDA_ERROR_INVALID_VALUE;
        }

        if (!have_buffer || !have_size || (packed_size % sizeof(CUdeviceptr) != 0)) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        const std::size_t arg_count = packed_size / sizeof(CUdeviceptr);
        if (arg_count > 31) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        const auto* packed_args = static_cast<const CUdeviceptr*>(packed_buffer);
        packed_arg_values.assign(packed_args, packed_args + arg_count);
        launch_params.reserve(arg_count);
        arg_info.reserve(arg_count);
        for (std::size_t i = 0; i < packed_arg_values.size(); ++i) {
            launch_params.push_back(&packed_arg_values[i]);
            if (is_runtime_device_pointer_word(packed_arg_values[i])) {
                arg_info.push_back(cumetalKernelArgInfo_t{
                    .kind = CUMETAL_ARG_BUFFER,
                    .size_bytes = 0,
                });
            } else {
                const std::uint32_t scalar_size =
                    (packed_arg_values[i] <= 0xFFFFFFFFull) ? 4u : 8u;
                arg_info.push_back(cumetalKernelArgInfo_t{
                    .kind = CUMETAL_ARG_BYTES,
                    .size_bytes = scalar_size,
                });
            }
        }
    } else if (kernelParams != nullptr) {
        std::size_t arg_count = 0;
        for (; arg_count < 31; ++arg_count) {
            if (kernelParams[arg_count] == nullptr) {
                break;
            }
        }
        if (arg_count == 31 && kernelParams[31] != nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        launch_params.reserve(arg_count);
        arg_info.reserve(arg_count);
        for (std::size_t i = 0; i < arg_count; ++i) {
            launch_params.push_back(kernelParams[i]);
            arg_info.push_back(cumetalKernelArgInfo_t{
                .kind = CUMETAL_ARG_BUFFER,
                .size_bytes = 0,
            });
        }
    }

    const cumetalKernel_t kernel{
        .metallib_path = f->module->metallib_path.c_str(),
        .kernel_name = f->kernel_name.c_str(),
        .arg_count = static_cast<std::uint32_t>(arg_info.size()),
        .arg_info = arg_info.empty() ? nullptr : arg_info.data(),
    };

    const cudaError_t status = cudaLaunchKernel(&kernel,
                                                dim3(gridDimX, gridDimY, gridDimZ),
                                                dim3(blockDimX, blockDimY, blockDimZ),
                                                launch_params.empty() ? nullptr : launch_params.data(),
                                                sharedMemBytes,
                                                reinterpret_cast<cudaStream_t>(hStream));
    return map_cuda_error(status);
}

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    if (dptr == nullptr || bytesize == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    void* allocated = nullptr;
    const cudaError_t status = cudaMalloc(&allocated, bytesize);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *dptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(allocated));
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    if (dptr == nullptr || bytesize == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    void* allocated = nullptr;
    const cudaError_t status = cudaMallocManaged(&allocated, bytesize, flags);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *dptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(allocated));
    return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaFree(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dptr))));
}

CUresult cuMemAllocHost(void** pp, size_t bytesize) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaHostAlloc(pp, bytesize, cudaHostAllocDefault));
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaHostAlloc(pp, bytesize, flags));
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int flags) {
    if (pdptr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }

    void* device_ptr = nullptr;
    const cudaError_t status = cudaHostGetDevicePointer(&device_ptr, p, flags);
    if (status != cudaSuccess) {
        return map_cuda_error(status);
    }

    *pdptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device_ptr));
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaHostGetFlags(pFlags, p));
}

CUresult cuMemFreeHost(void* p) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaFreeHost(p));
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                     srcHost,
                                     ByteCount,
                                     cudaMemcpyHostToDevice));
}

CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpy(dstHost,
                                     reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                     ByteCount,
                                     cudaMemcpyDeviceToHost));
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                     reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                     ByteCount,
                                     cudaMemcpyDeviceToDevice));
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,
                           const void* srcHost,
                           size_t ByteCount,
                           CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                          srcHost,
                                          ByteCount,
                                          cudaMemcpyHostToDevice,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuMemcpyDtoHAsync(void* dstHost,
                           CUdeviceptr srcDevice,
                           size_t ByteCount,
                           CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpyAsync(dstHost,
                                          reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                          ByteCount,
                                          cudaMemcpyDeviceToHost,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice,
                           CUdeviceptr srcDevice,
                           size_t ByteCount,
                           CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemcpyAsync(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                          reinterpret_cast<void*>(static_cast<std::uintptr_t>(srcDevice)),
                                          ByteCount,
                                          cudaMemcpyDeviceToDevice,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

// cuMemcpyAsync — generic async D2D copy (direction inferred; on UMA, same as D2D).
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    return map_cuda_error(cudaMemcpyAsync(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)),
        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(src)),
        byteCount, cudaMemcpyDefault,
        reinterpret_cast<cudaStream_t>(hStream)));
}

// cuMemcpyPeer / cuMemcpyPeerAsync — single GPU on Apple Silicon; same as D2D copy.
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext /*dstContext*/,
                      CUdeviceptr srcDevice, CUcontext /*srcContext*/,
                      size_t ByteCount) {
    return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext /*dstContext*/,
                            CUdeviceptr srcDevice, CUcontext /*srcContext*/,
                            size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoDAsync(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) {
    if (pCopy == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    // Resolve src / dst to host-accessible pointers (UMA — device pointers are host pointers).
    const char* src_base =
        (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST)
            ? static_cast<const char*>(pCopy->srcHost)
            : reinterpret_cast<const char*>(static_cast<std::uintptr_t>(pCopy->srcDevice));
    char* dst_base =
        (pCopy->dstMemoryType == CU_MEMORYTYPE_HOST)
            ? static_cast<char*>(pCopy->dstHost)
            : reinterpret_cast<char*>(static_cast<std::uintptr_t>(pCopy->dstDevice));

    if (src_base == nullptr || dst_base == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const size_t src_pitch  = pCopy->srcPitch  ? pCopy->srcPitch  : pCopy->WidthInBytes;
    const size_t dst_pitch  = pCopy->dstPitch  ? pCopy->dstPitch  : pCopy->WidthInBytes;
    const size_t src_height = pCopy->srcHeight ? pCopy->srcHeight : pCopy->Height;
    const size_t dst_height = pCopy->dstHeight ? pCopy->dstHeight : pCopy->Height;

    for (size_t z = 0; z < pCopy->Depth; ++z) {
        const size_t sz_off = (pCopy->srcZ + z) * src_pitch * src_height;
        const size_t dz_off = (pCopy->dstZ + z) * dst_pitch * dst_height;
        for (size_t y = 0; y < pCopy->Height; ++y) {
            const char* src_row = src_base + sz_off + (pCopy->srcY + y) * src_pitch
                                + pCopy->srcXInBytes;
            char*       dst_row = dst_base + dz_off + (pCopy->dstY + y) * dst_pitch
                                + pCopy->dstXInBytes;
            std::memcpy(dst_row, src_row, pCopy->WidthInBytes);
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream /*hStream*/) {
    return cuMemcpy3D(pCopy);
}

CUresult cuMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemGetInfo(freeBytes, totalBytes));
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                     static_cast<int>(uc),
                                     N));
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return map_cuda_error(cudaMemsetAsync(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dstDevice)),
                                          static_cast<int>(uc),
                                          N,
                                          reinterpret_cast<cudaStream_t>(hStream)));
}

CUresult cuGetErrorName(CUresult error, const char** pStr) {
    if (pStr == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    switch (error) {
        case CUDA_SUCCESS:
            *pStr = "CUDA_SUCCESS";
            break;
        case CUDA_ERROR_INVALID_VALUE:
            *pStr = "CUDA_ERROR_INVALID_VALUE";
            break;
        case CUDA_ERROR_OUT_OF_MEMORY:
            *pStr = "CUDA_ERROR_OUT_OF_MEMORY";
            break;
        case CUDA_ERROR_NOT_INITIALIZED:
            *pStr = "CUDA_ERROR_NOT_INITIALIZED";
            break;
        case CUDA_ERROR_DEVICES_UNAVAILABLE:
            *pStr = "CUDA_ERROR_DEVICES_UNAVAILABLE";
            break;
        case CUDA_ERROR_INVALID_DEVICE:
            *pStr = "CUDA_ERROR_INVALID_DEVICE";
            break;
        case CUDA_ERROR_INVALID_IMAGE:
            *pStr = "CUDA_ERROR_INVALID_IMAGE";
            break;
        case CUDA_ERROR_INVALID_CONTEXT:
            *pStr = "CUDA_ERROR_INVALID_CONTEXT";
            break;
        case CUDA_ERROR_NOT_FOUND:
            *pStr = "CUDA_ERROR_NOT_FOUND";
            break;
        case CUDA_ERROR_NOT_READY:
            *pStr = "CUDA_ERROR_NOT_READY";
            break;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            *pStr = "CUDA_ERROR_ILLEGAL_ADDRESS";
            break;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            *pStr = "CUDA_ERROR_LAUNCH_TIMEOUT";
            break;
        case CUDA_ERROR_UNKNOWN:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
        default:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char** pStr) {
    return cuGetErrorName(error, pStr);
}

CUresult cuProfilerStart(void) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return CUDA_SUCCESS;
}

CUresult cuProfilerStop(void) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    return CUDA_SUCCESS;
}

// Stream with priority — priority ignored; Metal has no priority queue.
CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int /*priority*/) {
    return cuStreamCreate(phStream, flags);
}

// Priority range — Metal has no stream priority; both bounds are 0.
CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;
    return CUDA_SUCCESS;
}

// Cooperative kernel launch — forwards to cuLaunchKernel (threadgroup CG works; spec §8).
CUresult cuLaunchCooperativeKernel(CUfunction f,
                                    unsigned int gridDimX, unsigned int gridDimY,
                                    unsigned int gridDimZ, unsigned int blockDimX,
                                    unsigned int blockDimY, unsigned int blockDimZ,
                                    unsigned int sharedMemBytes, CUstream hStream,
                                    void** kernelParams) {
    return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                          sharedMemBytes, hStream, kernelParams, nullptr);
}

// Memset 16/32-bit variants backed by the existing 8-bit path (via loop for correctness).
CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    if (dstDevice == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsigned short* ptr = reinterpret_cast<unsigned short*>(static_cast<uintptr_t>(dstDevice));
    for (size_t i = 0; i < N; ++i) {
        ptr[i] = us;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    if (dstDevice == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsigned int* ptr = reinterpret_cast<unsigned int*>(static_cast<uintptr_t>(dstDevice));
    for (size_t i = 0; i < N; ++i) {
        ptr[i] = ui;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                           CUstream /*hStream*/) {
    return cuMemsetD16(dstDevice, us, N);
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                           CUstream /*hStream*/) {
    return cuMemsetD32(dstDevice, ui, N);
}

// 2D strided memset — fills Width elements per row for Height rows (pitch stride).
CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                       unsigned char uc, size_t Width, size_t Height) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* base = reinterpret_cast<unsigned char*>(static_cast<std::uintptr_t>(dstDevice));
    for (size_t row = 0; row < Height; ++row) {
        std::memset(base + row * dstPitch, static_cast<int>(uc), Width);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                        unsigned short us, size_t Width, size_t Height) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* base = reinterpret_cast<unsigned char*>(static_cast<std::uintptr_t>(dstDevice));
    for (size_t row = 0; row < Height; ++row) {
        auto* row_ptr = reinterpret_cast<unsigned short*>(base + row * dstPitch);
        for (size_t col = 0; col < Width; ++col) row_ptr[col] = us;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                        unsigned int ui, size_t Width, size_t Height) {
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    auto* base = reinterpret_cast<unsigned char*>(static_cast<std::uintptr_t>(dstDevice));
    for (size_t row = 0; row < Height; ++row) {
        auto* row_ptr = reinterpret_cast<unsigned int*>(base + row * dstPitch);
        for (size_t col = 0; col < Width; ++col) row_ptr[col] = ui;
    }
    return CUDA_SUCCESS;
}

// Async variants — UMA: stream ignored; same as synchronous.
CUresult cuMemsetD2D8Async(CUdeviceptr d, size_t p, unsigned char uc,
                            size_t W, size_t H, CUstream /*s*/) {
    return cuMemsetD2D8(d, p, uc, W, H);
}
CUresult cuMemsetD2D16Async(CUdeviceptr d, size_t p, unsigned short us,
                             size_t W, size_t H, CUstream /*s*/) {
    return cuMemsetD2D16(d, p, us, W, H);
}
CUresult cuMemsetD2D32Async(CUdeviceptr d, size_t p, unsigned int ui,
                             size_t W, size_t H, CUstream /*s*/) {
    return cuMemsetD2D32(d, p, ui, W, H);
}

// cuMemGetAddressRange — returns the base address and size of the allocation
// that contains dptr.
CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    if (pbase == nullptr && psize == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    void* base = nullptr;
    size_t sz = 0;
    const void* raw = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(dptr));
    if (!cumetalRuntimeGetAllocationInfo(raw, &base, &sz)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (pbase) *pbase = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(base));
    if (psize) *psize = sz;
    return CUDA_SUCCESS;
}

// cuPointerGetAttribute — query a single attribute of a device pointer.
CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    if (data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) return ready;
    const void* raw = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(ptr));
    const bool is_known = (cumetalRuntimeIsDevicePointer(raw) != 0);
    switch (attribute) {
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: {
            auto* out = static_cast<unsigned int*>(data);
            // On UMA all allocations are accessible from both host and device.
            // Return CU_MEMORYTYPE_UNIFIED for known allocations, HOST for host ptrs.
            *out = is_known ? static_cast<unsigned int>(CU_MEMORYTYPE_UNIFIED)
                            : static_cast<unsigned int>(CU_MEMORYTYPE_HOST);
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: {
            auto* out = static_cast<CUdeviceptr*>(data);
            *out = ptr;  // on UMA, device ptr == host ptr
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_HOST_POINTER: {
            auto* out = static_cast<void**>(data);
            *out = reinterpret_cast<void*>(static_cast<std::uintptr_t>(ptr));
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_IS_MANAGED: {
            auto* out = static_cast<unsigned int*>(data);
            *out = static_cast<unsigned int>(cumetalRuntimeIsManaged(raw));
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_MAPPED: {
            auto* out = static_cast<unsigned int*>(data);
            *out = is_known ? 1u : 0u;
            return CUDA_SUCCESS;
        }
        case CU_POINTER_ATTRIBUTE_CONTEXT: {
            auto* out = static_cast<CUcontext*>(data);
            cuCtxGetCurrent(out);
            return CUDA_SUCCESS;
        }
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
}

// Compute capability — synthetic 8.0 (Ampere-equivalent, spec §6.8).
CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
    if (major == nullptr || minor == nullptr || dev != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *major = 8;
    *minor = 0;
    return CUDA_SUCCESS;
}

// Peer access — Apple Silicon is single GPU; no peer-to-peer.
CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice /*dev*/, CUdevice /*peerDev*/) {
    if (canAccessPeer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *canAccessPeer = 0;
    return CUDA_SUCCESS;
}

// Peer context access — Apple Silicon has a single GPU; always rejected.
CUresult cuCtxEnablePeerAccess(CUcontext /*peerContext*/, unsigned int /*flags*/) {
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuCtxDisablePeerAccess(CUcontext /*peerContext*/) {
    return CUDA_ERROR_INVALID_VALUE;
}

// Pitched 2D allocation — align pitch to 512 bytes (matching texture alignment).
CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes,
                          size_t Height, unsigned int /*ElementSizeBytes*/) {
    if (dptr == nullptr || pPitch == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    constexpr size_t kAlign = 512;
    *pPitch = (WidthInBytes + kAlign - 1) & ~(kAlign - 1);
    return cuMemAlloc(dptr, *pPitch * Height);
}

// Occupancy API — conservative estimates (spec §8, driver-API counterparts).
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                     CUfunction /*func*/,
                                                     int /*blockSize*/,
                                                     size_t /*dynamicSMemSize*/) {
    if (numBlocks == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    *numBlocks = 2;
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize,
                                          int* blockSize,
                                          CUfunction /*func*/,
                                          size_t /*dynamicSMemSize*/,
                                          int blockSizeLimit) {
    if (minGridSize == nullptr || blockSize == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    const int chosen = (blockSizeLimit > 0 && blockSizeLimit < 256) ? blockSizeLimit : 256;
    *blockSize = chosen;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.multiProcessorCount > 0) {
        *minGridSize = prop.multiProcessorCount * 2;
    } else {
        *minGridSize = 16;
    }
    return CUDA_SUCCESS;
}

// Function attribute query — returns zeroed/default values (spec §8).
CUresult cuFuncGetAttribute(int* pi, CUfunc_attribute attrib, CUfunction /*hfunc*/) {
    if (pi == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult ready = require_initialized_context();
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    switch (attrib) {
        case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            *pi = 1024;
            break;
        case CU_FUNC_ATTRIBUTE_PTX_VERSION:
            *pi = 80;
            break;
        case CU_FUNC_ATTRIBUTE_BINARY_VERSION:
            *pi = 80;
            break;
        default:
            *pi = 0;
            break;
    }
    return CUDA_SUCCESS;
}

// No-op — Metal has no L1/shared-memory cache configuration.
CUresult cuFuncSetCacheConfig(CUfunction /*hfunc*/, CUfunc_cache /*config*/) {
    return CUDA_SUCCESS;
}

// No-op — Metal manages thread occupancy automatically.
CUresult cuFuncSetAttribute(CUfunction /*hfunc*/, CUfunc_attribute /*attrib*/, int /*value*/) {
    return CUDA_SUCCESS;
}

// Delegates to base function, ignoring flags.
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                              CUfunction func,
                                                              int blockSize,
                                                              size_t dynamicSMemSize,
                                                              unsigned int /*flags*/) {
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

}  // extern "C"
