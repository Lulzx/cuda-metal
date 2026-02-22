#include "registration.h"

#include "cumetal/air_emitter/emitter.h"
#include "cumetal/common/metallib.h"
#include "cumetal/ptx/lower_to_metal.h"
#include "cumetal/ptx/lower_to_llvm.h"
#include "cumetal/ptx/parser.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ─── debug trace ─────────────────────────────────────────────────────────────
// Set CUMETAL_DEBUG_REGISTRATION=1 to enable per-event diagnostic output to stderr.
// Defined at file scope so the REG_DEBUG macro works both inside and outside
// the cumetal::registration namespace (the __cuda* symbols live outside it).

namespace {

bool is_debug_registration() {
    static const bool kEnabled = []() {
        const char* v = std::getenv("CUMETAL_DEBUG_REGISTRATION");
        if (v == nullptr || v[0] == '\0') return false;
        const char c = v[0];
        return c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y';
    }();
    return kEnabled;
}

// ─── JIT cache ───────────────────────────────────────────────────────────────
// Compiled metallibs are stored persistently at:
//   $CUMETAL_CACHE_DIR/registration-jit/<hash>.metallib
// where hash = FNV-1a-64 over (ptx_source + '\0' + kernel_name).
// This avoids recompiling the same kernel across process restarts.

std::uint64_t fnv1a64_registration(const std::uint8_t* bytes, std::size_t size) {
    constexpr std::uint64_t kOffset = 1469598103934665603ull;
    constexpr std::uint64_t kPrime  = 1099511628211ull;
    std::uint64_t hash = kOffset;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= kPrime;
    }
    return hash;
}

std::string jit_cache_key(const std::string& ptx_source, const std::string& kernel_name) {
    // Hash ptx_source + NUL + kernel_name so different kernels from the same PTX get different keys.
    std::string blob;
    blob.reserve(ptx_source.size() + 1 + kernel_name.size());
    blob.append(ptx_source);
    blob.push_back('\0');
    blob.append(kernel_name);
    const std::uint64_t h = fnv1a64_registration(
        reinterpret_cast<const std::uint8_t*>(blob.data()), blob.size());
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << h;
    return oss.str();
}

std::filesystem::path jit_cache_root() {
    if (const char* d = std::getenv("CUMETAL_CACHE_DIR"); d != nullptr && d[0] != '\0') {
        return std::filesystem::path(d) / "registration-jit";
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        return std::filesystem::path(home) / "Library" / "Caches" / "io.cumetal" / "registration-jit";
    }
    return std::filesystem::temp_directory_path() / "io.cumetal" / "registration-jit";
}

// Returns the persistent cache path for a (ptx_source, kernel_name) pair.
// Returns an empty path if the cache directory cannot be created.
std::filesystem::path jit_cache_path_for(const std::string& ptx_source,
                                          const std::string& kernel_name) {
    const std::filesystem::path root = jit_cache_root();
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        return {};
    }
    return root / (jit_cache_key(ptx_source, kernel_name) + ".metallib");
}

}  // namespace

#define REG_DEBUG(fmt, ...)                                                       \
    do {                                                                          \
        if (is_debug_registration()) {                                            \
            std::fprintf(stderr, "[cumetal-reg] " fmt "\n", ##__VA_ARGS__);      \
        }                                                                         \
    } while (0)

namespace cumetal::registration {

constexpr std::uint32_t kCumetalFatbinMagic = 0x4C544D43u;  // "CMTL"
constexpr std::uint32_t kCumetalFatbinVersion = 1u;
constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr std::uint32_t kFatbinBlobMagic = 0xBA55ED50u;
constexpr std::uint16_t kFatbinHeaderMinSize = 16u;
constexpr std::size_t kMaxFatbinImageBytes = 64ull * 1024ull * 1024ull;

struct CumetalFatbinImage {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    const char* metallib_path = nullptr;
};

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

struct ParsedFatbinImage {
    std::string metallib_path;
    std::string ptx_source;
};

struct RegistrationModule {
    std::string metallib_path;
    std::string ptx_source;
    std::unordered_map<std::string, std::string> emitted_kernel_metallibs;
    std::unordered_map<std::string, std::vector<std::string>> emitted_kernel_printf_formats;
    std::vector<std::string> owned_metallibs;
};

struct RegistrationRecord {
    void* module_handle = nullptr;
    std::string metallib_path;
    std::string kernel_name;
    std::vector<cumetalKernelArgInfo_t> arg_info;
    std::vector<std::string> printf_formats;
};

struct RegistrationSymbolRecord {
    void* module_handle = nullptr;
    const void* device_address = nullptr;
    std::size_t size = 0;
};

struct RegistrationState {
    std::mutex mutex;
    std::unordered_map<void*, std::unique_ptr<RegistrationModule>> modules;
    std::unordered_map<const void*, RegistrationRecord> kernels;
    std::unordered_map<const void*, RegistrationSymbolRecord> symbols;
};

RegistrationState& state() {
    static RegistrationState s;
    return s;
}

std::string fallback_metallib_path_from_env() {
    const char* value = std::getenv("CUMETAL_FATBIN_METALLIB");
    if (value == nullptr) {
        return {};
    }
    return std::string(value);
}

void remove_path_if_exists(const std::string& path) {
    if (path.empty()) {
        return;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
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

bool parse_direct_ptx_image(const void* fat_cubin, std::string* out_ptx) {
    if (fat_cubin == nullptr || out_ptx == nullptr) {
        return false;
    }

    const auto* chars = static_cast<const char*>(fat_cubin);
    if (chars[0] != '.') {
        return false;
    }

    return extract_ptx_cstr(chars, 1ull << 20, out_ptx);
}

bool extract_ptx_from_blob(const std::uint8_t* bytes,
                           std::size_t size,
                           std::string* out_ptx) {
    if (bytes == nullptr || out_ptx == nullptr || size == 0) {
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

bool parse_fatbin_blob_ptx(const void* fat_cubin, std::string* out_ptx) {
    if (fat_cubin == nullptr || out_ptx == nullptr) {
        return false;
    }

    const auto* blob = static_cast<const std::uint8_t*>(fat_cubin);
    FatbinBlobHeader header{};
    std::memcpy(&header, blob, sizeof(header));
    if (header.magic != kFatbinBlobMagic || header.header_size < kFatbinHeaderMinSize) {
        return false;
    }

    const std::size_t header_size = static_cast<std::size_t>(header.header_size);
    const std::size_t fat_size = static_cast<std::size_t>(header.fat_size);
    if (fat_size == 0 || header_size > kMaxFatbinImageBytes || fat_size > kMaxFatbinImageBytes ||
        header_size > (kMaxFatbinImageBytes - fat_size)) {
        return false;
    }

    return extract_ptx_from_blob(blob + header_size, fat_size, out_ptx);
}

bool parse_fatbin_wrapper_ptx(const void* fat_cubin, std::string* out_ptx) {
    if (fat_cubin == nullptr || out_ptx == nullptr) {
        return false;
    }

    // Some fatbin wrappers prepend private fields before the canonical wrapper.
    const auto* raw = static_cast<const std::uint8_t*>(fat_cubin);
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

ParsedFatbinImage parse_fatbin_image(const void* fat_cubin) {
    ParsedFatbinImage parsed;
    if (fat_cubin == nullptr) {
        REG_DEBUG("parse_fatbin_image: null fat_cubin, using env fallback");
        parsed.metallib_path = fallback_metallib_path_from_env();
        return parsed;
    }

    CumetalFatbinImage image{};
    std::memcpy(&image, fat_cubin, sizeof(image));
    if (image.magic == kCumetalFatbinMagic && image.version == kCumetalFatbinVersion &&
        image.metallib_path != nullptr) {
        REG_DEBUG("parse_fatbin_image: CMTL native format -> %s", image.metallib_path);
        parsed.metallib_path = image.metallib_path;
        return parsed;
    }

    if (parse_fatbin_wrapper_ptx(fat_cubin, &parsed.ptx_source)) {
        REG_DEBUG("parse_fatbin_image: fatbin wrapper format, ptx_size=%zu",
                  parsed.ptx_source.size());
        return parsed;
    }
    if (parse_fatbin_blob_ptx(fat_cubin, &parsed.ptx_source)) {
        REG_DEBUG("parse_fatbin_image: fatbin blob format, ptx_size=%zu",
                  parsed.ptx_source.size());
        return parsed;
    }
    if (parse_direct_ptx_image(fat_cubin, &parsed.ptx_source)) {
        REG_DEBUG("parse_fatbin_image: direct PTX image, ptx_size=%zu",
                  parsed.ptx_source.size());
        return parsed;
    }

    REG_DEBUG("parse_fatbin_image: unrecognized format, using env fallback");
    parsed.metallib_path = fallback_metallib_path_from_env();
    return parsed;
}

std::uint32_t scalar_size_bytes_for_ptx_type(std::string_view ptx_type) {
    if (ptx_type == ".u8" || ptx_type == ".s8" || ptx_type == ".b8") {
        return 1u;
    }
    if (ptx_type == ".u16" || ptx_type == ".s16" || ptx_type == ".b16") {
        return 2u;
    }
    if (ptx_type == ".u64" || ptx_type == ".s64" || ptx_type == ".b64" || ptx_type == ".f64") {
        return 8u;
    }
    return 4u;
}

std::vector<cumetalKernelArgInfo_t> infer_arg_info_from_ptx_entry(const std::string& ptx_source,
                                                                  const std::string& kernel_name) {
    if (ptx_source.empty() || kernel_name.empty()) {
        return {};
    }

    cumetal::ptx::ParseOptions parse_options;
    parse_options.strict = false;
    const auto parsed = cumetal::ptx::parse_ptx(ptx_source, parse_options);
    if (!parsed.ok) {
        return {};
    }

    for (const auto& entry : parsed.module.entries) {
        if (entry.name != kernel_name) {
            continue;
        }

        std::vector<cumetalKernelArgInfo_t> arg_info;
        arg_info.reserve(entry.params.size());
        for (const auto& param : entry.params) {
            cumetalKernelArgInfo_t info{};
            if (param.is_pointer) {
                info.kind = CUMETAL_ARG_BUFFER;
                info.size_bytes = static_cast<std::uint32_t>(sizeof(void*));
            } else {
                info.kind = CUMETAL_ARG_BYTES;
                info.size_bytes = scalar_size_bytes_for_ptx_type(param.type);
            }
            arg_info.push_back(info);
        }
        return arg_info;
    }

    return {};
}

// out_is_persistent: set to true if the output lives in the persistent JIT cache
// (and therefore must NOT be deleted on __cudaUnregisterFatBinary).
bool emit_ptx_entry_to_temp_metallib(const std::string& ptx_source,
                                     const std::string& kernel_name,
                                     std::string* out_path,
                                     std::vector<std::string>* out_printf_formats = nullptr,
                                     bool* out_is_persistent = nullptr) {
    if (ptx_source.empty() || kernel_name.empty() || out_path == nullptr) {
        REG_DEBUG("emit_ptx_entry_to_temp_metallib: invalid argument (ptx=%zu, kernel=%s)",
                  ptx_source.size(), kernel_name.c_str());
        return false;
    }

    REG_DEBUG("emit kernel '%s' ptx_size=%zu", kernel_name.c_str(), ptx_source.size());

    // ── Persistent JIT cache lookup ────────────────────────────────────────
    // If a metallib for this exact (ptx_source, kernel_name) pair was compiled
    // in a prior run it lives at jit_cache_path_for(...).  Reuse it and skip
    // the expensive xcrun compile step.
    const std::filesystem::path cached_metallib = jit_cache_path_for(ptx_source, kernel_name);
    if (!cached_metallib.empty()) {
        std::error_code ec;
        if (std::filesystem::exists(cached_metallib, ec) && !ec) {
            REG_DEBUG("jit cache hit: %s", cached_metallib.c_str());
            *out_path = cached_metallib.string();
            if (out_is_persistent != nullptr) *out_is_persistent = true;
            // printf_formats are not cached across runs; callers handle the empty case gracefully.
            return true;
        }
        REG_DEBUG("jit cache miss: %s", cached_metallib.c_str());
    }

    // ── Compilation ───────────────────────────────────────────────────────
    // Use a timestamp-based name for intermediate files (ll/metal) that are
    // cleaned up immediately.  The final metallib lands in the persistent cache.
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path ll_path    = tmp / ("cumetal-registration-" + std::to_string(stamp) + ".ll");
    const std::filesystem::path metal_path = tmp / ("cumetal-registration-" + std::to_string(stamp) + ".metal");
    // Output goes to persistent cache if possible, otherwise /tmp.
    const std::filesystem::path metallib_path =
        cached_metallib.empty()
            ? tmp / ("cumetal-registration-" + std::to_string(stamp) + ".metallib")
            : cached_metallib;

    cumetal::air_emitter::EmitOptions emit_options;
    emit_options.output = metallib_path;
    emit_options.mode = cumetal::air_emitter::EmitMode::kXcrun;
    emit_options.overwrite = true;
    emit_options.validate_output = false;
    emit_options.fallback_to_experimental = true;
    emit_options.kernel_name = kernel_name;

    std::string io_error;
    cumetal::ptx::LowerToMetalOptions lower_to_metal_options;
    lower_to_metal_options.entry_name = kernel_name;
    const auto lowered_metal = cumetal::ptx::lower_ptx_to_metal_source(ptx_source, lower_to_metal_options);
    if (!lowered_metal.ok) {
        REG_DEBUG("lower_ptx_to_metal_source failed for kernel '%s'", kernel_name.c_str());
        return false;
    }

    std::filesystem::path staged_input = ll_path;
    if (lowered_metal.matched && !lowered_metal.metal_source.empty()) {
        REG_DEBUG("using direct Metal lowering path for '%s'", kernel_name.c_str());
        const std::vector<std::uint8_t> metal_bytes(lowered_metal.metal_source.begin(),
                                                    lowered_metal.metal_source.end());
        if (!cumetal::common::write_file_bytes(metal_path, metal_bytes, &io_error)) {
            REG_DEBUG("write metal source failed: %s", io_error.c_str());
            return false;
        }
        staged_input = metal_path;
        emit_options.kernel_name =
            lowered_metal.entry_name.empty() ? kernel_name : lowered_metal.entry_name;
        if (out_printf_formats != nullptr) {
            *out_printf_formats = lowered_metal.printf_formats;
        }
    } else {
        REG_DEBUG("using LLVM IR lowering path for '%s'", kernel_name.c_str());
        cumetal::ptx::LowerToLlvmOptions lower_options;
        lower_options.entry_name = kernel_name;
        const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(ptx_source, lower_options);
        if (!lowered.ok || lowered.llvm_ir.empty()) {
            REG_DEBUG("lower_ptx_to_llvm_ir failed for kernel '%s'", kernel_name.c_str());
            return false;
        }
        const std::vector<std::uint8_t> ll_bytes(lowered.llvm_ir.begin(), lowered.llvm_ir.end());
        if (!cumetal::common::write_file_bytes(ll_path, ll_bytes, &io_error)) {
            REG_DEBUG("write LLVM IR failed: %s", io_error.c_str());
            return false;
        }
        emit_options.kernel_name = lowered.entry_name.empty() ? kernel_name : lowered.entry_name;
    }
    emit_options.input = staged_input;

    REG_DEBUG("invoking emit_metallib: input=%s output=%s",
              staged_input.c_str(), metallib_path.c_str());
    const auto emitted = cumetal::air_emitter::emit_metallib(emit_options);
    remove_path_if_exists(ll_path.string());
    remove_path_if_exists(metal_path.string());
    if (!emitted.ok || emitted.output.empty()) {
        REG_DEBUG("emit_metallib failed for kernel '%s'", kernel_name.c_str());
        remove_path_if_exists(metallib_path.string());
        return false;
    }

    REG_DEBUG("emit success: %s", emitted.output.c_str());
    *out_path = emitted.output.string();
    // Persistent cache entries (those routed through jit_cache_path_for) should
    // survive process exit and __cudaUnregisterFatBinary cleanup.
    if (out_is_persistent != nullptr) *out_is_persistent = !cached_metallib.empty();
    return true;
}

std::string resolve_metallib_path_for_kernel(void* module_handle,
                                              const std::string& kernel_name,
                                              std::vector<std::string>* out_printf_formats) {
    if (module_handle == nullptr || kernel_name.empty()) {
        return fallback_metallib_path_from_env();
    }

    std::string ptx_source;
    {
        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto found = s.modules.find(module_handle);
        if (found == s.modules.end()) {
            return fallback_metallib_path_from_env();
        }

        const RegistrationModule& module = *found->second;
        if (!module.metallib_path.empty()) {
            REG_DEBUG("resolve_metallib '%s': prebuilt metallib '%s'",
                      kernel_name.c_str(), module.metallib_path.c_str());
            return module.metallib_path;
        }

        const auto cached = module.emitted_kernel_metallibs.find(kernel_name);
        if (cached != module.emitted_kernel_metallibs.end()) {
            REG_DEBUG("resolve_metallib '%s': in-process cache hit '%s'",
                      kernel_name.c_str(), cached->second.c_str());
            if (out_printf_formats != nullptr) {
                const auto pf_it = module.emitted_kernel_printf_formats.find(kernel_name);
                if (pf_it != module.emitted_kernel_printf_formats.end()) {
                    *out_printf_formats = pf_it->second;
                }
            }
            return cached->second;
        }

        ptx_source = module.ptx_source;
    }

    if (ptx_source.empty()) {
        REG_DEBUG("resolve_metallib '%s': no PTX, using env fallback", kernel_name.c_str());
        return fallback_metallib_path_from_env();
    }

    REG_DEBUG("resolve_metallib '%s': JIT compiling...", kernel_name.c_str());
    std::string emitted_path;
    std::vector<std::string> local_printf_formats;
    bool is_persistent = false;
    if (!emit_ptx_entry_to_temp_metallib(ptx_source, kernel_name, &emitted_path,
                                         &local_printf_formats, &is_persistent)) {
        REG_DEBUG("resolve_metallib '%s': JIT compile failed, using env fallback",
                  kernel_name.c_str());
        return fallback_metallib_path_from_env();
    }
    REG_DEBUG("resolve_metallib '%s': JIT compiled -> '%s' (persistent=%d)",
              kernel_name.c_str(), emitted_path.c_str(), static_cast<int>(is_persistent));

    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto found = s.modules.find(module_handle);
    if (found == s.modules.end()) {
        remove_path_if_exists(emitted_path);
        return fallback_metallib_path_from_env();
    }

    RegistrationModule& module = *found->second;
    if (!module.metallib_path.empty()) {
        if (!is_persistent) remove_path_if_exists(emitted_path);
        return module.metallib_path;
    }

    const auto inserted = module.emitted_kernel_metallibs.emplace(kernel_name, emitted_path);
    if (!inserted.second) {
        if (!is_persistent) remove_path_if_exists(emitted_path);
        if (out_printf_formats != nullptr) {
            const auto pf_it = module.emitted_kernel_printf_formats.find(kernel_name);
            if (pf_it != module.emitted_kernel_printf_formats.end()) {
                *out_printf_formats = pf_it->second;
            }
        }
        return inserted.first->second;
    }

    module.emitted_kernel_printf_formats.emplace(kernel_name, local_printf_formats);
    if (out_printf_formats != nullptr) {
        *out_printf_formats = std::move(local_printf_formats);
    }
    // Persistent cache files (in registration-jit/) survive process exit and
    // __cudaUnregisterFatBinary — do not track them for deletion.
    if (!is_persistent) {
        module.owned_metallibs.push_back(emitted_path);
    }
    return emitted_path;
}

std::vector<std::string> release_owned_metallibs_locked(RegistrationModule* module) {
    if (module == nullptr) {
        return {};
    }
    std::vector<std::string> owned = std::move(module->owned_metallibs);
    module->emitted_kernel_metallibs.clear();
    return owned;
}

void remove_owned_metallibs(const std::vector<std::string>& owned) {
    for (const std::string& path : owned) {
        remove_path_if_exists(path);
    }
}

thread_local std::vector<LaunchConfiguration> tls_launch_stack;

bool lookup_registered_kernel(const void* host_function, RegisteredKernel* out) {
    if (host_function == nullptr || out == nullptr) {
        return false;
    }

    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto found = s.kernels.find(host_function);
    if (found == s.kernels.end()) {
        return false;
    }

    out->metallib_path = found->second.metallib_path;
    out->kernel_name = found->second.kernel_name;
    out->arg_info = found->second.arg_info;
    out->printf_formats = found->second.printf_formats;
    return true;
}

bool lookup_registered_symbol(const void* host_symbol,
                              const void** out_device_symbol,
                              std::size_t* out_size) {
    if (host_symbol == nullptr || out_device_symbol == nullptr) {
        return false;
    }

    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto found = s.symbols.find(host_symbol);
    if (found == s.symbols.end() || found->second.device_address == nullptr) {
        return false;
    }

    *out_device_symbol = found->second.device_address;
    if (out_size != nullptr) {
        *out_size = found->second.size;
    }
    return true;
}

void clear() {
    std::vector<std::string> owned;
    RegistrationState& s = state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        for (auto& [handle, module] : s.modules) {
            (void)handle;
            if (module) {
                std::vector<std::string> module_owned =
                    release_owned_metallibs_locked(module.get());
                owned.insert(owned.end(),
                             std::make_move_iterator(module_owned.begin()),
                             std::make_move_iterator(module_owned.end()));
            }
        }
        s.kernels.clear();
        s.symbols.clear();
        s.modules.clear();
        tls_launch_stack.clear();
    }
    remove_owned_metallibs(owned);
}

}  // namespace cumetal::registration

extern "C" {

void** __cudaRegisterFatBinary(const void* fat_cubin) {
    REG_DEBUG("__cudaRegisterFatBinary fat_cubin=%p", fat_cubin);
    auto module = std::make_unique<cumetal::registration::RegistrationModule>();
    const cumetal::registration::ParsedFatbinImage parsed =
        cumetal::registration::parse_fatbin_image(fat_cubin);
    module->metallib_path = parsed.metallib_path;
    module->ptx_source = parsed.ptx_source;

    if (module->metallib_path.empty() && module->ptx_source.empty()) {
        module->metallib_path = cumetal::registration::fallback_metallib_path_from_env();
    }

    REG_DEBUG("__cudaRegisterFatBinary: metallib='%s' ptx_size=%zu",
              module->metallib_path.c_str(), module->ptx_source.size());

    void* handle = module.get();

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.modules.emplace(handle, std::move(module));
    REG_DEBUG("__cudaRegisterFatBinary: handle=%p", handle);
    return reinterpret_cast<void**>(handle);
}

void** __cudaRegisterFatBinary2(const void* fat_cubin, ...) {
    return __cudaRegisterFatBinary(fat_cubin);
}

void** __cudaRegisterFatBinary3(const void* fat_cubin, ...) {
    return __cudaRegisterFatBinary(fat_cubin);
}

void __cudaRegisterFatBinaryEnd(void** fat_cubin_handle) {
    (void)fat_cubin_handle;
}

void __cudaUnregisterFatBinary(void** fat_cubin_handle) {
    REG_DEBUG("__cudaUnregisterFatBinary handle=%p", fat_cubin_handle);
    if (fat_cubin_handle == nullptr) {
        return;
    }

    void* handle = reinterpret_cast<void*>(fat_cubin_handle);
    std::vector<std::string> owned;
    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);

        for (auto it = s.kernels.begin(); it != s.kernels.end();) {
            if (it->second.module_handle == handle) {
                it = s.kernels.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = s.symbols.begin(); it != s.symbols.end();) {
            if (it->second.module_handle == handle) {
                it = s.symbols.erase(it);
            } else {
                ++it;
            }
        }

        const auto module = s.modules.find(handle);
        if (module != s.modules.end() && module->second != nullptr) {
            owned = cumetal::registration::release_owned_metallibs_locked(module->second.get());
        }
        s.modules.erase(handle);
    }
    cumetal::registration::remove_owned_metallibs(owned);
}

void __cudaRegisterFunction(void** fat_cubin_handle,
                            const void* host_function,
                            char* device_function,
                            const char* device_name,
                            int thread_limit,
                            void* thread_id,
                            void* block_id,
                            void* block_dim,
                            void* grid_dim,
                            int* warp_size) {
    (void)thread_limit;
    (void)thread_id;
    (void)block_id;
    (void)block_dim;
    (void)grid_dim;
    (void)warp_size;

    if (host_function == nullptr) {
        return;
    }

    const char* chosen_name = device_name;
    if ((chosen_name == nullptr || chosen_name[0] == '\0') && device_function != nullptr &&
        device_function[0] != '\0') {
        chosen_name = device_function;
    }
    if (chosen_name == nullptr || chosen_name[0] == '\0') {
        return;
    }

    void* handle = fat_cubin_handle == nullptr ? nullptr : reinterpret_cast<void*>(fat_cubin_handle);

    std::vector<std::string> printf_formats;
    std::string metallib_path =
        cumetal::registration::resolve_metallib_path_for_kernel(handle, chosen_name, &printf_formats);
    if (metallib_path.empty()) {
        metallib_path = cumetal::registration::fallback_metallib_path_from_env();
    }

    std::string module_ptx_source;
    {
        cumetal::registration::RegistrationState& s = cumetal::registration::state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto module_it = s.modules.find(handle);
        if (module_it != s.modules.end() && module_it->second != nullptr) {
            module_ptx_source = module_it->second->ptx_source;
        }
    }
    std::vector<cumetalKernelArgInfo_t> inferred_arg_info =
        cumetal::registration::infer_arg_info_from_ptx_entry(module_ptx_source, chosen_name);

    REG_DEBUG("__cudaRegisterFunction: kernel='%s' metallib='%s' args=%zu",
              chosen_name, metallib_path.c_str(), inferred_arg_info.size());

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.kernels[host_function] = cumetal::registration::RegistrationRecord{
        .module_handle = handle,
        .metallib_path = std::move(metallib_path),
        .kernel_name = chosen_name,
        .arg_info = std::move(inferred_arg_info),
        .printf_formats = std::move(printf_formats),
    };
    REG_DEBUG("__cudaRegisterFunction: registered host_fn=%p", host_function);
}

void __cudaRegisterVar(void** fat_cubin_handle,
                       char* host_var,
                       char* device_address,
                       const char* device_name,
                       int ext,
                       std::size_t size,
                       int constant,
                       int global) {
    (void)fat_cubin_handle;
    (void)host_var;
    (void)device_address;
    (void)device_name;
    (void)ext;
    (void)constant;
    (void)global;

    if (host_var == nullptr) {
        return;
    }

    const void* mapped = device_address == nullptr ? static_cast<const void*>(host_var)
                                                   : static_cast<const void*>(device_address);
    void* handle = fat_cubin_handle == nullptr ? nullptr : reinterpret_cast<void*>(fat_cubin_handle);

    REG_DEBUG("__cudaRegisterVar: name='%s' host_var=%p mapped=%p size=%zu",
              device_name != nullptr ? device_name : "(null)", static_cast<void*>(host_var),
              mapped, size);

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.symbols[host_var] = cumetal::registration::RegistrationSymbolRecord{
        .module_handle = handle,
        .device_address = mapped,
        .size = size,
    };
}

void __cudaRegisterManagedVar(void** fat_cubin_handle,
                              void** host_var_ptr_address,
                              char* device_address,
                              const char* device_name,
                              int ext,
                              std::size_t size,
                              int constant,
                              int global) {
    char* host_var = nullptr;
    if (host_var_ptr_address != nullptr) {
        host_var = static_cast<char*>(*host_var_ptr_address);
    }

    __cudaRegisterVar(fat_cubin_handle,
                      host_var,
                      device_address,
                      device_name,
                      ext,
                      size,
                      constant,
                      global);
}

cudaError_t __cudaPushCallConfiguration(dim3 grid_dim,
                                        dim3 block_dim,
                                        std::size_t shared_mem,
                                        cudaStream_t stream) {
    cumetal::registration::tls_launch_stack.push_back(cumetal::registration::LaunchConfiguration{
        .grid_dim = grid_dim,
        .block_dim = block_dim,
        .shared_mem = shared_mem,
        .stream = stream,
    });
    return cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3* grid_dim,
                                       dim3* block_dim,
                                       std::size_t* shared_mem,
                                       void** stream) {
    if (cumetal::registration::tls_launch_stack.empty()) {
        return cudaErrorInvalidValue;
    }

    const cumetal::registration::LaunchConfiguration config =
        cumetal::registration::tls_launch_stack.back();
    cumetal::registration::tls_launch_stack.pop_back();

    if (grid_dim != nullptr) {
        *grid_dim = config.grid_dim;
    }
    if (block_dim != nullptr) {
        *block_dim = config.block_dim;
    }
    if (shared_mem != nullptr) {
        *shared_mem = config.shared_mem;
    }
    if (stream != nullptr) {
        *stream = reinterpret_cast<void*>(config.stream);
    }

    return cudaSuccess;
}

}  // extern "C"
