#include "registration.h"

#include "cumetal/air_emitter/emitter.h"
#include "cumetal/common/metallib.h"
#include "cumetal_diag.h"
#include "cumetal/ptx/lower_to_metal.h"
#include "cumetal/ptx/lower_to_llvm.h"
#include "cumetal/ptx/parser.h"
#include "fatbin_elf.h"
#include "fatbin_ptx.h"
#include "metal_math_mode.h"

#include <dlfcn.h>
#include <mach-o/loader.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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
// where hash = FNV-1a-64 over
// (cache_schema + '\0' + libcumetal LC_UUID + '\0' + policy + '\0' + ptx_source + '\0' +
//  kernel_name).
// This avoids recompiling the same kernel across process restarts. The binary UUID is what keeps
// an entry from outliving the compiler that produced it -- see cumetal_binary_uuid() below.
constexpr std::string_view kRegistrationJitCacheSchema =
    "cumetal-registration-jit-v13-grid-y-chunk-offset";

constexpr std::string_view kRegistrationMetadataMagic = "CUMETA01";
constexpr std::size_t kMaxRegistrationMetadataBytes = 4u * 1024u * 1024u;
constexpr std::uint32_t kMaxRegistrationPrintfFormats = 4096u;

struct RegistrationCacheMetadata {
    std::vector<std::string> printf_formats;
    bool uses_device_heap = false;
    bool uses_device_launch_queue = false;
};

// Identity of the libcumetal binary currently executing, taken from its Mach-O LC_UUID.
//
// The cache key used to be (hand-maintained schema string + policy + PTX + kernel name). Nothing
// in that tuple describes *how this build lowers PTX*, so any change to the lowering -- an MSL
// template, an instruction handler, a legalization rule -- produced the same key and the runtime
// silently reused a metallib compiled by the previous compiler. Correctness depended on a human
// remembering to bump the magic string on every lowering change.
//
// That is not hypothetical. Editing the fused_classifier_kernel3 MSL template and re-running
// llm.c left the *old* kernel in the cache and executing; the edit had no effect. Worse, a cache
// populated across several builds holds kernels from different compiler versions at once, which
// is what made the llm.c parity gate fail a few runs in fifteen with a varying step and an
// occasional -inf loss. It also crosses build trees: a worktree at an older commit shares this
// cache and will happily consume entries a newer build wrote.
//
// The linker regenerates LC_UUID whenever the binary's contents change, so keying on it ties every
// cache entry to the exact build that produced it. A developer's rebuild invalidates the cache
// automatically; a user who never rebuilds keeps full cache reuse across runs. Read once via the
// Mach-O load commands -- no hashing of a multi-megabyte dylib.
const std::string& cumetal_binary_uuid() {
    static const std::string uuid = [] {
        Dl_info info{};
        // Any symbol in this library resolves to the image containing it.
        if (dladdr(reinterpret_cast<const void*>(&kRegistrationJitCacheSchema), &info) == 0 ||
            info.dli_fbase == nullptr) {
            return std::string("unknown-image");
        }

        const auto* header = static_cast<const struct mach_header_64*>(info.dli_fbase);
        if (header->magic != MH_MAGIC_64) {
            return std::string("unknown-macho");
        }

        const auto* command = reinterpret_cast<const struct load_command*>(header + 1);
        for (std::uint32_t i = 0; i < header->ncmds; ++i) {
            if (command->cmd == LC_UUID) {
                const auto* uuid_cmd = reinterpret_cast<const struct uuid_command*>(command);
                std::ostringstream out;
                out << std::hex << std::setfill('0');
                for (unsigned char byte : uuid_cmd->uuid) {
                    out << std::setw(2) << static_cast<unsigned int>(byte);
                }
                return out.str();
            }
            command = reinterpret_cast<const struct load_command*>(
                reinterpret_cast<const std::uint8_t*>(command) + command->cmdsize);
        }
        // No LC_UUID (unusual). Fall back to a value that disables cross-run reuse rather than
        // one that silently shares entries between builds.
        return std::string("no-uuid");
    }();
    return uuid;
}

std::string registration_lowering_policy() {
    const char* backend = std::getenv("CUMETAL_PTX_BACKEND");
    std::string policy = "frontend=ptx;backend=";
    policy += backend != nullptr && backend[0] != '\0' ? backend : "legacy";
    policy += ";fp64=";
    policy += cumetal::ptx::fp64_mode_name(cumetal::ptx::fp64_mode_from_env());
    policy += ";vf64_support_sha256=";
    policy += CUMETAL_VF64_SUPPORT_SHA256;
    policy += ";workload_specializations=";
    policy += cumetal::diag_env_truthy("CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS")
                  ? "enabled"
                  : "disabled";
    policy += ";msl_math=";
    policy += cumetal::metal_math_mode_name(
        cumetal::current_metal_math_mode());
    policy += ";ir_schema=1;metal_legalization=1;msl=3.1";
    return policy;
}

constexpr std::uint64_t kFnv1a64Offset = 1469598103934665603ull;

std::uint64_t fnv1a64_registration_update(std::uint64_t hash,
                                          const std::uint8_t* bytes,
                                          std::size_t size) {
    constexpr std::uint64_t kPrime  = 1099511628211ull;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= kPrime;
    }
    return hash;
}

std::uint64_t stable_device_kernel_token(std::string_view symbol) {
    std::uint64_t hash = fnv1a64_registration_update(
        kFnv1a64Offset,
        reinterpret_cast<const std::uint8_t*>(symbol.data()), symbol.size());
    hash &= 0x7fffffffffffffffull;
    return hash == 0 ? 1 : hash;
}

std::uint64_t jit_cache_prefix_hash(const std::string& ptx_source) {
    // Hash incrementally instead of materializing another full-sized PTX blob.
    // GGML modules can exceed 10 MiB, and several kernels from the same module
    // are resolved during one-layer llama.cpp inference.
    const std::string policy = registration_lowering_policy();
    std::uint64_t hash = kFnv1a64Offset;
    const auto append = [&hash](std::string_view bytes) {
        hash = fnv1a64_registration_update(
            hash, reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    };
    append(kRegistrationJitCacheSchema);
    append(std::string_view("\0", 1));
    // Ties the entry to the exact libcumetal build that lowered it; see cumetal_binary_uuid().
    append(cumetal_binary_uuid());
    append(std::string_view("\0", 1));
    append(policy);
    append(std::string_view("\0", 1));
    append(ptx_source);
    return hash;
}

std::string jit_cache_key(std::uint64_t prefix_hash, const std::string& kernel_name) {
    std::uint64_t hash = fnv1a64_registration_update(
        prefix_hash, reinterpret_cast<const std::uint8_t*>("\0"), 1);
    hash = fnv1a64_registration_update(
        hash,
        reinterpret_cast<const std::uint8_t*>(kernel_name.data()),
        kernel_name.size());
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return oss.str();
}

std::string sanitize_filename_component(std::string s) {
    for (char& c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        if (!ok) {
            c = '_';
        }
    }
    if (s.empty()) {
        s = "kernel";
    }
    return s;
}

void maybe_dump_ptx_for_llvm_debug(const std::string& kernel_name, const std::string& ptx_source) {
    const char* dir_env = std::getenv("CUMETAL_DEBUG_DUMP_PTX_DIR");
    if (dir_env == nullptr || dir_env[0] == '\0') {
        return;
    }

    std::error_code ec;
    const std::filesystem::path dir(dir_env);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        return;
    }

    const std::filesystem::path out = dir / (sanitize_filename_component(kernel_name) + ".ptx");
    if (std::filesystem::exists(out, ec) && !ec) {
        return;
    }

    std::string io_error;
    const std::vector<std::uint8_t> bytes(ptx_source.begin(), ptx_source.end());
    (void) cumetal::common::write_file_bytes(out, bytes, &io_error);
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
std::filesystem::path jit_cache_path_for(std::uint64_t prefix_hash,
                                          const std::string& kernel_name) {
    const std::filesystem::path root = jit_cache_root();
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        return {};
    }
    return root / (jit_cache_key(prefix_hash, kernel_name) + ".metallib");
}

std::filesystem::path jit_metadata_path_for(const std::filesystem::path& artifact_path) {
    std::filesystem::path metadata_path = artifact_path;
    metadata_path.replace_extension(".metadata");
    return metadata_path;
}

void append_u32_le(std::vector<std::uint8_t>* bytes, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes->push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }
}

bool consume_u32_le(const std::vector<std::uint8_t>& bytes,
                    std::size_t* offset,
                    std::uint32_t* value) {
    if (offset == nullptr || value == nullptr || *offset > bytes.size() ||
        bytes.size() - *offset < 4) {
        return false;
    }
    *value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        *value |= static_cast<std::uint32_t>(bytes[(*offset)++]) << shift;
    }
    return true;
}

bool write_registration_metadata(const std::filesystem::path& artifact_path,
                                 const RegistrationCacheMetadata& metadata) {
    if (artifact_path.empty() ||
        metadata.printf_formats.size() > kMaxRegistrationPrintfFormats) {
        return false;
    }

    std::vector<std::uint8_t> bytes(kRegistrationMetadataMagic.begin(),
                                    kRegistrationMetadataMagic.end());
    std::uint32_t flags = 0;
    if (metadata.uses_device_heap) flags |= 1u;
    if (metadata.uses_device_launch_queue) flags |= 2u;
    append_u32_le(&bytes, flags);
    append_u32_le(&bytes, static_cast<std::uint32_t>(metadata.printf_formats.size()));
    for (const std::string& format : metadata.printf_formats) {
        if (format.size() > std::numeric_limits<std::uint32_t>::max() ||
            bytes.size() + 4 + format.size() > kMaxRegistrationMetadataBytes) {
            return false;
        }
        append_u32_le(&bytes, static_cast<std::uint32_t>(format.size()));
        bytes.insert(bytes.end(), format.begin(), format.end());
    }

    const std::filesystem::path destination = jit_metadata_path_for(artifact_path);
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path temporary = destination;
    temporary += ".tmp-" + std::to_string(::getpid()) + "-" + std::to_string(stamp);
    std::string error;
    if (!cumetal::common::write_file_bytes(temporary, bytes, &error)) {
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(temporary, destination, ec);
    if (ec) {
        std::filesystem::remove(temporary, ec);
        return false;
    }
    return true;
}

bool read_registration_metadata(const std::filesystem::path& artifact_path,
                                RegistrationCacheMetadata* metadata) {
    if (artifact_path.empty() || metadata == nullptr) return false;
    std::string error;
    const std::vector<std::uint8_t> bytes = cumetal::common::read_file_bytes(
        jit_metadata_path_for(artifact_path), &error);
    if (bytes.size() < kRegistrationMetadataMagic.size() + 8 ||
        bytes.size() > kMaxRegistrationMetadataBytes ||
        !std::equal(kRegistrationMetadataMagic.begin(),
                    kRegistrationMetadataMagic.end(), bytes.begin())) {
        return false;
    }

    std::size_t offset = kRegistrationMetadataMagic.size();
    std::uint32_t flags = 0;
    std::uint32_t count = 0;
    if (!consume_u32_le(bytes, &offset, &flags) ||
        !consume_u32_le(bytes, &offset, &count) ||
        (flags & ~3u) != 0 || count > kMaxRegistrationPrintfFormats) {
        return false;
    }

    RegistrationCacheMetadata parsed;
    parsed.uses_device_heap = (flags & 1u) != 0;
    parsed.uses_device_launch_queue = (flags & 2u) != 0;
    parsed.printf_formats.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t size = 0;
        if (!consume_u32_le(bytes, &offset, &size) || offset > bytes.size() ||
            size > bytes.size() - offset) {
            return false;
        }
        parsed.printf_formats.emplace_back(
            reinterpret_cast<const char*>(bytes.data() + offset), size);
        offset += size;
    }
    if (offset != bytes.size()) return false;
    *metadata = std::move(parsed);
    return true;
}

}  // namespace

#define REG_DEBUG(fmt, ...)                                                       \
    do {                                                                          \
        if (is_debug_registration()) {                                            \
            std::fprintf(stderr, "[cumetal-reg] " fmt "\n"                    \
                                 __VA_OPT__(,) __VA_ARGS__);                     \
        }                                                                         \
    } while (0)

namespace cumetal::registration {

constexpr std::uint32_t kCumetalFatbinMagic = 0x4C544D43u;  // "CMTL"
constexpr std::uint32_t kCumetalFatbinVersion = 1u;
constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
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

struct ParsedFatbinImage {
    std::string metallib_path;
    std::string ptx_source;
    bool allow_environment_fallback = true;
};

struct RegistrationModule {
    std::string metallib_path;
    std::shared_ptr<const std::string> ptx_source;
    bool allow_environment_fallback = true;
    std::optional<std::uint64_t> jit_cache_prefix_hash;
    std::unordered_map<std::string, std::string> emitted_kernel_metallibs;
    std::unordered_map<std::string, std::vector<std::string>> emitted_kernel_printf_formats;
    std::unordered_map<std::string, bool> emitted_kernel_uses_device_heap;
    std::unordered_map<std::string, bool> emitted_kernel_uses_device_launch_queue;
    std::unordered_map<std::string, std::size_t> emitted_kernel_static_shared_bytes;
    // How each JIT-emitted kernel actually reached the GPU. Without this the
    // PTX->LLVM->AIR->metallib path is indistinguishable from a user-supplied
    // prebuilt binary, because a loaded .metallib carries no lowering marker
    // the way direct-MSL output does (its `// cumetal-provenance:` comment).
    std::unordered_map<std::string, std::string> emitted_kernel_provenance;
    std::unordered_map<std::uint64_t, std::unique_ptr<unsigned char>>
        device_kernel_aliases;
    // CUDA emits no __cudaRegisterVar record for translation-unit-private
    // initialized `.global` definitions. These buffers are therefore owned by
    // the module itself and shared by every kernel that references the symbol.
    std::unordered_map<std::string,
                       std::shared_ptr<cumetal::metal_backend::Buffer>>
        private_global_buffers;
    std::vector<std::string> owned_metallibs;
};

struct RegistrationRecord {
    void* module_handle = nullptr;
    std::string metallib_path;
    std::string kernel_name;
    std::vector<cumetalKernelArgInfo_t> arg_info;
    std::vector<cumetal::ptx::ExternalConstantSymbol> external_constant_symbols;
    std::size_t external_constant_buffer_size = 0;
    std::vector<cumetal::ptx::ExternalGlobalSymbol> external_global_symbols;
    bool arg_info_resolved = false;
    std::vector<std::string> printf_formats;
    bool uses_device_heap = false;
    bool uses_device_launch_queue = false;
    std::size_t static_shared_bytes = 0;
    std::string provenance;
};

struct RegistrationSymbolRecord {
    void* module_handle = nullptr;
    const void* device_address = nullptr;
    std::string device_name;
    std::size_t size = 0;
    bool constant = false;
    bool requires_metal_storage = false;
    std::shared_ptr<cumetal::metal_backend::Buffer> global_buffer;
};

struct RegistrationState {
    std::mutex mutex;
    std::unordered_map<void*, std::unique_ptr<RegistrationModule>> modules;
    std::unordered_map<const void*, RegistrationRecord> kernels;
    std::unordered_map<const void*, RegistrationSymbolRecord> symbols;
};

RegistrationState& state() {
    // Immortal on purpose. This is process-lifetime state guarded by a mutex, and a
    // function-local static gets an atexit destructor: anything that touches it during
    // teardown -- another static's destructor, a detached worker, a Metal completion
    // handler -- then locks a destroyed mutex. That surfaced as an intermittent
    // "mutex lock failed: Invalid argument" abort *after* a test had already printed
    // PASS. Leaking one object at exit is the fix; the OS reclaims it.
    static RegistrationState* s = new RegistrationState();
    return *s;
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

bool parse_fatbin_wrapper_ptx(const void* fat_cubin,
                              std::string* out_ptx,
                              bool* recognized_invalid) {
    if (fat_cubin == nullptr || out_ptx == nullptr) {
        return false;
    }
    if (recognized_invalid != nullptr) {
        *recognized_invalid = false;
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
        const cumetal::fatbin::PtxExtractStatus fatbin_status =
            cumetal::fatbin::extract_fatbin_ptx(
                data, kMaxFatbinImageBytes, out_ptx);
        if (fatbin_status == cumetal::fatbin::PtxExtractStatus::kFound) {
            return true;
        }
        if (fatbin_status != cumetal::fatbin::PtxExtractStatus::kNotFatbin) {
            if (recognized_invalid != nullptr) *recognized_invalid = true;
            return false;
        }
        const cumetal::fatbin::ElfPtxStatus elf_status =
            cumetal::fatbin::extract_elf_ptx(
                data, kMaxFatbinImageBytes, out_ptx);
        if (elf_status == cumetal::fatbin::ElfPtxStatus::kFound) {
            return true;
        }
        if (elf_status != cumetal::fatbin::ElfPtxStatus::kNotElf &&
            recognized_invalid != nullptr) {
            *recognized_invalid = true;
            return false;
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

    bool wrapper_recognized_invalid = false;
    if (parse_fatbin_wrapper_ptx(
            fat_cubin, &parsed.ptx_source, &wrapper_recognized_invalid)) {
        REG_DEBUG("parse_fatbin_image: fatbin wrapper format, ptx_size=%zu",
                  parsed.ptx_source.size());
        return parsed;
    }
    if (wrapper_recognized_invalid) {
        parsed.allow_environment_fallback = false;
        REG_DEBUG("%s", "parse_fatbin_image: malformed/unsupported ELF in fatbin wrapper");
        return parsed;
    }
    const cumetal::fatbin::PtxExtractStatus fatbin_status =
        cumetal::fatbin::extract_fatbin_ptx(
            fat_cubin, kMaxFatbinImageBytes, &parsed.ptx_source);
    if (fatbin_status == cumetal::fatbin::PtxExtractStatus::kFound) {
        REG_DEBUG("parse_fatbin_image: fatbin blob format, ptx_size=%zu",
                  parsed.ptx_source.size());
        return parsed;
    }
    if (fatbin_status != cumetal::fatbin::PtxExtractStatus::kNotFatbin) {
        parsed.allow_environment_fallback = false;
        REG_DEBUG("%s", "parse_fatbin_image: non-PTX/malformed/unsupported fatbin refused");
        return parsed;
    }
    if (parse_direct_ptx_image(fat_cubin, &parsed.ptx_source)) {
        REG_DEBUG("parse_fatbin_image: direct PTX image, ptx_size=%zu",
                  parsed.ptx_source.size());
        return parsed;
    }
    // NVCC-generated ELF objects carry PTX in named sections. Parse only the
    // ELF section-table ranges instead of scanning an arbitrary memory window.
    const cumetal::fatbin::ElfPtxStatus elf_status =
        cumetal::fatbin::extract_elf_ptx(
            fat_cubin, kMaxFatbinImageBytes, &parsed.ptx_source);
    if (elf_status == cumetal::fatbin::ElfPtxStatus::kFound) {
        REG_DEBUG("parse_fatbin_image: ELF-embedded PTX, ptx_size=%zu",
                  parsed.ptx_source.size());
        return parsed;
    }
    if (elf_status != cumetal::fatbin::ElfPtxStatus::kNotElf) {
        parsed.allow_environment_fallback = false;
        REG_DEBUG("%s", "parse_fatbin_image: malformed/unsupported ELF refused");
        return parsed;
    }

    REG_DEBUG("parse_fatbin_image: unrecognized format, using env fallback");
    parsed.metallib_path = fallback_metallib_path_from_env();
    return parsed;
}

// Extract the array element count from a PTX param name like "foo[12]" → 12.
// Returns 0 if the name has no array suffix.
std::uint32_t parse_param_array_bytes(std::string_view name) {
    const std::size_t open = name.rfind('[');
    const std::size_t close = name.rfind(']');
    if (open == std::string_view::npos || close == std::string_view::npos || close <= open + 1) {
        return 0u;
    }
    std::uint32_t value = 0;
    for (std::size_t i = open + 1; i < close; ++i) {
        const unsigned char c = static_cast<unsigned char>(name[i]);
        if (c < '0' || c > '9') {
            return 0u;
        }
        value = value * 10u + static_cast<std::uint32_t>(c - '0');
    }
    return value;
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

bool is_ptx_token_boundary(char c) {
    const unsigned char value = static_cast<unsigned char>(c);
    return std::isspace(value) || c == ',' || c == '(' || c == ')' || c == '{' || c == '}' ||
           c == ';';
}

void skip_ptx_trivia(std::string_view source, std::size_t* cursor) {
    while (*cursor < source.size()) {
        const unsigned char c = static_cast<unsigned char>(source[*cursor]);
        if (std::isspace(c)) {
            ++*cursor;
            continue;
        }
        if (source[*cursor] == '/' && *cursor + 1 < source.size()) {
            if (source[*cursor + 1] == '/') {
                *cursor += 2;
                while (*cursor < source.size() && source[*cursor] != '\n') ++*cursor;
                continue;
            }
            if (source[*cursor + 1] == '*') {
                *cursor += 2;
                while (*cursor + 1 < source.size() &&
                       !(source[*cursor] == '*' && source[*cursor + 1] == '/')) {
                    ++*cursor;
                }
                if (*cursor + 1 < source.size()) *cursor += 2;
                continue;
            }
        }
        break;
    }
}

void skip_ptx_string(std::string_view source, std::size_t* cursor) {
    if (*cursor >= source.size() || source[*cursor] != '"') return;
    ++*cursor;
    while (*cursor < source.size()) {
        if (source[*cursor] == '\\' && *cursor + 1 < source.size()) {
            *cursor += 2;
        } else if (source[(*cursor)++] == '"') {
            return;
        }
    }
}

bool ptx_offset_is_code(std::string_view source, std::size_t target) {
    std::size_t cursor = 0;
    while (cursor < target && cursor < source.size()) {
        if (source[cursor] == '"') {
            skip_ptx_string(source, &cursor);
            continue;
        }
        if (source[cursor] == '/' && cursor + 1 < source.size()) {
            if (source[cursor + 1] == '/') {
                cursor += 2;
                while (cursor < source.size() && source[cursor] != '\n') ++cursor;
                continue;
            }
            if (source[cursor + 1] == '*') {
                cursor += 2;
                while (cursor + 1 < source.size() &&
                       !(source[cursor] == '*' && source[cursor + 1] == '/')) {
                    ++cursor;
                }
                cursor = std::min(cursor + 2, source.size());
                continue;
            }
        }
        ++cursor;
    }
    return cursor == target;
}

std::string_view next_ptx_token(std::string_view source, std::size_t* cursor) {
    skip_ptx_trivia(source, cursor);
    const std::size_t begin = *cursor;
    while (*cursor < source.size() && !is_ptx_token_boundary(source[*cursor])) {
        if (source[*cursor] == '/' && *cursor + 1 < source.size() &&
            (source[*cursor + 1] == '/' || source[*cursor + 1] == '*')) {
            break;
        }
        ++*cursor;
    }
    return source.substr(begin, *cursor - begin);
}

std::vector<cumetalKernelArgInfo_t> scan_ptx_entry_params(std::string_view params) {
    std::vector<cumetalKernelArgInfo_t> result;
    std::size_t segment_begin = 0;
    while (segment_begin < params.size()) {
        std::size_t segment_end = segment_begin;
        while (segment_end < params.size() && params[segment_end] != ',') ++segment_end;
        const std::string_view segment = params.substr(segment_begin, segment_end - segment_begin);
        bool is_param = false;
        bool has_pointer_qualifier = false;
        std::string_view type;
        std::size_t cursor = 0;
        while (cursor < segment.size()) {
            const std::string_view token = next_ptx_token(segment, &cursor);
            if (token.empty()) {
                if (cursor < segment.size()) ++cursor;
                continue;
            }
            if (token == ".param") is_param = true;
            if (token == ".ptr") has_pointer_qualifier = true;
            // Match the parser's tolerant type rule: the first dotted token
            // other than a declaration qualifier is the parameter type.
            if (type.empty() && token.size() > 1 && token.front() == '.' &&
                token != ".param" && token != ".ptr" && token != ".align") {
                type = token;
            }
        }
        if (is_param && !type.empty()) {
            // NVCC commonly emits kernel pointers as unqualified .u64/.s64/.b64
            // parameters. The full PTX parser refines ambiguous 64-bit values from
            // body data flow, but deliberately defaults them to pointers when it
            // cannot prove scalar use. Keep that compatibility rule here. At launch,
            // allocation lookup safely reclassifies an over-inferred small scalar.
            const bool is_unannotated_pointer_width =
                type == ".u64" || type == ".s64" || type == ".b64";
            const bool is_pointer = has_pointer_qualifier || is_unannotated_pointer_width;
            cumetalKernelArgInfo_t info{};
            info.kind = is_pointer ? CUMETAL_ARG_BUFFER : CUMETAL_ARG_BYTES;
            const std::uint32_t array_bytes = parse_param_array_bytes(segment);
            info.size_bytes = is_pointer
                ? static_cast<std::uint32_t>(sizeof(void*))
                : (array_bytes > 0 ? array_bytes : scalar_size_bytes_for_ptx_type(type));
            result.push_back(info);
        }
        segment_begin = segment_end + (segment_end < params.size() ? 1u : 0u);
    }
    return result;
}

std::unordered_map<std::string, std::vector<cumetalKernelArgInfo_t>>
build_arg_info_index_from_ptx(const std::string& ptx_source) {
    std::unordered_map<std::string, std::vector<cumetalKernelArgInfo_t>> out;
    if (ptx_source.empty()) {
        return out;
    }

    const std::string_view source(ptx_source);
    std::size_t cursor = 0;
    while (cursor < source.size()) {
        skip_ptx_trivia(source, &cursor);
        if (cursor >= source.size()) break;
        if (source[cursor] == '"') {
            skip_ptx_string(source, &cursor);
            continue;
        }
        constexpr std::string_view kEntry = ".entry";
        const bool boundary_before = cursor == 0 || is_ptx_token_boundary(source[cursor - 1]);
        const bool matches = boundary_before && source.substr(cursor, kEntry.size()) == kEntry;
        const std::size_t after_entry = cursor + kEntry.size();
        const bool boundary_after = after_entry >= source.size() ||
                                    is_ptx_token_boundary(source[after_entry]);
        if (!matches || !boundary_after) {
            ++cursor;
            continue;
        }

        cursor = after_entry;
        const std::string_view name = next_ptx_token(source, &cursor);
        if (name.empty()) continue;
        skip_ptx_trivia(source, &cursor);
        while (cursor < source.size() && source[cursor] != '(' && source[cursor] != '{') {
            if (source[cursor] == '"') skip_ptx_string(source, &cursor);
            else ++cursor;
            skip_ptx_trivia(source, &cursor);
        }
        if (cursor >= source.size() || source[cursor] != '(') continue;
        const std::size_t params_begin = ++cursor;
        std::size_t depth = 1;
        while (cursor < source.size() && depth > 0) {
            if (source[cursor] == '"') {
                skip_ptx_string(source, &cursor);
                continue;
            }
            if (source[cursor] == '/' && cursor + 1 < source.size() &&
                (source[cursor + 1] == '/' || source[cursor + 1] == '*')) {
                skip_ptx_trivia(source, &cursor);
                continue;
            }
            if (source[cursor] == '(') ++depth;
            if (source[cursor] == ')') --depth;
            ++cursor;
        }
        if (depth != 0) break;
        const std::size_t params_end = cursor - 1;
        out.emplace(std::string(name),
                    scan_ptx_entry_params(source.substr(params_begin, params_end - params_begin)));
    }

    return out;
}

bool find_arg_info_for_ptx_entry(const std::string& ptx_source,
                                 std::string_view entry_name,
                                 std::vector<cumetalKernelArgInfo_t>* out) {
    if (ptx_source.empty() || entry_name.empty() || out == nullptr) {
        return false;
    }

    const std::string_view source(ptx_source);
    constexpr std::string_view kEntry = ".entry";
    std::size_t cursor = 0;
    while ((cursor = source.find(entry_name, cursor)) != std::string_view::npos) {
        const std::size_t name_end = cursor + entry_name.size();
        const bool boundary_before = cursor == 0 || is_ptx_token_boundary(source[cursor - 1]);
        const bool boundary_after = name_end >= source.size() ||
                                    is_ptx_token_boundary(source[name_end]);
        if (!boundary_before || !boundary_after || !ptx_offset_is_code(source, cursor)) {
            cursor = name_end;
            continue;
        }

        const std::size_t entry_pos = source.rfind(kEntry, cursor);
        if (entry_pos == std::string_view::npos ||
            !ptx_offset_is_code(source, entry_pos)) {
            cursor = name_end;
            continue;
        }

        std::size_t declaration_cursor = entry_pos + kEntry.size();
        const std::string_view name = next_ptx_token(source, &declaration_cursor);
        if (name != entry_name) {
            cursor = name_end;
            continue;
        }
        skip_ptx_trivia(source, &declaration_cursor);
        while (declaration_cursor < source.size() &&
               source[declaration_cursor] != '(' && source[declaration_cursor] != '{') {
            if (source[declaration_cursor] == '"') {
                skip_ptx_string(source, &declaration_cursor);
            } else {
                ++declaration_cursor;
            }
            skip_ptx_trivia(source, &declaration_cursor);
        }
        if (declaration_cursor >= source.size() || source[declaration_cursor] != '(') {
            return false;
        }

        const std::size_t params_begin = ++declaration_cursor;
        std::size_t depth = 1;
        while (declaration_cursor < source.size() && depth > 0) {
            if (source[declaration_cursor] == '"') {
                skip_ptx_string(source, &declaration_cursor);
                continue;
            }
            if (source[declaration_cursor] == '/' && declaration_cursor + 1 < source.size() &&
                (source[declaration_cursor + 1] == '/' ||
                 source[declaration_cursor + 1] == '*')) {
                skip_ptx_trivia(source, &declaration_cursor);
                continue;
            }
            if (source[declaration_cursor] == '(') ++depth;
            if (source[declaration_cursor] == ')') --depth;
            ++declaration_cursor;
        }
        if (depth != 0) {
            return false;
        }
        const std::size_t params_end = declaration_cursor - 1;
        *out = scan_ptx_entry_params(
            source.substr(params_begin, params_end - params_begin));
        return true;
    }
    return false;
}

// out_is_persistent: set to true if the output lives in the persistent JIT cache
// (and therefore must NOT be deleted on __cudaUnregisterFatBinary).
bool emit_ptx_entry_to_temp_metallib(const std::string& ptx_source,
                                     const std::string& kernel_name,
                                     std::uint64_t cache_prefix_hash,
                                     std::string* out_path,
                                     std::vector<std::string>* out_printf_formats = nullptr,
                                     bool* out_uses_device_heap = nullptr,
                                     bool* out_uses_device_launch_queue = nullptr,
                                     bool* out_is_persistent = nullptr,
                                     std::string* out_provenance = nullptr) {
    if (ptx_source.empty() || kernel_name.empty() || out_path == nullptr) {
        REG_DEBUG("emit_ptx_entry_to_temp_metallib: invalid argument (ptx=%zu, kernel=%s)",
                  ptx_source.size(), kernel_name.c_str());
        return false;
    }

    // Whether this kernel's FP64 arithmetic runs on the FP32-pair emulation.
    // Computed from the PTX rather than from the lowering result so a warm JIT
    // cache reports the same provenance as a cold one. Only the LLVM path can
    // lower FP64 (the direct-MSL lowering declines it), so this is the single
    // place that needs to know.
    const auto fp64_mode = cumetal::ptx::fp64_mode_from_env();
    const bool uses_fp64 = ptx_source.find(".f64") != std::string::npos;
    const char* generic_ptx_provenance = "generic_ptx_lowering";
    if (uses_fp64 && fp64_mode == cumetal::ptx::Fp64Mode::kEmulate) {
        generic_ptx_provenance = "generic_ptx_lowering_fp64_emulated";
    } else if (uses_fp64 && fp64_mode == cumetal::ptx::Fp64Mode::kWide48) {
        generic_ptx_provenance = "generic_ptx_lowering_fp64_wide48";
    } else if (uses_fp64 && fp64_mode == cumetal::ptx::Fp64Mode::kIEEE64) {
        generic_ptx_provenance = "generic_ptx_lowering_fp64_ieee64";
    }

    // Multiple host threads may launch the same newly registered kernel at
    // once. Cache lookup and publication must be one transaction: otherwise
    // they all observe a miss and concurrently overwrite the same .metal or
    // .metallib path, so some launches compile a partial artifact and silently
    // do no work. JIT is a cold-path operation; serializing it globally favors
    // correctness and still allows all warm in-process launches to bypass this
    // function through emitted_kernel_metallibs.
    static std::mutex jit_emit_mutex;
    std::lock_guard<std::mutex> jit_emit_lock(jit_emit_mutex);

    REG_DEBUG("emit kernel '%s' ptx_size=%zu", kernel_name.c_str(), ptx_source.size());

    const auto publish_metadata = [&](const RegistrationCacheMetadata& metadata) {
        if (out_printf_formats != nullptr) {
            *out_printf_formats = metadata.printf_formats;
        }
        if (out_uses_device_heap != nullptr) {
            *out_uses_device_heap = metadata.uses_device_heap;
        }
        if (out_uses_device_launch_queue != nullptr) {
            *out_uses_device_launch_queue = metadata.uses_device_launch_queue;
        }
    };

    // Check the persistent artifact and its metadata sidecar before invoking
    // either PTX lowering pipeline. A warm process must not repeat compiler
    // work merely to reconstruct host-side launch metadata.
    const std::filesystem::path cached_metallib =
        jit_cache_path_for(cache_prefix_hash, kernel_name);
    std::filesystem::path cached_hit_path;
    if (!cached_metallib.empty()) {
        std::error_code ec;
        if (std::filesystem::exists(cached_metallib, ec) && !ec) {
            // Check for stale unrunnable experimental containers from before we started
            // refusing to cache them. Treat as miss so we get a clean failure path.
            bool bad_experimental = false;
            {
                auto f = std::fopen(cached_metallib.string().c_str(), "rb");
                if (f) {
                    char buf[256] = {0};
                    size_t n = std::fread(buf, 1, sizeof(buf)-1, f);
                    std::fclose(f);
                    if (n > 0 && memmem(buf, n, "cumetal-experimental", 20) != nullptr) {
                        bad_experimental = true;
                    }
                }
            }
            if (bad_experimental) {
                REG_DEBUG("jit cache hit but contains experimental container (unusable); removing and treating as miss: %s",
                          cached_metallib.c_str());
                std::filesystem::remove(cached_metallib, ec);
                std::filesystem::remove(jit_metadata_path_for(cached_metallib), ec);
            } else {
                cached_hit_path = cached_metallib;
            }
        }
        if (cached_hit_path.empty()) {
            // Support direct MSL source caches (written as <hash>.metal for
            // runtime newLibraryWithSource).
            auto msl_path = cached_metallib;
            msl_path.replace_extension(".metal");
            if (std::filesystem::exists(msl_path, ec) && !ec) {
                cached_hit_path = msl_path;
            }
        }
        if (!cached_hit_path.empty()) {
            RegistrationCacheMetadata cached_metadata;
            if (read_registration_metadata(cached_hit_path, &cached_metadata)) {
                REG_DEBUG("jit metadata cache hit: %s",
                          jit_metadata_path_for(cached_hit_path).c_str());
                REG_DEBUG("jit cache hit: %s", cached_hit_path.c_str());
                publish_metadata(cached_metadata);
                *out_path = cached_hit_path.string();
                if (out_is_persistent != nullptr) *out_is_persistent = true;
                if (out_provenance != nullptr) {
                    *out_provenance = cached_hit_path.extension() == ".metallib"
                                          ? generic_ptx_provenance
                                          : "";
                }
                return true;
            }
            REG_DEBUG("jit metadata cache miss: %s",
                      jit_metadata_path_for(cached_hit_path).c_str());
        } else {
            REG_DEBUG("jit cache miss: %s", cached_metallib.c_str());
        }
    }

    // A cold artifact, or a legacy artifact without a metadata sidecar, still
    // needs lowering once. The resulting metadata is persisted below so later
    // processes take the fast path above.
    cumetal::ptx::LowerToMetalOptions lower_to_metal_options;
    lower_to_metal_options.entry_name = kernel_name;
    lower_to_metal_options.fp64_mode = std::string(cumetal::ptx::fp64_mode_name(
        cumetal::ptx::fp64_mode_from_env()));
    lower_to_metal_options.allow_workload_specializations =
        cumetal::diag_env_truthy("CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS");
    if (const char* backend = std::getenv("CUMETAL_PTX_BACKEND");
        backend != nullptr && std::string_view(backend) == "cumetal-ir") {
        lower_to_metal_options.backend =
            cumetal::ptx::PtxMetalBackend::kCumetalIr;
    }
    const auto lowered_metal =
        cumetal::ptx::lower_ptx_to_metal_source(ptx_source, lower_to_metal_options);
    if (!lowered_metal.ok) {
        REG_DEBUG("lower_ptx_to_metal_source failed for kernel '%s'", kernel_name.c_str());
        return false;
    }
    RegistrationCacheMetadata metadata{
        .printf_formats = lowered_metal.printf_formats,
        .uses_device_heap = lowered_metal.uses_device_heap,
    };
    if (ptx_source.find("cudaLaunchDevice") != std::string::npos ||
        ptx_source.find("cudaMemcpyAsync") != std::string::npos) {
        cumetal::ptx::LowerToLlvmOptions metadata_options;
        metadata_options.entry_name = kernel_name;
        metadata_options.strict = true;
        metadata_options.fp64_mode = cumetal::ptx::fp64_mode_from_env();
        const auto lowered_llvm_metadata =
            cumetal::ptx::lower_ptx_to_llvm_ir(ptx_source, metadata_options);
        if (!lowered_llvm_metadata.ok) {
            REG_DEBUG("device launch metadata lowering failed for '%s': %s",
                      kernel_name.c_str(), lowered_llvm_metadata.error.c_str());
            return false;
        }
        metadata.uses_device_launch_queue = lowered_llvm_metadata.uses_device_launch_queue;
    }

    if (!cached_hit_path.empty()) {
        (void)write_registration_metadata(cached_hit_path, metadata);
        REG_DEBUG("jit cache hit: %s", cached_hit_path.c_str());
        publish_metadata(metadata);
        *out_path = cached_hit_path.string();
        if (out_is_persistent != nullptr) *out_is_persistent = true;
        if (out_provenance != nullptr) {
            *out_provenance = cached_hit_path.extension() == ".metallib"
                                  ? generic_ptx_provenance
                                  : "";
        }
        return true;
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
    emit_options.validate_output = true;
    emit_options.fallback_to_experimental = true;
    emit_options.kernel_name = kernel_name;

    std::string io_error;
    bool use_direct_msl = lowered_metal.matched && !lowered_metal.metal_source.empty();
    if (use_direct_msl && lowered_metal.approximate) {
        cumetal::warn_once(
            "approx-refused:" + kernel_name,
            "kernel '" + kernel_name +
                "' produced an obsolete approximate lowering; refusing known-wrong output");
        use_direct_msl = false;
    }

    std::filesystem::path staged_input = ll_path;
    const bool typed_msl_needs_fp64_support =
        use_direct_msl && ptx_source.find(".f64") != std::string::npos;
    if (use_direct_msl) {
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
        if (!typed_msl_needs_fp64_support) {
            // Short-circuit non-linked MSL through newLibraryWithSource. FP64
            // needs private inline support in the same translation unit and
            // therefore takes the offline compile path below.
            std::filesystem::path msl_final = metal_path;
            if (!cached_metallib.empty()) {
                msl_final = cached_metallib;
                msl_final.replace_extension(".metal");
                std::error_code ec;
                std::filesystem::copy_file(
                    metal_path, msl_final,
                    std::filesystem::copy_options::overwrite_existing, ec);
            }
            *out_path = msl_final.string();
            if (out_is_persistent != nullptr) {
                *out_is_persistent = !cached_metallib.empty();
            }
            if (out_provenance != nullptr) *out_provenance = "";
            publish_metadata(metadata);
            if (!cached_metallib.empty()) {
                (void)write_registration_metadata(msl_final, metadata);
            }
            return true;
        }
        emit_options.textual_include_inputs.push_back(
            std::filesystem::path(CUMETAL_SOURCE_DIR) / "compiler" / "metal" /
            "support" / "cumetal_fp64_inline_support.metal");
    } else {
        REG_DEBUG("using LLVM IR lowering path for '%s'", kernel_name.c_str());
        maybe_dump_ptx_for_llvm_debug(kernel_name, ptx_source);
        // Match the driver JIT path: default kEmulate. Native AIR `double` ALU is
        // accepted by xcrun metal but fails at Metal pipeline creation on Apple GPU
        // (XPC_ERROR_CONNECTION_INTERRUPTED). See cuda_driver.cpp emit_ptx_to_temp_metallib.
        cumetal::ptx::LowerToLlvmOptions lower_options;
        lower_options.entry_name = kernel_name;
        lower_options.strict = true;
        lower_options.fp64_mode = cumetal::ptx::fp64_mode_from_env();
        if (lower_options.fp64_mode == cumetal::ptx::Fp64Mode::kEmulate &&
            ptx_source.find(".f64") != std::string::npos) {
            cumetal::warn_once(
                "fp64-emulate",
                "kernel uses FP64 (double) instructions, emulated with Dekker FP32-pair "
                "arithmetic (~48-bit significand with binary32 exponent range, not full "
                "IEEE-754 double); results lose precision. Set CUMETAL_FP64_MODE=native to "
                "compile true doubles (fails at launch on current Apple Silicon)");
        }
        const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(ptx_source, lower_options);
        if (!lowered.ok || lowered.llvm_ir.empty()) {
            if (!lowered.error.empty()) {
                REG_DEBUG("lower_ptx_to_llvm_ir error for '%s': %s",
                          kernel_name.c_str(), lowered.error.c_str());
            }
            REG_DEBUG("lower_ptx_to_llvm_ir failed for kernel '%s'", kernel_name.c_str());
            return false;
        }
        metadata.printf_formats = lowered.printf_formats;
        metadata.uses_device_heap = lowered.uses_device_heap;
        metadata.uses_device_launch_queue = lowered.uses_device_launch_queue;
        const std::vector<std::uint8_t> ll_bytes(lowered.llvm_ir.begin(), lowered.llvm_ir.end());
        if (!cumetal::common::write_file_bytes(ll_path, ll_bytes, &io_error)) {
            REG_DEBUG("write LLVM IR failed: %s", io_error.c_str());
            return false;
        }
        emit_options.kernel_name = lowered.entry_name.empty() ? kernel_name : lowered.entry_name;
        if (ptx_source.find(".f64") != std::string::npos &&
            cumetal::ptx::fp64_mode_links_vf64_support(lower_options.fp64_mode)) {
            emit_options.additional_link_inputs.push_back(
                std::filesystem::path(CUMETAL_SOURCE_DIR) / "compiler" /
                "metal" / "support" / "cumetal_fp64_support.metal"
            );
        }
    }
    emit_options.input = staged_input;

    REG_DEBUG("invoking emit_metallib: input=%s output=%s",
              staged_input.c_str(), metallib_path.c_str());
    // Optionally save the generated LLVM IR for debugging (set CUMETAL_DEBUG_DUMP_IR_DIR).
    if (const char* dump_dir = std::getenv("CUMETAL_DEBUG_DUMP_IR_DIR")) {
        if (dump_dir[0] != '\0' && std::filesystem::exists(ll_path)) {
            const std::string sanitized = [&]() {
                std::string s = kernel_name;
                for (char& c : s) { if (c != '_' && !std::isalnum(static_cast<unsigned char>(c))) c = '_'; }
                return s.substr(0, 80);
            }();
            const std::filesystem::path dest =
                std::filesystem::path(dump_dir) / (sanitized + ".ll");
            std::error_code ec;
            std::filesystem::copy_file(ll_path, dest,
                                       std::filesystem::copy_options::overwrite_existing, ec);
        }
    }
    const auto emitted = cumetal::air_emitter::emit_metallib(emit_options);
    remove_path_if_exists(ll_path.string());
    remove_path_if_exists(metal_path.string());
    for (const std::string& log_line : emitted.logs) {
        if (!log_line.empty()) {
            REG_DEBUG("emit_metallib log: %s", log_line.c_str());
        }
    }
    if (!emitted.ok || emitted.output.empty()) {
        REG_DEBUG("emit_metallib error for '%s': %s",
                  kernel_name.c_str(),
                  emitted.error.empty() ? "<none>" : emitted.error.c_str());
        REG_DEBUG("emit_metallib failed for kernel '%s'", kernel_name.c_str());
        remove_path_if_exists(metallib_path.string());
        return false;
    }

    if (emitted.mode_used == cumetal::air_emitter::EmitMode::kExperimentalContainer) {
        REG_DEBUG("emit_metallib for '%s' only produced an experimental test container "
                  "(metal toolchain was not available during lowering/JIT) - this kernel "
                  "is not executable on Metal; refusing to cache or use it",
                  kernel_name.c_str());
        remove_path_if_exists(emitted.output.string());
        // Do not treat as success; caller will fall back (leading to clear launch error)
        return false;
    }

    REG_DEBUG("emit success: %s", emitted.output.c_str());
    *out_path = emitted.output.string();
    if (out_provenance != nullptr) {
        // A real translation of this kernel's PTX, via LLVM IR and AIR. It is
        // emitted as a .metallib, which the Metal backend would otherwise
        // report as `precompiled_metallib` -- the same label a user-supplied
        // prebuilt binary gets. Naming it keeps the provenance contract able
        // to distinguish "we translated this" from "we loaded this".
        *out_provenance = generic_ptx_provenance;
    }
    publish_metadata(metadata);
    if (!cached_metallib.empty()) {
        (void)write_registration_metadata(emitted.output, metadata);
    }
    // Persistent cache entries (those routed through jit_cache_path_for) should
    // survive process exit and __cudaUnregisterFatBinary cleanup.
    if (out_is_persistent != nullptr) *out_is_persistent = !cached_metallib.empty();
    return true;
}

std::string resolve_metallib_path_for_kernel(void* module_handle,
                                              const std::string& kernel_name,
                                              std::vector<std::string>* out_printf_formats,
                                              bool* out_uses_device_heap,
                                              bool* out_uses_device_launch_queue,
                                              std::size_t* out_static_shared_bytes,
                                              std::string* out_provenance = nullptr) {
    if (module_handle == nullptr || kernel_name.empty()) {
        return fallback_metallib_path_from_env();
    }

    std::shared_ptr<const std::string> ptx_source;
    bool allow_environment_fallback = true;
    std::uint64_t cache_prefix_hash = kFnv1a64Offset;
    {
        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto found = s.modules.find(module_handle);
        if (found == s.modules.end()) {
            return fallback_metallib_path_from_env();
        }

        RegistrationModule& module = *found->second;
        allow_environment_fallback = module.allow_environment_fallback;
        if (!module.metallib_path.empty()) {
            REG_DEBUG("resolve_metallib '%s': prebuilt metallib '%s'",
                      kernel_name.c_str(), module.metallib_path.c_str());
            if (out_provenance != nullptr) *out_provenance = "precompiled_metallib";
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
            if (out_uses_device_heap != nullptr) {
                const auto heap_it =
                    module.emitted_kernel_uses_device_heap.find(kernel_name);
                if (heap_it != module.emitted_kernel_uses_device_heap.end()) {
                    *out_uses_device_heap = heap_it->second;
                }
            }
            if (out_uses_device_launch_queue != nullptr) {
                const auto launch_it =
                    module.emitted_kernel_uses_device_launch_queue.find(kernel_name);
                if (launch_it != module.emitted_kernel_uses_device_launch_queue.end()) {
                    *out_uses_device_launch_queue = launch_it->second;
                }
            }
            if (out_static_shared_bytes != nullptr) {
                const auto ssb_it = module.emitted_kernel_static_shared_bytes.find(kernel_name);
                if (ssb_it != module.emitted_kernel_static_shared_bytes.end()) {
                    *out_static_shared_bytes = ssb_it->second;
                }
            }
            if (out_provenance != nullptr) {
                const auto pv_it = module.emitted_kernel_provenance.find(kernel_name);
                if (pv_it != module.emitted_kernel_provenance.end()) {
                    *out_provenance = pv_it->second;
                }
            }
            return cached->second;
        }

        ptx_source = module.ptx_source;
        if (ptx_source != nullptr) {
            if (!module.jit_cache_prefix_hash.has_value()) {
                module.jit_cache_prefix_hash = ::jit_cache_prefix_hash(*ptx_source);
            }
            cache_prefix_hash = *module.jit_cache_prefix_hash;
        }
    }

    if (ptx_source == nullptr || ptx_source->empty()) {
        REG_DEBUG("resolve_metallib '%s': no PTX, env fallback allowed=%d",
                  kernel_name.c_str(),
                  static_cast<int>(allow_environment_fallback));
        return allow_environment_fallback ? fallback_metallib_path_from_env()
                                          : std::string{};
    }

    // Compute static shared memory size from the PTX source before JIT compilation.
    const std::size_t static_shared =
        cumetal::ptx::compute_static_shared_bytes(*ptx_source, kernel_name);

    REG_DEBUG("resolve_metallib '%s': JIT compiling... (static_shared=%zu)",
              kernel_name.c_str(), static_shared);
    std::string emitted_path;
    std::vector<std::string> local_printf_formats;
    bool local_uses_device_heap = false;
    bool local_uses_device_launch_queue = false;
    bool is_persistent = false;
    std::string local_provenance;
    if (!emit_ptx_entry_to_temp_metallib(*ptx_source, kernel_name, cache_prefix_hash,
                                         &emitted_path,
                                         &local_printf_formats, &local_uses_device_heap,
                                         &local_uses_device_launch_queue,
                                         &is_persistent,
                                         &local_provenance)) {
        REG_DEBUG("resolve_metallib '%s': JIT compile failed, using env fallback",
                  kernel_name.c_str());
        return allow_environment_fallback ? fallback_metallib_path_from_env()
                                          : std::string{};
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
        if (out_uses_device_heap != nullptr) {
            const auto heap_it =
                module.emitted_kernel_uses_device_heap.find(kernel_name);
            if (heap_it != module.emitted_kernel_uses_device_heap.end()) {
                *out_uses_device_heap = heap_it->second;
            }
        }
        if (out_uses_device_launch_queue != nullptr) {
            const auto launch_it =
                module.emitted_kernel_uses_device_launch_queue.find(kernel_name);
            if (launch_it != module.emitted_kernel_uses_device_launch_queue.end()) {
                *out_uses_device_launch_queue = launch_it->second;
            }
        }
        if (out_static_shared_bytes != nullptr) {
            const auto ssb_it = module.emitted_kernel_static_shared_bytes.find(kernel_name);
            if (ssb_it != module.emitted_kernel_static_shared_bytes.end()) {
                *out_static_shared_bytes = ssb_it->second;
            }
        }
        if (out_provenance != nullptr) {
            const auto pv_it = module.emitted_kernel_provenance.find(kernel_name);
            if (pv_it != module.emitted_kernel_provenance.end()) {
                *out_provenance = pv_it->second;
            }
        }
        return inserted.first->second;
    }

    module.emitted_kernel_printf_formats.emplace(kernel_name, local_printf_formats);
    module.emitted_kernel_uses_device_heap.emplace(kernel_name,
                                                   local_uses_device_heap);
    module.emitted_kernel_uses_device_launch_queue.emplace(
        kernel_name, local_uses_device_launch_queue);
    module.emitted_kernel_static_shared_bytes.emplace(kernel_name, static_shared);
    module.emitted_kernel_provenance.emplace(kernel_name, local_provenance);
    if (out_provenance != nullptr) {
        *out_provenance = local_provenance;
    }
    if (out_printf_formats != nullptr) {
        *out_printf_formats = std::move(local_printf_formats);
    }
    if (out_uses_device_heap != nullptr) {
        *out_uses_device_heap = local_uses_device_heap;
    }
    if (out_uses_device_launch_queue != nullptr) {
        *out_uses_device_launch_queue = local_uses_device_launch_queue;
    }
    if (out_static_shared_bytes != nullptr) {
        *out_static_shared_bytes = static_shared;
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

bool ensure_registered_global_symbol_buffer(
    const void* host_symbol,
    std::shared_ptr<cumetal::metal_backend::Buffer>* out_buffer,
    std::size_t* out_size) {
    if (host_symbol == nullptr || out_buffer == nullptr) {
        return false;
    }

    const void* initial_bytes = nullptr;
    std::size_t symbol_size = 0;
    {
        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto found = s.symbols.find(host_symbol);
        if (found == s.symbols.end() || found->second.constant ||
            !found->second.requires_metal_storage ||
            found->second.device_address == nullptr || found->second.size == 0) {
            return false;
        }
        if (found->second.global_buffer != nullptr) {
            *out_buffer = found->second.global_buffer;
            if (out_size != nullptr) {
                *out_size = found->second.size;
            }
            return true;
        }
        initial_bytes = found->second.device_address;
        symbol_size = found->second.size;
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> candidate;
    std::string allocation_error;
    if (cumetal::metal_backend::allocate_buffer(
            symbol_size, &candidate, &allocation_error) != cudaSuccess ||
        candidate == nullptr || candidate->contents() == nullptr) {
        return false;
    }
    std::memcpy(candidate->contents(), initial_bytes, symbol_size);

    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto found = s.symbols.find(host_symbol);
    if (found == s.symbols.end() || found->second.constant ||
        !found->second.requires_metal_storage ||
        found->second.size != symbol_size) {
        return false;
    }
    if (found->second.global_buffer == nullptr) {
        found->second.global_buffer = std::move(candidate);
    }
    *out_buffer = found->second.global_buffer;
    if (out_size != nullptr) {
        *out_size = found->second.size;
    }
    return *out_buffer != nullptr;
}

bool ensure_private_global_symbol_buffer(
    void* module_handle,
    const cumetal::ptx::ExternalGlobalSymbol& symbol,
    std::shared_ptr<cumetal::metal_backend::Buffer>* out_buffer) {
    if (module_handle == nullptr || out_buffer == nullptr || symbol.name.empty() ||
        symbol.size_bytes == 0 || !symbol.module_private_initialized) {
        return false;
    }

    std::shared_ptr<const std::string> ptx_source;
    {
        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto module = s.modules.find(module_handle);
        if (module == s.modules.end() || module->second == nullptr) return false;
        const auto existing =
            module->second->private_global_buffers.find(symbol.name);
        if (existing != module->second->private_global_buffers.end()) {
            *out_buffer = existing->second;
            return *out_buffer != nullptr;
        }
        ptx_source = module->second->ptx_source;
    }
    if (ptx_source == nullptr) return false;
    const auto initializer =
        cumetal::ptx::find_initialized_global_bytes(*ptx_source, symbol.name);
    if (!initializer.has_value() || initializer->size() != symbol.size_bytes) {
        return false;
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> candidate;
    std::string allocation_error;
    if (cumetal::metal_backend::allocate_buffer(
            symbol.size_bytes, &candidate, &allocation_error) != cudaSuccess ||
        candidate == nullptr || candidate->contents() == nullptr) {
        return false;
    }
    std::memcpy(candidate->contents(), initializer->data(), initializer->size());

    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto module = s.modules.find(module_handle);
    if (module == s.modules.end() || module->second == nullptr) return false;
    auto [stored, inserted] =
        module->second->private_global_buffers.emplace(symbol.name, candidate);
    (void)inserted;
    *out_buffer = stored->second;
    return *out_buffer != nullptr;
}

bool lookup_registered_kernel(const void* host_function, RegisteredKernel* out) {
    if (host_function == nullptr || out == nullptr) {
        return false;
    }

    RegistrationRecord record;
    {
        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto found = s.kernels.find(host_function);
        if (found == s.kernels.end()) {
            if (is_debug_registration()) {
                std::fprintf(stderr,
                             "[cumetal-reg] lookup MISS host_fn=%p (%zu kernels registered)\n",
                             host_function, s.kernels.size());
            }
            return false;
        }

        // Fatbins register every compiled CUDA kernel at process startup, while
        // an application normally launches only a small subset. Resolve only
        // the requested entry rather than allocating an ABI index for thousands
        // of unused GGML kernels in the same module.
        if (!found->second.arg_info_resolved) {
            const auto module_it = s.modules.find(found->second.module_handle);
            if (module_it != s.modules.end() && module_it->second != nullptr &&
                module_it->second->ptx_source != nullptr) {
                RegistrationModule& module = *module_it->second;
                (void)find_arg_info_for_ptx_entry(*module.ptx_source,
                                                  found->second.kernel_name,
                                                  &found->second.arg_info);
                found->second.external_constant_symbols =
                    cumetal::ptx::find_referenced_external_constant_symbols(
                        *module.ptx_source, found->second.kernel_name);
                found->second.external_constant_buffer_size =
                    cumetal::ptx::compute_external_constant_buffer_bytes(
                        *module.ptx_source);
                found->second.external_global_symbols =
                    cumetal::ptx::find_referenced_external_global_symbols(
                        *module.ptx_source, found->second.kernel_name);
            }
            found->second.arg_info_resolved = true;
        }
        record = found->second;
    }

    if (record.metallib_path.empty()) {
        std::vector<std::string> printf_formats;
        bool uses_device_heap = false;
        bool uses_device_launch_queue = false;
        std::size_t static_shared_bytes = 0;
        std::string provenance;
        std::string metallib_path =
            resolve_metallib_path_for_kernel(record.module_handle, record.kernel_name,
                                             &printf_formats, &uses_device_heap,
                                             &uses_device_launch_queue,
                                             &static_shared_bytes,
                                             &provenance);

        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto found = s.kernels.find(host_function);
        if (found == s.kernels.end()) {
            return false;
        }
        if (found->second.metallib_path.empty()) {
            found->second.metallib_path = std::move(metallib_path);
            if (!printf_formats.empty()) {
                found->second.printf_formats = std::move(printf_formats);
            }
            found->second.uses_device_heap = uses_device_heap;
            found->second.uses_device_launch_queue = uses_device_launch_queue;
            if (static_shared_bytes > 0) {
                found->second.static_shared_bytes = static_shared_bytes;
            }
            found->second.provenance = std::move(provenance);
        }
        record = found->second;
    }

    out->module_handle = record.module_handle;
    out->metallib_path = record.metallib_path;
    out->kernel_name = record.kernel_name;
    out->provenance = record.provenance;
    out->arg_info = record.arg_info;
    out->arg_info_resolved = record.arg_info_resolved;
    out->printf_formats = record.printf_formats;
    out->uses_device_heap = record.uses_device_heap;
    out->uses_device_launch_queue = record.uses_device_launch_queue;
    out->static_shared_bytes = record.static_shared_bytes;
    out->constant_symbols.clear();
    out->constant_buffer_size = record.external_constant_buffer_size;
    if (!record.external_constant_symbols.empty()) {
        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        for (const auto& expected : record.external_constant_symbols) {
            RegisteredConstantSymbol binding{
                .name = expected.name,
                .address = nullptr,
                .offset = expected.offset_bytes,
                .size = expected.size_bytes,
            };
            for (const auto& [host_symbol, registered] : s.symbols) {
                (void)host_symbol;
                if (registered.module_handle == record.module_handle &&
                    registered.constant && registered.device_name == expected.name &&
                    registered.size == expected.size_bytes) {
                    binding.address = registered.device_address;
                    break;
                }
            }
            out->constant_symbols.push_back(std::move(binding));
        }
    }
    out->global_symbols.clear();
    for (const auto& expected : record.external_global_symbols) {
        const void* host_symbol = nullptr;
        {
            RegistrationState& s = state();
            std::lock_guard<std::mutex> lock(s.mutex);
            for (const auto& [candidate_host, registered] : s.symbols) {
                if (registered.module_handle == record.module_handle &&
                    !registered.constant && registered.device_name == expected.name &&
                    registered.size == expected.size_bytes) {
                    host_symbol = candidate_host;
                    break;
                }
            }
        }

        RegisteredGlobalSymbol binding{
            .name = expected.name,
            .buffer = nullptr,
            .size = expected.size_bytes,
        };
        std::size_t actual_size = 0;
        if (host_symbol != nullptr) {
            (void)ensure_registered_global_symbol_buffer(
                host_symbol, &binding.buffer, &actual_size);
        } else if (expected.module_private_initialized &&
                   ensure_private_global_symbol_buffer(
                       record.module_handle, expected, &binding.buffer)) {
            actual_size = expected.size_bytes;
        }
        if (actual_size != expected.size_bytes) {
            binding.buffer.reset();
        }
        out->global_symbols.push_back(std::move(binding));
    }
    return true;
}

bool lookup_device_kernel_alias(void* module_handle,
                                std::uint64_t token,
                                const void** out_host_function,
                                RegisteredKernel* out_kernel) {
    if (module_handle == nullptr || token == 0 || out_host_function == nullptr) {
        return false;
    }

    const void* alias = nullptr;
    {
        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        RegistrationModule* target_module = nullptr;
        void* target_handle = nullptr;
        std::string matched_name;
        std::vector<cumetalKernelArgInfo_t> matched_args;
        const auto consider_module = [&](void* candidate_handle,
                                         RegistrationModule* candidate) {
            if (candidate == nullptr || candidate->ptx_source == nullptr ||
                target_module != nullptr) {
                return;
            }
            if (const auto existing = candidate->device_kernel_aliases.find(token);
                existing != candidate->device_kernel_aliases.end()) {
                target_module = candidate;
                target_handle = candidate_handle;
                alias = existing->second.get();
                return;
            }
            const auto entries = build_arg_info_index_from_ptx(*candidate->ptx_source);
            for (const auto& [name, args] : entries) {
                if (stable_device_kernel_token(name) == token) {
                    target_module = candidate;
                    target_handle = candidate_handle;
                    matched_name = name;
                    matched_args = args;
                    return;
                }
            }
        };

        // Prefer the parent module, then search other RDC fatbins for external
        // child entries referenced by this module.
        if (const auto preferred = s.modules.find(module_handle);
            preferred != s.modules.end()) {
            consider_module(preferred->first, preferred->second.get());
        }
        for (const auto& [candidate_handle, candidate] : s.modules) {
            consider_module(candidate_handle, candidate.get());
        }
        if (target_module == nullptr) {
            return false;
        }
        if (alias == nullptr) {
            RegistrationModule& module = *target_module;

            auto storage = std::make_unique<unsigned char>(0);
            alias = storage.get();
            RegistrationRecord record;
            record.module_handle = target_handle;
            record.metallib_path = module.metallib_path;
            record.kernel_name = matched_name;
            record.arg_info = matched_args;
            record.arg_info_resolved = true;
            record.external_constant_symbols =
                cumetal::ptx::find_referenced_external_constant_symbols(
                    *module.ptx_source, record.kernel_name);
            record.external_constant_buffer_size =
                cumetal::ptx::compute_external_constant_buffer_bytes(
                    *module.ptx_source);
            record.external_global_symbols =
                cumetal::ptx::find_referenced_external_global_symbols(
                    *module.ptx_source, record.kernel_name);
            s.kernels.emplace(alias, std::move(record));
            module.device_kernel_aliases.emplace(token, std::move(storage));
        }
    }

    *out_host_function = alias;
    if (out_kernel != nullptr) {
        return lookup_registered_kernel(alias, out_kernel);
    }
    return true;
}

bool lookup_registered_symbol(const void* host_symbol,
                              const void** out_device_symbol,
                              std::size_t* out_size) {
    if (host_symbol == nullptr || out_device_symbol == nullptr) {
        return false;
    }

    {
        RegistrationState& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto found = s.symbols.find(host_symbol);
        if (found == s.symbols.end() || found->second.device_address == nullptr) {
            return false;
        }
        if (found->second.constant || !found->second.requires_metal_storage) {
            *out_device_symbol = found->second.device_address;
            if (out_size != nullptr) {
                *out_size = found->second.size;
            }
            return true;
        }
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> buffer;
    if (!ensure_registered_global_symbol_buffer(host_symbol, &buffer, out_size) ||
        buffer == nullptr || buffer->contents() == nullptr) {
        return false;
    }
    *out_device_symbol = buffer->contents();
    return true;
}

// Drop only the state that belongs to the device context: the Metal buffers
// backing __device__ globals, which cudaDeviceReset is defined to destroy.
// The kernel and module tables are NOT context state -- they come from
// __cudaRegisterFatBinary when the image loads, and real CUDA keeps them across
// a reset so the next launch re-creates a context and re-loads the modules.
// Clearing them here made every launch after GROMACS's device-detection
// cudaDeviceReset fail with "inline kernel descriptor invalid".
void reset_device_state() {
    RegistrationState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (auto& [host_symbol, record] : s.symbols) {
        (void)host_symbol;
        record.global_buffer.reset();
    }
    for (auto& [module_handle, module] : s.modules) {
        (void)module_handle;
        if (module != nullptr) module->private_global_buffers.clear();
    }
    tls_launch_stack.clear();
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
    module->ptx_source =
        std::make_shared<const std::string>(std::move(parsed.ptx_source));
    module->allow_environment_fallback = parsed.allow_environment_fallback;

    if (parsed.allow_environment_fallback && module->metallib_path.empty() &&
        module->ptx_source->empty()) {
        module->metallib_path = cumetal::registration::fallback_metallib_path_from_env();
    }

    REG_DEBUG("__cudaRegisterFatBinary: metallib='%s' ptx_size=%zu",
              module->metallib_path.c_str(), module->ptx_source->size());

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

void __cudaRegisterLinkedBinary(void (*register_globals)(void**),
                                void* fatbin_wrapper,
                                void* module_id,
                                void (*callback)(void)) {
    (void)module_id;
    (void)callback;
    if (register_globals == nullptr || fatbin_wrapper == nullptr) {
        return;
    }
    void** handle = __cudaRegisterFatBinary(fatbin_wrapper);
    if (handle == nullptr) {
        return;
    }
    register_globals(handle);
    __cudaRegisterFatBinaryEnd(handle);
}

void __cudaRegisterFatBinaryEnd(void** fat_cubin_handle) {
    (void)fat_cubin_handle;
}

void __cudaUnregisterFatBinary(void** fat_cubin_handle) {
    REG_DEBUG("__cudaUnregisterFatBinary handle=%p",
              static_cast<void*>(fat_cubin_handle));
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
    bool lazy_metallib_resolution = true;
    bool allow_environment_fallback = true;
    std::string metallib_path;
    {
        cumetal::registration::RegistrationState& s = cumetal::registration::state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto module_it = s.modules.find(handle);
        if (module_it != s.modules.end() && module_it->second != nullptr) {
            metallib_path = module_it->second->metallib_path;
            lazy_metallib_resolution = metallib_path.empty();
            allow_environment_fallback =
                module_it->second->allow_environment_fallback;
        }
    }
    if (metallib_path.empty() && allow_environment_fallback) {
        metallib_path = cumetal::registration::fallback_metallib_path_from_env();
    }

    REG_DEBUG("__cudaRegisterFunction: kernel='%s' metallib='%s' args=lazy (lazy=%d)",
              chosen_name, metallib_path.c_str(),
              lazy_metallib_resolution ? 1 : 0);

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.kernels[host_function] = cumetal::registration::RegistrationRecord{
        .module_handle = handle,
        .metallib_path = std::move(metallib_path),
        .kernel_name = chosen_name,
        .arg_info = {},
        .arg_info_resolved = false,
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
    (void)ext;
    (void)global;

    if (host_var == nullptr) {
        return;
    }

    // `device_address` is NOT an address under the clang/NVCC registration ABI:
    // CodeGen emits __cudaRegisterVar(handle, (char *)&Var, VarName, VarName, ...),
    // passing the device-side *name string* for both the third and fourth argument.
    // Registering it as the device address meant cudaMemcpyToSymbol memcpy'd user
    // data straight over a string literal -- a SIGBUS when the constant landed in a
    // read-only page (cuda-samples/LargeKernelParameter, 27 KB over "excess_params"),
    // and silent corruption of the binary's own data when it did not.
    //
    // The host shadow is the storage the runtime reads and writes, so map to it.
    // Hand-built registrations that do pass a genuine distinct address still win.
    const bool device_address_is_name =
        device_address != nullptr && device_name != nullptr &&
        (device_address == device_name || std::strcmp(device_address, device_name) == 0);
    const void* mapped = (device_address == nullptr || device_address_is_name)
                             ? static_cast<const void*>(host_var)
                             : static_cast<const void*>(device_address);
    void* handle = fat_cubin_handle == nullptr ? nullptr : reinterpret_cast<void*>(fat_cubin_handle);

    // CUDA Clang emits initialized `__device__` storage as exact bytes in PTX,
    // but leaves its host registration shadow in zero-filled BSS. Recover the
    // PTX-owned initializer before publishing the symbol so the ordinary
    // registration-backed persistent buffer starts with the source value. A
    // genuine distinct caller-supplied address remains authoritative.
    std::shared_ptr<const std::string> module_ptx;
    if (device_address_is_name || device_address == nullptr) {
        cumetal::registration::RegistrationState& lookup_state =
            cumetal::registration::state();
        std::lock_guard<std::mutex> lookup_lock(lookup_state.mutex);
        const auto module = lookup_state.modules.find(handle);
        if (module != lookup_state.modules.end() && module->second != nullptr) {
            module_ptx = module->second->ptx_source;
        }
    }
    if (module_ptx != nullptr && device_name != nullptr) {
        const auto initializer =
            cumetal::ptx::find_initialized_global_bytes(*module_ptx, device_name);
        if (initializer.has_value() && initializer->size() == size) {
            std::memcpy(host_var, initializer->data(), size);
            mapped = host_var;
        }
    }

    REG_DEBUG("__cudaRegisterVar: name='%s' host_var=%p mapped=%p size=%zu",
              device_name != nullptr ? device_name : "(null)", static_cast<void*>(host_var),
              mapped, size);

    cumetal::registration::RegistrationState& s = cumetal::registration::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.symbols[host_var] = cumetal::registration::RegistrationSymbolRecord{
        .module_handle = handle,
        .device_address = mapped,
        .device_name = device_name != nullptr ? device_name : "",
        .size = size,
        .constant = constant != 0,
        .requires_metal_storage = device_address_is_name || device_address == nullptr,
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
