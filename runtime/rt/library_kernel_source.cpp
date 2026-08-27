#include "library_kernel_source.h"

#include "f64_dekker_msl.h"
#include "runtime_internal.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>

namespace cumetal::rt {
namespace {

std::filesystem::path cache_dir() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path dir;
    if (const char* d = std::getenv("CUMETAL_CACHE_DIR"); d != nullptr && d[0] != '\0') {
        dir = fs::path(d);
    } else if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        dir = fs::path(home) / "Library" / "Caches" / "io.cumetal";
    } else {
        dir = fs::temp_directory_path(ec);
        if (ec) return {};
    }
    dir /= "library-kernels";
    fs::create_directories(dir, ec);
    if (ec) return {};
    return dir;
}

}  // namespace

const std::string* stage_library_kernel_source(std::string_view name,
                                               std::string_view body) {
    static std::mutex mutex;
    static std::map<std::string, std::string, std::less<>> staged;

    const std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = staged.find(name); it != staged.end()) {
        return &it->second;
    }

    const std::filesystem::path dir = cache_dir();
    if (dir.empty()) return nullptr;
    const std::filesystem::path out = dir / (std::string(name) + ".metal");

    std::FILE* f = std::fopen(out.c_str(), "wb");
    if (f == nullptr) return nullptr;
    const auto write = [f](std::string_view s) {
        return std::fwrite(s.data(), 1, s.size(), f) == s.size();
    };
    // The Dekker prelude first: it opens with the provenance and math-mode
    // markers, which the backend reads from the head of the file.
    const bool wrote = write(kF64DekkerMsl) && write(body);
    std::fclose(f);
    if (!wrote) return nullptr;

    return &staged.emplace(std::string(name), out.string()).first->second;
}

bool resolve_kernel_buffer_arg(const void* ptr,
                               std::size_t required_bytes,
                               std::size_t alignment,
                               cumetal::metal_backend::KernelArg* out) {
    if (ptr == nullptr) return false;
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(ptr, &resolved)) return false;
    if (resolved.buffer == nullptr || resolved.remaining_size < required_bytes) return false;
    if (resolved.offset % alignment != 0) return false;
    out->kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
    out->buffer = resolved.buffer;
    out->offset = resolved.offset;
    return true;
}

}  // namespace cumetal::rt
