#include "module_cache.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace cumetal::cache {
namespace {

std::uint64_t fnv1a64(const std::uint8_t* bytes, std::size_t size) {
    constexpr std::uint64_t kOffset = 1469598103934665603ull;
    constexpr std::uint64_t kPrime = 1099511628211ull;

    std::uint64_t hash = kOffset;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= kPrime;
    }
    return hash;
}

std::filesystem::path default_cache_root() {
    if (const char* explicit_root = std::getenv("CUMETAL_CACHE_DIR");
        explicit_root != nullptr && explicit_root[0] != '\0') {
        return std::filesystem::path(explicit_root);
    }

    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        return std::filesystem::path(home) / "Library" / "Caches" / "io.cumetal" / "kernels";
    }

    return std::filesystem::temp_directory_path() / "io.cumetal" / "kernels";
}

std::string hash_to_hex(std::uint64_t hash) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

bool ensure_directory(const std::filesystem::path& path, std::string* error_message) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (!ec) {
        return true;
    }
    if (error_message != nullptr) {
        *error_message = "failed to create cache directory: " + path.string() + " (" + ec.message() +
                         ")";
    }
    return false;
}

bool write_file_bytes(const std::filesystem::path& path,
                      const std::uint8_t* bytes,
                      std::size_t size,
                      std::string* error_message) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (error_message != nullptr) {
            *error_message = "failed to open cache file for write: " + path.string();
        }
        return false;
    }

    out.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    out.close();
    if (!out.good()) {
        if (error_message != nullptr) {
            *error_message = "failed to write cache file: " + path.string();
        }
        return false;
    }
    return true;
}

bool file_size_matches(const std::filesystem::path& path, std::size_t expected_size) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return !ec && size == expected_size;
}

}  // namespace

bool stage_metallib_bytes(const void* image,
                          std::size_t size,
                          std::filesystem::path* out_path,
                          std::string* error_message) {
    if (image == nullptr || size == 0 || out_path == nullptr) {
        if (error_message != nullptr) {
            *error_message = "stage_metallib_bytes invalid argument";
        }
        return false;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(image);
    const std::filesystem::path root = default_cache_root();
    if (!ensure_directory(root, error_message)) {
        return false;
    }

    const std::uint64_t hash = fnv1a64(bytes, size);
    const std::filesystem::path target =
        root / ("metallib-" + hash_to_hex(hash) + "-" + std::to_string(size) + ".metallib");

    std::error_code ec;
    if (std::filesystem::exists(target, ec) && !ec && file_size_matches(target, size)) {
        *out_path = target;
        return true;
    }

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temp = target.string() + ".tmp." + std::to_string(nonce);
    if (!write_file_bytes(temp, bytes, size, error_message)) {
        std::filesystem::remove(temp, ec);
        return false;
    }

    ec.clear();
    std::filesystem::rename(temp, target, ec);
    if (ec) {
        if (!std::filesystem::exists(target, ec) || ec || !file_size_matches(target, size)) {
            std::filesystem::remove(temp, ec);
            if (error_message != nullptr) {
                *error_message = "failed to publish cache file: " + target.string() + " (" +
                                 ec.message() + ")";
            }
            return false;
        }
    }

    *out_path = target;
    return true;
}


bool stage_metallib_abi_sidecar(const std::filesystem::path& metallib_path,
                                const void* sidecar,
                                std::size_t size,
                                std::string* error_message) {
    if (sidecar == nullptr || size == 0 || metallib_path.empty()) {
        if (error_message != nullptr) {
            *error_message = "stage_metallib_abi_sidecar invalid argument";
        }
        return false;
    }

    const std::filesystem::path target = metallib_path.string() + ".cumetal-abi";
    std::error_code ec;
    if (std::filesystem::exists(target, ec) && !ec && file_size_matches(target, size)) {
        return true;
    }

    // Same publish-by-rename as the metallib itself: a reader must never see a
    // sidecar that is half written, since a truncated one parses as invalid and
    // sends the launch back to guessing.
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path temp = target.string() + ".tmp." + std::to_string(nonce);
    if (!write_file_bytes(temp, static_cast<const std::uint8_t*>(sidecar), size, error_message)) {
        std::filesystem::remove(temp, ec);
        return false;
    }

    ec.clear();
    std::filesystem::rename(temp, target, ec);
    if (ec) {
        std::filesystem::remove(temp, ec);
        if (error_message != nullptr) {
            *error_message = "failed to publish ABI sidecar: " + target.string();
        }
        return false;
    }
    return true;
}


namespace {

constexpr char kModuleImageMagic[8] = {'C', 'U', 'M', 'T', 'L', 'M', 'D', '1'};
constexpr std::size_t kModuleImageHeader = sizeof(kModuleImageMagic) + 2 * sizeof(std::uint64_t);

}  // namespace

std::string pack_module_image(const void* metallib,
                              std::size_t metallib_size,
                              const void* sidecar,
                              std::size_t sidecar_size) {
    std::string image;
    image.reserve(kModuleImageHeader + metallib_size + sidecar_size);
    image.append(kModuleImageMagic, sizeof(kModuleImageMagic));

    const std::uint64_t sizes[2] = {static_cast<std::uint64_t>(metallib_size),
                                    static_cast<std::uint64_t>(sidecar_size)};
    image.append(reinterpret_cast<const char*>(sizes), sizeof(sizes));
    image.append(static_cast<const char*>(metallib), metallib_size);
    image.append(static_cast<const char*>(sidecar), sidecar_size);
    return image;
}

bool parse_module_image(const void* image, std::size_t max_size, ModuleImageParts* out) {
    if (image == nullptr || out == nullptr || max_size < kModuleImageHeader) {
        return false;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(image);
    if (std::memcmp(bytes, kModuleImageMagic, sizeof(kModuleImageMagic)) != 0) {
        return false;
    }

    std::uint64_t sizes[2] = {0, 0};
    std::memcpy(sizes, bytes + sizeof(kModuleImageMagic), sizeof(sizes));
    // A sidecar of zero length is not a container worth writing, and either
    // length overrunning the caller's bound means the header is not ours.
    if (sizes[0] == 0 || sizes[1] == 0 || sizes[0] > max_size || sizes[1] > max_size ||
        kModuleImageHeader + sizes[0] + sizes[1] > max_size) {
        return false;
    }

    out->metallib = bytes + kModuleImageHeader;
    out->metallib_size = static_cast<std::size_t>(sizes[0]);
    out->sidecar = out->metallib + out->metallib_size;
    out->sidecar_size = static_cast<std::size_t>(sizes[1]);
    return true;
}

}  // namespace cumetal::cache
