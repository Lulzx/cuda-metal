#pragma once

// Locates the Metal support sources (software FP64, libdevice) that cumetalc
// and the runtime compile alongside generated MSL. They ship in two layouts:
//
//   build tree:  <source>/compiler/metal/support/<name>
//   installed:   <prefix>/share/cumetal/metal/compiler/metal/support/<name>
//
// Both keep VF64 at its source-relative path, which the support files
// #include. The installed copy is found from $CUMETAL_ROOT or from the binary
// image (cumetalc, or libcumetal.dylib) that calls this, so an installed
// toolchain never depends on the source checkout it was built from.

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>
#include <string_view>
#include <system_error>

namespace cumetal::common {

inline std::filesystem::path installed_metal_support_dir(const std::filesystem::path& prefix) {
    return prefix / "share" / "cumetal" / "metal" / "compiler" / "metal" / "support";
}

inline std::filesystem::path metal_support_path(std::string_view name,
                                                std::string_view source_dir) {
    std::error_code ec;
    if (const char* root = std::getenv("CUMETAL_ROOT"); root != nullptr && root[0] != '\0') {
        const auto candidate = installed_metal_support_dir(root) / name;
        if (std::filesystem::exists(candidate, ec)) return candidate;
    }
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&installed_metal_support_dir), &info) != 0 &&
        info.dli_fname != nullptr) {
        const auto image = std::filesystem::weakly_canonical(info.dli_fname, ec);
        if (!ec) {
            // <prefix>/bin/cumetalc or <prefix>/lib/libcumetal.dylib.
            const auto candidate =
                installed_metal_support_dir(image.parent_path().parent_path()) / name;
            if (std::filesystem::exists(candidate, ec)) return candidate;
        }
    }
    return std::filesystem::path(source_dir) / "compiler" / "metal" / "support" / name;
}

}  // namespace cumetal::common
