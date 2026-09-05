#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace cumetal::cache {

bool stage_metallib_bytes(const void* image,
                          std::size_t size,
                          std::filesystem::path* out_path,
                          std::string* error_message);

// Publishes a `.cumetal-abi` sidecar beside a metallib already staged by
// stage_metallib_bytes.
//
// The sidecar records each kernel argument's kind and size, and cuLaunchKernel
// needs it to bind arguments without guessing an argument count. It normally
// sits next to the metallib cumetalc wrote, but a caller that hands the driver
// an in-memory image -- NVRTC's, for instance -- has only the bytes. Since the
// cache is keyed on those bytes, staging the sidecar here puts it exactly where
// the later cuModuleLoadData of the same image will look.
bool stage_metallib_abi_sidecar(const std::filesystem::path& metallib_path,
                                const void* sidecar,
                                std::size_t size,
                                std::string* error_message);

}  // namespace cumetal::cache
