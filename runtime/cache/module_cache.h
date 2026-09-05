#pragma once

#include <cstddef>
#include <cstdint>
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
// needs it to bind arguments without guessing an argument count.
bool stage_metallib_abi_sidecar(const std::filesystem::path& metallib_path,
                                const void* sidecar,
                                std::size_t size,
                                std::string* error_message);

// A metallib and its ABI sidecar carried together as one byte image.
//
// cumetalc writes the sidecar as a file beside the metallib, which works for
// anyone who names a path. NVRTC callers do not: they take bytes from
// nvrtcGetCUBIN, cache them wherever they like, and hand them back to
// cuModuleLoadData in some later process. Nothing in that round trip carries a
// second file, so the sidecar has to travel inside the image or not at all --
// and without it cuLaunchKernel is reduced to guessing an argument count from a
// NULL terminator CUDA never promises.
//
// Layout: an 8-byte magic, then the two little-endian 64-bit lengths, then the
// metallib and the sidecar. The magic cannot collide with the "MTLB" a bare
// metallib starts with, or with PTX text, so cuModuleLoadData can tell the
// three apart by inspection.
struct ModuleImageParts {
    const std::uint8_t* metallib = nullptr;
    std::size_t metallib_size = 0;
    const std::uint8_t* sidecar = nullptr;
    std::size_t sidecar_size = 0;
};

std::string pack_module_image(const void* metallib,
                              std::size_t metallib_size,
                              const void* sidecar,
                              std::size_t sidecar_size);

// Reads the header only; the spans point into `image`, which must outlive them.
// `max_size` bounds the trusted lengths, since cuModuleLoadData is handed a
// bare pointer with no size of its own.
bool parse_module_image(const void* image, std::size_t max_size, ModuleImageParts* out);

}  // namespace cumetal::cache
