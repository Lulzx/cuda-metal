// hash_shim.cpp — CuMetal
// Provides std::__1::__hash_memory for binaries/objects that were compiled against
// Homebrew LLVM libc++ (which references this internal) but are linked against
// Apple's system libc++ (or when mixing). This was previously in cuda_runtime.cpp
// but moved here to a TU with *no* heavy includes that pull in the conflicting
// inline definition from the SDK's <__functional/hash.h>.
//
// The implementation is FNV-1a (stable, sufficient for the unordered_map use in
// e.g. llama.cpp GGML CUDA host objects).

#include <cstddef>
#include <cstdint>

namespace std {
namespace __1 {

size_t __hash_memory(const void* data, size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    // 64-bit FNV-1a
    size_t hash = static_cast<size_t>(1469598103934665603ull);
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<size_t>(bytes[i]);
        hash *= static_cast<size_t>(1099511628211ull);
    }
    return hash;
}

}  // namespace __1
}  // namespace std
