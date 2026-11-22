// #230: the driver library exports NVIDIA's _v2 entry-point names. A program
// that resolves the driver with dlopen/dlsym (as xgrid-gpu does) must find
// every _v2 name, each must be the same function as its unsuffixed spelling,
// and a call sequence made only through _v2 pointers must work.
//
// usage: driver_v2_symbols_test <driver dylib> <driver_v2_aliases.txt>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUdeviceptr = std::uint64_t;

std::vector<std::pair<std::string, std::string>> read_aliases(const char* path) {
    std::vector<std::pair<std::string, std::string>> aliases;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string from;
        std::string to;
        if (!(fields >> from >> to)) continue;
        // ld64 alias lists use Mach-O names; dlsym takes the C name.
        if (from[0] == '_') from.erase(0, 1);
        if (to[0] == '_') to.erase(0, 1);
        aliases.emplace_back(from, to);
    }
    return aliases;
}

template <typename Fn>
Fn lookup(void* library, const char* name) {
    return reinterpret_cast<Fn>(dlsym(library, name));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <driver dylib> <alias list>\n", argv[0]);
        return 64;
    }
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        std::fprintf(stderr, "FAIL: dlopen(%s): %s\n", argv[1], dlerror());
        return 1;
    }

    const auto aliases = read_aliases(argv[2]);
    if (aliases.size() < 6) {
        std::fprintf(stderr, "FAIL: alias list %s has %zu entries\n", argv[2], aliases.size());
        return 1;
    }
    int failures = 0;
    for (const auto& [plain, versioned] : aliases) {
        void* plain_fn = dlsym(library, plain.c_str());
        void* versioned_fn = dlsym(library, versioned.c_str());
        if (plain_fn == nullptr || versioned_fn == nullptr) {
            std::fprintf(stderr, "FAIL: %s=%p %s=%p\n", plain.c_str(), plain_fn,
                         versioned.c_str(), versioned_fn);
            ++failures;
        } else if (plain_fn != versioned_fn) {
            std::fprintf(stderr, "FAIL: %s and %s are different functions\n",
                         plain.c_str(), versioned.c_str());
            ++failures;
        }
    }
    // The six names xgrid-gpu needs must be in the list itself.
    for (const char* required : {"cuCtxCreate_v2", "cuCtxDestroy_v2", "cuMemAlloc_v2",
                                 "cuMemFree_v2", "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2"}) {
        bool listed = false;
        for (const auto& alias : aliases) listed = listed || alias.second == required;
        if (!listed) {
            std::fprintf(stderr, "FAIL: %s missing from the alias list\n", required);
            ++failures;
        }
    }
    // A name NVIDIA never versioned must not have grown a _v2.
    if (dlsym(library, "cuInit_v2") != nullptr) {
        std::fprintf(stderr, "FAIL: unexpected cuInit_v2 export\n");
        ++failures;
    }
    // cuGetProcAddress is the exception: the unsuffixed export keeps CUDA
    // 11.3's 4-argument ABI and _v2 is CUDA 12's 5-argument function, so they
    // must be two functions, and each must work when called with exactly its
    // own argument count.
    using GetProc4 = CUresult (*)(const char*, void**, int, std::uint64_t);
    using GetProc5 = CUresult (*)(const char*, void**, int, std::uint64_t, int*);
    auto get_proc4 = lookup<GetProc4>(library, "cuGetProcAddress");
    auto get_proc5 = lookup<GetProc5>(library, "cuGetProcAddress_v2");
    if (get_proc4 == nullptr || get_proc5 == nullptr ||
        reinterpret_cast<void*>(get_proc4) == reinterpret_cast<void*>(get_proc5)) {
        std::fprintf(stderr, "FAIL: cuGetProcAddress and _v2 must be distinct exports\n");
        return 1;
    }
    for (const auto& [plain, versioned] : aliases) {
        if (plain == "cuGetProcAddress" || versioned == "cuGetProcAddress_v2") {
            std::fprintf(stderr, "FAIL: the alias list must not alias cuGetProcAddress\n");
            return 1;
        }
    }
    void* resolved = nullptr;
    if (get_proc4("cuMemAlloc", &resolved, 12000, 0) != 0 ||
        resolved != dlsym(library, "cuMemAlloc")) {
        std::fprintf(stderr, "FAIL: 4-argument cuGetProcAddress(cuMemAlloc)\n");
        ++failures;
    }
    resolved = nullptr;
    if (get_proc4("cuNoSuchEntryPoint", &resolved, 12000, 0) == 0 || resolved != nullptr) {
        std::fprintf(stderr, "FAIL: 4-argument cuGetProcAddress must report a missing name\n");
        ++failures;
    }
    int status = -1;
    resolved = nullptr;
    if (get_proc5("cuMemAlloc_v2", &resolved, 12000, 0, &status) != 0 || status != 0 ||
        resolved != dlsym(library, "cuMemAlloc")) {
        std::fprintf(stderr, "FAIL: 5-argument cuGetProcAddress_v2(cuMemAlloc_v2)\n");
        ++failures;
    }
    status = -1;
    if (get_proc5("cuNoSuchEntryPoint", &resolved, 12000, 0, &status) == 0 || status != 1) {
        std::fprintf(stderr, "FAIL: cuGetProcAddress_v2 must set SYMBOL_NOT_FOUND\n");
        ++failures;
    }
    // Looking up "cuGetProcAddress" itself returns the ABI the caller's CUDA
    // version expects: the 4-argument function before 12.0, _v2 from 12.0.
    resolved = nullptr;
    if (get_proc4("cuGetProcAddress", &resolved, 11080, 0) != 0 ||
        resolved != reinterpret_cast<void*>(get_proc4)) {
        std::fprintf(stderr, "FAIL: cuGetProcAddress at 11.8 must return the 4-argument form\n");
        ++failures;
    }
    resolved = nullptr;
    if (get_proc5("cuGetProcAddress", &resolved, 12000, 0, &status) != 0 ||
        resolved != reinterpret_cast<void*>(get_proc5)) {
        std::fprintf(stderr, "FAIL: cuGetProcAddress at 12.0 must return cuGetProcAddress_v2\n");
        ++failures;
    }
    if (failures != 0) return 1;

    // Drive a context, allocation and round-trip copy only through _v2 names.
    auto cu_init = lookup<CUresult (*)(unsigned int)>(library, "cuInit");
    auto cu_device_get = lookup<CUresult (*)(CUdevice*, int)>(library, "cuDeviceGet");
    auto ctx_create = lookup<CUresult (*)(CUcontext*, unsigned int, CUdevice)>(library, "cuCtxCreate_v2");
    auto ctx_destroy = lookup<CUresult (*)(CUcontext)>(library, "cuCtxDestroy_v2");
    auto mem_alloc = lookup<CUresult (*)(CUdeviceptr*, std::size_t)>(library, "cuMemAlloc_v2");
    auto mem_free = lookup<CUresult (*)(CUdeviceptr)>(library, "cuMemFree_v2");
    auto htod = lookup<CUresult (*)(CUdeviceptr, const void*, std::size_t)>(library, "cuMemcpyHtoD_v2");
    auto dtoh = lookup<CUresult (*)(void*, CUdeviceptr, std::size_t)>(library, "cuMemcpyDtoH_v2");
    auto memset_d32 = lookup<CUresult (*)(CUdeviceptr, unsigned int, std::size_t)>(library, "cuMemsetD32_v2");
    auto mem_info = lookup<CUresult (*)(std::size_t*, std::size_t*)>(library, "cuMemGetInfo_v2");

    CUdevice device = 0;
    CUcontext context = nullptr;
    if (cu_init(0) != 0 || cu_device_get(&device, 0) != 0) {
        std::fprintf(stderr, "SKIP: no CUDA device through %s\n", argv[1]);
        return 77;
    }
    if (ctx_create(&context, 0, device) != 0 || context == nullptr) {
        std::fprintf(stderr, "FAIL: cuCtxCreate_v2\n");
        return 1;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (mem_info(&free_bytes, &total_bytes) != 0 || total_bytes == 0) {
        std::fprintf(stderr, "FAIL: cuMemGetInfo_v2\n");
        return 1;
    }
    constexpr std::size_t kCount = 4096;
    std::vector<std::uint32_t> host(kCount);
    for (std::size_t i = 0; i < kCount; ++i) host[i] = static_cast<std::uint32_t>(i * 2654435761u);
    CUdeviceptr device_buffer = 0;
    if (mem_alloc(&device_buffer, kCount * sizeof(std::uint32_t)) != 0 || device_buffer == 0 ||
        htod(device_buffer, host.data(), kCount * sizeof(std::uint32_t)) != 0) {
        std::fprintf(stderr, "FAIL: cuMemAlloc_v2 / cuMemcpyHtoD_v2\n");
        return 1;
    }
    // Overwrite the second half, then read everything back.
    if (memset_d32(device_buffer + (kCount / 2) * sizeof(std::uint32_t), 0xA5A5A5A5u,
                   kCount / 2) != 0) {
        std::fprintf(stderr, "FAIL: cuMemsetD32_v2\n");
        return 1;
    }
    std::vector<std::uint32_t> back(kCount, 0);
    if (dtoh(back.data(), device_buffer, kCount * sizeof(std::uint32_t)) != 0) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH_v2\n");
        return 1;
    }
    for (std::size_t i = 0; i < kCount; ++i) {
        const std::uint32_t expected = i < kCount / 2 ? host[i] : 0xA5A5A5A5u;
        if (back[i] != expected) {
            std::fprintf(stderr, "FAIL: element %zu = %08x, expected %08x\n", i, back[i], expected);
            return 1;
        }
    }
    if (mem_free(device_buffer) != 0 || ctx_destroy(context) != 0) {
        std::fprintf(stderr, "FAIL: cuMemFree_v2 / cuCtxDestroy_v2\n");
        return 1;
    }
    std::printf("PASS: %zu driver _v2 aliases resolve to their implementations\n", aliases.size());
    return 0;
}
