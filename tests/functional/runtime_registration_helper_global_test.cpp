// A module-scope writable global referenced only inside a device helper.
//
// The typed compiler gives the kernel a hidden buffer binding for the promoted
// global and threads it through the helper call chain; registration metadata
// must discover the same symbol through the reachable helper graph so the
// binding is backed by the registered host shadow. When either half is missing
// the kernel silently reads zeroed storage -- both launches observe 0 + 7 and
// nothing persists -- instead of the registered initializer.
#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
void** __cudaRegisterFatBinary(const void* fat_cubin);
void __cudaUnregisterFatBinary(void** fat_cubin_handle);
void __cudaRegisterFunction(void** fat_cubin_handle,
                            const void* host_function,
                            char* device_function,
                            const char* device_name,
                            int thread_limit,
                            void* thread_id,
                            void* block_id,
                            void* block_dim,
                            void* grid_dim,
                            int* warp_size);
void __cudaRegisterVar(void** fat_cubin_handle,
                       char* host_var,
                       char* device_address,
                       const char* device_name,
                       int ext,
                       std::size_t size,
                       int constant,
                       int global);
}

namespace {

constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr std::uint32_t kFatbinBlobMagic = 0xBA55ED50u;
constexpr std::uint32_t kGuard = 0xa5a5a5a5u;
constexpr std::uint32_t kInitialState = 5;

struct FatbinWrapper {
    std::uint32_t magic = kFatbinWrapperMagic;
    std::uint32_t version = 1;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

struct FatbinBlobHeader {
    std::uint32_t magic = kFatbinBlobMagic;
    std::uint16_t version = 1;
    std::uint16_t header_size = 16;
    std::uint64_t fat_size = 0;
};

// kernel -> wrapper -> leaf -> helper_state. `helper_state` appears nowhere in
// the entry body. (Reachability exclusion is covered by the compiler unit
// tests; the legacy JIT path lowers every device function in the module.)
const char kPtx[] = R"PTX(
.version 7.0
.target sm_80
.address_size 64

.visible .global .align 4 .u32 helper_state = 5;

.visible .func (.param .b32 rv) leaf()
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<3>;
    mov.b64 %rd1, helper_state;
    ld.global.u32 %r1, [%rd1];
    add.u32 %r2, %r1, 7;
    st.global.u32 [%rd1], %r2;
    st.param.b32 [rv], %r2;
    ret;
}

.visible .func (.param .b32 rv) wrapper()
{
    .reg .b32 %r<2>;
    .param .b32 inner;
    call.uni (inner), leaf, ();
    ld.param.b32 %r1, [inner];
    st.param.b32 [rv], %r1;
    ret;
}

.visible .entry helper_only_global(.param .u64 helper_only_global_param_0)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<2>;
    .param .b32 slot;
    ld.param.u64 %rd1, [helper_only_global_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    call.uni (slot), wrapper, ();
    ld.param.b32 %r1, [slot];
    st.global.u32 [%rd2], %r1;
    ret;
}
)PTX";

void helper_only_global_host_stub() {}

// CUDA's host shadow for `__device__ unsigned helper_state = 5;`. Clang leaves
// the shadow in zero-filled BSS and keeps the initializer in the PTX, which is
// where registration recovers it from.
std::uint32_t g_helper_state_shadow = 0;

bool launch_and_read(void* output, std::uint32_t* observed) {
    void* arg_output = output;
    void* args[] = {&arg_output, nullptr};
    const dim3 block_dim(1, 1, 1);
    const dim3 grid_dim(1, 1, 1);
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&helper_only_global_host_stub),
                         grid_dim, block_dim, args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return false;
    }
    std::uint32_t host[3] = {0, 0, 0};
    if (cudaMemcpy(host, output, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return false;
    }
    if (host[1] != kGuard || host[2] != kGuard) {
        std::fprintf(stderr, "FAIL: output guard words were overwritten (%u, %u)\n",
                     host[1], host[2]);
        return false;
    }
    *observed = host[0];
    return true;
}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    const std::string ptx(kPtx);
    std::vector<std::uint8_t> fatbin_blob(sizeof(FatbinBlobHeader) + ptx.size() + 1, 0);
    FatbinBlobHeader header{};
    header.fat_size = static_cast<std::uint64_t>(ptx.size() + 1);
    std::memcpy(fatbin_blob.data(), &header, sizeof(header));
    std::memcpy(fatbin_blob.data() + sizeof(header), ptx.data(), ptx.size());

    FatbinWrapper wrapper{};
    wrapper.data = fatbin_blob.data();

    void** fatbin_handle = __cudaRegisterFatBinary(&wrapper);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary returned null\n");
        return 1;
    }

    char device_function[] = "helper_only_global";
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&helper_only_global_host_stub),
                           device_function, nullptr, 0,
                           nullptr, nullptr, nullptr, nullptr, nullptr);
    // Clang passes the device-side name string for both the address and the
    // name arguments; the host shadow is the storage the runtime maps.
    char device_var_name[] = "helper_state";
    __cudaRegisterVar(fatbin_handle,
                      reinterpret_cast<char*>(&g_helper_state_shadow),
                      device_var_name, device_var_name, 0,
                      sizeof(g_helper_state_shadow), 0, 1);

    void* output = nullptr;
    if (cudaMalloc(&output, 3 * sizeof(std::uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    const std::uint32_t seed[3] = {kGuard, kGuard, kGuard};
    if (cudaMemcpy(output, seed, sizeof(seed), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    // The registered initializer is 5, so the first launch stores 12 and the
    // second must observe that persisted value and store 19.
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    if (!launch_and_read(output, &first)) return 1;
    if (!launch_and_read(output, &second)) return 1;

    if (first != kInitialState + 7) {
        std::fprintf(stderr,
                     "FAIL: first launch observed %u, expected the registered "
                     "initializer plus seven (%u)\n",
                     first, kInitialState + 7);
        return 1;
    }
    if (second != kInitialState + 14) {
        std::fprintf(stderr,
                     "FAIL: second launch observed %u, expected the helper-only "
                     "global to persist (%u)\n",
                     second, kInitialState + 14);
        return 1;
    }

    std::uint32_t device_state = 0;
    if (cudaMemcpyFromSymbol(&device_state, &g_helper_state_shadow,
                             sizeof(device_state), 0,
                             cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyFromSymbol failed\n");
        return 1;
    }
    if (device_state != kInitialState + 14) {
        std::fprintf(stderr,
                     "FAIL: host view of the helper-only global is %u, expected %u\n",
                     device_state, kInitialState + 14);
        return 1;
    }

    cudaFree(output);
    __cudaUnregisterFatBinary(fatbin_handle);
    std::printf("HELPER_ONLY_GLOBAL_OK %u %u\n", first, second);
    return 0;
}
