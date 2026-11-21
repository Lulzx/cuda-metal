// #225: device pointers embedded in data survive copies between device memory
// and pinned host memory, in both directions, synchronous and stream-ordered,
// through the runtime and the driver API. PhysX writes pointers into
// contact-report buffers on the GPU, copies them into pinned memory and
// dereferences them on the host.
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int kN = 256;

// Each entry points into either a pinned (mapped) array or a device array.
__global__ void write_pointers(const int* pinned_dev, const int* device_values, const int** table) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kN) table[i] = (i & 1) ? device_values + i : pinned_dev + i;
}

__global__ void read_pointers(const int* const* table, int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kN) out[i] = *table[i];
}

static int fail(const char* what) {
    std::printf("FAIL: %s\n", what);
    return 1;
}

static int expected(int i) { return (i & 1) ? 1000 + i : 3 * i; }

// Host side: entries read back must be dereferenceable and point at the same
// element: pinned entries are the host pointers, device entries are the
// public device pointers, exactly as CUDA hands them back.
static bool check_host_table(const int* const* table, const int* pinned, const int* device_values,
                             const char* what) {
    for (int i = 0; i < kN; ++i) {
        if ((i & 1) == 0) {
            if (table[i] != pinned + i || *table[i] != expected(i)) {
                std::printf("FAIL: %s: pinned entry %d is %p, want %p\n", what, i,
                            static_cast<const void*>(table[i]),
                            static_cast<const void*>(pinned + i));
                return false;
            }
        } else if (table[i] != device_values + i) {
            std::printf("FAIL: %s: device entry %d is %p, want %p\n", what, i,
                        static_cast<const void*>(table[i]),
                        static_cast<const void*>(device_values + i));
            return false;
        }
    }
    return true;
}

int main() {
    if (cuInit(0) != CUDA_SUCCESS) return fail("cuInit");
    CUdevice device = 0;
    CUcontext context = nullptr;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS ||
        cuCtxSetCurrent(context) != CUDA_SUCCESS)
        return fail("primary context");

    int* pinned = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void**>(&pinned), kN * sizeof(int), cudaHostAllocMapped) !=
        cudaSuccess)
        return fail("cudaHostAlloc");
    int* pinned_dev = nullptr;
    if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&pinned_dev), pinned, 0) != cudaSuccess)
        return fail("cudaHostGetDevicePointer");
    int host_values[kN];
    for (int i = 0; i < kN; ++i) {
        pinned[i] = 3 * i;
        host_values[i] = 1000 + i;
    }
    int* device_values = nullptr;
    const int** table = nullptr;
    int* out = nullptr;
    if (cudaMalloc(&device_values, sizeof host_values) != cudaSuccess ||
        cudaMalloc(&table, kN * sizeof(int*)) != cudaSuccess ||
        cudaMalloc(&out, kN * sizeof(int)) != cudaSuccess)
        return fail("cudaMalloc");
    cudaMemcpy(device_values, host_values, sizeof host_values, cudaMemcpyHostToDevice);

    const int** pinned_table = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void**>(&pinned_table), kN * sizeof(int*), 0) != cudaSuccess)
        return fail("cudaHostAlloc table");
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        return fail("stream");

    // 1. GPU-written pointers, copied device-to-host into pinned memory.
    const struct {
        const char* what;
        int mode;
    } d2h[] = {{"cudaMemcpyAsync D2H", 0},
               {"cudaMemcpyAsync Default", 1},
               {"cuMemcpyDtoHAsync", 2},
               {"cudaMemcpy D2H", 3}};
    for (const auto& c : d2h) {
        for (int i = 0; i < kN; ++i) pinned_table[i] = nullptr;
        write_pointers<<<1, kN, 0, stream>>>(pinned_dev, device_values, table);
        cudaError_t e = cudaSuccess;
        switch (c.mode) {
        case 0: e = cudaMemcpyAsync(pinned_table, table, kN * sizeof(int*), cudaMemcpyDeviceToHost, stream); break;
        case 1: e = cudaMemcpyAsync(pinned_table, table, kN * sizeof(int*), cudaMemcpyDefault, stream); break;
        case 2:
            e = cuMemcpyDtoHAsync(pinned_table, reinterpret_cast<CUdeviceptr>(table), kN * sizeof(int*),
                                  reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS
                    ? cudaSuccess
                    : cudaErrorUnknown;
            break;
        default:
            cudaStreamSynchronize(stream);
            e = cudaMemcpy(pinned_table, table, kN * sizeof(int*), cudaMemcpyDeviceToHost);
            break;
        }
        if (e != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) return fail(c.what);
        if (!check_host_table(pinned_table, pinned, device_values, c.what)) return 1;
    }

    // 2. Host-written pointers in pinned memory, copied host-to-device and
    //    dereferenced by a kernel queued behind the copy.
    const char* h2d[] = {"cudaMemcpyAsync H2D", "cudaMemcpyAsync Default H2D", "cuMemcpyHtoDAsync"};
    for (int mode = 0; mode < 3; ++mode) {
        for (int i = 0; i < kN; ++i) pinned_table[i] = (i & 1) ? device_values + i : pinned + i;
        cudaMemsetAsync(out, 0, kN * sizeof(int), stream);
        cudaError_t e = cudaSuccess;
        if (mode == 0)
            e = cudaMemcpyAsync(table, pinned_table, kN * sizeof(int*), cudaMemcpyHostToDevice, stream);
        else if (mode == 1)
            e = cudaMemcpyAsync(table, pinned_table, kN * sizeof(int*), cudaMemcpyDefault, stream);
        else
            e = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(table), pinned_table, kN * sizeof(int*),
                                  reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS
                    ? cudaSuccess
                    : cudaErrorUnknown;
        read_pointers<<<1, kN, 0, stream>>>(table, out);
        int got[kN];
        if (e != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess ||
            cudaMemcpy(got, out, sizeof got, cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail(h2d[mode]);
        for (int i = 0; i < kN; ++i)
            if (got[i] != expected(i)) {
                std::printf("FAIL: %s: kernel read %d through entry %d, want %d\n", h2d[mode], got[i],
                            i, expected(i));
                return 1;
            }
    }

    // 3. Plain data that is not a pointer is copied unchanged.
    auto* raw = reinterpret_cast<std::uint64_t*>(pinned_table);
    for (int i = 0; i < kN; ++i) raw[i] = 0x0123456789abcdefull + static_cast<std::uint64_t>(i);
    if (cudaMemcpyAsync(table, pinned_table, kN * 8, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(pinned_table, table, kN * 8, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return fail("plain data round trip");
    for (int i = 0; i < kN; ++i)
        if (raw[i] != 0x0123456789abcdefull + static_cast<std::uint64_t>(i)) return fail("plain data changed");

    std::printf("PASS: embedded pointers survive pinned copies\n");
    return 0;
}
