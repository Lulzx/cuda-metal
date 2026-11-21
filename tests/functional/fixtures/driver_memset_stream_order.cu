// Driver-API 16/32-bit and pitched memsets are ordered like CUDA's (#211):
// an *Async fill runs after the work already queued on its stream, a
// synchronous fill runs after prior device work, a cross-stream fill waits
// for its event, and a captured fill runs when the graph is launched, not
// while it is being captured.
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

// Slow on purpose: the fill is enqueued while this kernel is still running.
__global__ void slow_fill(unsigned* out, unsigned n, unsigned value, unsigned spin) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = i;
    for (unsigned k = 0; k < spin; ++k) acc = acc * 1664525u + 1013904223u;
    // The loop result feeds the store, so it cannot be optimised away.
    if (i < n) out[i] = acc == 0xffffffffu ? value ^ 1u : value;
}

static constexpr unsigned kN = 1u << 16;
static constexpr unsigned kSpin = 20000;
static unsigned host[kN];

static int fail(const char* what) {
    std::printf("FAIL: %s\n", what);
    return 1;
}

static bool all_bytes(const void* data, size_t bytes, size_t elem, unsigned value) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; i += elem)
        if (std::memcmp(p + i, &value, elem) != 0) return false;
    return true;
}

static bool read_back(CUdeviceptr ptr) {
    return cuMemcpyDtoH(host, ptr, sizeof host) == CUDA_SUCCESS;
}

int main() {
    if (cuInit(0) != CUDA_SUCCESS) return fail("cuInit");
    CUdevice device = 0;
    CUcontext context = nullptr;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS ||
        cuCtxSetCurrent(context) != CUDA_SUCCESS)
        return fail("primary context");
    unsigned* buf = nullptr;
    if (cudaMalloc(&buf, sizeof host) != cudaSuccess) return fail("cudaMalloc");
    const auto ptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(buf));
    const unsigned blocks = kN / 256;

    cudaStream_t a = nullptr, b = nullptr;
    if (cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking) != cudaSuccess)
        return fail("stream create");
    auto sa = reinterpret_cast<CUstream>(a), sb = reinterpret_cast<CUstream>(b);

    // 1. Same stream: the async fill must land after the kernel's writes.
    slow_fill<<<blocks, 256, 0, a>>>(buf, kN, 0xdeadbeefu, kSpin);
    if (cuMemsetD32Async(ptr, 0x12345678u, kN, sa) != CUDA_SUCCESS) return fail("cuMemsetD32Async");
    if (cuStreamSynchronize(sa) != CUDA_SUCCESS || !read_back(ptr)) return fail("sync 1");
    if (!all_bytes(host, sizeof host, 4, 0x12345678u)) return fail("cuMemsetD32Async ran before the queued kernel");

    slow_fill<<<blocks, 256, 0, a>>>(buf, kN, 0xdeadbeefu, kSpin);
    if (cuMemsetD16Async(ptr, 0xa1b2u, kN * 2, sa) != CUDA_SUCCESS) return fail("cuMemsetD16Async");
    if (cuStreamSynchronize(sa) != CUDA_SUCCESS || !read_back(ptr)) return fail("sync 2");
    if (!all_bytes(host, sizeof host, 2, 0xa1b2u)) return fail("cuMemsetD16Async ran before the queued kernel");

    // Pitched async forms: 256-byte rows, 192 bytes filled per row.
    const size_t pitch = 256, rows = sizeof host / 256;
    slow_fill<<<blocks, 256, 0, a>>>(buf, kN, 0u, kSpin);
    if (cuMemsetD2D32Async(ptr, pitch, 0x0badf00du, 48, rows, sa) != CUDA_SUCCESS)
        return fail("cuMemsetD2D32Async");
    if (cuStreamSynchronize(sa) != CUDA_SUCCESS || !read_back(ptr)) return fail("sync 3");
    for (size_t r = 0; r < rows; ++r) {
        const auto* row = reinterpret_cast<const unsigned char*>(host) + r * pitch;
        if (!all_bytes(row, 192, 4, 0x0badf00du) || !all_bytes(row + 192, 64, 4, 0u))
            return fail("cuMemsetD2D32Async row contents");
    }
    slow_fill<<<blocks, 256, 0, a>>>(buf, kN, 0u, kSpin);
    if (cuMemsetD2D16Async(ptr, pitch, 0x7e7fu, 96, rows, sa) != CUDA_SUCCESS ||
        cuMemsetD2D8Async(ptr + 192, pitch, 0x5au, 64, rows, sa) != CUDA_SUCCESS)
        return fail("cuMemsetD2D16Async/D2D8Async");
    if (cuStreamSynchronize(sa) != CUDA_SUCCESS || !read_back(ptr)) return fail("sync 4");
    for (size_t r = 0; r < rows; ++r) {
        const auto* row = reinterpret_cast<const unsigned char*>(host) + r * pitch;
        if (!all_bytes(row, 192, 2, 0x7e7fu) || !all_bytes(row + 192, 64, 1, 0x5au))
            return fail("cuMemsetD2D16Async/D2D8Async row contents");
    }

    // 2. Synchronous fill after a kernel on the legacy default stream.
    slow_fill<<<blocks, 256>>>(buf, kN, 0xdeadbeefu, kSpin);
    if (cuMemsetD32(ptr, 0x0f0f0f0fu, kN) != CUDA_SUCCESS || !read_back(ptr))
        return fail("cuMemsetD32");
    if (!all_bytes(host, sizeof host, 4, 0x0f0f0f0fu)) return fail("cuMemsetD32 ran before the queued kernel");
    slow_fill<<<blocks, 256>>>(buf, kN, 0xdeadbeefu, kSpin);
    if (cuMemsetD2D16(ptr, 256, 0x3c3cu, 128, rows) != CUDA_SUCCESS || !read_back(ptr))
        return fail("cuMemsetD2D16");
    if (!all_bytes(host, sizeof host, 2, 0x3c3cu)) return fail("cuMemsetD2D16 ran before the queued kernel");
    slow_fill<<<blocks, 256>>>(buf, kN, 0xdeadbeefu, kSpin);
    if (cudaMemset2D(buf, 256, 0x11, 256, rows) != cudaSuccess || !read_back(ptr))
        return fail("cudaMemset2D");
    if (!all_bytes(host, sizeof host, 1, 0x11u)) return fail("cudaMemset2D ran before the queued kernel");

    // 3. Cross-stream: stream b waits on an event recorded after a's kernel.
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return fail("event");
    slow_fill<<<blocks, 256, 0, a>>>(buf, kN, 0xdeadbeefu, kSpin);
    if (cudaEventRecord(ev, a) != cudaSuccess || cudaStreamWaitEvent(b, ev, 0) != cudaSuccess)
        return fail("event record/wait");
    if (cuMemsetD32Async(ptr, 0x600dcafeu, kN, sb) != CUDA_SUCCESS) return fail("cross-stream memset");
    if (cuStreamSynchronize(sb) != CUDA_SUCCESS || !read_back(ptr)) return fail("sync 5");
    if (!all_bytes(host, sizeof host, 4, 0x600dcafeu)) return fail("cross-stream memset ignored the event");

    // 4. Captured: nothing is written until the graph is launched.
    if (cuMemsetD32(ptr, 0u, kN) != CUDA_SUCCESS) return fail("clear");
    if (cudaStreamBeginCapture(a, cudaStreamCaptureModeGlobal) != cudaSuccess) return fail("begin capture");
    if (cuMemsetD32Async(ptr, 0xabcdef01u, kN, sa) != CUDA_SUCCESS) return fail("captured memset");
    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(a, &graph) != cudaSuccess || graph == nullptr) return fail("end capture");
    if (!read_back(ptr) || !all_bytes(host, sizeof host, 4, 0u)) return fail("captured memset ran during capture");
    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) return fail("instantiate");
    if (cudaGraphLaunch(exec, a) != cudaSuccess || cudaStreamSynchronize(a) != cudaSuccess) return fail("graph launch");
    if (!read_back(ptr) || !all_bytes(host, sizeof host, 4, 0xabcdef01u)) return fail("graph replay of memset");

    // 5. Refusals.
    if (cuMemsetD32Async(0, 1u, 4, sa) != CUDA_ERROR_INVALID_VALUE) return fail("null destination accepted");
    if (cuMemsetD2D32Async(ptr, 8, 1u, 4, 2, sa) != CUDA_ERROR_INVALID_VALUE)
        return fail("pitch narrower than a row accepted");
    if (cuMemsetD32(ptr, 7u, 0) != CUDA_SUCCESS) return fail("empty fill refused");

    std::printf("PASS: driver memsets are ordered on their streams\n");
    return 0;
}
