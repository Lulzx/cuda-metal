// cudaMemcpyAsync ordered against a kernel, above the MTLHeap threshold.
//
// The shape is NVIDIA's asyncAPI sample: copy in, increment on the GPU, copy
// out, check every element. It is a first-party test because of how it failed
// once -- silently, on some elements, only on large allocations, with no error
// reported anywhere.
//
// cudaMemcpyAsync is a Metal blit and a kernel is a compute dispatch. Metal
// turns encoder order inside a command buffer into a memory dependency only for
// hazard-tracked resources, and an MTLHeapDescriptor leaves hazard tracking off
// by default, which the resources allocated from it inherit. CuMetal allocates
// from a heap above CUMETAL_MTLHEAP_THRESHOLD_BYTES, so its large buffers are
// the untracked ones. Batching the blit and the dispatch into one command
// buffer therefore let the copy-out read elements the kernel had not written:
// they came back one increment short, a few hundred elements in.
//
// The allocation is far above the 4 MiB default on purpose. A small one takes
// the individually-allocated path, which is tracked, and would pass whether the
// ordering were right or not.
#include <cstdio>

__global__ void increment_kernel(int* data, int value, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] + value;
}

int main() {
    const int n = 4 * 1024 * 1024;  // 16 MiB of ints
    const size_t bytes = static_cast<size_t>(n) * sizeof(int);
    const int increment = 26;

    int* host = nullptr;
    if (cudaMallocHost(reinterpret_cast<void**>(&host), bytes) != cudaSuccess) {
        std::printf("SKIP: cudaMallocHost failed\n");
        return 0;
    }
    int* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), bytes) != cudaSuccess) {
        std::printf("SKIP: cudaMalloc failed\n");
        return 0;
    }
    for (int i = 0; i < n; ++i) host[i] = i & 0xFF;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate\n");
        return 1;
    }

    // Two runs: the second reuses a warm stream, where an open batch is most
    // likely to still be holding the previous iteration's encoder.
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < n; ++i) host[i] = i & 0xFF;

        cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream);
        increment_kernel<<<(n + 255) / 256, 256, 0, stream>>>(device, increment, n);
        cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream);

        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: cudaStreamSynchronize (pass %d)\n", pass);
            return 1;
        }
        if (cudaGetLastError() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: launch error (pass %d): %s\n", pass,
                         cudaGetErrorString(cudaGetLastError()));
            return 1;
        }

        int stale = 0;
        int first_bad = -1;
        for (int i = 0; i < n; ++i) {
            const int want = (i & 0xFF) + increment;
            if (host[i] != want) {
                if (first_bad < 0) first_bad = i;
                ++stale;
            }
        }
        if (stale != 0) {
            std::fprintf(stderr,
                         "FAIL: pass %d, %d of %d elements wrong; first at %d, got %d want %d\n",
                         pass, stale, n, first_bad, host[first_bad],
                         (first_bad & 0xFF) + increment);
            return 1;
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(device);
    cudaFreeHost(host);
    std::printf("PASS: async memcpy is ordered against a kernel on heap allocations\n");
    return 0;
}
