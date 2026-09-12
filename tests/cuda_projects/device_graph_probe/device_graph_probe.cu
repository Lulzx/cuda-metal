// Device-side CUDA graph capability probes.
//
// cudaGetCurrentGraphExec() and the cudaStreamGraphTailLaunch sentinel were
// missing, and cudaGraphLaunch was host-only, so kernels that probe device
// graph support could not compile. CuMetal does not execute device graph
// launches: the probe must report no current graph and an attempted launch
// must return cudaErrorNotSupported. The launch below is unconditional so
// dead-code elimination cannot hide an unresolved call.
#include <cuda_runtime.h>

#include <cstdio>

__global__ void probe(int* result) {
    cudaGraphExec_t graph = cudaGetCurrentGraphExec();
    result[0] = (graph == nullptr) ? 1 : 0;
    // Guarded probe: the branch is dead but must still resolve and return the
    // documented error if it were reached.
    result[1] = graph ? cudaGraphLaunch(graph, cudaStreamGraphTailLaunch) : -1;
    // Unconditional launch attempt: must return cudaErrorNotSupported.
    result[2] = (cudaGraphLaunch(graph, cudaStreamGraphTailLaunch) ==
                 cudaErrorNotSupported)
                    ? 1
                    : 0;
}

int main() {
    int* result = nullptr;
    if (cudaMallocManaged(&result, 3 * sizeof(int)) != cudaSuccess) {
        std::printf("FAIL: cudaMallocManaged\n");
        return 1;
    }
    result[0] = -1;
    result[1] = -2;
    result[2] = -1;

    probe<<<1, 1>>>(result);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }

    if (result[0] != 1) {
        std::printf("FAIL: cudaGetCurrentGraphExec should report no current graph, got %d\n",
                    result[0]);
        return 1;
    }
    if (result[1] != -1) {
        std::printf("FAIL: guarded launch should not have run, got %d\n", result[1]);
        return 1;
    }
    if (result[2] != 1) {
        std::printf("FAIL: device cudaGraphLaunch should return cudaErrorNotSupported\n");
        return 1;
    }

    std::printf("PASS: device graph probes report unsupported launch\n");
    return 0;
}
