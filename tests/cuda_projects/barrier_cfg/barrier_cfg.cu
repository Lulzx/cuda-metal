#include <cuda_runtime.h>

#include <cstdio>

__device__ __noinline__ int barrier_round(int* tile, unsigned lane,
                                          int round, int skip_round) {
    tile[lane] = static_cast<int>(lane) + round * 3;
    __syncthreads();

    int contribution = 0;
    if (round == skip_round) {
        contribution = tile[(lane + 1) & 31] - tile[lane];
        __syncthreads();
        return contribution;
    }

    if ((round & 1) == 0) {
        contribution = tile[(lane + 1) & 31] + tile[lane];
    } else {
        contribution = tile[lane] - tile[(lane + 31) & 31];
    }
    __syncthreads();
    return contribution;
}

__global__ void barrier_cfg_probe(int* output, int rounds, int skip_round) {
    __shared__ int tile[32];
    const unsigned lane = threadIdx.x;
    int sum = 0;
    for (int round = 0; round < rounds; ++round) {
        sum += barrier_round(tile, lane, round, skip_round);
    }
    output[lane] = sum;
}

int main() {
    constexpr int kThreads = 32;
    constexpr int kRounds = 5;
    constexpr int kSkipRound = 2;
    int expected[kThreads] = {};
    for (int lane = 0; lane < kThreads; ++lane) {
        for (int round = 0; round < kRounds; ++round) {
            const int current = lane + round * 3;
            const int next = ((lane + 1) & 31) + round * 3;
            const int previous = ((lane + 31) & 31) + round * 3;
            if (round == kSkipRound) {
                expected[lane] += next - current;
            } else if ((round & 1) == 0) {
                expected[lane] += next + current;
            } else {
                expected[lane] += current - previous;
            }
        }
    }

    int* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(expected)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }
    barrier_cfg_probe<<<1, kThreads>>>(device_output, kRounds, kSkipRound);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    int actual[kThreads] = {};
    const cudaError_t copy_status =
        cudaMemcpy(actual, device_output, sizeof(actual), cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    if (sync_status != cudaSuccess || copy_status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or copy: %s / %s\n",
                     cudaGetErrorString(sync_status),
                     cudaGetErrorString(copy_status));
        return 1;
    }
    for (int lane = 0; lane < kThreads; ++lane) {
        if (actual[lane] != expected[lane]) {
            std::fprintf(stderr, "FAIL: lane %d got %d expected %d\n", lane,
                         actual[lane], expected[lane]);
            return 1;
        }
    }
    std::puts("PASS: uniform multi-exit barrier CFG preserves all 32 lanes");
    return 0;
}
