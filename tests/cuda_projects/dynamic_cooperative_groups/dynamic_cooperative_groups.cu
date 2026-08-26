#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <stdio.h>

namespace cg = cooperative_groups;

__global__ void dynamic_cooperative_groups_probe(int* output) {
    const unsigned int lane = threadIdx.x;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    const int parity = static_cast<int>(lane & 1u);
    cg::coalesced_group binary = cg::binary_partition(tile, parity);
    const unsigned int base = lane * 17u;
    output[base + 0u] = static_cast<int>(binary.size());
    output[base + 1u] = static_cast<int>(binary.thread_rank());
    output[base + 2u] = binary.shfl(static_cast<int>(lane), 0u);
    output[base + 3u] = cg::reduce(binary, static_cast<int>(lane), cg::plus<int>());
    output[base + 12u] = binary.shfl(static_cast<int>(lane), binary.size());
    cg::thread_group erased_binary = binary;
    output[base + 13u] = static_cast<int>(erased_binary.size());
    output[base + 14u] = static_cast<int>(erased_binary.thread_rank());
    output[base + 15u] = binary.any(parity == 0);
    output[base + 16u] = binary.all(parity == 0);

    const int label = static_cast<int>(lane % 3u);
    cg::coalesced_group labeled = cg::labeled_partition(tile, label);
    output[base + 4u] = static_cast<int>(labeled.size());
    output[base + 5u] = static_cast<int>(labeled.thread_rank());
    output[base + 6u] = labeled.shfl(static_cast<int>(lane), 0u);
    output[base + 7u] = cg::reduce(labeled, static_cast<int>(lane), cg::greater<int>());

    if ((lane & 3u) != 3u) {
        cg::coalesced_group active = cg::coalesced_threads();
        output[base + 8u] = static_cast<int>(active.size());
        output[base + 9u] = static_cast<int>(active.thread_rank());
        output[base + 10u] = active.shfl(static_cast<int>(lane), 0u);
        output[base + 11u] = cg::reduce(active, static_cast<int>(lane), cg::plus<int>());
    }
}

int main() {
    constexpr unsigned int kThreads = 32u;
    constexpr unsigned int kFields = 17u;
    constexpr unsigned int kValues = kThreads * kFields;
    int* device_output = nullptr;
    int host_output[kValues];

    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(host_output)) != cudaSuccess ||
        cudaMemset(device_output, 0xff, sizeof(host_output)) != cudaSuccess) {
        fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }

    dynamic_cooperative_groups_probe<<<1, kThreads>>>(device_output);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        fprintf(stderr, "FAIL: kernel synchronization: %s\n", cudaGetErrorString(sync_status));
        cudaFree(device_output);
        return 1;
    }
    if (cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "FAIL: copy back\n");
        cudaFree(device_output);
        return 1;
    }
    cudaFree(device_output);

    constexpr int kLabelSize[3] = {11, 11, 10};
    constexpr int kLabelMax[3] = {30, 31, 29};
    for (unsigned int lane = 0; lane < kThreads; ++lane) {
        const unsigned int base = lane * kFields;
        const int parity = static_cast<int>(lane & 1u);
        const int expected_binary_sum = parity == 0 ? 240 : 256;
        if (host_output[base + 0u] != 16 ||
            host_output[base + 1u] != static_cast<int>(lane / 2u) ||
            host_output[base + 2u] != parity ||
            host_output[base + 3u] != expected_binary_sum ||
            host_output[base + 12u] != static_cast<int>(lane) ||
            host_output[base + 13u] != 16 ||
            host_output[base + 14u] != static_cast<int>(lane / 2u) ||
            host_output[base + 15u] != (parity == 0) ||
            host_output[base + 16u] != (parity == 0)) {
            fprintf(stderr,
                    "FAIL: binary group lane %u: size=%d rank=%d leader=%d sum=%d "
                    "invalid=%d erased=(%d,%d) votes=(%d,%d)\n",
                    lane, host_output[base + 0u], host_output[base + 1u],
                    host_output[base + 2u], host_output[base + 3u],
                    host_output[base + 12u], host_output[base + 13u],
                    host_output[base + 14u], host_output[base + 15u],
                    host_output[base + 16u]);
            return 1;
        }

        const int label = static_cast<int>(lane % 3u);
        if (host_output[base + 4u] != kLabelSize[label] ||
            host_output[base + 5u] != static_cast<int>(lane / 3u) ||
            host_output[base + 6u] != label ||
            host_output[base + 7u] != kLabelMax[label]) {
            fprintf(stderr,
                    "FAIL: labeled group lane %u: size=%d rank=%d leader=%d max=%d\n",
                    lane, host_output[base + 4u], host_output[base + 5u],
                    host_output[base + 6u], host_output[base + 7u]);
            return 1;
        }

        if ((lane & 3u) == 3u) {
            if (host_output[base + 8u] != -1 || host_output[base + 9u] != -1 ||
                host_output[base + 10u] != -1 || host_output[base + 11u] != -1) {
                fprintf(stderr, "FAIL: inactive lane %u wrote coalesced-group output\n", lane);
                return 1;
            }
        } else if (host_output[base + 8u] != 24 ||
                   host_output[base + 9u] != static_cast<int>(lane - lane / 4u) ||
                   host_output[base + 10u] != 0 ||
                   host_output[base + 11u] != 360) {
            fprintf(stderr,
                    "FAIL: active group lane %u: size=%d rank=%d leader=%d sum=%d\n",
                    lane, host_output[base + 8u], host_output[base + 9u],
                    host_output[base + 10u], host_output[base + 11u]);
            return 1;
        }
    }

    printf("PASS: coalesced, binary, and labeled cooperative groups\n");
    return 0;
}
