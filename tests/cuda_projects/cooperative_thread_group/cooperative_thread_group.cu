#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <stdio.h>

namespace cg = cooperative_groups;

__device__ int sum_group(cg::thread_group group, int* workspace, int value) {
    const unsigned int rank = group.thread_rank();
    for (unsigned int offset = group.size() / 2u; offset > 0u; offset >>= 1u) {
        workspace[rank] = value;
        group.sync();
        if (rank < offset) {
            value += workspace[rank + offset];
        }
        group.sync();
    }
    return rank == 0u ? value : -1;
}

__global__ void cooperative_thread_group_probe(int* output) {
    __shared__ int block_workspace[64];
    __shared__ int tile_workspace[64];

    const unsigned int tid = threadIdx.x;
    cg::thread_block block = cg::this_thread_block();
    const int block_sum = sum_group(block, block_workspace, static_cast<int>(tid));
    if (tid == 0u) {
        output[0] = block_sum;
    }

    block.sync();
    cg::thread_block_tile<16> tile = cg::tiled_partition<16>(block);
    const unsigned int tile_base = tid - tile.thread_rank();
    const int tile_sum = sum_group(tile, tile_workspace + tile_base,
                                   static_cast<int>(tile.thread_rank()));
    if (tile.thread_rank() == 0u) {
        output[1u + tile.meta_group_rank()] = tile_sum;
    }

    const int value = static_cast<int>(tid);
    output[5u + tid] = tile.shfl(value, 0u);
    output[69u + tid] = tile.shfl_up(value, 1u);
    output[133u + tid] = tile.shfl_down(value, 1u);
    output[197u + tid] = tile.shfl_xor(value, 1u);

    const int in_first_tile = tid < 16u;
    output[261u + tid] = tile.any(in_first_tile);
    output[325u + tid] = tile.all(in_first_tile);

    int prefix = 1;
    for (unsigned int offset = 1u; offset < tile.size(); offset <<= 1u) {
        const int previous = tile.shfl_up(prefix, offset);
        if (tile.thread_rank() >= offset) {
            prefix += previous;
        }
    }
    output[389u + tid] = prefix;

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    int prefix32 = 1;
    for (unsigned int offset = 1u; offset < tile32.size(); offset <<= 1u) {
        const int previous = tile32.shfl_up(prefix32, offset);
        if (tile32.thread_rank() >= offset) {
            prefix32 += previous;
        }
    }
    output[453u + tid] = prefix32;

    output[517u + tid] =
        cg::reduce(tile32, static_cast<int>(tid), cg::plus<int>());
    if (tile32.meta_group_rank() == 0u) {
        output[581u + tid] =
            cg::reduce(tile32, static_cast<int>(tile32.thread_rank()),
                       cg::plus<int>());
    }
    // This write must remain reachable for the other warp when only the first
    // warp executes the nested reduction above.
    output[645u + tid] = static_cast<int>(tid + 7u);
}

int main() {
    constexpr unsigned int kThreads = 64u;
    constexpr unsigned int kOutputs = 709u;
    int* device_output = nullptr;
    int host_output[kOutputs] = {};

    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(host_output)) != cudaSuccess ||
        cudaMemset(device_output, 0, sizeof(host_output)) != cudaSuccess) {
        fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }

    cooperative_thread_group_probe<<<1, kThreads>>>(device_output);
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

    if (host_output[0] != 2016) {
        fprintf(stderr, "FAIL: generic block reduction got %d, expected 2016\n", host_output[0]);
        return 1;
    }
    for (unsigned int tile_index = 0; tile_index < 4u; ++tile_index) {
        if (host_output[1u + tile_index] != 120) {
            fprintf(stderr, "FAIL: generic tile %u reduction got %d, expected 120\n",
                    tile_index, host_output[1u + tile_index]);
            return 1;
        }
    }

    for (unsigned int tid = 0; tid < kThreads; ++tid) {
        const unsigned int rank = tid & 15u;
        const unsigned int base = tid - rank;
        if (host_output[5u + tid] != static_cast<int>(base) ||
            (rank > 0u && host_output[69u + tid] != static_cast<int>(tid - 1u)) ||
            (rank < 15u && host_output[133u + tid] != static_cast<int>(tid + 1u)) ||
            host_output[197u + tid] != static_cast<int>(tid ^ 1u)) {
            fprintf(stderr,
                    "FAIL: tile shuffle mismatch at thread %u: idx=%d up=%d down=%d xor=%d\n",
                    tid, host_output[5u + tid], host_output[69u + tid],
                    host_output[133u + tid], host_output[197u + tid]);
            return 1;
        }

        const int expected_vote = tid < 16u ? 1 : 0;
        if (host_output[261u + tid] != expected_vote ||
            host_output[325u + tid] != expected_vote) {
            fprintf(stderr, "FAIL: tile vote mismatch at thread %u: any=%d all=%d expected=%d\n",
                    tid, host_output[261u + tid], host_output[325u + tid], expected_vote);
            return 1;
        }
        if (host_output[389u + tid] != static_cast<int>(rank + 1u)) {
            fprintf(stderr,
                    "FAIL: tile prefix scan mismatch at thread %u: got %d expected %u\n",
                    tid, host_output[389u + tid], rank + 1u);
            return 1;
        }
        const unsigned int rank32 = tid & 31u;
        if (host_output[453u + tid] != static_cast<int>(rank32 + 1u)) {
            fprintf(stderr,
                    "FAIL: 32-lane prefix scan mismatch at thread %u: got %d expected %u\n",
                    tid, host_output[453u + tid], rank32 + 1u);
            return 1;
        }
        const int expected_reduce = tid < 32u ? 496 : 1520;
        if (host_output[517u + tid] != expected_reduce ||
            (tid < 32u && host_output[581u + tid] != 496) ||
            host_output[645u + tid] != static_cast<int>(tid + 7u)) {
            fprintf(stderr,
                    "FAIL: tile reduction/divergence mismatch at thread %u: "
                    "all=%d nested=%d tail=%d\n",
                    tid, host_output[517u + tid], host_output[581u + tid],
                    host_output[645u + tid]);
            return 1;
        }
    }

    printf("PASS: generic cooperative thread groups and independent 16-lane tiles\n");
    return 0;
}
