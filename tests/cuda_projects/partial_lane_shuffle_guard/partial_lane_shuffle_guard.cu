#include <cuda_runtime.h>

#include <cstdio>

struct Words4 {
    unsigned x;
    unsigned y;
    unsigned z;
    unsigned w;
};

// Reproduces a CUDA warp idiom used by PhysX narrowphase code: only the lanes
// selected as shuffle sources initialize their local aggregate.  CUDA shuffle
// semantics make the remaining lane-local values irrelevant.  Treating those
// values as C++ undefined behavior lets Clang incorrectly assume lane < 4 and
// simplify the later lane < 3 guard to lane != 3.  Lane 4 then overwrites the
// first adjacent object.
__global__ void partial_lane_shuffle_guard(const Words4* input, Words4* output) {
    const unsigned lane = threadIdx.x;
    Words4 selected_lane_value;
    if (lane < 4u) {
        selected_lane_value = input[lane];
    }

    const unsigned lane_three =
        __shfl_sync(0xffffffffu, selected_lane_value.x, 3);
    if (lane_three != 0u && lane < 3u) {
        const Words4 value = lane == 0u
                                 ? Words4{11u, 12u, 13u, 14u}
                                 : lane == 1u
                                       ? Words4{21u, 22u, 23u, 24u}
                                       : Words4{31u, 32u, 33u, 0u};
        output[lane] = value;
    }
}

static bool equal(const Words4& lhs, const Words4& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}

int main() {
    const Words4 host_input[4] = {
        {1u, 2u, 3u, 4u},
        {5u, 6u, 7u, 8u},
        {9u, 10u, 11u, 12u},
        {13u, 14u, 15u, 16u},
    };
    const Words4 initial_output[5] = {
        {101u, 102u, 103u, 104u},
        {111u, 112u, 113u, 114u},
        {121u, 122u, 123u, 124u},
        {0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu},
        {0x13579bdfu, 0x2468ace0u, 0x55aa55aau, 0xaa55aa55u},
    };
    const Words4 expected_output[5] = {
        {11u, 12u, 13u, 14u},
        {21u, 22u, 23u, 24u},
        {31u, 32u, 33u, 0u},
        initial_output[3],
        initial_output[4],
    };

    Words4* device_input = nullptr;
    Words4* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(host_input)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(initial_output)) !=
            cudaSuccess ||
        cudaMemcpy(device_input, host_input, sizeof(host_input),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_output, initial_output, sizeof(initial_output),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation or upload\n");
        return 1;
    }

    partial_lane_shuffle_guard<<<1, 32>>>(device_input, device_output);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    Words4 host_output[5] = {};
    const cudaError_t copy_status =
        cudaMemcpy(host_output, device_output, sizeof(host_output),
                   cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    cudaFree(device_input);
    if (sync_status != cudaSuccess || copy_status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(sync_status), cudaGetErrorString(copy_status));
        return 1;
    }

    for (unsigned i = 0; i < 5u; ++i) {
        if (!equal(host_output[i], expected_output[i])) {
            std::fprintf(stderr,
                         "FAIL: output[%u] = {%u,%u,%u,%u}, expected {%u,%u,%u,%u}\n",
                         i, host_output[i].x, host_output[i].y, host_output[i].z,
                         host_output[i].w, expected_output[i].x,
                         expected_output[i].y, expected_output[i].z,
                         expected_output[i].w);
            return 1;
        }
    }

    std::printf("PASS: partial-lane shuffle preserved both adjacent sentinels\n");
    return 0;
}
