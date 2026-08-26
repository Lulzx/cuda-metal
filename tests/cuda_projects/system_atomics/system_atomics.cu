#include <cuda_runtime.h>

#include <cstdio>

__global__ void system_atomic_kernel(int* slots) {
    const int tid = static_cast<int>(threadIdx.x);
    atomicAdd_system(&slots[0], 3);
    atomicExch_system(&slots[1], tid);
    atomicMax_system(&slots[2], tid);
    atomicMin_system(&slots[3], tid);
    atomicInc_system(reinterpret_cast<unsigned*>(&slots[4]), 17);
    atomicDec_system(reinterpret_cast<unsigned*>(&slots[5]), 137);
    if (tid == 0) {
        atomicCAS_system(&slots[6], 0, 77);
        atomicAnd_system(&slots[7], 0x3c);
        atomicOr_system(&slots[8], 0x12);
        atomicXor_system(&slots[9], 0x55);
    }
}

int main() {
    void* unsupported = nullptr;
    if (cudaMallocManaged(&unsupported, sizeof(int), cudaMemAttachHost) !=
        cudaErrorInvalidValue) {
        std::printf("FAIL: unimplemented cudaMemAttachHost was accepted\n");
        return 1;
    }
    void* explicit_global = nullptr;
    if (cudaMallocManaged(&explicit_global, sizeof(int), cudaMemAttachGlobal) !=
        cudaSuccess) {
        std::printf("FAIL: explicit cudaMemAttachGlobal was rejected\n");
        return 1;
    }
    cudaFree(explicit_global);
    void* legacy_zero = nullptr;
    if (cudaMallocManaged(&legacy_zero, sizeof(int), 0) != cudaSuccess) {
        std::printf("FAIL: legacy zero managed-memory flags were rejected\n");
        return 1;
    }
    cudaFree(legacy_zero);
    int* slots = nullptr;
    if (cudaMallocManaged(&slots, 10 * sizeof(int)) != cudaSuccess) {
        std::printf("FAIL: cudaMallocManaged\n");
        return 1;
    }
    for (int i = 0; i < 10; ++i) slots[i] = 0;
    slots[2] = -1;
    slots[3] = 999;
    slots[7] = 0xff;

    system_atomic_kernel<<<1, 32>>>(slots);
    // Registered launches are asynchronous. This host atomic can execute before
    // or during the GPU dispatch; either ordering must contribute exactly once
    // to the same managed-memory word.
    __sync_fetch_and_add(&slots[0], 7);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }

    const int expected[10] = {103, 0, 31, 0, 14, 106, 77, 0x3c, 0x12, 0x55};
    int failures = 0;
    for (int i = 0; i < 10; ++i) {
        if (i == 1) continue;  // Any one of the 32 racing thread IDs may win.
        if (slots[i] != expected[i]) {
            std::printf("FAIL: system atomic slot %d got %d expected %d\n",
                        i, slots[i], expected[i]);
            ++failures;
        }
    }
    if (slots[1] < 0 || slots[1] >= 32) {
        std::printf("FAIL: atomicExch_system result %d is not a thread id\n", slots[1]);
        ++failures;
    }
    cudaFree(slots);
    if (failures != 0) return 1;
    std::printf("PASS: system atomics share managed-memory bytes with the host\n");
    return 0;
}
