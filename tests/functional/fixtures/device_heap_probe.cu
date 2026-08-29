#include <cuda_runtime.h>

#include <cstdio>

class DeviceContainer {
  public:
    __device__ virtual ~DeviceContainer() {}
    __device__ virtual void push(int value) = 0;
    __device__ virtual bool pop(int& value) = 0;
};

class DeviceStack final : public DeviceContainer {
  public:
    __device__ DeviceStack() : count_(0) {}

    __device__ void push(int value) override {
        if (count_ < 4) values_[count_++] = value;
    }

    __device__ bool pop(int& value) override {
        if (count_ == 0) return false;
        value = values_[--count_];
        return true;
    }

  private:
    int values_[4];
    int count_;
};

__global__ void allocate_values(int** slot, int* status) {
    if (threadIdx.x != 0) return;
    int* values = new int[8];
    if (values == nullptr) {
        *status = -1;
        return;
    }
    for (int i = 0; i < 8; ++i) values[i] = 10 + i;
    *slot = values;
    *status = 1;
}

__global__ void consume_values(int** slot, int* status) {
    if (threadIdx.x != 0) return;
    int* values = *slot;
    int sum = 0;
    for (int i = 0; i < 8; ++i) sum += values[i];
    *status = sum;
}

__global__ void release_values(int** slot, unsigned long long* released_address) {
    if (threadIdx.x != 0) return;
    int* values = *slot;
    *released_address = reinterpret_cast<unsigned long long>(values);
    delete[] values;
    *slot = nullptr;
}

__global__ void verify_reuse_and_exhaustion(
    int** slot, const unsigned long long* released_address,
    unsigned long long* too_large_address, int* status) {
    if (threadIdx.x != 0) return;
    int* reused = new int[8];
    void* too_large = malloc(8192);
    const bool reused_block =
        reinterpret_cast<unsigned long long>(reused) == *released_address;
    *too_large_address = reinterpret_cast<unsigned long long>(too_large);
    *status = (reused != nullptr ? 1 : 0) | (reused_block ? 2 : 0);
    delete[] reused;
    free(too_large);
    *slot = nullptr;
}

__global__ void create_polymorphic(DeviceContainer** slot) {
    if (threadIdx.x == 0) *slot = new DeviceStack();
}

__global__ void exercise_polymorphic(DeviceContainer** slot, int* output) {
    if (threadIdx.x != 0 || *slot == nullptr) return;
    DeviceContainer* container = *slot;
    container->push(17);
    container->push(29);
    int first = -1;
    int second = -1;
    const bool first_ok = container->pop(first);
    const bool second_ok = container->pop(second);
    output[0] = first_ok ? first : -1;
    output[1] = second_ok ? second : -1;
}

__global__ void destroy_polymorphic(DeviceContainer** slot) {
    if (threadIdx.x != 0) return;
    delete *slot;
    *slot = nullptr;
}

int main() {
    if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4096) != cudaSuccess) return 1;

    int** slot = nullptr;
    int* status = nullptr;
    unsigned long long* released_address = nullptr;
    unsigned long long* too_large_address = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&slot), sizeof(*slot)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&status), sizeof(*status)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&released_address),
                   sizeof(*released_address)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&too_large_address),
                   sizeof(*too_large_address)) != cudaSuccess) {
        return 1;
    }
    if (cudaMemset(slot, 0, sizeof(*slot)) != cudaSuccess ||
        cudaMemset(status, 0, sizeof(*status)) != cudaSuccess) {
        return 1;
    }

    allocate_values<<<1, 1>>>(slot, status);
    consume_values<<<1, 1>>>(slot, status);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    int host_status = 0;
    if (cudaMemcpy(&host_status, status, sizeof(host_status),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        host_status != 108) {
        std::fprintf(stderr, "FAIL: device allocation did not persist (status=%d)\n",
                     host_status);
        return 1;
    }

    release_values<<<1, 1>>>(slot, released_address);
    unsigned long long host_released_address = 0;
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(&host_released_address, released_address,
                   sizeof(host_released_address), cudaMemcpyDeviceToHost) != cudaSuccess ||
        host_released_address == 0) {
        return 1;
    }
    verify_reuse_and_exhaustion<<<1, 1>>>(slot, released_address,
                                           too_large_address, status);
    unsigned long long host_too_large_address = 1;
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(&host_status, status, sizeof(host_status),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&host_too_large_address, too_large_address,
                   sizeof(host_too_large_address), cudaMemcpyDeviceToHost) != cudaSuccess ||
        host_status != 3 || host_too_large_address != 0) {
        std::fprintf(stderr,
                     "FAIL: device free/reuse/exhaustion semantics (status=%d)\n",
                     host_status);
        return 1;
    }

    size_t heap_size = 0;
    if (cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize) != cudaSuccess ||
        heap_size != 4096 ||
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8192) != cudaErrorInvalidValue ||
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4096) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: device heap limit lifecycle\n");
        return 1;
    }

    DeviceContainer** polymorphic_slot = nullptr;
    int* polymorphic_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&polymorphic_slot),
                   sizeof(*polymorphic_slot)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&polymorphic_output),
                   2 * sizeof(*polymorphic_output)) != cudaSuccess ||
        cudaMemset(polymorphic_slot, 0, sizeof(*polymorphic_slot)) != cudaSuccess ||
        cudaMemset(polymorphic_output, 0, 2 * sizeof(*polymorphic_output)) !=
            cudaSuccess) {
        return 1;
    }
    create_polymorphic<<<1, 1>>>(polymorphic_slot);
    exercise_polymorphic<<<1, 1>>>(polymorphic_slot, polymorphic_output);
    destroy_polymorphic<<<1, 1>>>(polymorphic_slot);
    int host_polymorphic_output[2] = {};
    DeviceContainer* host_polymorphic_slot = reinterpret_cast<DeviceContainer*>(1);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(host_polymorphic_output, polymorphic_output,
                   sizeof(host_polymorphic_output), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&host_polymorphic_slot, polymorphic_slot,
                   sizeof(host_polymorphic_slot), cudaMemcpyDeviceToHost) != cudaSuccess ||
        host_polymorphic_output[0] != 29 || host_polymorphic_output[1] != 17 ||
        host_polymorphic_slot != nullptr) {
        std::fprintf(stderr,
                     "FAIL: virtual calls or local reference return (%d, %d, %p)\n",
                     host_polymorphic_output[0], host_polymorphic_output[1],
                     static_cast<void*>(host_polymorphic_slot));
        return 1;
    }

    cudaFree(polymorphic_output);
    cudaFree(polymorphic_slot);

    cudaFree(too_large_address);
    cudaFree(released_address);
    cudaFree(status);
    cudaFree(slot);
    std::printf("PASS: device new/delete, virtual dispatch, references, reuse, and exhaustion\n");
    return 0;
}
