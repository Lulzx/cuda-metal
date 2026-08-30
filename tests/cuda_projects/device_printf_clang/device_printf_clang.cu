#include <cuda_runtime.h>

#include <cstdio>

__device__ char module_global_string[] = "CuMetal-global-string";

__device__ __noinline__ void print_coordinate_record(int block, int thread,
                                                      int value) {
    printf("PRINTF[%d,%d]=%d\n", block, thread, value);
}

__device__ __noinline__ void print_coordinate_helper(int block, int thread,
                                                      int value) {
    print_coordinate_record(block, thread, value);
}

__global__ void print_coordinates(int value) {
    const int block = static_cast<int>(blockIdx.y * gridDim.x + blockIdx.x);
    const int thread = static_cast<int>(
        threadIdx.z * blockDim.x * blockDim.y +
        threadIdx.y * blockDim.x + threadIdx.x);
    print_coordinate_helper(block, thread, value);
}

__global__ void print_wide_values(const int* pointer) {
    const long long signed_value = -1234567890123LL;
    const unsigned long long unsigned_value = 0x1122334455667788ULL;
    const size_t size_value = static_cast<size_t>(0x200000007ULL);
    printf("WIDE signed=%lld unsigned=%llu hex=%#llx size=%zu ptr=%p "
           "float=%.3f char=%c percent=%%\n",
           signed_value, unsigned_value, unsigned_value, size_value, pointer,
           3.125, 'Q');
}

__global__ void print_dynamic_values() {
    printf("DYNAMIC int=%*d float=%*.*f left=%-*u\n",
           6, -42, 8, 2, 3.125, 5, 7u);
}

__global__ void print_device_string(const char* value) {
    printf("STRING value=%s\n", value);
}

__global__ void print_module_string() {
    printf("MODULE global=%s\n", module_global_string);
}

__global__ void print_untracked_string() {
    printf("UNTRACKED value=%s\n",
           reinterpret_cast<const char*>(0x12345678ULL));
}

__global__ void capture_printf_returns(int* values) {
    values[0] = printf("RETURN zero\n");
    values[1] = printf("RETURN args=%d,%d\n", 11, 22);
}

int main() {
    print_coordinates<<<dim3(2, 2), dim3(2, 2, 2)>>>(37);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }
    int* pointer = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(int)) != cudaSuccess) {
        std::printf("FAIL: cudaMalloc\n");
        return 1;
    }
    print_wide_values<<<1, 1>>>(pointer);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: wide cudaDeviceSynchronize: %s\n",
                    cudaGetErrorString(error));
        cudaFree(pointer);
        return 1;
    }
    print_dynamic_values<<<1, 1>>>();
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: dynamic cudaDeviceSynchronize: %s\n",
                    cudaGetErrorString(error));
        cudaFree(pointer);
        return 1;
    }
    constexpr char host_string[] = "CuMetal-device-string";
    char* device_string = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_string), sizeof(host_string)) !=
            cudaSuccess ||
        cudaMemcpy(device_string, host_string, sizeof(host_string),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::printf("FAIL: device string allocation/copy\n");
        cudaFree(pointer);
        return 1;
    }
    print_device_string<<<1, 1>>>(device_string);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: string cudaDeviceSynchronize: %s\n",
                    cudaGetErrorString(error));
        cudaFree(device_string);
        cudaFree(pointer);
        return 1;
    }
    cudaFree(device_string);
    print_module_string<<<1, 1>>>();
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: module string cudaDeviceSynchronize: %s\n",
                    cudaGetErrorString(error));
        cudaFree(pointer);
        return 1;
    }
    print_untracked_string<<<1, 1>>>();
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: untracked string cudaDeviceSynchronize: %s\n",
                    cudaGetErrorString(error));
        cudaFree(pointer);
        return 1;
    }
    constexpr size_t unterminated_size = 256;
    char* unterminated = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&unterminated), unterminated_size) !=
            cudaSuccess ||
        cudaMemset(unterminated, 'A', unterminated_size) != cudaSuccess) {
        std::printf("FAIL: unterminated string allocation/fill\n");
        cudaFree(pointer);
        return 1;
    }
    print_device_string<<<1, 1>>>(unterminated);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: unterminated string cudaDeviceSynchronize: %s\n",
                    cudaGetErrorString(error));
        cudaFree(unterminated);
        cudaFree(pointer);
        return 1;
    }
    cudaFree(unterminated);
    int* return_values = nullptr;
    int host_return_values[2] = {-99, -99};
    if (cudaMalloc(reinterpret_cast<void**>(&return_values),
                   sizeof(host_return_values)) != cudaSuccess ||
        cudaMemset(return_values, 0xff, sizeof(host_return_values)) != cudaSuccess ||
        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 12) != cudaSuccess) {
        std::printf("FAIL: printf return setup\n");
        cudaFree(return_values);
        cudaFree(pointer);
        return 1;
    }
    capture_printf_returns<<<1, 1>>>(return_values);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess ||
        cudaMemcpy(host_return_values, return_values, sizeof(host_return_values),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL: printf return launch/copy: %s\n",
                    cudaGetErrorString(error));
        cudaFree(return_values);
        cudaFree(pointer);
        return 1;
    }
    cudaFree(return_values);
    if (host_return_values[0] != 0 || host_return_values[1] != 2) {
        std::printf("FAIL: printf returns zero=%d args=%d\n",
                    host_return_values[0], host_return_values[1]);
        cudaFree(pointer);
        return 1;
    }
    std::printf("RETURN_VALUES zero=%d args=%d\n",
                host_return_values[0], host_return_values[1]);
    cudaFree(pointer);
    std::printf("HOST_DONE\n");
    return 0;
}
