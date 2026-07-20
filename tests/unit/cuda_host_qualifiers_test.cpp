#include "vector_types.h"
#include "cuda.h"

#include <cstdio>

__host__ __device__ __forceinline__ int host_device_max(int a, int b)
{
    return a > b ? a : b;
}

__global__ void host_parsed_kernel(int* value)
{
    *value = host_device_max(*value, 7);
}

int main()
{
    static_assert(CUDA_VERSION >= 12000, "CUDA_VERSION must select typed wrappers");
    static_assert(CU_EVENT_DISABLE_TIMING == 2, "CUDA event flag ABI mismatch");
    static_assert(CU_JIT_MAX_REGISTERS == 0, "CUDA JIT option ABI mismatch");
    static_assert(CU_AD_FORMAT_FLOAT == 0x20, "CUDA array format ABI mismatch");
    static_assert(CU_TR_FILTER_MODE_LINEAR == 1, "CUDA texture filter ABI mismatch");

    int value = 3;
    host_parsed_kernel(&value);
    if (value != 7) {
        std::fprintf(stderr, "FAIL: host compiler did not parse CUDA qualifiers\n");
        return 1;
    }

    std::printf("PASS: CUDA qualifiers are host-compiler compatible\n");
    return 0;
}
