// `%s` arguments that are string literals. The literals live in Metal
// constant memory, which the host cannot read, so the typed NVVM path passes
// them through the printf ring as tagged handles. One word is chosen at run
// time, one literal is passed directly, and `%p` of a literal must stay an
// address, not a handle.
#include <cuda_runtime.h>

#include <cstdio>

__global__ void print_words(int odd) {
    const char* word = odd ? "odd" : "even";
    printf("LITERAL i=%d s=%s lit=%s pct=%%\n", odd, word, "literal");
}

__global__ void print_literal_address() {
    printf("ADDRESS p=%p\n", "address-only");
}

int main() {
    print_words<<<1, 1>>>(1);
    print_words<<<1, 1>>>(0);
    print_literal_address<<<1, 1>>>();
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }
    std::printf("HOST_DONE\n");
    return 0;
}
