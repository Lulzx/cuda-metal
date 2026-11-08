// Kernel-free unit linking two kernel units. Each unit embeds its own
// metallib and declares `bump` (anonymous namespace) and `twice` (static);
// the sum proves every launch reached its own unit's kernel.
#include <cstdio>
#include <cuda_runtime.h>
void run_a(int*);
void run_b(int*);
int main() {
    int* d = nullptr;
    if (cudaMalloc(&d, 32 * sizeof(int)) != cudaSuccess ||
        cudaMemset(d, 0, 32 * sizeof(int)) != cudaSuccess) {
        printf("FAIL: allocation\n");
        return 1;
    }
    run_a(d);
    run_b(d);
    int h[32];
    if (cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("FAIL: copy\n");
        return 1;
    }
    for (int i = 0; i < 32; ++i) {
        if (h[i] != 111111) {
            printf("FAIL: element %d is %d, want 111111\n", i, h[i]);
            return 1;
        }
    }
    printf("PASS: native units with same-named internal kernels link and stay distinct\n");
    return 0;
}
