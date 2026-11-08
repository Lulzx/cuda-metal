// Native AOT registers kernels with internal linkage: an anonymous-namespace
// kernel (its stub name ends a component in a digit, `_GLOBAL__N_1`, which
// the stub-to-kernel name mapping used to misread) and a static kernel (whose
// host stub the separate registration object could not name).
#include <cstdio>
namespace { __global__ void k_anon(int* o) { o[threadIdx.x] = 1; } }
static __global__ void k_static(int* o) { o[threadIdx.x] += 2; }
__global__ void k_ext(int* o) { o[threadIdx.x] += 4; }
int main() {
  int* d; cudaMalloc(&d, 32 * sizeof(int));
  k_anon<<<1, 32>>>(d); k_static<<<1, 32>>>(d); k_ext<<<1, 32>>>(d);
  int h[32]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; ++i) if (h[i] != 7) { printf("FAIL: %d at %d\n", h[i], i); return 1; }
  printf("PASS: internal-linkage kernels\n"); return 0;
}
