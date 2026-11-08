// One of two native units with same-named internal-linkage kernels; see
// run_native_multi_unit.sh.
#include <cuda_runtime.h>
namespace {
__global__ void bump(int* o) { o[threadIdx.x] += 1; }
}  // namespace
static __global__ void twice(int* o) { o[threadIdx.x] += 1 * 100; }
__global__ void external_a(int* o) { o[threadIdx.x] += 1 * 10000; }
void run_a(int* d) {
    bump<<<1, 32>>>(d);
    twice<<<1, 32>>>(d);
    external_a<<<1, 32>>>(d);
}
