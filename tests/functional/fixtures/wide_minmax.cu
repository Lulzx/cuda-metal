#include <cuda_runtime.h>
__global__ void wide_minmax(unsigned long long* lo, unsigned long long* hi,
                            unsigned long long* old_out,
                            const unsigned long long* vals) {
    int i = threadIdx.x;
    atomicMin(lo, vals[i]);
    atomicMax(hi, vals[i]);
    unsigned long long old = atomicMin(old_out, vals[i]);
    if (i == 0) old_out[1] = old;
}
