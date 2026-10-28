#include <cuda_runtime.h>
extern "C" __global__ void mm(int* mn, int* mx, int* olds_mn, int* olds_mx) {
    int lane = (int)threadIdx.x;
    int o1 = atomicMin(mn, 100 - lane);
    int o2 = atomicMax(mx, lane);
    olds_mn[lane] = o1;
    olds_mx[lane] = o2;
}
