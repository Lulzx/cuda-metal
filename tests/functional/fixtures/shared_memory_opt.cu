#include <cuda_runtime.h>

// float4-shaped shared access: each thread stages four adjacent floats
// through shared memory; the typed backend should coalesce the shared
// accesses into vector (uint4) load/store.
__global__ void quad_copy(const float* in, float* out) {
    __shared__ float tile[64];
    int t = threadIdx.x;
    int base = t * 4;
    tile[base + 0] = in[base + 0];
    tile[base + 1] = in[base + 1];
    tile[base + 2] = in[base + 2];
    tile[base + 3] = in[base + 3];
    __syncthreads();
    float a = tile[base + 0];
    float b = tile[base + 1];
    float c = tile[base + 2];
    float d = tile[base + 3];
    out[t] = a + b + c + d;
}

// Bank-conflict-shaped shared access: 32x32 tile written row-major and
// read back transposed; rows are 128 bytes, so the typed backend pads
// one element per row.
__global__ void transpose32(const float* in, float* out) {
    __shared__ float tile[32][32];
    int tx = threadIdx.x, ty = threadIdx.y;
    tile[ty][tx] = in[ty * 32 + tx];
    __syncthreads();
    out[ty * 32 + tx] = tile[tx][ty];
}
