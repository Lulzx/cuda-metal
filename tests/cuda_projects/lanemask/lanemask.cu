// The PTX %lanemask_{eq,lt,le,ge,gt} special registers, read through inline
// asm the way CUB's warp primitives read them (cub::LaneMaskLt and friends
// feed WarpScan and the radix-sort rank). The typed backend once had no
// lowering for them and refused every kernel that used one; each mask is
// checked against its definition for every lane of two warps.
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

__device__ unsigned lanemask_eq() { unsigned m; asm("mov.u32 %0, %%lanemask_eq;" : "=r"(m)); return m; }
__device__ unsigned lanemask_lt() { unsigned m; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m)); return m; }
__device__ unsigned lanemask_le() { unsigned m; asm("mov.u32 %0, %%lanemask_le;" : "=r"(m)); return m; }
__device__ unsigned lanemask_ge() { unsigned m; asm("mov.u32 %0, %%lanemask_ge;" : "=r"(m)); return m; }
__device__ unsigned lanemask_gt() { unsigned m; asm("mov.u32 %0, %%lanemask_gt;" : "=r"(m)); return m; }

__global__ void lanemask_kernel(unsigned* out) {
    const unsigned t = threadIdx.x;
    out[t * 5 + 0] = lanemask_eq();
    out[t * 5 + 1] = lanemask_lt();
    out[t * 5 + 2] = lanemask_le();
    out[t * 5 + 3] = lanemask_ge();
    out[t * 5 + 4] = lanemask_gt();
}

int main() {
    constexpr int kThreads = 64;
    unsigned* device = nullptr;
    if (cudaMalloc(&device, kThreads * 5 * sizeof(unsigned)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        return 1;
    }
    lanemask_kernel<<<1, kThreads>>>(device);
    std::vector<unsigned> host(kThreads * 5);
    if (cudaMemcpy(host.data(), device, host.size() * sizeof(unsigned), cudaMemcpyDeviceToHost) !=
        cudaSuccess || cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: lanemask kernel did not run\n");
        return 1;
    }
    cudaFree(device);
    const char* const names[] = {"eq", "lt", "le", "ge", "gt"};
    int failures = 0;
    for (int t = 0; t < kThreads; ++t) {
        const unsigned lane = t % 32;
        const unsigned eq = 1u << lane;
        const unsigned lt = eq - 1u;
        const unsigned le = lt | eq;
        const unsigned want[5] = {eq, lt, le, ~lt, ~le};
        for (int k = 0; k < 5; ++k) {
            if (host[t * 5 + k] != want[k] && failures++ < 8) {
                std::fprintf(stderr, "FAIL: thread %d lanemask_%s = 0x%08x, want 0x%08x\n", t,
                             names[k], host[t * 5 + k], want[k]);
            }
        }
    }
    if (failures != 0) {
        std::fprintf(stderr, "FAIL: %d wrong lane masks\n", failures);
        return 1;
    }
    std::printf("PASS: lanemask special registers\n");
    return 0;
}
