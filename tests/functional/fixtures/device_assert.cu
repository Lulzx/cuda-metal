// Device assert on the native source path. A failing assert records
// cudaErrorAssert (710) for its launch; the asserting thread carries on, so a
// kernel with __syncthreads and warp shuffles still compiles and cannot hang.
// The error is sticky, as in CUDA, so the failing phase runs last.
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void block_sum(const unsigned* in, unsigned* out, unsigned bad) {
    __shared__ unsigned partial[2];
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    assert(i != bad);
    unsigned v = in[i];
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x / 32u] = v;
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = partial[0] + partial[1];
}

// printf on the asserting path cannot carry the trap status; the assert is
// dropped there (documented) but the kernel must still compile and run.
__global__ void assert_with_printf(unsigned* out, unsigned bad) {
    assert(threadIdx.x != bad);
    if (threadIdx.x == 0) printf("assert_with_printf ran\n");
    out[threadIdx.x] = threadIdx.x + 1u;
}

static int fail(const char* what, cudaError_t got) {
    std::printf("FAIL: %s (got %d %s)\n", what, static_cast<int>(got), cudaGetErrorName(got));
    return 1;
}

int main() {
    const unsigned blocks = 4, threads = 64, n = blocks * threads;
    unsigned host_in[n], host_out[blocks], expected[blocks] = {};
    for (unsigned i = 0; i < n; ++i) {
        host_in[i] = i * 7u + 3u;
        expected[i / threads] += host_in[i];
    }
    unsigned *in = nullptr, *out = nullptr, *flat = nullptr;
    cudaMalloc(&in, sizeof host_in);
    cudaMalloc(&out, sizeof host_out);
    cudaMalloc(&flat, threads * sizeof(unsigned));
    cudaMemcpy(in, host_in, sizeof host_in, cudaMemcpyHostToDevice);

    // Phase A: assertions hold.
    block_sum<<<blocks, threads>>>(in, out, 0xffffffffu);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail("passing assert reported an error", e);
    cudaMemcpy(host_out, out, sizeof host_out, cudaMemcpyDeviceToHost);
    for (unsigned b = 0; b < blocks; ++b)
        if (host_out[b] != expected[b]) return fail("block_sum result", cudaSuccess);

    assert_with_printf<<<1, threads>>>(flat, 0xffffffffu);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fail("printf kernel with passing assert", e);

    // Phase B: one thread in block 2 fails its assertion.
    cudaMemset(out, 0, sizeof host_out);
    block_sum<<<blocks, threads>>>(in, out, 2u * threads + 5u);
    e = cudaGetLastError();
    if (e != cudaSuccess) return fail("launch of an asserting kernel failed early", e);
    e = cudaDeviceSynchronize();
    if (e != cudaErrorAssert) return fail("failed assert did not report cudaErrorAssert", e);
    // Sticky: the next synchronization and the error state keep reporting it.
    e = cudaDeviceSynchronize();
    if (e != cudaErrorAssert) return fail("cudaErrorAssert was not sticky", e);
    // Reaching here means the launch completed: the asserting thread carried
    // on, so the block barrier it shares with its peers did not hang.

    std::printf("PASS: device assert reports sticky cudaErrorAssert without breaking barriers\n");
    return 0;
}
