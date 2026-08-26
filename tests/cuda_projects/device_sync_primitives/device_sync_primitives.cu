// Regression tests for two silent-wrong-answer defects found by the cuda-samples sweep.
// Both produced zeros with cudaSuccess everywhere the caller looked.
//
// 1. Apple's AIR backend cannot lower LLVM's `fence` instruction. Emitting one crashed
//    the Metal compiler service at pipeline-creation time, so the kernel never ran --
//    and because cudaDeviceSynchronize() did not report the launch failure, the caller
//    just read back zeros. clang plants a membar next to atomicCAS, so this took out
//    every CAS-bearing kernel, and atomicInc/atomicDec which are built on CAS.
//
// 2. compute_static_shared_bytes() only counted `.bN name[N]` array declarations, while
//    the layout pass also accepted scalars. A scalar `__shared__ unsigned x` -- which
//    clang emits as `.shared .align 4 .u32 name;` -- was assigned an offset but counted
//    as zero bytes, so the threadgroup allocation came out at length 0: every store
//    dropped, every load zero.
//
// Everything here is checked against a value the host computes independently.
#include <cuda_runtime.h>

#include <cstdio>

#define NB 64
#define NT 256
#define NTOTAL (NB * NT)

__global__ void all_atomics(int *o) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    atomicAdd(&o[0], 10);
    atomicSub(&o[1], 10);
    atomicMax(&o[2], (int)tid);
    atomicMin(&o[3], (int)tid);
    // The CAS family: the reason kernel 1 above never ran.
    atomicCAS(&o[4], (int)tid, -1);
    atomicInc((unsigned int *)&o[5], 0xffffffffu);
    atomicDec((unsigned int *)&o[6], 0xffffffffu);
    atomicOr(&o[7], 1);
    atomicAnd(&o[8], ~1);
    atomicXor(&o[9], 0);
}

__global__ void threadfence_kernel(int *o) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    o[tid] = (int)tid;
    __threadfence();
    atomicAdd(&o[NTOTAL], 1);
    __threadfence_block();
}

// uniformUpdate() from cuda-samples/scan, reduced: one thread writes a scalar
// __shared__, the rest read it after __syncthreads().
__global__ void shared_scalar_broadcast(unsigned *out, const unsigned *in) {
    __shared__ unsigned buf;
    const unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) {
        buf = in[blockIdx.x];
    }
    __syncthreads();
    out[pos] = buf;
}

// Same broadcast through a scalar plus an array, so the two are laid out together
// and a mis-sized allocation shows up as overlap rather than only as zeros.
__global__ void shared_mixed(unsigned *out, const unsigned *in) {
    __shared__ unsigned scalar_slot;
    __shared__ unsigned array_slot[NT];
    const unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) {
        scalar_slot = in[blockIdx.x];
    }
    array_slot[threadIdx.x] = threadIdx.x;
    __syncthreads();
    out[pos] = scalar_slot + array_slot[threadIdx.x];
}

static int failures = 0;

static void expect(const char *what, long long got, long long want) {
    if (got != want) {
        std::printf("FAIL: %s: expected %lld, got %lld\n", what, want, got);
        ++failures;
    }
}

int main() {
    const int kSlots = 10;
    int h[kSlots];
    for (int i = 0; i < kSlots; ++i) h[i] = 0;
    h[3] = NTOTAL;      // atomicMin identity
    h[8] = -1;          // atomicAnd identity

    int *d_atomics = nullptr;
    cudaMalloc(&d_atomics, sizeof(h));
    cudaMemcpy(d_atomics, h, sizeof(h), cudaMemcpyHostToDevice);
    all_atomics<<<NB, NT>>>(d_atomics);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("FAIL: all_atomics launch: %s\n", cudaGetErrorString(e));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h, d_atomics, sizeof(h), cudaMemcpyDeviceToHost);

    expect("atomicAdd", h[0], (long long)NTOTAL * 10);
    expect("atomicSub", h[1], -(long long)NTOTAL * 10);
    expect("atomicMax", h[2], NTOTAL - 1);
    expect("atomicMin", h[3], 0);
    // Exactly one thread sees o[4]==its tid (thread 0, against the initial 0) and swaps.
    expect("atomicCAS", h[4], -1);
    expect("atomicInc", h[5], NTOTAL);
    // atomicDec counts down from 0, wrapping to the supplied maximum each time.
    expect("atomicDec", h[6], (long long)(unsigned)(0u - (unsigned)NTOTAL) - 4294967296LL);
    expect("atomicOr", h[7], 1);
    expect("atomicAnd", h[8], -2);
    expect("atomicXor", h[9], 0);

    int *d_fence = nullptr;
    cudaMalloc(&d_fence, (NTOTAL + 1) * sizeof(int));
    cudaMemset(d_fence, 0, (NTOTAL + 1) * sizeof(int));
    threadfence_kernel<<<NB, NT>>>(d_fence);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("FAIL: threadfence launch: %s\n", cudaGetErrorString(e));
        return 1;
    }
    cudaDeviceSynchronize();
    int *h_fence = new int[NTOTAL + 1];
    cudaMemcpy(h_fence, d_fence, (NTOTAL + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    expect("__threadfence counter", h_fence[NTOTAL], NTOTAL);
    for (int i = 0; i < NTOTAL; ++i) {
        if (h_fence[i] != i) {
            std::printf("FAIL: __threadfence store[%d]: expected %d, got %d\n", i, i, h_fence[i]);
            ++failures;
            break;
        }
    }
    delete[] h_fence;

    unsigned h_in[NB], *h_out = new unsigned[NTOTAL];
    for (int i = 0; i < NB; ++i) h_in[i] = 1000u + (unsigned)i;
    unsigned *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, sizeof(h_in));
    cudaMalloc(&d_out, NTOTAL * sizeof(unsigned));
    cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);

    cudaMemset(d_out, 0, NTOTAL * sizeof(unsigned));
    shared_scalar_broadcast<<<NB, NT>>>(d_out, d_in);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("FAIL: shared_scalar_broadcast launch: %s\n", cudaGetErrorString(e));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, NTOTAL * sizeof(unsigned), cudaMemcpyDeviceToHost);
    for (int b = 0; b < NB; ++b) {
        for (int t = 0; t < NT; ++t) {
            if (h_out[b * NT + t] != h_in[b]) {
                std::printf("FAIL: scalar __shared__ broadcast block %d thread %d: "
                            "expected %u, got %u\n", b, t, h_in[b], h_out[b * NT + t]);
                ++failures;
                b = NB;
                break;
            }
        }
    }

    cudaMemset(d_out, 0, NTOTAL * sizeof(unsigned));
    shared_mixed<<<NB, NT>>>(d_out, d_in);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("FAIL: shared_mixed launch: %s\n", cudaGetErrorString(e));
        return 1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, NTOTAL * sizeof(unsigned), cudaMemcpyDeviceToHost);
    for (int b = 0; b < NB; ++b) {
        for (int t = 0; t < NT; ++t) {
            const unsigned want = h_in[b] + (unsigned)t;
            if (h_out[b * NT + t] != want) {
                std::printf("FAIL: mixed scalar+array __shared__ block %d thread %d: "
                            "expected %u, got %u\n", b, t, want, h_out[b * NT + t]);
                ++failures;
                b = NB;
                break;
            }
        }
    }
    delete[] h_out;

    if (failures != 0) {
        std::printf("FAIL: %d device sync primitive check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS: atomics, __threadfence, and scalar __shared__ all correct\n");
    return 0;
}
