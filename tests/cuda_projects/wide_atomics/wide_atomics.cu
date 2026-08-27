// 64-bit atomics. Metal has no atomic of that width -- `atomic_ulong` fails the
// standard library's own compare-exchange type check, there is no atomic
// double -- so every one of these is serialized behind a bank of the 32-bit
// locks Metal does have, hashed on the target address.
//
// What this pins is the part a "did it compile" check cannot see. A GPU lock
// deadlocks if a lane spins while a lane it executes in lockstep with holds the
// lock, so the contended cases below (1024 threads on one address, and a CAS
// loop, which is a retry loop wrapped around the lock's own retry loop) are the
// point of the harness: they either return the exact expected total or they
// hang. The uncontended per-thread case is here for the opposite reason -- it
// only passes if distinct addresses map to independent locks.
#include <cuda_runtime.h>

#include <cstdio>

#define BLOCKS 8
#define THREADS 128
#define TOTAL (BLOCKS * THREADS)

__global__ void wide_atomic_kernel(double* d, unsigned long long* u, long long* s,
                                   double* per_thread) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    // Contended: every thread onto one address. 1.0 and 0.25 are exact in
    // binary64 and stay exact through the FP32-pair ALU, so the totals are
    // equalities, not tolerances -- a dropped or doubled update shows up.
    atomicAdd(&d[0], 1.0);
    atomicAdd(&d[1], 0.25);

    // Uncontended: one address per thread, so this passes only if the address
    // hash actually spreads across the bank.
    atomicAdd(&per_thread[tid], 2.0);

    atomicAdd(&u[0], 3ull);
    atomicExch(&u[1], static_cast<unsigned long long>(tid));

    // A CAS loop of our own around the lock's retry loop. This is also the
    // shape clang gives atomicAdd(double*) on its own, so it is the exact
    // nesting HiGHS's PDLP convergence kernels perform.
    unsigned long long seen = u[2];
    unsigned long long prev;
    while ((prev = atomicCAS(&u[2], seen, seen + 1)) != seen) {
        seen = prev;
    }

    atomicMax(&s[0], static_cast<long long>(tid));
    atomicMin(&s[1], static_cast<long long>(tid));

    if (tid == 0) {
        atomicAnd(&u[3], 0x00000000ffffffffull);
        atomicOr(&u[4], 0x0123456789abcdefull);
        atomicXor(&u[5], 0xffffffffffffffffull);
    }

    // Threadgroup address space: the lock is device memory either way, but the
    // payload load/store is not.
    __shared__ double block_sum;
    if (threadIdx.x == 0) block_sum = 0.0;
    __syncthreads();
    atomicAdd(&block_sum, 0.5);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&d[2], block_sum);
}

static int check_u64(const char* name, unsigned long long got, unsigned long long want) {
    if (got == want) return 0;
    std::printf("FAIL: %s got %llu (0x%llx) expected %llu (0x%llx)\n", name, got, got, want, want);
    return 1;
}

static int check_f64(const char* name, double got, double want) {
    if (got == want) return 0;
    std::printf("FAIL: %s got %.17g expected %.17g\n", name, got, want);
    return 1;
}

int main() {
    double* d = nullptr;
    unsigned long long* u = nullptr;
    long long* s = nullptr;
    double* per_thread = nullptr;
    if (cudaMallocManaged(&d, 3 * sizeof(double)) != cudaSuccess ||
        cudaMallocManaged(&u, 6 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMallocManaged(&s, 2 * sizeof(long long)) != cudaSuccess ||
        cudaMallocManaged(&per_thread, TOTAL * sizeof(double)) != cudaSuccess) {
        std::printf("FAIL: cudaMallocManaged\n");
        return 1;
    }
    d[0] = 0.0;
    d[1] = 0.0;
    d[2] = 0.0;
    u[0] = 0;
    u[1] = 0;
    u[2] = 0;
    u[3] = 0xfedcba9876543210ull;
    u[4] = 0;
    u[5] = 0x00ff00ff00ff00ffull;
    s[0] = -1;
    s[1] = 999999;
    for (int i = 0; i < TOTAL; ++i) per_thread[i] = 0.0;

    wide_atomic_kernel<<<BLOCKS, THREADS>>>(d, u, s, per_thread);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }

    int failures = 0;
    failures += check_f64("atomicAdd(double) contended", d[0], static_cast<double>(TOTAL));
    failures += check_f64("atomicAdd(double) fractional", d[1], TOTAL * 0.25);
    failures += check_f64("atomicAdd(shared double)", d[2], BLOCKS * THREADS * 0.5);
    failures += check_u64("atomicAdd(unsigned long long)", u[0], 3ull * TOTAL);
    failures += check_u64("atomicCAS loop", u[2], TOTAL);
    failures += check_u64("atomicAnd(unsigned long long)", u[3], 0x0000000076543210ull);
    failures += check_u64("atomicOr(unsigned long long)", u[4], 0x0123456789abcdefull);
    failures += check_u64("atomicXor(unsigned long long)", u[5], 0xff00ff00ff00ff00ull);
    if (s[0] != TOTAL - 1) {
        std::printf("FAIL: atomicMax(long long) got %lld expected %d\n", s[0], TOTAL - 1);
        ++failures;
    }
    if (s[1] != 0) {
        std::printf("FAIL: atomicMin(long long) got %lld expected 0\n", s[1]);
        ++failures;
    }
    if (u[1] >= static_cast<unsigned long long>(TOTAL)) {
        std::printf("FAIL: atomicExch result %llu is not a thread id\n", u[1]);
        ++failures;
    }
    for (int i = 0; i < TOTAL; ++i) {
        if (per_thread[i] != 2.0) {
            std::printf("FAIL: per-thread atomicAdd slot %d got %.17g expected 2\n",
                        i, per_thread[i]);
            ++failures;
            break;
        }
    }

    cudaFree(d);
    cudaFree(u);
    cudaFree(s);
    cudaFree(per_thread);
    if (failures != 0) return 1;
    std::printf("PASS: 64-bit atomics are serialized correctly through the lock bank\n");
    return 0;
}
