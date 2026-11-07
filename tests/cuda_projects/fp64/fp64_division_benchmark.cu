// Exact binary64 divides in dependent and independent chains; many threads hide latency.
// Every output uses a bitwise host-divide oracle; paired raw GPU/host timings describe
// this workload, not intrinsic divide latency (ieee64 only).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 65536;  // 256 blocks x 256 threads
constexpr int kBlock = 256;
constexpr int kIters = 32;       // divides per chain per launch
constexpr int kLanes = 4;        // independent chains per thread in div_independent
constexpr int kTimed = 7;        // timed repetitions per kernel, after one warmup
constexpr int kGuard = 2;        // guard elements after each output buffer

// Quiet NaN written over the whole output buffer before every launch: an
// element the kernel never wrote reads back as NaN and is reported.
constexpr uint64_t kPoison = 0x7ff8000000000000ull;

uint64_t bits(double v) {
    uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

double from_bits(uint64_t b) {
    double v;
    std::memcpy(&v, &b, sizeof v);
    return v;
}

double poison_double() {
    double v;
    const uint64_t b = kPoison;
    std::memcpy(&v, &b, sizeof v);
    return v;
}

// Deterministic input, no library RNG: splitmix64 over the element index.
uint64_t splitmix64(uint64_t s) {
    s += 0x9e3779b97f4a7c15ull;
    s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9ull;
    s = (s ^ (s >> 27)) * 0x94d049bb133111ebull;
    return s ^ (s >> 31);
}

// Dividend in [1 + 2^-13, 2), assembled from bits so FMA contraction cannot move
// the input set; bit 39 sets the floor and excludes powers of two.
constexpr uint64_t kOne = 0x3FF0000000000000ull;
constexpr uint64_t kMantissa = 0x000FFFFFFFFFFFFFull;

double make_x(uint64_t index, uint64_t seed) {
    const uint64_t m = (splitmix64(index ^ seed) >> 11) | (1ull << 39);
    return from_bits(kOne | (m & kMantissa));
}

// Divisor in (1, 1 + 2^-10]: above 1 so chains change, close enough to 1
// that kIters divides stay in the normal binary64 range.
double make_d(uint64_t index, uint64_t seed) {
    const uint64_t m = (splitmix64(index ^ seed) % (1ull << 42)) + 1;
    return from_bits(kOne | m);
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                            \
        const cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                                \
            std::printf("FAIL: %s at %s:%d -> %s\n", #call, __FILE__, __LINE__, \
                      cudaGetErrorString(err_));                                  \
            return 1;                                                             \
        }                                                                         \
    } while (0)

// n_threads is the launch width; n_outputs is the number of live elements the
// kernel writes, which is kLanes * n_threads for div_independent.
__global__ void div_chain(double* __restrict__ out, const double* __restrict__ x,
                          const double* __restrict__ d, int n_threads, int iters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_threads) return;
    double v = x[i];
    const double div = d[i];
    for (int k = 0; k < iters; ++k) v = v / div;
    out[i] = v;
}

__global__ void div_independent(double* __restrict__ out, const double* __restrict__ x,
                                const double* __restrict__ d, int n_threads, int iters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_threads) return;
    const int b = kLanes * i;
    double x0 = x[b], x1 = x[b + 1], x2 = x[b + 2], x3 = x[b + 3];
    const double d0 = d[b], d1 = d[b + 1], d2 = d[b + 2], d3 = d[b + 3];
    for (int k = 0; k < iters; ++k) {
        x0 = x0 / d0;
        x1 = x1 / d1;
        x2 = x2 / d2;
        x3 = x3 / d3;
    }
    out[b] = x0;
    out[b + 1] = x1;
    out[b + 2] = x2;
    out[b + 3] = x3;
}

// The oracle: the host's native binary64 divide over the arrays the GPU was
// given, computed before any GPU result exists. One chain per element covers both
// kernels, since div_independent is the same chain over a flat kLanes * n_threads
// array; a second copy would only be a second place for the rule to drift.
void oracle(const double* x, const double* d, size_t n, int iters, double* want) {
    for (size_t i = 0; i < n; ++i) {
        double v = x[i];
        const double div = d[i];
        for (int k = 0; k < iters; ++k) v = v / div;
        want[i] = v;
    }
}

// `got` holds n + kGuard values: every live element is finite and matches the
// oracle bitwise, every guard element still holds the poison.
int check_outputs(const char* workload, const double* got, const double* want, int n) {
    int failures = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(want[i])) {
            std::printf("FAIL: %s oracle element %d is not finite: %.17g\n", workload, i, want[i]);
            ++failures;
            continue;
        }
        if (!std::isfinite(got[i]) || bits(got[i]) != bits(want[i])) {
            if (failures < 8) {
                std::printf("FAIL: %s element %d = %.17g (0x%016llx), expected %.17g (0x%016llx)\n",
                           workload, i, got[i], (unsigned long long)bits(got[i]), want[i],
                           (unsigned long long)bits(want[i]));
            }
            ++failures;
        }
    }
    for (int g = 0; g < kGuard; ++g) {
        if (bits(got[n + g]) != kPoison) {
            std::printf("FAIL: %s guard element %d overwritten: 0x%016llx\n", workload, n + g,
                       (unsigned long long)bits(got[n + g]));
            ++failures;
        }
    }
    return failures;
}

int fail(const char* msg) {
    std::printf("FAIL: %s\n", msg);
    return 1;
}

// Out-of-range input would make the range argument for the oracle void, so a
// bad value here is a harness bug and stops the run.
int make_inputs(uint64_t seed_x, uint64_t seed_d, size_t n, double* x, double* d) {
    for (size_t j = 0; j < n; ++j) {
        // Divisors differ per element, so parallel lanes are distinct work.
        x[j] = make_x((uint64_t)j, seed_x);
        d[j] = make_d((uint64_t)j, seed_d + (j % kLanes));
        if (!(x[j] >= 1.0 + 0x1.0p-13 && x[j] < 2.0) ||
            !(d[j] > 1.0 && d[j] <= 1.0 + 0x1.0p-10)) {
            return fail("input generation produced an out-of-range or identity value");
        }
    }
    return 0;
}

// An input that did not survive the upload would make the oracle wrong rather
// than the kernel, so the device copy is read back before any timing.
int check_upload(const double* dev, const double* host, size_t n) {
    double* back = new double[n];
    CUDA_CHECK(cudaMemcpy(back, dev, sizeof(double) * n, cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < n; ++j) {
        if (bits(back[j]) != bits(host[j])) {
            delete[] back;
            return fail("input upload did not round-trip");
        }
    }
    delete[] back;
    return 0;
}

// One timed launch. The host clock starts before the start-event record and stops
// after synchronization, so host_ms covers submit plus wait and not just the wait
// left over after recording; the GPU clock stays a separate column. A non-finite
// or non-positive elapsed time fails, it is never a dropped sample.
int run_timed(const char* workload, int sample, double* out, const double* want, int n_threads,
              int n_outputs, long long divides,
              void (*launch)(double*, const double*, const double*, int, int), const double* x,
              const double* d) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const size_t count = (size_t)n_outputs + kGuard;
    double* host = new double[count];
    for (size_t i = 0; i < count; ++i) host[i] = poison_double();
    CUDA_CHECK(cudaMemcpy(out, host, sizeof(double) * count, cudaMemcpyHostToDevice));

    const auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    launch(out, x, d, n_threads, kIters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const auto t1 = std::chrono::steady_clock::now();

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    const double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int rc = 0;
    if (!(gpu_ms > 0.0f) || !std::isfinite(gpu_ms)) {
        std::printf("FAIL: %s sample %d: gpu_ms is not finite and positive: %g\n", workload,
                   sample, (double)gpu_ms);
        rc = 1;
    }
    if (!(host_ms > 0.0) || !std::isfinite(host_ms)) {
        std::printf("FAIL: %s sample %d: host_ms is not finite and positive: %g\n", workload,
                   sample, host_ms);
        rc = 1;
    }

    CUDA_CHECK(cudaMemcpy(host, out, sizeof(double) * count, cudaMemcpyDeviceToHost));
    const int failures = check_outputs(workload, host, want, n_outputs);
    if (failures != 0) {
        std::printf("FAIL: %s sample %d: %d of %d outputs wrong\n", workload, sample, failures,
                   n_outputs);
        rc = 1;
    }
    delete[] host;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (rc != 0) return 1;

    std::printf("%d,%s,%d,%d,%d,%lld,%.6f,%.6f\n", sample, workload, n_threads, n_outputs, kIters,
               divides, (double)gpu_ms, host_ms);
    return 0;
}

void launch_chain(double* out, const double* x, const double* d, int n_threads, int iters) {
    div_chain<<<kThreads / kBlock, kBlock>>>(out, x, d, n_threads, iters);
}

void launch_independent(double* out, const double* x, const double* d, int n_threads, int iters) {
    div_independent<<<kThreads / kBlock, kBlock>>>(out, x, d, n_threads, iters);
}

}  // namespace

int main() {
    const char* mode = std::getenv("CUMETAL_FP64_MODE");
    if (mode == nullptr || std::strcmp(mode, "ieee64") != 0) {
        return fail("this benchmark requires CUMETAL_FP64_MODE=ieee64: the pair modes do not "
                   "promise a correctly rounded divide, so a bitwise oracle would be wrong");
    }

    // One launch width, two output widths: div_independent is kLanes chains per
    // thread, so its live elements are kLanes * kThreads.
    const int n_threads = kThreads;
    const int n_chain_out = n_threads;
    const int n_ind_out = kLanes * n_threads;

    double* hx = new double[n_chain_out];
    double* hd = new double[n_chain_out];
    double* hx4 = new double[n_ind_out];
    double* hd4 = new double[n_ind_out];
    double* want_chain = new double[n_chain_out];
    double* want_ind = new double[n_ind_out];
    if (make_inputs(0x5eed01ull, 0x5eed02ull, n_chain_out, hx, hd)) return 1;
    if (make_inputs(0x5eed03ull, 0x5eed04ull, n_ind_out, hx4, hd4)) return 1;

    // Oracle before any GPU result exists.
    oracle(hx, hd, n_chain_out, kIters, want_chain);
    oracle(hx4, hd4, n_ind_out, kIters, want_ind);

    const size_t c_count = (size_t)n_chain_out + kGuard;
    const size_t i_count = (size_t)n_ind_out + kGuard;
    double *dx = nullptr, *dd = nullptr, *dout = nullptr;
    double *dx4 = nullptr, *dd4 = nullptr, *dout4 = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, sizeof(double) * c_count));
    CUDA_CHECK(cudaMalloc(&dd, sizeof(double) * c_count));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(double) * c_count));
    CUDA_CHECK(cudaMalloc(&dx4, sizeof(double) * i_count));
    CUDA_CHECK(cudaMalloc(&dd4, sizeof(double) * i_count));
    CUDA_CHECK(cudaMalloc(&dout4, sizeof(double) * i_count));

    CUDA_CHECK(cudaMemcpy(dx, hx, sizeof(double) * n_chain_out, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd, hd, sizeof(double) * n_chain_out, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx4, hx4, sizeof(double) * n_ind_out, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd4, hd4, sizeof(double) * n_ind_out, cudaMemcpyHostToDevice));
    if (check_upload(dx, hx, n_chain_out)) return 1;
    if (check_upload(dx4, hx4, n_ind_out)) return 1;

    std::printf("sample,workload,n_threads,n_outputs,iterations,operations,gpu_ms,host_ms\n");

    // One untimed launch of each, so every timed sample runs on a pipeline that
    // is already resolved.
    launch_chain(dout, dx, dd, n_threads, kIters);
    CUDA_CHECK(cudaGetLastError());
    launch_independent(dout4, dx4, dd4, n_threads, kIters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Alternate the kernels across the timed repetitions, so neither one is
    // measured only in the colder or only in the warmer half of the run.
    const long long divides_chain = (long long)n_chain_out * kIters;
    const long long divides_ind = (long long)n_ind_out * kIters;
    int sample = 0;
    for (int rep = 0; rep < kTimed; ++rep) {
        for (int which = 0; which < 2; ++which) {
            const bool do_chain = (which == 0) == (rep % 2 == 0);
            if (do_chain) {
                if (run_timed("fp64_div_chain", sample++, dout, want_chain, n_threads, n_chain_out,
                              divides_chain, launch_chain, dx, dd))
                    return 1;
            } else {
                if (run_timed("fp64_div_independent", sample++, dout4, want_ind, n_threads,
                              n_ind_out, divides_ind, launch_independent, dx4, dd4))
                    return 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dd));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(dx4));
    CUDA_CHECK(cudaFree(dd4));
    CUDA_CHECK(cudaFree(dout4));
    delete[] hx;
    delete[] hd;
    delete[] hx4;
    delete[] hd4;
    delete[] want_chain;
    delete[] want_ind;

    std::printf("PASS: exact binary64 division benchmark verified\n");
    return 0;
}
