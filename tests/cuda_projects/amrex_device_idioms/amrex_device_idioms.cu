// Device-code idioms AMReX's CUDA backend depends on, each one a defect this
// harness was written to pin down. AMReX is a block-structured AMR framework
// whose GPU path is ordinary portable CUDA C++, and building it against CuMetal
// found every check below failing -- all of them silently, with a wrong number
// rather than an error. See demos/amrex/README.md.
//
// Each check compares against a host reference computed in the same process, so
// the harness cannot pass by agreeing with itself.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <curand_kernel.h>

namespace {

int g_failures = 0;

void report(const char* what, bool ok, const char* detail = "") {
    std::printf("  %-42s %s%s%s\n", what, ok ? "PASS" : "FAIL",
                detail[0] ? "  " : "", detail);
    if (!ok) ++g_failures;
}

// ── 1. `std::`-qualified math in device code ─────────────────────────────────
// Clang's standalone CUDA math header declares only the C entry points in the
// global namespace, so libc++ had never nominated a __device__ overload into
// `std` and `std::sqrt(x)` in a __device__ function did not compile at all.
// AMReX writes every ParallelFor body in `std::` form, so this is load-bearing.
// The float overloads matter separately: without them an unqualified float call
// promotes to binary64, which Metal has no hardware for.
__global__ void std_math_kernel(const double* x, double* od, const float* xf, float* of) {
    const int i = threadIdx.x;
    const double v = x[i];
    od[i] = std::sqrt(v) + std::fabs(v) + std::floor(v) + std::ceil(v) + std::trunc(v) +
            std::fmin(v, 3.0) + std::fmax(v, 1.0) + std::copysign(v, -1.0);
    const float f = xf[i];
    of[i] = std::sqrt(f) + std::fabs(f) + std::floor(f) + std::ceil(f) + std::trunc(f) +
            std::fmin(f, 3.0f) + std::fmax(f, 1.0f) + std::copysign(f, -1.0f);
}

// Transcendentals are split out because they are held to a different bar:
// under FP64 emulation a `double` exp/log/sin evaluates through binary32
// (docs/fp64-policy.md), so this checks the value, not binary64 accuracy.
__global__ void std_transcendental_kernel(const double* x, double* o) {
    const int i = threadIdx.x;
    const double v = x[i];
    o[i] = std::exp(v) + std::log(v) + std::sin(v) + std::tanh(v) + std::atan2(v, 2.0) +
           std::pow(v, 2.0) + std::hypot(v, 1.0);
}

// Integral and mixed arguments must promote to double the way <cmath> says,
// rather than becoming ambiguous between the float and double overloads.
__global__ void std_promotion_kernel(double* o) {
    const int n = 3;
    o[0] = std::pow(2, n) + std::sqrt(9) + std::log(1) + std::fmax(1, 2.0f);
}

// ── 2. 64-bit high multiply ──────────────────────────────────────────────────
// AMReX's FastDivmodU64 replaces the 64-bit division in every 3-D ParallelFor's
// index decomposition with a multiply-and-shift built on __umul64hi, so an
// unlowerable or wrong __umul64hi takes out every AMReX GPU kernel at once.
__global__ void mul64hi_kernel(const unsigned long long* a, const unsigned long long* b,
                               unsigned long long* hi_u, long long* hi_s) {
    const int i = threadIdx.x;
    hi_u[i] = __umul64hi(a[i], b[i]);
    hi_s[i] = __mul64hi(static_cast<long long>(a[i]), static_cast<long long>(b[i]));
}

// ── 3. `long` warp shuffles ──────────────────────────────────────────────────
// On LP64, `long` and `long long` are distinct types of the same width. CUDA
// declares the shuffle family for both; with only the `long long` overload a
// shuffle of a `long` is ambiguous against the `int` one. AMReX warp-reduces a
// `long` cell counter.
__global__ void long_shuffle_kernel(long* o, unsigned long* ou) {
    const int lane = threadIdx.x;
    long v = static_cast<long>(lane) + 1;
    unsigned long u = static_cast<unsigned long>(lane) * 3u + 7u;
    long sum = v;
    for (int off = 16; off > 0; off /= 2) sum += __shfl_down_sync(0xffffffffu, sum, off);
    const long bcast = __shfl_sync(0xffffffffu, v, 5);
    const long up = __shfl_up_sync(0xffffffffu, v, 1);
    const long xr = __shfl_xor_sync(0xffffffffu, v, 1);
    const unsigned long ubcast = __shfl_sync(0xffffffffu, u, 3);
    o[lane] = bcast + up + xr;
    if (lane == 0) { o[0] = sum; ou[0] = ubcast; }
}

// ── 4. Word-wise shuffle of a punned struct ──────────────────────────────────
// AMReX reduces a tuple across a warp by reinterpreting it as an array of
// 32-bit words and shuffling one word at a time (multi_shuffle_down in
// AMReX_GpuReduce.H). It is formally a strict-aliasing violation, which nvcc
// does not act on and Clang does: at -O2 Clang's TBAA concluded the punned
// stores could not alias the struct and deleted them, so every AMReX GPU
// min/max reduction returned zero. Fixed by compiling device code with
// -fno-strict-aliasing, matching nvcc.
struct MinMax { double lo; double hi; };

__device__ __forceinline__ MinMax multi_shuffle_down(MinMax x, int offset) {
    constexpr int nwords = sizeof(MinMax) / sizeof(unsigned int);
    MinMax y;
    auto* py = reinterpret_cast<unsigned int*>(&y);
    auto* px = reinterpret_cast<unsigned int*>(&x);
    for (int i = 0; i < nwords; ++i) {
        py[i] = __shfl_down_sync(0xffffffffu, px[i], offset);
    }
    return y;
}

__global__ void tuple_reduce_kernel(const double* in, double* out) {
    const int lane = threadIdx.x;
    MinMax x{in[lane], in[lane]};
    for (int off = 16; off > 0; off /= 2) {
        const MinMax y = multi_shuffle_down(x, off);
        x.lo = x.lo < y.lo ? x.lo : y.lo;
        x.hi = x.hi > y.hi ? x.hi : y.hi;
    }
    if (lane == 0) { out[0] = x.lo; out[1] = x.hi; }
}

// ── 5. __nanosleep ───────────────────────────────────────────────────────────
// sm_70's backoff hint, used inside AMReX's FillBoundary device-side lock loop.
// Metal has no sleep instruction, so CuMetal's is an optimization barrier only;
// the requirement here is just that it compiles and does not perturb the result.
__global__ void nanosleep_kernel(int* o) {
    int v = 0;
    for (int i = 0; i < 4; ++i) { __nanosleep(1); v += i; }
    o[0] = v;
}

// ── 6. binary64 device RNG ───────────────────────────────────────────────────
// AMReX's amrex::RandomNormal calls curand_normal_double when Real is double.
__global__ void curand_double_kernel(double* uniform, double* normal, int n) {
    curandState_t state;
    curand_init(1234ULL, 0ULL, 0ULL, &state);
    for (int i = 0; i < n; ++i) {
        uniform[i] = curand_uniform_double(&state);
        normal[i] = curand_normal_double(&state);
    }
}

#define CUDA_OK(expr)                                                              \
    do {                                                                           \
        const cudaError_t err_ = (expr);                                           \
        if (err_ != cudaSuccess) {                                                 \
            std::printf("FAIL: %s -> %s\n", #expr, cudaGetErrorString(err_));      \
            return 1;                                                              \
        }                                                                          \
    } while (0)

bool close_rel(double got, double want, double tol) {
    const double denom = std::fabs(want) > 1.0 ? std::fabs(want) : 1.0;
    return std::fabs(got - want) / denom <= tol;
}

}  // namespace

int main() {
    std::printf("AMReX device idioms\n");

    constexpr int kLanes = 32;

    // ── std:: math ───────────────────────────────────────────────────────────
    {
        double hx[kLanes];
        float hxf[kLanes];
        for (int i = 0; i < kLanes; ++i) {
            hx[i] = 0.5 + 0.25 * i;
            hxf[i] = static_cast<float>(hx[i]);
        }
        double *dx, *dod;
        float *dxf, *dof;
        CUDA_OK(cudaMalloc(&dx, sizeof(hx)));
        CUDA_OK(cudaMalloc(&dod, sizeof(hx)));
        CUDA_OK(cudaMalloc(&dxf, sizeof(hxf)));
        CUDA_OK(cudaMalloc(&dof, sizeof(hxf)));
        CUDA_OK(cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dxf, hxf, sizeof(hxf), cudaMemcpyHostToDevice));
        std_math_kernel<<<1, kLanes>>>(dx, dod, dxf, dof);
        CUDA_OK(cudaDeviceSynchronize());
        double god[kLanes];
        float gof[kLanes];
        CUDA_OK(cudaMemcpy(god, dod, sizeof(god), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(gof, dof, sizeof(gof), cudaMemcpyDeviceToHost));

        double worst_d = 0.0, worst_f = 0.0;
        for (int i = 0; i < kLanes; ++i) {
            const double v = hx[i];
            const double want = std::sqrt(v) + std::fabs(v) + std::floor(v) + std::ceil(v) +
                                std::trunc(v) + std::fmin(v, 3.0) + std::fmax(v, 1.0) +
                                std::copysign(v, -1.0);
            worst_d = std::fmax(worst_d, std::fabs(god[i] - want) / std::fabs(want));
            const float f = hxf[i];
            const float wantf = std::sqrt(f) + std::fabs(f) + std::floor(f) + std::ceil(f) +
                                std::trunc(f) + std::fmin(f, 3.0f) + std::fmax(f, 1.0f) +
                                std::copysign(f, -1.0f);
            worst_f = std::fmax(worst_f, std::fabs(gof[i] - wantf) / std::fabs(wantf));
        }
        char detail[96];
        // Algebraic binary64 under emulation carries a ~48-bit significand.
        std::snprintf(detail, sizeof(detail), "worst rel double %.2e float %.2e", worst_d, worst_f);
        report("std:: math, algebraic", worst_d < 1e-13 && worst_f < 1e-6, detail);
    }
    {
        double hx[kLanes];
        for (int i = 0; i < kLanes; ++i) hx[i] = 0.25 + 0.125 * i;
        double *dx, *dout;
        CUDA_OK(cudaMalloc(&dx, sizeof(hx)));
        CUDA_OK(cudaMalloc(&dout, sizeof(hx)));
        CUDA_OK(cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice));
        std_transcendental_kernel<<<1, kLanes>>>(dx, dout);
        CUDA_OK(cudaDeviceSynchronize());
        double got[kLanes];
        CUDA_OK(cudaMemcpy(got, dout, sizeof(got), cudaMemcpyDeviceToHost));
        double worst = 0.0;
        for (int i = 0; i < kLanes; ++i) {
            const double v = hx[i];
            const double want = std::exp(v) + std::log(v) + std::sin(v) + std::tanh(v) +
                                std::atan2(v, 2.0) + std::pow(v, 2.0) + std::hypot(v, 1.0);
            worst = std::fmax(worst, std::fabs(got[i] - want) / std::fabs(want));
        }
        char detail[96];
        std::snprintf(detail, sizeof(detail), "worst rel %.2e (binary32-accurate by policy)", worst);
        // Deliberately loose: the contract being checked is that the call
        // reaches the right function, not that it is accurate to binary64.
        report("std:: math, transcendental", worst < 1e-5, detail);
    }
    {
        double* dout;
        CUDA_OK(cudaMalloc(&dout, sizeof(double)));
        std_promotion_kernel<<<1, 1>>>(dout);
        CUDA_OK(cudaDeviceSynchronize());
        double got = 0.0;
        CUDA_OK(cudaMemcpy(&got, dout, sizeof(got), cudaMemcpyDeviceToHost));
        const double want = std::pow(2, 3) + std::sqrt(9) + std::log(1) + std::fmax(1, 2.0f);
        char detail[64];
        std::snprintf(detail, sizeof(detail), "got %.6f want %.6f", got, want);
        report("std:: math, integral promotion", close_rel(got, want, 1e-12), detail);
    }

    // ── __umul64hi / __mul64hi ───────────────────────────────────────────────
    {
        const unsigned long long ha[] = {
            0ULL, 1ULL, 0xFFFFFFFFULL, 0x100000000ULL, 0xFFFFFFFFFFFFFFFFULL,
            0x0123456789ABCDEFULL, 0xDEADBEEFCAFEBABEULL, 3037000499ULL,
        };
        const unsigned long long hb[] = {
            12345ULL, 1ULL, 0xFFFFFFFFULL, 0x100000000ULL, 0xFFFFFFFFFFFFFFFFULL,
            0x0FEDCBA987654321ULL, 0x0123456789ABCDEFULL, 3037000499ULL,
        };
        constexpr int n = sizeof(ha) / sizeof(ha[0]);
        unsigned long long *da, *db, *dhu;
        long long* dhs;
        CUDA_OK(cudaMalloc(&da, sizeof(ha)));
        CUDA_OK(cudaMalloc(&db, sizeof(hb)));
        CUDA_OK(cudaMalloc(&dhu, sizeof(ha)));
        CUDA_OK(cudaMalloc(&dhs, sizeof(ha)));
        CUDA_OK(cudaMemcpy(da, ha, sizeof(ha), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(db, hb, sizeof(hb), cudaMemcpyHostToDevice));
        mul64hi_kernel<<<1, n>>>(da, db, dhu, dhs);
        CUDA_OK(cudaDeviceSynchronize());
        unsigned long long gu[n];
        long long gs[n];
        CUDA_OK(cudaMemcpy(gu, dhu, sizeof(gu), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(gs, dhs, sizeof(gs), cudaMemcpyDeviceToHost));
        int bad_u = 0, bad_s = 0;
        for (int i = 0; i < n; ++i) {
            const auto want_u = static_cast<unsigned long long>(
                (static_cast<__uint128_t>(ha[i]) * hb[i]) >> 64);
            const auto want_s = static_cast<long long>(
                (static_cast<__int128_t>(static_cast<long long>(ha[i])) *
                 static_cast<long long>(hb[i])) >> 64);
            if (gu[i] != want_u) ++bad_u;
            if (gs[i] != want_s) ++bad_s;
        }
        char detail[64];
        std::snprintf(detail, sizeof(detail), "%d/%d exact", 2 * n - bad_u - bad_s, 2 * n);
        report("__umul64hi / __mul64hi", bad_u == 0 && bad_s == 0, detail);
    }

    // ── long shuffles ────────────────────────────────────────────────────────
    {
        long* dl;
        unsigned long* dul;
        CUDA_OK(cudaMalloc(&dl, kLanes * sizeof(long)));
        CUDA_OK(cudaMalloc(&dul, sizeof(unsigned long)));
        long_shuffle_kernel<<<1, kLanes>>>(dl, dul);
        CUDA_OK(cudaDeviceSynchronize());
        long gl[kLanes];
        unsigned long gul = 0;
        CUDA_OK(cudaMemcpy(gl, dl, sizeof(gl), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&gul, dul, sizeof(gul), cudaMemcpyDeviceToHost));
        // 1..32 summed by the same down-shuffle ladder AMReX's warpReduce uses.
        const long want_sum = kLanes * (kLanes + 1) / 2;
        const unsigned long want_u = 3UL * 3UL + 7UL;
        char detail[80];
        std::snprintf(detail, sizeof(detail), "sum %ld want %ld, idx %lu want %lu",
                      gl[0], want_sum, gul, want_u);
        report("long / unsigned long warp shuffle", gl[0] == want_sum && gul == want_u, detail);
    }

    // ── punned tuple warp reduce ─────────────────────────────────────────────
    {
        double hin[kLanes];
        for (int i = 0; i < kLanes; ++i) hin[i] = 1.0 + i;
        double *din, *dout;
        CUDA_OK(cudaMalloc(&din, sizeof(hin)));
        CUDA_OK(cudaMalloc(&dout, 2 * sizeof(double)));
        CUDA_OK(cudaMemcpy(din, hin, sizeof(hin), cudaMemcpyHostToDevice));
        tuple_reduce_kernel<<<1, kLanes>>>(din, dout);
        CUDA_OK(cudaDeviceSynchronize());
        double got[2] = {0.0, 0.0};
        CUDA_OK(cudaMemcpy(got, dout, sizeof(got), cudaMemcpyDeviceToHost));
        char detail[80];
        std::snprintf(detail, sizeof(detail), "min %.1f max %.1f want 1.0 %.1f",
                      got[0], got[1], static_cast<double>(kLanes));
        report("punned tuple warp reduce (min/max)",
               got[0] == 1.0 && got[1] == static_cast<double>(kLanes), detail);
    }

    // ── __nanosleep ──────────────────────────────────────────────────────────
    {
        int* dout;
        CUDA_OK(cudaMalloc(&dout, sizeof(int)));
        nanosleep_kernel<<<1, 1>>>(dout);
        CUDA_OK(cudaDeviceSynchronize());
        int got = -1;
        CUDA_OK(cudaMemcpy(&got, dout, sizeof(got), cudaMemcpyDeviceToHost));
        char detail[48];
        std::snprintf(detail, sizeof(detail), "got %d want 6", got);
        report("__nanosleep in a spin loop", got == 6, detail);
    }

    // ── binary64 device RNG ──────────────────────────────────────────────────
    {
        constexpr int n = 4096;
        double *du, *dn;
        CUDA_OK(cudaMalloc(&du, n * sizeof(double)));
        CUDA_OK(cudaMalloc(&dn, n * sizeof(double)));
        curand_double_kernel<<<1, 1>>>(du, dn, n);
        CUDA_OK(cudaDeviceSynchronize());
        static double hu[n], hn[n];
        CUDA_OK(cudaMemcpy(hu, du, sizeof(hu), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(hn, dn, sizeof(hn), cudaMemcpyDeviceToHost));
        double umin = hu[0], umax = hu[0], usum = 0.0, nsum = 0.0, nsq = 0.0;
        int distinct_low_bits = 0;
        for (int i = 0; i < n; ++i) {
            umin = std::fmin(umin, hu[i]);
            umax = std::fmax(umax, hu[i]);
            usum += hu[i];
            nsum += hn[i];
            nsq += hn[i] * hn[i];
            // A binary64 uniform drawn by widening a binary32 one would leave
            // the low mantissa bits zero in every sample.
            std::uint64_t bits;
            std::memcpy(&bits, &hu[i], sizeof(bits));
            if ((bits & 0x1FFFFFFFULL) != 0) ++distinct_low_bits;
        }
        const double umean = usum / n;
        const double nmean = nsum / n;
        const double nvar = nsq / n - nmean * nmean;
        const bool ok = umin >= 0.0 && umax < 1.0 && std::fabs(umean - 0.5) < 0.05 &&
                        std::fabs(nmean) < 0.1 && std::fabs(nvar - 1.0) < 0.15 &&
                        distinct_low_bits > n / 2;
        char detail[128];
        std::snprintf(detail, sizeof(detail),
                      "uniform mean %.3f, normal mean %.3f var %.3f, %d/%d full-width",
                      umean, nmean, nvar, distinct_low_bits, n);
        report("curand binary64 device RNG", ok, detail);
    }

    if (g_failures == 0) {
        std::printf("PASS: AMReX device idioms match host references on Apple GPU\n");
        return 0;
    }
    std::printf("FAIL: %d AMReX device idiom check(s) disagreed with the host\n", g_failures);
    return 1;
}
