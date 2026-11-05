// Double libdevice transcendentals under CUMETAL_FP64_MODE=ieee64 call VF64's
// correctly rounded binary64 functions (vf64_<name>_rne). Before, every one of
// them was evaluated in binary32 and re-encoded: about seven significant
// digits, and anything outside float range -- exp(100), log(1e300) -- was
// infinity or zero. The gate is <= 1 ulp from the host's libm, which is itself
// within 1 ulp of the exact result; the binary32 path is ~2^29 ulps away.
// The complete surface is registered for ieee64. CUMETAL_TEST_WIDE_HYPOT
// selects the separate reduced-hypot contract without exercising other math.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#if defined(CUMETAL_TEST_WIDE_HYPOT)
#include <limits>
#include <vector>

__global__ void reduced_hypot(const double* x, const double* y, double* out,
                              int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = hypot(x[i], y[i]);
}

static double from_bits(uint64_t bits) {
    double value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

int main() {
    std::vector<double> x, y, expected;
    auto add = [&](double a, double b, double result) {
        x.push_back(a); y.push_back(b); expected.push_back(result);
    };
    // Exact scaled 3-4-5 norms cover subnormals and both binary32 range gaps.
    for (int exponent : {-1074, -1073, -1050, -1022, -1000, -600, -200,
                          0, 200, 600, 1000, 1019}) {
        for (int sign : {-1, 1}) {
            add(sign * std::scalbn(3.0, exponent), std::scalbn(4.0, exponent),
                std::scalbn(5.0, exponent));
        }
    }
    // Independently rounded by VF64 tools/m9/m9_ref.c, MPFR 4.2.2,
    // f64_hypot rnear_even random 16 37. Inputs retain nonzero low limbs.
    static const uint64_t vectors[][3] = {
        {0xf13e58c354986f2dull, 0xf13e58c35030a7b4ull, 0x7145755850b914faull},
        {0x2cfcb788394f8607ull, 0xacfcb7883ceca6d8ull, 0x2d044e5177f81a0cull},
        {0xdbe01a2d14000000ull, 0xdc0cf861ec000000ull, 0x5c0d3fa3cd000000ull},
        {0xf83a3023fdc22dbdull, 0x73b39ed627a68e0aull, 0x783a3023fdc22dbdull},
        {0x1290eaed443ea3feull, 0x0de64a75a7d99daeull, 0x1290eaed443ea3feull},
        {0x800da8a962c5e1f0ull, 0x0006c80bec9676cbull, 0x000f3fe88a85cba5ull},
        {0xe846e0216f67d1d2ull, 0xec7b046c2f384443ull, 0x6c7b046c2f384443ull},
        {0x7e268804a3e4a513ull, 0x7f49bc3ce744634bull, 0x7f49bc3ce745011aull},
        {0xdfc781e23e446646ull, 0xdf7187f5da6007f4ull, 0x5fc783848f5cd5eaull},
        {0x9a264eb732400000ull, 0x1a446087a7800000ull, 0x1a45205e5e900000ull},
        {0x8e115deef0000000ull, 0x0e196e8854000001ull, 0x0e1ecbbd74000001ull},
        {0xe38f2854a37f851cull, 0x61fe84ed1c00898full, 0x638f2854a37f8520ull},
        {0x916ce201451c995cull, 0x1174598a0f5f25a7ull, 0x1178f40baeb2d38aull},
        {0xdbde172b82465196ull, 0x5b9e77e15a017001ull, 0x5bde26947722f272ull},
        {0xdaaef4211311973aull, 0xdc6904ec763bac3aull, 0x5c6904ec763bac3aull},
        {0xf94b81e217f8ed86ull, 0x7cd5564152c58538ull, 0x7cd5564152c58538ull},
    };
    for (const auto& v : vectors) add(from_bits(v[0]), from_bits(v[1]), from_bits(v[2]));
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    add(0.0, -0.0, 0.0);
    add(-0.0, -0.0, 0.0);
    add(infinity, nan, infinity);
    add(nan, -infinity, infinity);
    add(nan, 1.0, nan);
    add(1.0, nan, nan);
    add(from_bits(0x7ff4000000000000ull), infinity, infinity);
    add(-3.0, 0.0, 3.0);
    add(0.0, -4.0, 4.0);

    const int count = static_cast<int>(x.size());
    const size_t bytes = x.size() * sizeof(double);
    double *dx, *dy, *dout;
    if (cudaMalloc(&dx, bytes) != cudaSuccess ||
        cudaMalloc(&dy, bytes) != cudaSuccess ||
        cudaMalloc(&dout, bytes) != cudaSuccess ||
        cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dy, y.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("FAIL: wide48 hypot allocation or upload failed\n");
        return 1;
    }
    reduced_hypot<<<(count + 31) / 32, 32>>>(dx, dy, dout, count);
    std::vector<double> observed(x.size());
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(observed.data(), dout, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("FAIL: wide48 hypot launch or download failed\n");
        return 1;
    }
    int failures = 0;
    double worst_normal = 0;
    for (int i = 0; i < count; ++i) {
        const double got = observed[i], want = expected[i];
        bool passes;
        if (std::isnan(want)) passes = std::isnan(got);
        else if (std::isinf(want)) passes = got == want;
        else if (want == 0) passes = got == 0 && !std::signbit(got);
        else {
            const double error = std::abs(got - want);
            const double budget = std::fmax(0x1p-44 * want,
                std::nextafter(want, infinity) - want);
            passes = std::isfinite(got) && got >= 0 && error <= budget;
            if (want >= std::numeric_limits<double>::min()) {
                worst_normal = std::fmax(worst_normal, error / want);
            }
        }
        if (!passes && failures++ < 10) {
            printf("FAIL: wide48 hypot(%.17g, %.17g) = %.17g, want %.17g\n",
                   x[i], y[i], got, want);
        }
    }
    cudaFree(dx); cudaFree(dy); cudaFree(dout);
    if (failures) {
        printf("FAIL: wide48 hypot %d of %d results exceed the reduced-mode gate\n",
               failures, count);
        return 1;
    }
    printf("PASS: wide48 hypot analytic and MPFR vectors on Apple GPU (%d results, "
           "worst normal relative error %.3g)\n", count, worst_normal);
    return 0;
}
#else
#define UNARY(F) F(exp) F(exp2) F(expm1) F(log) F(log2) F(log1p) F(cbrt) F(atan) F(asin) \
    F(acos) F(sin) F(cos) F(tan) F(sinh) F(cosh) F(tanh) F(asinh) F(acosh) F(atanh)
#define BINARY(F) F(hypot) F(pow) F(atan2)

constexpr int kUnary = 19, kBinary = 3, kN = 12;

__global__ void unary_math(const double* x, double* out) {
    const int i = threadIdx.x;
    int f = 0;
#define APPLY(name) out[f++ * kN + i] = name(x[i]);
    UNARY(APPLY)
#undef APPLY
}

__global__ void binary_math(const double* x, const double* y, double* out) {
    const int i = threadIdx.x;
    int f = 0;
#define APPLY(name) out[f++ * kN + i] = name(x[i], y[i]);
    BINARY(APPLY)
#undef APPLY
}

static int64_t ordered(double v) {
    int64_t bits;
    memcpy(&bits, &v, 8);
    return bits < 0 ? INT64_MIN - bits : bits;
}

static bool close(double got, double want) {
    if (std::isnan(want)) return std::isnan(got);
    if (std::isinf(want) || want == 0) return got == want;
    const int64_t d = ordered(got) - ordered(want);
    return d >= -1 && d <= 1;
}

int main() {
    // Inside every function's domain where it matters, plus values far outside
    // binary32 range and tiny arguments that binary32 flushes or rounds away.
    const double xs[kN] = {0.5, 1.0e-12, 0.75, 0.3, 2.5, 1.0e-300, 0.999, 0.1, 100.0, 700.0,
                           1.0e300, 3.0e-7};
    const double ys[kN] = {2.0, 1.5, -0.5, 3.0, 0.25, 1.0e-300, 2.0, -3.0, 0.5, 1.0e-2,
                           1.0e-300, 7.0};
    // Out-of-domain lanes (asin(100), acosh(0.5), ...) must be NaN on both.
    const double* xu = xs;
    double *dx, *dy, *du, *db;
    cudaMalloc(&dx, sizeof xs);
    cudaMalloc(&dy, sizeof ys);
    cudaMalloc(&du, sizeof(double) * kUnary * kN);
    cudaMalloc(&db, sizeof(double) * kBinary * kN);
    cudaMemcpy(dx, xs, sizeof xs, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, ys, sizeof ys, cudaMemcpyHostToDevice);
    unary_math<<<1, kN>>>(dx, du);
    binary_math<<<1, kN>>>(dx, dy, db);
    double u[kUnary * kN], b[kBinary * kN];
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(u, du, sizeof u, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(b, db, sizeof b, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("FAIL: launch failed\n");
        return 1;
    }
    int failures = 0, checked = 0, f = 0;
    const char* names[] = {
#define NAME(name) #name,
        UNARY(NAME) BINARY(NAME)
#undef NAME
    };
#define CHECK_U(name) \
    for (int i = 0; i < kN; ++i, ++checked) { \
        const double want = std::name(xu[i]), got = u[f * kN + i]; \
        if (!close(got, want)) { \
            if (failures++ < 12) printf("FAIL: %s(%.17g) = %.17g, host %.17g\n", names[f], xu[i], got, want); \
        } \
    } \
    ++f;
#define CHECK_B(name) \
    for (int i = 0; i < kN; ++i, ++checked) { \
        const double want = std::name(xu[i], ys[i]), got = b[(f - kUnary) * kN + i]; \
        if (!close(got, want)) { \
            if (failures++ < 12) printf("FAIL: %s(%.17g, %.17g) = %.17g, host %.17g\n", names[f], xu[i], ys[i], got, want); \
        } \
    } \
    ++f;
    UNARY(CHECK_U)
    BINARY(CHECK_B)
    if (failures) {
        printf("FAIL: %d of %d results more than 1 ulp from host libm\n", failures, checked);
        return 1;
    }
    printf("PASS: 22 double libdevice functions within 1 ulp on Apple GPU (%d results)\n", checked);
    return 0;
}

#endif
