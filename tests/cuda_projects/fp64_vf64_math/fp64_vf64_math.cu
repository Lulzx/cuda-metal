// Double libdevice transcendentals under CUMETAL_FP64_MODE=ieee64 call VF64's
// correctly rounded binary64 functions (vf64_<name>_rne). Before, every one of
// them was evaluated in binary32 and re-encoded: about seven significant
// digits, and anything outside float range -- exp(100), log(1e300) -- was
// infinity or zero. The gate is <= 1 ulp from the host's libm, which is itself
// within 1 ulp of the exact result; the binary32 path is ~2^29 ulps away.
// Under fast48/wide48 the same calls still run in binary32, so this probe is
// registered for ieee64 only.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

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
