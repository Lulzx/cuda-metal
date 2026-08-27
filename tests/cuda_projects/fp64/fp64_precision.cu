// fp64_precision.cu — contract probe for CuMetal's FP64 emulation.
//
// Metal has no `double`, so FP64 arithmetic is emulated with Dekker FP32 pairs.
// Storage stays IEEE-754 binary64 everywhere the CUDA ABI is visible (register
// slots, .local spills, global memory, libdevice call slots); the pair exists
// only inside a single instruction's ALU. This probe pins down that contract:
//
//   * relative error <= 2^-48, plus an absolute floor at binary32's min normal
//   * splitting and re-packing is idempotent, so error does not compound
//   * signed zero, infinity and NaN survive
//   * the same eight bytes read back as a uint64_t are unchanged
//   * shared memory, warp shuffles and store/reload keep the extra bits
//
// It exists because "the kernel compiled and ran" says nothing about how many
// significand bits survived: the previous implementation dropped the low limb
// at every register write and silently ran at ~24 bits, which compiles and runs
// perfectly while quietly turning an LP solver's residuals to noise.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 4096
static const double kBound = 1.0 / 281474976710656.0;          // 2^-48
static const double kFltMin = 1.17549435082228751e-38;

// in[n] is a runtime 1.0, so the multiply survives constant folding and the
// value really does make a round trip through the pair ALU.
__global__ void trip(double* out, const double* in, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = in[i] * in[n];
}
__global__ void chain(double* out, const double* a, const double* b, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        double x = a[i], y = b[i];
        double t = x * y; t = t + x; t = t - y; t = t * x; t = t / y;
        out[i] = t;
    }
}
__global__ void shared_reduce(double* out, const double* in) {
    __shared__ double s[32];
    s[threadIdx.x] = in[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) { double v = 0.0; for (int i = 0; i < 32; ++i) v += s[i]; out[0] = v; }
}
__global__ void shuffle_reduce(double* out, const double* in) {
    double v = in[threadIdx.x];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, off);
    if (threadIdx.x == 0) out[0] = v;
}
__global__ void store_a(double* scratch, const double* in) { scratch[0] = in[0] * in[1]; }
__global__ void store_b(double* out, const double* scratch, const double* in) {
    out[0] = scratch[0] + in[0];
}
// Widen each limb and add: exercises the packer over arbitrary pairs, including
// alignment shifts large enough that the join must round rather than be exact.
__global__ void pair_join(double* out, const float* hi, const float* lo, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = (double) hi[i] + (double) lo[i];
}
__global__ void alias(unsigned long long* out, const double* in) {
    double v = in[0] * in[1];
    out[0] = *reinterpret_cast<unsigned long long*>(&v);
}

static uint64_t bits_of(double d) { uint64_t u; memcpy(&u, &d, sizeof u); return u; }

static int failures = 0;
static void report(const char* what, int bad, int total, double worst, const char* unit) {
    printf("  %-30s %5d/%-5d violations, worst %s %.3g\n", what, bad, total, unit, worst);
    failures += bad;
}
static void expect(const char* what, double got, double want, double tol) {
    const double rel = want == 0.0 ? fabs(got) : fabs(got - want) / fabs(want);
    const bool ok = rel <= tol;
    if (!ok) ++failures;
    printf("  %-30s got %-24.17g want %-24.17g %s\n", what, got, want, ok ? "ok" : "FAIL");
}

int main() {
    double* h_in = (double*) malloc((N + 1) * sizeof(double));
    double* h_b  = (double*) malloc((N + 1) * sizeof(double));
    double* h_y  = (double*) malloc((N + 1) * sizeof(double));
    double* h_z  = (double*) malloc(N * sizeof(double));
    double *d_in, *d_b, *d_y, *d_z;
    unsigned long long* d_bits;
    cudaMalloc(&d_in, (N + 1) * sizeof(double));
    cudaMalloc(&d_b,  (N + 1) * sizeof(double));
    cudaMalloc(&d_y,  (N + 1) * sizeof(double));
    cudaMalloc(&d_z,  N * sizeof(double));
    cudaMalloc(&d_bits, sizeof(unsigned long long));

    int n = 0;
    double payload_nan;
    { const uint64_t b = 0x7FF8000ABCDEF123ull; memcpy(&payload_nan, &b, sizeof b); }
    const double boundaries[] = {
        0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 16777217.0, -16777217.0, 16777216.0, 16777215.0,
        1.0000000000000002, 0.9999999999999999, 3.0000000000000004, 123456789.123456789,
        1.0 + 1.0 / 4503599627370496.0,   // 1 + 2^-52: needs all 53 bits
        1.0 + 1.0 / 281474976710656.0,    // 1 + 2^-48: exactly at the retained edge
        1.0 + 1.0 / 8796093022208.0,      // 1 + 2^-43: comfortably retained
        // Special values, and values outside the FP32 exponent envelope the pair
        // inherits. Without these the special-value check below passes vacuously.
        INFINITY, -INFINITY, NAN, -NAN, payload_nan,
        1e300, -1e300, 1e-300, 5e-324,    // beyond FLT_MAX / below FLT_MIN
        1.7976931348623157e308,           // DBL_MAX
    };
    for (double d : boundaries) h_in[n++] = d;
    srand(12345);
    while (n < N) {
        // Exponents stay inside binary32's range: the pair inherits it, and the
        // edges are a documented limit rather than a rounding question.
        uint64_t m = ((uint64_t) rand() << 40) ^ ((uint64_t) rand() << 20) ^ (uint64_t) rand();
        double v = ldexp(1.0 + (double) (m & 0xFFFFFFFFFFFFFull) / 9007199254740992.0,
                         -90 + (rand() % 180));
        h_in[n++] = (rand() & 1) ? -v : v;
    }
    h_in[N] = 1.0;
    cudaMemcpy(d_in, h_in, (N + 1) * sizeof(double), cudaMemcpyHostToDevice);

    printf("fp64 emulation contract probe\n");

    // ── round-trip error bound, and idempotence of split/pack ───────────────
    trip<<<16, 128>>>(d_y, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);
    h_y[N] = 1.0;
    cudaMemcpy(d_b, h_y, (N + 1) * sizeof(double), cudaMemcpyHostToDevice);
    trip<<<16, 128>>>(d_z, d_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_z, d_z, N * sizeof(double), cudaMemcpyDeviceToHost);

    int bound_bad = 0, idem_bad = 0, special_bad = 0;
    int precision_n = 0, special_n = 0, range_n = 0;
    double worst = 0.0;
    for (int i = 0; i < N; ++i) {
        const double x = h_in[i], y = h_y[i];
        if (x == 0.0) {
            ++special_n;
            if (bits_of(y) != bits_of(x)) ++special_bad;      // including -0.0
        } else if (std::isinf(x)) {
            ++special_n;
            if (y != x) ++special_bad;
        } else if (std::isnan(x)) {
            ++special_n;
            if (!std::isnan(y)) ++special_bad;
        } else if (fabs(x) > 3.4028234663852886e38 || fabs(x) < kFltMin) {
            // Outside binary32's normal range the pair cannot represent the
            // value at all: hi overflows to infinity or flushes to zero. That is
            // the documented envelope, reported separately rather than counted
            // as a precision failure.
            ++range_n;
        } else {
            ++precision_n;
            const double err = fabs(y - x);
            if (err > kBound * fabs(x) + kFltMin) ++bound_bad;
            const double rel = err / fabs(x);
            if (err <= kBound * fabs(x) && rel > worst) worst = rel;
        }
        if (bits_of(h_z[i]) != bits_of(y)) ++idem_bad;
    }
    report("round-trip <= 2^-48", bound_bad, precision_n, worst, "rel");
    report("pack(split(x)) idempotent", idem_bad, N, 0.0, "rel");
    report("signed zero / inf / nan", special_bad, special_n, 0.0, "rel");
    printf("  %-30s %5d values outside binary32's exponent envelope\n", "range-excluded", range_n);
    // The special-value line must not be able to pass vacuously.
    if (special_n < 7) {
        printf("  FAIL: only %d special values exercised; corpus lost its specials\n", special_n);
        ++failures;
    }

    // ── chained separate instructions against a host double reference ───────
    for (int i = 0; i < N; ++i) {
        const double v = h_in[(i * 7 + 3) % N];
        h_b[i] = v == 0.0 ? 1.0 : v;
    }
    cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice);
    chain<<<16, 128>>>(d_z, d_in, d_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_z, d_z, N * sizeof(double), cudaMemcpyDeviceToHost);
    int chain_bad = 0, chain_n = 0;
    double chain_worst = 0.0;
    for (int i = 0; i < N; ++i) {
        const double x = h_in[i], y = h_b[i];
        // Error is measured against the largest intermediate, not the result: a
        // chain that cancels leaves a result far below the scale it ran at, and
        // no finite precision recovers relative accuracy there.
        double scale = 0.0, floor_mag = INFINITY, t = x * y;
        #define STEP(e) do { t = (e); scale = fmax(scale, fabs(t)); \
                             if (t != 0.0) floor_mag = fmin(floor_mag, fabs(t)); } while (0)
        STEP(t); STEP(t + x); STEP(t - y); STEP(t * x); STEP(t / y);
        #undef STEP
        if (!std::isfinite(t) || !std::isfinite(h_z[i]) || scale == 0.0) continue;
        // Skip chains leaving binary32's exponent range: an intermediate below
        // FLT_MIN flushes to zero on the pair path. Documented range limit.
        if (ilogb(scale) > 100 || ilogb(floor_mag) < -100) continue;
        if (abs(ilogb(x)) > 100 || abs(ilogb(y)) > 100) continue;
        ++chain_n;
        const double rel = fabs(h_z[i] - t) / scale;
        if (rel > chain_worst) chain_worst = rel;
        // Deliberately loose: this is a collapse detector, not a precision gate.
        // Per-value error is 2^-48, but each operation composes its operands'
        // errors with its own, and the cancelling subtract feeding a divide
        // amplifies further -- ordinary finite-precision behaviour, just at 48
        // bits instead of 53. What it does catch is the low limb being dropped
        // again: that lands around 1e-7, four orders of magnitude away.
        if (rel > 1e-7) ++chain_bad;
    }
    report("5-op chain (collapse gate)", chain_bad, chain_n, chain_worst, "err/scale");

    // ── paths that must not quietly drop the low limb ───────────────────────
    // ── the packer over arbitrary limb pairs, not just ones a split produced ──
    // (double)hi + (double)lo widens each limb exactly and then runs the pair
    // add + pack, so sweeping the exponent gap exercises the alignment shift
    // across the exact region (<=29) and the rounded region (>29).
    {
        const int kPairs = 61 * 8;
        float* h_hi = (float*) malloc(kPairs * sizeof(float));
        float* h_lo = (float*) malloc(kPairs * sizeof(float));
        double* h_out = (double*) malloc(kPairs * sizeof(double));
        float *d_hi, *d_lo;
        cudaMalloc(&d_hi, kPairs * sizeof(float));
        cudaMalloc(&d_lo, kPairs * sizeof(float));
        int k = 0;
        for (int gap = 0; gap < 61; ++gap) {
            for (int v = 0; v < 8; ++v) {
                const float hi = ldexpf(1.0f + (float) v / 8.0f, (v & 1) ? 12 : -7);
                h_hi[k] = (v & 2) ? -hi : hi;
                h_lo[k] = ldexpf(hi, -gap) * ((v & 4) ? -0.75f : 0.625f);
                ++k;
            }
        }
        cudaMemcpy(d_hi, h_hi, kPairs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lo, h_lo, kPairs * sizeof(float), cudaMemcpyHostToDevice);
        pair_join<<<4, 128>>>(d_z, d_hi, d_lo, kPairs);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_z, kPairs * sizeof(double), cudaMemcpyDeviceToHost);
        int join_bad = 0; double join_worst = 0.0;
        for (int i = 0; i < kPairs; ++i) {
            const double want = (double) h_hi[i] + (double) h_lo[i];
            if (want == 0.0) continue;
            const double rel = fabs(h_out[i] - want) / fabs(want);
            if (rel > join_worst) join_worst = rel;
            if (rel > kBound) ++join_bad;
        }
        report("(double)hi + (double)lo", join_bad, kPairs, join_worst, "rel");
        free(h_hi); free(h_lo); free(h_out);
        cudaFree(d_hi); cudaFree(d_lo);
    }

    double vec[33], sum = 0.0;
    for (int i = 0; i < 32; ++i) { vec[i] = 1048576.0 + i * 0.000001; sum += vec[i]; }
    vec[32] = 1.0;
    cudaMemcpy(d_in, vec, sizeof vec, cudaMemcpyHostToDevice);
    double got = 0.0;
    shared_reduce<<<1, 32>>>(d_z, d_in);
    cudaDeviceSynchronize();
    cudaMemcpy(&got, d_z, sizeof got, cudaMemcpyDeviceToHost);
    expect("shared-memory reduction", got, sum, 1e-13);
    shuffle_reduce<<<1, 32>>>(d_z, d_in);
    cudaDeviceSynchronize();
    cudaMemcpy(&got, d_z, sizeof got, cudaMemcpyDeviceToHost);
    expect("warp-shuffle reduction", got, sum, 1e-13);

    double ab[2] = {1.0000000000000002, 3.0000000000000004};
    cudaMemcpy(d_in, ab, sizeof ab, cudaMemcpyHostToDevice);
    store_a<<<1,1>>>(d_z, d_in);
    store_b<<<1,1>>>(d_y, d_z, d_in);
    cudaDeviceSynchronize();
    cudaMemcpy(&got, d_y, sizeof got, cudaMemcpyDeviceToHost);
    expect("arith/store/reload/arith", got, ab[0] * ab[1] + ab[0], 1e-13);

    double av[2] = {1.0000000000000002, 1.0};
    cudaMemcpy(d_in, av, sizeof av, cudaMemcpyHostToDevice);
    alias<<<1,1>>>(d_bits, d_in);
    cudaDeviceSynchronize();
    unsigned long long ubits = 0;
    cudaMemcpy(&ubits, d_bits, sizeof ubits, cudaMemcpyDeviceToHost);
    double as_double;
    memcpy(&as_double, &ubits, sizeof as_double);
    expect("same bytes as uint64_t", as_double, av[0], 0.0);

    double rt[2] = {16777217.0, 1.0};       // 2^24 + 1: unrepresentable in binary32
    cudaMemcpy(d_in, rt, sizeof rt, cudaMemcpyHostToDevice);
    trip<<<1,1>>>(d_z, d_in, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(&got, d_z, sizeof got, cudaMemcpyDeviceToHost);
    expect("2^24+1 through fp64 mul", got, rt[0], 0.0);

    if (failures == 0) printf("PASS: fp64 emulation meets the ~48-bit significand contract\n");
    else printf("FAIL: %d fp64 contract violation(s)\n", failures);
    return failures != 0;
}
