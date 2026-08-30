// Libdevice math surface probe.
//
// The PTX->LLVM lowering maps CUDA math calls (__nv_sqrtf, __nv_tanf, ...) onto
// Metal AIR builtins. Those mappings were previously asserted only by checking
// that the emitted IR text contained the expected symbol name -- which proves
// nothing about whether the AIR symbol exists, links, or computes the right
// function. A wrong or missing mapping does not degrade gracefully: the whole
// kernel fails to lower, or links to something that silently computes garbage.
//
// This harness measures the surface instead of asserting it. Every function
// gets its OWN kernel, so one unsupported call cannot mask the rest, and each
// is scored independently:
//
//   SUPPORTED   launched and matched the host libm within tolerance
//   WRONG       launched but the numbers disagree -> mis-mapped builtin
//   UNSUPPORTED failed to lower -> mapping missing from lower_to_llvm.cpp
//
// Every input is confined to (0,1) and each entry folds its own domain shift
// into the expression, so the same text is valid on host and device.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#define N 1024
#define BLOCK 128

// id, device expression in x/y, host expression in x/y, relative tolerance.
//
// Tolerances split into two regimes. Exact ops (copies, sign, rounding, IEEE
// sqrt) must match to the bit. Everything routed through a Metal `fast_*`
// builtin is a lower-precision approximation by construction, so it gets 2e-3
// relative -- tight enough that a mis-mapped function (e.g. sin bound to cos)
// fails by orders of magnitude, loose enough not to flag honest fast-math error.
#define UNARY_LIST(X)                                                          \
    X(sqrtf,      sqrtf(x),         sqrtf(x),                     1e-6f)       \
    X(rsqrtf,     rsqrtf(x),        1.0f / sqrtf(x),              2e-3f)       \
    X(fabsf,      fabsf(x),         fabsf(x),                     0.0f)        \
    X(expf,       expf(x),          expf(x),                      2e-3f)       \
    X(exp2f,      exp2f(x),         exp2f(x),                     2e-3f)       \
    X(exp10f,     exp10f(x),        powf(10.0f, x),               2e-3f)       \
    X(expm1f,     expm1f(x),        expm1f(x),                    2e-3f)       \
    X(logf,       logf(x),          logf(x),                      2e-3f)       \
    X(log2f,      log2f(x),         log2f(x),                     2e-3f)       \
    X(log10f,     log10f(x),        log10f(x),                    2e-3f)       \
    X(log1pf,     log1pf(x),        log1pf(x),                    2e-3f)       \
    X(sinf,       sinf(x),          sinf(x),                      2e-3f)       \
    X(cosf,       cosf(x),          cosf(x),                      2e-3f)       \
    X(tanf,       tanf(x),          tanf(x),                      2e-3f)       \
    X(asinf,      asinf(x),         asinf(x),                     2e-3f)       \
    X(acosf,      acosf(x),         acosf(x),                     2e-3f)       \
    X(atanf,      atanf(x),         atanf(x),                     2e-3f)       \
    X(sinhf,      sinhf(x),         sinhf(x),                     2e-3f)       \
    X(coshf,      coshf(x),         coshf(x),                     2e-3f)       \
    X(tanhf,      tanhf(x),         tanhf(x),                     2e-3f)       \
    X(asinhf,     asinhf(x),        asinhf(x),                    2e-3f)       \
    X(acoshf,     acoshf(1.0f + x), acoshf(1.0f + x),             2e-3f)       \
    X(atanhf,     atanhf(x),        atanhf(x),                    2e-3f)       \
    X(cbrtf,      cbrtf(x),         cbrtf(x),                     2e-3f)       \
    X(erff,       erff(x),          erff(x),                      2e-3f)       \
    X(erfcf,      erfcf(x),         erfcf(x),                     2e-3f)       \
    X(floorf,     floorf(4.0f * x), floorf(4.0f * x),             0.0f)        \
    X(ceilf,      ceilf(4.0f * x),  ceilf(4.0f * x),              0.0f)        \
    X(truncf,     truncf(4.0f * x), truncf(4.0f * x),             0.0f)        \
    X(roundf,     roundf(4.0f * x), roundf(4.0f * x),             0.0f)        \
    X(rintf,      rintf(4.0f * x),  rintf(4.0f * x),              0.0f)       \
    /* Double rsqrt. GROMACS's nbnxm kernels call it, and until it was mapped   \
       an unlisted libdevice name aborted the whole kernel rather than          \
       degrading. Evaluated through the FP64 pair path, hence the fast-math     \
       tolerance. */                                                           \
    X(rsqrt_d,    (float)rsqrt((double)(x + 0.25f)),                            \
                  (float)(1.0 / sqrt((double)(x + 0.25f))),       2e-3f)       \
    /* Float->int conversions. The suffix names the rounding mode, which has to \
       be applied to the float before the cast; dropping it is the same defect  \
       class as cvt.rni silently truncating. The 4x-2 shift puts negatives in   \
       range, where a dropped mode shows up worst. */                          \
    X(f2i_rn,     (float)__float2int_rn(4.0f * x - 2.0f),                       \
                  nearbyintf(4.0f * x - 2.0f),                    0.0f)        \
    X(f2i_rz,     (float)__float2int_rz(4.0f * x - 2.0f),                       \
                  truncf(4.0f * x - 2.0f),                        0.0f)        \
    X(f2i_ru,     (float)__float2int_ru(4.0f * x - 2.0f),                       \
                  ceilf(4.0f * x - 2.0f),                         0.0f)        \
    X(f2i_rd,     (float)__float2int_rd(4.0f * x - 2.0f),                       \
                  floorf(4.0f * x - 2.0f),                        0.0f)

#define BINARY_LIST(X)                                                         \
    X(fmaxf,      fmaxf(x, y),      fmaxf(x, y),                  0.0f)        \
    X(fminf,      fminf(x, y),      fminf(x, y),                  0.0f)        \
    X(powf,       powf(x, y),       powf(x, y),                   2e-3f)       \
    X(atan2f,     atan2f(x, y),     atan2f(x, y),                 2e-3f)       \
    X(hypotf,     hypotf(x, y),     hypotf(x, y),                 2e-3f)       \
    X(fmodf,      fmodf(x, y),      fmodf(x, y),                  2e-3f)       \
    X(copysignf,  copysignf(x, -y), copysignf(x, -y),             0.0f)        \
    X(fdimf,      fdimf(x, y),      fdimf(x, y),                  0.0f)        \
    X(remainderf, remainderf(x, y), remainderf(x, y),             2e-3f)       \
    X(fmaf,       fmaf(x, y, x),    fmaf(x, y, x),                1e-6f)

// ------------------------------------------------------------- kernels

#define GEN_UNARY_KERNEL(id, dexpr, hexpr, tol)                                \
    __global__ void k_##id(const float* in, float* out, int n) {               \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                         \
        if (i < n) {                                                           \
            float x = in[i];                                                   \
            out[i] = (dexpr);                                                  \
        }                                                                      \
    }
UNARY_LIST(GEN_UNARY_KERNEL)

#define GEN_BINARY_KERNEL(id, dexpr, hexpr, tol)                               \
    __global__ void k_##id(const float* a, const float* b, float* out, int n) { \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                         \
        if (i < n) {                                                           \
            float x = a[i];                                                    \
            float y = b[i];                                                    \
            out[i] = (dexpr);                                                  \
        }                                                                      \
    }
BINARY_LIST(GEN_BINARY_KERNEL)

// PTX has two lowering paths, and `cvt` is implemented separately in each. The
// scalar kernels above are simple enough to be handled by the direct-MSL
// emitter, so they only ever exercise one of them. Shared memory plus a barrier
// forces this one down the LLVM path, where `cvt.rni.f32.f32` (what clang emits
// for rintf) previously degraded to a plain register copy -- rounding silently
// not happening at all.
__global__ void k_rint_shared(const float* in, float* out, int n) {
    __shared__ float tile[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = (i < n) ? in[i] * 4.0f : 0.0f;
    __syncthreads();
    if (i < n) out[i] = rintf(tile[threadIdx.x]);
}
static float ref_rint_shared(float x, float) { return rintf(x * 4.0f); }

// --------------------------------------------------------- host refs

#define GEN_UNARY_REF(id, dexpr, hexpr, tol)                                   \
    static float ref_##id(float x, float) { return (hexpr); }
UNARY_LIST(GEN_UNARY_REF)

#define GEN_BINARY_REF(id, dexpr, hexpr, tol)                                  \
    static float ref_##id(float x, float y) { return (hexpr); }
BINARY_LIST(GEN_BINARY_REF)

// ---------------------------------------------------------- scoring

static int g_supported = 0;
static int g_wrong = 0;
static int g_unsupported = 0;
static std::vector<std::string> g_wrong_names;
static std::vector<std::string> g_unsupported_names;

static std::vector<float> h_a, h_b, h_out;
static float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;

// Returns true if the launch lowered and ran. `launch` has already been issued
// by the caller; this only classifies the outcome.
static void score(const char* name, float (*ref)(float, float), float tol) {
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err == cudaSuccess) {
        launch_err = cudaDeviceSynchronize();
    }
    if (launch_err != cudaSuccess) {
        printf("  %-12s UNSUPPORTED  (%s)\n", name, cudaGetErrorString(launch_err));
        ++g_unsupported;
        g_unsupported_names.push_back(name);
        cudaGetLastError();  // clear sticky state before the next probe
        return;
    }

    if (cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("  %-12s UNSUPPORTED  (result copy failed)\n", name);
        ++g_unsupported;
        g_unsupported_names.push_back(name);
        cudaGetLastError();
        return;
    }

    float worst = 0.0f;
    int worst_i = 0;
    for (int i = 0; i < N; ++i) {
        float expect = ref(h_a[i], h_b[i]);
        float got = h_out[i];
        if (std::isnan(expect) && std::isnan(got)) continue;
        float denom = fmaxf(fabsf(expect), 1e-6f);
        float rel = fabsf(got - expect) / denom;
        if (rel > worst) { worst = rel; worst_i = i; }
    }
    if (!(worst <= tol)) {
        printf("  %-12s WRONG        rel err %.4g > %.4g (x=%.5f y=%.5f got=%.7g want=%.7g)\n",
               name, worst, tol, h_a[worst_i], h_b[worst_i], h_out[worst_i],
               ref(h_a[worst_i], h_b[worst_i]));
        ++g_wrong;
        g_wrong_names.push_back(name);
        return;
    }
    printf("  %-12s SUPPORTED    rel err %.3g\n", name, worst);
    ++g_supported;
}

int main(int argc, char** argv) {
    // A single unsupported call used to abort the whole probe; with one kernel
    // per function the run always completes and reports the full table.
    //
    // Strict by default: every function listed here is currently lowered, so a
    // newly-unsupported one is a regression. `--allow-missing` exists for
    // bringing up a new function without turning the suite red first.
    bool require_all = !(argc > 1 && std::string(argv[1]) == "--allow-missing");

    h_a.resize(N);
    h_b.resize(N);
    h_out.resize(N);
    for (int i = 0; i < N; ++i) {
        // Strictly inside (0,1): valid for log/asin/acos/atanh/sqrt alike, and
        // never hits the exact endpoints where fast builtins are allowed to
        // return inf and the relative check would be meaningless.
        h_a[i] = 0.01f + 0.98f * ((float)i / (float)(N - 1));
        h_b[i] = 0.99f - 0.97f * ((float)i / (float)(N - 1));
    }

    if (cudaMalloc((void**)&d_a, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&d_b, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&d_out, N * sizeof(float)) != cudaSuccess) {
        printf("FAIL: cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    printf("libdevice float surface probe (%d values per function)\n", N);
    printf("unary:\n");
#define RUN_UNARY(id, dexpr, hexpr, tol)                                       \
    cudaMemset(d_out, 0, N * sizeof(float));                                   \
    k_##id<<<grid, block>>>(d_a, d_out, N);                                    \
    score(#id, ref_##id, tol);
    UNARY_LIST(RUN_UNARY)

    printf("binary/ternary:\n");
#define RUN_BINARY(id, dexpr, hexpr, tol)                                      \
    cudaMemset(d_out, 0, N * sizeof(float));                                   \
    k_##id<<<grid, block>>>(d_a, d_b, d_out, N);                               \
    score(#id, ref_##id, tol);
    BINARY_LIST(RUN_BINARY)

    printf("shared-memory path (forces the LLVM lowering path):\n");
    cudaMemset(d_out, 0, N * sizeof(float));
    k_rint_shared<<<grid, block>>>(d_a, d_out, N);
    score("rint_shared", ref_rint_shared, 0.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    const int total = g_supported + g_wrong + g_unsupported;
    printf("\nsummary: %d/%d supported, %d wrong, %d unsupported\n",
           g_supported, total, g_wrong, g_unsupported);
    if (!g_wrong_names.empty()) {
        printf("wrong:");
        for (const auto& n : g_wrong_names) printf(" %s", n.c_str());
        printf("\n");
    }
    if (!g_unsupported_names.empty()) {
        printf("unsupported:");
        for (const auto& n : g_unsupported_names) printf(" %s", n.c_str());
        printf("\n");
    }

    // A mis-mapped builtin is always a bug: it computes the wrong function and
    // no caller can detect that. A missing mapping is a coverage gap; the test
    // gates on it only under --require-all so the probe stays runnable while
    // the surface is still being filled in.
    if (g_wrong > 0) {
        printf("FAIL: %d libdevice function(s) lower to the wrong builtin.\n", g_wrong);
        return 1;
    }
    if (require_all && g_unsupported > 0) {
        printf("FAIL: %d libdevice function(s) have no lowering.\n", g_unsupported);
        return 1;
    }
    printf("PASS: libdevice float surface probe (%d supported, %d unsupported)\n",
           g_supported, g_unsupported);
    return 0;
}
