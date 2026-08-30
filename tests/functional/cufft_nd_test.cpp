// Multidimensional cuFFT: rank-2 and rank-3 R2C/C2R/C2C against a direct DFT.
//
// The reference is a literal O(N^2) evaluation of the transform's definition in
// long double, computed here rather than taken from another FFT. That matters:
// two FFTs can agree on a wrong convention (a transposed axis order, a missing
// conjugate, an inverse that normalizes) and a round-trip test alone cannot see
// any of it, because the error cancels. Comparing forward output against the
// definition pins the convention down.
//
// Padded strides are exercised on purpose. GROMACS's PME grid is padded on the
// fastest axis and passes inembed/onembed to describe it, so a shim that quietly
// assumes contiguity produces a plausible-looking wrong grid.
#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

namespace {

using Ref = std::complex<long double>;

int g_failures = 0;

void report(const char* what, double worst, double tol) {
    if (worst <= tol) {
        std::printf("  %-34s ok    max err %.3e\n", what, worst);
    } else {
        std::printf("  %-34s FAIL  max err %.3e > %.3e\n", what, worst, tol);
        ++g_failures;
    }
}

// Direct 3-D DFT of a real grid at one output bin.
Ref reference_bin(const std::vector<float>& grid, int n0, int n1, int n2, int k0, int k1,
                  int k2, bool inverse) {
    const long double sign = inverse ? 1.0L : -1.0L;
    const long double two_pi = 2.0L * std::numbers::pi_v<long double>;
    Ref sum{0.0L, 0.0L};
    for (int i0 = 0; i0 < n0; ++i0) {
        for (int i1 = 0; i1 < n1; ++i1) {
            for (int i2 = 0; i2 < n2; ++i2) {
                const long double phase =
                    sign * two_pi *
                    (static_cast<long double>(i0) * k0 / n0 +
                     static_cast<long double>(i1) * k1 / n1 +
                     static_cast<long double>(i2) * k2 / n2);
                const long double value =
                    grid[(static_cast<std::size_t>(i0) * n1 + i1) * n2 + i2];
                sum += Ref{value * std::cos(phase), value * std::sin(phase)};
            }
        }
    }
    return sum;
}

// R2C on a grid padded on the fastest axis, then C2R back.
void test_r2c_c2r_3d(int n0, int n1, int n2, int real_pad, int complex_pad) {
    const int nc = n2 / 2 + 1;
    const int real_fast = n2 + real_pad;         // inembed[2]
    const int complex_fast = nc + complex_pad;   // onembed[2]

    std::vector<float> host(static_cast<std::size_t>(n0) * n1 * n2);
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = std::sin(0.7f * static_cast<float>(i)) + 0.25f * static_cast<float>(i % 5);
    }

    const std::size_t real_elems = static_cast<std::size_t>(n0) * n1 * real_fast;
    const std::size_t complex_elems = static_cast<std::size_t>(n0) * n1 * complex_fast;
    cufftReal* d_real = nullptr;
    cufftComplex* d_complex = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_real), real_elems * sizeof(cufftReal)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_complex),
                   complex_elems * sizeof(cufftComplex)) != cudaSuccess) {
        std::printf("  FAIL: cudaMalloc\n");
        ++g_failures;
        return;
    }

    // Fill the padding with a value that must never reach the output, so a shim
    // that ignores inembed shows up as a large error rather than a subtle one.
    std::vector<float> staged(real_elems, -12345.0f);
    for (int i0 = 0; i0 < n0; ++i0) {
        for (int i1 = 0; i1 < n1; ++i1) {
            for (int i2 = 0; i2 < n2; ++i2) {
                staged[(static_cast<std::size_t>(i0) * n1 + i1) * real_fast + i2] =
                    host[(static_cast<std::size_t>(i0) * n1 + i1) * n2 + i2];
            }
        }
    }
    cudaMemcpy(d_real, staged.data(), real_elems * sizeof(cufftReal), cudaMemcpyHostToDevice);

    int dims[3] = {n0, n1, n2};
    int inembed[3] = {n0, n1, real_fast};
    int onembed[3] = {n0, n1, complex_fast};

    cufftHandle forward = 0;
    if (cufftPlanMany(&forward, 3, dims, inembed, 1, static_cast<int>(real_elems), onembed, 1,
                      static_cast<int>(complex_elems), CUFFT_R2C, 1) != CUFFT_SUCCESS) {
        std::printf("  FAIL: cufftPlanMany R2C rank 3\n");
        ++g_failures;
        return;
    }
    if (cufftExecR2C(forward, d_real, d_complex) != CUFFT_SUCCESS) {
        std::printf("  FAIL: cufftExecR2C rank 3\n");
        ++g_failures;
        return;
    }

    std::vector<cufftComplex> spectrum(complex_elems);
    cudaMemcpy(spectrum.data(), d_complex, complex_elems * sizeof(cufftComplex),
               cudaMemcpyDeviceToHost);

    double worst = 0.0;
    double scale = 0.0;
    for (int k0 = 0; k0 < n0; ++k0) {
        for (int k1 = 0; k1 < n1; ++k1) {
            for (int k2 = 0; k2 < nc; ++k2) {
                const Ref want = reference_bin(host, n0, n1, n2, k0, k1, k2, false);
                const cufftComplex got =
                    spectrum[(static_cast<std::size_t>(k0) * n1 + k1) * complex_fast + k2];
                worst = std::max(worst, static_cast<double>(std::abs(
                                            Ref{got.x, got.y} - want)));
                scale = std::max(scale, static_cast<double>(std::abs(want)));
            }
        }
    }
    char label[96];
    std::snprintf(label, sizeof(label), "R2C %dx%dx%d pad %d/%d", n0, n1, n2, real_pad,
                  complex_pad);
    report(label, worst / (scale > 0 ? scale : 1.0), 2e-5);

    // Round trip: C2R of the spectrum must return N times the input.
    cufftHandle inverse = 0;
    if (cufftPlanMany(&inverse, 3, dims, onembed, 1, static_cast<int>(complex_elems), inembed,
                      1, static_cast<int>(real_elems), CUFFT_C2R, 1) != CUFFT_SUCCESS) {
        std::printf("  FAIL: cufftPlanMany C2R rank 3\n");
        ++g_failures;
        return;
    }
    cudaMemset(d_real, 0, real_elems * sizeof(cufftReal));
    if (cufftExecC2R(inverse, d_complex, d_real) != CUFFT_SUCCESS) {
        std::printf("  FAIL: cufftExecC2R rank 3\n");
        ++g_failures;
        return;
    }
    std::vector<float> back(real_elems);
    cudaMemcpy(back.data(), d_real, real_elems * sizeof(cufftReal), cudaMemcpyDeviceToHost);

    const double norm = static_cast<double>(n0) * n1 * n2;
    worst = 0.0;
    scale = 0.0;
    for (int i0 = 0; i0 < n0; ++i0) {
        for (int i1 = 0; i1 < n1; ++i1) {
            for (int i2 = 0; i2 < n2; ++i2) {
                const double want =
                    norm * host[(static_cast<std::size_t>(i0) * n1 + i1) * n2 + i2];
                const double got =
                    back[(static_cast<std::size_t>(i0) * n1 + i1) * real_fast + i2];
                worst = std::max(worst, std::abs(got - want));
                scale = std::max(scale, std::abs(want));
            }
        }
    }
    std::snprintf(label, sizeof(label), "C2R round trip %dx%dx%d", n0, n1, n2);
    report(label, worst / (scale > 0 ? scale : 1.0), 2e-5);

    cufftDestroy(forward);
    cufftDestroy(inverse);
    cudaFree(d_real);
    cudaFree(d_complex);
}

// Rank-2 C2C forward against the definition, with the second axis distinguishable
// from the first so a transposed traversal cannot pass.
void test_c2c_2d(int n0, int n1) {
    std::vector<float> real_part(static_cast<std::size_t>(n0) * n1);
    for (std::size_t i = 0; i < real_part.size(); ++i) {
        real_part[i] = std::cos(0.31f * static_cast<float>(i)) * (1.0f + (i % 3));
    }
    std::vector<cufftComplex> host(real_part.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i].x = real_part[i];
        host[i].y = 0.0f;
    }

    cufftComplex* d_data = nullptr;
    cufftComplex* d_out = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_data), host.size() * sizeof(cufftComplex));
    cudaMalloc(reinterpret_cast<void**>(&d_out), host.size() * sizeof(cufftComplex));
    cudaMemcpy(d_data, host.data(), host.size() * sizeof(cufftComplex),
               cudaMemcpyHostToDevice);

    cufftHandle plan = 0;
    if (cufftPlan2d(&plan, n0, n1, CUFFT_C2C) != CUFFT_SUCCESS ||
        cufftExecC2C(plan, d_data, d_out, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::printf("  FAIL: rank-2 C2C plan/exec\n");
        ++g_failures;
        return;
    }
    std::vector<cufftComplex> got(host.size());
    cudaMemcpy(got.data(), d_out, host.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    double worst = 0.0;
    double scale = 0.0;
    for (int k0 = 0; k0 < n0; ++k0) {
        for (int k1 = 0; k1 < n1; ++k1) {
            const Ref want = reference_bin(real_part, n0, n1, 1, k0, k1, 0, false);
            const cufftComplex value = got[static_cast<std::size_t>(k0) * n1 + k1];
            worst = std::max(worst,
                             static_cast<double>(std::abs(Ref{value.x, value.y} - want)));
            scale = std::max(scale, static_cast<double>(std::abs(want)));
        }
    }
    char label[96];
    std::snprintf(label, sizeof(label), "C2C %dx%d vs definition", n0, n1);
    report(label, worst / (scale > 0 ? scale : 1.0), 2e-5);

    cufftDestroy(plan);
    cudaFree(d_data);
    cudaFree(d_out);
}

}  // namespace

int main() {
    std::printf("cuFFT multidimensional transforms\n");

    // 40x32x32 is villin's PME grid. 12x10x14 carries a factor of 7 on the
    // fastest axis, which vDSP cannot factor -- that axis goes through Bluestein,
    // and this is the only place it is exercised.
    test_r2c_c2r_3d(4, 5, 8, 2, 1);
    test_r2c_c2r_3d(40, 32, 32, 2, 0);
    test_r2c_c2r_3d(12, 10, 14, 0, 3);
    test_c2c_2d(6, 9);
    test_c2c_2d(8, 7);

    if (g_failures != 0) {
        std::printf("FAIL: %d cuFFT multidimensional check(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS: cuFFT multidimensional transforms\n");
    return 0;
}
