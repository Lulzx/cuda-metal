// Double-precision reciprocal on the Apple GPU. Clang lowers `1.0 / x` to
// PTX `rcp.rn.f64`, which the typed backend imports as a division by the
// literal 1.0. binary64 travels as raw bits in a ulong there, so a decimal
// `1.0` became the integer 1 -- the smallest subnormal -- and every
// reciprocal underflowed to zero. LAMMPS' double-precision Lennard-Jones
// forces and energies all came out as exactly 0 this way, under every FP64
// mode, with every launch reporting success.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void reciprocal(double* out, const double* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0 / in[i];
}

// The shape of LAMMPS' PairComputeFunctor: cutoff gate, r^-2 via reciprocal,
// force and energy accumulated over a neighbour list.
__global__ void lj_forces(const double* x, int n, double cutsq, double* f, double* e) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double fi = 0, ei = 0;
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        double d = x[i] - x[j], rsq = d * d;
        if (rsq < cutsq) {
            double r2inv = 1.0 / rsq, r6inv = r2inv * r2inv * r2inv;
            fi += d * r6inv * (48.0 * r6inv - 24.0) * r2inv;
            ei += 4.0 * r6inv * (r6inv - 1.0);
        }
    }
    f[i] = fi;
    e[i] = ei;
}

static uint64_t bits(double v) { uint64_t b; memcpy(&b, &v, 8); return b; }

int main() {
    const char* mode = getenv("CUMETAL_FP64_MODE");
    const bool exact = mode != nullptr && strcmp(mode, "ieee64") == 0;
    const double tol = exact ? 0.0 : 1.0 / 281474976710656.0;  // 2^-48 for the pair modes
    const double inputs[] = {1.0, 2.0, 3.0, 0.1, -7.25, 1e300, 1e-300, 4.5,
                             1.0 / 3.0, 6.02214076e23, INFINITY, -INFINITY, 0.0, -0.0};
    const int n = sizeof inputs / sizeof inputs[0];
    double *d_in, *d_out;
    cudaMalloc(&d_in, sizeof inputs);
    cudaMalloc(&d_out, sizeof inputs);
    cudaMemcpy(d_in, inputs, sizeof inputs, cudaMemcpyHostToDevice);
    reciprocal<<<1, 32>>>(d_out, d_in, n);
    double got[n];
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(got, d_out, sizeof got, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("FAIL: reciprocal launch failed\n");
        return 1;
    }
    int failures = 0;
    for (int i = 0; i < n; ++i) {
        const double want = 1.0 / inputs[i];
        // fast48 keeps binary32's exponent range: 1/1e300 and 1/1e-300 are
        // outside it, and only wide48/ieee64 promise them.
        const bool in_range = std::fabs(want) == 0 || std::isinf(want) ||
                              (std::fabs(want) > 1e-37 && std::fabs(want) < 1e37);
        if (!exact && !in_range && strcmp(mode ? mode : "fast48", "wide48") != 0) continue;
        bool ok;
        if (std::isinf(want) || want == 0) ok = bits(got[i]) == bits(want);
        else if (exact) ok = bits(got[i]) == bits(want);
        else ok = std::fabs(got[i] - want) <= tol * std::fabs(want);
        if (!ok) {
            printf("FAIL: 1/%.17g = %.17g (0x%016llx), expected %.17g (0x%016llx)\n", inputs[i],
                   got[i], (unsigned long long)bits(got[i]), want, (unsigned long long)bits(want));
            ++failures;
        }
    }

    const int m = 48;
    double hx[m];
    for (int i = 0; i < m; ++i) hx[i] = 1.12 * i + 0.01 * (i % 5);
    double *d_x, *d_f, *d_e;
    cudaMalloc(&d_x, sizeof hx);
    cudaMalloc(&d_f, sizeof hx);
    cudaMalloc(&d_e, sizeof hx);
    cudaMemcpy(d_x, hx, sizeof hx, cudaMemcpyHostToDevice);
    lj_forces<<<2, 32>>>(d_x, m, 6.25, d_f, d_e);
    double gf[m], ge[m];
    cudaDeviceSynchronize();
    cudaMemcpy(gf, d_f, sizeof gf, cudaMemcpyDeviceToHost);
    cudaMemcpy(ge, d_e, sizeof ge, cudaMemcpyDeviceToHost);
    double energy = 0, ref_energy = 0, worst = 0;
    for (int i = 0; i < m; ++i) {
        double fi = 0, ei = 0;
        for (int j = 0; j < m; ++j) {
            if (j == i) continue;
            double d = hx[i] - hx[j], rsq = d * d;
            if (rsq < 6.25) {
                double r2inv = 1.0 / rsq, r6inv = r2inv * r2inv * r2inv;
                fi += d * r6inv * (48.0 * r6inv - 24.0) * r2inv;
                ei += 4.0 * r6inv * (r6inv - 1.0);
            }
        }
        energy += ge[i];
        ref_energy += ei;
        worst = std::fmax(worst, std::fabs(gf[i] - fi) / std::fmax(1.0, std::fabs(fi)));
    }
    // Pair-mode rounding compounds over the sum; 1e-10 is far above it and
    // far below the all-zero failure.
    if (std::fabs(energy - ref_energy) > 1e-10 * std::fabs(ref_energy) || worst > 1e-10 ||
        ref_energy == 0) {
        printf("FAIL: LJ energy %.17g vs host %.17g, worst scaled force error %g\n", energy,
               ref_energy, worst);
        ++failures;
    }
    if (failures == 0) printf("PASS: fp64 reciprocal matches host on Apple GPU\n");
    return failures != 0;
}
