#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

constexpr int kRelations = 15;

__global__ void relations(int* out, const double* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = in[2*i], b = in[2*i+1];
    int* r = out + kRelations*i;
    r[0] = a == b; r[1] = a != b;
    r[2] = a < b; r[3] = a <= b;
    r[4] = a > b; r[5] = a >= b;
    bool unordered = isnan(a) || isnan(b);
    r[6] = a == b || unordered; r[7] = !(a >= b);
    r[8] = !(a > b); r[9] = !(a <= b); r[10] = !(a < b);
    r[11] = unordered; r[12] = !unordered;
    r[13] = !unordered && a != b;
    r[14] = (a - b) == 0.0;
}

// A fixed bound turns endpoint nonprogress into a test failure, not a GPU hang.
__global__ void endpoint(int* out, const double* in) {
    if (threadIdx.x != 0) return;
    double t = in[0], target = in[1];
    int steps = 0, stalled = 0;
    while (t < target && steps < 8) {
        double next = t + (target - t);
        stalled |= next == t;
        t = next;
        ++steps;
    }
    out[0] = steps; out[1] = stalled; out[2] = t < target;
    out[3] = 0x12345678;
}

static double from_bits(uint64_t bits) {
    double value; memcpy(&value, &bits, sizeof value); return value;
}

static bool checked(cudaError_t error) {
    if (error == cudaSuccess) return true;
    printf("FAIL: CUDA error: %s\n", cudaGetErrorString(error));
    return false;
}

int main() {
    const char* mode = getenv("CUMETAL_FP64_MODE");
    bool exact = mode && strcmp(mode, "ieee64") == 0;
    bool fast = mode && strcmp(mode, "fast48") == 0;
    const double t = from_bits(0x3f947ae147ae1478ULL);
    const double target = from_bits(0x3f947ae147ae147bULL);
    const double nan = from_bits(0x7ff8000000000001ULL);
    const double snan = from_bits(0x7ff0000000000001ULL);
    double pairs[][2] = {
        {t, target}, {target, t}, {-t, -target}, {0.0, -0.0}, {-0.0, 0.0},
        {1.0, 2.0}, {-2.0, -1.0}, {INFINITY, INFINITY}, {-INFINITY, INFINITY},
        {nan, 1.0}, {1.0, nan}, {nan, nan}, {snan, 0.0},
        {0x1.fffffffffffffp-1, 1.0}, {1.0, 0x1.fffffffffffffp-1},
        {-0x1.fffffffffffffp-1, -1.0}, {-1.0, -0x1.fffffffffffffp-1},
        {0x1p900, 0x1p901}, {-0x1p901, -0x1p900},
        {0x1p-900, 0x1p-899}, {0x1p-1074, 0x1p-1073},
        {from_bits(0x7fefffffffffffffULL), INFINITY},
        {from_bits(0x000fffffffffffffULL), 0x1p-1022},
        {0x1.fffffffffffffp899, 0x1p900},
        {0x1.fffffffffffffp-901, 0x1p-900},
        {0x1.ffffffp0, 2.0}, {0x1.ffffffp0, 0x1.fffffep0},
        {-0x1.ffffffp0, -2.0}, {-0x1.ffffffp0, -0x1.fffffep0}
    };
    int n = sizeof pairs / sizeof pairs[0];
    // The following cases exercise wide48's distinct codec and exponent range.
    if (fast) n = 13;
    double* d_in;
    int* d_out;
    if (!checked(cudaMalloc(&d_in, sizeof pairs)) ||
        !checked(cudaMalloc(&d_out, (n*kRelations+4)*sizeof(int))) ||
        !checked(cudaMemcpy(d_in, pairs, sizeof pairs, cudaMemcpyHostToDevice)) ||
        !checked(cudaMemset(d_out, 0xa5, (n*kRelations+4)*sizeof(int)))) return 1;
    relations<<<1, 32>>>(d_out, d_in, n);
    int got[sizeof pairs / sizeof pairs[0] * kRelations];
    if (!checked(cudaGetLastError()) || !checked(cudaDeviceSynchronize()) ||
        !checked(cudaMemcpy(got, d_out, n*kRelations*sizeof(int), cudaMemcpyDeviceToHost))) return 1;
    int failures = 0;
    for (int i = 0; i < n; ++i) {
        double a = pairs[i][0], b = pairs[i][1];
        // These inputs differ only below the retained pair precision.
        if (!exact && (i < 3 || (i >= 13 && i <= 16) || (i >= 22 && i <= 24))) b = a;
        bool u = std::isnan(a) || std::isnan(b);
        int want[] = {a == b, a != b, a < b, a <= b, a > b, a >= b,
                      a == b || u, !(a >= b), !(a > b), !(a <= b), !(a < b),
                      u, !u, !u && a != b, (a - b) == 0.0};
        for (int j = 0; j < kRelations; ++j) {
            if (got[kRelations*i+j] != want[j]) {
                printf("FAIL: pair %d relation %d got %d expected %d\n", i, j,
                       got[kRelations*i+j], want[j]);
                ++failures;
            }
        }
    }
    for (int arrival = 0; arrival < 2; ++arrival) {
        double input[] = {arrival ? 0.0 : t, target};
        int result[4] = {-1, -1, -1, -1};
        if (!checked(cudaMemcpy(d_in, input, sizeof input, cudaMemcpyHostToDevice))) return 1;
        endpoint<<<1, 1>>>(d_out, d_in);
        if (!checked(cudaGetLastError()) || !checked(cudaDeviceSynchronize()) ||
            !checked(cudaMemcpy(result, d_out, sizeof result, cudaMemcpyDeviceToHost))) return 1;
        int steps = arrival || exact ? 1 : 0;
        if (result[0] != steps || result[1] || result[2] || result[3] != 0x12345678) {
            printf("FAIL: endpoint %d: steps=%d stalled=%d remaining=%d marker=%x\n",
                   arrival, result[0], result[1], result[2], result[3]);
            ++failures;
        }
    }
    if (!checked(cudaFree(d_in)) || !checked(cudaFree(d_out))) return 1;
    if (!failures) printf("PASS: fp64 comparisons and endpoint progress on Apple GPU\n");
    return failures != 0;
}
