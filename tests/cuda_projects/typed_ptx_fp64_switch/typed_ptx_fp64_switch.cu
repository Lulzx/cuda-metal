// Reduced from generativeqc's SCF/DFT kernels, which the typed PTX backend
// refused (#218):
//   * CUDART_INF is __longlong_as_double(0x7ff0000000000000ULL). Under
//     -nocudalib clang leaves __nv_longlong_as_double / __nv_double_as_longlong
//     as libdevice calls with no typed PTX definition.
//   * A dense switch becomes a `brx.idx` jump table over `.branchtargets`
//     ("ptx.branchtargets is not normalized"). The one-electron integrals
//     dispatch on angular momentum this way.
// Run with CUMETAL_PTX_BACKEND=cumetal-ir; every result is compared bit for bit.
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

__global__ void initialize_state(int n, bool reuse, const double* energy, double* previous,
                                 double* change, long long* bits, double* rebuilt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    previous[i] = reuse ? energy[i] : CUDART_INF;
    change[i] = CUDART_INF;
    bits[i] = __double_as_longlong(energy[i]);
    rebuilt[i] = __longlong_as_double(bits[i] ^ (1ll << 63));
}

// noinline keeps the switch in a helper, as in the generated integrals.
__device__ __noinline__ unsigned dispatch(unsigned selector, unsigned value) {
    switch (selector) {
    case 0: return value * 3u + 1u;
    case 1: return value ^ 0x5555u;
    case 2: return value << 3;
    case 3: return value - 17u;
    case 4: return value * value;
    case 5: return ~value;
    case 6: return value / 7u;
    case 7: return value + (value >> 5);
    case 8: return value | 0x80000000u;
    default: return 0xdeadbeefu;
    }
}

__global__ void switch_kernel(const unsigned* in, unsigned* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dispatch(static_cast<unsigned>(i) % 11u, in[i]);
}

static unsigned host_dispatch(unsigned selector, unsigned value) {
    switch (selector) {
    case 0: return value * 3u + 1u;
    case 1: return value ^ 0x5555u;
    case 2: return value << 3;
    case 3: return value - 17u;
    case 4: return value * value;
    case 5: return ~value;
    case 6: return value / 7u;
    case 7: return value + (value >> 5);
    case 8: return value | 0x80000000u;
    default: return 0xdeadbeefu;
    }
}

static int check_state() {
    const int n = 256;
    double energy[n];
    for (int i = 0; i < n; ++i) energy[i] = 1.5 * i - 7.25;
    double *d_energy, *d_previous, *d_change, *d_rebuilt;
    long long* d_bits;
    if (cudaMalloc(&d_energy, n * 8) || cudaMalloc(&d_previous, n * 8) ||
        cudaMalloc(&d_change, n * 8) || cudaMalloc(&d_rebuilt, n * 8) ||
        cudaMalloc(&d_bits, n * 8)) return 1;
    cudaMemcpy(d_energy, energy, n * 8, cudaMemcpyHostToDevice);
    for (const bool reuse : {false, true}) {
        initialize_state<<<(n + 63) / 64, 64>>>(n, reuse, d_energy, d_previous, d_change, d_bits,
                                               d_rebuilt);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: initialize_state launch\n");
            return 1;
        }
        double previous[n], change[n], rebuilt[n];
        long long bits[n];
        cudaMemcpy(previous, d_previous, n * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(change, d_change, n * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(rebuilt, d_rebuilt, n * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(bits, d_bits, n * 8, cudaMemcpyDeviceToHost);
        const uint64_t inf_bits = 0x7ff0000000000000ull;
        for (int i = 0; i < n; ++i) {
            uint64_t prev_bits, change_bits, energy_bits, rebuilt_bits;
            std::memcpy(&prev_bits, &previous[i], 8);
            std::memcpy(&change_bits, &change[i], 8);
            std::memcpy(&energy_bits, &energy[i], 8);
            std::memcpy(&rebuilt_bits, &rebuilt[i], 8);
            const uint64_t want_prev = reuse ? energy_bits : inf_bits;
            if (prev_bits != want_prev || change_bits != inf_bits ||
                static_cast<uint64_t>(bits[i]) != energy_bits ||
                rebuilt_bits != (energy_bits ^ (1ull << 63))) {
                std::fprintf(stderr, "FAIL: state[%d] reuse=%d prev=%016llx change=%016llx bits=%016llx"
                             " rebuilt=%016llx\n", i, reuse ? 1 : 0, (unsigned long long)prev_bits,
                             (unsigned long long)change_bits, (unsigned long long)bits[i],
                             (unsigned long long)rebuilt_bits);
                return 1;
            }
        }
    }
    cudaFree(d_energy); cudaFree(d_previous); cudaFree(d_change); cudaFree(d_rebuilt); cudaFree(d_bits);
    return 0;
}

static int check_switch() {
    const int n = 4096;
    static unsigned in[n], out[n];
    uint64_t x = 0x218u;
    for (int i = 0; i < n; ++i) {
        x = 6364136223846793005ull * x + 1442695040888963407ull;
        in[i] = static_cast<unsigned>(x >> 32);
    }
    unsigned *d_in, *d_out;
    if (cudaMalloc(&d_in, sizeof(in)) || cudaMalloc(&d_out, sizeof(out))) return 1;
    cudaMemcpy(d_in, in, sizeof(in), cudaMemcpyHostToDevice);
    switch_kernel<<<n / 128, 128>>>(d_in, d_out, n);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: switch launch\n");
        return 1;
    }
    cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        const unsigned want = host_dispatch(static_cast<unsigned>(i) % 11u, in[i]);
        if (out[i] != want) {
            std::fprintf(stderr, "FAIL: switch[%d] selector=%u got %08x want %08x\n", i,
                         static_cast<unsigned>(i) % 11u, out[i], want);
            return 1;
        }
    }
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}

int main() {
    if (check_state() != 0 || check_switch() != 0) return 1;
    std::puts("PASS: typed PTX FP64 bit casts and switch jump tables");
    return 0;
}
