// A kernel whose own arguments reach Metal buffer indices 25-30, the indices
// reserved for hidden bindings (grid Y offset, grid barrier, device clock,
// atomic lock bank, constant store, trap status). Pipeline reflection used to
// flag those hidden features by index alone, and the hidden buffers were bound
// over the caller's: writes through the upper arguments landed in the hidden
// buffers and read back zero with cudaSuccess everywhere.
//
// Enrolled for the legacy and native-AOT paths only. The typed PTX backend
// cannot yet prove the pointer loaded from `slots[i]` (a local pointer table
// walked by an unrolled cursor loop) and refuses the kernel; see the
// registration-default blocker table in docs/known-gaps/compiler.md.
#include <cuda_runtime.h>

#include <cstdio>

#define N_WIDE_ARGS 31
#define NT 32

__global__ void wide_args(unsigned *a0, unsigned *a1, unsigned *a2,
                          unsigned *a3, unsigned *a4, unsigned *a5,
                          unsigned *a6, unsigned *a7, unsigned *a8,
                          unsigned *a9, unsigned *a10, unsigned *a11,
                          unsigned *a12, unsigned *a13, unsigned *a14,
                          unsigned *a15, unsigned *a16, unsigned *a17,
                          unsigned *a18, unsigned *a19, unsigned *a20,
                          unsigned *a21, unsigned *a22, unsigned *a23,
                          unsigned *a24, unsigned *a25, unsigned *a26,
                          unsigned *a27, unsigned *a28, unsigned *a29,
                          unsigned *a30) {
    unsigned *slots[N_WIDE_ARGS] = {a0,  a1,  a2,  a3,  a4,  a5,  a6,  a7,
                                    a8,  a9,  a10, a11, a12, a13, a14, a15,
                                    a16, a17, a18, a19, a20, a21, a22, a23,
                                    a24, a25, a26, a27, a28, a29, a30};
    const unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < N_WIDE_ARGS; ++i) {
        slots[i][pos] = 0xC000u + (unsigned)i;
    }
}

int main() {
    int failures = 0;
    unsigned *d_wide[N_WIDE_ARGS];
    for (int i = 0; i < N_WIDE_ARGS; ++i) {
        cudaMalloc(&d_wide[i], NT * sizeof(unsigned));
        cudaMemset(d_wide[i], 0, NT * sizeof(unsigned));
    }
    wide_args<<<1, NT>>>(d_wide[0],  d_wide[1],  d_wide[2],  d_wide[3],
                         d_wide[4],  d_wide[5],  d_wide[6],  d_wide[7],
                         d_wide[8],  d_wide[9],  d_wide[10], d_wide[11],
                         d_wide[12], d_wide[13], d_wide[14], d_wide[15],
                         d_wide[16], d_wide[17], d_wide[18], d_wide[19],
                         d_wide[20], d_wide[21], d_wide[22], d_wide[23],
                         d_wide[24], d_wide[25], d_wide[26], d_wide[27],
                         d_wide[28], d_wide[29], d_wide[30]);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("FAIL: wide_args launch: %s\n", cudaGetErrorString(e));
        return 1;
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < N_WIDE_ARGS; ++i) {
        unsigned h_wide[NT];
        cudaMemcpy(h_wide, d_wide[i], sizeof(h_wide), cudaMemcpyDeviceToHost);
        for (int t = 0; t < NT; ++t) {
            if (h_wide[t] != 0xC000u + (unsigned)i) {
                std::printf("FAIL: wide_args arg %d (buffer index %d) thread %d: "
                            "expected %u, got %u\n",
                            i, i, t, 0xC000u + (unsigned)i, h_wide[t]);
                ++failures;
                break;
            }
        }
        cudaFree(d_wide[i]);
    }
    if (failures != 0) {
        std::printf("FAIL: %d argument buffer(s) were clobbered\n", failures);
        return 1;
    }
    std::printf("PASS: all 31 argument buffers written, none clobbered\n");
    return 0;
}
