// FP64 support arithmetic must not depend on the user's float math flags.
//
// VF64's pair and wide48 operations are error-free transforms: fast math may
// reassociate the cancellations they rely on, and FMA contraction fuses a
// product whose rounding error they recover. The user's --use_fast_math and
// -fmad choices govern user float code only, so every FP64 result here must be
// bit-identical across those flags. The driver script builds this probe under
// each flag set and compares the printed digests.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

constexpr int kOps = 6;

__global__ void fp64_ops(const double* a, const double* b, const double* c, double* out,
                         int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i * kOps + 0] = a[i] + b[i];
    out[i * kOps + 1] = a[i] * b[i];
    out[i * kOps + 2] = a[i] / b[i];
    out[i * kOps + 3] = sqrt(fabs(a[i]));
    out[i * kOps + 4] = fma(a[i], b[i], c[i]);
    out[i * kOps + 5] = hypot(a[i], b[i]);
}

int main() {
    // Operands near one another and spanning a few decades exercise the
    // cancellation and carry paths of the pair and wide48 transforms, while
    // staying inside the range every FP64 mode represents.
    const int n = 1 << 16;
    std::vector<double> a(n), b(n), c(n);
    uint64_t s = 0x243f6a8885a308d3ull;
    auto next = [&] {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    };
    auto unit = [&] { return double(next() >> 11) * 0x1p-53; };
    for (int i = 0; i < n; ++i) {
        const double scale = std::ldexp(1.0, int(next() % 40) - 20);
        a[i] = (unit() * 2 - 1) * scale;
        b[i] = (i & 3) == 0 ? -a[i] * (1 + unit() * 0x1p-30) : (unit() * 2 - 1) * scale;
        c[i] = (unit() * 2 - 1) * scale * scale;
    }
    const size_t in_bytes = n * sizeof(double), out_bytes = in_bytes * kOps;
    double *da, *db, *dc, *dout;
    if (cudaMalloc(&da, in_bytes) != cudaSuccess || cudaMalloc(&db, in_bytes) != cudaSuccess ||
        cudaMalloc(&dc, in_bytes) != cudaSuccess || cudaMalloc(&dout, out_bytes) != cudaSuccess ||
        cudaMemcpy(da, a.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(db, b.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dc, c.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("FAIL: allocation or upload failed\n");
        return 1;
    }
    fp64_ops<<<(n + 255) / 256, 256>>>(da, db, dc, dout, n);
    std::vector<double> out(size_t(n) * kOps);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(out.data(), dout, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("FAIL: launch or download failed\n");
        return 1;
    }
    // One FNV-1a digest per operation, so a mismatch names the operation.
    printf("DIGEST");
    for (int op = 0; op < kOps; ++op) {
        uint64_t h = 0xcbf29ce484222325ull;
        for (int i = 0; i < n; ++i) {
            uint64_t bits;
            std::memcpy(&bits, &out[size_t(i) * kOps + op], sizeof bits);
            for (int k = 0; k < 8; ++k) {
                h ^= (bits >> (8 * k)) & 0xff;
                h *= 0x100000001b3ull;
            }
        }
        printf(" %016llx", (unsigned long long)h);
    }
    printf("\n");
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dout);
    return 0;
}
