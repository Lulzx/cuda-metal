// Bitwise conformance of ieee64 binary64 division over the whole input domain.
//
// VF64 divides normal finite operands through a certified table reciprocal and
// falls back to the restoring divider for everything else; the rounding packer
// then handles overflow and subnormal results. fp64_division_benchmark covers
// only dividends and divisors in [1, 2). This probe covers the rest: every
// exponent, quotients that overflow or land in (or below) the subnormal range,
// every reciprocal-table segment edge, near-halfway quotients, and the special
// value cross product. Every quotient must equal the host's binary64 result bit
// for bit; NaN results need only be NaN, since payloads are not CUDA-visible.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

namespace {

constexpr uint64_t kPoison = 0x7ff8dead0000beefull;
constexpr int kGuard = 2;

uint64_t bits(double v) {
    uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

double from_bits(uint64_t b) {
    double v;
    std::memcpy(&v, &b, sizeof v);
    return v;
}

struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

uint64_t make(uint64_t sign, uint64_t exponent, uint64_t mantissa) {
    return (sign << 63) | ((exponent & 0x7ff) << 52) | (mantissa & ((1ull << 52) - 1));
}

}  // namespace

__global__ void divide(const double* x, const double* y, double* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] / y[i];
}

__global__ void reciprocal(const double* y, double* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __drcp_rn(y[i]);
}

int main() {
    std::vector<double> xs, ys;
    // Category boundaries, for failure reports.
    std::vector<std::pair<size_t, const char*>> categories;
    auto add = [&](uint64_t a, uint64_t b) {
        xs.push_back(from_bits(a));
        ys.push_back(from_bits(b));
    };
    auto category = [&](const char* name) { categories.emplace_back(xs.size(), name); };
    Rng rng;

    category("random normal operands over every exponent");
    for (int i = 0; i < 1 << 19; ++i) {
        const uint64_t a = rng.next(), b = rng.next();
        add(make(a >> 63, 1 + (a >> 52) % 2046, a), make(b >> 63, 1 + (b >> 52) % 2046, b));
    }

    // The quotient's unbiased exponent is ea - eb (+/- 1); steer it through the
    // overflow edge and the normal/subnormal/zero transitions.
    category("overflowing and subnormal quotients");
    for (int i = 0; i < 1 << 17; ++i) {
        const uint64_t a = rng.next(), b = rng.next(), c = rng.next();
        const int target = (c & 1) ? 1020 + int(c >> 58) % 10        // near 2^1024
                                   : -1080 + int((c >> 8) % 70);      // -1080..-1011
        const int ea = 1 + int((a >> 52) % 2046);
        const int eb = ea - target;
        if (eb < 1 || eb > 2046) continue;
        add(make(a >> 63, ea, a), make(b >> 63, eb, b));
    }

    // VF64 selects reciprocal coefficients from divisor mantissa bits 48..51.
    // Exercise both ends of each of the 16 segments with random dividends.
    category("reciprocal table segment edges");
    for (uint64_t segment = 0; segment < 16; ++segment) {
        const uint64_t lo = segment << 48;
        const uint64_t hi = lo | ((1ull << 48) - 1);
        for (uint64_t m : {lo, lo + 1, lo + 2, hi - 2, hi - 1, hi}) {
            for (int i = 0; i < 64; ++i) {
                const uint64_t a = rng.next();
                add(make(a >> 63, 1 + (a >> 52) % 2046, a), make(i & 1, 1023, m));
            }
        }
    }

    // Quotients one or two ulps either side of 1 and of significands with long
    // runs of ones or zeros sit next to a rounding threshold; the leaf's exact
    // residual correction decides those.
    category("near-halfway quotients");
    const uint64_t patterns[] = {0, 1, 2, 3, (1ull << 52) - 1, (1ull << 52) - 2,
                                 (1ull << 52) - 3, 1ull << 51, (1ull << 51) - 1,
                                 (1ull << 51) + 1, 0x5555555555555ull, 0xaaaaaaaaaaaaaull};
    for (uint64_t ma : patterns) {
        for (uint64_t mb : patterns) {
            for (int ea : {1, 2, 1022, 1023, 1024, 2045, 2046}) {
                for (int eb : {1, 2, 1022, 1023, 1024, 2045, 2046}) {
                    add(make(0, ea, ma), make(0, eb, mb));
                    add(make(1, ea, ma), make(0, eb, mb));
                }
            }
        }
    }
    for (int i = 0; i < 1 << 16; ++i) {
        // x = y * (1 + small k ulp), so x / y lands a few ulps from a
        // representable value and the exact residual decides.
        const uint64_t b = rng.next();
        const uint64_t mb = b & ((1ull << 52) - 1);
        const int64_t k = int64_t(rng.next() % 9) - 4;
        add(make(0, 1023, mb + uint64_t(k)), make(0, 1023, mb));
    }

    category("special and subnormal operand cross product");
    const uint64_t specials[] = {
        0x0000000000000000ull, 0x8000000000000000ull,  // +-0
        0x0000000000000001ull, 0x8000000000000001ull,  // +-min subnormal
        0x000fffffffffffffull, 0x0008000000000000ull,  // largest / mid subnormal
        0x0010000000000000ull, 0x8010000000000000ull,  // +-min normal
        0x7fefffffffffffffull, 0xffefffffffffffffull,  // +-max finite
        0x3ff0000000000000ull, 0xbff0000000000000ull,  // +-1
        0x4000000000000000ull, 0x3fe0000000000000ull,  // 2, 0.5
        0x3ff8000000000000ull, 0x400921fb54442d18ull,  // 1.5, pi
        0x7ff0000000000000ull, 0xfff0000000000000ull,  // +-inf
        0x7ff8000000000000ull, 0x7ff4000000000000ull,  // qNaN, sNaN
        0x0000000012345678ull, 0x8007654321fedcbaull,  // other subnormals
    };
    for (uint64_t a : specials) {
        for (uint64_t b : specials) add(a, b);
        for (int i = 0; i < 32; ++i) {
            const uint64_t r = rng.next();
            const uint64_t normal = make(r >> 63, 1 + (r >> 52) % 2046, r);
            add(a, normal);
            add(normal, a);
        }
    }

    const int n = static_cast<int>(xs.size());
    const size_t in_bytes = xs.size() * sizeof(double);
    const size_t out_count = xs.size() + kGuard;
    double *dx = nullptr, *dy = nullptr, *dout = nullptr;
    if (cudaMalloc(&dx, in_bytes) != cudaSuccess || cudaMalloc(&dy, in_bytes) != cudaSuccess ||
        cudaMalloc(&dout, out_count * sizeof(double)) != cudaSuccess ||
        cudaMemcpy(dx, xs.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dy, ys.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("FAIL: division domain allocation or upload failed\n");
        return 1;
    }

    std::vector<double> got(out_count);
    int failures = 0;
    auto category_of = [&](size_t i) {
        const char* name = "?";
        for (const auto& [start, label] : categories) {
            if (i >= start) name = label;
        }
        return name;
    };
    auto run = [&](const char* op, bool is_reciprocal) {
        std::vector<double> poison(out_count, from_bits(kPoison));
        if (cudaMemcpy(dout, poison.data(), out_count * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("FAIL: %s poison upload failed\n", op);
            ++failures;
            return;
        }
        const int block = 256;
        if (is_reciprocal) {
            reciprocal<<<(n + block - 1) / block, block>>>(dy, dout, n);
        } else {
            divide<<<(n + block - 1) / block, block>>>(dx, dy, dout, n);
        }
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(got.data(), dout, out_count * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("FAIL: %s launch or download failed\n", op);
            ++failures;
            return;
        }
        int local = 0;
        for (int i = 0; i < n; ++i) {
            const double x = is_reciprocal ? 1.0 : xs[i];
            const double want = x / ys[i];
            const bool ok = std::isnan(want) ? std::isnan(got[i]) : bits(got[i]) == bits(want);
            if (!ok && local++ < 8) {
                printf("FAIL: %s [%s] %.17g (0x%016llx) / %.17g (0x%016llx) = 0x%016llx, "
                       "want 0x%016llx\n",
                       op, category_of(i), x, (unsigned long long)bits(x), ys[i],
                       (unsigned long long)bits(ys[i]), (unsigned long long)bits(got[i]),
                       (unsigned long long)bits(want));
            }
        }
        for (int g = 0; g < kGuard; ++g) {
            if (bits(got[n + g]) != kPoison) {
                printf("FAIL: %s wrote past the end of its output\n", op);
                ++local;
            }
        }
        if (local) printf("FAIL: %s %d of %d results differ from host binary64\n", op, local, n);
        failures += local;
    };
    run("x / y", false);
    run("__drcp_rn(y)", true);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dout);
    if (failures) return 1;
    printf("PASS: ieee64 division matches host binary64 bitwise over the domain (%d pairs x 2 ops)\n", n);
    return 0;
}
