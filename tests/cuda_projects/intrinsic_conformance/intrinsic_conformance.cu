// CUDA intrinsic semantics that an Apple GPU does not give for free:
//   * __shfl_{xor,down,up}_sync with width < 32 follow the PTX shfl.sync
//     segment rules: a butterfly partner in an EARLIER segment is readable,
//     one past the end of the caller's segment keeps the caller's value.
//   * __syncthreads_count/_and/_or reduce a predicate over the whole block.
//   * __fmul_rn/__fadd_rn are never contracted into an FMA, while a plain
//     a * b + c is (nvcc's default --fmad=true), even in the same kernel.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void shuffles(const unsigned* in, unsigned* xor_out, unsigned* down_out,
                         unsigned* up_out) {
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned v = in[gid];
    const int widths[5] = {32, 16, 8, 4, 2};
    for (int w = 0; w < 5; ++w) {
        for (int m = 0; m < 5; ++m) {
            const unsigned slot = gid * 25 + w * 5 + m;
            xor_out[slot] = __shfl_xor_sync(0xffffffffu, v, 1 << m, widths[w]);
            down_out[slot] = __shfl_down_sync(0xffffffffu, v, 1u << m, widths[w]);
            up_out[slot] = __shfl_up_sync(0xffffffffu, v, 1u << m, widths[w]);
        }
    }
}

__global__ void votes(const unsigned* in, unsigned* out) {
    const unsigned v = in[blockIdx.x * blockDim.x + threadIdx.x];
    const int count = __syncthreads_count((v & 7u) == 0u);
    const int all = __syncthreads_and((v & 0xffu) != 0u);
    const int any = __syncthreads_or((v & 0xfffu) == 0u);
    // Every thread must see the block result, not just thread 0.
    out[(blockIdx.x * blockDim.x + threadIdx.x) * 3 + 0] = (unsigned)count;
    out[(blockIdx.x * blockDim.x + threadIdx.x) * 3 + 1] = all != 0;
    out[(blockIdx.x * blockDim.x + threadIdx.x) * 3 + 2] = any != 0;
}

__global__ void contraction(const float* a, const float* b, const float* c, float* expr,
                            float* split, float* sub_split, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        expr[i] = a[i] * b[i] + c[i];
        split[i] = __fadd_rn(__fmul_rn(a[i], b[i]), c[i]);
        sub_split[i] = __fsub_rn(__fmul_rn(a[i], b[i]), -c[i]);
    }
}

static unsigned lcg(uint64_t* x) {
    *x = 6364136223846793005ull * *x + 1442695040888963407ull;
    return (unsigned)(*x >> 32);
}

static float bits_to_float(unsigned u) {
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

static unsigned float_to_bits(float f) {
    unsigned u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// PTX ISA shfl.sync, with c = ((32 - width) << 8) | cval.
static unsigned reference_shuffle(const std::vector<unsigned>& in, unsigned warp_base,
                                  unsigned lane, unsigned b, unsigned width, char mode) {
    const unsigned segmask = 32u - width;
    const unsigned cval = mode == 'u' ? 0u : 31u;
    const unsigned min_lane = lane & segmask;
    const unsigned max_lane = min_lane | (cval & ~segmask & 31u);
    int j = (int)lane;
    bool valid = false;
    if (mode == 'x') { j = (int)(lane ^ b); valid = j <= (int)max_lane; }
    if (mode == 'd') { j = (int)(lane + b); valid = j <= (int)max_lane; }
    if (mode == 'u') { j = (int)lane - (int)b; valid = j >= (int)max_lane; }
    return in[warp_base + (valid ? (unsigned)j : lane)];
}

static int check_shuffles() {
    const unsigned blocks = 4, threads = 64, n = blocks * threads;
    std::vector<unsigned> in(n);
    uint64_t seed = 0x5eedu;
    for (unsigned& v : in) v = lcg(&seed);
    unsigned *d_in = nullptr, *d_out = nullptr;
    if (cudaMalloc(&d_in, n * 4) != cudaSuccess ||
        cudaMalloc(&d_out, 3 * n * 25 * 4) != cudaSuccess) return 1;
    cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice);
    shuffles<<<blocks, threads>>>(d_in, d_out, d_out + n * 25, d_out + 2 * n * 25);
    std::vector<unsigned> out(3 * n * 25);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: shuffle launch\n");
        return 1;
    }
    const unsigned widths[5] = {32, 16, 8, 4, 2};
    const char modes[3] = {'x', 'd', 'u'};
    for (unsigned k = 0; k < 3; ++k) {
        for (unsigned gid = 0; gid < n; ++gid) {
            for (unsigned w = 0; w < 5; ++w) {
                for (unsigned m = 0; m < 5; ++m) {
                    const unsigned expected = reference_shuffle(in, gid & ~31u, gid & 31u,
                                                                1u << m, widths[w], modes[k]);
                    const unsigned actual = out[k * n * 25 + gid * 25 + w * 5 + m];
                    if (actual != expected) {
                        std::fprintf(stderr,
                                     "FAIL: shfl mode=%c lane=%u width=%u b=%u got %08x want %08x\n",
                                     modes[k], gid & 31u, widths[w], 1u << m, actual, expected);
                        return 1;
                    }
                }
            }
        }
    }
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

static int check_votes() {
    const unsigned blocks = 64;
    for (unsigned threads : {32u, 96u, 128u, 256u}) {
        const unsigned n = blocks * threads;
        std::vector<unsigned> in(n);
        uint64_t seed = 0x7075u + threads;
        for (unsigned& v : in) v = lcg(&seed);
        // Force some blocks to all-true / none-true so and/or see both outcomes.
        for (unsigned t = 0; t < threads; ++t) in[t] |= 1u;            // block 0: and = 1
        for (unsigned t = 0; t < threads; ++t) in[threads + t] = 0u;   // block 1: or = 1, count = all
        unsigned *d_in = nullptr, *d_out = nullptr;
        if (cudaMalloc(&d_in, n * 4) != cudaSuccess || cudaMalloc(&d_out, 3 * n * 4) != cudaSuccess)
            return 1;
        cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice);
        votes<<<blocks, threads>>>(d_in, d_out);
        std::vector<unsigned> out(3 * n);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: vote launch\n");
            return 1;
        }
        for (unsigned block = 0; block < blocks; ++block) {
            unsigned count = 0, all = 1, any = 0;
            for (unsigned t = 0; t < threads; ++t) {
                const unsigned v = in[block * threads + t];
                count += (v & 7u) == 0u;
                all &= (v & 0xffu) != 0u;
                any |= (v & 0xfffu) == 0u;
            }
            for (unsigned t = 0; t < threads; ++t) {
                const unsigned* got = &out[(block * threads + t) * 3];
                if (got[0] != count || got[1] != all || got[2] != any) {
                    std::fprintf(stderr,
                                 "FAIL: votes threads=%u block=%u thread=%u got (%u,%u,%u) want (%u,%u,%u)\n",
                                 threads, block, t, got[0], got[1], got[2], count, all, any);
                    return 1;
                }
            }
        }
        cudaFree(d_in);
        cudaFree(d_out);
    }
    return 0;
}

static int check_contraction() {
    const unsigned n = 4096;
    std::vector<float> a(n), b(n), c(n);
    uint64_t seed = 0xf3au;
    for (unsigned i = 0; i < n; ++i) {
        a[i] = bits_to_float(0x3f800000u | (lcg(&seed) >> 9));
        b[i] = bits_to_float(0x3f800000u | (lcg(&seed) >> 9));
        const float product = a[i] * b[i];  // one host multiply, nothing to fuse
        c[i] = -product;
    }
    float* d[6] = {};
    for (float*& p : d)
        if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) return 1;
    cudaMemcpy(d[0], a.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d[1], b.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d[2], c.data(), n * 4, cudaMemcpyHostToDevice);
    contraction<<<(n + 127) / 128, 128>>>(d[0], d[1], d[2], d[3], d[4], d[5], n);
    std::vector<float> expr(n), split(n), sub_split(n);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: contraction launch\n");
        return 1;
    }
    cudaMemcpy(expr.data(), d[3], n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(split.data(), d[4], n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(sub_split.data(), d[5], n * 4, cudaMemcpyDeviceToHost);
    unsigned fused_nonzero = 0;
    for (unsigned i = 0; i < n; ++i) {
        // c = -round(a*b): two roundings give exactly 0; one rounding gives
        // the product's rounding error, which is nonzero for most inputs.
        if (float_to_bits(split[i]) != 0u || float_to_bits(sub_split[i]) != 0u) {
            std::fprintf(stderr, "FAIL: __fadd_rn/__fsub_rn(__fmul_rn) fused at %u: %08x %08x\n",
                         i, float_to_bits(split[i]), float_to_bits(sub_split[i]));
            return 1;
        }
        const double exact = (double)a[i] * (double)b[i] + (double)c[i];
        if (expr[i] != (float)exact) {
            std::fprintf(stderr, "FAIL: a*b+c not contracted at %u: %08x want %08x\n", i,
                         float_to_bits(expr[i]), float_to_bits((float)exact));
            return 1;
        }
        fused_nonzero += expr[i] != 0.0f;
    }
    if (fused_nonzero < n / 2) {
        std::fprintf(stderr, "FAIL: only %u of %u contracted results carry a rounding error\n",
                     fused_nonzero, n);
        return 1;
    }
    for (float* p : d) cudaFree(p);
    return 0;
}

int main() {
    if (check_shuffles() != 0 || check_votes() != 0 || check_contraction() != 0) return 1;
    std::puts("PASS: CUDA intrinsic conformance on Apple GPU");
    return 0;
}
