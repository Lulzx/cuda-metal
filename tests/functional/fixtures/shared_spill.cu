#include <cstdlib>
// Shared memory beyond Metal's 32 KB threadgroup limit. CUDA allows 48 KB per
// block without opt-in and more with cudaFuncAttributeMaxDynamicSharedMemorySize;
// CuMetal serves such launches from the kernel's spill variant, whose
// __shared__ storage is a per-block slice of device memory. Run under
// MTL_DEBUG_LAYER=1 so an oversized threadgroup length would abort.
#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

constexpr unsigned kThreads = 256;

// Fills all of its dynamic shared memory, then reads it back reversed after a
// barrier, so every thread reads words other threads wrote.
__global__ void fill_reverse(const unsigned* in, unsigned words, unsigned* out) {
    extern __shared__ unsigned s[];
    for (unsigned i = threadIdx.x; i < words; i += blockDim.x) s[i] = in[i] ^ (blockIdx.x * 0x9E3779B9u);
    __syncthreads();
    unsigned acc = 0;
    for (unsigned i = threadIdx.x; i < words; i += blockDim.x) acc += s[words - 1 - i] * (i | 1u);
    __syncthreads();
    s[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

// Contended shared atomics, integer and float, in a block whose dynamic shared
// memory forces the spill variant. The counters sit past the 32 KB mark.
__global__ void shared_atomics(unsigned pad_words, unsigned* int_out, float* float_out) {
    extern __shared__ unsigned s[];
    unsigned* counters = s + pad_words;  // 4 sums, then a max
    float* sums = reinterpret_cast<float*>(counters + 8);
    if (threadIdx.x < 5) counters[threadIdx.x] = 0;
    if (threadIdx.x < 4) sums[threadIdx.x] = 0.0f;
    __syncthreads();
    for (int r = 0; r < 8; ++r) {
        atomicAdd(&counters[threadIdx.x & 3u], threadIdx.x + 1u);
        atomicAdd(&sums[threadIdx.x & 3u], 0.5f);
    }
    atomicMax(&counters[4], threadIdx.x * 1000u + blockIdx.x);
    __syncthreads();
    if (threadIdx.x < 5) int_out[blockIdx.x * 5 + threadIdx.x] = counters[threadIdx.x];
    if (threadIdx.x < 4) float_out[blockIdx.x * 4 + threadIdx.x] = sums[threadIdx.x];
}

// Static shared memory alone exceeds 32 KB (40 KB), so the kernel's own
// pipeline cannot exist and every launch runs the spill variant.
__global__ void big_static(unsigned* out) {
    __shared__ unsigned table[10240];
    for (unsigned i = threadIdx.x; i < 10240; i += blockDim.x) table[i] = i * 3u + blockIdx.x;
    __syncthreads();
    unsigned acc = 0;
    for (unsigned i = threadIdx.x; i < 10240; i += blockDim.x) acc += table[10239 - i];
    __shared__ unsigned partial[kThreads];
    partial[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned total = 0;
        for (unsigned t = 0; t < blockDim.x; ++t) total += partial[t];
        out[blockIdx.x] = total;
    }
}

// 25 launch arguments reach Metal buffer index 24, where a spill variant
// binds its shared memory, so this kernel gets no variant and a launch past
// Metal's limit must be refused rather than run with too little memory.
#define CM_ARGS(X) X(a0) X(a1) X(a2) X(a3) X(a4) X(a5) X(a6) X(a7) X(a8) X(a9) X(a10) X(a11) \
    X(a12) X(a13) X(a14) X(a15) X(a16) X(a17) X(a18) X(a19) X(a20) X(a21) X(a22) X(a23)
#define CM_PARAM(n) unsigned n,
#define CM_SUM(n) + n
#define CM_ONE(n) 1u,
__global__ void many_args(CM_ARGS(CM_PARAM) unsigned* out) {
    extern __shared__ unsigned s[];
    s[threadIdx.x] = 0 CM_ARGS(CM_SUM);
    __syncthreads();
    if (threadIdx.x == 0) out[0] = s[blockDim.x - 1];
}

// Tensor-core tiles staged in shared memory past the 32 KB mark (the
// cuda-samples bf16/tf32 GEMMs): in the spill variant the simdgroup matrix
// loads and stores address device memory instead of threadgroup memory.
__global__ void wmma_past_limit(const half* a, const half* b, float* out) {
    using namespace nvcuda;
    extern __shared__ unsigned char smem[];
    half* sa = reinterpret_cast<half*>(smem + 36 * 1024);
    half* sb = sa + 256;
    float* sc = reinterpret_cast<float*>(sb + 256);
    for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) {
        sa[i] = a[blockIdx.x * 256 + i];
        sb[i] = b[blockIdx.x * 256 + i];
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::load_matrix_sync(af, sa, 16);
    wmma::load_matrix_sync(bf, sb, 16);
    wmma::fill_fragment(cf, 1.0f);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(sc, cf, 16, wmma::mem_row_major);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) out[blockIdx.x * 256 + i] = sc[i];
}

// The bf16 shape goes through mma.h's 8x8 simdgroup-matrix marker; its
// spill variant hands simdgroup_load/store device pointers.
__global__ void wmma_bf16_past_limit(const __nv_bfloat16* a, const __nv_bfloat16* b, float* out) {
    using namespace nvcuda;
    extern __shared__ unsigned char smem[];
    __nv_bfloat16* sa = reinterpret_cast<__nv_bfloat16*>(smem + 36 * 1024);
    __nv_bfloat16* sb = sa + 256;
    float* sc = reinterpret_cast<float*>(sb + 256);
    for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) {
        sa[i] = a[blockIdx.x * 256 + i];
        sb[i] = b[blockIdx.x * 256 + i];
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::load_matrix_sync(af, sa, 16);
    wmma::load_matrix_sync(bf, sb, 16);
    wmma::fill_fragment(cf, 1.0f);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(sc, cf, 16, wmma::mem_row_major);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) out[blockIdx.x * 256 + i] = sc[i];
}

static int fails = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++fails;
    }
}

static unsigned expected_fold(const unsigned* in, unsigned words, unsigned block) {
    unsigned total = 0;
    for (unsigned t = 0; t < kThreads; ++t) {
        unsigned acc = 0;
        for (unsigned i = t; i < words; i += kThreads) acc += (in[words - 1 - i] ^ (block * 0x9E3779B9u)) * (i | 1u);
        total += acc;
    }
    return total;
}

static bool run_fold(const unsigned* d_in, const unsigned* h_in, unsigned* d_out, unsigned words,
                     unsigned blocks, cudaError_t* launch) {
    fill_reverse<<<blocks, kThreads, words * 4>>>(d_in, words, d_out);
    *launch = cudaGetLastError();
    if (*launch != cudaSuccess) return false;
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    static unsigned host[512];
    cudaMemcpy(host, d_out, blocks * 4, cudaMemcpyDeviceToHost);
    for (unsigned b = 0; b < blocks; ++b)
        if (host[b] != expected_fold(h_in, words, b)) return false;
    return true;
}

int main() {
    int optin = 0, per_block = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    check(optin == 101376, "opt-in shared memory ceiling is 99 KB");
    check(per_block > 0 && per_block <= optin, "per-block shared memory is within the opt-in ceiling");

    const unsigned max_words = 24 * 1024;  // 96 KB
    static unsigned h_in[max_words];
    for (unsigned i = 0; i < max_words; ++i) h_in[i] = i * 2654435761u + 12345u;
    unsigned *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, sizeof h_in);
    cudaMalloc(&d_out, 512 * 4);
    cudaMemcpy(d_in, h_in, sizeof h_in, cudaMemcpyHostToDevice);

    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, fill_reverse) == cudaSuccess &&
              attributes.maxDynamicSharedSizeBytes == 48 * 1024 - static_cast<int>(attributes.sharedSizeBytes),
          "default dynamic shared limit is 48 KB less static shared memory");

    cudaError_t launch = cudaSuccess;
    // Within Metal's limit (real threadgroup memory), exactly at it, one word
    // past it, and up to CUDA's 48 KB default, across many blocks.
    check(run_fold(d_in, h_in, d_out, 2048, 64, &launch), "8 KB fold");
    check(run_fold(d_in, h_in, d_out, 8192, 64, &launch), "32 KB fold");
    check(run_fold(d_in, h_in, d_out, 8193, 64, &launch), "32 KB + 4 B fold (spilled)");
    check(run_fold(d_in, h_in, d_out, 12 * 1024, 300, &launch), "48 KB fold over 300 blocks (spilled)");

    // Past 48 KB without opt-in: refused, nothing runs, not sticky.
    cudaMemset(d_out, 0xAB, 4);
    fill_reverse<<<1, kThreads, 64 * 1024>>>(d_in, 16 * 1024, d_out);
    check(cudaGetLastError() == cudaErrorInvalidValue, "64 KB without opt-in is cudaErrorInvalidValue");
    check(cudaDeviceSynchronize() == cudaSuccess, "refused launch is not sticky");
    unsigned untouched = 0;
    cudaMemcpy(&untouched, d_out, 4, cudaMemcpyDeviceToHost);
    check(untouched == 0xABABABABu, "refused launch did not run");

    // Opt in: 96 KB of dynamic shared memory.
    check(cudaFuncSetAttribute(fill_reverse, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               optin + 4) == cudaErrorInvalidValue,
          "attribute above the opt-in ceiling is refused");
    (void)cudaGetLastError();
    check(cudaFuncSetAttribute(fill_reverse, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               max_words * 4) == cudaSuccess,
          "attribute raised to 96 KB");
    check(cudaFuncGetAttributes(&attributes, fill_reverse) == cudaSuccess &&
              attributes.maxDynamicSharedSizeBytes == static_cast<int>(max_words * 4),
          "cudaFuncGetAttributes reports the raised limit");
    check(run_fold(d_in, h_in, d_out, max_words, 16, &launch), "96 KB fold with opt-in (spilled)");
    fill_reverse<<<1, kThreads, max_words * 4 + 16>>>(d_in, max_words, d_out);
    check(cudaGetLastError() == cudaErrorInvalidValue, "launch above the raised attribute is refused");
    check(cudaDeviceSynchronize() == cudaSuccess, "attribute refusal is not sticky");

    // Shared atomics in spilled memory.
    {
        const unsigned blocks = 32, pad = 9000;  // counters at 36 KB
        unsigned *d_int = nullptr;
        float* d_float = nullptr;
        cudaMalloc(&d_int, blocks * 20);
        cudaMalloc(&d_float, blocks * 16);
        shared_atomics<<<blocks, kThreads, (pad + 12) * 4>>>(pad, d_int, d_float);
        check(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
              "spilled shared atomics launch");
        unsigned h_int[blocks * 5];
        float h_float[blocks * 4];
        cudaMemcpy(h_int, d_int, sizeof h_int, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_float, d_float, sizeof h_float, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (unsigned b = 0; b < blocks; ++b) {
            for (unsigned c = 0; c < 4; ++c) {
                unsigned want = 0;
                for (unsigned t = c; t < kThreads; t += 4) want += 8u * (t + 1u);
                ok &= h_int[b * 5 + c] == want;
            }
            ok &= h_int[b * 5 + 4] == (kThreads - 1) * 1000u + b;
            for (unsigned c = 0; c < 4; ++c) ok &= h_float[b * 4 + c] == 0.5f * 8 * (kThreads / 4);
        }
        check(ok, "spilled shared atomics are exact");
        cudaFree(d_int);
        cudaFree(d_float);
    }

    // Static shared memory above 32 KB.
    {
        big_static<<<8, kThreads>>>(d_out);
        check(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
              "40 KB static shared launch");
        unsigned host[8];
        cudaMemcpy(host, d_out, sizeof host, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (unsigned b = 0; b < 8; ++b) {
            unsigned want = 0;
            for (unsigned i = 0; i < 10240; ++i) want += i * 3u + b;
            ok &= host[b] == want;
        }
        check(ok, "40 KB static shared result");
        check(cudaFuncGetAttributes(&attributes, big_static) == cudaSuccess &&
                  attributes.sharedSizeBytes > 32 * 1024,
              "cudaFuncGetAttributes reports static shared memory above 32 KB");
    }

    // Tensor-core tiles in spilled shared memory: 40 KB of dynamic shared
    // memory, tiles at 36 KB. Small integer inputs keep every product exact.
    // The typed PTX importer does not lower mma.h's tile markers, so the
    // registration run (CUMETAL_SPILL_SKIP_WMMA=1) covers only the native path.
    if (std::getenv("CUMETAL_SPILL_SKIP_WMMA") == nullptr) {
        constexpr unsigned kBlocks = 4;
        half h_a[kBlocks * 256], h_b[kBlocks * 256];
        for (unsigned i = 0; i < kBlocks * 256; ++i) {
            h_a[i] = __float2half(static_cast<float>(static_cast<int>(i * 7u % 9u) - 4));
            h_b[i] = __float2half(static_cast<float>(static_cast<int>(i * 5u % 7u) - 3));
        }
        half *d_a = nullptr, *d_b = nullptr;
        float* d_c = nullptr;
        cudaMalloc(&d_a, sizeof h_a);
        cudaMalloc(&d_b, sizeof h_b);
        cudaMalloc(&d_c, kBlocks * 256 * sizeof(float));
        cudaMemcpy(d_a, h_a, sizeof h_a, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof h_b, cudaMemcpyHostToDevice);
        wmma_past_limit<<<kBlocks, 32, 40 * 1024>>>(d_a, d_b, d_c);
        check(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
              "wmma tiles past 32 KB launch (spilled)");
        float host[kBlocks * 256];
        cudaMemcpy(host, d_c, sizeof host, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (unsigned blk = 0; blk < kBlocks; ++blk) {
            for (unsigned r = 0; r < 16; ++r) {
                for (unsigned c = 0; c < 16; ++c) {
                    float want = 1.0f;
                    for (unsigned k = 0; k < 16; ++k) {
                        // A row-major [r][k]; B column-major [k][c] at c*16+k.
                        want += __half2float(h_a[blk * 256 + r * 16 + k]) *
                                __half2float(h_b[blk * 256 + c * 16 + k]);
                    }
                    ok &= host[blk * 256 + r * 16 + c] == want;
                }
            }
        }
        check(ok, "wmma tiles past 32 KB are exact");

        // Same product through bf16 inputs (exact for these small integers).
        __nv_bfloat16 g_a[kBlocks * 256], g_b[kBlocks * 256];
        for (unsigned i = 0; i < kBlocks * 256; ++i) {
            g_a[i] = __float2bfloat16(__half2float(h_a[i]));
            g_b[i] = __float2bfloat16(__half2float(h_b[i]));
        }
        __nv_bfloat16 *e_a = nullptr, *e_b = nullptr;
        cudaMalloc(&e_a, sizeof g_a);
        cudaMalloc(&e_b, sizeof g_b);
        cudaMemcpy(e_a, g_a, sizeof g_a, cudaMemcpyHostToDevice);
        cudaMemcpy(e_b, g_b, sizeof g_b, cudaMemcpyHostToDevice);
        cudaMemset(d_c, 0, kBlocks * 256 * sizeof(float));
        // mma.h stages bf16 tiles in 16 KB of its own static shared memory,
        // so 40 KB of dynamic shared memory needs the opt-in, as the
        // cuda-samples tensor-core GEMMs request it.
        check(cudaFuncSetAttribute(wmma_bf16_past_limit, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   40 * 1024) == cudaSuccess,
              "bf16 tile kernel opts in to 40 KB of dynamic shared memory");
        wmma_bf16_past_limit<<<kBlocks, 32, 40 * 1024>>>(e_a, e_b, d_c);
        check(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
              "bf16 simdgroup-matrix tiles past 32 KB launch (spilled)");
        float bf_host[kBlocks * 256];
        cudaMemcpy(bf_host, d_c, sizeof bf_host, cudaMemcpyDeviceToHost);
        bool bf_ok = true;
        for (unsigned i = 0; i < kBlocks * 256; ++i) bf_ok &= bf_host[i] == host[i];
        check(bf_ok, "bf16 simdgroup-matrix tiles past 32 KB are exact");
        cudaFree(e_a);
        cudaFree(e_b);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    // A device allocation larger than the existing heap arenas: offering it to
    // a 64 MB arena first was a Metal validation failure (this test runs under
    // MTL_DEBUG_LAYER=1), not just a nil result.
    {
        void* small = nullptr;
        void* large = nullptr;
        check(cudaMalloc(&small, 8u << 20) == cudaSuccess && cudaMalloc(&large, 100u << 20) == cudaSuccess,
              "a 100 MB allocation after a heap arena exists");
        cudaFree(large);
        cudaFree(small);
    }

    // No spill variant: within Metal's limit it runs, past it it is refused.
    many_args<<<1, kThreads, 8 * 1024>>>(CM_ARGS(CM_ONE) d_out);
    unsigned answer = 0;
    check(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
              cudaMemcpy(&answer, d_out, 4, cudaMemcpyDeviceToHost) == cudaSuccess && answer == 24u,
          "25-argument kernel within the threadgroup limit");
    cudaMemset(d_out, 0xAB, 4);
    many_args<<<1, kThreads, 40 * 1024>>>(CM_ARGS(CM_ONE) d_out);
    check(cudaGetLastError() == cudaErrorInvalidValue,
          "kernel without a spill variant is refused past the threadgroup limit");
    check(cudaDeviceSynchronize() == cudaSuccess, "missing-variant refusal is not sticky");
    cudaMemcpy(&answer, d_out, 4, cudaMemcpyDeviceToHost);
    check(answer == 0xABABABABu, "refused launch without a spill variant did not run");

    // A grid whose shared memory cannot be allocated is refused.
    fill_reverse<<<dim3(65535, 65535), kThreads, max_words * 4>>>(d_in, max_words, d_out);
    check(cudaGetLastError() == cudaErrorMemoryAllocation,
          "unallocatable spill buffer is cudaErrorMemoryAllocation");
    (void)cudaDeviceSynchronize();
    (void)cudaGetLastError();

    if (fails == 0) std::printf("PASS: shared memory beyond the Metal threadgroup limit\n");
    return fails == 0 ? 0 : 1;
}
