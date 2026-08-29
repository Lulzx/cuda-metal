#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>
#include <cstdio>

namespace wmma = nvcuda::wmma;

__global__ void half_wmma_probe(const half* a, const half* b, float* output) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::fill_fragment(cf, 3.0f);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(output, cf, 16, wmma::mem_row_major);
}

__global__ void integer_wmma_probe(const unsigned char* a,
                                   const unsigned char* b, int* output) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char,
                   wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char,
                   wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf;
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::fill_fragment(cf, 1);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(output, cf, 16, wmma::mem_row_major);
}

__global__ void tf32_wmma_probe(const float* a, const float* b, float* output) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                   wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> cf;
    wmma::load_matrix_sync(af, a, 8);
    wmma::load_matrix_sync(bf, b, 8);
    wmma::fill_fragment(cf, 2.0f);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(output, cf, 16, wmma::mem_row_major);
}

__global__ void bf16_wmma_probe(const __nv_bfloat16* a,
                                const __nv_bfloat16* b,
                                float* output) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                   wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                   wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::fill_fragment(cf, 4.0f);
#pragma unroll
    for (int i = 0; i < cf.num_elements; ++i) cf.x[i] += 1.0f;
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(output, cf, 16, wmma::mem_row_major);
}

__global__ void double_wmma_probe(const double* a, const double* b,
                                  double* output) {
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> cf;
    wmma::load_matrix_sync(af, a, 4);
    wmma::load_matrix_sync(bf, b, 4);
    wmma::fill_fragment(cf, 3.0);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(output, cf, 8, wmma::mem_row_major);
}

template <typename T>
bool copy_to_device(T** device, const T* host, size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(device), count * sizeof(T)) ==
               cudaSuccess &&
           cudaMemcpy(*device, host, count * sizeof(T), cudaMemcpyHostToDevice) ==
               cudaSuccess;
}

int main() {
    half half_a[16 * 16];
    half half_b[16 * 16];
    unsigned char int_a[16 * 16];
    unsigned char int_b[16 * 16];
    float tf32_a[16 * 8];
    float tf32_b[8 * 16];
    __nv_bfloat16 bf16_a[16 * 16];
    __nv_bfloat16 bf16_b[16 * 16];
    double double_a[8 * 4];
    double double_b[4 * 8];
    for (int i = 0; i < 16 * 16; ++i) {
        half_a[i] = static_cast<half>(1.0f);
        half_b[i] = static_cast<half>(2.0f);
        int_a[i] = 2;
        int_b[i] = 3;
        bf16_a[i] = __nv_bfloat16(1.5f);
        bf16_b[i] = __nv_bfloat16(2.0f);
    }
    for (int i = 0; i < 16 * 8; ++i) tf32_a[i] = 1.0f;
    for (int i = 0; i < 8 * 16; ++i) tf32_b[i] = 0.5f;
    for (double& value : double_a) value = 1.25;
    for (double& value : double_b) value = 2.0;

    half *d_half_a = nullptr, *d_half_b = nullptr;
    unsigned char *d_int_a = nullptr, *d_int_b = nullptr;
    float *d_tf32_a = nullptr, *d_tf32_b = nullptr;
    __nv_bfloat16 *d_bf16_a = nullptr, *d_bf16_b = nullptr;
    double *d_double_a = nullptr, *d_double_b = nullptr, *d_double_out = nullptr;
    float *d_half_out = nullptr, *d_tf32_out = nullptr, *d_bf16_out = nullptr;
    int* d_int_out = nullptr;
    if (!copy_to_device(&d_half_a, half_a, 16 * 16) ||
        !copy_to_device(&d_half_b, half_b, 16 * 16) ||
        !copy_to_device(&d_int_a, int_a, 16 * 16) ||
        !copy_to_device(&d_int_b, int_b, 16 * 16) ||
        !copy_to_device(&d_tf32_a, tf32_a, 16 * 8) ||
        !copy_to_device(&d_tf32_b, tf32_b, 8 * 16) ||
        !copy_to_device(&d_bf16_a, bf16_a, 16 * 16) ||
        !copy_to_device(&d_bf16_b, bf16_b, 16 * 16) ||
        !copy_to_device(&d_double_a, double_a, 8 * 4) ||
        !copy_to_device(&d_double_b, double_b, 4 * 8) ||
        cudaMalloc(reinterpret_cast<void**>(&d_half_out), 256 * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_int_out), 256 * sizeof(int)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_tf32_out), 256 * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_bf16_out), 256 * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_double_out), 64 * sizeof(double)) !=
            cudaSuccess) {
        return 1;
    }

    half_wmma_probe<<<1, 32>>>(d_half_a, d_half_b, d_half_out);
    integer_wmma_probe<<<1, 32>>>(d_int_a, d_int_b, d_int_out);
    tf32_wmma_probe<<<1, 32>>>(d_tf32_a, d_tf32_b, d_tf32_out);
    bf16_wmma_probe<<<1, 32>>>(d_bf16_a, d_bf16_b, d_bf16_out);
    double_wmma_probe<<<1, 32>>>(d_double_a, d_double_b, d_double_out);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;

    float half_out[256]{};
    int int_out[256]{};
    float tf32_out[256]{};
    float bf16_out[256]{};
    double double_out[64]{};
    if (cudaMemcpy(half_out, d_half_out, sizeof(half_out),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(int_out, d_int_out, sizeof(int_out),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(tf32_out, d_tf32_out, sizeof(tf32_out),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(bf16_out, d_bf16_out, sizeof(bf16_out),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(double_out, d_double_out, sizeof(double_out),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }

    bool ok = true;
    for (int i = 0; i < 256; ++i) {
        ok = ok && std::fabs(half_out[i] - 35.0f) < 1.0e-5f;
        ok = ok && int_out[i] == 97;
        ok = ok && std::fabs(tf32_out[i] - 6.0f) < 1.0e-5f;
        ok = ok && std::fabs(bf16_out[i] - 53.0f) < 1.0e-5f;
    }
    for (double value : double_out) {
        ok = ok && std::fabs(value - 13.0) < 1.0e-5;
    }
    std::printf("%s: WMMA half=%g integer=%d tf32=%g bf16=%g double=%g\n",
                ok ? "PASS" : "FAIL", half_out[0], int_out[0], tf32_out[0],
                bf16_out[0], double_out[0]);
    return ok ? 0 : 1;
}
