#include <cuda_runtime.h>

// Article algorithms, with names shortened and repeated scalar code expressed
// as fixed-size loops. Original vector path intentionally requires K,N % 4 ==
// 0.
extern "C" __global__ void naive(const float *A, const float *B, float *C,
                                 int M, int K, int N) {
  int r = threadIdx.x + blockDim.x * blockIdx.x,
      c = threadIdx.y + blockDim.y * blockIdx.y;
  if (r < M && c < N) {
    float s = 0;
    for (int j = 0; j < K; j++)
      s += A[r * K + j] * B[j * N + c];
    C[r * N + c] = s;
  }
}
extern "C" __global__ void coalesced(const float *A, const float *B, float *C,
                                     int M, int K, int N) {
  int r = threadIdx.y + blockDim.y * blockIdx.y,
      c = threadIdx.x + blockDim.x * blockIdx.x;
  if (r < M && c < N) {
    float s = 0;
    for (int j = 0; j < K; j++)
      s += A[r * K + j] * B[j * N + c];
    C[r * N + c] = s;
  }
}
extern "C" __global__ void tiled(const float *A, const float *B, float *C,
                                 int M, int K, int N) {
  __shared__ float a[16][16], b[16][16];
  int x = threadIdx.x, y = threadIdx.y, r = blockIdx.y * 16 + y,
      c = blockIdx.x * 16 + x;
  float s = 0;
  for (int t = 0; t < (K + 15) / 16; t++) {
    a[y][x] = (r < M && t * 16 + x < K) ? A[r * K + t * 16 + x] : 0;
    b[y][x] = (t * 16 + y < K && c < N) ? B[(t * 16 + y) * N + c] : 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; j++)
      s += a[y][j] * b[j][x];
    __syncthreads();
  }
  if (r < M && c < N)
    C[r * N + c] = s;
}
template <bool VECTOR>
__device__ __forceinline__ void reg_impl(const float *A, const float *B,
                                         float *C, int M, int K, int N) {
  __shared__ float a[16][16], b[16][16];
  int x = threadIdx.x, y = threadIdx.y, r = blockIdx.y * 16 + y * 2,
      c = blockIdx.x * 16 + x * 2;
  float c00 = 0, c01 = 0, c10 = 0, c11 = 0;
  for (int t = 0; t < (K + 15) / 16; t++) {
    if (VECTOR) {
      int tid = y * blockDim.x + x, lr = tid / 4, lc = (tid % 4) * 4;
      int ar = blockIdx.y * 16 + lr, ac = t * 16 + lc, br = t * 16 + lr,
          bc = blockIdx.x * 16 + lc;
      if (ar < M && ac + 3 < K) {
        float4 v = *reinterpret_cast<const float4 *>(A + ar * K + ac);
        a[lr][lc] = v.x;
        a[lr][lc + 1] = v.y;
        a[lr][lc + 2] = v.z;
        a[lr][lc + 3] = v.w;
      } else
        for (int i = 0; i < 4; i++)
          a[lr][lc + i] = (ar < M && ac + i < K) ? A[ar * K + ac + i] : 0;
      if (br < K && bc + 3 < N) {
        float4 v = *reinterpret_cast<const float4 *>(B + br * N + bc);
        b[lr][lc] = v.x;
        b[lr][lc + 1] = v.y;
        b[lr][lc + 2] = v.z;
        b[lr][lc + 3] = v.w;
      } else
        for (int i = 0; i < 4; i++)
          b[lr][lc + i] = (br < K && bc + i < N) ? B[br * N + bc + i] : 0;
    } else {
      for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++) {
          int ar = blockIdx.y * 16 + y * 2 + dy, ac = t * 16 + x * 2 + dx;
          a[y * 2 + dy][x * 2 + dx] = (ar < M && ac < K) ? A[ar * K + ac] : 0;
          int br = t * 16 + y * 2 + dy, bc = blockIdx.x * 16 + x * 2 + dx;
          b[y * 2 + dy][x * 2 + dx] = (br < K && bc < N) ? B[br * N + bc] : 0;
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; j++) {
      float a0 = a[y * 2][j], a1 = a[y * 2 + 1][j], b0 = b[j][x * 2],
            b1 = b[j][x * 2 + 1];
      c00 += a0 * b0;
      c01 += a0 * b1;
      c10 += a1 * b0;
      c11 += a1 * b1;
    }
    __syncthreads();
  }
  if (r < M && c < N)
    C[r * N + c] = c00;
  if (r < M && c + 1 < N)
    C[r * N + c + 1] = c01;
  if (r + 1 < M && c < N)
    C[(r + 1) * N + c] = c10;
  if (r + 1 < M && c + 1 < N)
    C[(r + 1) * N + c + 1] = c11;
}
extern "C" __global__ void registers(const float *A, const float *B, float *C,
                                     int M, int K, int N) {
  reg_impl<false>(A, B, C, M, K, N);
}
extern "C" __global__ void vectorized(const float *A, const float *B, float *C,
                                      int M, int K, int N) {
  reg_impl<true>(A, B, C, M, K, N);
}

// Portable CUDA optimization: larger output tile, more reuse per load/barrier.
// Strided per-thread columns keep neighboring lanes on neighboring addresses.
template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void blocked_impl(const float *A, const float *B,
                                             float *C, int M, int K, int N) {
  constexpr int NT = BM * BN / (TM * TN);
  __shared__ float a[BM][BK], b[BK][BN];
  int tid = threadIdx.x, tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  for (int base = 0; base < K; base += BK) {
    for (int q = tid; q < BM * BK; q += NT) {
      int r = q / BK, k = q % BK, gr = blockIdx.y * BM + r;
      a[r][k] = (gr < M && base + k < K) ? A[gr * K + base + k] : 0;
    }
    for (int q = tid; q < BK * BN; q += NT) {
      int k = q / BN, c = q % BN, gc = blockIdx.x * BN + c;
      b[k][c] = (base + k < K && gc < N) ? B[(base + k) * N + gc] : 0;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
      float av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; i++)
        av[i] = a[tr + i * (BM / TM)][k];
#pragma unroll
      for (int j = 0; j < TN; j++)
        bv[j] = b[k][tc + j * (BN / TN)];
#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] += av[i] * bv[j];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; i++) {
#pragma unroll
    for (int j = 0; j < TN; j++) {
      int r = blockIdx.y * BM + tr + i * (BM / TM),
          c = blockIdx.x * BN + tc + j * (BN / TN);
      if (r < M && c < N)
        C[r * N + c] = acc[i][j];
    }
  }
}
#define BLOCK(NAME, BM, BN, BK, TM, TN)                                        \
  extern "C" __global__ void NAME(const float *A, const float *B, float *C,    \
                                  int M, int K, int N) {                       \
    blocked_impl<BM, BN, BK, TM, TN>(A, B, C, M, K, N);                        \
  }
BLOCK(block64, 64, 64, 16, 4, 4)
BLOCK(block64x32, 64, 32, 16, 4, 4)
BLOCK(block32, 32, 32, 16, 4, 4)
BLOCK(block64_8, 64, 64, 8, 8, 4)
BLOCK(block64_32, 64, 64, 32, 4, 4)
BLOCK(block64_8x8, 64, 64, 32, 8, 8)
BLOCK(block64_8x4, 64, 64, 32, 8, 4)
BLOCK(block64_4x8, 64, 64, 32, 4, 8)
BLOCK(block128x64, 128, 64, 16, 8, 4)
BLOCK(block64x128, 64, 128, 16, 4, 8)
BLOCK(block64_k64, 64, 64, 64, 4, 4)
