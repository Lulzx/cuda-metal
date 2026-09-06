// Same blocked algorithm written directly in MSL to isolate translation
// overhead.
#include <metal_stdlib>
using namespace metal;
kernel void block64_32(device const float *A [[buffer(0)]],
                       device const float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       constant int &M [[buffer(3)]],
                       constant int &K [[buffer(4)]],
                       constant int &N [[buffer(5)]],
                       uint3 tid3 [[thread_position_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;

  constexpr int NT = BM * BN / (TM * TN);
  threadgroup float a[BM][BK], b[BK][BN];
  int tid = tid3.x, tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  for (int base = 0; base < K; base += BK) {
    for (int q = tid; q < BM * BK; q += NT) {
      int r = q / BK, k = q % BK, gr = group.y * BM + r;
      a[r][k] = (gr < M && base + k < K) ? A[gr * K + base + k] : 0;
    }
    for (int q = tid; q < BK * BN; q += NT) {
      int k = q / BN, c = q % BN, gc = group.x * BN + c;
      b[k][c] = (base + k < K && gc < N) ? B[(base + k) * N + gc] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (int i = 0; i < TM; i++) {
#pragma unroll
    for (int j = 0; j < TN; j++) {
      int r = group.y * BM + tr + i * (BM / TM),
          c = group.x * BN + tc + j * (BN / TN);
      if (r < M && c < N)
        C[r * N + c] = acc[i][j];
    }
  }
}
kernel void block64_8x8(device const float *A [[buffer(0)]],
                        device const float *B [[buffer(1)]],
                        device float *C [[buffer(2)]],
                        constant int &M [[buffer(3)]],
                        constant int &K [[buffer(4)]],
                        constant int &N [[buffer(5)]],
                        uint3 tid3 [[thread_position_in_threadgroup]],
                        uint3 group [[threadgroup_position_in_grid]]) {
  constexpr int BM = 64, BN = 64, BK = 32, TM = 8, TN = 8;

  constexpr int NT = BM * BN / (TM * TN);
  threadgroup float a[BM][BK], b[BK][BN];
  int tid = tid3.x, tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  for (int base = 0; base < K; base += BK) {
    for (int q = tid; q < BM * BK; q += NT) {
      int r = q / BK, k = q % BK, gr = group.y * BM + r;
      a[r][k] = (gr < M && base + k < K) ? A[gr * K + base + k] : 0;
    }
    for (int q = tid; q < BK * BN; q += NT) {
      int k = q / BN, c = q % BN, gc = group.x * BN + c;
      b[k][c] = (base + k < K && gc < N) ? B[(base + k) * N + gc] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (int i = 0; i < TM; i++) {
#pragma unroll
    for (int j = 0; j < TN; j++) {
      int r = group.y * BM + tr + i * (BM / TM),
          c = group.x * BN + tc + j * (BN / TN);
      if (r < M && c < N)
        C[r * N + c] = acc[i][j];
    }
  }
}
kernel void block64_k64(device const float *A [[buffer(0)]],
                        device const float *B [[buffer(1)]],
                        device float *C [[buffer(2)]],
                        constant int &M [[buffer(3)]],
                        constant int &K [[buffer(4)]],
                        constant int &N [[buffer(5)]],
                        uint3 tid3 [[thread_position_in_threadgroup]],
                        uint3 group [[threadgroup_position_in_grid]]) {
  constexpr int BM = 64, BN = 64, BK = 64, TM = 4, TN = 4;

  constexpr int NT = BM * BN / (TM * TN);
  threadgroup float a[BM][BK], b[BK][BN];
  int tid = tid3.x, tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  for (int base = 0; base < K; base += BK) {
    for (int q = tid; q < BM * BK; q += NT) {
      int r = q / BK, k = q % BK, gr = group.y * BM + r;
      a[r][k] = (gr < M && base + k < K) ? A[gr * K + base + k] : 0;
    }
    for (int q = tid; q < BK * BN; q += NT) {
      int k = q / BN, c = q % BN, gc = group.x * BN + c;
      b[k][c] = (base + k < K && gc < N) ? B[(base + k) * N + gc] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (int i = 0; i < TM; i++) {
#pragma unroll
    for (int j = 0; j < TN; j++) {
      int r = group.y * BM + tr + i * (BM / TM),
          c = group.x * BN + tc + j * (BN / TN);
      if (r < M && c < N)
        C[r * N + c] = acc[i][j];
    }
  }
}
