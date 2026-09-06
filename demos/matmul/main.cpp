#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

extern "C" int matmul_mps_reference(const float *, const float *, float *, int,
                                    int, int, int, double *);
using Clock = std::chrono::steady_clock;
#define CUDA(x)                                                                \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(e));           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define BLAS(x)                                                                \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "%s failed: %d\n", #x, int(e));                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
struct Variant {
  const char *name;
  int bm, bn, bx, by;
};
const Variant variants[] = {
    {"naive", 16, 16, 16, 16},        {"coalesced", 16, 16, 16, 16},
    {"tiled", 16, 16, 16, 16},        {"registers", 16, 16, 8, 8},
    {"vectorized", 16, 16, 8, 8},     {"block64", 64, 64, 256, 1},
    {"block64x32", 64, 32, 128, 1},   {"block32", 32, 32, 64, 1},
    {"block64_8", 64, 64, 128, 1},    {"block64_32", 64, 64, 256, 1},
    {"block64_8x8", 64, 64, 64, 1},   {"block64_8x4", 64, 64, 128, 1},
    {"block64_4x8", 64, 64, 128, 1},  {"block128x64", 128, 64, 256, 1},
    {"block64x128", 64, 128, 256, 1}, {"block64_k64", 64, 64, 256, 1}};
static void validate(const char *name, const std::vector<float> &c,
                     const std::vector<float> &ref, int K) {
  double maxerr = 0, sse = 0, ssref = 0;
  size_t bad = 0;
  // Forward-error allowance with an absolute floor for cancellation near zero.
  double tol = 2e-5 * K;
  for (size_t i = 0; i < c.size(); i++) {
    double d = std::abs(double(c[i]) - ref[i]);
    maxerr = std::max(maxerr, d);
    sse += d * d;
    ssref += double(ref[i]) * ref[i];
    if (!std::isfinite(c[i]) || d > tol + 2e-4 * std::abs(ref[i]))
      bad++;
  }
  printf("check,%s,max_abs=%.9g,relative_l2=%.9g,bad=%zu\n", name, maxerr,
         std::sqrt(sse / std::max(ssref, 1e-30)), bad);
  if (bad)
    exit(1);
}
static void report(const char *name, int M, int K, int N,
                   std::vector<double> times) {
  double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  std::sort(times.begin(), times.end());
  double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) / 2;
  printf("result,%s,%d,%d,%d,%zu,%.6f,%.6f,%.6f,%.6f,%.3f\n", name, M, K, N,
         times.size(), mean, median, times.front(), times.back(),
         2.0 * M * K * N / (mean * 1e6));
  fflush(stdout);
}
int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s metallib M K N iterations [variant|all]\n",
            argv[0]);
    return 2;
  }
  int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]),
      iterations = atoi(argv[5]);
  std::string selected = argc > 6 ? argv[6] : "all";
  if (M <= 0 || K <= 0 || N <= 0 || M > 8192 || K > 8192 || N > 8192 ||
      iterations <= 0 || iterations > 1000)
    return 2;
  bool known = selected == "all" || selected == "mps" ||
               selected == "cublas_mps" || selected == "accelerate";
  for (auto v : variants)
    known |= selected == v.name;
  if (!known)
    return 2;
  CUDA(cudaInit(0));
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("device,%s; FP32; warmup=2; synchronized wall ms; excludes "
         "allocation, transfers, compilation\n",
         prop.name);
  std::vector<float> a(size_t(M) * K), b(size_t(K) * N), c(size_t(M) * N),
      ref(c.size());
  std::mt19937 rng(20260906);
  std::uniform_real_distribution<float> d(-1, 1);
  for (auto &v : a)
    v = d(rng);
  for (auto &v : b)
    v = d(rng);
  auto cpu = [&] {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, a.data(),
                K, b.data(), N, 0, ref.data(), N);
  };
  cpu();
  // Independently check reference dot products in double precision.
  for (int q = 0; q < 64; q++) {
    size_t idx = (size_t(q) * 104729 + 17) % ref.size();
    int r = idx / N, col = idx % N;
    double s = 0;
    for (int k = 0; k < K; k++)
      s += double(a[r * K + k]) * b[k * N + col];
    if (std::abs(s - ref[idx]) > 2e-5 * K + 2e-4 * std::abs(s)) {
      fprintf(stderr, "CPU reference failed\n");
      return 1;
    }
  }
  puts("columns,name,M,K,N,iterations,mean_ms,median_ms,min_ms,max_ms,GFLOPs_"
       "from_mean");
  if (selected == "all" || selected == "accelerate") {
    std::vector<double> t;
    cpu();
    for (int i = 0; i < iterations; i++) {
      auto start = Clock::now();
      cpu();
      t.push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count());
    }
    report("accelerate", M, K, N, t);
  }
  if (selected == "all" || selected == "mps") {
    std::vector<double> t(iterations);
    if (matmul_mps_reference(a.data(), b.data(), c.data(), M, K, N, iterations,
                             t.data()))
      return 1;
    validate("mps", c, ref, K);
    report("mps", M, K, N, t);
  }
  float *da, *db, *dc;
  CUDA(cudaMalloc(&da, a.size() * 4));
  CUDA(cudaMalloc(&db, b.size() * 4));
  CUDA(cudaMalloc(&dc, c.size() * 4));
  CUDA(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  static const cumetalKernelArgInfo_t info[] = {
      {CUMETAL_ARG_BUFFER, 0}, {CUMETAL_ARG_BUFFER, 0}, {CUMETAL_ARG_BUFFER, 0},
      {CUMETAL_ARG_BYTES, 4},  {CUMETAL_ARG_BYTES, 4},  {CUMETAL_ARG_BYTES, 4}};
  void *args[] = {&da, &db, &dc, &M, &K, &N};
  for (auto v : variants) {
    if (selected != "all" && selected != v.name)
      continue;
    if (std::string(v.name) == "vectorized" && (K % 4 || N % 4)) {
      puts("skip,vectorized,article requires aligned K and N");
      if (selected == "vectorized")
        return 2;
      continue;
    }
    cumetalKernel_t kernel{argv[1], v.name, 6, info};
    dim3 grid((N + v.bn - 1) / v.bn, (M + v.bm - 1) / v.bm), block(v.bx, v.by);
    if (std::string(v.name) == "naive")
      grid = dim3((M + 15) / 16, (N + 15) / 16);
    CUDA(cudaMemset(dc, 0xff,
                    c.size() * 4)); // NaN sentinel catches unwritten output.
    auto launch = [&] {
      CUDA(cudaLaunchKernel(&kernel, grid, block, args, 0, nullptr));
      CUDA(cudaDeviceSynchronize());
    };
    auto cold = Clock::now();
    launch();
    printf(
        "first_launch,%s,%.6f ms\n", v.name,
        std::chrono::duration<double, std::milli>(Clock::now() - cold).count());
    launch();
    std::vector<double> t;
    for (int i = 0; i < iterations; i++) {
      auto start = Clock::now();
      launch();
      t.push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count());
    }
    CUDA(cudaMemcpy(c.data(), dc, c.size() * 4, cudaMemcpyDeviceToHost));
    validate(v.name, c, ref, K);
    report(v.name, M, K, N, t);
  }
  if (selected == "all" || selected == "cublas_mps") {
    cublasHandle_t handle;
    BLAS(cublasCreate(&handle));
    float alpha = 1, beta = 0;
    CUDA(cudaMemset(dc, 0xff, c.size() * 4));
    auto launch = [&] {
      BLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, db, N,
                       da, K, &beta, dc, N));
      CUDA(cudaDeviceSynchronize());
    };
    launch();
    launch();
    std::vector<double> t;
    for (int i = 0; i < iterations; i++) {
      auto start = Clock::now();
      launch();
      t.push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count());
    }
    CUDA(cudaMemcpy(c.data(), dc, c.size() * 4, cudaMemcpyDeviceToHost));
    validate("cublas_mps", c, ref, K);
    report("cublas_mps", M, K, N, t);
    BLAS(cublasDestroy(handle));
  }
  CUDA(cudaFree(da));
  CUDA(cudaFree(db));
  CUDA(cudaFree(dc));
}
