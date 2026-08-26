// Tiny MNIST diffusion model (DDPM) sampled entirely with CUDA kernels,
// running on an Apple GPU through CuMetal.
//
//   ./run.sh --check          # one forward pass vs the PyTorch reference
//   ./run.sh                  # sample a 4x4 grid of digits -> out/samples.png
//
// Weights come from train.py (~290k params, no attention).  The layer order
// below is the same contract train.py implements; the two must stay in sync.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),    \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const int kT = 1000;    // diffusion steps used in training
static const int kTdim = 64;   // time-embedding width

// ---------------------------------------------------------------- kernels

__global__ void k_conv3(const float *__restrict__ x, const float *__restrict__ w,
                        const float *__restrict__ b, float *__restrict__ y,
                        int N, int Cin, int Cout, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * Cout * H * W) return;
  int ox = idx % W, oy = (idx / W) % H;
  int co = (idx / (W * H)) % Cout, n = idx / (W * H * Cout);
  float acc = b[co];
  for (int ci = 0; ci < Cin; ++ci) {
    const float *xp = x + (size_t)(n * Cin + ci) * H * W;
    const float *wp = w + (size_t)(co * Cin + ci) * 9;
    for (int ky = 0; ky < 3; ++ky) {
      int yy = oy + ky - 1;
      if (yy < 0 || yy >= H) continue;
      for (int kx = 0; kx < 3; ++kx) {
        int xx = ox + kx - 1;
        if (xx < 0 || xx >= W) continue;
        acc += xp[yy * W + xx] * wp[ky * 3 + kx];
      }
    }
  }
  y[idx] = acc;
}

// Per-channel bias: how the time embedding enters each block.
__global__ void k_add_cbias(float *y, const float *b, int N, int C, int HW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C * HW) return;
  y[idx] += b[(idx / HW) % C];
}

// GroupNorm + SiLU, one block per (image, group).  blockDim must be 128.
__global__ void k_gn_silu(float *y, const float *g, const float *b, int C, int G,
                          int HW) {
  int gi = blockIdx.x % G, n = blockIdx.x / G;
  int cpg = C / G, cnt = cpg * HW;
  float *base = y + (size_t)(n * C + gi * cpg) * HW;
  __shared__ float ssum[128], ssq[128];
  float s = 0.f, q = 0.f;
  for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
    float v = base[i];
    s += v;
    q += v * v;
  }
  ssum[threadIdx.x] = s;
  ssq[threadIdx.x] = q;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) {
      ssum[threadIdx.x] += ssum[threadIdx.x + st];
      ssq[threadIdx.x] += ssq[threadIdx.x + st];
    }
    __syncthreads();
  }
  float mean = ssum[0] / cnt;
  float inv = rsqrtf(ssq[0] / cnt - mean * mean + 1e-5f);
  for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
    int c = gi * cpg + i / HW;
    float v = (base[i] - mean) * inv * g[c] + b[c];
    base[i] = v / (1.f + expf(-v));  // SiLU
  }
}

__global__ void k_avgpool2(const float *x, float *y, int N, int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = H / 2, ow = W / 2;
  if (idx >= N * C * oh * ow) return;
  int px = idx % ow, py = (idx / ow) % oh, nc = idx / (ow * oh);
  const float *xp = x + (size_t)nc * H * W;
  y[idx] = 0.25f * (xp[(2 * py) * W + 2 * px] + xp[(2 * py) * W + 2 * px + 1] +
                    xp[(2 * py + 1) * W + 2 * px] + xp[(2 * py + 1) * W + 2 * px + 1]);
}

__global__ void k_upsample2(const float *x, float *y, int N, int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = H * 2, ow = W * 2;
  if (idx >= N * C * oh * ow) return;
  int px = idx % ow, py = (idx / ow) % oh, nc = idx / (ow * oh);
  y[idx] = x[(size_t)nc * H * W + (py / 2) * W + px / 2];
}

__global__ void k_concat(const float *a, const float *b, float *y, int N, int Ca,
                         int Cb, int HW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int C = Ca + Cb;
  if (idx >= N * C * HW) return;
  int i = idx % HW, c = (idx / HW) % C, n = idx / (HW * C);
  y[idx] = (c < Ca) ? a[(size_t)(n * Ca + c) * HW + i]
                    : b[(size_t)(n * Cb + (c - Ca)) * HW + i];
}

// One ancestral DDPM step: x <- c1 * (x - c2 * eps) + sigma * z
__global__ void k_ddpm_step(float *x, const float *eps, const float *z, float c1,
                            float c2, float sigma, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) x[idx] = c1 * (x[idx] - c2 * eps[idx]) + sigma * z[idx];
}

// ---------------------------------------------------------------- weights

struct Tensor {
  int ndim = 0, dims[4] = {1, 1, 1, 1};
  std::vector<float> host;
  float *dev = nullptr;
  size_t size() const { return host.size(); }
};

static std::map<std::string, Tensor> g_w;

static Tensor &W(const std::string &name) {
  auto it = g_w.find(name);
  if (it == g_w.end()) {
    fprintf(stderr, "missing tensor '%s' in model.bin\n", name.c_str());
    exit(1);
  }
  return it->second;
}

static void load_model(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s (run train.py first)\n", path);
    exit(1);
  }
  char magic[4];
  int version = 0, count = 0;
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CMDF", 4) != 0) {
    fprintf(stderr, "%s: bad magic\n", path);
    exit(1);
  }
  fread(&version, 4, 1, f);
  fread(&count, 4, 1, f);
  size_t total = 0;
  for (int i = 0; i < count; ++i) {
    char name[32];
    Tensor t;
    fread(name, 1, 32, f);
    fread(&t.ndim, 4, 1, f);
    fread(t.dims, 4, 4, f);
    size_t n = 1;
    for (int d = 0; d < t.ndim; ++d) n *= (size_t)t.dims[d];
    t.host.resize(n);
    fread(t.host.data(), 4, n, f);
    CHECK(cudaMalloc(&t.dev, n * sizeof(float)));
    CHECK(cudaMemcpy(t.dev, t.host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    total += n;
    g_w[std::string(name)] = std::move(t);
  }
  fclose(f);
  printf("model: %d tensors, %zu params\n", count, total);
}

// Scratch activations, allocated once and reused across sampling steps.
static std::map<std::string, std::pair<float *, size_t>> g_buf;

static float *buf(const std::string &name, size_t n) {
  auto &e = g_buf[name];
  if (e.second < n) {
    if (e.first) CHECK(cudaFree(e.first));
    CHECK(cudaMalloc(&e.first, n * sizeof(float)));
    e.second = n;
  }
  return e.first;
}

// ---------------------------------------------------------------- model

static int grid_for(size_t n) { return (int)((n + 255) / 256); }

static void silu_host(std::vector<float> &v) {
  for (float &x : v) x = x / (1.f + expf(-x));
}

// dense: y = W x + b, with W stored [out][in]
static std::vector<float> dense(const Tensor &w, const Tensor &b,
                                const std::vector<float> &x) {
  int out = w.dims[0], in = w.dims[1];
  std::vector<float> y(out);
  for (int o = 0; o < out; ++o) {
    float acc = b.host[o];
    for (int i = 0; i < in; ++i) acc += w.host[(size_t)o * in + i] * x[i];
    y[o] = acc;
  }
  return y;
}

// The time-embedding MLP is 8k params on a batch of one timestep: host is fine.
static std::vector<float> time_embed(int t) {
  int half = kTdim / 2;
  std::vector<float> e(kTdim);
  for (int i = 0; i < half; ++i) {
    float a = t * expf(-logf(10000.f) * i / half);
    e[i] = sinf(a);
    e[half + i] = cosf(a);
  }
  std::vector<float> h = dense(W("fc1.weight"), W("fc1.bias"), e);
  silu_host(h);
  return dense(W("fc2.weight"), W("fc2.bias"), h);
}

static void conv3(const std::string &p, const float *in, float *out, int N, int cin,
                  int cout, int H) {
  Tensor &w = W(p + ".weight");
  Tensor &b = W(p + ".bias");
  size_t n = (size_t)N * cout * H * H;
  k_conv3<<<grid_for(n), 256>>>(in, w.dev, b.dev, out, N, cin, cout, H, H);
}

static void gn_silu(const std::string &p, float *y, int N, int C, int H) {
  k_gn_silu<<<N * (C / 8), 128>>>(y, W(p + ".weight").dev, W(p + ".bias").dev, C,
                                  C / 8, H * H);
}

// conv -> +t -> GN -> SiLU -> conv -> GN -> SiLU
static float *block(const std::string &p, const float *in, int N, int cin, int cout,
                    int H, const std::vector<float> &temb) {
  size_t n = (size_t)N * cout * H * H;
  float *a = buf(p + ".a", n);
  float *o = buf(p + ".o", n);

  conv3(p + ".conv1", in, a, N, cin, cout, H);
  std::vector<float> tb = dense(W(p + ".proj.weight"), W(p + ".proj.bias"), temb);
  float *dtb = buf(p + ".tb", cout);
  CHECK(cudaMemcpy(dtb, tb.data(), cout * sizeof(float), cudaMemcpyHostToDevice));
  k_add_cbias<<<grid_for(n), 256>>>(a, dtb, N, cout, H * H);
  gn_silu(p + ".gn1", a, N, cout, H);

  conv3(p + ".conv2", a, o, N, cout, cout, H);
  gn_silu(p + ".gn2", o, N, cout, H);
  return o;
}

// Predict the noise in x at timestep t.  Returns a device buffer of N*1*28*28.
static float *unet(const float *x, int t, int N) {
  std::vector<float> temb = time_embed(t);

  float *h1 = block("d1", x, N, 1, 32, 28, temb);
  float *p1 = buf("p1", (size_t)N * 32 * 14 * 14);
  k_avgpool2<<<grid_for((size_t)N * 32 * 196), 256>>>(h1, p1, N, 32, 28, 28);

  float *h2 = block("d2", p1, N, 32, 64, 14, temb);
  float *p2 = buf("p2", (size_t)N * 64 * 7 * 7);
  k_avgpool2<<<grid_for((size_t)N * 64 * 49), 256>>>(h2, p2, N, 64, 14, 14);

  float *hm = block("m", p2, N, 64, 64, 7, temb);

  float *um = buf("um", (size_t)N * 64 * 196);
  k_upsample2<<<grid_for((size_t)N * 64 * 196), 256>>>(hm, um, N, 64, 7, 7);
  float *c2 = buf("c2", (size_t)N * 128 * 196);
  k_concat<<<grid_for((size_t)N * 128 * 196), 256>>>(um, h2, c2, N, 64, 64, 196);

  float *b2 = block("u2", c2, N, 128, 64, 14, temb);

  float *ub = buf("ub", (size_t)N * 64 * 784);
  k_upsample2<<<grid_for((size_t)N * 64 * 784), 256>>>(b2, ub, N, 64, 14, 14);
  float *c1 = buf("c1", (size_t)N * 96 * 784);
  k_concat<<<grid_for((size_t)N * 96 * 784), 256>>>(ub, h1, c1, N, 64, 32, 784);

  float *b1 = block("u1", c1, N, 96, 32, 28, temb);

  float *eps = buf("eps", (size_t)N * 784);
  conv3("out", b1, eps, N, 32, 1, 28);
  return eps;
}

// ---------------------------------------------------------------- PNG out

static unsigned crc32_of(const unsigned char *p, size_t n, unsigned crc) {
  static unsigned tab[256];
  static bool init = false;
  if (!init) {
    for (unsigned i = 0; i < 256; ++i) {
      unsigned c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      tab[i] = c;
    }
    init = true;
  }
  crc = ~crc;
  for (size_t i = 0; i < n; ++i) crc = tab[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

static void put32(std::vector<unsigned char> &v, unsigned x) {
  v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x);
}

static void chunk(FILE *f, const char *tag, const std::vector<unsigned char> &data) {
  std::vector<unsigned char> hdr;
  put32(hdr, (unsigned)data.size());
  fwrite(hdr.data(), 1, 4, f);
  std::vector<unsigned char> body(tag, tag + 4);
  body.insert(body.end(), data.begin(), data.end());
  fwrite(body.data(), 1, body.size(), f);
  std::vector<unsigned char> c;
  put32(c, crc32_of(body.data(), body.size(), 0));
  fwrite(c.data(), 1, 4, f);
}

// 8-bit grayscale PNG, zlib "stored" blocks -- no compression, no libz.
static void write_png(const char *path, const unsigned char *pix, int w, int h) {
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
  const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  fwrite(sig, 1, 8, f);

  std::vector<unsigned char> ihdr;
  put32(ihdr, w); put32(ihdr, h);
  ihdr.push_back(8); ihdr.push_back(0);          // 8-bit, grayscale
  ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
  chunk(f, "IHDR", ihdr);

  std::vector<unsigned char> raw;
  for (int y = 0; y < h; ++y) {
    raw.push_back(0);                            // filter: none
    raw.insert(raw.end(), pix + (size_t)y * w, pix + (size_t)y * w + w);
  }
  std::vector<unsigned char> z{0x78, 0x01};
  for (size_t off = 0; off < raw.size();) {
    size_t n = raw.size() - off < 65535 ? raw.size() - off : 65535;
    bool last = (off + n == raw.size());
    z.push_back(last ? 1 : 0);
    z.push_back(n & 0xFF); z.push_back(n >> 8);
    z.push_back(~n & 0xFF); z.push_back((~n >> 8) & 0xFF);
    z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
    off += n;
  }
  unsigned a = 1, b = 0;
  for (unsigned char c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
  put32(z, (b << 16) | a);
  chunk(f, "IDAT", z);
  chunk(f, "IEND", {});
  fclose(f);
}

// ---------------------------------------------------------------- driver

static std::string dir_of(const std::string &p) {
  size_t s = p.find_last_of('/');
  return s == std::string::npos ? std::string(".") : p.substr(0, s);
}

static std::vector<float> read_bin(const std::string &path, size_t n) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  std::vector<float> v(n);
  if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "%s: short read\n", path.c_str()); exit(1); }
  fclose(f);
  return v;
}

// Single forward pass against PyTorch's output for the same input.
static int run_check(const std::string &out_dir) {
  const int N = 2;
  std::vector<float> x = read_bin(out_dir + "/check_in.bin", N * 784);
  std::vector<float> ref = read_bin(out_dir + "/check_ref.bin", N * 784);
  std::vector<int> t(N);
  FILE *tf = fopen((out_dir + "/check_t.bin").c_str(), "rb");
  if (!tf) { fprintf(stderr, "missing check_t.bin\n"); exit(1); }
  fread(t.data(), 4, N, tf);
  fclose(tf);

  double worst = 0.0;
  for (int i = 0; i < N; ++i) {
    float *dx = buf("check_x", 784);
    CHECK(cudaMemcpy(dx, x.data() + i * 784, 784 * sizeof(float), cudaMemcpyHostToDevice));
    float *eps = unet(dx, t[i], 1);
    std::vector<float> got(784);
    CHECK(cudaMemcpy(got.data(), eps, 784 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    for (int k = 0; k < 784; ++k) {
      double d = fabs((double)got[k] - (double)ref[i * 784 + k]);
      if (d > worst) worst = d;
    }
  }
  printf("max |cumetal - pytorch| over 2 forward passes: %.3e\n", worst);
  if (worst > 2e-3) {
    printf("FAIL: eps mismatch\n");
    return 1;
  }
  printf("PASS: CUDA sampler matches the PyTorch model\n");
  return 0;
}

int main(int argc, char **argv) {
  std::string model = "out/model.bin", out_png;
  int N = 16, seed = 0, zoom = 4;
  bool check = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--check") check = true;
    else if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--out" && i + 1 < argc) out_png = argv[++i];
    else if (a == "--n" && i + 1 < argc) N = atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = atoi(argv[++i]);
    else if (a == "--zoom" && i + 1 < argc) zoom = atoi(argv[++i]);
    else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
  }
  std::string out_dir = dir_of(model);
  if (out_png.empty()) out_png = out_dir + "/samples.png";

  load_model(model.c_str());
  if (check) return run_check(out_dir);

  // Same linear beta schedule as training.
  std::vector<float> beta(kT), alpha(kT), abar(kT);
  for (int i = 0; i < kT; ++i) {
    beta[i] = 1e-4f + (0.02f - 1e-4f) * i / (kT - 1);
    alpha[i] = 1.f - beta[i];
    abar[i] = (i ? abar[i - 1] : 1.f) * alpha[i];
  }

  size_t n = (size_t)N * 784;
  std::mt19937 rng(seed);
  std::normal_distribution<float> gauss(0.f, 1.f);
  std::vector<float> h(n);

  float *x = buf("x", n), *z = buf("z", n);
  for (float &v : h) v = gauss(rng);
  CHECK(cudaMemcpy(x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  printf("sampling %d images, %d denoising steps...\n", N, kT);
  for (int t = kT - 1; t >= 0; --t) {
    float *eps = unet(x, t, N);
    float c1 = 1.f / sqrtf(alpha[t]);
    float c2 = beta[t] / sqrtf(1.f - abar[t]);
    float sigma = t > 0 ? sqrtf(beta[t]) : 0.f;
    if (sigma > 0.f) {
      for (float &v : h) v = gauss(rng);
      CHECK(cudaMemcpy(z, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    k_ddpm_step<<<grid_for(n), 256>>>(x, eps, z, c1, c2, sigma, (int)n);
    if ((kT - t) % 200 == 0) { CHECK(cudaDeviceSynchronize()); printf("  t=%d\n", t); }
  }
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost));

  // Tile the batch into a square-ish grid.
  int cols = 1;
  while (cols * cols < N) ++cols;
  int rows = (N + cols - 1) / cols;
  int cell = 28 * zoom;
  int gw = cols * cell, gh = rows * cell;
  std::vector<unsigned char> img((size_t)gw * gh, 0);
  for (int i = 0; i < N; ++i) {
    int r = i / cols, c = i % cols;
    for (int y = 0; y < cell; ++y)
      for (int xx = 0; xx < cell; ++xx) {
        float v = (h[(size_t)i * 784 + (y / zoom) * 28 + xx / zoom] + 1.f) * 0.5f;
        v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        img[(size_t)(r * cell + y) * gw + c * cell + xx] = (unsigned char)(v * 255.f + 0.5f);
      }
  }
  write_png(out_png.c_str(), img.data(), gw, gh);
  printf("PASS: wrote %s (%dx%d)\n", out_png.c_str(), gw, gh);
  return 0;
}
