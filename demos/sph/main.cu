// 3D SPH dam break, solved and rendered entirely in CUDA, executed on Apple
// Silicon through CuMetal.
//
//   physics : weakly-compressible SPH (Wendland C2 kernel, Tait EOS, Monaghan
//             artificial viscosity, dynamic boundary particles), uniform-grid
//             neighbour search with a GPU counting sort every step.
//   render  : sphere-impostor point splatting with an atomicMin depth pass and
//             a second shading pass, jet colormap on speed.
//
// Everything below that touches particles is a __global__ kernel; the host only
// sets up the scene, drives the loop, and stamps 2D overlay chrome onto the
// finished frame.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#define CK(call)                                                               \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                   __FILE__, __LINE__);                                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static const int kTypeFluid = 0;
static const int kTypeBound = 1;

// ---------------------------------------------------------------------------
// SPH kernels
// ---------------------------------------------------------------------------

// Wendland C2, 3D: W(q) = aD (1 - q/2)^4 (2q + 1), support q < 2, q = r/h.
__device__ __forceinline__ float wendland_w(float q, float aD) {
  float t = 1.0f - 0.5f * q;
  float t2 = t * t;
  return aD * t2 * t2 * (2.0f * q + 1.0f);
}

// dW/dr / r, so that grad_i W_ij = wendland_f(...) * (r_i - r_j).
__device__ __forceinline__ float wendland_f(float q, float aD, float h) {
  float t = 1.0f - 0.5f * q;
  return -5.0f * aD * t * t * t / (h * h);
}

// (rho/rho0)^7 without powf: cheaper and bit-identical to the host reference.
__device__ __forceinline__ float pow7(float t) {
  float t2 = t * t;
  float t4 = t2 * t2;
  return t4 * t2 * t;
}

__device__ __forceinline__ int cell_of(float x, float y, float z, float gminx,
                                       float gminy, float gminz, float invCell,
                                       int gx, int gy, int gz) {
  int cx = (int)floorf((x - gminx) * invCell);
  int cy = (int)floorf((y - gminy) * invCell);
  int cz = (int)floorf((z - gminz) * invCell);
  cx = cx < 0 ? 0 : (cx >= gx ? gx - 1 : cx);
  cy = cy < 0 ? 0 : (cy >= gy ? gy - 1 : cy);
  cz = cz < 0 ? 0 : (cz >= gz ? gz - 1 : cz);
  return (cz * gy + cy) * gx + cx;
}

__global__ void k_fill_uint(unsigned int *dst, unsigned int v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = v;
}

__global__ void k_hash_count(const float4 *pos, int n, float gminx, float gminy,
                             float gminz, float invCell, int gx, int gy, int gz,
                             unsigned int *hash, unsigned int *count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pos[i];
  unsigned int c =
      (unsigned int)cell_of(p.x, p.y, p.z, gminx, gminy, gminz, invCell, gx, gy, gz);
  hash[i] = c;
  atomicAdd(&count[c], 1u);
}

#define SCAN_BLOCK 512

__global__ void k_scan_block(const unsigned int *in, unsigned int *out,
                             unsigned int *blockSum, int n) {
  __shared__ unsigned int s[SCAN_BLOCK];
  int t = threadIdx.x;
  int i = blockIdx.x * SCAN_BLOCK + t;
  unsigned int v = (i < n) ? in[i] : 0u;
  s[t] = v;
  __syncthreads();
  for (int off = 1; off < SCAN_BLOCK; off <<= 1) {
    unsigned int add = (t >= off) ? s[t - off] : 0u;
    __syncthreads();
    s[t] += add;
    __syncthreads();
  }
  if (i < n) out[i] = s[t] - v;  // exclusive
  if (t == SCAN_BLOCK - 1) blockSum[blockIdx.x] = s[t];
}

__global__ void k_scan_blocksums(unsigned int *blockSum, int nb) {
  __shared__ unsigned int s[SCAN_BLOCK];
  int t = threadIdx.x;
  unsigned int v = (t < nb) ? blockSum[t] : 0u;
  s[t] = v;
  __syncthreads();
  for (int off = 1; off < SCAN_BLOCK; off <<= 1) {
    unsigned int add = (t >= off) ? s[t - off] : 0u;
    __syncthreads();
    s[t] += add;
    __syncthreads();
  }
  if (t < nb) blockSum[t] = s[t] - v;
}

__global__ void k_scan_add(unsigned int *out, const unsigned int *blockOff,
                           int n) {
  int i = blockIdx.x * SCAN_BLOCK + threadIdx.x;
  if (i < n) out[i] += blockOff[blockIdx.x];
}

__global__ void k_scatter(const float4 *pos, const float4 *vel,
                          const float *rho, const int *type,
                          const unsigned int *hash, unsigned int *cursor,
                          float4 *posOut, float4 *velOut, float *rhoOut,
                          int *typeOut, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned int dst = atomicAdd(&cursor[hash[i]], 1u);
  posOut[dst] = pos[i];
  velOut[dst] = vel[i];
  rhoOut[dst] = rho[i];
  typeOut[dst] = type[i];
}

// Single neighbour pass: continuity (drho/dt) + momentum (acceleration).
__global__ void k_forces(const float4 *pos, const float4 *vel, const float *rho,
                         const int *type, const unsigned int *cellStart, int n,
                         float gminx, float gminy, float gminz, float invCell,
                         int gx, int gy, int gz, float h, float aD, float mass,
                         float rho0, float B, float c0, float alpha, float delta,
                         float gz_acc, float4 *acc, float *drho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float4 pi = pos[i];
  float4 vi = vel[i];
  float ri = rho[i];
  int ti = type[i];
  float pri = B * (pow7(ri / rho0) - 1.0f);
  if (ti == kTypeBound && pri < 0.0f) pri = 0.0f;
  float pOverRho2_i = pri / (ri * ri);

  int cx = (int)floorf((pi.x - gminx) * invCell);
  int cy = (int)floorf((pi.y - gminy) * invCell);
  int cz = (int)floorf((pi.z - gminz) * invCell);

  float ax = 0.0f, ay = 0.0f, az = 0.0f, dr = 0.0f;
  float h2 = h * h;
  float support2 = 4.0f * h2;

  for (int ddz = -1; ddz <= 1; ++ddz) {
    int zz = cz + ddz;
    if (zz < 0 || zz >= gz) continue;
    for (int ddy = -1; ddy <= 1; ++ddy) {
      int yy = cy + ddy;
      if (yy < 0 || yy >= gy) continue;
      for (int ddx = -1; ddx <= 1; ++ddx) {
        int xx = cx + ddx;
        if (xx < 0 || xx >= gx) continue;
        int c = (zz * gy + yy) * gx + xx;
        unsigned int s = cellStart[c];
        unsigned int e = cellStart[c + 1];
        for (unsigned int j = s; j < e; ++j) {
          if ((int)j == i) continue;
          float4 pj = pos[j];
          float dx = pi.x - pj.x;
          float dy = pi.y - pj.y;
          float dz = pi.z - pj.z;
          float r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= support2 || r2 < 1e-12f) continue;
          float r = sqrtf(r2);
          float q = r / h;
          float f = wendland_f(q, aD, h);  // grad W = f * (dx,dy,dz)

          float4 vj = vel[j];
          float dvx = vi.x - vj.x;
          float dvy = vi.y - vj.y;
          float dvz = vi.z - vj.z;

          // Continuity.
          dr += mass * (dvx * dx + dvy * dy + dvz * dz) * f;

          float rj = rho[j];
          int tj = type[j];

          // delta-SPH density diffusion (Antuono/Marrone), fluid-fluid only.
          if (ti == kTypeFluid && tj == kTypeFluid)
            dr += -delta * h * c0 * 2.0f * (rj - ri) * f * r2 /
                  (r2 + 0.01f * h2) * (mass / rj);

          float prj = B * (pow7(rj / rho0) - 1.0f);
          if (tj == kTypeBound && prj < 0.0f) prj = 0.0f;

          float term = pOverRho2_i + prj / (rj * rj);

          // Monaghan artificial viscosity (approaching pairs only).
          float vr = dvx * dx + dvy * dy + dvz * dz;
          if (vr < 0.0f) {
            float mu = h * vr / (r2 + 0.01f * h2);
            term += -alpha * c0 * mu * 2.0f / (ri + rj);
          }

          float w = -mass * term * f;
          ax += w * dx;
          ay += w * dy;
          az += w * dz;
        }
      }
    }
  }

  acc[i] = make_float4(ax, ay, az + gz_acc, 0.0f);
  drho[i] = dr;
}

__global__ void k_integrate(float4 *pos, float4 *vel, float *rho,
                            const int *type, const float4 *acc,
                            const float *drho, int n, float dt, float rho0,
                            float vmax, float bxlo, float bxhi, float bylo,
                            float byhi, float bzlo, float bzhi) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float r = rho[i] + drho[i] * dt;
  if (type[i] == kTypeBound) {
    if (r < rho0) r = rho0;
  }
  if (r < 0.5f * rho0) r = 0.5f * rho0;
  if (r > 2.0f * rho0) r = 2.0f * rho0;
  rho[i] = r;

  if (type[i] != kTypeFluid) return;

  float4 a = acc[i];
  float4 v = vel[i];
  v.x += a.x * dt;
  v.y += a.y * dt;
  v.z += a.z * dt;
  float sp = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  if (sp > vmax) {
    float s = vmax / sp;
    v.x *= s;
    v.y *= s;
    v.z *= s;
  }
  float4 p = pos[i];
  p.x += v.x * dt;
  p.y += v.y * dt;
  p.z += v.z * dt;
  // Hard containment: the dynamic boundary does the real work, this only stops
  // a stray particle from leaving the search grid.
  if (p.x < bxlo) { p.x = bxlo; v.x = fabsf(v.x) * 0.1f; }
  if (p.x > bxhi) { p.x = bxhi; v.x = -fabsf(v.x) * 0.1f; }
  if (p.y < bylo) { p.y = bylo; v.y = fabsf(v.y) * 0.1f; }
  if (p.y > byhi) { p.y = byhi; v.y = -fabsf(v.y) * 0.1f; }
  if (p.z < bzlo) { p.z = bzlo; v.z = fabsf(v.z) * 0.1f; }
  if (p.z > bzhi) { p.z = bzhi; v.z = -fabsf(v.z) * 0.1f; }
  vel[i] = v;
  pos[i] = p;
}

// Reduction helpers for the run summary / gates. Floats are compared through
// their bit patterns, which is monotonic for the non-negative values here.
__global__ void k_stats(const float4 *vel, const float *rho, const int *type,
                        int n, int *out) {
  // out[0] = max speed bits, out[1] = -(min rho bits), out[2] = max rho bits.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (type[i] != kTypeFluid) return;
  float4 v = vel[i];
  float sp = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  atomicMax(&out[0], __float_as_int(sp));
  int ri = __float_as_int(rho[i]);
  atomicMax(&out[1], -ri);
  atomicMax(&out[2], ri);
}

// ---------------------------------------------------------------------------
// Rasterizer kernels
// ---------------------------------------------------------------------------

__global__ void k_clear(unsigned int *depth, unsigned int *color, int npix,
                        unsigned int bg) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npix) return;
  depth[i] = 0xFFFFFFFFu;
  color[i] = bg;
}

// Camera is passed as a flat float array to keep kernel signatures simple:
//   0..2  eye, 3..5 right, 6..8 up, 9..11 fwd, 12 focal, 13 cx, 14 cy
__device__ __forceinline__ bool project(const float *cam, float4 p, float &sx,
                                        float &sy, float &zv) {
  float dx = p.x - cam[0], dy = p.y - cam[1], dz = p.z - cam[2];
  float xv = dx * cam[3] + dy * cam[4] + dz * cam[5];
  float yv = dx * cam[6] + dy * cam[7] + dz * cam[8];
  zv = dx * cam[9] + dy * cam[10] + dz * cam[11];
  if (zv < 0.05f) return false;
  sx = cam[13] + cam[12] * xv / zv;
  sy = cam[14] - cam[12] * yv / zv;
  return true;
}

__device__ __forceinline__ unsigned int depth_key(float z) {
  return (unsigned int)(z * 1.0e6f);
}

__global__ void k_splat_depth(const float4 *pos, const int *type, int n,
                              const float *cam, unsigned int *depth, int W,
                              int H, float rFluid, float rBound) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sx, sy, zv;
  if (!project(cam, pos[i], sx, sy, zv)) return;
  int bound = (type[i] == kTypeBound);
  float rw = bound ? rBound : rFluid;
  float rp = cam[12] * rw / zv;
  if (rp < 0.6f) rp = 0.6f;
  if (rp > 24.0f) rp = 24.0f;
  int R = (int)ceilf(rp);
  int x0 = (int)sx - R, x1 = (int)sx + R;
  int y0 = (int)sy - R, y1 = (int)sy + R;
  if (x1 < 0 || y1 < 0 || x0 >= W || y0 >= H) return;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 >= W) x1 = W - 1;
  if (y1 >= H) y1 = H - 1;
  float inv = 1.0f / rp;
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      float ox = ((float)x + 0.5f - sx) * inv;
      float oy = ((float)y + 0.5f - sy) * inv;
      float d2 = ox * ox + oy * oy;
      if (d2 > 1.0f) continue;
      float nz = sqrtf(1.0f - d2);
      unsigned int key = depth_key(zv - nz * rw);
      atomicMin(&depth[y * W + x], key);
    }
  }
}

__device__ __forceinline__ unsigned int pack_rgb(float r, float g, float b) {
  int ri = (int)(255.0f * fminf(fmaxf(r, 0.0f), 1.0f) + 0.5f);
  int gi = (int)(255.0f * fminf(fmaxf(g, 0.0f), 1.0f) + 0.5f);
  int bi = (int)(255.0f * fminf(fmaxf(b, 0.0f), 1.0f) + 0.5f);
  return ((unsigned int)ri << 16) | ((unsigned int)gi << 8) | (unsigned int)bi;
}

__device__ __forceinline__ void jet(float t, float &r, float &g, float &b) {
  t = fminf(fmaxf(t, 0.0f), 1.0f);
  r = fminf(fmaxf(1.5f - fabsf(4.0f * t - 3.0f), 0.0f), 1.0f);
  g = fminf(fmaxf(1.5f - fabsf(4.0f * t - 2.0f), 0.0f), 1.0f);
  b = fminf(fmaxf(1.5f - fabsf(4.0f * t - 1.0f), 0.0f), 1.0f);
}

__global__ void k_splat_shade(const float4 *pos, const float4 *vel,
                              const int *type, int n, const float *cam,
                              const unsigned int *depth, unsigned int *color,
                              int W, int H, float rFluid, float rBound,
                              float vScale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sx, sy, zv;
  if (!project(cam, pos[i], sx, sy, zv)) return;
  int bound = (type[i] == kTypeBound);
  float rw = bound ? rBound : rFluid;
  float rp = cam[12] * rw / zv;
  if (rp < 0.6f) rp = 0.6f;
  if (rp > 24.0f) rp = 24.0f;
  int R = (int)ceilf(rp);
  int x0 = (int)sx - R, x1 = (int)sx + R;
  int y0 = (int)sy - R, y1 = (int)sy + R;
  if (x1 < 0 || y1 < 0 || x0 >= W || y0 >= H) return;
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 >= W) x1 = W - 1;
  if (y1 >= H) y1 = H - 1;

  float cr, cg, cb;
  if (bound) {
    cr = 0.11f; cg = 0.17f; cb = 0.22f;
  } else {
    float4 v = vel[i];
    float sp = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    jet(sp / vScale, cr, cg, cb);
  }

  float inv = 1.0f / rp;
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      float ox = ((float)x + 0.5f - sx) * inv;
      float oy = ((float)y + 0.5f - sy) * inv;
      float d2 = ox * ox + oy * oy;
      if (d2 > 1.0f) continue;
      float nz = sqrtf(1.0f - d2);
      unsigned int key = depth_key(zv - nz * rw);
      int pix = y * W + x;
      if (key != depth[pix]) continue;
      // Lambert + rim, light fixed in view space.
      float lam = 0.45f * ox + 0.55f * (-oy) + 0.70f * nz;
      if (lam < 0.0f) lam = 0.0f;
      float sh = bound ? 1.0f : (0.30f + 0.85f * lam);
      float spec = 0.0f;
      if (!bound) {
        float t = lam;
        t = t * t;
        t = t * t;
        spec = 0.35f * t * t;
      }
      color[pix] = pack_rgb(cr * sh + spec, cg * sh + spec, cb * sh + spec);
    }
  }
}

// ---------------------------------------------------------------------------
// Host: scene construction
// ---------------------------------------------------------------------------

struct Scene {
  std::vector<float> px, py, pz;
  std::vector<int> tp;
  int nFluid = 0, nBound = 0;
};

struct Config {
  float dp = 0.021f;
  float tankX = 3.2f, tankY = 1.4f, tankZ = 1.0f;
  float fluidX = 1.20f, fluidZ = 0.55f;
  float obsX0 = 1.80f, obsX1 = 1.95f;
  float obsY0 = 0.40f, obsY1 = 1.00f;
  float obsZ = 0.35f;
  int boundLayers = 3;
  float rho0 = 1000.0f;
  float g = 9.81f;
  float alpha = 0.10f;
  float delta = 0.10f;
  float coefh = 1.3f;
  float dt = 2.5e-5f;
  int stepsPerFrame = 200;
  int frames = 300;
  int W = 1920, H = 1080;
  float vScale = 4.0f;
};

static void add_particle(Scene &s, float x, float y, float z, int t) {
  s.px.push_back(x);
  s.py.push_back(y);
  s.pz.push_back(z);
  s.tp.push_back(t);
}

static void build_scene(const Config &c, Scene &s) {
  const float dp = c.dp;
  // Fluid block at the left end of the tank.
  int nx = (int)(c.fluidX / dp);
  int ny = (int)(c.tankY / dp);
  int nz = (int)(c.fluidZ / dp);
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        add_particle(s, (i + 0.5f) * dp, (j + 0.5f) * dp, (k + 0.5f) * dp,
                     kTypeFluid);
  s.nFluid = (int)s.px.size();

  // Boundary: floor + 4 walls, `boundLayers` layers thick, growing outward.
  int wx = (int)(c.tankX / dp) + 1;
  int wy = (int)(c.tankY / dp) + 1;
  int wz = (int)(c.tankZ / dp) + 1;
  for (int L = 0; L < c.boundLayers; ++L) {
    float o = (L + 0.5f) * dp;  // outward offset
    // Floor (z < 0), extended under the walls.
    for (int j = -c.boundLayers; j < wy + c.boundLayers; ++j)
      for (int i = -c.boundLayers; i < wx + c.boundLayers; ++i)
        add_particle(s, i * dp, j * dp, -o, kTypeBound);
    // x walls.
    for (int k = 0; k < wz; ++k)
      for (int j = -c.boundLayers; j < wy + c.boundLayers; ++j) {
        add_particle(s, -o, j * dp, k * dp, kTypeBound);
        add_particle(s, c.tankX + o, j * dp, k * dp, kTypeBound);
      }
    // y walls.
    for (int k = 0; k < wz; ++k)
      for (int i = 0; i < wx; ++i) {
        add_particle(s, i * dp, -o, k * dp, kTypeBound);
        add_particle(s, i * dp, c.tankY + o, k * dp, kTypeBound);
      }
  }
  // Obstacle block (solid, sampled at dp).
  int ox = (int)((c.obsX1 - c.obsX0) / dp) + 1;
  int oy = (int)((c.obsY1 - c.obsY0) / dp) + 1;
  int oz = (int)(c.obsZ / dp) + 1;
  for (int k = 0; k < oz; ++k)
    for (int j = 0; j < oy; ++j)
      for (int i = 0; i < ox; ++i)
        add_particle(s, c.obsX0 + i * dp, c.obsY0 + j * dp, k * dp, kTypeBound);

  s.nBound = (int)s.px.size() - s.nFluid;
}

// ---------------------------------------------------------------------------
// Host: camera + 2D overlay chrome
// ---------------------------------------------------------------------------

struct Cam {
  float f[15];
};

static void build_camera(const Config &c, float azDeg, float elDeg, float dist,
                         Cam &cam) {
  float tx = c.tankX * 0.5f, ty = c.tankY * 0.5f, tz = c.tankZ * 0.30f;
  float az = azDeg * (float)M_PI / 180.0f;
  float el = elDeg * (float)M_PI / 180.0f;
  float ex = tx + dist * cosf(el) * cosf(az);
  float ey = ty + dist * cosf(el) * sinf(az);
  float ez = tz + dist * sinf(el);
  float fx = tx - ex, fy = ty - ey, fz = tz - ez;
  float fl = sqrtf(fx * fx + fy * fy + fz * fz);
  fx /= fl; fy /= fl; fz /= fl;
  // right = normalize(fwd x worldUp)
  float rx = fy * 1.0f - fz * 0.0f;
  float ry = fz * 0.0f - fx * 1.0f;
  float rz = fx * 0.0f - fy * 0.0f;
  float rl = sqrtf(rx * rx + ry * ry + rz * rz);
  rx /= rl; ry /= rl; rz /= rl;
  // up = right x fwd
  float ux = ry * fz - rz * fy;
  float uy = rz * fx - rx * fz;
  float uz = rx * fy - ry * fx;
  float fovy = 32.0f * (float)M_PI / 180.0f;
  float focal = (0.5f * (float)c.H) / tanf(0.5f * fovy);
  cam.f[0] = ex; cam.f[1] = ey; cam.f[2] = ez;
  cam.f[3] = rx; cam.f[4] = ry; cam.f[5] = rz;
  cam.f[6] = ux; cam.f[7] = uy; cam.f[8] = uz;
  cam.f[9] = fx; cam.f[10] = fy; cam.f[11] = fz;
  cam.f[12] = focal;
  cam.f[13] = 0.5f * (float)c.W;
  cam.f[14] = 0.5f * (float)c.H;
}

// 5x7 bitmap font, columns LSB = top row. Only the glyphs this demo needs.
struct Glyph { char ch; unsigned char col[5]; };
static const Glyph kFont[] = {
  {'0', {0x3E, 0x51, 0x49, 0x45, 0x3E}},
  {'1', {0x00, 0x42, 0x7F, 0x40, 0x00}},
  {'2', {0x42, 0x61, 0x51, 0x49, 0x46}},
  {'3', {0x21, 0x41, 0x45, 0x4B, 0x31}},
  {'4', {0x18, 0x14, 0x12, 0x7F, 0x10}},
  {'5', {0x27, 0x45, 0x45, 0x45, 0x39}},
  {'6', {0x3C, 0x4A, 0x49, 0x49, 0x30}},
  {'7', {0x01, 0x71, 0x09, 0x05, 0x03}},
  {'8', {0x36, 0x49, 0x49, 0x49, 0x36}},
  {'9', {0x06, 0x49, 0x49, 0x29, 0x1E}},
  {'.', {0x00, 0x60, 0x60, 0x00, 0x00}},
  {'=', {0x14, 0x14, 0x14, 0x14, 0x14}},
  {' ', {0x00, 0x00, 0x00, 0x00, 0x00}},
  {'t', {0x04, 0x3F, 0x44, 0x40, 0x20}},
  {'s', {0x48, 0x54, 0x54, 0x54, 0x20}},
  {'p', {0x7C, 0x14, 0x14, 0x14, 0x08}},
  {'e', {0x38, 0x54, 0x54, 0x54, 0x18}},
  {'d', {0x38, 0x44, 0x44, 0x48, 0x7C}},
};

static void put_px(unsigned char *img, int W, int H, int x, int y, float r,
                   float g, float b, float a) {
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  unsigned char *p = img + 3 * (y * W + x);
  p[0] = (unsigned char)(p[0] * (1 - a) + 255.0f * r * a);
  p[1] = (unsigned char)(p[1] * (1 - a) + 255.0f * g * a);
  p[2] = (unsigned char)(p[2] * (1 - a) + 255.0f * b * a);
}

static void draw_text(unsigned char *img, int W, int H, int x, int y,
                      const char *s, int scale, float r, float g, float b,
                      float a) {
  int cursor = x;
  for (const char *c = s; *c; ++c) {
    const Glyph *gl = nullptr;
    for (size_t i = 0; i < sizeof(kFont) / sizeof(kFont[0]); ++i)
      if (kFont[i].ch == *c) { gl = &kFont[i]; break; }
    if (gl) {
      for (int cx = 0; cx < 5; ++cx)
        for (int ry = 0; ry < 7; ++ry)
          if (gl->col[cx] & (1 << ry))
            for (int sy = 0; sy < scale; ++sy)
              for (int sx = 0; sx < scale; ++sx)
                put_px(img, W, H, cursor + cx * scale + sx, y + ry * scale + sy,
                       r, g, b, a);
    }
    cursor += 6 * scale;
  }
}

static int text_width(const char *s, int scale) {
  return (int)std::strlen(s) * 6 * scale;
}

static void draw_line(unsigned char *img, int W, int H, float x0, float y0,
                      float x1, float y1, float r, float g, float b, float a) {
  float dx = x1 - x0, dy = y1 - y0;
  int steps = (int)(fmaxf(fabsf(dx), fabsf(dy))) + 1;
  for (int i = 0; i <= steps; ++i) {
    float t = (float)i / (float)steps;
    put_px(img, W, H, (int)(x0 + dx * t), (int)(y0 + dy * t), r, g, b, a);
  }
}

static bool project_host(const Cam &cam, float x, float y, float z, float &sx,
                         float &sy) {
  const float *c = cam.f;
  float dx = x - c[0], dy = y - c[1], dz = z - c[2];
  float xv = dx * c[3] + dy * c[4] + dz * c[5];
  float yv = dx * c[6] + dy * c[7] + dz * c[8];
  float zv = dx * c[9] + dy * c[10] + dz * c[11];
  if (zv < 0.05f) return false;
  sx = c[13] + c[12] * xv / zv;
  sy = c[14] - c[12] * yv / zv;
  return true;
}

static void jet_host(float t, float &r, float &g, float &b) {
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  r = fminf(fmaxf(1.5f - fabsf(4 * t - 3), 0.0f), 1.0f);
  g = fminf(fmaxf(1.5f - fabsf(4 * t - 2), 0.0f), 1.0f);
  b = fminf(fmaxf(1.5f - fabsf(4 * t - 1), 0.0f), 1.0f);
}

static void draw_overlay(unsigned char *img, const Config &c, const Cam &cam,
                         float simTime) {
  const int W = c.W, H = c.H;
  // Tank wireframe.
  float X = c.tankX, Y = c.tankY, Z = c.tankZ;
  float corner[8][3] = {{0, 0, 0}, {X, 0, 0}, {X, Y, 0}, {0, Y, 0},
                        {0, 0, Z}, {X, 0, Z}, {X, Y, Z}, {0, Y, Z}};
  int edge[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                     {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  float sx[8], sy[8];
  bool ok[8];
  for (int i = 0; i < 8; ++i)
    ok[i] = project_host(cam, corner[i][0], corner[i][1], corner[i][2], sx[i],
                         sy[i]);
  for (int e = 0; e < 12; ++e) {
    int a = edge[e][0], b = edge[e][1];
    if (ok[a] && ok[b])
      draw_line(img, W, H, sx[a], sy[a], sx[b], sy[b], 0.55f, 0.60f, 0.65f,
                0.30f);
  }

  // Colorbar, bottom left.
  int bx = 60, by = H - 110, bw = 240, bh = 12;
  for (int i = 0; i < bw; ++i) {
    float r, g, b;
    jet_host((float)i / (float)(bw - 1), r, g, b);
    for (int j = 0; j < bh; ++j) put_px(img, W, H, bx + i, by + j, r, g, b, 1.0f);
  }
  draw_text(img, W, H, bx + bw / 2 - 15, by - 22, "speed", 2, 0.85f, 0.85f,
            0.85f, 1.0f);
  char lab[32];
  draw_text(img, W, H, bx - 24, by + bh + 6, "0.0", 2, 0.8f, 0.8f, 0.8f, 1.0f);
  std::snprintf(lab, sizeof(lab), "%.1f", c.vScale * 0.5f);
  draw_text(img, W, H, bx + bw / 2 - 16, by + bh + 6, lab, 2, 0.8f, 0.8f, 0.8f,
            1.0f);
  std::snprintf(lab, sizeof(lab), "%.1f", c.vScale);
  draw_text(img, W, H, bx + bw - 12, by + bh + 6, lab, 2, 0.8f, 0.8f, 0.8f, 1.0f);

  // Timer, top right.
  std::snprintf(lab, sizeof(lab), "t = %.2f s", simTime);
  draw_text(img, W, H, W - text_width(lab, 4) - 40, 28, lab, 4, 0.95f, 0.95f,
            0.95f, 1.0f);
}

// ---------------------------------------------------------------------------
// Host reference SPH (brute force) used by --selftest
// ---------------------------------------------------------------------------

struct RefState {
  std::vector<float> px, py, pz, vx, vy, vz, rho;
  std::vector<int> tp;
};

static inline float pow7_host(float t) {
  float t2 = t * t;
  float t4 = t2 * t2;
  return t4 * t2 * t;
}

static void ref_step(RefState &s, const Config &c, float h, float aD, float mass,
                     float B, float c0, float vmax) {
  int n = (int)s.px.size();
  std::vector<float> ax(n, 0.0f), ay(n, 0.0f), az(n, 0.0f), dr(n, 0.0f);
  float h2 = h * h, sup2 = 4.0f * h2;
  for (int i = 0; i < n; ++i) {
    float pri = B * (pow7_host(s.rho[i] / c.rho0) - 1.0f);
    if (s.tp[i] == kTypeBound && pri < 0.0f) pri = 0.0f;
    float pi2 = pri / (s.rho[i] * s.rho[i]);
    float sax = 0, say = 0, saz = 0, sdr = 0;
    for (int j = 0; j < n; ++j) {
      if (j == i) continue;
      float dx = s.px[i] - s.px[j];
      float dy = s.py[i] - s.py[j];
      float dz = s.pz[i] - s.pz[j];
      float r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= sup2 || r2 < 1e-12f) continue;
      float r = sqrtf(r2), q = r / h;
      float t = 1.0f - 0.5f * q;
      float f = -5.0f * aD * t * t * t / (h * h);
      float dvx = s.vx[i] - s.vx[j];
      float dvy = s.vy[i] - s.vy[j];
      float dvz = s.vz[i] - s.vz[j];
      sdr += mass * (dvx * dx + dvy * dy + dvz * dz) * f;
      if (s.tp[i] == kTypeFluid && s.tp[j] == kTypeFluid)
        sdr += -c.delta * h * c0 * 2.0f * (s.rho[j] - s.rho[i]) * f * r2 /
               (r2 + 0.01f * h2) * (mass / s.rho[j]);
      float prj = B * (pow7_host(s.rho[j] / c.rho0) - 1.0f);
      if (s.tp[j] == kTypeBound && prj < 0.0f) prj = 0.0f;
      float term = pi2 + prj / (s.rho[j] * s.rho[j]);
      float vr = dvx * dx + dvy * dy + dvz * dz;
      if (vr < 0.0f) {
        float mu = h * vr / (r2 + 0.01f * h2);
        term += -c.alpha * c0 * mu * 2.0f / (s.rho[i] + s.rho[j]);
      }
      float w = -mass * term * f;
      sax += w * dx; say += w * dy; saz += w * dz;
    }
    ax[i] = sax; ay[i] = say; az[i] = saz - c.g; dr[i] = sdr;
  }
  for (int i = 0; i < n; ++i) {
    float r = s.rho[i] + dr[i] * c.dt;
    if (s.tp[i] == kTypeBound && r < c.rho0) r = c.rho0;
    if (r < 0.5f * c.rho0) r = 0.5f * c.rho0;
    if (r > 2.0f * c.rho0) r = 2.0f * c.rho0;
    s.rho[i] = r;
    if (s.tp[i] != kTypeFluid) continue;
    s.vx[i] += ax[i] * c.dt;
    s.vy[i] += ay[i] * c.dt;
    s.vz[i] += az[i] * c.dt;
    float sp = sqrtf(s.vx[i] * s.vx[i] + s.vy[i] * s.vy[i] + s.vz[i] * s.vz[i]);
    if (sp > vmax) {
      float k = vmax / sp;
      s.vx[i] *= k; s.vy[i] *= k; s.vz[i] *= k;
    }
    s.px[i] += s.vx[i] * c.dt;
    s.py[i] += s.vy[i] * c.dt;
    s.pz[i] += s.vz[i] * c.dt;
  }
}

// ---------------------------------------------------------------------------

int run_selftest(const Config &c, const Scene &scene);

struct Gpu {
  float4 *pos[2] = {nullptr, nullptr};
  float4 *vel[2] = {nullptr, nullptr};
  float *rho[2] = {nullptr, nullptr};
  int *type[2] = {nullptr, nullptr};
  float4 *acc = nullptr;
  float *drho = nullptr;
  unsigned int *hash = nullptr;
  unsigned int *cellCount = nullptr;
  unsigned int *cellStart = nullptr;
  unsigned int *cellCursor = nullptr;
  unsigned int *blockSum = nullptr;
  float *cam = nullptr;
  unsigned int *depth = nullptr;
  unsigned int *color = nullptr;
  int *stats = nullptr;
  int cur = 0;
};

int main(int argc, char **argv) {
  Config c;
  std::string outPath = "demos/sph/out/dambreak.mp4";
  std::string ppmPath;
  bool selftest = false;
  bool noVideo = false;
  float az = -62.0f, el = 20.0f, dist = 5.3f;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : (char *)"0"; };
    if (a == "--dp") c.dp = (float)atof(next());
    else if (a == "--frames") c.frames = atoi(next());
    else if (a == "--steps-per-frame") c.stepsPerFrame = atoi(next());
    else if (a == "--dt") c.dt = (float)atof(next());
    else if (a == "--alpha") c.alpha = (float)atof(next());
    else if (a == "--delta") c.delta = (float)atof(next());
    else if (a == "--vscale") c.vScale = (float)atof(next());
    else if (a == "--coefh") c.coefh = (float)atof(next());
    else if (a == "--width") c.W = atoi(next());
    else if (a == "--height") c.H = atoi(next());
    else if (a == "--out") outPath = next();
    else if (a == "--ppm") ppmPath = next();
    else if (a == "--selftest") selftest = true;
    else if (a == "--no-video") noVideo = true;
    else if (a == "--az") az = (float)atof(next());
    else if (a == "--el") el = (float)atof(next());
    else if (a == "--dist") dist = (float)atof(next());
    else {
      std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
      return 2;
    }
  }

  const float h = c.coefh * c.dp;
  const float aD = 21.0f / (16.0f * (float)M_PI * h * h * h);
  const float mass = c.rho0 * c.dp * c.dp * c.dp;
  const float c0 = 10.0f * sqrtf(2.0f * c.g * c.fluidZ);
  const float B = c0 * c0 * c.rho0 / 7.0f;
  const float vmax = 0.5f * c0;

  if (selftest) {
    // Small, dense, brute-force-comparable configuration.
    Config sc = c;
    sc.dp = 0.05f;
    sc.tankX = 0.6f; sc.tankY = 0.4f; sc.tankZ = 0.4f;
    sc.fluidX = 0.3f; sc.fluidZ = 0.25f;
    sc.obsX0 = 0.45f; sc.obsX1 = 0.50f; sc.obsY0 = 0.1f; sc.obsY1 = 0.3f;
    sc.obsZ = 0.15f;
    sc.boundLayers = 1;
    sc.frames = 0;
    Scene s;
    build_scene(sc, s);
    std::printf("selftest: %d particles (%d fluid, %d boundary)\n",
                (int)s.px.size(), s.nFluid, s.nBound);
    return run_selftest(sc, s);
  }

  Scene scene;
  build_scene(c, scene);
  const int N = (int)scene.px.size();
  std::printf("particles: %d total (%d fluid, %d boundary), dp=%.4f h=%.4f\n", N,
              scene.nFluid, scene.nBound, c.dp, h);

  // Search grid.
  float cell = 2.0f * h;
  float pad = 4.0f * c.dp + cell;
  float gminx = -pad, gminy = -pad, gminz = -pad;
  int gx = (int)ceilf((c.tankX + 2 * pad) / cell);
  int gy = (int)ceilf((c.tankY + 2 * pad) / cell);
  int gz = (int)ceilf((c.tankZ + 2 * pad) / cell);
  int numCells = gx * gy * gz;
  int nScanBlocks = (numCells + 1 + SCAN_BLOCK - 1) / SCAN_BLOCK;
  if (nScanBlocks > SCAN_BLOCK) {
    std::fprintf(stderr, "grid too large for the single-pass scan (%d cells)\n",
                 numCells);
    return 1;
  }
  std::printf("grid: %d x %d x %d = %d cells, cell=%.4f\n", gx, gy, gz, numCells,
              cell);

  Gpu g;
  size_t f4 = sizeof(float4) * (size_t)N;
  for (int b = 0; b < 2; ++b) {
    CK(cudaMalloc(&g.pos[b], f4));
    CK(cudaMalloc(&g.vel[b], f4));
    CK(cudaMalloc(&g.rho[b], sizeof(float) * (size_t)N));
    CK(cudaMalloc(&g.type[b], sizeof(int) * (size_t)N));
  }
  CK(cudaMalloc(&g.acc, f4));
  CK(cudaMalloc(&g.drho, sizeof(float) * (size_t)N));
  CK(cudaMalloc(&g.hash, sizeof(unsigned int) * (size_t)N));
  CK(cudaMalloc(&g.cellCount, sizeof(unsigned int) * (size_t)(numCells + 1)));
  CK(cudaMalloc(&g.cellStart, sizeof(unsigned int) * (size_t)(numCells + 1)));
  CK(cudaMalloc(&g.cellCursor, sizeof(unsigned int) * (size_t)(numCells + 1)));
  CK(cudaMalloc(&g.blockSum, sizeof(unsigned int) * (size_t)SCAN_BLOCK));
  CK(cudaMalloc(&g.cam, sizeof(float) * 15));
  CK(cudaMalloc(&g.stats, sizeof(int) * 4));
  const int npix = c.W * c.H;
  CK(cudaMalloc(&g.depth, sizeof(unsigned int) * (size_t)npix));
  CK(cudaMalloc(&g.color, sizeof(unsigned int) * (size_t)npix));

  {
    std::vector<float4> hp(N), hv(N, make_float4(0, 0, 0, 0));
    std::vector<float> hr(N, c.rho0);
    for (int i = 0; i < N; ++i) {
      hp[i] = make_float4(scene.px[i], scene.py[i], scene.pz[i], 0.0f);
      // Hydrostatic density profile, so the column does not start with a
      // pressure shock it has to radiate away.
      if (scene.tp[i] == kTypeFluid) {
        float depth = c.fluidZ - scene.pz[i];
        if (depth < 0.0f) depth = 0.0f;
        hr[i] = c.rho0 * powf(1.0f + c.rho0 * c.g * depth / B, 1.0f / 7.0f);
      }
    }
    CK(cudaMemcpy(g.pos[0], hp.data(), f4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.vel[0], hv.data(), f4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.rho[0], hr.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.type[0], scene.tp.data(), sizeof(int) * N,
                  cudaMemcpyHostToDevice));
  }

  Cam cam;
  build_camera(c, az, el, dist, cam);
  CK(cudaMemcpy(g.cam, cam.f, sizeof(float) * 15, cudaMemcpyHostToDevice));

  FILE *ff = nullptr;
  if (!noVideo) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt "
                  "rgb24 -s %dx%d -r 60 -i - -c:v libx264 -preset medium -crf 18 "
                  "-pix_fmt yuv420p \"%s\"",
                  c.W, c.H, outPath.c_str());
    ff = popen(cmd, "w");
    if (!ff) {
      std::fprintf(stderr, "failed to start ffmpeg\n");
      return 1;
    }
  }

  std::vector<unsigned int> hostColor(npix);
  std::vector<unsigned char> rgb(3 * (size_t)npix);

  const int TPB = 256;
  const int nb = (N + TPB - 1) / TPB;
  const int nbCells = (numCells + 1 + TPB - 1) / TPB;
  const int nbPix = (npix + TPB - 1) / TPB;

  const int kPrintEvery = getenv("SPH_PRINT_EVERY") ? atoi(getenv("SPH_PRINT_EVERY")) : 20;
  float simTime = 0.0f;
  float maxSpeedAll = 0.0f, minRhoAll = 1e30f, maxRhoAll = 0.0f;
  float lastSpeed = 0.0f, lastRhoMin = 0.0f, lastRhoMax = 0.0f;

  auto tStart = std::chrono::steady_clock::now();

  for (int frame = 0; frame < c.frames; ++frame) {
    for (int step = 0; step < c.stepsPerFrame; ++step) {
      int a = g.cur, b = 1 - g.cur;
      k_fill_uint<<<nbCells, TPB>>>(g.cellCount, 0u, numCells + 1);
      k_hash_count<<<nb, TPB>>>(g.pos[a], N, gminx, gminy, gminz, 1.0f / cell,
                                gx, gy, gz, g.hash, g.cellCount);
      k_scan_block<<<nScanBlocks, SCAN_BLOCK>>>(g.cellCount, g.cellStart,
                                                g.blockSum, numCells + 1);
      k_scan_blocksums<<<1, SCAN_BLOCK>>>(g.blockSum, nScanBlocks);
      k_scan_add<<<nScanBlocks, SCAN_BLOCK>>>(g.cellStart, g.blockSum,
                                              numCells + 1);
      CK(cudaMemcpy(g.cellCursor, g.cellStart,
                    sizeof(unsigned int) * (size_t)(numCells + 1),
                    cudaMemcpyDeviceToDevice));
      k_scatter<<<nb, TPB>>>(g.pos[a], g.vel[a], g.rho[a], g.type[a], g.hash,
                             g.cellCursor, g.pos[b], g.vel[b], g.rho[b],
                             g.type[b], N);
      k_forces<<<nb, TPB>>>(g.pos[b], g.vel[b], g.rho[b], g.type[b], g.cellStart,
                            N, gminx, gminy, gminz, 1.0f / cell, gx, gy, gz, h,
                            aD, mass, c.rho0, B, c0, c.alpha, c.delta, -c.g,
                            g.acc, g.drho);
      k_integrate<<<nb, TPB>>>(g.pos[b], g.vel[b], g.rho[b], g.type[b], g.acc,
                               g.drho, N, c.dt, c.rho0, vmax, gminx + cell,
                               gminx + (gx - 1) * cell, gminy + cell,
                               gminy + (gy - 1) * cell, gminz + cell,
                               gminz + (gz - 1) * cell);
      g.cur = b;
      simTime += c.dt;
    }

    // Per-frame health stats.
    {
      int init[4] = {0, -2147483647, 0, 0};
      CK(cudaMemcpy(g.stats, init, sizeof(int) * 4, cudaMemcpyHostToDevice));
      k_stats<<<nb, TPB>>>(g.vel[g.cur], g.rho[g.cur], g.type[g.cur], N,
                           g.stats);
      int st[4];
      CK(cudaMemcpy(st, g.stats, sizeof(int) * 4, cudaMemcpyDeviceToHost));
      float spMax, rhoMin, rhoMax;
      std::memcpy(&spMax, &st[0], 4);
      int minBits = -st[1];
      std::memcpy(&rhoMin, &minBits, 4);
      std::memcpy(&rhoMax, &st[2], 4);
      lastSpeed = spMax;
      lastRhoMin = rhoMin;
      lastRhoMax = rhoMax;
      if (spMax > maxSpeedAll) maxSpeedAll = spMax;
      if (rhoMin < minRhoAll) minRhoAll = rhoMin;
      if (rhoMax > maxRhoAll) maxRhoAll = rhoMax;
      if (!(spMax == spMax) || spMax > 1e5f) {
        std::fprintf(stderr, "FAIL: simulation diverged at frame %d\n", frame);
        return 1;
      }
    }

    // Render.
    k_clear<<<nbPix, TPB>>>(g.depth, g.color, npix, 0x00000000u);
    k_splat_depth<<<nb, TPB>>>(g.pos[g.cur], g.type[g.cur], N, g.cam, g.depth,
                               c.W, c.H, 0.55f * c.dp, 0.13f * c.dp);
    k_splat_shade<<<nb, TPB>>>(g.pos[g.cur], g.vel[g.cur], g.type[g.cur], N,
                               g.cam, g.depth, g.color, c.W, c.H, 0.55f * c.dp,
                               0.13f * c.dp, c.vScale);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hostColor.data(), g.color,
                  sizeof(unsigned int) * (size_t)npix, cudaMemcpyDeviceToHost));
    for (int i = 0; i < npix; ++i) {
      unsigned int v = hostColor[i];
      rgb[3 * i + 0] = (unsigned char)((v >> 16) & 0xFF);
      rgb[3 * i + 1] = (unsigned char)((v >> 8) & 0xFF);
      rgb[3 * i + 2] = (unsigned char)(v & 0xFF);
    }
    draw_overlay(rgb.data(), c, cam, simTime);

    if (ff) std::fwrite(rgb.data(), 1, rgb.size(), ff);
    if (!ppmPath.empty() && frame == c.frames / 2) {
      FILE *pf = std::fopen(ppmPath.c_str(), "wb");
      if (pf) {
        std::fprintf(pf, "P6\n%d %d\n255\n", c.W, c.H);
        std::fwrite(rgb.data(), 1, rgb.size(), pf);
        std::fclose(pf);
      }
    }
    if ((frame % kPrintEvery) == 0 || frame == c.frames - 1)
      std::printf("frame %3d/%d  t=%.3f s  vmax=%.2f m/s  rho=[%.1f, %.1f]\n",
                  frame + 1, c.frames, simTime, maxSpeedAll, minRhoAll,
                  maxRhoAll);
    std::fflush(stdout);
  }

  CK(cudaDeviceSynchronize());
  float ms = std::chrono::duration<float, std::milli>(
                 std::chrono::steady_clock::now() - tStart)
                 .count();

  if (ff) pclose(ff);

  long long steps = (long long)c.frames * c.stepsPerFrame;
  std::printf("\n--- run summary ---\n");
  std::printf("particles      : %d (%d fluid)\n", N, scene.nFluid);
  std::printf("SPH steps      : %lld  (dt=%.1e, t_end=%.3f s)\n", steps, c.dt,
              simTime);
  std::printf("frames rendered: %d @ %dx%d\n", c.frames, c.W, c.H);
  std::printf("wall time      : %.2f s  (%.2f ms/step)\n", ms / 1000.0f,
              ms / (float)(steps > 0 ? steps : 1));
  std::printf("max fluid speed: %.3f m/s\n", maxSpeedAll);
  std::printf("fluid density  : [%.1f, %.1f] kg/m^3 (rho0=%.0f)\n", minRhoAll,
              maxRhoAll, c.rho0);

  std::printf("final frame   : vmax=%.3f m/s, rho=[%.1f, %.1f]\n", lastSpeed,
              lastRhoMin, lastRhoMax);

  bool ok = true;
  // Physical gates. A dam break of height H drives a front at roughly
  // sqrt(2 g H); WCSPH must stay weakly compressible throughout.
  float expected = sqrtf(2.0f * c.g * c.fluidZ);
  if (!(maxSpeedAll > 0.6f * expected && maxSpeedAll < 3.0f * expected)) {
    std::printf("FAIL: front speed %.3f outside [%.3f, %.3f]\n", maxSpeedAll,
                0.6f * expected, 3.0f * expected);
    ok = false;
  }
  // Whole-run extreme: the single worst particle out of ~1e5 x 6e4 samples,
  // which peaks during the impact on the far wall. Loose bound -- this catches
  // a blow-up, not ordinary impact compression.
  if (!(minRhoAll > 0.80f * c.rho0 && maxRhoAll < 1.20f * c.rho0)) {
    std::printf("FAIL: peak density excursion outside +/-20%% of rho0\n");
    ok = false;
  }
  // The real stability statement: by the end of the run the instantaneous
  // density spread must be small. An unstable WCSPH run cannot satisfy this,
  // because its extremes keep growing instead of relaxing.
  if (!(lastRhoMin > 0.92f * c.rho0 && lastRhoMax < 1.08f * c.rho0)) {
    std::printf("FAIL: final-frame density spread outside +/-8%% of rho0\n");
    ok = false;
  }
  if (ok) std::printf("PASS: SPH dam break simulated and rendered on CuMetal\n");
  return ok ? 0 : 1;
}

// Selftest: run the same GPU pipeline and a host brute-force reference for a
// few steps on a small scene, then compare particle state.
int run_selftest(const Config &c, const Scene &scene) {
  const int N = (int)scene.px.size();
  const float h = c.coefh * c.dp;
  const float aD = 21.0f / (16.0f * (float)M_PI * h * h * h);
  const float mass = c.rho0 * c.dp * c.dp * c.dp;
  const float c0 = 10.0f * sqrtf(2.0f * c.g * c.fluidZ);
  const float B = c0 * c0 * c.rho0 / 7.0f;
  const float vmax = 0.5f * c0;
  const int steps = 20;

  float cell = 2.0f * h;
  float pad = 4.0f * c.dp + cell;
  float gminx = -pad, gminy = -pad, gminz = -pad;
  int gx = (int)ceilf((c.tankX + 2 * pad) / cell);
  int gy = (int)ceilf((c.tankY + 2 * pad) / cell);
  int gz = (int)ceilf((c.tankZ + 2 * pad) / cell);
  int numCells = gx * gy * gz;
  int nScanBlocks = (numCells + 1 + SCAN_BLOCK - 1) / SCAN_BLOCK;

  Gpu g;
  size_t f4 = sizeof(float4) * (size_t)N;
  for (int b = 0; b < 2; ++b) {
    CK(cudaMalloc(&g.pos[b], f4));
    CK(cudaMalloc(&g.vel[b], f4));
    CK(cudaMalloc(&g.rho[b], sizeof(float) * (size_t)N));
    CK(cudaMalloc(&g.type[b], sizeof(int) * (size_t)N));
  }
  CK(cudaMalloc(&g.acc, f4));
  CK(cudaMalloc(&g.drho, sizeof(float) * (size_t)N));
  CK(cudaMalloc(&g.hash, sizeof(unsigned int) * (size_t)N));
  CK(cudaMalloc(&g.cellCount, sizeof(unsigned int) * (size_t)(numCells + 1)));
  CK(cudaMalloc(&g.cellStart, sizeof(unsigned int) * (size_t)(numCells + 1)));
  CK(cudaMalloc(&g.cellCursor, sizeof(unsigned int) * (size_t)(numCells + 1)));
  CK(cudaMalloc(&g.blockSum, sizeof(unsigned int) * (size_t)SCAN_BLOCK));

  std::vector<float4> hp(N), hv(N, make_float4(0, 0, 0, 0));
  std::vector<float> hr(N, c.rho0);
  for (int i = 0; i < N; ++i)
    hp[i] = make_float4(scene.px[i], scene.py[i], scene.pz[i], 0.0f);
  CK(cudaMemcpy(g.pos[0], hp.data(), f4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g.vel[0], hv.data(), f4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g.rho[0], hr.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(g.type[0], scene.tp.data(), sizeof(int) * N,
                cudaMemcpyHostToDevice));

  const int TPB = 256;
  const int nb = (N + TPB - 1) / TPB;
  const int nbCells = (numCells + 1 + TPB - 1) / TPB;

  for (int s = 0; s < steps; ++s) {
    int a = g.cur, b = 1 - g.cur;
    k_fill_uint<<<nbCells, TPB>>>(g.cellCount, 0u, numCells + 1);
    k_hash_count<<<nb, TPB>>>(g.pos[a], N, gminx, gminy, gminz, 1.0f / cell, gx,
                              gy, gz, g.hash, g.cellCount);
    k_scan_block<<<nScanBlocks, SCAN_BLOCK>>>(g.cellCount, g.cellStart,
                                              g.blockSum, numCells + 1);
    k_scan_blocksums<<<1, SCAN_BLOCK>>>(g.blockSum, nScanBlocks);
    k_scan_add<<<nScanBlocks, SCAN_BLOCK>>>(g.cellStart, g.blockSum,
                                            numCells + 1);
    CK(cudaMemcpy(g.cellCursor, g.cellStart,
                  sizeof(unsigned int) * (size_t)(numCells + 1),
                  cudaMemcpyDeviceToDevice));
    k_scatter<<<nb, TPB>>>(g.pos[a], g.vel[a], g.rho[a], g.type[a], g.hash,
                           g.cellCursor, g.pos[b], g.vel[b], g.rho[b],
                           g.type[b], N);
    k_forces<<<nb, TPB>>>(g.pos[b], g.vel[b], g.rho[b], g.type[b], g.cellStart,
                          N, gminx, gminy, gminz, 1.0f / cell, gx, gy, gz, h, aD,
                          mass, c.rho0, B, c0, c.alpha, c.delta, -c.g, g.acc,
                          g.drho);
    k_integrate<<<nb, TPB>>>(g.pos[b], g.vel[b], g.rho[b], g.type[b], g.acc,
                             g.drho, N, c.dt, c.rho0, vmax, gminx + cell,
                             gminx + (gx - 1) * cell, gminy + cell,
                             gminy + (gy - 1) * cell, gminz + cell,
                             gminz + (gz - 1) * cell);
    g.cur = b;
  }
  CK(cudaDeviceSynchronize());

  std::vector<float4> gp(N), gv(N);
  std::vector<float> gr(N);
  std::vector<int> gt(N);
  CK(cudaMemcpy(gp.data(), g.pos[g.cur], f4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gv.data(), g.vel[g.cur], f4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gr.data(), g.rho[g.cur], sizeof(float) * N,
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gt.data(), g.type[g.cur], sizeof(int) * N,
                cudaMemcpyDeviceToHost));

  RefState rs;
  rs.px = scene.px; rs.py = scene.py; rs.pz = scene.pz;
  rs.vx.assign(N, 0.0f); rs.vy.assign(N, 0.0f); rs.vz.assign(N, 0.0f);
  rs.rho.assign(N, c.rho0);
  rs.tp = scene.tp;
  for (int s = 0; s < steps; ++s) ref_step(rs, c, h, aD, mass, B, c0, vmax);

  // GPU output is permuted by the counting sort; match by nearest initial
  // position is unnecessary because the sort is a pure permutation of a
  // deterministic scene -- instead compare sorted scalar spectra plus a
  // position-keyed lookup on the reference.
  double maxPosErr = 0.0, maxVelErr = 0.0, maxRhoErr = 0.0;
  // Build a spatial hash of reference particles keyed by their *initial*
  // position, which the GPU also carries implicitly: re-derive by matching
  // each GPU particle to the reference particle with the closest state.
  // Displacements over 20 steps are far below dp, so nearest-initial-position
  // matching is unambiguous.
  std::vector<int> bucketStart, bucketItems;
  {
    float bcell = c.dp;
    int bx = (int)ceilf((c.tankX + 2 * pad) / bcell) + 2;
    int by = (int)ceilf((c.tankY + 2 * pad) / bcell) + 2;
    int bz = (int)ceilf((c.tankZ + 2 * pad) / bcell) + 2;
    int nb2 = bx * by * bz;
    std::vector<int> cnt(nb2 + 1, 0);
    auto bidx = [&](float x, float y, float z) {
      int ix = (int)floorf((x + pad) / bcell);
      int iy = (int)floorf((y + pad) / bcell);
      int iz = (int)floorf((z + pad) / bcell);
      ix = ix < 0 ? 0 : (ix >= bx ? bx - 1 : ix);
      iy = iy < 0 ? 0 : (iy >= by ? by - 1 : iy);
      iz = iz < 0 ? 0 : (iz >= bz ? bz - 1 : iz);
      return (iz * by + iy) * bx + ix;
    };
    for (int i = 0; i < N; ++i) cnt[bidx(scene.px[i], scene.py[i], scene.pz[i])]++;
    bucketStart.assign(nb2 + 1, 0);
    for (int i = 0; i < nb2; ++i) bucketStart[i + 1] = bucketStart[i] + cnt[i];
    bucketItems.assign(N, 0);
    std::vector<int> cur = bucketStart;
    for (int i = 0; i < N; ++i)
      bucketItems[cur[bidx(scene.px[i], scene.py[i], scene.pz[i])]++] = i;

    int matched = 0;
    for (int i = 0; i < N; ++i) {
      // Match on the reference's *current* position (both moved identically).
      int best = -1;
      float bestd = 1e30f;
      int ix = (int)floorf((gp[i].x + pad) / bcell);
      int iy = (int)floorf((gp[i].y + pad) / bcell);
      int iz = (int)floorf((gp[i].z + pad) / bcell);
      for (int dz2 = -1; dz2 <= 1; ++dz2)
        for (int dy2 = -1; dy2 <= 1; ++dy2)
          for (int dx2 = -1; dx2 <= 1; ++dx2) {
            int jx = ix + dx2, jy = iy + dy2, jz = iz + dz2;
            if (jx < 0 || jy < 0 || jz < 0 || jx >= bx || jy >= by || jz >= bz)
              continue;
            int cidx = (jz * by + jy) * bx + jx;
            for (int t = bucketStart[cidx]; t < bucketStart[cidx + 1]; ++t) {
              int j = bucketItems[t];
              float ddx = gp[i].x - rs.px[j];
              float ddy = gp[i].y - rs.py[j];
              float ddz = gp[i].z - rs.pz[j];
              float d = ddx * ddx + ddy * ddy + ddz * ddz;
              if (d < bestd) { bestd = d; best = j; }
            }
          }
      if (best < 0) continue;
      matched++;
      double pe = sqrt((double)bestd);
      double ve = sqrt((double)((gv[i].x - rs.vx[best]) * (gv[i].x - rs.vx[best]) +
                                (gv[i].y - rs.vy[best]) * (gv[i].y - rs.vy[best]) +
                                (gv[i].z - rs.vz[best]) * (gv[i].z - rs.vz[best])));
      double re = fabs((double)(gr[i] - rs.rho[best])) / c.rho0;
      if (pe > maxPosErr) maxPosErr = pe;
      if (ve > maxVelErr) maxVelErr = ve;
      if (re > maxRhoErr) maxRhoErr = re;
    }
    std::printf("selftest: matched %d/%d particles\n", matched, N);
  }

  std::printf("selftest after %d steps vs host brute-force reference:\n", steps);
  std::printf("  max |dx|   = %.3e m   (dp = %.3f m)\n", maxPosErr, c.dp);
  std::printf("  max |dv|   = %.3e m/s\n", maxVelErr);
  std::printf("  max drho/rho0 = %.3e\n", maxRhoErr);
  bool ok = (maxPosErr < 1e-6) && (maxVelErr < 1e-3) && (maxRhoErr < 1e-4);
  std::printf(ok ? "PASS: GPU SPH matches host reference\n"
                 : "FAIL: GPU SPH diverges from host reference\n");
  return ok ? 0 : 1;
}
