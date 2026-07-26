// Ray Tracing in One Weekend, GPU edition -- a cuda_projects conformance harness.
//
// Why this project: it stresses compiler surface the other harnesses do not.
// Every helper is __host__ __device__, so the identical source runs on the CPU
// and on Metal; the harness renders both and compares pixels. There is no RNG
// state carried across launches (the sampler is a pure hash of pixel/sample/
// bounce), so CPU and GPU must agree to the last ulp of a float8 quantisation.
//
// Specifically exercised:
//   * a deep per-thread stack frame (Ray + Hit + Material by value, 50-deep
//     bounce loop) -- the PTX .local depot path
//   * branchy float math: sqrtf/fabsf/fminf/fmaxf/powf, early-out returns
//   * struct-by-value returns out of non-inlined-in-PTX helpers
//   * integer hashing (mul/shift/xor) interleaved with float work
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            printf("FAIL: %s -> %s\n", #call, cudaGetErrorString(_e));         \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------- vec3

struct Vec3 {
    float x, y, z;
};

__host__ __device__ inline Vec3 mkv(float x, float y, float z) { Vec3 v; v.x = x; v.y = y; v.z = z; return v; }
__host__ __device__ inline Vec3 add(Vec3 a, Vec3 b) { return mkv(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline Vec3 sub(Vec3 a, Vec3 b) { return mkv(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline Vec3 mul(Vec3 a, Vec3 b) { return mkv(a.x * b.x, a.y * b.y, a.z * b.z); }
__host__ __device__ inline Vec3 scale(Vec3 a, float s) { return mkv(a.x * s, a.y * s, a.z * s); }
__host__ __device__ inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline Vec3 cross3(Vec3 a, Vec3 b) {
    return mkv(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ inline float len(Vec3 a) { return sqrtf(dot(a, a)); }
__host__ __device__ inline Vec3 unit(Vec3 a) { return scale(a, 1.0f / len(a)); }
__host__ __device__ inline Vec3 reflect3(Vec3 v, Vec3 n) { return sub(v, scale(n, 2.0f * dot(v, n))); }

__host__ __device__ inline bool refract3(Vec3 v, Vec3 n, float ni_over_nt, Vec3 *out) {
    Vec3 uv = unit(v);
    float dt = dot(uv, n);
    float disc = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (disc <= 0.0f) return false;
    *out = sub(scale(sub(uv, scale(n, dt)), ni_over_nt), scale(n, sqrtf(disc)));
    return true;
}

__host__ __device__ inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

// ------------------------------------------------------- hash sampler
//
// Stateless on purpose: seeded from (pixel, sample, bounce, stream) so the
// sequence is a pure function of position. No curand state to diverge.

__host__ __device__ inline unsigned pcg_hash(unsigned x) {
    x = x * 747796405u + 2891336453u;
    unsigned w = ((x >> ((x >> 28) + 4u)) ^ x) * 277803737u;
    return (w >> 22) ^ w;
}

__host__ __device__ inline float rnd(unsigned *s) {
    *s = pcg_hash(*s);
    return (float)(*s >> 8) * (1.0f / 16777216.0f);
}

__host__ __device__ inline Vec3 rnd_in_sphere(unsigned *s) {
    // Rejection sampling: a data-dependent loop trip count per thread.
    for (int i = 0; i < 32; ++i) {
        Vec3 p = mkv(2.0f * rnd(s) - 1.0f, 2.0f * rnd(s) - 1.0f, 2.0f * rnd(s) - 1.0f);
        if (dot(p, p) < 1.0f) return p;
    }
    return mkv(0.0f, 0.0f, 0.0f);
}

// ------------------------------------------------------------- scene

#define MAT_LAMBERT 0
#define MAT_METAL   1
#define MAT_GLASS   2

struct Sphere {
    Vec3 center;
    float radius;
    int mat;
    Vec3 albedo;
    float fuzz;      // metal
    float ref_idx;   // glass
};

#define NUM_SPHERES 5
#define MAX_BOUNCE  50

__host__ __device__ inline void build_scene(Sphere *s) {
    s[0].center = mkv(0.0f, -100.5f, -1.0f); s[0].radius = 100.0f;
    s[0].mat = MAT_LAMBERT; s[0].albedo = mkv(0.8f, 0.8f, 0.0f); s[0].fuzz = 0.0f; s[0].ref_idx = 1.0f;

    s[1].center = mkv(0.0f, 0.0f, -1.0f); s[1].radius = 0.5f;
    s[1].mat = MAT_LAMBERT; s[1].albedo = mkv(0.1f, 0.2f, 0.5f); s[1].fuzz = 0.0f; s[1].ref_idx = 1.0f;

    s[2].center = mkv(1.0f, 0.0f, -1.0f); s[2].radius = 0.5f;
    s[2].mat = MAT_METAL; s[2].albedo = mkv(0.8f, 0.6f, 0.2f); s[2].fuzz = 0.15f; s[2].ref_idx = 1.0f;

    s[3].center = mkv(-1.0f, 0.0f, -1.0f); s[3].radius = 0.5f;
    s[3].mat = MAT_GLASS; s[3].albedo = mkv(1.0f, 1.0f, 1.0f); s[3].fuzz = 0.0f; s[3].ref_idx = 1.5f;

    s[4].center = mkv(-1.0f, 0.0f, -1.0f); s[4].radius = -0.45f;  // hollow inner shell
    s[4].mat = MAT_GLASS; s[4].albedo = mkv(1.0f, 1.0f, 1.0f); s[4].fuzz = 0.0f; s[4].ref_idx = 1.5f;
}

struct Hit {
    float t;
    Vec3 p;
    Vec3 n;
    int idx;
};

__host__ __device__ inline bool hit_sphere(const Sphere *s, Vec3 ro, Vec3 rd, float tmin, float tmax, Hit *h) {
    Vec3 oc = sub(ro, s->center);
    float a = dot(rd, rd);
    float b = dot(oc, rd);
    float c = dot(oc, oc) - s->radius * s->radius;
    float disc = b * b - a * c;
    if (disc <= 0.0f) return false;
    float sd = sqrtf(disc);
    float t = (-b - sd) / a;
    if (t <= tmin || t >= tmax) {
        t = (-b + sd) / a;
        if (t <= tmin || t >= tmax) return false;
    }
    h->t = t;
    h->p = add(ro, scale(rd, t));
    h->n = scale(sub(h->p, s->center), 1.0f / s->radius);
    return true;
}

__host__ __device__ inline bool hit_world(const Sphere *sp, Vec3 ro, Vec3 rd, Hit *best) {
    bool any = false;
    float closest = 1e20f;
    for (int i = 0; i < NUM_SPHERES; ++i) {
        Hit tmp;
        if (hit_sphere(&sp[i], ro, rd, 0.001f, closest, &tmp)) {
            any = true;
            closest = tmp.t;
            tmp.idx = i;
            *best = tmp;
        }
    }
    return any;
}

// Iterative path trace. `sp` lives in the caller's frame (a 5 x 48-byte array),
// which together with the per-bounce Hit is what pushes this over the small
// fixed local-frame guesses that used to silently truncate.
__host__ __device__ inline Vec3 trace(Vec3 ro, Vec3 rd, const Sphere *sp, unsigned *seed) {
    Vec3 atten = mkv(1.0f, 1.0f, 1.0f);
    for (int bounce = 0; bounce < MAX_BOUNCE; ++bounce) {
        Hit h;
        if (!hit_world(sp, ro, rd, &h)) {
            Vec3 u = unit(rd);
            float t = 0.5f * (u.y + 1.0f);
            Vec3 sky = add(scale(mkv(1.0f, 1.0f, 1.0f), 1.0f - t), scale(mkv(0.5f, 0.7f, 1.0f), t));
            return mul(atten, sky);
        }
        const Sphere *s = &sp[h.idx];
        if (s->mat == MAT_LAMBERT) {
            Vec3 target = add(add(h.p, h.n), rnd_in_sphere(seed));
            ro = h.p;
            rd = sub(target, h.p);
            atten = mul(atten, s->albedo);
        } else if (s->mat == MAT_METAL) {
            Vec3 refl = reflect3(unit(rd), h.n);
            Vec3 dir = add(refl, scale(rnd_in_sphere(seed), s->fuzz));
            if (dot(dir, h.n) <= 0.0f) return mkv(0.0f, 0.0f, 0.0f);
            ro = h.p;
            rd = dir;
            atten = mul(atten, s->albedo);
        } else {
            Vec3 outward;
            float ni_over_nt;
            float cosine;
            float dn = dot(rd, h.n);
            if (dn > 0.0f) {
                outward = scale(h.n, -1.0f);
                ni_over_nt = s->ref_idx;
                cosine = s->ref_idx * dn / len(rd);
            } else {
                outward = h.n;
                ni_over_nt = 1.0f / s->ref_idx;
                cosine = -dn / len(rd);
            }
            Vec3 refracted;
            float reflect_prob = refract3(rd, outward, ni_over_nt, &refracted)
                                     ? schlick(cosine, s->ref_idx)
                                     : 1.0f;
            ro = h.p;
            rd = (rnd(seed) < reflect_prob) ? reflect3(rd, h.n) : refracted;
        }
    }
    return mkv(0.0f, 0.0f, 0.0f);
}

// ------------------------------------------------------------ camera

__host__ __device__ inline void camera_ray(int px, int py, float u, float v, int w, int h,
                                           Vec3 *ro, Vec3 *rd) {
    Vec3 lookfrom = mkv(3.0f, 1.4f, 2.0f);
    Vec3 lookat = mkv(0.0f, 0.0f, -1.0f);
    Vec3 vup = mkv(0.0f, 1.0f, 0.0f);
    float theta = 30.0f * 3.14159265358979f / 180.0f;
    float half_h = tanf(theta * 0.5f);
    float aspect = (float)w / (float)h;
    float half_w = aspect * half_h;

    Vec3 wv = unit(sub(lookfrom, lookat));
    Vec3 uv = unit(cross3(vup, wv));
    Vec3 vv = cross3(wv, uv);

    Vec3 lower_left = sub(sub(sub(lookfrom, scale(uv, half_w)), scale(vv, half_h)), wv);
    Vec3 horiz = scale(uv, 2.0f * half_w);
    Vec3 vert = scale(vv, 2.0f * half_h);

    float s = ((float)px + u) / (float)w;
    float t = ((float)py + v) / (float)h;
    *ro = lookfrom;
    *rd = sub(add(add(lower_left, scale(horiz, s)), scale(vert, t)), lookfrom);
}

// One pixel, shared by CPU reference and GPU kernel. Byte-identical source.
__host__ __device__ inline Vec3 render_pixel(int px, int py, int w, int h, int spp) {
    Sphere sp[NUM_SPHERES];
    build_scene(sp);

    Vec3 acc = mkv(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < spp; ++s) {
        unsigned seed = pcg_hash((unsigned)(py * w + px) * 9781u + (unsigned)s * 6271u + 1u);
        float ju = rnd(&seed);
        float jv = rnd(&seed);
        Vec3 ro, rd;
        camera_ray(px, py, ju, jv, w, h, &ro, &rd);
        acc = add(acc, trace(ro, rd, sp, &seed));
    }
    float inv = 1.0f / (float)spp;
    return mkv(sqrtf(acc.x * inv), sqrtf(acc.y * inv), sqrtf(acc.z * inv));
}

__global__ void render_kernel(float *out, int w, int h, int spp) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    Vec3 c = render_pixel(px, py, w, h, spp);
    int i = (py * w + px) * 3;
    out[i + 0] = c.x;
    out[i + 1] = c.y;
    out[i + 2] = c.z;
}

// -------------------------------------------------------------- main

int main(int argc, char **argv) {
    int w = 160, h = 120, spp = 8;
    if (argc > 1) w = atoi(argv[1]);
    if (argc > 2) h = atoi(argv[2]);
    if (argc > 3) spp = atoi(argv[3]);

    const size_t n = (size_t)w * h * 3;
    const size_t bytes = n * sizeof(float);
    printf("rtiow: %dx%d, %d spp, %d bounces max\n", w, h, spp, MAX_BOUNCE);

    float *d_out = nullptr;
    CHECK(cudaMalloc((void **)&d_out, bytes));

    dim3 block(8, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    render_kernel<<<grid, block>>>(d_out, w, h, spp);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu(n);
    CHECK(cudaMemcpy(gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_out));

    // CPU reference: same inline functions, same order of operations.
    std::vector<float> ref(n);
    for (int py = 0; py < h; ++py) {
        for (int px = 0; px < w; ++px) {
            Vec3 c = render_pixel(px, py, w, h, spp);
            int i = (py * w + px) * 3;
            ref[i + 0] = c.x;
            ref[i + 1] = c.y;
            ref[i + 2] = c.z;
        }
    }

    // A blank / constant image would sail through a pure diff check, so assert
    // the reference itself has real dynamic range before trusting the compare.
    float rmin = 1e30f, rmax = -1e30f, rsum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        rmin = fminf(rmin, ref[i]);
        rmax = fmaxf(rmax, ref[i]);
        rsum += ref[i];
    }
    printf("reference range: min=%.4f max=%.4f mean=%.4f\n", rmin, rmax, rsum / (float)n);
    if (rmax - rmin < 0.25f) {
        printf("FAIL: CPU reference image is degenerate (range %.4f); harness cannot validate.\n",
               rmax - rmin);
        return 1;
    }

    double sse = 0.0;
    float maxdiff = 0.0f;
    size_t worst = 0, outliers = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(gpu[i] - ref[i]);
        sse += (double)d * d;
        if (d > maxdiff) { maxdiff = d; worst = i; }
        if (d > 0.02f) ++outliers;  // a visible, more-than-quantum deviation
    }
    double rmse = sqrt(sse / (double)n);
    double psnr = (rmse > 0.0) ? 20.0 * log10(1.0 / rmse) : 99.0;
    double outlier_frac = (double)outliers / (double)n;

    printf("max abs diff: %.6g at pixel (%zu,%zu) ch %zu (gpu=%.6f ref=%.6f)\n",
           maxdiff, (worst / 3) % (size_t)w, (worst / 3) / (size_t)w, worst % 3,
           gpu[worst], ref[worst]);
    printf("rmse: %.6g   psnr: %.2f dB   outliers(>0.02): %zu / %zu (%.4f%%)\n",
           rmse, psnr, outliers, n, 100.0 * outlier_frac);

    // Tolerance rationale. Both sides run the same source with the same
    // deterministic sampler, so ordinary divergence is fused-multiply-add and
    // transcendental ulp noise, far below a display quantum. But a path tracer
    // also branches on float compares (Schlick reflect-vs-refract, the
    // rejection-sampling `dot(p,p) < 1`), so a single ulp can flip one ray onto
    // a completely different bounce chain. Those isolated pixels are physics,
    // not a lowering bug, and max-abs-diff cannot tell them apart from one.
    //
    // Gate on the two statistics that CAN: whole-image PSNR, and the fraction
    // of visibly-wrong components. Every failure mode this harness is built to
    // catch -- a truncated local frame, an aliased pointer base, a dropped
    // bounce, a mis-lowered transcendental -- is systematic: it darkens or
    // flattens whole surfaces and drags PSNR into the teens with outliers in
    // the tens of percent. Chaotic ray flips stay in the single digits of
    // pixels. The thresholds sit in the wide gap between the two regimes.
    const double kMinPsnr = 45.0;
    const double kMaxOutlierFrac = 0.001;  // 0.1% of components
    if (psnr < kMinPsnr || outlier_frac > kMaxOutlierFrac) {
        printf("FAIL: GPU render diverges from CPU reference "
               "(psnr %.2f dB < %.2f, or outliers %.4f%% > %.4f%%).\n",
               psnr, kMinPsnr, 100.0 * outlier_frac, 100.0 * kMaxOutlierFrac);
        return 1;
    }

    printf("OK: GPU render matches CPU reference.\n");
    return 0;
}
