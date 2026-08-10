// CuMetal demo host for 3D Gaussian Splatting forward rasterization.
//
// Device kernels are the Inria GRAPHDECO differential Gaussian rasterizer
// (graphdeco-inria/diff-gaussian-rasterization), vendored under vendor/ and
// trimmed to the forward path only. This host replaces the PyTorch extension
// entry points with a plain CUDA Runtime API harness so the kernels can be
// compiled and launched by CuMetal on Apple Silicon.
//
// Original rasterizer license: non-commercial research (see vendor LICENSE notes
// and upstream LICENSE.md). Demo code in this file is Apache-2.0 with the rest
// of CuMetal.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << what << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

// Growable device buffer used the same way as the torch Tensor resizer in the
// upstream PyTorch extension.
struct DeviceScratch {
    char* ptr = nullptr;
    size_t capacity = 0;

    ~DeviceScratch() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    char* ensure(size_t n) {
        if (n <= capacity) {
            return ptr;
        }
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
        check(cudaMalloc(reinterpret_cast<void**>(&ptr), n), "cudaMalloc scratch");
        capacity = n;
        return ptr;
    }
};

// Column-major 4x4 view for the 3DGS convention: camera-space +Z is in front
// of the camera (p_view.z > 0.2 is required by in_frustum). Place the camera
// at world (0,0,-eye_z) looking toward +Z so the origin sits at z = +eye_z.
void make_lookat_view(float* m, float eye_z) {
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = 1.f;
    m[5] = 1.f;
    m[10] = 1.f;
    m[15] = 1.f;
    m[14] = eye_z; // world origin -> camera (0,0,+eye_z)
}

// OpenGL-style perspective projection, column-major, matches common 3DGS usage.
void make_perspective(float* m, float fov_y_deg, float aspect, float znear, float zfar) {
    const float f = 1.f / std::tan(fov_y_deg * 0.5f * static_cast<float>(M_PI) / 180.f);
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (zfar + znear) / (znear - zfar);
    m[11] = -1.f;
    m[14] = (2.f * zfar * znear) / (znear - zfar);
}

bool write_ppm(const char* path, int W, int H, const float* out_chw, const float* bg) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f << "P6\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix = y * W + x;
            for (int c = 0; c < 3; ++c) {
                float v = out_chw[c * H * W + pix];
                if (!std::isfinite(v)) {
                    v = bg[c];
                }
                v = std::min(1.f, std::max(0.f, v));
                const unsigned char b = static_cast<unsigned char>(v * 255.f + 0.5f);
                f.put(static_cast<char>(b));
            }
        }
    }
    return true;
}

// Count pixels that differ from background by more than eps in any channel.
int count_non_background(int W, int H, const float* out_chw, const float* bg, float eps) {
    int n = 0;
    for (int i = 0; i < W * H; ++i) {
        for (int c = 0; c < 3; ++c) {
            if (std::fabs(out_chw[c * H * W + i] - bg[c]) > eps) {
                ++n;
                break;
            }
        }
    }
    return n;
}

float mean_abs_diff(int N, const float* a, const float* b) {
    double acc = 0.0;
    for (int i = 0; i < N; ++i) {
        acc += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return static_cast<float>(acc / std::max(1, N));
}

} // namespace

int main(int argc, char** argv) {
    const char* out_ppm = "out/gaussians.ppm";
    int W = 128;
    int H = 128;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) {
            out_ppm = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            W = H = std::atoi(argv[++i]);
        }
    }

    // Synthetic scene: three bright anisotropic Gaussians in front of the camera.
    // Precomputed RGB (no SH) keeps the first gate focused on the tile rasterizer.
    constexpr int P = 3;
    constexpr int D = 0; // SH degree unused when colors are precomputed
    constexpr int M = 0;

    float means_h[P * 3] = {
        -0.4f,  0.0f, 0.0f,
         0.0f,  0.2f, 0.0f,
         0.4f, -0.1f, 0.0f,
    };
    float colors_h[P * 3] = {
        1.0f, 0.15f, 0.15f,
        0.15f, 1.0f, 0.15f,
        0.2f, 0.35f, 1.0f,
    };
    float opacities_h[P] = { 0.95f, 0.95f, 0.95f };
    float scales_h[P * 3] = {
        0.18f, 0.12f, 0.10f,
        0.14f, 0.18f, 0.10f,
        0.16f, 0.14f, 0.10f,
    };
    // Identity quaternion (w, x, y, z) as used by the rasterizer.
    float rots_h[P * 4] = {
        1.f, 0.f, 0.f, 0.f,
        1.f, 0.f, 0.f, 0.f,
        1.f, 0.f, 0.f, 0.f,
    };
    float bg_h[3] = { 0.05f, 0.05f, 0.08f };

    float view_h[16];
    float proj_h[16];
    make_lookat_view(view_h, /*eye_z=*/2.5f);
    make_perspective(proj_h, /*fov_y_deg=*/50.f, static_cast<float>(W) / H, 0.1f, 100.f);
    // Full transform used as projmatrix in the original pipeline is often
    // proj * view. Compose here (column-major).
    float full_proj_h[16];
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            full_proj_h[c * 4 + r] =
                proj_h[0 * 4 + r] * view_h[c * 4 + 0] +
                proj_h[1 * 4 + r] * view_h[c * 4 + 1] +
                proj_h[2 * 4 + r] * view_h[c * 4 + 2] +
                proj_h[3 * 4 + r] * view_h[c * 4 + 3];
        }
    }
    float cam_pos_h[3] = { 0.f, 0.f, -2.5f };
    const float tan_fovy = std::tan(50.f * 0.5f * static_cast<float>(M_PI) / 180.f);
    const float tan_fovx = tan_fovy * static_cast<float>(W) / H;

    float *d_means = nullptr, *d_colors = nullptr, *d_opac = nullptr;
    float *d_scales = nullptr, *d_rots = nullptr, *d_view = nullptr, *d_proj = nullptr;
    float *d_cam = nullptr, *d_bg = nullptr, *d_out = nullptr;
    int* d_radii = nullptr;

    check(cudaMalloc(&d_means, P * 3 * sizeof(float)), "means");
    check(cudaMalloc(&d_colors, P * 3 * sizeof(float)), "colors");
    check(cudaMalloc(&d_opac, P * sizeof(float)), "opac");
    check(cudaMalloc(&d_scales, P * 3 * sizeof(float)), "scales");
    check(cudaMalloc(&d_rots, P * 4 * sizeof(float)), "rots");
    check(cudaMalloc(&d_view, 16 * sizeof(float)), "view");
    check(cudaMalloc(&d_proj, 16 * sizeof(float)), "proj");
    check(cudaMalloc(&d_cam, 3 * sizeof(float)), "cam");
    check(cudaMalloc(&d_bg, 3 * sizeof(float)), "bg");
    check(cudaMalloc(&d_out, 3 * W * H * sizeof(float)), "out");
    check(cudaMalloc(&d_radii, P * sizeof(int)), "radii");

    check(cudaMemcpy(d_means, means_h, sizeof(means_h), cudaMemcpyHostToDevice), "H2D means");
    check(cudaMemcpy(d_colors, colors_h, sizeof(colors_h), cudaMemcpyHostToDevice), "H2D colors");
    check(cudaMemcpy(d_opac, opacities_h, sizeof(opacities_h), cudaMemcpyHostToDevice), "H2D opac");
    check(cudaMemcpy(d_scales, scales_h, sizeof(scales_h), cudaMemcpyHostToDevice), "H2D scales");
    check(cudaMemcpy(d_rots, rots_h, sizeof(rots_h), cudaMemcpyHostToDevice), "H2D rots");
    check(cudaMemcpy(d_view, view_h, sizeof(view_h), cudaMemcpyHostToDevice), "H2D view");
    check(cudaMemcpy(d_proj, full_proj_h, sizeof(full_proj_h), cudaMemcpyHostToDevice), "H2D proj");
    check(cudaMemcpy(d_cam, cam_pos_h, sizeof(cam_pos_h), cudaMemcpyHostToDevice), "H2D cam");
    check(cudaMemcpy(d_bg, bg_h, sizeof(bg_h), cudaMemcpyHostToDevice), "H2D bg");
    check(cudaMemset(d_out, 0, 3 * W * H * sizeof(float)), "memset out");
    check(cudaMemset(d_radii, 0, P * sizeof(int)), "memset radii");

    DeviceScratch geom, binning, image;
    auto geomFunc = [&](size_t n) { return geom.ensure(n); };
    auto binningFunc = [&](size_t n) { return binning.ensure(n); };
    auto imageFunc = [&](size_t n) { return image.ensure(n); };

    std::cout << "3D Gaussian Splatting forward (Inria CUDA kernels on CuMetal)\n";
    std::cout << "  resolution: " << W << "x" << H << "  gaussians: " << P << "\n";

    int rendered = 0;
    try {
        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binningFunc,
            imageFunc,
            P, D, M,
            d_bg,
            W, H,
            d_means,
            /*shs=*/nullptr,
            d_colors,
            d_opac,
            d_scales,
            /*scale_modifier=*/1.f,
            d_rots,
            /*cov3D_precomp=*/nullptr,
            d_view,
            d_proj,
            d_cam,
            tan_fovx, tan_fovy,
            /*prefiltered=*/false,
            d_out,
            d_radii,
            /*debug=*/true);
    } catch (const std::exception& ex) {
        std::cerr << "Rasterizer::forward threw: " << ex.what() << "\n";
        return 1;
    }
    check(cudaDeviceSynchronize(), "sync after forward");

    std::vector<float> out_h(static_cast<size_t>(3 * W * H));
    std::vector<int> radii_h(P);
    check(cudaMemcpy(out_h.data(), d_out, out_h.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "D2H out");
    check(cudaMemcpy(radii_h.data(), d_radii, P * sizeof(int), cudaMemcpyDeviceToHost), "D2H radii");

    // Rerun for determinism check (same inputs → same image).
    check(cudaMemset(d_out, 0, 3 * W * H * sizeof(float)), "memset out2");
    int rendered2 = CudaRasterizer::Rasterizer::forward(
        geomFunc, binningFunc, imageFunc,
        P, D, M, d_bg, W, H, d_means, nullptr, d_colors, d_opac, d_scales, 1.f, d_rots,
        nullptr, d_view, d_proj, d_cam, tan_fovx, tan_fovy, false, d_out, d_radii, true);
    check(cudaDeviceSynchronize(), "sync after forward2");
    std::vector<float> out2_h(out_h.size());
    check(cudaMemcpy(out2_h.data(), d_out, out2_h.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "D2H out2");

    const int painted = count_non_background(W, H, out_h.data(), bg_h, 0.02f);
    const float rerun_mad = mean_abs_diff(static_cast<int>(out_h.size()), out_h.data(), out2_h.data());

    float maxv = 0.f;
    for (float v : out_h) {
        maxv = std::max(maxv, v);
    }

    std::cout << "  rendered instances: " << rendered << " (rerun " << rendered2 << ")\n";
    std::cout << "  radii:";
    for (int r : radii_h) {
        std::cout << " " << r;
    }
    std::cout << "\n";
    std::cout << "  non-background pixels: " << painted << " / " << (W * H) << "\n";
    std::cout << "  max channel: " << maxv << "\n";
    std::cout << "  rerun mean abs diff: " << rerun_mad << "\n";

    if (!write_ppm(out_ppm, W, H, out_h.data(), bg_h)) {
        std::cerr << "failed to write " << out_ppm << "\n";
        return 1;
    }
    std::cout << "  wrote " << out_ppm << "\n";

    bool ok = true;
    if (rendered <= 0) {
        std::cerr << "FAIL: no Gaussian tile instances were rendered\n";
        ok = false;
    }
    if (painted < 20) {
        std::cerr << "FAIL: image is essentially background (painted=" << painted << ")\n";
        ok = false;
    }
    if (maxv < 0.2f) {
        std::cerr << "FAIL: image peak too low (max=" << maxv << ")\n";
        ok = false;
    }
    if (rerun_mad > 1e-5f) {
        std::cerr << "FAIL: non-deterministic rerun (mad=" << rerun_mad << ")\n";
        ok = false;
    }
    for (int r : radii_h) {
        if (r <= 0) {
            std::cerr << "FAIL: a Gaussian has non-positive screen radius\n";
            ok = false;
            break;
        }
    }

    cudaFree(d_means);
    cudaFree(d_colors);
    cudaFree(d_opac);
    cudaFree(d_scales);
    cudaFree(d_rots);
    cudaFree(d_view);
    cudaFree(d_proj);
    cudaFree(d_cam);
    cudaFree(d_bg);
    cudaFree(d_out);
    cudaFree(d_radii);

    if (!ok) {
        return 1;
    }
    std::cout << "PASS: 3D Gaussian Splatting forward rendered on CuMetal\n";
    return 0;
}
