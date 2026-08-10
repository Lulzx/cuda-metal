// Host-side preprocess for the CuMetal 3DGS demo.
// Same projection / EWA conic math as the Inria forward preprocess, run on CPU
// so the GPU path can focus on tile binning + alpha-blend renderCUDA.

#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_rasterizer/config.h"

namespace host_preprocess {

inline float3 transform_point4x3(const float3& p, const float* m) {
    return float3{
        m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
        m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
        m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14],
    };
}

inline float4 transform_point4x4(const float3& p, const float* m) {
    return float4{
        m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
        m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
        m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14],
        m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15],
    };
}

inline float ndc2pix(float v, int S) {
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

inline void cov3d_from_scale_rot(const float* scale, float mod, const float* rot, float* cov3D) {
    const float sx = mod * scale[0];
    const float sy = mod * scale[1];
    const float sz = mod * scale[2];
    const float r = rot[0], x = rot[1], y = rot[2], z = rot[3];

    const float R00 = 1.f - 2.f * (y * y + z * z);
    const float R01 = 2.f * (x * y - r * z);
    const float R02 = 2.f * (x * z + r * y);
    const float R10 = 2.f * (x * y + r * z);
    const float R11 = 1.f - 2.f * (x * x + z * z);
    const float R12 = 2.f * (y * z - r * x);
    const float R20 = 2.f * (x * z - r * y);
    const float R21 = 2.f * (y * z + r * x);
    const float R22 = 1.f - 2.f * (x * x + y * y);

    const float M00 = sx * R00, M01 = sx * R01, M02 = sx * R02;
    const float M10 = sy * R10, M11 = sy * R11, M12 = sy * R12;
    const float M20 = sz * R20, M21 = sz * R21, M22 = sz * R22;

    cov3D[0] = M00 * M00 + M10 * M10 + M20 * M20;
    cov3D[1] = M00 * M01 + M10 * M11 + M20 * M21;
    cov3D[2] = M00 * M02 + M10 * M12 + M20 * M22;
    cov3D[3] = M01 * M01 + M11 * M11 + M21 * M21;
    cov3D[4] = M01 * M02 + M11 * M12 + M21 * M22;
    cov3D[5] = M02 * M02 + M12 * M12 + M22 * M22;
}

inline float3 cov2d(const float3& mean, float focal_x, float focal_y,
                    float tan_fovx, float tan_fovy,
                    const float* cov3D, const float* viewmatrix) {
    float3 t = transform_point4x3(mean, viewmatrix);
    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = std::min(limx, std::max(-limx, txtz)) * t.z;
    t.y = std::min(limy, std::max(-limy, tytz)) * t.z;

    const float J00 = focal_x / t.z;
    const float J02 = -(focal_x * t.x) / (t.z * t.z);
    const float J11 = focal_y / t.z;
    const float J12 = -(focal_y * t.y) / (t.z * t.z);

    const float W00 = viewmatrix[0], W01 = viewmatrix[4], W02 = viewmatrix[8];
    const float W10 = viewmatrix[1], W11 = viewmatrix[5], W12 = viewmatrix[9];
    const float W20 = viewmatrix[2], W21 = viewmatrix[6], W22 = viewmatrix[10];
    (void)W02; (void)W12; (void)W22;

    const float T00 = W00 * J00;
    const float T01 = W01 * J11;
    const float T02 = W00 * J02 + W01 * J12;
    const float T10 = W10 * J00;
    const float T11 = W11 * J11;
    const float T12 = W10 * J02 + W11 * J12;
    const float T20 = W20 * J00;
    const float T21 = W21 * J11;
    const float T22 = W20 * J02 + W21 * J12;

    const float V00 = cov3D[0], V01 = cov3D[1], V02 = cov3D[2];
    const float V11 = cov3D[3], V12 = cov3D[4], V22 = cov3D[5];

    const float M00 = V00 * T00 + V01 * T10 + V02 * T20;
    const float M01 = V00 * T01 + V01 * T11 + V02 * T21;
    const float M02 = V00 * T02 + V01 * T12 + V02 * T22;
    const float M10 = V01 * T00 + V11 * T10 + V12 * T20;
    const float M11 = V01 * T01 + V11 * T11 + V12 * T21;
    const float M12 = V01 * T02 + V11 * T12 + V12 * T22;
    const float M20 = V02 * T00 + V12 * T10 + V22 * T20;
    const float M21 = V02 * T01 + V12 * T11 + V22 * T21;
    const float M22 = V02 * T02 + V12 * T12 + V22 * T22;
    (void)M02; (void)M12; (void)M22;

    float c00 = T00 * M00 + T10 * M10 + T20 * M20;
    float c01 = T00 * M01 + T10 * M11 + T20 * M21;
    float c11 = T01 * M01 + T11 * M11 + T21 * M21;
    c00 += 0.3f;
    c11 += 0.3f;
    return float3{c00, c01, c11};
}

inline void get_rect(float2 p, int max_radius, uint32_t grid_x, uint32_t grid_y,
                     uint32_t& rect_min_x, uint32_t& rect_min_y,
                     uint32_t& rect_max_x, uint32_t& rect_max_y) {
    rect_min_x = static_cast<uint32_t>(std::min<int>(grid_x, std::max(0, (int)((p.x - max_radius) / BLOCK_X))));
    rect_min_y = static_cast<uint32_t>(std::min<int>(grid_y, std::max(0, (int)((p.y - max_radius) / BLOCK_Y))));
    rect_max_x = static_cast<uint32_t>(std::min<int>(grid_x, std::max(0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))));
    rect_max_y = static_cast<uint32_t>(std::min<int>(grid_y, std::max(0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y))));
}

struct Output {
    std::vector<int> radii;
    std::vector<float2> means2D;
    std::vector<float> depths;
    std::vector<float> cov3D;
    std::vector<float> rgb;
    std::vector<float4> conic_opacity;
    std::vector<uint32_t> tiles_touched;
};

inline Output run(int P, int W, int H,
                  float scale_modifier,
                  float focal_x, float focal_y,
                  float tan_fovx, float tan_fovy,
                  uint32_t grid_x, uint32_t grid_y,
                  const float* means3D,
                  const float* scales,
                  const float* rotations,
                  const float* opacities,
                  const float* colors,
                  const float* viewmatrix,
                  const float* projmatrix) {
    Output o;
    o.radii.assign(P, 0);
    o.means2D.assign(P, float2{0, 0});
    o.depths.assign(P, 0);
    o.cov3D.assign(P * 6, 0);
    o.rgb.assign(P * 3, 0);
    o.conic_opacity.assign(P, float4{0, 0, 0, 0});
    o.tiles_touched.assign(P, 0);

    for (int idx = 0; idx < P; ++idx) {
        float3 p_orig{means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
        float3 p_view = transform_point4x3(p_orig, viewmatrix);
        if (p_view.z <= 0.2f) {
            continue;
        }

        float4 p_hom = transform_point4x4(p_orig, projmatrix);
        float p_w = 1.0f / (p_hom.w + 1e-7f);
        float3 p_proj{p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

        float cov3D_local[6];
        cov3d_from_scale_rot(scales + 3 * idx, scale_modifier, rotations + 4 * idx, cov3D_local);
        for (int i = 0; i < 6; ++i) {
            o.cov3D[idx * 6 + i] = cov3D_local[i];
        }

        float3 cov = cov2d(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D_local, viewmatrix);
        float det = cov.x * cov.z - cov.y * cov.y;
        if (det == 0.0f) {
            continue;
        }
        float det_inv = 1.f / det;
        float3 conic{cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

        float mid = 0.5f * (cov.x + cov.z);
        float lambda1 = mid + std::sqrt(std::max(0.1f, mid * mid - det));
        float lambda2 = mid - std::sqrt(std::max(0.1f, mid * mid - det));
        float my_radius = std::ceil(3.f * std::sqrt(std::max(lambda1, lambda2)));
        float2 point_image{ndc2pix(p_proj.x, W), ndc2pix(p_proj.y, H)};

        uint32_t rminx, rminy, rmaxx, rmaxy;
        get_rect(point_image, (int)my_radius, grid_x, grid_y, rminx, rminy, rmaxx, rmaxy);
        if ((rmaxx - rminx) * (rmaxy - rminy) == 0) {
            continue;
        }

        o.rgb[idx * 3 + 0] = colors[idx * 3 + 0];
        o.rgb[idx * 3 + 1] = colors[idx * 3 + 1];
        o.rgb[idx * 3 + 2] = colors[idx * 3 + 2];
        o.depths[idx] = p_view.z;
        o.radii[idx] = (int)my_radius;
        o.means2D[idx] = point_image;
        o.conic_opacity[idx] = float4{conic.x, conic.y, conic.z, opacities[idx]};
        o.tiles_touched[idx] = (rmaxy - rminy) * (rmaxx - rminx);
    }
    return o;
}

} // namespace host_preprocess
