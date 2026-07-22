#include <vector_types.h>
#include <cuda.h>
#include <sm_35_intrinsics.h>

#include "cuda_device_frontend_config.h"

// Match the opaque declarations used by CUDA consumers that avoid including
// cuda.h themselves. Repeating the real CUDA spelling must be harmless.
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;

struct __builtin_align__(16) CumetalAlignedProbe {
    unsigned long long value;
};

static_assert(sizeof(CUtexObject) == 8, "CUDA texture object ABI");
static_assert(alignof(CumetalAlignedProbe) == 16, "CUDA builtin alignment spelling");

// Keep these helpers externally visible and representative of project CUDA
// device code. CuMetal cannot lower standalone .func bodies yet, so the test
// driver verifies that --cuda-inline-threshold requests viable-call inlining.
__device__ float4 cumetal_frontend_inline_probe(float4 value, unsigned int lane) {
    float components[4] = {value.x, value.y, value.z, value.w};
    for (unsigned int pass = 0; pass != 8; ++pass) {
        const unsigned int current = (lane + pass) & 3u;
        const unsigned int next = (current + 1u) & 3u;
        const float scale = (pass & 1u) ? 0.125f : 0.25f;
        components[current] += components[next] * scale;
        components[next] -= components[current] * (scale * 0.5f);
        if ((lane + pass) & 4u)
            components[current] += static_cast<float>(pass) * 0.03125f;
        else
            components[next] -= static_cast<float>(pass) * 0.015625f;
    }
    value.x = components[0];
    value.y = components[1];
    value.z = components[2];
    value.w = components[3];
    return value;
}

struct FrontendVec3 {
    float x, y, z;
};

struct FrontendTransform {
    FrontendVec3 columns[3];
    FrontendVec3 translation;
};

__device__ FrontendVec3 cumetal_frontend_incident_probe(
    FrontendVec3& face_normal, const FrontendVec3& axis,
    const FrontendTransform& transform, const FrontendVec3& extents,
    unsigned int lane, unsigned int group_start, unsigned int group_mask) {
    float dot = 0.0f;
    float absolute_dot = 0.0f;
    if (lane < 3u) {
        const FrontendVec3 column = transform.columns[lane];
        dot = column.x * axis.x + column.y * axis.y + column.z * axis.z;
        absolute_dot = dot < 0.0f ? -dot : dot;
    }

    const float sign0 = (lane == 0u || lane == 3u) ? -1.0f : 1.0f;
    const float sign1 = (lane & 2u) ? 1.0f : -1.0f;
    const float dot0 = __shfl_sync(group_mask, absolute_dot, group_start);
    const float dot1 = __shfl_sync(group_mask, absolute_dot, group_start + 1u);
    const float dot2 = __shfl_sync(group_mask, absolute_dot, group_start + 2u);
    FrontendVec3 sign;
    if (dot0 >= dot1 && dot0 >= dot2) {
        sign.x = __shfl_sync(group_mask, dot, group_start) > 0.0f ? -1.0f : 1.0f;
        sign.y = sign0;
        sign.z = sign1;
        face_normal = {transform.columns[0].x * sign.x,
                       transform.columns[0].y * sign.x,
                       transform.columns[0].z * sign.x};
    } else if (dot1 >= dot2) {
        sign.y = __shfl_sync(group_mask, dot, group_start + 1u) > 0.0f ? -1.0f : 1.0f;
        sign.x = sign0;
        sign.z = sign1;
        face_normal = {transform.columns[1].x * sign.y,
                       transform.columns[1].y * sign.y,
                       transform.columns[1].z * sign.y};
    } else {
        sign.z = __shfl_sync(group_mask, dot, group_start + 2u) > 0.0f ? -1.0f : 1.0f;
        sign.x = sign0;
        sign.y = sign1;
        face_normal = {transform.columns[2].x * sign.z,
                       transform.columns[2].y * sign.z,
                       transform.columns[2].z * sign.z};
    }

    const FrontendVec3 vertex = {
        extents.x * sign.x, extents.y * sign.y, extents.z * sign.z};
    return {
        transform.columns[0].x * vertex.x + transform.columns[1].x * vertex.y +
            transform.columns[2].x * vertex.z + transform.translation.x,
        transform.columns[0].y * vertex.x + transform.columns[1].y * vertex.y +
            transform.columns[2].y * vertex.z + transform.translation.y,
        transform.columns[0].z * vertex.x + transform.columns[1].z * vertex.y +
            transform.columns[2].z * vertex.z + transform.translation.z};
}

extern "C" __global__ void cuda_device_probe(
    const float4* input, float4* output, unsigned long long* shuffled) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    float4 value = input[index];
    value.x += cumetal_frontend_test_bias();
    value.w = sqrt(value.w * value.w + 1.0f);
    value.z = fabs(value.z);
    value = cumetal_frontend_inline_probe(value, threadIdx.x & 31u);
    const float zero = value.w - value.w;
    const float one = zero + 1.0f;
    const FrontendTransform transform = {
        {{one, zero, zero}, {zero, one, zero}, {zero, zero, one}},
        {value.x, value.y, value.z}};
    FrontendVec3 face_normal;
    const FrontendVec3 incident = cumetal_frontend_incident_probe(
        face_normal, {value.x, value.y, value.z}, transform,
        {1.0f, 1.0f, 1.0f}, threadIdx.x & 31u, 0u, 0xffffffffu);
    value.x = incident.x + face_normal.x;
    value.y = incident.y + face_normal.y;
    value.z = incident.z + face_normal.z;
    output[index] = value;
    const unsigned long long cached = __ldcg(shuffled + index);
    unsigned long long shuffle_input = static_cast<unsigned long long>(index) + cached;
    double shuffle_as_double;
    __builtin_memcpy(&shuffle_as_double, &shuffle_input, sizeof(shuffle_as_double));
    shuffle_as_double = __shfl_sync(0xffffffffu, shuffle_as_double, 0);
    __builtin_memcpy(shuffled + index, &shuffle_as_double, sizeof(shuffle_as_double));
}
