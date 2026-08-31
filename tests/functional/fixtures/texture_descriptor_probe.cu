#include <cuda_runtime.h>

#include <cstdio>

__global__ void texture_descriptor_probe(float* output,
                                         cudaTextureObject_t texture,
                                         cudaTextureObject_t linear_clamp,
                                         cudaTextureObject_t linear_border) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const auto* descriptor =
            reinterpret_cast<const __cumetal_texture_descriptor*>(texture);
        output[0] = static_cast<float>(descriptor->width);
        output[1] = static_cast<float>(descriptor->height);
        output[2] = tex2D<float>(texture, 1.5f, 1.5f);
        const float4 fetched = tex1Dfetch<float4>(linear_clamp, 1);
        const float4 clamped = tex1Dfetch<float4>(linear_clamp, -1);
        const float4 bordered = tex1Dfetch<float4>(linear_border, 4);
        output[3] = fetched.x;
        output[4] = fetched.y;
        output[5] = fetched.z;
        output[6] = fetched.w;
        output[7] = clamped.x;
        output[8] = clamped.w;
        output[9] = bordered.x;
        output[10] = bordered.w;
    }
}

int main() {
    float source[16];
    for (int i = 0; i < 16; ++i) source[i] = static_cast<float>(i + 1);

    float* data = nullptr;
    float* output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&data), sizeof(source)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&output), 12 * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(data, source, sizeof(source), cudaMemcpyHostToDevice) != cudaSuccess) {
        return 1;
    }

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypePitch2D;
    resource.res.pitch2D.devPtr = data;
    resource.res.pitch2D.desc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    resource.res.pitch2D.width = 4;
    resource.res.pitch2D.height = 4;
    resource.res.pitch2D.pitchInBytes = 4 * sizeof(float);

    cudaTextureDesc sampler{};
    sampler.addressMode[0] = cudaAddressModeClamp;
    sampler.addressMode[1] = cudaAddressModeClamp;
    sampler.filterMode = cudaFilterModePoint;
    sampler.readMode = cudaReadModeElementType;

    cudaTextureObject_t texture = 0;
    if (cudaCreateTextureObject(&texture, &resource, &sampler, nullptr) != cudaSuccess) {
        return 1;
    }

    cudaResourceDesc linear_resource{};
    linear_resource.resType = cudaResourceTypeLinear;
    linear_resource.res.linear.devPtr = data;
    linear_resource.res.linear.desc =
        cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    linear_resource.res.linear.sizeInBytes = sizeof(source);
    cudaTextureDesc linear_sampler{};
    linear_sampler.addressMode[0] = cudaAddressModeClamp;
    linear_sampler.filterMode = cudaFilterModePoint;
    linear_sampler.readMode = cudaReadModeElementType;
    cudaTextureObject_t linear_clamp = 0;
    cudaTextureObject_t linear_border = 0;
    if (cudaCreateTextureObject(&linear_clamp, &linear_resource,
                                &linear_sampler, nullptr) != cudaSuccess) {
        return 1;
    }
    linear_sampler.addressMode[0] = cudaAddressModeBorder;
    if (cudaCreateTextureObject(&linear_border, &linear_resource,
                                &linear_sampler, nullptr) != cudaSuccess) {
        return 1;
    }

    texture_descriptor_probe<<<1, 1>>>(output, texture, linear_clamp, linear_border);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;

    float result[12]{};
    if (cudaMemcpy(result, output, sizeof(result), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }
    const bool ok = result[0] == 4.0f && result[1] == 4.0f &&
                    result[2] == 6.0f && result[3] == 5.0f &&
                    result[4] == 6.0f && result[5] == 7.0f &&
                    result[6] == 8.0f && result[7] == 1.0f &&
                    result[8] == 4.0f && result[9] == 0.0f &&
                    result[10] == 0.0f;
    std::printf("%s: descriptor=%gx%g tex2D=%g tex1Dfetch={%g,%g,%g,%g} "
                "clamp={%g,%g} border={%g,%g}\n",
                ok ? "PASS" : "FAIL", result[0], result[1], result[2],
                result[3], result[4], result[5], result[6], result[7],
                result[8], result[9], result[10]);
    return ok ? 0 : 1;
}
