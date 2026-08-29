#include <cuda_runtime.h>

#include <cstdio>

__global__ void texture_descriptor_probe(float* output,
                                         cudaTextureObject_t texture) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const auto* descriptor =
            reinterpret_cast<const __cumetal_texture_descriptor*>(texture);
        output[0] = static_cast<float>(descriptor->width);
        output[1] = static_cast<float>(descriptor->height);
        output[2] = tex2D<float>(texture, 1.5f, 1.5f);
    }
}

int main() {
    float source[16];
    for (int i = 0; i < 16; ++i) source[i] = static_cast<float>(i + 1);

    float* data = nullptr;
    float* output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&data), sizeof(source)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&output), 4 * sizeof(float)) != cudaSuccess ||
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
    texture_descriptor_probe<<<1, 1>>>(output, texture);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;

    float result[4]{};
    if (cudaMemcpy(result, output, sizeof(result), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }
    const bool ok = result[0] == 4.0f && result[1] == 4.0f &&
                    result[2] == 6.0f;
    std::printf("%s: texture descriptor width=%g height=%g sampled=%g\n",
                ok ? "PASS" : "FAIL", result[0], result[1], result[2]);
    return ok ? 0 : 1;
}
