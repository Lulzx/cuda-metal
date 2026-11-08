// Texture sampling semantics COLMAP's PatchMatch depends on, checked against
// host-computed CUDA results:
//   * border addressing returns the border colour (0) outside the texture,
//     for unnormalized coordinates too (wrap/mirror need normalized ones);
//   * clamp addressing returns the edge texel;
//   * tex2DLayered honours cudaFilterModeLinear (bilinear within the layer,
//     each tap addressed on its own, so a border tap contributes 0);
//   * a one-layer float array read with tex2D at integer coordinates under
//     point filtering returns exact texels (COLMAP's poses texture).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kW = 8, kH = 6, kL = 3;

template <typename T>
cudaTextureObject_t make_layered(const std::vector<T>& host, int w, int h, int d,
                                 cudaTextureFilterMode filter, cudaTextureReadMode read,
                                 cudaTextureAddressMode address) {
    cudaArray_t array;
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<T>();
    const cudaExtent extent = make_cudaExtent(w, h, d);
    if (cudaMalloc3DArray(&array, &fmt, extent, cudaArrayLayered) != cudaSuccess) return 0;
    cudaMemcpy3DParms copy;
    std::memset(&copy, 0, sizeof(copy));
    copy.extent = extent;
    copy.kind = cudaMemcpyHostToDevice;
    copy.srcPtr = make_cudaPitchedPtr((void*)host.data(), w * sizeof(T), w, h);
    copy.dstArray = array;
    if (cudaMemcpy3D(&copy) != cudaSuccess) return 0;
    cudaResourceDesc resource;
    std::memset(&resource, 0, sizeof(resource));
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;
    cudaTextureDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.addressMode[0] = desc.addressMode[1] = desc.addressMode[2] = address;
    desc.filterMode = filter;
    desc.readMode = read;
    desc.normalizedCoords = false;
    cudaTextureObject_t object = 0;
    if (cudaCreateTextureObject(&object, &resource, &desc, nullptr) != cudaSuccess) return 0;
    return object;
}

struct Query {
    float x, y;
    int layer;
};

__global__ void sample_layered(cudaTextureObject_t texture, const Query* queries, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tex2DLayered<float>(texture, queries[i].x, queries[i].y, queries[i].layer);
}

__global__ void sample_2d(cudaTextureObject_t texture, const Query* queries, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tex2D<float>(texture, queries[i].x, queries[i].y);
}

int failures = 0;

void expect_near(float got, float want, const char* what, int index) {
    if (std::fabs(got - want) > 1e-5f) {
        std::printf("FAIL: %s query %d got %.7f want %.7f\n", what, index, got, want);
        ++failures;
    }
}

std::vector<float> run(bool layered, cudaTextureObject_t texture, const std::vector<Query>& queries) {
    Query* d_queries = nullptr;
    float* d_out = nullptr;
    const int n = static_cast<int>(queries.size());
    std::vector<float> out(n, -1.0f);
    if (cudaMalloc(&d_queries, n * sizeof(Query)) != cudaSuccess || cudaMalloc(&d_out, n * sizeof(float)) != cudaSuccess)
        return out;
    cudaMemcpy(d_queries, queries.data(), n * sizeof(Query), cudaMemcpyHostToDevice);
    if (layered) sample_layered<<<1, 64>>>(texture, d_queries, d_out, n);
    else sample_2d<<<1, 64>>>(texture, d_queries, d_out, n);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL: kernel launch\n");
        ++failures;
    }
    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_queries);
    cudaFree(d_out);
    return out;
}

}  // namespace

int main() {
    std::vector<uint8_t> image(kW * kH * kL);
    for (int i = 0; i < static_cast<int>(image.size()); ++i) image[i] = static_cast<uint8_t>((i * 37 + 11) & 255);
    const auto texel = [&](int x, int y, int l) { return image[(l * kH + y) * kW + x] / 255.0f; };

    // Point filtering, border vs clamp, unnormalized coordinates.
    const std::vector<Query> edge = {{-0.5f, 2.5f, 1}, {kW + 2.5f, 2.5f, 1}, {3.5f, -4.0f, 0}, {3.5f, 2.5f, 2}};
    const auto border = run(true, make_layered(image, kW, kH, kL, cudaFilterModePoint, cudaReadModeNormalizedFloat,
                                               cudaAddressModeBorder), edge);
    expect_near(border[0], 0.0f, "point border left", 0);
    expect_near(border[1], 0.0f, "point border right", 1);
    expect_near(border[2], 0.0f, "point border top", 2);
    expect_near(border[3], texel(3, 2, 2), "point border inside", 3);
    const auto clamp = run(true, make_layered(image, kW, kH, kL, cudaFilterModePoint, cudaReadModeNormalizedFloat,
                                              cudaAddressModeClamp), edge);
    expect_near(clamp[0], texel(0, 2, 1), "point clamp left", 0);
    expect_near(clamp[1], texel(kW - 1, 2, 1), "point clamp right", 1);
    expect_near(clamp[2], texel(3, 0, 0), "point clamp top", 2);

    // Bilinear in a layer. (x, y) = (2.75, 1.5): taps (2,1),(3,1) with fx = 0.25.
    // (0.0, 3.5): taps (-1,3),(0,3) with fx = 0.5; the border tap is 0.
    // (1.5, 0.25): taps rows -1 and 0 with fy = 0.75; the border row is 0.
    const std::vector<Query> linear_queries = {{2.75f, 1.5f, 1}, {0.0f, 3.5f, 2}, {1.5f, 0.25f, 0}};
    const auto linear = run(true, make_layered(image, kW, kH, kL, cudaFilterModeLinear, cudaReadModeNormalizedFloat,
                                               cudaAddressModeBorder), linear_queries);
    expect_near(linear[0], 0.75f * texel(2, 1, 1) + 0.25f * texel(3, 1, 1), "layered bilinear", 0);
    expect_near(linear[1], 0.5f * texel(0, 3, 2), "layered bilinear border tap", 1);
    expect_near(linear[2], 0.75f * texel(1, 0, 0), "layered bilinear border row", 2);

    // A one-layer float array read through tex2D at integer coordinates.
    std::vector<float> poses(43 * 9);
    for (int i = 0; i < static_cast<int>(poses.size()); ++i) poses[i] = 0.25f * i - 7.0f;
    const std::vector<Query> pose_queries = {{0.0f, 0.0f, 0}, {16.0f, 3.0f, 0}, {42.0f, 8.0f, 0}};
    const auto pose = run(false, make_layered(poses, 43, 9, 1, cudaFilterModePoint, cudaReadModeElementType,
                                              cudaAddressModeBorder), pose_queries);
    expect_near(pose[0], poses[0], "poses texel", 0);
    expect_near(pose[1], poses[3 * 43 + 16], "poses texel", 1);
    expect_near(pose[2], poses[8 * 43 + 42], "poses texel", 2);

    if (failures != 0) return 1;
    std::printf("PASS: texture border, clamp and layered bilinear sampling match CUDA\n");
    return 0;
}
