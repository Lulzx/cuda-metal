// An out-of-line device helper that reads a pose table through tex2D into
// two private arrays and samples a layered depth texture, the shape of
// COLMAP's ComputeGeomConsistencyCost when Clang leaves it out of line. The
// kernel only forwards both texture objects to the helper, so they must still
// be bound as pointers. Each result is checked against the same arithmetic on
// the host.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kPoseWidth = 43;
constexpr int kImages = 3;
constexpr int kDepthW = 16;
constexpr int kDepthH = 12;

template <typename T>
cudaTextureObject_t make_texture(const std::vector<T>& host, int w, int h, int d) {
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
    desc.addressMode[0] = desc.addressMode[1] = desc.addressMode[2] = cudaAddressModeBorder;
    desc.filterMode = cudaFilterModePoint;
    desc.readMode = cudaReadModeElementType;
    desc.normalizedCoords = false;
    cudaTextureObject_t object = 0;
    if (cudaCreateTextureObject(&object, &resource, &desc, nullptr) != cudaSuccess) return 0;
    return object;
}

__device__ __noinline__ float reproject(cudaTextureObject_t poses, cudaTextureObject_t depths, float row,
                                        float col, float depth, int image, float max_cost) {
    float P[12];
    for (int i = 0; i < 12; ++i) P[i] = tex2D<float>(poses, i + 19, image);
    float inv_P[12];
    for (int i = 0; i < 12; ++i) inv_P[i] = tex2D<float>(poses, i + 31, image);
    const float x = col * depth;
    const float y = row * depth;
    const float z = depth;
    const float inv_z = 1.0f / (P[8] * x + P[9] * y + P[10] * z + P[11]);
    float src_col = inv_z * (P[0] * x + P[1] * y + P[2] * z + P[3]);
    float src_row = inv_z * (P[4] * x + P[5] * y + P[6] * z + P[7]);
    const float src_depth = tex2DLayered<float>(depths, src_col + 0.5f, src_row + 0.5f, image);
    if (src_depth == 0.0f) return max_cost;
    src_col *= src_depth;
    src_row *= src_depth;
    const float bx = inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
    const float by = inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
    const float bz = inv_P[8] * src_col + inv_P[9] * src_row + inv_P[10] * src_depth + inv_P[11];
    const float dc = col - bx / bz;
    const float dr = row - by / bz;
    return fminf(max_cost, sqrtf(dc * dc + dr * dr));
}

__global__ void costs(cudaTextureObject_t poses, cudaTextureObject_t depths, float* out) {
    const int image = threadIdx.x % kImages;
    const float row = 2.0f + threadIdx.x / kImages;
    const float col = 3.0f + threadIdx.x / kImages;
    // Called twice with different arguments so the helper stays out of line.
    out[threadIdx.x] = reproject(poses, depths, row, col, 2.0f, image, 3.0f) +
                       10.0f * reproject(poses, depths, row, col, 4.0f, image, 3.0f);
}

float host_reproject(const std::vector<float>& poses, const std::vector<float>& depths, float row, float col,
                     float depth, int image, float max_cost) {
    const float* P = &poses[image * kPoseWidth + 19];
    const float* inv_P = &poses[image * kPoseWidth + 31];
    const float x = col * depth, y = row * depth, z = depth;
    const float inv_z = 1.0f / (P[8] * x + P[9] * y + P[10] * z + P[11]);
    float src_col = inv_z * (P[0] * x + P[1] * y + P[2] * z + P[3]);
    float src_row = inv_z * (P[4] * x + P[5] * y + P[6] * z + P[7]);
    const int ix = static_cast<int>(std::floor(src_col + 0.5f));
    const int iy = static_cast<int>(std::floor(src_row + 0.5f));
    const float src_depth = (ix < 0 || iy < 0 || ix >= kDepthW || iy >= kDepthH)
                                ? 0.0f : depths[(image * kDepthH + iy) * kDepthW + ix];
    if (src_depth == 0.0f) return max_cost;
    src_col *= src_depth;
    src_row *= src_depth;
    const float bx = inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
    const float by = inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
    const float bz = inv_P[8] * src_col + inv_P[9] * src_row + inv_P[10] * src_depth + inv_P[11];
    const float dc = col - bx / bz;
    const float dr = row - by / bz;
    return std::fmin(max_cost, std::sqrt(dc * dc + dr * dr));
}

}  // namespace

int main() {
    // P maps (x, y, z) to (x + shift, y, z): a horizontal shift of `image`
    // pixels; inv_P undoes it with a small per-image error so the cost is
    // neither zero nor the cap.
    std::vector<float> poses(kPoseWidth * kImages, 0.0f);
    for (int image = 0; image < kImages; ++image) {
        float* P = &poses[image * kPoseWidth + 19];
        float* inv_P = &poses[image * kPoseWidth + 31];
        P[0] = 1; P[2] = static_cast<float>(image); P[5] = 1; P[10] = 1;
        inv_P[0] = 1; inv_P[2] = -static_cast<float>(image) + 0.1f * (image + 1); inv_P[5] = 1; inv_P[10] = 1;
        inv_P[7] = 0.05f * image;
    }
    std::vector<float> depths(kDepthW * kDepthH * kImages);
    for (int i = 0; i < static_cast<int>(depths.size()); ++i) depths[i] = 1.0f + 0.01f * (i % 97);
    const cudaTextureObject_t poses_texture = make_texture(poses, kPoseWidth, kImages, 1);
    const cudaTextureObject_t depth_texture = make_texture(depths, kDepthW, kDepthH, kImages);
    if (poses_texture == 0 || depth_texture == 0) {
        std::printf("FAIL: texture setup\n");
        return 1;
    }
    constexpr int kThreads = 30;
    float* d_out = nullptr;
    cudaMalloc(&d_out, kThreads * sizeof(float));
    costs<<<1, kThreads>>>(poses_texture, depth_texture, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL: kernel launch\n");
        return 1;
    }
    std::vector<float> out(kThreads);
    cudaMemcpy(out.data(), d_out, kThreads * sizeof(float), cudaMemcpyDeviceToHost);
    int failures = 0;
    for (int t = 0; t < kThreads; ++t) {
        const int image = t % kImages;
        const float row = 2.0f + t / kImages;
        const float col = 3.0f + t / kImages;
        const float want = host_reproject(poses, depths, row, col, 2.0f, image, 3.0f) +
                           10.0f * host_reproject(poses, depths, row, col, 4.0f, image, 3.0f);
        if (std::fabs(out[t] - want) > 1e-3f * std::fmax(1.0f, std::fabs(want))) {
            std::printf("FAIL: thread %d got %.6f want %.6f\n", t, out[t], want);
            ++failures;
        }
    }
    if (failures != 0) return 1;
    std::printf("PASS: out-of-line texture helper matches the host\n");
    return 0;
}
