#include <cuda_runtime.h>

#include <cstdio>

__global__ void fill_pitched_rows(float* output, size_t pitch_bytes) {
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    if (x < 4 && y < 4) {
        float* row = reinterpret_cast<float*>(
            reinterpret_cast<unsigned char*>(output) + y * pitch_bytes);
        row[x] = static_cast<float>(y * 4 + x + 1);
    }
}

int main() {
    float* device = nullptr;
    size_t device_pitch = 0;
    if (cudaMallocPitch(reinterpret_cast<void**>(&device), &device_pitch,
                        4 * sizeof(float), 4) != cudaSuccess) {
        return 1;
    }

    fill_pitched_rows<<<1, dim3(4, 4)>>>(device, device_pitch);

    // No explicit synchronization: synchronous cudaMemcpy2D must wait for the
    // producer kernel before its host-side UMA row copies begin.
    float host[4][6]{};
    if (cudaMemcpy2D(host, sizeof(host[0]), device, device_pitch,
                     4 * sizeof(float), 4,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }

    bool ok = true;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            ok = ok && host[y][x] == static_cast<float>(y * 4 + x + 1);
        }
        ok = ok && host[y][4] == 0.0f && host[y][5] == 0.0f;
    }
    const cudaError_t wrong_direction =
        cudaMemcpy2D(host, sizeof(host[0]), host, sizeof(host[0]),
                     4 * sizeof(float), 4, cudaMemcpyDeviceToDevice);
    ok = ok && wrong_direction == cudaErrorInvalidDevicePointer;

    cudaFree(device);
    std::printf("%s: synchronous cudaMemcpy2D observed GPU rows and rejected a wrong direction\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
