// A helper that returns one of its reference arguments -- the shape of
// `T& operator+=(T&, const T&)` and every chained setter -- is called with a
// local object at one site and a device object at another. The return then
// carries a different address space at each call site, which used to be
// refused with "unresolved generic pointer return address space". Now each
// address-space clone of the helper returns its own space and each call site
// takes the space of the object it passed.
#include <cuda_runtime.h>

#include <cstdio>

struct Vec3 {
    float x;
    float y;
    float z;
};

__device__ __noinline__ Vec3& scale_in_place(Vec3& value, float factor) {
    value.x *= factor;
    value.y *= factor;
    value.z *= factor;
    return value;
}

__device__ __noinline__ Vec3& shift_x(Vec3& value, float delta) {
    value.x += delta;
    return value;
}

__global__ void ref_return_probe(Vec3* output, float factor) {
    if (threadIdx.x == 0) {
        Vec3 local{1.0f, 2.0f, 3.0f};
        // Chained through the returned reference on a private object.
        Vec3& scaled_local = shift_x(scale_in_place(local, factor), 100.0f);
        // The same helpers on a device object, chained the same way.
        Vec3& scaled_device = shift_x(scale_in_place(output[1], factor), 0.5f);
        output[0].x = scaled_local.x;
        output[0].y = scaled_local.y + scaled_device.y;
        output[0].z = local.z + scaled_device.z;
    }
}

int main() {
    Vec3 host_output[2] = {{0.0f, 0.0f, 0.0f}, {10.0f, 20.0f, 30.0f}};
    const float factor = 2.0f;
    // local = (2, 4, 6) then x += 100 -> (102, 4, 6)
    // output[1] = (20, 40, 60) then x += 0.5 -> (20.5, 40, 60)
    const float expected[2][3] = {{102.0f, 4.0f + 40.0f, 6.0f + 60.0f}, {20.5f, 40.0f, 60.0f}};

    Vec3* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(host_output)) != cudaSuccess ||
        cudaMemcpy(device_output, host_output, sizeof(host_output), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation or upload\n");
        return 1;
    }

    ref_return_probe<<<1, 32>>>(device_output, factor);

    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess || sync_status != cudaSuccess ||
        cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(launch_status), cudaGetErrorString(sync_status));
        return 1;
    }

    const float* got[2] = {&host_output[0].x, &host_output[1].x};
    for (int element = 0; element < 2; ++element) {
        for (int component = 0; component < 3; ++component) {
            if (got[element][component] != expected[element][component]) {
                std::fprintf(stderr, "FAIL: output[%d].%c = %f, expected %f\n", element,
                             "xyz"[component], got[element][component],
                             expected[element][component]);
                return 1;
            }
        }
    }
    std::printf("PASS: reference-returning helpers take each call site's address space\n");
    return 0;
}
