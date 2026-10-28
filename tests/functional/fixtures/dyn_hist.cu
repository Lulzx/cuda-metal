#include <cuda_runtime.h>
__global__ void dyn_hist(const float* in, float* out, int n, int bins) {
    float acc[4];
    acc[0] = 0.0f; acc[1] = 0.0f; acc[2] = 0.0f; acc[3] = 0.0f;
    for (int i = 0; i < n; ++i) {
        int b = static_cast<int>(in[i]) % bins;
        if (b < 0) b += bins;
        acc[b] += in[i];
    }
    out[0] = acc[0]; out[1] = acc[1]; out[2] = acc[2]; out[3] = acc[3];
}
