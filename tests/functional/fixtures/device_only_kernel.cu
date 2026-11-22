// A kernel only the device pass sees has no host launch stub, so cumetalc
// builds no metallib for it. That is valid, but it must be said (#223).
#include <cstdio>

#ifdef __CUDA_ARCH__
__global__ void device_only(float* out) { out[0] = 1.0f; }
#endif

int main() {
    std::printf("PASS: host program without launchable kernels\n");
    return 0;
}
