// A kernel whose first parameter is a by-value aggregate, the shape NVIDIA
// Warp's generated kernels always take: every one of them begins with a
// `launch_bounds_t` holding the grid shape and total thread count. Clang lowers
// such a parameter to `ptr byval(%T)`, so the CUDA caller passes the address of
// its own copy of the struct rather than a device pointer, and the launch has
// to bind the struct's bytes.
#include <cstddef>

struct LaunchBounds {
    int shape[4];
    int ndim;
    size_t size;
};

extern "C" __global__ void byval_aggregate_launch(LaunchBounds bounds,
                                                  float scale,
                                                  float* out) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < bounds.size) {
        out[index] = scale * static_cast<float>(bounds.ndim) +
                     static_cast<float>(bounds.shape[3]);
    }
}
