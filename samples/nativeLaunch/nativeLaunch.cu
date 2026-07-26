// Device code for the native-launch sample. `extern "C"` keeps the symbol name unmangled so the
// host can name it directly in a cumetalKernel_t descriptor.
extern "C" __global__ void vector_add(float* a, float* b, float* c) {
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    c[id] = a[id] + b[id];
}
