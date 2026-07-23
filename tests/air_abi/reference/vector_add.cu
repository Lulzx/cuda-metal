extern "C" __global__ void vector_add(float* a, float* b, float* c) {
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    c[id] = a[id] + b[id];
}
