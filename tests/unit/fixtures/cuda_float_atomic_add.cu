extern "C" __global__ void cuda_float_atomic_add(float* total, const float* values,
                                                 float* old_out, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        old_out[index] = atomicAdd(total, values[index]);
        __fAtomicAdd(total + 1, values[index]);
    }
}
