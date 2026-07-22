extern "C" __global__ void switch_normalization(const int* input, int* output) {
    const int index = static_cast<int>(threadIdx.x);
    switch (input[index]) {
        case 0: output[index] = 11; break;
        case 3: output[index] = 29; break;
        default: output[index] = -7; break;
    }
}
