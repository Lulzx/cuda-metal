struct Descriptor {
    float4* data;
    int size;
};

extern "C" __global__ void byval_aggregate_memcpy(Descriptor source,
                                                    Descriptor destination) {
    const int index = static_cast<int>(threadIdx.x);
    if (index < source.size && index < destination.size) {
        destination.data[index] = source.data[index];
    }
}
