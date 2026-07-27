#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] + b[id];
}

// A single-lane, data-dependent loop gives stream-ordering tests a command
// buffer that remains pending long enough to observe forbidden synchronization.
kernel void spin_store(device uint* output [[buffer(0)]],
                       constant uint& iterations [[buffer(1)]]) {
    uint value = 0x1234567u;
    for (uint i = 0; i < iterations; ++i) {
        value = value * 1664525u + 1013904223u;
    }
    output[0] = value;
}

kernel void marker_store(device uint* output [[buffer(0)]],
                         constant uint& value [[buffer(1)]]) {
    output[0] = value;
}
