// Native Metal baseline kernels for cumetal_bench Phase 5 benchmarks.
// Compiled to bench_kernels.metallib via scripts/generate_bench_metallib.sh.

#include <metal_stdlib>
using namespace metal;

// Memory-bound: c[i] = a[i] + b[i]
kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] + b[id];
}

// Memory-bound: y[i] = alpha[0] * x[i] + y[i]
kernel void saxpy(device float* y [[buffer(0)]],
                  device const float* x [[buffer(1)]],
                  device const float* alpha [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    y[id] = alpha[0] * x[id] + y[id];
}

// STREAM copy: one read stream and one write stream.
kernel void copy_f32(device float* output [[buffer(0)]],
                     device const float* input [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    output[id] = input[id];
}

// STREAM triad: output[i] = a[i] + alpha[0] * b[i].
kernel void triad_f32(device float* output [[buffer(0)]],
                      device const float* a [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device const float* alpha [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = a[id] + alpha[0] * b[id];
}

// Memory-bound + shared memory: tree reduction.
// Each threadgroup writes its partial sum to partial_sums[group_id].
kernel void reduce_f32(device const float* input [[buffer(0)]],
                       device float* partial_sums [[buffer(1)]],
                       threadgroup float* scratch [[threadgroup(0)]],
                       uint id [[thread_position_in_threadgroup]],
                       uint group_id [[threadgroup_position_in_grid]],
                       uint group_size [[threads_per_threadgroup]]) {
    scratch[id] = input[group_id * group_size + id];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (id < stride) {
            scratch[id] += scratch[id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (id == 0) {
        partial_sums[group_id] = scratch[0];
    }
}
