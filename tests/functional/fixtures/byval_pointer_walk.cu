// A kernel that walks a by-value aggregate of pointers with a runtime loop.
// Clang keeps the loop (#pragma unroll 1) and emits it as a pointer moving
// through parameter space: `mov.b64 %rd, param`, then `ld.param.b64 [%rd]`
// with `add.s64 %rd, %rd, 8` on the back edge. stwo-zig's Cairo
// memory-execution split kernel has this shape (#221); the typed PTX importer
// used to reject it with "conflicting PTX incoming type" because the loop
// join mixed the aggregate value with the advanced address.
struct Columns {
    unsigned int* values[8];
};

extern "C" __global__ void byval_pointer_walk(const unsigned int* source,
                                              unsigned int rows,
                                              Columns columns) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
#pragma unroll 1
    for (int column = 0; column < 8; ++column) {
        columns.values[column][row] =
            source[row] * static_cast<unsigned int>(column + 1) +
            static_cast<unsigned int>(column);
    }
}
