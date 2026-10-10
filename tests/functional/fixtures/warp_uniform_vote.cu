#include "cuda_runtime.h"

extern "C" __global__ void warp_uniform_vote(const unsigned int* predicates,
                                              unsigned int* output,
                                              unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int bit = 1u << lane;
    const int predicate = predicates[index] != 0u;
    const unsigned int base = index * 6u;

    output[base] = __nvvm_vote_uni_sync(0xffffffffu, predicate);
    // Excluded callers leave their pre-filled output sentinel untouched.
    if ((0xa5a55a5au & bit) != 0u) {
        output[base + 1u] = __nvvm_vote_uni_sync(0xa5a55a5au, predicate);
    }
    if (lane == 11u) {
        output[base + 2u] = __nvvm_vote_uni_sync(1u << 11u, predicate);
    }

    // Both groups execute the same instruction with their own consistent mask.
    const unsigned int half_mask = lane < 16u ? 0x0000ffffu : 0xffff0000u;
    output[base + 3u] = __nvvm_vote_uni_sync(half_mask, predicate);
    // Clang can fold source !predicate into a separate comparison. Keep the
    // literal negated PTX operand to exercise that distinct importer case.
    unsigned int negated_any, negated_ballot;
    asm volatile("{ .reg .pred p, voted;\n"
                 "setp.ne.u32 p, %2, 0;\n"
                 "vote.sync.any.pred voted, !p, %3;\n"
                 "selp.u32 %0, 1, 0, voted;\n"
                 "vote.sync.ballot.b32 %1, !p, %3;\n"
                 "}"
                 : "=r"(negated_any), "=r"(negated_ballot)
                 : "r"(predicate), "r"(half_mask));
    output[base + 4u] = negated_any;
    output[base + 5u] = negated_ballot;
}
