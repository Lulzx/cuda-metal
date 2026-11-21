// Reduced forms of the CUDA constructs stwo-zig used to patch or shim for
// CuMetal (#221): funnel shifts (PTX shf.{l,r}.{wrap,clamp}.b32), __brev,
// __match_any_sync on 32- and 64-bit keys, a pointer select between two
// device arrays, atomicExch on shared memory, 64-bit atomicMin, and 64-bit
// scalar kernel arguments (sizes, strides and hashes, not pointers). Rows
// 12-15 issue PTX shf directly in all four forms, since the header funnel
// shifts are portable C++ and never reach the shf lowering; row 16 indexes a
// local constant table, which clang emits as a `__const_$` global array.
#include <cstdint>

constexpr unsigned int kOutputsPerThread = 17;

__device__ __forceinline__ unsigned int ptx_shf_l_wrap(unsigned int lo, unsigned int hi,
                                                       unsigned int n) {
    unsigned int r;
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(n));
    return r;
}
__device__ __forceinline__ unsigned int ptx_shf_r_wrap(unsigned int lo, unsigned int hi,
                                                       unsigned int n) {
    unsigned int r;
    asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(n));
    return r;
}
__device__ __forceinline__ unsigned int ptx_shf_l_clamp(unsigned int lo, unsigned int hi,
                                                        unsigned int n) {
    unsigned int r;
    asm("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(n));
    return r;
}
__device__ __forceinline__ unsigned int ptx_shf_r_clamp(unsigned int lo, unsigned int hi,
                                                        unsigned int n) {
    unsigned int r;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(n));
    return r;
}

extern "C" __global__ void stwo_compat_probe(const unsigned int* words,
                                             const unsigned int* left,
                                             const unsigned int* right,
                                             unsigned int* out,
                                             unsigned long long* minimum,
                                             unsigned long long stride,
                                             unsigned long long hash,
                                             unsigned int count) {
    __shared__ unsigned int exchanged;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) exchanged = 0xffffffffu;
    __syncthreads();
    const unsigned int lane = threadIdx.x & 31u;
    // Every lane of a warp reaches the vote, so the full mask is valid even
    // for the padding threads beyond count.
    const unsigned int index = tid < count ? tid : 0u;
    const unsigned int low = words[index];
    const unsigned int high = words[(index + 1u) % count];
    const unsigned int shift = (low ^ high) % 41u;  // covers 0, 31, 32 and > 32
    const unsigned int key32 = lane % 3u;
    const unsigned long long key64 =
        (static_cast<unsigned long long>(lane % 5u) << 40) | (lane & 1u);
    const unsigned int match32 = __match_any_sync(0xffffffffu, key32);
    const unsigned int match64 = __match_any_sync(0xffffffffu, key64);
    const unsigned int* chosen = (low & 1u) ? left : right;
    const unsigned int previous = atomicExch(&exchanged, tid);
    __syncthreads();
    if (tid >= count) return;
    atomicMin(minimum, (static_cast<unsigned long long>(low) << 32) | high);
    unsigned int* row = out + static_cast<unsigned long long>(tid) * stride;
    row[0] = __funnelshift_l(low, high, shift);
    row[1] = __funnelshift_r(low, high, shift);
    row[2] = __funnelshift_lc(low, high, shift);
    row[3] = __funnelshift_rc(low, high, shift);
    row[4] = __brev(low);
    row[5] = match32;
    row[6] = match64;
    row[7] = chosen[tid];
    row[8] = previous == tid ? 1u : 0u;  // no thread can read its own id back
    row[9] = static_cast<unsigned int>(hash >> 32);
    row[10] = static_cast<unsigned int>(hash);
    row[11] = static_cast<unsigned int>(stride);
    row[12] = ptx_shf_l_wrap(low, high, shift);
    row[13] = ptx_shf_r_wrap(low, high, shift);
    row[14] = ptx_shf_l_clamp(low, high, shift);
    row[15] = ptx_shf_r_clamp(low, high, shift);
    const unsigned int round_constants[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    row[16] = round_constants[(low >> 3) & 7u];
}
