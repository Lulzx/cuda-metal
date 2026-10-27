#include <cuda_runtime.h>
extern "C" __global__ void heapk(int* out, unsigned long long* addrs, unsigned block_bytes) {
    if (threadIdx.x != 0) return;
    int* p = (int*)malloc(32);
    if (p == nullptr) { out[0] = -1; return; }
    for (int i = 0; i < 8; ++i) p[i] = i * 3;
    int s = 0;
    for (int i = 0; i < 8; ++i) s += p[i];
    out[0] = s;
    addrs[0] = reinterpret_cast<unsigned long long>(p);
    free(p);
    int* q = (int*)malloc(32);
    if (q == nullptr) { out[1] = -1; return; }
    for (int i = 0; i < 8; ++i) q[i] = i + 100;
    int s2 = 0;
    for (int i = 0; i < 8; ++i) s2 += q[i];
    out[1] = s2;
    addrs[1] = reinterpret_cast<unsigned long long>(q);
    free(q);
    char* big = (char*)malloc(block_bytes);
    addrs[2] = reinterpret_cast<unsigned long long>(big);
    if (big == nullptr) { out[2] = 9; } else { big[0] = 5; out[2] = big[0] + 1; }
}
