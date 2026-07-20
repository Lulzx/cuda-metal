#include <vector_types.h>
#include <vector_functions.h>

#include <cstdio>
#include <cstring>

// Verify struct sizes match expected CUDA ABI sizes.
static_assert(sizeof(char2) == 2,   "char2 size");
static_assert(sizeof(char3) == 3,   "char3 size");
static_assert(sizeof(char4) == 4,   "char4 size");
static_assert(sizeof(short2) == 4,  "short2 size");
static_assert(sizeof(short3) == 6,  "short3 size");
static_assert(sizeof(short4) == 8,  "short4 size");
static_assert(sizeof(int2) == 8,    "int2 size");
static_assert(sizeof(int3) == 12,   "int3 size");
static_assert(sizeof(int4) == 16,   "int4 size");
static_assert(sizeof(uint2) == 8,   "uint2 size");
static_assert(sizeof(uint4) == 16,  "uint4 size");
static_assert(sizeof(float2) == 8,  "float2 size");
static_assert(sizeof(float3) == 12, "float3 size");
static_assert(sizeof(float4) == 16, "float4 size");
static_assert(sizeof(double2) == 16, "double2 size");

int main() {
    // make_int2
    int2 a = make_int2(3, 7);
    if (a.x != 3 || a.y != 7) {
        std::fprintf(stderr, "FAIL: make_int2 field mismatch\n");
        return 1;
    }

    // make_int3
    int3 b = make_int3(-1, 2, -3);
    if (b.x != -1 || b.y != 2 || b.z != -3) {
        std::fprintf(stderr, "FAIL: make_int3 field mismatch\n");
        return 1;
    }

    // make_int4
    int4 c = make_int4(1, 2, 3, 4);
    if (c.x != 1 || c.y != 2 || c.z != 3 || c.w != 4) {
        std::fprintf(stderr, "FAIL: make_int4 field mismatch\n");
        return 1;
    }

    // make_uint2
    uint2 d = make_uint2(10u, 20u);
    if (d.x != 10u || d.y != 20u) {
        std::fprintf(stderr, "FAIL: make_uint2 field mismatch\n");
        return 1;
    }

    // make_uint4
    uint4 e = make_uint4(1u, 2u, 3u, 4u);
    if (e.x != 1u || e.y != 2u || e.z != 3u || e.w != 4u) {
        std::fprintf(stderr, "FAIL: make_uint4 field mismatch\n");
        return 1;
    }

    // make_float2
    float2 f = make_float2(1.5f, -2.5f);
    if (f.x != 1.5f || f.y != -2.5f) {
        std::fprintf(stderr, "FAIL: make_float2 field mismatch\n");
        return 1;
    }

    // make_float3
    float3 g = make_float3(1.0f, 2.0f, 3.0f);
    if (g.x != 1.0f || g.y != 2.0f || g.z != 3.0f) {
        std::fprintf(stderr, "FAIL: make_float3 field mismatch\n");
        return 1;
    }

    // make_float4 (already existed; verify still works)
    float4 h = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    if (h.x != 1.0f || h.y != 2.0f || h.z != 3.0f || h.w != 4.0f) {
        std::fprintf(stderr, "FAIL: make_float4 field mismatch\n");
        return 1;
    }

    // make_double2
    double2 i = make_double2(1.0, -1.0);
    if (i.x != 1.0 || i.y != -1.0) {
        std::fprintf(stderr, "FAIL: make_double2 field mismatch\n");
        return 1;
    }

    // make_char2
    char2 j = make_char2(10, -5);
    if (j.x != 10 || j.y != -5) {
        std::fprintf(stderr, "FAIL: make_char2 field mismatch\n");
        return 1;
    }

    // make_short4
    short4 k = make_short4(1, 2, 3, 4);
    if (k.x != 1 || k.y != 2 || k.z != 3 || k.w != 4) {
        std::fprintf(stderr, "FAIL: make_short4 field mismatch\n");
        return 1;
    }

    // Verify memcpy-safe layout: int4 with aggregate init
    int4 m{};
    m.x = 100; m.y = 200; m.z = 300; m.w = 400;
    int vals[4];
    std::memcpy(vals, &m, sizeof(m));
    if (vals[0] != 100 || vals[1] != 200 || vals[2] != 300 || vals[3] != 400) {
        std::fprintf(stderr, "FAIL: int4 memory layout unexpected\n");
        return 1;
    }

    std::printf("PASS: CUDA vector types have correct fields, sizes, and make_* constructors\n");
    return 0;
}
