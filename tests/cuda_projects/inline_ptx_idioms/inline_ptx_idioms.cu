// Inline PTX the way real CUDA code writes it: half conversions, approximate
// division, wide and high multiplies, an immediate operand, a tied operand, a
// special register, and a predicate declared inside the block. On the direct
// `.cu` path the NVVM importer binds the operands and runs the template through
// the PTX instruction lowering; on the PTX path the same text is simply part
// of the PTX. Both must agree with the host oracle.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

__device__ __forceinline__ unsigned short float_to_half_bits(float x) {
    unsigned short h;
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(h) : "f"(x));
    return h;
}

__device__ __forceinline__ float half_bits_to_float(unsigned short h) {
    float f;
    asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ float approx_div(float a, float b) {
    float r;
    asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ unsigned mul_hi_u32(unsigned a, unsigned b) {
    unsigned r;
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ long long mul_wide_s32(int a, int b) {
    long long r;
    asm("mul.wide.s32 %0, %1, %2;" : "=l"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ unsigned shift_left_3(unsigned a) {
    unsigned r;
    asm("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(a), "n"(3));
    return r;
}

__device__ __forceinline__ unsigned add_tied(unsigned a, unsigned b) {
    unsigned r = a;
    asm("add.u32 %0, %0, %1;" : "+r"(r) : "r"(b));
    return r;
}

__device__ __forceinline__ unsigned select_if_less(unsigned a, unsigned b) {
    unsigned r;
    asm("{\n\t.reg .pred p;\n\tsetp.lt.u32 p, %1, %2;\n\tselp.u32 %0, %1, %2, p;\n}\n"
        : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ unsigned lane_id() {
    unsigned r;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(r));
    return r;
}

struct Result {
    unsigned half_bits;
    float half_roundtrip;
    float quotient;
    unsigned high;
    long long wide;
    unsigned shifted;
    unsigned tied;
    unsigned selected;
    unsigned lane;
};

__global__ void inline_ptx_probe(Result* output, const float* input, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    const unsigned a = 0x9E3779B9u ^ static_cast<unsigned>(index * 2654435761u);
    const unsigned b = 0x7F4A7C15u + static_cast<unsigned>(index) * 40503u;
    Result r;
    r.half_bits = float_to_half_bits(x);
    r.half_roundtrip = half_bits_to_float(r.half_bits);
    r.quotient = approx_div(x, 3.0f);
    r.high = mul_hi_u32(a, b);
    r.wide = mul_wide_s32(static_cast<int>(a), static_cast<int>(b));
    r.shifted = shift_left_3(a);
    r.tied = add_tied(a, b);
    r.selected = select_if_less(a, b);
    r.lane = lane_id();
    output[index] = r;
}

// Round-to-nearest-even float -> IEEE half, the conversion cvt.rn.f16.f32
// performs.
static unsigned short host_float_to_half(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        return static_cast<unsigned short>(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    }
    if (exponent >= 31) return static_cast<unsigned short>(sign | 0x7C00u);
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<unsigned short>(sign);
        mantissa |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) ++half_mantissa;
        return static_cast<unsigned short>(sign | half_mantissa);
    }
    std::uint32_t half = sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
    const std::uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;
    return static_cast<unsigned short>(half);
}

static float host_half_to_float(unsigned short h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int main() {
    constexpr int kCount = 32;
    float host_input[kCount];
    for (int i = 0; i < kCount; ++i) {
        host_input[i] = (i % 7 == 3) ? -0.375f * i : 1.0f + 0.01f * i * i;
    }
    float* device_input = nullptr;
    Result* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(host_input)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), kCount * sizeof(Result)) != cudaSuccess ||
        cudaMemcpy(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation or upload\n");
        return 1;
    }
    inline_ptx_probe<<<1, 32>>>(device_output, device_input, kCount);
    Result host_output[kCount];
    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess || sync_status != cudaSuccess ||
        cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(launch_status), cudaGetErrorString(sync_status));
        return 1;
    }
    for (int i = 0; i < kCount; ++i) {
        const float x = host_input[i];
        const unsigned a = 0x9E3779B9u ^ static_cast<unsigned>(i * 2654435761u);
        const unsigned b = 0x7F4A7C15u + static_cast<unsigned>(i) * 40503u;
        const Result& r = host_output[i];
        const unsigned short half = host_float_to_half(x);
        if (r.half_bits != half) {
            std::fprintf(stderr, "FAIL: [%d] half bits 0x%x, expected 0x%x\n", i, r.half_bits, half);
            return 1;
        }
        if (r.half_roundtrip != host_half_to_float(half)) {
            std::fprintf(stderr, "FAIL: [%d] half round trip %g, expected %g\n", i,
                         r.half_roundtrip, host_half_to_float(half));
            return 1;
        }
        const float quotient = x / 3.0f;
        if (std::fabs(r.quotient - quotient) > 4.0f * std::fabs(quotient) * 1.1920929e-7f) {
            std::fprintf(stderr, "FAIL: [%d] approx div %g, expected %g\n", i, r.quotient, quotient);
            return 1;
        }
        const unsigned high = static_cast<unsigned>((static_cast<std::uint64_t>(a) * b) >> 32);
        const long long wide = static_cast<long long>(static_cast<int>(a)) * static_cast<int>(b);
        if (r.high != high || r.wide != wide || r.shifted != (a << 3) || r.tied != a + b ||
            r.selected != (a < b ? a : b) || r.lane != static_cast<unsigned>(i % 32)) {
            std::fprintf(stderr,
                         "FAIL: [%d] high %u/%u wide %lld/%lld shifted %u/%u tied %u/%u "
                         "selected %u/%u lane %u/%d\n",
                         i, r.high, high, r.wide, wide, r.shifted, a << 3, r.tied, a + b,
                         r.selected, a < b ? a : b, r.lane, i % 32);
            return 1;
        }
    }
    std::printf("PASS: inline PTX idioms match the host on the device\n");
    return 0;
}
