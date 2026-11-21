#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include <climits>
#include <cstdint>
#include <cstring>

#include <cassert>
#include <cmath>
#include <cstdio>

// Unit tests for host-side cuda_fp16.h __half type.

static bool test_conversion_contracts() {
    // Exhaust every binary16 bit pattern against the native host storage type,
    // including subnormals, signed zeros, and all infinity/NaN encodings.
    for (unsigned bits = 0; bits < 65536; ++bits) {
        const uint16_t raw = static_cast<uint16_t>(bits);
        _Float16 native;
        std::memcpy(&native, &raw, sizeof(raw));
        const float expected = static_cast<float>(native);
        const __half half = __cumetal_half_from_bits(raw);
        const float actual = __half2float(half);
        if (std::isnan(expected)) {
            if (!std::isnan(actual) || !__hisnan(__float2half_rn(actual))) return false;
        } else {
            if (actual != expected || std::signbit(actual) != std::signbit(expected) ||
                __float2half_rn(actual).__x != raw) return false;
        }
        const __nv_bfloat16 bfloat = __cumetal_bfloat_from_bits(raw);
        const float bfloat_value = __bfloat162float(bfloat);
        const __nv_bfloat16 roundtrip = __float2bfloat16_rn(bfloat_value);
        if (std::isnan(bfloat_value)) {
            if (!__hisnan(roundtrip)) return false;
        } else if (roundtrip.__x != raw) return false;
    }
    // Values on opposite sides of a midpoint must not first round to FP32.
    const double halfway_half = 1.0 + 1.0 / 2048.0;
    if (__double2half(halfway_half).__x != 0x3c00 ||
        __double2half(halfway_half + 0x1p-40).__x != 0x3c01 ||
        __double2half(halfway_half - 0x1p-40).__x != 0x3c00 ||
        __double2bfloat16(1.0 + 1.0 / 256.0 + 0x1p-40).__x != 0x3f81 ||
        __ll2bfloat16_rn(16777216LL + 65536 + 1).__x != 0x4b81 ||
        __ull2bfloat16_rn(ULLONG_MAX).__x != 0x5f80) return false;
    const __half nan = __cumetal_half_from_bits(0x7fff);
    const __half inf = __cumetal_half_from_bits(0x7c00);
    const __half neg_inf = __cumetal_half_from_bits(0xfc00);
    if (__half2int_rz(nan) != 0 || __half2ll_rz(nan) != LLONG_MIN ||
        __half2ull_rz(nan) != (1ULL << 63) ||
        __half2int_rz(inf) != INT_MAX || __half2int_rz(neg_inf) != INT_MIN ||
        __half2ushort_rz(__float2half_rn(-1.75f)) != 0 ||
        __half2short_rz(__float2half_rn(65504.0f)) != SHRT_MAX ||
        __half2int_rz(__float2half_rn(-1.75f)) != -1 ||
        __half2int_rn(__float2half_rn(2.5f)) != 2 ||
        __half2int_rn(__float2half_rn(3.5f)) != 4) return false;
    if (__bfloat162int_rz(__cumetal_bfloat_from_bits(0x7fc1)) != 0 ||
        __bfloat162ll_rz(__cumetal_bfloat_from_bits(0x7fc1)) != LLONG_MIN ||
        __bfloat162ull_rz(__float2bfloat16_rn(-1.0f)) != 0 ||
        __bfloat162uint_rz(__float2bfloat16_rn(0x1p32f)) != UINT_MAX) return false;
    if (__half2float(hrint(__float2half_rn(2.5f))) != 2.0f ||
        __half2float(hrint(__float2half_rn(3.5f))) != 4.0f ||
        !std::signbit(__half2float(hrint(__float2half_rn(-0.5f)))) ||
        __half2float(hceil(__float2half_rn(-1.75f))) != -1.0f ||
        __bfloat162float(hfloor(__float2bfloat16_rn(-1.75f))) != -2.0f ||
        __bfloat162float(hsqrt(__float2bfloat16_rn(4.0f))) != 2.0f ||
        __half2float(hrcp(__float2half_rn(2.0f))) != 0.5f) return false;
    return true;
}

// Neighbouring 16-bit floats in value order (sign-magnitude encoding).
static uint16_t next_up16(uint16_t bits) {
    if (bits == 0x8000u) return 0x0001u;
    return (bits & 0x8000u) ? static_cast<uint16_t>(bits - 1) : static_cast<uint16_t>(bits + 1);
}
static uint16_t next_down16(uint16_t bits) {
    if (bits == 0x0000u) return 0x8001u;
    return (bits & 0x8000u) ? static_cast<uint16_t>(bits + 1) : static_cast<uint16_t>(bits - 1);
}

// Directed conversions must equal the round-to-nearest result moved to the
// representable neighbour on the required side of x. Zero results keep the
// sign of x, overflow saturates toward zero for rz and the inward direction.
template <typename ToFloat>
static bool directed_matches(float x, uint16_t rn, uint16_t rz, uint16_t ru, uint16_t rd,
                             ToFloat to_float) {
    uint16_t up = rn;
    if (to_float(up) < x) up = next_up16(up);
    uint16_t down = rn;
    if (to_float(down) > x) down = next_down16(down);
    const uint16_t toward_zero = std::signbit(x) ? up : down;
    return ru == up && rd == down && rz == toward_zero;
}

static bool test_directed_rounding() {
    const auto half_value = [](uint16_t bits) { return __half2float(__cumetal_half_from_bits(bits)); };
    const auto bfloat_value = [](uint16_t bits) {
        return __bfloat162float(__cumetal_bfloat_from_bits(bits));
    };
    // Every float exponent, sign and many significands, plus fixed edges:
    // the half overflow boundary and its tie, subnormal edges, float extremes.
    static const uint32_t edges[] = {
        0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x477fe000u, 0x477ff000u,
        0x477fefffu, 0xc77ff000u, 0xc77ff001u, 0x33800000u, 0x33000000u, 0x33000001u,
        0xb3000001u, 0x387fc000u, 0x38800000u, 0x7f7fffffu, 0xff7fffffu, 0x7f7f8000u,
        0x00000001u, 0x80000001u, 0x3f808000u, 0x3f818000u,
    };
    const auto check = [&](uint32_t bits) {
        float x;
        std::memcpy(&x, &bits, sizeof(x));
        if (std::isnan(x)) return true;
        const bool half_ok = directed_matches(
            x, __half_as_ushort(__float2half_rn(x)), __half_as_ushort(__float2half_rz(x)),
            __half_as_ushort(__float2half_ru(x)), __half_as_ushort(__float2half_rd(x)),
            half_value);
        const bool bfloat_ok = directed_matches(
            x, __bfloat16_as_ushort(__float2bfloat16_rn(x)),
            __bfloat16_as_ushort(__float2bfloat16_rz(x)),
            __bfloat16_as_ushort(__float2bfloat16_ru(x)),
            __bfloat16_as_ushort(__float2bfloat16_rd(x)), bfloat_value);
        if (!half_ok || !bfloat_ok) {
            std::fprintf(stderr, "directed rounding mismatch for 0x%08x (half %d, bfloat16 %d)\n",
                         bits, half_ok, bfloat_ok);
            return false;
        }
        return true;
    };
    for (const uint32_t bits : edges) {
        if (!check(bits)) return false;
    }
    for (uint64_t bits = 0; bits <= 0xffffffffull; bits += 4099) {
        if (!check(static_cast<uint32_t>(bits))) return false;
    }
    // Infinity inputs stay infinite in every direction.
    if (__half_as_ushort(__float2half_rz(INFINITY)) != 0x7c00u ||
        __bfloat16_as_ushort(__float2bfloat16_rd(-INFINITY)) != 0xff80u) {
        return false;
    }
    // A bit-identity check of the reinterpretation helpers.
    return __bfloat16_as_ushort(__ushort_as_bfloat16(0x3f81u)) == 0x3f81u &&
           __bfloat16_as_short(__short_as_bfloat16(-2)) == -2;
}

int main() {
    if (!test_conversion_contracts()) {
        std::fprintf(stderr, "FAIL: half/bfloat rounding, saturation, or special-value contract\n");
        return 1;
    }
    if (!test_directed_rounding()) {
        std::fprintf(stderr, "FAIL: half/bfloat16 directed-rounding conversions\n");
        return 1;
    }
    // float → half → float roundtrip for simple values
    const float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 65504.0f, -65504.0f, 1.0f / 1024.0f};
    const int n = static_cast<int>(sizeof(vals) / sizeof(vals[0]));

    for (int i = 0; i < n; ++i) {
        const __half h = __float2half(vals[i]);
        const float back = __half2float(h);
        // Allow 0.1% relative error (half has ~3 significant decimal digits)
        const float tol = std::fabs(vals[i]) * 0.001f + 1e-6f;
        if (std::fabs(back - vals[i]) > tol) {
            std::fprintf(stderr,
                         "FAIL: __float2half/half2float roundtrip: input=%g, output=%g, diff=%g\n",
                         static_cast<double>(vals[i]),
                         static_cast<double>(back),
                         static_cast<double>(std::fabs(back - vals[i])));
            return 1;
        }
    }

    // Basic arithmetic via operator overloads
    const __half a = __float2half(3.0f);
    const __half b = __float2half(2.0f);

    const __half2 duplicated = __float2half2_rn(-1.5f);
    if (std::fabs(__low2float(duplicated) + 1.5f) > 0.01f ||
        std::fabs(__high2float(duplicated) + 1.5f) > 0.01f) {
        std::fprintf(stderr, "FAIL: __float2half2_rn did not duplicate its input\n");
        return 1;
    }
    const __half2 fma2 = __hfma2(__half2(2.0f, 3.0f),
                                 __half2(4.0f, 5.0f),
                                 __half2(1.0f, 2.0f));
    if (std::fabs(__low2float(fma2) - 9.0f) > 0.01f ||
        std::fabs(__high2float(fma2) - 17.0f) > 0.01f) {
        std::fprintf(stderr, "FAIL: __hfma2 produced the wrong lane values\n");
        return 1;
    }

    const float sum = __half2float(a + b);
    if (std::fabs(sum - 5.0f) > 0.01f) {
        std::fprintf(stderr, "FAIL: __half add: got %g, expected 5.0\n", static_cast<double>(sum));
        return 1;
    }

    const float diff = __half2float(a - b);
    if (std::fabs(diff - 1.0f) > 0.01f) {
        std::fprintf(stderr, "FAIL: __half sub: got %g, expected 1.0\n", static_cast<double>(diff));
        return 1;
    }

    const float prod = __half2float(a * b);
    if (std::fabs(prod - 6.0f) > 0.01f) {
        std::fprintf(stderr,
                     "FAIL: __half mul: got %g, expected 6.0\n",
                     static_cast<double>(prod));
        return 1;
    }

    // Comparison operators
    if (!(a > b)) {
        std::fprintf(stderr, "FAIL: __half operator> failed\n");
        return 1;
    }
    if (!(b < a)) {
        std::fprintf(stderr, "FAIL: __half operator< failed\n");
        return 1;
    }
    if (!(a == a)) {
        std::fprintf(stderr, "FAIL: __half operator== failed\n");
        return 1;
    }
    if (!(a != b)) {
        std::fprintf(stderr, "FAIL: __half operator!= failed\n");
        return 1;
    }

    // __hadd / __hmul functions
    const float hadd_val = __half2float(__hadd(a, b));
    if (std::fabs(hadd_val - 5.0f) > 0.01f) {
        std::fprintf(stderr,
                     "FAIL: __hadd: got %g, expected 5.0\n",
                     static_cast<double>(hadd_val));
        return 1;
    }

    std::printf("PASS: cuda_fp16.h host-side __half type (roundtrip, arithmetic, comparison)\n");
    return 0;
}
