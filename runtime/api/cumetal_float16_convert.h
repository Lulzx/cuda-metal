#pragma once
// Clean-room scalar conversion primitives shared by CUDA half/bfloat headers.
#include <stdint.h>
#include <limits.h>

#if defined(__clang__) && defined(__CUDA__)
#define CUMETAL_F16_HD __host__ __device__
#else
#define CUMETAL_F16_HD
#endif

namespace cumetal_float16 {

CUMETAL_F16_HD inline uint64_t round_shift(uint64_t value, int shift) {
    if (shift <= 0) return value << -shift;
    if (shift > 64) return 0;
    if (shift == 64) return value > (uint64_t(1) << 63) ? 1 : 0;
    const uint64_t truncated = value >> shift;
    const uint64_t remainder = value & ((uint64_t(1) << shift) - 1);
    const uint64_t halfway = uint64_t(1) << (shift - 1);
    return truncated + (remainder > halfway ||
                        (remainder == halfway && (truncated & 1)));
}

// value = significand * 2^exponent. Round only once to the destination;
// converting through float first can double-round integer/double inputs.
template <int ExponentBits, int FractionBits>
CUMETAL_F16_HD inline uint16_t pack(bool negative, uint64_t significand, int exponent) {
    const uint16_t sign = negative ? 0x8000u : 0;
    if (significand == 0) return sign;
    int top = 0;
    for (uint64_t scan = significand; scan >>= 1;) ++top;
    constexpr int bias = (1 << (ExponentBits - 1)) - 1;
    constexpr int max_exponent = (1 << ExponentBits) - 1;
    int destination_exponent = exponent + top + bias;
    uint64_t rounded;
    if (destination_exponent <= 0) {
        rounded = round_shift(significand, (1 - bias - FractionBits) - exponent);
        // Rounding a subnormal can carry into the minimum normal value.
        return sign | static_cast<uint16_t>(rounded);
    }
    if (destination_exponent >= max_exponent)
        return sign | static_cast<uint16_t>(max_exponent << FractionBits);
    rounded = round_shift(significand, top - FractionBits);
    if (rounded == (uint64_t(1) << (FractionBits + 1))) {
        rounded >>= 1;
        ++destination_exponent;
    }
    if (destination_exponent >= max_exponent)
        return sign | static_cast<uint16_t>(max_exponent << FractionBits);
    return sign | static_cast<uint16_t>((destination_exponent << FractionBits) |
                 (rounded & ((uint64_t(1) << FractionBits) - 1)));
}

template <int ExponentBits, int FractionBits>
CUMETAL_F16_HD inline uint16_t from_float(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const unsigned exponent = (bits >> 23) & 255;
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 255) {
        return static_cast<uint16_t>((bits >> 16 & 0x8000u) |
            (((1 << ExponentBits) - 1) << FractionBits) |
            (fraction ? (1 << (FractionBits - 1)) : 0));
    }
    return pack<ExponentBits, FractionBits>(bits >> 31,
        fraction | (exponent ? 0x800000u : 0),
        exponent ? static_cast<int>(exponent) - 127 - 23 : -149);
}

template <int ExponentBits, int FractionBits>
CUMETAL_F16_HD inline uint16_t from_double(double value) {
    uint64_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const unsigned exponent = (bits >> 52) & 2047;
    const uint64_t fraction = bits & ((uint64_t(1) << 52) - 1);
    if (exponent == 2047) {
        return static_cast<uint16_t>((bits >> 48 & 0x8000u) |
            (((1 << ExponentBits) - 1) << FractionBits) |
            (fraction ? (1 << (FractionBits - 1)) : 0));
    }
    return pack<ExponentBits, FractionBits>(bits >> 63,
        fraction | (exponent ? uint64_t(1) << 52 : 0),
        exponent ? static_cast<int>(exponent) - 1023 - 52 : -1074);
}

// Decode FP32 exactly before casting. NaN and overflow must never reach a
// C++ float-to-integer conversion (whose behavior would be undefined).
template <bool Nearest>
CUMETAL_F16_HD inline uint64_t integer_magnitude(uint32_t bits, bool* overflow) {
    const int exponent = static_cast<int>((bits >> 23) & 255) - 127;
    *overflow = exponent >= 64;
    if (*overflow || exponent < (Nearest ? -1 : 0)) return 0;
    const uint64_t significand = (bits & 0x7fffffu) | 0x800000u;
    const int shift = 23 - exponent;
    if constexpr (Nearest) return round_shift(significand, shift);
    return shift > 0 ? significand >> shift : significand << -shift;
}

CUMETAL_F16_HD inline float round_even(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    if (((bits >> 23) & 255) >= 150) return value;  // Already integral, Inf, or NaN.
    bool overflow;
    const float rounded = static_cast<float>(integer_magnitude<true>(bits, &overflow));
    return bits >> 31 ? -rounded : rounded;
}

template <int Width, bool Nearest = false>
CUMETAL_F16_HD inline long long signed_integer(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const bool negative = bits >> 31;
    const uint64_t minimum_magnitude = uint64_t(1) << (Width - 1);
    const long long minimum = Width == 64 ? LLONG_MIN : -static_cast<long long>(minimum_magnitude);
    const long long maximum = static_cast<long long>(minimum_magnitude - 1);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return Width == 64 ? LLONG_MIN : 0;
    bool overflow;
    const uint64_t magnitude = integer_magnitude<Nearest>(bits, &overflow);
    if (negative) {
        if (overflow || magnitude >= minimum_magnitude) return minimum;
        return -static_cast<long long>(magnitude);
    }
    if (overflow || magnitude >= minimum_magnitude) return maximum;
    return static_cast<long long>(magnitude);
}

template <int Width, bool Nearest = false>
CUMETAL_F16_HD inline unsigned long long unsigned_integer(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const uint64_t maximum = ULLONG_MAX >> (64 - Width);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return Width == 64 ? uint64_t(1) << 63 : 0;
    if (bits >> 31) return 0;
    bool overflow;
    const uint64_t magnitude = integer_magnitude<Nearest>(bits, &overflow);
    return overflow || magnitude > maximum ? maximum : magnitude;
}

}  // namespace cumetal_float16
