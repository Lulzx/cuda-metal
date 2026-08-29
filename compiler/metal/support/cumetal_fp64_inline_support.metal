#include "../../../third_party/VF64-metal/Sources/VF64Metal/Shaders/Core/Preamble.metal"
#include "../../../third_party/VF64-metal/Sources/VF64Metal/Shaders/Pair/Arithmetic.metal"
#include "../../../third_party/VF64-metal/Sources/VF64Metal/Shaders/Pair/Codec.metal"
#include "../../../third_party/VF64-metal/Sources/VF64Metal/Shaders/IEEE/Arithmetic.metal"
#include "../../../third_party/VF64-metal/Sources/VF64Metal/Shaders/Wide/Arithmetic.metal"

// Private helpers for typed MSL. These deliberately live in the kernel's
// translation unit: [[visible]] functions require MTL function stitching at
// pipeline creation, whereas these helpers are ordinary statically inlined ALU.
inline ulong cm_fp64_fast_canonicalize(ulong value) {
    bool range_flag = false;
    return pack_binary64(unpack_binary64(value, range_flag));
}

inline ulong cm_fp64_fast_add(ulong a, ulong b) {
    bool ar = false, br = false;
    return pack_binary64(add_ff(unpack_binary64(a, ar), unpack_binary64(b, br)));
}

inline ulong cm_fp64_fast_sub(ulong a, ulong b) {
    bool ar = false, br = false;
    return pack_binary64(sub_ff(unpack_binary64(a, ar), unpack_binary64(b, br)));
}

inline ulong cm_fp64_fast_mul(ulong a, ulong b) {
    bool ar = false, br = false;
    return pack_binary64(mul_ff(unpack_binary64(a, ar), unpack_binary64(b, br)));
}

inline ulong cm_fp64_fast_div(ulong a, ulong b) {
    bool ar = false, br = false;
    return pack_binary64(div_ff(unpack_binary64(a, ar), unpack_binary64(b, br)));
}

inline ulong cm_fp64_fast_sqrt(ulong a) {
    bool range = false;
    return pack_binary64(sqrt_ff(unpack_binary64(a, range)));
}

inline ulong cm_fp64_fast_fma(ulong a, ulong b, ulong c) {
    bool ar = false, br = false, cr = false;
    return pack_binary64(fma_ff(unpack_binary64(a, ar), unpack_binary64(b, br),
                                unpack_binary64(c, cr)));
}

inline bool cm_fp64_nan(ulong value) {
    return (value & 0x7ff0000000000000ul) == 0x7ff0000000000000ul &&
           (value & 0x000ffffffffffffful) != 0ul;
}

inline bool vf64_eq(ulong a, ulong b) {
    uint flags = 0;
    return soft_equal64_status(a, b, false, flags);
}

inline bool vf64_lt(ulong a, ulong b) {
    uint flags = 0;
    return soft_less64_status(a, b, false, false, flags);
}

inline bool vf64_le(ulong a, ulong b) {
    uint flags = 0;
    return soft_less64_status(a, b, true, false, flags);
}

inline ulong vf64_min(ulong a, ulong b) {
    if (cm_fp64_nan(a)) return b;
    if (cm_fp64_nan(b)) return a;
    if (vf64_eq(a, b)) return a | b;
    uint flags = 0;
    return soft_less64_status(a, b, false, true, flags) ? a : b;
}

inline ulong vf64_max(ulong a, ulong b) {
    if (cm_fp64_nan(a)) return b;
    if (cm_fp64_nan(b)) return a;
    if (vf64_eq(a, b)) return a & b;
    uint flags = 0;
    return soft_less64_status(a, b, false, true, flags) ? b : a;
}

inline bool cm_fp64_fast_eq(ulong a, ulong b) {
    return vf64_eq(cm_fp64_fast_canonicalize(a), cm_fp64_fast_canonicalize(b));
}

inline bool cm_fp64_fast_lt(ulong a, ulong b) {
    uint flags = 0;
    return soft_less64_status(cm_fp64_fast_canonicalize(a),
                              cm_fp64_fast_canonicalize(b), false, true, flags);
}

inline bool cm_fp64_fast_le(ulong a, ulong b) {
    uint flags = 0;
    return soft_less64_status(cm_fp64_fast_canonicalize(a),
                              cm_fp64_fast_canonicalize(b), true, true, flags);
}

inline ulong cm_fp64_fast_min(ulong a, ulong b) {
    a = cm_fp64_fast_canonicalize(a);
    b = cm_fp64_fast_canonicalize(b);
    if (cm_fp64_nan(a)) return b;
    if (cm_fp64_nan(b)) return a;
    if (vf64_eq(a, b)) return a | b;
    uint flags = 0;
    return soft_less64_status(a, b, false, true, flags) ? a : b;
}

inline ulong cm_fp64_fast_max(ulong a, ulong b) {
    a = cm_fp64_fast_canonicalize(a);
    b = cm_fp64_fast_canonicalize(b);
    if (cm_fp64_nan(a)) return b;
    if (cm_fp64_nan(b)) return a;
    if (vf64_eq(a, b)) return a & b;
    uint flags = 0;
    return soft_less64_status(a, b, false, true, flags) ? b : a;
}

inline ulong vf64_remainder(ulong a, ulong b) {
    uint flags = 0;
    return soft_remainder64_status(a, b, flags);
}

inline ulong vf64_round_to_int(ulong a, uint mode, bool exact) {
    uint flags = 0;
    return soft_round_to_int64_status(a, mode, exact, flags);
}

inline ulong cm_fp64_fast_remainder(ulong a, ulong b) {
    return cm_fp64_fast_canonicalize(vf64_remainder(
        cm_fp64_fast_canonicalize(a), cm_fp64_fast_canonicalize(b)));
}

inline ulong cm_fp64_fast_round_int(ulong a, uint mode) {
    return cm_fp64_fast_canonicalize(
        vf64_round_to_int(cm_fp64_fast_canonicalize(a), mode, false));
}

inline ulong vf64_f32_to_f64(uint raw) {
    uint flags = 0;
    return soft_format_to_f64_status(ulong(raw), 8u, 23u, 127, flags);
}

inline ulong cm_fp64_fast_f32_to_f64(uint value) {
    return cm_fp64_fast_canonicalize(vf64_f32_to_f64(value));
}

inline ulong vf64_add_rne(ulong a, ulong b) {
    uint flags = 0;
    return soft_add64_status(a, b, soft_round_near_even, flags);
}

inline ulong vf64_sub_rne(ulong a, ulong b) {
    uint flags = 0;
    return soft_sub64_status(a, b, soft_round_near_even, flags);
}

inline ulong vf64_mul_rne(ulong a, ulong b) {
    uint flags = 0;
    return soft_mul64_status(a, b, soft_round_near_even, flags);
}

inline ulong vf64_div_rne(ulong a, ulong b) {
    uint flags = 0;
    return soft_div64_status(a, b, soft_round_near_even, flags);
}

inline ulong vf64_sqrt_rne(ulong a) {
    uint flags = 0;
    return soft_sqrt64_status(a, soft_round_near_even, flags);
}

inline ulong vf64_fma_rne(ulong a, ulong b, ulong c) {
    uint flags = 0;
    return soft_fma64_status(a, b, c, soft_round_near_even, flags);
}

inline ulong vf64_wide_add(ulong a, ulong b) {
    return wide_pack64(wide_add(wide_unpack64(a), wide_unpack64(b)));
}

inline ulong vf64_wide_sub(ulong a, ulong b) {
    return wide_pack64(wide_sub(wide_unpack64(a), wide_unpack64(b)));
}

inline ulong vf64_wide_mul(ulong a, ulong b) {
    return wide_pack64(wide_mul(wide_unpack64(a), wide_unpack64(b)));
}

inline ulong vf64_wide_div(ulong a, ulong b) {
    return wide_pack64(wide_div(wide_unpack64(a), wide_unpack64(b)));
}

inline ulong vf64_wide_sqrt(ulong a) {
    return wide_pack64(wide_sqrt(wide_unpack64(a)));
}

inline ulong vf64_wide_fma(ulong a, ulong b, ulong c) {
    return wide_pack64(wide_fma(wide_unpack64(a), wide_unpack64(b),
                                wide_unpack64(c)));
}

inline uint vf64_f64_to_f32(ulong value, uint mode) {
    uint flags = 0;
    return uint(soft_f64_to_format_status(value, mode, 8u, 23u, 127, flags));
}

inline ulong vf64_ui32_to_f64(uint value, uint mode) {
    uint flags = 0;
    return soft_uint_to_f64_status(ulong(value), false, mode, flags);
}

inline ulong vf64_ui64_to_f64(ulong value, uint mode) {
    uint flags = 0;
    return soft_uint_to_f64_status(value, false, mode, flags);
}

inline ulong vf64_i32_to_f64(int value, uint mode) {
    const bool sign = value < 0;
    const uint magnitude = sign ? uint(-(value + 1)) + 1u : uint(value);
    uint flags = 0;
    return soft_uint_to_f64_status(ulong(magnitude), sign, mode, flags);
}

inline ulong vf64_i64_to_f64(long value, uint mode) {
    const bool sign = value < 0;
    const ulong magnitude = sign ? ulong(-(value + 1l)) + 1ul : ulong(value);
    uint flags = 0;
    return soft_uint_to_f64_status(magnitude, sign, mode, flags);
}

inline uint vf64_f64_to_ui32(ulong value, uint mode, bool exact) {
    uint flags = 0;
    return uint(soft_f64_to_int_status(value, mode, exact, false, 32u, flags));
}

inline ulong vf64_f64_to_ui64(ulong value, uint mode, bool exact) {
    uint flags = 0;
    return soft_f64_to_int_status(value, mode, exact, false, 64u, flags);
}

inline int vf64_f64_to_i32(ulong value, uint mode, bool exact) {
    uint flags = 0;
    return as_type<int>(uint(
        soft_f64_to_int_status(value, mode, exact, true, 32u, flags)));
}

inline long vf64_f64_to_i64(ulong value, uint mode, bool exact) {
    uint flags = 0;
    return as_type<long>(
        soft_f64_to_int_status(value, mode, exact, true, 64u, flags));
}
