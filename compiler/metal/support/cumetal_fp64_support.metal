#include "../../../third_party/VF64-metal/Sources/VF64Metal/Shaders/Interop/VF64Support.metal"

// Linkable raw-binary64 entry points for CuMetal's typed MSL backend. The
// fast48 functions decode storage to an FP32 expansion for each ALU operation
// and repack it immediately, preserving CUDA-visible eight-byte storage while
// retaining the documented ~48-bit/binary32-range contract.
inline ulong cm_fp64_fast_canonicalize(ulong value) {
    bool range_flag = false;
    return pack_binary64(unpack_binary64(value, range_flag));
}

[[visible]] ulong cm_fp64_fast_add(ulong a, ulong b) {
    bool ar = false, br = false;
    return pack_binary64(add_ff(unpack_binary64(a, ar), unpack_binary64(b, br)));
}

[[visible]] ulong cm_fp64_fast_sub(ulong a, ulong b) {
    bool ar = false, br = false;
    return pack_binary64(sub_ff(unpack_binary64(a, ar), unpack_binary64(b, br)));
}

[[visible]] ulong cm_fp64_fast_mul(ulong a, ulong b) {
    bool ar = false, br = false;
    return pack_binary64(mul_ff(unpack_binary64(a, ar), unpack_binary64(b, br)));
}

[[visible]] ulong cm_fp64_fast_div(ulong a, ulong b) {
    bool ar = false, br = false;
    return pack_binary64(div_ff(unpack_binary64(a, ar), unpack_binary64(b, br)));
}

[[visible]] ulong cm_fp64_fast_sqrt(ulong a) {
    bool range = false;
    return pack_binary64(sqrt_ff(unpack_binary64(a, range)));
}

[[visible]] ulong cm_fp64_fast_fma(ulong a, ulong b, ulong c) {
    bool ar = false, br = false, cr = false;
    return pack_binary64(fma_ff(unpack_binary64(a, ar), unpack_binary64(b, br),
                                unpack_binary64(c, cr)));
}

inline bool cm_fp64_nan(ulong value) {
    return (value & 0x7ff0000000000000ul) == 0x7ff0000000000000ul &&
           (value & 0x000ffffffffffffful) != 0ul;
}

[[visible]] bool cm_fp64_fast_eq(ulong a, ulong b) {
    return vf64_eq(cm_fp64_fast_canonicalize(a),
                   cm_fp64_fast_canonicalize(b));
}

[[visible]] bool cm_fp64_fast_lt(ulong a, ulong b) {
    return vf64_lt_quiet(cm_fp64_fast_canonicalize(a),
                         cm_fp64_fast_canonicalize(b));
}

[[visible]] bool cm_fp64_fast_le(ulong a, ulong b) {
    return vf64_le_quiet(cm_fp64_fast_canonicalize(a),
                         cm_fp64_fast_canonicalize(b));
}

[[visible]] ulong cm_fp64_fast_min(ulong a, ulong b) {
    a = cm_fp64_fast_canonicalize(a);
    b = cm_fp64_fast_canonicalize(b);
    if (cm_fp64_nan(a)) return b;
    if (cm_fp64_nan(b)) return a;
    if (vf64_eq(a, b)) return (a | b); // fmin(+0,-0) = -0
    return vf64_lt_quiet(a, b) ? a : b;
}

[[visible]] ulong cm_fp64_fast_max(ulong a, ulong b) {
    a = cm_fp64_fast_canonicalize(a);
    b = cm_fp64_fast_canonicalize(b);
    if (cm_fp64_nan(a)) return b;
    if (cm_fp64_nan(b)) return a;
    if (vf64_eq(a, b)) return (a & b); // fmax(+0,-0) = +0
    return vf64_lt_quiet(a, b) ? b : a;
}

[[visible]] ulong cm_fp64_fast_remainder(ulong a, ulong b) {
    return cm_fp64_fast_canonicalize(vf64_remainder(
        cm_fp64_fast_canonicalize(a), cm_fp64_fast_canonicalize(b)));
}

[[visible]] ulong cm_fp64_fast_round_int(ulong a, uint mode) {
    return cm_fp64_fast_canonicalize(vf64_round_to_int(
        cm_fp64_fast_canonicalize(a), mode, false));
}

[[visible]] ulong cm_fp64_fast_f32_to_f64(uint value) {
    return cm_fp64_fast_canonicalize(vf64_f32_to_f64(value));
}
