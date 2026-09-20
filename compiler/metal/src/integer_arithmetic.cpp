#include "integer_arithmetic.h"

namespace cumetal::metal::detail {

MslExpr integer_high_product_64(MslExpr left, MslExpr right, bool is_signed) {
    // Exact high half using four 32x32 products. The two
    // middle sums fit in u64; carry propagation is explicit.
    const MslType u64 = MslType::uint(64);
    const auto literal = [&](const char* value) {
        return MslExpression::literal(value, u64);
    };
    const auto binary64 = [&](const char* op, MslExpr a, MslExpr b) {
        return MslExpression::binary(op, a, b, u64);
    };
    const auto high32 = [&](MslExpr value) {
        return binary64(">>", value, literal("32ul"));
    };
    const auto low32 = [&](MslExpr value) {
        return binary64("&", value, literal("0xfffffffful"));
    };
    const MslExpr a = MslExpression::cast(u64, left);
    const MslExpr b = MslExpression::cast(u64, right);
    const MslExpr a0 = low32(a), a1 = high32(a);
    const MslExpr b0 = low32(b), b1 = high32(b);
    const MslExpr t = binary64("+", binary64("*", a1, b0),
                              high32(binary64("*", a0, b0)));
    const MslExpr middle = binary64("+", low32(t), binary64("*", a0, b1));
    MslExpr high = binary64("+", binary64("+", binary64("*", a1, b1), high32(t)),
                           high32(middle));
    if (is_signed) {
        const auto correction = [&](MslExpr x, MslExpr y) {
            return MslExpression::conditional(
                MslExpression::binary("!=", binary64(">>", x, literal("63ul")),
                                      literal("0ul"), MslType::boolean()),
                y, literal("0ul"), u64);
        };
        high = binary64("-", binary64("-", high, correction(a, b)), correction(b, a));
    }
    return high;
}

}  // namespace cumetal::metal::detail
