#pragma once

#include "cumetal/metal/msl_ast.h"

namespace cumetal::metal::detail {

// Exact high 64 bits of a signed/unsigned 64x64 product, expressed using
// 32x32 partial products because MSL has no 128-bit integer type.
MslExpr integer_high_product_64(MslExpr left, MslExpr right, bool is_signed);

}  // namespace cumetal::metal::detail
