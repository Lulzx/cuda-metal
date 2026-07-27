#pragma once

namespace cumetal {

enum class MetalMathMode {
    kFast,
    kSafe,
};

// Process-wide policy selected by CUMETAL_MSL_MATH_MODE. The first query
// freezes the normalized mode so cache identity and Metal compilation cannot
// disagree if the environment is mutated later.
MetalMathMode current_metal_math_mode();
const char* metal_math_mode_name(MetalMathMode mode);

}  // namespace cumetal
