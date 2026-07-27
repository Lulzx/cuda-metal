#include "metal_math_mode.h"

#include "cumetal_diag.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace cumetal {

MetalMathMode current_metal_math_mode() {
    static const MetalMathMode mode = [] {
        const char* value = std::getenv("CUMETAL_MSL_MATH_MODE");
        if (value == nullptr || value[0] == '\0') {
            return MetalMathMode::kFast;
        }
        std::string normalized(value);
        std::transform(
            normalized.begin(), normalized.end(), normalized.begin(),
            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (normalized == "fast") {
            return MetalMathMode::kFast;
        }
        if (normalized == "safe") {
            return MetalMathMode::kSafe;
        }
        warn_once(
            "invalid-msl-math-mode",
            "invalid CUMETAL_MSL_MATH_MODE='" + std::string(value) +
                "'; expected 'fast' or 'safe', using compatibility default 'fast'");
        return MetalMathMode::kFast;
    }();
    return mode;
}

const char* metal_math_mode_name(MetalMathMode mode) {
    switch (mode) {
        case MetalMathMode::kFast:
            return "fast";
        case MetalMathMode::kSafe:
            return "safe";
    }
    return "fast";
}

}  // namespace cumetal
