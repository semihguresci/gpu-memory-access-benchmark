#pragma once

#include <algorithm>
#include <cmath>

namespace FloatValidation {

inline bool nearly_equal(float lhs, float rhs, float relative_tolerance) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs) || !std::isfinite(relative_tolerance) || relative_tolerance < 0.0F) {
        return false;
    }
    const float scale = std::max({1.0F, std::fabs(lhs), std::fabs(rhs)});
    return std::fabs(lhs - rhs) <= relative_tolerance * scale;
}

} // namespace FloatValidation
