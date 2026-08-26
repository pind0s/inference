#pragma once
#include <cmath>

namespace test::util {
    template <typename T>
    [[nodiscard]] float tolerance_for(const T expected, const float absolute_tolerance = 0.01F, const float relative_tolerance = 0.01F) {
        return absolute_tolerance + relative_tolerance * std::abs(static_cast<float>(expected));
    }
} // namespace test::util
