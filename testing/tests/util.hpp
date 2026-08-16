#pragma once
#include <filesystem>

namespace test {
    [[nodiscard]] inline std::filesystem::path resource_path() {
        return std::filesystem::path{TEST_RESOURCE_DIR};
    }

    template <typename T>
    [[nodiscard]] float tolerance_for(const T expected, const float absolute_tolerance = 0.01f, const float relative_tolerance = 0.01f) {
        return absolute_tolerance + relative_tolerance * std::abs(static_cast<float>(expected));
    }
} // namespace test
