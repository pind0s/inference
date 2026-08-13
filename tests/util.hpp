#pragma once
#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace test {
    [[nodiscard]] inline std::filesystem::path resource_path() {
        return std::filesystem::path{TEST_RESOURCE_DIR};
    }

    [[nodiscard]] inline std::vector<__bf16> random_bf16_vector(const std::size_t size) {
        static std::mt19937 engine{0};
        auto min = std::numeric_limits<__bf16>::lowest();
        auto max = std::numeric_limits<__bf16>::max();
        std::uniform_real_distribution distribution(static_cast<float>(min), static_cast<float>(max));
        std::vector<__bf16> values(size);
        std::ranges::generate(values, [&] { return static_cast<__bf16>(distribution(engine)); });
        return values;
    }
} // namespace test
