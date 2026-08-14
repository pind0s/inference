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
        constexpr auto elem = -2.0f;
        std::uniform_real_distribution distribution(-2.0f, 2.0f);
        std::vector<__bf16> values(size);
        std::ranges::generate(values, [&] { return static_cast<__bf16>(distribution(engine)); });
        return values;
    }
} // namespace test
