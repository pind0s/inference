#pragma once
#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "cpu/bf16.hpp"

namespace test {
    [[nodiscard]] inline std::filesystem::path resource_path() {
        return std::filesystem::path{TEST_RESOURCE_DIR};
    }

    [[nodiscard]] inline std::vector<__bf16> random_bf16_vector(const std::size_t size) {
        static std::mt19937 engine{0};
        std::uniform_real_distribution distribution{-1.0F, 1.0F};
        std::vector<__bf16> values(size);
        std::ranges::generate(values, [&] { return inference::cpu::bf16::from_float(distribution(engine)); });
        return values;
    }
} // namespace test
