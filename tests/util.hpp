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

    [[nodiscard]] inline std::vector<inference::cpu::bf16_t> random_bf16_vector(const std::size_t size, const std::uint32_t seed = 0) {
        static std::mt19937 engine{seed};
        std::uniform_real_distribution distribution{-1.0F, 1.0F};
        std::vector<inference::cpu::bf16_t> values(size);
        std::ranges::generate(values, [&] { return inference::cpu::bf16_t::from_float(distribution(engine)); });
        return values;
    }
} // namespace test
