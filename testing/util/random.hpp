#pragma once
#include "make_tensor.hpp"
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace test {
    struct RandomSeed {
        std::uint32_t value;
    };

    [[nodiscard]] inline inference::Tensor random_bf16(const inference::TensorShape& shape, inference::Backend& backend, const RandomSeed seed = { 0 },
                                                       const float min = -2.0F, const float max = 2.0F) {
        std::mt19937 engine{ seed.value };
        std::uniform_real_distribution distribution(min, max);
        std::vector<inference::cpu::bf16_t> values(shape.size());
        std::ranges::generate(values, [&] { return distribution(engine); });
        return make_tensor(values, shape, backend);
    }
} // namespace test
