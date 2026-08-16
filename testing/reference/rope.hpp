#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor_view.hpp"
#include <cmath>

namespace test::reference::cpu {
    inline void rope(const inference::TensorView<inference::cpu::bf16_t> values, const float theta, const std::size_t pos) {
        const auto head_count = values.dim(0);
        const auto head_size = values.dim(1);
        const auto half_size = head_size / 2;

        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t index = 0; index < half_size; ++index) {
                const auto exponent = static_cast<float>(2 * index) / static_cast<float>(head_size);
                const auto angle = static_cast<float>(pos) / std::pow(theta, exponent);
                const auto cosine = std::cos(angle);
                const auto sine = std::sin(angle);
                const auto first = static_cast<float>(values(head, index));
                const auto second = static_cast<float>(values(head, half_size + index));

                values(head, index) = first * cosine - second * sine;
                values(head, half_size + index) = second * cosine + first * sine;
            }
        }
    }
} // namespace test::reference::cpu
