#pragma once
#include "cpu/bf16.hpp"

#include <cmath>

namespace inference::cpu::kernels {
    inline void rope(bf16_t* __restrict values, const std::size_t head_count, const std::size_t head_size, const float theta, const std::size_t pos) {
        const auto half_size = head_size / 2;

        for (std::size_t head = 0; head < head_count; ++head) {
            const auto base = head * head_size;
            for (std::size_t index = 0; index < half_size; ++index) {
                const auto exponent = static_cast<float>(2 * index) / static_cast<float>(head_size);
                const auto angle = static_cast<float>(pos) / std::pow(theta, exponent);
                const auto cosine = std::cos(angle);
                const auto sine = std::sin(angle);
                const auto first = values[base + index].to_float();
                const auto second = values[base + half_size + index].to_float();

                values[base + index] = bf16_t::from_float(first * cosine - second * sine);
                values[base + half_size + index] = bf16_t::from_float(second * cosine + first * sine);
            }
        }
    }
} // namespace inference::cpu::kernels
