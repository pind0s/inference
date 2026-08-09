#pragma once

#include <cmath>

#include "cpu/bf16.hpp"
namespace inference::cpu::kernels {
    inline void rmsnorm(const bf16_t* __restrict input, const bf16_t* __restrict weight, bf16_t* __restrict output, const std::size_t row_count,
                        const std::size_t hidden_size, const float epsilon) noexcept {
        for (std::size_t row = 0; row < row_count; ++row) {
            float sum_of_squares = 0.0F;
            for (std::size_t hidden = 0; hidden < hidden_size; ++hidden) {
                const auto value = input[row * hidden_size + hidden].to_float();
                sum_of_squares += value * value;
            }

            const auto scale = 1.0F / std::sqrt(sum_of_squares / static_cast<float>(hidden_size) + epsilon);
            for (std::size_t hidden = 0; hidden < hidden_size; ++hidden) {
                const auto value = input[row * hidden_size + hidden].to_float();
                const auto gain = weight[hidden].to_float();
                output[row * hidden_size + hidden] = bf16_t::from_float(value * scale * gain);
            }
        }
    }
} // namespace inference::cpu::kernels
