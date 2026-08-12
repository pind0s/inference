#pragma once
#include "cpu/bf16.hpp"

#include <cmath>

namespace inference::cpu::kernels {
    inline void rmsnorm(const bf16_t* __restrict input, const bf16_t* __restrict weight, bf16_t* __restrict output, const std::size_t element_count,
                        const std::size_t row_size, const float epsilon) {

        for (std::size_t row = 0; row < element_count; row += row_size) {
            float sum_of_squares = 0.0F;
            for (std::size_t index = 0; index < row_size; ++index) {
                const auto value = input[row + index].to_float();
                sum_of_squares += value * value;
            }

            const auto scale = 1.0F / std::sqrt(sum_of_squares / static_cast<float>(row_size) + epsilon);
            for (std::size_t index = 0; index < row_size; ++index) {
                const auto value = input[row + index].to_float();
                const auto gain = weight[index].to_float();
                output[row + index] = bf16_t::from_float(value * scale * gain);
            }
        }
    }
} // namespace inference::cpu::kernels
