#pragma once
#include "cpu/bf16.hpp"

#include <cmath>


namespace inference::cpu::kernels {
    inline void rmsnorm(const __bf16* __restrict input, const __bf16* __restrict weight, __bf16* __restrict output, const std::size_t element_count,
                        const std::size_t row_size, const float epsilon) {
        for (std::size_t row = 0; row < element_count; row += row_size) {
            float sum_of_squares = 0.0F;
            for (std::size_t index = 0; index < row_size; ++index) {
                const auto value = bf16::to_float(input[row + index]);
                sum_of_squares += value * value;
            }

            const auto scale = 1.0F / std::sqrt(sum_of_squares / static_cast<float>(row_size) + epsilon);
            for (std::size_t index = 0; index < row_size; ++index) {
                const auto value = bf16::to_float(input[row + index]);
                const auto gain = bf16::to_float(weight[index]);
                output[row + index] = bf16::from_float(value * scale * gain);
            }
        }
    }
} // namespace inference::cpu::kernels
