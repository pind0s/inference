#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor_view.hpp"
#include <cmath>
namespace inference::cpu::kernels {
    inline void rmsnorm(TensorView<const bf16_t> input, TensorView<const bf16_t> weight, TensorView<bf16_t> output, float epsilon) {
        const auto element_count = input.size();
        const auto row_size = weight.size();

        for (std::size_t row = 0; row < element_count; row += row_size) {
            float sum_of_squares = 0.0F;
            for (std::size_t index = 0; index < row_size; ++index) {
                const auto value = input[row + index].to_float();
                sum_of_squares += value * value;
            }

            const auto scale = 1.0F / std::sqrt(sum_of_squares / static_cast<float>(row_size) + epsilon);
            for (std::size_t index = 0; index < row_size; ++index) {
                const auto value = input[row + index].to_float();
                const auto gain = weight(index).to_float();
                output[row + index] = value * scale * gain;
            }
        }
    }
} // namespace inference::cpu::kernels
