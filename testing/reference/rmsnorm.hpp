#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor_view.hpp"
#include <cmath>

namespace test::reference {
    inline void rmsnorm(inference::TensorView<inference::cpu::bf16_t> input, inference::TensorView<const inference::cpu::bf16_t> weight,
                        inference::TensorView<inference::cpu::bf16_t> output, const float epsilon) {
        const auto row_size = weight.size();
        const auto row_count = input.size() / row_size;

        for (std::size_t row = 0; row < row_count; ++row) {
            const auto row_offset = row * row_size;
            float sum_of_squares = 0.0F;

            for (std::size_t index = 0; index < row_size; ++index) {
                const auto value = input[row_offset + index].to_float();
                sum_of_squares += value * value;
            }

            const auto scale = 1.0F / std::sqrt(sum_of_squares / static_cast<float>(row_size) + epsilon);
            for (std::size_t index = 0; index < row_size; ++index) {
                output[row_offset + index] = input[row_offset + index].to_float() * weight(index).to_float() * scale;
            }
        }
    }
} // namespace test::reference
