#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor_view.hpp"
#include <cmath>
namespace inference::cpu::kernels {

    inline void rmsnorm_add(TensorView<bf16_t> residual, TensorView<const bf16_t> residual_update, TensorView<const bf16_t> weight,
                            TensorView<bf16_t> output, float epsilon) {

        const auto hidden_size = residual.size();
        float sum_of_squares = 0.0F;
        for (std::size_t index = 0; index < hidden_size; ++index) {
            const auto updated = residual[index].to_float() + residual_update[index].to_float();
            residual[index] = updated;
            sum_of_squares += updated * updated;
        }

        const auto scale = 1.0F / std::sqrt(sum_of_squares / static_cast<float>(hidden_size) + epsilon);
        for (std::size_t index = 0; index < hidden_size; ++index) {
            const auto value = residual[index].to_float();
            const auto gain = weight[index].to_float();
            output[index] = value * scale * gain;
        }
    }

    inline void rmsnorm(TensorView<bf16_t> input, TensorView<const bf16_t> weight, TensorView<bf16_t> output, float epsilon) {
        const auto hidden_size = input.size();
        float sum_of_squares = 0.0F;
        for (std::size_t index = 0; index < hidden_size; ++index) {
            const auto value = input[index].to_float();
            sum_of_squares += value * value;
        }

        const auto scale = 1.0F / std::sqrt(sum_of_squares / static_cast<float>(hidden_size) + epsilon);
        for (std::size_t index = 0; index < hidden_size; ++index) {
            const auto value = input[index].to_float();
            const auto gain = weight(index).to_float();
            output[index] = value * scale * gain;
        }
    }
} // namespace inference::cpu::kernels
