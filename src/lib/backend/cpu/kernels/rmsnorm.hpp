#pragma once
#include "backend/cpu/avx.hpp"
#include "tensor/tensor_view.hpp"
#include <cmath>
namespace inference::cpu::kernels {
    namespace detail {
        inline void apply_rmsnorm(TensorView<bf16_t> input, TensorView<const bf16_t> weight, TensorView<bf16_t> output, const float scale) {
            constexpr auto lanes = avx::lanes_v<avx::f32x16>;
            for (std::size_t index = 0; index < input.size(); index += lanes) {
                const auto value = avx::load_bf16_as_f32(&input[index]);
                const auto gain = avx::load_bf16_as_f32(&weight[index]);
                avx::store_f32_as_bf16(&output[index], value * scale * gain);
            }
        }
    } // namespace detail

    inline void rmsnorm(TensorView<bf16_t> input, TensorView<const bf16_t> weight, TensorView<bf16_t> output, float epsilon) {
        const auto hidden_size = input.size();
        avx::f32x16 accumulator{};
        constexpr auto lanes = avx::lanes_v<avx::bf16x32>;
        for (std::size_t index = 0; index < hidden_size; index += lanes) {
            const auto value = avx::load<avx::bf16x32>(&input[index]);
            accumulator = avx::dot_bf16(value, value, accumulator);
        }

        const auto sum_of_squares = avx::reduce_add(accumulator);
        const auto scale = 1.0F / std::sqrt(sum_of_squares / static_cast<float>(hidden_size) + epsilon);
        detail::apply_rmsnorm(input, weight, output, scale);
    }

    inline void rmsnorm_add(TensorView<bf16_t> residual, TensorView<const bf16_t> residual_update, TensorView<const bf16_t> weight,
                            TensorView<bf16_t> output, float epsilon) {

        const auto hidden_size = residual.size();
        avx::f32x16 accumulator{};
        constexpr auto lanes = avx::lanes_v<avx::f32x16>;
        for (std::size_t index = 0; index < hidden_size; index += lanes) {
            const auto updated = avx::load_bf16_as_f32(&residual[index]) + avx::load_bf16_as_f32(&residual_update[index]);
            avx::store_f32_as_bf16(&residual[index], updated);
            accumulator += updated * updated;
        }

        const auto sum_of_squares = avx::reduce_add(accumulator);
        const auto scale = 1.0F / std::sqrt(sum_of_squares / static_cast<float>(hidden_size) + epsilon);
        detail::apply_rmsnorm(residual, weight, output, scale);
    }
} // namespace inference::cpu::kernels
